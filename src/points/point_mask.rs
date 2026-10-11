//! Conversion of points to mask and count grids (port of `openvdb/points/PointMask.h`).

use super::attribute_array::AttributeHandle;
use super::index_filter::IndexFilter;
use super::index_iterator::{IndexState, VoxelSelection};
use super::point_data_grid::{PointDataGrid, PointDataLeaf, PointDataTree, PointLeaf};
use crate::grid::Grid;
use crate::math::{Coord, Transform, Vec3d, Vec3f};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::Value;
use rayon::prelude::*;

/// Voxel value types of count and mask grids built from points.
pub trait CountValue: Value {
    /// The value for `n` points.
    fn from_count(n: u64) -> Self;
    /// The value after adding one point (`SumOp(1)`).
    fn add_one(self) -> Self;
    /// The sum of two counts (or for bool).
    fn sum(self, other: Self) -> Self;
}

macro_rules! impl_count {
    ($($t:ty),*) => {$(
        impl CountValue for $t {
            #[inline]
            fn from_count(n: u64) -> Self { n as $t }
            #[inline]
            fn add_one(self) -> Self { self + 1 as $t }
            #[inline]
            fn sum(self, other: Self) -> Self { self + other }
        }
    )*};
}
impl_count!(i32, i64, u32, u64, f32, f64);

impl CountValue for bool {
    fn from_count(n: u64) -> Self {
        n > 0
    }
    fn add_one(self) -> Self {
        true
    }
    fn sum(self, other: Self) -> Self {
        self || other
    }
}

/// Moves points before they are binned (OpenVDB's deformer concept, used by
/// [`convert_points_to_mask_transformed`] and [`move_points`](super::move_points)).
pub trait PointDeformer: Clone + Sync {
    /// True if [`apply`](Self::apply) works on index-space positions (`DeformerTraits`).
    const INDEX_SPACE: bool = false;
    /// True only for [`NullDeformer`] (enables shortcuts).
    const IS_NULL: bool = false;
    /// Prepare for the points of a leaf (`reset(leaf, idx)`).
    fn reset(&mut self, leaf: &PointDataLeaf<'_>, leaf_index: usize);
    /// Move point `index` (in voxel `ijk`) whose position is `pos`.
    fn apply(&self, pos: &mut Vec3d, index: u32, ijk: &Coord);
}

/// A deformer that leaves points in place (`NullDeformer`).
#[derive(Clone, Copy, Debug, Default)]
pub struct NullDeformer;

impl PointDeformer for NullDeformer {
    const IS_NULL: bool = true;
    fn reset(&mut self, _: &PointDataLeaf<'_>, _: usize) {}
    fn apply(&self, _: &mut Vec3d, _: u32, _: &Coord) {}
}

/// A tree with the topology of the active voxels of `points` whose values are the number of
/// points accepted by `filter` (voxels without accepted points are deactivated).
pub(crate) fn convert_points_to_scalar<T: CountValue, F: IndexFilter>(points: &PointDataTree, filter: &F) -> Tree<T> {
    let mut tree = Tree::<T>::new(T::default());
    tree.topology_union(points.offsets());
    let is_bool = std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>();
    if points.leaf_count() == 0 {
        return tree;
    }
    if is_bool && filter.state() == IndexState::All {
        tree.par_for_each_leaf_mut(|leaf| {
            let (buf, mask) = leaf.buffer_and_mask_mut();
            for n in mask.iter_on() {
                buf[n] = T::from_count(1);
            }
        });
        return tree;
    }
    tree.par_for_each_leaf_mut(|leaf| {
        let Some(pl) = points.probe_leaf(&leaf.origin()) else { return };
        let bound_all = filter.state() == IndexState::All;
        let on: Vec<usize> = leaf.value_mask().iter_on().collect();
        for n in on {
            let count = if bound_all {
                pl.voxel_range(n).len() as u64
            } else {
                pl.index_filtered(VoxelSelection::Voxel(n), filter).count_remaining()
            };
            if count > 0 {
                leaf.set_value_only_at(n, T::from_count(count));
            } else {
                leaf.set_active_state_at(n, false);
            }
        }
    });
    tree
}

/// Bin the (deformed) points accepted by `filter` into a grid with `transform`, counting
/// points per voxel (`convertPointsToScalar(grid, transform, filter, deformer)`).
pub(crate) fn convert_points_to_scalar_transformed<T: CountValue, F: IndexFilter, D: PointDeformer>(
    points: &PointDataGrid,
    transform: &Transform,
    filter: &F,
    deformer: &D,
) -> Grid<T> {
    let is_null = D::IS_NULL;
    if *transform == *points.transform() && is_null {
        return Grid::from_tree(convert_points_to_scalar::<T, F>(points.tree(), filter))
            .with_transform(transform.clone());
    }
    let mut grid = Grid::<T>::new(T::default()).with_transform(transform.clone());
    if points.tree().leaf_count() == 0 {
        return grid;
    }
    let source = points.transform();
    let leaves = points.tree().leaf_vec();
    let bin = |idx: usize, leaf: &PointDataLeaf<'_>, tree: &mut Tree<T>| {
        let mut deformer = deformer.clone();
        deformer.reset(leaf, idx);
        let Ok(handle) = AttributeHandle::<Vec3f>::new(match leaf.attribute_array("P") {
            Ok(a) => a,
            Err(_) => return,
        }) else {
            return;
        };
        for (ijk, i) in leaf.index_filtered(VoxelSelection::On, filter).with_coords() {
            let p = handle.get(i);
            let mut pos = Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64) + ijk.as_vec3d();
            if D::INDEX_SPACE {
                deformer.apply(&mut pos, i, &ijk);
                pos = source.index_to_world(pos);
            } else {
                pos = source.index_to_world(pos);
                deformer.apply(&mut pos, i, &ijk);
            }
            let c = transform.world_to_index_cell_centered(pos);
            let target = tree.touch_leaf(&c);
            let n = LeafNode::<T>::coord_to_offset(&c);
            let v = target.get_value_at(n).add_one();
            target.set_value_on_at(n, v);
        }
    };
    let tree = leaves
        .par_iter()
        .enumerate()
        .fold(
            || Tree::<T>::new(T::default()),
            |mut tree, (idx, leaf)| {
                bin(idx, leaf, &mut tree);
                tree
            },
        )
        .reduce(
            || Tree::<T>::new(T::default()),
            |mut a, b| {
                merge_counts(&mut a, b);
                a
            },
        );
    grid.set_tree(tree);
    grid
}

/// Merge two count trees (bool trees are or-ed, numeric trees summed).
fn merge_counts<T: CountValue>(dst: &mut Tree<T>, src: Tree<T>) {
    for leaf in src.leaves() {
        match dst.probe_leaf_mut(&leaf.origin()) {
            Some(d) => {
                for (n, v) in leaf.iter_on() {
                    let cur = *d.get_value_at(n);
                    d.set_value_on_at(n, cur.sum(*v));
                }
            }
            None => dst.add_leaf(Box::new(leaf.clone())),
        }
    }
}

/// A mask (bool tree whose values equal the active states) of the voxels containing points
/// accepted by `filter` (`convertPointsToMask(tree, filter)`).
pub fn convert_points_to_mask_tree<F: IndexFilter>(tree: &PointDataTree, filter: &F) -> Tree<bool> {
    convert_points_to_scalar::<bool, F>(tree, filter)
}

/// A mask grid of the voxels containing points accepted by `filter`, with the grid's
/// transform (`convertPointsToMask(grid, filter)`). Values equal active states, as in
/// [`AnyGrid::Mask`](crate::AnyGrid::Mask).
pub fn convert_points_to_mask<F: IndexFilter>(grid: &PointDataGrid, filter: &F) -> Grid<bool> {
    Grid::from_tree(convert_points_to_mask_tree(grid.tree(), filter)).with_transform(grid.transform().clone())
}

/// A mask grid with `transform` of the voxels containing points accepted by `filter`
/// (`convertPointsToMask(grid, transform, filter)`).
pub fn convert_points_to_mask_transformed<F: IndexFilter>(
    grid: &PointDataGrid,
    transform: &Transform,
    filter: &F,
) -> Grid<bool> {
    convert_points_to_scalar_transformed::<bool, F, NullDeformer>(grid, transform, filter, &NullDeformer)
}

/// Like [`convert_points_to_mask_transformed`], moving the points with `deformer` first.
pub fn convert_points_to_mask_deformed<F: IndexFilter, D: PointDeformer>(
    grid: &PointDataGrid,
    transform: &Transform,
    filter: &F,
    deformer: &D,
) -> Grid<bool> {
    convert_points_to_scalar_transformed::<bool, F, D>(grid, transform, filter, deformer)
}
