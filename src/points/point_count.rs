//! Point counting (port of `openvdb/points/PointCount.h`).

use super::index_filter::{ActiveFilter, GroupFilter, InactiveFilter, IndexFilter, NullFilter};
use super::index_iterator::IndexState;
use super::point_data_grid::{PointDataGrid, PointDataLeaf, PointDataTree, PointLeaf};
use super::point_mask::{convert_points_to_scalar, convert_points_to_scalar_transformed, CountValue, NullDeformer};
use crate::grid::Grid;
use crate::math::Transform;
use rayon::prelude::*;

/// Number of points of one leaf accepted by `filter`.
pub(crate) fn leaf_point_count<F: IndexFilter>(leaf: &PointDataLeaf<'_>, filter: &F) -> u64 {
    match filter.leaf_state(leaf.leaf, leaf.attributes) {
        IndexState::All => leaf.point_count(),
        IndexState::None => 0,
        IndexState::Partial => {
            if filter.state() == IndexState::None {
                0
            } else {
                leaf.index_all_filtered(filter).count_remaining()
            }
        }
    }
}

/// Total number of points in the tree.
pub fn point_count(tree: &PointDataTree) -> u64 {
    point_count_filtered(tree, &NullFilter)
}

/// Number of points accepted by `filter` (`pointCount(tree, filter)`).
pub fn point_count_filtered<F: IndexFilter>(tree: &PointDataTree, filter: &F) -> u64 {
    tree.leaf_vec().par_iter().map(|l| leaf_point_count(l, filter)).sum()
}

/// Number of points in active voxels.
pub fn active_point_count(tree: &PointDataTree) -> u64 {
    point_count_filtered(tree, &ActiveFilter::default())
}

/// Number of points in inactive voxels.
pub fn inactive_point_count(tree: &PointDataTree) -> u64 {
    point_count_filtered(tree, &InactiveFilter::default())
}

/// Number of points in group `name` (0 if the tree is empty or the group does not exist).
pub fn group_point_count(tree: &PointDataTree, name: &str) -> u64 {
    let Some(descriptor) = tree.descriptor() else { return 0 };
    match GroupFilter::new(name, descriptor) {
        Ok(f) => point_count_filtered(tree, &f),
        Err(_) => 0,
    }
}

/// Cumulative per-leaf counts of the points accepted by `filter`, in leaf order, and the
/// total (`pointOffsets()`): `offsets[i]` is the number of points in leaves `0..=i`.
pub fn point_offsets<F: IndexFilter>(tree: &PointDataTree, filter: &F) -> (Vec<u64>, u64) {
    let mut offsets: Vec<u64> = tree.leaf_vec().par_iter().map(|l| leaf_point_count(l, filter)).collect();
    let mut total = 0u64;
    for o in offsets.iter_mut() {
        total += *o;
        *o = total;
    }
    (offsets, total)
}

/// A grid with the number of points accepted by `filter` in each active voxel
/// (`pointCountGrid()`; OpenVDB defaults to `i32` values).
pub fn point_count_grid<T: CountValue, F: IndexFilter>(grid: &PointDataGrid, filter: &F) -> Grid<T> {
    let tree = convert_points_to_scalar::<T, F>(grid.tree(), filter);
    Grid::from_tree(tree).with_transform(grid.transform().clone())
}

/// A grid with the number of points accepted by `filter` in each voxel of a grid with a
/// different transform (`pointCountGrid(grid, transform, filter)`).
pub fn point_count_grid_transformed<T: CountValue, F: IndexFilter>(
    grid: &PointDataGrid,
    transform: &Transform,
    filter: &F,
) -> Grid<T> {
    convert_points_to_scalar_transformed::<T, F, NullDeformer>(grid, transform, filter, &NullDeformer)
}
