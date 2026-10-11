//! Apply differential operators to whole grids (port of `openvdb/tools/GridOperators.h`).
//!
//! Each function evaluates a world-space operator from [`crate::math::operators`] at every
//! active value of the input grid and returns a new grid with the same transform and active
//! topology (optionally intersected with the active topology of a mask grid). The `_masked`
//! variants restrict the output to voxels that are also active in the mask.
//!
//! | function | input | output | scheme |
//! |---|---|---|---|
//! | [`gradient`] | scalar | vector | `CD_2ND` |
//! | [`cpt`] | scalar (SDF) | vector (world-space closest point) | `CD_2ND` |
//! | [`curl`] | vector | vector | `CD_2ND` |
//! | [`divergence`] | vector | scalar | `CD_2ND` (`FD_1ST` for staggered grids) |
//! | [`laplacian`] | scalar | scalar | `CD_SECOND` |
//! | [`mean_curvature`] | scalar | scalar | `CD_SECOND`/`CD_2ND` |
//! | [`magnitude`] | vector | scalar | — |
//! | [`normalize`] | vector | vector | — |

use super::tree_ops::topology_copy;
use crate::grid::{Grid, GridClass, VecType};
use crate::math::finite_difference::{DDScheme, DScheme, StencilValues};
use crate::math::operators::{self as ops, MapJacobian};
use crate::math::{Coord, Vec3};
use crate::tree::{Internal2, LeafNode, Node, Tree, ValueAccessor};
use crate::types::{RealValue, Value};
use rayon::prelude::*;

type Acc<'a, T> = ValueAccessor<'a, Internal2<T>>;

/// The input values around a voxel. Neighbors in the voxel's own leaf (if it has one) are
/// read from the leaf buffer, the others through the accessor.
pub struct LeafAt<'s, 't, T: Value> {
    acc: &'s Acc<'t, T>,
    ijk: Coord,
    buf: Option<&'t [T]>,
}

impl<'s, 't, T: Value> LeafAt<'s, 't, T> {
    #[inline]
    fn new(acc: &'s Acc<'t, T>, ijk: Coord, leaf: Option<&'t LeafNode<T>>) -> Self {
        LeafAt { acc, ijk, buf: leaf.map(|l| l.buffer()) }
    }
}

impl<T: Value> StencilValues for LeafAt<'_, '_, T> {
    type Value = T;
    #[inline]
    fn value_at(&self, i: i32, j: i32, k: i32) -> T {
        let (x, y, z) = ((self.ijk[0] & 7) + i, (self.ijk[1] & 7) + j, (self.ijk[2] & 7) + k);
        match self.buf {
            Some(buf) if ((x | y | z) as u32) < 8 => buf[((x << 6) | (y << 3) | z) as usize],
            _ => self.acc.get_value(&self.ijk.offset_by(i, j, k)),
        }
    }
    #[inline]
    fn center_coord(&self) -> Coord {
        self.ijk
    }
}

/// The generic driver (`gridop::GridOperator::process`).
fn process<T, U, M, F>(input: &Grid<T>, mask: Option<&Grid<M>>, densify: bool, op: F) -> Grid<U>
where
    T: Value,
    U: Value,
    M: Value,
    F: Fn(&MapJacobian, &LeafAt<'_, '_, T>) -> U + Sync + Send,
{
    let map = MapJacobian::new(input.transform());
    // The background is the operator applied to a constant (empty) tree.
    let empty = Tree::new(*input.background());
    let bg = {
        let acc = empty.accessor();
        op(&map, &LeafAt::new(&acc, Coord::origin(), None))
    };
    let mut tree = topology_copy(input.tree(), bg);
    if densify || mask.is_some() {
        tree.voxelize_active_tiles();
    }
    if let Some(mask) = mask {
        let mtree = mask.tree();
        tree.leaf_refs_mut().into_par_iter().for_each_init(
            || mtree.accessor(),
            |macc, leaf: &mut LeafNode<U>| {
                let origin = leaf.origin();
                let on: Vec<usize> = leaf.value_mask().iter_on().collect();
                for n in on {
                    if !macc.is_value_on(&(origin + LeafNode::<U>::offset_to_local_coord(n))) {
                        leaf.set_active_state_at(n, false);
                    }
                }
            },
        );
    }
    let itree = input.tree();
    tree.leaf_refs_mut().into_par_iter().for_each_init(
        || itree.accessor(),
        |acc, leaf: &mut LeafNode<U>| {
            let origin = leaf.origin();
            let input_leaf = itree.probe_leaf(&origin);
            let (buf, vmask) = leaf.buffer_and_mask_mut();
            for n in vmask.iter_on() {
                let ijk = origin + LeafNode::<U>::offset_to_local_coord(n);
                buf[n] = op(&map, &LeafAt::new(acc, ijk, input_leaf));
            }
        },
    );
    if !densify {
        let acc = itree.accessor();
        tree.visit_tiles_mut(&mut |bbox, v, on, _| {
            if *on {
                *v = op(&map, &LeafAt::new(&acc, bbox.min, None));
            }
        });
    } else {
        tree.prune(<U as Value>::zero());
    }
    Grid::from_tree(tree).with_transform(input.transform().clone())
}

/// World-space gradient of a scalar grid (`tools::gradient`).
pub fn gradient<T: RealValue>(grid: &Grid<T>) -> Grid<Vec3<T>> {
    gradient_impl::<T, bool>(grid, None)
}
/// [`gradient`] restricted to the active voxels of `mask`.
pub fn gradient_masked<T: RealValue, M: Value>(grid: &Grid<T>, mask: &Grid<M>) -> Grid<Vec3<T>> {
    gradient_impl(grid, Some(mask))
}
fn gradient_impl<T: RealValue, M: Value>(grid: &Grid<T>, mask: Option<&Grid<M>>) -> Grid<Vec3<T>>
where
    Vec3<T>: Value,
{
    let mut out = process(grid, mask, true, |map, s| ops::gradient(map, DScheme::Cd2nd, s));
    out.set_vector_type(VecType::Covariant);
    out
}

/// World-space closest points on the zero level set of a signed distance field (`tools::cpt`
/// with `useWorldTransform = true`).
pub fn cpt<T: RealValue>(grid: &Grid<T>) -> Grid<Vec3<T>> {
    cpt_impl::<T, bool>(grid, None, true)
}
/// [`cpt`] restricted to the active voxels of `mask`.
pub fn cpt_masked<T: RealValue, M: Value>(grid: &Grid<T>, mask: &Grid<M>) -> Grid<Vec3<T>> {
    cpt_impl(grid, Some(mask), true)
}
/// Closest points in index space (`Cpt::process(threaded, /*useWorldTransform=*/false)`).
pub fn cpt_index_space<T: RealValue>(grid: &Grid<T>) -> Grid<Vec3<T>> {
    cpt_impl::<T, bool>(grid, None, false)
}
fn cpt_impl<T: RealValue, M: Value>(grid: &Grid<T>, mask: Option<&Grid<M>>, world: bool) -> Grid<Vec3<T>>
where
    Vec3<T>: Value,
{
    let mut out = if world {
        process(grid, mask, false, |map, s| ops::cpt_range(map, DScheme::Cd2nd, s))
    } else {
        process(grid, mask, false, |map, s| ops::cpt(map, DScheme::Cd2nd, s))
    };
    out.set_vector_type(VecType::ContravariantAbsolute);
    out
}

/// World-space curl of a vector grid (`tools::curl`).
pub fn curl<T: RealValue>(grid: &Grid<Vec3<T>>) -> Grid<Vec3<T>>
where
    Vec3<T>: Value,
{
    curl_impl::<T, bool>(grid, None)
}
/// [`curl`] restricted to the active voxels of `mask`.
pub fn curl_masked<T: RealValue, M: Value>(grid: &Grid<Vec3<T>>, mask: &Grid<M>) -> Grid<Vec3<T>>
where
    Vec3<T>: Value,
{
    curl_impl(grid, Some(mask))
}
fn curl_impl<T: RealValue, M: Value>(grid: &Grid<Vec3<T>>, mask: Option<&Grid<M>>) -> Grid<Vec3<T>>
where
    Vec3<T>: Value,
{
    let mut out = process(grid, mask, true, |map, s| ops::curl(map, DScheme::Cd2nd, s));
    out.set_vector_type(VecType::Covariant);
    out
}

/// World-space divergence of a vector grid (`tools::divergence`); staggered grids use
/// first-order forward differences.
pub fn divergence<T: RealValue>(grid: &Grid<Vec3<T>>) -> Grid<T>
where
    Vec3<T>: Value,
{
    divergence_impl::<T, bool>(grid, None)
}
/// [`divergence`] restricted to the active voxels of `mask`.
pub fn divergence_masked<T: RealValue, M: Value>(grid: &Grid<Vec3<T>>, mask: &Grid<M>) -> Grid<T>
where
    Vec3<T>: Value,
{
    divergence_impl(grid, Some(mask))
}
fn divergence_impl<T: RealValue, M: Value>(grid: &Grid<Vec3<T>>, mask: Option<&Grid<M>>) -> Grid<T>
where
    Vec3<T>: Value,
{
    let scheme = if grid.grid_class() == GridClass::Staggered { DScheme::Fd1st } else { DScheme::Cd2nd };
    process(grid, mask, true, move |map, s| ops::divergence(map, scheme, s))
}

/// World-space Laplacian of a scalar grid (`tools::laplacian`).
pub fn laplacian<T: RealValue>(grid: &Grid<T>) -> Grid<T> {
    laplacian_impl::<T, bool>(grid, None)
}
/// [`laplacian`] restricted to the active voxels of `mask`.
pub fn laplacian_masked<T: RealValue, M: Value>(grid: &Grid<T>, mask: &Grid<M>) -> Grid<T> {
    laplacian_impl(grid, Some(mask))
}
fn laplacian_impl<T: RealValue, M: Value>(grid: &Grid<T>, mask: Option<&Grid<M>>) -> Grid<T> {
    let mut out = process(grid, mask, true, |map, s| ops::laplacian(map, DDScheme::CdSecond, s));
    out.set_vector_type(VecType::Covariant);
    out
}

/// World-space mean curvature of a scalar grid (`tools::meanCurvature`).
pub fn mean_curvature<T: RealValue>(grid: &Grid<T>) -> Grid<T> {
    mean_curvature_impl::<T, bool>(grid, None)
}
/// [`mean_curvature`] restricted to the active voxels of `mask`.
pub fn mean_curvature_masked<T: RealValue, M: Value>(grid: &Grid<T>, mask: &Grid<M>) -> Grid<T> {
    mean_curvature_impl(grid, Some(mask))
}
fn mean_curvature_impl<T: RealValue, M: Value>(grid: &Grid<T>, mask: Option<&Grid<M>>) -> Grid<T> {
    let mut out = process(grid, mask, true, |map, s| ops::mean_curvature(map, DDScheme::CdSecond, DScheme::Cd2nd, s));
    out.set_vector_type(VecType::Covariant);
    out
}

/// Length of every vector of a vector grid (`tools::magnitude`).
pub fn magnitude<T: RealValue>(grid: &Grid<Vec3<T>>) -> Grid<T>
where
    Vec3<T>: Value,
{
    magnitude_impl::<T, bool>(grid, None)
}
/// [`magnitude`] restricted to the active voxels of `mask`.
pub fn magnitude_masked<T: RealValue, M: Value>(grid: &Grid<Vec3<T>>, mask: &Grid<M>) -> Grid<T>
where
    Vec3<T>: Value,
{
    magnitude_impl(grid, Some(mask))
}
fn magnitude_impl<T: RealValue, M: Value>(grid: &Grid<Vec3<T>>, mask: Option<&Grid<M>>) -> Grid<T>
where
    Vec3<T>: Value,
{
    use crate::math::finite_difference::StencilValues;
    process(grid, mask, false, |_, s| s.center_value().length())
}

/// Unit-length versions of every vector of a vector grid; zero vectors stay zero
/// (`tools::normalize`).
pub fn normalize<T: RealValue>(grid: &Grid<Vec3<T>>) -> Grid<Vec3<T>>
where
    Vec3<T>: Value,
{
    normalize_impl::<T, bool>(grid, None)
}
/// [`normalize`] restricted to the active voxels of `mask`.
pub fn normalize_masked<T: RealValue, M: Value>(grid: &Grid<Vec3<T>>, mask: &Grid<M>) -> Grid<Vec3<T>>
where
    Vec3<T>: Value,
{
    normalize_impl(grid, Some(mask))
}
fn normalize_impl<T: RealValue, M: Value>(grid: &Grid<Vec3<T>>, mask: Option<&Grid<M>>) -> Grid<Vec3<T>>
where
    Vec3<T>: Value,
{
    use crate::math::finite_difference::StencilValues;
    process(grid, mask, false, |_, s| {
        let mut v = s.center_value();
        if !v.normalize() {
            v = Vec3::zero();
        }
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{CoordBBox, Vec3d};
    use crate::tools::test_util::{assert_close, make_sphere};

    /// Reference values from the C++ operators (`scratchpad/cppref/fd.cpp`, OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let grid = make_sphere::<f32>(1.0, Vec3d::new(0.1, -0.05, 0.02), 0.05, 4.0);
        let g = gradient(&grid);
        let l = laplacian(&grid);
        let mc = mean_curvature(&grid);
        let cp = cpt(&grid);
        assert_eq!(g.active_voxel_count(), grid.active_voxel_count());
        assert_eq!(g.vector_type(), VecType::Covariant);
        let p = Coord::new(20, 3, -2);
        let gv = g.get_value(&p);
        for (i, e) in [0.967933476, 0.214819789, -0.128886476].iter().enumerate() {
            assert_close(gv[i] as f64, *e, 1e-6);
        }
        assert_close(l.get_value(&p) as f64, 2.15007067, 1e-5);
        assert_close(mc.get_value(&p) as f64, 1.07504153, 1e-5);
        let c = cp.get_value(&p);
        for (i, e) in [1.06801558, 0.165095136, -0.109056704].iter().enumerate() {
            assert_close(c[i] as f64, *e, 1e-6);
        }
        assert_eq!(*g.background(), Vec3::zero());
        let n = normalize(&g);
        assert_close(n.get_value(&p).length() as f64, 1.0, 1e-6);
        let m = magnitude(&g);
        assert_close(m.get_value(&p) as f64, gv.length() as f64, 1e-6);
        let d = divergence(&n);
        // Divergence of the unit normal is twice the mean curvature.
        assert_close(d.get_value(&p) as f64, 2.0 * 1.07504153, 0.05);
        let cu = curl(&n);
        assert!(cu.get_value(&p).length() < 0.05);
    }

    #[test]
    fn masks_and_tiles() {
        let mut grid = Grid::<f32>::new(0.0);
        grid.tree_mut().fill(&CoordBBox::from_coords(0, 0, 0, 31, 31, 31), 1.0, true);
        let mut mask = Grid::<bool>::new(false);
        mask.tree_mut().fill(&CoordBBox::from_coords(0, 0, 0, 3, 3, 3), true, true);
        let g = gradient_masked(&grid, &mask);
        assert_eq!(g.active_voxel_count(), 64);
        // Constant field: zero gradient everywhere, densified then pruned to tiles.
        let g = gradient(&grid);
        assert_eq!(g.active_voxel_count(), 32 * 32 * 32);
        assert_eq!(g.get_value(&Coord::new(5, 5, 5)), Vec3::zero());
        let mut vg = Grid::<Vec3<f32>>::new(Vec3::zero());
        vg.tree_mut().fill(&CoordBBox::from_coords(0, 0, 0, 15, 15, 15), Vec3::new(3.0, 4.0, 0.0), true);
        let m = magnitude(&vg);
        assert_eq!(m.get_value(&Coord::new(1, 2, 3)), 5.0);
        assert!(m.tree().active_tile_count() > 0);
    }
}
