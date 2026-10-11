//! Filtering (smoothing, offsetting) of narrow-band level sets (port of
//! `openvdb/tools/LevelSetFilter.h`).
//!
//! Every operation modifies the active voxels and then re-establishes the narrow band with
//! [`LevelSetTracker::track`]. Like the C++ class, [`LevelSetFilter`] dereferences to its
//! [`LevelSetTracker`], so the tracker's schemes and parameters can be set directly. All
//! filters accept an optional alpha mask (see [`super::interpolation::AlphaMask`]).

use super::leaf_buffers::{masked_voxel_pass, write_leaf_buffers, MaskSpec};
use super::level_set_tracker::{LevelSetError, LevelSetTracker};
use crate::grid::Grid;
use crate::math::stencils::{CurvatureStencil, DenseStencil, GradStencil};
use crate::types::{ArithValue, RealValue, Value};
use std::ops::{Deref, DerefMut};

/// Level-set filtering (`tools::LevelSetFilter`).
pub struct LevelSetFilter<'a, T: RealValue> {
    tracker: LevelSetTracker<'a, T>,
    min_mask: f64,
    max_mask: f64,
    invert_mask: bool,
}

impl<'a, T: RealValue> Deref for LevelSetFilter<'a, T> {
    type Target = LevelSetTracker<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.tracker
    }
}
impl<'a, T: RealValue> DerefMut for LevelSetFilter<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tracker
    }
}

impl<'a, T: RealValue + ArithValue> LevelSetFilter<'a, T> {
    /// A filter for the level set `grid`.
    pub fn new(grid: &'a mut Grid<T>) -> Result<Self, LevelSetError> {
        Ok(LevelSetFilter { tracker: LevelSetTracker::new(grid)?, min_mask: 0.0, max_mask: 1.0, invert_mask: false })
    }
    /// The mask value mapped to zero weight.
    pub fn min_mask(&self) -> f64 {
        self.min_mask
    }
    /// The mask value mapped to full weight.
    pub fn max_mask(&self) -> f64 {
        self.max_mask
    }
    /// Set the range of mask values remapped to `[0, 1]`.
    ///
    /// # Panics
    /// If `min >= max`.
    pub fn set_mask_range(&mut self, min: f64, max: f64) {
        assert!(min < max, "Invalid mask range (expects min < max)");
        self.min_mask = min;
        self.max_mask = max;
    }
    /// True if the mask is inverted.
    pub fn is_mask_inverted(&self) -> bool {
        self.invert_mask
    }
    /// Invert the mask.
    pub fn invert_mask(&mut self, invert: bool) {
        self.invert_mask = invert;
    }

    fn spec<'m, M: RealValue + ArithValue>(&self, mask: Option<&'m Grid<M>>) -> MaskSpec<'m, M> {
        mask.map(|m| (m, self.min_mask, self.max_mask, self.invert_mask))
    }

    fn box_pass<M: RealValue + ArithValue>(&mut self, axis: usize, w: i32, mask: Option<&Grid<M>>) {
        let spec = self.spec(mask);
        let frac = T::from_f64(1.0) / T::from_f64((2 * w + 1) as f64);
        let bufs = masked_voxel_pass(
            self.tracker.grid(),
            spec,
            |g| g.accessor(),
            move |acc, xyz, _| {
                let mut c = *xyz;
                let mut sum = <T as Value>::zero();
                for i in xyz[axis] - w..=xyz[axis] + w {
                    c[axis] = i;
                    sum += acc.get_value(&c);
                }
                sum * frac
            },
        );
        write_leaf_buffers(self.tracker.grid_mut().tree_mut(), &bufs);
    }

    fn box_impl<M: RealValue + ArithValue>(&mut self, width: i32, mask: Option<&Grid<M>>) {
        let w = width.max(1);
        // Same axis order as OpenVDB (x, then z, then y).
        self.box_pass(0, w, mask);
        self.box_pass(2, w, mask);
        self.box_pass(1, w, mask);
        self.tracker.track();
    }

    fn mean_curvature_impl<M: RealValue + ArithValue>(&mut self, mask: Option<&Grid<M>>) {
        let spec = self.spec(mask);
        let dx = self.tracker.voxel_size();
        let dt = dx * dx / T::from_f64(3.0);
        let bufs = masked_voxel_pass(
            self.tracker.grid(),
            spec,
            |g| CurvatureStencil::with_dx(g, dx.to_f64()),
            move |st, xyz, v| {
                st.move_to_with_center(xyz, v);
                v + dt * st.mean_curvature_norm_grad()
            },
        );
        write_leaf_buffers(self.tracker.grid_mut().tree_mut(), &bufs);
        self.tracker.track();
    }

    fn fillet_impl<M: RealValue + ArithValue>(&mut self, mask: Option<&Grid<M>>) {
        let spec = self.spec(mask);
        let dx = self.tracker.voxel_size();
        let dt = dx * dx / T::from_f64(3.0);
        let z = <T as Value>::zero();
        let bufs = masked_voxel_pass(
            self.tracker.grid(),
            spec,
            |g| CurvatureStencil::with_dx(g, dx.to_f64()),
            move |st, xyz, v| {
                st.move_to_with_center(xyz, v);
                let kappa = st.principal_curvatures().0;
                v + (dt * kappa).min(z)
            },
        );
        write_leaf_buffers(self.tracker.grid_mut().tree_mut(), &bufs);
        self.tracker.track();
    }

    fn laplacian_impl<M: RealValue + ArithValue>(&mut self, mask: Option<&Grid<M>>) {
        let spec = self.spec(mask);
        let dx = self.tracker.voxel_size();
        let dt = dx * dx / T::from_f64(6.0);
        let bufs = masked_voxel_pass(
            self.tracker.grid(),
            spec,
            |g| GradStencil::with_dx(g, dx.to_f64()),
            move |st, xyz, v| {
                st.move_to_with_center(xyz, v);
                v + dt * st.laplacian()
            },
        );
        write_leaf_buffers(self.tracker.grid_mut().tree_mut(), &bufs);
        self.tracker.track();
    }

    fn median_impl<M: RealValue + ArithValue>(&mut self, width: i32, mask: Option<&Grid<M>>) {
        let spec = self.spec(mask);
        let w = width.max(1);
        let bufs = masked_voxel_pass(
            self.tracker.grid(),
            spec,
            move |g| DenseStencil::new(g, w),
            |st, xyz, v| {
                st.move_to_with_center(xyz, v);
                st.median()
            },
        );
        write_leaf_buffers(self.tracker.grid_mut().tree_mut(), &bufs);
        self.tracker.track();
    }

    fn offset_impl<M: RealValue + ArithValue>(&mut self, value: T, mask: Option<&Grid<M>>) {
        let cfl = T::from_f64(0.5) * self.tracker.voxel_size();
        let offset = value.abs();
        let mut dist = <T as Value>::zero();
        while offset - dist > T::from_f64(0.001) * cfl {
            let delta = (offset - dist).min(cfl);
            dist += delta;
            let step = delta.copysign(value);
            match mask {
                None => {
                    self.tracker.grid_mut().tree_mut().par_for_each_leaf_mut(|l| l.for_each_on_mut(|_, v| *v += step))
                }
                Some(_) => {
                    let spec = self.spec(mask);
                    let bufs = masked_voxel_pass(self.tracker.grid(), spec, |_| (), move |_, _, v| v + step);
                    write_leaf_buffers(self.tracker.grid_mut().tree_mut(), &bufs);
                }
            }
            self.tracker.track();
        }
    }

    /// Mean-curvature flow (one explicit step of `∂φ/∂t = κ|∇φ|`).
    pub fn mean_curvature(&mut self) {
        self.mean_curvature_impl::<f32>(None)
    }
    /// [`mean_curvature`](Self::mean_curvature) blended by `mask`.
    pub fn mean_curvature_masked<M: RealValue + ArithValue>(&mut self, mask: &Grid<M>) {
        self.mean_curvature_impl(Some(mask))
    }
    /// Fillet concave regions: flow by the minimum principal curvature where it is negative.
    pub fn fillet(&mut self) {
        self.fillet_impl::<f32>(None)
    }
    /// [`fillet`](Self::fillet) blended by `mask`.
    pub fn fillet_masked<M: RealValue + ArithValue>(&mut self, mask: &Grid<M>) {
        self.fillet_impl(Some(mask))
    }
    /// Laplacian flow (one explicit step of `∂φ/∂t = Δφ`).
    pub fn laplacian(&mut self) {
        self.laplacian_impl::<f32>(None)
    }
    /// [`laplacian`](Self::laplacian) blended by `mask`.
    pub fn laplacian_masked<M: RealValue + ArithValue>(&mut self, mask: &Grid<M>) {
        self.laplacian_impl(Some(mask))
    }
    /// Approximate Gaussian smoothing: four separable box filters of half-width `width`.
    pub fn gaussian(&mut self, width: i32) {
        for _ in 0..4 {
            self.box_impl::<f32>(width, None);
        }
    }
    /// [`gaussian`](Self::gaussian) blended by `mask`.
    pub fn gaussian_masked<M: RealValue + ArithValue>(&mut self, width: i32, mask: &Grid<M>) {
        for _ in 0..4 {
            self.box_impl(width, Some(mask));
        }
    }
    /// Mean-value (separable box) smoothing of half-width `width`.
    pub fn mean(&mut self, width: i32) {
        self.box_impl::<f32>(width, None)
    }
    /// [`mean`](Self::mean) blended by `mask`.
    pub fn mean_masked<M: RealValue + ArithValue>(&mut self, width: i32, mask: &Grid<M>) {
        self.box_impl(width, Some(mask))
    }
    /// Median-value smoothing over a cube of half-width `width`.
    pub fn median(&mut self, width: i32) {
        self.median_impl::<f32>(width, None)
    }
    /// [`median`](Self::median) blended by `mask`.
    pub fn median_masked<M: RealValue + ArithValue>(&mut self, width: i32, mask: &Grid<M>) {
        self.median_impl(width, Some(mask))
    }
    /// Offset the surface by `value` world units along its normal (positive values dilate, i.e.
    /// add to φ), in steps of at most half a voxel with re-tracking in between.
    pub fn offset(&mut self, value: T) {
        self.offset_impl::<f32>(value, None)
    }
    /// [`offset`](Self::offset) scaled by `mask`.
    pub fn offset_masked<M: RealValue + ArithValue>(&mut self, value: T, mask: &Grid<M>) {
        self.offset_impl(value, Some(mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, Vec3d};
    use crate::tools::level_set_tracker::tests::{check_ref, distorted_sphere};
    use crate::tools::test_util::{assert_close, make_sphere};

    /// Reference values from `scratchpad/cppref/tracker.cpp` (OpenVDB 10 `LevelSetFilter`).
    #[test]
    fn matches_openvdb_reference() {
        let run = |f: &dyn Fn(&mut LevelSetFilter<'_, f32>), expect: [f64; 4], count: u64| {
            let mut grid = distorted_sphere();
            let mut filter = LevelSetFilter::new(&mut grid).unwrap();
            f(&mut filter);
            check_ref(&grid, expect, count, 2e-6);
        };
        run(&|f| f.mean_curvature(), [-0.078630738, 0.0151150487, -0.150000006, 0.0535619259], 22489);
        run(&|f| f.laplacian(), [-0.0790883601, 0.0150658255, -0.150000006, 0.0538162254], 22589);
        run(&|f| f.gaussian(1), [-0.0588995479, 0.0194477476, -0.150000006, 0.0518786609], 28530);
        run(&|f| f.mean(1), [-0.0778643414, 0.0160310566, -0.150000006, 0.0546221845], 22835);
        run(&|f| f.median(1), [-0.0801469162, 0.0142274341, -0.150000006, 0.052914951], 22649);
        run(&|f| f.offset(0.07), [-0.00799824856, 0.0709287897, -0.106700636, 0.104083918], 24359);
    }

    #[test]
    fn fillet_and_masks() {
        let mut grid = make_sphere::<f32>(1.0, Vec3d::zero(), 0.05, 3.0);
        let before = grid.get_value(&Coord::new(19, 0, 0));
        {
            // A convex sphere has positive curvature everywhere: fillet leaves it unchanged.
            let mut f = LevelSetFilter::new(&mut grid).unwrap();
            f.fillet();
        }
        assert_close(grid.get_value(&Coord::new(19, 0, 0)) as f64, before as f64, 2e-3);
        // A zero mask disables the offset entirely.
        let mask = Grid::<f32>::new(0.0);
        {
            let mut f = LevelSetFilter::new(&mut grid).unwrap();
            f.offset_masked(0.1, &mask);
        }
        assert_close(grid.get_value(&Coord::new(19, 0, 0)) as f64, before as f64, 2e-3);
        {
            let mut f = LevelSetFilter::new(&mut grid).unwrap();
            f.invert_mask(true);
            f.offset_masked(0.1, &mask);
        }
        assert_close(grid.get_value(&Coord::new(19, 0, 0)) as f64, before as f64 + 0.1, 5e-3);
    }
}
