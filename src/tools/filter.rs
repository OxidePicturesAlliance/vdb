//! Volume filtering: mean, Gaussian, median and offset (port of `openvdb/tools/Filter.h`).
//!
//! [`Filter`] modifies the active voxels of a grid in place. Mean and Gaussian filters are
//! applied as separable box filters (a Gaussian is approximated by four successive box passes),
//! exactly like OpenVDB. Every filter optionally takes a scalar mask grid whose values, remapped
//! from `[min, max]` to `[0, 1]` with a smooth step (see
//! [`AlphaMask`](super::interpolation::AlphaMask)), blend between the original and the filtered
//! values.
//!
//! ```
//! use vdb::prelude::*;
//! use vdb::tools::filter::Filter;
//! let mut grid = FloatGrid::new(0.0);
//! grid.tree_mut().fill(&CoordBBox::from_coords(0, 0, 0, 7, 7, 7), 1.0, true);
//! grid.tree_mut().voxelize_active_tiles();
//! grid.set_value(&Coord::new(3, 3, 3), 28.0);
//! Filter::new(&mut grid).mean(1, 1);
//! assert_eq!(grid.get_value(&Coord::new(3, 3, 3)), 2.0);
//! ```

use super::internal::{masked_voxel_pass, write_leaf_buffers, MaskSpec};
use super::interpolation::AlphaMask;
use crate::grid::Grid;
use crate::math::stencils::DenseStencil;
use crate::types::{ArithValue, RealValue, Value};

/// In-place filtering of a grid's active voxels (`tools::Filter`).
pub struct Filter<'a, T: Value> {
    grid: &'a mut Grid<T>,
    min_mask: f64,
    max_mask: f64,
    invert_mask: bool,
    process_tiles: bool,
}

impl<'a, T: ArithValue> Filter<'a, T> {
    /// A filter operating on `grid`.
    pub fn new(grid: &'a mut Grid<T>) -> Self {
        Filter { grid, min_mask: 0.0, max_mask: 1.0, invert_mask: false, process_tiles: false }
    }
    /// The mask value mapped to zero blending.
    pub fn min_mask(&self) -> f64 {
        self.min_mask
    }
    /// The mask value mapped to full blending.
    pub fn max_mask(&self) -> f64 {
        self.max_mask
    }
    /// Set the range of mask values that is remapped to `[0, 1]`.
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
    /// Invert the mask (filter where the mask is low).
    pub fn invert_mask(&mut self, invert: bool) {
        self.invert_mask = invert;
    }
    /// True if active tiles are processed.
    pub fn process_tiles(&self) -> bool {
        self.process_tiles
    }
    /// If set, active tiles are voxelized before filtering so that they are filtered too.
    /// (OpenVDB only densifies tiles near leaf nodes; this port densifies all active tiles.)
    pub fn set_process_tiles(&mut self, on: bool) {
        self.process_tiles = on;
    }

    fn mask_spec<'g, M: RealValue + ArithValue>(&self, mask: Option<&'g Grid<M>>) -> MaskSpec<'g, M> {
        mask.map(|m| (m, self.min_mask, self.max_mask, self.invert_mask))
    }

    fn box_pass<M: RealValue + ArithValue>(&mut self, axis: usize, w: i32, mask: Option<&Grid<M>>) {
        let frac = (1.0f32 / (2 * w + 1) as f32) as f64;
        let spec = self.mask_spec(mask);
        let bufs = masked_voxel_pass(
            &*self.grid,
            spec,
            |g| g.accessor(),
            move |acc, xyz, _| {
                let mut c = *xyz;
                let mut sum = <T as Value>::zero();
                for i in xyz[axis] - w..=xyz[axis] + w {
                    c[axis] = i;
                    sum = sum + acc.get_value(&c);
                }
                sum.scale(frac)
            },
        );
        write_leaf_buffers(self.grid.tree_mut(), &bufs);
    }

    fn prepare(&mut self) {
        if self.process_tiles {
            self.grid.tree_mut().voxelize_active_tiles();
        }
    }

    fn mean_impl<M: RealValue + ArithValue>(&mut self, width: i32, iterations: usize, mask: Option<&Grid<M>>) {
        if iterations == 0 {
            return;
        }
        self.prepare();
        let w = width.max(1);
        for _ in 0..iterations {
            self.box_pass(0, w, mask);
            self.box_pass(2, w, mask);
            self.box_pass(1, w, mask);
        }
    }

    fn gaussian_impl<M: RealValue + ArithValue>(&mut self, width: i32, iterations: usize, mask: Option<&Grid<M>>) {
        if iterations == 0 {
            return;
        }
        self.prepare();
        let w = width.max(1);
        for _ in 0..iterations {
            for _ in 0..4 {
                self.box_pass(0, w, mask);
                self.box_pass(2, w, mask);
                self.box_pass(1, w, mask);
            }
        }
    }

    fn offset_impl<M: RealValue + ArithValue>(&mut self, value: T, mask: Option<&Grid<M>>) {
        if self.process_tiles {
            match mask {
                None => self.grid.tree_mut().visit_tiles_mut(&mut |_, v, on, _| {
                    if *on {
                        *v = *v + value;
                    }
                }),
                Some(m) => {
                    let xf = self.grid.transform().clone();
                    let am = AlphaMask::new(&xf, m, self.min_mask, self.max_mask, self.invert_mask);
                    self.grid.tree_mut().visit_tiles_mut(&mut |b, v, on, _| {
                        if *on {
                            if let Some((a, _)) = am.weights(&b.min) {
                                *v = *v + value.scale(a);
                            }
                        }
                    });
                }
            }
        }
        match mask {
            None => self.grid.tree_mut().par_for_each_leaf_mut(|leaf| leaf.for_each_on_mut(|_, v| *v = *v + value)),
            Some(_) => {
                let spec = self.mask_spec(mask);
                let bufs = masked_voxel_pass(&*self.grid, spec, |_| (), move |_, _, v| v + value);
                write_leaf_buffers(self.grid.tree_mut(), &bufs);
            }
        }
    }

    /// Mean (box) filter of half-width `width`, applied `iterations` times.
    pub fn mean(&mut self, width: i32, iterations: usize) {
        self.mean_impl::<f32>(width, iterations, None)
    }
    /// [`mean`](Self::mean) blended by `mask`.
    pub fn mean_masked<M: RealValue + ArithValue>(&mut self, width: i32, iterations: usize, mask: &Grid<M>) {
        self.mean_impl(width, iterations, Some(mask))
    }
    /// Approximate Gaussian filter (four box passes of half-width `width`), applied
    /// `iterations` times.
    pub fn gaussian(&mut self, width: i32, iterations: usize) {
        self.gaussian_impl::<f32>(width, iterations, None)
    }
    /// [`gaussian`](Self::gaussian) blended by `mask`.
    pub fn gaussian_masked<M: RealValue + ArithValue>(&mut self, width: i32, iterations: usize, mask: &Grid<M>) {
        self.gaussian_impl(width, iterations, Some(mask))
    }
    /// Add `value` to every active voxel.
    pub fn offset(&mut self, value: T) {
        self.offset_impl::<f32>(value, None)
    }
    /// Add `value`, scaled by the mask weight, to every active voxel.
    pub fn offset_masked<M: RealValue + ArithValue>(&mut self, value: T, mask: &Grid<M>) {
        self.offset_impl(value, Some(mask))
    }
}

impl<'a, T: ArithValue + PartialOrd> Filter<'a, T> {
    fn median_impl<M: RealValue + ArithValue>(&mut self, width: i32, iterations: usize, mask: Option<&Grid<M>>) {
        if iterations == 0 {
            return;
        }
        self.prepare();
        let w = width.max(1);
        for _ in 0..iterations {
            let spec = self.mask_spec(mask);
            let bufs = masked_voxel_pass(
                &*self.grid,
                spec,
                move |g| DenseStencil::new(g, w),
                |st, xyz, v| {
                    st.move_to_with_center(xyz, v);
                    st.median()
                },
            );
            write_leaf_buffers(self.grid.tree_mut(), &bufs);
        }
    }
    /// Median filter over a cube of half-width `width`, applied `iterations` times.
    pub fn median(&mut self, width: i32, iterations: usize) {
        self.median_impl::<f32>(width, iterations, None)
    }
    /// [`median`](Self::median) blended by `mask`.
    pub fn median_masked<M: RealValue + ArithValue>(&mut self, width: i32, iterations: usize, mask: &Grid<M>) {
        self.median_impl(width, iterations, Some(mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, CoordBBox};
    use crate::tools::test_util::assert_close;

    /// Deterministic test volume shared with `scratchpad/cppref/filter.cpp`.
    pub(crate) fn test_grid() -> Grid<f32> {
        let mut grid = Grid::<f32>::new(0.25);
        for c in CoordBBox::from_coords(-6, -6, -6, 9, 9, 9) {
            let v = ((0.7 * c[0] as f64).sin() * (0.3 * c[1] as f64).cos() + 0.05 * c[2] as f64) as f32;
            if (c[0] + c[1] + c[2]).rem_euclid(7) != 0 {
                grid.tree_mut().set_value_on(&c, v);
            } else {
                grid.tree_mut().set_value_off(&c, v);
            }
        }
        grid
    }

    fn mask_grid() -> Grid<f32> {
        let mut m = Grid::<f32>::new(0.0);
        for c in CoordBBox::from_coords(-6, -6, -6, 9, 9, 9) {
            m.tree_mut().set_value(&c, ((c[0] + 6) as f32 / 15.0).clamp(0.0, 1.0));
        }
        m
    }

    const PTS: [[i32; 3]; 4] = [[0, 0, 1], [3, -2, 5], [-6, 9, 0], [7, 7, 8]];

    fn check(grid: &Grid<f32>, expect: &[f64; 4]) {
        for (p, e) in PTS.iter().zip(expect.iter()) {
            assert_close(grid.get_value(&Coord::new(p[0], p[1], p[2])) as f64, *e, 2e-6);
        }
    }

    /// Reference values from the C++ `tools::Filter` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let mut g = test_grid();
        Filter::new(&mut g).mean(1, 1);
        check(&g, &[0.0500000082, 0.888364017, -0.0840511769, 0.826081753]);
        let mut g = test_grid();
        Filter::new(&mut g).mean(2, 2);
        check(&g, &[0.0395730063, 0.649464011, 0.123688355, 0.505738199]);
        let mut g = test_grid();
        Filter::new(&mut g).gaussian(1, 1);
        check(&g, &[0.0144406585, 0.801081896, 0.158171266, 0.642042041]);
        let mut g = test_grid();
        Filter::new(&mut g).median(1, 1);
        check(&g, &[0.0500000007, 0.862565398, 0.25, 0.815492034]);
        let mut g = test_grid();
        Filter::new(&mut g).offset(0.5);
        check(&g, &[0.550000012, 1.46243739, -0.287967384, 1.39598739]);
        let mask = mask_grid();
        let mut g = test_grid();
        Filter::new(&mut g).mean_masked(1, 1, &mask);
        check(&g, &[0.0500000082, 0.904687524, -0.787967384, 0.829192877]);
        let mut g = test_grid();
        let mut f = Filter::new(&mut g);
        f.invert_mask(true);
        f.set_mask_range(0.2, 0.8);
        f.median_masked(1, 1, &mask);
        check(&g, &[0.0500000007, 0.936544716, 0.25, 0.895987391]);
    }
}
