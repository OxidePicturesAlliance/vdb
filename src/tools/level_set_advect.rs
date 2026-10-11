//! Advection of narrow-band level sets in a velocity field (port of
//! `openvdb/tools/LevelSetAdvect.h`).
//!
//! [`LevelSetAdvection`] solves `∂φ/∂t + V·∇φ = 0` on the active voxels with upwind spatial
//! differencing, TVD Runge-Kutta time integration and CFL-limited sub-steps, and keeps the
//! narrow band valid with a [`LevelSetTracker`] after every step.

use super::level_set_tracker::{tvd_rk_step, LevelSetError, LevelSetTracker};
use super::velocity_fields::VelocityField;
use crate::grid::Grid;
use crate::math::finite_difference::{BiasedGradientScheme, TemporalIntegrationScheme};
use crate::math::operators::{gradient_biased, MapJacobian};
use crate::math::Vec3;
use crate::tree::{LeafNode, Node};
use crate::types::{RealValue, Value};
use rayon::prelude::*;

/// Hyperbolic advection of a level set (`tools::LevelSetAdvection`).
pub struct LevelSetAdvection<'a, 'f, T: RealValue, F: VelocityField<T>> {
    tracker: LevelSetTracker<'a, T>,
    field: &'f F,
    spatial: BiasedGradientScheme,
    temporal: TemporalIntegrationScheme,
}

impl<'a, 'f, T, F> LevelSetAdvection<'a, 'f, T, F>
where
    T: RealValue,
    Vec3<T>: Value,
    F: VelocityField<T>,
{
    /// Advect the level set `grid` in `field`.
    pub fn new(grid: &'a mut Grid<T>, field: &'f F) -> Result<Self, LevelSetError> {
        if !grid.transform().is_linear() {
            return Err(LevelSetError::UnsupportedTransform("level set advection requires a linear transform"));
        }
        Ok(LevelSetAdvection {
            tracker: LevelSetTracker::new(grid)?,
            field,
            spatial: BiasedGradientScheme::HjWeno5Bias,
            temporal: TemporalIntegrationScheme::TvdRk2,
        })
    }
    /// The level set grid.
    pub fn grid(&self) -> &Grid<T> {
        self.tracker.grid()
    }
    /// The tracker used to maintain the narrow band.
    pub fn tracker(&mut self) -> &mut LevelSetTracker<'a, T> {
        &mut self.tracker
    }
    /// The spatial scheme of the advection.
    pub fn spatial_scheme(&self) -> BiasedGradientScheme {
        self.spatial
    }
    /// Set the spatial scheme of the advection.
    pub fn set_spatial_scheme(&mut self, s: BiasedGradientScheme) {
        self.spatial = s;
    }
    /// The temporal scheme of the advection.
    pub fn temporal_scheme(&self) -> TemporalIntegrationScheme {
        self.temporal
    }
    /// Set the temporal scheme of the advection.
    pub fn set_temporal_scheme(&mut self, s: TemporalIntegrationScheme) {
        self.temporal = s;
    }
    /// The tracker's spatial scheme.
    pub fn tracker_spatial_scheme(&self) -> BiasedGradientScheme {
        self.tracker.spatial_scheme()
    }
    /// Set the tracker's spatial scheme.
    pub fn set_tracker_spatial_scheme(&mut self, s: BiasedGradientScheme) {
        self.tracker.set_spatial_scheme(s);
    }
    /// The tracker's temporal scheme.
    pub fn tracker_temporal_scheme(&self) -> TemporalIntegrationScheme {
        self.tracker.temporal_scheme()
    }
    /// Set the tracker's temporal scheme.
    pub fn set_tracker_temporal_scheme(&mut self, s: TemporalIntegrationScheme) {
        self.tracker.set_temporal_scheme(s);
    }
    /// Number of normalizations per tracking step.
    pub fn norm_count(&self) -> usize {
        self.tracker.norm_count()
    }
    /// Set the number of normalizations per tracking step.
    pub fn set_norm_count(&mut self, n: usize) {
        self.tracker.set_norm_count(n);
    }
    /// The parallel grain size.
    pub fn grain_size(&self) -> usize {
        self.tracker.grain_size()
    }
    /// Set the parallel grain size.
    pub fn set_grain_size(&mut self, g: usize) {
        self.tracker.set_grain_size(g);
    }

    /// Sample the velocity at every active voxel (negated for backward advection) and return
    /// the per-leaf velocities and the CFL-limited time step.
    fn sample_field(&self, time0: T, time1: T) -> (Vec<Vec<Vec3<T>>>, T) {
        let grid = self.tracker.grid();
        let forward = time0 < time1;
        let aligned = self.field.transform() == *grid.transform();
        let xform = grid.transform();
        let field = self.field;
        let vel: Vec<Vec<Vec3<T>>> = grid
            .tree()
            .leaf_refs()
            .par_iter()
            .map(|leaf: &&LeafNode<T>| {
                let mut out = vec![Vec3::zero(); LeafNode::<T>::SIZE];
                let origin = leaf.origin();
                for n in leaf.value_mask().iter_on() {
                    let ijk = origin + LeafNode::<T>::offset_to_local_coord(n);
                    let v = if aligned {
                        field.sample_index(&ijk, time0)
                    } else {
                        field.sample_world(xform.index_to_world(ijk.as_vec3d()), time0)
                    };
                    out[n] = if forward { v } else { -v };
                }
                out
            })
            .collect();
        let max_v2 = grid
            .tree()
            .leaf_refs()
            .par_iter()
            .zip(vel.par_iter())
            .map(|(leaf, v)| leaf.value_mask().iter_on().map(|n| v[n].length_sqr()).fold(<T as Value>::zero(), T::max))
            .reduce(<T as Value>::zero, T::max);
        if RealValue::to_f64(max_v2).abs() <= RealValue::to_f64(<T as RealValue>::delta()) {
            return (vel, <T as Value>::zero());
        }
        let cfl = match self.temporal {
            TemporalIntegrationScheme::TvdRk1 => 0.3,
            TemporalIntegrationScheme::TvdRk2 => 0.9,
            TemporalIntegrationScheme::TvdRk3 => 1.0,
        };
        let cfl = T::from_f64(cfl) / T::from_f64(3.0).sqrt();
        let dt = (time1 - time0).abs();
        let dx = self.tracker.voxel_size();
        (vel, dt.min(cfl * dx / max_v2.sqrt()))
    }

    /// Advect the level set from `time0` to `time1` (backwards if `time1 < time0`); returns the
    /// number of CFL-limited time steps taken.
    pub fn advect(&mut self, mut time0: T, time1: T) -> usize {
        let mut count = 0;
        if (time0 - time1).abs() <= T::tolerance() {
            return 0;
        }
        let forward = time0 < time1;
        let spatial = self.spatial;
        while if forward { time0 < time1 } else { time0 > time1 } {
            if self.tracker.grid().tree().leaf_count() == 0 {
                break;
            }
            let (vel, dt) = self.sample_field(time0, time1);
            if dt.to_f64().abs() <= T::tolerance().to_f64() {
                break;
            }
            let map = MapJacobian::new(self.tracker.grid().transform());
            tvd_rk_step(self.tracker.grid_mut(), self.temporal, spatial.stencil_kind(), None, |st, leaf, n| {
                let v = vel[leaf][n];
                Some(st.center_value() - dt * v.dot(&gradient_biased(&map, spatial, st, v)))
            });
            time0 = if forward { time0 + dt } else { time0 - dt };
            count += 1;
            self.tracker.track();
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, CoordBBox, Vec3d};
    use crate::tools::level_set_tracker::tests::REF_PTS;
    use crate::tools::test_util::{assert_close, make_sphere};
    use crate::tools::velocity_fields::{DiscreteField, EnrightField};

    /// Reference values from `scratchpad/cppref/advect.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        // Constant velocity field stored in a grid (translation by (0.5, 0, 0)*t).
        let mut vel = Grid::<Vec3<f32>>::new(Vec3::new(0.5, 0.25, 0.0));
        vel.tree_mut().fill(&CoordBBox::from_coords(-64, -64, -64, 63, 63, 63), Vec3::new(0.5, 0.25, 0.0), true);
        let field: DiscreteField<'_, f32> = DiscreteField::new(&vel);
        let mut grid = make_sphere::<f32>(1.0, Vec3d::new(0.1, -0.05, 0.02), 0.05, 3.0);
        let steps = LevelSetAdvection::new(&mut grid, &field).unwrap().advect(0.0, 0.1);
        assert_eq!(steps, 3);
        let expect = [-0.123869315, 0.00201789243, -0.150000006, 0.0411688909];
        for (p, e) in REF_PTS.iter().zip(expect.iter()) {
            assert_close(grid.get_value(p) as f64, *e, 2e-6);
        }
        assert_eq!(grid.active_voxel_count(), 30315);

        let enright = EnrightField::<f32>::new();
        let mut grid = make_sphere::<f32>(0.15, Vec3d::new(0.35, 0.35, 0.35), 0.01, 3.0);
        let mut adv = LevelSetAdvection::new(&mut grid, &enright).unwrap();
        adv.set_temporal_scheme(TemporalIntegrationScheme::TvdRk3);
        let steps = adv.advect(0.0, 0.05);
        assert_eq!(steps, 18);
        let pts = [
            Coord::new(48, 35, 35),
            Coord::new(49, 35, 35),
            Coord::new(21, 35, 35),
            Coord::new(35, 35, 49),
            Coord::new(35, 48, 35),
            Coord::new(35, 22, 35),
        ];
        let expect = [-0.0299999993, -0.0299999993, 0.0181409083, 0.0225733146, 0.0127704218, -0.0170300081];
        for (p, e) in pts.iter().zip(expect.iter()) {
            assert_close(grid.get_value(p) as f64, *e, 2e-6);
        }
        assert_eq!(grid.active_voxel_count(), 17383);
    }

    #[test]
    fn translation_moves_surface() {
        let mut vel = Grid::<Vec3<f32>>::new(Vec3::new(1.0, 0.0, 0.0));
        vel.set_transform(crate::math::Transform::linear(0.1));
        let field: DiscreteField<'_, f32> = DiscreteField::new(&vel);
        let mut grid = make_sphere::<f32>(1.0, Vec3d::zero(), 0.05, 3.0);
        let mut adv = LevelSetAdvection::new(&mut grid, &field).unwrap();
        assert!(adv.advect(0.0, 0.25) >= 1);
        // The surface point (1, 0, 0) moved to (1.25, 0, 0).
        assert_close(grid.get_value(&Coord::new(25, 0, 0)) as f64, 0.0, 5e-3);
        assert_close(grid.get_value(&Coord::new(-15, 0, 0)) as f64, 0.0, 5e-3);
        let mut adv = LevelSetAdvection::new(&mut grid, &field).unwrap();
        adv.advect(0.25, 0.0);
        assert_close(grid.get_value(&Coord::new(20, 0, 0)) as f64, 0.0, 5e-3);
    }
}
