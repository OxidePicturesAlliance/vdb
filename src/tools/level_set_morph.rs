//! Morphing of one level set into another (port of `openvdb/tools/LevelSetMorph.h`).
//!
//! [`LevelSetMorphing`] advects the source level set with the speed `φ_source - φ_target` along
//! its normal, so that its zero crossing moves towards that of the target. An optional alpha
//! mask (a scalar grid remapped with a smooth step) scales the speed locally.

use super::interpolation::{smooth_unit_step, BoxSampler, Sampler};
use super::level_set_tracker::{tvd_rk_step, LevelSetError, LevelSetTracker};
use crate::grid::Grid;
use crate::math::finite_difference::{BiasedGradientScheme, TemporalIntegrationScheme};
use crate::math::operators::{gradient_norm_sqrd, MapJacobian};
use crate::tree::{LeafNode, Node};
use crate::types::{ArithValue, RealValue, Value};
use rayon::prelude::*;

/// Morphs a source level set into a target level set (`tools::LevelSetMorphing`).
pub struct LevelSetMorphing<'a, 'b, T: RealValue> {
    tracker: LevelSetTracker<'a, T>,
    target: &'b Grid<T>,
    mask: Option<&'b Grid<T>>,
    spatial: BiasedGradientScheme,
    temporal: TemporalIntegrationScheme,
    min_mask: T,
    delta_mask: T,
    invert_mask: bool,
}

impl<'a, 'b, T: RealValue + ArithValue> LevelSetMorphing<'a, 'b, T> {
    /// Morph `source` (modified in place) towards `target`.
    pub fn new(source: &'a mut Grid<T>, target: &'b Grid<T>) -> Result<Self, LevelSetError> {
        if !source.transform().is_linear() {
            return Err(LevelSetError::UnsupportedTransform("level set morphing requires a linear transform"));
        }
        Ok(LevelSetMorphing {
            tracker: LevelSetTracker::new(source)?,
            target,
            mask: None,
            spatial: BiasedGradientScheme::HjWeno5Bias,
            temporal: TemporalIntegrationScheme::TvdRk2,
            min_mask: <T as Value>::zero(),
            delta_mask: T::from_f64(1.0),
            invert_mask: false,
        })
    }
    /// The source level set.
    pub fn grid(&self) -> &Grid<T> {
        self.tracker.grid()
    }
    /// The tracker used to maintain the narrow band.
    pub fn tracker(&mut self) -> &mut LevelSetTracker<'a, T> {
        &mut self.tracker
    }
    /// Replace the target level set.
    pub fn set_target(&mut self, target: &'b Grid<T>) {
        self.target = target;
    }
    /// Scale the speed by a smooth step of the values of `mask`.
    pub fn set_alpha_mask(&mut self, mask: &'b Grid<T>) {
        self.mask = Some(mask);
    }
    /// The spatial scheme (only `FirstBias` and `HjWeno5Bias` are supported, as in OpenVDB).
    pub fn spatial_scheme(&self) -> BiasedGradientScheme {
        self.spatial
    }
    /// Set the spatial scheme.
    pub fn set_spatial_scheme(&mut self, s: BiasedGradientScheme) {
        self.spatial = s;
    }
    /// The temporal scheme.
    pub fn temporal_scheme(&self) -> TemporalIntegrationScheme {
        self.temporal
    }
    /// Set the temporal scheme.
    pub fn set_temporal_scheme(&mut self, s: TemporalIntegrationScheme) {
        self.temporal = s;
    }
    /// Set the tracker's spatial scheme.
    pub fn set_tracker_spatial_scheme(&mut self, s: BiasedGradientScheme) {
        self.tracker.set_spatial_scheme(s);
    }
    /// Set the tracker's temporal scheme.
    pub fn set_tracker_temporal_scheme(&mut self, s: TemporalIntegrationScheme) {
        self.tracker.set_temporal_scheme(s);
    }
    /// Set the number of normalizations per tracking step.
    pub fn set_norm_count(&mut self, n: usize) {
        self.tracker.set_norm_count(n);
    }
    /// The mask value mapped to zero speed.
    pub fn min_mask(&self) -> T {
        self.min_mask
    }
    /// The mask value mapped to full speed.
    pub fn max_mask(&self) -> T {
        self.min_mask + self.delta_mask
    }
    /// Set the mask range.
    ///
    /// # Panics
    /// If `min >= max`.
    pub fn set_mask_range(&mut self, min: T, max: T) {
        assert!(min < max, "Invalid mask range (expects min < max)");
        self.min_mask = min;
        self.delta_mask = max - min;
    }
    /// True if the mask is inverted.
    pub fn is_mask_inverted(&self) -> bool {
        self.invert_mask
    }
    /// Invert the mask.
    pub fn invert_mask(&mut self, invert: bool) {
        self.invert_mask = invert;
    }

    /// Per-leaf speeds (indexed by voxel offset) and per-leaf "all zero" flags, plus the
    /// largest absolute speed.
    fn sample_speed(&self) -> (Vec<Vec<T>>, Vec<bool>, T) {
        let grid = self.tracker.grid();
        let xform = grid.transform();
        let aligned = self.target.transform() == xform && self.mask.map(|m| m.transform() == xform).unwrap_or(true);
        let target = self.target;
        let mask = self.mask;
        let (min, inv_norm, invert) = (self.min_mask, T::from_f64(1.0) / self.delta_mask, self.invert_mask);
        let res: Vec<(Vec<T>, bool, T)> = grid
            .tree()
            .leaf_refs()
            .par_iter()
            .map_init(
                || (target.accessor(), mask.map(|m| m.accessor())),
                |(tacc, macc), leaf: &&LeafNode<T>| {
                    let mut speed = leaf.buffer().to_vec();
                    let mut is_zero = true;
                    let mut max_s = <T as Value>::zero();
                    let origin = leaf.origin();
                    for n in leaf.value_mask().iter_on() {
                        let ijk = origin + LeafNode::<T>::offset_to_local_coord(n);
                        let (tv, mv) = if aligned {
                            (tacc.get_value(&ijk), macc.as_ref().map(|m| m.get_value(&ijk)))
                        } else {
                            let w = xform.index_to_world(ijk.as_vec3d());
                            (
                                BoxSampler::sample(&*tacc, target.transform().world_to_index(w)),
                                macc.as_ref()
                                    .map(|m| BoxSampler::sample(m, mask.unwrap().transform().world_to_index(w))),
                            )
                        };
                        let mut s = speed[n] - tv;
                        if let Some(mv) = mv {
                            let a = T::from_f64(smooth_unit_step(((mv - min) * inv_norm).to_f64()));
                            s *= if invert { T::from_f64(1.0) - a } else { a };
                        }
                        speed[n] = s;
                        if s.abs() > T::tolerance() {
                            is_zero = false;
                        }
                        max_s = max_s.max(s.abs());
                    }
                    (speed, is_zero, max_s)
                },
            )
            .collect();
        let mut max_abs = T::from_f64(1e-6);
        let mut speeds = Vec::with_capacity(res.len());
        let mut zero = Vec::with_capacity(res.len());
        for (s, z, m) in res {
            max_abs = max_abs.max(m);
            speeds.push(s);
            zero.push(z);
        }
        (speeds, zero, max_abs)
    }

    /// Morph from `time0` to `time1`; returns the number of CFL-limited steps taken.
    pub fn advect(&mut self, mut time0: T, time1: T) -> Result<usize, LevelSetError> {
        let spatial = self.spatial;
        if !matches!(spatial, BiasedGradientScheme::FirstBias | BiasedGradientScheme::HjWeno5Bias) {
            return Err(LevelSetError::UnsupportedTransform("spatial difference scheme not supported for morphing"));
        }
        let mut count = 0;
        while time0 < time1 {
            if self.tracker.grid().tree().leaf_count() == 0 {
                break;
            }
            let (speed, zero, max_abs) = self.sample_speed();
            let min_abs = T::from_f64(1e-6);
            if (max_abs - min_abs).abs() <= T::tolerance() {
                break;
            }
            let cfl = match self.temporal {
                TemporalIntegrationScheme::TvdRk1 => 0.3,
                TemporalIntegrationScheme::TvdRk2 => 0.9,
                TemporalIntegrationScheme::TvdRk3 => 1.0,
            };
            let cfl = T::from_f64(cfl) / T::from_f64(3.0).sqrt();
            let dx = self.tracker.voxel_size();
            let dt = (time1 - time0).abs().min(cfl * dx / max_abs);
            if dt == <T as Value>::zero() {
                break;
            }
            let map = MapJacobian::new(self.tracker.grid().transform());
            let tol = T::tolerance();
            tvd_rk_step(self.tracker.grid_mut(), self.temporal, spatial.stencil_kind(), None, |st, leaf, n| {
                if zero[leaf] {
                    return None;
                }
                let s = speed[leaf][n];
                if s.abs() <= tol {
                    return None;
                }
                Some(st.center_value() - dt * s * gradient_norm_sqrd(&map, spatial, st))
            });
            time0 += dt;
            count += 1;
            self.tracker.track();
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, Vec3d};
    use crate::tools::test_util::{assert_close, make_sphere};

    /// Reference values from `scratchpad/cppref/morph.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let mut src = make_sphere::<f32>(1.0, Vec3d::new(0.1, -0.05, 0.02), 0.05, 3.0);
        let dst = make_sphere::<f32>(0.8, Vec3d::new(0.25, 0.0, 0.0), 0.05, 3.0);
        let mut m = LevelSetMorphing::new(&mut src, &dst).unwrap();
        let steps = m.advect(0.0, 0.5).unwrap();
        assert_eq!(steps, 6);
        let pts = [Coord::new(20, 3, -2), Coord::new(-5, 17, 6), Coord::new(21, 0, 0), Coord::new(0, 0, 16)];
        let expect = [-0.0533356443, 0.0739268884, -0.028767664, -0.131487578];
        for (p, e) in pts.iter().zip(expect.iter()) {
            assert_close(src.get_value(p) as f64, *e, 2e-6);
        }
        assert_eq!(src.active_voxel_count(), 28466);
    }
}
