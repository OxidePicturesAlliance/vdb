//! Semi-Lagrangian advection of scalar and vector volumes in a velocity grid (port of
//! `openvdb/tools/VolumeAdvect.h`).
//!
//! [`VolumeAdvection`] back-traces the characteristics of every (dilated) active voxel through
//! the velocity field with a Runge-Kutta integrator and samples the input volume at the
//! departure point. Second-order MacCormack and BFECC schemes with clamping or reverting
//! limiters are supported, as are sub-stepping and masks.

use super::internal::dilate_face;
use super::interpolation::{BoxSampler, Sampler};
use super::velocity_fields::VelocityIntegrator;
use crate::grid::Grid;
use crate::math::{Coord, Vec3, Vec3d};
use crate::tree::{LeafNode, Node};
use crate::types::{ArithValue, RealValue, Value};
use rayon::prelude::*;
use std::marker::PhantomData;

/// The integration scheme (`Scheme::SemiLagrangian`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SemiLagrangian {
    /// First-order semi-Lagrangian (forward Euler back-tracing).
    #[default]
    Semi,
    /// Second-order Runge-Kutta (mid-point) back-tracing.
    Mid,
    /// Third-order Runge-Kutta back-tracing.
    Rk3,
    /// Fourth-order Runge-Kutta back-tracing.
    Rk4,
    /// MacCormack's second-order scheme.
    Mac,
    /// Back and Forth Error Compensation and Correction (second order).
    Bfecc,
}

/// Limiter applied to the second-order schemes (`Scheme::Limiter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Limiter {
    /// No limiting.
    NoLimiter,
    /// Clamp to the range of the values around the departure point.
    #[default]
    Clamp,
    /// Revert to first-order trilinear interpolation where the value is out of range.
    Revert,
}

/// Value types that can be advected: scalars and 3-vectors of `f32`/`f64`.
pub trait AdvectValue: ArithValue {
    /// OpenVDB's `math::cwiseLessThan` (all components less).
    fn cwise_less(&self, o: &Self) -> bool;
    /// OpenVDB's `math::cwiseGreaterThan` (all components greater).
    fn cwise_greater(&self, o: &Self) -> bool;
    /// Component-wise clamp.
    fn clamp_cw(self, lo: Self, hi: Self) -> Self;
    /// `math::Delta` of the component type, splatted.
    fn delta_value() -> Self;
}

macro_rules! advect_scalar {
    ($($t:ty),*) => {$(
        impl AdvectValue for $t {
            fn cwise_less(&self, o: &Self) -> bool { self < o }
            fn cwise_greater(&self, o: &Self) -> bool { self > o }
            fn clamp_cw(self, lo: Self, hi: Self) -> Self { if self < lo { lo } else if self > hi { hi } else { self } }
            fn delta_value() -> Self { <$t as RealValue>::delta() }
        }
        impl AdvectValue for Vec3<$t> {
            fn cwise_less(&self, o: &Self) -> bool { (0..3).all(|i| self[i] < o[i]) }
            fn cwise_greater(&self, o: &Self) -> bool { (0..3).all(|i| self[i] > o[i]) }
            fn clamp_cw(self, lo: Self, hi: Self) -> Self {
                Vec3::new(self[0].clamp_cw(lo[0], hi[0]), self[1].clamp_cw(lo[1], hi[1]), self[2].clamp_cw(lo[2], hi[2]))
            }
            fn delta_value() -> Self { Vec3::splat(<$t as RealValue>::delta()) }
        }
    )*};
}
advect_scalar!(f32, f64);

/// Advects volumes in a velocity grid (`tools::VolumeAdvection`). The velocity sampler `Sm`
/// selects collocated ([`BoxSampler`]) or staggered
/// ([`StaggeredBoxSampler`](super::interpolation::StaggeredBoxSampler)) velocities.
pub struct VolumeAdvection<'a, V: RealValue, Sm = BoxSampler> {
    vel: &'a Grid<Vec3<V>>,
    max_velocity: f64,
    integrator: SemiLagrangian,
    limiter: Limiter,
    sub_steps: usize,
    _sampler: PhantomData<Sm>,
}

impl<'a, V, Sm> VolumeAdvection<'a, V, Sm>
where
    V: RealValue,
    Vec3<V>: Value,
    Sm: Sampler<Vec3<V>> + Sync,
{
    /// Advect volumes in the velocity grid `vel`.
    pub fn new(vel: &'a Grid<Vec3<V>>) -> Self {
        let mut max_v = vel.background().length().to_f64();
        for leaf in vel.tree().leaves() {
            for v in leaf.buffer() {
                max_v = max_v.max(v.length().to_f64());
            }
        }
        vel.tree().visit_tiles(&mut |_, v, _, _| max_v = max_v.max(v.length().to_f64()));
        VolumeAdvection {
            vel,
            max_velocity: max_v,
            integrator: SemiLagrangian::Semi,
            limiter: Limiter::Clamp,
            sub_steps: 1,
            _sampler: PhantomData,
        }
    }
    /// Spatial order of the scheme (2 for MacCormack and BFECC).
    pub fn spatial_order(&self) -> usize {
        match self.integrator {
            SemiLagrangian::Mac | SemiLagrangian::Bfecc => 2,
            _ => 1,
        }
    }
    /// Temporal order of the scheme.
    pub fn temporal_order(&self) -> usize {
        match self.integrator {
            SemiLagrangian::Semi => 1,
            SemiLagrangian::Mid | SemiLagrangian::Bfecc | SemiLagrangian::Mac => 2,
            SemiLagrangian::Rk3 => 3,
            SemiLagrangian::Rk4 => 4,
        }
    }
    /// Set the integration scheme.
    pub fn set_integrator(&mut self, s: SemiLagrangian) {
        self.integrator = s;
    }
    /// The integration scheme.
    pub fn integrator(&self) -> SemiLagrangian {
        self.integrator
    }
    /// Set the limiter.
    pub fn set_limiter(&mut self, l: Limiter) {
        self.limiter = l;
    }
    /// The limiter.
    pub fn limiter(&self) -> Limiter {
        self.limiter
    }
    /// True if a limiter is applied (second-order scheme and a limiter other than none).
    pub fn is_limiter_on(&self) -> bool {
        self.spatial_order() > 1 && self.limiter != Limiter::NoLimiter
    }
    /// Number of sub-steps per call to `advect`.
    pub fn sub_steps(&self) -> usize {
        self.sub_steps
    }
    /// Set the number of sub-steps (at least one).
    pub fn set_sub_steps(&mut self, n: usize) {
        self.sub_steps = n.max(1);
    }
    /// The largest velocity magnitude in the velocity grid.
    pub fn max_velocity(&self) -> f64 {
        self.max_velocity
    }
    /// The largest distance, in voxels of `grid`, that values can travel in time `dt`.
    pub fn max_distance<T: Value>(&self, grid: &Grid<T>, dt: f64) -> usize {
        (self.max_velocity * dt.abs() / grid.voxel_size()[0]).ceil() as usize
    }

    /// Advect `grid` by `time_step`, sampling it with `S`; returns the advected grid.
    pub fn advect<T, S>(&self, grid: &Grid<T>, time_step: f64) -> Grid<T>
    where
        T: AdvectValue,
        S: Sampler<T> + Sync,
    {
        self.advect_impl::<T, S, bool>(grid, None, time_step)
    }

    /// Advect `grid` by `time_step`, restricting the output to the active voxels of `mask`
    /// (which must share the grid's transform) plus the input's active voxels.
    ///
    /// # Panics
    /// If the transforms of `grid` and `mask` differ.
    pub fn advect_masked<T, S, M>(&self, grid: &Grid<T>, mask: &Grid<M>, time_step: f64) -> Grid<T>
    where
        T: AdvectValue,
        S: Sampler<T> + Sync,
        M: Value,
    {
        assert!(
            grid.transform() == mask.transform(),
            "Volume grid and mask grid are misaligned! Consider resampling either of the two grids into the index \
             space of the other."
        );
        self.advect_impl::<T, S, M>(grid, Some(mask), time_step)
    }

    fn advect_impl<T, S, M>(&self, grid: &Grid<T>, mask: Option<&Grid<M>>, time_step: f64) -> Grid<T>
    where
        T: AdvectValue,
        S: Sampler<T> + Sync,
        M: Value,
    {
        let dt = time_step / self.sub_steps as f64;
        let n = self.max_distance(grid, dt);
        let mut out = grid.clone();
        for step in 0..self.sub_steps {
            let src = if step == 0 { grid.clone() } else { out.clone() };
            let mut tmp = src.clone();
            tmp.tree_mut().voxelize_active_tiles();
            dilate_face(tmp.tree_mut(), n);
            if let Some(m) = mask {
                intersect_with_mask(&mut tmp, m);
                tmp.tree_mut().prune_inactive();
            }
            self.cook::<T, S>(&mut tmp, &src, dt);
            if mask.is_some() {
                tmp.tree_mut().topology_union(grid.tree());
            }
            out = tmp;
        }
        out
    }

    /// Back-trace every active voxel of `out` by `dt` and sample `src` there: returns new
    /// per-leaf buffers of `out`.
    fn rk_pass<T, S>(&self, out: &Grid<T>, src: &Grid<T>, xform_grid: &Grid<T>, dt: V, order: usize) -> Vec<Vec<T>>
    where
        T: AdvectValue,
        S: Sampler<T> + Sync,
    {
        let xform = xform_grid.transform();
        let vel = self.vel;
        out.tree()
            .leaf_refs()
            .par_iter()
            .map_init(
                || (src.accessor(), VelocityIntegrator::<_, Sm>::new(vel.accessor(), vel.transform())),
                |(acc, integ), leaf: &&LeafNode<T>| {
                    let mut buf = leaf.buffer().to_vec();
                    let origin = leaf.origin();
                    for n in leaf.value_mask().iter_on() {
                        let ijk = origin + LeafNode::<T>::offset_to_local_coord(n);
                        let mut w = xform.index_to_world(ijk.as_vec3d());
                        integ.runge_kutta(order, dt, &mut w);
                        buf[n] = S::sample(&*acc, xform.world_to_index(w));
                    }
                    buf
                },
            )
            .collect()
    }

    fn cook<T, S>(&self, out: &mut Grid<T>, src: &Grid<T>, time_step: f64)
    where
        T: AdvectValue,
        S: Sampler<T> + Sync,
    {
        let dt = V::from_f64(-time_step);
        let order = match self.integrator {
            SemiLagrangian::Semi | SemiLagrangian::Mac | SemiLagrangian::Bfecc => 1,
            SemiLagrangian::Mid => 2,
            SemiLagrangian::Rk3 => 3,
            SemiLagrangian::Rk4 => 4,
        };
        let write = |g: &mut Grid<T>, bufs: &[Vec<T>]| super::internal::write_leaf_buffers(g.tree_mut(), bufs);
        match self.integrator {
            SemiLagrangian::Mac | SemiLagrangian::Bfecc => {
                let fwd = self.rk_pass::<T, S>(out, src, src, dt, order);
                write(out, &fwd);
                let bwd = self.rk_pass::<T, S>(out, out, src, -dt, order);
                // Combine with the input values (from the input leaf if present).
                let mac = self.integrator == SemiLagrangian::Mac;
                let combined: Vec<Vec<T>> = {
                    let o: &Grid<T> = out;
                    o.tree()
                        .leaf_refs()
                        .par_iter()
                        .zip(bwd.par_iter())
                        .map_init(
                            || src.accessor(),
                            |acc, (leaf, b)| {
                                let mut buf = leaf.buffer().to_vec();
                                let origin = leaf.origin();
                                for n in leaf.value_mask().iter_on() {
                                    let ijk = origin + LeafNode::<T>::offset_to_local_coord(n);
                                    let inv = acc.get_value(&ijk);
                                    buf[n] = if mac {
                                        buf[n] + (inv - b[n]).scale(0.5)
                                    } else {
                                        (inv.scale(3.0) - b[n]).scale(0.5)
                                    };
                                }
                                buf
                            },
                        )
                        .collect()
                };
                write(out, &combined);
                if !mac {
                    let fwd2 = self.rk_pass::<T, S>(out, out, src, dt, order);
                    write(out, &fwd2);
                }
            }
            _ => {
                let fwd = self.rk_pass::<T, S>(out, src, src, dt, order);
                write(out, &fwd);
            }
        }
        self.limit(out, src, dt);
        out.tree_mut().prune_inactive();
    }

    fn limit<T: AdvectValue>(&self, out: &mut Grid<T>, src: &Grid<T>, dt: V) {
        let do_limiter = self.is_limiter_on();
        let do_clamp = self.limiter == Limiter::Clamp;
        let xform = src.transform().clone();
        let bg = *src.background();
        let delta = T::delta_value();
        let vel = self.vel;
        let results: Vec<(Vec<T>, Vec<usize>)> = {
            let o: &Grid<T> = out;
            o.tree()
                .leaf_refs()
                .par_iter()
                .map_init(
                    || (src.accessor(), VelocityIntegrator::<_, Sm>::new(vel.accessor(), vel.transform())),
                    |(acc, integ), leaf| {
                        let mut buf = leaf.buffer().to_vec();
                        let mut off = Vec::new();
                        let origin = leaf.origin();
                        for n in leaf.value_mask().iter_on() {
                            let ijk = origin + LeafNode::<T>::offset_to_local_coord(n);
                            let mut value = buf[n];
                            if do_limiter {
                                let mut w = xform.index_to_world(ijk.as_vec3d());
                                integ.runge_kutta(1, dt, &mut w);
                                let ipos = xform.world_to_index(w);
                                let c = Coord::floor(ipos);
                                let data = BoxSampler::get_values(&*acc, c);
                                let (lo, hi) = cwise_extrema(&data);
                                if do_clamp {
                                    value = value.clamp_cw(lo, hi);
                                } else if value.cwise_less(&lo) || value.cwise_greater(&hi) {
                                    value = BoxSampler::trilinear_interpolation(&data, ipos - c.as_vec3d());
                                }
                            }
                            if value.approx_eq(&bg, &delta) {
                                value = bg;
                                off.push(n);
                            }
                            buf[n] = value;
                        }
                        (buf, off)
                    },
                )
                .collect()
        };
        out.tree_mut().leaf_refs_mut().into_par_iter().zip(results.into_par_iter()).for_each(|(leaf, (buf, off))| {
            leaf.buffer_mut().copy_from_slice(&buf);
            for n in off {
                leaf.set_active_state_at(n, false);
            }
        });
    }
}

/// `BoxSampler::extrema` with OpenVDB's `cwiseLessThan` comparisons.
fn cwise_extrema<T: AdvectValue>(data: &[[[T; 2]; 2]; 2]) -> (T, T) {
    let mut lo = data[0][0][0];
    let mut hi = lo;
    for v in [data[0][0][1], data[0][1][0], data[0][1][1], data[1][0][0], data[1][0][1], data[1][1][0], data[1][1][1]] {
        if !lo.cwise_less(&v) {
            lo = v;
        }
        if hi.cwise_less(&v) {
            hi = v;
        }
    }
    (lo, hi)
}

fn intersect_with_mask<T: Value, M: Value>(grid: &mut Grid<T>, mask: &Grid<M>) {
    let mtree = mask.tree();
    grid.tree_mut().leaf_refs_mut().into_par_iter().for_each_init(
        || mtree.accessor(),
        |acc, leaf| {
            let origin = leaf.origin();
            let on: Vec<usize> = leaf.value_mask().iter_on().collect();
            for n in on {
                if !acc.is_value_on(&(origin + LeafNode::<T>::offset_to_local_coord(n))) {
                    leaf.set_active_state_at(n, false);
                }
            }
        },
    );
}

/// Convenience: advect a world-space point cloud position through `vel` with the given
/// Runge-Kutta order (used by tests and examples).
pub fn advect_point<V: RealValue + ArithValue>(vel: &Grid<Vec3<V>>, order: usize, dt: V, p: Vec3d) -> Vec3d
where
    Vec3<V>: Value,
{
    let integ: VelocityIntegrator<'_, _, BoxSampler> = VelocityIntegrator::new(vel.tree(), vel.transform());
    let mut w = p;
    integ.runge_kutta(order, dt, &mut w);
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{CoordBBox, Transform};
    use crate::tools::test_util::assert_close;

    fn density() -> Grid<f32> {
        let mut g = Grid::<f32>::new(0.0).with_transform(Transform::linear(0.1));
        for c in CoordBBox::from_coords(-6, -6, -6, 6, 6, 6) {
            let r = c.as_vec3d().length();
            if r < 5.0 {
                g.tree_mut().set_value(&c, (1.0 - r / 5.0) as f32);
            }
        }
        g
    }

    fn velocity() -> Grid<Vec3<f32>> {
        let mut v = Grid::<Vec3<f32>>::new(Vec3::zero()).with_transform(Transform::linear(0.1));
        for c in CoordBBox::from_coords(-20, -20, -20, 20, 20, 20) {
            let p = c.as_vec3d() * 0.1;
            v.tree_mut().set_value(&c, Vec3::new(1.0 - 0.3 * p[1] as f32, 0.5 + 0.2 * p[0] as f32, 0.25));
        }
        v
    }

    /// Reference values from `scratchpad/cppref/volume_advect.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let vel = velocity();
        let den = density();
        let pts = [Coord::new(2, 1, 0), Coord::new(5, 3, 2), Coord::new(-1, 0, 1), Coord::new(8, 4, 3)];
        let cases = [
            (SemiLagrangian::Semi, Limiter::Clamp, 1, [0.703119338, 0.417528391, 0.138887525, 0.0], 709),
            (SemiLagrangian::Rk4, Limiter::Clamp, 2, [0.668256462, 0.406748682, 0.136055082, 0.0], 995),
            (SemiLagrangian::Mac, Limiter::Clamp, 1, [0.725429177, 0.417078793, 0.15751335, 0.0], 686),
            (SemiLagrangian::Bfecc, Limiter::Revert, 1, [0.727429569, 0.435099512, 0.141917452, 0.0], 709),
        ];
        for (scheme, lim, sub, expect, count) in cases {
            let mut adv: VolumeAdvection<'_, f32> = VolumeAdvection::new(&vel);
            assert_close(adv.max_velocity(), 1.85270071, 1e-6);
            adv.set_integrator(scheme);
            adv.set_limiter(lim);
            adv.set_sub_steps(sub);
            let out = adv.advect::<f32, BoxSampler>(&den, 0.3);
            for (p, e) in pts.iter().zip(expect.iter()) {
                assert_close(out.get_value(p) as f64, *e, 2e-6);
            }
            assert_eq!(out.active_voxel_count(), count, "{scheme:?}");
        }
    }

    #[test]
    fn constant_velocity_translates() {
        let mut vel = Grid::<Vec3<f32>>::new(Vec3::new(1.0, 0.0, 0.0));
        vel.set_transform(Transform::linear(0.1));
        let den = density();
        let adv: VolumeAdvection<'_, f32> = VolumeAdvection::new(&vel);
        assert_eq!(adv.max_velocity(), 1.0);
        assert_eq!(adv.max_distance(&den, 0.2), 2);
        let out = adv.advect::<f32, BoxSampler>(&den, 0.2);
        // Moved by two voxels in +x.
        assert_close(out.get_value(&Coord::new(3, 1, 0)) as f64, den.get_value(&Coord::new(1, 1, 0)) as f64, 1e-6);
        let p = advect_point(&vel, 4, 0.5f32, Vec3d::zero());
        assert_close(p[0], 0.5, 1e-6);
    }
}
