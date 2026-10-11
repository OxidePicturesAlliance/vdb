//! Velocity fields for advection (port of `openvdb/tools/VelocityFields.h`).
//!
//! * [`VelocityField`] — the interface used by
//!   [`LevelSetAdvection`](super::level_set_advect::LevelSetAdvection): sample a velocity at a
//!   world position or at a voxel of the field's own index space.
//! * [`DiscreteField`] — a velocity grid interpolated with a [`Sampler`].
//! * [`EnrightField`] — the analytic, divergence-free Enright test field.
//! * [`VelocitySampler`] and [`VelocityIntegrator`] — world-space sampling and Runge-Kutta
//!   integration of points through a velocity grid (used by
//!   [`VolumeAdvection`](super::volume_advect::VolumeAdvection)).

use super::interpolation::{BoxSampler, Sampler};
use crate::grid::Grid;
use crate::math::{Coord, Transform, Vec3, Vec3d};
use crate::tree::ValueSource;
use crate::types::{RealValue, Value};
use std::marker::PhantomData;

/// A (possibly time-dependent) velocity field.
pub trait VelocityField<T: RealValue>: Sync {
    /// The transform of the field's index space (identity for analytic fields).
    fn transform(&self) -> Transform;
    /// The velocity at world position `xyz` and `time`.
    fn sample_world(&self, xyz: Vec3d, time: T) -> Vec3<T>;
    /// The velocity at voxel `ijk` of the field's index space.
    fn sample_index(&self, ijk: &Coord, time: T) -> Vec3<T>;
}

/// A velocity grid interpolated in world space (`tools::DiscreteField`).
pub struct DiscreteField<'a, T: RealValue, Sm = BoxSampler> {
    grid: &'a Grid<Vec3<T>>,
    _sampler: PhantomData<Sm>,
}

impl<'a, T: RealValue, Sm> DiscreteField<'a, T, Sm> {
    /// Wrap a velocity grid.
    pub fn new(grid: &'a Grid<Vec3<T>>) -> Self
    where
        Vec3<T>: Value,
    {
        DiscreteField { grid, _sampler: PhantomData }
    }
}

impl<T, Sm> VelocityField<T> for DiscreteField<'_, T, Sm>
where
    T: RealValue,
    Vec3<T>: Value,
    Sm: Sampler<Vec3<T>> + Sync,
{
    fn transform(&self) -> Transform {
        self.grid.transform().clone()
    }
    #[inline]
    fn sample_world(&self, xyz: Vec3d, _time: T) -> Vec3<T> {
        Sm::sample(self.grid.tree(), self.grid.transform().world_to_index(xyz))
    }
    #[inline]
    fn sample_index(&self, ijk: &Coord, _time: T) -> Vec3<T> {
        self.grid.tree().get_value(ijk)
    }
}

/// The analytic Enright test velocity field on the unit cube with period 3
/// (`tools::EnrightField`).
#[derive(Clone, Copy, Debug, Default)]
pub struct EnrightField<T = f32>(PhantomData<T>);

impl<T: RealValue> EnrightField<T> {
    /// The field.
    pub fn new() -> Self {
        EnrightField(PhantomData)
    }
    /// The velocity at world position `xyz` and `time`.
    pub fn eval(&self, xyz: Vec3d, time: T) -> Vec3<T> {
        let pi = T::PI();
        let phase = pi / T::from_f64(3.0);
        let px = pi * T::from_f64(xyz[0]);
        let py = pi * T::from_f64(xyz[1]);
        let pz = pi * T::from_f64(xyz[2]);
        let two = T::from_f64(2.0);
        let tr = (time * phase).cos();
        let a = (two * py).sin();
        let b = -(two * px).sin();
        let c = (two * pz).sin();
        let p2 = |x: T| x * x;
        Vec3::new(tr * (two * p2(px.sin()) * a * c), tr * (b * p2(py.sin()) * c), tr * (b * a * p2(pz.sin())))
    }
}

impl<T: RealValue> VelocityField<T> for EnrightField<T> {
    fn transform(&self) -> Transform {
        Transform::identity()
    }
    fn sample_world(&self, xyz: Vec3d, time: T) -> Vec3<T> {
        self.eval(xyz, time)
    }
    fn sample_index(&self, ijk: &Coord, time: T) -> Vec3<T> {
        self.eval(ijk.as_vec3d(), time)
    }
}

/// Samples a velocity grid at world positions (`tools::VelocitySampler`). The sampler type
/// selects the interpolation order and whether the grid is staggered.
pub struct VelocitySampler<'a, S, Sm = BoxSampler> {
    source: S,
    transform: &'a Transform,
    _sampler: PhantomData<Sm>,
}

impl<'a, S: ValueSource, Sm: Sampler<S::Value>> VelocitySampler<'a, S, Sm> {
    /// Sample `source` (tree, grid or accessor) whose index space is mapped by `transform`.
    pub fn new(source: S, transform: &'a Transform) -> Self {
        VelocitySampler { source, transform, _sampler: PhantomData }
    }
    /// The velocity at world position `world` and whether any contributing voxel is active.
    #[inline]
    pub fn probe(&self, world: Vec3d) -> (S::Value, bool) {
        Sm::probe_sample(&self.source, self.transform.world_to_index(world))
    }
    /// The velocity at world position `world`.
    #[inline]
    pub fn sample(&self, world: Vec3d) -> S::Value {
        Sm::sample(&self.source, self.transform.world_to_index(world))
    }
}

/// Runge-Kutta integration of world-space points through a velocity grid
/// (`tools::VelocityIntegrator`).
pub struct VelocityIntegrator<'a, S, Sm = BoxSampler> {
    sampler: VelocitySampler<'a, S, Sm>,
}

impl<'a, T, S, Sm> VelocityIntegrator<'a, S, Sm>
where
    T: RealValue,
    Vec3<T>: Value,
    S: ValueSource<Value = Vec3<T>>,
    Sm: Sampler<Vec3<T>>,
{
    /// Integrate through `source` whose index space is mapped by `transform`.
    pub fn new(source: S, transform: &'a Transform) -> Self {
        VelocityIntegrator { sampler: VelocitySampler::new(source, transform) }
    }
    /// Advance `world` by one step of size `dt` with a Runge-Kutta scheme of order `order`
    /// (0 to 4; 0 does nothing).
    pub fn runge_kutta(&self, order: usize, dt: T, world: &mut Vec3d) {
        let p = world.map(T::from_f64);
        let s = |q: Vec3<T>| self.sampler.probe(q.map(|x| x.to_f64())).0;
        let half = T::from_f64(0.5);
        let delta = match order {
            0 => return,
            1 => s(p) * dt,
            2 => {
                let v0 = s(p);
                s(p + v0 * (half * dt)) * dt
            }
            3 => {
                let v0 = s(p);
                let v1 = s(p + v0 * (half * dt));
                let v2 = s(p + (v1 * T::from_f64(2.0) - v0) * dt);
                (v0 + v1 * T::from_f64(4.0) + v2) * dt * T::from_f64(1.0 / 6.0)
            }
            4 => {
                let v0 = s(p);
                let v1 = s(p + v0 * (half * dt));
                let v2 = s(p + v1 * (half * dt));
                let v3 = s(p + v2 * dt);
                (v0 + (v1 + v2) * T::from_f64(2.0) + v3) * dt * T::from_f64(1.0 / 6.0)
            }
            _ => panic!("Runge-Kutta order must be at most 4"),
        };
        *world += delta.map(|x| x.to_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::CoordBBox;

    #[test]
    fn enright_matches_openvdb() {
        // Values from the C++ EnrightField<float>.
        let f = EnrightField::<f32>::new();
        let v = f.eval(Vec3d::new(0.3, 0.6, 0.1), 0.5);
        let e = [-0.391663909f32, -0.437893242, 0.0462296456];
        for i in 0..3 {
            assert!((v[i] - e[i]).abs() < 1e-6, "{v:?}");
        }
    }

    #[test]
    fn integrator_constant_field() {
        let mut g = Grid::<Vec3<f32>>::new(Vec3::new(1.0, 0.0, -2.0));
        g.tree_mut().fill(&CoordBBox::from_coords(-50, -50, -50, 50, 50, 50), Vec3::new(1.0, 0.0, -2.0), true);
        let field: DiscreteField<'_, f32> = DiscreteField::new(&g);
        assert_eq!(field.sample_world(Vec3d::new(0.3, 0.2, 0.1), 0.0), Vec3::new(1.0, 0.0, -2.0));
        for order in 1..=4 {
            let integ: VelocityIntegrator<'_, _, BoxSampler> = VelocityIntegrator::new(g.tree(), g.transform());
            let mut p = Vec3d::new(0.5, 0.5, 0.5);
            integ.runge_kutta(order, 0.25, &mut p);
            assert!((p - Vec3d::new(0.75, 0.5, 0.0)).length() < 1e-6);
        }
    }
}
