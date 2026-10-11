//! Advecting points through a velocity field (port of `openvdb/points/PointAdvect.h`).

use super::index_filter::{BinaryFilter, BoundFilter, IndexFilter, NullFilter};
use super::index_iterator::IndexState;
use super::point_data_grid::{PointDataGrid, PointDataLeaf, PointLeaf};
use super::point_mask::PointDeformer;
use super::point_move::{move_points, CachedDeformer};
use super::{Error, Result};
use crate::grid::{Grid, GridClass};
use crate::math::{Coord, Vec3, Vec3d};
use crate::tools::interpolation::{BoxSampler, StaggeredBoxSampler};
use crate::tools::velocity_fields::VelocityIntegrator;
use crate::types::{RealValue, Value};

/// Moves points by integrating a velocity grid (`AdvectionDeformer`): `steps` Runge-Kutta
/// steps of order `order` and size `dt`, applied to the points accepted by a filter.
pub struct AdvectionDeformer<'v, T: RealValue, F: IndexFilter> {
    velocity: &'v Grid<Vec3<T>>,
    order: usize,
    dt: T,
    steps: u32,
    filter: &'v F,
    staggered: bool,
    /// Points of the current leaf accepted by the filter (`None` = all).
    valid: Option<Vec<bool>>,
}

impl<T: RealValue, F: IndexFilter> Clone for AdvectionDeformer<'_, T, F> {
    fn clone(&self) -> Self {
        AdvectionDeformer {
            velocity: self.velocity,
            order: self.order,
            dt: self.dt,
            steps: self.steps,
            filter: self.filter,
            staggered: self.staggered,
            valid: self.valid.clone(),
        }
    }
}

impl<'v, T: RealValue, F: IndexFilter> AdvectionDeformer<'v, T, F>
where
    Vec3<T>: Value + crate::types::ArithValue,
{
    /// A deformer integrating `velocity` (staggered if its class is `GridClass::Staggered`).
    pub fn new(velocity: &'v Grid<Vec3<T>>, order: usize, dt: f64, steps: u32, filter: &'v F) -> Self {
        AdvectionDeformer {
            velocity,
            order,
            dt: T::from_f64(dt),
            steps,
            filter,
            staggered: velocity.grid_class() == GridClass::Staggered,
            valid: None,
        }
    }
}

impl<T: RealValue, F: IndexFilter> PointDeformer for AdvectionDeformer<'_, T, F>
where
    Vec3<T>: Value + crate::types::ArithValue,
{
    fn reset(&mut self, leaf: &PointDataLeaf<'_>, _: usize) {
        self.valid = if self.filter.state() == IndexState::All {
            None
        } else {
            let mut bound = self.filter.bind(leaf.leaf, leaf.attributes);
            let mut v = vec![false; leaf.point_count() as usize];
            for n in 0..512 {
                let ijk = leaf.leaf.offset_to_global_coord(n);
                let active = leaf.leaf.is_value_on_at(n);
                for i in leaf.voxel_range(n) {
                    v[i as usize] = bound.valid(i, &ijk, active);
                }
            }
            Some(v)
        };
    }
    fn apply(&self, pos: &mut Vec3d, index: u32, _: &Coord) {
        if let Some(v) = &self.valid {
            if !v.get(index as usize).copied().unwrap_or(false) {
                return;
            }
        }
        let tree = self.velocity.tree();
        let xform = self.velocity.transform();
        if self.staggered {
            let integ: VelocityIntegrator<'_, _, StaggeredBoxSampler> = VelocityIntegrator::new(tree, xform);
            for _ in 0..self.steps {
                integ.runge_kutta(self.order, self.dt, pos);
            }
        } else {
            let integ: VelocityIntegrator<'_, _, BoxSampler> = VelocityIntegrator::new(tree, xform);
            for _ in 0..self.steps {
                integ.runge_kutta(self.order, self.dt, pos);
            }
        }
    }
}

/// Advect the points of `points` through `velocity` with `steps` Runge-Kutta steps of order
/// `order` (1-4) and size `dt` (`advectPoints()`). Only points accepted by `advect_filter`
/// move; points rejected by `filter` are deleted. With `cached`, positions are computed
/// first and the points moved afterwards (as OpenVDB does by default).
#[allow(clippy::too_many_arguments)]
pub fn advect_points<T: RealValue, A: IndexFilter, F: IndexFilter>(
    points: &mut PointDataGrid,
    velocity: &Grid<Vec3<T>>,
    order: usize,
    dt: f64,
    steps: u32,
    advect_filter: &A,
    filter: &F,
    cached: bool,
) -> Result<()>
where
    Vec3<T>: Value + crate::types::ArithValue,
{
    if steps == 0 {
        return Ok(());
    }
    if order > 4 {
        return Err(Error::Value("unknown integration order for advecting points".into()));
    }
    if points.tree().leaf_count() == 0 {
        return Ok(());
    }
    if cached {
        let null = NullFilter;
        let deformer = AdvectionDeformer::new(velocity, order, dt, steps, &null);
        let mut cache = CachedDeformer::new();
        if filter.state() == IndexState::All && advect_filter.state() == IndexState::All {
            cache.evaluate(points, &deformer, &NullFilter);
        } else {
            cache.evaluate(points, &deformer, &BinaryFilter::<&A, &F, true>::new(advect_filter, filter));
        }
        move_points(points, &cache, filter)
    } else {
        let deformer = AdvectionDeformer::new(velocity, order, dt, steps, advect_filter);
        move_points(points, &deformer, filter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{CoordBBox, Transform, Vec3f};
    use crate::points::{create_point_data_grid, point_count, Codec};

    #[test]
    fn advect_constant_velocity() {
        let pos: Vec<Vec3f> = (0..200).map(|i| Vec3f::new((i % 10) as f32 * 0.3, (i / 10) as f32 * 0.2, 0.5)).collect();
        let mut velocity = Grid::<Vec3f>::new(Vec3f::zero());
        velocity.tree_mut().fill(
            &CoordBBox::from_coords(-100, -100, -100, 100, 100, 100),
            Vec3f::new(1.0, -0.5, 0.25),
            true,
        );
        for (order, cached) in [(1, true), (4, false), (2, true)] {
            let mut points = create_point_data_grid(&pos, Codec::Null, &Transform::linear(0.1)).unwrap();
            let before = points.world_positions().unwrap();
            advect_points(&mut points, &velocity, order, 0.5, 4, &NullFilter, &NullFilter, cached).unwrap();
            assert_eq!(point_count(points.tree()), 200);
            let mut after = points.world_positions().unwrap();
            let mut expected: Vec<Vec3d> = before.iter().map(|p| *p + Vec3d::new(2.0, -1.0, 0.5)).collect();
            let key = |p: &Vec3d| ((p[0] * 1000.0).round() as i64, (p[1] * 1000.0).round() as i64);
            after.sort_by_key(key);
            expected.sort_by_key(key);
            for (a, e) in after.iter().zip(&expected) {
                assert!((*a - *e).length() < 1e-4, "{:?} vs {:?}", a, e);
            }
        }
    }
}
