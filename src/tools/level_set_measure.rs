//! Area, volume, curvature and topology (Euler characteristic, genus) of level sets (port
//! of `openvdb/tools/LevelSetMeasure.h`).
//!
//! The measures are computed with a smeared Dirac delta function (three voxels wide)
//! applied to the central-difference gradients of [`GradStencil`] and the curvatures of
//! [`CurvatureStencil`], exactly as in OpenVDB.

use crate::grid::{Grid, GridClass};
use crate::math::stencils::{CurvatureStencil, GradStencil};
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::LeafNode;
use crate::types::RealValue;
use rayon::prelude::*;

/// Smeared Dirac delta of half-width `eps` (`DiracDelta`).
#[derive(Clone, Copy)]
struct DiracDelta {
    c: f64,
    d: f64,
    e: f64,
}

impl DiracDelta {
    fn new(eps: f64) -> Self {
        let c = 0.5 / eps;
        DiracDelta { c, d: 2.0 * std::f64::consts::PI * c, e: eps }
    }
    #[inline]
    fn eval(&self, phi: f64) -> f64 {
        if phi.abs() > self.e {
            0.0
        } else {
            self.c * (1.0 + (self.d * phi).cos())
        }
    }
}

/// Sum per-leaf contributions after sorting them (to mitigate cancellation), as OpenVDB does.
fn sorted_sum(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v.into_iter().sum()
}

/// Measures of a narrow-band level set (`tools::LevelSetMeasure`). Results are cached.
pub struct LevelSetMeasure<'a, T: RealValue + ToolValue> {
    grid: &'a Grid<T>,
    dx: f64,
    area_volume: Option<(f64, f64)>,
    curvature: Option<(f64, f64)>,
}

impl<'a, T: RealValue + ToolValue> LevelSetMeasure<'a, T> {
    /// Returns an error unless `grid` is a non-empty level set with uniform voxels.
    pub fn new(grid: &'a Grid<T>) -> ToolResult<Self> {
        if !grid.transform().has_uniform_scale() {
            return Err(ToolError::Runtime(
                "The transform must have uniform scale for the LevelSetMeasure to function".into(),
            ));
        }
        if grid.grid_class() != GridClass::LevelSet {
            return Err(ToolError::Runtime(
                "LevelSetMeasure only supports level sets; try setting the grid class to \"level set\"".into(),
            ));
        }
        if grid.empty() {
            return Err(ToolError::Runtime("LevelSetMeasure does not support empty grids;".into()));
        }
        Ok(LevelSetMeasure { grid, dx: grid.voxel_size()[0], area_volume: None, curvature: None })
    }

    fn measure_area(&mut self) -> (f64, f64) {
        if let Some(r) = self.area_volume {
            return r;
        }
        let tree = self.grid.tree();
        let dx = self.dx;
        let inv_dx = 1.0 / dx;
        let dd = DiracDelta::new(1.5);
        let grid = self.grid;
        let parts: Vec<(f64, f64)> = tree
            .leaf_refs()
            .par_iter()
            .map_init(
                || GradStencil::with_dx(grid, dx),
                |st, leaf: &&LeafNode<T>| {
                    let (mut sa, mut sv) = (0.0, 0.0);
                    for (pos, v) in leaf.iter_on() {
                        let d = dd.eval(inv_dx * (*v).to_f64());
                        if d > 0.0 {
                            let p = leaf.offset_to_global_coord(pos);
                            st.move_to(&p);
                            let g = st.gradient();
                            let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                            sa += d * len.to_f64();
                            sv += d
                                * (g[0].to_f64() * p[0] as f64
                                    + g[1].to_f64() * p[1] as f64
                                    + g[2].to_f64() * p[2] as f64);
                        }
                    }
                    (sa, sv)
                },
            )
            .collect();
        let area = sorted_sum(parts.iter().map(|p| p.0).collect());
        let volume = sorted_sum(parts.iter().map(|p| p.1).collect()) / 3.0;
        self.area_volume = Some((area, volume));
        (area, volume)
    }

    fn measure_curvatures(&mut self) -> (f64, f64) {
        if let Some(r) = self.curvature {
            return r;
        }
        let tree = self.grid.tree();
        let dx = self.dx;
        let (dx2, inv_dx) = (dx * dx, 1.0 / dx);
        let dd = DiracDelta::new(1.5);
        let grid = self.grid;
        let parts: Vec<(f64, f64)> = tree
            .leaf_refs()
            .par_iter()
            .map_init(
                || CurvatureStencil::with_dx(grid, dx),
                |st, leaf: &&LeafNode<T>| {
                    let (mut sm, mut sg) = (0.0, 0.0);
                    for (pos, v) in leaf.iter_on() {
                        let d = dd.eval(inv_dx * (*v).to_f64());
                        if d > 0.0 {
                            let p = leaf.offset_to_global_coord(pos);
                            st.move_to(&p);
                            let g = st.gradient();
                            let da = d * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().to_f64();
                            let (mean, gauss) = st.curvatures();
                            sm += da * mean.to_f64() * dx;
                            sg += da * gauss.to_f64() * dx2;
                        }
                    }
                    (sm, sg)
                },
            )
            .collect();
        let m = sorted_sum(parts.iter().map(|p| p.0).collect());
        let g = sorted_sum(parts.iter().map(|p| p.1).collect());
        self.curvature = Some((m, g));
        (m, g)
    }

    /// Surface area, in world units if `world_units`, otherwise in voxel units.
    pub fn area(&mut self, world_units: bool) -> f64 {
        let a = self.measure_area().0;
        if world_units {
            a * self.dx * self.dx
        } else {
            a
        }
    }
    /// Enclosed volume, in world units if `world_units`, otherwise in voxel units.
    pub fn volume(&mut self, world_units: bool) -> f64 {
        let v = self.measure_area().1;
        if world_units {
            v * self.dx * self.dx * self.dx
        } else {
            v
        }
    }
    /// Total mean curvature (integral of the mean curvature over the surface).
    pub fn total_mean_curvature(&mut self, world_units: bool) -> f64 {
        self.measure_curvatures().0 * if world_units { self.dx } else { 1.0 }
    }
    /// Total Gaussian curvature (dimensionless).
    pub fn total_gaussian_curvature(&mut self, _world_units: bool) -> f64 {
        self.measure_curvatures().1
    }
    /// Average mean curvature.
    pub fn avg_mean_curvature(&mut self, world_units: bool) -> f64 {
        self.total_mean_curvature(world_units) / self.area(world_units)
    }
    /// Average Gaussian curvature.
    pub fn avg_gaussian_curvature(&mut self, world_units: bool) -> f64 {
        self.total_gaussian_curvature(world_units) / self.area(world_units)
    }
    /// Euler characteristic: total Gaussian curvature / 2π, rounded.
    pub fn euler_characteristic(&mut self) -> i32 {
        let x = self.total_gaussian_curvature(true) / (2.0 * std::f64::consts::PI);
        x.round() as i32
    }
    /// Genus: `1 - χ/2` (integer division).
    pub fn genus(&mut self) -> i32 {
        1 - self.euler_characteristic() / 2
    }
}

/// Surface area of a level set (`tools::levelSetArea`).
pub fn level_set_area<T: RealValue + ToolValue>(grid: &Grid<T>, world_units: bool) -> ToolResult<f64> {
    Ok(LevelSetMeasure::new(grid)?.area(world_units))
}

/// Enclosed volume of a level set (`tools::levelSetVolume`).
pub fn level_set_volume<T: RealValue + ToolValue>(grid: &Grid<T>, world_units: bool) -> ToolResult<f64> {
    Ok(LevelSetMeasure::new(grid)?.volume(world_units))
}

/// Euler characteristic of a level set (`tools::levelSetEulerCharacteristic`).
pub fn level_set_euler_characteristic<T: RealValue + ToolValue>(grid: &Grid<T>) -> ToolResult<i32> {
    Ok(LevelSetMeasure::new(grid)?.euler_characteristic())
}

/// Genus of a level set (`tools::levelSetGenus`).
pub fn level_set_genus<T: RealValue + ToolValue>(grid: &Grid<T>) -> ToolResult<i32> {
    Ok(LevelSetMeasure::new(grid)?.genus())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3f;
    use crate::tools::level_set_sphere::create_level_set_sphere;

    #[test]
    fn sphere_measures() {
        let r = 1.0f64;
        let g = create_level_set_sphere::<f32>(r as f32, Vec3f::splat(0.0), 0.05, 3.0, true).unwrap();
        let mut m = LevelSetMeasure::new(&g).unwrap();
        let area = m.area(true);
        let vol = m.volume(true);
        let pi = std::f64::consts::PI;
        assert!((area - 4.0 * pi * r * r).abs() / (4.0 * pi) < 0.01, "{area}");
        assert!((vol - 4.0 / 3.0 * pi).abs() / (4.0 / 3.0 * pi) < 0.01, "{vol}");
        assert_eq!(m.euler_characteristic(), 2);
        assert_eq!(m.genus(), 0);
        assert!((m.avg_mean_curvature(true) - 1.0 / r).abs() < 0.05);
        assert!(level_set_area(&Grid::<f32>::new(1.0), true).is_err());
    }
}
