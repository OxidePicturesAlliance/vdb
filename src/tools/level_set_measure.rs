//! Area, volume, curvature and topology (Euler characteristic, genus) of level sets (port
//! of `openvdb/tools/LevelSetMeasure.h`).
//!
//! The measures are computed with a smeared Dirac delta function (three voxels wide)
//! applied to finite-difference gradients and curvatures, exactly as in OpenVDB. The small
//! stencils needed here (central gradient, curvature, first-order and WENO5 upwind
//! gradients) are implemented privately in this module.

use crate::grid::{Grid, GridClass};
use crate::math::{Coord, Vec3};
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::{Internal2, LeafNode, ValueAccessor};
use crate::types::RealValue;
use rayon::prelude::*;

type Acc<'a, T> = ValueAccessor<'a, Internal2<T>>;

/// Central-difference gradient in world units (`GradStencil::gradient`), in `T` precision.
#[inline]
pub(crate) fn central_gradient<T: RealValue>(acc: &Acc<'_, T>, ijk: &Coord, inv_2dx: T) -> Vec3<T> {
    Vec3::new(
        acc.get_value(&ijk.offset_by(1, 0, 0)) - acc.get_value(&ijk.offset_by(-1, 0, 0)),
        acc.get_value(&ijk.offset_by(0, 1, 0)) - acc.get_value(&ijk.offset_by(0, -1, 0)),
        acc.get_value(&ijk.offset_by(0, 0, 1)) - acc.get_value(&ijk.offset_by(0, 0, -1)),
    ) * inv_2dx
}

/// Godunov's norm-squared gradient from backward (`m`) and forward (`p`) differences.
#[inline]
pub(crate) fn godunov_norm_sqrd<R: num_traits::Float>(outside: bool, m: [R; 3], p: [R; 3]) -> R {
    let zero = R::zero();
    let mut s = zero;
    for i in 0..3 {
        s = s + if outside {
            (m[i].max(zero) * m[i].max(zero)).max(p[i].min(zero) * p[i].min(zero))
        } else {
            (m[i].min(zero) * m[i].min(zero)).max(p[i].max(zero) * p[i].max(zero))
        };
    }
    s
}

/// `ISGradientNormSqrd<FIRST_BIAS>` at `ijk` (index-space differences, no 1/dx scaling).
#[inline]
pub(crate) fn first_bias_norm_sq_grad<T: RealValue>(acc: &Acc<'_, T>, ijk: &Coord, center: T) -> T {
    let v = |x, y, z| acc.get_value(&ijk.offset_by(x, y, z));
    let up = [v(1, 0, 0) - center, v(0, 1, 0) - center, v(0, 0, 1) - center];
    let down = [center - v(-1, 0, 0), center - v(0, -1, 0), center - v(0, 0, -1)];
    godunov_norm_sqrd(center > T::from_f64(0.0), down, up)
}

/// `math::WENO5` (computed in double precision, as in OpenVDB).
#[inline]
fn weno5(v1: f64, v2: f64, v3: f64, v4: f64, v5: f64, scale2: f32) -> f64 {
    let c = 13.0 / 12.0;
    let eps = 1.0e-6 * scale2 as f64;
    let sq = |x: f64| x * x;
    let a1 = 0.1 / sq(c * sq(v1 - 2.0 * v2 + v3) + 0.25 * sq(v1 - 4.0 * v2 + 3.0 * v3) + eps);
    let a2 = 0.6 / sq(c * sq(v2 - 2.0 * v3 + v4) + 0.25 * sq(v2 - v4) + eps);
    let a3 = 0.3 / sq(c * sq(v3 - 2.0 * v4 + v5) + 0.25 * sq(3.0 * v3 - 4.0 * v4 + v5) + eps);
    (a1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + a2 * (5.0 * v3 - v2 + 2.0 * v4) + a3 * (2.0 * v3 + 5.0 * v4 - v5))
        / (6.0 * (a1 + a2 + a3))
}

/// The 19-point WENO stencil values (`WenoStencil`): `[center, x-3..x+3 (6), y.., z..]`.
pub(crate) fn weno_values<T: RealValue>(acc: &Acc<'_, T>, ijk: &Coord, center: T) -> [T; 19] {
    let mut v = [center; 19];
    let offs = [-3, -2, -1, 1, 2, 3];
    for axis in 0..3 {
        for (i, o) in offs.iter().enumerate() {
            let mut d = [0i32; 3];
            d[axis] = *o;
            v[1 + axis * 6 + i] = acc.get_value(&ijk.offset_by(d[0], d[1], d[2]));
        }
    }
    v
}

/// `WenoStencil::normSqGrad` (world units) for the given stencil values.
pub(crate) fn weno_norm_sq_grad<T: RealValue>(v: &[T; 19], dx: f64, iso: T) -> T {
    let dx2 = (dx * dx) as f32;
    let inv_dx2 = 1.0 / (dx * dx);
    let d = |a: usize, b: usize| (v[a] - v[b]).to_f64();
    let mut m = [0.0f64; 3];
    let mut p = [0.0f64; 3];
    for axis in 0..3 {
        let b = 1 + axis * 6; // indices: b+0 = -3, b+1 = -2, b+2 = -1, b+3 = +1, b+4 = +2, b+5 = +3
        m[axis] = weno5(d(b + 1, b), d(b + 2, b + 1), d(0, b + 2), d(b + 3, 0), d(b + 4, b + 3), dx2);
        p[axis] = weno5(d(b + 5, b + 4), d(b + 4, b + 3), d(b + 3, 0), d(0, b + 2), d(b + 2, b + 1), dx2);
    }
    T::from_f64(inv_dx2 * godunov_norm_sqrd(v[0] > iso, m, p))
}

/// `WenoStencil::zeroCrossing`: the sign of the center differs from a face neighbor.
pub(crate) fn weno_zero_crossing<T: RealValue>(v: &[T; 19]) -> bool {
    let zero = T::from_f64(0.0);
    let n = [v[3], v[4], v[9], v[10], v[15], v[16]];
    if v[0] > zero {
        n.iter().any(|x| *x < zero)
    } else {
        n.iter().any(|x| *x > zero)
    }
}

/// `CurvatureStencil::curvatures`: mean and Gaussian curvature (in `T` precision).
fn curvatures<T: RealValue>(acc: &Acc<'_, T>, ijk: &Coord, center: T, inv_2dx: T, inv_dx2: T) -> (T, T) {
    let g = |x, y, z| acc.get_value(&ijk.offset_by(x, y, z)).to_f64();
    let c = center.to_f64();
    let (v1, v2, v3, v4, v5, v6) = (g(-1, 0, 0), g(1, 0, 0), g(0, -1, 0), g(0, 1, 0), g(0, 0, -1), g(0, 0, 1));
    let (v7, v8, v9, v10) = (g(-1, -1, 0), g(1, -1, 0), g(-1, 1, 0), g(1, 1, 0));
    let (v11, v12, v13, v14) = (g(-1, 0, -1), g(1, 0, -1), g(-1, 0, 1), g(1, 0, 1));
    let (v15, v16, v17, v18) = (g(0, -1, -1), g(0, 1, -1), g(0, -1, 1), g(0, 1, 1));
    let dx = 0.5 * (v2 - v1);
    let dy = 0.5 * (v4 - v3);
    let dz = 0.5 * (v6 - v5);
    let (dx2, dy2, dz2) = (dx * dx, dy * dy, dz * dz);
    let norm_grad2 = dx2 + dy2 + dz2;
    if norm_grad2 <= crate::math::TOLERANCE_F64 {
        return (T::from_f64(0.0), T::from_f64(0.0));
    }
    let dxx = v2 - 2.0 * c + v1;
    let dyy = v4 - 2.0 * c + v3;
    let dzz = v6 - 2.0 * c + v5;
    let dxy = 0.25 * (v10 - v8 + v7 - v9);
    let dxz = 0.25 * (v14 - v12 + v11 - v13);
    let dyz = 0.25 * (v18 - v16 + v15 - v17);
    let alpha_m =
        dx2 * (dyy + dzz) + dy2 * (dxx + dzz) + dz2 * (dxx + dyy) - 2.0 * (dx * (dy * dxy + dz * dxz) + dy * dz * dyz);
    let alpha_g = dx2 * (dyy * dzz - dyz * dyz)
        + dy2 * (dxx * dzz - dxz * dxz)
        + dz2 * (dxx * dyy - dxy * dxy)
        + 2.0
            * (dy * dz * (dxy * dxz - dyz * dxx)
                + dx * dz * (dxy * dyz - dxz * dyy)
                + dx * dy * (dxz * dyz - dxy * dzz));
    let norm_grad = norm_grad2.sqrt();
    let mean = T::from_f64(alpha_m * inv_2dx.to_f64() / norm_grad.powi(3));
    let gauss = T::from_f64(alpha_g * inv_dx2.to_f64() / norm_grad.powi(4));
    (mean, gauss)
}

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
        let inv_2dx = T::from_f64(0.5 / dx);
        let dd = DiracDelta::new(1.5);
        let parts: Vec<(f64, f64)> = tree
            .leaf_refs()
            .par_iter()
            .map(|leaf: &&LeafNode<T>| {
                let acc = tree.accessor();
                let (mut sa, mut sv) = (0.0, 0.0);
                for (pos, v) in leaf.iter_on() {
                    let d = dd.eval(inv_dx * (*v).to_f64());
                    if d > 0.0 {
                        let p = leaf.offset_to_global_coord(pos);
                        let g = central_gradient(&acc, &p, inv_2dx);
                        let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                        sa += d * len.to_f64();
                        sv += d
                            * (g[0].to_f64() * p[0] as f64 + g[1].to_f64() * p[1] as f64 + g[2].to_f64() * p[2] as f64);
                    }
                }
                (sa, sv)
            })
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
        let inv_2dx = T::from_f64(0.5 / dx);
        let inv_dx2 = T::from_f64(4.0 * (0.5 / dx) * (0.5 / dx));
        let inv_2dx_f = T::from_f64(0.5 / dx);
        let dd = DiracDelta::new(1.5);
        let parts: Vec<(f64, f64)> = tree
            .leaf_refs()
            .par_iter()
            .map(|leaf: &&LeafNode<T>| {
                let acc = tree.accessor();
                let (mut sm, mut sg) = (0.0, 0.0);
                for (pos, v) in leaf.iter_on() {
                    let d = dd.eval(inv_dx * (*v).to_f64());
                    if d > 0.0 {
                        let p = leaf.offset_to_global_coord(pos);
                        let g = central_gradient(&acc, &p, inv_2dx);
                        let da = d * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().to_f64();
                        let (mean, gauss) = curvatures(&acc, &p, *v, inv_2dx_f, inv_dx2);
                        sm += da * mean.to_f64() * dx;
                        sg += da * gauss.to_f64() * dx2;
                    }
                }
                (sm, sg)
            })
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
