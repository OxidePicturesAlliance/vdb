//! Narrow-band level set of a sphere (port of `openvdb/tools/LevelSetSphere.h`), plus the
//! `createLevelSet` grid factory from `openvdb/Grid.h`.

use crate::grid::{Grid, GridClass};
use crate::math::{Coord, Transform, Vec3f};
use crate::tools::signed_flood_fill::signed_flood_fill;
use crate::tools::tree_ops::{merge, MergePolicy};
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::Tree;
use crate::types::RealValue;
use rayon::prelude::*;

/// OpenVDB's default narrow-band half-width, in voxel units (`LEVEL_SET_HALF_WIDTH`).
pub const LEVEL_SET_HALF_WIDTH: f64 = 3.0;

/// An empty level-set grid with background `voxel_size * half_width`, a linear transform
/// with the given voxel size, and grid class [`GridClass::LevelSet`] (`createLevelSet`).
pub fn create_level_set<T: RealValue>(voxel_size: f64, half_width: f64) -> Grid<T> {
    let mut grid = Grid::new(T::from_f64(voxel_size * half_width)).with_transform(Transform::linear(voxel_size));
    grid.set_grid_class(GridClass::LevelSet);
    grid
}

/// Return a narrow-band level set of a sphere of the given world-space `radius` and
/// `center` (`tools::createLevelSetSphere`).
///
/// `voxel_size` is the world-space size of a voxel and `half_width` the half-width of the
/// narrow band in voxel units. Values are computed in the precision of `T`, exactly as
/// OpenVDB does, so the result matches its active voxels and values. If the radius is
/// smaller than 1.5 voxels the returned grid is empty.
pub fn create_level_set_sphere<T: RealValue + ToolValue>(
    radius: f32,
    center: Vec3f,
    voxel_size: f32,
    half_width: f32,
    threaded: bool,
) -> ToolResult<Grid<T>> {
    let r = T::from_f64(radius as f64);
    let dx = T::from_f64(voxel_size as f64);
    let w = T::from_f64(half_width as f64);
    if !(r > T::from_f64(0.0)) {
        return Err(ToolError::Value("radius must be positive".into()));
    }
    let mut grid = create_level_set::<T>(dx.to_f64(), w.to_f64());
    if !(dx > T::from_f64(0.0)) {
        return Err(ToolError::Value("voxel size must be positive".into()));
    }
    if !(w > T::from_f64(1.0)) {
        return Err(ToolError::Value("half-width must be larger than one".into()));
    }
    let r0 = r / dx;
    let rmax = r0 + w;
    if r0 < T::from_f64(1.5) {
        return Ok(grid);
    }
    let c =
        [T::from_f64(center[0] as f64) / dx, T::from_f64(center[1] as f64) / dx, T::from_f64(center[2] as f64) / dx];
    let floor = |v: T| v.floor().to_f64() as i32;
    let ceil = |v: T| v.ceil().to_f64() as i32;
    let (imin, imax) = (floor(c[0] - rmax), ceil(c[0] + rmax));
    let (jmin, jmax) = (floor(c[1] - rmax), ceil(c[1] + rmax));
    let (kmin, kmax) = (floor(c[2] - rmax), ceil(c[2] + rmax));
    let background = *grid.background();

    // Rasterize one slab of constant x into its own tree.
    let kernel = |i0: i32, i1: i32| -> Tree<T> {
        let mut tree = Tree::new(background);
        {
            let mut acc = tree.accessor_mut();
            for i in i0..i1 {
                let x2 = sqr(T::from_f64(i as f64) - c[0]);
                for j in jmin..=jmax {
                    let x2y2 = sqr(T::from_f64(j as f64) - c[1]) + x2;
                    let mut k = kmin;
                    while k <= kmax {
                        let mut m = 1;
                        let v = (x2y2 + sqr(T::from_f64(k as f64) - c[2])).sqrt() - r0;
                        let d = v.abs();
                        if d < w {
                            acc.set_value(&Coord::new(i, j, k), dx * v);
                        } else {
                            m += (d - w).floor().to_f64() as i32;
                        }
                        k += m;
                    }
                }
            }
        }
        tree
    };

    let tree = grid.tree_mut();
    if threaded {
        // Mirror tbb::blocked_range<int>(imin, imax, 128): split into chunks of x-slices.
        let chunks: Vec<(i32, i32)> = (imin..imax).step_by(64).map(|i0| (i0, (i0 + 64).min(imax))).collect();
        let parts: Vec<Tree<T>> = chunks.par_iter().map(|&(a, b)| kernel(a, b)).collect();
        for mut p in parts {
            merge(tree, &mut p, MergePolicy::ActiveStates);
        }
    } else {
        let mut p = kernel(imin, imax);
        merge(tree, &mut p, MergePolicy::ActiveStates);
    }
    signed_flood_fill(tree, threaded)?;
    Ok(grid)
}

#[inline]
fn sqr<T: RealValue>(v: T) -> T {
    v * v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_basics() {
        let g = create_level_set_sphere::<f32>(5.0, Vec3f::new(0.0, 0.0, 0.0), 0.5, 3.0, true).unwrap();
        assert_eq!(g.grid_class(), GridClass::LevelSet);
        assert_eq!(*g.background(), 1.5);
        assert!(g.get_value(&Coord::new(0, 0, 0)) == -1.5);
        assert!((g.get_value(&Coord::new(10, 0, 0))).abs() < 1e-6);
        let serial = create_level_set_sphere::<f32>(5.0, Vec3f::new(0.0, 0.0, 0.0), 0.5, 3.0, false).unwrap();
        assert_eq!(serial.active_voxel_count(), g.active_voxel_count());
        assert!(create_level_set_sphere::<f32>(-1.0, Vec3f::splat(0.0), 0.5, 3.0, true).is_err());
        let tiny = create_level_set_sphere::<f64>(0.5, Vec3f::splat(0.0), 0.5, 3.0, true).unwrap();
        assert!(tiny.empty());
    }
}
