//! Shared helpers for unit tests of the tools (not part of the public API).

use crate::grid::{Grid, GridClass};
use crate::math::{Coord, CoordBBox, Transform, Vec3d};
use crate::types::RealValue;

/// A narrow-band signed distance field of a sphere, built voxel by voxel.
///
/// Every voxel of the bounding box of the band is visited: voxels with `|d| < half_width * dx`
/// are active with value `d`, all others are set inactive to `±half_width * dx`. The C++
/// reference programs used to produce expected values build exactly the same grid.
pub(crate) fn make_sphere<T: RealValue>(radius: f64, center: Vec3d, voxel_size: f64, half_width: f64) -> Grid<T> {
    let bg = half_width * voxel_size;
    let mut grid = Grid::new(T::from_f64(bg)).with_transform(Transform::linear(voxel_size));
    grid.set_grid_class(GridClass::LevelSet);
    let ext = radius + bg;
    let lo = Coord::floor((center - Vec3d::splat(ext)) / voxel_size).offset(-1);
    let hi = Coord::ceil((center + Vec3d::splat(ext)) / voxel_size).offset(1);
    let xform = grid.transform().clone();
    {
        let mut acc = grid.accessor_mut();
        for c in CoordBBox::new(lo, hi) {
            let d = (xform.index_to_world(c.as_vec3d()) - center).length() - radius;
            if d.abs() < bg {
                acc.set_value_on(&c, T::from_f64(d));
            } else {
                acc.set_value_off(&c, T::from_f64(if d < 0.0 { -bg } else { bg }));
            }
        }
    }
    grid
}

/// Assert that two floats agree within `tol`, with a helpful message.
#[track_caller]
pub(crate) fn assert_close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol}, diff {})", (a - b).abs());
}
