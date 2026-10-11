//! Rebuild a narrow-band level set from any scalar volume by meshing its isosurface and
//! scan-converting the mesh back to a signed distance field (port of
//! `openvdb/tools/LevelSetRebuild.h`).
//!
//! The isosurface at `isovalue` is extracted with [`VolumeToMesh`] (no adaptivity), its points
//! are transformed into the index space of the target transform (the input's by default) and
//! the quads and triangles are converted with [`mesh_to_volume`] without renormalization,
//! exactly like OpenVDB (whose optional "oracle" interior test is disabled by default).

use super::mesh_to_volume::{mesh_to_volume, QuadAndTriangleDataAdapter, DISABLE_RENORMALIZATION};
use super::volume_to_mesh::{MeshValue, VolumeToMesh};
use crate::grid::Grid;
use crate::math::{Transform, Vec3d, Vec3f};
use crate::types::RealValue;
use crate::util::INVALID_IDX;
use rayon::prelude::*;

/// Return a new narrow-band level set of the `isovalue` surface of `grid`, with symmetric
/// half-width `half_width` (in voxel units), in the index space of `xform` (or of `grid` if
/// `None`) (`tools::levelSetRebuild(grid, isovalue, halfWidth, xform)`).
pub fn level_set_rebuild<T: RealValue + MeshValue>(
    grid: &Grid<T>,
    isovalue: f32,
    half_width: f32,
    xform: Option<&Transform>,
) -> Grid<T> {
    level_set_rebuild_asymmetric(grid, isovalue, half_width, half_width, xform)
}

/// [`level_set_rebuild`] with distinct exterior and interior band widths (in voxel units)
/// (`tools::levelSetRebuild(grid, isovalue, exBandWidth, inBandWidth, xform)`).
pub fn level_set_rebuild_asymmetric<T: RealValue + MeshValue>(
    grid: &Grid<T>,
    isovalue: f32,
    ex_band_width: f32,
    in_band_width: f32,
    xform: Option<&Transform>,
) -> Grid<T> {
    do_level_set_rebuild(
        grid,
        <T as RealValue>::from_f64(isovalue as f64),
        <T as RealValue>::from_f64(ex_band_width as f64),
        <T as RealValue>::from_f64(in_band_width as f64),
        xform,
    )
}

/// The rebuild itself, with value-typed parameters (`tools::doLevelSetRebuild`).
pub fn do_level_set_rebuild<T: RealValue + MeshValue>(
    grid: &Grid<T>,
    iso: T,
    ex_width: T,
    in_width: T,
    xform: Option<&Transform>,
) -> Grid<T> {
    let isovalue = <T as RealValue>::to_f64(iso) as f32;
    let ex_band_width = <T as RealValue>::to_f64(ex_width) as f32;
    let in_band_width = <T as RealValue>::to_f64(in_width) as f32;

    let mut mesher = VolumeToMesh::<T>::new(isovalue as f64, 0.0, true);
    mesher.run(grid);

    let transform = xform.unwrap_or_else(|| grid.transform());

    // Copy and transform the points to the index space of the target transform.
    let points: Vec<Vec3f> = mesher
        .point_list()
        .par_iter()
        .map(|p| {
            let q = transform.world_to_index(Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64));
            Vec3f::new(q[0] as f32, q[1] as f32, q[2] as f32)
        })
        .collect();

    // Copy the primitives: per polygon pool, quads first, then triangles (as 4-tuples).
    let count: usize = mesher.polygon_pool_list().iter().map(|p| p.quads.len() + p.triangles.len()).sum();
    let mut primitives: Vec<[u32; 4]> = Vec::with_capacity(count);
    for pool in mesher.polygon_pool_list() {
        primitives.extend_from_slice(&pool.quads);
        primitives.extend(pool.triangles.iter().map(|t| [t[0], t[1], t[2], INVALID_IDX]));
    }

    let mesh = QuadAndTriangleDataAdapter::new(&points, &primitives);
    mesh_to_volume(&mesh, transform, ex_band_width, in_band_width, DISABLE_RENORMALIZATION, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridClass;
    use crate::math::Coord;
    use crate::tools::level_set_sphere::create_level_set_sphere;

    fn report(g: &Grid<f32>) -> (u64, usize, f32, f64, [f32; 4]) {
        let sum: f64 = g.tree().active_voxels().map(|(_, v)| v as f64).sum();
        let pts = [Coord::new(0, 0, 0), Coord::new(14, 0, 0), Coord::new(10, 3, -2), Coord::new(40, 0, 0)];
        (g.active_voxel_count(), g.tree().leaf_count(), *g.background(), sum, pts.map(|p| g.get_value(&p)))
    }

    /// Reference values from `scratchpad/cppref/lsrebuild.cpp` (OpenVDB 10).
    #[test]
    fn rebuild_sphere_matches_openvdb() {
        let g = create_level_set_sphere::<f32>(1.0, Vec3f::new(0.1, -0.05, 0.02), 0.1, 3.0, true).unwrap();
        let r = level_set_rebuild(&g, 0.0, 3.0, None);
        assert_eq!(r.grid_class(), GridClass::LevelSet);
        assert_eq!(r.transform(), g.transform());
        let (n, leaves, bg, sum, vals) = report(&r);
        assert_eq!((n, leaves), (7709, 50));
        assert_eq!(bg, 0.3);
        assert!((sum - 450.685225).abs() < 1e-6 * 450.685225, "{sum}");
        assert_eq!(vals[0], -0.3);
        assert_eq!(vals[1], 0.3);
        assert!((vals[2] - -0.007_965_385).abs() < 1e-7);
        // Asymmetric band, other isovalue, into a finer transform.
        let xf = Transform::linear(0.05);
        let r = level_set_rebuild_asymmetric(&g, 0.02, 2.0, 4.0, Some(&xf));
        assert_eq!(r.transform(), &xf);
        let (n, leaves, bg, sum, vals) = report(&r);
        assert_eq!((n, leaves), (28334, 158));
        assert_eq!(bg, 0.1);
        assert!((sum - -971.830653).abs() < 1e-6 * 971.830653, "{sum}");
        assert_eq!(vals, [-0.2, -0.2, -0.2, 0.1]);
    }
}
