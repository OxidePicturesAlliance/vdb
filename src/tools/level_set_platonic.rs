//! Narrow-band level sets of the five Platonic solids and of axis-aligned boxes (port of
//! `openvdb/tools/LevelSetPlatonic.h` and `createLevelSetBox` from
//! `openvdb/tools/MeshToVolume.h`).
//!
//! Exactly like OpenVDB, each shape is built as a triangle/quad mesh which is scan-converted
//! with [`mesh_to_level_set`], so active voxels and values match OpenVDB 10.

use crate::grid::Grid;
use crate::math::{Transform, Vec3d, Vec3f};
use crate::tools::mesh_to_volume::mesh_to_level_set;
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::types::RealValue;

/// Split polygons (lists of vertex indices) into triangles and quads, fan-triangulating
/// polygons with more than four vertices.
fn split_polygons(faces: &[Vec<u32>]) -> (Vec<[u32; 3]>, Vec<[u32; 4]>) {
    let mut tri = Vec::new();
    let mut qua = Vec::new();
    for f in faces {
        match f.len() {
            3 => tri.push([f[0], f[1], f[2]]),
            4 => qua.push([f[0], f[1], f[2], f[3]]),
            _ => {
                for i in 1..f.len() - 1 {
                    tri.push([f[0], f[i], f[i + 1]]);
                }
            }
        }
    }
    (tri, qua)
}

/// Return a narrow-band level set of the closed polyhedron with the given world-space
/// `vertices` and planar `faces` (lists of vertex indices, consistently oriented), scan
/// converted with [`mesh_to_level_set`] using a linear transform of the given voxel size.
/// Triangles and quads are passed as such; larger polygons are fan-triangulated.
///
/// This is the shared implementation of the Platonic solids.
pub fn create_level_set_convex_polytope<T: RealValue + ToolValue>(
    vertices: &[Vec3d],
    faces: &[Vec<usize>],
    voxel_size: f64,
    half_width: f64,
) -> ToolResult<Grid<T>> {
    if faces.len() < 4 || faces.iter().any(|f| f.len() < 3 || f.iter().any(|&i| i >= vertices.len())) {
        return Err(ToolError::Value("invalid polytope".into()));
    }
    let points: Vec<Vec3f> = vertices.iter().map(|v| v.map(|c| c as f32)).collect();
    let faces: Vec<Vec<u32>> = faces.iter().map(|f| f.iter().map(|&i| i as u32).collect()).collect();
    mesh_level_set(&points, &faces, voxel_size, half_width as f32)
}

fn mesh_level_set<T: RealValue>(
    points: &[Vec3f],
    faces: &[Vec<u32>],
    voxel_size: f64,
    half_width: f32,
) -> ToolResult<Grid<T>> {
    if !(voxel_size > 0.0) {
        return Err(ToolError::Value("voxel size must be positive".into()));
    }
    let xform = Transform::linear(voxel_size);
    let (tri, qua) = split_polygons(faces);
    Ok(mesh_to_level_set(&xform, points, &tri, &qua, half_width))
}

/// Return a narrow-band level set of a Platonic solid with `face_count` faces (4, 6, 8, 12
/// or 20), uniformly scaled by `scale` and translated to `center` (`tools::createLevelSetPlatonic`).
pub fn create_level_set_platonic<T: RealValue + ToolValue>(
    face_count: u32,
    scale: f32,
    center: Vec3f,
    voxel_size: f32,
    half_width: f32,
) -> ToolResult<Grid<T>> {
    let (vtx, faces): (Vec<[f32; 3]>, Vec<Vec<u32>>) = match face_count {
        4 => (
            vec![
                [0.0, 1.0, 0.0],
                [-0.942810297, -0.333329707, 0.0],
                [0.471405149, -0.333329707, 0.816497624],
                [0.471405149, -0.333329707, -0.816497624],
            ],
            vec![vec![0, 2, 3], vec![0, 3, 1], vec![0, 1, 2], vec![1, 3, 2]],
        ),
        6 => (
            vec![
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            vec![
                vec![1, 0, 4, 5],
                vec![2, 1, 5, 6],
                vec![3, 2, 6, 7],
                vec![0, 3, 7, 4],
                vec![2, 3, 0, 1],
                vec![5, 4, 7, 6],
            ],
        ),
        8 => (
            vec![
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![
                vec![0, 4, 3],
                vec![0, 1, 4],
                vec![1, 2, 4],
                vec![2, 3, 4],
                vec![0, 3, 5],
                vec![0, 5, 1],
                vec![1, 5, 2],
                vec![2, 5, 3],
            ],
        ),
        12 => (
            vec![
                [0.354437858, 0.487842113, -0.789344311],
                [0.573492587, -0.186338872, -0.78934437],
                [0.0, -0.603005826, -0.78934443],
                [-0.573492587, -0.186338872, -0.78934437],
                [-0.354437858, 0.487842113, -0.789344311],
                [-0.573492587, 0.789345026, -0.186338797],
                [-0.927930415, -0.301502913, -0.186338872],
                [0.0, -0.975683928, -0.186338902],
                [0.927930415, -0.301502913, -0.186338872],
                [0.573492587, 0.789345026, -0.186338797],
                [0.0, 0.975683868, 0.186338902],
                [-0.927930415, 0.301502913, 0.186338872],
                [-0.573492587, -0.789345026, 0.186338797],
                [0.573492587, -0.789345026, 0.186338797],
                [0.927930415, 0.301502913, 0.186338872],
                [0.0, 0.603005826, 0.78934443],
                [0.573492587, 0.186338872, 0.78934437],
                [0.354437858, -0.487842113, 0.789344311],
                [-0.354437858, -0.487842113, 0.789344311],
                [-0.573492587, 0.186338872, 0.78934437],
            ],
            // Each pentagon is given as a quad plus a triangle, interleaved as in OpenVDB.
            vec![
                vec![0, 1, 2, 3],
                vec![0, 3, 4],
                vec![0, 4, 5, 10],
                vec![0, 10, 9],
                vec![0, 9, 14, 8],
                vec![0, 8, 1],
                vec![1, 8, 13, 7],
                vec![1, 7, 2],
                vec![2, 7, 12, 6],
                vec![2, 6, 3],
                vec![3, 6, 11, 5],
                vec![3, 5, 4],
                vec![5, 11, 19, 15],
                vec![5, 15, 10],
                vec![6, 12, 18, 19],
                vec![6, 19, 11],
                vec![7, 13, 17, 18],
                vec![7, 18, 12],
                vec![8, 14, 16, 17],
                vec![8, 17, 13],
                vec![9, 10, 15, 16],
                vec![9, 16, 14],
                vec![15, 19, 18, 17],
                vec![15, 17, 16],
            ],
        ),
        20 => (
            vec![
                [0.0, 0.0, -1.0],
                [0.0, 0.894427359, -0.447213143],
                [0.850650847, 0.276393682, -0.447213203],
                [0.525731206, -0.723606944, -0.447213262],
                [-0.525731206, -0.723606944, -0.447213262],
                [-0.850650847, 0.276393682, -0.447213203],
                [-0.525731206, 0.723606944, 0.447213262],
                [-0.850650847, -0.276393682, 0.447213203],
                [0.0, -0.894427359, 0.447213143],
                [0.850650847, -0.276393682, 0.447213203],
                [0.525731206, 0.723606944, 0.447213262],
                [0.0, 0.0, 1.0],
            ],
            vec![
                vec![2, 0, 1],
                vec![3, 0, 2],
                vec![4, 0, 3],
                vec![5, 0, 4],
                vec![1, 0, 5],
                vec![6, 1, 5],
                vec![7, 5, 4],
                vec![8, 4, 3],
                vec![9, 3, 2],
                vec![10, 2, 1],
                vec![10, 1, 6],
                vec![6, 5, 7],
                vec![7, 4, 8],
                vec![8, 3, 9],
                vec![9, 2, 10],
                vec![6, 11, 10],
                vec![10, 11, 9],
                vec![9, 11, 8],
                vec![8, 11, 7],
                vec![7, 11, 6],
            ],
        ),
        _ => return Err(ToolError::Runtime("Invalid face count".into())),
    };
    // Scale and translate in single precision, as OpenVDB does.
    let points: Vec<Vec3f> = vtx.iter().map(|v| Vec3f::new(v[0], v[1], v[2]) * scale + center).collect();
    mesh_level_set(&points, &faces, voxel_size as f64, half_width)
}

/// Level set of a regular tetrahedron (`tools::createLevelSetTetrahedron`).
pub fn create_level_set_tetrahedron<T: RealValue + ToolValue>(
    scale: f32,
    center: Vec3f,
    voxel_size: f32,
    half_width: f32,
) -> ToolResult<Grid<T>> {
    create_level_set_platonic(4, scale, center, voxel_size, half_width)
}

/// Level set of a cube (`tools::createLevelSetCube`).
pub fn create_level_set_cube<T: RealValue + ToolValue>(
    scale: f32,
    center: Vec3f,
    voxel_size: f32,
    half_width: f32,
) -> ToolResult<Grid<T>> {
    create_level_set_platonic(6, scale, center, voxel_size, half_width)
}

/// Level set of a regular octahedron (`tools::createLevelSetOctahedron`).
pub fn create_level_set_octahedron<T: RealValue + ToolValue>(
    scale: f32,
    center: Vec3f,
    voxel_size: f32,
    half_width: f32,
) -> ToolResult<Grid<T>> {
    create_level_set_platonic(8, scale, center, voxel_size, half_width)
}

/// Level set of a regular dodecahedron (`tools::createLevelSetDodecahedron`).
pub fn create_level_set_dodecahedron<T: RealValue + ToolValue>(
    scale: f32,
    center: Vec3f,
    voxel_size: f32,
    half_width: f32,
) -> ToolResult<Grid<T>> {
    create_level_set_platonic(12, scale, center, voxel_size, half_width)
}

/// Level set of a regular icosahedron (`tools::createLevelSetIcosahedron`).
pub fn create_level_set_icosahedron<T: RealValue + ToolValue>(
    scale: f32,
    center: Vec3f,
    voxel_size: f32,
    half_width: f32,
) -> ToolResult<Grid<T>> {
    create_level_set_platonic(20, scale, center, voxel_size, half_width)
}

/// Level set of the world-space axis-aligned box `[min, max]` with a linear transform of
/// the given voxel size (`tools::createLevelSetBox`; the half-width is in voxel units).
/// See [`crate::tools::mesh_to_volume::create_level_set_box`] for an arbitrary transform.
pub fn create_level_set_box<T: RealValue + ToolValue>(
    min: Vec3d,
    max: Vec3d,
    voxel_size: f64,
    half_width: f64,
) -> ToolResult<Grid<T>> {
    if !(voxel_size > 0.0) {
        return Err(ToolError::Value("voxel size must be positive".into()));
    }
    let xform = Transform::linear(voxel_size);
    Ok(crate::tools::mesh_to_volume::create_level_set_box(min, max, &xform, half_width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;

    #[test]
    fn cube_distances() {
        let g = create_level_set_cube::<f32>(2.0, Vec3f::splat(0.0), 0.1, 3.0).unwrap();
        // Face at x = 1 (index 10): voxel 10 is on the surface, 9 is one voxel inside.
        assert!(g.get_value(&Coord::new(10, 0, 0)).abs() < 1e-6);
        assert!((g.get_value(&Coord::new(9, 0, 0)) + 0.1).abs() < 1e-6);
        assert!((g.get_value(&Coord::new(12, 0, 0)) - 0.2).abs() < 1e-6);
        assert_eq!(g.get_value(&Coord::new(0, 0, 0)), -0.3f32);
        assert_eq!(g.get_value(&Coord::new(50, 0, 0)), 0.3f32);
        for faces in [4, 8, 12, 20] {
            let s = create_level_set_platonic::<f32>(faces, 1.0, Vec3f::splat(0.0), 0.05, 3.0).unwrap();
            assert!(s.active_voxel_count() > 0);
            assert!(s.get_value(&Coord::new(0, 0, 0)) < 0.0);
        }
        assert!(create_level_set_platonic::<f32>(5, 1.0, Vec3f::splat(0.0), 0.05, 3.0).is_err());
        let b = create_level_set_box::<f64>(Vec3d::splat(-1.0), Vec3d::splat(1.0), 0.1, 3.0).unwrap();
        assert!((b.get_value(&Coord::new(11, 0, 0)) - 0.1).abs() < 1e-6);
    }
}
