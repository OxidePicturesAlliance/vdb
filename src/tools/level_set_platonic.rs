//! Narrow-band level sets of the five Platonic solids and of axis-aligned boxes (port of
//! `openvdb/tools/LevelSetPlatonic.h` and `createLevelSetBox` from
//! `openvdb/tools/MeshToVolume.h`).
//!
//! OpenVDB builds these shapes by scan-converting a polygon mesh with `meshToLevelSet`.
//! Since every shape here is a convex polytope, this port computes the exact signed
//! distance to the polytope directly instead (the distance to the closest triangle of the
//! triangulated boundary, negative inside), which is what mesh scan conversion converges
//! to for closed convex meshes. Voxels whose distance is below the half-width (in voxel
//! units) are active, and the inactive interior is flood-filled with `-background`.
//!
//! Compared with OpenVDB 10 (`meshToLevelSet`), the values of the voxels active in both
//! agree to within single precision (except for a few voxels near edges that OpenVDB's
//! renormalization pass adjusts by up to a few hundredths of a voxel), but OpenVDB's band
//! is grown iteratively from the surface and omits some voxels near edges and corners
//! whose exact distance is just below the half-width, so this port's active set is a
//! slight superset (e.g. 10727 vs 10724 voxels for a tetrahedron, 12218 vs 11966 for a
//! cube with voxel-aligned faces).

use crate::grid::Grid;
use crate::math::{Coord, Vec3d, Vec3f};
use crate::tools::level_set_sphere::create_level_set;
use crate::tools::signed_flood_fill::signed_flood_fill;
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::{LeafNode, Node};
use crate::types::RealValue;
use rayon::prelude::*;

/// A convex polytope in index space, triangulated, with outward face planes.
struct ConvexPolytope {
    triangles: Vec<[Vec3d; 3]>,
    /// Outward unit normal and offset of each face plane: `n·p - d` is the signed distance.
    planes: Vec<(Vec3d, f64)>,
}

impl ConvexPolytope {
    fn new(points: &[Vec3d], faces: &[Vec<usize>]) -> Self {
        let n = points.len() as f64;
        let centroid = points.iter().fold(Vec3d::zero(), |a, p| a + *p) * (1.0 / n);
        let mut triangles = Vec::new();
        let mut planes = Vec::new();
        for f in faces {
            let a = points[f[0]];
            for i in 1..f.len() - 1 {
                triangles.push([a, points[f[i]], points[f[i + 1]]]);
            }
            let mut nrm = (points[f[1]] - a).cross(&(points[f[2]] - a));
            nrm = nrm * (1.0 / nrm.length());
            if nrm.dot(&(centroid - a)) > 0.0 {
                nrm = -nrm;
            }
            planes.push((nrm, nrm.dot(&a)));
        }
        ConvexPolytope { triangles, planes }
    }

    /// Signed distance from `p` to the boundary (negative inside).
    fn signed_distance(&self, p: Vec3d) -> f64 {
        let mut d2 = f64::MAX;
        for t in &self.triangles {
            let q = closest_point_on_triangle(p, t[0], t[1], t[2]);
            d2 = d2.min((p - q).length_sqr());
        }
        let d = d2.sqrt();
        let inside = self.planes.iter().all(|(n, off)| n.dot(&p) - off < 0.0);
        if inside {
            -d
        } else {
            d
        }
    }

    /// Unsigned distance from `p` to the boundary (used for culling).
    fn unsigned_distance(&self, p: Vec3d) -> f64 {
        self.signed_distance(p).abs()
    }
}

/// Closest point to `p` on triangle `abc` (Ericson, "Real-Time Collision Detection" 5.1.5).
fn closest_point_on_triangle(p: Vec3d, a: Vec3d, b: Vec3d, c: Vec3d) -> Vec3d {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Return a narrow-band level set of the convex polytope with the given world-space
/// `vertices` and (planar, convex) `faces` (lists of vertex indices).
///
/// This is the shared implementation of the Platonic solids and [`create_level_set_box`].
pub fn create_level_set_convex_polytope<T: RealValue + ToolValue>(
    vertices: &[Vec3d],
    faces: &[Vec<usize>],
    voxel_size: f64,
    half_width: f64,
) -> ToolResult<Grid<T>> {
    if !(voxel_size > 0.0) {
        return Err(ToolError::Value("voxel size must be positive".into()));
    }
    if faces.len() < 4 || faces.iter().any(|f| f.len() < 3 || f.iter().any(|&i| i >= vertices.len())) {
        return Err(ToolError::Value("invalid polytope".into()));
    }
    let mut grid = create_level_set::<T>(voxel_size, half_width);
    // OpenVDB's mesh scan conversion stores index-space points in single precision.
    let points: Vec<Vec3d> =
        vertices.iter().map(|v| grid.transform().world_to_index(*v).map(|c| c as f32 as f64)).collect();
    let poly = ConvexPolytope::new(&points, faces);
    let hw = half_width;

    // Leaf-aligned index-space bounds of the band.
    let mut lo = Vec3d::splat(f64::MAX);
    let mut hi = Vec3d::splat(f64::MIN);
    for p in &points {
        lo = lo.min_elem(p);
        hi = hi.max_elem(p);
    }
    let lo = Coord::floor(lo - Vec3d::splat(hw + 1.0)).align_down(8);
    let hi = Coord::ceil(hi + Vec3d::splat(hw + 1.0));

    // Hierarchically cull blocks that cannot intersect the band.
    let mut leaf_origins = Vec::new();
    let mut blocks = Vec::new();
    let block = 64;
    let mut x = lo[0];
    while x <= hi[0] {
        let mut y = lo[1];
        while y <= hi[1] {
            let mut z = lo[2];
            while z <= hi[2] {
                blocks.push(Coord::new(x, y, z));
                z += block;
            }
            y += block;
        }
        x += block;
    }
    let cull = |origin: Coord, size: i32| -> bool {
        let half = (size - 1) as f64 * 0.5;
        let center = origin.as_vec3d() + Vec3d::splat(half);
        let radius = half * 3f64.sqrt();
        poly.unsigned_distance(center) >= hw + radius
    };
    for b in blocks {
        if cull(b, block) {
            continue;
        }
        for dx in (0..block).step_by(8) {
            for dy in (0..block).step_by(8) {
                for dz in (0..block).step_by(8) {
                    let o = b.offset_by(dx, dy, dz);
                    if !cull(o, 8) {
                        leaf_origins.push(o);
                    }
                }
            }
        }
    }

    let dx = voxel_size;
    let background = *grid.background();
    let leaves: Vec<Box<LeafNode<T>>> = leaf_origins
        .par_iter()
        .filter_map(|&o| {
            let mut leaf = LeafNode::new(o, background, false);
            for n in 0..LeafNode::<T>::SIZE {
                let ijk = leaf.offset_to_global_coord(n);
                let d = poly.signed_distance(ijk.as_vec3d());
                if d.abs() < hw {
                    leaf.set_value_on_at(n, T::from_f64(d * dx));
                } else if d < 0.0 {
                    leaf.set_value_off_at(n, background.negative());
                }
            }
            (!leaf.is_empty()).then(|| Box::new(leaf))
        })
        .collect();
    let tree = grid.tree_mut();
    for leaf in leaves {
        tree.add_leaf(leaf);
    }
    signed_flood_fill(tree, true)?;
    Ok(grid)
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
    let (vtx, faces): (Vec<[f32; 3]>, Vec<Vec<usize>>) = match face_count {
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
            // Each pentagon is given in OpenVDB as a quad plus a triangle.
            vec![
                vec![0, 1, 2, 3, 4],
                vec![0, 4, 5, 10, 9],
                vec![0, 9, 14, 8, 1],
                vec![1, 8, 13, 7, 2],
                vec![2, 7, 12, 6, 3],
                vec![3, 6, 11, 5, 4],
                vec![5, 11, 19, 15, 10],
                vec![6, 12, 18, 19, 11],
                vec![7, 13, 17, 18, 12],
                vec![8, 14, 16, 17, 13],
                vec![9, 10, 15, 16, 14],
                vec![15, 19, 18, 17, 16],
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
    let vertices: Vec<Vec3d> = vtx
        .iter()
        .map(|v| {
            let p = Vec3f::new(v[0], v[1], v[2]) * scale + center;
            Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64)
        })
        .collect();
    create_level_set_convex_polytope(&vertices, &faces, voxel_size as f64, half_width as f64)
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
pub fn create_level_set_box<T: RealValue + ToolValue>(
    min: Vec3d,
    max: Vec3d,
    voxel_size: f64,
    half_width: f64,
) -> ToolResult<Grid<T>> {
    let p = [
        Vec3d::new(min[0], min[1], min[2]),
        Vec3d::new(min[0], min[1], max[2]),
        Vec3d::new(max[0], min[1], max[2]),
        Vec3d::new(max[0], min[1], min[2]),
        Vec3d::new(min[0], max[1], min[2]),
        Vec3d::new(min[0], max[1], max[2]),
        Vec3d::new(max[0], max[1], max[2]),
        Vec3d::new(max[0], max[1], min[2]),
    ];
    let faces = vec![
        vec![0, 1, 2, 3],
        vec![7, 6, 5, 4],
        vec![4, 5, 1, 0],
        vec![6, 7, 3, 2],
        vec![0, 3, 7, 4],
        vec![1, 5, 6, 2],
    ];
    create_level_set_convex_polytope(&p, &faces, voxel_size, half_width)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!((b.get_value(&Coord::new(11, 0, 0)) - 0.1).abs() < 1e-9);
    }
}
