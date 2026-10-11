//! Narrow-band level set of a dilated triangle/quad mesh, i.e. the set of points within a
//! given radius of the mesh (port of `openvdb/tools/LevelSetDilatedMesh.h`).
//!
//! OpenVDB decomposes the dilated mesh into open triangular prisms (one per triangle) and
//! open capsule wedges (one per edge) and rasterizes each with its `ConvexVoxelizer`. The
//! union of these pieces is the set of points within `radius` of the mesh, with signed
//! distance `distance(p, mesh) - radius`; this port rasterizes that distance directly, per
//! triangle (degenerate triangles become capsules around their longest edge, as in
//! OpenVDB). Results agree with OpenVDB up to floating-point rounding (see the module tests
//! and the report for the comparison against the C++ library).

use super::level_set_tubes::{level_set_params, rasterize_union, Capsule, DynShape, SdfShape};
use super::mesh_support::closest_point_on_triangle_to_point;
use crate::grid::Grid;
use crate::math::{Coord, CoordBBox, Vec3, Vec3d};
use crate::types::RealValue;

struct DilatedTriangle<T> {
    a: Vec3d,
    b: Vec3d,
    c: Vec3d,
    rad: f64,
    orad: f64,
    _t: std::marker::PhantomData<T>,
}

impl<T: RealValue> SdfShape<T> for DilatedTriangle<T> {
    #[inline]
    fn sd(&self, p: &Vec3<T>) -> T {
        let q = Vec3d::new(p[0].to_f64(), p[1].to_f64(), p[2].to_f64());
        let (cp, _) = closest_point_on_triangle_to_point(&self.a, &self.b, &self.c, &q);
        T::from_f64((q - cp).length() - self.rad)
    }
    fn bbox(&self) -> CoordBBox {
        let mut mn = [0i32; 3];
        let mut mx = [0i32; 3];
        for i in 0..3 {
            mn[i] = (self.a[i].min(self.b[i]).min(self.c[i]) - self.orad).floor() as i32 - 1;
            mx[i] = (self.a[i].max(self.b[i]).max(self.c[i]) + self.orad).ceil() as i32 + 1;
        }
        CoordBBox::new(Coord(mn), Coord(mx))
    }
}

/// Narrow-band level set of the region within `radius` of a triangle mesh
/// (`tools::createLevelSetDilatedMesh`). Vertices are in world units.
///
/// # Panics
/// If `voxel_size` or `half_width` is not positive or an index is out of bounds.
pub fn create_level_set_dilated_mesh<T: RealValue, S: RealValue>(
    vertices: &[Vec3<S>],
    triangles: &[[u32; 3]],
    radius: S,
    voxel_size: f32,
    half_width: f32,
) -> Grid<T> {
    let (mut grid, vox, hw, bg) = level_set_params::<T>(voxel_size, half_width);
    // Index-space positions and radius are computed in the grid's value precision, as in
    // OpenVDB, so that distances on band boundaries round identically.
    let to_index = |p: &Vec3<S>| {
        let q = |v: S| (T::from_f64(v.to_f64()) / vox).to_f64();
        Vec3d::new(q(p[0]), q(p[1]), q(p[2]))
    };
    let tol = S::tolerance();
    let shapes: Vec<DynShape<T>> = triangles
        .iter()
        .map(|tri| {
            let (p1, p2, p3) = (&vertices[tri[0] as usize], &vertices[tri[1] as usize], &vertices[tri[2] as usize]);
            let n = (*p2 - *p1).cross(&(*p3 - *p1));
            let independent = n[0].abs() > tol || n[1].abs() > tol || n[2].abs() > tol;
            if independent {
                let rad = (T::from_f64(radius.to_f64()) / vox).to_f64();
                Box::new(DilatedTriangle::<T> {
                    a: to_index(p1),
                    b: to_index(p2),
                    c: to_index(p3),
                    rad,
                    orad: rad + hw.to_f64(),
                    _t: std::marker::PhantomData,
                }) as DynShape<T>
            } else {
                let d1 = (*p2 - *p1).length_sqr();
                let d2 = (*p3 - *p2).length_sqr();
                let d3 = (*p1 - *p3).length_sqr();
                let maxd = d1.max(d2).max(d3);
                let (a, b) = if maxd == d1 {
                    (p1, p2)
                } else if maxd == d2 {
                    (p2, p3)
                } else {
                    (p3, p1)
                };
                Box::new(Capsule::new(a, b, radius, vox, hw)) as DynShape<T>
            }
        })
        .collect();
    grid.set_tree(rasterize_union(&shapes, vox, hw, bg, true));
    grid
}

/// Dilated mesh of quads (each quad `(a, b, c, d)` is split into triangles `(a, b, c)` and
/// `(a, c, d)`).
pub fn create_level_set_dilated_quad_mesh<T: RealValue, S: RealValue>(
    vertices: &[Vec3<S>],
    quads: &[[u32; 4]],
    radius: S,
    voxel_size: f32,
    half_width: f32,
) -> Grid<T> {
    create_level_set_dilated_mixed_mesh(vertices, &[], quads, radius, voxel_size, half_width)
}

/// Dilated mesh of triangles and quads.
pub fn create_level_set_dilated_mixed_mesh<T: RealValue, S: RealValue>(
    vertices: &[Vec3<S>],
    triangles: &[[u32; 3]],
    quads: &[[u32; 4]],
    radius: S,
    voxel_size: f32,
    half_width: f32,
) -> Grid<T> {
    let mut tris: Vec<[u32; 3]> = triangles.to_vec();
    tris.extend(quads.iter().map(|q| [q[0], q[1], q[2]]));
    tris.extend(quads.iter().map(|q| [q[0], q[2], q[3]]));
    create_level_set_dilated_mesh(vertices, &tris, radius, voxel_size, half_width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3f;

    #[test]
    fn dilated_triangle() {
        let v = vec![Vec3f::new(0.0, 0.0, 0.0), Vec3f::new(1.0, 0.0, 0.0), Vec3f::new(0.0, 1.0, 0.0)];
        let g: Grid<f32> = create_level_set_dilated_mesh(&v, &[[0, 1, 2]], 0.1, 0.05, 3.0);
        // 0.25 above the face: distance 0.15.
        assert!((g.get_value(&Coord::new(4, 4, 5)) - 0.15).abs() < 1e-5);
        assert!(g.get_value(&Coord::new(4, 4, 0)) < 0.0);
        assert!(g.get_value(&Coord::new(4, 4, 20)) > 0.0);
        let q: Grid<f32> = create_level_set_dilated_quad_mesh(
            &[
                Vec3f::new(0.0, 0.0, 0.0),
                Vec3f::new(1.0, 0.0, 0.0),
                Vec3f::new(1.0, 1.0, 0.0),
                Vec3f::new(0.0, 1.0, 0.0),
            ],
            &[[0, 1, 2, 3]],
            0.1,
            0.05,
            3.0,
        );
        assert!((q.get_value(&Coord::new(15, 15, 5)) - 0.15).abs() < 1e-5);
    }
}
