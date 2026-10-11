//! Index-to-world maps (port of `openvdb/math/Maps.h`).
//!
//! Initial version: every linear map is stored as an affine matrix plus its inverse.

use super::mat::{Axis, Mat4d};
use super::vec3::Vec3d;
use std::fmt;

/// An affine map `world = index * M` (row-vector convention).
#[derive(Clone, PartialEq)]
pub struct AffineMap {
    matrix: Mat4d,
    inverse: Mat4d,
}

impl AffineMap {
    pub fn new(matrix: Mat4d) -> Self {
        let inverse = matrix.inverse(0.0).expect("AffineMap: singular matrix");
        AffineMap { matrix, inverse }
    }
    pub fn matrix(&self) -> &Mat4d {
        &self.matrix
    }
    pub fn inverse_matrix(&self) -> &Mat4d {
        &self.inverse
    }
}

/// Any index-to-world map.
#[derive(Clone, PartialEq)]
pub enum MapBase {
    Affine(AffineMap),
}

impl MapBase {
    pub fn uniform_scale(s: f64) -> Self {
        MapBase::Affine(AffineMap::new(Mat4d::scale(Vec3d::splat(s))))
    }
    pub fn from_matrix(m: Mat4d) -> Self {
        MapBase::Affine(AffineMap::new(m))
    }
    pub fn type_name(&self) -> &'static str {
        "AffineMap"
    }
    pub fn is_linear(&self) -> bool {
        true
    }
    /// The affine matrix of a linear map.
    pub fn affine_matrix(&self) -> Mat4d {
        match self {
            MapBase::Affine(a) => a.matrix,
        }
    }
    pub fn has_uniform_scale(&self) -> bool {
        let m3 = self.affine_matrix().get_mat3();
        let mtm = m3 * m3.transpose();
        let s = mtm.0[0][0];
        let tol = 1e-6 * s.abs().max(1.0);
        (0..3).all(|i| {
            (0..3).all(|j| {
                let e = if i == j { s } else { 0.0 };
                (mtm.0[i][j] - e).abs() <= tol
            })
        })
    }
    pub fn is_identity(&self) -> bool {
        self.affine_matrix().is_approx_equal(&Mat4d::identity(), 1e-12)
    }
    pub fn voxel_size(&self) -> Vec3d {
        let m = self.affine_matrix();
        Vec3d::new(
            m.transform3x3(Vec3d::new(1.0, 0.0, 0.0)).length(),
            m.transform3x3(Vec3d::new(0.0, 1.0, 0.0)).length(),
            m.transform3x3(Vec3d::new(0.0, 0.0, 1.0)).length(),
        )
    }
    pub fn voxel_size_at(&self, _xyz: Vec3d) -> Vec3d {
        self.voxel_size()
    }
    pub fn apply_map(&self, xyz: Vec3d) -> Vec3d {
        match self {
            MapBase::Affine(a) => a.matrix.transform(xyz),
        }
    }
    pub fn apply_inverse_map(&self, xyz: Vec3d) -> Vec3d {
        match self {
            MapBase::Affine(a) => a.inverse.transform(xyz),
        }
    }
    fn with(&self, f: impl FnOnce(&mut Mat4d)) -> MapBase {
        let mut m = self.affine_matrix();
        f(&mut m);
        MapBase::from_matrix(m)
    }
    pub fn pre_rotate(&self, angle: f64, axis: Axis) -> MapBase {
        self.with(|m| m.pre_rotate(axis, angle))
    }
    pub fn pre_translate(&self, t: Vec3d) -> MapBase {
        self.with(|m| m.pre_translate(t))
    }
    pub fn pre_scale(&self, s: Vec3d) -> MapBase {
        self.with(|m| m.pre_scale(s))
    }
    pub fn post_rotate(&self, angle: f64, axis: Axis) -> MapBase {
        self.with(|m| m.post_rotate(axis, angle))
    }
    pub fn post_translate(&self, t: Vec3d) -> MapBase {
        self.with(|m| m.post_translate(t))
    }
    pub fn post_scale(&self, s: Vec3d) -> MapBase {
        self.with(|m| m.post_scale(s))
    }
    pub fn pre_mult(&self, o: &Mat4d) -> MapBase {
        self.with(|m| *m = *o * *m)
    }
    pub fn post_mult(&self, o: &Mat4d) -> MapBase {
        self.with(|m| *m = *m * *o)
    }
    pub fn is_approx_equal(&self, o: &MapBase, tol: f64) -> bool {
        self.affine_matrix().is_approx_equal(&o.affine_matrix(), tol)
    }
}

impl MapBase {
    /// Read a map serialized by OpenVDB (`MapBase::read`), given its registered type name
    /// (e.g. `"UniformScaleMap"`), from a little-endian stream.
    pub fn read_typed<R: std::io::Read>(type_name: &str, r: &mut R) -> std::io::Result<MapBase> {
        let _ = (type_name, r);
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "map serialization not implemented"))
    }
    /// Write this map's payload as OpenVDB does (`MapBase::write`); the type name is
    /// written separately by the caller.
    pub fn write<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        let _ = w;
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "map serialization not implemented"))
    }
}

impl fmt::Debug for MapBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({:?})", self.type_name(), self.affine_matrix())
    }
}
