//! Index-to-world maps (port of `openvdb/math/Maps.h` and `Maps.cc`).
//!
//! OpenVDB models maps as a class hierarchy rooted at `MapBase` plus a registry of factory
//! functions keyed by type name. Here every concrete map is a struct and [`MapBase`] is an enum
//! over them, so the registry reduces to [`MapBase::read_typed`]:
//!
//! | C++ class                  | Rust struct                  | registered name              |
//! |----------------------------|------------------------------|------------------------------|
//! | `AffineMap`                | [`AffineMap`]                | `"AffineMap"`                |
//! | `UnitaryMap`               | [`UnitaryMap`]               | `"UnitaryMap"`               |
//! | `ScaleMap`                 | [`ScaleMap`]                 | `"ScaleMap"`                 |
//! | `UniformScaleMap`          | [`UniformScaleMap`]          | `"UniformScaleMap"`          |
//! | `TranslationMap`           | [`TranslationMap`]           | `"TranslationMap"`           |
//! | `ScaleTranslateMap`        | [`ScaleTranslateMap`]        | `"ScaleTranslateMap"`        |
//! | `UniformScaleTranslateMap` | [`UniformScaleTranslateMap`] | `"UniformScaleTranslateMap"` |
//! | `NonlinearFrustumMap`      | [`NonlinearFrustumMap`]      | `"NonlinearFrustumMap"`      |
//!
//! All arithmetic, map-type simplification rules (e.g. a `UniformScaleMap` followed by a
//! translation becomes a `UniformScaleTranslateMap`, a rotation of an affine map is passed through
//! [`simplify`]) and the binary serialization layout follow the C++ code exactly. As in C++,
//! operations that would produce a singular map (e.g. scaling by zero) panic; fallible
//! constructors are provided as `try_new`.

use super::bbox::BBoxd;
use super::mat::{Axis, Mat3d, Mat4d};
use super::math_util::{is_approx_equal, is_approx_equal_tol};
use super::vec3::Vec3d;
use std::fmt;
use std::io::{self, Read, Write};

/// Error raised when a map cannot be constructed (OpenVDB's `ArithmeticError`), e.g. from a
/// singular or non-affine matrix.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ArithmeticError(pub String);

fn arith(msg: &str) -> ArithmeticError {
    ArithmeticError(msg.to_string())
}

const TOL: f64 = super::TOLERANCE_F64;

/// `Vec3d::eq` with its default tolerance of 1e-7.
#[inline]
fn veq(a: &Vec3d, b: &Vec3d) -> bool {
    (0..3).all(|i| is_approx_equal_tol(a[i], b[i], 1.0e-7))
}

#[inline]
fn is_uniform(v: &Vec3d) -> bool {
    is_approx_equal(v[0], v[1]) && is_approx_equal(v[0], v[2])
}

// ---------------------------------------------------------------------------------------------
// Little-endian serialization helpers (OpenVDB writes raw host-order, i.e. little-endian, data).

pub(crate) fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
pub(crate) fn write_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub(crate) fn read_vec3d<R: Read>(r: &mut R) -> io::Result<Vec3d> {
    Ok(Vec3d::new(read_f64(r)?, read_f64(r)?, read_f64(r)?))
}
pub(crate) fn write_vec3d<W: Write>(w: &mut W, v: &Vec3d) -> io::Result<()> {
    (0..3).try_for_each(|i| write_f64(w, v[i]))
}
pub(crate) fn read_mat4d<R: Read>(r: &mut R) -> io::Result<Mat4d> {
    let mut m = Mat4d::zero();
    for i in 0..4 {
        for j in 0..4 {
            m.0[i][j] = read_f64(r)?;
        }
    }
    Ok(m)
}
pub(crate) fn write_mat4d<W: Write>(w: &mut W, m: &Mat4d) -> io::Result<()> {
    m.0.iter().flatten().try_for_each(|&v| write_f64(w, v))
}
/// Read an OpenVDB string: a little-endian `u32` byte count followed by the bytes.
pub(crate) fn read_name<R: Read>(r: &mut R) -> io::Result<String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    let n = u32::from_le_bytes(b) as usize;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
/// Write an OpenVDB string: a little-endian `u32` byte count followed by the bytes.
pub(crate) fn write_name<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    w.write_all(s.as_bytes())
}
fn invalid(e: ArithmeticError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

// ---------------------------------------------------------------------------------------------
// AffineMap

/// A general affine transform (rotation, scale, shear and translation) stored as a 4x4 matrix,
/// together with cached inverse, inverse Jacobian, determinant and voxel size.
#[derive(Clone, Debug)]
pub struct AffineMap {
    matrix: Mat4d,
    matrix_inv: Mat4d,
    jacobian_inv: Mat3d,
    determinant: f64,
    voxel_size: Vec3d,
    is_diagonal: bool,
    is_identity: bool,
}

impl Default for AffineMap {
    fn default() -> Self {
        Self::identity()
    }
}

impl AffineMap {
    /// The registered type name, `"AffineMap"`.
    pub const TYPE_NAME: &'static str = "AffineMap";

    /// The identity map.
    pub fn identity() -> Self {
        AffineMap {
            matrix: Mat4d::identity(),
            matrix_inv: Mat4d::identity(),
            jacobian_inv: Mat3d::identity(),
            determinant: 1.0,
            voxel_size: Vec3d::splat(1.0),
            is_diagonal: true,
            is_identity: true,
        }
    }
    /// An affine map from a 4x4 matrix, failing if the matrix is not affine (last column
    /// `(0, 0, 0, 1)`) or is nearly singular.
    pub fn try_new(m: Mat4d) -> Result<Self, ArithmeticError> {
        if !m.is_affine() {
            return Err(arith("Tried to initialize an affine transform from a non-affine 4x4 matrix"));
        }
        let mut a = AffineMap::identity();
        a.matrix = m;
        a.update_acceleration()?;
        Ok(a)
    }
    /// An affine map from a 4x4 matrix.
    ///
    /// # Panics
    /// If the matrix is not affine or is nearly singular.
    pub fn new(m: Mat4d) -> Self {
        Self::try_new(m).unwrap_or_else(|e| panic!("AffineMap: {e}"))
    }
    /// An affine map with the given 3x3 linear part and no translation.
    ///
    /// # Panics
    /// If the matrix is nearly singular.
    pub fn from_mat3(m: &Mat3d) -> Self {
        Self::new(Mat4d::from_mat3_translation(m, Vec3d::zero()))
    }
    /// The composition `first` then `second` (matrix product `first * second`).
    pub fn compose(first: &AffineMap, second: &AffineMap) -> Self {
        Self::new(first.matrix * second.matrix)
    }
    fn update_acceleration(&mut self) -> Result<(), ArithmeticError> {
        let mat3 = self.matrix.get_mat3();
        self.determinant = mat3.det();
        if self.determinant.abs() < 3.0 * TOL {
            return Err(arith("Tried to initialize an affine transform from a nearly singular matrix"));
        }
        self.matrix_inv = self.matrix.inverse(0.0).ok_or_else(|| arith("Inversion of singular 4x4 matrix"))?;
        self.jacobian_inv = mat3.inverse(0.0).ok_or_else(|| arith("Inversion of singular 3x3 matrix"))?.transpose();
        self.is_diagonal = self.matrix.is_diagonal();
        self.is_identity = self.matrix.is_identity();
        let pos = self.apply_map(Vec3d::zero());
        self.voxel_size = Vec3d::new(
            (self.apply_map(Vec3d::new(1.0, 0.0, 0.0)) - pos).length(),
            (self.apply_map(Vec3d::new(0.0, 1.0, 0.0)) - pos).length(),
            (self.apply_map(Vec3d::new(0.0, 0.0, 1.0)) - pos).length(),
        );
        Ok(())
    }
    fn modified(&self, f: impl FnOnce(&mut Mat4d)) -> Self {
        let mut m = self.matrix;
        f(&mut m);
        Self::new(m)
    }

    /// The 4x4 matrix (`getMat4`).
    pub fn matrix(&self) -> &Mat4d {
        &self.matrix
    }
    /// The 4x4 matrix (`getMat4`).
    pub fn get_mat4(&self) -> Mat4d {
        self.matrix
    }
    /// The cached inverse matrix.
    pub fn inverse_matrix(&self) -> &Mat4d {
        &self.matrix_inv
    }
    /// The cached transpose of the inverse of the 3x3 part (`getConstJacobianInv`).
    pub fn jacobian_inv(&self) -> &Mat3d {
        &self.jacobian_inv
    }
    /// True if the matrix is approximately the identity.
    pub fn is_identity(&self) -> bool {
        self.is_identity
    }
    /// True if the 4x4 matrix is diagonal (no translation either).
    pub fn is_diagonal(&self) -> bool {
        self.is_diagonal
    }
    /// True if the map is equivalent to a [`ScaleMap`].
    pub fn is_scale(&self) -> bool {
        self.is_diagonal
    }
    /// True if the map is equivalent to a [`ScaleTranslateMap`].
    pub fn is_scale_translate(&self) -> bool {
        self.matrix.get_mat3().is_diagonal()
    }
    /// True if the 3x3 part is a uniform scale times a rotation/reflection.
    pub fn has_uniform_scale(&self) -> bool {
        let mat = self.matrix.get_mat3();
        let det = mat.det();
        if is_approx_equal(det, 0.0) {
            false
        } else {
            (mat * (1.0 / det.abs().powf(1.0 / 3.0))).is_unitary()
        }
    }
    /// The inverse map.
    pub fn inverse(&self) -> AffineMap {
        AffineMap::new(self.matrix_inv)
    }

    /// `in * M`.
    #[inline]
    pub fn apply_map(&self, xyz: Vec3d) -> Vec3d {
        self.matrix.transform(xyz)
    }
    /// `in * M⁻¹`.
    #[inline]
    pub fn apply_inverse_map(&self, xyz: Vec3d) -> Vec3d {
        self.matrix_inv.transform(xyz)
    }
    /// The map without translation.
    #[inline]
    pub fn apply_jacobian(&self, v: Vec3d) -> Vec3d {
        self.matrix.transform3x3(v)
    }
    /// The inverse map without translation.
    #[inline]
    pub fn apply_inverse_jacobian(&self, v: Vec3d) -> Vec3d {
        self.matrix_inv.transform3x3(v)
    }
    /// The transpose of the Jacobian applied to `v` (range-space to domain-space gradients).
    #[inline]
    pub fn apply_jt(&self, v: Vec3d) -> Vec3d {
        let m = &self.matrix.0;
        Vec3d::new(
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        )
    }
    /// The transpose of the inverse Jacobian applied to `v`.
    #[inline]
    pub fn apply_ijt(&self, v: Vec3d) -> Vec3d {
        self.jacobian_inv.transform(v)
    }
    /// `J⁻ᵀᵀ m J⁻ᵀ`: transforms a second-derivative matrix to range space.
    #[inline]
    pub fn apply_ijc(&self, m: &Mat3d) -> Mat3d {
        self.jacobian_inv.transpose() * *m * self.jacobian_inv
    }
    /// The determinant of the Jacobian.
    #[inline]
    pub fn determinant(&self) -> f64 {
        self.determinant
    }
    /// Lengths of the images of the unit axis segments.
    #[inline]
    pub fn voxel_size(&self) -> Vec3d {
        self.voxel_size
    }

    /// `M = R * M`.
    pub fn accum_pre_rotation(&mut self, axis: Axis, radians: f64) {
        *self = self.modified(|m| m.pre_rotate(axis, radians));
    }
    /// `M = S * M`.
    pub fn accum_pre_scale(&mut self, v: Vec3d) {
        *self = self.modified(|m| m.pre_scale(v));
    }
    /// `M = T * M`.
    pub fn accum_pre_translation(&mut self, v: Vec3d) {
        *self = self.modified(|m| m.pre_translate(v));
    }
    /// `M = Shear * M`.
    pub fn accum_pre_shear(&mut self, axis0: Axis, axis1: Axis, shear: f64) {
        *self = self.modified(|m| m.pre_shear(axis0, axis1, shear));
    }
    /// `M = M * R`.
    pub fn accum_post_rotation(&mut self, axis: Axis, radians: f64) {
        *self = self.modified(|m| m.post_rotate(axis, radians));
    }
    /// `M = M * S`.
    pub fn accum_post_scale(&mut self, v: Vec3d) {
        *self = self.modified(|m| m.post_scale(v));
    }
    /// `M = M * T`.
    pub fn accum_post_translation(&mut self, v: Vec3d) {
        *self = self.modified(|m| m.post_translate(v));
    }
    /// `M = M * Shear`.
    pub fn accum_post_shear(&mut self, axis0: Axis, axis1: Axis, shear: f64) {
        *self = self.modified(|m| m.post_shear(axis0, axis1, shear));
    }

    fn pre_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        simplify(self.modified(|m| m.pre_rotate(axis, radians)))
    }
    fn pre_translate(&self, t: Vec3d) -> MapBase {
        MapBase::Affine(self.modified(|m| m.pre_translate(t)))
    }
    fn pre_scale(&self, s: Vec3d) -> MapBase {
        MapBase::Affine(self.modified(|m| m.pre_scale(s)))
    }
    fn pre_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        simplify(self.modified(|m| m.pre_shear(axis0, axis1, shear)))
    }
    fn post_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        simplify(self.modified(|m| m.post_rotate(axis, radians)))
    }
    fn post_translate(&self, t: Vec3d) -> MapBase {
        MapBase::Affine(self.modified(|m| m.post_translate(t)))
    }
    fn post_scale(&self, s: Vec3d) -> MapBase {
        MapBase::Affine(self.modified(|m| m.post_scale(s)))
    }
    fn post_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        simplify(self.modified(|m| m.post_shear(axis0, axis1, shear)))
    }
    fn get_affine_map(&self) -> AffineMap {
        self.clone()
    }
    fn inverse_map(&self) -> Option<MapBase> {
        Some(MapBase::Affine(self.inverse()))
    }

    /// Read the matrix (16 little-endian doubles) and recompute the cached data.
    pub fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let m = read_mat4d(r)?;
        let mut a = AffineMap::identity();
        a.matrix = m;
        a.update_acceleration().map_err(invalid)?;
        Ok(a)
    }
    /// Write the matrix as 16 little-endian doubles.
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_mat4d(w, &self.matrix)
    }
}

impl PartialEq for AffineMap {
    /// Approximate comparison of the matrices and their inverses (tolerance 1e-8).
    fn eq(&self, o: &Self) -> bool {
        self.matrix.eq_default(&o.matrix) && self.matrix_inv.eq_default(&o.matrix_inv)
    }
}

// ---------------------------------------------------------------------------------------------
// ScaleMap / UniformScaleMap

/// A scale along the principal axes, not necessarily uniform.
#[derive(Clone, Debug)]
pub struct ScaleMap {
    scale_values: Vec3d,
    voxel_size: Vec3d,
    scale_values_inverse: Vec3d,
    inv_scale_sqr: Vec3d,
    inv_twice_scale: Vec3d,
}

impl Default for ScaleMap {
    fn default() -> Self {
        let one = Vec3d::splat(1.0);
        ScaleMap {
            scale_values: one,
            voxel_size: one,
            scale_values_inverse: one,
            inv_scale_sqr: one,
            inv_twice_scale: Vec3d::splat(0.5),
        }
    }
}

impl ScaleMap {
    /// The registered type name, `"ScaleMap"`.
    pub const TYPE_NAME: &'static str = "ScaleMap";

    /// A scale map, failing if the product of the scale values is nearly zero.
    pub fn try_new(scale: Vec3d) -> Result<Self, ArithmeticError> {
        let det = scale[0] * scale[1] * scale[2];
        if det.abs() < 3.0 * TOL {
            return Err(arith("Non-zero scale values required"));
        }
        let inv = Vec3d::new(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2]);
        Ok(ScaleMap {
            scale_values: scale,
            voxel_size: scale.abs(),
            scale_values_inverse: inv,
            inv_scale_sqr: inv * inv,
            inv_twice_scale: inv / 2.0,
        })
    }
    /// A scale map.
    ///
    /// # Panics
    /// If the product of the scale values is nearly zero.
    pub fn new(scale: Vec3d) -> Self {
        Self::try_new(scale).unwrap_or_else(|e| panic!("ScaleMap: {e}"))
    }
    /// The scale values.
    pub fn get_scale(&self) -> Vec3d {
        self.scale_values
    }
    /// `1 / scale²`, used by finite differencing.
    pub fn get_inv_scale_sqr(&self) -> Vec3d {
        self.inv_scale_sqr
    }
    /// `1 / (2 scale)`, used by finite differencing.
    pub fn get_inv_twice_scale(&self) -> Vec3d {
        self.inv_twice_scale
    }
    /// `1 / scale`.
    pub fn get_inv_scale(&self) -> Vec3d {
        self.scale_values_inverse
    }
    /// True if the absolute scale values agree to within 5e-7.
    pub fn has_uniform_scale(&self) -> bool {
        let s = self.scale_values;
        is_approx_equal_tol(s[0].abs(), s[1].abs(), 5e-7) && is_approx_equal_tol(s[0].abs(), s[2].abs(), 5e-7)
    }
    /// `in * scale`.
    #[inline]
    pub fn apply_map(&self, v: Vec3d) -> Vec3d {
        Vec3d::new(v[0] * self.scale_values[0], v[1] * self.scale_values[1], v[2] * self.scale_values[2])
    }
    /// `in / scale`.
    #[inline]
    pub fn apply_inverse_map(&self, v: Vec3d) -> Vec3d {
        let s = &self.scale_values_inverse;
        Vec3d::new(v[0] * s[0], v[1] * s[1], v[2] * s[2])
    }
    /// Same as [`apply_map`](Self::apply_map).
    #[inline]
    pub fn apply_jacobian(&self, v: Vec3d) -> Vec3d {
        self.apply_map(v)
    }
    /// Same as [`apply_inverse_map`](Self::apply_inverse_map).
    #[inline]
    pub fn apply_inverse_jacobian(&self, v: Vec3d) -> Vec3d {
        self.apply_inverse_map(v)
    }
    /// Same as [`apply_map`](Self::apply_map).
    #[inline]
    pub fn apply_jt(&self, v: Vec3d) -> Vec3d {
        self.apply_map(v)
    }
    /// Same as [`apply_inverse_map`](Self::apply_inverse_map).
    #[inline]
    pub fn apply_ijt(&self, v: Vec3d) -> Vec3d {
        self.apply_inverse_map(v)
    }
    /// Scale rows and columns of `m` by `1 / scale`.
    pub fn apply_ijc(&self, m: &Mat3d) -> Mat3d {
        scale_ijc(m, &self.scale_values_inverse)
    }
    /// Product of the scale values.
    #[inline]
    pub fn determinant(&self) -> f64 {
        self.scale_values[0] * self.scale_values[1] * self.scale_values[2]
    }
    /// Absolute scale values.
    #[inline]
    pub fn voxel_size(&self) -> Vec3d {
        self.voxel_size
    }
    fn get_affine_map(&self) -> AffineMap {
        AffineMap::new(Mat4d::scale(self.scale_values))
    }
    fn inverse_map(&self) -> Option<MapBase> {
        Some(MapBase::Scale(ScaleMap::new(self.scale_values_inverse)))
    }
    fn pre_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_rotation(axis, radians);
        simplify(a)
    }
    fn pre_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_shear(axis0, axis1, shear);
        simplify(a)
    }
    fn post_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_rotation(axis, radians);
        simplify(a)
    }
    fn post_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_shear(axis0, axis1, shear);
        simplify(a)
    }
    fn pre_translate(&self, t: Vec3d) -> MapBase {
        let s = &self.scale_values;
        let scaled = Vec3d::new(t[0] * s[0], t[1] * s[1], t[2] * s[2]);
        MapBase::ScaleTranslate(ScaleTranslateMap::new(self.scale_values, scaled))
    }
    fn post_translate(&self, t: Vec3d) -> MapBase {
        MapBase::ScaleTranslate(ScaleTranslateMap::new(self.scale_values, t))
    }
    fn pre_scale(&self, v: Vec3d) -> MapBase {
        let new_scale = v * self.scale_values;
        if is_uniform(&new_scale) {
            MapBase::UniformScale(UniformScaleMap::new(new_scale[0]))
        } else {
            MapBase::Scale(ScaleMap::new(new_scale))
        }
    }
    fn post_scale(&self, v: Vec3d) -> MapBase {
        self.pre_scale(v)
    }
    fn read_fields<R: Read>(r: &mut R) -> io::Result<Self> {
        Ok(ScaleMap {
            scale_values: read_vec3d(r)?,
            voxel_size: read_vec3d(r)?,
            scale_values_inverse: read_vec3d(r)?,
            inv_scale_sqr: read_vec3d(r)?,
            inv_twice_scale: read_vec3d(r)?,
        })
    }
    /// Write scale, voxel size, inverse scale, inverse squared scale and inverse twice scale
    /// (15 little-endian doubles).
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_vec3d(w, &self.scale_values)?;
        write_vec3d(w, &self.voxel_size)?;
        write_vec3d(w, &self.scale_values_inverse)?;
        write_vec3d(w, &self.inv_scale_sqr)?;
        write_vec3d(w, &self.inv_twice_scale)
    }
}

fn scale_ijc(m: &Mat3d, inv: &Vec3d) -> Mat3d {
    let mut tmp = Mat3d::zero();
    for i in 0..3 {
        tmp.set_row(i, m.row(i) * inv[i]);
    }
    for i in 0..3 {
        tmp.set_col(i, tmp.col(i) * inv[i]);
    }
    tmp
}

impl PartialEq for ScaleMap {
    /// Approximate comparison of the scale values (tolerance 1e-7).
    fn eq(&self, o: &Self) -> bool {
        veq(&self.scale_values, &o.scale_values)
    }
}

/// A uniform scale along the principal axes. It shares [`ScaleMap`]'s data and serialization
/// (and dereferences to it) but simplifies differently when composed with translations.
#[derive(Clone, Debug, PartialEq)]
pub struct UniformScaleMap(ScaleMap);

impl Default for UniformScaleMap {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl std::ops::Deref for UniformScaleMap {
    type Target = ScaleMap;
    fn deref(&self) -> &ScaleMap {
        &self.0
    }
}

impl UniformScaleMap {
    /// The registered type name, `"UniformScaleMap"`.
    pub const TYPE_NAME: &'static str = "UniformScaleMap";

    /// A uniform scale map, failing if `scale³` is nearly zero.
    pub fn try_new(scale: f64) -> Result<Self, ArithmeticError> {
        Ok(UniformScaleMap(ScaleMap::try_new(Vec3d::splat(scale))?))
    }
    /// A uniform scale map.
    ///
    /// # Panics
    /// If `scale³` is nearly zero.
    pub fn new(scale: f64) -> Self {
        Self::try_new(scale).unwrap_or_else(|e| panic!("UniformScaleMap: {e}"))
    }
    /// The underlying [`ScaleMap`] data.
    pub fn as_scale_map(&self) -> &ScaleMap {
        &self.0
    }
    fn inverse_map(&self) -> Option<MapBase> {
        Some(MapBase::UniformScale(UniformScaleMap::new(self.get_inv_scale()[0])))
    }
    fn pre_translate(&self, t: Vec3d) -> MapBase {
        let scale = self.get_scale()[0];
        MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(scale, t * scale))
    }
    fn post_translate(&self, t: Vec3d) -> MapBase {
        let scale = self.get_scale()[0];
        MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(scale, t))
    }
}

// ---------------------------------------------------------------------------------------------
// TranslationMap

/// A pure translation.
#[derive(Clone, Debug, Default)]
pub struct TranslationMap {
    translation: Vec3d,
}

impl TranslationMap {
    /// The registered type name, `"TranslationMap"`.
    pub const TYPE_NAME: &'static str = "TranslationMap";

    /// A translation by `t`.
    pub fn new(t: Vec3d) -> Self {
        TranslationMap { translation: t }
    }
    /// The translation vector.
    pub fn get_translation(&self) -> Vec3d {
        self.translation
    }
    /// `in + t`.
    #[inline]
    pub fn apply_map(&self, v: Vec3d) -> Vec3d {
        v + self.translation
    }
    /// `in - t`.
    #[inline]
    pub fn apply_inverse_map(&self, v: Vec3d) -> Vec3d {
        v - self.translation
    }
    /// Identity.
    #[inline]
    pub fn apply_jacobian(&self, v: Vec3d) -> Vec3d {
        v
    }
    /// Identity.
    #[inline]
    pub fn apply_inverse_jacobian(&self, v: Vec3d) -> Vec3d {
        v
    }
    /// Identity.
    #[inline]
    pub fn apply_jt(&self, v: Vec3d) -> Vec3d {
        v
    }
    /// Identity.
    #[inline]
    pub fn apply_ijt(&self, v: Vec3d) -> Vec3d {
        v
    }
    /// Identity.
    #[inline]
    pub fn apply_ijc(&self, m: &Mat3d) -> Mat3d {
        *m
    }
    /// Always 1.
    #[inline]
    pub fn determinant(&self) -> f64 {
        1.0
    }
    /// Always (1, 1, 1).
    #[inline]
    pub fn voxel_size(&self) -> Vec3d {
        Vec3d::splat(1.0)
    }
    fn get_affine_map(&self) -> AffineMap {
        AffineMap::new(Mat4d::translation(self.translation))
    }
    fn inverse_map(&self) -> Option<MapBase> {
        Some(MapBase::Translation(TranslationMap::new(-self.translation)))
    }
    fn pre_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_rotation(axis, radians);
        simplify(a)
    }
    fn pre_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_shear(axis0, axis1, shear);
        simplify(a)
    }
    fn post_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_rotation(axis, radians);
        simplify(a)
    }
    fn post_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_shear(axis0, axis1, shear);
        simplify(a)
    }
    fn pre_translate(&self, t: Vec3d) -> MapBase {
        MapBase::Translation(TranslationMap::new(t + self.translation))
    }
    fn post_translate(&self, t: Vec3d) -> MapBase {
        MapBase::Translation(TranslationMap::new(t + self.translation))
    }
    fn pre_scale(&self, v: Vec3d) -> MapBase {
        if is_uniform(&v) {
            MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(v[0], self.translation))
        } else {
            MapBase::ScaleTranslate(ScaleTranslateMap::new(v, self.translation))
        }
    }
    fn post_scale(&self, v: Vec3d) -> MapBase {
        if is_uniform(&v) {
            MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(v[0], self.translation * v[0]))
        } else {
            MapBase::ScaleTranslate(ScaleTranslateMap::new(v, self.translation * v))
        }
    }
    /// Write the translation (3 little-endian doubles).
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_vec3d(w, &self.translation)
    }
}

impl PartialEq for TranslationMap {
    /// Approximate comparison of the translations (tolerance 1e-7).
    fn eq(&self, o: &Self) -> bool {
        veq(&self.translation, &o.translation)
    }
}

// ---------------------------------------------------------------------------------------------
// ScaleTranslateMap / UniformScaleTranslateMap

/// A scale along the principal axes followed by a translation.
#[derive(Clone, Debug)]
pub struct ScaleTranslateMap {
    translation: Vec3d,
    scale_values: Vec3d,
    voxel_size: Vec3d,
    scale_values_inverse: Vec3d,
    inv_scale_sqr: Vec3d,
    inv_twice_scale: Vec3d,
}

impl Default for ScaleTranslateMap {
    fn default() -> Self {
        let one = Vec3d::splat(1.0);
        ScaleTranslateMap {
            translation: Vec3d::zero(),
            scale_values: one,
            voxel_size: one,
            scale_values_inverse: one,
            inv_scale_sqr: one,
            inv_twice_scale: Vec3d::splat(0.5),
        }
    }
}

impl ScaleTranslateMap {
    /// The registered type name, `"ScaleTranslateMap"`.
    pub const TYPE_NAME: &'static str = "ScaleTranslateMap";

    /// A scale-then-translate map, failing if the product of the scale values is nearly zero.
    pub fn try_new(scale: Vec3d, translate: Vec3d) -> Result<Self, ArithmeticError> {
        let s = ScaleMap::try_new(scale)?;
        Ok(ScaleTranslateMap {
            translation: translate,
            scale_values: s.scale_values,
            voxel_size: s.voxel_size,
            scale_values_inverse: s.scale_values_inverse,
            inv_scale_sqr: s.inv_scale_sqr,
            inv_twice_scale: s.inv_twice_scale,
        })
    }
    /// A scale-then-translate map.
    ///
    /// # Panics
    /// If the product of the scale values is nearly zero.
    pub fn new(scale: Vec3d, translate: Vec3d) -> Self {
        Self::try_new(scale, translate).unwrap_or_else(|e| panic!("ScaleTranslateMap: {e}"))
    }
    /// The scale values.
    pub fn get_scale(&self) -> Vec3d {
        self.scale_values
    }
    /// The translation.
    pub fn get_translation(&self) -> Vec3d {
        self.translation
    }
    /// `1 / scale²`.
    pub fn get_inv_scale_sqr(&self) -> Vec3d {
        self.inv_scale_sqr
    }
    /// `1 / (2 scale)`.
    pub fn get_inv_twice_scale(&self) -> Vec3d {
        self.inv_twice_scale
    }
    /// `1 / scale`.
    pub fn get_inv_scale(&self) -> Vec3d {
        self.scale_values_inverse
    }
    /// True if the absolute scale values agree to within 5e-7.
    pub fn has_uniform_scale(&self) -> bool {
        let s = self.scale_values;
        is_approx_equal_tol(s[0].abs(), s[1].abs(), 5e-7) && is_approx_equal_tol(s[0].abs(), s[2].abs(), 5e-7)
    }
    /// `in * scale + t`.
    #[inline]
    pub fn apply_map(&self, v: Vec3d) -> Vec3d {
        let (s, t) = (&self.scale_values, &self.translation);
        Vec3d::new(v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2])
    }
    /// `(in - t) / scale`.
    #[inline]
    pub fn apply_inverse_map(&self, v: Vec3d) -> Vec3d {
        let (s, t) = (&self.scale_values_inverse, &self.translation);
        Vec3d::new((v[0] - t[0]) * s[0], (v[1] - t[1]) * s[1], (v[2] - t[2]) * s[2])
    }
    /// `in * scale`.
    #[inline]
    pub fn apply_jacobian(&self, v: Vec3d) -> Vec3d {
        v * self.scale_values
    }
    /// `in / scale`.
    #[inline]
    pub fn apply_inverse_jacobian(&self, v: Vec3d) -> Vec3d {
        v * self.scale_values_inverse
    }
    /// `in * scale`.
    #[inline]
    pub fn apply_jt(&self, v: Vec3d) -> Vec3d {
        self.apply_jacobian(v)
    }
    /// `in / scale`.
    #[inline]
    pub fn apply_ijt(&self, v: Vec3d) -> Vec3d {
        let s = &self.scale_values_inverse;
        Vec3d::new(v[0] * s[0], v[1] * s[1], v[2] * s[2])
    }
    /// Scale rows and columns of `m` by `1 / scale`.
    pub fn apply_ijc(&self, m: &Mat3d) -> Mat3d {
        scale_ijc(m, &self.scale_values_inverse)
    }
    /// Product of the scale values.
    #[inline]
    pub fn determinant(&self) -> f64 {
        self.scale_values[0] * self.scale_values[1] * self.scale_values[2]
    }
    /// Absolute scale values.
    #[inline]
    pub fn voxel_size(&self) -> Vec3d {
        self.voxel_size
    }
    fn get_affine_map(&self) -> AffineMap {
        let mut a = AffineMap::new(Mat4d::scale(self.scale_values));
        a.accum_post_translation(self.translation);
        a
    }
    fn inverse_map(&self) -> Option<MapBase> {
        let inv = self.scale_values_inverse;
        Some(MapBase::ScaleTranslate(ScaleTranslateMap::new(inv, -inv * self.translation)))
    }
    fn pre_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_rotation(axis, radians);
        simplify(a)
    }
    fn pre_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_shear(axis0, axis1, shear);
        simplify(a)
    }
    fn post_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_rotation(axis, radians);
        simplify(a)
    }
    fn post_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_shear(axis0, axis1, shear);
        simplify(a)
    }
    fn pre_translate(&self, t: Vec3d) -> MapBase {
        let s = &self.scale_values;
        let scaled = Vec3d::new(t[0] * s[0], t[1] * s[1], t[2] * s[2]);
        MapBase::ScaleTranslate(ScaleTranslateMap::new(self.scale_values, self.translation + scaled))
    }
    fn post_translate(&self, t: Vec3d) -> MapBase {
        MapBase::ScaleTranslate(ScaleTranslateMap::new(self.scale_values, self.translation + t))
    }
    fn pre_scale(&self, v: Vec3d) -> MapBase {
        let new_scale = v * self.scale_values;
        if is_uniform(&new_scale) {
            MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(new_scale[0], self.translation))
        } else {
            MapBase::ScaleTranslate(ScaleTranslateMap::new(new_scale, self.translation))
        }
    }
    fn post_scale(&self, v: Vec3d) -> MapBase {
        let new_scale = v * self.scale_values;
        let t = &self.translation;
        let new_trans = Vec3d::new(t[0] * v[0], t[1] * v[1], t[2] * v[2]);
        if is_uniform(&new_scale) {
            MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(new_scale[0], new_trans))
        } else {
            MapBase::ScaleTranslate(ScaleTranslateMap::new(new_scale, new_trans))
        }
    }
    fn read_fields<R: Read>(r: &mut R) -> io::Result<Self> {
        Ok(ScaleTranslateMap {
            translation: read_vec3d(r)?,
            scale_values: read_vec3d(r)?,
            voxel_size: read_vec3d(r)?,
            scale_values_inverse: read_vec3d(r)?,
            inv_scale_sqr: read_vec3d(r)?,
            inv_twice_scale: read_vec3d(r)?,
        })
    }
    /// Write translation, scale, voxel size, inverse scale, inverse squared scale and inverse
    /// twice scale (18 little-endian doubles).
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_vec3d(w, &self.translation)?;
        write_vec3d(w, &self.scale_values)?;
        write_vec3d(w, &self.voxel_size)?;
        write_vec3d(w, &self.scale_values_inverse)?;
        write_vec3d(w, &self.inv_scale_sqr)?;
        write_vec3d(w, &self.inv_twice_scale)
    }
}

impl PartialEq for ScaleTranslateMap {
    /// Approximate comparison of scale and translation (tolerance 1e-7).
    fn eq(&self, o: &Self) -> bool {
        veq(&self.scale_values, &o.scale_values) && veq(&self.translation, &o.translation)
    }
}

/// A uniform scale followed by a translation. It shares [`ScaleTranslateMap`]'s data and
/// serialization (and dereferences to it).
#[derive(Clone, Debug, PartialEq)]
pub struct UniformScaleTranslateMap(ScaleTranslateMap);

impl Default for UniformScaleTranslateMap {
    fn default() -> Self {
        Self::new(1.0, Vec3d::zero())
    }
}

impl std::ops::Deref for UniformScaleTranslateMap {
    type Target = ScaleTranslateMap;
    fn deref(&self) -> &ScaleTranslateMap {
        &self.0
    }
}

impl UniformScaleTranslateMap {
    /// The registered type name, `"UniformScaleTranslateMap"`.
    pub const TYPE_NAME: &'static str = "UniformScaleTranslateMap";

    /// A uniform scale followed by a translation, failing if `scale³` is nearly zero.
    pub fn try_new(scale: f64, translate: Vec3d) -> Result<Self, ArithmeticError> {
        Ok(UniformScaleTranslateMap(ScaleTranslateMap::try_new(Vec3d::splat(scale), translate)?))
    }
    /// A uniform scale followed by a translation.
    ///
    /// # Panics
    /// If `scale³` is nearly zero.
    pub fn new(scale: f64, translate: Vec3d) -> Self {
        Self::try_new(scale, translate).unwrap_or_else(|e| panic!("UniformScaleTranslateMap: {e}"))
    }
    /// The underlying [`ScaleTranslateMap`] data.
    pub fn as_scale_translate_map(&self) -> &ScaleTranslateMap {
        &self.0
    }
    fn inverse_map(&self) -> Option<MapBase> {
        let s = self.get_inv_scale()[0];
        Some(MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(s, self.get_translation() * -s)))
    }
    fn pre_translate(&self, t: Vec3d) -> MapBase {
        let scale = self.get_scale()[0];
        let new_trans = self.get_translation() + t * scale;
        MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(scale, new_trans))
    }
    fn post_translate(&self, t: Vec3d) -> MapBase {
        let scale = self.get_scale()[0];
        MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(scale, self.get_translation() + t))
    }
}

// ---------------------------------------------------------------------------------------------
// UnitaryMap

/// A rotation and/or reflection.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnitaryMap {
    affine: AffineMap,
}

impl UnitaryMap {
    /// The registered type name, `"UnitaryMap"`.
    pub const TYPE_NAME: &'static str = "UnitaryMap";

    /// The identity.
    pub fn identity() -> Self {
        Self::default()
    }
    /// Rotation by `radians` about a coordinate axis.
    pub fn from_axis(axis: Axis, radians: f64) -> Self {
        UnitaryMap { affine: AffineMap::new(Mat4d::rotation(axis, radians)) }
    }
    /// Rotation by `radians` about an arbitrary axis.
    pub fn from_axis_angle(axis: Vec3d, radians: f64) -> Self {
        UnitaryMap { affine: AffineMap::from_mat3(&Mat3d::rotation_axis_angle(axis, radians)) }
    }
    /// A unitary map from a 3x3 matrix, failing if it is not a rotation or reflection.
    pub fn try_from_mat3(m: &Mat3d) -> Result<Self, ArithmeticError> {
        if !m.is_unitary() {
            return Err(arith("Matrix initializing unitary map was not unitary"));
        }
        Ok(UnitaryMap { affine: AffineMap::try_new(Mat4d::from_mat3_translation(m, Vec3d::zero()))? })
    }
    /// A unitary map from a 4x4 matrix, failing if it is not invertible, not affine, has a
    /// translation or its 3x3 part is not unitary.
    pub fn try_from_mat4(m: &Mat4d) -> Result<Self, ArithmeticError> {
        if !m.is_invertible() {
            return Err(arith("4x4 Matrix initializing unitary map was not unitary: not invertible"));
        }
        if !m.is_affine() {
            return Err(arith("4x4 Matrix initializing unitary map was not unitary: not affine"));
        }
        if m.has_translation() {
            return Err(arith("4x4 Matrix initializing unitary map was not unitary: had translation"));
        }
        if !m.get_mat3().is_unitary() {
            return Err(arith("4x4 Matrix initializing unitary map was not unitary"));
        }
        Ok(UnitaryMap { affine: AffineMap::try_new(*m)? })
    }
    /// The composition `first` then `second`.
    pub fn compose(first: &UnitaryMap, second: &UnitaryMap) -> Self {
        UnitaryMap { affine: AffineMap::compose(&first.affine, &second.affine) }
    }
    /// The underlying affine map.
    pub fn affine(&self) -> &AffineMap {
        &self.affine
    }
    /// Index to world.
    #[inline]
    pub fn apply_map(&self, v: Vec3d) -> Vec3d {
        self.affine.apply_map(v)
    }
    /// World to index.
    #[inline]
    pub fn apply_inverse_map(&self, v: Vec3d) -> Vec3d {
        self.affine.apply_inverse_map(v)
    }
    /// The rotation applied to `v`.
    #[inline]
    pub fn apply_jacobian(&self, v: Vec3d) -> Vec3d {
        self.affine.apply_jacobian(v)
    }
    /// The inverse rotation applied to `v`.
    #[inline]
    pub fn apply_inverse_jacobian(&self, v: Vec3d) -> Vec3d {
        self.affine.apply_inverse_jacobian(v)
    }
    /// The transpose, i.e. the inverse map, applied to `v`.
    #[inline]
    pub fn apply_jt(&self, v: Vec3d) -> Vec3d {
        self.apply_inverse_map(v)
    }
    /// The inverse transpose (the rotation itself) applied to `v`.
    #[inline]
    pub fn apply_ijt(&self, v: Vec3d) -> Vec3d {
        self.affine.apply_ijt(v)
    }
    /// See [`AffineMap::apply_ijc`].
    #[inline]
    pub fn apply_ijc(&self, m: &Mat3d) -> Mat3d {
        self.affine.apply_ijc(m)
    }
    /// The determinant (±1).
    #[inline]
    pub fn determinant(&self) -> f64 {
        self.affine.determinant()
    }
    /// The voxel size (1, 1, 1 up to rounding).
    #[inline]
    pub fn voxel_size(&self) -> Vec3d {
        self.affine.voxel_size()
    }
    fn get_affine_map(&self) -> AffineMap {
        self.affine.clone()
    }
    fn inverse_map(&self) -> Option<MapBase> {
        let inv = self.affine.matrix().inverse(0.0)?;
        UnitaryMap::try_from_mat4(&inv).ok().map(MapBase::Unitary)
    }
    fn pre_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        MapBase::Unitary(UnitaryMap::compose(&UnitaryMap::from_axis(axis, radians), self))
    }
    fn post_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        MapBase::Unitary(UnitaryMap::compose(self, &UnitaryMap::from_axis(axis, radians)))
    }
    fn pre_translate(&self, t: Vec3d) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_translation(t);
        simplify(a)
    }
    fn pre_scale(&self, v: Vec3d) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_scale(v);
        simplify(a)
    }
    fn pre_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_pre_shear(axis0, axis1, shear);
        simplify(a)
    }
    fn post_translate(&self, t: Vec3d) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_translation(t);
        simplify(a)
    }
    fn post_scale(&self, v: Vec3d) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_scale(v);
        simplify(a)
    }
    fn post_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        let mut a = self.get_affine_map();
        a.accum_post_shear(axis0, axis1, shear);
        simplify(a)
    }
}

// ---------------------------------------------------------------------------------------------
// NonlinearFrustumMap

/// A frustum map: an index-space box is mapped onto a truncated pyramid (with near-plane width
/// 1 and the given depth and taper), which is then transformed by a linear second map.
#[derive(Clone, Debug)]
pub struct NonlinearFrustumMap {
    bbox: BBoxd,
    taper: f64,
    depth: f64,
    second_map: AffineMap,
    lx: f64,
    ly: f64,
    lz: f64,
    xo: f64,
    yo: f64,
    gamma: f64,
    depth_on_lz: f64,
    depth_on_lz_lx_lx: f64,
    has_simple_affine: bool,
}

impl Default for NonlinearFrustumMap {
    fn default() -> Self {
        Self::new(BBoxd::new(Vec3d::zero(), Vec3d::splat(1.0)), 1.0, 1.0)
    }
}

impl NonlinearFrustumMap {
    /// The registered type name, `"NonlinearFrustumMap"`.
    pub const TYPE_NAME: &'static str = "NonlinearFrustumMap";

    fn build(bbox: BBoxd, taper: f64, depth: f64, second_map: AffineMap) -> Result<Self, ArithmeticError> {
        let mut m = NonlinearFrustumMap {
            bbox,
            taper,
            depth,
            second_map,
            lx: 0.0,
            ly: 0.0,
            lz: 0.0,
            xo: 0.0,
            yo: 0.0,
            gamma: 0.0,
            depth_on_lz: 0.0,
            depth_on_lz_lx_lx: 0.0,
            has_simple_affine: true,
        };
        m.init()?;
        Ok(m)
    }
    /// A frustum mapping the index-space `bbox` with the given `taper` (near-plane width over
    /// far-plane width) and `depth` (in units of the near-plane width), with an identity second
    /// map. Fails if the box is degenerate.
    pub fn try_new(bbox: BBoxd, taper: f64, depth: f64) -> Result<Self, ArithmeticError> {
        Self::build(bbox, taper, depth, AffineMap::identity())
    }
    /// See [`try_new`](Self::try_new).
    ///
    /// # Panics
    /// If the box is degenerate.
    pub fn new(bbox: BBoxd, taper: f64, depth: f64) -> Self {
        Self::try_new(bbox, taper, depth).unwrap_or_else(|e| panic!("NonlinearFrustumMap: {e}"))
    }
    /// A frustum followed by a linear `second_map`. Fails if `second_map` is not linear or the
    /// box is degenerate.
    pub fn try_with_second_map(
        bbox: BBoxd,
        taper: f64,
        depth: f64,
        second_map: &MapBase,
    ) -> Result<Self, ArithmeticError> {
        if !second_map.is_linear() {
            return Err(arith("The second map in the Frustum transfrom must be linear"));
        }
        Self::build(bbox, taper, depth, second_map.get_affine_map())
    }
    /// See [`try_with_second_map`](Self::try_with_second_map).
    ///
    /// # Panics
    /// If `second_map` is not linear or the box is degenerate.
    pub fn with_second_map(bbox: BBoxd, taper: f64, depth: f64, second_map: &MapBase) -> Self {
        Self::try_with_second_map(bbox, taper, depth, second_map).unwrap_or_else(|e| panic!("NonlinearFrustumMap: {e}"))
    }
    /// A frustum from a camera description.
    ///
    /// * `position`: the tip of the frustum (the camera position)
    /// * `direction`: a vector from `position` toward the near plane
    /// * `up`: a vector orthogonal to `direction` whose length is half the near-plane height
    /// * `aspect`: near-plane width / height
    /// * `z_near`, `depth`: distances from `position` to the near plane and from the near to the
    ///   far plane
    /// * `x_count`, `z_count`: the number of voxels across the near plane and along the depth
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_camera(
        position: Vec3d,
        direction: Vec3d,
        up: Vec3d,
        aspect: f64,
        z_near: f64,
        depth: f64,
        x_count: i32,
        z_count: i32,
    ) -> Result<Self, ArithmeticError> {
        if !(depth > 0.0) {
            return Err(arith("The frustum depth must be non-zero and positive"));
        }
        if !(up.length() > 0.0) {
            return Err(arith("The frustum height must be non-zero and positive"));
        }
        if !(aspect > 0.0) {
            return Err(arith("The frustum aspect ratio  must be non-zero and positive"));
        }
        if !is_approx_equal(up.dot(&direction), 0.0) {
            return Err(arith("The frustum up orientation must be perpendicular to into-frustum direction"));
        }
        let near_plane_height = 2.0 * up.length();
        let near_plane_width = aspect * near_plane_height;
        let y_count = (x_count as f64 / aspect + 0.5).floor() as i32;
        let bbox = BBoxd::new(Vec3d::zero(), Vec3d::new(x_count as f64, y_count as f64, z_count as f64));
        let fdepth = depth / near_plane_width;
        let gamma = near_plane_width / z_near;
        let taper = 1.0 / (fdepth * gamma + 1.0);

        let mut direction_unit = direction;
        let d = direction_unit.length();
        if !is_approx_equal_tol(d, 0.0, 1.0e-7) {
            direction_unit *= 1.0 / d;
        }
        let r1 = Mat4d::rotation_from_to(Vec3d::new(0.0, 0.0, 1.0), direction_unit);
        let temp = r1.inverse(0.0).ok_or_else(|| arith("Inversion of singular 4x4 matrix"))?.transform(up);
        let r2 = Mat4d::rotation_from_to(Vec3d::new(0.0, 1.0, 0.0), temp);
        let scale = Mat4d::scale(Vec3d::splat(near_plane_width));
        let mut mat = scale * r2 * r1;
        mat.set_translation(position + direction_unit * z_near);
        Self::build(bbox, taper, fdepth, AffineMap::try_new(mat)?)
    }

    fn init(&mut self) -> Result<(), ArithmeticError> {
        let e = self.bbox.extents();
        self.lx = e[0];
        self.ly = e[1];
        self.lz = e[2];
        if is_approx_equal(self.lx, 0.0) || is_approx_equal(self.ly, 0.0) || is_approx_equal(self.lz, 0.0) {
            return Err(arith("The index space bounding box must have at least two index points in each direction."));
        }
        self.xo = 0.5 * self.lx;
        self.yo = 0.5 * self.ly;
        self.gamma = (1.0 / self.taper - 1.0) / self.depth;
        self.depth_on_lz = self.depth / self.lz;
        self.depth_on_lz_lx_lx = self.depth_on_lz / (self.lx * self.lx);

        self.has_simple_affine = true;
        let tmp = self.second_map.voxel_size();
        if !is_approx_equal(tmp[0], tmp[1]) || !is_approx_equal(tmp[0], tmp[2]) {
            self.has_simple_affine = false;
            return Ok(());
        }
        let trans = self.second_map.apply_map(Vec3d::zero());
        let t1 = self.second_map.apply_map(Vec3d::new(1.0, 0.0, 0.0)) - trans;
        let t2 = self.second_map.apply_map(Vec3d::new(0.0, 1.0, 0.0)) - trans;
        let t3 = self.second_map.apply_map(Vec3d::new(0.0, 0.0, 1.0)) - trans;
        if !is_approx_equal_tol(t1.dot(&t2), 0.0, 1.0e-7)
            || !is_approx_equal_tol(t2.dot(&t3), 0.0, 1.0e-7)
            || !is_approx_equal_tol(t3.dot(&t1), 0.0, 1.0e-7)
        {
            self.has_simple_affine = false;
        }
        Ok(())
    }

    /// The taper: near-plane width / far-plane width.
    pub fn get_taper(&self) -> f64 {
        self.taper
    }
    /// Set the taper.
    pub fn set_taper(&mut self, t: f64) {
        self.taper = t;
        self.init().expect("NonlinearFrustumMap::set_taper");
    }
    /// The depth, in units of the near-plane width.
    pub fn get_depth(&self) -> f64 {
        self.depth
    }
    /// Set the depth.
    pub fn set_depth(&mut self, d: f64) {
        self.depth = d;
        self.init().expect("NonlinearFrustumMap::set_depth");
    }
    /// Near-plane width / camera-to-near-plane distance.
    pub fn get_gamma(&self) -> f64 {
        self.gamma
    }
    /// The index-space box that is mapped onto the frustum.
    pub fn get_bbox(&self) -> &BBoxd {
        &self.bbox
    }
    /// The linear second map.
    pub fn second_map(&self) -> &AffineMap {
        &self.second_map
    }
    /// True if the index-space box is not empty.
    pub fn is_valid(&self) -> bool {
        !self.bbox.is_empty()
    }
    /// True if the second map is a uniform scale, rotation and translation (no shear).
    pub fn has_simple_affine(&self) -> bool {
        self.has_simple_affine
    }
    /// True if the map is equivalent to the identity.
    pub fn is_identity(&self) -> bool {
        if !is_approx_equal(self.taper, 1.0) {
            return false;
        }
        [Vec3d::new(1.0, 0.0, 0.0), Vec3d::new(0.0, 1.0, 0.0), Vec3d::new(0.0, 0.0, 1.0)]
            .iter()
            .all(|e| veq(&self.apply_map(*e), e))
    }

    fn apply_frustum_map(&self, v: Vec3d) -> Vec3d {
        let mut out = v - self.bbox.min();
        out[0] -= self.xo;
        out[1] -= self.yo;
        out[2] *= self.depth_on_lz;
        let scale = (self.gamma * out[2] + 1.0) / self.lx;
        out[0] *= scale;
        out[1] *= scale;
        out
    }
    fn apply_frustum_inverse_map(&self, v: Vec3d) -> Vec3d {
        let mut out = v;
        let inv_scale = self.lx / (self.gamma * out[2] + 1.0);
        out[0] *= inv_scale;
        out[1] *= inv_scale;
        out[0] += self.xo;
        out[1] += self.yo;
        out[2] /= self.depth_on_lz;
        out + self.bbox.min()
    }
    fn centered(&self, isloc: Vec3d) -> Vec3d {
        let mut c = isloc - self.bbox.min();
        c[0] -= self.xo;
        c[1] -= self.yo;
        c
    }
    /// Index to world.
    pub fn apply_map(&self, v: Vec3d) -> Vec3d {
        self.second_map.apply_map(self.apply_frustum_map(v))
    }
    /// World to index.
    pub fn apply_inverse_map(&self, v: Vec3d) -> Vec3d {
        self.apply_frustum_inverse_map(self.second_map.apply_inverse_map(v))
    }
    /// The Jacobian at index-space location `isloc` applied to `v`.
    pub fn apply_jacobian_at(&self, v: Vec3d, isloc: Vec3d) -> Vec3d {
        let c = self.centered(isloc);
        let zprime = c[2] * self.depth_on_lz;
        let scale = (self.gamma * zprime + 1.0) / self.lx;
        let scale2 = self.gamma * self.depth_on_lz / self.lx;
        let tmp = Vec3d::new(
            scale * v[0] + scale2 * c[0] * v[2],
            scale * v[1] + scale2 * c[1] * v[2],
            self.depth_on_lz * v[2],
        );
        self.second_map.apply_jacobian(tmp)
    }
    /// The inverse Jacobian at index-space location `isloc` applied to `v`.
    pub fn apply_inverse_jacobian_at(&self, v: Vec3d, isloc: Vec3d) -> Vec3d {
        let c = self.centered(isloc);
        let zprime = c[2] * self.depth_on_lz;
        let scale = (self.gamma * zprime + 1.0) / self.lx;
        let scale2 = self.gamma * self.depth_on_lz / self.lx;
        let mut out = self.second_map.apply_inverse_jacobian(v);
        out[0] = (out[0] - scale2 * c[0] * out[2] / self.depth_on_lz) / scale;
        out[1] = (out[1] - scale2 * c[1] * out[2] / self.depth_on_lz) / scale;
        out[2] /= self.depth_on_lz;
        out
    }
    /// The Jacobian transpose at index-space location `isloc` applied to `v`.
    pub fn apply_jt_at(&self, v: Vec3d, isloc: Vec3d) -> Vec3d {
        let tmp = self.second_map.apply_jt(v);
        let c = self.centered(isloc);
        let zprime = c[2] * self.depth_on_lz;
        let scale = (self.gamma * zprime + 1.0) / self.lx;
        let scale2 = self.gamma * self.depth_on_lz / self.lx;
        Vec3d::new(
            scale * tmp[0],
            scale * tmp[1],
            scale2 * c[0] * tmp[0] + scale2 * c[1] * tmp[1] + self.depth_on_lz * tmp[2],
        )
    }
    fn grad_e(&self, loc: &Vec3d) -> (Mat3d, f64, f64, f64) {
        let s = self.gamma * loc[2] + 1.0;
        assert!(
            !is_approx_equal(s, 0.0),
            "Tried to evaluate the frustum transform at the singular focal point (e.g. camera)"
        );
        let sinv = 1.0 / s;
        let pt0 = self.lx * sinv;
        let pt1 = self.gamma * pt0;
        let pt2 = pt1 * sinv;
        let jacinv = self.second_map.jacobian_inv();
        let mut g = Mat3d::zero();
        for j in 0..3 {
            g.0[0][j] = pt0 * jacinv.0[0][j] - pt2 * loc[0] * jacinv.0[2][j];
            g.0[1][j] = pt0 * jacinv.0[1][j] - pt2 * loc[1] * jacinv.0[2][j];
            g.0[2][j] = (1.0 / self.depth_on_lz) * jacinv.0[2][j];
        }
        (g, sinv, pt2, pt2 * sinv)
    }
    /// The transpose of the inverse Jacobian at index-space location `ijk` applied to the
    /// index-space gradient `d1_is`.
    ///
    /// # Panics
    /// At the singular focal point of the frustum.
    pub fn apply_ijt_at(&self, d1_is: Vec3d, ijk: Vec3d) -> Vec3d {
        let loc = self.apply_frustum_map(ijk);
        let (g, ..) = self.grad_e(&loc);
        let mut r = Vec3d::zero();
        for i in 0..3 {
            r[i] = d1_is[0] * g.0[0][i] + d1_is[1] * g.0[1][i] + d1_is[2] * g.0[2][i];
        }
        r
    }
    /// The second derivatives in range space from the index-space second derivatives `d2_is`
    /// and gradient `d1_is` at index-space location `ijk`.
    ///
    /// # Panics
    /// At the singular focal point of the frustum.
    pub fn apply_ijc_at(&self, d2_is: &Mat3d, d1_is: Vec3d, ijk: Vec3d) -> Mat3d {
        let loc = self.apply_frustum_map(ijk);
        let (grad_e, _sinv, pt2, pt3) = self.grad_e(&loc);
        let jacinv = self.second_map.jacobian_inv().0;
        let mut mat_e0 = Mat3d::zero();
        let mut mat_e1 = Mat3d::zero();
        for j in 0..3 {
            for k in 0..3 {
                let pt4 = 2.0 * jacinv[2][j] * jacinv[2][k] * pt3;
                mat_e0.0[j][k] = -(jacinv[0][j] * jacinv[2][k] + jacinv[2][j] * jacinv[0][k]) * pt2 + pt4 * loc[0];
                mat_e1.0[j][k] = -(jacinv[1][j] * jacinv[2][k] + jacinv[2][j] * jacinv[1][k]) * pt2 + pt4 * loc[1];
            }
        }
        let mut result = Mat3d::zero();
        for m in 0..3 {
            for n in 0..3 {
                for i in 0..3 {
                    for j in 0..3 {
                        result.0[m][n] += grad_e.0[j][m] * grad_e.0[i][n] * d2_is.0[i][j];
                    }
                }
            }
        }
        for m in 0..3 {
            for n in 0..3 {
                result.0[m][n] += mat_e0.0[m][n] * d1_is[0] + mat_e1.0[m][n] * d1_is[1];
            }
        }
        result
    }
    /// The determinant of the Jacobian at index-space location `loc`.
    pub fn determinant_at(&self, loc: Vec3d) -> f64 {
        let s = self.gamma * loc[2] + 1.0;
        let frustum_determinant = s * s * self.depth_on_lz_lx_lx;
        self.second_map.determinant() * frustum_determinant
    }
    /// The voxel size at the center of the near plane.
    pub fn voxel_size(&self) -> Vec3d {
        let (lo, hi) = (self.bbox.min(), self.bbox.max());
        self.voxel_size_at(Vec3d::new(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]))
    }
    /// Lengths of the images of the unit segments from index-space location `loc`.
    pub fn voxel_size_at(&self, loc: Vec3d) -> Vec3d {
        let pos = self.apply_map(loc);
        Vec3d::new(
            (self.apply_map(loc + Vec3d::new(1.0, 0.0, 0.0)) - pos).length(),
            (self.apply_map(loc + Vec3d::new(0.0, 1.0, 0.0)) - pos).length(),
            (self.apply_map(loc + Vec3d::new(0.0, 0.0, 1.0)) - pos).length(),
        )
    }
    /// The second map's Jacobian applied to `v`.
    pub fn apply_jacobian(&self, v: Vec3d) -> Vec3d {
        self.second_map.apply_jacobian(v)
    }
    /// The second map's inverse Jacobian applied to `v`.
    pub fn apply_inverse_jacobian(&self, v: Vec3d) -> Vec3d {
        self.second_map.apply_inverse_jacobian(v)
    }
    /// The second map's Jacobian transpose applied to `v`.
    pub fn apply_jt(&self, v: Vec3d) -> Vec3d {
        self.second_map.apply_jt(v)
    }
    /// The second map's inverse Jacobian transpose applied to `v`.
    pub fn apply_ijt(&self, v: Vec3d) -> Vec3d {
        self.second_map.apply_ijt(v)
    }
    /// The second map's Jacobian curvature.
    pub fn apply_ijc(&self, m: &Mat3d) -> Mat3d {
        self.second_map.apply_ijc(m)
    }
    /// The second map's determinant.
    pub fn determinant(&self) -> f64 {
        self.second_map.determinant()
    }
    fn get_affine_map(&self) -> AffineMap {
        self.second_map.clone()
    }
    fn inverse_map(&self) -> Option<MapBase> {
        None
    }
    fn with_second(&self, second: MapBase) -> MapBase {
        MapBase::NonlinearFrustum(NonlinearFrustumMap::with_second_map(self.bbox, self.taper, self.depth, &second))
    }
    fn pre_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        self.with_second(self.second_map.pre_rotate(radians, axis))
    }
    fn pre_translate(&self, t: Vec3d) -> MapBase {
        self.with_second(self.second_map.pre_translate(t))
    }
    fn pre_scale(&self, s: Vec3d) -> MapBase {
        self.with_second(self.second_map.pre_scale(s))
    }
    fn pre_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        self.with_second(self.second_map.pre_shear(shear, axis0, axis1))
    }
    fn post_rotate(&self, radians: f64, axis: Axis) -> MapBase {
        self.with_second(self.second_map.post_rotate(radians, axis))
    }
    fn post_translate(&self, t: Vec3d) -> MapBase {
        self.with_second(self.second_map.post_translate(t))
    }
    fn post_scale(&self, s: Vec3d) -> MapBase {
        self.with_second(self.second_map.post_scale(s))
    }
    fn post_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        self.with_second(self.second_map.post_shear(shear, axis0, axis1))
    }

    /// Read bbox, taper, depth and the typed second map.
    pub fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let min = read_vec3d(r)?;
        let max = read_vec3d(r)?;
        let taper = read_f64(r)?;
        let depth = read_f64(r)?;
        let type_name = read_name(r)?;
        let second = MapBase::read_typed(&type_name, r)?;
        if !second.is_linear() {
            return Err(invalid(arith("The second map in the Frustum transfrom must be linear")));
        }
        Self::build(BBoxd::new(min, max), taper, depth, second.get_affine_map()).map_err(invalid)
    }
    /// Write bbox (6 doubles), taper, depth, the second map's type name and the second map.
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_vec3d(w, &self.bbox.min())?;
        write_vec3d(w, &self.bbox.max())?;
        write_f64(w, self.taper)?;
        write_f64(w, self.depth)?;
        write_name(w, AffineMap::TYPE_NAME)?;
        self.second_map.write(w)
    }
}

impl PartialEq for NonlinearFrustumMap {
    /// Approximate comparison of bbox, taper, depth and the second map's action on the origin
    /// and the unit axes.
    fn eq(&self, o: &Self) -> bool {
        if self.bbox != o.bbox || !is_approx_equal(self.taper, o.taper) || !is_approx_equal(self.depth, o.depth) {
            return false;
        }
        [Vec3d::zero(), Vec3d::new(1.0, 0.0, 0.0), Vec3d::new(0.0, 1.0, 0.0), Vec3d::new(0.0, 0.0, 1.0)]
            .iter()
            .all(|e| veq(&self.second_map.apply_map(*e), &o.second_map.apply_map(*e)))
    }
}

// ---------------------------------------------------------------------------------------------
// Free functions

/// Reduce an affine map to a [`ScaleMap`]/[`UniformScaleMap`] or a
/// [`ScaleTranslateMap`]/[`UniformScaleTranslateMap`] when possible (`math::simplify`).
pub fn simplify(affine: AffineMap) -> MapBase {
    if affine.is_scale() {
        let scale = affine.apply_map(Vec3d::splat(1.0));
        if is_uniform(&scale) {
            MapBase::UniformScale(UniformScaleMap::new(scale[0]))
        } else {
            MapBase::Scale(ScaleMap::new(scale))
        }
    } else if affine.is_scale_translate() {
        let translate = affine.apply_map(Vec3d::zero());
        let scale = affine.apply_map(Vec3d::splat(1.0)) - translate;
        if is_uniform(&scale) {
            MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(scale[0], translate))
        } else {
            MapBase::ScaleTranslate(ScaleTranslateMap::new(scale, translate))
        }
    } else {
        MapBase::Affine(affine)
    }
}

/// The inverse of `m` if it is invertible; otherwise, if its 3x3 part is symmetric, a
/// pseudo-inverse built from its eigen-decomposition, else a matrix with a zero 3x3 part and the
/// negated translation (`math::approxInverse`).
pub fn approx_inverse(m: &Mat4d) -> Mat4d {
    if m.det().abs() >= 3.0 * TOL {
        if let Some(inv) = m.inverse(0.0) {
            return inv;
        }
    }
    let mat3 = m.get_mat3();
    let mat3t = mat3.transpose();
    let trans = m.get_translation();
    let symmetric = (0..3).all(|i| (0..3).all(|j| is_approx_equal_tol(mat3.0[i][j], mat3t.0[i][j], 1.0e-6)));
    if !symmetric {
        let mut r = Mat4d::zero();
        r.set_translation(-trans);
        r.0[3][3] = 1.0;
        return r;
    }
    let (eigen_vectors, eigen_values) =
        super::mat::diagonalize_symmetric_matrix(&mat3, 250).unwrap_or((Mat3d::identity(), Vec3d::zero()));
    let mut d = Mat3d::identity();
    for i in 0..3 {
        d.0[i][i] = if eigen_values[i].abs() < 10.0 * TOL { 0.0 } else { 1.0 / eigen_values[i] };
    }
    let pseudo_inv = eigen_vectors * d * eigen_vectors.transpose();
    let inv_trans = pseudo_inv.transform(-trans);
    Mat4d::from_mat3_translation(&pseudo_inv, inv_trans)
}

// ---------------------------------------------------------------------------------------------
// MapBase

/// Any index-to-world map.
#[derive(Clone)]
pub enum MapBase {
    /// A general affine map.
    Affine(AffineMap),
    /// A rotation/reflection.
    Unitary(UnitaryMap),
    /// A non-uniform scale.
    Scale(ScaleMap),
    /// A uniform scale.
    UniformScale(UniformScaleMap),
    /// A translation.
    Translation(TranslationMap),
    /// A non-uniform scale followed by a translation.
    ScaleTranslate(ScaleTranslateMap),
    /// A uniform scale followed by a translation.
    UniformScaleTranslate(UniformScaleTranslateMap),
    /// A frustum followed by an affine map.
    NonlinearFrustum(NonlinearFrustumMap),
}

impl Default for MapBase {
    /// OpenVDB's default transform map: a unit `ScaleMap`.
    fn default() -> Self {
        MapBase::Scale(ScaleMap::default())
    }
}

/// Dispatch to every concrete map type.
macro_rules! each_map {
    ($self:expr, $m:ident => $e:expr) => {
        match $self {
            MapBase::Affine($m) => $e,
            MapBase::Unitary($m) => $e,
            MapBase::Scale($m) => $e,
            MapBase::UniformScale($m) => $e,
            MapBase::Translation($m) => $e,
            MapBase::ScaleTranslate($m) => $e,
            MapBase::UniformScaleTranslate($m) => $e,
            MapBase::NonlinearFrustum($m) => $e,
        }
    };
}

impl MapBase {
    /// A [`UniformScaleMap`] (the map of `Transform::createLinearTransform(voxelSize)`).
    pub fn uniform_scale(s: f64) -> Self {
        MapBase::UniformScale(UniformScaleMap::new(s))
    }
    /// An [`AffineMap`] from a 4x4 matrix (not simplified; see [`simplify`]).
    ///
    /// # Panics
    /// If the matrix is not affine or is nearly singular.
    pub fn from_matrix(m: Mat4d) -> Self {
        MapBase::Affine(AffineMap::new(m))
    }
    /// The registered OpenVDB type name of the map (e.g. `"UniformScaleMap"`).
    pub fn type_name(&self) -> &'static str {
        match self {
            MapBase::Affine(_) => AffineMap::TYPE_NAME,
            MapBase::Unitary(_) => UnitaryMap::TYPE_NAME,
            MapBase::Scale(_) => ScaleMap::TYPE_NAME,
            MapBase::UniformScale(_) => UniformScaleMap::TYPE_NAME,
            MapBase::Translation(_) => TranslationMap::TYPE_NAME,
            MapBase::ScaleTranslate(_) => ScaleTranslateMap::TYPE_NAME,
            MapBase::UniformScaleTranslate(_) => UniformScaleTranslateMap::TYPE_NAME,
            MapBase::NonlinearFrustum(_) => NonlinearFrustumMap::TYPE_NAME,
        }
    }
    /// True for every map type except the frustum.
    pub fn is_linear(&self) -> bool {
        !matches!(self, MapBase::NonlinearFrustum(_))
    }
    /// True if the spacing of the image of the lattice is uniform in all directions.
    pub fn has_uniform_scale(&self) -> bool {
        match self {
            MapBase::Affine(m) => m.has_uniform_scale(),
            MapBase::Unitary(_) | MapBase::Translation(_) => true,
            MapBase::Scale(m) => m.has_uniform_scale(),
            MapBase::UniformScale(m) => m.has_uniform_scale(),
            MapBase::ScaleTranslate(m) => m.has_uniform_scale(),
            MapBase::UniformScaleTranslate(m) => m.has_uniform_scale(),
            MapBase::NonlinearFrustum(_) => false,
        }
    }
    /// True if the map is (approximately) the identity (`Transform::isIdentity`).
    pub fn is_identity(&self) -> bool {
        match self {
            MapBase::NonlinearFrustum(f) => f.is_identity(),
            m => m.get_affine_map().is_identity(),
        }
    }
    /// An affine map equivalent to this linear map; for a frustum, its second map
    /// (`getAffineMap`).
    pub fn get_affine_map(&self) -> AffineMap {
        each_map!(self, m => m.get_affine_map())
    }
    /// The 4x4 matrix of [`get_affine_map`](Self::get_affine_map).
    pub fn affine_matrix(&self) -> Mat4d {
        match self {
            MapBase::Affine(a) => a.matrix,
            MapBase::Unitary(u) => u.affine.matrix,
            MapBase::NonlinearFrustum(f) => f.second_map.matrix,
            m => m.get_affine_map().matrix,
        }
    }
    /// The inverse map, or `None` for a frustum (OpenVDB throws `NotImplementedError`).
    pub fn inverse_map(&self) -> Option<MapBase> {
        each_map!(self, m => m.inverse_map())
    }
    /// Index to world.
    pub fn apply_map(&self, xyz: Vec3d) -> Vec3d {
        each_map!(self, m => m.apply_map(xyz))
    }
    /// World to index.
    pub fn apply_inverse_map(&self, xyz: Vec3d) -> Vec3d {
        each_map!(self, m => m.apply_inverse_map(xyz))
    }
    /// The Jacobian applied to `v` (the map without translation for linear maps; the second
    /// map's Jacobian for a frustum).
    pub fn apply_jacobian(&self, v: Vec3d) -> Vec3d {
        each_map!(self, m => m.apply_jacobian(v))
    }
    /// The Jacobian at index-space location `domain_pos` applied to `v`.
    pub fn apply_jacobian_at(&self, v: Vec3d, domain_pos: Vec3d) -> Vec3d {
        match self {
            MapBase::NonlinearFrustum(f) => f.apply_jacobian_at(v, domain_pos),
            _ => self.apply_jacobian(v),
        }
    }
    /// The inverse Jacobian applied to `v`.
    pub fn apply_inverse_jacobian(&self, v: Vec3d) -> Vec3d {
        each_map!(self, m => m.apply_inverse_jacobian(v))
    }
    /// The inverse Jacobian at index-space location `domain_pos` applied to `v`.
    pub fn apply_inverse_jacobian_at(&self, v: Vec3d, domain_pos: Vec3d) -> Vec3d {
        match self {
            MapBase::NonlinearFrustum(f) => f.apply_inverse_jacobian_at(v, domain_pos),
            _ => self.apply_inverse_jacobian(v),
        }
    }
    /// The Jacobian transpose applied to `v` (range-space to domain-space gradients).
    pub fn apply_jt(&self, v: Vec3d) -> Vec3d {
        each_map!(self, m => m.apply_jt(v))
    }
    /// The Jacobian transpose at index-space location `domain_pos` applied to `v`.
    pub fn apply_jt_at(&self, v: Vec3d, domain_pos: Vec3d) -> Vec3d {
        match self {
            MapBase::NonlinearFrustum(f) => f.apply_jt_at(v, domain_pos),
            _ => self.apply_jt(v),
        }
    }
    /// The transpose of the inverse Jacobian applied to `v` (domain-space to range-space
    /// gradients).
    pub fn apply_ijt(&self, v: Vec3d) -> Vec3d {
        each_map!(self, m => m.apply_ijt(v))
    }
    /// The transpose of the inverse Jacobian at index-space location `domain_pos` applied to `v`.
    pub fn apply_ijt_at(&self, v: Vec3d, domain_pos: Vec3d) -> Vec3d {
        match self {
            MapBase::NonlinearFrustum(f) => f.apply_ijt_at(v, domain_pos),
            _ => self.apply_ijt(v),
        }
    }
    /// The Jacobian curvature: transforms a matrix of second derivatives to range space
    /// (linear maps only use the inverse Jacobian).
    pub fn apply_ijc(&self, m: &Mat3d) -> Mat3d {
        each_map!(self, a => a.apply_ijc(m))
    }
    /// The Jacobian curvature at index-space location `domain_pos`, given the index-space second
    /// derivatives `m` and gradient `v`.
    pub fn apply_ijc_at(&self, m: &Mat3d, v: Vec3d, domain_pos: Vec3d) -> Mat3d {
        match self {
            MapBase::NonlinearFrustum(f) => f.apply_ijc_at(m, v, domain_pos),
            _ => self.apply_ijc(m),
        }
    }
    /// The determinant of the Jacobian (the second map's for a frustum).
    pub fn determinant(&self) -> f64 {
        each_map!(self, m => m.determinant())
    }
    /// The determinant of the Jacobian at index-space location `xyz`.
    pub fn determinant_at(&self, xyz: Vec3d) -> f64 {
        match self {
            MapBase::NonlinearFrustum(f) => f.determinant_at(xyz),
            _ => self.determinant(),
        }
    }
    /// The world-space size of a voxel (at the center of the near plane for a frustum).
    pub fn voxel_size(&self) -> Vec3d {
        each_map!(self, m => m.voxel_size())
    }
    /// The world-space size of the voxel at index-space location `xyz`.
    pub fn voxel_size_at(&self, xyz: Vec3d) -> Vec3d {
        match self {
            MapBase::NonlinearFrustum(f) => f.voxel_size_at(xyz),
            _ => self.voxel_size(),
        }
    }

    /// The map with a rotation by `angle` radians about `axis` applied first (in index space).
    pub fn pre_rotate(&self, angle: f64, axis: Axis) -> MapBase {
        each_map!(self, m => m.pre_rotate(angle, axis))
    }
    /// The map with a translation applied first.
    pub fn pre_translate(&self, t: Vec3d) -> MapBase {
        each_map!(self, m => m.pre_translate(t))
    }
    /// The map with a scale applied first.
    pub fn pre_scale(&self, s: Vec3d) -> MapBase {
        each_map!(self, m => m.pre_scale(s))
    }
    /// The map with a shear (along `axis0` by a fraction of `axis1`) applied first.
    pub fn pre_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        each_map!(self, m => m.pre_shear(shear, axis0, axis1))
    }
    /// The map followed by a rotation by `angle` radians about `axis` (in world space).
    pub fn post_rotate(&self, angle: f64, axis: Axis) -> MapBase {
        each_map!(self, m => m.post_rotate(angle, axis))
    }
    /// The map followed by a translation.
    pub fn post_translate(&self, t: Vec3d) -> MapBase {
        each_map!(self, m => m.post_translate(t))
    }
    /// The map followed by a scale.
    pub fn post_scale(&self, s: Vec3d) -> MapBase {
        each_map!(self, m => m.post_scale(s))
    }
    /// The map followed by a shear.
    pub fn post_shear(&self, shear: f64, axis0: Axis, axis1: Axis) -> MapBase {
        each_map!(self, m => m.post_shear(shear, axis0, axis1))
    }
    /// The map with the matrix `o` applied first (`Transform::preMult`): linear maps become
    /// `simplify(AffineMap(o * M))`; for a frustum, `o` is pre-multiplied into the second map.
    pub fn pre_mult(&self, o: &Mat4d) -> MapBase {
        match self {
            MapBase::NonlinearFrustum(f) => f.with_second(MapBase::Affine(AffineMap::new(*o * f.second_map.matrix))),
            m => simplify(AffineMap::new(*o * m.affine_matrix())),
        }
    }
    /// The map followed by the matrix `o` (`Transform::postMult`).
    pub fn post_mult(&self, o: &Mat4d) -> MapBase {
        match self {
            MapBase::NonlinearFrustum(f) => f.with_second(MapBase::Affine(AffineMap::new(f.second_map.matrix * *o))),
            m => simplify(AffineMap::new(m.affine_matrix() * *o)),
        }
    }
    /// OpenVDB's `MapBase::isEqual`: same map type and approximately equal parameters.
    pub fn is_equal(&self, o: &MapBase) -> bool {
        match (self, o) {
            (MapBase::Affine(a), MapBase::Affine(b)) => a == b,
            (MapBase::Unitary(a), MapBase::Unitary(b)) => a == b,
            (MapBase::Scale(a), MapBase::Scale(b)) => a == b,
            (MapBase::UniformScale(a), MapBase::UniformScale(b)) => a == b,
            (MapBase::Translation(a), MapBase::Translation(b)) => a == b,
            (MapBase::ScaleTranslate(a), MapBase::ScaleTranslate(b)) => a == b,
            (MapBase::UniformScaleTranslate(a), MapBase::UniformScaleTranslate(b)) => a == b,
            (MapBase::NonlinearFrustum(a), MapBase::NonlinearFrustum(b)) => a == b,
            _ => false,
        }
    }
    /// True if both maps are linear with matrices equal to within `tol`, or both are frustums
    /// with equal frustum parameters and second-map matrices equal to within `tol`.
    pub fn is_approx_equal(&self, o: &MapBase, tol: f64) -> bool {
        match (self, o) {
            (MapBase::NonlinearFrustum(a), MapBase::NonlinearFrustum(b)) => {
                a.bbox == b.bbox
                    && is_approx_equal_tol(a.taper, b.taper, tol)
                    && is_approx_equal_tol(a.depth, b.depth, tol)
                    && a.second_map.matrix.is_approx_equal(&b.second_map.matrix, tol)
            }
            (MapBase::NonlinearFrustum(_), _) | (_, MapBase::NonlinearFrustum(_)) => false,
            _ => self.affine_matrix().is_approx_equal(&o.affine_matrix(), tol),
        }
    }
}

impl PartialEq for MapBase {
    /// Same as [`MapBase::is_equal`].
    fn eq(&self, o: &Self) -> bool {
        self.is_equal(o)
    }
}

impl MapBase {
    /// Read a map serialized by OpenVDB (`MapBase::read`), given its registered type name
    /// (e.g. `"UniformScaleMap"`), from a little-endian stream.
    pub fn read_typed<R: Read>(type_name: &str, r: &mut R) -> io::Result<MapBase> {
        Ok(match type_name {
            AffineMap::TYPE_NAME => MapBase::Affine(AffineMap::read(r)?),
            UnitaryMap::TYPE_NAME => MapBase::Unitary(UnitaryMap { affine: AffineMap::read(r)? }),
            ScaleMap::TYPE_NAME => MapBase::Scale(ScaleMap::read_fields(r)?),
            UniformScaleMap::TYPE_NAME => MapBase::UniformScale(UniformScaleMap(ScaleMap::read_fields(r)?)),
            TranslationMap::TYPE_NAME => MapBase::Translation(TranslationMap::new(read_vec3d(r)?)),
            ScaleTranslateMap::TYPE_NAME => MapBase::ScaleTranslate(ScaleTranslateMap::read_fields(r)?),
            UniformScaleTranslateMap::TYPE_NAME => {
                MapBase::UniformScaleTranslate(UniformScaleTranslateMap(ScaleTranslateMap::read_fields(r)?))
            }
            NonlinearFrustumMap::TYPE_NAME => MapBase::NonlinearFrustum(NonlinearFrustumMap::read(r)?),
            other => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Map {other} is not registered")));
            }
        })
    }
    /// Write this map's payload as OpenVDB does (`MapBase::write`); the type name is
    /// written separately by the caller.
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            MapBase::Affine(m) => m.write(w),
            MapBase::Unitary(m) => m.affine.write(w),
            MapBase::Scale(m) => m.write(w),
            MapBase::UniformScale(m) => m.0.write(w),
            MapBase::Translation(m) => m.write(w),
            MapBase::ScaleTranslate(m) => m.write(w),
            MapBase::UniformScaleTranslate(m) => m.0.write(w),
            MapBase::NonlinearFrustum(m) => m.write(w),
        }
    }
    /// True if `name` is one of the registered map type names.
    pub fn is_registered(name: &str) -> bool {
        matches!(
            name,
            AffineMap::TYPE_NAME
                | UnitaryMap::TYPE_NAME
                | ScaleMap::TYPE_NAME
                | UniformScaleMap::TYPE_NAME
                | TranslationMap::TYPE_NAME
                | ScaleTranslateMap::TYPE_NAME
                | UniformScaleTranslateMap::TYPE_NAME
                | NonlinearFrustumMap::TYPE_NAME
        )
    }
}

impl fmt::Debug for MapBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapBase::NonlinearFrustum(m) => write!(
                f,
                "NonlinearFrustumMap(bbox: {:?}, taper: {}, depth: {}, second: {:?})",
                m.bbox,
                m.taper,
                m.depth,
                m.second_map.matrix()
            ),
            m => write!(f, "{}({:?})", m.type_name(), m.affine_matrix()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
    fn serialize(m: &MapBase) -> String {
        let mut buf = Vec::new();
        write_name(&mut buf, m.type_name()).unwrap();
        m.write(&mut buf).unwrap();
        hex(&buf)
    }
    fn roundtrip(m: &MapBase) -> MapBase {
        let mut buf = Vec::new();
        write_name(&mut buf, m.type_name()).unwrap();
        m.write(&mut buf).unwrap();
        let mut r = &buf[..];
        let name = read_name(&mut r).unwrap();
        let back = MapBase::read_typed(&name, &mut r).unwrap();
        assert!(r.is_empty());
        let mut buf2 = Vec::new();
        write_name(&mut buf2, back.type_name()).unwrap();
        back.write(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
        back
    }

    // Byte streams produced by OpenVDB 10: `writeString(os, map.type()); map.write(os);`.
    const SCALE: &str = "080000005363616c654d6170000000000000f03f00000000000000400000000000000840000000000000f03f00000000000000400000000000000840000000000000f03f000000000000e03f555555555555d53f000000000000f03f000000000000d03f1cc7711cc771bc3f000000000000e03f000000000000d03f555555555555c53f";
    const USCALE: &str = "0f000000556e69666f726d5363616c654d6170000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000004000000000000000400000000000000040000000000000104000000000000010400000000000001040000000000000f03f000000000000f03f000000000000f03f";
    const TRANS: &str = "0e0000005472616e736c6174696f6e4d6170000000000000f03f00000000000000400000000000000840";
    const ST: &str = "110000005363616c655472616e736c6174654d6170000000000000104000000000000014400000000000001840000000000000f03f00000000000000400000000000000840000000000000f03f00000000000000400000000000000840000000000000f03f000000000000e03f555555555555d53f000000000000f03f000000000000d03f1cc7711cc771bc3f000000000000e03f000000000000d03f555555555555c53f";
    const UST: &str = "18000000556e69666f726d5363616c655472616e736c6174654d6170000000000000104000000000000014400000000000001840000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000004000000000000000400000000000000040000000000000104000000000000010400000000000001040000000000000f03f000000000000f03f000000000000f03f";
    const AFFINE: &str = "09000000416666696e654d6170ba092fd41d92ee3f000000000000000033baba95cde9d2bf000000000000000000000000000000000000000000000040000000000000000000000000000000004c179860b45eec3f00000000000000004c47235f96ed06400000000000000000000000000000f03f00000000000000c00000000000001440000000000000f03f";
    const UNITARY: &str = "0a000000556e69746172794d6170939d9f529679e83fb81946696e9de43f00000000000000000000000000000000b81946696e9de4bf939d9f529679e83f0000000000000000000000000000000000000000000000000000000000000000000000000000f03f0000000000000000000000000000000000000000000000000000000000000000000000000000f03f";
    const FRUSTUM: &str = "130000004e6f6e6c696e6561724672757374756d4d6170000000000000000000000000000000000000000000000000000000000000244000000000000034400000000000003e40000000000000e03f000000000000004009000000416666696e654d6170000000000000f03f0000000000000000000000000000000000000000000000000000000000000000000000000000f03f0000000000000000000000000000000000000000000000000000000000000000000000000000f03f0000000000000000000000000000000000000000000000000000000000000000000000000000f03f";
    const FRUSTCAM: &str = "130000004e6f6e6c696e6561724672757374756d4d6170000000000000000000000000000000000000000000000000000000000000504000000000008045400000000000004040fbd96265f846843fabaaaaaaaaaa1a4009000000416666696e654d6170000000000000f8bf0000000000000000000000000000000000000000000000000000000000000000000000000000f83f0000000000000000000000000000000000000000000000000000000000000000000000000000f8bf0000000000000000000000000000f03f00000000000000403333333333330740000000000000f03f";

    #[test]
    fn serialization_matches_openvdb() {
        let mut m = Mat4d::identity();
        m.pre_scale(Vec3d::new(1.0, 2.0, 3.0));
        m.post_rotate(Axis::Y, 0.3);
        m.post_translate(Vec3d::new(1.0, -2.0, 5.0));
        let cases = [
            (MapBase::Scale(ScaleMap::new(Vec3d::new(1.0, 2.0, 3.0))), SCALE),
            (MapBase::uniform_scale(0.5), USCALE),
            (MapBase::Translation(TranslationMap::new(Vec3d::new(1.0, 2.0, 3.0))), TRANS),
            (MapBase::ScaleTranslate(ScaleTranslateMap::new(Vec3d::new(1.0, 2.0, 3.0), Vec3d::new(4.0, 5.0, 6.0))), ST),
            (MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(0.5, Vec3d::new(4.0, 5.0, 6.0))), UST),
            (MapBase::from_matrix(m), AFFINE),
            (MapBase::Unitary(UnitaryMap::from_axis(Axis::Z, 0.7)), UNITARY),
            (
                MapBase::NonlinearFrustum(NonlinearFrustumMap::new(
                    BBoxd::new(Vec3d::zero(), Vec3d::new(10.0, 20.0, 30.0)),
                    0.5,
                    2.0,
                )),
                FRUSTUM,
            ),
            (
                MapBase::NonlinearFrustum(
                    NonlinearFrustumMap::try_from_camera(
                        Vec3d::new(1.0, 2.0, 3.0),
                        Vec3d::new(0.0, 0.0, -1.0),
                        Vec3d::new(0.0, 0.5, 0.0),
                        1.5,
                        0.1,
                        10.0,
                        64,
                        32,
                    )
                    .unwrap(),
                ),
                FRUSTCAM,
            ),
        ];
        for (map, expected) in &cases {
            assert_eq!(serialize(map), *expected, "{}", map.type_name());
            let back = roundtrip(map);
            assert_eq!(back.type_name(), map.type_name());
            assert!(back.is_equal(map));
        }
    }

    #[test]
    fn simplification_rules() {
        let u = MapBase::uniform_scale(2.0);
        assert_eq!(u.pre_translate(Vec3d::splat(1.0)).type_name(), "UniformScaleTranslateMap");
        assert_eq!(u.post_scale(Vec3d::new(1.0, 2.0, 1.0)).type_name(), "ScaleMap");
        assert_eq!(u.post_rotate(0.3, Axis::X).type_name(), "AffineMap");
        assert_eq!(u.post_rotate(0.0, Axis::X).type_name(), "UniformScaleMap");
        let t = MapBase::Translation(TranslationMap::new(Vec3d::new(1.0, 2.0, 3.0)));
        assert_eq!(t.pre_scale(Vec3d::splat(2.0)).type_name(), "UniformScaleTranslateMap");
        assert_eq!(t.post_scale(Vec3d::new(1.0, 2.0, 3.0)).type_name(), "ScaleTranslateMap");
        assert_eq!(t.post_translate(Vec3d::splat(1.0)).type_name(), "TranslationMap");
        let r = MapBase::Unitary(UnitaryMap::from_axis(Axis::X, 0.4));
        assert_eq!(r.post_rotate(0.2, Axis::Y).type_name(), "UnitaryMap");
        assert_eq!(r.post_translate(Vec3d::splat(1.0)).type_name(), "AffineMap");
        assert_eq!(
            r.pre_rotate(-0.4, Axis::X).post_translate(Vec3d::splat(1.0)).type_name(),
            "UniformScaleTranslateMap"
        );
        // An AffineMap scaled stays an AffineMap, as in OpenVDB.
        let a = MapBase::from_matrix(Mat4d::identity());
        assert_eq!(a.pre_scale(Vec3d::splat(2.0)).type_name(), "AffineMap");
        assert_eq!(simplify(AffineMap::identity()).type_name(), "UniformScaleMap");
    }

    #[test]
    fn inverses_and_jacobians() {
        let maps = [
            MapBase::Scale(ScaleMap::new(Vec3d::new(1.0, 2.0, -3.0))),
            MapBase::uniform_scale(0.25),
            MapBase::Translation(TranslationMap::new(Vec3d::new(1.0, 2.0, 3.0))),
            MapBase::ScaleTranslate(ScaleTranslateMap::new(Vec3d::new(1.0, 2.0, 3.0), Vec3d::new(4.0, 5.0, 6.0))),
            MapBase::UniformScaleTranslate(UniformScaleTranslateMap::new(0.5, Vec3d::new(4.0, 5.0, 6.0))),
            MapBase::uniform_scale(0.5).post_rotate(0.3, Axis::Y).post_shear(0.2, Axis::X, Axis::Z),
            MapBase::Unitary(UnitaryMap::from_axis_angle(Vec3d::new(1.0, 1.0, 0.0), 0.7)),
        ];
        let p = Vec3d::new(0.3, -1.2, 2.5);
        let d2 = Mat3d::new(1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0);
        for m in &maps {
            let a = m.get_affine_map();
            assert!(m.apply_inverse_map(m.apply_map(p)).is_approx_equal(&p, 1e-12), "{}", m.type_name());
            assert!(m.apply_jacobian(p).is_approx_equal(&a.apply_jacobian(p), 1e-12));
            assert!(m.apply_inverse_jacobian(p).is_approx_equal(&a.apply_inverse_jacobian(p), 1e-12));
            assert!(m.apply_jt(p).is_approx_equal(&a.apply_jt(p), 1e-12), "{}", m.type_name());
            assert!(m.apply_ijt(p).is_approx_equal(&a.apply_ijt(p), 1e-12));
            assert!(m.apply_ijc(&d2).is_approx_equal(&a.apply_ijc(&d2), 1e-12));
            assert!((m.determinant() - a.determinant()).abs() < 1e-12);
            assert!(m.voxel_size().is_approx_equal(&a.voxel_size(), 1e-12));
            let inv = m.inverse_map().unwrap();
            assert!(inv.apply_map(m.apply_map(p)).is_approx_equal(&p, 1e-12), "{}", m.type_name());
            assert_eq!(inv.type_name(), m.type_name());
        }
    }

    #[test]
    fn frustum_against_openvdb() {
        // createFrustumTransform(BBoxd((0,0,0),(10,20,30)), 0.5, 2.0, 0.25)
        let f = MapBase::NonlinearFrustum(NonlinearFrustumMap::new(
            BBoxd::new(Vec3d::zero(), Vec3d::new(10.0, 20.0, 30.0)),
            0.5,
            2.0,
        ))
        .pre_scale(Vec3d::splat(0.25));
        assert!(!f.is_linear());
        let close = |a: Vec3d, b: [f64; 3]| a.is_approx_equal(&Vec3d::from(b), 1e-13);
        assert!(close(f.voxel_size(), [0.025, 0.025, 0.016666666666666666]));
        assert!(close(
            f.apply_map(Vec3d::new(1.5, -2.25, 3.0)),
            [-0.096_250_000_000_000_02, -0.33687500000000004, 0.05]
        ));
        assert!(close(f.apply_inverse_map(Vec3d::new(0.1, 0.2, 0.3)), [7.5, 15.0, 18.0]));
        let loc = Vec3d::new(3.0, 4.0, 5.0);
        assert!(close(f.voxel_size_at(loc), [0.029166666666666667, 0.029166666666666646, 0.017480147469502536]));
        assert!((f.determinant_at(loc) - 0.00012760416666666665).abs() < 1e-17);
        assert!((f.determinant() - 0.015625).abs() < 1e-17);
        let v = Vec3d::new(1.0, 2.0, 3.0);
        assert!(close(f.apply_jacobian_at(v, loc), [0.02416666666666667, 0.043333333333333335, 0.05]));
        assert!(f
            .apply_inverse_jacobian_at(v, loc)
            .is_approx_equal(&Vec3d::new(44.571_428_571_428_56, 99.428_571_428_571_43, 180.0), 1e-11));
        assert!(close(f.apply_jt_at(v, loc), [0.029166666666666667, 0.058333333333333334, 0.038_333_333_333_333_34]));
        assert!(f
            .apply_ijt_at(v, loc)
            .is_approx_equal(&Vec3d::new(34.285714285714285, 68.571_428_571_428_57, 204.0), 1e-11));
        let d2 = Mat3d::new(1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0);
        let c = f.apply_ijc_at(&d2, v, loc);
        let c_ref = Mat3d::new(
            1175.5102040816325,
            2_351.020_408_163_265,
            6_935.510_204_081_632,
            2_351.020_408_163_265,
            5_877.551_020_408_163,
            14223.673469387753,
            6_935.510_204_081_632,
            14223.673469387753,
            41_557.224_489_795_92,
        );
        assert!(c.is_approx_equal(&c_ref, 1e-9));
        assert!(f.inverse_map().is_none());
        let g = f.post_translate(Vec3d::new(1.0, 2.0, 3.0));
        assert!(close(g.apply_map(Vec3d::new(1.5, -2.25, 3.0)), [0.903_749_999_999_999_9, 1.663125, 3.05]));
        assert!(!f.is_equal(&g));
        assert!(f.is_equal(&f.clone()));
    }

    #[test]
    fn approx_inverse_handles_singular() {
        let m = Mat4d::scale(Vec3d::new(2.0, 0.0, 4.0));
        let inv = approx_inverse(&m);
        assert!(inv.get_mat3().is_approx_equal(&Mat3d::from_diagonal(Vec3d::new(0.5, 0.0, 0.25)), 1e-15));
        let r = Mat4d::rotation(Axis::X, 0.5);
        assert!((approx_inverse(&r) * r).is_approx_equal(&Mat4d::identity(), 1e-14));
    }

    #[test]
    fn errors() {
        assert!(ScaleMap::try_new(Vec3d::new(1.0, 0.0, 1.0)).is_err());
        assert!(AffineMap::try_new(Mat4d::zero()).is_err());
        assert!(UnitaryMap::try_from_mat3(&Mat3d::from_diagonal(Vec3d::splat(2.0))).is_err());
        assert!(NonlinearFrustumMap::try_new(BBoxd::new(Vec3d::zero(), Vec3d::new(1.0, 0.0, 1.0)), 1.0, 1.0).is_err());
        assert!(MapBase::read_typed("Bogus", &mut &[][..]).is_err());
        assert!(MapBase::is_registered("UnitaryMap"));
    }
}
