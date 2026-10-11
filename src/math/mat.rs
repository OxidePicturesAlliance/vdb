//! 3x3 and 4x4 matrices (port of `openvdb/math/Mat.h`, `openvdb/math/Mat3.h` and
//! `openvdb/math/Mat4.h`).
//!
//! Like OpenVDB, matrices are stored row-major and use the *row-vector* convention:
//! a point is transformed as `p' = p * M`, so the translation lives in the last row and
//! `A * B` applies `A` first, then `B`.
//!
//! The arithmetic (products, inverses, determinants, pre/post operations) follows the C++
//! code operation by operation, so results are bit-identical to OpenVDB's. This matters
//! because transforms are serialized as raw doubles.

use super::math_util::{is_approx_equal, is_approx_equal_tol, FloatTolerance};
use super::vec3::Vec3;
use super::vec4::Vec4;
use num_traits::Float;
use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// Coordinate axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// The x axis.
    X = 0,
    /// The y axis.
    Y = 1,
    /// The z axis.
    Z = 2,
}

impl Axis {
    /// The axis index (0, 1 or 2).
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// A 3x3 matrix, row-major.
#[derive(Clone, Copy, PartialEq)]
pub struct Mat3<T>(pub [[T; 3]; 3]);
/// Single-precision 3x3 matrix.
pub type Mat3s = Mat3<f32>;
/// Double-precision 3x3 matrix.
pub type Mat3d = Mat3<f64>;

/// A 4x4 matrix, row-major.
#[derive(Clone, Copy, PartialEq)]
pub struct Mat4<T>(pub [[T; 4]; 4]);
/// Single-precision 4x4 matrix.
pub type Mat4s = Mat4<f32>;
/// Double-precision 4x4 matrix.
pub type Mat4d = Mat4<f64>;

impl<T: Float> Default for Mat3<T> {
    fn default() -> Self {
        Self::identity()
    }
}
impl<T: Float> Default for Mat4<T> {
    fn default() -> Self {
        Self::identity()
    }
}

#[inline]
fn lit<T: Float>(v: f64) -> T {
    T::from(v).unwrap()
}

impl<T: Float> Mat3<T> {
    /// Construct from nine values in row-major order.
    #[allow(clippy::too_many_arguments)]
    pub fn new(a: T, b: T, c: T, d: T, e: T, f: T, g: T, h: T, i: T) -> Self {
        Mat3([[a, b, c], [d, e, f], [g, h, i]])
    }
    /// The zero matrix.
    pub fn zero() -> Self {
        Mat3([[T::zero(); 3]; 3])
    }
    /// The identity matrix.
    pub fn identity() -> Self {
        let (o, z) = (T::one(), T::zero());
        Mat3([[o, z, z], [z, o, z], [z, z, o]])
    }
    /// Construct from three row vectors.
    pub fn from_rows(r0: Vec3<T>, r1: Vec3<T>, r2: Vec3<T>) -> Self {
        Mat3([r0.0, r1.0, r2.0])
    }
    /// Construct from three column vectors.
    pub fn from_cols(c0: Vec3<T>, c1: Vec3<T>, c2: Vec3<T>) -> Self {
        Self::from_rows(c0, c1, c2).transpose()
    }
    /// A diagonal matrix.
    pub fn from_diagonal(d: Vec3<T>) -> Self {
        let mut m = Self::zero();
        for i in 0..3 {
            m.0[i][i] = d[i];
        }
        m
    }
    /// Row `i`.
    pub fn row(&self, i: usize) -> Vec3<T> {
        Vec3(self.0[i])
    }
    /// Column `j`.
    pub fn col(&self, j: usize) -> Vec3<T> {
        Vec3([self.0[0][j], self.0[1][j], self.0[2][j]])
    }
    /// Set row `i`.
    pub fn set_row(&mut self, i: usize, v: Vec3<T>) {
        self.0[i] = v.0;
    }
    /// Set column `j`.
    pub fn set_col(&mut self, j: usize, v: Vec3<T>) {
        for i in 0..3 {
            self.0[i][j] = v[i];
        }
    }
    /// The transpose.
    pub fn transpose(&self) -> Self {
        let mut r = *self;
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = self.0[j][i];
            }
        }
        r
    }
    /// The determinant.
    pub fn det(&self) -> T {
        let m = &self.0;
        let co00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let co10 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
        let co20 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
        m[0][0] * co00 + m[0][1] * co10 + m[0][2] * co20
    }
    /// Adjugate (classical adjoint) matrix.
    pub fn adjoint(&self) -> Self {
        let m = &self.0;
        Mat3([
            [
                m[1][1] * m[2][2] - m[1][2] * m[2][1],
                m[0][2] * m[2][1] - m[0][1] * m[2][2],
                m[0][1] * m[1][2] - m[0][2] * m[1][1],
            ],
            [
                m[1][2] * m[2][0] - m[1][0] * m[2][2],
                m[0][0] * m[2][2] - m[0][2] * m[2][0],
                m[0][2] * m[1][0] - m[0][0] * m[1][2],
            ],
            [
                m[1][0] * m[2][1] - m[1][1] * m[2][0],
                m[0][1] * m[2][0] - m[0][0] * m[2][1],
                m[0][0] * m[1][1] - m[0][1] * m[1][0],
            ],
        ])
    }
    /// Inverse, or `None` if the matrix is singular (|det| <= `tol`).
    pub fn inverse(&self, tol: T) -> Option<Self> {
        let inv = self.adjoint();
        let d = inv.0[0][0] * self.0[0][0] + inv.0[0][1] * self.0[1][0] + inv.0[0][2] * self.0[2][0];
        if is_approx_equal_tol(d, T::zero(), tol) {
            return None;
        }
        Some(inv * (T::one() / d))
    }
    /// Row-vector transform: `v * M`.
    pub fn transform(&self, v: Vec3<T>) -> Vec3<T> {
        let m = &self.0;
        Vec3([
            v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
            v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
            v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
        ])
    }
    /// Column-vector transform: `M * v` (OpenVDB's `pretransform`).
    pub fn mul_vec(&self, v: Vec3<T>) -> Vec3<T> {
        let m = &self.0;
        Vec3([
            v[0] * m[0][0] + v[1] * m[0][1] + v[2] * m[0][2],
            v[0] * m[1][0] + v[1] * m[1][1] + v[2] * m[1][2],
            v[0] * m[2][0] + v[1] * m[2][1] + v[2] * m[2][2],
        ])
    }
    /// Sum of the diagonal.
    pub fn trace(&self) -> T {
        self.0[0][0] + self.0[1][1] + self.0[2][2]
    }
    /// True if every element differs from `o`'s by at most `tol`.
    pub fn is_approx_equal(&self, o: &Self, tol: T) -> bool {
        (0..3).all(|i| (0..3).all(|j| is_approx_equal_tol(self.0[i][j], o.0[i][j], tol)))
    }
    /// OpenVDB's `Mat3::eq` with its default tolerance of 1e-8.
    pub fn eq_default(&self, o: &Self) -> bool {
        self.is_approx_equal(o, lit(1.0e-8))
    }
    /// Rotation by `angle` radians about the given coordinate axis (row-vector convention).
    pub fn rotation(axis: Axis, angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        let (o, z) = (T::one(), T::zero());
        match axis {
            Axis::X => Mat3([[o, z, z], [z, c, s], [z, -s, c]]),
            Axis::Y => Mat3([[c, z, -s], [z, o, z], [s, z, c]]),
            Axis::Z => Mat3([[c, s, z], [-s, c, z], [z, z, o]]),
        }
    }
    /// Rotation by `angle` radians about an arbitrary axis (`math::rotation<Mat3>(axis, angle)`).
    /// The axis is normalized first.
    pub fn rotation_axis_angle(axis: Vec3<T>, angle: T) -> Self {
        let len = axis.length();
        let a = axis / len;
        let c = angle.cos();
        let s = angle.sin();
        let t = T::one() - c;
        let mut r = Self::zero();
        r.0[0][0] = a[0] * a[0] * t + c;
        r.0[1][1] = a[1] * a[1] * t + c;
        r.0[2][2] = a[2] * a[2] * t + c;
        let txy = a[0] * a[1] * t;
        let sz = a[2] * s;
        let txz = a[0] * a[2] * t;
        let sy = a[1] * s;
        let tyz = a[1] * a[2] * t;
        let sx = a[0] * s;
        r.0[0][1] = txy + sz;
        r.0[1][0] = txy - sz;
        r.0[0][2] = txz - sy;
        r.0[2][0] = txz + sy;
        r.0[1][2] = tyz + sx;
        r.0[2][1] = tyz - sx;
        r
    }
    /// The rotation that maps `v1` onto `v2` about their cross product
    /// (`math::rotation<Mat3>(v1, v2)`).
    pub fn rotation_from_to(v1: Vec3<T>, v2: Vec3<T>) -> Self {
        rotation_from_to_3x3(v1, v2, lit(1.0e-8))
    }
    /// A scale matrix (`math::scale<Mat3>`).
    pub fn scale(s: Vec3<T>) -> Self {
        Self::from_diagonal(s)
    }
    /// Shear matrix along `axis0` by a fraction of `axis1` (`math::shear<Mat3>`).
    pub fn shear(axis0: Axis, axis1: Axis, shear: T) -> Self {
        let mut r = Self::identity();
        let (i0, i1) = (axis0.index(), axis1.index());
        r.0[i1][i0] = if axis0 == axis1 { shear + T::one() } else { shear };
        r
    }
    /// The lengths of the rows (`math::getScale`).
    pub fn get_scale(&self) -> Vec3<T> {
        Vec3::new(self.row(0).length(), self.row(1).length(), self.row(2).length())
    }
    /// `M * diag(d)` (`Mat3::timesDiagonal`).
    pub fn times_diagonal(&self, d: Vec3<T>) -> Self {
        let mut r = *self;
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = r.0[i][j] * d[j];
            }
        }
        r
    }
    /// Outer product `v1 v2ᵀ` (`math::outerProduct`).
    pub fn outer_product(v1: Vec3<T>, v2: Vec3<T>) -> Self {
        let mut r = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = v1[i] * v2[j];
            }
        }
        r
    }
    /// The L∞ norm: maximum absolute column sum (`math::lInfinityNorm`, which despite its
    /// name sums columns).
    pub fn l_infinity_norm(&self) -> T {
        let mut norm = T::zero();
        for j in 0..3 {
            let mut s = T::zero();
            for i in 0..3 {
                s = s + self.0[i][j].abs();
            }
            norm = if s > norm { s } else { norm };
        }
        norm
    }
    /// The L1 norm: maximum absolute row sum (`math::lOneNorm`).
    pub fn l_one_norm(&self) -> T {
        let mut norm = T::zero();
        for i in 0..3 {
            let mut s = T::zero();
            for j in 0..3 {
                s = s + self.0[i][j].abs();
            }
            norm = if s > norm { s } else { norm };
        }
        norm
    }
    /// Maximum absolute element.
    pub fn abs_max(&self) -> T {
        self.0.iter().flatten().fold(T::zero(), |a, &b| if b.abs() > a { b.abs() } else { a })
    }
    /// Element-wise cast.
    pub fn cast<U: Float>(&self) -> Mat3<U> {
        let mut r = Mat3::<U>::zero();
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = U::from(self.0[i][j]).unwrap();
            }
        }
        r
    }
}

impl<T: FloatTolerance> Mat3<T> {
    /// True if approximately the identity (`math::isIdentity`, tolerance 1e-8).
    pub fn is_identity(&self) -> bool {
        self.eq_default(&Self::identity())
    }
    /// True if the determinant is not approximately zero (`math::isInvertible`).
    pub fn is_invertible(&self) -> bool {
        !is_approx_equal(self.det(), T::zero())
    }
    /// True if approximately symmetric (`math::isSymmetric`, tolerance 1e-8).
    pub fn is_symmetric(&self) -> bool {
        self.eq_default(&self.transpose())
    }
    /// True if the matrix is a rotation or reflection (`math::isUnitary`).
    pub fn is_unitary(&self) -> bool {
        if !is_approx_equal(self.det().abs(), T::one()) {
            return false;
        }
        (*self * self.transpose()).eq_default(&Self::identity())
    }
    /// True if the sum of the absolute off-diagonal elements is approximately zero
    /// (`math::isDiagonal`).
    pub fn is_diagonal(&self) -> bool {
        let mut t = T::zero();
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    t = t + self.0[i][j].abs();
                }
            }
        }
        is_approx_equal(t, T::zero())
    }
    /// Polar decomposition `M = U * S` with `U` unitary and `S` symmetric positive semi-definite,
    /// or `None` if the matrix is singular or the iteration does not converge
    /// (`math::polarDecomposition`).
    pub fn polar_decomposition(&self) -> Option<(Self, Self)> {
        polar_decomposition(self, 100)
    }
    /// Jacobi diagonalization of a symmetric matrix: returns `(Q, D)` with `M = Q diag(D) Qᵀ`,
    /// or `None` if it did not converge (`math::diagonalizeSymmetricMatrix`).
    pub fn diagonalize_symmetric(&self) -> Option<(Self, Vec3<T>)> {
        diagonalize_symmetric_matrix(self, 250)
    }
}

fn rotation_from_to_3x3<T: Float>(v1: Vec3<T>, v2: Vec3<T>, eps: T) -> Mat3<T> {
    let mut v1 = v1;
    let mut v2 = v2;
    let normalize = |v: &mut Vec3<T>| {
        let d = v.length();
        if !is_approx_equal_tol(d, T::zero(), lit(1.0e-7)) {
            *v *= T::one() / d;
        }
    };
    if !is_approx_equal_tol(T::one(), v1.dot(&v1), eps) {
        normalize(&mut v1);
    }
    if !is_approx_equal_tol(T::one(), v2.dot(&v2), eps) {
        normalize(&mut v2);
    }
    let cross = v1.cross(&v2);
    let mut r = Mat3::identity();
    if is_approx_equal_tol(cross[0], T::zero(), eps)
        && is_approx_equal_tol(cross[1], T::zero(), eps)
        && is_approx_equal_tol(cross[2], T::zero(), eps)
    {
        // Nearly parallel: compose two reflections (Moller & Hughes).
        let mut p = Vec3::<T>::zero();
        let (x, y, z) = (v1[0].abs(), v1[1].abs(), v1[2].abs());
        if x < y {
            if z < x {
                p[2] = T::one();
            } else {
                p[0] = T::one();
            }
        } else if z < y {
            p[2] = T::one();
        } else {
            p[1] = T::one();
        }
        let u = p - v1;
        let v = p - v2;
        let udot = u.dot(&u);
        let vdot = v.dot(&v);
        let a = lit::<T>(-2.0) / udot;
        let b = lit::<T>(-2.0) / vdot;
        let c = lit::<T>(4.0) * u.dot(&v) / (udot * vdot);
        for j in 0..3 {
            for i in 0..3 {
                r.0[i][j] = a * u[i] * u[j] + b * v[i] * v[j] + c * v[j] * u[i];
            }
        }
        for i in 0..3 {
            r.0[i][i] = r.0[i][i] + T::one();
        }
    } else {
        let c = v1.dot(&v2);
        let a = (T::one() - c) / cross.dot(&cross);
        let a0 = a * cross[0];
        let a1 = a * cross[1];
        let a2 = a * cross[2];
        let a01 = a0 * cross[1];
        let a02 = a0 * cross[2];
        let a12 = a1 * cross[2];
        r.0[0][0] = c + a0 * cross[0];
        r.0[0][1] = a01 + cross[2];
        r.0[0][2] = a02 - cross[1];
        r.0[1][0] = a01 - cross[2];
        r.0[1][1] = c + a1 * cross[1];
        r.0[1][2] = a12 + cross[0];
        r.0[2][0] = a02 + cross[1];
        r.0[2][1] = a12 - cross[0];
        r.0[2][2] = c + a2 * cross[2];
    }
    r
}

/// Polar decomposition `M = U * S` (Higham's iteration), returning `(U, S)`
/// (`math::polarDecomposition`).
pub fn polar_decomposition<T: FloatTolerance>(input: &Mat3<T>, max_iterations: u32) -> Option<(Mat3<T>, Mat3<T>)> {
    let mut unitary = *input;
    if unitary.det().abs() < T::tolerance() {
        return None;
    }
    let mut iteration = 0u32;
    loop {
        let unitary_inv = unitary.inverse(T::zero())?;
        let linf_of_u = unitary.l_infinity_norm();
        let l1nm_of_u = unitary.l_one_norm();
        let linf_of_u_inv = unitary_inv.l_infinity_norm();
        let l1nm_of_u_inv = unitary_inv.l_one_norm();
        let gamma = ((l1nm_of_u_inv * linf_of_u_inv) / (l1nm_of_u * linf_of_u)).sqrt().sqrt();
        let new_unitary = (unitary * gamma + unitary_inv.transpose() * (T::one() / gamma)) * lit::<T>(0.5);
        let l1_error = (unitary - new_unitary).l_infinity_norm();
        unitary = new_unitary;
        if iteration > max_iterations {
            return None;
        }
        iteration += 1;
        if !(l1_error > T::tolerance()) {
            break;
        }
    }
    let hermitian = unitary.transpose() * *input;
    Some((unitary, hermitian))
}

fn jacobi_pivot<T: FloatTolerance>(i: usize, j: usize, s: &mut Mat3<T>, d: &mut Vec3<T>, q: &mut Mat3<T>) {
    let sij = s.0[i][j];
    let sjj_minus_sii = d[j] - d[i];
    let tan_of_theta;
    if sjj_minus_sii.abs() * (lit::<T>(10.0) * T::tolerance()) > sij.abs() {
        tan_of_theta = sij / sjj_minus_sii;
    } else {
        let cotan_of_2_theta = lit::<T>(0.5) * sjj_minus_sii / sij;
        let root = (T::one() + cotan_of_2_theta * cotan_of_2_theta).sqrt();
        tan_of_theta = if cotan_of_2_theta < T::zero() {
            -T::one() / (root - cotan_of_2_theta)
        } else {
            T::one() / (root + cotan_of_2_theta)
        };
    }
    let cos_t = T::one() / (T::one() + tan_of_theta * tan_of_theta).sqrt();
    let sin_t = cos_t * tan_of_theta;
    let z = tan_of_theta * sij;
    s.0[i][j] = T::zero();
    d[i] = d[i] - z;
    d[j] = d[j] + z;
    for k in 0..i {
        let temp = s.0[k][i];
        s.0[k][i] = cos_t * temp - sin_t * s.0[k][j];
        s.0[k][j] = sin_t * temp + cos_t * s.0[k][j];
    }
    for k in (i + 1)..j {
        let temp = s.0[i][k];
        s.0[i][k] = cos_t * temp - sin_t * s.0[k][j];
        s.0[k][j] = sin_t * temp + cos_t * s.0[k][j];
    }
    for k in (j + 1)..3 {
        let temp = s.0[i][k];
        s.0[i][k] = cos_t * temp - sin_t * s.0[j][k];
        s.0[j][k] = sin_t * temp + cos_t * s.0[j][k];
    }
    for k in 0..3 {
        let temp = q.0[k][i];
        q.0[k][i] = cos_t * temp - sin_t * q.0[k][j];
        q.0[k][j] = sin_t * temp + cos_t * q.0[k][j];
    }
}

/// Diagonalize a symmetric 3x3 matrix with Jacobi rotations (largest pivot first), returning the
/// eigenvector matrix `Q` (eigenvectors in columns) and eigenvalues `D` such that
/// `M = Q diag(D) Qᵀ`, or `None` if it did not converge in `max_iterations`
/// (`math::diagonalizeSymmetricMatrix`).
pub fn diagonalize_symmetric_matrix<T: FloatTolerance>(
    input: &Mat3<T>,
    max_iterations: u32,
) -> Option<(Mat3<T>, Vec3<T>)> {
    let mut q = Mat3::identity();
    let mut s = *input;
    let mut d = Vec3::new(s.0[0][0], s.0[1][1], s.0[2][2]);
    let mut iterations = 0u32;
    loop {
        let mut er = T::zero();
        for i in 0..3 {
            for j in (i + 1)..3 {
                er = er + s.0[i][j].abs();
            }
        }
        if er.abs() < T::tolerance() {
            return Some((q, d));
        }
        iterations += 1;
        let mut max_element = T::zero();
        let (mut ip, mut jp) = (0, 0);
        for i in 0..3 {
            for j in (i + 1)..3 {
                if d[i].abs() * (lit::<T>(10.0) * T::tolerance()) > s.0[i][j].abs() {
                    s.0[i][j] = T::zero();
                }
                if s.0[i][j].abs() > max_element {
                    max_element = s.0[i][j].abs();
                    ip = i;
                    jp = j;
                }
            }
        }
        jacobi_pivot(ip, jp, &mut s, &mut d, &mut q);
        if iterations >= max_iterations {
            return None;
        }
    }
}

impl<T: Float> Mul for Mat3<T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let mut r = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = self.0[i][0] * o.0[0][j] + self.0[i][1] * o.0[1][j] + self.0[i][2] * o.0[2][j];
            }
        }
        r
    }
}
impl<T: Float> Mul<T> for Mat3<T> {
    type Output = Self;
    fn mul(self, s: T) -> Self {
        let mut r = self;
        for row in r.0.iter_mut() {
            for v in row.iter_mut() {
                *v = *v * s;
            }
        }
        r
    }
}
impl<T: Float> Add for Mat3<T> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut r = self;
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = r.0[i][j] + o.0[i][j];
            }
        }
        r
    }
}
impl<T: Float> Sub for Mat3<T> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        let mut r = self;
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = r.0[i][j] - o.0[i][j];
            }
        }
        r
    }
}
impl<T: Float> Neg for Mat3<T> {
    type Output = Self;
    fn neg(self) -> Self {
        self * -T::one()
    }
}

impl<T: Float> Mat4<T> {
    /// Construct from four rows.
    pub fn new(r0: [T; 4], r1: [T; 4], r2: [T; 4], r3: [T; 4]) -> Self {
        Mat4([r0, r1, r2, r3])
    }
    /// The zero matrix.
    pub fn zero() -> Self {
        Mat4([[T::zero(); 4]; 4])
    }
    /// The identity matrix.
    pub fn identity() -> Self {
        let mut m = Self::zero();
        for i in 0..4 {
            m.0[i][i] = T::one();
        }
        m
    }
    /// An affine matrix with the given 3x3 block and translation.
    pub fn from_mat3_translation(m3: &Mat3<T>, t: Vec3<T>) -> Self {
        let mut m = Self::identity();
        m.set_mat3(m3);
        m.set_translation(t);
        m
    }
    /// Row `i`.
    pub fn row(&self, i: usize) -> Vec4<T> {
        Vec4(self.0[i])
    }
    /// Column `j`.
    pub fn col(&self, j: usize) -> Vec4<T> {
        Vec4([self.0[0][j], self.0[1][j], self.0[2][j], self.0[3][j]])
    }
    /// Set row `i`.
    pub fn set_row(&mut self, i: usize, v: Vec4<T>) {
        self.0[i] = v.0;
    }
    /// Set column `j`.
    pub fn set_col(&mut self, j: usize, v: Vec4<T>) {
        for i in 0..4 {
            self.0[i][j] = v[i];
        }
    }
    /// The upper-left 3x3 block.
    pub fn get_mat3(&self) -> Mat3<T> {
        let mut r = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = self.0[i][j];
            }
        }
        r
    }
    /// Set the upper-left 3x3 block.
    pub fn set_mat3(&mut self, m: &Mat3<T>) {
        for i in 0..3 {
            for j in 0..3 {
                self.0[i][j] = m.0[i][j];
            }
        }
    }
    /// The translation (first three elements of the last row).
    pub fn get_translation(&self) -> Vec3<T> {
        Vec3([self.0[3][0], self.0[3][1], self.0[3][2]])
    }
    /// Set the translation.
    pub fn set_translation(&mut self, t: Vec3<T>) {
        self.0[3][0] = t[0];
        self.0[3][1] = t[1];
        self.0[3][2] = t[2];
    }
    /// A translation matrix.
    pub fn translation(t: Vec3<T>) -> Self {
        let mut m = Self::identity();
        m.set_translation(t);
        m
    }
    /// A scale matrix.
    pub fn scale(s: Vec3<T>) -> Self {
        let mut m = Self::identity();
        for i in 0..3 {
            m.0[i][i] = s[i];
        }
        m
    }
    /// Rotation by `angle` radians about a coordinate axis.
    pub fn rotation(axis: Axis, angle: T) -> Self {
        Self::from_mat3_translation(&Mat3::rotation(axis, angle), Vec3([T::zero(); 3]))
    }
    /// Rotation by `angle` radians about an arbitrary axis (`Mat4::setToRotation(axis, angle)`).
    pub fn rotation_axis_angle(axis: Vec3<T>, angle: T) -> Self {
        Self::from_mat3_translation(&Mat3::rotation_axis_angle(axis, angle), Vec3::zero())
    }
    /// The rotation mapping `v1` onto `v2` (`Mat4::setToRotation(v1, v2)`).
    pub fn rotation_from_to(v1: Vec3<T>, v2: Vec3<T>) -> Self {
        Self::from_mat3_translation(&Mat3::rotation_from_to(v1, v2), Vec3::zero())
    }
    /// Shear matrix along `axis0` by a fraction of `axis1` (`math::shear<Mat4>`).
    pub fn shear(axis0: Axis, axis1: Axis, shear: T) -> Self {
        let mut r = Self::identity();
        let (i0, i1) = (axis0.index(), axis1.index());
        r.0[i1][i0] = if axis0 == axis1 { shear + T::one() } else { shear };
        r
    }
    /// The transpose.
    pub fn transpose(&self) -> Self {
        let mut r = *self;
        for i in 0..4 {
            for j in 0..4 {
                r.0[i][j] = self.0[j][i];
            }
        }
        r
    }
    /// The determinant (cofactor expansion along the first row, as in OpenVDB).
    pub fn det(&self) -> T {
        let m = &self.0;
        let mut d = T::zero();
        let mut sign = T::one();
        for j in 0..4 {
            let minor = self.minor3(0, j);
            d = d + sign * m[0][j] * minor.det();
            sign = -sign;
        }
        d
    }
    fn minor3(&self, row: usize, col: usize) -> Mat3<T> {
        let mut r = Mat3::zero();
        let mut ii = 0;
        for i in 0..4 {
            if i == row {
                continue;
            }
            let mut jj = 0;
            for j in 0..4 {
                if j == col {
                    continue;
                }
                r.0[ii][jj] = self.0[i][j];
                jj += 1;
            }
            ii += 1;
        }
        r
    }
    fn det3(&self, j0: usize, j1: usize, j2: usize, i0: usize, i1: usize, i2: usize) -> T {
        let a = &self.0;
        let det2 = |j0: usize, j1: usize, i0: usize, i1: usize| a[j0][i0] * a[j1][i1] - a[j0][i1] * a[j1][i0];
        a[j0][i0] * det2(j1, j2, i1, i2) + a[j0][i1] * det2(j1, j2, i2, i0) + a[j0][i2] * det2(j1, j2, i0, i1)
    }
    /// Inverse, or `None` if the matrix is singular (|det| <= `tol`). Uses OpenVDB's block
    /// inversion, falling back to Gauss-Jordan elimination when the 3x3 block is nearly singular.
    pub fn inverse(&self, tol: T) -> Option<Self> {
        let m = &self.0;
        let m0011 = m[0][0] * m[1][1];
        let m0012 = m[0][0] * m[1][2];
        let m0110 = m[0][1] * m[1][0];
        let m0210 = m[0][2] * m[1][0];
        let m0120 = m[0][1] * m[2][0];
        let m0220 = m[0][2] * m[2][0];
        let mut det_a =
            m0011 * m[2][2] - m0012 * m[2][1] - m0110 * m[2][2] + m0210 * m[2][1] + m0120 * m[1][2] - m0220 * m[1][1];
        let has_perspective =
            m[0][3] != T::zero() || m[1][3] != T::zero() || m[2][3] != T::zero() || m[3][3] != T::one();
        let det = if has_perspective {
            m[0][3] * self.det3(1, 2, 3, 0, 2, 1)
                + m[1][3] * self.det3(2, 0, 3, 0, 2, 1)
                + m[2][3] * self.det3(3, 0, 1, 0, 2, 1)
                + m[3][3] * det_a
        } else {
            det_a * m[3][3]
        };
        if is_approx_equal_tol(det, T::zero(), tol) {
            return None;
        }
        if is_approx_equal_tol(det_a, T::zero(), lit(1e-8)) {
            return self.invert_gauss_jordan(tol);
        }
        let mut inv = Self::zero();
        det_a = T::one() / det_a;
        inv.0[0][0] = det_a * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
        inv.0[0][1] = det_a * (-m[0][1] * m[2][2] + m[0][2] * m[2][1]);
        inv.0[0][2] = det_a * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
        inv.0[1][0] = det_a * (-m[1][0] * m[2][2] + m[1][2] * m[2][0]);
        inv.0[1][1] = det_a * (m[0][0] * m[2][2] - m0220);
        inv.0[1][2] = det_a * (m0210 - m0012);
        inv.0[2][0] = det_a * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        inv.0[2][1] = det_a * (m0120 - m[0][0] * m[2][1]);
        inv.0[2][2] = det_a * (m0011 - m0110);
        if has_perspective {
            let mut r = Vec3::<T>::zero();
            let mut p = Vec3::<T>::zero();
            for k in 0..3 {
                r[k] = m[3][0] * inv.0[0][k] + m[3][1] * inv.0[1][k] + m[3][2] * inv.0[2][k];
                p[k] = inv.0[k][0] * m[0][3] + inv.0[k][1] * m[1][3] + inv.0[k][2] * m[2][3];
            }
            let mut h = m[3][3] - p.dot(&Vec3::new(m[3][0], m[3][1], m[3][2]));
            if is_approx_equal_tol(h, T::zero(), tol) {
                return None;
            }
            h = T::one() / h;
            inv.0[3][3] = h;
            for k in 0..3 {
                inv.0[3][k] = -h * r[k];
                inv.0[k][3] = -h * p[k];
            }
            p = p * h;
            for i in 0..3 {
                for j in 0..3 {
                    inv.0[i][j] = inv.0[i][j] + p[i] * r[j];
                }
            }
        } else {
            for k in 0..3 {
                inv.0[3][k] = -(m[3][0] * inv.0[0][k] + m[3][1] * inv.0[1][k] + m[3][2] * inv.0[2][k]);
            }
            inv.0[0][3] = T::zero();
            inv.0[1][3] = T::zero();
            inv.0[2][3] = T::zero();
            inv.0[3][3] = T::one();
        }
        Some(inv)
    }
    fn invert_gauss_jordan(&self, tol: T) -> Option<Self> {
        let mut temp = self.0;
        let mut inv = Self::identity().0;
        let mut det = T::one();
        for i in 0..4 {
            let mut row = i;
            let mut max = temp[i][i].abs();
            for (k, t) in temp.iter().enumerate().skip(i + 1) {
                if t[i].abs() > max {
                    row = k;
                    max = t[i].abs();
                }
            }
            if max == T::zero() {
                return None;
            }
            if row != i {
                det = -det;
                temp.swap(row, i);
                inv.swap(row, i);
            }
            let pivot = temp[i][i];
            det = det * pivot;
            for k in 0..4 {
                temp[i][k] = temp[i][k] / pivot;
                inv[i][k] = inv[i][k] / pivot;
            }
            for j in (i + 1)..4 {
                let t = temp[j][i];
                if t != T::zero() {
                    for k in 0..4 {
                        temp[j][k] = temp[j][k] - temp[i][k] * t;
                        inv[j][k] = inv[j][k] - inv[i][k] * t;
                    }
                }
            }
        }
        for i in (1..4).rev() {
            for j in 0..i {
                let t = temp[j][i];
                if t != T::zero() {
                    for k in 0..4 {
                        inv[j][k] = inv[j][k] - inv[i][k] * t;
                    }
                }
            }
        }
        if det * det >= tol * tol {
            Some(Mat4(inv))
        } else {
            None
        }
    }
    /// Transform a point (row vector, w = 1): `[p 1] * M`, without perspective division.
    pub fn transform(&self, p: Vec3<T>) -> Vec3<T> {
        let m = &self.0;
        Vec3([
            p[0] * m[0][0] + p[1] * m[1][0] + p[2] * m[2][0] + m[3][0],
            p[0] * m[0][1] + p[1] * m[1][1] + p[2] * m[2][1] + m[3][1],
            p[0] * m[0][2] + p[1] * m[1][2] + p[2] * m[2][2] + m[3][2],
        ])
    }
    /// Transform a point with homogeneous division.
    pub fn transform_h(&self, p: Vec3<T>) -> Vec3<T> {
        let m = &self.0;
        let w = p[0] * m[0][3] + p[1] * m[1][3] + p[2] * m[2][3] + m[3][3];
        self.transform(p) / w
    }
    /// Transform a 4-vector by post-multiplication: `v * M`.
    pub fn transform4(&self, v: Vec4<T>) -> Vec4<T> {
        let m = &self.0;
        let mut r = Vec4([T::zero(); 4]);
        for j in 0..4 {
            r[j] = v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j] + v[3] * m[3][j];
        }
        r
    }
    /// Transform a direction (ignores translation).
    pub fn transform3x3(&self, v: Vec3<T>) -> Vec3<T> {
        let m = &self.0;
        Vec3([
            v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
            v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
            v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
        ])
    }
    /// Transform a normal by the upper 3x3 block from the left: `M3 * n`
    /// (`math::transformNormal`).
    pub fn transform_normal(&self, n: Vec3<T>) -> Vec3<T> {
        let m = &self.0;
        Vec3([
            m[0][0] * n[0] + m[0][1] * n[1] + m[0][2] * n[2],
            m[1][0] * n[0] + m[1][1] * n[1] + m[1][2] * n[2],
            m[2][0] * n[0] + m[2][1] * n[1] + m[2][2] * n[2],
        ])
    }
    /// `M = S * M` (scale applied before the existing transform).
    pub fn pre_scale(&mut self, s: Vec3<T>) {
        for i in 0..3 {
            for j in 0..4 {
                self.0[i][j] = self.0[i][j] * s[i];
            }
        }
    }
    /// `M = M * S` (scale applied after the existing transform).
    pub fn post_scale(&mut self, s: Vec3<T>) {
        for i in 0..4 {
            for j in 0..3 {
                self.0[i][j] = self.0[i][j] * s[j];
            }
        }
    }
    /// `M = T * M` (translation applied before the existing transform).
    pub fn pre_translate(&mut self, t: Vec3<T>) {
        *self = Self::translation(t) * *self;
    }
    /// `M = M * T` (translation applied after the existing transform).
    pub fn post_translate(&mut self, t: Vec3<T>) {
        *self = *self * Self::translation(t);
    }
    /// `M = R * M` (rotation applied before the existing transform).
    pub fn pre_rotate(&mut self, axis: Axis, angle: T) {
        let c = angle.cos();
        let s = -angle.sin();
        let m = &mut self.0;
        let (a, b) = match axis {
            Axis::X => (1, 2),
            Axis::Y => (0, 2),
            Axis::Z => (0, 1),
        };
        match axis {
            Axis::X | Axis::Z => {
                for k in 0..4 {
                    let na = c * m[a][k] - s * m[b][k];
                    m[b][k] = s * m[a][k] + c * m[b][k];
                    m[a][k] = na;
                }
            }
            Axis::Y => {
                for k in 0..4 {
                    let na = c * m[a][k] + s * m[b][k];
                    m[b][k] = -s * m[a][k] + c * m[b][k];
                    m[a][k] = na;
                }
            }
        }
    }
    /// `M = M * R` (rotation applied after the existing transform).
    pub fn post_rotate(&mut self, axis: Axis, angle: T) {
        let c = angle.cos();
        let s = -angle.sin();
        let m = &mut self.0;
        match axis {
            Axis::X => {
                for row in m.iter_mut() {
                    let a2 = c * row[2] - s * row[1];
                    row[1] = c * row[1] + s * row[2];
                    row[2] = a2;
                }
            }
            Axis::Y => {
                for row in m.iter_mut() {
                    let a2 = c * row[2] + s * row[0];
                    row[0] = c * row[0] - s * row[2];
                    row[2] = a2;
                }
            }
            Axis::Z => {
                for row in m.iter_mut() {
                    let a1 = c * row[1] - s * row[0];
                    row[0] = c * row[0] + s * row[1];
                    row[1] = a1;
                }
            }
        }
    }
    /// Left-multiply by a shear: add `shear` times row `axis0` to row `axis1` (`Mat4::preShear`).
    pub fn pre_shear(&mut self, axis0: Axis, axis1: Axis, shear: T) {
        let (i0, i1) = (axis0.index(), axis1.index());
        for k in 0..4 {
            self.0[i1][k] = self.0[i1][k] + shear * self.0[i0][k];
        }
    }
    /// Right-multiply by a shear: add `shear` times column `axis1` to column `axis0`
    /// (`Mat4::postShear`).
    pub fn post_shear(&mut self, axis0: Axis, axis1: Axis, shear: T) {
        let (i0, i1) = (axis0.index(), axis1.index());
        for k in 0..4 {
            self.0[k][i0] = self.0[k][i0] + shear * self.0[k][i1];
        }
    }
    /// True if every element differs from `o`'s by at most `tol`.
    pub fn is_approx_equal(&self, o: &Self, tol: T) -> bool {
        (0..4).all(|i| (0..4).all(|j| is_approx_equal_tol(self.0[i][j], o.0[i][j], tol)))
    }
    /// OpenVDB's `Mat4::eq` with its default tolerance of 1e-8.
    pub fn eq_default(&self, o: &Self) -> bool {
        self.is_approx_equal(o, lit(1.0e-8))
    }
    /// True if the last column is exactly (0, 0, 0, 1) (`math::isAffine`).
    pub fn is_affine(&self) -> bool {
        let m = &self.0;
        m[0][3] == T::zero() && m[1][3] == T::zero() && m[2][3] == T::zero() && m[3][3] == T::one()
    }
    /// True if the last row is not exactly (0, 0, 0, 1) (`math::hasTranslation`).
    pub fn has_translation(&self) -> bool {
        let m = &self.0;
        !(m[3][0] == T::zero() && m[3][1] == T::zero() && m[3][2] == T::zero() && m[3][3] == T::one())
    }
    /// Element-wise cast.
    pub fn cast<U: Float>(&self) -> Mat4<U> {
        let mut r = Mat4::<U>::zero();
        for i in 0..4 {
            for j in 0..4 {
                r.0[i][j] = U::from(self.0[i][j]).unwrap();
            }
        }
        r
    }
}

impl<T: FloatTolerance> Mat4<T> {
    /// True if approximately the identity (`math::isIdentity`, tolerance 1e-8).
    pub fn is_identity(&self) -> bool {
        self.eq_default(&Self::identity())
    }
    /// True if the determinant is not approximately zero (`math::isInvertible`).
    pub fn is_invertible(&self) -> bool {
        !is_approx_equal(self.det(), T::zero())
    }
    /// True if approximately symmetric (`math::isSymmetric`).
    pub fn is_symmetric(&self) -> bool {
        self.eq_default(&self.transpose())
    }
    /// True if the sum of the absolute off-diagonal elements (all 12 of them) is approximately
    /// zero (`math::isDiagonal`).
    pub fn is_diagonal(&self) -> bool {
        let mut t = T::zero();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    t = t + self.0[i][j].abs();
                }
            }
        }
        is_approx_equal(t, T::zero())
    }
}

impl<T: Float> Mul for Mat4<T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let mut r = Self::zero();
        for i in 0..4 {
            for j in 0..4 {
                r.0[i][j] = self.0[i][0] * o.0[0][j]
                    + self.0[i][1] * o.0[1][j]
                    + self.0[i][2] * o.0[2][j]
                    + self.0[i][3] * o.0[3][j];
            }
        }
        r
    }
}
impl<T: Float> Mul<T> for Mat4<T> {
    type Output = Self;
    fn mul(self, s: T) -> Self {
        let mut r = self;
        for row in r.0.iter_mut() {
            for v in row.iter_mut() {
                *v = *v * s;
            }
        }
        r
    }
}
impl<T: Float> Add for Mat4<T> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut r = self;
        for i in 0..4 {
            for j in 0..4 {
                r.0[i][j] = r.0[i][j] + o.0[i][j];
            }
        }
        r
    }
}
impl<T: Float> Sub for Mat4<T> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        let mut r = self;
        for i in 0..4 {
            for j in 0..4 {
                r.0[i][j] = r.0[i][j] - o.0[i][j];
            }
        }
        r
    }
}
impl<T: Float> Neg for Mat4<T> {
    type Output = Self;
    fn neg(self) -> Self {
        self * -T::one()
    }
}

impl<T> Index<usize> for Mat3<T> {
    type Output = [T; 3];
    fn index(&self, i: usize) -> &[T; 3] {
        &self.0[i]
    }
}
impl<T> IndexMut<usize> for Mat3<T> {
    fn index_mut(&mut self, i: usize) -> &mut [T; 3] {
        &mut self.0[i]
    }
}
impl<T> Index<usize> for Mat4<T> {
    type Output = [T; 4];
    fn index(&self, i: usize) -> &[T; 4] {
        &self.0[i]
    }
}
impl<T> IndexMut<usize> for Mat4<T> {
    fn index_mut(&mut self, i: usize) -> &mut [T; 4] {
        &mut self.0[i]
    }
}

impl<T: fmt::Debug> fmt::Debug for Mat3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl<T: fmt::Debug> fmt::Debug for Mat4<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3d;

    fn m3(v: [f64; 9]) -> Mat3d {
        Mat3d::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
    }

    #[test]
    fn mat4_inverse_and_transform() {
        let mut m = Mat4d::identity();
        m.pre_scale(Vec3d::new(2.0, 3.0, 4.0));
        m.post_rotate(Axis::Z, 0.3);
        m.post_translate(Vec3d::new(1.0, -2.0, 5.0));
        let inv = m.inverse(1e-12).unwrap();
        let p = Vec3d::new(0.5, 0.25, -1.0);
        let q = inv.transform(m.transform(p));
        assert!(p.is_approx_equal(&q, 1e-12));
        assert!((m * inv).is_approx_equal(&Mat4d::identity(), 1e-12));
        // Gauss-Jordan fallback (tiny 3x3 block) and perspective matrices.
        let mut tiny = Mat4d::scale(Vec3d::new(1e-3, 1e-3, 1e-3));
        tiny.0[0][3] = 0.5;
        let ti = tiny.inverse(0.0).unwrap();
        assert!((tiny * ti).is_approx_equal(&Mat4d::identity(), 1e-9));
        let mut persp = m;
        persp.0[2][3] = 0.25;
        let pi = persp.inverse(0.0).unwrap();
        assert!((persp * pi).is_approx_equal(&Mat4d::identity(), 1e-12));
        assert!(Mat4d::zero().inverse(0.0).is_none());
    }

    #[test]
    fn pre_post_ops_match_products() {
        let mut base = Mat4d::identity();
        base.0 = [[1.0, 0.2, 0.3, 0.0], [0.4, 2.0, 0.1, 0.0], [0.0, 0.5, 3.0, 0.0], [1.0, 2.0, 3.0, 1.0]];
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let mut a = base;
            a.pre_rotate(axis, 0.7);
            assert!(a.is_approx_equal(&(Mat4d::rotation(axis, 0.7) * base), 1e-14));
            let mut b = base;
            b.post_rotate(axis, 0.7);
            assert!(b.is_approx_equal(&(base * Mat4d::rotation(axis, 0.7)), 1e-14));
        }
        let mut a = base;
        a.pre_shear(Axis::X, Axis::Z, 0.3);
        assert!(a.is_approx_equal(&(Mat4d::shear(Axis::X, Axis::Z, 0.3) * base), 1e-14));
        let mut b = base;
        b.post_shear(Axis::X, Axis::Z, 0.3);
        assert!(b.is_approx_equal(&(base * Mat4d::shear(Axis::X, Axis::Z, 0.3)), 1e-14));
        let mut c = base;
        c.pre_scale(Vec3d::new(2.0, 3.0, 4.0));
        assert!(c.is_approx_equal(&(Mat4d::scale(Vec3d::new(2.0, 3.0, 4.0)) * base), 1e-14));
        let mut d = base;
        d.post_scale(Vec3d::new(2.0, 3.0, 4.0));
        assert!(d.is_approx_equal(&(base * Mat4d::scale(Vec3d::new(2.0, 3.0, 4.0))), 1e-14));
    }

    #[test]
    fn row_vector_convention() {
        let m = Mat4d::translation(Vec3d::new(1.0, 2.0, 3.0));
        assert_eq!(m.transform(Vec3d::zero()), Vec3d::new(1.0, 2.0, 3.0));
        let r = Mat3d::rotation(Axis::Z, std::f64::consts::FRAC_PI_2);
        assert!(r.transform(Vec3d::new(1.0, 0.0, 0.0)).is_approx_equal(&Vec3d::new(0.0, 1.0, 0.0), 1e-12));
    }

    #[test]
    fn predicates() {
        let r = Mat3d::rotation(Axis::Y, 0.4);
        assert!(r.is_unitary());
        assert!(!r.is_diagonal());
        assert!(Mat3d::from_diagonal(Vec3d::new(1.0, 2.0, 3.0)).is_diagonal());
        assert!(m3([1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0]).is_symmetric());
        assert!(!Mat3d::zero().is_invertible());
        assert!(Mat4d::identity().is_identity());
        assert!(!Mat4d::translation(Vec3d::new(1.0, 0.0, 0.0)).is_diagonal());
        assert!(Mat4d::translation(Vec3d::new(1.0, 0.0, 0.0)).has_translation());
        let o = Mat3d::outer_product(Vec3d::new(1.0, 2.0, 3.0), Vec3d::new(4.0, 5.0, 6.0));
        assert_eq!(o.0[1][2], 12.0);
    }

    // Reference values from OpenVDB 10 (diagonalizeSymmetricMatrix, polarDecomposition,
    // rotation<Mat4d>(v1, v2) and rotation<Mat3d>(axis, angle)).
    #[test]
    fn decompositions_match_openvdb() {
        let s = m3([4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0]);
        let (q, d) = s.diagonalize_symmetric().unwrap();
        let d_ref = Vec3d::new(2.0965216172967196, 3.0722239520050745, 6.8312544306982064);
        let q_ref = m3([
            0.71674306628934192,
            0.32576727330278094,
            0.61656715820016206,
            -0.57487262482462442,
            0.77650188137195364,
            0.25800444463819533,
            -0.39471615390480141,
            -0.53937047738144539,
            0.74382702691950731,
        ]);
        assert!(d.is_approx_equal(&d_ref, 1e-14));
        assert!(q.is_approx_equal(&q_ref, 1e-14));

        let a = m3([1.0, 2.0, 0.5, -0.3, 2.0, 1.0, 0.2, 0.1, 3.0]);
        let (u, p) = a.polar_decomposition().unwrap();
        let u_ref = m3([
            0.78298059355741167,
            0.62189052742903295,
            0.013912656343148765,
            -0.61008254164883113,
            0.76336431731271948,
            0.21230688031476561,
            0.12141121236436792,
            -0.17472003590809243,
            0.97710399987129781,
        ]);
        let p_ref = m3([
            0.99028759852493453,
            0.35793722505359787,
            0.14564139222297845,
            0.3579372250535986,
            2.7530376858926955,
            0.55014947330295882,
            0.14564139222297867,
            0.55014947330295849,
            3.1505752081002334,
        ]);
        assert!(u.is_approx_equal(&u_ref, 1e-14));
        assert!(p.is_approx_equal(&p_ref, 1e-14));

        let r = Mat4d::rotation_from_to(Vec3d::new(0.0, 0.0, 1.0), Vec3d::new(0.3, 0.4, -0.2));
        let r_ref = m3([
            0.50629935651252289,
            -0.65826752464996985,
            -0.55708601453115547,
            -0.65826752464996985,
            0.12230996713337355,
            -0.74278135270820744,
            0.55708601453115547,
            0.74278135270820744,
            -0.37139067635410372,
        ]);
        assert!(r.get_mat3().is_approx_equal(&r_ref, 1e-15));
        let ra = Mat3d::rotation_axis_angle(Vec3d::new(1.0, 2.0, 3.0), 0.6);
        let ra_ref = m3([
            0.83781164241612993,
            0.47767320104108368,
            -0.26438601483276575,
            -0.42776909101527749,
            0.87523972493548452,
            0.22576321371476948,
            0.33924217987147504,
            -0.076050883637350908,
            0.93761986246774232,
        ]);
        assert!(ra.is_approx_equal(&ra_ref, 1e-15));
    }
}
