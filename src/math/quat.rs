//! Quaternions (port of `openvdb/math/Quat.h`).
//!
//! Components are stored as `[x, y, z, w]` with `w` the scalar part. Rotation matrices built
//! from quaternions follow OpenVDB's row-vector convention (they are the transpose of the usual
//! column-vector matrices).

use super::mat::{Axis, Mat3};
use super::math_util::{is_approx_equal_tol, FloatTolerance};
use super::vec3::Vec3;
use num_traits::Float;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// A quaternion `x i + y j + z k + w`.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Quat<T>(pub [T; 4]);

/// Single-precision quaternion.
pub type Quats = Quat<f32>;
/// Double-precision quaternion.
pub type Quatd = Quat<f64>;

#[inline]
fn lit<T: Float>(v: f64) -> T {
    T::from(v).unwrap()
}

impl<T: Float> Quat<T> {
    /// Construct from components.
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Quat([x, y, z, w])
    }
    /// The zero quaternion.
    pub fn zero() -> Self {
        Quat([T::zero(); 4])
    }
    /// The identity rotation `(0, 0, 0, 1)`.
    pub fn identity() -> Self {
        Quat([T::zero(), T::zero(), T::zero(), T::one()])
    }
    /// Rotation by `angle` radians about the (unit) `axis`.
    pub fn from_axis_angle(axis: Vec3<T>, angle: T) -> Self {
        let s = (angle * lit(0.5)).sin();
        Quat([axis[0] * s, axis[1] * s, axis[2] * s, (angle * lit(0.5)).cos()])
    }
    /// Rotation by `angle` radians about a coordinate axis.
    pub fn from_axis(axis: Axis, angle: T) -> Self {
        let s = (angle * lit(0.5)).sin();
        let pick = |a: Axis| if axis == a { s } else { T::zero() };
        Quat([pick(Axis::X), pick(Axis::Y), pick(Axis::Z), (angle * lit(0.5)).cos()])
    }
    /// Quaternion from a rotation matrix without checking that it is a rotation
    /// (the `UnsafeConstruct` constructor).
    pub fn from_mat3_unchecked(rot: &Mat3<T>) -> Self {
        let r = &rot.0;
        let trace = rot.trace();
        let quarter: T = lit(0.25);
        let half: T = lit(0.5);
        if trace > T::zero() {
            let q_w = half * (trace + T::one()).sqrt();
            let f = quarter / q_w;
            Quat([f * (r[1][2] - r[2][1]), f * (r[2][0] - r[0][2]), f * (r[0][1] - r[1][0]), q_w])
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let q_x = half * (r[0][0] - r[1][1] - r[2][2] + T::one()).sqrt();
            let f = quarter / q_x;
            Quat([q_x, f * (r[0][1] + r[1][0]), f * (r[2][0] + r[0][2]), f * (r[1][2] - r[2][1])])
        } else if r[1][1] > r[2][2] {
            let q_y = half * (r[1][1] - r[0][0] - r[2][2] + T::one()).sqrt();
            let f = quarter / q_y;
            Quat([f * (r[0][1] + r[1][0]), q_y, f * (r[1][2] + r[2][1]), f * (r[2][0] - r[0][2])])
        } else {
            let q_z = half * (r[2][2] - r[0][0] - r[1][1] + T::one()).sqrt();
            let f = quarter / q_z;
            Quat([f * (r[2][0] + r[0][2]), f * (r[1][2] + r[2][1]), q_z, f * (r[0][1] - r[1][0])])
        }
    }
    /// The x component.
    pub fn x(&self) -> T {
        self.0[0]
    }
    /// The y component.
    pub fn y(&self) -> T {
        self.0[1]
    }
    /// The z component.
    pub fn z(&self) -> T {
        self.0[2]
    }
    /// The scalar component.
    pub fn w(&self) -> T {
        self.0[3]
    }
    /// The rotation angle, in radians (0 for a near-identity quaternion).
    pub fn angle(&self) -> T {
        let sqr = self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2];
        if sqr > lit(1.0e-8) {
            lit::<T>(2.0) * self.0[3].acos()
        } else {
            T::zero()
        }
    }
    /// The rotation axis (`(1, 0, 0)` for a near-identity quaternion).
    pub fn axis(&self) -> Vec3<T> {
        let sqr = self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2];
        if sqr > lit(1.0e-8) {
            let inv = T::one() / sqr.sqrt();
            Vec3::new(self.0[0] * inv, self.0[1] * inv, self.0[2] * inv)
        } else {
            Vec3::new(T::one(), T::zero(), T::zero())
        }
    }
    /// Set to the rotation by `angle` about `axis` (`setAxisAngle`).
    pub fn set_axis_angle(&mut self, axis: Vec3<T>, angle: T) -> &mut Self {
        *self = Self::from_axis_angle(axis, angle);
        self
    }
    /// True if all components differ by at most `eps` (`Quat::eq`, default 1e-7).
    pub fn is_approx_equal(&self, q: &Self, eps: T) -> bool {
        (0..4).all(|i| is_approx_equal_tol(self.0[i], q.0[i], eps))
    }
    /// Dot product.
    pub fn dot(&self, q: &Self) -> T {
        self.0[0] * q.0[0] + self.0[1] * q.0[1] + self.0[2] * q.0[2] + self.0[3] * q.0[3]
    }
    /// The time derivative of this quaternion for angular velocity `omega`.
    pub fn derivative(&self, omega: Vec3<T>) -> Self {
        let (x, y, z, w) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        Quat([
            w * omega[0] - z * omega[1] + y * omega[2],
            z * omega[0] + w * omega[1] - x * omega[2],
            -y * omega[0] + x * omega[1] + w * omega[2],
            -x * omega[0] - y * omega[1] - z * omega[2],
        ])
    }
    /// Normalize in place; returns `false` if the length is within `eps` (1e-8 in OpenVDB) of 0.
    pub fn normalize(&mut self, eps: T) -> bool {
        let d = self.dot(self).sqrt();
        if is_approx_equal_tol(d, T::zero(), eps) {
            return false;
        }
        *self = *self * (T::one() / d);
        true
    }
    /// A unit-length copy, or `None` for the zero quaternion.
    pub fn unit(&self) -> Option<Self> {
        let d = self.dot(self).sqrt();
        if d == T::zero() {
            None
        } else {
            Some(*self / d)
        }
    }
    /// The multiplicative inverse, or `None` if the squared length is within `tolerance` of 0.
    pub fn inverse(&self, tolerance: T) -> Option<Self> {
        let d = self.dot(self);
        if is_approx_equal_tol(d, T::zero(), tolerance) {
            return None;
        }
        let mut r = *self / -d;
        r.0[3] = -r.0[3];
        Some(r)
    }
    /// The conjugate `(-x, -y, -z, w)`.
    pub fn conjugate(&self) -> Self {
        Quat([-self.0[0], -self.0[1], -self.0[2], self.0[3]])
    }
    /// The rotation matrix of this quaternion (`math::rotation<Mat3>(q)`), row-vector
    /// convention. The quaternion need not be normalized.
    pub fn to_mat3(&self) -> Mat3<T> {
        let qdot = self.dot(self);
        let s = if !is_approx_equal_tol(qdot, T::zero(), lit(1.0e-8)) { lit::<T>(2.0) / qdot } else { T::zero() };
        let (qx, qy, qz, qw) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let x = s * qx;
        let y = s * qy;
        let z = s * qz;
        let wx = x * qw;
        let wy = y * qw;
        let wz = z * qw;
        let xx = x * qx;
        let xy = y * qx;
        let xz = z * qx;
        let yy = y * qy;
        let yz = z * qy;
        let zz = z * qz;
        let o = T::one();
        Mat3([[o - (yy + zz), xy + wz, xz - wy], [xy - wz, o - (xx + zz), yz + wx], [xz + wy, yz - wx, o - (xx + yy)]])
    }
    /// Rotate a vector by this quaternion (`rotateVector`).
    pub fn rotate_vector(&self, v: Vec3<T>) -> Vec3<T> {
        self.to_mat3().transform(v)
    }
}

impl<T: FloatTolerance> Quat<T> {
    /// Quaternion from a rotation matrix; `None` if the matrix is not a rotation (non-unitary or
    /// a reflection).
    pub fn from_mat3(rot: &Mat3<T>) -> Option<Self> {
        if !rot.is_unitary() || !super::math_util::is_approx_equal(rot.det(), T::one()) {
            return None;
        }
        Some(Self::from_mat3_unchecked(rot))
    }
}

impl<T: Float> Mat3<T> {
    /// The rotation matrix of a quaternion (`Mat3(Quat)`), row-vector convention.
    pub fn from_quat(q: &Quat<T>) -> Self {
        q.to_mat3()
    }
}

/// Spherical linear interpolation between two unit quaternions (`math::slerp`, tolerance
/// 1e-5 in OpenVDB).
pub fn slerp<T: Float>(q1: &Quat<T>, q2: &Quat<T>, t: T, tolerance: T) -> Quat<T> {
    let qdot = q1.dot(q2);
    let (angle, sine_angle) = if qdot.abs() >= T::one() {
        (T::zero(), T::zero())
    } else {
        let a = qdot.acos();
        (a, a.sin())
    };
    if sine_angle <= tolerance {
        let s = T::one() - t;
        let mut q = Quat([
            s * q1.0[0] + t * q2.0[0],
            s * q1.0[1] + t * q2.0[1],
            s * q1.0[2] + t * q2.0[2],
            s * q1.0[3] + t * q2.0[3],
        ]);
        let length_sqr = q.dot(&q);
        if length_sqr <= tolerance * tolerance {
            q = if t < lit(0.5) { *q1 } else { *q2 };
        } else {
            q = q * (T::one() / length_sqr.sqrt());
        }
        q
    } else {
        let sine = T::one() / sine_angle;
        let a = ((T::one() - t) * angle).sin() * sine;
        let b = (t * angle).sin() * sine;
        Quat([
            a * q1.0[0] + b * q2.0[0],
            a * q1.0[1] + b * q2.0[1],
            a * q1.0[2] + b * q2.0[2],
            a * q1.0[3] + b * q2.0[3],
        ])
    }
}

/// Interpolate between two rotation matrices via quaternion slerp (`math::slerp(Mat3, Mat3, t)`).
/// Returns `None` if either matrix is not a rotation.
pub fn slerp_mat3<T: FloatTolerance>(m1: &Mat3<T>, m2: &Mat3<T>, t: T) -> Option<Mat3<T>> {
    let q1 = Quat::from_mat3(m1)?;
    let mut q2 = Quat::from_mat3(m2)?;
    if q1.dot(&q2) < T::zero() {
        q2 = q2 * -T::one();
    }
    Some(slerp(&q1, &q2, t, lit(0.00001)).to_mat3())
}

impl<T> Index<usize> for Quat<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}
impl<T> IndexMut<usize> for Quat<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

impl<T: Float> Add for Quat<T> {
    type Output = Self;
    fn add(self, q: Self) -> Self {
        Quat([self.0[0] + q.0[0], self.0[1] + q.0[1], self.0[2] + q.0[2], self.0[3] + q.0[3]])
    }
}
impl<T: Float> Sub for Quat<T> {
    type Output = Self;
    fn sub(self, q: Self) -> Self {
        Quat([self.0[0] - q.0[0], self.0[1] - q.0[1], self.0[2] - q.0[2], self.0[3] - q.0[3]])
    }
}
impl<T: Float> AddAssign for Quat<T> {
    fn add_assign(&mut self, q: Self) {
        *self = *self + q;
    }
}
impl<T: Float> SubAssign for Quat<T> {
    fn sub_assign(&mut self, q: Self) {
        *self = *self - q;
    }
}
impl<T: Float> Mul for Quat<T> {
    type Output = Self;
    /// Hamilton product.
    fn mul(self, q: Self) -> Self {
        let (a, b) = (&self.0, &q.0);
        Quat([
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2],
            a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ])
    }
}
impl<T: Float> MulAssign for Quat<T> {
    fn mul_assign(&mut self, q: Self) {
        *self = *self * q;
    }
}
impl<T: Float> Mul<T> for Quat<T> {
    type Output = Self;
    fn mul(self, s: T) -> Self {
        Quat([self.0[0] * s, self.0[1] * s, self.0[2] * s, self.0[3] * s])
    }
}
impl<T: Float> MulAssign<T> for Quat<T> {
    fn mul_assign(&mut self, s: T) {
        *self = *self * s;
    }
}
impl<T: Float> Div<T> for Quat<T> {
    type Output = Self;
    fn div(self, s: T) -> Self {
        Quat([self.0[0] / s, self.0[1] / s, self.0[2] / s, self.0[3] / s])
    }
}
impl<T: Float> Neg for Quat<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Quat([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

impl<T: fmt::Debug> fmt::Debug for Quat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}, {:?}, {:?}, {:?}]", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}
impl<T: fmt::Display> fmt::Display for Quat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Mat3d, Vec3d};

    #[test]
    fn rotation_matches_matrix() {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let q = Quatd::from_axis(axis, 0.6);
            assert!(q.to_mat3().is_approx_equal(&Mat3d::rotation(axis, 0.6), 1e-15));
        }
        let axis = Vec3d::new(1.0, 2.0, 3.0).unit();
        let q = Quatd::from_axis_angle(axis, 0.6);
        assert!(q.to_mat3().is_approx_equal(&Mat3d::rotation_axis_angle(axis, 0.6), 1e-15));
        assert!((q.angle() - 0.6).abs() < 1e-12);
        assert!(q.axis().is_approx_equal(&axis, 1e-12));
        let back = Quatd::from_mat3(&q.to_mat3()).unwrap();
        assert!(back.is_approx_equal(&q, 1e-12));
        assert!(Quatd::from_mat3(&Mat3d::from_diagonal(Vec3d::new(1.0, 1.0, -1.0))).is_none());
        let v = Vec3d::new(0.3, -0.2, 0.9);
        assert!(q.rotate_vector(v).is_approx_equal(&Mat3d::from_quat(&q).transform(v), 1e-15));
    }

    #[test]
    fn algebra() {
        let a = Quatd::from_axis(Axis::Z, 0.3);
        let b = Quatd::from_axis(Axis::Z, 0.5);
        assert!((a * b).is_approx_equal(&Quatd::from_axis(Axis::Z, 0.8), 1e-15));
        assert!((a * a.inverse(0.0).unwrap()).is_approx_equal(&Quatd::identity(), 1e-15));
        assert_eq!(a.conjugate().w(), a.w());
        let mut z = Quatd::zero();
        assert!(!z.normalize(1e-8));
        assert!(z.unit().is_none());
        let m = slerp(&Quatd::identity(), &b, 0.5, 1e-5);
        assert!(m.is_approx_equal(&Quatd::from_axis(Axis::Z, 0.25), 1e-12));
        let r = slerp_mat3(&Mat3d::identity(), &Mat3d::rotation(Axis::X, 1.0), 0.25).unwrap();
        assert!(r.is_approx_equal(&Mat3d::rotation(Axis::X, 0.25), 1e-12));
        let d = Quatd::identity().derivative(Vec3d::new(1.0, 0.0, 0.0));
        assert_eq!(d, Quatd::new(1.0, 0.0, 0.0, 0.0));
    }
}
