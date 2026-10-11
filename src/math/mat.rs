//! 3x3 and 4x4 matrices (port of `openvdb/math/Mat3.h` and `openvdb/math/Mat4.h`).
//!
//! Like OpenVDB, matrices are stored row-major and use the *row-vector* convention:
//! a point is transformed as `p' = p * M`, so the translation lives in the last row and
//! `A * B` applies `A` first, then `B`.

use super::vec3::Vec3;
use num_traits::Float;
use std::fmt;
use std::ops::{Index, IndexMut, Mul};

/// Coordinate axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

/// A 3x3 matrix, row-major.
#[derive(Clone, Copy, PartialEq)]
pub struct Mat3<T>(pub [[T; 3]; 3]);
pub type Mat3s = Mat3<f32>;
pub type Mat3d = Mat3<f64>;

/// A 4x4 matrix, row-major.
#[derive(Clone, Copy, PartialEq)]
pub struct Mat4<T>(pub [[T; 4]; 4]);
pub type Mat4s = Mat4<f32>;
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

impl<T: Float> Mat3<T> {
    pub fn zero() -> Self {
        Mat3([[T::zero(); 3]; 3])
    }
    pub fn identity() -> Self {
        let (o, z) = (T::one(), T::zero());
        Mat3([[o, z, z], [z, o, z], [z, z, o]])
    }
    pub fn from_rows(r0: Vec3<T>, r1: Vec3<T>, r2: Vec3<T>) -> Self {
        Mat3([r0.0, r1.0, r2.0])
    }
    pub fn from_diagonal(d: Vec3<T>) -> Self {
        let mut m = Self::zero();
        for i in 0..3 {
            m.0[i][i] = d[i];
        }
        m
    }
    pub fn row(&self, i: usize) -> Vec3<T> {
        Vec3(self.0[i])
    }
    pub fn col(&self, j: usize) -> Vec3<T> {
        Vec3([self.0[0][j], self.0[1][j], self.0[2][j]])
    }
    pub fn set_row(&mut self, i: usize, v: Vec3<T>) {
        self.0[i] = v.0;
    }
    pub fn set_col(&mut self, j: usize, v: Vec3<T>) {
        for i in 0..3 {
            self.0[i][j] = v[i];
        }
    }
    pub fn transpose(&self) -> Self {
        let mut r = *self;
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = self.0[j][i];
            }
        }
        r
    }
    pub fn det(&self) -> T {
        let m = &self.0;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
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
        let d = self.det();
        if d.abs() <= tol {
            return None;
        }
        let a = self.adjoint();
        let inv = T::one() / d;
        let mut r = a;
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = a.0[i][j] * inv;
            }
        }
        Some(r)
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
    /// Column-vector transform: `M * v`.
    pub fn mul_vec(&self, v: Vec3<T>) -> Vec3<T> {
        let m = &self.0;
        Vec3([
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ])
    }
    pub fn trace(&self) -> T {
        self.0[0][0] + self.0[1][1] + self.0[2][2]
    }
    pub fn is_approx_equal(&self, o: &Self, tol: T) -> bool {
        (0..3).all(|i| (0..3).all(|j| (self.0[i][j] - o.0[i][j]).abs() <= tol))
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

impl<T: Float> Mat4<T> {
    pub fn zero() -> Self {
        Mat4([[T::zero(); 4]; 4])
    }
    pub fn identity() -> Self {
        let mut m = Self::zero();
        for i in 0..4 {
            m.0[i][i] = T::one();
        }
        m
    }
    pub fn from_mat3_translation(m3: &Mat3<T>, t: Vec3<T>) -> Self {
        let mut m = Self::identity();
        m.set_mat3(m3);
        m.set_translation(t);
        m
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
    pub fn set_mat3(&mut self, m: &Mat3<T>) {
        for i in 0..3 {
            for j in 0..3 {
                self.0[i][j] = m.0[i][j];
            }
        }
    }
    pub fn get_translation(&self) -> Vec3<T> {
        Vec3([self.0[3][0], self.0[3][1], self.0[3][2]])
    }
    pub fn set_translation(&mut self, t: Vec3<T>) {
        self.0[3][0] = t[0];
        self.0[3][1] = t[1];
        self.0[3][2] = t[2];
    }
    pub fn translation(t: Vec3<T>) -> Self {
        let mut m = Self::identity();
        m.set_translation(t);
        m
    }
    pub fn scale(s: Vec3<T>) -> Self {
        let mut m = Self::identity();
        for i in 0..3 {
            m.0[i][i] = s[i];
        }
        m
    }
    pub fn rotation(axis: Axis, angle: T) -> Self {
        Self::from_mat3_translation(&Mat3::rotation(axis, angle), Vec3([T::zero(); 3]))
    }
    pub fn transpose(&self) -> Self {
        let mut r = *self;
        for i in 0..4 {
            for j in 0..4 {
                r.0[i][j] = self.0[j][i];
            }
        }
        r
    }
    pub fn det(&self) -> T {
        let m = &self.0;
        let mut d = T::zero();
        for j in 0..4 {
            let minor = self.minor3(0, j);
            let sign = if j % 2 == 0 { T::one() } else { -T::one() };
            d = d + sign * m[0][j] * minor.det();
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
    /// General inverse via Gauss-Jordan elimination with partial pivoting.
    pub fn inverse(&self, tol: T) -> Option<Self> {
        let mut a = self.0;
        let mut inv = Self::identity().0;
        for c in 0..4 {
            let mut p = c;
            for r in (c + 1)..4 {
                if a[r][c].abs() > a[p][c].abs() {
                    p = r;
                }
            }
            if a[p][c].abs() <= tol {
                return None;
            }
            a.swap(c, p);
            inv.swap(c, p);
            let d = T::one() / a[c][c];
            for j in 0..4 {
                a[c][j] = a[c][j] * d;
                inv[c][j] = inv[c][j] * d;
            }
            for r in 0..4 {
                if r != c {
                    let f = a[r][c];
                    if f != T::zero() {
                        for j in 0..4 {
                            a[r][j] = a[r][j] - f * a[c][j];
                            inv[r][j] = inv[r][j] - f * inv[c][j];
                        }
                    }
                }
            }
        }
        Some(Mat4(inv))
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
    /// Transform a direction (ignores translation).
    pub fn transform3x3(&self, v: Vec3<T>) -> Vec3<T> {
        let m = &self.0;
        Vec3([
            v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
            v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
            v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
        ])
    }
    /// `M = S * M` (scale applied before the existing transform).
    pub fn pre_scale(&mut self, s: Vec3<T>) {
        *self = Self::scale(s) * *self;
    }
    /// `M = M * S` (scale applied after the existing transform).
    pub fn post_scale(&mut self, s: Vec3<T>) {
        *self = *self * Self::scale(s);
    }
    pub fn pre_translate(&mut self, t: Vec3<T>) {
        *self = Self::translation(t) * *self;
    }
    pub fn post_translate(&mut self, t: Vec3<T>) {
        *self = *self * Self::translation(t);
    }
    pub fn pre_rotate(&mut self, axis: Axis, angle: T) {
        *self = Self::rotation(axis, angle) * *self;
    }
    pub fn post_rotate(&mut self, axis: Axis, angle: T) {
        *self = *self * Self::rotation(axis, angle);
    }
    pub fn is_approx_equal(&self, o: &Self, tol: T) -> bool {
        (0..4).all(|i| (0..4).all(|j| (self.0[i][j] - o.0[i][j]).abs() <= tol))
    }
    /// True if the last column is (0, 0, 0, 1).
    pub fn is_affine(&self) -> bool {
        let m = &self.0;
        m[0][3] == T::zero() && m[1][3] == T::zero() && m[2][3] == T::zero() && m[3][3] == T::one()
    }
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

impl<T: Float> Mul for Mat4<T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let mut r = Self::zero();
        for i in 0..4 {
            for j in 0..4 {
                let mut s = T::zero();
                for k in 0..4 {
                    s = s + self.0[i][k] * o.0[k][j];
                }
                r.0[i][j] = s;
            }
        }
        r
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
    }

    #[test]
    fn row_vector_convention() {
        let m = Mat4d::translation(Vec3d::new(1.0, 2.0, 3.0));
        assert_eq!(m.transform(Vec3d::zero()), Vec3d::new(1.0, 2.0, 3.0));
        let r = Mat3d::rotation(Axis::Z, std::f64::consts::FRAC_PI_2);
        assert!(r.transform(Vec3d::new(1.0, 0.0, 0.0)).is_approx_equal(&Vec3d::new(0.0, 1.0, 0.0), 1e-12));
    }
}
