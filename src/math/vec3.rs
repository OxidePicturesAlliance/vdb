//! Three-component vector type (port of `openvdb/math/Vec3.h`).

use num_traits::{Float, NumCast, Zero};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// A generic three-component vector.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Hash, Eq)]
#[repr(C)]
pub struct Vec3<T>(pub [T; 3]);

pub type Vec3f = Vec3<f32>;
pub type Vec3s = Vec3<f32>;
pub type Vec3d = Vec3<f64>;
pub type Vec3i = Vec3<i32>;
pub type Vec3ui = Vec3<u32>;
pub type Vec3h = Vec3<half::f16>;

impl<T: Copy> Vec3<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Vec3([x, y, z])
    }
    #[inline]
    pub const fn splat(v: T) -> Self {
        Vec3([v, v, v])
    }
    #[inline]
    pub fn x(&self) -> T {
        self.0[0]
    }
    #[inline]
    pub fn y(&self) -> T {
        self.0[1]
    }
    #[inline]
    pub fn z(&self) -> T {
        self.0[2]
    }
    #[inline]
    pub fn as_array(&self) -> [T; 3] {
        self.0
    }
    /// Apply `f` to every component.
    #[inline]
    pub fn map<U: Copy>(self, f: impl Fn(T) -> U) -> Vec3<U> {
        Vec3([f(self.0[0]), f(self.0[1]), f(self.0[2])])
    }
    /// Combine two vectors component-wise.
    #[inline]
    pub fn zip_map<U: Copy, R: Copy>(self, o: Vec3<U>, f: impl Fn(T, U) -> R) -> Vec3<R> {
        Vec3([f(self.0[0], o.0[0]), f(self.0[1], o.0[1]), f(self.0[2], o.0[2])])
    }
}

impl<T: Copy + NumCast> Vec3<T> {
    /// Numeric cast of every component (panics if a component is not representable).
    #[inline]
    pub fn cast<U: Copy + NumCast>(self) -> Vec3<U> {
        self.map(|v| U::from(v).expect("Vec3::cast: value not representable"))
    }
}

impl<T> Vec3<T>
where
    T: Copy + Zero + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + PartialOrd,
{
    #[inline]
    pub fn zero() -> Self {
        Vec3([T::zero(); 3])
    }
    #[inline]
    pub fn dot(&self, o: &Self) -> T {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }
    #[inline]
    pub fn cross(&self, o: &Self) -> Self {
        Vec3([
            self.0[1] * o.0[2] - self.0[2] * o.0[1],
            self.0[2] * o.0[0] - self.0[0] * o.0[2],
            self.0[0] * o.0[1] - self.0[1] * o.0[0],
        ])
    }
    #[inline]
    pub fn length_sqr(&self) -> T {
        self.dot(self)
    }
    #[inline]
    pub fn sum(&self) -> T {
        self.0[0] + self.0[1] + self.0[2]
    }
    #[inline]
    pub fn product(&self) -> T {
        self.0[0] * self.0[1] * self.0[2]
    }
    #[inline]
    pub fn min_component(&self) -> T {
        let m = if self.0[0] < self.0[1] { self.0[0] } else { self.0[1] };
        if m < self.0[2] {
            m
        } else {
            self.0[2]
        }
    }
    #[inline]
    pub fn max_component(&self) -> T {
        let m = if self.0[0] > self.0[1] { self.0[0] } else { self.0[1] };
        if m > self.0[2] {
            m
        } else {
            self.0[2]
        }
    }
    /// Component-wise minimum.
    #[inline]
    pub fn min_elem(&self, o: &Self) -> Self {
        self.zip_map(*o, |a, b| if b < a { b } else { a })
    }
    /// Component-wise maximum.
    #[inline]
    pub fn max_elem(&self, o: &Self) -> Self {
        self.zip_map(*o, |a, b| if b > a { b } else { a })
    }
    /// Component-wise product.
    #[inline]
    pub fn mul_elem(&self, o: &Self) -> Self {
        self.zip_map(*o, |a, b| a * b)
    }
    /// Index (0, 1 or 2) of the largest component.
    #[inline]
    pub fn max_index(&self) -> usize {
        let v = &self.0;
        if v[0] > v[1] {
            if v[0] > v[2] {
                0
            } else {
                2
            }
        } else if v[1] > v[2] {
            1
        } else {
            2
        }
    }
    /// Index (0, 1 or 2) of the smallest component.
    #[inline]
    pub fn min_index(&self) -> usize {
        let v = &self.0;
        if v[0] < v[1] {
            if v[0] < v[2] {
                0
            } else {
                2
            }
        } else if v[1] < v[2] {
            1
        } else {
            2
        }
    }
}

impl<T: Float> Vec3<T> {
    #[inline]
    pub fn length(&self) -> T {
        self.length_sqr().sqrt()
    }
    /// Return a unit-length copy of this vector (or the zero vector if its length is ~0).
    #[inline]
    pub fn unit(&self) -> Self {
        let l = self.length();
        if l <= T::epsilon() {
            Vec3::zero()
        } else {
            *self / l
        }
    }
    /// Normalize in place; returns `false` if the vector had (near) zero length.
    #[inline]
    pub fn normalize(&mut self) -> bool {
        let l = self.length();
        if l <= T::epsilon() {
            return false;
        }
        *self /= l;
        true
    }
    #[inline]
    pub fn abs(&self) -> Self {
        self.map(|v| v.abs())
    }
    #[inline]
    pub fn floor(&self) -> Self {
        self.map(|v| v.floor())
    }
    #[inline]
    pub fn ceil(&self) -> Self {
        self.map(|v| v.ceil())
    }
    #[inline]
    pub fn round(&self) -> Self {
        self.map(|v| v.round())
    }
    #[inline]
    pub fn is_approx_equal(&self, o: &Self, tol: T) -> bool {
        (0..3).all(|i| (self.0[i] - o.0[i]).abs() <= tol)
    }
    /// OpenVDB's `Vec3::eq(v, eps)`: every component is equal to within `eps`, absolutely or
    /// relatively (`math::isRelOrApproxEqual(a, b, eps, eps)`). With `eps = 1e-7` this is
    /// `math::isApproxEqual(Vec3, Vec3)`.
    #[inline]
    pub fn eq_eps(&self, o: &Self, eps: T) -> bool {
        (0..3).all(|i| super::math_util::is_rel_or_approx_equal(self.0[i], o.0[i], eps, eps))
    }
    /// OpenVDB's `Vec3::normalize(eps)`: returns false (leaving the vector untouched) if the
    /// length is within `eps` of zero; otherwise multiplies by the reciprocal length.
    #[inline]
    pub fn normalize_eps(&mut self, eps: T) -> bool {
        let d = self.length();
        if !(d.abs() > eps) {
            return false;
        }
        let inv = T::one() / d;
        *self = Vec3([self.0[0] * inv, self.0[1] * inv, self.0[2] * inv]);
        true
    }
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.0.iter().any(|v| v.is_nan())
    }
    #[inline]
    pub fn lerp(&self, o: &Self, t: T) -> Self {
        *self + (*o - *self) * t
    }
}

impl<T> Index<usize> for Vec3<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}
impl<T> IndexMut<usize> for Vec3<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

macro_rules! vec3_binop {
    ($tr:ident, $f:ident, $atr:ident, $af:ident, $op:tt) => {
        impl<T: Copy + $tr<Output = T>> $tr for Vec3<T> {
            type Output = Self;
            #[inline]
            fn $f(self, o: Self) -> Self {
                Vec3([self.0[0] $op o.0[0], self.0[1] $op o.0[1], self.0[2] $op o.0[2]])
            }
        }
        impl<T: Copy + $tr<Output = T>> $tr<T> for Vec3<T> {
            type Output = Self;
            #[inline]
            fn $f(self, s: T) -> Self {
                Vec3([self.0[0] $op s, self.0[1] $op s, self.0[2] $op s])
            }
        }
        impl<T: Copy + $tr<Output = T>> $atr for Vec3<T> {
            #[inline]
            fn $af(&mut self, o: Self) {
                *self = *self $op o;
            }
        }
        impl<T: Copy + $tr<Output = T>> $atr<T> for Vec3<T> {
            #[inline]
            fn $af(&mut self, s: T) {
                *self = *self $op s;
            }
        }
    };
}
vec3_binop!(Add, add, AddAssign, add_assign, +);
vec3_binop!(Sub, sub, SubAssign, sub_assign, -);
vec3_binop!(Mul, mul, MulAssign, mul_assign, *);
vec3_binop!(Div, div, DivAssign, div_assign, /);

impl<T: Copy + Neg<Output = T>> Neg for Vec3<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Vec3([-self.0[0], -self.0[1], -self.0[2]])
    }
}

macro_rules! scalar_mul_vec3 {
    ($($t:ty),*) => {$(
        impl Mul<Vec3<$t>> for $t {
            type Output = Vec3<$t>;
            #[inline]
            fn mul(self, v: Vec3<$t>) -> Vec3<$t> { v * self }
        }
    )*};
}
scalar_mul_vec3!(f32, f64, i32, i64, u32);

impl<T: fmt::Debug> fmt::Debug for Vec3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}, {:?}, {:?}]", self.0[0], self.0[1], self.0[2])
    }
}
impl<T: fmt::Display> fmt::Display for Vec3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.0[0], self.0[1], self.0[2])
    }
}

impl<T> From<[T; 3]> for Vec3<T> {
    fn from(a: [T; 3]) -> Self {
        Vec3(a)
    }
}
impl<T> From<Vec3<T>> for [T; 3] {
    fn from(v: Vec3<T>) -> Self {
        v.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_ops() {
        let a = Vec3d::new(1.0, 2.0, 3.0);
        let b = Vec3d::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3d::new(5.0, 7.0, 9.0));
        assert_eq!(a.dot(&b), 32.0);
        assert_eq!(a.cross(&b), Vec3d::new(-3.0, 6.0, -3.0));
        assert!((Vec3d::new(3.0, 4.0, 0.0).length() - 5.0).abs() < 1e-12);
        assert_eq!(b.max_index(), 2);
        assert_eq!(2.0 * a, a * 2.0);
    }
}
