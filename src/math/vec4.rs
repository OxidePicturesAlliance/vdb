//! Four-component vector type (port of `openvdb/math/Vec4.h`).

use super::vec3::Vec3;
use num_traits::{Float, NumCast, Zero};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// A generic four-component vector.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Hash, Eq)]
#[repr(C)]
pub struct Vec4<T>(pub [T; 4]);

/// Single-precision 4-vector.
pub type Vec4f = Vec4<f32>;
/// Single-precision 4-vector (OpenVDB's `Vec4s`).
pub type Vec4s = Vec4<f32>;
/// Double-precision 4-vector.
pub type Vec4d = Vec4<f64>;
/// 32-bit integer 4-vector.
pub type Vec4i = Vec4<i32>;
/// Unsigned 32-bit integer 4-vector.
pub type Vec4ui = Vec4<u32>;
/// Half-precision 4-vector.
pub type Vec4h = Vec4<half::f16>;

impl<T: Copy> Vec4<T> {
    /// Construct from components.
    #[inline]
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Vec4([x, y, z, w])
    }
    /// All components equal to `v`.
    #[inline]
    pub const fn splat(v: T) -> Self {
        Vec4([v, v, v, v])
    }
    /// Construct from a 3-vector and a fourth component.
    #[inline]
    pub fn from_vec3(v: Vec3<T>, w: T) -> Self {
        Vec4([v[0], v[1], v[2], w])
    }
    /// The x component.
    #[inline]
    pub fn x(&self) -> T {
        self.0[0]
    }
    /// The y component.
    #[inline]
    pub fn y(&self) -> T {
        self.0[1]
    }
    /// The z component.
    #[inline]
    pub fn z(&self) -> T {
        self.0[2]
    }
    /// The w component.
    #[inline]
    pub fn w(&self) -> T {
        self.0[3]
    }
    /// The first three components.
    #[inline]
    pub fn xyz(&self) -> Vec3<T> {
        Vec3([self.0[0], self.0[1], self.0[2]])
    }
    /// The components as an array.
    #[inline]
    pub fn as_array(&self) -> [T; 4] {
        self.0
    }
    /// Apply `f` to every component.
    #[inline]
    pub fn map<U: Copy>(self, f: impl Fn(T) -> U) -> Vec4<U> {
        Vec4([f(self.0[0]), f(self.0[1]), f(self.0[2]), f(self.0[3])])
    }
    /// Combine two vectors component-wise.
    #[inline]
    pub fn zip_map<U: Copy, R: Copy>(self, o: Vec4<U>, f: impl Fn(T, U) -> R) -> Vec4<R> {
        Vec4([f(self.0[0], o.0[0]), f(self.0[1], o.0[1]), f(self.0[2], o.0[2]), f(self.0[3], o.0[3])])
    }
}

impl<T: Copy + NumCast> Vec4<T> {
    /// Numeric cast of every component (panics if a component is not representable).
    #[inline]
    pub fn cast<U: Copy + NumCast>(self) -> Vec4<U> {
        self.map(|v| U::from(v).expect("Vec4::cast: value not representable"))
    }
}

impl<T> Vec4<T>
where
    T: Copy + Zero + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + PartialOrd,
{
    /// The zero vector.
    #[inline]
    pub fn zero() -> Self {
        Vec4([T::zero(); 4])
    }
    /// Dot product.
    #[inline]
    pub fn dot(&self, o: &Self) -> T {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2] + self.0[3] * o.0[3]
    }
    /// Squared length.
    #[inline]
    pub fn length_sqr(&self) -> T {
        self.dot(self)
    }
    /// Sum of the components.
    #[inline]
    pub fn sum(&self) -> T {
        self.0[0] + self.0[1] + self.0[2] + self.0[3]
    }
    /// Product of the components.
    #[inline]
    pub fn product(&self) -> T {
        self.0[0] * self.0[1] * self.0[2] * self.0[3]
    }
    /// Smallest component.
    #[inline]
    pub fn min_component(&self) -> T {
        self.0.iter().skip(1).fold(self.0[0], |a, &b| if b < a { b } else { a })
    }
    /// Largest component.
    #[inline]
    pub fn max_component(&self) -> T {
        self.0.iter().skip(1).fold(self.0[0], |a, &b| if b > a { b } else { a })
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
}

impl<T: Float> Vec4<T> {
    /// Euclidean length.
    #[inline]
    pub fn length(&self) -> T {
        self.length_sqr().sqrt()
    }
    /// A unit-length copy (or the zero vector if the length is ~0).
    #[inline]
    pub fn unit(&self) -> Self {
        let l = self.length();
        if l <= T::epsilon() {
            Vec4::zero()
        } else {
            *self / l
        }
    }
    /// Normalize in place; returns `false` if the length was approximately zero (1e-7).
    #[inline]
    pub fn normalize(&mut self) -> bool {
        let d = self.length();
        if d.abs() <= T::from(1.0e-7).unwrap() {
            return false;
        }
        *self = *self * (T::one() / d);
        true
    }
    /// Component-wise absolute value.
    #[inline]
    pub fn abs(&self) -> Self {
        self.map(|v| v.abs())
    }
    /// True if all components differ by at most `tol` (`Vec4::eq`).
    #[inline]
    pub fn is_approx_equal(&self, o: &Self, tol: T) -> bool {
        (0..4).all(|i| !((self.0[i] - o.0[i]).abs() > tol))
    }
    /// True if all components are finite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
    /// True if any component is NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.0.iter().any(|v| v.is_nan())
    }
}

impl<T> Index<usize> for Vec4<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}
impl<T> IndexMut<usize> for Vec4<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

macro_rules! vec4_binop {
    ($tr:ident, $f:ident, $atr:ident, $af:ident, $op:tt) => {
        impl<T: Copy + $tr<Output = T>> $tr for Vec4<T> {
            type Output = Self;
            #[inline]
            fn $f(self, o: Self) -> Self {
                Vec4([self.0[0] $op o.0[0], self.0[1] $op o.0[1], self.0[2] $op o.0[2], self.0[3] $op o.0[3]])
            }
        }
        impl<T: Copy + $tr<Output = T>> $tr<T> for Vec4<T> {
            type Output = Self;
            #[inline]
            fn $f(self, s: T) -> Self {
                Vec4([self.0[0] $op s, self.0[1] $op s, self.0[2] $op s, self.0[3] $op s])
            }
        }
        impl<T: Copy + $tr<Output = T>> $atr for Vec4<T> {
            #[inline]
            fn $af(&mut self, o: Self) {
                *self = *self $op o;
            }
        }
        impl<T: Copy + $tr<Output = T>> $atr<T> for Vec4<T> {
            #[inline]
            fn $af(&mut self, s: T) {
                *self = *self $op s;
            }
        }
    };
}
vec4_binop!(Add, add, AddAssign, add_assign, +);
vec4_binop!(Sub, sub, SubAssign, sub_assign, -);
vec4_binop!(Mul, mul, MulAssign, mul_assign, *);
vec4_binop!(Div, div, DivAssign, div_assign, /);

impl<T: Copy + Neg<Output = T>> Neg for Vec4<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Vec4([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

macro_rules! scalar_mul_vec4 {
    ($($t:ty),*) => {$(
        impl Mul<Vec4<$t>> for $t {
            type Output = Vec4<$t>;
            #[inline]
            fn mul(self, v: Vec4<$t>) -> Vec4<$t> { v * self }
        }
    )*};
}
scalar_mul_vec4!(f32, f64, i32, i64, u32);

impl<T: fmt::Debug> fmt::Debug for Vec4<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}, {:?}, {:?}, {:?}]", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}
impl<T: fmt::Display> fmt::Display for Vec4<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

impl<T> From<[T; 4]> for Vec4<T> {
    fn from(a: [T; 4]) -> Self {
        Vec4(a)
    }
}
impl<T> From<Vec4<T>> for [T; 4] {
    fn from(v: Vec4<T>) -> Self {
        v.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_ops() {
        let a = Vec4d::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4d::splat(2.0);
        assert_eq!(a + b, Vec4d::new(3.0, 4.0, 5.0, 6.0));
        assert_eq!(a.dot(&b), 20.0);
        assert_eq!(a.max_component(), 4.0);
        assert_eq!(a.min_component(), 1.0);
        assert_eq!(a.xyz(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(2.0 * a, a * 2.0);
        let mut u = Vec4d::new(0.0, 3.0, 0.0, 4.0);
        assert!(u.normalize());
        assert!(u.is_approx_equal(&Vec4d::new(0.0, 0.6, 0.0, 0.8), 1e-15));
        assert_eq!(Vec4i::new(1, 2, 3, 4).cast::<f32>(), Vec4f::new(1.0, 2.0, 3.0, 4.0));
    }
}
