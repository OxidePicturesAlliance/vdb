//! Two-component vector type (port of `openvdb/math/Vec2.h`).

use num_traits::{Float, NumCast, Zero};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// A generic two-component vector.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Hash, Eq)]
#[repr(C)]
pub struct Vec2<T>(pub [T; 2]);

/// Single-precision 2-vector.
pub type Vec2f = Vec2<f32>;
/// Single-precision 2-vector (OpenVDB's `Vec2s`).
pub type Vec2s = Vec2<f32>;
/// Double-precision 2-vector.
pub type Vec2d = Vec2<f64>;
/// 32-bit integer 2-vector.
pub type Vec2i = Vec2<i32>;
/// Unsigned 32-bit integer 2-vector.
pub type Vec2ui = Vec2<u32>;
/// Half-precision 2-vector.
pub type Vec2h = Vec2<half::f16>;

impl<T: Copy> Vec2<T> {
    /// Construct from components.
    #[inline]
    pub const fn new(x: T, y: T) -> Self {
        Vec2([x, y])
    }
    /// Both components equal to `v`.
    #[inline]
    pub const fn splat(v: T) -> Self {
        Vec2([v, v])
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
    /// The components as an array.
    #[inline]
    pub fn as_array(&self) -> [T; 2] {
        self.0
    }
    /// Apply `f` to every component.
    #[inline]
    pub fn map<U: Copy>(self, f: impl Fn(T) -> U) -> Vec2<U> {
        Vec2([f(self.0[0]), f(self.0[1])])
    }
    /// Combine two vectors component-wise.
    #[inline]
    pub fn zip_map<U: Copy, R: Copy>(self, o: Vec2<U>, f: impl Fn(T, U) -> R) -> Vec2<R> {
        Vec2([f(self.0[0], o.0[0]), f(self.0[1], o.0[1])])
    }
}

impl<T: Copy + NumCast> Vec2<T> {
    /// Numeric cast of every component (panics if a component is not representable).
    #[inline]
    pub fn cast<U: Copy + NumCast>(self) -> Vec2<U> {
        self.map(|v| U::from(v).expect("Vec2::cast: value not representable"))
    }
}

impl<T> Vec2<T>
where
    T: Copy + Zero + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + PartialOrd,
{
    /// The zero vector.
    #[inline]
    pub fn zero() -> Self {
        Vec2([T::zero(); 2])
    }
    /// Dot product.
    #[inline]
    pub fn dot(&self, o: &Self) -> T {
        self.0[0] * o.0[0] + self.0[1] * o.0[1]
    }
    /// Squared length.
    #[inline]
    pub fn length_sqr(&self) -> T {
        self.dot(self)
    }
    /// Sum of the components.
    #[inline]
    pub fn sum(&self) -> T {
        self.0[0] + self.0[1]
    }
    /// Product of the components.
    #[inline]
    pub fn product(&self) -> T {
        self.0[0] * self.0[1]
    }
    /// Smallest component.
    #[inline]
    pub fn min_component(&self) -> T {
        if self.0[1] < self.0[0] {
            self.0[1]
        } else {
            self.0[0]
        }
    }
    /// Largest component.
    #[inline]
    pub fn max_component(&self) -> T {
        if self.0[1] > self.0[0] {
            self.0[1]
        } else {
            self.0[0]
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
}

impl<T: Float> Vec2<T> {
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
            Vec2::zero()
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
        *self *= T::one() / d;
        true
    }
    /// Component-wise absolute value.
    #[inline]
    pub fn abs(&self) -> Self {
        self.map(|v| v.abs())
    }
    /// The vector rotated by 90 degrees counter-clockwise: `(-y, x)`.
    #[inline]
    pub fn perp(&self) -> Self {
        Vec2([-self.0[1], self.0[0]])
    }
    /// True if all components differ by at most `tol` (`Vec2::eq`).
    #[inline]
    pub fn is_approx_equal(&self, o: &Self, tol: T) -> bool {
        (0..2).all(|i| !((self.0[i] - o.0[i]).abs() > tol))
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

impl<T> Index<usize> for Vec2<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}
impl<T> IndexMut<usize> for Vec2<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

macro_rules! vec2_binop {
    ($tr:ident, $f:ident, $atr:ident, $af:ident, $op:tt) => {
        impl<T: Copy + $tr<Output = T>> $tr for Vec2<T> {
            type Output = Self;
            #[inline]
            fn $f(self, o: Self) -> Self {
                Vec2([self.0[0] $op o.0[0], self.0[1] $op o.0[1]])
            }
        }
        impl<T: Copy + $tr<Output = T>> $tr<T> for Vec2<T> {
            type Output = Self;
            #[inline]
            fn $f(self, s: T) -> Self {
                Vec2([self.0[0] $op s, self.0[1] $op s])
            }
        }
        impl<T: Copy + $tr<Output = T>> $atr for Vec2<T> {
            #[inline]
            fn $af(&mut self, o: Self) {
                *self = *self $op o;
            }
        }
        impl<T: Copy + $tr<Output = T>> $atr<T> for Vec2<T> {
            #[inline]
            fn $af(&mut self, s: T) {
                *self = *self $op s;
            }
        }
    };
}
vec2_binop!(Add, add, AddAssign, add_assign, +);
vec2_binop!(Sub, sub, SubAssign, sub_assign, -);
vec2_binop!(Mul, mul, MulAssign, mul_assign, *);
vec2_binop!(Div, div, DivAssign, div_assign, /);

impl<T: Copy + Neg<Output = T>> Neg for Vec2<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Vec2([-self.0[0], -self.0[1]])
    }
}

macro_rules! scalar_mul_vec2 {
    ($($t:ty),*) => {$(
        impl Mul<Vec2<$t>> for $t {
            type Output = Vec2<$t>;
            #[inline]
            fn mul(self, v: Vec2<$t>) -> Vec2<$t> { v * self }
        }
    )*};
}
scalar_mul_vec2!(f32, f64, i32, i64, u32);

impl<T: fmt::Debug> fmt::Debug for Vec2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}, {:?}]", self.0[0], self.0[1])
    }
}
impl<T: fmt::Display> fmt::Display for Vec2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.0[0], self.0[1])
    }
}

impl<T> From<[T; 2]> for Vec2<T> {
    fn from(a: [T; 2]) -> Self {
        Vec2(a)
    }
}
impl<T> From<Vec2<T>> for [T; 2] {
    fn from(v: Vec2<T>) -> Self {
        v.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_ops() {
        let a = Vec2d::new(3.0, 4.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(a.dot(&Vec2d::new(1.0, 2.0)), 11.0);
        assert_eq!(a - Vec2d::splat(1.0), Vec2d::new(2.0, 3.0));
        assert_eq!(a.perp(), Vec2d::new(-4.0, 3.0));
        assert_eq!(a.min_component(), 3.0);
        assert!(a.unit().is_approx_equal(&Vec2d::new(0.6, 0.8), 1e-15));
        assert_eq!(2.0 * a, a * 2.0);
        assert_eq!(Vec2i::new(1, -2).cast::<f64>(), Vec2d::new(1.0, -2.0));
    }
}
