//! Core value-type traits (port of the relevant parts of `openvdb/Types.h` and `math/Math.h`).

use crate::math::{Vec3, Vec3d, Vec3f, Vec3i};
use std::fmt::Debug;

/// Index type used for voxel offsets within nodes.
pub type Index = u32;
/// 64-bit counts (voxel counts, memory usage, ...).
pub type Index64 = u64;

/// Requirements for a type that can be stored in a VDB tree.
pub trait Value: Copy + PartialEq + Default + Debug + Send + Sync + 'static {
    /// The additive identity (OpenVDB's `zeroVal<T>()`).
    #[inline]
    fn zero() -> Self {
        Self::default()
    }
    /// Approximate equality within a tolerance (OpenVDB's `math::isApproxEqual(a, b, tol)`),
    /// used by pruning.
    fn approx_eq(&self, other: &Self, tolerance: &Self) -> bool;
}

macro_rules! impl_value_float {
    ($($t:ty),*) => {$(
        impl Value for $t {
            #[inline]
            fn approx_eq(&self, o: &Self, tol: &Self) -> bool {
                // Matches math::isApproxEqual: also true for exact equality (e.g. infinities).
                self == o || (self - o).abs() <= *tol
            }
        }
    )*};
}
impl_value_float!(f32, f64);

macro_rules! impl_value_int {
    ($($t:ty),*) => {$(
        impl Value for $t {
            #[inline]
            fn approx_eq(&self, o: &Self, tol: &Self) -> bool {
                (*self as i128 - *o as i128).abs() <= *tol as i128
            }
        }
    )*};
}
impl_value_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl Value for bool {
    #[inline]
    fn approx_eq(&self, o: &Self, _tol: &Self) -> bool {
        self == o
    }
}

impl Value for half::f16 {
    #[inline]
    fn approx_eq(&self, o: &Self, tol: &Self) -> bool {
        self == o || (self.to_f32() - o.to_f32()).abs() <= tol.to_f32()
    }
}

impl<T: Value> Value for Vec3<T> {
    #[inline]
    fn approx_eq(&self, o: &Self, tol: &Self) -> bool {
        (0..3).all(|i| self.0[i].approx_eq(&o.0[i], &tol.0[i]))
    }
}

/// The type of a value that can be stored in a grid, identified by OpenVDB's type-name strings.
pub trait TypeName {
    /// OpenVDB's name for this value type (e.g. `"float"`, `"vec3s"`).
    fn type_name() -> &'static str;
}

macro_rules! impl_type_name {
    ($($t:ty => $n:expr),* $(,)?) => {$(
        impl TypeName for $t { #[inline] fn type_name() -> &'static str { $n } }
    )*};
}
impl_type_name!(
    bool => "bool",
    f32 => "float",
    f64 => "double",
    half::f16 => "half",
    i8 => "int8",
    i16 => "int16",
    i32 => "int32",
    i64 => "int64",
    u8 => "uint8",
    u16 => "uint16",
    u32 => "uint32",
    u64 => "uint64",
    Vec3f => "vec3s",
    Vec3d => "vec3d",
    Vec3i => "vec3i",
);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn approx() {
        assert!(1.0f32.approx_eq(&1.05, &0.1));
        assert!(!1.0f32.approx_eq(&1.2, &0.1));
        assert!(Vec3f::splat(1.0).approx_eq(&Vec3f::splat(1.0), &Vec3f::zero()));
        assert!(5i32.approx_eq(&6, &1));
    }
}

/// Floating-point scalar values (`f32`, `f64`), as required by level-set and filtering tools.
pub trait RealValue:
    Value + TypeName + num_traits::Float + num_traits::FloatConst + std::iter::Sum + std::ops::AddAssign + std::ops::SubAssign
    + std::ops::MulAssign + std::ops::DivAssign
{
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    /// OpenVDB's `math::Tolerance<T>::value()`.
    fn tolerance() -> Self;
    /// OpenVDB's `math::Delta<T>::value()`.
    fn delta() -> Self;
}
impl RealValue for f32 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn tolerance() -> Self {
        crate::math::TOLERANCE_F32
    }
    fn delta() -> Self {
        crate::math::DELTA_F32
    }
}
impl RealValue for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    fn tolerance() -> Self {
        crate::math::TOLERANCE_F64
    }
    fn delta() -> Self {
        crate::math::DELTA_F64
    }
}

/// Values supporting addition, subtraction and scaling by a real number: scalars and vectors.
/// Used by interpolation, filtering and compositing.
pub trait ArithValue: Value + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> {
    /// Multiply by a scalar.
    fn scale(self, s: f64) -> Self;
    /// Absolute value (component-wise for vectors).
    fn abs_val(self) -> Self;
}
macro_rules! impl_arith_scalar {
    ($($t:ty),*) => {$(
        impl ArithValue for $t {
            #[inline] fn scale(self, s: f64) -> Self { ((self as f64) * s) as $t }
            #[inline] fn abs_val(self) -> Self { self.abs() }
        }
    )*};
}
impl_arith_scalar!(f32, f64, i32, i64);
impl<T: ArithValue> ArithValue for Vec3<T> {
    #[inline]
    fn scale(self, s: f64) -> Self {
        self.map(|v| v.scale(s))
    }
    #[inline]
    fn abs_val(self) -> Self {
        self.map(|v| v.abs_val())
    }
}
