//! General-purpose scalar math utilities (port of the free functions in `openvdb/math/Math.h`).
//!
//! Function names follow the C++ ones in snake case. Where C++ overloads a function with and
//! without an explicit tolerance (e.g. `isApproxEqual(a, b)` / `isApproxEqual(a, b, tol)`), the
//! Rust port provides two functions: the default-tolerance one keeps the plain name and the
//! explicit one gets a `_tol` suffix.

use super::vec3::Vec3;
use num_traits::{Float, NumCast};

/// Floating-point types with OpenVDB's default comparison tolerances
/// (`math::Tolerance<T>` and `math::Delta<T>`).
pub trait FloatTolerance: Float {
    /// `math::Tolerance<T>::value()`: 1e-8 for `f32`, 1e-15 for `f64`.
    fn tolerance() -> Self;
    /// `math::Delta<T>::value()`: 1e-5 for `f32`, 1e-9 for `f64`.
    fn delta() -> Self;
}

impl FloatTolerance for f32 {
    #[inline]
    fn tolerance() -> Self {
        super::TOLERANCE_F32
    }
    #[inline]
    fn delta() -> Self {
        super::DELTA_F32
    }
}

impl FloatTolerance for f64 {
    #[inline]
    fn tolerance() -> Self {
        super::TOLERANCE_F64
    }
    #[inline]
    fn delta() -> Self {
        super::DELTA_F64
    }
}

/// `math::pi<T>()`.
#[inline]
pub fn pi<T: Float>() -> T {
    T::from(std::f64::consts::PI).unwrap()
}

// ==========> Clamp <==================

/// Return `x` clamped to `[min, max]` (`math::Clamp`).
#[inline]
pub fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    debug_assert!(!(min > max));
    if x > min {
        if x < max {
            x
        } else {
            max
        }
    } else {
        min
    }
}

/// Return `x` clamped to `[0, 1]` (`math::Clamp01`).
#[inline]
pub fn clamp01<T: Float>(x: T) -> T {
    if x > T::zero() {
        if x < T::one() {
            x
        } else {
            T::one()
        }
    } else {
        T::zero()
    }
}

/// If `x` lies outside `[0, 1]`, clamp it and return `true`; otherwise return `false`
/// (`math::ClampTest01`).
#[inline]
pub fn clamp_test01<T: Float>(x: &mut T) -> bool {
    if *x >= T::zero() && *x <= T::one() {
        return false;
    }
    *x = if *x < T::zero() { T::zero() } else { T::one() };
    true
}

/// Return 0 if `x < 0`, 1 if `x > 1`, else `(3 - 2x) x²` (`math::SmoothUnitStep`).
#[inline]
pub fn smooth_unit_step<T: Float>(x: T) -> T {
    let two = T::one() + T::one();
    let three = two + T::one();
    if x > T::zero() {
        if x < T::one() {
            (three - two * x) * x * x
        } else {
            T::one()
        }
    } else {
        T::zero()
    }
}

/// Smooth step of `(x - min) / (max - min)` (`math::SmoothUnitStep(x, min, max)`).
#[inline]
pub fn smooth_unit_step_range<T: Float>(x: T, min: T, max: T) -> T {
    debug_assert!(min < max);
    smooth_unit_step((x - min) / (max - min))
}

// ==========> Value comparison <==================

/// True if `x` is exactly zero (`math::isZero`).
#[inline]
pub fn is_zero<T: num_traits::Zero + PartialEq>(x: T) -> bool {
    x == T::zero()
}

/// True if `|x| <= tol` (`math::isApproxZero(x, tol)`).
#[inline]
pub fn is_approx_zero_tol<T: Float>(x: T, tol: T) -> bool {
    !(x > tol) && !(x < -tol)
}

/// True if `x` is zero to within the default tolerance (`math::isApproxZero(x)`).
#[inline]
pub fn is_approx_zero<T: FloatTolerance>(x: T) -> bool {
    is_approx_zero_tol(x, T::tolerance())
}

/// True if `x < 0` (`math::isNegative`).
#[inline]
pub fn is_negative<T: num_traits::Zero + PartialOrd>(x: T) -> bool {
    x < T::zero()
}

/// True if `a` equals `b` to within `tol`, i.e. `!(|a - b| > tol)` (`math::isApproxEqual(a, b, tol)`).
#[inline]
pub fn is_approx_equal_tol<T: Float>(a: T, b: T, tol: T) -> bool {
    !((a - b).abs() > tol)
}

/// True if `a` equals `b` to within the default tolerance `math::Tolerance<T>`
/// (`math::isApproxEqual(a, b)`): 1e-15 for `f64`, 1e-8 for `f32`.
#[inline]
pub fn is_approx_equal<T: FloatTolerance>(a: T, b: T) -> bool {
    is_approx_equal_tol(a, b, T::tolerance())
}

/// True if `a` is larger than `b` to within `tol`, i.e. `b - a < tol` (`math::isApproxLarger`).
#[inline]
pub fn is_approx_larger<T: Float>(a: T, b: T, tol: T) -> bool {
    b - a < tol
}

/// True if `a == b` exactly (`math::isExactlyEqual`).
#[inline]
pub fn is_exactly_equal<T: PartialEq>(a: T, b: T) -> bool {
    a == b
}

/// True if `a` and `b` are equal to within an absolute tolerance, or failing that, a relative
/// tolerance (`math::isRelOrApproxEqual`).
#[inline]
pub fn is_rel_or_approx_equal<T: Float>(a: T, b: T, abs_tol: T, rel_tol: T) -> bool {
    if !((a - b).abs() > abs_tol) {
        return true;
    }
    let rel_error = if b.abs() > a.abs() { ((a - b) / b).abs() } else { ((a - b) / a).abs() };
    rel_error <= rel_tol
}

/// Compare two doubles by units in the last place (`math::isUlpsEqual(double, double, int64)`).
pub fn is_ulps_equal_f64(a: f64, b: f64, ulps: i64) -> bool {
    let fix = |v: f64| {
        let i = v.to_bits() as i64;
        if i < 0 {
            i64::MIN.wrapping_sub(i)
        } else {
            i
        }
    };
    fix(a).wrapping_sub(fix(b)).wrapping_abs() <= ulps
}

/// Compare two floats by units in the last place (`math::isUlpsEqual(float, float, int32)`).
pub fn is_ulps_equal_f32(a: f32, b: f32, ulps: i32) -> bool {
    let fix = |v: f32| {
        let i = v.to_bits() as i32;
        if i < 0 {
            i32::MIN.wrapping_sub(i)
        } else {
            i
        }
    };
    fix(a).wrapping_sub(fix(b)).wrapping_abs() <= ulps
}

// ==========> Pow <==================

/// `x²` (`math::Pow2`).
#[inline]
pub fn pow2<T: Copy + std::ops::Mul<Output = T>>(x: T) -> T {
    x * x
}

/// `x³` (`math::Pow3`).
#[inline]
pub fn pow3<T: Copy + std::ops::Mul<Output = T>>(x: T) -> T {
    x * x * x
}

/// `x⁴` (`math::Pow4`).
#[inline]
pub fn pow4<T: Copy + std::ops::Mul<Output = T>>(x: T) -> T {
    pow2(pow2(x))
}

/// `xⁿ` for an integer exponent, by repeated multiplication (`math::Pow(x, n)`).
pub fn pow_int<T: Float>(mut x: T, mut n: i32) -> T {
    let mut ans = T::one();
    if n < 0 {
        n = -n;
        x = T::one() / x;
    }
    while n > 0 {
        ans = ans * x;
        n -= 1;
    }
    ans
}

// ==========> Sign <==================

/// The sign of `x` as -1, 0 or 1 (`math::Sign`).
#[inline]
pub fn sign<T: num_traits::Zero + PartialOrd>(x: T) -> i32 {
    (T::zero() < x) as i32 - (x < T::zero()) as i32
}

/// True if `a` and `b` have different signs; zero counts as positive (`math::SignChange`).
#[inline]
pub fn sign_change<T: num_traits::Zero + PartialOrd>(a: T, b: T) -> bool {
    (a < T::zero()) ^ (b < T::zero())
}

/// True if `[a, b]` contains zero (`math::ZeroCrossing`).
#[inline]
pub fn zero_crossing<T: num_traits::Zero + PartialOrd + std::ops::Mul<Output = T>>(a: T, b: T) -> bool {
    a * b <= T::zero()
}

// ==========> Rounding <==================

/// `x` rounded up to the nearest integer (`math::RoundUp`).
#[inline]
pub fn round_up<T: Float>(x: T) -> T {
    x.ceil()
}

/// `x` rounded up to the nearest multiple of `base` (`math::RoundUp(x, base)`).
#[inline]
pub fn round_up_to<T: Float>(x: T, base: T) -> T {
    let r = x % base;
    if r != T::zero() {
        x - r + base
    } else {
        x
    }
}

/// `x` rounded down to the nearest integer (`math::RoundDown`).
#[inline]
pub fn round_down<T: Float>(x: T) -> T {
    x.floor()
}

/// `x` rounded down to the nearest multiple of `base` (`math::RoundDown(x, base)`).
#[inline]
pub fn round_down_to<T: Float>(x: T, base: T) -> T {
    let r = x % base;
    if r != T::zero() {
        x - r
    } else {
        x
    }
}

/// `x` rounded to the nearest integer, halves rounded up: `floor(x + 0.5)` (`math::Round`).
#[inline]
pub fn round<T: Float>(x: T) -> T {
    (x + T::from(0.5).unwrap()).floor()
}

/// `x - floor(x)`, always non-negative (`math::EuclideanRemainder`).
#[inline]
pub fn euclidean_remainder<T: Float>(x: T) -> T {
    x - x.floor()
}

/// The integer part of `x`, rounded toward zero (`math::IntegerPart`).
#[inline]
pub fn integer_part<T: Float>(x: T) -> T {
    if x > T::zero() {
        x.floor()
    } else {
        x.ceil()
    }
}

/// The fractional part of `x`: `fmod(x, 1)` (`math::FractionalPart`).
#[inline]
pub fn fractional_part<T: Float>(x: T) -> T {
    x % T::one()
}

/// `floor(x)` as an `i32` (`math::Floor`).
#[inline]
pub fn floor_i32<T: Float>(x: T) -> i32 {
    <i32 as NumCast>::from(x.floor()).unwrap_or(if x < T::zero() { i32::MIN } else { i32::MAX })
}

/// `ceil(x)` as an `i32` (`math::Ceil`).
#[inline]
pub fn ceil_i32<T: Float>(x: T) -> i32 {
    <i32 as NumCast>::from(x.ceil()).unwrap_or(if x < T::zero() { i32::MIN } else { i32::MAX })
}

/// `x` if `|x| >= delta`, otherwise zero (`math::Chop`).
#[inline]
pub fn chop<T: Float>(x: T, delta: T) -> T {
    if x.abs() < delta {
        T::zero()
    } else {
        x
    }
}

/// `x` truncated (rounded) to the given number of decimal digits (`math::Truncate`).
#[inline]
pub fn truncate<T: Float>(x: T, digits: u32) -> T {
    let tenth = T::from(10u64.pow(digits)).unwrap();
    (x * tenth + T::from(0.5).unwrap()).floor() / tenth
}

/// `1 / x` (`math::Inv`).
#[inline]
pub fn inv<T: Float>(x: T) -> T {
    debug_assert!(x != T::zero());
    T::one() / x
}

/// Index of the smallest component of a 3-vector; ties resolve to the *largest* index
/// (`math::MinIndex`).
#[inline]
pub fn min_index<T: Copy + PartialOrd>(v: &Vec3<T>) -> usize {
    let mut r = 0;
    for i in 1..3 {
        if v.0[i] <= v.0[r] {
            r = i;
        }
    }
    r
}

/// Index of the largest component of a 3-vector; ties resolve to the *largest* index
/// (`math::MaxIndex`).
#[inline]
pub fn max_index<T: Copy + PartialOrd>(v: &Vec3<T>) -> usize {
    let mut r = 0;
    for i in 1..3 {
        if v.0[i] >= v.0[r] {
            r = i;
        }
    }
    r
}

/// Minimum of two partially ordered values (`math::Min`), returning `a` if they are unordered.
#[inline]
pub fn min<T: PartialOrd>(a: T, b: T) -> T {
    if b < a {
        b
    } else {
        a
    }
}

/// Maximum of two partially ordered values (`math::Max`), returning `a` if they are unordered.
#[inline]
pub fn max<T: PartialOrd>(a: T, b: T) -> T {
    if b > a {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3d;

    #[test]
    fn comparisons() {
        assert!(is_approx_equal(1.0f64, 1.0 + 1e-16));
        assert!(!is_approx_equal(1.0f64, 1.0 + 1e-14));
        assert!(is_approx_equal_tol(1.0f32, 1.05, 0.1));
        assert!(is_rel_or_approx_equal(1e10f64, 1e10 + 1.0, 1e-6, 1e-9));
        assert!(!is_rel_or_approx_equal(1.0f64, 1.1, 1e-6, 1e-3));
        assert!(is_approx_zero(1e-16f64));
        assert!(is_ulps_equal_f64(1.0, 1.0 + f64::EPSILON, 1));
        assert!(!is_ulps_equal_f64(1.0, 1.0 + 4.0 * f64::EPSILON, 2));
        assert!(is_ulps_equal_f32(-0.0, 0.0, 0));
        assert!(is_negative(-1.0f32));
    }

    #[test]
    fn rounding() {
        assert_eq!(floor_i32(-1.5f64), -2);
        assert_eq!(ceil_i32(-1.5f64), -1);
        assert_eq!(round(-1.5f64), -1.0);
        assert_eq!(round(2.5f64), 3.0);
        assert_eq!(round_up_to(7.0f64, 4.0), 8.0);
        assert_eq!(round_down_to(7.0f64, 4.0), 4.0);
        assert_eq!(integer_part(-2.7f64), -2.0);
        assert!((fractional_part(2.25f64) - 0.25).abs() < 1e-15);
        assert!((euclidean_remainder(-0.25f64) - 0.75).abs() < 1e-15);
        assert_eq!(truncate(1.23456f64, 2), 1.23);
        assert_eq!(chop(1e-6f64, 1e-5), 0.0);
    }

    #[test]
    fn misc() {
        assert_eq!(clamp(5, 0, 3), 3);
        assert_eq!(clamp01(-0.5f64), 0.0);
        let mut x = 1.5f64;
        assert!(clamp_test01(&mut x));
        assert_eq!(x, 1.0);
        assert_eq!(smooth_unit_step(0.5f64), 0.5);
        assert_eq!(smooth_unit_step_range(15.0f64, 10.0, 20.0), 0.5);
        assert_eq!(sign(-3.0f64), -1);
        assert_eq!(sign(0i32), 0);
        assert!(sign_change(-1.0f64, 0.0));
        assert!(zero_crossing(-1.0f64, 0.0));
        assert_eq!(pow2(3), 9);
        assert_eq!(pow3(2.0f64), 8.0);
        assert_eq!(pow_int(2.0f64, -2), 0.25);
        assert_eq!(min_index(&Vec3d::new(1.0, 1.0, 2.0)), 1);
        assert_eq!(max_index(&Vec3d::new(2.0, 1.0, 2.0)), 2);
    }
}
