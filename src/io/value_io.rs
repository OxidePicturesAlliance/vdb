//! Binary encoding of voxel value types.

use crate::math::{Vec3, Vec3d, Vec3f, Vec3i};
use crate::types::Value;
use half::f16;

/// A value type that can be stored in a `.vdb` file.
///
/// Values are stored little-endian. Floating-point types ("real" types in OpenVDB's
/// `RealToHalf` sense) can additionally be stored as 16-bit half floats.
pub trait IoValue: Value {
    /// Size of the value in bytes (`sizeof(T)` in C++).
    const SIZE: usize;
    /// True for floating-point scalars and vectors, which support half-float storage.
    const IS_REAL: bool;
    /// Size of the half-float representation (equal to `SIZE` for non-real types).
    const HALF_SIZE: usize;

    /// Decode from `SIZE` little-endian bytes.
    fn from_le(b: &[u8]) -> Self;
    /// Append `SIZE` little-endian bytes.
    fn write_le(&self, out: &mut Vec<u8>);
    /// `math::negative()`: arithmetic negation (logical not for `bool`).
    fn negative(&self) -> Self;
    /// Decode from the half-float representation.
    fn from_half_le(b: &[u8]) -> Self {
        Self::from_le(b)
    }
    /// Append the half-float representation.
    fn write_half_le(&self, out: &mut Vec<u8>) {
        self.write_le(out)
    }
    /// The value rounded to half precision (`truncateRealToHalf`).
    fn truncate_half(&self) -> Self {
        *self
    }
    /// Convert from a bit of a bool or mask leaf buffer.
    fn from_bit(b: bool) -> Self;
    /// Convert to a bit of a bool or mask leaf buffer.
    fn to_bit(&self) -> bool;
}

macro_rules! impl_int {
    ($($t:ty),*) => {$(
        impl IoValue for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            const IS_REAL: bool = false;
            const HALF_SIZE: usize = Self::SIZE;
            #[inline]
            fn from_le(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b[..Self::SIZE].try_into().unwrap())
            }
            #[inline]
            fn write_le(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            #[inline]
            fn negative(&self) -> Self {
                self.wrapping_neg()
            }
            fn from_bit(b: bool) -> Self {
                b as $t
            }
            fn to_bit(&self) -> bool {
                *self != 0
            }
        }
    )*};
}
impl_int!(i32, i64);

impl IoValue for f32 {
    const SIZE: usize = 4;
    const IS_REAL: bool = true;
    const HALF_SIZE: usize = 2;
    #[inline]
    fn from_le(b: &[u8]) -> Self {
        f32::from_le_bytes(b[..4].try_into().unwrap())
    }
    #[inline]
    fn write_le(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    #[inline]
    fn negative(&self) -> Self {
        -*self
    }
    #[inline]
    fn from_half_le(b: &[u8]) -> Self {
        f16::from_le_bytes([b[0], b[1]]).to_f32()
    }
    #[inline]
    fn write_half_le(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&f16::from_f32(*self).to_le_bytes());
    }
    fn truncate_half(&self) -> Self {
        f16::from_f32(*self).to_f32()
    }
    fn from_bit(b: bool) -> Self {
        b as i32 as f32
    }
    fn to_bit(&self) -> bool {
        *self != 0.0
    }
}

impl IoValue for f64 {
    const SIZE: usize = 8;
    const IS_REAL: bool = true;
    const HALF_SIZE: usize = 2;
    #[inline]
    fn from_le(b: &[u8]) -> Self {
        f64::from_le_bytes(b[..8].try_into().unwrap())
    }
    #[inline]
    fn write_le(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    #[inline]
    fn negative(&self) -> Self {
        -*self
    }
    #[inline]
    fn from_half_le(b: &[u8]) -> Self {
        f16::from_le_bytes([b[0], b[1]]).to_f64()
    }
    #[inline]
    fn write_half_le(&self, out: &mut Vec<u8>) {
        // Like OpenVDB, convert to single precision first.
        out.extend_from_slice(&f16::from_f32(*self as f32).to_le_bytes());
    }
    fn truncate_half(&self) -> Self {
        f16::from_f32(*self as f32).to_f64()
    }
    fn from_bit(b: bool) -> Self {
        b as i32 as f64
    }
    fn to_bit(&self) -> bool {
        *self != 0.0
    }
}

impl IoValue for bool {
    const SIZE: usize = 1;
    const IS_REAL: bool = false;
    const HALF_SIZE: usize = 1;
    #[inline]
    fn from_le(b: &[u8]) -> Self {
        b[0] != 0
    }
    #[inline]
    fn write_le(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
    #[inline]
    fn negative(&self) -> Self {
        !*self
    }
    fn from_bit(b: bool) -> Self {
        b
    }
    fn to_bit(&self) -> bool {
        *self
    }
}

impl<T: IoValue> IoValue for Vec3<T> {
    const SIZE: usize = 3 * T::SIZE;
    const IS_REAL: bool = T::IS_REAL;
    const HALF_SIZE: usize = 3 * T::HALF_SIZE;
    fn from_le(b: &[u8]) -> Self {
        let n = T::SIZE;
        Vec3([T::from_le(&b[..n]), T::from_le(&b[n..2 * n]), T::from_le(&b[2 * n..3 * n])])
    }
    fn write_le(&self, out: &mut Vec<u8>) {
        for v in &self.0 {
            v.write_le(out);
        }
    }
    fn negative(&self) -> Self {
        Vec3([self.0[0].negative(), self.0[1].negative(), self.0[2].negative()])
    }
    fn from_half_le(b: &[u8]) -> Self {
        let n = T::HALF_SIZE;
        Vec3([T::from_half_le(&b[..n]), T::from_half_le(&b[n..2 * n]), T::from_half_le(&b[2 * n..3 * n])])
    }
    fn write_half_le(&self, out: &mut Vec<u8>) {
        for v in &self.0 {
            v.write_half_le(out);
        }
    }
    fn truncate_half(&self) -> Self {
        Vec3([self.0[0].truncate_half(), self.0[1].truncate_half(), self.0[2].truncate_half()])
    }
    fn from_bit(b: bool) -> Self {
        Vec3([T::from_bit(b); 3])
    }
    fn to_bit(&self) -> bool {
        self.0.iter().any(|v| v.to_bit())
    }
}

// Compile-time checks that the grid value types are covered.
const _: fn() = || {
    fn check<T: IoValue>() {}
    check::<Vec3f>();
    check::<Vec3d>();
    check::<Vec3i>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode() {
        let mut b = Vec::new();
        Vec3f::new(1.0, -2.0, 0.5).write_le(&mut b);
        assert_eq!(b.len(), 12);
        assert_eq!(Vec3f::from_le(&b), Vec3f::new(1.0, -2.0, 0.5));
        let mut h = Vec::new();
        Vec3d::new(1.0, -2.0, 0.5).write_half_le(&mut h);
        assert_eq!(h.len(), 6);
        assert_eq!(Vec3d::from_half_le(&h), Vec3d::new(1.0, -2.0, 0.5));
        assert_eq!(0.1f32.truncate_half(), f16::from_f32(0.1).to_f32());
        assert_eq!(i32::MIN.negative(), i32::MIN);
        assert!(!true.negative());
        assert_eq!(Vec3i::new(1, 2, 3).negative(), Vec3i::new(-1, -2, -3));
    }
}
