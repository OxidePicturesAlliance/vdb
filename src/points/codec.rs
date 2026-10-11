//! Attribute value types and compression codecs (port of the codec part of
//! `openvdb/points/AttributeArray.h`, `math/QuantizedUnitVec.h`, `AttributeGroup.h` and
//! `AttributeArrayString.h`).
//!
//! An attribute array stores each element in a *storage type* determined by its value type
//! and its [`Codec`]: e.g. a `vec3s` position with the 16-bit fixed-point codec is stored as
//! three `u16` (6 bytes). The codec names (`"null"`, `"trnc"`, `"fxpt16"`, ...) are the ones
//! OpenVDB writes into files.

use crate::math::{Mat3, Mat4, Quat, Vec3, Vec3f, Vec3i};
use crate::metadata::MetaValue;
use half::f16;
use std::fmt::Debug;
use std::sync::OnceLock;

/// Value range of a fixed-point codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FixedPointRange {
    /// Voxel-space positions in `[-0.5, 0.5]` (OpenVDB's `PositionRange`, names `"fxpt8"`,
    /// `"fxpt16"`).
    Position,
    /// Unsigned values in `[0, 1]` (OpenVDB's `UnitRange`, names `"ufxpt8"`, `"ufxpt16"`).
    Unit,
}

/// An attribute compression codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// Values are stored as is (`"null"`).
    Null,
    /// `float` is stored as half, `int32` as `int16` (`"trnc"`).
    Truncate,
    /// Quantization of `[−0.5, 0.5]` or `[0, 1]` values to 8 or 16 bits per component.
    FixedPoint {
        /// `true` for 8-bit, `false` for 16-bit storage.
        one_byte: bool,
        /// The encoded value range.
        range: FixedPointRange,
    },
    /// Unit vectors quantized to 16 bits (`"uvec"`).
    UnitVec,
    /// Bit-packed group membership of `uint8` group arrays (`"grp"`).
    Group,
    /// String indices into descriptor metadata (`"str"`).
    String,
    /// String indices truncated to 16 bits (`"str_trnc"`).
    TruncatedString,
}

impl Codec {
    /// 8-bit position fixed-point codec (`FixedPointCodec<true, PositionRange>`).
    pub const FIXED_POINT_8: Codec = Codec::FixedPoint { one_byte: true, range: FixedPointRange::Position };
    /// 16-bit position fixed-point codec (`FixedPointCodec<false, PositionRange>`).
    pub const FIXED_POINT_16: Codec = Codec::FixedPoint { one_byte: false, range: FixedPointRange::Position };
    /// 8-bit unit-range fixed-point codec (`FixedPointCodec<true, UnitRange>`).
    pub const UNIT_FIXED_POINT_8: Codec = Codec::FixedPoint { one_byte: true, range: FixedPointRange::Unit };
    /// 16-bit unit-range fixed-point codec (`FixedPointCodec<false, UnitRange>`).
    pub const UNIT_FIXED_POINT_16: Codec = Codec::FixedPoint { one_byte: false, range: FixedPointRange::Unit };

    /// The name OpenVDB uses for this codec.
    pub fn name(&self) -> &'static str {
        match self {
            Codec::Null => "null",
            Codec::Truncate => "trnc",
            Codec::FixedPoint { one_byte: true, range: FixedPointRange::Position } => "fxpt8",
            Codec::FixedPoint { one_byte: false, range: FixedPointRange::Position } => "fxpt16",
            Codec::FixedPoint { one_byte: true, range: FixedPointRange::Unit } => "ufxpt8",
            Codec::FixedPoint { one_byte: false, range: FixedPointRange::Unit } => "ufxpt16",
            Codec::UnitVec => "uvec",
            Codec::Group => "grp",
            Codec::String => "str",
            Codec::TruncatedString => "str_trnc",
        }
    }

    /// Parse a codec name.
    pub fn from_name(name: &str) -> Option<Codec> {
        Some(match name {
            "null" => Codec::Null,
            "trnc" => Codec::Truncate,
            "fxpt8" => Codec::FIXED_POINT_8,
            "fxpt16" => Codec::FIXED_POINT_16,
            "ufxpt8" => Codec::UNIT_FIXED_POINT_8,
            "ufxpt16" => Codec::UNIT_FIXED_POINT_16,
            "uvec" => Codec::UnitVec,
            "grp" => Codec::Group,
            "str" => Codec::String,
            "str_trnc" => Codec::TruncatedString,
            _ => return None,
        })
    }
}

impl std::fmt::Display for Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Size in bytes of a value type, by OpenVDB type name (`sizeof(ValueType)`).
pub fn value_type_size(value_type: &str) -> Option<usize> {
    Some(match value_type {
        "bool" | "int8" | "uint8" => 1,
        "int16" | "uint16" | "half" => 2,
        "int32" | "uint32" | "float" => 4,
        "int64" | "uint64" | "double" => 8,
        "vec2i" | "vec2s" => 8,
        "vec2d" => 16,
        "vec3i" | "vec3s" | "vec3u32" => 12,
        "vec3d" => 24,
        "vec4i" | "vec4s" => 16,
        "vec4d" => 32,
        "mat3s" => 36,
        "mat3d" => 72,
        "mat4s" => 64,
        "mat4d" => 128,
        "quats" => 16,
        "quatd" => 32,
        _ => return None,
    })
}

/// Number of scalar components of a value type (3 for `vec3*`, 1 for scalars).
fn component_count(value_type: &str) -> usize {
    if value_type.starts_with("vec3") {
        3
    } else {
        1
    }
}

/// Size in bytes of one stored element of an array with the given value type and codec
/// (`TypedAttributeArray::storageTypeSize()`), or `None` if the combination is not supported.
pub fn storage_type_size(value_type: &str, codec: &str) -> Option<usize> {
    let vsize = value_type_size(value_type)?;
    let n = component_count(value_type);
    let real = matches!(value_type, "float" | "double" | "vec3s" | "vec3d");
    Some(match Codec::from_name(codec)? {
        Codec::Null => vsize,
        Codec::Truncate => match value_type {
            "float" | "int32" => 2,
            "vec3s" | "vec3i" => 6,
            _ => return None,
        },
        Codec::FixedPoint { one_byte, .. } if real => n * if one_byte { 1 } else { 2 },
        Codec::UnitVec if matches!(value_type, "vec3s" | "vec3d") => 2,
        Codec::Group if value_type == "uint8" => 1,
        Codec::String if value_type == "uint32" => 4,
        Codec::TruncatedString if value_type == "uint32" => 2,
        _ => return None,
    })
}

/// A type that can be stored in an attribute array.
///
/// Implemented for the scalar and vector types OpenVDB registers by default. The codec
/// methods encode a value into the codec's storage representation (little-endian bytes) and
/// back; codecs that do not apply to a type return `None` from
/// [`storage_size`](Self::storage_size).
pub trait AttributeValue: Copy + PartialEq + Default + Debug + Send + Sync + 'static {
    /// OpenVDB's type name (`typeNameAsString<T>()`), e.g. `"vec3s"`.
    const TYPE_NAME: &'static str;
    /// Size of the value in bytes.
    const SIZE: usize;
    /// True for floating-point scalars and vectors.
    const IS_FLOAT: bool = false;

    /// Decode from `SIZE` little-endian bytes.
    fn read_le(b: &[u8]) -> Self;
    /// Encode into `SIZE` little-endian bytes.
    fn write_le(&self, out: &mut [u8]);

    /// Storage size of one element with the given codec, or `None` if not supported.
    fn storage_size(codec: Codec) -> Option<usize> {
        match codec {
            Codec::Null => Some(Self::SIZE),
            _ => None,
        }
    }
    /// Decode a stored element (`Codec::decode`).
    fn decode(codec: Codec, b: &[u8]) -> Self {
        let _ = codec;
        Self::read_le(b)
    }
    /// Encode an element for storage (`Codec::encode`).
    fn encode(&self, codec: Codec, out: &mut [u8]) {
        let _ = codec;
        self.write_le(out)
    }
    /// The value as metadata (for default values), if the type has a metadata equivalent.
    fn to_meta(&self) -> Option<MetaValue> {
        None
    }
    /// A value from metadata of the matching type.
    fn from_meta(m: &MetaValue) -> Option<Self> {
        let _ = m;
        None
    }
}

macro_rules! impl_int_value {
    ($($t:ty => $name:expr),* $(,)?) => {$(
        impl AttributeValue for $t {
            const TYPE_NAME: &'static str = $name;
            const SIZE: usize = std::mem::size_of::<$t>();
            #[inline]
            fn read_le(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b[..Self::SIZE].try_into().unwrap())
            }
            #[inline]
            fn write_le(&self, out: &mut [u8]) {
                out[..Self::SIZE].copy_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}
impl_int_value!(i8 => "int8", i16 => "int16", u16 => "uint16", u64 => "uint64");

impl AttributeValue for bool {
    const TYPE_NAME: &'static str = "bool";
    const SIZE: usize = 1;
    fn read_le(b: &[u8]) -> Self {
        b[0] != 0
    }
    fn write_le(&self, out: &mut [u8]) {
        out[0] = *self as u8;
    }
    fn to_meta(&self) -> Option<MetaValue> {
        Some(MetaValue::Bool(*self))
    }
    fn from_meta(m: &MetaValue) -> Option<Self> {
        match m {
            MetaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl AttributeValue for u8 {
    const TYPE_NAME: &'static str = "uint8";
    const SIZE: usize = 1;
    fn read_le(b: &[u8]) -> Self {
        b[0]
    }
    fn write_le(&self, out: &mut [u8]) {
        out[0] = *self;
    }
    fn storage_size(codec: Codec) -> Option<usize> {
        matches!(codec, Codec::Null | Codec::Group).then_some(1)
    }
}

impl AttributeValue for u32 {
    const TYPE_NAME: &'static str = "uint32";
    const SIZE: usize = 4;
    fn read_le(b: &[u8]) -> Self {
        u32::from_le_bytes(b[..4].try_into().unwrap())
    }
    fn write_le(&self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.to_le_bytes());
    }
    fn storage_size(codec: Codec) -> Option<usize> {
        match codec {
            Codec::Null | Codec::String => Some(4),
            Codec::TruncatedString => Some(2),
            _ => None,
        }
    }
    fn decode(codec: Codec, b: &[u8]) -> Self {
        match codec {
            Codec::TruncatedString => u16::from_le_bytes([b[0], b[1]]) as u32,
            _ => Self::read_le(b),
        }
    }
    fn encode(&self, codec: Codec, out: &mut [u8]) {
        match codec {
            Codec::TruncatedString => out[..2].copy_from_slice(&(*self as u16).to_le_bytes()),
            _ => self.write_le(out),
        }
    }
}

impl AttributeValue for i32 {
    const TYPE_NAME: &'static str = "int32";
    const SIZE: usize = 4;
    fn read_le(b: &[u8]) -> Self {
        i32::from_le_bytes(b[..4].try_into().unwrap())
    }
    fn write_le(&self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.to_le_bytes());
    }
    fn storage_size(codec: Codec) -> Option<usize> {
        match codec {
            Codec::Null => Some(4),
            Codec::Truncate => Some(2),
            _ => None,
        }
    }
    fn decode(codec: Codec, b: &[u8]) -> Self {
        match codec {
            Codec::Truncate => i16::from_le_bytes([b[0], b[1]]) as i32,
            _ => Self::read_le(b),
        }
    }
    fn encode(&self, codec: Codec, out: &mut [u8]) {
        match codec {
            Codec::Truncate => out[..2].copy_from_slice(&(*self as i16).to_le_bytes()),
            _ => self.write_le(out),
        }
    }
    fn to_meta(&self) -> Option<MetaValue> {
        Some(MetaValue::Int32(*self))
    }
    fn from_meta(m: &MetaValue) -> Option<Self> {
        match m {
            MetaValue::Int32(v) => Some(*v),
            _ => None,
        }
    }
}

impl AttributeValue for f16 {
    const TYPE_NAME: &'static str = "half";
    const SIZE: usize = 2;
    const IS_FLOAT: bool = true;
    fn read_le(b: &[u8]) -> Self {
        f16::from_le_bytes([b[0], b[1]])
    }
    fn write_le(&self, out: &mut [u8]) {
        out[..2].copy_from_slice(&self.to_le_bytes());
    }
}

/// `floatingPointToFixedPoint`: quantize `s` in `[0, 1]` to an unsigned integer with
/// `max` as the largest value.
#[inline]
fn to_fixed<F: num_traits::Float>(s: F, max: u32) -> u32 {
    let maxf = F::from(max).unwrap();
    if F::zero() > s {
        0
    } else if F::one() <= s {
        max
    } else {
        (s * maxf).to_u32().unwrap_or(0)
    }
}

/// `fixedPointToFloatingPoint`.
#[inline]
fn from_fixed<F: num_traits::Float>(v: u32, max: u32) -> F {
    F::from(v).unwrap() / F::from(max).unwrap()
}

#[inline]
fn fixed_encode<F: num_traits::Float>(v: F, one_byte: bool, range: FixedPointRange, out: &mut [u8]) {
    let shifted = match range {
        FixedPointRange::Position => v + F::from(0.5).unwrap(),
        FixedPointRange::Unit => v,
    };
    if one_byte {
        out[0] = to_fixed(shifted, u8::MAX as u32) as u8;
    } else {
        out[..2].copy_from_slice(&(to_fixed(shifted, u16::MAX as u32) as u16).to_le_bytes());
    }
}

#[inline]
fn fixed_decode<F: num_traits::Float>(b: &[u8], one_byte: bool, range: FixedPointRange) -> F {
    let v: F = if one_byte {
        from_fixed(b[0] as u32, u8::MAX as u32)
    } else {
        from_fixed(u16::from_le_bytes([b[0], b[1]]) as u32, u16::MAX as u32)
    };
    match range {
        FixedPointRange::Position => v - F::from(0.5).unwrap(),
        FixedPointRange::Unit => v,
    }
}

impl AttributeValue for f32 {
    const TYPE_NAME: &'static str = "float";
    const SIZE: usize = 4;
    const IS_FLOAT: bool = true;
    fn read_le(b: &[u8]) -> Self {
        f32::from_le_bytes(b[..4].try_into().unwrap())
    }
    fn write_le(&self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.to_le_bytes());
    }
    fn storage_size(codec: Codec) -> Option<usize> {
        match codec {
            Codec::Null => Some(4),
            Codec::Truncate => Some(2),
            Codec::FixedPoint { one_byte, .. } => Some(if one_byte { 1 } else { 2 }),
            _ => None,
        }
    }
    fn decode(codec: Codec, b: &[u8]) -> Self {
        match codec {
            Codec::Truncate => f16::from_le_bytes([b[0], b[1]]).to_f32(),
            Codec::FixedPoint { one_byte, range } => fixed_decode(b, one_byte, range),
            _ => Self::read_le(b),
        }
    }
    fn encode(&self, codec: Codec, out: &mut [u8]) {
        match codec {
            Codec::Truncate => out[..2].copy_from_slice(&f16::from_f32(*self).to_le_bytes()),
            Codec::FixedPoint { one_byte, range } => fixed_encode(*self, one_byte, range, out),
            _ => self.write_le(out),
        }
    }
    fn to_meta(&self) -> Option<MetaValue> {
        Some(MetaValue::Float(*self))
    }
    fn from_meta(m: &MetaValue) -> Option<Self> {
        match m {
            MetaValue::Float(v) => Some(*v),
            _ => None,
        }
    }
}

impl AttributeValue for f64 {
    const TYPE_NAME: &'static str = "double";
    const SIZE: usize = 8;
    const IS_FLOAT: bool = true;
    fn read_le(b: &[u8]) -> Self {
        f64::from_le_bytes(b[..8].try_into().unwrap())
    }
    fn write_le(&self, out: &mut [u8]) {
        out[..8].copy_from_slice(&self.to_le_bytes());
    }
    fn storage_size(codec: Codec) -> Option<usize> {
        match codec {
            Codec::Null => Some(8),
            Codec::FixedPoint { one_byte, .. } => Some(if one_byte { 1 } else { 2 }),
            _ => None,
        }
    }
    fn decode(codec: Codec, b: &[u8]) -> Self {
        match codec {
            Codec::FixedPoint { one_byte, range } => fixed_decode(b, one_byte, range),
            _ => Self::read_le(b),
        }
    }
    fn encode(&self, codec: Codec, out: &mut [u8]) {
        match codec {
            Codec::FixedPoint { one_byte, range } => fixed_encode(*self, one_byte, range, out),
            _ => self.write_le(out),
        }
    }
    fn to_meta(&self) -> Option<MetaValue> {
        Some(MetaValue::Double(*self))
    }
    fn from_meta(m: &MetaValue) -> Option<Self> {
        match m {
            MetaValue::Double(v) => Some(*v),
            _ => None,
        }
    }
}

impl AttributeValue for i64 {
    const TYPE_NAME: &'static str = "int64";
    const SIZE: usize = 8;
    fn read_le(b: &[u8]) -> Self {
        i64::from_le_bytes(b[..8].try_into().unwrap())
    }
    fn write_le(&self, out: &mut [u8]) {
        out[..8].copy_from_slice(&self.to_le_bytes());
    }
    fn to_meta(&self) -> Option<MetaValue> {
        Some(MetaValue::Int64(*self))
    }
    fn from_meta(m: &MetaValue) -> Option<Self> {
        match m {
            MetaValue::Int64(v) => Some(*v),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit vector quantization (math::QuantizedUnitVec)

const MASK_SLOTS: u16 = 0x1FFF;
const MASK_XSLOT: u16 = 0x1F80;
const MASK_YSLOT: u16 = 0x007F;
const MASK_XSIGN: u16 = 0x8000;
const MASK_YSIGN: u16 = 0x4000;
const MASK_ZSIGN: u16 = 0x2000;

/// The normalization weights of `QuantizedUnitVec` (computed exactly as OpenVDB's table).
fn normalization_weights() -> &'static [f32; 8192] {
    static TABLE: OnceLock<Box<[f32; 8192]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0f32; 8192]);
        for (d, w) in t.iter_mut().enumerate() {
            let mut x = ((d as u16 & MASK_XSLOT) >> 7) as i32;
            let mut y = (d as u16 & MASK_YSLOT) as i32;
            if x + y > 126 {
                x = 127 - x;
                y = 127 - y;
            }
            let z = 126 - x - y;
            *w = (1.0 / ((x * x + y * y + z * z) as f64).sqrt()) as f32;
        }
        t
    })
}

/// Quantize a vector to 16 bits (`QuantizedUnitVec::pack`); the vector is normalized
/// implicitly (only its direction is stored).
pub fn pack_unit_vec<T: num_traits::Float>(v: Vec3<T>) -> u16 {
    let zero = T::zero();
    if v.0.iter().all(|c| *c == zero) {
        return 0;
    }
    let mut data = 0u16;
    let (mut x, mut y, mut z) = (v.0[0], v.0[1], v.0[2]);
    if x < zero {
        data |= MASK_XSIGN;
        x = -x;
    }
    if y < zero {
        data |= MASK_YSIGN;
        y = -y;
    }
    if z < zero {
        data |= MASK_ZSIGN;
        z = -z;
    }
    let w = T::from(126.0).unwrap() / (x + y + z);
    let mut xbits = (x * w).to_u16().unwrap_or(0);
    let mut ybits = (y * w).to_u16().unwrap_or(0);
    if xbits > 63 {
        xbits = 127 - xbits;
        ybits = 127u16.wrapping_sub(ybits);
    }
    data | (xbits << 7) | ybits
}

/// Decode a quantized unit vector (`QuantizedUnitVec::unpack`).
pub fn unpack_unit_vec(data: u16) -> Vec3f {
    let w = normalization_weights()[(data & MASK_SLOTS) as usize];
    let mut xbits = (data & MASK_XSLOT) >> 7;
    let mut ybits = data & MASK_YSLOT;
    if xbits + ybits > 126 {
        xbits = 127 - xbits;
        ybits = 127 - ybits;
    }
    let mut v = Vec3f::new(xbits as f32 * w, ybits as f32 * w, (126 - xbits as i32 - ybits as i32) as f32 * w);
    if data & MASK_XSIGN != 0 {
        v.0[0] = -v.0[0];
    }
    if data & MASK_YSIGN != 0 {
        v.0[1] = -v.0[1];
    }
    if data & MASK_ZSIGN != 0 {
        v.0[2] = -v.0[2];
    }
    v
}

macro_rules! impl_vec3_real {
    ($s:ty, $name:expr, $meta:ident) => {
        impl AttributeValue for Vec3<$s> {
            const TYPE_NAME: &'static str = $name;
            const SIZE: usize = 3 * std::mem::size_of::<$s>();
            const IS_FLOAT: bool = true;
            fn read_le(b: &[u8]) -> Self {
                let n = std::mem::size_of::<$s>();
                Vec3(std::array::from_fn(|i| <$s>::read_le(&b[i * n..])))
            }
            fn write_le(&self, out: &mut [u8]) {
                let n = std::mem::size_of::<$s>();
                for i in 0..3 {
                    self.0[i].write_le(&mut out[i * n..]);
                }
            }
            fn storage_size(codec: Codec) -> Option<usize> {
                match codec {
                    Codec::Null => Some(Self::SIZE),
                    Codec::Truncate if Self::SIZE == 12 => Some(6),
                    Codec::FixedPoint { one_byte, .. } => Some(if one_byte { 3 } else { 6 }),
                    Codec::UnitVec => Some(2),
                    _ => None,
                }
            }
            fn decode(codec: Codec, b: &[u8]) -> Self {
                match codec {
                    Codec::Truncate => {
                        Vec3(std::array::from_fn(|i| <$s>::from(f16::from_le_bytes([b[2 * i], b[2 * i + 1]]).to_f32())))
                    }
                    Codec::FixedPoint { one_byte, range } => {
                        let n = if one_byte { 1 } else { 2 };
                        Vec3(std::array::from_fn(|i| fixed_decode::<$s>(&b[i * n..], one_byte, range)))
                    }
                    Codec::UnitVec => unpack_unit_vec(u16::from_le_bytes([b[0], b[1]])).map(|c| c as $s),
                    _ => Self::read_le(b),
                }
            }
            fn encode(&self, codec: Codec, out: &mut [u8]) {
                match codec {
                    Codec::Truncate => {
                        for i in 0..3 {
                            out[2 * i..2 * i + 2].copy_from_slice(&f16::from_f64(self.0[i] as f64).to_le_bytes());
                        }
                    }
                    Codec::FixedPoint { one_byte, range } => {
                        let n = if one_byte { 1 } else { 2 };
                        for i in 0..3 {
                            fixed_encode(self.0[i], one_byte, range, &mut out[i * n..]);
                        }
                    }
                    Codec::UnitVec => out[..2].copy_from_slice(&pack_unit_vec(*self).to_le_bytes()),
                    _ => self.write_le(out),
                }
            }
            fn to_meta(&self) -> Option<MetaValue> {
                Some(MetaValue::$meta(*self))
            }
            fn from_meta(m: &MetaValue) -> Option<Self> {
                match m {
                    MetaValue::$meta(v) => Some(*v),
                    _ => None,
                }
            }
        }
    };
}
impl_vec3_real!(f32, "vec3s", Vec3s);
impl_vec3_real!(f64, "vec3d", Vec3d);

impl AttributeValue for Vec3i {
    const TYPE_NAME: &'static str = "vec3i";
    const SIZE: usize = 12;
    fn read_le(b: &[u8]) -> Self {
        Vec3(std::array::from_fn(|i| i32::read_le(&b[4 * i..])))
    }
    fn write_le(&self, out: &mut [u8]) {
        for i in 0..3 {
            self.0[i].write_le(&mut out[4 * i..]);
        }
    }
    fn to_meta(&self) -> Option<MetaValue> {
        Some(MetaValue::Vec3i(*self))
    }
    fn from_meta(m: &MetaValue) -> Option<Self> {
        match m {
            MetaValue::Vec3i(v) => Some(*v),
            _ => None,
        }
    }
}

macro_rules! impl_matrix_value {
    ($t:ty, $s:ty, $n:expr, $name:expr) => {
        impl AttributeValue for $t {
            const TYPE_NAME: &'static str = $name;
            const SIZE: usize = $n * $n * std::mem::size_of::<$s>();
            const IS_FLOAT: bool = true;
            fn read_le(b: &[u8]) -> Self {
                let s = std::mem::size_of::<$s>();
                let mut m = <$t>::default();
                for i in 0..$n {
                    for j in 0..$n {
                        m.0[i][j] = <$s>::read_le(&b[(i * $n + j) * s..]);
                    }
                }
                m
            }
            fn write_le(&self, out: &mut [u8]) {
                let s = std::mem::size_of::<$s>();
                for i in 0..$n {
                    for j in 0..$n {
                        self.0[i][j].write_le(&mut out[(i * $n + j) * s..]);
                    }
                }
            }
        }
    };
}
impl_matrix_value!(Mat3<f32>, f32, 3, "mat3s");
impl_matrix_value!(Mat3<f64>, f64, 3, "mat3d");
impl_matrix_value!(Mat4<f32>, f32, 4, "mat4s");
impl_matrix_value!(Mat4<f64>, f64, 4, "mat4d");

macro_rules! impl_quat_value {
    ($s:ty, $name:expr) => {
        impl AttributeValue for Quat<$s> {
            const TYPE_NAME: &'static str = $name;
            const SIZE: usize = 4 * std::mem::size_of::<$s>();
            const IS_FLOAT: bool = true;
            fn read_le(b: &[u8]) -> Self {
                let s = std::mem::size_of::<$s>();
                Quat(std::array::from_fn(|i| <$s>::read_le(&b[i * s..])))
            }
            fn write_le(&self, out: &mut [u8]) {
                let s = std::mem::size_of::<$s>();
                for i in 0..4 {
                    self.0[i].write_le(&mut out[i * s..]);
                }
            }
        }
    };
}
impl_quat_value!(f32, "quats");
impl_quat_value!(f64, "quatd");

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math::Vec3d;

    fn roundtrip<T: AttributeValue>(v: T, codec: Codec) -> T {
        let mut b = vec![0u8; T::storage_size(codec).unwrap()];
        v.encode(codec, &mut b);
        T::decode(codec, &b)
    }

    #[test]
    fn names() {
        for name in ["null", "trnc", "fxpt8", "fxpt16", "ufxpt8", "ufxpt16", "uvec", "grp", "str", "str_trnc"] {
            assert_eq!(Codec::from_name(name).unwrap().name(), name);
        }
        assert_eq!(storage_type_size("vec3s", "fxpt16"), Some(6));
        assert_eq!(storage_type_size("vec3s", "fxpt8"), Some(3));
        assert_eq!(storage_type_size("float", "trnc"), Some(2));
        assert_eq!(storage_type_size("vec3s", "uvec"), Some(2));
        assert_eq!(storage_type_size("uint8", "grp"), Some(1));
        assert_eq!(storage_type_size("mat4d", "null"), Some(128));
        assert_eq!(storage_type_size("int32", "uvec"), None);
        assert_eq!(<Vec3f as AttributeValue>::storage_size(Codec::FIXED_POINT_16), Some(6));
    }

    #[test]
    fn fixed_point() {
        let v = Vec3f::new(-0.5, 0.0, 0.4999);
        let r = roundtrip(v, Codec::FIXED_POINT_16);
        assert!((r - v).abs().max_component() < 1.0 / 65535.0 + 1e-6);
        let r = roundtrip(v, Codec::FIXED_POINT_8);
        assert!((r - v).abs().max_component() < 1.0 / 255.0 + 1e-6);
        // Out-of-range values are clamped.
        assert_eq!(roundtrip(Vec3f::new(-3.0, 3.0, 0.5), Codec::FIXED_POINT_8), Vec3f::new(-0.5, 0.5, 0.5));
        assert_eq!(roundtrip(1.0f32, Codec::UNIT_FIXED_POINT_16), 1.0);
        assert_eq!(roundtrip(0.0f32, Codec::UNIT_FIXED_POINT_8), 0.0);
        // OpenVDB truncates toward zero when quantizing.
        let mut b = [0u8; 1];
        0.5f32.encode(Codec::UNIT_FIXED_POINT_8, &mut b);
        assert_eq!(b[0], 127);
    }

    #[test]
    fn truncate_and_null() {
        assert_eq!(roundtrip(1.5f32, Codec::Truncate), 1.5);
        assert_eq!(roundtrip(Vec3f::new(1.0, -2.5, 0.25), Codec::Truncate), Vec3f::new(1.0, -2.5, 0.25));
        assert_eq!(roundtrip(-1234i32, Codec::Truncate), -1234);
        assert_eq!(roundtrip(Vec3d::new(1.0, 2.0, 3.0), Codec::Null), Vec3d::new(1.0, 2.0, 3.0));
        assert_eq!(roundtrip(70000u32, Codec::TruncatedString), 70000 & 0xFFFF);
    }

    #[test]
    fn unit_vec() {
        assert_eq!(pack_unit_vec(Vec3f::zero()), 0);
        assert_eq!(normalization_weights()[0], 0.007_936_508);
        assert_eq!(normalization_weights()[127], 0.007_873_772);
        for v in [
            Vec3f::new(1.0, 0.0, 0.0),
            Vec3f::new(0.0, -1.0, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
            Vec3f::new(1.0, 2.0, -3.0).unit(),
            Vec3f::new(-0.3, -0.9, 0.1).unit(),
        ] {
            let r = roundtrip(v, Codec::UnitVec);
            assert!((r - v).length() < 0.02, "{:?} -> {:?}", v, r);
        }
    }
}
