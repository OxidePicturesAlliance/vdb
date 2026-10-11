//! Build types: the value types a NanoVDB grid can be built with, and how each one lays
//! out its nodes (port of the `LeafData<T>` specializations and `BuildTraits`).

use super::data::{leaf_off, rd_f32, rd_u16, rd_u64, rd_u8, wr_f32, wr_u16, LeafLayout, MaskView, NodeLayout};
use super::{GridClass, GridType};
use crate::math::{Vec3d, Vec3f};
use std::fmt::Debug;

/// A plain value that can be stored in a NanoVDB buffer (little endian, natural alignment).
pub trait NanoPod: Copy + Default + PartialEq + Debug + Send + Sync + 'static {
    /// `sizeof(T)` in C++.
    const SIZE: usize;
    /// `alignof(T)` in C++.
    const ALIGN: usize;
    /// Read a value at byte offset `o`.
    fn read(b: &[u8], o: usize) -> Self;
    /// Write a value at byte offset `o`.
    fn write(self, b: &mut [u8], o: usize);
    /// Conversion from a double (used to store statistics; vectors splat the value).
    fn from_f64(v: f64) -> Self;
}

macro_rules! pod_num {
    ($($t:ty => $n:expr),*) => {$(
        impl NanoPod for $t {
            const SIZE: usize = $n;
            const ALIGN: usize = $n;
            #[inline]
            fn read(b: &[u8], o: usize) -> Self { <$t>::from_le_bytes(b[o..o + $n].try_into().unwrap()) }
            #[inline]
            fn write(self, b: &mut [u8], o: usize) { b[o..o + $n].copy_from_slice(&self.to_le_bytes()) }
            #[inline]
            fn from_f64(v: f64) -> Self { v as $t }
        }
    )*};
}
pod_num!(f32 => 4, f64 => 8, i32 => 4, i64 => 8, u32 => 4, u64 => 8);

impl NanoPod for bool {
    const SIZE: usize = 1;
    const ALIGN: usize = 1;
    #[inline]
    fn read(b: &[u8], o: usize) -> Self {
        b[o] != 0
    }
    #[inline]
    fn write(self, b: &mut [u8], o: usize) {
        b[o] = self as u8;
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
}

macro_rules! pod_vec {
    ($($t:ty, $s:ty => $n:expr),*) => {$(
        impl NanoPod for $t {
            const SIZE: usize = 3 * $n;
            const ALIGN: usize = $n;
            #[inline]
            fn read(b: &[u8], o: usize) -> Self {
                <$t>::new(<$s>::read(b, o), <$s>::read(b, o + $n), <$s>::read(b, o + 2 * $n))
            }
            #[inline]
            fn write(self, b: &mut [u8], o: usize) {
                self.0[0].write(b, o);
                self.0[1].write(b, o + $n);
                self.0[2].write(b, o + 2 * $n);
            }
            #[inline]
            fn from_f64(v: f64) -> Self { <$t>::splat(v as $s) }
        }
    )*};
}
pod_vec!(Vec3f, f32 => 4, Vec3d, f64 => 8);

/// Values for which NanoVDB computes statistics (`Extrema`/`Stats` in `GridStats.h`).
///
/// Scalars are compared and averaged directly; vectors are compared and averaged by their
/// squared length, as in the C++ implementation.
pub trait StatsValue: NanoPod {
    /// The quantity used for ordering.
    type Key: Copy + PartialOrd;
    /// The ordering key of a value.
    fn key(&self) -> Self::Key;
    /// The key as a double (the quantity accumulated by the average and variance).
    fn key_f64(k: Self::Key) -> f64;
    /// Initial minimum (`numeric_limits::max()`).
    fn init_min() -> (Self::Key, Self);
    /// Initial maximum (`numeric_limits::lowest()`).
    fn init_max() -> (Self::Key, Self);
}

macro_rules! stats_scalar {
    ($($t:ty: $lo:expr, $hi:expr);*) => {$(
        impl StatsValue for $t {
            type Key = $t;
            #[inline] fn key(&self) -> $t { *self }
            #[inline] fn key_f64(k: $t) -> f64 { k as f64 }
            fn init_min() -> ($t, $t) { ($hi, $hi) }
            fn init_max() -> ($t, $t) { ($lo, $lo) }
        }
    )*};
}
stats_scalar!(f32: f32::MIN, f32::MAX; f64: f64::MIN, f64::MAX; i32: i32::MIN, i32::MAX; i64: i64::MIN, i64::MAX;
    u32: 0, u32::MAX; u64: 0, u64::MAX);

impl StatsValue for bool {
    type Key = bool;
    fn key(&self) -> bool {
        *self
    }
    fn key_f64(k: bool) -> f64 {
        k as u8 as f64
    }
    fn init_min() -> (bool, bool) {
        (true, true)
    }
    fn init_max() -> (bool, bool) {
        (false, false)
    }
}

macro_rules! stats_vec {
    ($($t:ty, $s:ty);*) => {$(
        impl StatsValue for $t {
            type Key = $s;
            #[inline]
            fn key(&self) -> $s { self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2] }
            #[inline]
            fn key_f64(k: $s) -> f64 { k as f64 }
            fn init_min() -> ($s, $t) { (<$s>::MAX, <$t>::splat(<$s>::MAX)) }
            fn init_max() -> ($s, $t) { (<$s>::MIN, <$t>::splat(<$s>::MIN)) }
        }
    )*};
}
stats_vec!(Vec3f, f32; Vec3d, f64);

/// C-style conversion of a float to an unsigned integer as performed by x86-64 code
/// generated for `static_cast<uintN_t>(float)`: truncation through a signed 32-bit integer,
/// with NaN and out-of-range inputs producing zero in the low bits.
#[inline]
pub(crate) fn c_cast_u32(x: f64) -> u32 {
    const LIM: f64 = 9_223_372_036_854_775_808.0; // 2^63
    if x.is_nan() || !(-LIM..LIM).contains(&x) {
        0
    } else {
        (x as i64) as u32
    }
}
#[inline]
pub(crate) fn c_cast_u16(x: f64) -> u16 {
    if x.is_nan() || !(-2147483648.0..2147483648.0).contains(&x) {
        0
    } else {
        (x as i32) as u16
    }
}
#[inline]
pub(crate) fn c_cast_u8(x: f64) -> u8 {
    c_cast_u16(x) as u8
}

/// A NanoVDB build type: determines the value type of the tree, the statistics type and
/// the layout of every node (`BuildT` in C++).
pub trait BuildType: Send + Sync + 'static {
    /// The type returned by value lookups (`BuildToValueMap<BuildT>::type`).
    type Value: StatsValue;
    /// The type of averages and standard deviations (`FloatTraits<ValueT>::FloatType`).
    type Stats: NanoPod;
    /// The `GridType` tag stored in the grid.
    const GRID_TYPE: GridType;
    /// The grid class forced by this build type, if any (`toGridClass<BuildT>()`).
    const FORCED_CLASS: Option<GridClass> = None;
    /// Layout of the root, its tiles and the internal nodes.
    const NODE: NodeLayout = NodeLayout::new(
        <Self::Value as NanoPod>::SIZE,
        <Self::Value as NanoPod>::ALIGN,
        <Self::Stats as NanoPod>::SIZE,
        <Self::Stats as NanoPod>::ALIGN,
    );
    /// Size of a leaf node in bytes, or 0 if it varies per leaf ([`FpN`]).
    const LEAF_SIZE: usize;
    /// True if statistics (min/max/avg/stddev) are computed for this type.
    const HAS_STATS: bool = true;
    /// True for the index build types.
    const IS_INDEX: bool = false;
    /// True for the quantized float build types.
    const IS_FP: bool = false;

    /// Size of the leaf at byte offset `off`.
    #[inline]
    fn leaf_size(_buf: &[u8], _off: usize) -> usize {
        Self::LEAF_SIZE
    }
    /// The value of voxel `n` of the leaf at `off`.
    fn leaf_value(buf: &[u8], off: usize, n: usize) -> Self::Value;
    /// The stored minimum of the leaf.
    fn leaf_min(buf: &[u8], off: usize) -> Self::Value;
    /// The stored maximum of the leaf.
    fn leaf_max(buf: &[u8], off: usize) -> Self::Value;
    /// The stored average of the leaf.
    fn leaf_avg(buf: &[u8], off: usize) -> Self::Stats;
    /// The stored standard deviation of the leaf.
    fn leaf_dev(buf: &[u8], off: usize) -> Self::Stats;
    /// Store the minimum and maximum of a leaf (no-op for types without statistics).
    fn leaf_set_min_max(buf: &mut [u8], off: usize, min: Self::Value, max: Self::Value);
    /// Store the average and standard deviation of a leaf.
    fn leaf_set_avg_dev(buf: &mut [u8], off: usize, avg: f64, dev: f64);
}

macro_rules! plain_build_type {
    ($($t:ty, $s:ty => $gt:ident);*) => {$(
        impl BuildType for $t {
            type Value = $t;
            type Stats = $s;
            const GRID_TYPE: GridType = GridType::$gt;
            const LEAF_SIZE: usize = Self::LEAF.size;
            #[inline]
            fn leaf_value(buf: &[u8], off: usize, n: usize) -> $t {
                <$t>::read(buf, off + Self::LEAF.values + n * <$t as NanoPod>::SIZE)
            }
            fn leaf_min(buf: &[u8], off: usize) -> $t { <$t>::read(buf, off + Self::LEAF.min) }
            fn leaf_max(buf: &[u8], off: usize) -> $t { <$t>::read(buf, off + Self::LEAF.max) }
            fn leaf_avg(buf: &[u8], off: usize) -> $s { <$s>::read(buf, off + Self::LEAF.avg) }
            fn leaf_dev(buf: &[u8], off: usize) -> $s { <$s>::read(buf, off + Self::LEAF.dev) }
            fn leaf_set_min_max(buf: &mut [u8], off: usize, min: $t, max: $t) {
                min.write(buf, off + Self::LEAF.min);
                max.write(buf, off + Self::LEAF.max);
            }
            fn leaf_set_avg_dev(buf: &mut [u8], off: usize, avg: f64, dev: f64) {
                <$s>::from_f64(avg).write(buf, off + Self::LEAF.avg);
                <$s>::from_f64(dev).write(buf, off + Self::LEAF.dev);
            }
        }
        impl DenseLeaf for $t {
            const LEAF: LeafLayout = LeafLayout::new(
                <$t as NanoPod>::SIZE, <$t as NanoPod>::ALIGN, <$s as NanoPod>::SIZE, <$s as NanoPod>::ALIGN);
        }
    )*};
}

/// Build types whose leaves store a dense array of 512 values (`LeafData<ValueT>`).
pub trait DenseLeaf: BuildType {
    /// Layout of the leaf.
    const LEAF: LeafLayout;
}

plain_build_type!(f32, f32 => Float; f64, f64 => Double; i32, f32 => Int32; i64, f64 => Int64; u32, f32 => UInt32;
    Vec3f, f32 => Vec3f; Vec3d, f64 => Vec3d);

/// Boolean grids: values are stored as a bit mask (`LeafData<bool>`, 160-byte leaves).
impl BuildType for bool {
    type Value = bool;
    type Stats = bool;
    const GRID_TYPE: GridType = GridType::Boolean;
    const LEAF_SIZE: usize = 160;
    const HAS_STATS: bool = false;
    #[inline]
    fn leaf_value(buf: &[u8], off: usize, n: usize) -> bool {
        (rd_u8(buf, off + 80 + (n >> 3)) >> (n & 7)) & 1 != 0
    }
    fn leaf_min(_: &[u8], _: usize) -> bool {
        false
    }
    fn leaf_max(_: &[u8], _: usize) -> bool {
        false
    }
    fn leaf_avg(_: &[u8], _: usize) -> bool {
        false
    }
    fn leaf_dev(_: &[u8], _: usize) -> bool {
        false
    }
    fn leaf_set_min_max(_: &mut [u8], _: usize, _: bool, _: bool) {}
    fn leaf_set_avg_dev(_: &mut [u8], _: usize, _: f64, _: f64) {}
}

/// Build type of topology-only grids: the value of a voxel is its active state
/// (`nanovdb::ValueMask`, 96-byte leaves).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueMask;

impl BuildType for ValueMask {
    type Value = bool;
    type Stats = bool;
    const GRID_TYPE: GridType = GridType::Mask;
    const FORCED_CLASS: Option<GridClass> = Some(GridClass::Topology);
    const LEAF_SIZE: usize = 96;
    const HAS_STATS: bool = false;
    #[inline]
    fn leaf_value(buf: &[u8], off: usize, n: usize) -> bool {
        (rd_u8(buf, off + leaf_off::VALUE_MASK + (n >> 3)) >> (n & 7)) & 1 != 0
    }
    fn leaf_min(_: &[u8], _: usize) -> bool {
        false
    }
    fn leaf_max(_: &[u8], _: usize) -> bool {
        false
    }
    fn leaf_avg(_: &[u8], _: usize) -> bool {
        false
    }
    fn leaf_dev(_: &[u8], _: usize) -> bool {
        false
    }
    fn leaf_set_min_max(_: &mut [u8], _: usize, _: bool, _: bool) {}
    fn leaf_set_avg_dev(_: &mut [u8], _: usize, _: f64, _: f64) {}
}

/// Offsets within the leaves of the quantized types (`LeafFnBase`).
pub(crate) mod fn_off {
    pub const MINIMUM: usize = 80;
    pub const QUANTUM: usize = 84;
    pub const MIN: usize = 88;
    pub const MAX: usize = 90;
    pub const AVG: usize = 92;
    pub const DEV: usize = 94;
    pub const CODE: usize = 96;
}

#[inline]
fn fn_decode(buf: &[u8], off: usize, code: f32) -> f32 {
    code * rd_f32(buf, off + fn_off::QUANTUM) + rd_f32(buf, off + fn_off::MINIMUM)
}

fn fn_set_min_max(buf: &mut [u8], off: usize, min: f32, max: f32) {
    let (m, q) = (rd_f32(buf, off + fn_off::MINIMUM), rd_f32(buf, off + fn_off::QUANTUM));
    wr_u16(buf, off + fn_off::MIN, c_cast_u16(((min - m) / q + 0.5f32) as f64));
    wr_u16(buf, off + fn_off::MAX, c_cast_u16(((max - m) / q + 0.5f32) as f64));
}

fn fn_set_avg_dev(buf: &mut [u8], off: usize, avg: f64, dev: f64) {
    let (m, q) = (rd_f32(buf, off + fn_off::MINIMUM), rd_f32(buf, off + fn_off::QUANTUM));
    let (avg, dev) = (avg as f32, dev as f32);
    wr_u16(buf, off + fn_off::AVG, c_cast_u16(((avg - m) / q + 0.5f32) as f64));
    wr_u16(buf, off + fn_off::DEV, c_cast_u16((dev / q + 0.5f32) as f64));
}

/// Initialize the quantization range of an Fp leaf (`LeafFnBase::init`).
pub(crate) fn fn_init(buf: &mut [u8], off: usize, min: f32, max: f32, bit_width: u32) {
    wr_f32(buf, off + fn_off::MINIMUM, min);
    wr_f32(buf, off + fn_off::QUANTUM, (max - min) / (((1u32 << bit_width) - 1) as f32));
}

macro_rules! fp_build_type {
    ($(#[$m:meta])* $name:ident, $gt:ident, $size:expr, |$b:ident, $o:ident, $n:ident| $code:expr) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name;
        impl BuildType for $name {
            type Value = f32;
            type Stats = f32;
            const GRID_TYPE: GridType = GridType::$gt;
            const LEAF_SIZE: usize = $size;
            const IS_FP: bool = true;
            #[inline]
            fn leaf_value($b: &[u8], $o: usize, $n: usize) -> f32 {
                fn_decode($b, $o, $code)
            }
            fn leaf_min(buf: &[u8], off: usize) -> f32 {
                fn_decode(buf, off, rd_u16(buf, off + fn_off::MIN) as f32)
            }
            fn leaf_max(buf: &[u8], off: usize) -> f32 {
                fn_decode(buf, off, rd_u16(buf, off + fn_off::MAX) as f32)
            }
            fn leaf_avg(buf: &[u8], off: usize) -> f32 {
                fn_decode(buf, off, rd_u16(buf, off + fn_off::AVG) as f32)
            }
            fn leaf_dev(buf: &[u8], off: usize) -> f32 {
                rd_u16(buf, off + fn_off::DEV) as f32 * rd_f32(buf, off + fn_off::QUANTUM)
            }
            fn leaf_set_min_max(buf: &mut [u8], off: usize, min: f32, max: f32) {
                fn_set_min_max(buf, off, min, max)
            }
            fn leaf_set_avg_dev(buf: &mut [u8], off: usize, avg: f64, dev: f64) {
                fn_set_avg_dev(buf, off, avg, dev)
            }
        }
    };
}

fp_build_type!(
    /// 4-bit quantization of floats relative to the per-leaf value range (`nanovdb::Fp4`).
    Fp4, Fp4, 352, |b, o, n| ((b[o + fn_off::CODE + (n >> 1)] >> ((n & 1) << 2)) & 15) as f32
);
fp_build_type!(
    /// 8-bit quantization of floats relative to the per-leaf value range (`nanovdb::Fp8`).
    Fp8, Fp8, 608, |b, o, n| b[o + fn_off::CODE + n] as f32
);
fp_build_type!(
    /// 16-bit quantization of floats relative to the per-leaf value range (`nanovdb::Fp16`).
    Fp16, Fp16, 1120, |b, o, n| rd_u16(b, o + fn_off::CODE + 2 * n) as f32
);

/// Variable bit-width (1, 2, 4, 8 or 16 bits per voxel) quantization of floats
/// (`nanovdb::FpN`). Each leaf stores its log2 bit width in the 3 high bits of its flags and
/// is followed by `bitWidth * 64` bytes of codes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FpN;

impl FpN {
    /// log2 of the bit width of the leaf at `off`.
    #[inline]
    pub fn log2_bit_width(buf: &[u8], off: usize) -> u32 {
        (buf[off + leaf_off::FLAGS] >> 5) as u32
    }
}

impl BuildType for FpN {
    type Value = f32;
    type Stats = f32;
    const GRID_TYPE: GridType = GridType::FpN;
    const LEAF_SIZE: usize = 0;
    const IS_FP: bool = true;
    #[inline]
    fn leaf_size(buf: &[u8], off: usize) -> usize {
        96 + (1usize << Self::log2_bit_width(buf, off)) * 64
    }
    #[inline]
    fn leaf_value(buf: &[u8], off: usize, n: usize) -> f32 {
        let b = Self::log2_bit_width(buf, off);
        let mut code = super::data::rd_u32(buf, off + fn_off::CODE + 4 * (n >> (5 - b)));
        code >>= ((n as u32) & ((32 >> b) - 1)) << b;
        code &= ((1u64 << (1u32 << b)) - 1) as u32;
        fn_decode(buf, off, code as f32)
    }
    fn leaf_min(buf: &[u8], off: usize) -> f32 {
        fn_decode(buf, off, rd_u16(buf, off + fn_off::MIN) as f32)
    }
    fn leaf_max(buf: &[u8], off: usize) -> f32 {
        fn_decode(buf, off, rd_u16(buf, off + fn_off::MAX) as f32)
    }
    fn leaf_avg(buf: &[u8], off: usize) -> f32 {
        fn_decode(buf, off, rd_u16(buf, off + fn_off::AVG) as f32)
    }
    fn leaf_dev(buf: &[u8], off: usize) -> f32 {
        rd_u16(buf, off + fn_off::DEV) as f32 * rd_f32(buf, off + fn_off::QUANTUM)
    }
    fn leaf_set_min_max(buf: &mut [u8], off: usize, min: f32, max: f32) {
        fn_set_min_max(buf, off, min, max)
    }
    fn leaf_set_avg_dev(buf: &mut [u8], off: usize, avg: f64, dev: f64) {
        fn_set_avg_dev(buf, off, avg, dev)
    }
}

/// Offsets within index leaves (`LeafIndexBase`).
pub(crate) mod idx_off {
    pub const OFFSET: usize = 80;
    pub const PREFIX_SUM: usize = 88;
}

/// Common interface of [`ValueIndex`] and [`ValueOnIndex`].
pub trait IndexBuildType: BuildType<Value = u64, Stats = u64> {
    /// True if only active values are indexed.
    const ON_INDEX: bool;
    /// Number of values indexed by the leaf at `off` (excluding statistics).
    fn leaf_value_count(buf: &[u8], off: usize) -> u64;
}

#[inline]
fn idx_has_stats(buf: &[u8], off: usize) -> bool {
    buf[off + leaf_off::FLAGS] & 16 != 0
}

/// Index grids whose voxels map to offsets into an external array of all (active and
/// inactive) values (`nanovdb::ValueIndex`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueIndex;

impl BuildType for ValueIndex {
    type Value = u64;
    type Stats = u64;
    const GRID_TYPE: GridType = GridType::Index;
    const FORCED_CLASS: Option<GridClass> = Some(GridClass::IndexGrid);
    const LEAF_SIZE: usize = 96;
    const HAS_STATS: bool = false;
    const IS_INDEX: bool = true;
    #[inline]
    fn leaf_value(buf: &[u8], off: usize, n: usize) -> u64 {
        rd_u64(buf, off + idx_off::OFFSET) + n as u64
    }
    fn leaf_min(buf: &[u8], off: usize) -> u64 {
        if idx_has_stats(buf, off) {
            rd_u64(buf, off + idx_off::OFFSET) + 512
        } else {
            0
        }
    }
    fn leaf_max(buf: &[u8], off: usize) -> u64 {
        if idx_has_stats(buf, off) {
            rd_u64(buf, off + idx_off::OFFSET) + 513
        } else {
            0
        }
    }
    fn leaf_avg(buf: &[u8], off: usize) -> u64 {
        if idx_has_stats(buf, off) {
            rd_u64(buf, off + idx_off::OFFSET) + 514
        } else {
            0
        }
    }
    fn leaf_dev(buf: &[u8], off: usize) -> u64 {
        if idx_has_stats(buf, off) {
            rd_u64(buf, off + idx_off::OFFSET) + 515
        } else {
            0
        }
    }
    fn leaf_set_min_max(_: &mut [u8], _: usize, _: u64, _: u64) {}
    fn leaf_set_avg_dev(_: &mut [u8], _: usize, _: f64, _: f64) {}
}

impl IndexBuildType for ValueIndex {
    const ON_INDEX: bool = false;
    fn leaf_value_count(_: &[u8], _: usize) -> u64 {
        512
    }
}

/// Index grids whose active voxels map to offsets into an external array of active values;
/// inactive voxels map to 0, the background (`nanovdb::ValueOnIndex`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueOnIndex;

impl ValueOnIndex {
    fn last_offset(buf: &[u8], off: usize) -> u64 {
        rd_u64(buf, off + idx_off::OFFSET) + Self::leaf_value_count(buf, off) - 1
    }
}

impl BuildType for ValueOnIndex {
    type Value = u64;
    type Stats = u64;
    const GRID_TYPE: GridType = GridType::OnIndex;
    const FORCED_CLASS: Option<GridClass> = Some(GridClass::IndexGrid);
    const LEAF_SIZE: usize = 96;
    const HAS_STATS: bool = false;
    const IS_INDEX: bool = true;
    #[inline]
    fn leaf_value(buf: &[u8], off: usize, i: usize) -> u64 {
        let mask = MaskView::<3> { buf, off: off + leaf_off::VALUE_MASK };
        let n = i >> 6;
        let w = mask.word(n);
        let bit = 1u64 << (i & 63);
        if w & bit == 0 {
            return 0;
        }
        let mut sum = rd_u64(buf, off + idx_off::OFFSET) + (w & (bit - 1)).count_ones() as u64;
        if n > 0 {
            sum += (rd_u64(buf, off + idx_off::PREFIX_SUM) >> (9 * (n - 1))) & 511;
        }
        sum
    }
    fn leaf_min(buf: &[u8], off: usize) -> u64 {
        if idx_has_stats(buf, off) {
            Self::last_offset(buf, off) + 1
        } else {
            0
        }
    }
    fn leaf_max(buf: &[u8], off: usize) -> u64 {
        if idx_has_stats(buf, off) {
            Self::last_offset(buf, off) + 2
        } else {
            0
        }
    }
    fn leaf_avg(buf: &[u8], off: usize) -> u64 {
        if idx_has_stats(buf, off) {
            Self::last_offset(buf, off) + 3
        } else {
            0
        }
    }
    fn leaf_dev(buf: &[u8], off: usize) -> u64 {
        if idx_has_stats(buf, off) {
            Self::last_offset(buf, off) + 4
        } else {
            0
        }
    }
    fn leaf_set_min_max(_: &mut [u8], _: usize, _: u64, _: u64) {}
    fn leaf_set_avg_dev(_: &mut [u8], _: usize, _: f64, _: f64) {}
}

impl IndexBuildType for ValueOnIndex {
    const ON_INDEX: bool = true;
    fn leaf_value_count(buf: &[u8], off: usize) -> u64 {
        let mask = MaskView::<3> { buf, off: off + leaf_off::VALUE_MASK };
        mask.word(7).count_ones() as u64 + ((rd_u64(buf, off + idx_off::PREFIX_SUM) >> 54) & 511)
    }
}
