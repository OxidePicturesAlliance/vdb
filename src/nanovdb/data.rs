//! Low-level helpers for the NanoVDB byte layout: little-endian field access, structure
//! layouts and small value types stored in the buffer (`Version`, `Checksum`, `Map`,
//! `GridBlindMetaData`).

use super::{GridBlindDataClass, GridBlindDataSemantic, GridType, MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION};
use crate::math::{Coord, CoordBBox, Vec3d};

// ------------------------------------------------------------------------------------------
// Little-endian field access
// ------------------------------------------------------------------------------------------

#[inline]
pub(crate) fn rd_u8(b: &[u8], o: usize) -> u8 {
    b[o]
}
#[inline]
pub(crate) fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
pub(crate) fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
#[inline]
pub(crate) fn rd_i32(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
#[inline]
pub(crate) fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
#[inline]
pub(crate) fn rd_i64(b: &[u8], o: usize) -> i64 {
    i64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
#[inline]
pub(crate) fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
#[inline]
pub(crate) fn rd_f64(b: &[u8], o: usize) -> f64 {
    f64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
#[inline]
pub(crate) fn rd_coord(b: &[u8], o: usize) -> Coord {
    Coord::new(rd_i32(b, o), rd_i32(b, o + 4), rd_i32(b, o + 8))
}
#[inline]
pub(crate) fn rd_bbox(b: &[u8], o: usize) -> CoordBBox {
    CoordBBox::new(rd_coord(b, o), rd_coord(b, o + 12))
}
#[inline]
pub(crate) fn rd_vec3d(b: &[u8], o: usize) -> Vec3d {
    Vec3d::new(rd_f64(b, o), rd_f64(b, o + 8), rd_f64(b, o + 16))
}

#[inline]
pub(crate) fn wr_u8(b: &mut [u8], o: usize, v: u8) {
    b[o] = v;
}
#[inline]
pub(crate) fn wr_u16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
#[inline]
pub(crate) fn wr_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
pub(crate) fn wr_i32(b: &mut [u8], o: usize, v: i32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
pub(crate) fn wr_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
#[inline]
pub(crate) fn wr_i64(b: &mut [u8], o: usize, v: i64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
#[inline]
pub(crate) fn wr_f32(b: &mut [u8], o: usize, v: f32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
pub(crate) fn wr_f64(b: &mut [u8], o: usize, v: f64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
#[inline]
pub(crate) fn wr_coord(b: &mut [u8], o: usize, c: Coord) {
    wr_i32(b, o, c[0]);
    wr_i32(b, o + 4, c[1]);
    wr_i32(b, o + 8, c[2]);
}
#[inline]
pub(crate) fn wr_bbox(b: &mut [u8], o: usize, bb: &CoordBBox) {
    wr_coord(b, o, bb.min);
    wr_coord(b, o + 12, bb.max);
}
#[inline]
pub(crate) fn wr_vec3d(b: &mut [u8], o: usize, v: Vec3d) {
    wr_f64(b, o, v[0]);
    wr_f64(b, o + 8, v[1]);
    wr_f64(b, o + 16, v[2]);
}

/// Round `n` up to a multiple of `a` (a power of two).
#[inline]
pub(crate) const fn align_up(n: usize, a: usize) -> usize {
    (n + a - 1) & !(a - 1)
}

/// The empty CoordBBox used by NanoVDB (`CoordBBox()`: min = INT_MAX, max = INT_MIN).
#[inline]
pub(crate) fn empty_bbox() -> CoordBBox {
    CoordBBox::new(Coord::splat(i32::MAX), Coord::splat(i32::MIN))
}

/// True if `bbox` is empty in the NanoVDB sense (any min component greater than max).
#[inline]
pub(crate) fn bbox_is_empty(b: &CoordBBox) -> bool {
    let (lo, hi) = (b.min, b.max);
    lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]
}

// ------------------------------------------------------------------------------------------
// Field offsets
// ------------------------------------------------------------------------------------------

/// Byte offsets of the `GridData` fields.
pub(crate) mod grid_off {
    pub const MAGIC: usize = 0;
    pub const CHECKSUM: usize = 8;
    pub const VERSION: usize = 16;
    pub const FLAGS: usize = 20;
    pub const GRID_INDEX: usize = 24;
    pub const GRID_COUNT: usize = 28;
    pub const GRID_SIZE: usize = 32;
    pub const GRID_NAME: usize = 40;
    pub const MAP: usize = 296;
    pub const WORLD_BBOX: usize = 560;
    pub const VOXEL_SIZE: usize = 608;
    pub const GRID_CLASS: usize = 632;
    pub const GRID_TYPE: usize = 636;
    pub const BLIND_META_OFFSET: usize = 640;
    pub const BLIND_META_COUNT: usize = 648;
    pub const DATA0: usize = 652;
    pub const DATA1: usize = 656;
    pub const DATA2: usize = 664;
}

/// Byte offsets of the `TreeData` fields (relative to the tree).
pub(crate) mod tree_off {
    pub const NODE_OFFSET: usize = 0; // 4 x i64: leaf, lower, upper, root
    pub const NODE_COUNT: usize = 32; // 3 x u32
    pub const TILE_COUNT: usize = 44; // 3 x u32
    pub const VOXEL_COUNT: usize = 56;
}

/// Byte offsets of the `GridBlindMetaData` fields.
pub(crate) mod blind_off {
    pub const DATA_OFFSET: usize = 0;
    pub const VALUE_COUNT: usize = 8;
    pub const VALUE_SIZE: usize = 16;
    pub const SEMANTIC: usize = 20;
    pub const DATA_CLASS: usize = 24;
    pub const DATA_TYPE: usize = 28;
    pub const NAME: usize = 32;
}

/// Layout of `RootData`, its tiles and the two `InternalData` levels for a given value
/// type (size/alignment) and statistics type (size/alignment), following the C++ struct
/// layout rules (natural alignment, 32-byte aligned structs and node tables).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeLayout {
    /// Offset of `RootData::mBackground`.
    pub root_background: usize,
    /// Offset of `RootData::mMinimum`.
    pub root_min: usize,
    /// Offset of `RootData::mMaximum`.
    pub root_max: usize,
    /// Offset of `RootData::mAverage`.
    pub root_avg: usize,
    /// Offset of `RootData::mStdDevi`.
    pub root_dev: usize,
    /// `sizeof(RootData)`.
    pub root_size: usize,
    /// Offset of `Tile::value`.
    pub tile_value: usize,
    /// `sizeof(RootData::Tile)`.
    pub tile_size: usize,
    /// Upper (index 0, LOG2DIM = 5) and lower (index 1, LOG2DIM = 4) internal-node layouts.
    pub internal: [InternalLayout; 2],
}

/// Layout of an `InternalData<ChildT, LOG2DIM>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InternalLayout {
    /// Offset of `mValueMask`.
    pub value_mask: usize,
    /// Offset of `mChildMask`.
    pub child_mask: usize,
    /// Offset of `mMinimum`.
    pub min: usize,
    /// Offset of `mMaximum`.
    pub max: usize,
    /// Offset of `mAverage`.
    pub avg: usize,
    /// Offset of `mStdDevi`.
    pub dev: usize,
    /// Offset of `mTable`.
    pub table: usize,
    /// Size of one table entry (a union of a value and an `int64_t`).
    pub stride: usize,
    /// `sizeof(InternalData)`.
    pub size: usize,
}

const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

impl InternalLayout {
    const fn new(log2dim: u32, vs: usize, va: usize, ss: usize, sa: usize) -> Self {
        let mask = 8 * ((1usize << (3 * log2dim)) >> 6);
        let value_mask = 32;
        let child_mask = value_mask + mask;
        let min = align_up(child_mask + mask, va);
        let max_ = min + vs;
        let avg = align_up(max_ + vs, sa);
        let dev = avg + ss;
        let table = align_up(dev + ss, 32);
        let stride = align_up(max(vs, 8), max(va, 8));
        let size = align_up(table + (1usize << (3 * log2dim)) * stride, 32);
        InternalLayout { value_mask, child_mask, min, max: max_, avg, dev, table, stride, size }
    }
}

impl NodeLayout {
    /// Compute the layout for a value type of size `vs`/alignment `va` and a statistics type
    /// of size `ss`/alignment `sa`.
    pub const fn new(vs: usize, va: usize, ss: usize, sa: usize) -> Self {
        let root_background = align_up(28, va);
        let root_min = root_background + vs;
        let root_max = root_min + vs;
        let root_avg = align_up(root_max + vs, sa);
        let root_dev = root_avg + ss;
        let root_size = align_up(root_dev + ss, 32);
        let tile_value = align_up(20, va);
        let tile_size = align_up(tile_value + vs, 32);
        NodeLayout {
            root_background,
            root_min,
            root_max,
            root_avg,
            root_dev,
            root_size,
            tile_value,
            tile_size,
            internal: [InternalLayout::new(5, vs, va, ss, sa), InternalLayout::new(4, vs, va, ss, sa)],
        }
    }
}

/// Layout of a generic `LeafData<ValueT>` (values stored as a dense array).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafLayout {
    /// Offset of `mMinimum`.
    pub min: usize,
    /// Offset of `mMaximum`.
    pub max: usize,
    /// Offset of `mAverage`.
    pub avg: usize,
    /// Offset of `mStdDevi`.
    pub dev: usize,
    /// Offset of `mValues`.
    pub values: usize,
    /// `sizeof(LeafData)`.
    pub size: usize,
}

impl LeafLayout {
    /// Compute the layout of a dense leaf for the given value and statistics types.
    pub const fn new(vs: usize, va: usize, ss: usize, sa: usize) -> Self {
        let min = align_up(80, va);
        let max = min + vs;
        let avg = align_up(max + vs, sa);
        let dev = avg + ss;
        let values = align_up(dev + ss, 32);
        let size = align_up(values + 512 * vs, 32);
        LeafLayout { min, max, avg, dev, values, size }
    }
}

/// Common leaf offsets.
pub(crate) mod leaf_off {
    pub const BBOX_MIN: usize = 0;
    pub const BBOX_DIF: usize = 12;
    pub const FLAGS: usize = 15;
    pub const VALUE_MASK: usize = 16;
}

/// Common internal-node offsets.
pub(crate) mod internal_off {
    pub const BBOX: usize = 0;
    pub const FLAGS: usize = 24;
}

/// Common root offsets.
pub(crate) mod root_off {
    pub const BBOX: usize = 0;
    pub const TABLE_SIZE: usize = 24;
}

/// Root tile offsets.
pub(crate) mod tile_off {
    pub const KEY: usize = 0;
    pub const CHILD: usize = 8;
    pub const STATE: usize = 16;
}

// ------------------------------------------------------------------------------------------
// Mask view
// ------------------------------------------------------------------------------------------

/// A read-only view of a `nanovdb::Mask<LOG2DIM>` stored in a buffer.
#[derive(Clone, Copy)]
pub struct MaskView<'a, const LOG2DIM: u32> {
    pub(crate) buf: &'a [u8],
    pub(crate) off: usize,
}

impl<'a, const LOG2DIM: u32> MaskView<'a, LOG2DIM> {
    /// Number of bits.
    pub const SIZE: usize = 1 << (3 * LOG2DIM);
    /// Number of 64-bit words.
    pub const WORD_COUNT: usize = Self::SIZE >> 6;

    /// The `n`-th 64-bit word.
    #[inline]
    pub fn word(&self, n: usize) -> u64 {
        rd_u64(self.buf, self.off + 8 * n)
    }
    /// True if bit `n` is set.
    #[inline]
    pub fn is_on(&self, n: usize) -> bool {
        (self.buf[self.off + (n >> 3)] >> (n & 7)) & 1 != 0
    }
    /// True if bit `n` is not set.
    #[inline]
    pub fn is_off(&self, n: usize) -> bool {
        !self.is_on(n)
    }
    /// Number of set bits.
    pub fn count_on(&self) -> u32 {
        (0..Self::WORD_COUNT).map(|i| self.word(i).count_ones()).sum()
    }
    /// Number of set bits below bit `i` (`Mask::countOn(i)`).
    pub fn count_on_below(&self, i: usize) -> u32 {
        let n = i >> 6;
        let mut sum = (self.word(n) & ((1u64 << (i & 63)) - 1)).count_ones();
        for w in 0..n {
            sum += self.word(w).count_ones();
        }
        sum
    }
    /// True if no bit is set.
    pub fn is_all_off(&self) -> bool {
        (0..Self::WORD_COUNT).all(|i| self.word(i) == 0)
    }
    /// True if all bits are set.
    pub fn is_all_on(&self) -> bool {
        (0..Self::WORD_COUNT).all(|i| self.word(i) == !0)
    }
    /// Iterate over the indices of the set bits in increasing order.
    pub fn iter_on(&self) -> impl Iterator<Item = usize> + 'a {
        let (buf, off) = (self.buf, self.off);
        (0..Self::WORD_COUNT).flat_map(move |w| {
            let mut word = rd_u64(buf, off + 8 * w);
            std::iter::from_fn(move || {
                if word == 0 {
                    None
                } else {
                    let b = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some((w << 6) + b)
                }
            })
        })
    }
    /// Copy the mask into a [`NodeMask`](crate::util::NodeMask).
    pub fn to_node_mask(&self) -> crate::util::NodeMask<LOG2DIM> {
        let mut m = crate::util::NodeMask::<LOG2DIM>::new();
        for (i, w) in m.words_mut().iter_mut().enumerate() {
            *w = self.word(i);
        }
        m
    }
}

/// Write a [`NodeMask`](crate::util::NodeMask) into a buffer.
pub(crate) fn wr_mask<const L: u32>(b: &mut [u8], o: usize, m: &crate::util::NodeMask<L>) {
    for (i, w) in m.words().iter().enumerate() {
        wr_u64(b, o + 8 * i, *w);
    }
}

// ------------------------------------------------------------------------------------------
// Version
// ------------------------------------------------------------------------------------------

/// Bit-packed major (11 bits), minor (11 bits) and patch (10 bits) version numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(pub u32);

impl Default for Version {
    fn default() -> Self {
        Version::new(MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
    }
}

impl Version {
    /// Pack version numbers.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Version((major << 21) | (minor << 10) | patch)
    }
    /// The raw packed value.
    pub fn id(&self) -> u32 {
        self.0
    }
    /// Major version.
    pub fn major(&self) -> u32 {
        (self.0 >> 21) & ((1 << 11) - 1)
    }
    /// Minor version.
    pub fn minor(&self) -> u32 {
        (self.0 >> 10) & ((1 << 11) - 1)
    }
    /// Patch version.
    pub fn patch(&self) -> u32 {
        self.0 & ((1 << 10) - 1)
    }
    /// True if the major version equals [`MAJOR_VERSION`].
    pub fn is_compatible(&self) -> bool {
        self.major() == MAJOR_VERSION
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major(), self.minor(), self.patch())
    }
}

// ------------------------------------------------------------------------------------------
// Checksum
// ------------------------------------------------------------------------------------------

/// Two CRC32 checksums: `head` covers the grid and tree headers, `tail` the remainder of
/// the grid (nodes and blind data). All bits set means "disabled".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Checksum(pub u64);

impl Default for Checksum {
    fn default() -> Self {
        Checksum::EMPTY
    }
}

impl Checksum {
    /// The disabled checksum.
    pub const EMPTY: Checksum = Checksum(!0u64);
    /// 32-bit "empty" marker.
    pub const EMPTY32: u32 = !0u32;
    /// Build from the two halves.
    pub fn from_parts(head: u32, tail: u32) -> Self {
        Checksum(head as u64 | ((tail as u64) << 32))
    }
    /// CRC32 of the grid and tree headers.
    pub fn head(&self) -> u32 {
        self.0 as u32
    }
    /// CRC32 of the nodes and blind data.
    pub fn tail(&self) -> u32 {
        (self.0 >> 32) as u32
    }
    /// True if disabled.
    pub fn is_empty(&self) -> bool {
        self.0 == !0
    }
    /// True if only the head checksum is set.
    pub fn is_half(&self) -> bool {
        self.head() != Self::EMPTY32 && self.tail() == Self::EMPTY32
    }
    /// True if both checksums are set.
    pub fn is_full(&self) -> bool {
        self.0 != !0 && self.tail() != Self::EMPTY32
    }
    /// The mode this checksum was computed with.
    pub fn mode(&self) -> super::CheckMode {
        if self.is_empty() {
            super::CheckMode::Disable
        } else if self.tail() == Self::EMPTY32 {
            super::CheckMode::Partial
        } else {
            super::CheckMode::Full
        }
    }
}

// ------------------------------------------------------------------------------------------
// Map
// ------------------------------------------------------------------------------------------

/// The affine index-to-world transform stored in every grid, in single and double
/// precision (`nanovdb::Map`, 264 bytes). Matrices are stored transposed relative to
/// OpenVDB, i.e. `mat[3 * i + j]` is OpenVDB's `mat[j][i]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Map {
    /// Forward 3x3 matrix (single precision).
    pub mat_f: [f32; 9],
    /// Inverse 3x3 matrix (single precision).
    pub inv_mat_f: [f32; 9],
    /// Translation (single precision).
    pub vec_f: [f32; 3],
    /// Taper (single precision, unused).
    pub taper_f: f32,
    /// Forward 3x3 matrix (double precision).
    pub mat_d: [f64; 9],
    /// Inverse 3x3 matrix (double precision).
    pub inv_mat_d: [f64; 9],
    /// Translation (double precision).
    pub vec_d: [f64; 3],
    /// Taper (double precision, unused).
    pub taper_d: f64,
}

impl Default for Map {
    fn default() -> Self {
        Map::from_scale_translation(1.0, Vec3d::zero())
    }
}

#[inline]
fn mat_mult(m: &[f64; 9], v: [f64; 3]) -> Vec3d {
    Vec3d::new(
        v[0].mul_add(m[0], v[1].mul_add(m[1], v[2] * m[2])),
        v[0].mul_add(m[3], v[1].mul_add(m[4], v[2] * m[5])),
        v[0].mul_add(m[6], v[1].mul_add(m[7], v[2] * m[8])),
    )
}
#[inline]
fn mat_mult_add(m: &[f64; 9], t: &[f64; 3], v: [f64; 3]) -> Vec3d {
    Vec3d::new(
        v[0].mul_add(m[0], v[1].mul_add(m[1], v[2].mul_add(m[2], t[0]))),
        v[0].mul_add(m[3], v[1].mul_add(m[4], v[2].mul_add(m[5], t[1]))),
        v[0].mul_add(m[6], v[1].mul_add(m[7], v[2].mul_add(m[8], t[2]))),
    )
}
#[inline]
fn mat_mult_t(m: &[f64; 9], v: [f64; 3]) -> Vec3d {
    Vec3d::new(
        v[0].mul_add(m[0], v[1].mul_add(m[3], v[2] * m[6])),
        v[0].mul_add(m[1], v[1].mul_add(m[4], v[2] * m[7])),
        v[0].mul_add(m[2], v[1].mul_add(m[5], v[2] * m[8])),
    )
}
#[inline]
fn mat_mult_f(m: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        v[0].mul_add(m[0], v[1].mul_add(m[1], v[2] * m[2])),
        v[0].mul_add(m[3], v[1].mul_add(m[4], v[2] * m[5])),
        v[0].mul_add(m[6], v[1].mul_add(m[7], v[2] * m[8])),
    ]
}
#[inline]
fn mat_mult_add_f(m: &[f32; 9], t: &[f32; 3], v: [f32; 3]) -> [f32; 3] {
    [
        v[0].mul_add(m[0], v[1].mul_add(m[1], v[2].mul_add(m[2], t[0]))),
        v[0].mul_add(m[3], v[1].mul_add(m[4], v[2].mul_add(m[5], t[1]))),
        v[0].mul_add(m[6], v[1].mul_add(m[7], v[2].mul_add(m[8], t[2]))),
    ]
}

impl Map {
    /// Size in bytes.
    pub const SIZE: usize = 264;

    /// A uniform scale followed by a translation (`Map(double s, Vec3d t)` / `Map::set(dx, t)`).
    pub fn from_scale_translation(s: f64, t: Vec3d) -> Self {
        let mat = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]];
        let inv = 1.0 / s;
        let inv_mat = [[inv, 0.0, 0.0], [0.0, inv, 0.0], [0.0, 0.0, inv]];
        Self::from_mat3(&mat, &inv_mat, [t[0], t[1], t[2]], 1.0)
    }

    /// Initialize from OpenVDB-convention 3x3 matrices (`Map::set(mat, invMat, translate)`).
    pub fn from_mat3(mat: &[[f64; 3]; 3], inv_mat: &[[f64; 3]; 3], translate: [f64; 3], taper: f64) -> Self {
        let mut m = Map {
            mat_f: [0.0; 9],
            inv_mat_f: [0.0; 9],
            vec_f: [0.0; 3],
            taper_f: taper as f32,
            mat_d: [0.0; 9],
            inv_mat_d: [0.0; 9],
            vec_d: [0.0; 3],
            taper_d: taper,
        };
        for i in 0..3 {
            m.vec_d[i] = translate[i];
            m.vec_f[i] = translate[i] as f32;
            for j in 0..3 {
                m.mat_d[3 * i + j] = mat[j][i];
                m.inv_mat_d[3 * i + j] = inv_mat[j][i];
                m.mat_f[3 * i + j] = mat[j][i] as f32;
                m.inv_mat_f[3 * i + j] = inv_mat[j][i] as f32;
            }
        }
        m
    }

    /// Initialize from OpenVDB 4x4 matrices (translation in row 3).
    pub fn from_mat4(mat: &crate::math::Mat4d, inv: &crate::math::Mat4d) -> Self {
        let m3 = |m: &crate::math::Mat4d| {
            let a = &m.0;
            [[a[0][0], a[0][1], a[0][2]], [a[1][0], a[1][1], a[1][2]], [a[2][0], a[2][1], a[2][2]]]
        };
        Self::from_mat3(&m3(mat), &m3(inv), [mat.0[3][0], mat.0[3][1], mat.0[3][2]], 1.0)
    }

    /// The map of a [`Transform`](crate::math::Transform): `mMap.set(mat4, mat4.inverse())`
    /// with the transform's affine matrix.
    pub fn from_transform(t: &crate::math::Transform) -> Self {
        let mat = t.base_map().affine_matrix();
        let inv = mat.inverse(0.0).unwrap_or_else(crate::math::Mat4d::identity);
        Self::from_mat4(&mat, &inv)
    }

    /// Read a map from a buffer.
    pub fn read(b: &[u8], o: usize) -> Self {
        let mut m = Map::default();
        for i in 0..9 {
            m.mat_f[i] = rd_f32(b, o + 4 * i);
            m.inv_mat_f[i] = rd_f32(b, o + 36 + 4 * i);
            m.mat_d[i] = rd_f64(b, o + 88 + 8 * i);
            m.inv_mat_d[i] = rd_f64(b, o + 160 + 8 * i);
        }
        for i in 0..3 {
            m.vec_f[i] = rd_f32(b, o + 72 + 4 * i);
            m.vec_d[i] = rd_f64(b, o + 232 + 8 * i);
        }
        m.taper_f = rd_f32(b, o + 84);
        m.taper_d = rd_f64(b, o + 256);
        m
    }

    /// Write the map into a buffer.
    pub fn write(&self, b: &mut [u8], o: usize) {
        for i in 0..9 {
            wr_f32(b, o + 4 * i, self.mat_f[i]);
            wr_f32(b, o + 36 + 4 * i, self.inv_mat_f[i]);
            wr_f64(b, o + 88 + 8 * i, self.mat_d[i]);
            wr_f64(b, o + 160 + 8 * i, self.inv_mat_d[i]);
        }
        for i in 0..3 {
            wr_f32(b, o + 72 + 4 * i, self.vec_f[i]);
            wr_f64(b, o + 232 + 8 * i, self.vec_d[i]);
        }
        wr_f32(b, o + 84, self.taper_f);
        wr_f64(b, o + 256, self.taper_d);
    }

    /// Index to world (double precision): `mat * ijk + translation`.
    pub fn apply_map(&self, ijk: Vec3d) -> Vec3d {
        mat_mult_add(&self.mat_d, &self.vec_d, ijk.0)
    }
    /// World to index (double precision).
    pub fn apply_inverse_map(&self, xyz: Vec3d) -> Vec3d {
        mat_mult(&self.inv_mat_d, [xyz[0] - self.vec_d[0], xyz[1] - self.vec_d[1], xyz[2] - self.vec_d[2]])
    }
    /// Linear part of the forward map (directions, index to world).
    pub fn apply_jacobian(&self, ijk: Vec3d) -> Vec3d {
        mat_mult(&self.mat_d, ijk.0)
    }
    /// Linear part of the inverse map (directions, world to index).
    pub fn apply_inverse_jacobian(&self, xyz: Vec3d) -> Vec3d {
        mat_mult(&self.inv_mat_d, xyz.0)
    }
    /// Transposed inverse Jacobian (gradients, index to world).
    pub fn apply_ijt(&self, xyz: Vec3d) -> Vec3d {
        mat_mult_t(&self.inv_mat_d, xyz.0)
    }
    /// Index to world (single precision).
    pub fn apply_map_f(&self, ijk: [f32; 3]) -> [f32; 3] {
        mat_mult_add_f(&self.mat_f, &self.vec_f, ijk)
    }
    /// World to index (single precision).
    pub fn apply_inverse_map_f(&self, xyz: [f32; 3]) -> [f32; 3] {
        mat_mult_f(&self.inv_mat_f, [xyz[0] - self.vec_f[0], xyz[1] - self.vec_f[1], xyz[2] - self.vec_f[2]])
    }
    /// Linear forward map (single precision).
    pub fn apply_jacobian_f(&self, ijk: [f32; 3]) -> [f32; 3] {
        mat_mult_f(&self.mat_f, ijk)
    }
    /// Linear inverse map (single precision).
    pub fn apply_inverse_jacobian_f(&self, xyz: [f32; 3]) -> [f32; 3] {
        mat_mult_f(&self.inv_mat_f, xyz)
    }
    /// Size of a voxel along each axis, measured at the origin.
    pub fn voxel_size(&self) -> Vec3d {
        self.apply_map(Vec3d::splat(1.0)) - self.apply_map(Vec3d::splat(0.0))
    }
    /// The equivalent OpenVDB 4x4 matrix (translation in row 3).
    pub fn to_mat4(&self) -> crate::math::Mat4d {
        let mut m = crate::math::Mat4d::identity();
        for i in 0..3 {
            for j in 0..3 {
                m.0[j][i] = self.mat_d[3 * i + j];
            }
            m.0[3][i] = self.vec_d[i];
        }
        m
    }
    /// World-space bounds of the index-space box `bbox` (`CoordBBox::transform(map)`),
    /// treating `bbox` as a box of continuous coordinates.
    pub fn transform_bbox(&self, lo: Coord, hi: Coord) -> (Vec3d, Vec3d) {
        let (a, b) = (lo.as_vec3d(), hi.as_vec3d());
        let first = self.apply_map(a);
        let (mut mn, mut mx) = (first, first);
        let corners = [
            [a[0], a[1], b[2]],
            [a[0], b[1], a[2]],
            [b[0], a[1], a[2]],
            [b[0], b[1], a[2]],
            [b[0], a[1], b[2]],
            [a[0], b[1], b[2]],
            [b[0], b[1], b[2]],
        ];
        for c in corners {
            let p = self.apply_map(Vec3d::new(c[0], c[1], c[2]));
            for k in 0..3 {
                if p[k] < mn[k] {
                    mn[k] = p[k];
                }
                if p[k] > mx[k] {
                    mx[k] = p[k];
                }
            }
        }
        (mn, mx)
    }
}

// ------------------------------------------------------------------------------------------
// GridBlindMetaData
// ------------------------------------------------------------------------------------------

/// Description of a blind-data block attached to a grid (`nanovdb::GridBlindMetaData`,
/// 288 bytes). `data_offset` is relative to the start of this structure.
#[derive(Clone, Debug, PartialEq)]
pub struct GridBlindMetaData {
    /// Byte offset of the blind data relative to this meta data.
    pub data_offset: i64,
    /// Number of values.
    pub value_count: u64,
    /// Size of each value in bytes.
    pub value_size: u32,
    /// Semantic meaning of the data.
    pub semantic: GridBlindDataSemantic,
    /// Class of the data.
    pub data_class: GridBlindDataClass,
    /// Value type of the data.
    pub data_type: GridType,
    /// Name (at most 255 bytes).
    pub name: String,
}

impl GridBlindMetaData {
    /// Read meta data at offset `o`.
    pub fn read(b: &[u8], o: usize) -> Self {
        let name_bytes = &b[o + blind_off::NAME..o + blind_off::NAME + super::MAX_NAME_SIZE];
        let end = name_bytes.iter().position(|&c| c == 0).unwrap_or(name_bytes.len());
        GridBlindMetaData {
            data_offset: rd_i64(b, o + blind_off::DATA_OFFSET),
            value_count: rd_u64(b, o + blind_off::VALUE_COUNT),
            value_size: rd_u32(b, o + blind_off::VALUE_SIZE),
            semantic: GridBlindDataSemantic::from_u32(rd_u32(b, o + blind_off::SEMANTIC)),
            data_class: GridBlindDataClass::from_u32(rd_u32(b, o + blind_off::DATA_CLASS)),
            data_type: GridType::from_u32(rd_u32(b, o + blind_off::DATA_TYPE)),
            name: String::from_utf8_lossy(&name_bytes[..end]).into_owned(),
        }
    }
    /// Write the meta data at offset `o` (the name is zero padded).
    pub fn write(&self, b: &mut [u8], o: usize) {
        wr_i64(b, o + blind_off::DATA_OFFSET, self.data_offset);
        wr_u64(b, o + blind_off::VALUE_COUNT, self.value_count);
        wr_u32(b, o + blind_off::VALUE_SIZE, self.value_size);
        wr_u32(b, o + blind_off::SEMANTIC, self.semantic.to_u32());
        wr_u32(b, o + blind_off::DATA_CLASS, self.data_class.to_u32());
        wr_u32(b, o + blind_off::DATA_TYPE, self.data_type.to_u32());
        let n = &mut b[o + blind_off::NAME..o + blind_off::NAME + super::MAX_NAME_SIZE];
        n.fill(0);
        let src = self.name.as_bytes();
        let len = src.len().min(super::MAX_NAME_SIZE - 1);
        n[..len].copy_from_slice(&src[..len]);
    }
    /// Size of the blind data in bytes, padded to 32 bytes.
    pub fn blind_data_size(&self) -> u64 {
        align_up((self.value_count * self.value_size as u64) as usize, 32) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_cpp() {
        // Values printed by a C++ program using sizeof/offsetof on NanoVDB 32.9.2.
        let f = NodeLayout::new(4, 4, 4, 4);
        assert_eq!((f.root_background, f.root_min, f.root_avg, f.root_size), (28, 32, 40, 64));
        assert_eq!((f.tile_value, f.tile_size), (20, 32));
        assert_eq!(
            (f.internal[0].min, f.internal[0].avg, f.internal[0].table, f.internal[0].size),
            (8224, 8232, 8256, 270400)
        );
        assert_eq!((f.internal[1].table, f.internal[1].size), (1088, 33856));
        let d = NodeLayout::new(8, 8, 8, 8);
        assert_eq!((d.root_background, d.root_min, d.root_avg, d.root_size, d.tile_value), (32, 40, 56, 96, 24));
        let v3f = NodeLayout::new(12, 4, 4, 4);
        assert_eq!((v3f.root_min, v3f.root_avg, v3f.root_size, v3f.internal[0].size), (40, 64, 96, 532544));
        let v3d = NodeLayout::new(24, 8, 8, 8);
        assert_eq!(
            (v3d.root_size, v3d.tile_size, v3d.internal[0].table, v3d.internal[0].size),
            (128, 64, 8288, 794720)
        );
        assert_eq!(v3d.internal[1].size, 99424);
        let b = NodeLayout::new(1, 1, 1, 1);
        assert_eq!((b.root_min, b.root_avg, b.root_dev, b.root_size, b.internal[0].avg), (29, 31, 32, 64, 8226));
        assert_eq!(LeafLayout::new(4, 4, 4, 4).size, 2144);
        assert_eq!(LeafLayout::new(8, 8, 8, 8).size, 4224);
        let l = LeafLayout::new(12, 4, 4, 4);
        assert_eq!((l.avg, l.values, l.size), (104, 128, 6272));
        let l = LeafLayout::new(24, 8, 8, 8);
        assert_eq!((l.avg, l.values, l.size), (128, 160, 12448));
    }

    #[test]
    fn version_packing() {
        assert_eq!(Version::default().id(), 67118082);
        assert_eq!(Version::default().to_string(), "32.9.2");
    }
}
