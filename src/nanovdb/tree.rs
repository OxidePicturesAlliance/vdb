//! Zero-copy views of the tree, root, internal and leaf nodes of a NanoVDB grid, and the
//! caching [`ReadAccessor`].

use super::build_types::{BuildType, NanoPod};
use super::data::{
    internal_off, leaf_off, rd_bbox, rd_coord, rd_i64, rd_u32, rd_u64, rd_u8, root_off, tile_off, tree_off,
    InternalLayout, MaskView,
};
use crate::math::{Coord, CoordBBox};
use std::cell::Cell;
use std::marker::PhantomData;

/// The 64-bit key of a root-table entry (`RootData::CoordToKey` with
/// `NANOVDB_USE_SINGLE_ROOT_KEY`): 21 bits per axis of the coordinate shifted by 12.
#[inline]
pub fn coord_to_key(ijk: Coord) -> u64 {
    ((ijk[2] as u32 >> 12) as u64) | (((ijk[1] as u32 >> 12) as u64) << 21) | (((ijk[0] as u32 >> 12) as u64) << 42)
}

/// Inverse of [`coord_to_key`].
#[inline]
pub fn key_to_coord(key: u64) -> Coord {
    const MASK: u64 = (1 << 21) - 1;
    Coord::new(
        (((key >> 42) & MASK) << 12) as u32 as i32,
        (((key >> 21) & MASK) << 12) as u32 as i32,
        ((key & MASK) << 12) as u32 as i32,
    )
}

// ------------------------------------------------------------------------------------------
// Tree
// ------------------------------------------------------------------------------------------

/// View of `TreeData` (`nanovdb::NanoTree<BuildT>`).
pub struct NanoTree<'a, B: BuildType> {
    pub(crate) buf: &'a [u8],
    pub(crate) off: usize,
    _m: PhantomData<B>,
}

impl<'a, B: BuildType> NanoTree<'a, B> {
    pub(crate) fn new(buf: &'a [u8], off: usize) -> Self {
        NanoTree { buf, off, _m: PhantomData }
    }
    /// Byte offset of the first node at `level` (0 = leaf ... 3 = root) relative to the
    /// grid, or `None` if there are no such nodes.
    pub fn first_node_offset(&self, level: usize) -> Option<usize> {
        let o = rd_i64(self.buf, self.off + tree_off::NODE_OFFSET + 8 * level);
        (o != 0).then(|| (self.off as i64 + o) as usize)
    }
    /// The root node.
    pub fn root(&self) -> NanoRoot<'a, B> {
        NanoRoot { buf: self.buf, off: self.first_node_offset(3).expect("tree without root"), _m: PhantomData }
    }
    /// Number of nodes at `level` (0 = leaf, 1 = lower, 2 = upper).
    pub fn node_count(&self, level: usize) -> u32 {
        rd_u32(self.buf, self.off + tree_off::NODE_COUNT + 4 * level)
    }
    /// Number of active tiles at `level` (1 = lower, 2 = upper, 3 = root).
    pub fn active_tile_count(&self, level: usize) -> u32 {
        rd_u32(self.buf, self.off + tree_off::TILE_COUNT + 4 * (level - 1))
    }
    /// Total number of active voxels (including those covered by active tiles).
    pub fn active_voxel_count(&self) -> u64 {
        rd_u64(self.buf, self.off + tree_off::VOXEL_COUNT)
    }
    /// Value at `ijk`.
    pub fn get_value(&self, ijk: Coord) -> B::Value {
        self.root().get_value(ijk)
    }
    /// Active state at `ijk`.
    pub fn is_active(&self, ijk: Coord) -> bool {
        self.root().is_active(ijk)
    }
    /// Value and active state at `ijk`.
    pub fn probe_value(&self, ijk: Coord) -> (B::Value, bool) {
        self.root().probe_value(ijk)
    }
    /// The background value.
    pub fn background(&self) -> B::Value {
        self.root().background()
    }
    /// Minimum and maximum active values stored in the root (`Tree::extrema`).
    pub fn extrema(&self) -> (B::Value, B::Value) {
        let r = self.root();
        (r.minimum(), r.maximum())
    }
    /// The leaves, in memory (breadth-first) order.
    pub fn leaves(&self) -> impl Iterator<Item = NanoLeaf<'a, B>> + 'a {
        let buf = self.buf;
        let mut off = self.first_node_offset(0).unwrap_or(0);
        (0..self.node_count(0)).map(move |_| {
            let leaf = NanoLeaf { buf, off, _m: PhantomData };
            off += B::leaf_size(buf, off);
            leaf
        })
    }
    /// The lower internal nodes, in memory order.
    pub fn lower_nodes(&self) -> impl Iterator<Item = NanoLower<'a, B>> + 'a {
        let (buf, start) = (self.buf, self.first_node_offset(1).unwrap_or(0));
        let size = B::NODE.internal[1].size;
        (0..self.node_count(1) as usize).map(move |i| NanoInternal::at(buf, start + i * size))
    }
    /// The upper internal nodes, in memory order.
    pub fn upper_nodes(&self) -> impl Iterator<Item = NanoUpper<'a, B>> + 'a {
        let (buf, start) = (self.buf, self.first_node_offset(2).unwrap_or(0));
        let size = B::NODE.internal[0].size;
        (0..self.node_count(2) as usize).map(move |i| NanoInternal::at(buf, start + i * size))
    }
}

// ------------------------------------------------------------------------------------------
// Value location
// ------------------------------------------------------------------------------------------

/// Where the value of a voxel is stored, as found by a top-down traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueLocation {
    /// Outside the root table: the background value.
    Background,
    /// A root tile, at the given byte offset.
    RootTile(usize),
    /// Slot `n` of the upper node at the given offset.
    Upper(usize, usize),
    /// Slot `n` of the lower node at the given offset.
    Lower(usize, usize),
    /// Voxel `n` of the leaf at the given offset.
    Leaf(usize, usize),
}

impl ValueLocation {
    /// The tree level holding the value (0 = voxel, 3 = root tile or background).
    pub fn level(&self) -> u32 {
        match self {
            ValueLocation::Leaf(..) => 0,
            ValueLocation::Lower(..) => 1,
            ValueLocation::Upper(..) => 2,
            _ => 3,
        }
    }
}

// ------------------------------------------------------------------------------------------
// Root
// ------------------------------------------------------------------------------------------

/// View of `RootData` and its table of tiles (`nanovdb::NanoRoot<BuildT>`).
pub struct NanoRoot<'a, B: BuildType> {
    pub(crate) buf: &'a [u8],
    pub(crate) off: usize,
    _m: PhantomData<B>,
}

impl<'a, B: BuildType> NanoRoot<'a, B> {
    /// Byte offset of this node relative to the grid.
    pub fn offset(&self) -> usize {
        self.off
    }
    /// Index-space bounding box of the active values.
    pub fn bbox(&self) -> CoordBBox {
        rd_bbox(self.buf, self.off + root_off::BBOX)
    }
    /// Number of tiles (values and children) in the table.
    pub fn tile_count(&self) -> u32 {
        rd_u32(self.buf, self.off + root_off::TABLE_SIZE)
    }
    /// True if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.tile_count() == 0
    }
    /// The background value.
    pub fn background(&self) -> B::Value {
        B::Value::read(self.buf, self.off + B::NODE.root_background)
    }
    /// Minimum active value.
    pub fn minimum(&self) -> B::Value {
        B::Value::read(self.buf, self.off + B::NODE.root_min)
    }
    /// Maximum active value.
    pub fn maximum(&self) -> B::Value {
        B::Value::read(self.buf, self.off + B::NODE.root_max)
    }
    /// Average of the active values.
    pub fn average(&self) -> B::Stats {
        B::Stats::read(self.buf, self.off + B::NODE.root_avg)
    }
    /// Standard deviation of the active values.
    pub fn std_deviation(&self) -> B::Stats {
        B::Stats::read(self.buf, self.off + B::NODE.root_dev)
    }
    /// The `n`-th tile.
    pub fn tile(&self, n: u32) -> RootTile<'a, B> {
        RootTile {
            buf: self.buf,
            root: self.off,
            off: self.off + B::NODE.root_size + n as usize * B::NODE.tile_size,
            _m: PhantomData,
        }
    }
    /// All tiles in table order.
    pub fn tiles(&self) -> impl Iterator<Item = RootTile<'a, B>> + 'a {
        let r = *self;
        (0..self.tile_count()).map(move |n| r.tile(n))
    }
    /// The tile whose key matches `ijk`, if any (`RootData::probeTile`).
    pub fn probe_tile(&self, ijk: Coord) -> Option<RootTile<'a, B>> {
        let key = coord_to_key(ijk);
        self.tiles().find(|t| t.key() == key)
    }
    /// The upper node containing `ijk`, if any.
    pub fn probe_child(&self, ijk: Coord) -> Option<NanoUpper<'a, B>> {
        self.probe_tile(ijk).and_then(|t| t.child())
    }
    /// Locate the value of `ijk` without caching.
    pub fn locate(&self, ijk: Coord) -> ValueLocation {
        match self.probe_tile(ijk) {
            None => ValueLocation::Background,
            Some(t) => match t.child() {
                None => ValueLocation::RootTile(t.off),
                Some(upper) => upper.locate(ijk),
            },
        }
    }
    /// Value stored at a location.
    pub fn value_at(&self, loc: ValueLocation) -> B::Value {
        let n = &B::NODE;
        match loc {
            ValueLocation::Background => self.background(),
            ValueLocation::RootTile(o) => B::Value::read(self.buf, o + n.tile_value),
            ValueLocation::Upper(o, i) => B::Value::read(self.buf, o + n.internal[0].table + i * n.internal[0].stride),
            ValueLocation::Lower(o, i) => B::Value::read(self.buf, o + n.internal[1].table + i * n.internal[1].stride),
            ValueLocation::Leaf(o, i) => B::leaf_value(self.buf, o, i),
        }
    }
    /// Active state at a location.
    pub fn active_at(&self, loc: ValueLocation) -> bool {
        let n = &B::NODE;
        let bit = |o: usize, i: usize| (self.buf[o + (i >> 3)] >> (i & 7)) & 1 != 0;
        match loc {
            ValueLocation::Background => false,
            ValueLocation::RootTile(o) => rd_u32(self.buf, o + tile_off::STATE) > 0,
            ValueLocation::Upper(o, i) => bit(o + n.internal[0].value_mask, i),
            ValueLocation::Lower(o, i) => bit(o + n.internal[1].value_mask, i),
            ValueLocation::Leaf(o, i) => bit(o + leaf_off::VALUE_MASK, i),
        }
    }
    /// Value at `ijk`.
    pub fn get_value(&self, ijk: Coord) -> B::Value {
        self.value_at(self.locate(ijk))
    }
    /// Active state at `ijk`.
    pub fn is_active(&self, ijk: Coord) -> bool {
        self.active_at(self.locate(ijk))
    }
    /// Value and active state at `ijk`.
    pub fn probe_value(&self, ijk: Coord) -> (B::Value, bool) {
        let loc = self.locate(ijk);
        (self.value_at(loc), self.active_at(loc))
    }
    /// The leaf containing `ijk`, if any.
    pub fn probe_leaf(&self, ijk: Coord) -> Option<NanoLeaf<'a, B>> {
        match self.locate(ijk) {
            ValueLocation::Leaf(o, _) => Some(NanoLeaf { buf: self.buf, off: o, _m: PhantomData }),
            _ => None,
        }
    }
    /// A caching accessor.
    pub fn accessor(&self) -> ReadAccessor<'a, B> {
        ReadAccessor::new(*self)
    }
}

/// An entry of the root table (`RootData::Tile`): either a child upper node or a constant
/// tile value spanning 4096³ voxels.
pub struct RootTile<'a, B: BuildType> {
    buf: &'a [u8],
    root: usize,
    pub(crate) off: usize,
    _m: PhantomData<B>,
}

impl<'a, B: BuildType> RootTile<'a, B> {
    /// The 64-bit key.
    pub fn key(&self) -> u64 {
        rd_u64(self.buf, self.off + tile_off::KEY)
    }
    /// Origin of the tile.
    pub fn origin(&self) -> Coord {
        key_to_coord(self.key())
    }
    /// Byte offset of the child relative to the root, or 0 for a value tile.
    pub fn child_offset(&self) -> i64 {
        rd_i64(self.buf, self.off + tile_off::CHILD)
    }
    /// True if this entry holds a child node.
    pub fn is_child(&self) -> bool {
        self.child_offset() != 0
    }
    /// The raw state of a value tile.
    pub fn state(&self) -> u32 {
        rd_u32(self.buf, self.off + tile_off::STATE)
    }
    /// True for an active value tile.
    pub fn is_active(&self) -> bool {
        !self.is_child() && self.state() != 0
    }
    /// The tile value (meaningless for child entries).
    pub fn value(&self) -> B::Value {
        B::Value::read(self.buf, self.off + B::NODE.tile_value)
    }
    /// The child upper node, if any.
    pub fn child(&self) -> Option<NanoUpper<'a, B>> {
        let c = self.child_offset();
        (c != 0).then(|| NanoInternal::at(self.buf, (self.root as i64 + c) as usize))
    }
}

// ------------------------------------------------------------------------------------------
// Internal nodes
// ------------------------------------------------------------------------------------------

/// View of an `InternalData<ChildT, LOG2DIM>`: [`NanoUpper`] (`LOG2DIM` = 5, children are
/// lower nodes) or [`NanoLower`] (`LOG2DIM` = 4, children are leaves).
pub struct NanoInternal<'a, B: BuildType, const LOG2DIM: u32> {
    pub(crate) buf: &'a [u8],
    pub(crate) off: usize,
    _m: PhantomData<B>,
}

/// Upper internal node (32³ children of 128³ voxels).
pub type NanoUpper<'a, B> = NanoInternal<'a, B, 5>;
/// Lower internal node (16³ children of 8³ voxels).
pub type NanoLower<'a, B> = NanoInternal<'a, B, 4>;

impl<'a, B: BuildType, const LOG2DIM: u32> NanoInternal<'a, B, LOG2DIM> {
    /// log2 of the child extent in voxels.
    pub const CHILD_TOTAL: u32 = if LOG2DIM == 5 { 7 } else { 3 };
    /// log2 of the node extent in voxels.
    pub const TOTAL: u32 = LOG2DIM + Self::CHILD_TOTAL;
    /// Node extent in voxels.
    pub const DIM: i32 = 1 << Self::TOTAL;
    /// Number of table entries.
    pub const SIZE: usize = 1 << (3 * LOG2DIM);
    /// Tree level (2 = upper, 1 = lower).
    pub const LEVEL: u32 = if LOG2DIM == 5 { 2 } else { 1 };
    const MASK: i32 = (1 << Self::TOTAL) - 1;

    pub(crate) fn at(buf: &'a [u8], off: usize) -> Self {
        NanoInternal { buf, off, _m: PhantomData }
    }
    #[inline]
    pub(crate) fn layout() -> &'static InternalLayout {
        &B::NODE.internal[if LOG2DIM == 5 { 0 } else { 1 }]
    }
    /// Byte offset of this node relative to the grid.
    pub fn offset(&self) -> usize {
        self.off
    }
    /// Bounding box of the active values (after statistics; the origin before).
    pub fn bbox(&self) -> CoordBBox {
        rd_bbox(self.buf, self.off + internal_off::BBOX)
    }
    /// Node flags: bit 0 = skip during rendering, bit 1 = has active values.
    pub fn flags(&self) -> u64 {
        rd_u64(self.buf, self.off + internal_off::FLAGS)
    }
    /// Origin of the node (`mBBox.min() & ~MASK`). For nodes without active values the
    /// stored bounding box is empty and this is not the node origin; traversals from the
    /// root should use [`child_origin`](Self::child_origin) of the parent instead.
    pub fn origin(&self) -> Coord {
        let m = rd_coord(self.buf, self.off + internal_off::BBOX);
        Coord::new(m[0] & !Self::MASK, m[1] & !Self::MASK, m[2] & !Self::MASK)
    }
    /// The value mask.
    pub fn value_mask(&self) -> MaskView<'a, LOG2DIM> {
        MaskView { buf: self.buf, off: self.off + Self::layout().value_mask }
    }
    /// The child mask.
    pub fn child_mask(&self) -> MaskView<'a, LOG2DIM> {
        MaskView { buf: self.buf, off: self.off + Self::layout().child_mask }
    }
    /// Minimum active value.
    pub fn minimum(&self) -> B::Value {
        B::Value::read(self.buf, self.off + Self::layout().min)
    }
    /// Maximum active value.
    pub fn maximum(&self) -> B::Value {
        B::Value::read(self.buf, self.off + Self::layout().max)
    }
    /// Average of the active values.
    pub fn average(&self) -> B::Stats {
        B::Stats::read(self.buf, self.off + Self::layout().avg)
    }
    /// Standard deviation of the active values.
    pub fn std_deviation(&self) -> B::Stats {
        B::Stats::read(self.buf, self.off + Self::layout().dev)
    }
    /// True if slot `n` holds a child.
    pub fn is_child(&self, n: usize) -> bool {
        self.child_mask().is_on(n)
    }
    /// The tile value of slot `n` (meaningless for child slots).
    pub fn tile_value(&self, n: usize) -> B::Value {
        let l = Self::layout();
        B::Value::read(self.buf, self.off + l.table + n * l.stride)
    }
    /// True if slot `n` is an active tile.
    pub fn is_tile_active(&self, n: usize) -> bool {
        self.value_mask().is_on(n)
    }
    /// Byte offset of the child in slot `n` relative to the grid.
    pub fn child_offset(&self, n: usize) -> usize {
        let l = Self::layout();
        (self.off as i64 + rd_i64(self.buf, self.off + l.table + n * l.stride)) as usize
    }
    /// Slot containing `ijk` (`InternalNode::CoordToOffset`).
    #[inline]
    pub fn coord_to_offset(ijk: Coord) -> usize {
        let m = Self::MASK;
        let t = Self::CHILD_TOTAL;
        ((((ijk[0] & m) >> t) as usize) << (2 * LOG2DIM))
            | ((((ijk[1] & m) >> t) as usize) << LOG2DIM)
            | (((ijk[2] & m) >> t) as usize)
    }
    /// Local coordinates of slot `n`.
    pub fn offset_to_local_coord(n: usize) -> Coord {
        let m = n & ((1 << (2 * LOG2DIM)) - 1);
        Coord::new((n >> (2 * LOG2DIM)) as i32, (m >> LOG2DIM) as i32, (m & ((1 << LOG2DIM) - 1)) as i32)
    }
    /// Origin of the child or tile in slot `n`, given this node's `origin`.
    pub fn child_origin(origin: Coord, n: usize) -> Coord {
        let l = Self::offset_to_local_coord(n);
        Coord::new(
            (l[0] << Self::CHILD_TOTAL) + origin[0],
            (l[1] << Self::CHILD_TOTAL) + origin[1],
            (l[2] << Self::CHILD_TOTAL) + origin[2],
        )
    }
    /// Origin of slot `n` using this node's stored origin (`offsetToGlobalCoord`).
    pub fn offset_to_global_coord(&self, n: usize) -> Coord {
        Self::child_origin(self.origin(), n)
    }
}

impl<'a, B: BuildType> NanoUpper<'a, B> {
    /// The lower node in slot `n`, if any.
    pub fn child(&self, n: usize) -> Option<NanoLower<'a, B>> {
        self.is_child(n).then(|| NanoInternal::at(self.buf, self.child_offset(n)))
    }
    /// Locate the value of `ijk` below this node.
    pub fn locate(&self, ijk: Coord) -> ValueLocation {
        let n = Self::coord_to_offset(ijk);
        match self.child(n) {
            Some(lower) => lower.locate(ijk),
            None => ValueLocation::Upper(self.off, n),
        }
    }
}

impl<'a, B: BuildType> NanoLower<'a, B> {
    /// The leaf in slot `n`, if any.
    pub fn child(&self, n: usize) -> Option<NanoLeaf<'a, B>> {
        self.is_child(n).then(|| NanoLeaf { buf: self.buf, off: self.child_offset(n), _m: PhantomData })
    }
    /// Locate the value of `ijk` below this node.
    pub fn locate(&self, ijk: Coord) -> ValueLocation {
        let n = Self::coord_to_offset(ijk);
        if self.is_child(n) {
            ValueLocation::Leaf(self.child_offset(n), NanoLeaf::<B>::coord_to_offset(ijk))
        } else {
            ValueLocation::Lower(self.off, n)
        }
    }
}

// ------------------------------------------------------------------------------------------
// Leaf
// ------------------------------------------------------------------------------------------

/// View of a `LeafData<BuildT>`: a dense 8³ block of voxels.
pub struct NanoLeaf<'a, B: BuildType> {
    pub(crate) buf: &'a [u8],
    pub(crate) off: usize,
    _m: PhantomData<B>,
}

impl<'a, B: BuildType> NanoLeaf<'a, B> {
    /// Number of voxels.
    pub const SIZE: usize = 512;
    /// Extent in voxels.
    pub const DIM: i32 = 8;

    pub(crate) fn at(buf: &'a [u8], off: usize) -> Self {
        NanoLeaf { buf, off, _m: PhantomData }
    }
    /// Byte offset of this leaf relative to the grid.
    pub fn offset(&self) -> usize {
        self.off
    }
    /// Size of this leaf in bytes.
    pub fn mem_usage(&self) -> usize {
        B::leaf_size(self.buf, self.off)
    }
    /// The bytes of this leaf.
    pub fn bytes(&self) -> &'a [u8] {
        &self.buf[self.off..self.off + self.mem_usage()]
    }
    /// Voxel offset of `ijk` within its leaf.
    #[inline]
    pub fn coord_to_offset(ijk: Coord) -> usize {
        (((ijk[0] & 7) as usize) << 6) | (((ijk[1] & 7) as usize) << 3) | ((ijk[2] & 7) as usize)
    }
    /// Local coordinates of voxel `n`.
    pub fn offset_to_local_coord(n: usize) -> Coord {
        Coord::new((n >> 6) as i32, ((n >> 3) & 7) as i32, (n & 7) as i32)
    }
    /// Global coordinates of voxel `n`.
    pub fn offset_to_global_coord(&self, n: usize) -> Coord {
        self.origin() + Self::offset_to_local_coord(n)
    }
    /// Origin of the leaf.
    pub fn origin(&self) -> Coord {
        let m = rd_coord(self.buf, self.off + leaf_off::BBOX_MIN);
        Coord::new(m[0] & !7, m[1] & !7, m[2] & !7)
    }
    /// Leaf flags: bit 0 = skip rendering, bit 1 = has bbox, bit 4 = has stats (index
    /// grids), bits 5-7 = log2 bit width ([`FpN`](super::FpN)).
    pub fn flags(&self) -> u8 {
        rd_u8(self.buf, self.off + leaf_off::FLAGS)
    }
    /// True if the leaf stores a bounding box of its active voxels.
    pub fn has_bbox(&self) -> bool {
        self.flags() & 2 != 0
    }
    /// Bounding box of the active voxels (empty if none).
    pub fn bbox(&self) -> CoordBBox {
        if !self.has_bbox() {
            return CoordBBox::empty();
        }
        let m = rd_coord(self.buf, self.off + leaf_off::BBOX_MIN);
        let d = |i: usize| self.buf[self.off + leaf_off::BBOX_DIF + i] as i32;
        CoordBBox::new(m, Coord::new(m[0] + d(0), m[1] + d(1), m[2] + d(2)))
    }
    /// The value mask.
    pub fn value_mask(&self) -> MaskView<'a, 3> {
        MaskView { buf: self.buf, off: self.off + leaf_off::VALUE_MASK }
    }
    /// Value of voxel `n`.
    #[inline]
    pub fn get_value_at(&self, n: usize) -> B::Value {
        B::leaf_value(self.buf, self.off, n)
    }
    /// Value at `ijk` (which must lie in this leaf).
    pub fn get_value(&self, ijk: Coord) -> B::Value {
        self.get_value_at(Self::coord_to_offset(ijk))
    }
    /// Active state of voxel `n`.
    pub fn is_active_at(&self, n: usize) -> bool {
        self.value_mask().is_on(n)
    }
    /// Active state at `ijk`.
    pub fn is_active(&self, ijk: Coord) -> bool {
        self.is_active_at(Self::coord_to_offset(ijk))
    }
    /// True if any voxel is active.
    pub fn has_active(&self) -> bool {
        !self.value_mask().is_all_off()
    }
    /// Value and active state at `ijk`.
    pub fn probe_value(&self, ijk: Coord) -> (B::Value, bool) {
        let n = Self::coord_to_offset(ijk);
        (self.get_value_at(n), self.is_active_at(n))
    }
    /// Stored minimum of the active values.
    pub fn minimum(&self) -> B::Value {
        B::leaf_min(self.buf, self.off)
    }
    /// Stored maximum of the active values.
    pub fn maximum(&self) -> B::Value {
        B::leaf_max(self.buf, self.off)
    }
    /// Stored average of the active values.
    pub fn average(&self) -> B::Stats {
        B::leaf_avg(self.buf, self.off)
    }
    /// Stored standard deviation of the active values.
    pub fn std_deviation(&self) -> B::Stats {
        B::leaf_dev(self.buf, self.off)
    }
    /// Iterate over `(global coordinate, value)` of the active voxels.
    pub fn iter_on(&self) -> impl Iterator<Item = (Coord, B::Value)> + 'a {
        let leaf = *self;
        let origin = self.origin();
        self.value_mask().iter_on().map(move |n| (origin + Self::offset_to_local_coord(n), leaf.get_value_at(n)))
    }
}

// ------------------------------------------------------------------------------------------
// ReadAccessor
// ------------------------------------------------------------------------------------------

/// A read accessor that caches the last visited leaf, lower and upper nodes so that
/// spatially coherent lookups skip the top of the tree (`ReadAccessor<BuildT, 0, 1, 2>`).
pub struct ReadAccessor<'a, B: BuildType> {
    root: NanoRoot<'a, B>,
    keys: [Cell<Coord>; 3],
    nodes: [Cell<usize>; 3],
}

const NODE_MASKS: [i32; 3] = [7, (1 << 7) - 1, (1 << 12) - 1];

impl<'a, B: BuildType> ReadAccessor<'a, B> {
    /// Create an accessor for a tree's root.
    pub fn new(root: NanoRoot<'a, B>) -> Self {
        ReadAccessor {
            root,
            keys: [
                Cell::new(Coord::splat(i32::MAX)),
                Cell::new(Coord::splat(i32::MAX)),
                Cell::new(Coord::splat(i32::MAX)),
            ],
            nodes: [Cell::new(0), Cell::new(0), Cell::new(0)],
        }
    }
    /// The root node.
    pub fn root(&self) -> NanoRoot<'a, B> {
        self.root
    }
    /// Forget all cached nodes.
    pub fn clear(&self) {
        for l in 0..3 {
            self.keys[l].set(Coord::splat(i32::MAX));
            self.nodes[l].set(0);
        }
    }
    #[inline]
    fn is_cached(&self, level: usize, ijk: Coord) -> bool {
        let m = !NODE_MASKS[level];
        let k = self.keys[level].get();
        (ijk[0] & m) == k[0] && (ijk[1] & m) == k[1] && (ijk[2] & m) == k[2]
    }
    #[inline]
    fn insert(&self, level: usize, ijk: Coord, off: usize) {
        let m = !NODE_MASKS[level];
        self.keys[level].set(Coord::new(ijk[0] & m, ijk[1] & m, ijk[2] & m));
        self.nodes[level].set(off);
    }
    fn locate_lower(&self, off: usize, ijk: Coord) -> ValueLocation {
        let node = NanoLower::<B>::at(self.root.buf, off);
        let n = NanoLower::<B>::coord_to_offset(ijk);
        if node.is_child(n) {
            let leaf = node.child_offset(n);
            self.insert(0, ijk, leaf);
            ValueLocation::Leaf(leaf, NanoLeaf::<B>::coord_to_offset(ijk))
        } else {
            ValueLocation::Lower(off, n)
        }
    }
    fn locate_upper(&self, off: usize, ijk: Coord) -> ValueLocation {
        let node = NanoUpper::<B>::at(self.root.buf, off);
        let n = NanoUpper::<B>::coord_to_offset(ijk);
        if node.is_child(n) {
            let lower = node.child_offset(n);
            self.insert(1, ijk, lower);
            self.locate_lower(lower, ijk)
        } else {
            ValueLocation::Upper(off, n)
        }
    }
    fn locate_root(&self, ijk: Coord) -> ValueLocation {
        match self.root.probe_tile(ijk) {
            None => ValueLocation::Background,
            Some(t) => match t.child() {
                None => ValueLocation::RootTile(t.off),
                Some(upper) => {
                    self.insert(2, ijk, upper.off);
                    self.locate_upper(upper.off, ijk)
                }
            },
        }
    }
    /// Locate the value of `ijk`, using and updating the node cache.
    pub fn locate(&self, ijk: Coord) -> ValueLocation {
        if self.is_cached(0, ijk) {
            ValueLocation::Leaf(self.nodes[0].get(), NanoLeaf::<B>::coord_to_offset(ijk))
        } else if self.is_cached(1, ijk) {
            self.locate_lower(self.nodes[1].get(), ijk)
        } else if self.is_cached(2, ijk) {
            self.locate_upper(self.nodes[2].get(), ijk)
        } else {
            self.locate_root(ijk)
        }
    }
    /// Value at `ijk`.
    #[inline]
    pub fn get_value(&self, ijk: Coord) -> B::Value {
        self.root.value_at(self.locate(ijk))
    }
    /// Active state at `ijk`.
    #[inline]
    pub fn is_active(&self, ijk: Coord) -> bool {
        self.root.active_at(self.locate(ijk))
    }
    /// Value and active state at `ijk`.
    pub fn probe_value(&self, ijk: Coord) -> (B::Value, bool) {
        let loc = self.locate(ijk);
        (self.root.value_at(loc), self.root.active_at(loc))
    }
    /// The leaf containing `ijk`, if any.
    pub fn probe_leaf(&self, ijk: Coord) -> Option<NanoLeaf<'a, B>> {
        match self.locate(ijk) {
            ValueLocation::Leaf(o, _) => Some(NanoLeaf::at(self.root.buf, o)),
            _ => None,
        }
    }
    /// The cached node at `level` (0 = leaf, 1 = lower, 2 = upper), as a byte offset.
    pub fn cached_node(&self, level: usize) -> Option<usize> {
        let o = self.nodes[level].get();
        (o != 0 && self.keys[level].get()[0] != i32::MAX).then_some(o)
    }
    /// The step size used by hierarchical DDA at `ijk` (`ReadAccessor::getDim`): the extent
    /// of the node or tile containing `ijk`, honoring the "skip" flag (bit 0) that
    /// statistics set on nodes outside a level set's narrow band.
    pub fn get_dim(&self, ijk: Coord) -> u32 {
        if self.is_cached(0, ijk) {
            self.leaf_dim(self.nodes[0].get())
        } else if self.is_cached(1, ijk) {
            self.lower_dim(self.nodes[1].get(), ijk)
        } else if self.is_cached(2, ijk) {
            self.upper_dim(self.nodes[2].get(), ijk)
        } else {
            match self.root.probe_tile(ijk) {
                None => 4096,
                Some(t) => match t.child() {
                    None => 4096,
                    Some(upper) => {
                        self.insert(2, ijk, upper.off);
                        self.upper_dim(upper.off, ijk)
                    }
                },
            }
        }
    }
    fn leaf_dim(&self, off: usize) -> u32 {
        if self.root.buf[off + leaf_off::FLAGS] & 1 != 0 {
            8
        } else {
            1
        }
    }
    fn lower_dim(&self, off: usize, ijk: Coord) -> u32 {
        let node = NanoLower::<B>::at(self.root.buf, off);
        if node.flags() & 1 != 0 {
            return 128;
        }
        let n = NanoLower::<B>::coord_to_offset(ijk);
        if node.is_child(n) {
            let leaf = node.child_offset(n);
            self.insert(0, ijk, leaf);
            self.leaf_dim(leaf)
        } else {
            8
        }
    }
    fn upper_dim(&self, off: usize, ijk: Coord) -> u32 {
        let node = NanoUpper::<B>::at(self.root.buf, off);
        if node.flags() & 1 != 0 {
            return 4096;
        }
        let n = NanoUpper::<B>::coord_to_offset(ijk);
        if node.is_child(n) {
            let lower = node.child_offset(n);
            self.insert(1, ijk, lower);
            self.lower_dim(lower, ijk)
        } else {
            128
        }
    }
}

impl<B: BuildType> Clone for NanoTree<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: BuildType> Copy for NanoTree<'_, B> {}

impl<B: BuildType> Clone for NanoRoot<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: BuildType> Copy for NanoRoot<'_, B> {}

impl<B: BuildType> Clone for RootTile<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: BuildType> Copy for RootTile<'_, B> {}

impl<B: BuildType> Clone for NanoLeaf<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: BuildType> Copy for NanoLeaf<'_, B> {}
impl<B: BuildType, const L: u32> Clone for NanoInternal<'_, B, L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: BuildType, const L: u32> Copy for NanoInternal<'_, B, L> {}
impl<B: BuildType> Clone for ReadAccessor<'_, B> {
    fn clone(&self) -> Self {
        ReadAccessor { root: self.root, keys: self.keys.clone(), nodes: self.nodes.clone() }
    }
}
