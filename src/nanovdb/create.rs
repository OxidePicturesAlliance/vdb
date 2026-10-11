//! Conversion of the crate's OpenVDB-style [`Grid`] into NanoVDB grids
//! (port of `nanovdb/tools/CreateNanoGrid.h` for OpenVDB source grids).
//!
//! The output is byte-for-byte identical to what the C++ `createNanoGrid` produces for the
//! same source tree (nodes are serialized breadth first, in the order of the root table and
//! child masks), apart from bytes the C++ code leaves uninitialized, which are zero here.

use super::build_types::{
    c_cast_u16, c_cast_u32, c_cast_u8, fn_init, fn_off, idx_off, BuildType, DenseLeaf, Fp16, Fp4, Fp8, FpN,
    IndexBuildType, NanoPod, ValueMask,
};
use super::data::{
    empty_bbox, grid_off, internal_off, leaf_off, root_off, tile_off, tree_off, wr_bbox, wr_coord, wr_i64, wr_mask,
    wr_u16, wr_u32, wr_u64, wr_u8, wr_vec3d, Checksum, GridBlindMetaData, Map, Version,
};
use super::dither::DitherLut;
use super::handle::{AlignedBuffer, GridHandle};
use super::stats::{update_grid_stats, StatsMode};
use super::tree::coord_to_key;
use super::{
    grid_flags, CheckMode, GridBlindDataClass, GridBlindDataSemantic, GridClass, GridType, BLIND_META_DATA_SIZE,
    GRID_DATA_SIZE, MAGIC_GRID, MAX_NAME_SIZE, TREE_DATA_SIZE,
};
use crate::grid::{Grid, GridClass as VdbGridClass};
use crate::math::{Vec3d, Vec3f};
use crate::tree::{Internal1, Internal2, LeafNode, Node, RootEntry};
use crate::types::Value;

/// Map the crate's grid class to NanoVDB's (`NodeAccessor<openvdb::Grid>::gridClass`).
fn map_class(c: VdbGridClass) -> GridClass {
    match c {
        VdbGridClass::LevelSet => GridClass::LevelSet,
        VdbGridClass::FogVolume => GridClass::FogVolume,
        VdbGridClass::Staggered => GridClass::Staggered,
        VdbGridClass::Unknown => GridClass::Unknown,
    }
}

/// Build types that can be produced from source values of type `T`.
pub trait BuildFrom<T: Value>: BuildType {
    /// Convert a tile or background value (`static_cast<DstValueT>(value)`).
    fn convert(v: T) -> Self::Value;
    /// Write the leaf payload (values or codes, and quantization parameters) of `src` into
    /// the zero-initialized leaf at `dst[off..]`; the origin and value mask are written by
    /// the caller.
    fn write_leaf(src: &LeafNode<T>, dst: &mut [u8], off: usize, dither: &DitherLut);
}

macro_rules! build_from_same {
    ($($t:ty),*) => {$(
        impl BuildFrom<$t> for $t {
            #[inline]
            fn convert(v: $t) -> $t { v }
            fn write_leaf(src: &LeafNode<$t>, dst: &mut [u8], off: usize, _: &DitherLut) {
                let base = off + <$t as DenseLeaf>::LEAF.values;
                for (i, v) in src.buffer().iter().enumerate() {
                    v.write(dst, base + i * <$t as NanoPod>::SIZE);
                }
            }
        }
    )*};
}
build_from_same!(f32, f64, i32, i64, u32, Vec3f, Vec3d);

impl BuildFrom<bool> for bool {
    fn convert(v: bool) -> bool {
        v
    }
    fn write_leaf(src: &LeafNode<bool>, dst: &mut [u8], off: usize, _: &DitherLut) {
        for (i, &v) in src.buffer().iter().enumerate() {
            if v {
                dst[off + 80 + (i >> 3)] |= 1 << (i & 7);
            }
        }
    }
}

impl<T: Value> BuildFrom<T> for ValueMask {
    fn convert(v: T) -> bool {
        v != T::zero()
    }
    fn write_leaf(_: &LeafNode<T>, _: &mut [u8], _: usize, _: &DitherLut) {}
}

/// Minimum and maximum of all 512 values of a leaf (as computed by the C++ quantizers).
fn leaf_range(src: &LeafNode<f32>) -> (f32, f32) {
    let (mut min, mut max) = (f32::MAX, -f32::MAX);
    for &v in src.buffer() {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

impl BuildFrom<f32> for Fp4 {
    fn convert(v: f32) -> f32 {
        v
    }
    fn write_leaf(src: &LeafNode<f32>, dst: &mut [u8], off: usize, lut: &DitherLut) {
        let (min, max) = leaf_range(src);
        fn_init(dst, off, min, max, 4);
        let encode = 15.0f32 / (max - min);
        let vals = src.buffer();
        let q = |i: usize| c_cast_u8((encode * (vals[i] - min) + lut.get(i)) as f64);
        for j in 0..256 {
            let (a, b) = (q(2 * j), q(2 * j + 1));
            dst[off + fn_off::CODE + j] = (b << 4) | a;
        }
    }
}

impl BuildFrom<f32> for Fp8 {
    fn convert(v: f32) -> f32 {
        v
    }
    fn write_leaf(src: &LeafNode<f32>, dst: &mut [u8], off: usize, lut: &DitherLut) {
        let (min, max) = leaf_range(src);
        fn_init(dst, off, min, max, 8);
        let encode = 255.0f32 / (max - min);
        for (i, &v) in src.buffer().iter().enumerate() {
            dst[off + fn_off::CODE + i] = c_cast_u8((encode * (v - min) + lut.get(i)) as f64);
        }
    }
}

impl BuildFrom<f32> for Fp16 {
    fn convert(v: f32) -> f32 {
        v
    }
    fn write_leaf(src: &LeafNode<f32>, dst: &mut [u8], off: usize, lut: &DitherLut) {
        let (min, max) = leaf_range(src);
        fn_init(dst, off, min, max, 16);
        let encode = 65535.0f64 / (max - min) as f64;
        for (i, &v) in src.buffer().iter().enumerate() {
            let code = c_cast_u16(encode * (v - min) as f64 + lut.get(i) as f64);
            wr_u16(dst, off + fn_off::CODE + 2 * i, code);
        }
    }
}

/// Tile values of [`FpN`] grids are plain floats; leaves are written by
/// [`CreateNanoGrid::get_handle_fpn`] since their size depends on the chosen bit width.
impl BuildFrom<f32> for FpN {
    fn convert(v: f32) -> f32 {
        v
    }
    fn write_leaf(_: &LeafNode<f32>, _: &mut [u8], _: usize, _: &DitherLut) {
        unreachable!("FpN leaves are written by CreateNanoGrid::get_handle_fpn")
    }
}

/// A compression oracle deciding whether a quantized value is acceptable ([`FpN`]).
pub trait Oracle {
    /// True if the tolerance has been set.
    fn is_initialized(&self) -> bool;
    /// Derive a default tolerance from the grid class and background (only [`AbsDiff`]
    /// does this; other oracles keep their tolerance).
    fn init(&mut self, _class: GridClass, _background: f32) {}
    /// True if `approx` is an acceptable approximation of `exact`.
    fn accept(&self, exact: f32, approx: f32) -> bool;
}

/// Oracle based on the absolute error (`nanovdb::tools::AbsDiff`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbsDiff {
    /// Absolute tolerance; negative means uninitialized.
    pub tolerance: f32,
}

impl Default for AbsDiff {
    fn default() -> Self {
        AbsDiff { tolerance: -1.0 }
    }
}

impl AbsDiff {
    /// An oracle with the given tolerance.
    pub fn new(tolerance: f32) -> Self {
        AbsDiff { tolerance }
    }
}

impl Oracle for AbsDiff {
    fn is_initialized(&self) -> bool {
        self.tolerance >= 0.0
    }
    fn init(&mut self, class: GridClass, background: f32) {
        self.tolerance = match class {
            GridClass::LevelSet => 0.1f32 * background / 3.0f32,
            GridClass::FogVolume => 0.01,
            _ => 0.0,
        };
    }
    fn accept(&self, exact: f32, approx: f32) -> bool {
        (exact - approx).abs() <= self.tolerance
    }
}

/// Oracle based on the relative error (`nanovdb::tools::RelDiff`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelDiff {
    /// Relative tolerance; negative means uninitialized.
    pub tolerance: f32,
}

impl Default for RelDiff {
    fn default() -> Self {
        RelDiff { tolerance: -1.0 }
    }
}

impl RelDiff {
    /// An oracle with the given tolerance.
    pub fn new(tolerance: f32) -> Self {
        RelDiff { tolerance }
    }
}

impl Oracle for RelDiff {
    fn is_initialized(&self) -> bool {
        self.tolerance >= 0.0
    }
    fn accept(&self, exact: f32, approx: f32) -> bool {
        (exact - approx).abs() / exact.abs().max(approx.abs()) <= self.tolerance
    }
}

/// A pending blind data block.
struct BlindSpec {
    meta: GridBlindMetaData,
    data: Vec<u8>,
}

/// Offsets of the parts of the output buffer (`CreateNanoGrid::BufferOffsets`).
#[derive(Clone, Copy, Default)]
struct Offsets {
    root: usize,
    upper: usize,
    lower: usize,
    leaf: usize,
    meta: usize,
    blind: usize,
    size: usize,
}

/// Converter from a [`Grid<T>`] to NanoVDB grids of various build types
/// (`nanovdb::tools::CreateNanoGrid`).
pub struct CreateNanoGrid<'g, T: Value> {
    grid: &'g Grid<T>,
    upper: Vec<&'g Internal2<T>>,
    lower: Vec<&'g Internal1<T>>,
    leaves: Vec<&'g LeafNode<T>>,
    stats: StatsMode,
    checksum: CheckMode,
    dither: bool,
    name: Option<String>,
    map: Option<Map>,
    class: Option<GridClass>,
}

impl<'g, T: Value> CreateNanoGrid<'g, T> {
    /// Prepare the conversion of `grid` (collects its nodes breadth first).
    pub fn new(grid: &'g Grid<T>) -> Self {
        let (mut upper, mut lower, mut leaves) = (Vec::new(), Vec::new(), Vec::new());
        for entry in grid.tree().root().table().values() {
            if let RootEntry::Child(u) = entry {
                upper.push(&**u);
                for (_, l) in u.children() {
                    lower.push(l);
                    for (_, leaf) in l.children() {
                        leaves.push(leaf);
                    }
                }
            }
        }
        CreateNanoGrid {
            grid,
            upper,
            lower,
            leaves,
            stats: StatsMode::default(),
            checksum: CheckMode::default(),
            dither: false,
            name: None,
            map: None,
            class: None,
        }
    }
    /// Set the statistics mode (default [`StatsMode::All`]).
    pub fn stats(mut self, mode: StatsMode) -> Self {
        self.stats = mode;
        self
    }
    /// Set the checksum mode (default [`CheckMode::Partial`]).
    pub fn checksum(mut self, mode: CheckMode) -> Self {
        self.checksum = mode;
        self
    }
    /// Enable or disable dithering of quantization errors (default off).
    pub fn dither(mut self, on: bool) -> Self {
        self.dither = on;
        self
    }
    /// Override the grid name (defaults to the source grid's name).
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    /// Override the index-to-world map (defaults to the source transform's affine map).
    pub fn map(mut self, map: Map) -> Self {
        self.map = Some(map);
        self
    }
    /// Override the source grid class.
    pub fn grid_class(mut self, class: GridClass) -> Self {
        self.class = Some(class);
        self
    }

    fn src_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.grid.name().to_string())
    }
    fn src_class(&self) -> GridClass {
        self.class.unwrap_or_else(|| map_class(self.grid.grid_class()))
    }
    fn src_map(&self) -> Map {
        self.map.unwrap_or_else(|| Map::from_transform(self.grid.transform()))
    }
    fn long_name_blind(&self) -> Option<BlindSpec> {
        let name = self.src_name();
        (name.len() >= MAX_NAME_SIZE).then(|| {
            let mut data = name.clone().into_bytes();
            data.push(0);
            BlindSpec {
                meta: GridBlindMetaData {
                    data_offset: 0,
                    value_count: data.len() as u64,
                    value_size: 1,
                    semantic: GridBlindDataSemantic::Unknown,
                    data_class: GridBlindDataClass::GridName,
                    data_type: GridType::Unknown,
                    name: "grid name".into(),
                },
                data,
            }
        })
    }

    fn offsets<B: BuildType>(&self, leaf_bytes: usize, blind: &[BlindSpec]) -> Offsets {
        let n = B::NODE;
        let root = GRID_DATA_SIZE + TREE_DATA_SIZE;
        let upper = root + n.root_size + self.grid.tree().root().table().len() * n.tile_size;
        let lower = upper + n.internal[0].size * self.upper.len();
        let leaf = lower + n.internal[1].size * self.lower.len();
        let meta = leaf + leaf_bytes;
        let blind_start = meta + BLIND_META_DATA_SIZE * blind.len();
        let size = blind_start + blind.iter().map(|b| b.meta.blind_data_size() as usize).sum::<usize>();
        Offsets { root, upper, lower, leaf, meta, blind: blind_start, size }
    }

    /// Write the common leaf header (origin and value mask) of leaf `i`.
    fn write_leaf_header(&self, buf: &mut [u8], off: usize, i: usize) {
        let src = self.leaves[i];
        wr_coord(buf, off + leaf_off::BBOX_MIN, src.origin());
        wr_mask(buf, off + leaf_off::VALUE_MASK, src.value_mask());
    }

    /// Serialize internal nodes; `tile` produces the bytes of a tile value for slot `n`.
    fn write_internal<B: BuildType>(
        &self,
        buf: &mut [u8],
        o: &Offsets,
        leaf_offsets: &dyn Fn(usize) -> usize,
        mut tile_upper: impl FnMut(&mut [u8], usize, usize, usize, T, bool),
        mut tile_lower: impl FnMut(&mut [u8], usize, usize, usize, T, bool),
    ) {
        let n = B::NODE;
        let (lu, ll) = (n.internal[0], n.internal[1]);
        let mut child_id = 0usize;
        for (i, src) in self.upper.iter().enumerate() {
            let off = o.upper + i * lu.size;
            wr_coord(buf, off + internal_off::BBOX, src.origin());
            wr_mask(buf, off + lu.value_mask, src.value_mask());
            wr_mask(buf, off + lu.child_mask, src.child_mask());
            for s in 0..Internal2::<T>::NUM_VALUES {
                let entry = off + lu.table + s * lu.stride;
                if src.is_child(s) {
                    let child = o.lower + child_id * ll.size;
                    child_id += 1;
                    wr_i64(buf, entry, child as i64 - off as i64);
                } else {
                    tile_upper(buf, i, s, entry, src.values()[s], src.value_mask().is_on(s));
                }
            }
        }
        child_id = 0;
        for (i, src) in self.lower.iter().enumerate() {
            let off = o.lower + i * ll.size;
            wr_coord(buf, off + internal_off::BBOX, src.origin());
            wr_mask(buf, off + ll.value_mask, src.value_mask());
            wr_mask(buf, off + ll.child_mask, src.child_mask());
            for s in 0..Internal1::<T>::NUM_VALUES {
                let entry = off + ll.table + s * ll.stride;
                if src.is_child(s) {
                    let child = leaf_offsets(child_id);
                    child_id += 1;
                    wr_i64(buf, entry, child as i64 - off as i64);
                } else {
                    tile_lower(buf, i, s, entry, src.values()[s], src.value_mask().is_on(s));
                }
            }
        }
    }

    /// Serialize the root; `tile` writes a value tile (`buf`, tile offset, value, active).
    fn write_root<B: BuildType>(&self, buf: &mut [u8], o: &Offsets, mut tile: impl FnMut(&mut [u8], usize, T, bool)) {
        let n = B::NODE;
        let root = self.grid.tree().root();
        wr_u32(buf, o.root + root_off::TABLE_SIZE, root.table().len() as u32);
        wr_bbox(buf, o.root + root_off::BBOX, &empty_bbox());
        let mut child_id = 0usize;
        for (k, (origin, entry)) in root.table().iter().enumerate() {
            let t = o.root + n.root_size + k * n.tile_size;
            wr_u64(buf, t + tile_off::KEY, coord_to_key(*origin));
            match entry {
                RootEntry::Child(_) => {
                    let child = o.upper + child_id * n.internal[0].size;
                    child_id += 1;
                    wr_i64(buf, t + tile_off::CHILD, child as i64 - o.root as i64);
                    wr_u32(buf, t + tile_off::STATE, 0);
                }
                RootEntry::Tile { value, active } => {
                    wr_u32(buf, t + tile_off::STATE, *active as u32);
                    tile(buf, t, *value, *active);
                }
            }
        }
    }

    /// Fill `TreeData` (`processTree`).
    fn write_tree(&self, buf: &mut [u8], o: &Offsets) {
        let t = GRID_DATA_SIZE;
        let rel = |x: usize, count: usize| if count > 0 { (x - t) as i64 } else { 0 };
        wr_i64(buf, t + tree_off::NODE_OFFSET, rel(o.leaf, self.leaves.len()));
        wr_i64(buf, t + tree_off::NODE_OFFSET + 8, rel(o.lower, self.lower.len()));
        wr_i64(buf, t + tree_off::NODE_OFFSET + 16, rel(o.upper, self.upper.len()));
        wr_i64(buf, t + tree_off::NODE_OFFSET + 24, (o.root - t) as i64);
        wr_u32(buf, t + tree_off::NODE_COUNT, self.leaves.len() as u32);
        wr_u32(buf, t + tree_off::NODE_COUNT + 4, self.lower.len() as u32);
        wr_u32(buf, t + tree_off::NODE_COUNT + 8, self.upper.len() as u32);
        let tc0: u32 = self.lower.iter().map(|n| n.value_mask().count_on()).sum();
        let tc1: u32 = self.upper.iter().map(|n| n.value_mask().count_on()).sum();
        let tc2 = self
            .grid
            .tree()
            .root()
            .table()
            .values()
            .filter(|e| matches!(e, RootEntry::Tile { active: true, .. }))
            .count() as u32;
        wr_u32(buf, t + tree_off::TILE_COUNT, tc0);
        wr_u32(buf, t + tree_off::TILE_COUNT + 4, tc1);
        wr_u32(buf, t + tree_off::TILE_COUNT + 8, tc2);
        let mut voxels: u64 = self.leaves.iter().map(|l| l.value_mask().count_on() as u64).sum();
        voxels += (tc0 as u64) << 9;
        voxels += (tc1 as u64) << 21;
        voxels += (tc2 as u64) << 36;
        wr_u64(buf, t + tree_off::VOXEL_COUNT, voxels);
    }

    /// Fill `GridData` and the blind meta data and data (`processGrid`).
    fn write_grid<B: BuildType>(&self, buf: &mut [u8], o: &Offsets, value_count: u64, blind: &[BlindSpec]) {
        let map = self.src_map();
        wr_u64(buf, grid_off::MAGIC, MAGIC_GRID);
        wr_u64(buf, grid_off::CHECKSUM, Checksum::EMPTY.0);
        wr_u32(buf, grid_off::VERSION, Version::default().id());
        let name = self.src_name();
        let mut flags = grid_flags::IS_BREADTH_FIRST;
        if name.len() >= MAX_NAME_SIZE {
            flags |= grid_flags::HAS_LONG_GRID_NAME;
        }
        wr_u32(buf, grid_off::FLAGS, flags);
        wr_u32(buf, grid_off::GRID_INDEX, 0);
        wr_u32(buf, grid_off::GRID_COUNT, 1);
        wr_u64(buf, grid_off::GRID_SIZE, o.size as u64);
        let nb = name.as_bytes();
        let len = nb.len().min(MAX_NAME_SIZE - 1);
        buf[grid_off::GRID_NAME..grid_off::GRID_NAME + len].copy_from_slice(&nb[..len]);
        map.write(buf, grid_off::MAP);
        wr_vec3d(buf, grid_off::WORLD_BBOX, Vec3d::splat(f64::MAX));
        wr_vec3d(buf, grid_off::WORLD_BBOX + 24, Vec3d::splat(-f64::MAX));
        wr_vec3d(buf, grid_off::VOXEL_SIZE, map.voxel_size());
        let class = B::FORCED_CLASS.unwrap_or_else(|| self.src_class());
        wr_u32(buf, grid_off::GRID_CLASS, class.to_u32());
        wr_u32(buf, grid_off::GRID_TYPE, B::GRID_TYPE.to_u32());
        let blind_off = if blind.is_empty() { o.size } else { o.meta };
        wr_i64(buf, grid_off::BLIND_META_OFFSET, blind_off as i64);
        wr_u32(buf, grid_off::BLIND_META_COUNT, blind.len() as u32);
        wr_u32(buf, grid_off::DATA0, 0);
        wr_u64(buf, grid_off::DATA1, value_count);
        wr_u64(buf, grid_off::DATA2, 0);
        let mut data_pos = o.blind;
        for (i, b) in blind.iter().enumerate() {
            let mo = o.meta + i * BLIND_META_DATA_SIZE;
            let mut meta = b.meta.clone();
            meta.data_offset = data_pos as i64 - mo as i64;
            meta.write(buf, mo);
            buf[data_pos..data_pos + b.data.len()].copy_from_slice(&b.data);
            data_pos += b.meta.blind_data_size() as usize;
        }
    }

    fn finish<B: BuildType>(&self, mut buffer: AlignedBuffer, stats: StatsMode) -> GridHandle {
        update_grid_stats::<B>(buffer.as_mut_slice(), stats);
        super::checksum::update_checksum(buffer.as_mut_slice(), self.checksum);
        GridHandle::new(buffer).expect("CreateNanoGrid produced an invalid grid")
    }

    /// Convert to a grid with build type `B` (any value type, `bool`, [`ValueMask`] or the
    /// fixed-rate quantized [`Fp4`]/[`Fp8`]/[`Fp16`]).
    pub fn get_handle<B: BuildFrom<T>>(&self) -> GridHandle {
        assert!(B::LEAF_SIZE > 0, "use get_handle_fpn for variable-size leaves");
        let blind: Vec<BlindSpec> = self.long_name_blind().into_iter().collect();
        let o = self.offsets::<B>(self.leaves.len() * B::LEAF_SIZE, &blind);
        let mut buffer = AlignedBuffer::zeroed(o.size);
        let buf = buffer.as_mut_slice();
        let lut = DitherLut::new(self.dither);
        for (i, src) in self.leaves.iter().enumerate() {
            let off = o.leaf + i * B::LEAF_SIZE;
            self.write_leaf_header(buf, off, i);
            B::write_leaf(src, buf, off, &lut);
        }
        let leaf_off = |i: usize| o.leaf + i * B::LEAF_SIZE;
        let tile = |buf: &mut [u8], _: usize, _: usize, entry: usize, v: T, _: bool| B::convert(v).write(buf, entry);
        self.write_internal::<B>(buf, &o, &leaf_off, tile, tile);
        self.write_root_values::<B>(buf, &o);
        self.write_tree(buf, &o);
        self.write_grid::<B>(buf, &o, 0, &blind);
        self.finish::<B>(buffer, self.stats)
    }

    fn write_root_values<B: BuildFrom<T>>(&self, buf: &mut [u8], o: &Offsets) {
        let n = B::NODE;
        let bg = B::convert(*self.grid.background());
        bg.write(buf, o.root + n.root_background);
        bg.write(buf, o.root + n.root_min);
        bg.write(buf, o.root + n.root_max);
        self.write_root::<B>(buf, o, |buf, t, v, _| B::convert(v).write(buf, t + n.tile_value));
    }

    /// Convert to an index grid ([`ValueIndex`](super::ValueIndex) or
    /// [`ValueOnIndex`](super::ValueOnIndex)). `channels` copies of the indexed values are
    /// appended as blind data; `include_stats` reserves 4 statistics slots per node and
    /// `include_tiles` also indexes tile values.
    pub fn get_index_handle<B: IndexBuildType>(
        &self,
        channels: u32,
        include_stats: bool,
        include_tiles: bool,
    ) -> GridHandle
    where
        T: NanoPod + super::grid::BlindType,
    {
        let on = B::ON_INDEX;
        let stats = if include_stats { 4u64 } else { 0 };
        let root = self.grid.tree().root();
        // countValues
        let mut value_count = 1u64;
        let mut idx2 = vec![0u64; self.upper.len() + 1];
        let mut idx1 = vec![0u64; self.lower.len() + 1];
        if include_tiles {
            value_count += root
                .table()
                .values()
                .filter(|e| match e {
                    RootEntry::Tile { active, .. } => !on || *active,
                    RootEntry::Child(_) => false,
                })
                .count() as u64;
            value_count += stats;
            idx2[0] = value_count;
            for (i, n) in self.upper.iter().enumerate() {
                let c = if on { n.value_mask().count_on() as u64 } else { 32768 - n.child_mask().count_on() as u64 };
                idx2[i + 1] = idx2[i] + stats + c;
            }
            value_count = *idx2.last().unwrap();
            idx1[0] = value_count;
            for (i, n) in self.lower.iter().enumerate() {
                let c = if on { n.value_mask().count_on() as u64 } else { 4096 - n.child_mask().count_on() as u64 };
                idx1[i + 1] = idx1[i] + stats + c;
            }
            value_count = *idx1.last().unwrap();
        }
        let mut idx0 = vec![value_count; self.leaves.len() + 1];
        for (i, l) in self.leaves.iter().enumerate() {
            let c = if on { l.value_mask().count_on() as u64 } else { 512 };
            idx0[i + 1] = idx0[i] + stats + c;
        }
        let value_count = *idx0.last().unwrap();

        // Blind data: channels, then the long grid name.
        let mut blind = Vec::new();
        if channels > 0 {
            let values = self.index_values(on, include_stats, include_tiles, value_count, &idx2, &idx1, &idx0);
            let mut data = vec![0u8; values.len() * T::SIZE];
            for (i, v) in values.iter().enumerate() {
                v.write(&mut data, i * T::SIZE);
            }
            for c in 0..channels {
                blind.push(BlindSpec {
                    meta: GridBlindMetaData {
                        data_offset: 0,
                        value_count,
                        value_size: T::SIZE as u32,
                        semantic: GridBlindDataSemantic::from_grid_class(self.src_class()),
                        data_class: GridBlindDataClass::ChannelArray,
                        data_type: T::BLIND_TYPE,
                        name: format!("channel_{}", c),
                    },
                    data: data.clone(),
                });
            }
        }
        blind.extend(self.long_name_blind());

        let o = self.offsets::<B>(self.leaves.len() * B::LEAF_SIZE, &blind);
        let mut buffer = AlignedBuffer::zeroed(o.size);
        let buf = buffer.as_mut_slice();
        // Leaves.
        for (i, src) in self.leaves.iter().enumerate() {
            let off = o.leaf + i * B::LEAF_SIZE;
            self.write_leaf_header(buf, off, i);
            wr_u8(buf, off + leaf_off::FLAGS, if include_stats { 16 } else { 0 });
            wr_u64(buf, off + idx_off::OFFSET, idx0[i]);
            if on {
                let w = src.value_mask().words();
                let mut sum = w[0].count_ones() as u64;
                let mut prefix = sum;
                for (k, n) in (9..55).step_by(9).enumerate() {
                    sum += w[k + 1].count_ones() as u64;
                    prefix |= sum << n;
                }
                wr_u64(buf, off + idx_off::PREFIX_SUM, prefix);
            }
        }
        // Internal nodes: tile indices, then statistics indices.
        let leaf_off = |i: usize| o.leaf + i * B::LEAF_SIZE;
        let counters = std::cell::RefCell::new((usize::MAX, 0u64, usize::MAX, 0u64));
        let n = B::NODE;
        let (lu, ll) = (n.internal[0], n.internal[1]);
        {
            let tile_upper = |buf: &mut [u8], i: usize, _: usize, entry: usize, _: T, active: bool| {
                let mut c = counters.borrow_mut();
                if c.0 != i {
                    *c = (i, idx2[i], c.2, c.3);
                }
                let mut m = 0;
                if include_tiles && !(on && !active) {
                    m = c.1;
                    c.1 += 1;
                }
                wr_u64(buf, entry, m);
            };
            let tile_lower = |buf: &mut [u8], i: usize, _: usize, entry: usize, _: T, active: bool| {
                let mut c = counters.borrow_mut();
                if c.2 != i {
                    *c = (c.0, c.1, i, idx1[i]);
                }
                let mut m = 0;
                if include_tiles && !(on && !active) {
                    m = c.3;
                    c.3 += 1;
                }
                wr_u64(buf, entry, m);
            };
            self.write_internal::<B>(buf, &o, &leaf_off, tile_upper, tile_lower);
        }
        let node_stats = |buf: &mut [u8], off: usize, l: &super::data::InternalLayout, start: u64, tiles: u64| {
            let (a, b, c, d) = if include_tiles && include_stats {
                let s = start + tiles;
                (s, s + 1, s + 2, s + 3)
            } else {
                (0, 0, 0, 0)
            };
            wr_u64(buf, off + l.min, a);
            wr_u64(buf, off + l.max, b);
            wr_u64(buf, off + l.avg, c);
            wr_u64(buf, off + l.dev, d);
        };
        for (i, src) in self.upper.iter().enumerate() {
            let t = if on { src.value_mask().count_on() as u64 } else { 32768 - src.child_mask().count_on() as u64 };
            node_stats(buf, o.upper + i * lu.size, &lu, idx2[i], t);
        }
        for (i, src) in self.lower.iter().enumerate() {
            let t = if on { src.value_mask().count_on() as u64 } else { 4096 - src.child_mask().count_on() as u64 };
            node_stats(buf, o.lower + i * ll.size, &ll, idx1[i], t);
        }
        // Root.
        let mut count = 0u64;
        self.write_root::<B>(buf, &o, |buf, t, _, active| {
            let mut v = 0u64;
            if include_tiles && !(on && !active) {
                count += 1;
                v = count;
            }
            wr_u64(buf, t + n.tile_value, v);
        });
        wr_u64(buf, o.root + n.root_background, 0);
        if include_tiles && include_stats {
            for (k, f) in [n.root_min, n.root_max, n.root_avg, n.root_dev].iter().enumerate() {
                wr_u64(buf, o.root + f, count + 1 + k as u64);
            }
        }
        self.write_tree(buf, &o);
        self.write_grid::<B>(buf, &o, value_count, &blind);
        self.finish::<B>(buffer, self.stats.min(StatsMode::BBox))
    }

    /// The values indexed by an index grid, in index order (`CreateNanoGrid::copyValues`).
    #[allow(clippy::too_many_arguments)]
    fn index_values(
        &self,
        on: bool,
        include_stats: bool,
        include_tiles: bool,
        value_count: u64,
        idx2: &[u64],
        idx1: &[u64],
        idx0: &[u64],
    ) -> Vec<T> {
        let bg = *self.grid.background();
        let mut out = vec![T::zero(); value_count as usize];
        out[0] = bg;
        let put_stats = |out: &mut Vec<T>, mut p: usize| {
            if include_stats {
                for _ in 0..4 {
                    out[p] = bg;
                    p += 1;
                }
            }
        };
        if include_tiles {
            let mut p = 1usize;
            for e in self.grid.tree().root().table().values() {
                if let RootEntry::Tile { value, active } = e {
                    if !on || *active {
                        out[p] = *value;
                        p += 1;
                    }
                }
            }
            put_stats(&mut out, p);
            for (i, n) in self.upper.iter().enumerate() {
                let mut p = idx2[i] as usize;
                for s in 0..32768 {
                    if (on && n.value_mask().is_on(s)) || (!on && !n.is_child(s)) {
                        out[p] = n.values()[s];
                        p += 1;
                    }
                }
                put_stats(&mut out, p);
            }
            for (i, n) in self.lower.iter().enumerate() {
                let mut p = idx1[i] as usize;
                for s in 0..4096 {
                    if (on && n.value_mask().is_on(s)) || (!on && !n.is_child(s)) {
                        out[p] = n.values()[s];
                        p += 1;
                    }
                }
                put_stats(&mut out, p);
            }
        }
        for (i, l) in self.leaves.iter().enumerate() {
            let mut p = idx0[i] as usize;
            for (s, v) in l.buffer().iter().enumerate() {
                if !on || l.value_mask().is_on(s) {
                    out[p] = *v;
                    p += 1;
                }
            }
            put_stats(&mut out, p);
        }
        out
    }
}

impl<'g> CreateNanoGrid<'g, f32> {
    /// Convert to an [`FpN`] grid, choosing the bit width of each leaf with `oracle`
    /// (`createNanoGrid<SrcGridT, FpN>`). An uninitialized [`AbsDiff`] derives its tolerance
    /// from the grid class and background.
    pub fn get_handle_fpn<O: Oracle>(&self, mut oracle: O) -> GridHandle {
        let lut = DitherLut::new(self.dither);
        if !oracle.is_initialized() {
            oracle.init(self.src_class(), *self.grid.background());
        }
        // Choose the bit width of every leaf.
        struct Codec {
            min: f32,
            max: f32,
            log2: u32,
        }
        let codecs: Vec<Codec> = self
            .leaves
            .iter()
            .map(|leaf| {
                let vals: Vec<f32> = leaf.buffer().to_vec();
                let (mut min, mut max) = (f32::MAX, -f32::MAX);
                for &v in &vals {
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
                let range = max - min;
                let mut log2 = 0u32;
                while range > 0.0 && log2 < 4 {
                    let mask = ((1u32 << (1u32 << log2)) - 1) as f32;
                    let (encode, decode) = (mask / range, range / mask);
                    let ok = (0..512).all(|j| {
                        let exact = vals[j];
                        let code = c_cast_u32((encode * (exact - min) + lut.get(j)) as f64);
                        let approx = code as f32 * decode + min;
                        oracle.accept(exact, approx)
                    });
                    if ok {
                        break;
                    }
                    log2 += 1;
                }
                Codec { min, max, log2 }
            })
            .collect();
        let blind: Vec<BlindSpec> = self.long_name_blind().into_iter().collect();
        let sizes: Vec<usize> = codecs.iter().map(|c| 96 + (1usize << c.log2) * 64).collect();
        // C++ 32.9.2 computes the size of the leaf section as the absolute end offset of the
        // last leaf (`mLeafNodeSize = getOffset(leafCount)`), so FpN grids carry an unused
        // (here zero-filled) region of the size of the headers and internal nodes before
        // their blind data. This is reproduced so that buffers stay byte-for-byte identical.
        let leaf_start = self.offsets::<FpN>(0, &[]).leaf;
        let leaf_bytes: usize = sizes.iter().sum();
        let o = self.offsets::<FpN>(if sizes.is_empty() { 0 } else { leaf_start + leaf_bytes }, &blind);
        let mut starts = Vec::with_capacity(sizes.len());
        let mut pos = o.leaf;
        for s in &sizes {
            starts.push(pos);
            pos += s;
        }
        let mut buffer = AlignedBuffer::zeroed(o.size);
        let buf = buffer.as_mut_slice();
        for (i, src) in self.leaves.iter().enumerate() {
            let off = starts[i];
            let c = &codecs[i];
            self.write_leaf_header(buf, off, i);
            wr_u8(buf, off + leaf_off::FLAGS, (c.log2 << 5) as u8);
            fn_init(buf, off, c.min, c.max, 1 << c.log2);
            let (min, max) = (c.min, c.max);
            let vals: Vec<f32> = src.buffer().to_vec();
            let dst = off + fn_off::CODE;
            let q8 = |encode: f32, i: usize| c_cast_u8((encode * (vals[i] - min) + lut.get(i)) as f64);
            match c.log2 {
                0 => {
                    let e = 1.0f32 / (max - min);
                    for j in 0..64 {
                        let mut a = 0u8;
                        for k in 0..8 {
                            a |= q8(e, 8 * j + k) << k;
                        }
                        buf[dst + j] = a;
                    }
                }
                1 => {
                    let e = 3.0f32 / (max - min);
                    for j in 0..128 {
                        let a = q8(e, 4 * j) | (q8(e, 4 * j + 1) << 2) | (q8(e, 4 * j + 2) << 4);
                        buf[dst + j] = (q8(e, 4 * j + 3) << 6) | a;
                    }
                }
                2 => {
                    let e = 15.0f32 / (max - min);
                    for j in 0..256 {
                        buf[dst + j] = (q8(e, 2 * j + 1) << 4) | q8(e, 2 * j);
                    }
                }
                3 => {
                    let e = 255.0f32 / (max - min);
                    for j in 0..512 {
                        buf[dst + j] = q8(e, j);
                    }
                }
                _ => {
                    let e = 65535.0f64 / (max - min) as f64;
                    for (j, &v) in vals.iter().enumerate() {
                        wr_u16(buf, dst + 2 * j, c_cast_u16(e * (v - min) as f64 + lut.get(j) as f64));
                    }
                }
            }
        }
        let leaf_off = |i: usize| starts[i];
        let tile =
            |buf: &mut [u8], _: usize, _: usize, entry: usize, v: f32, _: bool| FpN::convert(v).write(buf, entry);
        self.write_internal::<FpN>(buf, &o, &leaf_off, tile, tile);
        self.write_root_values::<FpN>(buf, &o);
        self.write_tree(buf, &o);
        self.write_grid::<FpN>(buf, &o, 0, &blind);
        self.finish::<FpN>(buffer, self.stats)
    }
}

/// Convert `grid` to a NanoVDB grid with build type `B`, default statistics
/// ([`StatsMode::All`]) and a partial checksum (`nanovdb::tools::createNanoGrid`).
///
/// For [`FpN`] use [`CreateNanoGrid::get_handle_fpn`]; for index grids
/// [`CreateNanoGrid::get_index_handle`].
pub fn create_nano_grid<T: Value, B: BuildFrom<T>>(grid: &Grid<T>) -> GridHandle {
    CreateNanoGrid::new(grid).get_handle::<B>()
}
