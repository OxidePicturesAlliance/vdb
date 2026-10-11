//! Views of `GridData` (untyped [`GridHeader`]) and typed grids ([`NanoGrid`]).

use super::build_types::{BuildType, NanoPod};
use super::data::{
    grid_off, rd_bbox, rd_i64, rd_u32, rd_u64, rd_vec3d, root_off, tree_off, Checksum, GridBlindMetaData, Map, Version,
};
use super::tree::{NanoTree, ReadAccessor};
use super::{
    grid_flags, to_magic, GridBlindDataClass, GridClass, GridType, MagicType, NanoError, Result, BLIND_META_DATA_SIZE,
    GRID_DATA_SIZE, MAX_NAME_SIZE,
};
use crate::math::{CoordBBox, Vec3d};
use std::marker::PhantomData;

/// Untyped view of the header (`GridData`) of a grid, plus a few tree-level counters.
///
/// The slice starts at the grid and normally spans the whole grid (`grid_size()` bytes);
/// accessors that need data beyond the 672-byte header return defaults when it is shorter.
#[derive(Clone, Copy)]
pub struct GridHeader<'a> {
    pub(crate) buf: &'a [u8],
}

impl<'a> GridHeader<'a> {
    /// Wrap the bytes of a grid.
    pub fn new(buf: &'a [u8]) -> Self {
        GridHeader { buf }
    }
    /// The bytes of the grid.
    pub fn bytes(&self) -> &'a [u8] {
        self.buf
    }
    /// The magic number.
    pub fn magic(&self) -> u64 {
        rd_u64(self.buf, grid_off::MAGIC)
    }
    /// The checksum.
    pub fn checksum(&self) -> Checksum {
        Checksum(rd_u64(self.buf, grid_off::CHECKSUM))
    }
    /// The version the grid was written with.
    pub fn version(&self) -> Version {
        Version(rd_u32(self.buf, grid_off::VERSION))
    }
    /// The raw grid flags (see [`grid_flags`]).
    pub fn flags(&self) -> u32 {
        rd_u32(self.buf, grid_off::FLAGS)
    }
    /// Index of this grid in its buffer.
    pub fn grid_index(&self) -> u32 {
        rd_u32(self.buf, grid_off::GRID_INDEX)
    }
    /// Number of grids in the buffer this grid belongs to.
    pub fn grid_count(&self) -> u32 {
        rd_u32(self.buf, grid_off::GRID_COUNT)
    }
    /// Size of this grid in bytes (including blind data).
    pub fn grid_size(&self) -> u64 {
        rd_u64(self.buf, grid_off::GRID_SIZE)
    }
    /// The grid name stored in the header (at most 255 bytes).
    pub fn short_grid_name(&self) -> String {
        let n = &self.buf[grid_off::GRID_NAME..grid_off::GRID_NAME + MAX_NAME_SIZE];
        let end = n.iter().position(|&c| c == 0).unwrap_or(n.len());
        String::from_utf8_lossy(&n[..end]).into_owned()
    }
    /// The full grid name; long names are read from their blind data.
    pub fn grid_name(&self) -> String {
        if self.has_long_grid_name() {
            for i in 0..self.blind_data_count() {
                let m = self.blind_meta_data(i);
                if m.data_class == GridBlindDataClass::GridName {
                    let data = self.blind_data(i);
                    let end = data.iter().position(|&c| c == 0).unwrap_or(data.len());
                    return String::from_utf8_lossy(&data[..end]).into_owned();
                }
            }
        }
        self.short_grid_name()
    }
    /// The index-to-world map.
    pub fn map(&self) -> Map {
        Map::read(self.buf, grid_off::MAP)
    }
    /// World-space bounds of the active values (min inclusive, max exclusive).
    pub fn world_bbox(&self) -> (Vec3d, Vec3d) {
        (rd_vec3d(self.buf, grid_off::WORLD_BBOX), rd_vec3d(self.buf, grid_off::WORLD_BBOX + 24))
    }
    /// Size of a voxel in world units.
    pub fn voxel_size(&self) -> Vec3d {
        rd_vec3d(self.buf, grid_off::VOXEL_SIZE)
    }
    /// The grid class.
    pub fn grid_class(&self) -> GridClass {
        GridClass::from_u32(rd_u32(self.buf, grid_off::GRID_CLASS))
    }
    /// The value type.
    pub fn grid_type(&self) -> GridType {
        GridType::from_u32(rd_u32(self.buf, grid_off::GRID_TYPE))
    }
    /// Byte offset of the first blind meta data, relative to the grid.
    pub fn blind_meta_data_offset(&self) -> i64 {
        rd_i64(self.buf, grid_off::BLIND_META_OFFSET)
    }
    /// Number of blind data blocks.
    pub fn blind_data_count(&self) -> u32 {
        rd_u32(self.buf, grid_off::BLIND_META_COUNT)
    }
    /// `mData1`: the number of indexed values of an index grid.
    pub fn data1(&self) -> u64 {
        rd_u64(self.buf, grid_off::DATA1)
    }
    /// Number of values indexed by an index grid (`Grid::valueCount`).
    pub fn value_count(&self) -> u64 {
        self.data1()
    }
    /// True if the magic number and version are valid (`GridData::isValid`).
    pub fn is_valid(&self) -> bool {
        if self.buf.len() < GRID_DATA_SIZE {
            return false;
        }
        let magic = to_magic(self.magic());
        if magic == MagicType::NanoGrid || to_magic(rd_u64(self.buf, grid_off::DATA2)) == MagicType::NanoGrid {
            return true;
        }
        magic == MagicType::NanoVDB
            && self.version().is_compatible()
            && self.grid_count() > 0
            && self.grid_index() < self.grid_count()
            && self.grid_class() < GridClass::End
            && self.grid_type() < GridType::End
    }
    /// The grid has bounding boxes.
    pub fn has_bbox(&self) -> bool {
        self.flags() & grid_flags::HAS_BBOX != 0
    }
    /// Nodes store min/max values.
    pub fn has_min_max(&self) -> bool {
        self.flags() & grid_flags::HAS_MIN_MAX != 0
    }
    /// Nodes store averages.
    pub fn has_average(&self) -> bool {
        self.flags() & grid_flags::HAS_AVERAGE != 0
    }
    /// Nodes store standard deviations.
    pub fn has_std_deviation(&self) -> bool {
        self.flags() & grid_flags::HAS_STD_DEVIATION != 0
    }
    /// The name is stored as blind data.
    pub fn has_long_grid_name(&self) -> bool {
        self.flags() & grid_flags::HAS_LONG_GRID_NAME != 0
    }
    /// Nodes are stored breadth first.
    pub fn is_breadth_first(&self) -> bool {
        self.flags() & grid_flags::IS_BREADTH_FIRST != 0
    }
    /// True if this is a level set.
    pub fn is_level_set(&self) -> bool {
        self.grid_class() == GridClass::LevelSet
    }
    /// True if this is a fog volume.
    pub fn is_fog_volume(&self) -> bool {
        self.grid_class() == GridClass::FogVolume
    }
    /// Byte offset of the root node relative to the grid.
    pub(crate) fn root_offset(&self) -> usize {
        GRID_DATA_SIZE + rd_i64(self.buf, GRID_DATA_SIZE + tree_off::NODE_OFFSET + 24) as usize
    }
    /// Index-space bounds of the active values (stored in the root node).
    pub fn index_bbox(&self) -> CoordBBox {
        rd_bbox(self.buf, self.root_offset() + root_off::BBOX)
    }
    /// Number of entries in the root table.
    pub fn root_table_size(&self) -> u32 {
        rd_u32(self.buf, self.root_offset() + root_off::TABLE_SIZE)
    }
    /// True if the root table is empty.
    pub fn is_empty(&self) -> bool {
        self.root_table_size() == 0
    }
    /// Number of nodes at `level` (0 = leaf, 1 = lower, 2 = upper).
    pub fn node_count(&self, level: usize) -> u32 {
        rd_u32(self.buf, GRID_DATA_SIZE + tree_off::NODE_COUNT + 4 * level)
    }
    /// Number of active tiles at `level` (1 = lower, 2 = upper, 3 = root).
    pub fn tile_count(&self, level: usize) -> u32 {
        rd_u32(self.buf, GRID_DATA_SIZE + tree_off::TILE_COUNT + 4 * (level - 1))
    }
    /// Total number of active voxels.
    pub fn active_voxel_count(&self) -> u64 {
        rd_u64(self.buf, GRID_DATA_SIZE + tree_off::VOXEL_COUNT)
    }
    /// The `n`-th blind meta data.
    pub fn blind_meta_data(&self, n: u32) -> GridBlindMetaData {
        GridBlindMetaData::read(self.buf, self.blind_meta_data_offset() as usize + n as usize * BLIND_META_DATA_SIZE)
    }
    /// The bytes of the `n`-th blind data block (`value_count * value_size` bytes).
    pub fn blind_data(&self, n: u32) -> &'a [u8] {
        let mo = self.blind_meta_data_offset() as usize + n as usize * BLIND_META_DATA_SIZE;
        let m = GridBlindMetaData::read(self.buf, mo);
        let start = (mo as i64 + m.data_offset) as usize;
        &self.buf[start..start + (m.value_count * m.value_size as u64) as usize]
    }
    /// Decode the `n`-th blind data block as values of type `T`, if its type matches.
    pub fn blind_values<T: NanoPod + BlindType>(&self, n: u32) -> Option<Vec<T>> {
        if n >= self.blind_data_count() {
            return None;
        }
        let m = self.blind_meta_data(n);
        if m.data_type != T::BLIND_TYPE || m.value_size as usize != T::SIZE {
            return None;
        }
        let d = self.blind_data(n);
        Some((0..m.value_count as usize).map(|i| T::read(d, i * T::SIZE)).collect())
    }
    /// Index of the first blind data with the given name (`Grid::findBlindData`).
    pub fn find_blind_data(&self, name: &str) -> Option<u32> {
        (0..self.blind_data_count()).find(|&i| self.blind_meta_data(i).name == name)
    }
}

/// Types that can be stored as typed blind data.
pub trait BlindType {
    /// The matching `GridType`.
    const BLIND_TYPE: GridType;
}
impl BlindType for f32 {
    const BLIND_TYPE: GridType = GridType::Float;
}
impl BlindType for f64 {
    const BLIND_TYPE: GridType = GridType::Double;
}
impl BlindType for i32 {
    const BLIND_TYPE: GridType = GridType::Int32;
}
impl BlindType for i64 {
    const BLIND_TYPE: GridType = GridType::Int64;
}
impl BlindType for u32 {
    const BLIND_TYPE: GridType = GridType::UInt32;
}
impl BlindType for crate::math::Vec3f {
    const BLIND_TYPE: GridType = GridType::Vec3f;
}
impl BlindType for crate::math::Vec3d {
    const BLIND_TYPE: GridType = GridType::Vec3d;
}

impl std::fmt::Debug for GridHeader<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GridHeader")
            .field("name", &self.grid_name())
            .field("type", &self.grid_type())
            .field("class", &self.grid_class())
            .field("size", &self.grid_size())
            .field("version", &self.version())
            .finish()
    }
}

/// A typed, zero-copy view of a NanoVDB grid (`nanovdb::NanoGrid<BuildT>`).
///
/// Dereferences to its [`GridHeader`].
pub struct NanoGrid<'a, B: BuildType> {
    header: GridHeader<'a>,
    _m: PhantomData<B>,
}

impl<'a, B: BuildType> std::ops::Deref for NanoGrid<'a, B> {
    type Target = GridHeader<'a>;
    fn deref(&self) -> &GridHeader<'a> {
        &self.header
    }
}

impl<'a, B: BuildType> NanoGrid<'a, B> {
    /// View `buf` (the bytes of one grid) as a grid with build type `B`.
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        let header = GridHeader::new(buf);
        if !header.is_valid() {
            return Err(NanoError::InvalidGrid("bad magic number or version".into()));
        }
        if (header.grid_size() as usize) > buf.len() {
            return Err(NanoError::InvalidGrid("buffer shorter than grid".into()));
        }
        if header.grid_type() != B::GRID_TYPE {
            return Err(NanoError::TypeMismatch { expected: B::GRID_TYPE, found: header.grid_type() });
        }
        Ok(NanoGrid { header: GridHeader::new(&buf[..header.grid_size() as usize]), _m: PhantomData })
    }
    /// The untyped header.
    pub fn header(&self) -> GridHeader<'a> {
        self.header
    }
    /// The tree.
    pub fn tree(&self) -> NanoTree<'a, B> {
        NanoTree::new(self.header.buf, GRID_DATA_SIZE)
    }
    /// A caching accessor (`Grid::getAccessor`).
    pub fn accessor(&self) -> ReadAccessor<'a, B> {
        ReadAccessor::new(self.tree().root())
    }
    /// World to index (double precision).
    pub fn world_to_index(&self, xyz: Vec3d) -> Vec3d {
        self.map().apply_inverse_map(xyz)
    }
    /// Index to world (double precision).
    pub fn index_to_world(&self, ijk: Vec3d) -> Vec3d {
        self.map().apply_map(ijk)
    }
    /// Direction from index to world space.
    pub fn index_to_world_dir(&self, dir: Vec3d) -> Vec3d {
        self.map().apply_jacobian(dir)
    }
    /// Direction from world to index space.
    pub fn world_to_index_dir(&self, dir: Vec3d) -> Vec3d {
        self.map().apply_inverse_jacobian(dir)
    }
    /// Gradient from index to world space.
    pub fn index_to_world_grad(&self, grad: Vec3d) -> Vec3d {
        self.map().apply_ijt(grad)
    }
    /// World to index (single precision).
    pub fn world_to_index_f(&self, xyz: [f32; 3]) -> [f32; 3] {
        self.map().apply_inverse_map_f(xyz)
    }
    /// Index to world (single precision).
    pub fn index_to_world_f(&self, ijk: [f32; 3]) -> [f32; 3] {
        self.map().apply_map_f(ijk)
    }
}

impl<B: BuildType> std::fmt::Debug for NanoGrid<'_, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.header.fmt(f)
    }
}

impl<B: BuildType> Clone for NanoGrid<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: BuildType> Copy for NanoGrid<'_, B> {}
