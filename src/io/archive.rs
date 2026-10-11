//! File header, grid descriptors and per-grid reading and writing (port of
//! `openvdb/io/Archive.cc` and `openvdb/io/GridDescriptor.cc`).

use super::bytes::{Reader, WriteLe};
use super::compression::{Compression, FILE_VERSION_NODE_MASK_COMPRESSION};
use super::error::{Error, Result};
use super::metadata_io::{read_meta_map, write_meta_map};
use super::transform_io::{read_map, write_transform};
use super::tree_io::{read_tree, write_ctx, write_tree_buffers, write_tree_topology, LeafFormat};
use super::value_io::IoValue;
use super::{WriteOptions, FILE_VERSION, LIBRARY_MAJOR_VERSION, LIBRARY_MINOR_VERSION};
use crate::grid::{meta_names, AnyGrid, Grid, GridClass};
use crate::math::{Transform, Vec3i};
use crate::metadata::{MetaMap, MetaValue};
use crate::tree::{Internal2, Tree};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

/// The magic number at the start of every `.vdb` file (`"VDB "` as a little-endian `i64`).
pub const MAGIC: i64 = 0x5644_4220;
/// Oldest supported file format version (`OPENVDB_FILE_VERSION_SELECTIVE_COMPRESSION`).
pub const MIN_FILE_VERSION: u32 = 220;
/// Newest readable file format version (`OPENVDB_FILE_VERSION_HALF_GRID`).
pub const MAX_FILE_VERSION: u32 = 225;
/// File version from which the header no longer carries a compression flag.
const FILE_VERSION_BLOSC_COMPRESSION: u32 = 223;

const HALF_FLOAT_TYPENAME_SUFFIX: &str = "_HalfFloat";
const SEP: char = '\x1e';

/// The header of a `.vdb` file or stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileHeader {
    /// File format version (e.g. 224).
    pub file_version: u32,
    /// Version of the library that wrote the file, as (major, minor).
    pub library_version: (u32, u32),
    /// True if grid descriptors carry valid byte offsets (files written by `io::File`);
    /// false for streams, which must be read sequentially.
    pub has_grid_offsets: bool,
    /// The file's 36-character UUID string.
    pub uuid: String,
    /// File-wide compression for formats older than 222; newer files store compression
    /// per grid (see [`GridMetadata::compression`](super::GridMetadata::compression)).
    pub compression: Compression,
}

impl FileHeader {
    /// OpenVDB's version string, `"<major>.<minor>/<file version>"`.
    pub fn version_string(&self) -> String {
        format!("{}.{}/{}", self.library_version.0, self.library_version.1, self.file_version)
    }
}

/// Read and validate a file header.
pub(crate) fn read_header<R: Read>(r: &mut Reader<R>) -> Result<FileHeader> {
    let magic = r.read_i64().map_err(|_| Error::NotVdb)?;
    if magic != MAGIC {
        return Err(Error::NotVdb);
    }
    let file_version = r.read_u32()?;
    if !(MIN_FILE_VERSION..=MAX_FILE_VERSION).contains(&file_version) {
        return Err(Error::UnsupportedVersion(file_version));
    }
    let major = r.read_u32()?;
    let minor = r.read_u32()?;
    let has_grid_offsets = r.read_u8()? != 0;
    let mut compression = if file_version < FILE_VERSION_BLOSC_COMPRESSION {
        Compression::ZIP | Compression::ACTIVE_MASK
    } else {
        Compression::BLOSC | Compression::ACTIVE_MASK
    };
    if file_version < FILE_VERSION_NODE_MASK_COMPRESSION {
        compression = if r.read_u8()? != 0 { Compression::ZIP } else { Compression::NONE };
    }
    let uuid = r.read_bytes(36)?;
    Ok(FileHeader {
        file_version,
        library_version: (major, minor),
        has_grid_offsets,
        uuid: String::from_utf8_lossy(&uuid).into_owned(),
        compression,
    })
}

/// Generate a random UUID string in OpenVDB's format.
fn new_uuid() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut hex = String::with_capacity(36);
    for i in 0..4u64 {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(i);
        if let Ok(d) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            h.write_u128(d.as_nanos());
        }
        hex.push_str(&format!("{:08X}", h.finish() as u32));
    }
    format!("{}-{}-{}-{}-{}", &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32])
}

/// Write a file header (current format version, no header compression flag).
pub(crate) fn write_header(out: &mut Vec<u8>, seekable: bool) {
    out.put_i64(MAGIC);
    out.put_u32(FILE_VERSION);
    out.put_u32(LIBRARY_MAJOR_VERSION);
    out.put_u32(LIBRARY_MINOR_VERSION);
    out.put_u8(seekable as u8);
    out.extend_from_slice(new_uuid().as_bytes());
}

/// Describes one grid stored in a file: its names, type and byte offsets
/// (port of `io::GridDescriptor`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridDescriptor {
    /// The grid's name (the unique name without any `"\x1e<n>"` suffix).
    pub grid_name: String,
    /// The name under which the grid is stored, unique within the file. Duplicate and empty
    /// names get a suffix of an ASCII record separator (`'\x1e'`) and a number.
    pub unique_name: String,
    /// The tree type, e.g. `"Tree_float_5_4_3"` (without the `"_HalfFloat"` suffix).
    pub grid_type: String,
    /// True if floating-point values are stored as 16-bit half floats.
    pub save_float_as_half: bool,
    /// Unique name of the grid whose tree this grid shares, or empty.
    pub instance_parent: String,
    /// Offset of the grid's data (compression flags, metadata, transform, topology).
    pub grid_pos: i64,
    /// Offset of the grid's leaf buffers.
    pub block_pos: i64,
    /// Offset just past the grid's data.
    pub end_pos: i64,
}

impl GridDescriptor {
    /// True if this grid shares the tree of another grid ([`instance_parent`](Self::instance_parent)).
    pub fn is_instance(&self) -> bool {
        !self.instance_parent.is_empty()
    }
    /// The value type name parsed from the grid type (e.g. `"float"`, `"mask"`,
    /// `"ptdataidx32"`), if the type has the standard `Tree_<value>_5_4_3` form.
    pub fn value_type(&self) -> Option<&str> {
        self.grid_type.strip_prefix("Tree_")?.strip_suffix("_5_4_3")
    }
    /// True if grids of this type can be loaded by this library.
    pub fn is_supported(&self) -> bool {
        GridKind::from_type(&self.grid_type).is_some()
    }
    /// The unique name in printable form: `"name[n]"` for suffixed names (OpenVDB's
    /// `GridDescriptor::nameAsString`).
    pub fn display_name(&self) -> String {
        name_as_string(&self.unique_name)
    }

    pub(crate) fn read<R: Read>(r: &mut Reader<R>) -> Result<Self> {
        let unique_name = r.read_string()?;
        let mut grid_type = r.read_string()?;
        let mut save_float_as_half = false;
        if let Some(t) = grid_type.strip_suffix(HALF_FLOAT_TYPENAME_SUFFIX) {
            grid_type = t.to_string();
            save_float_as_half = true;
        }
        let instance_parent = r.read_string()?;
        let grid_pos = r.read_i64()?;
        let block_pos = r.read_i64()?;
        let end_pos = r.read_i64()?;
        Ok(GridDescriptor {
            grid_name: strip_suffix(&unique_name).to_string(),
            unique_name,
            grid_type,
            save_float_as_half,
            instance_parent,
            grid_pos,
            block_pos,
            end_pos,
        })
    }

    fn write_header(&self, out: &mut Vec<u8>) {
        out.put_string(&self.unique_name);
        let mut t = self.grid_type.clone();
        if self.save_float_as_half {
            t.push_str(HALF_FLOAT_TYPENAME_SUFFIX);
        }
        out.put_string(&t);
        out.put_string(&self.instance_parent);
    }
}

/// Append a numeric suffix to a grid name (`GridDescriptor::addSuffix`).
pub fn add_suffix(name: &str, n: usize) -> String {
    format!("{}{}{}", name, SEP, n)
}

/// Remove a numeric suffix from a unique grid name (`GridDescriptor::stripSuffix`).
pub fn strip_suffix(name: &str) -> &str {
    name.split(SEP).next().unwrap_or("")
}

/// Printable form of a unique name: `"name[n]"` (`GridDescriptor::nameAsString`).
pub fn name_as_string(name: &str) -> String {
    match name.find(SEP) {
        Some(p) => format!("{}[{}]", &name[..p], &name[p + 1..]),
        None => name.to_string(),
    }
}

/// Inverse of [`name_as_string`]: `"name[n]"` becomes `"name\x1en"`
/// (`GridDescriptor::stringAsUniqueName`).
pub fn string_as_unique_name(s: &str) -> String {
    if s.ends_with(']') {
        if let Some(p) = s.find('[') {
            return format!("{}{}{}", &s[..p], SEP, &s[p + 1..s.len() - 1]);
        }
    }
    s.to_string()
}

/// The grid types this library can load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GridKind {
    Float,
    Double,
    Int32,
    Int64,
    Bool,
    Mask,
    Vec3S,
    Vec3D,
    Vec3I,
    PointData,
}

impl GridKind {
    pub fn from_type(grid_type: &str) -> Option<GridKind> {
        Some(match grid_type {
            "Tree_float_5_4_3" => GridKind::Float,
            "Tree_double_5_4_3" => GridKind::Double,
            "Tree_int32_5_4_3" => GridKind::Int32,
            "Tree_int64_5_4_3" => GridKind::Int64,
            "Tree_bool_5_4_3" => GridKind::Bool,
            "Tree_mask_5_4_3" => GridKind::Mask,
            "Tree_vec3s_5_4_3" => GridKind::Vec3S,
            "Tree_vec3d_5_4_3" => GridKind::Vec3D,
            "Tree_vec3i_5_4_3" => GridKind::Vec3I,
            "Tree_ptdataidx32_5_4_3" => GridKind::PointData,
            _ => return None,
        })
    }
}

/// Everything stored for a grid except its tree.
#[derive(Clone, Debug)]
pub(crate) struct GridHead {
    pub compression: Compression,
    pub metadata: MetaMap,
    pub map_type: String,
    /// `Err` if the transform's map type is not supported.
    pub transform: std::result::Result<Transform, String>,
}

/// Read a grid's compression flags, metadata and transform. The reader must be positioned
/// at the descriptor's `grid_pos`.
pub(crate) fn read_grid_head<R: Read>(r: &mut Reader<R>, header: &FileHeader) -> Result<GridHead> {
    let compression = if header.file_version >= FILE_VERSION_NODE_MASK_COMPRESSION {
        Compression(r.read_u32()?)
    } else {
        header.compression
    };
    let mut metadata = read_meta_map(r)?;
    // Delayed loading is not supported; drop its bookkeeping like OpenVDB does.
    metadata.remove(meta_names::FILE_DELAYED_LOAD);
    let map_type = r.read_string()?;
    let transform = match read_map(r, &map_type) {
        Ok(m) => Ok(Transform::from_map(m)),
        Err(Error::Unsupported(msg)) => Err(msg),
        Err(e) => return Err(e),
    };
    Ok(GridHead { compression, metadata, map_type, transform })
}

fn typed_grid<T: IoValue>(tree: Tree<T>, transform: Transform, metadata: MetaMap) -> Grid<T> {
    let mut g = Grid::from_tree(tree).with_transform(transform);
    *g.metadata_mut() = metadata;
    g
}

/// Read a grid's tree (if `read_tree_data`) and assemble the grid. The reader must be
/// positioned right after the grid head.
pub(crate) fn read_grid_tree<R: Read>(
    r: &mut Reader<R>,
    header: &FileHeader,
    desc: &GridDescriptor,
    head: GridHead,
    read_tree_data: bool,
) -> Result<AnyGrid> {
    let kind =
        GridKind::from_type(&desc.grid_type).ok_or_else(|| Error::UnsupportedGridType(desc.grid_type.clone()))?;
    let transform = head.transform.map_err(Error::Unsupported)?;
    let mut meta = head.metadata;
    if !meta.contains(meta_names::GRID_NAME) && !desc.grid_name.is_empty() {
        meta.insert(meta_names::GRID_NAME, desc.grid_name.as_str());
    }
    let (v, c, h) = (header.file_version, head.compression, desc.save_float_as_half);
    macro_rules! load {
        ($t:ty, $variant:ident, $fmt:expr) => {{
            let tree: Tree<$t> = if read_tree_data {
                read_tree::<$t, Internal2<$t>, R>(r, v, c, h, $fmt)?
            } else {
                Tree::new(<$t>::default())
            };
            AnyGrid::$variant(typed_grid(tree, transform, meta))
        }};
    }
    Ok(match kind {
        GridKind::Float => load!(f32, Float, LeafFormat::Standard),
        GridKind::Double => load!(f64, Double, LeafFormat::Standard),
        GridKind::Int32 => load!(i32, Int32, LeafFormat::Standard),
        GridKind::Int64 => load!(i64, Int64, LeafFormat::Standard),
        GridKind::Bool => load!(bool, Bool, LeafFormat::Bool),
        GridKind::Mask => load!(bool, Mask, LeafFormat::Mask),
        GridKind::Vec3S => load!(crate::math::Vec3f, Vec3S, LeafFormat::Standard),
        GridKind::Vec3D => load!(crate::math::Vec3d, Vec3D, LeafFormat::Standard),
        GridKind::Vec3I => load!(Vec3i, Vec3I, LeafFormat::Standard),
        GridKind::PointData => {
            let tree = if read_tree_data {
                super::points_io::read_point_data_tree(r, v, c)?
            } else {
                crate::points::PointDataTree::new()
            };
            let mut g = crate::points::PointDataGrid::from_tree(tree, transform);
            *g.metadata_mut() = meta;
            AnyGrid::PointData(g)
        }
    })
}

/// Give `grid` a copy of `parent`'s tree (for instanced grids). Fails if the value types
/// differ.
pub(crate) fn copy_tree_from(grid: &mut AnyGrid, parent: &AnyGrid) -> Result<()> {
    macro_rules! copy {
        ($($v:ident),*) => {
            match (grid, parent) {
                $((AnyGrid::$v(g), AnyGrid::$v(p)) => g.set_tree(p.tree().clone()),)*
                _ => return Err(Error::corrupt("instanced grid and its parent have different types")),
            }
        };
    }
    copy!(Float, Double, Int32, Int64, Bool, Mask, Vec3S, Vec3D, Vec3I, PointData);
    Ok(())
}

/// Grid statistics metadata as OpenVDB's `GridBase::addStatsMetadata` computes it
/// (an empty grid gets an inverted bounding box).
fn add_stats_metadata(meta: &mut MetaMap, grid: &AnyGrid) {
    let bbox = grid.eval_active_voxel_bbox();
    let (min, max) = if bbox.is_empty() {
        (Vec3i::splat(i32::MAX), Vec3i::splat(i32::MIN))
    } else {
        (bbox.min.as_vec3i(), bbox.max.as_vec3i())
    };
    let mem = crate::with_any_grid!(grid, g => g.mem_usage());
    meta.insert(meta_names::FILE_BBOX_MIN, min);
    meta.insert(meta_names::FILE_BBOX_MAX, max);
    meta.insert(meta_names::FILE_MEM_BYTES, mem as i64);
    meta.insert(meta_names::FILE_VOXEL_COUNT, grid.active_voxel_count() as i64);
}

/// Compression actually used for a grid: zip is never applied to level sets and fog volumes.
pub(crate) fn grid_compression(c: Compression, grid: &AnyGrid) -> Compression {
    match grid.grid_class() {
        GridClass::LevelSet | GridClass::FogVolume => c.without(Compression::ZIP),
        _ => c,
    }
}

/// Write a complete archive (header, file metadata, grids) into a buffer
/// (port of `Archive::write`). Offsets are filled in when `seekable` is set.
pub(crate) fn write_archive(
    grids: &[&AnyGrid],
    meta: &MetaMap,
    opts: &WriteOptions,
    seekable: bool,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    write_header(&mut out, seekable);
    write_meta_map(&mut out, meta);
    out.put_i32(grids.len() as i32);

    // Make grid names unique, as OpenVDB does.
    let mut name_count: BTreeMap<&str, usize> = BTreeMap::new();
    for g in grids {
        *name_count.entry(g.name()).or_default() += 1;
    }
    let mut unique_names = BTreeSet::new();
    for grid in grids {
        let name = grid.name();
        let mut unique = name.to_string();
        if name.is_empty() || name_count[name] > 1 {
            unique = add_suffix(name, 0);
        }
        let mut n = 1;
        while unique_names.contains(&unique) {
            unique = add_suffix(name, n);
            n += 1;
        }
        unique_names.insert(unique.clone());
        write_grid(&mut out, grid, unique, opts, seekable)?;
    }
    Ok(out)
}

fn write_grid(
    out: &mut Vec<u8>,
    grid: &AnyGrid,
    unique_name: String,
    opts: &WriteOptions,
    seekable: bool,
) -> Result<()> {
    let half = crate::with_any_grid!(grid, g => g.save_float_as_half());
    let mut desc = GridDescriptor {
        grid_name: strip_suffix(&unique_name).to_string(),
        unique_name,
        grid_type: grid.grid_type(),
        save_float_as_half: half,
        instance_parent: String::new(),
        grid_pos: 0,
        block_pos: 0,
        end_pos: 0,
    };
    desc.write_header(out);
    let offset_pos = out.len();
    out.extend_from_slice(&[0u8; 24]);
    desc.grid_pos = out.len() as i64;

    let compression = grid_compression(opts.compression, grid);
    out.put_u32(compression.bits());
    let mut meta = grid.metadata().clone();
    if opts.write_grid_stats {
        add_stats_metadata(&mut meta, grid);
        meta.insert(meta_names::FILE_COMPRESSION, MetaValue::String(compression.to_string()));
    }
    write_meta_map(out, &meta);
    write_transform(out, grid.transform())?;

    macro_rules! tree {
        ($g:expr, $fmt:expr) => {{
            let ctx = write_ctx($g.tree(), compression, half, $fmt);
            write_tree_topology(out, $g.tree(), &ctx);
            desc.block_pos = out.len() as i64;
            write_tree_buffers(out, $g.tree(), &ctx);
        }};
    }
    match grid {
        AnyGrid::Float(g) => tree!(g, LeafFormat::Standard),
        AnyGrid::Double(g) => tree!(g, LeafFormat::Standard),
        AnyGrid::Int32(g) => tree!(g, LeafFormat::Standard),
        AnyGrid::Int64(g) => tree!(g, LeafFormat::Standard),
        AnyGrid::Bool(g) => tree!(g, LeafFormat::Bool),
        AnyGrid::Mask(g) => tree!(g, LeafFormat::Mask),
        AnyGrid::Vec3S(g) => tree!(g, LeafFormat::Standard),
        AnyGrid::Vec3D(g) => tree!(g, LeafFormat::Standard),
        AnyGrid::Vec3I(g) => tree!(g, LeafFormat::Standard),
        AnyGrid::PointData(g) => {
            super::points_io::write_point_data_tree(out, g.tree(), compression, &mut desc.block_pos);
        }
    }
    desc.end_pos = out.len() as i64;
    if seekable {
        let mut pos = Vec::with_capacity(24);
        pos.put_i64(desc.grid_pos);
        pos.put_i64(desc.block_pos);
        pos.put_i64(desc.end_pos);
        out[offset_pos..offset_pos + 24].copy_from_slice(&pos);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let u = add_suffix("density", 2);
        assert_eq!(u, "density\x1e2");
        assert_eq!(strip_suffix(&u), "density");
        assert_eq!(name_as_string(&u), "density[2]");
        assert_eq!(string_as_unique_name("density[2]"), u);
        assert_eq!(string_as_unique_name("density"), "density");
    }

    #[test]
    fn header_roundtrip() {
        let mut out = Vec::new();
        write_header(&mut out, true);
        let h = read_header(&mut Reader::new(&out[..])).unwrap();
        assert_eq!(h.file_version, FILE_VERSION);
        assert!(h.has_grid_offsets);
        assert_eq!(h.uuid.len(), 36);
        assert_eq!(h.version_string(), format!("{}.{}/{}", LIBRARY_MAJOR_VERSION, LIBRARY_MINOR_VERSION, FILE_VERSION));
        assert!(matches!(read_header(&mut Reader::new(&b"not a vdb file"[..])), Err(Error::NotVdb)));
    }

    #[test]
    fn descriptor_value_type() {
        let d = GridDescriptor {
            grid_name: "p".into(),
            unique_name: "p".into(),
            grid_type: "Tree_foo_5_4_3".into(),
            save_float_as_half: false,
            instance_parent: String::new(),
            grid_pos: 0,
            block_pos: 0,
            end_pos: 0,
        };
        assert_eq!(d.value_type(), Some("foo"));
        assert!(!d.is_supported());
    }
}
