//! Reading and writing `.nvdb` files (port of `nanovdb/io/IO.h`).
//!
//! A NanoVDB file is a sequence of *segments*, each made of a 16-byte `FileHeader`, one
//! 176-byte `FileMetaData` plus NUL-terminated name per grid, and the (optionally ZIP or
//! BLOSC compressed) grid buffers. Since v32.6 a file may instead be a raw dump of a grid
//! buffer holding one or more grids; both forms are read.
//!
//! The module exists when the `read` or `write` feature is enabled; the readers require
//! `read` and the writers `write`. It contains no `unsafe` code.

#![forbid(unsafe_code)]

#[cfg(feature = "read")]
use super::checksum::update_grid_count;
use super::data::Version;
#[cfg(feature = "read")]
use super::data::{rd_bbox, rd_u16, rd_u32, rd_u64, rd_vec3d};
#[cfg(feature = "write")]
use super::data::{wr_bbox, wr_u16, wr_u32, wr_u64, wr_vec3d};
#[cfg(feature = "read")]
use super::handle::AlignedBuffer;
use super::handle::GridHandle;
#[cfg(feature = "read")]
use super::{to_magic, MagicType, MAGIC_NUMB};
use super::{GridClass, GridType, NanoError, Result, MAGIC_FILE};
use crate::math::{CoordBBox, Vec3d};
#[cfg(feature = "read")]
use std::io::Read;
#[cfg(feature = "write")]
use std::io::Write;
use std::path::Path;

/// Compression codec of the grids in a segment (`nanovdb::io::Codec`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Codec {
    /// Uncompressed.
    #[default]
    None = 0,
    /// zlib (`compress`/`uncompress`).
    Zip = 1,
    /// Blosc (LZ4, byte shuffle), in chunks of at most 1 GB.
    Blosc = 2,
}

impl Codec {
    #[cfg(feature = "read")]
    fn from_u16(v: u16) -> Result<Self> {
        match v {
            0 => Ok(Codec::None),
            1 => Ok(Codec::Zip),
            2 => Ok(Codec::Blosc),
            _ => Err(NanoError::InvalidFile(format!("unknown codec {}", v))),
        }
    }
    /// The C++ name of the codec.
    pub fn as_str(&self) -> &'static str {
        match self {
            Codec::None => "NONE",
            Codec::Zip => "ZIP",
            Codec::Blosc => "BLOSC",
        }
    }
}

/// Size of a `FileHeader`.
pub const FILE_HEADER_SIZE: usize = 16;
/// Size of a `FileMetaData` (without the grid name).
pub const FILE_META_DATA_SIZE: usize = 176;
/// Maximum size of an uncompressed BLOSC chunk.
#[cfg(any(feature = "read", feature = "write"))]
const BLOSC_MAX_CHUNK: usize = 1 << 30;

/// Per-grid meta data stored in a segment (`FileMetaData` + grid name).
#[derive(Clone, Debug, PartialEq)]
pub struct FileGridMetaData {
    /// Size of the grid in memory.
    pub grid_size: u64,
    /// Size of the grid on disk (compressed, including chunk size prefixes).
    pub file_size: u64,
    /// Hash of the grid name ([`string_hash`]).
    pub name_key: u64,
    /// Number of active voxels.
    pub voxel_count: u64,
    /// Value type.
    pub grid_type: GridType,
    /// Grid class.
    pub grid_class: GridClass,
    /// World-space bounding box.
    pub world_bbox: (Vec3d, Vec3d),
    /// Index-space bounding box.
    pub index_bbox: CoordBBox,
    /// Voxel size.
    pub voxel_size: Vec3d,
    /// Length of the name including the NUL terminator.
    pub name_size: u32,
    /// Number of leaf, lower, upper and root nodes.
    pub node_count: [u32; 4],
    /// Number of active tiles at the lower, upper and root levels.
    pub tile_count: [u32; 3],
    /// Codec of the grid data.
    pub codec: Codec,
    /// Number of blind data blocks.
    pub blind_data_count: u16,
    /// Version of the library that wrote the file.
    pub version: Version,
    /// The grid name.
    pub grid_name: String,
}

impl FileGridMetaData {
    /// Meta data describing grid `n` of `handle`.
    pub fn from_grid(handle: &GridHandle, n: usize, codec: Codec) -> Self {
        let h = handle.header(n).expect("grid index out of range");
        let name = h.grid_name();
        FileGridMetaData {
            grid_size: h.grid_size(),
            file_size: h.grid_size(),
            name_key: string_hash(&name),
            voxel_count: h.active_voxel_count(),
            grid_type: h.grid_type(),
            grid_class: h.grid_class(),
            world_bbox: h.world_bbox(),
            index_bbox: h.index_bbox(),
            voxel_size: h.voxel_size(),
            name_size: name.len() as u32 + 1,
            node_count: [h.node_count(0), h.node_count(1), h.node_count(2), 1],
            tile_count: [h.tile_count(1), h.tile_count(2), h.tile_count(3)],
            codec,
            blind_data_count: h.blind_data_count() as u16,
            version: Version::default(),
            grid_name: name,
        }
    }

    #[cfg(feature = "write")]
    fn to_bytes(&self) -> Vec<u8> {
        let mut b = vec![0u8; FILE_META_DATA_SIZE];
        wr_u64(&mut b, 0, self.grid_size);
        wr_u64(&mut b, 8, self.file_size);
        wr_u64(&mut b, 16, self.name_key);
        wr_u64(&mut b, 24, self.voxel_count);
        wr_u32(&mut b, 32, self.grid_type.to_u32());
        wr_u32(&mut b, 36, self.grid_class.to_u32());
        wr_vec3d(&mut b, 40, self.world_bbox.0);
        wr_vec3d(&mut b, 64, self.world_bbox.1);
        wr_bbox(&mut b, 88, &self.index_bbox);
        wr_vec3d(&mut b, 112, self.voxel_size);
        wr_u32(&mut b, 136, self.name_size);
        for i in 0..4 {
            wr_u32(&mut b, 140 + 4 * i, self.node_count[i]);
        }
        for i in 0..3 {
            wr_u32(&mut b, 156 + 4 * i, self.tile_count[i]);
        }
        wr_u16(&mut b, 168, self.codec as u16);
        wr_u16(&mut b, 170, self.blind_data_count);
        wr_u32(&mut b, 172, self.version.id());
        b.extend_from_slice(self.grid_name.as_bytes());
        b.push(0);
        b
    }

    #[cfg(feature = "read")]
    fn parse(b: &[u8], pos: &mut usize) -> Result<Self> {
        let o = *pos;
        if b.len() < o + FILE_META_DATA_SIZE {
            return Err(NanoError::InvalidFile("truncated grid meta data".into()));
        }
        let name_size = rd_u32(b, o + 136) as usize;
        let end = o + FILE_META_DATA_SIZE + name_size;
        if b.len() < end {
            return Err(NanoError::InvalidFile("truncated grid name".into()));
        }
        let name_bytes = &b[o + FILE_META_DATA_SIZE..end];
        let len = name_bytes.iter().position(|&c| c == 0).unwrap_or(name_bytes.len());
        *pos = end;
        Ok(FileGridMetaData {
            grid_size: rd_u64(b, o),
            file_size: rd_u64(b, o + 8),
            name_key: rd_u64(b, o + 16),
            voxel_count: rd_u64(b, o + 24),
            grid_type: GridType::from_u32(rd_u32(b, o + 32)),
            grid_class: GridClass::from_u32(rd_u32(b, o + 36)),
            world_bbox: (rd_vec3d(b, o + 40), rd_vec3d(b, o + 64)),
            index_bbox: rd_bbox(b, o + 88),
            voxel_size: rd_vec3d(b, o + 112),
            name_size: name_size as u32,
            node_count: [rd_u32(b, o + 140), rd_u32(b, o + 144), rd_u32(b, o + 148), rd_u32(b, o + 152)],
            tile_count: [rd_u32(b, o + 156), rd_u32(b, o + 160), rd_u32(b, o + 164)],
            codec: Codec::from_u16(rd_u16(b, o + 168))?,
            blind_data_count: rd_u16(b, o + 170),
            version: Version(rd_u32(b, o + 172)),
            grid_name: String::from_utf8_lossy(&name_bytes[..len]).into_owned(),
        })
    }
}

/// The hash of a grid name stored in the file meta data (`nanovdb::io::stringHash`).
pub fn string_hash(s: &str) -> u64 {
    let mut hash: u64 = 0;
    for &c in s.as_bytes() {
        let overflow = hash >> (64 - 8);
        hash = hash.wrapping_mul(67);
        hash = hash.wrapping_add(c as u64 + overflow);
    }
    hash
}

#[cfg(feature = "write")]
fn compress(data: &[u8], codec: Codec) -> Result<Vec<u8>> {
    match codec {
        Codec::None => Ok(data.to_vec()),
        Codec::Zip => {
            let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(data)?;
            let z = enc.finish()?;
            let mut out = Vec::with_capacity(8 + z.len());
            out.extend_from_slice(&(z.len() as u64).to_le_bytes());
            out.extend_from_slice(&z);
            Ok(out)
        }
        Codec::Blosc => {
            let mut out = Vec::new();
            for chunk in data.chunks(BLOSC_MAX_CHUNK) {
                let c = crate::io::blosc::compress(chunk, 4);
                out.extend_from_slice(&(c.len() as u64).to_le_bytes());
                out.extend_from_slice(&c);
            }
            Ok(out)
        }
    }
}

/// Decompress grid data of `size` bytes starting at `src[*pos..]`, advancing `pos`.
#[cfg(feature = "read")]
fn decompress_into(src: &[u8], pos: &mut usize, dst: &mut [u8], codec: Codec) -> Result<()> {
    let need = |n: usize, pos: usize| {
        if src.len() < pos + n {
            Err(NanoError::InvalidFile("truncated grid data".into()))
        } else {
            Ok(())
        }
    };
    match codec {
        Codec::None => {
            need(dst.len(), *pos)?;
            dst.copy_from_slice(&src[*pos..*pos + dst.len()]);
            *pos += dst.len();
        }
        Codec::Zip => {
            need(8, *pos)?;
            let n = rd_u64(src, *pos) as usize;
            *pos += 8;
            need(n, *pos)?;
            let mut dec = flate2::read::ZlibDecoder::new(&src[*pos..*pos + n]);
            dec.read_exact(dst).map_err(|e| NanoError::InvalidFile(format!("ZIP: {}", e)))?;
            *pos += n;
        }
        Codec::Blosc => {
            let mut done = 0;
            loop {
                need(8, *pos)?;
                let n = rd_u64(src, *pos) as usize;
                *pos += 8;
                need(n, *pos)?;
                let chunk = crate::io::blosc::decompress(&src[*pos..*pos + n])
                    .map_err(|e| NanoError::InvalidFile(format!("BLOSC: {}", e)))?;
                *pos += n;
                let expected = (dst.len() - done).min(BLOSC_MAX_CHUNK);
                if chunk.len() != expected {
                    return Err(NanoError::InvalidFile("BLOSC failed on byte size".into()));
                }
                dst[done..done + chunk.len()].copy_from_slice(&chunk);
                done += chunk.len();
                if done >= dst.len() {
                    break;
                }
            }
        }
    }
    Ok(())
}

/// Serialize the grids of `handle` as one segment.
#[cfg(feature = "write")]
fn segment_bytes(handle: &GridHandle, codec: Codec) -> Result<Vec<u8>> {
    if handle.grid_count() == 0 {
        return Err(NanoError::Unsupported("segment contains no grids".into()));
    }
    let mut payloads = Vec::with_capacity(handle.grid_count());
    let mut metas = Vec::with_capacity(handle.grid_count());
    for n in 0..handle.grid_count() {
        let data = compress(handle.grid_bytes(n), codec)?;
        let mut m = FileGridMetaData::from_grid(handle, n, codec);
        m.file_size = data.len() as u64;
        metas.push(m);
        payloads.push(data);
    }
    let mut out = vec![0u8; FILE_HEADER_SIZE];
    wr_u64(&mut out, 0, MAGIC_FILE);
    wr_u32(&mut out, 8, Version::default().id());
    wr_u16(&mut out, 12, handle.grid_count() as u16);
    wr_u16(&mut out, 14, codec as u16);
    for m in &metas {
        out.extend_from_slice(&m.to_bytes());
    }
    for p in payloads {
        out.extend_from_slice(&p);
    }
    Ok(out)
}

/// Write the grids of `handle` as one segment to a stream (`nanovdb::io::writeGrid`).
#[cfg(feature = "write")]
pub fn write_grid_to<W: Write>(w: &mut W, handle: &GridHandle, codec: Codec) -> Result<()> {
    w.write_all(&segment_bytes(handle, codec)?)?;
    Ok(())
}

/// Write the grids of `handle` to a file as one segment (`nanovdb::io::writeGrid`).
#[cfg(feature = "write")]
pub fn write_grid<P: AsRef<Path>>(path: P, handle: &GridHandle, codec: Codec) -> Result<()> {
    std::fs::write(path, segment_bytes(handle, codec)?)?;
    Ok(())
}

/// Write several handles to a file, one segment per handle (`nanovdb::io::writeGrids`).
#[cfg(feature = "write")]
pub fn write_grids<P: AsRef<Path>>(path: P, handles: &[GridHandle], codec: Codec) -> Result<()> {
    let mut out = Vec::new();
    for h in handles {
        out.extend_from_slice(&segment_bytes(h, codec)?);
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// A parsed segment: its codec, grid meta data and the offset of the first grid's data.
#[cfg(feature = "read")]
struct Segment {
    codec: Codec,
    meta: Vec<FileGridMetaData>,
    data: usize,
}

/// The contents of a NanoVDB file: segments or a raw grid buffer.
#[cfg(feature = "read")]
enum Contents {
    Raw,
    Segments(Vec<Segment>),
}

#[cfg(feature = "read")]
fn parse(bytes: &[u8]) -> Result<Contents> {
    if bytes.len() < 8 {
        return Err(NanoError::InvalidFile("file too short".into()));
    }
    let magic = to_magic(rd_u64(bytes, 0));
    let is_file_header = match magic {
        MagicType::NanoGrid => false,
        MagicType::NanoFile => true,
        // Legacy magic: a file header holds a compatible version right after the magic.
        MagicType::NanoVDB => bytes.len() >= 16 && Version(rd_u32(bytes, 8)).is_compatible(),
        MagicType::OpenVDB => {
            return Err(NanoError::InvalidFile("expected a NanoVDB file, found an OpenVDB file".into()))
        }
        MagicType::Unknown => {
            let m = rd_u64(bytes, 0);
            if m == MAGIC_NUMB.swap_bytes() || m == MAGIC_FILE.swap_bytes() {
                return Err(NanoError::InvalidFile("this nvdb file has reversed endianness".into()));
            }
            return Err(NanoError::InvalidFile("expected a NanoVDB file, found a file of unknown type".into()));
        }
    };
    if !is_file_header {
        return Ok(Contents::Raw);
    }
    let mut segments = Vec::new();
    let mut pos = 0;
    while pos + FILE_HEADER_SIZE <= bytes.len() {
        let magic = to_magic(rd_u64(bytes, pos));
        if magic != MagicType::NanoVDB && magic != MagicType::NanoFile {
            return Err(NanoError::InvalidFile(format!("bad segment magic at byte {}", pos)));
        }
        let version = Version(rd_u32(bytes, pos + 8));
        if !version.is_compatible() {
            return Err(NanoError::InvalidFile(format!("incompatible file version {}", version)));
        }
        let count = rd_u16(bytes, pos + 12) as usize;
        let codec = Codec::from_u16(rd_u16(bytes, pos + 14))?;
        pos += FILE_HEADER_SIZE;
        let mut meta = Vec::with_capacity(count);
        for _ in 0..count {
            let mut m = FileGridMetaData::parse(bytes, &mut pos)?;
            m.version = version;
            meta.push(m);
        }
        let data = pos;
        pos += meta.iter().map(|m| m.file_size as usize).sum::<usize>();
        if pos > bytes.len() {
            return Err(NanoError::InvalidFile("truncated segment".into()));
        }
        segments.push(Segment { codec, meta, data });
    }
    Ok(Contents::Segments(segments))
}

#[cfg(feature = "read")]
fn read_file<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

/// Read all grids from the bytes of a file: one handle per segment, or a single handle for
/// a raw grid buffer (`nanovdb::io::readGrids`).
#[cfg(feature = "read")]
pub fn read_grids_from_bytes(bytes: &[u8]) -> Result<Vec<GridHandle>> {
    match parse(bytes)? {
        Contents::Raw => Ok(vec![GridHandle::from_bytes(bytes)?]),
        Contents::Segments(segs) => {
            let mut out = Vec::with_capacity(segs.len());
            for s in segs {
                let total: usize = s.meta.iter().map(|m| m.grid_size as usize).sum();
                let mut buf = AlignedBuffer::zeroed(total);
                let (mut src, mut dst) = (s.data, 0usize);
                let count = s.meta.len() as u32;
                for (i, m) in s.meta.iter().enumerate() {
                    let g = &mut buf.as_mut_slice()[dst..dst + m.grid_size as usize];
                    decompress_into(bytes, &mut src, g, s.codec)?;
                    update_grid_count(g, i as u32, count);
                    dst += m.grid_size as usize;
                }
                out.push(GridHandle::new(buf)?);
            }
            Ok(out)
        }
    }
}

/// Read all grids of a file, one handle per segment (`nanovdb::io::readGrids`).
#[cfg(feature = "read")]
pub fn read_grids<P: AsRef<Path>>(path: P) -> Result<Vec<GridHandle>> {
    read_grids_from_bytes(&read_file(path)?)
}

/// Read every grid of a file into a single handle (`nanovdb::io::readGrid(file, -1)`).
#[cfg(feature = "read")]
pub fn read_all_grids<P: AsRef<Path>>(path: P) -> Result<GridHandle> {
    let handles = read_grids(path)?;
    if handles.len() == 1 {
        return Ok(handles.into_iter().next().unwrap());
    }
    GridHandle::merge(&handles)
}

#[cfg(feature = "read")]
fn single(bytes: &[u8], src: &mut usize, m: &FileGridMetaData, codec: Codec) -> Result<GridHandle> {
    let mut buf = AlignedBuffer::zeroed(m.grid_size as usize);
    decompress_into(bytes, src, buf.as_mut_slice(), codec)?;
    update_grid_count(buf.as_mut_slice(), 0, 1);
    GridHandle::new(buf)
}

#[cfg(feature = "read")]
fn select<F: Fn(usize, &FileGridMetaData) -> bool>(bytes: &[u8], pred: F) -> Result<Option<GridHandle>> {
    match parse(bytes)? {
        Contents::Raw => {
            let all = GridHandle::from_bytes(bytes)?;
            for (i, h) in all.split().into_iter().enumerate() {
                let m = FileGridMetaData::from_grid(&h, 0, Codec::None);
                if pred(i, &m) {
                    return Ok(Some(h));
                }
            }
            Ok(None)
        }
        Contents::Segments(segs) => {
            let mut counter = 0;
            for s in segs {
                let mut src = s.data;
                for m in &s.meta {
                    if pred(counter, m) {
                        return single(bytes, &mut src, m, s.codec).map(Some);
                    }
                    src += m.file_size as usize;
                    counter += 1;
                }
            }
            Ok(None)
        }
    }
}

/// Read the `n`-th grid of a file (`nanovdb::io::readGrid(file, n)`).
#[cfg(feature = "read")]
pub fn read_grid<P: AsRef<Path>>(path: P, n: usize) -> Result<GridHandle> {
    select(&read_file(path)?, |i, _| i == n)?
        .ok_or_else(|| NanoError::InvalidFile(format!("file does not contain a #{} grid", n)))
}

/// Read the first grid with the given name (`nanovdb::io::readGrid(file, name)`).
#[cfg(feature = "read")]
pub fn read_grid_by_name<P: AsRef<Path>>(path: P, name: &str) -> Result<GridHandle> {
    let key = string_hash(name);
    select(&read_file(path)?, |_, m| (m.name_key == 0 || m.name_key == key) && m.grid_name == name)?
        .ok_or_else(|| NanoError::InvalidFile(format!("grid name '{}' not found in file", name)))
}

/// Read the meta data of all grids in a file without decompressing them
/// (`nanovdb::io::readGridMetaData`).
#[cfg(feature = "read")]
pub fn read_grid_meta_data<P: AsRef<Path>>(path: P) -> Result<Vec<FileGridMetaData>> {
    let bytes = read_file(path)?;
    match parse(&bytes)? {
        Contents::Raw => {
            let h = GridHandle::from_bytes(&bytes)?;
            Ok((0..h.grid_count()).map(|n| FileGridMetaData::from_grid(&h, n, Codec::None)).collect())
        }
        Contents::Segments(segs) => Ok(segs.into_iter().flat_map(|s| s.meta).collect()),
    }
}

/// True if the file contains a grid with the given name (`nanovdb::io::hasGrid`).
#[cfg(feature = "read")]
pub fn has_grid<P: AsRef<Path>>(path: P, name: &str) -> Result<bool> {
    let key = string_hash(name);
    Ok(read_grid_meta_data(path)?.iter().any(|m| m.name_key == key && m.grid_name == name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash() {
        assert_eq!(string_hash(""), 0);
        assert_eq!(string_hash("a"), 97);
        assert_eq!(string_hash("ab"), 97 * 67 + 98);
    }
}
