//! Serialization of point data grids (port of the I/O parts of
//! `openvdb/points/AttributeArray.h`, `AttributeSet.cc`, `StreamCompression.cc` and
//! `PointDataGrid.h`).
//!
//! A point data tree is written like any tree's topology, followed by its leaf buffers in
//! several *passes* over all leaves (`io::MultiPass`), preceded by the pass count (`u16`):
//!
//! | pass | contents (per leaf) |
//! |------|---------------------|
//! | 0 | compressed size (`u16`) of the voxel offsets |
//! | 1 | descriptor (once if shared by all leaves) and attribute array headers |
//! | 2 ..= n+1 | page headers of attribute `pass - 2` (Blosc only) |
//! | n+2 | voxel offsets: value mask, `u16` size, Blosc frame or raw 512 × `u32` |
//! | n+3 ..= 2n+2 | data of attribute `pass - n - 3` |
//! | 2n+3 | cleanup (no data) |
//!
//! With Blosc compression, attribute data is written through *paged streams*: the arrays
//! of one attribute in all leaves are concatenated into pages of up to 1 MiB, each Blosc
//! compressed; the page headers (compressed and uncompressed sizes) are written in the size
//! pass and the page data in the data pass. Without Blosc, each array is written directly.

use super::blosc;
#[cfg(feature = "read")]
use super::bytes::Reader;
#[cfg(feature = "write")]
use super::bytes::WriteLe;
#[cfg(feature = "read")]
use super::compression::read_mask;
#[cfg(feature = "write")]
use super::compression::write_mask;
use super::compression::Compression;
#[cfg(feature = "read")]
use super::error::{Error, Result};
#[cfg(feature = "read")]
use super::metadata_io::read_meta_map;
#[cfg(feature = "write")]
use super::metadata_io::write_meta_map;
#[cfg(feature = "read")]
use super::tree_io::read_tree_topology;
#[cfg(feature = "write")]
use super::tree_io::{write_ctx, write_tree_topology};
use super::tree_io::{LeafFormat, NodeIo};
#[cfg(feature = "read")]
use crate::points::AttributeType;
#[cfg(feature = "write")]
use crate::points::PointLeaf;
use crate::points::{AttributeArray, AttributeSet, Descriptor, PointDataTree};
use crate::tools::point_index_grid::PointIndexTree;
use crate::tree::Node;
#[cfg(feature = "write")]
use crate::tree::Tree;
#[cfg(feature = "read")]
use crate::tree::{Internal2, LeafNode};
#[cfg(feature = "write")]
use rayon::prelude::*;
#[cfg(feature = "write")]
use std::borrow::Cow;
#[cfg(feature = "read")]
use std::collections::{BTreeMap, HashMap};
#[cfg(feature = "read")]
use std::io::Read;
#[cfg(feature = "read")]
use std::sync::Arc;

/// File format version that introduced multi-pass leaf I/O.
#[cfg(feature = "read")]
const FILE_VERSION_MULTIPASS_IO: u32 = 224;
/// Size of the pages of paged streams (`compression::PageSize`).
#[cfg(feature = "write")]
pub(crate) const PAGE_SIZE: usize = 1024 * 1024;
/// Inputs up to this size are never Blosc compressed (`BLOSC_MINIMUM_BYTES`).
#[cfg(feature = "write")]
const BLOSC_MINIMUM_BYTES: usize = 48;
/// Inputs smaller than this are zero-padded before compression (`BLOSC_PAD_BYTES`).
const BLOSC_PAD_BYTES: usize = 128;

/// Attribute serialization flags (`AttributeArray::SerializationFlag`).
#[cfg(any(feature = "read", feature = "write"))]
const WRITE_STRIDED: u8 = 0x1;
const WRITE_UNIFORM: u8 = 0x2;
const WRITE_PAGED: u8 = 0x8;
/// Array flag set while an array is partially read; never written.
const PARTIAL_READ: u8 = 0x20;

/// Blosc compression as `compression::bloscCompress` does it: no compression for inputs of
/// at most 48 bytes, zero padding to 128 bytes for small inputs, and `None` unless the
/// result is smaller than the input.
#[cfg(feature = "write")]
pub(crate) fn blosc_compress(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() <= BLOSC_MINIMUM_BYTES {
        return None;
    }
    let frame = if data.len() < BLOSC_PAD_BYTES {
        let mut padded = data.to_vec();
        padded.resize(BLOSC_PAD_BYTES, 0);
        blosc::compress(&padded, 4)
    } else {
        blosc::compress(data, 4)
    };
    (frame.len() < data.len()).then_some(frame)
}

/// Decompress a Blosc frame of `expected` bytes (frames of small inputs may decompress to the
/// 128 padded bytes).
#[cfg(feature = "read")]
pub(crate) fn blosc_decompress(frame: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut out = blosc::decompress(frame)?;
    if out.len() == BLOSC_PAD_BYTES && expected <= BLOSC_PAD_BYTES {
        out.truncate(expected);
    } else if out.len() != expected {
        return Err(Error::corrupt(format!("expected to decompress {} bytes, got {}", expected, out.len())));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Descriptors

/// Read an attribute set descriptor (`Descriptor::read`).
#[cfg(feature = "read")]
pub(crate) fn read_descriptor<R: Read>(r: &mut Reader<R>) -> Result<Descriptor> {
    let n = r.read_u64()? as usize;
    if n > 1 << 20 {
        return Err(Error::corrupt("implausible attribute count"));
    }
    let mut types = Vec::with_capacity(n);
    for _ in 0..n {
        let v = r.read_string()?;
        let c = r.read_string()?;
        types.push(AttributeType::new(v, c));
    }
    let mut name_map = BTreeMap::new();
    for _ in 0..n {
        let name = r.read_string()?;
        if !crate::points::attribute_set::valid_name(&name) {
            return Err(Error::corrupt(format!("attribute name contains invalid characters - {}", name)));
        }
        let pos = r.read_u64()? as usize;
        if pos >= n {
            return Err(Error::corrupt(format!("attribute position {} out of range", pos)));
        }
        name_map.insert(name, pos);
    }
    let ngroups = r.read_u64()?;
    let mut group_map = BTreeMap::new();
    for _ in 0..ngroups {
        let name = r.read_string()?;
        if !crate::points::attribute_set::valid_name(&name) {
            return Err(Error::corrupt(format!("group name contains invalid characters - {}", name)));
        }
        let offset = r.read_u64()? as usize;
        group_map.insert(name, offset);
    }
    let metadata = read_meta_map(r)?;
    Ok(Descriptor::from_raw(types, name_map, group_map, metadata))
}

/// Write an attribute set descriptor (`Descriptor::write`).
#[cfg(feature = "write")]
pub(crate) fn write_descriptor(out: &mut Vec<u8>, d: &Descriptor) {
    out.put_u64(d.size() as u64);
    for t in d.types() {
        out.put_string(&t.value_type);
        out.put_string(&t.codec);
    }
    for (name, &pos) in d.map() {
        out.put_string(name);
        out.put_u64(pos as u64);
    }
    out.put_u64(d.group_map().len() as u64);
    for (name, &offset) in d.group_map() {
        out.put_string(name);
        out.put_u64(offset as u64);
    }
    write_meta_map(out, d.metadata());
}

// ---------------------------------------------------------------------------
// Attribute array headers and buffers

/// An attribute array header (`TypedAttributeArray::readMetadata`).
#[cfg(feature = "read")]
#[derive(Clone, Debug)]
struct ArrayHeader {
    flags: u8,
    uniform: bool,
    paged: bool,
    size: u32,
    stride_or_total: u32,
    /// Number of data bytes (compressed size for unpaged Blosc data).
    bytes: usize,
}

#[cfg(feature = "read")]
fn read_array_header<R: Read>(r: &mut Reader<R>) -> Result<ArrayHeader> {
    let bytes = r.read_u64()?;
    let flags = r.read_u8()?;
    let ser = r.read_u8()?;
    let size = r.read_u32()?;
    if ser >= 0x10 {
        return Err(Error::corrupt("unknown attribute serialization flags"));
    }
    let stride_or_total = if ser & WRITE_STRIDED != 0 { r.read_u32()? } else { 1 };
    let bytes = bytes.checked_sub(6).ok_or_else(|| Error::corrupt("invalid attribute array size"))? as usize;
    Ok(ArrayHeader {
        flags: flags & !PARTIAL_READ,
        uniform: ser & WRITE_UNIFORM != 0,
        paged: ser & WRITE_PAGED != 0,
        size,
        stride_or_total,
        bytes,
    })
}

/// Write an array header for multi-pass leaf output (`writeMetadata(os, false, paged=true)`).
#[cfg(feature = "write")]
fn write_array_header(out: &mut Vec<u8>, a: &AttributeArray, blosc: bool) {
    let strided = a.stride() != 1;
    let mut ser = 0u8;
    if strided {
        ser |= WRITE_STRIDED;
    }
    if a.is_uniform() {
        ser |= WRITE_UNIFORM;
    }
    if blosc {
        ser |= WRITE_PAGED;
    }
    out.put_u64(6 + a.storage().len() as u64);
    out.put_u8(a.flags() & !PARTIAL_READ);
    out.put_u8(ser);
    out.put_u32(a.size());
    if strided {
        out.put_u32(a.stride_or_total_size());
    }
}

/// Write unpaged array data (`TypedAttributeArray::writeBuffers` without Blosc).
#[cfg(feature = "write")]
fn write_array_buffers(out: &mut Vec<u8>, a: &AttributeArray) {
    if !a.is_uniform() {
        out.put_u8(0);
    }
    out.extend_from_slice(a.storage());
}

/// Read unpaged array data (`TypedAttributeArray::readBuffers`).
#[cfg(feature = "read")]
fn read_array_buffers<R: Read>(r: &mut Reader<R>, h: &ArrayHeader, storage_size: usize) -> Result<Vec<u8>> {
    let compressed = if h.uniform { false } else { r.read_u8()? == 1 };
    let data = r.read_bytes(h.bytes)?;
    if compressed {
        let expected = data_size(h) as usize * storage_size;
        blosc_decompress(&data, expected)
    } else {
        Ok(data)
    }
}

#[cfg(feature = "read")]
fn data_size(h: &ArrayHeader) -> u32 {
    if h.flags & crate::points::attribute_array::FLAG_CONSTANT_STRIDE != 0 {
        h.size * h.stride_or_total
    } else {
        h.stride_or_total
    }
}

// ---------------------------------------------------------------------------
// Paged streams

/// Writes data in Blosc-compressed pages (`compression::PagedOutputStream`). In size-only
/// mode only the page headers are written; otherwise only the page data.
#[cfg(feature = "write")]
struct PagedOutputStream {
    page: Vec<u8>,
    size_only: bool,
}

#[cfg(feature = "write")]
impl PagedOutputStream {
    fn new(size_only: bool) -> Self {
        PagedOutputStream { page: Vec::new(), size_only }
    }

    fn write(&mut self, out: &mut Vec<u8>, data: &[u8]) {
        if data.len() > PAGE_SIZE {
            self.flush(out);
            self.compress_and_write(out, data);
        } else {
            if data.len() + self.page.len() > PAGE_SIZE {
                self.flush(out);
            }
            self.page.extend_from_slice(data);
        }
    }

    fn flush(&mut self, out: &mut Vec<u8>) {
        let page = std::mem::take(&mut self.page);
        self.compress_and_write(out, &page);
    }

    fn compress_and_write(&self, out: &mut Vec<u8>, buf: &[u8]) {
        if buf.is_empty() {
            return;
        }
        match blosc_compress(buf) {
            None if self.size_only => out.put_i32(-(buf.len() as i32)),
            None => out.extend_from_slice(buf),
            Some(c) if self.size_only => {
                out.put_i32(c.len() as i32);
                out.put_i32(buf.len() as i32);
            }
            Some(c) => out.extend_from_slice(&c),
        }
    }
}

/// A reference to a part of a page (`compression::PageHandle`).
#[cfg(feature = "read")]
#[derive(Clone, Copy, Debug)]
struct PageHandle {
    page: usize,
    offset: usize,
    size: usize,
}

/// Reads paged data (`compression::PagedInputStream`): handles are created while reading
/// the page headers (size pass) and resolved while reading the page data (data pass).
#[cfg(feature = "read")]
#[derive(Default)]
struct PagedInputStream {
    /// (compressed bytes, uncompressed bytes) of each page; negative compressed sizes mark
    /// uncompressed pages.
    pages: Vec<(i32, i32)>,
    data: Vec<Option<Vec<u8>>>,
    byte_index: usize,
    uncompressed: usize,
}

#[cfg(feature = "read")]
impl PagedInputStream {
    fn create_handle<R: Read>(&mut self, r: &mut Reader<R>, n: usize) -> Result<PageHandle> {
        if self.byte_index == self.uncompressed || self.pages.is_empty() {
            let compressed = r.read_i32()?;
            let uncompressed = if compressed > 0 { r.read_i32()? } else { -compressed };
            if compressed == 0 || uncompressed <= 0 {
                return Err(Error::corrupt("invalid page header"));
            }
            self.pages.push((compressed, uncompressed));
            self.data.push(None);
            self.uncompressed = uncompressed as usize;
            self.byte_index = 0;
        }
        let h = PageHandle { page: self.pages.len() - 1, offset: self.byte_index, size: n };
        self.byte_index += n;
        if self.byte_index > self.uncompressed {
            return Err(Error::corrupt("attribute data exceeds its page"));
        }
        Ok(h)
    }

    fn read<R: Read>(&mut self, r: &mut Reader<R>, h: PageHandle) -> Result<Vec<u8>> {
        if self.byte_index == self.uncompressed && self.data[h.page].is_none() {
            let (compressed, uncompressed) = self.pages[h.page];
            self.uncompressed = uncompressed as usize;
            let bytes = r.read_bytes(compressed.unsigned_abs() as usize)?;
            let page = if compressed > 0 {
                let out = blosc::decompress(&bytes)?;
                if out.len() < uncompressed as usize {
                    return Err(Error::corrupt("page decompressed to too few bytes"));
                }
                out
            } else {
                bytes
            };
            self.data[h.page] = Some(page);
            self.byte_index = 0;
        }
        self.byte_index += h.size;
        let page = self.data[h.page].as_ref().ok_or_else(|| Error::corrupt("paged data read out of order"))?;
        page.get(h.offset..h.offset + h.size)
            .map(|s| s.to_vec())
            .ok_or_else(|| Error::corrupt("attribute data exceeds its page"))
    }
}

// ---------------------------------------------------------------------------
// Point data trees

/// Per-leaf state while reading the passes.
#[cfg(feature = "read")]
#[derive(Default)]
struct LeafState {
    descriptor: Option<Arc<Descriptor>>,
    headers: Vec<ArrayHeader>,
    handles: Vec<Option<PageHandle>>,
    arrays: Vec<Option<AttributeArray>>,
}

/// Read a point data tree (topology and multi-pass buffers).
#[cfg(feature = "read")]
pub(crate) fn read_point_data_tree<R: Read>(
    r: &mut Reader<R>,
    file_version: u32,
    compression: Compression,
) -> Result<PointDataTree> {
    let (mut tree, _) =
        read_tree_topology::<u32, Internal2<u32>, R>(r, file_version, compression, false, LeafFormat::Standard)?;
    if file_version < FILE_VERSION_MULTIPASS_IO {
        return Err(Error::Unsupported(format!(
            "point data grids in file format version {} (multi-pass I/O requires {})",
            file_version, FILE_VERSION_MULTIPASS_IO
        )));
    }
    let num_passes = r.read_u16()? as u32;
    let attributes = num_passes.wrapping_sub(4) / 2;
    let mut leaves: Vec<&mut LeafNode<u32>> = tree.leaf_refs_mut();
    let mut states: Vec<LeafState> = (0..leaves.len()).map(|_| LeafState::default()).collect();
    let mut paged: HashMap<usize, PagedInputStream> = HashMap::new();
    let mut matching: Option<Arc<Descriptor>> = None;

    for pass in 0..num_passes {
        if pass == 0 {
            for _ in 0..leaves.len() {
                r.read_u16()?; // voxel buffer size, only used for seeking
            }
            matching = None;
        } else if pass == 1 {
            for st in states.iter_mut() {
                let descriptor = match &matching {
                    Some(d) => d.clone(),
                    None => {
                        let header = r.read_u8()?;
                        let d = Arc::new(read_descriptor(r)?);
                        if header & 1 != 0 {
                            matching = Some(d.clone());
                        }
                        if header & 2 != 0 {
                            let skip = r.read_u64()?;
                            r.skip(skip)?;
                        }
                        if header > 3 {
                            return Err(Error::corrupt("unrecognised header flags in point data leaf"));
                        }
                        d
                    }
                };
                st.headers = (0..descriptor.size()).map(|_| read_array_header(r)).collect::<Result<_>>()?;
                st.handles = vec![None; descriptor.size()];
                st.arrays = (0..descriptor.size()).map(|_| None).collect();
                st.descriptor = Some(descriptor);
            }
        } else if pass < attributes.wrapping_add(2) {
            let idx = (pass - 2) as usize;
            for st in states.iter_mut() {
                if idx < st.headers.len() && st.headers[idx].paged {
                    let n = st.headers[idx].bytes;
                    st.handles[idx] = Some(paged.entry(idx).or_default().create_handle(r, n)?);
                }
            }
        } else if pass == attributes + 2 {
            for leaf in leaves.iter_mut() {
                read_mask(r, leaf.value_mask_mut())?;
                let bytes16 = r.read_u16()?;
                let raw = if bytes16 == u16::MAX {
                    r.read_bytes(512 * 4)?
                } else {
                    let frame = r.read_bytes(bytes16 as usize)?;
                    blosc_decompress(&frame, 512 * 4)?
                };
                for (v, c) in leaf.buffer_mut().iter_mut().zip(raw.chunks_exact(4)) {
                    *v = u32::from_le_bytes(c.try_into().unwrap());
                }
            }
        } else if pass < attributes * 2 + 3 {
            let idx = (pass - attributes - 3) as usize;
            for st in states.iter_mut() {
                if idx >= st.headers.len() {
                    continue;
                }
                let h = st.headers[idx].clone();
                let ty = st.descriptor.as_ref().unwrap().attribute_type(idx).clone();
                let storage_size =
                    ty.storage_size().ok_or_else(|| Error::Unsupported(format!("attribute type \"{}\"", ty)))?;
                let data = if h.paged {
                    let handle = st.handles[idx].ok_or_else(|| Error::corrupt("missing page handle"))?;
                    paged.entry(idx).or_default().read(r, handle)?
                } else {
                    read_array_buffers(r, &h, storage_size)?
                };
                let array = AttributeArray::from_raw(ty, h.size, h.stride_or_total, h.flags, h.uniform, data)
                    .map_err(|e| Error::corrupt(e.to_string()))?;
                st.arrays[idx] = Some(array);
            }
            if idx > 0 {
                paged.remove(&(idx - 1));
            }
        }
    }

    let mut attrs = HashMap::with_capacity(leaves.len());
    for (leaf, st) in leaves.iter().zip(states) {
        let set = match st.descriptor {
            Some(d) => {
                let arrays = st
                    .arrays
                    .into_iter()
                    .map(|a| a.ok_or_else(|| Error::corrupt("missing attribute array data")))
                    .collect::<Result<Vec<_>>>()?;
                AttributeSet::from_parts(d, arrays)
            }
            None => AttributeSet::new(),
        };
        attrs.insert(leaf.origin(), set);
    }
    drop(leaves);
    Ok(PointDataTree::from_parts(tree, attrs))
}

/// The attribute set of a leaf without its transient arrays.
#[cfg(feature = "write")]
fn persistent_set(set: &AttributeSet) -> Cow<'_, AttributeSet> {
    let transient: Vec<usize> = (0..set.size()).filter(|&i| set.get(i).is_some_and(|a| a.is_transient())).collect();
    if transient.is_empty() {
        return Cow::Borrowed(set);
    }
    let mut s = set.clone();
    // Dropping transient arrays from a valid descriptor cannot fail.
    let _ = s.drop_attributes(&transient);
    Cow::Owned(s)
}

/// Write a point data tree's topology and multi-pass buffers. `block_pos` receives the offset
/// of the buffers.
#[cfg(feature = "write")]
pub(crate) fn write_point_data_tree(
    out: &mut Vec<u8>,
    tree: &PointDataTree,
    compression: Compression,
    block_pos: &mut i64,
) {
    let offsets: &Tree<u32> = tree.offsets();
    let ctx = write_ctx(offsets, compression, false, LeafFormat::Standard);
    write_tree_topology(out, offsets, &ctx);
    *block_pos = out.len() as i64;

    let blosc = compression.contains(Compression::BLOSC);
    let leaves = tree.leaf_vec();
    let sets: Vec<Cow<'_, AttributeSet>> = leaves.iter().map(|l| persistent_set(l.attribute_set())).collect();
    if leaves.is_empty() {
        out.put_u16(0);
        return;
    }
    let attributes = sets.iter().map(|s| s.size()).max().unwrap_or(0);
    out.put_u16((4 + 2 * attributes) as u16);

    // Voxel offsets, compressed in parallel.
    let voxels: Vec<(Vec<u8>, Option<Vec<u8>>)> = leaves
        .par_iter()
        .map(|l| {
            let raw: Vec<u8> = l.leaf().buffer().iter().flat_map(|v| v.to_le_bytes()).collect();
            let c = blosc_compress(&raw);
            (raw, c)
        })
        .collect();

    // Pass 0: voxel buffer sizes.
    for (_, c) in &voxels {
        out.put_u16(c.as_ref().map(|c| c.len() as u16).unwrap_or(u16::MAX));
    }
    // Pass 1: descriptors and array headers.
    let first = sets[0].descriptor();
    let matching = sets.iter().all(|s| s.descriptor() == first);
    for (k, set) in sets.iter().enumerate() {
        if !matching {
            out.put_u8(0);
            write_descriptor(out, set.descriptor());
        } else if k == 0 {
            out.put_u8(1);
            write_descriptor(out, set.descriptor());
        }
        for a in set.arrays() {
            write_array_header(out, a, blosc);
        }
    }
    // Passes 2..: page headers.
    if blosc {
        for idx in 0..attributes {
            let mut ps = PagedOutputStream::new(true);
            for set in &sets {
                if let Some(a) = set.get(idx) {
                    ps.write(out, a.storage());
                }
            }
            ps.flush(out);
        }
    }
    // Voxel offsets.
    for (leaf, (raw, c)) in leaves.iter().zip(&voxels) {
        write_mask(out, leaf.leaf().value_mask());
        match c {
            Some(c) => {
                out.put_u16(c.len() as u16);
                out.extend_from_slice(c);
            }
            None => {
                out.put_u16(u16::MAX);
                out.extend_from_slice(raw);
            }
        }
    }
    // Attribute data.
    for idx in 0..attributes {
        if blosc {
            let mut ps = PagedOutputStream::new(false);
            for set in &sets {
                if let Some(a) = set.get(idx) {
                    ps.write(out, a.storage());
                }
            }
            ps.flush(out);
        } else {
            for set in &sets {
                if let Some(a) = set.get(idx) {
                    write_array_buffers(out, a);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Point index trees

/// Read a point index tree (`PointIndexLeafNode::readBuffers`: the standard leaf buffers
/// followed by the leaf's index array and an auxiliary data block).
#[cfg(feature = "read")]
pub(crate) fn read_point_index_tree<R: Read>(
    r: &mut Reader<R>,
    file_version: u32,
    compression: Compression,
) -> Result<PointIndexTree> {
    let (mut tree, ctx) =
        read_tree_topology::<u32, Internal2<u32>, R>(r, file_version, compression, false, LeafFormat::Standard)?;
    let mut indices = HashMap::new();
    for leaf in tree.leaf_refs_mut() {
        leaf.read_buffers(r, &ctx)?;
        let n = r.read_u64()? as usize;
        let bytes = r.read_bytes(n.checked_mul(4).ok_or_else(|| Error::corrupt("index count overflow"))?)?;
        let idx: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let aux = r.read_u64()?;
        r.skip(aux)?;
        indices.insert(leaf.origin(), idx);
    }
    Ok(PointIndexTree::from_parts(tree, indices))
}

/// Write a point index tree's topology and buffers.
#[cfg(feature = "write")]
pub(crate) fn write_point_index_tree(
    out: &mut Vec<u8>,
    tree: &PointIndexTree,
    compression: Compression,
    block_pos: &mut i64,
) {
    let offsets = tree.tree();
    let ctx = write_ctx(offsets, compression, false, LeafFormat::Standard);
    write_tree_topology(out, offsets, &ctx);
    *block_pos = out.len() as i64;
    for leaf in offsets.leaves() {
        leaf.write_buffers(out, &ctx);
        let idx = tree.indices_map().get(&leaf.origin()).map(|v| v.as_slice()).unwrap_or(&[]);
        out.put_u64(idx.len() as u64);
        for i in idx {
            out.put_u32(*i);
        }
        out.put_u64(0); // auxiliary data, reserved
    }
}

#[cfg(all(test, feature = "read", feature = "write"))]
mod tests {
    use super::*;
    use crate::math::{Coord, Vec3f};
    use crate::points::{AttributeArray, Codec, PointDataLeafNode, PointLeafMut};

    fn sample_tree(n_leaves: i32, points_per_leaf: u32) -> PointDataTree {
        let mut tree = PointDataTree::new();
        let d = Arc::new(Descriptor::with_position(AttributeType::of::<Vec3f>(Codec::FIXED_POINT_16)).unwrap());
        for k in 0..n_leaves {
            let mut node = PointDataLeafNode::new(Coord::new(8 * k, -8 * (k % 3), 16));
            node.initialize_attributes(d.clone(), points_per_leaf).unwrap();
            let mut offsets = vec![points_per_leaf; 512];
            for (i, o) in offsets.iter_mut().enumerate().take(points_per_leaf as usize) {
                *o = i as u32 + 1;
            }
            node.set_offsets(&offsets, true).unwrap();
            let p = node.attribute_array_mut("P").unwrap();
            for i in 0..points_per_leaf {
                p.set(i, Vec3f::new(0.1 * (i % 5) as f32 - 0.2, 0.0, (k % 7) as f32 * 0.05));
            }
            let mut id = AttributeArray::with_size::<i32>(Codec::Null, points_per_leaf).unwrap();
            for i in 0..points_per_leaf {
                id.set(i, k * 1000 + i as i32);
            }
            node.attributes.append_attribute("id", AttributeType::of::<i32>(Codec::Null), 1, true, None).unwrap();
            node.attributes.replace(1, id).unwrap();
            tree.add_leaf(node);
        }
        tree
    }

    fn roundtrip(tree: &PointDataTree, c: Compression) -> PointDataTree {
        let mut out = Vec::new();
        let mut bp = 0;
        write_point_data_tree(&mut out, tree, c, &mut bp);
        let mut r = Reader::new(&out[..]);
        let back = read_point_data_tree(&mut r, 224, c).unwrap();
        assert_eq!(r.pos() as usize, out.len());
        back
    }

    #[test]
    fn blosc_helpers() {
        assert!(blosc_compress(&[0u8; 48]).is_none());
        let small = [7u8; 100];
        let f = blosc_compress(&small).unwrap();
        assert_eq!(blosc_decompress(&f, 100).unwrap(), small.to_vec());
        let noise: Vec<u8> = (0..1000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        if let Some(f) = blosc_compress(&noise) {
            assert_eq!(blosc_decompress(&f, 1000).unwrap(), noise);
        }
    }

    #[test]
    fn tree_roundtrip_all_compressions() {
        let tree = sample_tree(5, 37);
        for c in [
            Compression::NONE,
            Compression::ZIP | Compression::ACTIVE_MASK,
            Compression::BLOSC | Compression::ACTIVE_MASK,
        ] {
            let back = roundtrip(&tree, c);
            assert!(back == tree, "compression {}", c);
        }
        let empty = PointDataTree::new();
        assert!(roundtrip(&empty, Compression::BLOSC).is_empty());
    }

    #[test]
    fn multi_page_roundtrip() {
        // 300 leaves × 512 points × 6 bytes of positions span two pages.
        let tree = sample_tree(300, 512);
        let back = roundtrip(&tree, Compression::BLOSC);
        assert!(back == tree);
    }
}
