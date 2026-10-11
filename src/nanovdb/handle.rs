//! [`GridHandle`]: owner of a 32-byte aligned buffer holding one or more NanoVDB grids
//! (port of `nanovdb/GridHandle.h` with a host buffer).

use super::build_types::BuildType;
use super::data::{grid_off, rd_u32, rd_u64};
use super::grid::{GridHeader, NanoGrid};
use super::{NanoError, Result, DATA_ALIGNMENT, GRID_DATA_SIZE};

/// A 32-byte block; the unit of allocation of [`AlignedBuffer`].
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Block([u8; DATA_ALIGNMENT]);

/// A heap buffer of bytes whose start is aligned to [`DATA_ALIGNMENT`] (32) bytes, as
/// required by the NanoVDB memory layout.
#[derive(Clone, Default)]
pub struct AlignedBuffer {
    blocks: Vec<Block>,
    len: usize,
}

impl AlignedBuffer {
    /// A zero-filled buffer of `len` bytes.
    pub fn zeroed(len: usize) -> Self {
        AlignedBuffer { blocks: vec![Block([0; DATA_ALIGNMENT]); len.div_ceil(DATA_ALIGNMENT)], len }
    }
    /// A buffer holding a copy of `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut b = Self::zeroed(bytes.len());
        b.as_mut_slice().copy_from_slice(bytes);
        b
    }
    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }
    /// True if the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The bytes.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `Block` is a `repr(C)` wrapper of `[u8; 32]` without padding, so the vector's
        // allocation is `blocks.len() * 32` initialized bytes, and `len <= blocks.len() * 32`.
        // The returned slice borrows `self`, so the allocation outlives it.
        unsafe { std::slice::from_raw_parts(self.blocks.as_ptr() as *const u8, self.len) }
    }
    /// The bytes, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_slice`; the exclusive borrow of `self` guarantees uniqueness.
        unsafe { std::slice::from_raw_parts_mut(self.blocks.as_mut_ptr() as *mut u8, self.len) }
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuffer({} bytes)", self.len)
    }
}

/// Owns a buffer containing one or more consecutive NanoVDB grids and gives typed,
/// zero-copy access to them.
#[derive(Clone, Debug, Default)]
pub struct GridHandle {
    buffer: AlignedBuffer,
    /// Byte offset of each grid in the buffer.
    offsets: Vec<usize>,
}

impl GridHandle {
    /// Wrap a buffer holding one or more grids, validating their headers.
    pub fn new(buffer: AlignedBuffer) -> Result<Self> {
        let mut offsets = Vec::new();
        let bytes = buffer.as_slice();
        let mut off = 0usize;
        while off < bytes.len() {
            if bytes.len() - off < GRID_DATA_SIZE {
                return Err(NanoError::InvalidGrid("truncated grid header".into()));
            }
            let header = GridHeader::new(&bytes[off..off + GRID_DATA_SIZE]);
            if !header.is_valid() {
                return Err(NanoError::InvalidGrid(format!("bad magic or version at byte {}", off)));
            }
            let size = rd_u64(bytes, off + grid_off::GRID_SIZE) as usize;
            if size < GRID_DATA_SIZE || off + size > bytes.len() {
                return Err(NanoError::InvalidGrid(format!("grid size {} exceeds buffer", size)));
            }
            offsets.push(off);
            off += size;
        }
        Ok(GridHandle { buffer, offsets })
    }

    /// Wrap a copy of `bytes` (which may be unaligned).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::new(AlignedBuffer::from_bytes(bytes))
    }

    /// Concatenate the grids of several handles into one buffer, updating each grid's index
    /// and count (and checksum head) as `nanovdb::mergeGrids` does.
    pub fn merge(handles: &[GridHandle]) -> Result<Self> {
        let total: usize = handles.iter().map(|h| h.size()).sum();
        let count: usize = handles.iter().map(|h| h.grid_count()).sum();
        let mut buffer = AlignedBuffer::zeroed(total);
        let mut pos = 0;
        let mut index = 0u32;
        for h in handles {
            for n in 0..h.grid_count() {
                let src = h.grid_bytes(n);
                let dst = &mut buffer.as_mut_slice()[pos..pos + src.len()];
                dst.copy_from_slice(src);
                super::checksum::update_grid_count(dst, index, count as u32);
                index += 1;
                pos += src.len();
            }
        }
        Self::new(buffer)
    }

    /// Split a handle into one handle per grid (`nanovdb::splitGrids`).
    pub fn split(&self) -> Vec<GridHandle> {
        (0..self.grid_count())
            .map(|n| {
                let mut b = AlignedBuffer::from_bytes(self.grid_bytes(n));
                super::checksum::update_grid_count(b.as_mut_slice(), 0, 1);
                GridHandle::new(b).expect("valid grid")
            })
            .collect()
    }

    /// The whole buffer.
    pub fn data(&self) -> &[u8] {
        self.buffer.as_slice()
    }
    /// The whole buffer, mutably. Callers must keep the grids valid.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }
    /// The underlying aligned buffer.
    pub fn buffer(&self) -> &AlignedBuffer {
        &self.buffer
    }
    /// Consume the handle, returning its buffer.
    pub fn into_buffer(self) -> AlignedBuffer {
        self.buffer
    }
    /// Total size in bytes.
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
    /// True if the handle holds no grid.
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
    /// Number of grids in the buffer.
    pub fn grid_count(&self) -> usize {
        self.offsets.len()
    }
    /// Size in bytes of grid `n`.
    pub fn grid_size(&self, n: usize) -> usize {
        rd_u64(self.data(), self.offsets[n] + grid_off::GRID_SIZE) as usize
    }
    /// The bytes of grid `n`.
    pub fn grid_bytes(&self, n: usize) -> &[u8] {
        let o = self.offsets[n];
        &self.data()[o..o + self.grid_size(n)]
    }
    /// Mutable bytes of grid `n`.
    pub fn grid_bytes_mut(&mut self, n: usize) -> &mut [u8] {
        let o = self.offsets[n];
        let s = self.grid_size(n);
        &mut self.buffer.as_mut_slice()[o..o + s]
    }
    /// Untyped header of grid `n` (`GridHandle::gridMetaData`).
    pub fn header(&self, n: usize) -> Option<GridHeader<'_>> {
        (n < self.grid_count()).then(|| GridHeader::new(self.grid_bytes(n)))
    }
    /// Typed view of grid `n`, or `None` if it does not exist or has a different type.
    pub fn grid<B: BuildType>(&self, n: usize) -> Option<NanoGrid<'_, B>> {
        if n >= self.grid_count() {
            return None;
        }
        NanoGrid::new(self.grid_bytes(n)).ok()
    }
    /// The grid type of grid `n`.
    pub fn grid_type(&self, n: usize) -> Option<super::GridType> {
        (n < self.grid_count())
            .then(|| super::GridType::from_u32(rd_u32(self.data(), self.offsets[n] + grid_off::GRID_TYPE)))
    }
    /// Write the raw buffer to a file (`GridHandle::write`; readable by [`io::read_grids`](super::io::read_grids)).
    pub fn write_raw<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        std::fs::write(path, self.data())?;
        Ok(())
    }
}
