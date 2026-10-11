//! CRC32 checksums of NanoVDB grids (port of `nanovdb/tools/GridChecksum.h`).
//!
//! Since v32.6 the checksum consists of two CRC32 codes: the *head* covers the grid and
//! tree headers (excluding the magic number and the checksum itself), and the *tail* covers
//! everything after them (root, nodes and blind data), computed as the CRC32 of the CRC32s
//! of 4 KB blocks.

use super::data::{grid_off, rd_u32, rd_u64, wr_u32, wr_u64, Checksum, Version};
use super::{CheckMode, GRID_DATA_SIZE, TREE_DATA_SIZE};

const fn make_lut() -> [u32; 256] {
    let mut lut = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut cs = n as u32;
        let mut i = 0;
        while i < 8 {
            cs = (cs >> 1) ^ (if cs & 1 != 0 { 0xEDB88320 } else { 0 });
            i += 1;
        }
        lut[n] = cs;
        n += 1;
    }
    lut
}

static CRC32_LUT: [u32; 256] = make_lut();

/// Standard (zlib) CRC32 of `data`, continuing from `crc`.
pub fn crc32(data: &[u8], crc: u32) -> u32 {
    let mut crc = !crc;
    for &b in data {
        crc = CRC32_LUT[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

const LOG2_BLOCK_SIZE: usize = 12;

/// CRC32 of the CRC32s of consecutive 4 KB blocks; the last block absorbs the remainder
/// (`util::blockedCrc32`).
fn blocked_crc32(data: &[u8]) -> u32 {
    if data.is_empty() {
        return !0;
    }
    let block_count = data.len() >> LOG2_BLOCK_SIZE;
    let mut sums = Vec::with_capacity(4 * block_count);
    for i in 0..block_count {
        let start = i << LOG2_BLOCK_SIZE;
        let end = if i + 1 == block_count { data.len() } else { start + (1 << LOG2_BLOCK_SIZE) };
        sums.extend_from_slice(&crc32(&data[start..end], 0).to_le_bytes());
    }
    crc32(&sums, 0)
}

/// CRC32 of the grid head (`crc32Head`).
fn crc32_head(grid: &[u8]) -> u32 {
    crc32(&grid[16..GRID_DATA_SIZE + TREE_DATA_SIZE], 0)
}

/// CRC32 of the grid tail (`crc32Tail`, format of version > 32.6.0).
fn crc32_tail(grid: &[u8]) -> u32 {
    let size = rd_u64(grid, grid_off::GRID_SIZE) as usize;
    blocked_crc32(&grid[GRID_DATA_SIZE + TREE_DATA_SIZE..size])
}

/// Compute the checksum of a grid with the given mode (`evalChecksum`).
///
/// Only the checksum layout of version > 32.6.0 is supported; older grids yield a partial
/// checksum computed as if they were current.
pub fn eval_checksum(grid: &[u8], mode: CheckMode) -> Checksum {
    match mode {
        CheckMode::Disable => Checksum::EMPTY,
        CheckMode::Partial => Checksum::from_parts(crc32_head(grid), Checksum::EMPTY32),
        CheckMode::Full => Checksum::from_parts(crc32_head(grid), crc32_tail(grid)),
    }
}

/// Recompute and store the checksum of a grid (`updateChecksum`).
pub fn update_checksum(grid: &mut [u8], mode: CheckMode) {
    let cs = eval_checksum(grid, mode);
    wr_u64(grid, grid_off::CHECKSUM, cs.0);
}

/// Validate the stored checksum of a grid (`validateChecksum`). Disabled checksums are
/// always valid; `mode` may restrict the check to the head.
pub fn validate_checksum(grid: &[u8], mode: CheckMode) -> bool {
    let cs = Checksum(rd_u64(grid, grid_off::CHECKSUM));
    if cs.is_empty() || mode == CheckMode::Disable {
        return true;
    }
    if Version(rd_u32(grid, grid_off::VERSION)) <= Version::new(32, 6, 0) {
        // Older checksum layouts are not supported.
        return true;
    }
    let head_ok = cs.head() == crc32_head(grid);
    if cs.is_half() || mode == CheckMode::Partial || !head_ok {
        head_ok
    } else {
        cs.tail() == crc32_tail(grid)
    }
}

/// Set a grid's index and count, updating the checksum head if needed (`updateGridCount`).
pub fn update_grid_count(grid: &mut [u8], index: u32, count: u32) {
    if rd_u32(grid, grid_off::GRID_INDEX) != index || rd_u32(grid, grid_off::GRID_COUNT) != count {
        wr_u32(grid, grid_off::GRID_INDEX, index);
        wr_u32(grid, grid_off::GRID_COUNT, count);
        let cs = Checksum(rd_u64(grid, grid_off::CHECKSUM));
        if !cs.is_empty() {
            let new = Checksum::from_parts(crc32_head(grid), cs.tail());
            wr_u64(grid, grid_off::CHECKSUM, new.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        assert_eq!(crc32(b"123456789", 0), 0xCBF43926);
        assert_eq!(crc32(b"", 0), 0);
        assert_eq!(blocked_crc32(&[]), !0);
        // Fewer than 4096 bytes: no blocks, CRC32 of nothing.
        assert_eq!(blocked_crc32(&[1, 2, 3]), 0);
    }
}
