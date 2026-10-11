//! A pure-Rust implementation of the subset of the [Blosc](https://www.blosc.org) 1.x
//! container format used by OpenVDB.
//!
//! OpenVDB compresses node buffers with
//! `blosc_compress_ctx(9, BLOSC_SHUFFLE, sizeof(float), n, src, dst, n + 16, "lz4", n, 1)`,
//! i.e. one block per buffer, byte shuffle with a type size of 4, and the LZ4 codec.
//!
//! A Blosc 1.x frame is a 16-byte header
//!
//! | offset | size | field |
//! |--------|------|-------|
//! | 0 | 1 | format version (2) |
//! | 1 | 1 | codec format version |
//! | 2 | 1 | flags: `0x1` byte shuffle, `0x2` memcpy'd, `0x4` bit shuffle, `0x10` no split, bits 5-7 codec |
//! | 3 | 1 | type size |
//! | 4 | 4 | uncompressed size |
//! | 8 | 4 | block size |
//! | 12 | 4 | total compressed size (including the header) |
//!
//! followed (unless the frame is memcpy'd) by one `i32` start offset per block. Each block
//! holds one or (when split) `typesize` streams, each prefixed by its compressed size; a
//! stream whose size equals its uncompressed size is stored raw, and a non-positive size
//! encodes a run of a single byte.
//!
//! Decompression supports the BloscLZ, LZ4 (and LZ4HC) and zlib codecs, byte shuffling,
//! split and unsplit blocks, multiple blocks and memcpy'd frames. Compression always emits
//! a single LZ4 block, matching what OpenVDB writes.

use super::error::{Error, Result};

/// Size of a Blosc 1.x header; also the maximum overhead of a compressed frame.
pub const MAX_OVERHEAD: usize = 16;

const MIN_BUFFERSIZE: usize = 128;
const MAX_SPLITS: usize = 16;
const FORMAT_VERSION: u8 = 2;
const LZ4_VERSION_FORMAT: u8 = 1;

const DOSHUFFLE: u8 = 0x1;
const MEMCPYED: u8 = 0x2;
const DOBITSHUFFLE: u8 = 0x4;
const NOSPLIT: u8 = 0x10;

const CODEC_BLOSCLZ: u8 = 0;
const CODEC_LZ4: u8 = 1;
const CODEC_ZLIB: u8 = 3;

fn err(msg: impl Into<String>) -> Error {
    Error::Blosc(msg.into())
}

#[inline]
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The uncompressed size recorded in a Blosc frame header.
pub fn decompressed_size(src: &[u8]) -> Result<usize> {
    if src.len() < MAX_OVERHEAD {
        return Err(err("frame shorter than its header"));
    }
    Ok(u32_at(src, 4) as usize)
}

/// Decompress a Blosc 1.x frame.
pub fn decompress(src: &[u8]) -> Result<Vec<u8>> {
    let nbytes = decompressed_size(src)?;
    let version = src[0];
    if version == 0 || version > FORMAT_VERSION {
        return Err(err(format!("unsupported Blosc format version {}", version)));
    }
    let flags = src[2];
    let typesize = (src[3] as usize).max(1);
    let blocksize = u32_at(src, 8) as usize;
    let ctbytes = u32_at(src, 12) as usize;
    if ctbytes > src.len() {
        return Err(err("frame is truncated"));
    }
    let src = &src[..ctbytes.max(MAX_OVERHEAD)];
    let mut dest = vec![0u8; nbytes];
    if nbytes == 0 {
        return Ok(dest);
    }
    if flags & MEMCPYED != 0 {
        if src.len() < MAX_OVERHEAD + nbytes {
            return Err(err("memcpy'd frame is truncated"));
        }
        dest.copy_from_slice(&src[MAX_OVERHEAD..MAX_OVERHEAD + nbytes]);
        return Ok(dest);
    }
    if flags & DOBITSHUFFLE != 0 {
        return Err(err("bit shuffle is not supported"));
    }
    if blocksize == 0 {
        return Err(err("zero block size"));
    }
    let codec = flags >> 5;
    let leftover = nbytes % blocksize;
    let nblocks = nbytes / blocksize + usize::from(leftover > 0);
    if src.len() < MAX_OVERHEAD + 4 * nblocks {
        return Err(err("frame is truncated"));
    }
    let mut tmp = vec![0u8; blocksize];
    for j in 0..nblocks {
        let is_leftover = leftover > 0 && j == nblocks - 1;
        let bsize = if is_leftover { leftover } else { blocksize };
        let start = u32_at(src, MAX_OVERHEAD + 4 * j) as usize;
        let dont_split = flags & NOSPLIT != 0;
        let nsplits = if !dont_split && typesize <= MAX_SPLITS && bsize / typesize >= MIN_BUFFERSIZE && !is_leftover {
            typesize
        } else {
            1
        };
        let neblock = bsize / nsplits;
        let mut p = start;
        for s in 0..nsplits {
            if p + 4 > src.len() {
                return Err(err("frame is truncated"));
            }
            let cbytes = i32::from_le_bytes([src[p], src[p + 1], src[p + 2], src[p + 3]]);
            p += 4;
            let out = &mut tmp[s * neblock..(s + 1) * neblock];
            if cbytes <= 0 {
                // A run of a single byte value.
                if cbytes < -255 {
                    return Err(err("invalid run length"));
                }
                out.fill((-cbytes) as u8);
                continue;
            }
            let cbytes = cbytes as usize;
            if p + cbytes > src.len() {
                return Err(err("frame is truncated"));
            }
            let input = &src[p..p + cbytes];
            p += cbytes;
            if cbytes == neblock {
                out.copy_from_slice(input);
                continue;
            }
            let n = match codec {
                CODEC_BLOSCLZ => blosclz_decompress(input, out).ok_or_else(|| err("corrupt BloscLZ stream"))?,
                CODEC_LZ4 => lz4_flex::block::decompress_into(input, out).map_err(|e| err(format!("LZ4: {}", e)))?,
                CODEC_ZLIB => zlib_decompress(input, out)?,
                c => return Err(err(format!("unsupported Blosc codec {}", c))),
            };
            if n != neblock {
                return Err(err("decompressed stream has the wrong size"));
            }
        }
        let dst = &mut dest[j * blocksize..j * blocksize + bsize];
        if flags & DOSHUFFLE != 0 && typesize > 1 {
            unshuffle(typesize, &tmp[..bsize], dst);
        } else {
            dst.copy_from_slice(&tmp[..bsize]);
        }
    }
    Ok(dest)
}

/// Compress `src` into a Blosc frame as OpenVDB does: byte shuffle with the given type
/// size, LZ4, and a single block. Small or incompressible inputs produce a memcpy'd frame.
pub fn compress(src: &[u8], typesize: usize) -> Vec<u8> {
    let nbytes = src.len();
    let mut typesize = typesize.clamp(1, 255);
    if !nbytes.is_multiple_of(typesize) {
        // Splitting and shuffling need whole elements; fall back to bytes.
        typesize = 1;
    }
    let header = |flags: u8, ctbytes: usize| -> Vec<u8> {
        let mut h = Vec::with_capacity(ctbytes);
        h.extend_from_slice(&[FORMAT_VERSION, LZ4_VERSION_FORMAT, flags, typesize as u8]);
        h.extend_from_slice(&(nbytes as u32).to_le_bytes());
        h.extend_from_slice(&(nbytes as u32).to_le_bytes());
        h.extend_from_slice(&(ctbytes as u32).to_le_bytes());
        h
    };
    let memcpyed = || {
        let mut out = header(DOSHUFFLE | MEMCPYED | NOSPLIT | (CODEC_LZ4 << 5), MAX_OVERHEAD + nbytes);
        out.extend_from_slice(src);
        out
    };
    if nbytes < MIN_BUFFERSIZE {
        return memcpyed();
    }
    let split = typesize <= MAX_SPLITS && nbytes / typesize >= MIN_BUFFERSIZE;
    let nsplits = if split { typesize } else { 1 };
    let flags = DOSHUFFLE | (CODEC_LZ4 << 5) | if split { 0 } else { NOSPLIT };
    let shuffled;
    let data = if typesize > 1 {
        let mut s = vec![0u8; nbytes];
        shuffle(typesize, src, &mut s);
        shuffled = s;
        &shuffled[..]
    } else {
        src
    };
    let mut out = header(flags, 0);
    out.extend_from_slice(&((MAX_OVERHEAD + 4) as u32).to_le_bytes()); // start of block 0
    let neblock = nbytes / nsplits;
    for part in data.chunks_exact(neblock) {
        let c = lz4_flex::block::compress(part);
        if c.is_empty() || c.len() >= neblock {
            out.extend_from_slice(&(neblock as i32).to_le_bytes());
            out.extend_from_slice(part);
        } else {
            out.extend_from_slice(&(c.len() as i32).to_le_bytes());
            out.extend_from_slice(&c);
        }
        if out.len() > nbytes + MAX_OVERHEAD {
            return memcpyed();
        }
    }
    let ctbytes = out.len() as u32;
    out[12..16].copy_from_slice(&ctbytes.to_le_bytes());
    out
}

/// Byte shuffle: gather byte `j` of every element into a contiguous run.
fn shuffle(typesize: usize, src: &[u8], dest: &mut [u8]) {
    let n = src.len() / typesize;
    for i in 0..n {
        for j in 0..typesize {
            dest[j * n + i] = src[i * typesize + j];
        }
    }
    let tail = n * typesize;
    dest[tail..].copy_from_slice(&src[tail..]);
}

/// Inverse of [`shuffle`].
fn unshuffle(typesize: usize, src: &[u8], dest: &mut [u8]) {
    let n = src.len() / typesize;
    for i in 0..n {
        for j in 0..typesize {
            dest[i * typesize + j] = src[j * n + i];
        }
    }
    let tail = n * typesize;
    dest[tail..].copy_from_slice(&src[tail..]);
}

fn zlib_decompress(input: &[u8], out: &mut [u8]) -> Result<usize> {
    use std::io::Read;
    let mut d = flate2::read::ZlibDecoder::new(input);
    let mut n = 0;
    while n < out.len() {
        let k = d.read(&mut out[n..]).map_err(|e| err(format!("zlib: {}", e)))?;
        if k == 0 {
            break;
        }
        n += k;
    }
    Ok(n)
}

/// Decompress a BloscLZ stream (a FastLZ-derived LZ77 format) into `out`, returning the
/// number of bytes produced, or `None` if the stream is malformed.
fn blosclz_decompress(input: &[u8], out: &mut [u8]) -> Option<usize> {
    const MAX_DISTANCE: usize = 8191;
    if input.is_empty() {
        return Some(0);
    }
    let ip_limit = input.len();
    let mut ip = 0usize;
    let mut op = 0usize;
    let mut ctrl = (input[ip] & 31) as usize;
    ip += 1;
    loop {
        if ctrl >= 32 {
            // Back reference.
            let mut len = (ctrl >> 5) - 1;
            let mut ofs = (ctrl & 31) << 8;
            if len == 7 - 1 {
                loop {
                    if ip + 1 >= ip_limit {
                        return None;
                    }
                    let code = input[ip];
                    ip += 1;
                    len += code as usize;
                    if code != 255 {
                        break;
                    }
                }
            } else if ip + 1 >= ip_limit {
                return None;
            }
            let code = input[ip] as usize;
            ip += 1;
            len += 3;
            let mut back = ofs + code;
            if code == 255 && ofs == (31 << 8) {
                // Match from a 16-bit distance.
                if ip + 1 >= ip_limit {
                    return None;
                }
                ofs = ((input[ip] as usize) << 8) + input[ip + 1] as usize;
                ip += 2;
                back = ofs + MAX_DISTANCE;
            }
            let dist = back + 1;
            if op + len > out.len() || dist > op {
                return None;
            }
            for k in op..op + len {
                out[k] = out[k - dist];
            }
            op += len;
            if ip >= ip_limit {
                break;
            }
            ctrl = input[ip] as usize;
            ip += 1;
        } else {
            // Literal run.
            ctrl += 1;
            if op + ctrl > out.len() || ip + ctrl > ip_limit {
                return None;
            }
            out[op..op + ctrl].copy_from_slice(&input[ip..ip + ctrl]);
            op += ctrl;
            ip += ctrl;
            if ip >= ip_limit {
                break;
            }
            ctrl = input[ip] as usize;
            ip += 1;
        }
    }
    Some(op)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i / 7) as u8).wrapping_mul(31) ^ if i % 13 == 0 { 0x5a } else { 0 }).collect()
    }

    #[test]
    fn roundtrip_sizes() {
        for &n in &[0usize, 1, 17, 64, 127, 128, 129, 512, 1026, 2048, 4095, 6000] {
            for &ts in &[1usize, 2, 4, 12] {
                let src = sample(n);
                let c = compress(&src, ts);
                assert_eq!(decompressed_size(&c).unwrap(), n);
                assert_eq!(decompress(&c).unwrap(), src, "n={} ts={}", n, ts);
            }
        }
    }

    #[test]
    fn compresses_redundant_data() {
        let src: Vec<u8> = (0..2048u32).flat_map(|i| ((i % 16) as f32).to_le_bytes()).collect();
        let c = compress(&src, 4);
        assert!(c.len() < src.len() / 4);
        assert_eq!(c[2] & MEMCPYED, 0);
        assert_eq!(decompress(&c).unwrap(), src);
    }

    #[test]
    fn shuffle_inverse() {
        let src = sample(1003);
        let mut s = vec![0; src.len()];
        let mut u = vec![0; src.len()];
        shuffle(4, &src, &mut s);
        unshuffle(4, &s, &mut u);
        assert_eq!(src, u);
    }

    #[test]
    fn rejects_garbage() {
        assert!(decompress(&[1, 2, 3]).is_err());
        let mut c = compress(&sample(4096), 4);
        c.truncate(40);
        assert!(decompress(&c).is_err());
    }

    #[test]
    fn blosclz_literals_and_matches() {
        // Literal run "abc", then a match of length 3 at distance 3 ("abc").
        let stream = [2u8, b'a', b'b', b'c', 1 << 5, 2, 0, b'x'];
        let mut out = [0u8; 7];
        // ctrl=(1<<5): len=(1)-1=0 -> +3 = 3, ofs=0, code=2 -> dist 3
        assert_eq!(blosclz_decompress(&stream, &mut out), Some(7));
        assert_eq!(&out, b"abcabcx");
    }
}
