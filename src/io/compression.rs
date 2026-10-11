//! Compression of node value buffers (port of `openvdb/io/Compression.h/.cc`).

use super::blosc;
use super::bytes::{Reader, WriteLe};
use super::error::{Error, Result};
use super::value_io::IoValue;
use crate::util::NodeMask;
use std::cell::RefCell;
use std::fmt;
use std::io::{Read, Write};

/// Compression flags of a grid (OpenVDB's `COMPRESS_*` bit set).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Compression(pub u32);

impl Compression {
    /// No compression.
    pub const NONE: Compression = Compression(0);
    /// zlib compression of node value buffers (not applied to level sets and fog volumes).
    pub const ZIP: Compression = Compression(0x1);
    /// Omit inactive values that can be reconstructed from the background value.
    pub const ACTIVE_MASK: Compression = Compression(0x2);
    /// Blosc (shuffle + LZ4) compression of node value buffers.
    pub const BLOSC: Compression = Compression(0x4);

    /// The raw flag bits.
    pub fn bits(self) -> u32 {
        self.0
    }
    /// True if all flags in `other` are set.
    pub fn contains(self, other: Compression) -> bool {
        self.0 & other.0 == other.0 && other.0 != 0
    }
    /// These flags without those in `other`.
    pub fn without(self, other: Compression) -> Compression {
        Compression(self.0 & !other.0)
    }
}

impl std::ops::BitOr for Compression {
    type Output = Compression;
    fn bitor(self, o: Compression) -> Compression {
        Compression(self.0 | o.0)
    }
}

impl fmt::Display for Compression {
    /// Matches OpenVDB's `compressionToString()`, e.g. `"zip + active values"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return f.write_str("none");
        }
        let mut parts = Vec::new();
        if self.contains(Self::ZIP) {
            parts.push("zip");
        }
        if self.contains(Self::BLOSC) {
            parts.push("blosc");
        }
        if self.contains(Self::ACTIVE_MASK) {
            parts.push("active values");
        }
        f.write_str(&parts.join(" + "))
    }
}

// Per-node metadata byte describing how inactive values were stored.
pub(crate) const NO_MASK_OR_INACTIVE_VALS: i8 = 0;
pub(crate) const NO_MASK_AND_MINUS_BG: i8 = 1;
pub(crate) const NO_MASK_AND_ONE_INACTIVE_VAL: i8 = 2;
pub(crate) const MASK_AND_NO_INACTIVE_VALS: i8 = 3;
pub(crate) const MASK_AND_ONE_INACTIVE_VAL: i8 = 4;
pub(crate) const MASK_AND_TWO_INACTIVE_VALS: i8 = 5;
pub(crate) const NO_MASK_AND_ALL_VALS: i8 = 6;

/// File format version that introduced the per-node metadata byte.
pub(crate) const FILE_VERSION_NODE_MASK_COMPRESSION: u32 = 222;

/// Settings in effect while reading or writing one grid's tree.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueCodec<T> {
    pub file_version: u32,
    pub compression: Compression,
    /// Values of floating-point types are stored as 16-bit half floats.
    pub half: bool,
    pub background: T,
}

impl<T: IoValue> ValueCodec<T> {
    fn half_values(&self) -> bool {
        self.half && T::IS_REAL
    }
}

/// Buffers and zlib state reused across the node buffers read by one thread, so reading a
/// leaf allocates nothing.
struct ReadScratch {
    bytes: Vec<u8>,
    compressed: Vec<u8>,
    inflate: flate2::Decompress,
}

thread_local! {
    static SCRATCH: RefCell<ReadScratch> = RefCell::new(ReadScratch {
        bytes: Vec::new(),
        compressed: Vec::new(),
        inflate: flate2::Decompress::new(true),
    });
}

/// Read a (possibly compressed) chunk that decompresses to exactly `out.len()` bytes into `out`
/// (OpenVDB's `readData`, `unzipFromStream` and `bloscFromStream`).
fn read_data_into<R: Read>(r: &mut Reader<R>, out: &mut [u8], c: Compression, s: &mut ReadScratch) -> Result<()> {
    let nbytes = out.len();
    let is_blosc = c.contains(Compression::BLOSC);
    if !is_blosc && !c.contains(Compression::ZIP) {
        return r.read_exact(out);
    }
    let n = r.read_i64()?;
    if n <= 0 {
        if n.unsigned_abs() != nbytes as u64 {
            return Err(Error::corrupt(format!("expected a {}-byte chunk, got a {}-byte chunk", nbytes, -n)));
        }
        return r.read_exact(out);
    }
    let n = n as usize;
    // Writers only compress chunks that shrink (a Blosc frame adds a 16-byte header), so a
    // larger size is corrupt: read it incrementally rather than allocating it up front.
    if n <= nbytes + 64 {
        s.compressed.resize(n, 0);
        r.read_exact(&mut s.compressed)?;
    } else {
        s.compressed = r.read_bytes(n)?;
    }
    if is_blosc {
        return blosc::decompress_into(&s.compressed, out);
    }
    s.inflate.reset(true);
    let status = s
        .inflate
        .decompress(&s.compressed, out, flate2::FlushDecompress::Finish)
        .map_err(|e| Error::Zip(e.to_string()))?;
    let got = s.inflate.total_out() as usize;
    if status != flate2::Status::StreamEnd || got != nbytes {
        let got = if got == nbytes { format!("more than {}", nbytes) } else { got.to_string() };
        return Err(Error::corrupt(format!("expected to decompress {} bytes, got {}", nbytes, got)));
    }
    Ok(())
}

/// Write a chunk with the given compression (OpenVDB's `writeData`, `zipToStream` and
/// `bloscToStream`).
pub(crate) fn write_data(out: &mut Vec<u8>, data: &[u8], c: Compression) {
    if c.contains(Compression::BLOSC) {
        if data.is_empty() {
            out.put_i64(0);
            return;
        }
        let frame = blosc::compress(data, 4);
        out.put_i64(frame.len() as i64);
        out.extend_from_slice(&frame);
    } else if c.contains(Compression::ZIP) {
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        let zipped = enc.write_all(data).and_then(|_| enc.finish());
        match zipped {
            Ok(z) if z.len() < data.len() => {
                out.put_i64(z.len() as i64);
                out.extend_from_slice(&z);
            }
            _ => {
                out.put_i64(-(data.len() as i64));
                out.extend_from_slice(data);
            }
        }
    } else {
        out.extend_from_slice(data);
    }
}

/// Read the encoded bytes of `count` values into `s.bytes` and return the size of one encoded
/// value (values of real types are stored as half floats if `half` is set).
fn read_value_bytes<T: IoValue, R: Read>(
    r: &mut Reader<R>,
    count: usize,
    codec: &ValueCodec<T>,
    s: &mut ReadScratch,
) -> Result<usize> {
    let size = if codec.half_values() { T::HALF_SIZE } else { T::SIZE };
    let mut bytes = std::mem::take(&mut s.bytes);
    bytes.clear();
    if count == 0 && codec.half_values() {
        // Empty half-float buffers are not written at all.
        s.bytes = bytes;
        return Ok(size);
    }
    bytes.resize(count * size, 0);
    let res = read_data_into(r, &mut bytes, codec.compression, s);
    s.bytes = bytes;
    res.map(|_| size)
}

/// Decode consecutive values of `size` bytes from `bytes` into `dest`.
#[inline(always)]
fn decode_all<T>(dest: &mut [T], bytes: &[u8], size: usize, decode: impl Fn(&[u8]) -> T) {
    for (d, b) in dest.iter_mut().zip(bytes.chunks_exact(size)) {
        *d = decode(b);
    }
}

/// Decode consecutive values from `bytes` into the slots of `dest` whose `on` bit is set, and
/// fill the other slots with `inactive1` where the `sel` bit is set and `inactive0` elsewhere.
#[inline(always)]
fn decode_masked<T: Copy>(
    dest: &mut [T],
    bytes: &[u8],
    size: usize,
    decode: impl Fn(&[u8]) -> T,
    (on_words, sel_words): (&[u64], &[u64]),
    (inactive0, inactive1): (T, T),
) {
    let mut n = 0;
    for ((out, &on), &sel) in dest.chunks_exact_mut(64).zip(on_words).zip(sel_words) {
        if on == !0 {
            decode_all(out, &bytes[n..n + 64 * size], size, &decode);
            n += 64 * size;
            continue;
        }
        out.fill(inactive0);
        let mut w = sel & !on;
        while w != 0 {
            out[w.trailing_zeros() as usize] = inactive1;
            w &= w - 1;
        }
        let mut w = on;
        while w != 0 {
            out[w.trailing_zeros() as usize] = decode(&bytes[n..n + size]);
            n += size;
            w &= w - 1;
        }
    }
}

fn write_values<T: IoValue>(out: &mut Vec<u8>, values: &[T], codec: &ValueCodec<T>) {
    if codec.half_values() {
        if values.is_empty() {
            return;
        }
        let mut b = Vec::with_capacity(values.len() * T::HALF_SIZE);
        for v in values {
            v.write_half_le(&mut b);
        }
        write_data(out, &b, codec.compression);
    } else {
        let mut b = Vec::with_capacity(values.len() * T::SIZE);
        for v in values {
            v.write_le(&mut b);
        }
        write_data(out, &b, codec.compression);
    }
}

/// Read a node's value buffer of `dest_count` values (port of `readCompressedValues`).
///
/// `dest_count` is normally `NodeMask::SIZE`; files older than version 222 store only the
/// non-child values of internal nodes, in which case it may be smaller.
pub(crate) fn read_compressed_values<T: IoValue, R: Read, const L: u32>(
    r: &mut Reader<R>,
    dest_count: usize,
    value_mask: &NodeMask<L>,
    codec: &ValueCodec<T>,
) -> Result<Vec<T>> {
    let mut dest = vec![codec.background; dest_count];
    read_compressed_values_into(r, &mut dest, value_mask, codec)?;
    Ok(dest)
}

/// [`read_compressed_values`] into an existing buffer of `dest_count` values.
pub(crate) fn read_compressed_values_into<T: IoValue, R: Read, const L: u32>(
    r: &mut Reader<R>,
    dest: &mut [T],
    value_mask: &NodeMask<L>,
    codec: &ValueCodec<T>,
) -> Result<()> {
    let dest_count = dest.len();
    let mask_compressed = codec.compression.contains(Compression::ACTIVE_MASK);
    let mut metadata = NO_MASK_AND_ALL_VALS;
    if codec.file_version >= FILE_VERSION_NODE_MASK_COMPRESSION {
        metadata = r.read_i8()?;
    }
    let bg = codec.background;
    let mut inactive1 = bg;
    let mut inactive0 = if metadata == NO_MASK_OR_INACTIVE_VALS { bg } else { bg.negative() };
    if matches!(metadata, NO_MASK_AND_ONE_INACTIVE_VAL | MASK_AND_ONE_INACTIVE_VAL | MASK_AND_TWO_INACTIVE_VALS) {
        let mut b = [0u8; 32];
        let b = &mut b[..T::SIZE];
        r.read_exact(b)?;
        inactive0 = T::from_le(b);
        if metadata == MASK_AND_TWO_INACTIVE_VALS {
            r.read_exact(b)?;
            inactive1 = T::from_le(b);
        }
    }
    let mut selection = NodeMask::<L>::new();
    if matches!(metadata, MASK_AND_NO_INACTIVE_VALS | MASK_AND_ONE_INACTIVE_VAL | MASK_AND_TWO_INACTIVE_VALS) {
        read_mask(r, &mut selection)?;
    }
    if !(0..=NO_MASK_AND_ALL_VALS).contains(&metadata) {
        return Err(Error::corrupt(format!("invalid node compression metadata {}", metadata)));
    }
    let mut temp_count = dest_count;
    if mask_compressed && metadata != NO_MASK_AND_ALL_VALS {
        temp_count = value_mask.count_on() as usize;
    }
    if temp_count != dest_count && dest_count != NodeMask::<L>::SIZE {
        return Err(Error::corrupt("mask-compressed values with an unexpected buffer size"));
    }
    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        let size = read_value_bytes(r, temp_count, codec, s)?;
        let bytes = &s.bytes;
        let half = codec.half_values();
        if temp_count == dest_count {
            if half {
                decode_all(dest, bytes, size, T::from_half_le);
            } else {
                decode_all(dest, bytes, size, T::from_le);
            }
        } else {
            // Restore the omitted inactive values.
            let masks = (value_mask.words(), selection.words());
            if half {
                decode_masked(dest, bytes, size, T::from_half_le, masks, (inactive0, inactive1));
            } else {
                decode_masked(dest, bytes, size, T::from_le, masks, (inactive0, inactive1));
            }
        }
        Ok(())
    })
}

/// Write a node's value buffer (port of `writeCompressedValues`). Values in slots whose
/// `child_mask` bit is on are ignored when choosing the inactive-value encoding.
pub(crate) fn write_compressed_values<T: IoValue, const L: u32>(
    out: &mut Vec<u8>,
    src: &[T],
    value_mask: &NodeMask<L>,
    child_mask: &NodeMask<L>,
    codec: &ValueCodec<T>,
) {
    if !codec.compression.contains(Compression::ACTIVE_MASK) {
        out.put_u8(NO_MASK_AND_ALL_VALS as u8);
        write_values(out, src, codec);
        return;
    }
    let bg = codec.background;
    let (metadata, inactive) = mask_compress(src, value_mask, child_mask, &bg);
    out.put_u8(metadata as u8);
    if matches!(metadata, NO_MASK_AND_ONE_INACTIVE_VAL | MASK_AND_ONE_INACTIVE_VAL | MASK_AND_TWO_INACTIVE_VALS) {
        let conv = |v: T| if codec.half { v.truncate_half() } else { v };
        conv(inactive[0]).write_le(out);
        if metadata == MASK_AND_TWO_INACTIVE_VALS {
            conv(inactive[1]).write_le(out);
        }
    }
    if metadata == NO_MASK_AND_ALL_VALS {
        write_values(out, src, codec);
        return;
    }
    let mut active = Vec::with_capacity(value_mask.count_on() as usize);
    if matches!(metadata, NO_MASK_OR_INACTIVE_VALS | NO_MASK_AND_MINUS_BG | NO_MASK_AND_ONE_INACTIVE_VAL) {
        active.extend(value_mask.iter_on().map(|i| src[i]));
    } else {
        let mut selection = NodeMask::<L>::new();
        for (i, v) in src.iter().enumerate() {
            if value_mask.is_on(i) {
                active.push(*v);
            } else if *v == inactive[1] {
                selection.set_on(i);
            }
        }
        write_mask(out, &selection);
    }
    write_values(out, &active, codec);
}

/// Choose the inactive-value encoding of a node (port of `MaskCompress`).
fn mask_compress<T: IoValue, const L: u32>(
    src: &[T],
    value_mask: &NodeMask<L>,
    child_mask: &NodeMask<L>,
    bg: &T,
) -> (i8, [T; 2]) {
    let mut inactive = [*bg, *bg];
    let mut unique = 0;
    for idx in value_mask.iter_off() {
        if unique >= 3 {
            break;
        }
        if child_mask.is_on(idx) {
            continue;
        }
        let v = src[idx];
        let is_new = !((unique > 0 && v == inactive[0]) || (unique > 1 && v == inactive[1]));
        if is_new {
            if unique < 2 {
                inactive[unique] = v;
            }
            unique += 1;
        }
    }
    let neg_bg = bg.negative();
    let metadata = match unique {
        0 => NO_MASK_OR_INACTIVE_VALS,
        1 => {
            if inactive[0] == *bg {
                NO_MASK_OR_INACTIVE_VALS
            } else if inactive[0] == neg_bg {
                NO_MASK_AND_MINUS_BG
            } else {
                NO_MASK_AND_ONE_INACTIVE_VAL
            }
        }
        2 => {
            if inactive[0] != *bg && inactive[1] != *bg {
                MASK_AND_TWO_INACTIVE_VALS
            } else if inactive[1] == *bg {
                if inactive[0] == neg_bg {
                    MASK_AND_NO_INACTIVE_VALS
                } else {
                    MASK_AND_ONE_INACTIVE_VAL
                }
            } else {
                // inactive[0] == bg
                inactive.swap(0, 1);
                if inactive[0] == neg_bg {
                    MASK_AND_NO_INACTIVE_VALS
                } else {
                    MASK_AND_ONE_INACTIVE_VAL
                }
            }
        }
        _ => NO_MASK_AND_ALL_VALS,
    };
    (metadata, inactive)
}

/// Read a node mask stored as little-endian 64-bit words.
pub(crate) fn read_mask<R: Read, const L: u32>(r: &mut Reader<R>, mask: &mut NodeMask<L>) -> Result<()> {
    let n = NodeMask::<L>::WORD_COUNT;
    let b = r.read_bytes(n * 8)?;
    for (w, c) in mask.words_mut().iter_mut().zip(b.chunks_exact(8)) {
        *w = u64::from_le_bytes(c.try_into().unwrap());
    }
    Ok(())
}

/// Write a node mask as little-endian 64-bit words.
pub(crate) fn write_mask<const L: u32>(out: &mut Vec<u8>, mask: &NodeMask<L>) {
    for w in mask.words() {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec(c: Compression, half: bool, bg: f32) -> ValueCodec<f32> {
        ValueCodec { file_version: 224, compression: c, half, background: bg }
    }

    fn roundtrip(values: &[f32], mask: &NodeMask<3>, c: &ValueCodec<f32>) -> (i8, Vec<f32>) {
        let mut out = Vec::new();
        write_compressed_values(&mut out, values, mask, &NodeMask::new(), c);
        let mut r = Reader::new(&out[..]);
        let back = read_compressed_values(&mut r, 512, mask, c).unwrap();
        assert_eq!(r.pos() as usize, out.len());
        (out[0] as i8, back)
    }

    #[test]
    fn display() {
        assert_eq!(Compression::NONE.to_string(), "none");
        assert_eq!((Compression::ZIP | Compression::ACTIVE_MASK).to_string(), "zip + active values");
        assert_eq!((Compression::BLOSC | Compression::ACTIVE_MASK).to_string(), "blosc + active values");
    }

    #[test]
    fn all_mask_schemes() {
        let bg = 2.0f32;
        let mut mask = NodeMask::<3>::new();
        for i in (0..512).step_by(5) {
            mask.set_on(i);
        }
        type Case<'a> = (Box<dyn Fn(usize) -> f32 + 'a>, i8);
        let cases: Vec<Case> = vec![
            (Box::new(|_| bg), NO_MASK_OR_INACTIVE_VALS),
            (Box::new(|_| -bg), NO_MASK_AND_MINUS_BG),
            (Box::new(|_| 7.0), NO_MASK_AND_ONE_INACTIVE_VAL),
            (Box::new(|i| if i & 2 != 0 { bg } else { -bg }), MASK_AND_NO_INACTIVE_VALS),
            (Box::new(|i| if i < 100 { bg } else { -bg }), MASK_AND_NO_INACTIVE_VALS),
            (Box::new(|i| if i & 2 != 0 { bg } else { 9.0 }), MASK_AND_ONE_INACTIVE_VAL),
            (Box::new(|i| if i < 100 { bg } else { 9.0 }), MASK_AND_ONE_INACTIVE_VAL),
            (Box::new(|i| if i & 2 != 0 { 3.0 } else { 4.0 }), MASK_AND_TWO_INACTIVE_VALS),
            (Box::new(|i| (i % 3) as f32), NO_MASK_AND_ALL_VALS),
        ];
        for comp in [
            Compression::ACTIVE_MASK,
            Compression::ACTIVE_MASK | Compression::ZIP,
            Compression::ACTIVE_MASK | Compression::BLOSC,
        ] {
            for (f, expected) in &cases {
                let values: Vec<f32> = (0..512).map(|i| if mask.is_on(i) { i as f32 * 0.5 } else { f(i) }).collect();
                let (meta, back) = roundtrip(&values, &mask, &codec(comp, false, bg));
                assert_eq!(meta, *expected);
                assert_eq!(back, values);
            }
        }
    }

    #[test]
    fn no_mask_compression_and_half() {
        let mask = NodeMask::<3>::new_with(true);
        let values: Vec<f32> = (0..512).map(|i| i as f32 * 0.25).collect();
        for comp in [Compression::NONE, Compression::ZIP, Compression::BLOSC] {
            let (meta, back) = roundtrip(&values, &mask, &codec(comp, false, 0.0));
            assert_eq!(meta, NO_MASK_AND_ALL_VALS);
            assert_eq!(back, values);
            let (_, back) = roundtrip(&values, &mask, &codec(comp, true, 0.0));
            assert_eq!(back, values); // exactly representable in half precision
        }
    }

    #[test]
    fn empty_active_set() {
        let mask = NodeMask::<3>::new();
        let values = vec![1.0f32; 512];
        for comp in [Compression::ACTIVE_MASK, Compression::ACTIVE_MASK | Compression::ZIP] {
            for half in [false, true] {
                let (_, back) = roundtrip(&values, &mask, &codec(comp, half, 1.0));
                assert_eq!(back, values);
            }
        }
    }

    #[test]
    fn full_and_empty_mask_words() {
        // Words 0 and 5 fully active, 2 and 6 fully inactive, the rest mixed.
        let bg = 3.0f32;
        let mut mask = NodeMask::<3>::new();
        for i in (0..128).chain(320..384).chain((128..512).filter(|i| i % 7 == 0)) {
            if !(128..192).contains(&i) && !(384..448).contains(&i) {
                mask.set_on(i);
            }
        }
        let values: Vec<f32> = (0..512)
            .map(|i| {
                if mask.is_on(i) {
                    i as f32 * 0.25
                } else if i % 3 == 0 {
                    9.0
                } else {
                    4.0
                }
            })
            .collect();
        for comp in [
            Compression::ACTIVE_MASK,
            Compression::ACTIVE_MASK | Compression::ZIP,
            Compression::ACTIVE_MASK | Compression::BLOSC,
        ] {
            for half in [false, true] {
                let (meta, back) = roundtrip(&values, &mask, &codec(comp, half, bg));
                assert_eq!(meta, MASK_AND_TWO_INACTIVE_VALS);
                let expected: Vec<f32> = values.iter().map(|v| if half { v.truncate_half() } else { *v }).collect();
                assert_eq!(back, expected, "{comp} half={half}");
            }
        }
    }

    #[test]
    fn corrupt_zip_chunk() {
        let values: Vec<f32> = (0..512).map(|i| (i % 4) as f32).collect();
        let c = codec(Compression::ZIP, false, 0.0);
        let mut out = Vec::new();
        write_compressed_values(&mut out, &values, &NodeMask::<3>::new(), &NodeMask::new(), &c);
        assert!(i64::from_le_bytes(out[1..9].try_into().unwrap()) > 0, "chunk should be zipped");
        let last = out.len() - 6;
        out[last] ^= 0xff;
        let mut r = Reader::new(&out[..]);
        assert!(read_compressed_values(&mut r, 512, &NodeMask::<3>::new(), &c).is_err());
        // A stream that holds fewer values than expected is rejected too.
        let mut r = Reader::new(&out[..]);
        assert!(read_compressed_values(&mut r, 256, &NodeMask::<3>::new(), &c).is_err());
    }
}
