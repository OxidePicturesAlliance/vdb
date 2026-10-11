//! Little-endian binary reading and writing helpers.

use super::error::{Error, Result};
use std::io::{Read, Seek, SeekFrom};

/// A reader that tracks its byte position and provides little-endian primitive reads.
pub(crate) struct Reader<R> {
    inner: R,
    pos: u64,
}

macro_rules! read_prim {
    ($($name:ident -> $t:ty),* $(,)?) => {$(
        #[inline]
        pub fn $name(&mut self) -> Result<$t> {
            let mut b = [0u8; std::mem::size_of::<$t>()];
            self.read_exact(&mut b)?;
            Ok(<$t>::from_le_bytes(b))
        }
    )*};
}

impl<R: Read> Reader<R> {
    pub fn new(inner: R) -> Self {
        Reader { inner, pos: 0 }
    }

    /// Number of bytes consumed so far (or the absolute position after a seek).
    pub fn pos(&self) -> u64 {
        self.pos
    }

    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.inner.read_exact(buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::corrupt("unexpected end of data")
            } else {
                Error::Io(e)
            }
        })?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    /// Read exactly `n` bytes. The buffer grows as data arrives, so a corrupt size does not
    /// trigger a huge up-front allocation.
    pub fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut v = Vec::with_capacity(n.min(1 << 24));
        let got = (&mut self.inner).take(n as u64).read_to_end(&mut v)?;
        self.pos += got as u64;
        if got != n {
            return Err(Error::corrupt("unexpected end of data"));
        }
        Ok(v)
    }

    /// Skip `n` bytes by reading and discarding them.
    pub fn skip(&mut self, n: u64) -> Result<()> {
        let got = std::io::copy(&mut (&mut self.inner).take(n), &mut std::io::sink())?;
        self.pos += got;
        if got != n {
            return Err(Error::corrupt("unexpected end of data"));
        }
        Ok(())
    }

    read_prim!(
        read_u8 -> u8, read_i8 -> i8, read_u32 -> u32, read_i32 -> i32, read_i64 -> i64, read_f64 -> f64,
    );

    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// Read an OpenVDB string: a `u32` length followed by that many bytes.
    pub fn read_string(&mut self) -> Result<String> {
        let n = self.read_u32()? as usize;
        let b = self.read_bytes(n)?;
        Ok(String::from_utf8(b).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
    }

    /// Read `N` consecutive `f64` values.
    pub fn read_f64s<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut out = [0.0; N];
        for v in out.iter_mut() {
            *v = self.read_f64()?;
        }
        Ok(out)
    }
}

impl<R: Read + Seek> Reader<R> {
    /// Seek to an absolute byte offset.
    pub fn seek_to(&mut self, pos: u64) -> Result<()> {
        self.pos = self.inner.seek(SeekFrom::Start(pos))?;
        Ok(())
    }
}

/// Little-endian primitive writes into a byte buffer.
pub(crate) trait WriteLe {
    fn put_u8(&mut self, v: u8);
    fn put_u32(&mut self, v: u32);
    fn put_i32(&mut self, v: i32);
    fn put_i64(&mut self, v: i64);
    fn put_f64(&mut self, v: f64);
    /// Write an OpenVDB string (`u32` length + bytes).
    fn put_string(&mut self, s: &str);
}

impl WriteLe for Vec<u8> {
    #[inline]
    fn put_u8(&mut self, v: u8) {
        self.push(v);
    }
    #[inline]
    fn put_u32(&mut self, v: u32) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    fn put_i32(&mut self, v: i32) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    fn put_i64(&mut self, v: i64) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    fn put_f64(&mut self, v: f64) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    fn put_string(&mut self, s: &str) {
        self.put_u32(s.len() as u32);
        self.extend_from_slice(s.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_primitives() {
        let mut b = Vec::new();
        b.put_u32(7);
        b.put_i64(-3);
        b.put_string("abc");
        b.put_f64(1.5);
        let mut r = Reader::new(&b[..]);
        assert_eq!(r.read_u32().unwrap(), 7);
        assert_eq!(r.read_i64().unwrap(), -3);
        assert_eq!(r.read_string().unwrap(), "abc");
        assert_eq!(r.read_f64().unwrap(), 1.5);
        assert_eq!(r.pos(), b.len() as u64);
        assert!(r.read_u8().is_err());
    }

    #[test]
    fn truncated_bytes() {
        let b = [1u8, 2, 3];
        let mut r = Reader::new(&b[..]);
        assert!(r.read_bytes(10).is_err());
        let mut r = Reader::new(&b[..]);
        r.skip(2).unwrap();
        assert_eq!(r.read_u8().unwrap(), 3);
    }
}
