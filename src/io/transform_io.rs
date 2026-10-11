//! Transform serialization: the map's registered type name followed by its payload,
//! delegating to the exact ports of OpenVDB's map `read`/`write` in [`crate::math::maps`].

use super::bytes::{Reader, WriteLe};
use super::error::{Error, Result};
use crate::math::{MapBase, Transform};
use std::io::Read;

/// Read the payload of a map of the given registered type.
pub(crate) fn read_map<R: Read>(r: &mut Reader<R>, type_name: &str) -> Result<MapBase> {
    if !MapBase::is_registered(type_name) {
        return Err(Error::Unsupported(format!("map type \"{}\"", type_name)));
    }
    MapBase::read_typed(type_name, r).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => Error::corrupt("unexpected end of data"),
        std::io::ErrorKind::Unsupported => Error::Unsupported(e.to_string()),
        _ => Error::Io(e),
    })
}

/// Write a transform as OpenVDB's `Transform::write` does.
pub(crate) fn write_transform(out: &mut Vec<u8>, t: &Transform) -> Result<()> {
    out.put_string(t.map().type_name());
    t.map().write(out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Axis, Vec3d};

    fn roundtrip(t: &Transform) -> (String, Transform) {
        let mut out = Vec::new();
        write_transform(&mut out, t).unwrap();
        let mut r = Reader::new(&out[..]);
        let name = r.read_string().unwrap();
        let back = Transform::from_map(read_map(&mut r, &name).unwrap());
        assert_eq!(r.pos() as usize, out.len());
        (name, back)
    }

    #[test]
    fn map_types_roundtrip() {
        let (n, b) = roundtrip(&Transform::linear(0.5));
        assert_eq!(n, "UniformScaleMap");
        assert_eq!(b, Transform::linear(0.5));
        let mut t = Transform::linear(0.1);
        t.post_translate(Vec3d::new(1.0, 2.0, 3.0));
        let (n, b) = roundtrip(&t);
        assert_eq!(n, "UniformScaleTranslateMap");
        assert_eq!(b, t);
        t.pre_rotate(0.3, Axis::Y);
        let (n, b) = roundtrip(&t);
        assert_eq!(n, "AffineMap");
        assert_eq!(b, t);
    }
}
