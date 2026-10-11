//! Serialization of metadata maps (port of `MetaMap::readMeta/writeMeta` and
//! `TypedMetadata<T>::readValue/writeValue`).
//!
//! A map is stored as a `u32` count followed, for each entry, by the name, the type name
//! (both as `u32`-length-prefixed strings), a `u32` byte size and the value bytes.

use super::bytes::{Reader, WriteLe};
use super::error::Result;
use crate::math::{Mat4, Vec3};
use crate::metadata::{MetaMap, MetaValue};
use std::io::Read;

fn f32s(b: &[u8]) -> impl Iterator<Item = f32> + '_ {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))
}
fn f64s(b: &[u8]) -> impl Iterator<Item = f64> + '_ {
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()))
}
fn i32s(b: &[u8]) -> impl Iterator<Item = i32> + '_ {
    b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()))
}
fn arr<T: Default + Copy, const N: usize>(it: impl Iterator<Item = T>) -> [T; N] {
    let mut a = [T::default(); N];
    for (d, s) in a.iter_mut().zip(it) {
        *d = s;
    }
    a
}

/// Decode a metadata value of the given type from its serialized bytes.
pub(crate) fn decode_meta(type_name: &str, data: Vec<u8>) -> MetaValue {
    let n = data.len();
    let b = &data[..];
    match (type_name, n) {
        ("bool", 1) => MetaValue::Bool(b[0] != 0),
        ("int32", 4) => MetaValue::Int32(i32::from_le_bytes(b.try_into().unwrap())),
        ("int64", 8) => MetaValue::Int64(i64::from_le_bytes(b.try_into().unwrap())),
        ("float", 4) => MetaValue::Float(f32::from_le_bytes(b.try_into().unwrap())),
        ("double", 8) => MetaValue::Double(f64::from_le_bytes(b.try_into().unwrap())),
        ("string", _) => MetaValue::String(String::from_utf8_lossy(b).into_owned()),
        ("vec2i", 8) => MetaValue::Vec2i(arr(i32s(b))),
        ("vec2s", 8) => MetaValue::Vec2s(arr(f32s(b))),
        ("vec2d", 16) => MetaValue::Vec2d(arr(f64s(b))),
        ("vec3i", 12) => MetaValue::Vec3i(Vec3(arr(i32s(b)))),
        ("vec3s", 12) => MetaValue::Vec3s(Vec3(arr(f32s(b)))),
        ("vec3d", 24) => MetaValue::Vec3d(Vec3(arr(f64s(b)))),
        ("vec4i", 16) => MetaValue::Vec4i(arr(i32s(b))),
        ("vec4s", 16) => MetaValue::Vec4s(arr(f32s(b))),
        ("vec4d", 32) => MetaValue::Vec4d(arr(f64s(b))),
        ("mat4s", 64) => {
            let v: [f32; 16] = arr(f32s(b));
            MetaValue::Mat4s(Mat4(std::array::from_fn(|i| std::array::from_fn(|j| v[4 * i + j]))))
        }
        ("mat4d", 128) => {
            let v: [f64; 16] = arr(f64s(b));
            MetaValue::Mat4d(Mat4(std::array::from_fn(|i| std::array::from_fn(|j| v[4 * i + j]))))
        }
        _ => MetaValue::Unknown { type_name: type_name.to_string(), data },
    }
}

/// Serialize a metadata value's payload (without the size prefix).
pub(crate) fn encode_meta(v: &MetaValue) -> Vec<u8> {
    let mut b = Vec::new();
    match v {
        MetaValue::Bool(x) => b.push(*x as u8),
        MetaValue::Int32(x) => b.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Int64(x) => b.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Float(x) => b.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Double(x) => b.extend_from_slice(&x.to_le_bytes()),
        MetaValue::String(s) => b.extend_from_slice(s.as_bytes()),
        MetaValue::Vec2i(a) => a.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Vec2s(a) => a.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Vec2d(a) => a.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Vec3i(a) => a.0.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Vec3s(a) => a.0.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Vec3d(a) => a.0.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Vec4i(a) => a.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Vec4s(a) => a.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Vec4d(a) => a.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Mat4s(m) => m.0.iter().flatten().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Mat4d(m) => m.0.iter().flatten().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        MetaValue::Unknown { data, .. } => b.extend_from_slice(data),
    }
    b
}

/// Read a metadata map. Unknown types are kept verbatim, except temporary ones whose
/// type name starts with `"__"`, which OpenVDB drops.
pub(crate) fn read_meta_map<R: Read>(r: &mut Reader<R>) -> Result<MetaMap> {
    let mut map = MetaMap::new();
    let count = r.read_u32()?;
    for _ in 0..count {
        let name = r.read_string()?;
        let type_name = r.read_string()?;
        let size = r.read_u32()? as usize;
        let data = r.read_bytes(size)?;
        let value = decode_meta(&type_name, data);
        if matches!(value, MetaValue::Unknown { .. }) && type_name.starts_with("__") {
            continue;
        }
        map.insert(name, value);
    }
    Ok(map)
}

/// Write a metadata map. Entries with empty names (which OpenVDB rejects) are skipped.
pub(crate) fn write_meta_map(out: &mut Vec<u8>, map: &MetaMap) {
    let entries: Vec<_> = map.iter().filter(|(k, _)| !k.is_empty()).collect();
    out.put_u32(entries.len() as u32);
    for (name, value) in entries {
        out.put_string(name);
        out.put_string(value.type_name());
        let data = encode_meta(value);
        out.put_u32(data.len() as u32);
        out.extend_from_slice(&data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Mat4d, Mat4s, Vec3d, Vec3f, Vec3i};

    #[test]
    fn roundtrip_all_types() {
        let mut m = MetaMap::new();
        m.insert("b", true);
        m.insert("i", -5i32);
        m.insert("l", 1i64 << 40);
        m.insert("f", 1.5f32);
        m.insert("d", 2.5f64);
        m.insert("s", "hello");
        m.insert("v2i", MetaValue::Vec2i([1, -2]));
        m.insert("v2s", MetaValue::Vec2s([1.0, -2.0]));
        m.insert("v2d", MetaValue::Vec2d([1.0, -2.0]));
        m.insert("v3i", Vec3i::new(1, 2, 3));
        m.insert("v3s", Vec3f::new(1.0, 2.0, 3.0));
        m.insert("v3d", Vec3d::new(1.0, 2.0, 3.0));
        m.insert("v4i", MetaValue::Vec4i([1, 2, 3, 4]));
        m.insert("v4s", MetaValue::Vec4s([1.0, 2.0, 3.0, 4.0]));
        m.insert("v4d", MetaValue::Vec4d([1.0, 2.0, 3.0, 4.0]));
        let mut md = Mat4d::identity();
        md.0[3][0] = 7.0;
        m.insert("m4d", md);
        m.insert("m4s", Mat4s::identity());
        m.insert("u", MetaValue::Unknown { type_name: "custom".into(), data: vec![1, 2, 3] });
        let mut out = Vec::new();
        write_meta_map(&mut out, &m);
        let back = read_meta_map(&mut Reader::new(&out[..])).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn drops_temporary_unknown() {
        let mut m = MetaMap::new();
        m.insert("t", MetaValue::Unknown { type_name: "__tmp".into(), data: vec![9] });
        m.insert("k", 1i32);
        let mut out = Vec::new();
        write_meta_map(&mut out, &m);
        let back = read_meta_map(&mut Reader::new(&out[..])).unwrap();
        assert_eq!(back.len(), 1);
        assert!(back.contains("k"));
    }
}
