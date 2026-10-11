//! Typed metadata and metadata maps (port of `openvdb/Metadata.h` and `openvdb/MetaMap.h`).

use crate::math::{Mat4d, Mat4s, Vec3d, Vec3f, Vec3i};
use std::collections::BTreeMap;
use std::fmt;

/// A metadata value. The variant names correspond to OpenVDB metadata type names.
#[derive(Clone, PartialEq, Debug)]
pub enum MetaValue {
    Bool(bool),
    Int32(i32),
    Int64(i64),
    Float(f32),
    Double(f64),
    String(String),
    Vec2i([i32; 2]),
    Vec2s([f32; 2]),
    Vec2d([f64; 2]),
    Vec3i(Vec3i),
    Vec3s(Vec3f),
    Vec3d(Vec3d),
    Vec4i([i32; 4]),
    Vec4s([f32; 4]),
    Vec4d([f64; 4]),
    Mat4s(Mat4s),
    Mat4d(Mat4d),
    /// Metadata of a type this library does not know, kept verbatim so it can be rewritten.
    Unknown {
        type_name: String,
        data: Vec<u8>,
    },
}

impl MetaValue {
    /// OpenVDB's type name for this metadata (e.g. `"string"`, `"vec3d"`).
    pub fn type_name(&self) -> &str {
        match self {
            MetaValue::Bool(_) => "bool",
            MetaValue::Int32(_) => "int32",
            MetaValue::Int64(_) => "int64",
            MetaValue::Float(_) => "float",
            MetaValue::Double(_) => "double",
            MetaValue::String(_) => "string",
            MetaValue::Vec2i(_) => "vec2i",
            MetaValue::Vec2s(_) => "vec2s",
            MetaValue::Vec2d(_) => "vec2d",
            MetaValue::Vec3i(_) => "vec3i",
            MetaValue::Vec3s(_) => "vec3s",
            MetaValue::Vec3d(_) => "vec3d",
            MetaValue::Vec4i(_) => "vec4i",
            MetaValue::Vec4s(_) => "vec4s",
            MetaValue::Vec4d(_) => "vec4d",
            MetaValue::Mat4s(_) => "mat4s",
            MetaValue::Mat4d(_) => "mat4d",
            MetaValue::Unknown { type_name, .. } => type_name,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// Numeric value as `f64`, for any scalar numeric variant.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::Int32(v) => Some(*v as f64),
            MetaValue::Int64(v) => Some(*v as f64),
            MetaValue::Float(v) => Some(*v as f64),
            MetaValue::Double(v) => Some(*v),
            MetaValue::Bool(b) => Some(*b as i32 as f64),
            _ => None,
        }
    }
    /// Numeric value as `i64`, for integer and boolean variants.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            MetaValue::Int32(v) => Some(*v as i64),
            MetaValue::Int64(v) => Some(*v),
            MetaValue::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    pub fn as_vec3i(&self) -> Option<Vec3i> {
        match self {
            MetaValue::Vec3i(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for MetaValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaValue::Bool(v) => write!(f, "{}", v),
            MetaValue::Int32(v) => write!(f, "{}", v),
            MetaValue::Int64(v) => write!(f, "{}", v),
            MetaValue::Float(v) => write!(f, "{}", v),
            MetaValue::Double(v) => write!(f, "{}", v),
            MetaValue::String(v) => write!(f, "{}", v),
            MetaValue::Vec2i(v) => write!(f, "{:?}", v),
            MetaValue::Vec2s(v) => write!(f, "{:?}", v),
            MetaValue::Vec2d(v) => write!(f, "{:?}", v),
            MetaValue::Vec3i(v) => write!(f, "{}", v),
            MetaValue::Vec3s(v) => write!(f, "{}", v),
            MetaValue::Vec3d(v) => write!(f, "{}", v),
            MetaValue::Vec4i(v) => write!(f, "{:?}", v),
            MetaValue::Vec4s(v) => write!(f, "{:?}", v),
            MetaValue::Vec4d(v) => write!(f, "{:?}", v),
            MetaValue::Mat4s(v) => write!(f, "{:?}", v),
            MetaValue::Mat4d(v) => write!(f, "{:?}", v),
            MetaValue::Unknown { type_name, data } => write!(f, "<{}: {} bytes>", type_name, data.len()),
        }
    }
}

macro_rules! meta_from {
    ($($t:ty => $v:ident),* $(,)?) => {$(
        impl From<$t> for MetaValue { fn from(x: $t) -> Self { MetaValue::$v(x) } }
    )*};
}
meta_from!(
    bool => Bool, i32 => Int32, i64 => Int64, f32 => Float, f64 => Double, String => String,
    Vec3i => Vec3i, Vec3f => Vec3s, Vec3d => Vec3d, Mat4s => Mat4s, Mat4d => Mat4d,
);
impl From<&str> for MetaValue {
    fn from(s: &str) -> Self {
        MetaValue::String(s.to_string())
    }
}

/// An ordered map from names to metadata values (OpenVDB's `MetaMap`).
#[derive(Clone, PartialEq, Debug, Default)]
pub struct MetaMap {
    map: BTreeMap<String, MetaValue>,
}

impl MetaMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get(&self, name: &str) -> Option<&MetaValue> {
        self.map.get(name)
    }
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<MetaValue>) {
        self.map.insert(name.into(), value.into());
    }
    pub fn remove(&mut self, name: &str) -> Option<MetaValue> {
        self.map.remove(name)
    }
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn clear(&mut self) {
        self.map.clear()
    }
    /// Iterate in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetaValue)> {
        self.map.iter()
    }
    pub fn get_str(&self, name: &str) -> Option<&str> {
        self.get(name)?.as_str()
    }
}

impl<'a> IntoIterator for &'a MetaMap {
    type Item = (&'a String, &'a MetaValue);
    type IntoIter = std::collections::btree_map::Iter<'a, String, MetaValue>;
    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

impl fmt::Display for MetaMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, v) in &self.map {
            writeln!(f, "{}: {}", k, v)?;
        }
        Ok(())
    }
}
