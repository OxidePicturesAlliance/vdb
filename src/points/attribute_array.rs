//! Attribute arrays: per-point data storage with optional compression codecs (port of
//! `openvdb/points/AttributeArray.h`).
//!
//! An [`AttributeArray`] stores `size()` elements (each of `stride()` values) of one value
//! type, encoded with one [`Codec`]. Arrays whose elements are all equal can be *uniform*:
//! they then store a single element. Elements are kept as encoded little-endian storage
//! bytes, exactly as OpenVDB writes them to disk, so arrays of any registered type can be
//! read and written; typed access goes through [`AttributeHandle`] and
//! [`AttributeWriteHandle`] (or [`AttributeArray::get`]/[`AttributeArray::set`]).

use super::codec::{storage_type_size, value_type_size, AttributeValue, Codec};
use super::{Error, Result};
use crate::metadata::MetaValue;
use std::fmt;
use std::marker::PhantomData;

/// The type of an attribute array: value type name and codec name (OpenVDB's `NamePair`),
/// e.g. `("vec3s", "fxpt16")`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AttributeType {
    /// Value type name, e.g. `"float"`.
    pub value_type: String,
    /// Codec name, e.g. `"null"`.
    pub codec: String,
}

impl AttributeType {
    /// A type from its value type and codec names.
    pub fn new(value_type: impl Into<String>, codec: impl Into<String>) -> Self {
        AttributeType { value_type: value_type.into(), codec: codec.into() }
    }
    /// The type of arrays of `T` encoded with `codec`.
    pub fn of<T: AttributeValue>(codec: Codec) -> Self {
        AttributeType::new(T::TYPE_NAME, codec.name())
    }
    /// The group array type `("uint8", "grp")`.
    pub fn group() -> Self {
        AttributeType::new("uint8", "grp")
    }
    /// The string array type `("uint32", "str")`.
    pub fn string() -> Self {
        AttributeType::new("uint32", "str")
    }
    /// Size of one stored element, if the type is supported.
    pub fn storage_size(&self) -> Option<usize> {
        storage_type_size(&self.value_type, &self.codec)
    }
    /// True if arrays of this type can be created (OpenVDB's attribute type registry).
    pub fn is_registered(&self) -> bool {
        self.storage_size().is_some()
    }
}

impl fmt::Display for AttributeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.value_type, self.codec)
    }
}

/// Array flag: not written to disk.
pub const FLAG_TRANSIENT: u8 = 0x1;
/// Array flag: hidden from UIs and iterators.
pub const FLAG_HIDDEN: u8 = 0x2;
/// Array flag: all elements have the same stride.
pub const FLAG_CONSTANT_STRIDE: u8 = 0x8;
/// Array flag: collapse the array after it has been read once (streaming mode).
pub const FLAG_STREAMING: u8 = 0x10;

/// A typed array of per-point values (OpenVDB's `TypedAttributeArray`, type-erased).
#[derive(Clone, PartialEq)]
pub struct AttributeArray {
    ty: AttributeType,
    codec: Option<Codec>,
    storage_size: usize,
    value_size: usize,
    size: u32,
    stride_or_total: u32,
    flags: u8,
    uniform: bool,
    data: Vec<u8>,
}

impl fmt::Debug for AttributeArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AttributeArray")
            .field("type", &self.ty.to_string())
            .field("size", &self.size)
            .field("stride", &self.stride())
            .field("uniform", &self.uniform)
            .field("flags", &self.flags)
            .finish()
    }
}

impl AttributeArray {
    /// Create an array of `T` encoded with `codec`, with `n` elements of `stride` values each,
    /// uniformly set to `value`.
    pub fn new<T: AttributeValue>(codec: Codec, n: u32, stride: u32, value: T) -> Result<Self> {
        let mut a = Self::create(&AttributeType::of::<T>(codec), n, stride, true, None)?;
        a.collapse_to(value)?;
        Ok(a)
    }

    /// Create a zero-initialized uniform array of `T` with stride 1.
    pub fn with_size<T: AttributeValue>(codec: Codec, n: u32) -> Result<Self> {
        Self::new(codec, n, 1, T::default())
    }

    /// Create a uniform array of a registered type (`AttributeArray::create`).
    ///
    /// With a constant stride, `stride_or_total` is the number of values per element; with
    /// a non-constant stride it is the total number of values (at least `n`). The array is
    /// initialized with `default` if it is metadata of the array's value type, else zero.
    pub fn create(
        ty: &AttributeType,
        n: u32,
        stride_or_total: u32,
        constant_stride: bool,
        default: Option<&MetaValue>,
    ) -> Result<Self> {
        let storage_size =
            ty.storage_size().ok_or_else(|| Error::Key(format!("unregistered attribute type \"{}\"", ty)))?;
        if constant_stride && stride_or_total == 0 {
            return Err(Error::Value("a constant-stride attribute array requires a stride of at least one".into()));
        }
        if !constant_stride && stride_or_total < n {
            return Err(Error::Value(
                "a non-constant-stride attribute array needs a total size of at least its length".into(),
            ));
        }
        let mut a = AttributeArray {
            codec: Codec::from_name(&ty.codec),
            value_size: value_type_size(&ty.value_type).unwrap_or(storage_size),
            ty: ty.clone(),
            storage_size,
            size: n.max(1),
            stride_or_total: stride_or_total.max(1),
            flags: if constant_stride { FLAG_CONSTANT_STRIDE } else { 0 },
            uniform: true,
            data: vec![0u8; storage_size],
        };
        if let Some(m) = default {
            a.set_uniform_from_meta(m);
        }
        Ok(a)
    }

    /// Build an array from raw parts (used by file I/O). `data` must hold one element if
    /// `uniform`, else `data_size()` elements.
    #[cfg(feature = "read")]
    pub(crate) fn from_raw(
        ty: AttributeType,
        size: u32,
        stride_or_total: u32,
        flags: u8,
        uniform: bool,
        data: Vec<u8>,
    ) -> Result<Self> {
        let storage_size =
            ty.storage_size().ok_or_else(|| Error::Key(format!("unregistered attribute type \"{}\"", ty)))?;
        let a = AttributeArray {
            codec: Codec::from_name(&ty.codec),
            value_size: value_type_size(&ty.value_type).unwrap_or(storage_size),
            ty,
            storage_size,
            size,
            stride_or_total,
            flags,
            uniform,
            data,
        };
        let expected = if uniform { storage_size } else { a.data_size() as usize * storage_size };
        if a.data.len() < expected {
            return Err(Error::Value(format!(
                "attribute array data has {} bytes, expected {}",
                a.data.len(),
                expected
            )));
        }
        let mut a = a;
        a.data.truncate(expected);
        Ok(a)
    }

    fn set_uniform_from_meta(&mut self, m: &MetaValue) {
        macro_rules! try_types {
            ($($t:ty),*) => {$(
                if self.ty.value_type == <$t as AttributeValue>::TYPE_NAME {
                    if let (Some(v), Some(c)) = (<$t as AttributeValue>::from_meta(m), self.codec) {
                        if <$t as AttributeValue>::storage_size(c) == Some(self.storage_size) {
                            v.encode(c, &mut self.data[..self.storage_size]);
                        }
                    }
                    return;
                }
            )*};
        }
        try_types!(bool, i32, i64, f32, f64, crate::math::Vec3i, crate::math::Vec3f, crate::math::Vec3d);
    }

    /// The array's type.
    pub fn attribute_type(&self) -> &AttributeType {
        &self.ty
    }
    /// The value type name (e.g. `"float"`).
    pub fn value_type(&self) -> &str {
        &self.ty.value_type
    }
    /// The codec name (e.g. `"trnc"`).
    pub fn codec_type(&self) -> &str {
        &self.ty.codec
    }
    /// The codec, if it is one this library knows.
    pub fn codec(&self) -> Option<Codec> {
        self.codec
    }
    /// True if the value type is `T`.
    pub fn has_value_type<T: AttributeValue>(&self) -> bool {
        self.ty.value_type == T::TYPE_NAME
    }
    /// True if this array has the given value type and codec (`isType<...>()`).
    pub fn is_type<T: AttributeValue>(&self, codec: Codec) -> bool {
        self.has_value_type::<T>() && self.codec == Some(codec)
    }
    /// True if this is a group array (`isGroup`).
    pub fn is_group(&self) -> bool {
        self.ty.value_type == "uint8" && self.codec == Some(Codec::Group)
    }
    /// True if this is a string array (`isString`).
    pub fn is_string(&self) -> bool {
        self.ty.value_type == "uint32" && self.codec == Some(Codec::String)
    }

    /// Number of elements (`size()`); each element has `stride()` values.
    pub fn size(&self) -> u32 {
        self.size
    }
    /// Values per element, or 0 for a non-constant stride.
    pub fn stride(&self) -> u32 {
        if self.has_constant_stride() {
            self.stride_or_total
        } else {
            0
        }
    }
    /// Total number of values (`dataSize()`).
    pub fn data_size(&self) -> u32 {
        if self.has_constant_stride() {
            self.size * self.stride_or_total
        } else {
            self.stride_or_total
        }
    }
    /// The raw stride-or-total-size field.
    #[cfg(feature = "write")]
    pub(crate) fn stride_or_total_size(&self) -> u32 {
        self.stride_or_total
    }
    /// True if every element has the same stride.
    pub fn has_constant_stride(&self) -> bool {
        self.flags & FLAG_CONSTANT_STRIDE != 0
    }
    /// Size in bytes of one stored (encoded) value.
    pub fn storage_type_size(&self) -> usize {
        self.storage_size
    }
    /// Size in bytes of one decoded value (`sizeof(ValueType)`).
    pub fn value_type_size(&self) -> usize {
        self.value_size
    }
    /// True if the value type is floating point (scalars, vectors, matrices, quaternions).
    pub fn value_type_is_floating_point(&self) -> bool {
        let t = self.value_type();
        matches!(t, "half" | "float" | "double")
            || t.ends_with('s') && (t.starts_with("vec") || t.starts_with("mat") || t.starts_with("quat"))
            || t.ends_with('d') && (t.starts_with("vec") || t.starts_with("mat") || t.starts_with("quat"))
    }
    /// The array flags (`FLAG_*`).
    pub fn flags(&self) -> u8 {
        self.flags
    }
    /// True if the array is hidden from UIs and iterators.
    pub fn is_hidden(&self) -> bool {
        self.flags & FLAG_HIDDEN != 0
    }
    /// Hide or show the array.
    pub fn set_hidden(&mut self, on: bool) {
        self.set_flag(FLAG_HIDDEN, on)
    }
    /// True if the array is not written to disk.
    pub fn is_transient(&self) -> bool {
        self.flags & FLAG_TRANSIENT != 0
    }
    /// Mark the array as transient (not written to disk).
    pub fn set_transient(&mut self, on: bool) {
        self.set_flag(FLAG_TRANSIENT, on)
    }
    /// True if the array is in streaming mode.
    pub fn is_streaming(&self) -> bool {
        self.flags & FLAG_STREAMING != 0
    }
    /// Toggle streaming mode.
    pub fn set_streaming(&mut self, on: bool) {
        self.set_flag(FLAG_STREAMING, on)
    }
    fn set_flag(&mut self, flag: u8, on: bool) {
        if on {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }

    /// True if the array stores a single uniform value.
    pub fn is_uniform(&self) -> bool {
        self.uniform
    }
    /// Make a uniform array non-uniform; if `fill`, every element gets the uniform value.
    pub fn expand(&mut self, fill: bool) {
        if !self.uniform {
            return;
        }
        let n = self.data_size() as usize;
        let val = self.data[..self.storage_size].to_vec();
        self.uniform = false;
        if fill {
            self.data = val.repeat(n);
        } else {
            self.data = vec![0u8; n * self.storage_size];
        }
    }
    /// Replace the contents with a uniform zero value.
    pub fn collapse(&mut self) {
        self.uniform = true;
        self.data = vec![0u8; self.storage_size];
    }
    /// Replace the contents with the uniform value `v`.
    pub fn collapse_to<T: AttributeValue>(&mut self, v: T) -> Result<()> {
        let codec = self.check_type::<T>()?;
        self.uniform = true;
        self.data = vec![0u8; self.storage_size];
        v.encode(codec, &mut self.data);
        Ok(())
    }
    /// Collapse the array if all its stored values are identical; returns true if the array
    /// is uniform afterwards.
    pub fn compact(&mut self) -> bool {
        if self.uniform {
            return true;
        }
        let s = self.storage_size;
        let first = &self.data[..s];
        if self.data.chunks_exact(s).skip(1).any(|c| c != first) {
            return false;
        }
        self.data.truncate(s);
        self.uniform = true;
        true
    }
    /// Set every value (the uniform value if uniform) to `v`.
    pub fn fill<T: AttributeValue>(&mut self, v: T) -> Result<()> {
        let codec = self.check_type::<T>()?;
        let mut enc = vec![0u8; self.storage_size];
        v.encode(codec, &mut enc);
        for c in self.data.chunks_exact_mut(self.storage_size) {
            c.copy_from_slice(&enc);
        }
        Ok(())
    }

    fn check_type<T: AttributeValue>(&self) -> Result<Codec> {
        if !self.has_value_type::<T>() {
            return Err(Error::Type(format!(
                "attribute array of type {} accessed as {}",
                self.ty.value_type,
                T::TYPE_NAME
            )));
        }
        let codec = self.codec.ok_or_else(|| Error::Type(format!("unknown codec \"{}\"", self.ty.codec)))?;
        if T::storage_size(codec) != Some(self.storage_size) {
            return Err(Error::Type(format!("codec {} is not supported for {}", codec, T::TYPE_NAME)));
        }
        Ok(codec)
    }

    /// The value at data index `n` (`get(n)`), decoded as `T`.
    ///
    /// # Panics
    /// If `T` is not the array's value type or `n >= data_size()`.
    pub fn get<T: AttributeValue>(&self, n: u32) -> T {
        let codec = self.check_type::<T>().unwrap_or_else(|e| panic!("{}", e));
        assert!(n < self.data_size(), "attribute index {} out of range ({})", n, self.data_size());
        let i = if self.uniform { 0 } else { n as usize * self.storage_size };
        T::decode(codec, &self.data[i..i + self.storage_size])
    }
    /// Set the value at data index `n`, expanding a uniform array first.
    ///
    /// # Panics
    /// If `T` is not the array's value type or `n >= data_size()`.
    pub fn set<T: AttributeValue>(&mut self, n: u32, v: T) {
        let codec = self.check_type::<T>().unwrap_or_else(|e| panic!("{}", e));
        assert!(n < self.data_size(), "attribute index {} out of range ({})", n, self.data_size());
        self.expand(true);
        let i = n as usize * self.storage_size;
        v.encode(codec, &mut self.data[i..i + self.storage_size]);
    }

    /// The encoded storage: one element if uniform, else `data_size()` elements.
    pub fn storage(&self) -> &[u8] {
        &self.data
    }
    /// Mutable encoded storage.
    pub fn storage_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
    /// The encoded bytes of data index `n`.
    pub fn storage_at(&self, n: u32) -> &[u8] {
        let i = if self.uniform { 0 } else { n as usize * self.storage_size };
        &self.data[i..i + self.storage_size]
    }

    /// Bytes of memory used by the array (`memUsage()`).
    pub fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len()
    }

    /// Copy stored values from `source` for each `(source index, target index)` pair, as
    /// OpenVDB's `copyValues`; the arrays must have the same storage size and stride. The
    /// target is expanded first and compacted afterwards if `compact`.
    pub fn copy_values(
        &mut self,
        source: &AttributeArray,
        pairs: impl IntoIterator<Item = (u32, u32)>,
        compact: bool,
    ) -> Result<()> {
        if source.storage_size * source.stride_or_total as usize != self.storage_size * self.stride_or_total as usize
            && source.storage_size != self.storage_size
        {
            return Err(Error::Type("cannot copy array data due to mismatching storage type sizes".into()));
        }
        self.expand(true);
        let stride = if self.has_constant_stride() { self.stride_or_total as usize } else { 1 };
        let bytes = self.storage_size * stride;
        let src_n = source.data_size() as usize / stride;
        let dst_n = self.data_size() as usize / stride;
        for (s, t) in pairs {
            let s = if source.uniform { 0 } else { s as usize };
            let t = t as usize;
            if (!source.uniform && s >= src_n) || t >= dst_n {
                return Err(Error::Index("cannot copy array data: index out of range".into()));
            }
            if source.uniform {
                for k in 0..stride {
                    let d = (t * stride + k) * self.storage_size;
                    self.data[d..d + self.storage_size].copy_from_slice(&source.data[..self.storage_size]);
                }
            } else {
                self.data[t * bytes..(t + 1) * bytes].copy_from_slice(&source.data[s * bytes..(s + 1) * bytes]);
            }
        }
        if compact {
            self.compact();
        }
        Ok(())
    }

    /// A new array of the same type, stride and flags with `n` elements, filled with the
    /// uniform value of this array if it is uniform (used when rebuilding attribute sets).
    pub fn empty_like(&self, n: u32) -> AttributeArray {
        let stride = if self.has_constant_stride() { self.stride_or_total } else { self.data_size().max(n) };
        AttributeArray {
            ty: self.ty.clone(),
            codec: self.codec,
            storage_size: self.storage_size,
            value_size: self.value_size,
            size: n.max(1),
            stride_or_total: stride.max(1),
            flags: self.flags,
            uniform: true,
            data: vec![0u8; self.storage_size],
        }
    }
}

/// Read access to an attribute array as values of type `T` (OpenVDB's `AttributeHandle`
/// with an unknown codec: any codec supported for `T` is accepted).
pub struct AttributeHandle<'a, T: AttributeValue> {
    array: &'a AttributeArray,
    codec: Codec,
    stride: u32,
    size: u32,
    _t: PhantomData<T>,
}

impl<'a, T: AttributeValue> AttributeHandle<'a, T> {
    /// Bind a handle; fails if the array's value type is not `T`.
    pub fn new(array: &'a AttributeArray) -> Result<Self> {
        let codec = array.check_type::<T>()?;
        let (stride, size) =
            if array.has_constant_stride() { (array.stride(), array.size()) } else { (1, array.data_size()) };
        Ok(AttributeHandle { array, codec, stride, size, _t: PhantomData })
    }
    /// Values per element (1 for non-constant strides).
    pub fn stride(&self) -> u32 {
        self.stride
    }
    /// Number of elements (the total value count for non-constant strides).
    pub fn size(&self) -> u32 {
        self.size
    }
    /// True if the array is uniform.
    pub fn is_uniform(&self) -> bool {
        self.array.uniform
    }
    /// True if the array has a constant stride.
    pub fn has_constant_stride(&self) -> bool {
        self.array.has_constant_stride()
    }
    /// The value of element `n` (first value of the element).
    #[inline]
    pub fn get(&self, n: u32) -> T {
        self.get_strided(n, 0)
    }
    /// Value `m` of element `n`.
    #[inline]
    pub fn get_strided(&self, n: u32, m: u32) -> T {
        let idx = n * self.stride + m;
        debug_assert!(idx < self.size * self.stride);
        let s = self.array.storage_size;
        let i = if self.array.uniform { 0 } else { idx as usize * s };
        T::decode(self.codec, &self.array.data[i..i + s])
    }
    /// The bound array.
    pub fn array(&self) -> &'a AttributeArray {
        self.array
    }
}

/// Read-write access to an attribute array as values of type `T`
/// (OpenVDB's `AttributeWriteHandle`).
pub struct AttributeWriteHandle<'a, T: AttributeValue> {
    array: &'a mut AttributeArray,
    codec: Codec,
    stride: u32,
    size: u32,
    _t: PhantomData<T>,
}

impl<'a, T: AttributeValue> AttributeWriteHandle<'a, T> {
    /// Bind a handle and expand the array (so that it can be written element-wise).
    pub fn new(array: &'a mut AttributeArray) -> Result<Self> {
        Self::with_expand(array, true)
    }
    /// Bind a handle, expanding the array only if `expand`.
    pub fn with_expand(array: &'a mut AttributeArray, expand: bool) -> Result<Self> {
        let codec = array.check_type::<T>()?;
        let (stride, size) =
            if array.has_constant_stride() { (array.stride(), array.size()) } else { (1, array.data_size()) };
        if expand {
            array.expand(true);
        }
        Ok(AttributeWriteHandle { array, codec, stride, size, _t: PhantomData })
    }
    /// Values per element.
    pub fn stride(&self) -> u32 {
        self.stride
    }
    /// Number of elements.
    pub fn size(&self) -> u32 {
        self.size
    }
    /// True if the array is uniform.
    pub fn is_uniform(&self) -> bool {
        self.array.uniform
    }
    /// The value of element `n`.
    #[inline]
    pub fn get(&self, n: u32) -> T {
        self.get_strided(n, 0)
    }
    /// Value `m` of element `n`.
    #[inline]
    pub fn get_strided(&self, n: u32, m: u32) -> T {
        let idx = n * self.stride + m;
        let s = self.array.storage_size;
        let i = if self.array.uniform { 0 } else { idx as usize * s };
        T::decode(self.codec, &self.array.data[i..i + s])
    }
    /// Set element `n`. On a uniform (unexpanded) array this sets the uniform value.
    #[inline]
    pub fn set(&mut self, n: u32, v: T) {
        self.set_strided(n, 0, v)
    }
    /// Set value `m` of element `n`.
    #[inline]
    pub fn set_strided(&mut self, n: u32, m: u32, v: T) {
        let idx = n * self.stride + m;
        debug_assert!(idx < self.size * self.stride);
        let s = self.array.storage_size;
        let i = if self.array.uniform { 0 } else { idx as usize * s };
        v.encode(self.codec, &mut self.array.data[i..i + s]);
    }
    /// Expand the array.
    pub fn expand(&mut self, fill: bool) {
        self.array.expand(fill)
    }
    /// Collapse to a uniform zero.
    pub fn collapse(&mut self) {
        self.array.collapse()
    }
    /// Collapse to a uniform value.
    pub fn collapse_to(&mut self, v: T) {
        self.array.uniform = true;
        let s = self.array.storage_size;
        self.array.data = vec![0u8; s];
        v.encode(self.codec, &mut self.array.data);
    }
    /// Collapse if all values are equal.
    pub fn compact(&mut self) -> bool {
        self.array.compact()
    }
    /// Set every value to `v`.
    pub fn fill(&mut self, v: T) {
        let s = self.array.storage_size;
        let mut enc = vec![0u8; s];
        v.encode(self.codec, &mut enc);
        for c in self.array.data.chunks_exact_mut(s) {
            c.copy_from_slice(&enc);
        }
    }
    /// The bound array.
    pub fn array(&mut self) -> &mut AttributeArray {
        self.array
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3f;

    #[test]
    fn uniform_expand_compact() {
        let mut a = AttributeArray::new(Codec::Null, 10, 1, 2.5f32).unwrap();
        assert!(a.is_uniform());
        assert_eq!(a.size(), 10);
        assert_eq!(a.get::<f32>(7), 2.5);
        a.set(3, 1.0f32);
        assert!(!a.is_uniform());
        assert_eq!(a.get::<f32>(3), 1.0);
        assert_eq!(a.get::<f32>(4), 2.5);
        assert!(!a.compact());
        a.set(3, 2.5f32);
        assert!(a.compact());
        assert_eq!(a.storage().len(), 4);
        a.collapse();
        assert_eq!(a.get::<f32>(0), 0.0);
    }

    #[test]
    fn handles_and_stride() {
        let mut a = AttributeArray::new(Codec::FIXED_POINT_16, 4, 2, Vec3f::zero()).unwrap();
        assert_eq!(a.data_size(), 8);
        {
            let mut h = AttributeWriteHandle::<Vec3f>::new(&mut a).unwrap();
            h.set_strided(1, 1, Vec3f::new(0.25, -0.25, 0.0));
        }
        let h = AttributeHandle::<Vec3f>::new(&a).unwrap();
        let v = h.get_strided(1, 1);
        assert!((v - Vec3f::new(0.25, -0.25, 0.0)).length() < 1e-4);
        // Zero quantizes to 32767/65535 - 0.5 with the 16-bit position codec.
        assert!(h.get(0).length() < 2e-5);
        assert!(AttributeHandle::<f32>::new(&a).is_err());
        assert_eq!(a.storage_type_size(), 6);
        assert_eq!(a.value_type_size(), 12);
    }

    #[test]
    fn create_and_copy() {
        let t = AttributeType::new("float", "null");
        let a = AttributeArray::create(&t, 3, 1, true, Some(&MetaValue::Float(4.0))).unwrap();
        assert_eq!(a.get::<f32>(2), 4.0);
        assert!(AttributeArray::create(&AttributeType::new("foo", "null"), 3, 1, true, None).is_err());
        assert!(AttributeArray::create(&t, 3, 0, true, None).is_err());
        let mut src = AttributeArray::with_size::<i32>(Codec::Null, 4).unwrap();
        for i in 0..4 {
            src.set(i, i as i32 * 10);
        }
        let mut dst = AttributeArray::with_size::<i32>(Codec::Null, 2).unwrap();
        dst.copy_values(&src, [(3, 0), (1, 1)], true).unwrap();
        assert_eq!(dst.get::<i32>(0), 30);
        assert_eq!(dst.get::<i32>(1), 10);
        assert!(dst.copy_values(&src, [(9, 0)], true).is_err());
    }
}
