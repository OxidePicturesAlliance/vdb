//! String attributes (port of `openvdb/points/AttributeArrayString.h`).
//!
//! A string array (`("uint32", "str")`) stores per-point indices into string metadata held
//! by the attribute set's descriptor: index `i > 0` refers to the metadata entry
//! `"string:<i-1>"`, and index 0 is the empty string.

use super::attribute_array::{AttributeArray, AttributeHandle, AttributeWriteHandle};
use super::{Error, Result};
use crate::metadata::{MetaMap, MetaValue};
use std::collections::HashMap;

/// The metadata key of string index `index` (> 0).
pub fn string_key(index: u32) -> String {
    format!("string:{}", index - 1)
}

/// Map from string to index for all string metadata in `metadata` (`StringMetaCache`).
pub fn string_cache(metadata: &MetaMap) -> HashMap<String, u32> {
    let mut cache = HashMap::new();
    for (key, value) in metadata.iter() {
        let (Some(rest), MetaValue::String(s)) = (key.strip_prefix("string:"), value) else {
            continue;
        };
        if let Ok(i) = rest.parse::<u32>() {
            cache.insert(s.clone(), i + 1);
        }
    }
    cache
}

/// Inserts strings into metadata, choosing unused indices (`StringMetaInserter`).
pub struct StringMetaInserter<'a> {
    metadata: &'a mut MetaMap,
    cache: HashMap<String, u32>,
    /// Sorted, non-overlapping blocks of used indices: (start, length).
    blocks: Vec<(u32, u32)>,
}

impl<'a> StringMetaInserter<'a> {
    /// Create an inserter for `metadata`.
    pub fn new(metadata: &'a mut MetaMap) -> Self {
        let mut s = StringMetaInserter { metadata, cache: HashMap::new(), blocks: Vec::new() };
        s.reset_cache();
        s
    }
    /// True if `name` is already stored.
    pub fn has_key(&self, name: &str) -> bool {
        self.cache.contains_key(name)
    }
    /// True if string index `index` is in use.
    pub fn has_index(&self, index: u32) -> bool {
        index > 0 && self.metadata.contains(&string_key(index))
    }
    /// Insert `name` (if not present) and return its index. A non-zero `hint` is used as the
    /// index if it is free.
    pub fn insert(&mut self, name: &str, hint: u32) -> u32 {
        if let Some(&i) = self.cache.get(name) {
            return i;
        }
        let index = if hint > 0 && !self.metadata.contains(&string_key(hint)) {
            hint
        } else {
            // The first index not covered by a block of used indices.
            let mut index = 1u32;
            for &(start, len) in &self.blocks {
                if index < start {
                    break;
                }
                index = index.max(start + len);
            }
            index
        };
        // Record the new index in the sorted block list, merging adjacent blocks.
        let pos = self.blocks.partition_point(|&(s, _)| s < index);
        self.blocks.insert(pos, (index, 1));
        if pos + 1 < self.blocks.len() && index + 1 == self.blocks[pos + 1].0 {
            self.blocks[pos].1 += self.blocks[pos + 1].1;
            self.blocks.remove(pos + 1);
        }
        if pos > 0 && self.blocks[pos - 1].0 + self.blocks[pos - 1].1 == index {
            self.blocks[pos - 1].1 += self.blocks[pos].1;
            self.blocks.remove(pos);
        }
        self.metadata.insert(string_key(index), MetaValue::String(name.to_string()));
        self.cache.insert(name.to_string(), index);
        index
    }
    /// Rebuild the cache from the metadata.
    pub fn reset_cache(&mut self) {
        self.cache = string_cache(self.metadata);
        let mut ids: Vec<u32> = self.cache.values().copied().collect();
        ids.sort_unstable();
        self.blocks.clear();
        for id in ids {
            match self.blocks.last_mut() {
                Some((s, l)) if *s + *l == id => *l += 1,
                _ => self.blocks.push((id, 1)),
            }
        }
    }
}

/// Read access to a string array (`StringAttributeHandle`).
pub struct StringAttributeHandle<'a> {
    handle: AttributeHandle<'a, u32>,
    metadata: &'a MetaMap,
}

impl<'a> StringAttributeHandle<'a> {
    /// Bind a handle to a string array and the descriptor metadata holding its strings.
    pub fn new(array: &'a AttributeArray, metadata: &'a MetaMap) -> Result<Self> {
        if !array.is_string() {
            return Err(Error::Type("cannot create a string handle for a non-string attribute array".into()));
        }
        Ok(StringAttributeHandle { handle: AttributeHandle::new(array)?, metadata })
    }
    /// Values per element.
    pub fn stride(&self) -> u32 {
        self.handle.stride()
    }
    /// Number of elements.
    pub fn size(&self) -> u32 {
        self.handle.size()
    }
    /// True if the array is uniform.
    pub fn is_uniform(&self) -> bool {
        self.handle.is_uniform()
    }
    /// The string of element `n`.
    pub fn get(&self, n: u32) -> Result<String> {
        self.get_strided(n, 0)
    }
    /// The string of value `m` of element `n`.
    pub fn get_strided(&self, n: u32, m: u32) -> Result<String> {
        let index = self.handle.get_strided(n, m);
        if index == 0 {
            return Ok(String::new());
        }
        match self.metadata.get(&string_key(index)) {
            Some(MetaValue::String(s)) => Ok(s.clone()),
            _ => Err(Error::Lookup(format!("string attribute cannot be found with index {}", index))),
        }
    }
}

/// Read-write access to a string array (`StringAttributeWriteHandle`). Strings must have been
/// inserted into the metadata beforehand (see [`StringMetaInserter`]).
pub struct StringAttributeWriteHandle<'a> {
    handle: AttributeWriteHandle<'a, u32>,
    cache: HashMap<String, u32>,
}

impl<'a> StringAttributeWriteHandle<'a> {
    /// Bind a handle (expanding the array if `expand`).
    pub fn new(array: &'a mut AttributeArray, metadata: &MetaMap, expand: bool) -> Result<Self> {
        if !array.is_string() {
            return Err(Error::Type("cannot create a string handle for a non-string attribute array".into()));
        }
        Ok(StringAttributeWriteHandle {
            handle: AttributeWriteHandle::with_expand(array, expand)?,
            cache: string_cache(metadata),
        })
    }
    fn index(&self, name: &str) -> Result<u32> {
        if name.is_empty() {
            return Ok(0);
        }
        self.cache.get(name).copied().ok_or_else(|| {
            Error::Lookup(format!("string does not exist in metadata, insert it and reset the cache - \"{}\"", name))
        })
    }
    /// True if `name` can be set (it is empty or present in the metadata).
    pub fn contains(&self, name: &str) -> bool {
        name.is_empty() || self.cache.contains_key(name)
    }
    /// Set element `n` to `name`.
    pub fn set(&mut self, n: u32, name: &str) -> Result<()> {
        let i = self.index(name)?;
        self.handle.set(n, i);
        Ok(())
    }
    /// Set value `m` of element `n` to `name`.
    pub fn set_strided(&mut self, n: u32, m: u32, name: &str) -> Result<()> {
        let i = self.index(name)?;
        self.handle.set_strided(n, m, i);
        Ok(())
    }
    /// Collapse to the uniform string `name`.
    pub fn collapse(&mut self, name: &str) -> Result<()> {
        let i = self.index(name)?;
        self.handle.collapse_to(i);
        Ok(())
    }
    /// Set every element to `name`.
    pub fn fill(&mut self, name: &str) -> Result<()> {
        let i = self.index(name)?;
        self.handle.fill(i);
        Ok(())
    }
    /// Collapse if uniform.
    pub fn compact(&mut self) -> bool {
        self.handle.compact()
    }
    /// Expand the array.
    pub fn expand(&mut self) {
        self.handle.expand(true)
    }
    /// Refresh the string cache from (updated) metadata.
    pub fn reset_cache(&mut self, metadata: &MetaMap) {
        self.cache = string_cache(metadata);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::{AttributeType, Codec};

    #[test]
    fn inserter_indices() {
        let mut meta = MetaMap::new();
        {
            let mut ins = StringMetaInserter::new(&mut meta);
            assert_eq!(ins.insert("a", 0), 1);
            assert_eq!(ins.insert("b", 0), 2);
            assert_eq!(ins.insert("a", 0), 1);
            assert_eq!(ins.insert("e", 5), 5);
            assert_eq!(ins.insert("c", 0), 3);
            assert_eq!(ins.insert("d", 0), 4);
            assert_eq!(ins.insert("f", 0), 6);
            assert_eq!(ins.insert("g", 2), 7);
            assert!(ins.has_index(5) && !ins.has_index(8));
        }
        assert_eq!(meta.get_str("string:0"), Some("a"));
        assert_eq!(meta.get_str("string:4"), Some("e"));
        let mut ins = StringMetaInserter::new(&mut meta);
        assert!(ins.has_key("g"));
        assert_eq!(ins.insert("h", 0), 8);
    }

    #[test]
    fn handles() {
        let mut meta = MetaMap::new();
        StringMetaInserter::new(&mut meta).insert("hello", 0);
        let mut a = AttributeArray::create(&AttributeType::string(), 3, 1, true, None).unwrap();
        assert_eq!(a.codec(), Some(Codec::String));
        {
            let mut w = StringAttributeWriteHandle::new(&mut a, &meta, true).unwrap();
            w.set(1, "hello").unwrap();
            assert!(w.set(0, "nope").is_err());
        }
        let h = StringAttributeHandle::new(&a, &meta).unwrap();
        assert_eq!(h.get(0).unwrap(), "");
        assert_eq!(h.get(1).unwrap(), "hello");
    }
}
