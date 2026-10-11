//! Attribute sets and their descriptors (port of `openvdb/points/AttributeSet.h/.cc`).
//!
//! Every leaf of a point data tree owns an [`AttributeSet`]: one [`AttributeArray`] per
//! attribute, all of the same length (the leaf's point count), and a shared [`Descriptor`]
//! that names the attributes, records their types, maps group names to group bits and
//! holds metadata (default values as `"default:<name>"`, strings as `"string:<n>"`).
//!
//! Arrays and descriptors are reference counted ([`Arc`]) and copied on write, like
//! OpenVDB's shared pointers, so copying a tree is cheap.

use super::attribute_array::{AttributeArray, AttributeType};
use super::attribute_group::{GroupIndex, GROUP_BITS};
use super::{Error, Result};
use crate::metadata::{MetaMap, MetaValue};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Marker for "not found" positions (`AttributeSet::INVALID_POS`).
pub const INVALID_POS: usize = usize::MAX;

/// Names, types, groups and metadata of the attributes of an attribute set
/// (`AttributeSet::Descriptor`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Descriptor {
    name_map: BTreeMap<String, usize>,
    types: Vec<AttributeType>,
    group_map: BTreeMap<String, usize>,
    metadata: MetaMap,
}

/// True if `name` is a valid attribute or group name: non-empty, made of alphanumeric
/// characters, `_`, `|` and `:` (`Descriptor::validName`).
pub fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '|' || c == ':')
}

/// Parse a space-separated list of names with `^`/`!` exclusions and `*` wildcard
/// (`Descriptor::parseNames`). Returns (include, exclude, include_all).
pub fn parse_names(names: &str) -> Result<(Vec<String>, Vec<String>, bool)> {
    let (mut include, mut exclude, mut all) = (Vec::new(), Vec::new(), false);
    for token in names.split_whitespace() {
        if let Some(t) = token.strip_prefix('^').or_else(|| token.strip_prefix('!')) {
            if t.is_empty() {
                return Err(Error::Value("negate character (^) must prefix a name".into()));
            }
            if !valid_name(t) {
                return Err(Error::Value(format!("name contains invalid characters - {}", t)));
            }
            exclude.push(t.to_string());
        } else if !all {
            if token == "*" {
                all = true;
                include.clear();
                continue;
            }
            if !valid_name(token) {
                return Err(Error::Value(format!("name contains invalid characters - {}", token)));
            }
            include.push(token.to_string());
        }
    }
    Ok((include, exclude, all))
}

impl Descriptor {
    /// An empty descriptor.
    pub fn new() -> Self {
        Self::default()
    }
    /// A descriptor with a single position attribute `"P"` of the given type
    /// (`Descriptor::create`).
    pub fn with_position(position_type: AttributeType) -> Result<Self> {
        let mut d = Descriptor::new();
        d.insert("P", position_type)?;
        Ok(d)
    }

    fn insert(&mut self, name: &str, ty: AttributeType) -> Result<usize> {
        if !valid_name(name) {
            return Err(Error::Value(format!("attribute name contains invalid characters - {}", name)));
        }
        if let Some(&pos) = self.name_map.get(name) {
            if self.types[pos] != ty {
                return Err(Error::Key(
                    "cannot insert into a descriptor with a duplicate name but different type".into(),
                ));
            }
            return Ok(pos);
        }
        if !ty.is_registered() {
            return Err(Error::Key(format!("failed to insert '{}' with unregistered attribute type '{}'", name, ty)));
        }
        let pos = self.types.len();
        self.types.push(ty);
        self.name_map.insert(name.to_string(), pos);
        Ok(pos)
    }

    /// Build a descriptor from its parts (used by file I/O; types need not be registered).
    pub(crate) fn from_raw(
        types: Vec<AttributeType>,
        name_map: BTreeMap<String, usize>,
        group_map: BTreeMap<String, usize>,
        metadata: MetaMap,
    ) -> Self {
        Descriptor { name_map, types, group_map, metadata }
    }

    /// Number of attributes.
    pub fn size(&self) -> usize {
        self.types.len()
    }
    /// Number of attributes of the given type.
    pub fn count(&self, ty: &AttributeType) -> usize {
        self.types.iter().filter(|t| *t == ty).count()
    }
    /// Position of the attribute `name`, or [`INVALID_POS`].
    pub fn find(&self, name: &str) -> usize {
        self.name_map.get(name).copied().unwrap_or(INVALID_POS)
    }
    /// Position of the attribute `name`.
    pub fn position(&self, name: &str) -> Option<usize> {
        self.name_map.get(name).copied()
    }
    /// The name of the attribute at `pos`.
    pub fn name_at(&self, pos: usize) -> Option<&str> {
        self.name_map.iter().find(|(_, p)| **p == pos).map(|(n, _)| n.as_str())
    }
    /// Attribute names in position order.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<(usize, &String)> = self.name_map.iter().map(|(n, p)| (*p, n)).collect();
        v.sort();
        v.into_iter().map(|(_, n)| n.clone()).collect()
    }
    /// The type of the attribute at `pos`.
    pub fn attribute_type(&self, pos: usize) -> &AttributeType {
        &self.types[pos]
    }
    /// The value type name of the attribute at `pos`.
    pub fn value_type(&self, pos: usize) -> &str {
        &self.types[pos].value_type
    }
    /// All attribute types, in position order.
    pub fn types(&self) -> &[AttributeType] {
        &self.types
    }
    /// Map from attribute name to position.
    pub fn map(&self) -> &BTreeMap<String, usize> {
        &self.name_map
    }
    /// Map from group name to group offset.
    pub fn group_map(&self) -> &BTreeMap<String, usize> {
        &self.group_map
    }
    /// The descriptor metadata.
    pub fn metadata(&self) -> &MetaMap {
        &self.metadata
    }
    /// The descriptor metadata, mutably.
    pub fn metadata_mut(&mut self) -> &mut MetaMap {
        &mut self.metadata
    }

    /// Rename an attribute; returns its position, or [`INVALID_POS`] if `from` does not exist
    /// or `to` is taken.
    pub fn rename(&mut self, from: &str, to: &str) -> Result<usize> {
        if !valid_name(to) {
            return Err(Error::Value(format!("attribute name contains invalid characters - {}", to)));
        }
        if self.name_map.contains_key(to) {
            return Ok(INVALID_POS);
        }
        let Some(pos) = self.name_map.remove(from) else {
            return Ok(INVALID_POS);
        };
        self.name_map.insert(to.to_string(), pos);
        if let Some(v) = self.metadata.remove(&format!("default:{}", from)) {
            self.metadata.insert(format!("default:{}", to), v);
        }
        Ok(pos)
    }

    /// True if attribute `name` has a default value.
    pub fn has_default_value(&self, name: &str) -> bool {
        self.metadata.contains(&format!("default:{}", name))
    }
    /// The default value of attribute `name`, if any.
    pub fn default_value(&self, name: &str) -> Option<&MetaValue> {
        self.metadata.get(&format!("default:{}", name))
    }
    /// Set the default value of attribute `name`; the metadata type must match the value type.
    pub fn set_default_value(&mut self, name: &str, value: MetaValue) -> Result<()> {
        let pos = self.position(name).ok_or_else(|| Error::Lookup(format!("cannot find attribute {}", name)))?;
        if self.types[pos].value_type != value.type_name() {
            return Err(Error::Type("mismatching default value type".into()));
        }
        self.metadata.insert(format!("default:{}", name), value);
        Ok(())
    }
    /// Remove the default value of attribute `name`.
    pub fn remove_default_value(&mut self, name: &str) {
        self.metadata.remove(&format!("default:{}", name));
    }
    /// Remove default values of attributes that no longer exist.
    pub fn prune_unused_default_values(&mut self) {
        let stale: Vec<String> = self
            .metadata
            .iter()
            .filter_map(|(k, _)| {
                k.strip_prefix("default:").filter(|n| !self.name_map.contains_key(*n)).map(|_| k.clone())
            })
            .collect();
        for k in stale {
            self.metadata.remove(&k);
        }
    }

    /// A copy with an extra attribute appended (`duplicateAppend`).
    pub fn duplicate_append(&self, name: &str, ty: AttributeType) -> Result<Descriptor> {
        let mut d = Descriptor {
            name_map: BTreeMap::new(),
            types: Vec::new(),
            group_map: self.group_map.clone(),
            metadata: self.metadata.clone(),
        };
        for (n, t) in self.names().iter().zip(&self.types) {
            d.insert(n, t.clone())?;
        }
        d.insert(name, ty)?;
        Ok(d)
    }

    /// A copy without the attributes at `positions`, with group offsets adjusted when group
    /// arrays are dropped (`duplicateDrop`).
    pub fn duplicate_drop(&self, positions: &[usize]) -> Result<Descriptor> {
        let names = self.names();
        let mut drop: Vec<usize> = positions.to_vec();
        drop.sort_unstable();
        drop.dedup();
        if drop.last().is_some_and(|&p| p >= names.len()) {
            return Err(Error::Lookup("cannot erase indices as index is out of range".into()));
        }
        let group_type = AttributeType::group();
        let groups: Vec<usize> = (0..self.types.len()).filter(|&i| self.types[i] == group_type).collect();
        let groups_to_drop: Vec<usize> = groups.iter().copied().filter(|g| drop.contains(g)).collect();
        let mut group_map = self.group_map.clone();
        if !groups_to_drop.is_empty() {
            group_map.retain(|_, off| {
                let array_pos = groups[*off / GROUP_BITS];
                !drop.contains(&array_pos)
            });
            for off in group_map.values_mut() {
                let array_pos = groups[*off / GROUP_BITS];
                let shift = groups_to_drop.iter().filter(|&&g| g < array_pos).count();
                *off -= shift * GROUP_BITS;
            }
        }
        let mut d =
            Descriptor { name_map: BTreeMap::new(), types: Vec::new(), group_map, metadata: self.metadata.clone() };
        for (i, (n, t)) in names.iter().zip(&self.types).enumerate() {
            if !drop.contains(&i) {
                d.insert(n, t.clone())?;
            }
        }
        d.prune_unused_default_values();
        Ok(d)
    }

    /// True if both descriptors have the same attributes (names and types, any order) and
    /// groups.
    pub fn has_same_attributes(&self, rhs: &Descriptor) -> bool {
        if self.types.len() != rhs.types.len() || self.group_map != rhs.group_map {
            return false;
        }
        self.name_map.iter().all(|(n, &p)| rhs.position(n).is_some_and(|q| self.types[p] == rhs.types[q]))
    }

    /// True if group `name` exists.
    pub fn has_group(&self, name: &str) -> bool {
        self.group_map.contains_key(name)
    }
    /// Map group `name` to the given offset.
    pub fn set_group(&mut self, name: &str, offset: usize, check_valid_offset: bool) -> Result<()> {
        if !valid_name(name) {
            return Err(Error::Value(format!("group name contains invalid characters - {}", name)));
        }
        if check_valid_offset {
            if offset >= self.available_groups() {
                return Err(Error::Value(format!("group offset is out-of-range - {}", name)));
            }
            if self.group_map.values().any(|&o| o == offset) {
                return Err(Error::Value(format!("group offset is already in use - {}", name)));
            }
        }
        self.group_map.insert(name.to_string(), offset);
        Ok(())
    }
    /// Remove group `name`.
    pub fn drop_group(&mut self, name: &str) {
        self.group_map.remove(name);
    }
    /// Remove all groups.
    pub fn clear_groups(&mut self) {
        self.group_map.clear();
    }
    /// Rename a group; returns its offset or [`INVALID_POS`].
    pub fn rename_group(&mut self, from: &str, to: &str) -> Result<usize> {
        if !valid_name(to) {
            return Err(Error::Value(format!("group name contains invalid characters - {}", to)));
        }
        if self.group_map.contains_key(to) {
            return Ok(INVALID_POS);
        }
        match self.group_map.remove(from) {
            Some(o) => {
                self.group_map.insert(to.to_string(), o);
                Ok(o)
            }
            None => Ok(INVALID_POS),
        }
    }
    /// `name`, or `name` with the smallest numeric suffix that is not an attribute name.
    pub fn unique_name(&self, name: &str) -> String {
        unique(name, |n| self.name_map.contains_key(n))
    }
    /// `name`, or `name` with the smallest numeric suffix that is not a group name.
    pub fn unique_group_name(&self, name: &str) -> String {
        unique(name, |n| self.group_map.contains_key(n))
    }
    /// The offset of group `name`, if it exists.
    pub fn group_offset(&self, name: &str) -> Option<usize> {
        self.group_map.get(name).copied()
    }
    /// The offset of a group index.
    pub fn group_offset_of_index(&self, index: GroupIndex) -> Result<usize> {
        if index.0 >= self.types.len() || self.types[index.0] != AttributeType::group() {
            return Err(Error::Lookup("group index invalid".into()));
        }
        let group_type = AttributeType::group();
        let relative = (0..index.0).filter(|&i| self.types[i] == group_type).count();
        Ok(relative * GROUP_BITS + index.1 as usize)
    }
    /// The group index (array position, bit) of group `name`.
    pub fn group_index(&self, name: &str) -> Result<GroupIndex> {
        let off = self.group_offset(name).ok_or_else(|| Error::Lookup(format!("group not found - {}", name)))?;
        self.group_index_of_offset(off)
    }
    /// The group index of a group offset.
    pub fn group_index_of_offset(&self, offset: usize) -> Result<GroupIndex> {
        let group_type = AttributeType::group();
        let groups: Vec<usize> = (0..self.types.len()).filter(|&i| self.types[i] == group_type).collect();
        if offset >= groups.len() * GROUP_BITS {
            return Err(Error::Lookup(format!("out of range group offset - {}", offset)));
        }
        Ok((groups[offset / GROUP_BITS], (offset % GROUP_BITS) as u8))
    }
    /// Number of group bits available in the group arrays.
    pub fn available_groups(&self) -> usize {
        self.count(&AttributeType::group()) * GROUP_BITS
    }
    /// Number of unused group bits.
    pub fn unused_groups(&self) -> usize {
        self.available_groups().saturating_sub(self.group_map.len())
    }
    /// True if at least a whole group array is unused.
    pub fn can_compact_groups(&self) -> bool {
        self.unused_groups() >= GROUP_BITS
    }
    /// An unused group offset (`hint` if free), or `None` if all are in use.
    pub fn unused_group_offset(&self, hint: Option<usize>) -> Option<usize> {
        if self.unused_groups() == 0 {
            return None;
        }
        let mut used: Vec<usize> = self.group_map.values().copied().collect();
        used.sort_unstable();
        if let Some(h) = hint {
            if h < self.available_groups() && !used.contains(&h) {
                return Some(h);
            }
        }
        let mut offset = 0;
        for i in used {
            if i != offset {
                break;
            }
            offset += 1;
        }
        Some(offset)
    }
    /// If a group lies at or beyond the first unused offset, return (group name, its offset,
    /// target offset) for compaction (`requiresGroupMove`).
    pub fn requires_group_move(&self) -> Option<(String, usize, usize)> {
        let target = self.unused_group_offset(None)?;
        self.group_map.iter().find(|(_, &o)| o >= target).map(|(n, &o)| (n.clone(), o, target))
    }
    /// True if a group exists in both descriptors with different offsets.
    pub fn group_index_collision(&self, rhs: &Descriptor) -> bool {
        self.group_map.iter().any(|(n, o)| rhs.group_map.get(n).is_some_and(|p| p != o))
    }
}

fn unique(name: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(name) {
        return name.to_string();
    }
    (0..).map(|i| format!("{}{}", name, i)).find(|n| !taken(n)).unwrap()
}

/// The attribute arrays of a leaf plus their descriptor (`AttributeSet`).
#[derive(Clone, Debug)]
pub struct AttributeSet {
    descr: Arc<Descriptor>,
    attrs: Vec<Arc<AttributeArray>>,
}

impl Default for AttributeSet {
    fn default() -> Self {
        AttributeSet { descr: Arc::new(Descriptor::new()), attrs: Vec::new() }
    }
}

impl PartialEq for AttributeSet {
    fn eq(&self, other: &Self) -> bool {
        *self.descr == *other.descr
            && self.attrs.len() == other.attrs.len()
            && self.attrs.iter().zip(&other.attrs).all(|(a, b)| **a == **b)
    }
}

impl AttributeSet {
    /// An empty attribute set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Arrays of length `n` for every attribute of `descr`, initialized with their default
    /// values.
    pub fn from_descriptor(descr: Arc<Descriptor>, n: u32) -> Result<Self> {
        let mut attrs = Vec::with_capacity(descr.size());
        for (pos, ty) in descr.types.iter().enumerate() {
            let name = descr.name_at(pos).unwrap_or("");
            let default = descr.default_value(name);
            attrs.push(Arc::new(AttributeArray::create(ty, n, 1, true, default)?));
        }
        Ok(AttributeSet { descr, attrs })
    }

    /// A set with the same descriptor, array types, strides and flags as `other` but arrays
    /// of length `n` (initialized with default values).
    pub fn like(other: &AttributeSet, n: u32) -> Result<Self> {
        let descr = other.descr.clone();
        let mut attrs = Vec::with_capacity(other.attrs.len());
        for (pos, existing) in other.attrs.iter().enumerate() {
            let name = descr.name_at(pos).unwrap_or("");
            let constant = existing.has_constant_stride();
            let stride = if constant { existing.stride() } else { existing.data_size() };
            let mut a =
                AttributeArray::create(existing.attribute_type(), n, stride, constant, descr.default_value(name))?;
            a.set_hidden(existing.is_hidden());
            a.set_transient(existing.is_transient());
            attrs.push(Arc::new(a));
        }
        Ok(AttributeSet { descr, attrs })
    }

    /// Assemble a set from a descriptor and arrays (used by file I/O).
    pub(crate) fn from_parts(descr: Arc<Descriptor>, attrs: Vec<AttributeArray>) -> Self {
        AttributeSet { descr, attrs: attrs.into_iter().map(Arc::new).collect() }
    }

    /// The descriptor.
    pub fn descriptor(&self) -> &Descriptor {
        &self.descr
    }
    /// The shared descriptor pointer.
    pub fn descriptor_ptr(&self) -> &Arc<Descriptor> {
        &self.descr
    }
    /// Mutable access to the descriptor (copied first if shared).
    pub fn descriptor_mut(&mut self) -> &mut Descriptor {
        Arc::make_mut(&mut self.descr)
    }
    /// Number of attributes.
    pub fn size(&self) -> usize {
        self.attrs.len()
    }
    /// True if there are no attributes.
    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }
    /// Position of attribute `name`, or [`INVALID_POS`].
    pub fn find(&self, name: &str) -> usize {
        self.descr.find(name)
    }
    /// The array at `pos`.
    pub fn get(&self, pos: usize) -> Option<&AttributeArray> {
        self.attrs.get(pos).map(|a| &**a)
    }
    /// The array at `pos`, mutably (copied first if shared).
    pub fn get_mut(&mut self, pos: usize) -> Option<&mut AttributeArray> {
        self.attrs.get_mut(pos).map(Arc::make_mut)
    }
    /// The array named `name`.
    pub fn get_by_name(&self, name: &str) -> Option<&AttributeArray> {
        self.descr.position(name).and_then(|p| self.get(p))
    }
    /// The array named `name`, mutably.
    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut AttributeArray> {
        let p = self.descr.position(name)?;
        self.get_mut(p)
    }
    /// The shared pointer of the array at `pos`.
    pub fn array_ptr(&self, pos: usize) -> &Arc<AttributeArray> {
        &self.attrs[pos]
    }
    /// All arrays, in position order.
    pub fn arrays(&self) -> impl Iterator<Item = &AttributeArray> + '_ {
        self.attrs.iter().map(|a| &**a)
    }
    /// The descriptor and all arrays, mutably (for parallel per-array updates).
    pub fn arrays_mut(&mut self) -> (&Descriptor, Vec<&mut AttributeArray>) {
        (&self.descr, self.attrs.iter_mut().map(Arc::make_mut).collect())
    }
    /// True if the array at `pos` is shared with another set.
    pub fn is_shared(&self, pos: usize) -> bool {
        Arc::strong_count(&self.attrs[pos]) > 1
    }
    /// Make the array at `pos` unique.
    pub fn make_unique(&mut self, pos: usize) {
        Arc::make_mut(&mut self.attrs[pos]);
    }
    /// Replace the array at `pos`; fails if the type differs from the descriptor's.
    pub fn replace(&mut self, pos: usize, array: AttributeArray) -> Result<()> {
        if pos >= self.attrs.len() || *array.attribute_type() != self.descr.types[pos] {
            return Err(Error::Type("cannot replace attribute with an array of a different type".into()));
        }
        self.attrs[pos] = Arc::new(array);
        Ok(())
    }
    /// The (array position, bit) of group `name`.
    pub fn group_index(&self, name: &str) -> Result<GroupIndex> {
        self.descr.group_index(name)
    }
    /// The offset of group `name`.
    pub fn group_offset(&self, name: &str) -> Option<usize> {
        self.descr.group_offset(name)
    }
    /// Positions of all group arrays.
    pub fn group_attribute_indices(&self) -> Vec<usize> {
        (0..self.attrs.len()).filter(|&i| self.attrs[i].is_group()).collect()
    }

    /// Append an attribute (creating a new descriptor); returns its position.
    pub fn append_attribute(
        &mut self,
        name: &str,
        ty: AttributeType,
        stride_or_total: u32,
        constant_stride: bool,
        default: Option<MetaValue>,
    ) -> Result<usize> {
        let mut d = self.descr.duplicate_append(name, ty)?;
        if let Some(v) = &default {
            d.set_default_value(name, v.clone())?;
        }
        let pos = d.find(name);
        let expected = (*self.descr).clone();
        self.append_attribute_with(&expected, &Arc::new(d), pos, stride_or_total, constant_stride, default.as_ref())?;
        Ok(pos)
    }

    /// Append the attribute at `pos` of `replacement` (which must extend `expected`, the
    /// current descriptor).
    pub fn append_attribute_with(
        &mut self,
        expected: &Descriptor,
        replacement: &Arc<Descriptor>,
        pos: usize,
        stride_or_total: u32,
        constant_stride: bool,
        default: Option<&MetaValue>,
    ) -> Result<()> {
        if *self.descr != *expected {
            return Err(Error::Lookup("cannot append attributes as descriptors do not match".into()));
        }
        let len = if self.attrs.is_empty() { 1 } else { self.attrs[0].size() };
        let ty = replacement.attribute_type(pos);
        let array = AttributeArray::create(ty, len, stride_or_total, constant_stride, default)?;
        self.descr = replacement.clone();
        self.attrs.push(Arc::new(array));
        Ok(())
    }

    /// Remove the attribute `name`, returning its array.
    pub fn remove_attribute(&mut self, name: &str) -> Result<Option<AttributeArray>> {
        let Some(pos) = self.descr.position(name) else {
            return Ok(None);
        };
        let a = self.attrs[pos].clone();
        self.drop_attributes(&[pos])?;
        Ok(Some(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())))
    }

    /// Drop the attributes at `positions`.
    pub fn drop_attributes(&mut self, positions: &[usize]) -> Result<()> {
        if positions.is_empty() {
            return Ok(());
        }
        let d = Arc::new(self.descr.duplicate_drop(positions)?);
        let expected = (*self.descr).clone();
        self.drop_attributes_with(positions, &expected, &d)
    }

    /// Drop the attributes at `positions`, replacing the descriptor with `replacement`.
    pub fn drop_attributes_with(
        &mut self,
        positions: &[usize],
        expected: &Descriptor,
        replacement: &Arc<Descriptor>,
    ) -> Result<()> {
        if positions.is_empty() {
            return Ok(());
        }
        if *self.descr != *expected {
            return Err(Error::Lookup("cannot drop attributes as descriptors do not match".into()));
        }
        let mut drop = positions.to_vec();
        drop.sort_unstable();
        drop.dedup();
        for &p in drop.iter().rev() {
            if p >= self.attrs.len() {
                return Err(Error::Lookup("cannot erase indices as index is out of range".into()));
            }
            self.attrs.remove(p);
        }
        self.descr = replacement.clone();
        Arc::make_mut(&mut self.descr).prune_unused_default_values();
        Ok(())
    }

    /// Replace the descriptor with a renamed equivalent.
    pub fn rename_attributes(&mut self, expected: &Descriptor, replacement: &Arc<Descriptor>) -> Result<()> {
        if *self.descr != *expected {
            return Err(Error::Lookup("cannot rename attribute as descriptors do not match".into()));
        }
        self.descr = replacement.clone();
        Ok(())
    }

    /// Reorder the arrays to match `replacement`, which must have the same attributes.
    pub fn reorder_attributes(&mut self, replacement: &Arc<Descriptor>) -> Result<()> {
        if *self.descr == **replacement {
            self.descr = replacement.clone();
            return Ok(());
        }
        if !self.descr.has_same_attributes(replacement) {
            return Err(Error::Lookup(
                "cannot reorder attributes as descriptors do not contain the same attributes".into(),
            ));
        }
        let mut attrs = self.attrs.clone();
        for (name, &pos) in self.descr.map() {
            attrs[replacement.find(name)] = self.attrs[pos].clone();
        }
        self.attrs = attrs;
        self.descr = replacement.clone();
        Ok(())
    }

    /// Replace the descriptor (which must be equal unless `allow_mismatch`).
    pub fn reset_descriptor(&mut self, replacement: Arc<Descriptor>, allow_mismatch: bool) -> Result<()> {
        if !allow_mismatch && *self.descr != *replacement {
            return Err(Error::Lookup("cannot swap descriptor as replacement does not match".into()));
        }
        self.descr = replacement;
        Ok(())
    }

    /// Approximate memory usage in bytes.
    pub fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.attrs.iter().map(|a| a.mem_usage()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::Codec;

    fn descriptor() -> Descriptor {
        Descriptor::with_position(AttributeType::new("vec3s", "fxpt16")).unwrap()
    }

    #[test]
    fn names_and_parsing() {
        assert!(valid_name("a_b|c:1"));
        assert!(!valid_name("a b") && !valid_name(""));
        let (inc, exc, all) = parse_names("a ^b !c").unwrap();
        assert_eq!(inc, vec!["a"]);
        assert_eq!(exc, vec!["b", "c"]);
        assert!(!all);
        let (inc, _, all) = parse_names("a * d").unwrap();
        assert!(all && inc.is_empty());
        assert!(parse_names("^").is_err());
    }

    #[test]
    fn append_drop_groups() {
        let d = descriptor();
        let d = d.duplicate_append("density", AttributeType::new("float", "null")).unwrap();
        let mut d = d.duplicate_append("grp0", AttributeType::group()).unwrap();
        let mut d2 = d.duplicate_append("grp1", AttributeType::group()).unwrap();
        assert_eq!(d2.available_groups(), 16);
        d2.set_group("a", 3, true).unwrap();
        d2.set_group("b", 9, true).unwrap();
        assert!(d2.set_group("c", 9, true).is_err());
        assert_eq!(d2.group_index("a").unwrap(), (2, 3));
        assert_eq!(d2.group_index("b").unwrap(), (3, 1));
        assert_eq!(d2.group_offset_of_index((3, 1)).unwrap(), 9);
        assert_eq!(d2.unused_group_offset(None), Some(0));
        assert_eq!(d2.unused_group_offset(Some(5)), Some(5));
        // Dropping the first group array removes group "a" and shifts "b".
        let d3 = d2.duplicate_drop(&[2]).unwrap();
        assert!(!d3.has_group("a"));
        assert_eq!(d3.group_offset("b"), Some(1));
        assert_eq!(d3.names(), vec!["P", "density", "grp1"]);
        d.set_default_value("density", MetaValue::Float(2.0)).unwrap();
        assert!(d.set_default_value("density", MetaValue::Int32(2)).is_err());
        let d4 = d.duplicate_drop(&[1]).unwrap();
        assert!(!d4.has_default_value("density"));
        assert_eq!(d.unique_name("P"), "P0");
        assert_eq!(d.rename("density", "rho").unwrap(), 1);
        assert!(d.has_default_value("rho"));
    }

    #[test]
    fn attribute_set_ops() {
        let d = Arc::new(descriptor());
        let mut set = AttributeSet::from_descriptor(d, 5).unwrap();
        assert_eq!(set.size(), 1);
        assert_eq!(set.get(0).unwrap().size(), 5);
        let pos = set
            .append_attribute("id", AttributeType::new("int32", "null"), 1, true, Some(MetaValue::Int32(7)))
            .unwrap();
        assert_eq!(pos, 1);
        assert_eq!(set.get(1).unwrap().get::<i32>(4), 7);
        assert_eq!(set.get(1).unwrap().size(), 5);
        let copy = set.clone();
        assert!(set.is_shared(1));
        set.get_mut(1).unwrap().set(0, 1i32);
        assert!(!set.is_shared(1));
        assert_ne!(set, copy);
        let removed = set.remove_attribute("id").unwrap().unwrap();
        assert_eq!(removed.get::<i32>(0), 1);
        assert_eq!(set.size(), 1);
        assert!(!set.descriptor().has_default_value("id"));
        let like = AttributeSet::like(&copy, 2).unwrap();
        assert_eq!(like.get(1).unwrap().size(), 2);
        assert_eq!(like.get(1).unwrap().get::<i32>(1), 7);
        assert!(set.get(0).unwrap().is_type::<crate::math::Vec3f>(Codec::FIXED_POINT_16));
    }
}
