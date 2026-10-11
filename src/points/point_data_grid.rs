//! Point data leaves, trees and grids (port of `openvdb/points/PointDataGrid.h`).
//!
//! A [`PointDataTree`] has the topology of a standard `Tree<u32>` (OpenVDB's
//! `Tree_ptdataidx32_5_4_3`): the voxel values of each leaf are *end offsets* into the leaf's
//! attribute arrays, so voxel `n` holds the points `value(n-1)..value(n)`, and the leaf's
//! last value is its point count. Each leaf additionally owns an [`AttributeSet`].
//!
//! The offsets live in a regular [`Tree<u32>`] (so all tree algorithms work on the
//! topology) and the attribute sets are stored per leaf origin; leaves are accessed through
//! the [`PointDataLeaf`] and [`PointDataLeafMut`] views (or the owned [`PointDataLeafNode`]),
//! whose common API is provided by the [`PointLeaf`] and [`PointLeafMut`] traits.

use super::attribute_array::{AttributeArray, AttributeHandle};
use super::attribute_group::{GroupHandle, GroupIndex, GroupWriteHandle};
use super::attribute_set::{AttributeSet, Descriptor};
use super::index_filter::{IndexFilter, NullBound};
use super::index_iterator::{IndexIter, IndexState, VoxelSelection};
use super::{Error, Result};
use crate::grid::{meta_names, GridClass, VecType};
use crate::math::{Coord, CoordBBox, Transform, Vec3d, Vec3f};
use crate::metadata::{MetaMap, MetaValue};
use crate::tree::{LeafNode, Node, Tree};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Read access shared by all point data leaf types.
pub trait PointLeaf {
    /// The voxel offsets and active mask.
    fn leaf(&self) -> &LeafNode<u32>;
    /// The attribute arrays.
    fn attribute_set(&self) -> &AttributeSet;

    /// Origin of the leaf.
    fn origin(&self) -> Coord {
        self.leaf().origin()
    }
    /// Number of points in the leaf (`pointCount()`, the last voxel offset).
    fn point_count(&self) -> u64 {
        self.leaf().buffer()[511] as u64
    }
    /// Number of points in active voxels (`onPointCount()`).
    fn on_point_count(&self) -> u64 {
        let leaf = self.leaf();
        if leaf.is_empty() {
            0
        } else if leaf.is_dense() {
            self.point_count()
        } else {
            leaf.value_mask().iter_on().map(|n| self.voxel_range(n).len() as u64).sum()
        }
    }
    /// Number of points in inactive voxels (`offPointCount()`).
    fn off_point_count(&self) -> u64 {
        self.point_count() - self.on_point_count()
    }
    /// Number of points in group `name` (0 if the group does not exist).
    fn group_point_count(&self, name: &str) -> u64 {
        let Ok(index) = self.attribute_set().group_index(name) else { return 0 };
        let Ok(h) = self.group_handle(index) else { return 0 };
        (0..self.point_count() as u32).filter(|&i| h.get(i)).count() as u64
    }
    /// The point index range of the voxel at linear offset `n`.
    fn voxel_range(&self, n: usize) -> std::ops::Range<u32> {
        let buf = self.leaf().buffer();
        let start = if n == 0 { 0 } else { buf[n - 1] };
        start..buf[n].max(start)
    }
    /// True if the leaf has attribute `name`.
    fn has_attribute(&self, name: &str) -> bool {
        self.attribute_set().descriptor().position(name).is_some()
    }
    /// The array of attribute `name`.
    fn attribute_array(&self, name: &str) -> Result<&AttributeArray> {
        self.attribute_set().get_by_name(name).ok_or_else(|| Error::Lookup(format!("attribute not found - {}", name)))
    }
    /// The array at position `pos`.
    fn attribute_array_at(&self, pos: usize) -> Result<&AttributeArray> {
        self.attribute_set().get(pos).ok_or_else(|| Error::Lookup(format!("attribute out of range - {}", pos)))
    }
    /// A read handle on attribute `name` as values of `T`.
    fn attribute_handle<T: super::AttributeValue>(&self, name: &str) -> Result<AttributeHandle<'_, T>> {
        AttributeHandle::new(self.attribute_array(name)?)
    }
    /// A handle on a group.
    fn group_handle(&self, index: GroupIndex) -> Result<GroupHandle<'_>> {
        GroupHandle::new(self.attribute_array_at(index.0)?, index.1)
    }
    /// A handle on group `name`.
    fn group_handle_by_name(&self, name: &str) -> Result<GroupHandle<'_>> {
        let index = self.attribute_set().group_index(name)?;
        self.group_handle(index)
    }

    /// Indices of all points (`beginIndexAll()`).
    fn index_all(&self) -> IndexIter<'_, NullBound> {
        IndexIter::new(self.leaf(), VoxelSelection::All, NullBound)
    }
    /// Indices of the points in active voxels (`beginIndexOn()`).
    fn index_on(&self) -> IndexIter<'_, NullBound> {
        IndexIter::new(self.leaf(), VoxelSelection::On, NullBound)
    }
    /// Indices of the points in inactive voxels (`beginIndexOff()`).
    fn index_off(&self) -> IndexIter<'_, NullBound> {
        IndexIter::new(self.leaf(), VoxelSelection::Off, NullBound)
    }
    /// Indices of the points in the voxel containing `ijk` (`beginIndexVoxel()`).
    fn index_voxel(&self, ijk: &Coord) -> IndexIter<'_, NullBound> {
        IndexIter::new(self.leaf(), VoxelSelection::Voxel(LeafNode::<u32>::coord_to_offset(ijk)), NullBound)
    }
    /// Filtered point indices of the selected voxels (`beginIndex<Iter, Filter>()`).
    fn index_filtered<'a, F: IndexFilter>(
        &'a self,
        selection: VoxelSelection,
        filter: &'a F,
    ) -> IndexIter<'a, F::Bound<'a>> {
        let sel = if filter.state() == IndexState::None { VoxelSelection::Empty } else { selection };
        IndexIter::new(self.leaf(), sel, filter.bind(self.leaf(), self.attribute_set()))
    }
    /// Filtered indices of the points in active voxels (`beginIndexOn(filter)`).
    fn index_on_filtered<'a, F: IndexFilter>(&'a self, filter: &'a F) -> IndexIter<'a, F::Bound<'a>> {
        self.index_filtered(VoxelSelection::On, filter)
    }
    /// Filtered indices of all points (`beginIndexAll(filter)`).
    fn index_all_filtered<'a, F: IndexFilter>(&'a self, filter: &'a F) -> IndexIter<'a, F::Bound<'a>> {
        self.index_filtered(VoxelSelection::All, filter)
    }

    /// Check that the offsets increase monotonically and match the attribute array lengths
    /// (`validateOffsets()`).
    fn validate_offsets(&self) -> Result<()> {
        let buf = self.leaf().buffer();
        if buf.windows(2).any(|w| w[0] > w[1]) {
            return Err(Error::Value("voxel offset values are not monotonically increasing".into()));
        }
        let set = self.attribute_set();
        let sizes: Vec<u32> = set.arrays().map(|a| a.size()).collect();
        if sizes.windows(2).any(|w| w[0] != w[1]) {
            return Err(Error::Value("attribute arrays have inconsistent length".into()));
        }
        if let Some(&s) = sizes.first() {
            if buf[511] != s {
                return Err(Error::Value("last voxel offset value does not match attribute array length".into()));
            }
        }
        Ok(())
    }
    /// Approximate memory usage in bytes.
    fn leaf_mem_usage(&self) -> usize {
        self.leaf().mem_usage() + self.attribute_set().mem_usage()
    }
}

/// Write access shared by mutable point data leaf types.
pub trait PointLeafMut: PointLeaf {
    /// The offsets and attributes, mutably.
    fn parts_mut(&mut self) -> (&mut LeafNode<u32>, &mut AttributeSet);

    /// The attribute set, mutably.
    fn attribute_set_mut(&mut self) -> &mut AttributeSet {
        self.parts_mut().1
    }
    /// The offset leaf, mutably.
    fn leaf_mut(&mut self) -> &mut LeafNode<u32> {
        self.parts_mut().0
    }
    /// Replace the attributes with arrays of length `n` for `descriptor`, which must contain
    /// only a `vec3s` position attribute (`initializeAttributes()`).
    fn initialize_attributes(&mut self, descriptor: Arc<Descriptor>, n: u32) -> Result<()> {
        if descriptor.size() != 1 || descriptor.position("P").is_none() || descriptor.value_type(0) != "vec3s" {
            return Err(Error::Index("initializing attributes only allowed with one Vec3f position attribute".into()));
        }
        *self.attribute_set_mut() = AttributeSet::from_descriptor(descriptor, n)?;
        Ok(())
    }
    /// Empty the attribute arrays and zero the offsets; deactivate all voxels if
    /// `update_value_mask` (`clearAttributes()`).
    fn clear_attributes(&mut self, update_value_mask: bool) -> Result<()> {
        let (leaf, set) = self.parts_mut();
        *set = AttributeSet::like(set, 0)?;
        leaf.buffer_mut().fill(0);
        if update_value_mask {
            leaf.value_mask_mut().set_all_off();
        }
        Ok(())
    }
    /// Set all 512 voxel offsets; update the active mask from the offsets if
    /// `update_value_mask` (`setOffsets()`).
    fn set_offsets(&mut self, offsets: &[u32], update_value_mask: bool) -> Result<()> {
        if offsets.len() != 512 {
            return Err(Error::Value("offset vector size doesn't match number of voxels".into()));
        }
        self.leaf_mut().buffer_mut().copy_from_slice(offsets);
        if update_value_mask {
            self.update_value_mask();
        }
        Ok(())
    }
    /// Activate exactly the voxels that contain points (`updateValueMask()`).
    fn update_value_mask(&mut self) {
        let leaf = self.leaf_mut();
        let (buf, mask) = leaf.buffer_and_mask_mut();
        let mut start = 0u32;
        for (n, &end) in buf.iter().enumerate() {
            mask.set(n, end > start);
            start = end;
        }
    }
    /// Set the offset of voxel `n` and activate it (`setOffsetOn()`).
    fn set_offset_on(&mut self, n: usize, value: u32) {
        self.leaf_mut().set_value_on_at(n, value);
    }
    /// Set the offset of voxel `n` without changing its state (`setOffsetOnly()`).
    fn set_offset_only(&mut self, n: usize, value: u32) {
        self.leaf_mut().set_value_only_at(n, value);
    }
    /// The array of attribute `name`, mutably.
    fn attribute_array_mut(&mut self, name: &str) -> Result<&mut AttributeArray> {
        self.attribute_set_mut()
            .get_by_name_mut(name)
            .ok_or_else(|| Error::Lookup(format!("attribute not found - {}", name)))
    }
    /// The array at position `pos`, mutably.
    fn attribute_array_at_mut(&mut self, pos: usize) -> Result<&mut AttributeArray> {
        self.attribute_set_mut().get_mut(pos).ok_or_else(|| Error::Lookup(format!("attribute out of range - {}", pos)))
    }
    /// A write handle on a group.
    fn group_write_handle(&mut self, index: GroupIndex) -> Result<GroupWriteHandle<'_>> {
        GroupWriteHandle::new(self.attribute_array_at_mut(index.0)?, index.1)
    }
    /// A write handle on group `name`.
    fn group_write_handle_by_name(&mut self, name: &str) -> Result<GroupWriteHandle<'_>> {
        let index = self.attribute_set().group_index(name)?;
        self.group_write_handle(index)
    }
    /// Collapse every array whose values are all equal (`compactAttributes()`).
    fn compact_attributes(&mut self) {
        let (_, mut arrays) = self.attribute_set_mut().arrays_mut();
        for a in arrays.iter_mut() {
            a.compact();
        }
    }
    /// Replace the attribute set; unless `allow_mismatch`, the descriptors must be equal.
    fn replace_attribute_set(&mut self, set: AttributeSet, allow_mismatch: bool) -> Result<()> {
        if !allow_mismatch && self.attribute_set().descriptor() != set.descriptor() {
            return Err(Error::Value("attribute set descriptors are not equal".into()));
        }
        *self.attribute_set_mut() = set;
        Ok(())
    }
}

/// A read-only view of a point data leaf.
#[derive(Clone, Copy)]
pub struct PointDataLeaf<'a> {
    /// Voxel offsets and active mask.
    pub leaf: &'a LeafNode<u32>,
    /// Attribute arrays.
    pub attributes: &'a AttributeSet,
}

impl PointLeaf for PointDataLeaf<'_> {
    fn leaf(&self) -> &LeafNode<u32> {
        self.leaf
    }
    fn attribute_set(&self) -> &AttributeSet {
        self.attributes
    }
}

/// A mutable view of a point data leaf.
pub struct PointDataLeafMut<'a> {
    /// Voxel offsets and active mask.
    pub leaf: &'a mut LeafNode<u32>,
    /// Attribute arrays.
    pub attributes: &'a mut AttributeSet,
}

impl PointLeaf for PointDataLeafMut<'_> {
    fn leaf(&self) -> &LeafNode<u32> {
        self.leaf
    }
    fn attribute_set(&self) -> &AttributeSet {
        self.attributes
    }
}

impl PointLeafMut for PointDataLeafMut<'_> {
    fn parts_mut(&mut self) -> (&mut LeafNode<u32>, &mut AttributeSet) {
        (self.leaf, self.attributes)
    }
}

/// An owned point data leaf (OpenVDB's `PointDataLeafNode`).
#[derive(Clone, PartialEq)]
pub struct PointDataLeafNode {
    /// Voxel offsets and active mask.
    pub leaf: LeafNode<u32>,
    /// Attribute arrays.
    pub attributes: AttributeSet,
}

impl fmt::Debug for PointDataLeafNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointDataLeafNode")
            .field("origin", &self.leaf.origin())
            .field("points", &self.point_count())
            .field("attributes", &self.attributes.descriptor().names())
            .finish()
    }
}

impl PointDataLeafNode {
    /// An empty leaf (all offsets zero, all voxels inactive) containing `xyz`.
    pub fn new(xyz: Coord) -> Self {
        PointDataLeafNode { leaf: LeafNode::new(xyz, 0, false), attributes: AttributeSet::new() }
    }
    /// A leaf from offsets and attributes.
    pub fn from_parts(leaf: LeafNode<u32>, attributes: AttributeSet) -> Self {
        PointDataLeafNode { leaf, attributes }
    }
    /// A view of this leaf.
    pub fn as_view(&self) -> PointDataLeaf<'_> {
        PointDataLeaf { leaf: &self.leaf, attributes: &self.attributes }
    }
}

impl PointLeaf for PointDataLeafNode {
    fn leaf(&self) -> &LeafNode<u32> {
        &self.leaf
    }
    fn attribute_set(&self) -> &AttributeSet {
        &self.attributes
    }
}

impl PointLeafMut for PointDataLeafNode {
    fn parts_mut(&mut self) -> (&mut LeafNode<u32>, &mut AttributeSet) {
        (&mut self.leaf, &mut self.attributes)
    }
}

/// A tree of point data leaves (OpenVDB's `PointDataTree`, type name
/// `"Tree_ptdataidx32_5_4_3"`).
#[derive(Clone, Default)]
pub struct PointDataTree {
    tree: Tree<u32>,
    attributes: HashMap<Coord, AttributeSet>,
}

impl PartialEq for PointDataTree {
    fn eq(&self, other: &Self) -> bool {
        self.tree == other.tree && self.leaves().zip(other.leaves()).all(|(a, b)| a.attributes == b.attributes)
    }
}

impl fmt::Debug for PointDataTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointDataTree")
            .field("leaves", &self.leaf_count())
            .field("points", &self.point_count())
            .finish()
    }
}

static EMPTY_SET: std::sync::OnceLock<AttributeSet> = std::sync::OnceLock::new();

fn empty_set() -> &'static AttributeSet {
    EMPTY_SET.get_or_init(AttributeSet::new)
}

impl PointDataTree {
    /// An empty tree.
    pub fn new() -> Self {
        Self::default()
    }
    /// A tree from an offset tree and per-leaf attribute sets keyed by leaf origin. Leaves
    /// without an attribute set get an empty one; sets without a leaf are dropped.
    pub fn from_parts(tree: Tree<u32>, mut attributes: HashMap<Coord, AttributeSet>) -> Self {
        let origins: Vec<Coord> = tree.leaves().map(|l| l.origin()).collect();
        let mut map = HashMap::with_capacity(origins.len());
        for o in origins {
            map.insert(o, attributes.remove(&o).unwrap_or_default());
        }
        PointDataTree { tree, attributes: map }
    }
    /// A tree with the topology and voxel offsets of `tree` (e.g. a point index tree) and
    /// empty attribute sets.
    pub fn from_topology(tree: Tree<u32>) -> Self {
        Self::from_parts(tree, HashMap::new())
    }
    /// Split into the offset tree and the attribute sets.
    pub fn into_parts(self) -> (Tree<u32>, HashMap<Coord, AttributeSet>) {
        (self.tree, self.attributes)
    }
    /// The offset tree (topology and voxel offsets).
    pub fn offsets(&self) -> &Tree<u32> {
        &self.tree
    }
    /// The offset tree, mutably. Adding or removing leaves through it leaves them without
    /// attributes (call [`sync`](Self::sync) afterwards); changing offsets requires keeping
    /// them consistent with the attribute arrays.
    pub fn offsets_mut(&mut self) -> &mut Tree<u32> {
        &mut self.tree
    }
    /// Re-associate attribute sets with the leaves of the offset tree (after the offset tree
    /// was modified directly).
    pub fn sync(&mut self) {
        let attrs = std::mem::take(&mut self.attributes);
        let tree = std::mem::take(&mut self.tree);
        *self = Self::from_parts(tree, attrs);
    }

    /// Number of leaves.
    pub fn leaf_count(&self) -> usize {
        self.tree.leaf_count()
    }
    /// True if the tree has no leaves or tiles.
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }
    /// Remove all leaves and tiles.
    pub fn clear(&mut self) {
        self.tree.clear();
        self.attributes.clear();
    }
    /// Iterate over the leaves in tree order.
    pub fn leaves(&self) -> impl Iterator<Item = PointDataLeaf<'_>> + '_ {
        self.tree.leaves().map(move |leaf| PointDataLeaf {
            leaf,
            attributes: self.attributes.get(&leaf.origin()).unwrap_or_else(|| empty_set()),
        })
    }
    /// All leaves in tree order, as a vector (for parallel iteration).
    pub fn leaf_vec(&self) -> Vec<PointDataLeaf<'_>> {
        self.leaves().collect()
    }
    /// All leaves in tree order, mutably.
    pub fn leaves_mut(&mut self) -> Vec<PointDataLeafMut<'_>> {
        let mut sets: HashMap<Coord, &mut AttributeSet> = self.attributes.iter_mut().map(|(k, v)| (*k, v)).collect();
        let leaves = self.tree.leaf_refs_mut();
        let mut out = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let attributes = sets.remove(&leaf.origin()).expect("point data leaf without attribute set");
            out.push(PointDataLeafMut { leaf, attributes });
        }
        out
    }
    /// Apply `f` to every leaf in parallel.
    pub fn par_for_each_leaf_mut<F>(&mut self, f: F)
    where
        F: Fn(&mut PointDataLeafMut<'_>) + Send + Sync,
    {
        self.leaves_mut().into_par_iter().for_each(|mut l| f(&mut l));
    }
    /// The leaf containing `ijk`.
    pub fn probe_leaf(&self, ijk: &Coord) -> Option<PointDataLeaf<'_>> {
        let leaf = self.tree.probe_leaf(ijk)?;
        Some(PointDataLeaf { leaf, attributes: self.attributes.get(&leaf.origin()).unwrap_or_else(|| empty_set()) })
    }
    /// The leaf containing `ijk`, mutably.
    pub fn probe_leaf_mut(&mut self, ijk: &Coord) -> Option<PointDataLeafMut<'_>> {
        let leaf = self.tree.probe_leaf_mut(ijk)?;
        let attributes = self.attributes.entry(leaf.origin()).or_default();
        Some(PointDataLeafMut { leaf, attributes })
    }
    /// Insert a leaf, replacing any leaf or tile at its position.
    pub fn add_leaf(&mut self, leaf: PointDataLeafNode) {
        let PointDataLeafNode { leaf, attributes } = leaf;
        self.attributes.insert(leaf.origin(), attributes);
        self.tree.add_leaf(Box::new(leaf));
    }
    /// The leaf containing `ijk`, created (empty) if needed.
    pub fn touch_leaf(&mut self, ijk: &Coord) -> PointDataLeafMut<'_> {
        let leaf = self.tree.touch_leaf(ijk);
        let attributes = self.attributes.entry(leaf.origin()).or_default();
        PointDataLeafMut { leaf, attributes }
    }
    /// Keep only the leaves for which `keep` returns true (tiles are preserved).
    pub fn retain_leaves(&mut self, mut keep: impl FnMut(&PointDataLeaf<'_>) -> bool) {
        let remove: Vec<Coord> = self.leaves().filter(|l| !keep(l)).map(|l| l.origin()).collect();
        if remove.is_empty() {
            return;
        }
        let mut tree = Tree::new(*self.tree.background());
        self.tree.visit_tiles(&mut |bbox, v, on, level| tree.add_tile(level, &bbox.min, *v, on));
        let old = std::mem::replace(&mut self.tree, Tree::new(0));
        let remove: std::collections::HashSet<Coord> = remove.into_iter().collect();
        for leaf in old.leaves() {
            if !remove.contains(&leaf.origin()) {
                tree.add_leaf(Box::new(leaf.clone()));
            }
        }
        for o in &remove {
            self.attributes.remove(o);
        }
        self.tree = tree;
    }
    /// Remove leaves without points.
    pub fn prune_empty_leaves(&mut self) {
        self.retain_leaves(|l| l.point_count() > 0);
    }

    /// Total number of points.
    pub fn point_count(&self) -> u64 {
        self.leaves().map(|l| l.point_count()).sum()
    }
    /// The background value (always 0).
    pub fn background(&self) -> &u32 {
        self.tree.background()
    }
    /// Number of active voxels.
    pub fn active_voxel_count(&self) -> u64 {
        self.tree.active_voxel_count()
    }
    /// Number of inactive voxels.
    pub fn inactive_voxel_count(&self) -> u64 {
        self.tree.inactive_voxel_count()
    }
    /// Number of active tiles.
    pub fn active_tile_count(&self) -> u64 {
        self.tree.active_tile_count()
    }
    /// Number of nodes per level (leaf to root).
    pub fn node_count(&self) -> Vec<usize> {
        self.tree.node_count()
    }
    /// Bounding box of the active voxels.
    pub fn eval_active_voxel_bbox(&self) -> CoordBBox {
        self.tree.eval_active_voxel_bbox()
    }
    /// Bounding box of the leaves.
    pub fn eval_leaf_bbox(&self) -> CoordBBox {
        self.tree.eval_leaf_bbox()
    }
    /// Approximate memory usage in bytes.
    pub fn mem_usage(&self) -> usize {
        self.tree.mem_usage() + self.attributes.values().map(|a| a.mem_usage()).sum::<usize>()
    }
    /// The descriptor of the first leaf, if any.
    pub fn descriptor(&self) -> Option<&Descriptor> {
        self.leaves().next().map(|l| l.attributes.descriptor())
    }

    /// Give every leaf a copy of the first leaf's descriptor, shared by all leaves
    /// (`makeDescriptorUnique()`); returns it.
    pub fn make_descriptor_unique(&mut self) -> Option<Arc<Descriptor>> {
        let d = Arc::new(self.descriptor()?.clone());
        for set in self.attributes.values_mut() {
            let _ = set.reset_descriptor(d.clone(), true);
        }
        Some(d)
    }
    /// Toggle streaming mode on all arrays (`setStreamingMode()`).
    pub fn set_streaming_mode(&mut self, on: bool) {
        for set in self.attributes.values_mut() {
            let (_, mut arrays) = set.arrays_mut();
            for a in arrays.iter_mut() {
                a.set_streaming(on);
            }
        }
    }
}

/// A grid of points (OpenVDB's `PointDataGrid`): a [`PointDataTree`], a transform and
/// metadata. Positions are stored per point relative to the voxel center, in index space
/// (attribute `"P"`, typically quantized).
#[derive(Clone, Default)]
pub struct PointDataGrid {
    tree: PointDataTree,
    transform: Transform,
    meta: MetaMap,
}

impl fmt::Debug for PointDataGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointDataGrid")
            .field("name", &self.name())
            .field("points", &self.tree.point_count())
            .field("transform", &self.transform)
            .finish()
    }
}

impl PointDataGrid {
    /// OpenVDB grid type name.
    pub const GRID_TYPE: &'static str = "Tree_ptdataidx32_5_4_3";

    /// An empty grid with an identity transform.
    pub fn new() -> Self {
        Self::default()
    }
    /// A grid from a tree and transform.
    pub fn from_tree(tree: PointDataTree, transform: Transform) -> Self {
        PointDataGrid { tree, transform, meta: MetaMap::new() }
    }
    /// OpenVDB grid type name, `"Tree_ptdataidx32_5_4_3"`.
    pub fn grid_type() -> String {
        Self::GRID_TYPE.to_string()
    }
    /// The value type name, `"ptdataidx32"`.
    pub fn value_type() -> &'static str {
        "ptdataidx32"
    }
    /// The tree.
    pub fn tree(&self) -> &PointDataTree {
        &self.tree
    }
    /// The tree, mutably.
    pub fn tree_mut(&mut self) -> &mut PointDataTree {
        &mut self.tree
    }
    /// Replace the tree.
    pub fn set_tree(&mut self, tree: PointDataTree) {
        self.tree = tree;
    }
    /// Take the tree.
    pub fn into_tree(self) -> PointDataTree {
        self.tree
    }
    /// The transform.
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
    /// The transform, mutably.
    pub fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
    /// Replace the transform.
    pub fn set_transform(&mut self, t: Transform) {
        self.transform = t;
    }
    /// The metadata.
    pub fn metadata(&self) -> &MetaMap {
        &self.meta
    }
    /// The metadata, mutably.
    pub fn metadata_mut(&mut self) -> &mut MetaMap {
        &mut self.meta
    }
    /// Insert a metadata entry.
    pub fn insert_meta(&mut self, name: &str, value: impl Into<MetaValue>) {
        self.meta.insert(name, value);
    }
    /// The grid name.
    pub fn name(&self) -> &str {
        self.meta.get_str(meta_names::GRID_NAME).unwrap_or("")
    }
    /// Set the grid name.
    pub fn set_name(&mut self, name: &str) {
        self.meta.insert(meta_names::GRID_NAME, name);
    }
    /// Builder-style [`set_name`](Self::set_name).
    pub fn with_name(mut self, name: &str) -> Self {
        self.set_name(name);
        self
    }
    /// The grid creator.
    pub fn creator(&self) -> &str {
        self.meta.get_str(meta_names::GRID_CREATOR).unwrap_or("")
    }
    /// Set the grid creator.
    pub fn set_creator(&mut self, creator: &str) {
        self.meta.insert(meta_names::GRID_CREATOR, creator);
    }
    /// The grid class.
    pub fn grid_class(&self) -> GridClass {
        self.meta.get_str(meta_names::GRID_CLASS).map(GridClass::from_str).unwrap_or_default()
    }
    /// Set the grid class.
    pub fn set_grid_class(&mut self, c: GridClass) {
        self.meta.insert(meta_names::GRID_CLASS, c.as_str());
    }
    /// The vector type.
    pub fn vector_type(&self) -> VecType {
        self.meta.get_str(meta_names::VECTOR_TYPE).map(VecType::from_str).unwrap_or_default()
    }
    /// True if floating-point voxel values are saved as half floats (irrelevant for points).
    pub fn save_float_as_half(&self) -> bool {
        self.meta.get(meta_names::SAVE_HALF_FLOAT).and_then(|m| m.as_bool()).unwrap_or(false)
    }
    /// Set the half-float flag.
    pub fn set_save_float_as_half(&mut self, on: bool) {
        self.meta.insert(meta_names::SAVE_HALF_FLOAT, on);
    }
    /// The background value (0).
    pub fn background(&self) -> &u32 {
        self.tree.background()
    }
    /// The voxel size.
    pub fn voxel_size(&self) -> Vec3d {
        self.transform.voxel_size()
    }
    /// Number of active voxels.
    pub fn active_voxel_count(&self) -> u64 {
        self.tree.active_voxel_count()
    }
    /// Bounding box of the active voxels.
    pub fn eval_active_voxel_bbox(&self) -> CoordBBox {
        self.tree.eval_active_voxel_bbox()
    }
    /// Approximate memory usage in bytes.
    pub fn mem_usage(&self) -> usize {
        self.tree.mem_usage()
    }
    /// True if the grid has no points or voxels.
    pub fn empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// World-space positions of all points, in leaf order (a convenience for
    /// [`convert_point_data_grid_position`](super::convert_point_data_grid_position)).
    pub fn world_positions(&self) -> Result<Vec<Vec3d>> {
        let mut out = Vec::with_capacity(self.tree.point_count() as usize);
        for leaf in self.tree.leaves() {
            let Ok(h) = leaf.attribute_handle::<Vec3f>("P") else { continue };
            for (ijk, i) in leaf.index_all().with_coords() {
                let p = h.get(i);
                out.push(
                    self.transform.index_to_world(ijk.as_vec3d() + Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64)),
                );
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::{AttributeType, Codec};

    fn leaf_with_points() -> PointDataLeafNode {
        let mut node = PointDataLeafNode::new(Coord::new(0, 0, 8));
        let d = Arc::new(Descriptor::with_position(AttributeType::of::<Vec3f>(Codec::Null)).unwrap());
        node.initialize_attributes(d, 3).unwrap();
        let mut offsets = vec![0u32; 512];
        for (i, o) in offsets.iter_mut().enumerate() {
            *o = if i >= 10 {
                3
            } else if i >= 2 {
                1
            } else {
                0
            };
        }
        node.set_offsets(&offsets, true).unwrap();
        node
    }

    #[test]
    fn leaf_counts() {
        let mut node = leaf_with_points();
        assert_eq!(node.point_count(), 3);
        assert_eq!(node.on_point_count(), 3);
        assert!(node.leaf.is_value_on_at(2) && node.leaf.is_value_on_at(10) && !node.leaf.is_value_on_at(3));
        node.leaf.set_active_state_at(10, false);
        assert_eq!(node.on_point_count(), 1);
        assert_eq!(node.off_point_count(), 2);
        assert_eq!(node.index_off().collect::<Vec<_>>(), vec![1, 2]);
        node.validate_offsets().unwrap();
        node.clear_attributes(true).unwrap();
        assert_eq!(node.point_count(), 0);
        assert!(node.leaf.is_empty());
        assert!(node.initialize_attributes(Arc::new(Descriptor::new()), 1).is_err());
    }

    #[test]
    fn tree_and_grid() {
        let mut tree = PointDataTree::new();
        tree.add_leaf(leaf_with_points());
        let mut other = leaf_with_points();
        other.leaf.set_origin(Coord::new(800, 0, 0));
        tree.add_leaf(other);
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.point_count(), 6);
        for mut l in tree.leaves_mut() {
            l.attribute_array_mut("P").unwrap().set(0, Vec3f::new(0.25, 0.0, 0.0));
        }
        let leaf = tree.probe_leaf(&Coord::new(801, 1, 1)).unwrap();
        assert_eq!(leaf.attribute_array("P").unwrap().get::<Vec3f>(0), Vec3f::new(0.25, 0.0, 0.0));
        let mut grid = PointDataGrid::from_tree(tree.clone(), Transform::linear(0.5)).with_name("pts");
        assert_eq!(grid.name(), "pts");
        let ws = grid.world_positions().unwrap();
        assert_eq!(ws.len(), 6);
        assert_eq!(ws[0], Vec3d::new(0.125, 0.0, 5.0));
        grid.tree_mut().retain_leaves(|l| l.origin() != Coord::new(800, 0, 0));
        assert_eq!(grid.tree().leaf_count(), 1);
        assert!(grid.tree() != &tree);
        let d = tree.make_descriptor_unique().unwrap();
        assert!(tree.leaves().all(|l| Arc::ptr_eq(l.attributes.descriptor_ptr(), &d)));
    }
}
