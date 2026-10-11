//! Leaf nodes: dense 8x8x8 blocks of voxels (port of `openvdb/tree/LeafNode.h`).

use super::node::Node;
use crate::math::{Coord, CoordBBox};
use crate::types::Value;
use crate::util::NodeMask;

/// A dense 8³ block of voxel values with a per-voxel active mask.
#[derive(Clone, PartialEq)]
pub struct LeafNode<T: Value> {
    origin: Coord,
    value_mask: NodeMask<3>,
    buffer: Box<[T]>,
}

impl<T: Value> LeafNode<T> {
    pub const LOG2DIM: u32 = 3;
    pub const DIM: i32 = 8;
    pub const SIZE: usize = 512;

    /// Linear offset of the voxel at `xyz` within its leaf.
    #[inline]
    pub fn coord_to_offset(xyz: &Coord) -> usize {
        ((((xyz[0] & 7) as usize) << 6) | (((xyz[1] & 7) as usize) << 3) | ((xyz[2] & 7) as usize)) as usize
    }
    /// Local (0..8) coordinates for a linear offset.
    #[inline]
    pub fn offset_to_local_coord(n: usize) -> Coord {
        Coord::new((n >> 6) as i32, ((n >> 3) & 7) as i32, (n & 7) as i32)
    }
    /// Global coordinates for a linear offset.
    #[inline]
    pub fn offset_to_global_coord(&self, n: usize) -> Coord {
        self.origin + Self::offset_to_local_coord(n)
    }

    /// Build a leaf from its parts (the buffer must have 512 entries).
    pub fn from_parts(origin: Coord, value_mask: NodeMask<3>, buffer: Vec<T>) -> Self {
        assert_eq!(buffer.len(), Self::SIZE);
        LeafNode { origin: origin.align_down(8), value_mask, buffer: buffer.into_boxed_slice() }
    }

    pub fn set_origin(&mut self, origin: Coord) {
        self.origin = origin.align_down(8);
    }
    pub fn value_mask(&self) -> &NodeMask<3> {
        &self.value_mask
    }
    pub fn value_mask_mut(&mut self) -> &mut NodeMask<3> {
        &mut self.value_mask
    }
    /// The 512 voxel values, in offset order.
    pub fn buffer(&self) -> &[T] {
        &self.buffer
    }
    pub fn buffer_mut(&mut self) -> &mut [T] {
        &mut self.buffer
    }
    /// Mutable access to both the buffer and the mask.
    pub fn buffer_and_mask_mut(&mut self) -> (&mut [T], &mut NodeMask<3>) {
        (&mut self.buffer, &mut self.value_mask)
    }

    #[inline]
    pub fn get_value_at(&self, n: usize) -> &T {
        &self.buffer[n]
    }
    #[inline]
    pub fn is_value_on_at(&self, n: usize) -> bool {
        self.value_mask.is_on(n)
    }
    #[inline]
    pub fn set_value_on_at(&mut self, n: usize, v: T) {
        self.buffer[n] = v;
        self.value_mask.set_on(n);
    }
    #[inline]
    pub fn set_value_off_at(&mut self, n: usize, v: T) {
        self.buffer[n] = v;
        self.value_mask.set_off(n);
    }
    #[inline]
    pub fn set_value_only_at(&mut self, n: usize, v: T) {
        self.buffer[n] = v;
    }
    #[inline]
    pub fn set_active_state_at(&mut self, n: usize, on: bool) {
        self.value_mask.set(n, on);
    }
    /// Mark the voxel at `n` active without changing its value.
    #[inline]
    pub fn set_value_mask_on(&mut self, n: usize) {
        self.value_mask.set_on(n);
    }

    /// Number of active voxels.
    pub fn on_voxel_count(&self) -> u64 {
        self.value_mask.count_on() as u64
    }
    pub fn off_voxel_count(&self) -> u64 {
        self.value_mask.count_off() as u64
    }
    pub fn is_empty(&self) -> bool {
        self.value_mask.is_all_off()
    }
    pub fn is_dense(&self) -> bool {
        self.value_mask.is_all_on()
    }

    /// Iterate over `(offset, &value)` for active voxels.
    pub fn iter_on(&self) -> impl Iterator<Item = (usize, &T)> + '_ {
        self.value_mask.iter_on().map(move |n| (n, &self.buffer[n]))
    }
    /// Iterate over `(global coord, value)` for active voxels.
    pub fn iter_on_coords(&self) -> impl Iterator<Item = (Coord, T)> + '_ {
        self.value_mask.iter_on().map(move |n| (self.offset_to_global_coord(n), self.buffer[n]))
    }
    /// Iterate over `(offset, &value)` for inactive voxels.
    pub fn iter_off(&self) -> impl Iterator<Item = (usize, &T)> + '_ {
        self.value_mask.iter_off().map(move |n| (n, &self.buffer[n]))
    }
    /// Iterate over all `(offset, &value, active)` triples.
    pub fn iter_all(&self) -> impl Iterator<Item = (usize, &T, bool)> + '_ {
        self.buffer.iter().enumerate().map(move |(n, v)| (n, v, self.value_mask.is_on(n)))
    }
    /// Apply `f(offset, &mut value)` to every active voxel.
    pub fn for_each_on_mut(&mut self, mut f: impl FnMut(usize, &mut T)) {
        let LeafNode { value_mask, buffer, .. } = self;
        for n in value_mask.iter_on() {
            f(n, &mut buffer[n]);
        }
    }

    /// Fill every voxel with `value` and state `active`.
    pub fn fill_all(&mut self, value: T, active: bool) {
        for v in self.buffer.iter_mut() {
            *v = value;
        }
        self.value_mask.set_all(active);
    }
    /// Union the active mask of `other` (of any value type) into this leaf.
    pub fn topology_union<U: Value>(&mut self, other: &LeafNode<U>) {
        self.value_mask.union_with(&other.value_mask);
    }
    /// Intersect this leaf's active mask with that of `other`.
    pub fn topology_intersection<U: Value>(&mut self, other: &LeafNode<U>) {
        self.value_mask.intersect_with(&other.value_mask);
    }
    /// Remove from this leaf's active mask the voxels active in `other`.
    pub fn topology_difference<U: Value>(&mut self, other: &LeafNode<U>) {
        self.value_mask.difference_with(&other.value_mask);
    }
    /// True if both leaves have the same active mask.
    pub fn has_same_topology<U: Value>(&self, other: &LeafNode<U>) -> bool {
        self.value_mask == other.value_mask
    }
    /// Index-space bounds of the active voxels.
    pub fn active_bbox(&self) -> CoordBBox {
        let mut b = CoordBBox::empty();
        self.eval_active_bbox(&mut b, true);
        b
    }
}

impl<T: Value> Node for LeafNode<T> {
    type Value = T;
    const LEVEL: u32 = 0;
    const LOG2DIM: u32 = 3;
    const TOTAL: u32 = 3;

    fn new(xyz: Coord, value: T, active: bool) -> Self {
        LeafNode {
            origin: xyz.align_down(8),
            value_mask: NodeMask::new_with(active),
            buffer: vec![value; 512].into_boxed_slice(),
        }
    }
    #[inline]
    fn origin(&self) -> Coord {
        self.origin
    }
    #[inline]
    fn get_value(&self, xyz: &Coord) -> &T {
        &self.buffer[Self::coord_to_offset(xyz)]
    }
    #[inline]
    fn is_value_on(&self, xyz: &Coord) -> bool {
        self.value_mask.is_on(Self::coord_to_offset(xyz))
    }
    #[inline]
    fn get_value_level(&self, _xyz: &Coord) -> u32 {
        0
    }
    #[inline]
    fn set_value_on(&mut self, xyz: &Coord, value: T) {
        self.set_value_on_at(Self::coord_to_offset(xyz), value)
    }
    #[inline]
    fn set_value_off(&mut self, xyz: &Coord, value: T) {
        self.set_value_off_at(Self::coord_to_offset(xyz), value)
    }
    #[inline]
    fn set_value_only(&mut self, xyz: &Coord, value: T) {
        self.buffer[Self::coord_to_offset(xyz)] = value;
    }
    #[inline]
    fn set_active_state(&mut self, xyz: &Coord, on: bool) {
        self.value_mask.set(Self::coord_to_offset(xyz), on)
    }
    #[inline]
    fn modify_value<F: FnOnce(&mut T)>(&mut self, xyz: &Coord, f: F) {
        let n = Self::coord_to_offset(xyz);
        f(&mut self.buffer[n]);
        self.value_mask.set_on(n);
    }
    #[inline]
    fn modify_value_and_active_state<F: FnOnce(&mut T, &mut bool)>(&mut self, xyz: &Coord, f: F) {
        let n = Self::coord_to_offset(xyz);
        let mut state = self.value_mask.is_on(n);
        f(&mut self.buffer[n], &mut state);
        self.value_mask.set(n, state);
    }
    fn fill(&mut self, bbox: &CoordBBox, value: T, active: bool) {
        let clipped = bbox.intersection(&self.node_bbox());
        if clipped.is_empty() {
            return;
        }
        for x in clipped.min[0]..=clipped.max[0] {
            for y in clipped.min[1]..=clipped.max[1] {
                for z in clipped.min[2]..=clipped.max[2] {
                    let n = Self::coord_to_offset(&Coord::new(x, y, z));
                    self.buffer[n] = value;
                    self.value_mask.set(n, active);
                }
            }
        }
    }
    fn activate_region(&mut self, bbox: &CoordBBox) {
        let clipped = bbox.intersection(&self.node_bbox());
        if clipped.is_empty() {
            return;
        }
        for x in clipped.min[0]..=clipped.max[0] {
            for y in clipped.min[1]..=clipped.max[1] {
                for z in clipped.min[2]..=clipped.max[2] {
                    self.value_mask.set_on(Self::coord_to_offset(&Coord::new(x, y, z)));
                }
            }
        }
    }
    fn touch_leaf(&mut self, _xyz: &Coord) -> &mut LeafNode<T> {
        self
    }
    fn probe_leaf(&self, _xyz: &Coord) -> Option<&LeafNode<T>> {
        Some(self)
    }
    fn probe_leaf_mut(&mut self, _xyz: &Coord) -> Option<&mut LeafNode<T>> {
        Some(self)
    }
    fn add_leaf(&mut self, leaf: Box<LeafNode<T>>) {
        *self = *leaf;
    }
    fn add_tile(&mut self, _level: u32, xyz: &Coord, value: T, active: bool) {
        let n = Self::coord_to_offset(xyz);
        self.buffer[n] = value;
        self.value_mask.set(n, active);
    }
    fn is_constant(&self, tolerance: &T) -> Option<(T, bool)> {
        let state = self.value_mask.is_constant()?;
        let first = self.buffer[0];
        if self.buffer.iter().all(|v| v.approx_eq(&first, tolerance)) {
            Some((first, state))
        } else {
            None
        }
    }
    fn prune(&mut self, _tolerance: &T) {}
    fn prune_inactive(&mut self, _background: &T) {}
    fn eval_active_bbox(&self, bbox: &mut CoordBBox, visit_voxels: bool) {
        if self.value_mask.is_all_off() {
            return;
        }
        let this_bbox = self.node_bbox();
        if bbox.is_inside_bbox(&this_bbox) {
            return;
        }
        if visit_voxels && !self.value_mask.is_all_on() {
            for n in self.value_mask.iter_on() {
                bbox.expand_coord(self.offset_to_global_coord(n));
            }
        } else {
            bbox.expand_bbox(&this_bbox);
        }
    }
    fn active_voxel_count(&self) -> u64 {
        self.on_voxel_count()
    }
    fn inactive_voxel_count(&self) -> u64 {
        self.off_voxel_count()
    }
    fn active_leaf_voxel_count(&self) -> u64 {
        self.on_voxel_count()
    }
    fn inactive_leaf_voxel_count(&self) -> u64 {
        self.off_voxel_count()
    }
    fn active_tile_count(&self) -> u64 {
        0
    }
    fn leaf_count(&self) -> usize {
        1
    }
    fn node_count(&self, counts: &mut [usize]) {
        counts[0] += 1;
    }
    fn is_inactive(&self) -> bool {
        self.value_mask.is_all_off()
    }
    fn leaves(&self) -> Box<dyn Iterator<Item = &LeafNode<T>> + '_> {
        Box::new(std::iter::once(self))
    }
    fn leaves_mut(&mut self) -> Box<dyn Iterator<Item = &mut LeafNode<T>> + '_> {
        Box::new(std::iter::once(self))
    }
    fn visit_tiles(&self, _f: &mut dyn FnMut(CoordBBox, &T, bool, u32)) {}
    fn visit_tiles_mut(&mut self, _f: &mut dyn FnMut(CoordBBox, &mut T, &mut bool, u32)) {}
    fn clip(&mut self, bbox: &CoordBBox, background: &T) {
        let node_bbox = self.node_bbox();
        if bbox.is_inside_bbox(&node_bbox) {
            return;
        }
        if !bbox.has_overlap(&node_bbox) {
            self.fill_all(*background, false);
            return;
        }
        for n in 0..Self::SIZE {
            let xyz = self.offset_to_global_coord(n);
            if !bbox.is_inside(&xyz) {
                self.buffer[n] = *background;
                self.value_mask.set_off(n);
            }
        }
    }
    fn voxelize_active_tiles(&mut self) {}
    fn set_values_on_off(&mut self, on: bool) {
        self.value_mask.set_all(on);
    }
    fn reset_background(&mut self, old: &T, new: &T) {
        let LeafNode { value_mask, buffer, .. } = self;
        for n in value_mask.iter_off() {
            if buffer[n] == *old {
                buffer[n] = *new;
            }
        }
    }
    fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.buffer.len() * std::mem::size_of::<T>() + self.value_mask.mem_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn leaf_basics() {
        let mut leaf = LeafNode::<f32>::new(Coord::new(9, -3, 17), 0.0, false);
        assert_eq!(leaf.origin(), Coord::new(8, -8, 16));
        let c = Coord::new(10, -1, 23);
        leaf.set_value_on(&c, 3.5);
        assert_eq!(*leaf.get_value(&c), 3.5);
        assert!(leaf.is_value_on(&c));
        assert_eq!(leaf.on_voxel_count(), 1);
        let n = LeafNode::<f32>::coord_to_offset(&c);
        assert_eq!(leaf.offset_to_global_coord(n), c);
        assert_eq!(leaf.active_bbox(), CoordBBox::new(c, c));
        assert!(leaf.is_constant(&0.0).is_none());
        leaf.fill_all(1.0, true);
        assert_eq!(leaf.is_constant(&0.0), Some((1.0, true)));
    }
}
