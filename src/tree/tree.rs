//! The tree container (port of `openvdb/tree/Tree.h`).

use super::accessor::{ValueAccessor, ValueAccessorMut};
use super::internal::InternalNode;
use super::leaf::LeafNode;
use super::node::Node;
use super::root::RootNode;
use crate::math::{Coord, CoordBBox};
use crate::types::Value;
use rayon::prelude::*;

/// A sparse tree of values whose root's children are of type `C`.
#[derive(Clone)]
pub struct TreeBase<C: Node> {
    root: RootNode<C>,
}

/// The standard OpenVDB tree configuration: Root → Internal(32³) → Internal(16³) → Leaf(8³).
pub type Tree<T> = TreeBase<InternalNode<InternalNode<LeafNode<T>, 4>, 5>>;
/// The level-1 internal node type of the standard tree.
pub type Internal1<T> = InternalNode<LeafNode<T>, 4>;
/// The level-2 internal node type of the standard tree.
pub type Internal2<T> = InternalNode<Internal1<T>, 5>;

pub type FloatTree = Tree<f32>;
pub type DoubleTree = Tree<f64>;
pub type Int32Tree = Tree<i32>;
pub type Int64Tree = Tree<i64>;
pub type BoolTree = Tree<bool>;
pub type Vec3STree = Tree<crate::math::Vec3f>;
pub type Vec3DTree = Tree<crate::math::Vec3d>;
pub type Vec3ITree = Tree<crate::math::Vec3i>;

impl<C: Node> Default for TreeBase<C> {
    fn default() -> Self {
        Self::new(C::Value::zero())
    }
}

impl<C: Node> TreeBase<C> {
    /// Depth of the tree, including the root (4 for the standard configuration).
    pub const DEPTH: u32 = C::LEVEL + 2;

    /// An empty tree with the given background value.
    pub fn new(background: C::Value) -> Self {
        TreeBase { root: RootNode::new(background) }
    }
    pub fn from_root(root: RootNode<C>) -> Self {
        TreeBase { root }
    }
    pub fn root(&self) -> &RootNode<C> {
        &self.root
    }
    pub fn root_mut(&mut self) -> &mut RootNode<C> {
        &mut self.root
    }
    #[inline]
    pub fn background(&self) -> &C::Value {
        self.root.background()
    }
    /// Change the background, replacing inactive values equal to the old background.
    pub fn set_background(&mut self, background: C::Value) {
        self.root.set_background(background, true);
    }
    /// Remove all nodes, leaving an empty tree with the same background.
    pub fn clear(&mut self) {
        self.root.clear();
    }
    /// True if the tree contains only background tiles.
    pub fn is_empty(&self) -> bool {
        self.root.is_background_only()
    }

    // ---- Value access ----

    #[inline]
    pub fn get_value(&self, xyz: &Coord) -> C::Value {
        *self.root.get_value(xyz)
    }
    #[inline]
    pub fn is_value_on(&self, xyz: &Coord) -> bool {
        self.root.is_value_on(xyz)
    }
    #[inline]
    pub fn probe_value(&self, xyz: &Coord) -> (C::Value, bool) {
        self.root.probe_value(xyz)
    }
    /// Depth at which the value is stored: 0 for root tiles, `DEPTH - 1` for voxels,
    /// -1 for background.
    pub fn get_value_depth(&self, xyz: &Coord) -> i32 {
        self.root.get_value_depth(xyz)
    }
    pub fn set_value(&mut self, xyz: &Coord, value: C::Value) {
        self.root.set_value_on(xyz, value)
    }
    pub fn set_value_on(&mut self, xyz: &Coord, value: C::Value) {
        self.root.set_value_on(xyz, value)
    }
    pub fn set_value_off(&mut self, xyz: &Coord, value: C::Value) {
        self.root.set_value_off(xyz, value)
    }
    pub fn set_value_only(&mut self, xyz: &Coord, value: C::Value) {
        self.root.set_value_only(xyz, value)
    }
    pub fn set_active_state(&mut self, xyz: &Coord, on: bool) {
        self.root.set_active_state(xyz, on)
    }
    pub fn modify_value<F: FnOnce(&mut C::Value)>(&mut self, xyz: &Coord, f: F) {
        self.root.modify_value(xyz, f)
    }
    pub fn modify_value_and_active_state<F: FnOnce(&mut C::Value, &mut bool)>(&mut self, xyz: &Coord, f: F) {
        self.root.modify_value_and_active_state(xyz, f)
    }
    /// Fill a box with a constant value, using tiles where possible (sparse fill).
    pub fn fill(&mut self, bbox: &CoordBBox, value: C::Value, active: bool) {
        self.root.fill(bbox, value, active)
    }
    /// Alias for [`fill`](Self::fill).
    pub fn sparse_fill(&mut self, bbox: &CoordBBox, value: C::Value, active: bool) {
        self.fill(bbox, value, active)
    }
    /// Fill a box with a constant value at voxel resolution (no tiles).
    pub fn dense_fill(&mut self, bbox: &CoordBBox, value: C::Value, active: bool) {
        self.fill(bbox, value, active);
        self.voxelize_active_tiles_in(bbox);
    }
    fn voxelize_active_tiles_in(&mut self, bbox: &CoordBBox) {
        let mut tiles = Vec::new();
        self.root.visit_tiles(&mut |b, _, on, _| {
            if on && b.has_overlap(bbox) {
                tiles.push(b);
            }
        });
        for b in tiles {
            let b = b.intersection(bbox);
            let mut x = b.min[0] & !7;
            while x <= b.max[0] {
                let mut y = b.min[1] & !7;
                while y <= b.max[1] {
                    let mut z = b.min[2] & !7;
                    while z <= b.max[2] {
                        self.touch_leaf(&Coord::new(x, y, z));
                        z += 8;
                    }
                    y += 8;
                }
                x += 8;
            }
        }
    }

    // ---- Node access ----

    pub fn touch_leaf(&mut self, xyz: &Coord) -> &mut LeafNode<C::Value> {
        self.root.touch_leaf(xyz)
    }
    pub fn probe_leaf(&self, xyz: &Coord) -> Option<&LeafNode<C::Value>> {
        self.root.probe_leaf(xyz)
    }
    pub fn probe_leaf_mut(&mut self, xyz: &Coord) -> Option<&mut LeafNode<C::Value>> {
        self.root.probe_leaf_mut(xyz)
    }
    pub fn add_leaf(&mut self, leaf: Box<LeafNode<C::Value>>) {
        self.root.add_leaf(leaf)
    }
    /// Add a tile at the given level (0 = voxel, `DEPTH - 1` = root tile).
    pub fn add_tile(&mut self, level: u32, xyz: &Coord, value: C::Value, active: bool) {
        self.root.add_tile(level, xyz, value, active)
    }

    // ---- Accessors ----

    /// A read-only accessor that caches recently visited nodes.
    pub fn accessor(&self) -> ValueAccessor<'_, C> {
        ValueAccessor::new(self)
    }
    /// A read/write accessor that caches recently visited nodes.
    pub fn accessor_mut(&mut self) -> ValueAccessorMut<'_, C> {
        ValueAccessorMut::new(self)
    }

    // ---- Iteration ----

    /// Iterate over all leaf nodes.
    pub fn leaves(&self) -> impl Iterator<Item = &LeafNode<C::Value>> + '_ {
        self.root.leaves()
    }
    pub fn leaves_mut(&mut self) -> impl Iterator<Item = &mut LeafNode<C::Value>> + '_ {
        self.root.leaves_mut()
    }
    /// Collect references to all leaves (useful for parallel processing).
    pub fn leaf_refs(&self) -> Vec<&LeafNode<C::Value>> {
        self.leaves().collect()
    }
    pub fn leaf_refs_mut(&mut self) -> Vec<&mut LeafNode<C::Value>> {
        self.leaves_mut().collect()
    }
    /// Apply `f` to every leaf in parallel.
    pub fn par_for_each_leaf_mut<F>(&mut self, f: F)
    where
        F: Fn(&mut LeafNode<C::Value>) + Send + Sync,
    {
        self.leaf_refs_mut().into_par_iter().for_each(|l| f(l));
    }
    pub fn par_for_each_leaf<F>(&self, f: F)
    where
        F: Fn(&LeafNode<C::Value>) + Send + Sync,
    {
        self.leaf_refs().into_par_iter().for_each(|l| f(l));
    }
    /// Iterate over `(coord, value)` for every active *voxel* (tiles are not included;
    /// see [`visit_tiles`](Self::visit_tiles) and [`iter_values_on`](Self::iter_values_on)).
    pub fn active_voxels(&self) -> impl Iterator<Item = (Coord, C::Value)> + '_ {
        self.leaves().flat_map(|l| l.iter_on_coords())
    }
    /// Iterate over every active value, voxels and tiles alike. Each item carries the
    /// bounding box the value covers (a single voxel for leaf values).
    pub fn iter_values_on(&self) -> impl Iterator<Item = ValueItem<C::Value>> + '_ {
        let mut tiles = Vec::new();
        self.root.visit_tiles(&mut |bbox, v, on, level| {
            if on {
                tiles.push(ValueItem { bbox, value: *v, level });
            }
        });
        tiles.into_iter().chain(self.active_voxels().map(|(c, v)| ValueItem {
            bbox: CoordBBox::new(c, c),
            value: v,
            level: 0,
        }))
    }
    /// Visit every tile (root and internal): `f(bbox, value, active, level)`.
    pub fn visit_tiles(&self, f: &mut dyn FnMut(CoordBBox, &C::Value, bool, u32)) {
        self.root.visit_tiles(f)
    }
    pub fn visit_tiles_mut(&mut self, f: &mut dyn FnMut(CoordBBox, &mut C::Value, &mut bool, u32)) {
        self.root.visit_tiles_mut(f)
    }
    /// Apply `f` to every active value (voxels and tiles).
    pub fn for_each_value_on_mut(&mut self, f: impl Fn(&mut C::Value) + Send + Sync) {
        self.root.visit_tiles_mut(&mut |_, v, on, _| {
            if *on {
                f(v)
            }
        });
        self.par_for_each_leaf_mut(|leaf| leaf.for_each_on_mut(|_, v| f(v)));
    }

    // ---- Statistics ----

    pub fn active_voxel_count(&self) -> u64 {
        self.root.active_voxel_count()
    }
    pub fn inactive_voxel_count(&self) -> u64 {
        self.root.inactive_voxel_count()
    }
    pub fn active_leaf_voxel_count(&self) -> u64 {
        self.root.active_leaf_voxel_count()
    }
    pub fn inactive_leaf_voxel_count(&self) -> u64 {
        self.root.inactive_leaf_voxel_count()
    }
    pub fn active_tile_count(&self) -> u64 {
        self.root.active_tile_count()
    }
    pub fn leaf_count(&self) -> usize {
        self.root.leaf_count()
    }
    /// Node counts per level, index 0 = leaves, last = root.
    pub fn node_count(&self) -> Vec<usize> {
        self.root.node_count()
    }
    /// Bounding box of all active values (voxels and tiles).
    pub fn eval_active_voxel_bbox(&self) -> CoordBBox {
        let mut b = CoordBBox::empty();
        self.root.eval_active_bbox(&mut b, true);
        b
    }
    /// Bounding box of all leaf nodes (and active tiles).
    pub fn eval_leaf_bbox(&self) -> CoordBBox {
        let mut b = CoordBBox::empty();
        self.root.eval_active_bbox(&mut b, false);
        for l in self.leaves() {
            b.expand_bbox(&l.node_bbox());
        }
        b
    }
    /// Dimensions of the active-voxel bounding box.
    pub fn eval_active_voxel_dim(&self) -> Coord {
        self.eval_active_voxel_bbox().dim()
    }
    pub fn mem_usage(&self) -> usize {
        self.root.mem_usage()
    }

    // ---- Topology editing ----

    /// Replace constant nodes (within `tolerance`) with tiles.
    pub fn prune(&mut self, tolerance: C::Value) {
        self.root.prune(&tolerance)
    }
    /// Replace nodes with no active values by inactive background tiles.
    pub fn prune_inactive(&mut self) {
        self.root.prune_inactive()
    }
    /// Deactivate and reset to background every value outside `bbox`.
    pub fn clip(&mut self, bbox: &CoordBBox) {
        self.root.clip(bbox)
    }
    /// Densify active tiles into leaf voxels.
    pub fn voxelize_active_tiles(&mut self) {
        self.root.voxelize_active_tiles()
    }
    /// Mark every value active or inactive.
    pub fn set_values_on_off(&mut self, on: bool) {
        self.root.set_values_on_off(on)
    }
    /// Union the active topology of `other` (any value type) into this tree: every value
    /// active in `other` becomes active here (keeping its existing value).
    pub fn topology_union<D: Node>(&mut self, other: &TreeBase<D>) {
        let mut tiles = Vec::new();
        other.visit_tiles(&mut |b, _, on, _| {
            if on {
                tiles.push(b);
            }
        });
        for b in tiles {
            self.root.activate_region(&b);
        }
        for leaf in other.leaves() {
            let mine = self.touch_leaf(&leaf.origin());
            mine.value_mask_mut().union_with(leaf.value_mask());
        }
    }
    /// Mark every value in `bbox` active without changing values (tiles are kept where
    /// possible).
    pub fn activate_region(&mut self, bbox: &CoordBBox) {
        self.root.activate_region(bbox)
    }
}

/// An active value yielded by [`TreeBase::iter_values_on`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueItem<T> {
    /// Region covered by the value (one voxel for leaf values).
    pub bbox: CoordBBox,
    pub value: T,
    /// 0 for voxels, higher for tiles.
    pub level: u32,
}

impl<T: Value> PartialEq for TreeBase<super::internal::InternalNode<super::internal::InternalNode<LeafNode<T>, 4>, 5>>
where
    T: PartialEq,
{
    /// Two trees are equal if they have the same background and the same active values
    /// and topology.
    fn eq(&self, other: &Self) -> bool {
        if self.background() != other.background()
            || self.active_voxel_count() != other.active_voxel_count()
            || self.leaf_count() != other.leaf_count()
        {
            return false;
        }
        for (a, b) in self.leaves().zip(other.leaves()) {
            if a.origin() != b.origin() || a.value_mask() != b.value_mask() {
                return false;
            }
            if a.iter_on().zip(b.iter_on()).any(|((_, x), (_, y))| x != y) {
                return false;
            }
        }
        let mut ta = Vec::new();
        let mut tb = Vec::new();
        self.visit_tiles(&mut |b, v, on, _| {
            if on {
                ta.push((b, *v))
            }
        });
        other.visit_tiles(&mut |b, v, on, _| {
            if on {
                tb.push((b, *v))
            }
        });
        ta == tb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_basics() {
        let mut tree = FloatTree::new(-1.0);
        assert_eq!(FloatTree::DEPTH, 4);
        assert_eq!(tree.get_value(&Coord::new(1000, -5000, 3)), -1.0);
        let coords = [Coord::new(0, 0, 0), Coord::new(-1, -1, -1), Coord::new(5000, 1, -70000)];
        for (i, c) in coords.iter().enumerate() {
            tree.set_value(c, i as f32);
        }
        for (i, c) in coords.iter().enumerate() {
            assert_eq!(tree.get_value(c), i as f32);
            assert!(tree.is_value_on(c));
            assert_eq!(tree.get_value_depth(c), 3);
        }
        assert_eq!(tree.active_voxel_count(), 3);
        assert_eq!(tree.leaf_count(), 3);
        assert_eq!(tree.node_count(), vec![3, 3, 3, 1]);
        let bbox = tree.eval_active_voxel_bbox();
        assert_eq!(bbox.min, Coord::new(-1, -1, -70000));
        assert_eq!(bbox.max, Coord::new(5000, 1, 0));
        assert_eq!(tree.active_voxels().count(), 3);

        tree.set_value_off(&coords[0], 7.0);
        assert!(!tree.is_value_on(&coords[0]));
        assert_eq!(tree.active_voxel_count(), 2);
    }

    #[test]
    fn fill_prune_clip() {
        let mut tree = FloatTree::new(0.0);
        let b = CoordBBox::from_coords(-100, -100, -100, 100, 100, 100);
        tree.fill(&b, 1.0, true);
        assert_eq!(tree.active_voxel_count(), b.volume());
        assert_eq!(tree.eval_active_voxel_bbox(), b);
        let small = CoordBBox::from_coords(-10, -10, -10, 10, 10, 10);
        tree.clip(&small);
        assert_eq!(tree.active_voxel_count(), small.volume());
        assert_eq!(tree.get_value(&Coord::new(50, 50, 50)), 0.0);
        tree.voxelize_active_tiles();
        assert_eq!(tree.active_voxel_count(), small.volume());
        assert_eq!(tree.active_tile_count(), 0);
        tree.prune(0.0);
        assert_eq!(tree.active_voxel_count(), small.volume());

        let mut dense = FloatTree::new(0.0);
        dense.dense_fill(&CoordBBox::from_coords(0, 0, 0, 15, 15, 15), 2.0, true);
        assert_eq!(dense.leaf_count(), 8);
        assert_eq!(dense.active_tile_count(), 0);
    }

    #[test]
    fn root_tiles_and_background() {
        let mut tree = FloatTree::new(0.0);
        tree.add_tile(3, &Coord::new(0, 0, 0), 5.0, true);
        assert_eq!(tree.get_value(&Coord::new(4095, 4095, 4095)), 5.0);
        assert_eq!(tree.get_value_depth(&Coord::new(1, 1, 1)), 0);
        assert_eq!(tree.active_voxel_count(), 4096u64.pow(3));
        tree.set_value(&Coord::new(1, 1, 1), 2.0);
        assert_eq!(tree.get_value(&Coord::new(1, 1, 1)), 2.0);
        assert_eq!(tree.get_value(&Coord::new(1, 1, 2)), 5.0);
        tree.set_value_off(&Coord::new(-5, 0, 0), 0.0);
        assert_eq!(tree.leaf_count(), 1);
        tree.set_value_off(&Coord::new(-5, 0, 0), 3.0);
        tree.set_background(10.0);
        assert_eq!(tree.get_value(&Coord::new(-5, 0, 1)), 10.0);
        assert_eq!(tree.get_value(&Coord::new(-5, 0, 0)), 3.0);
    }

    #[test]
    fn topology_union() {
        let mut a = FloatTree::new(0.0);
        let mut b = BoolTree::new(false);
        b.set_value(&Coord::new(10, 10, 10), true);
        b.fill(&CoordBBox::from_coords(0, 0, 0, 15, 15, 15), true, true);
        a.set_value(&Coord::new(-3, 0, 0), 1.0);
        a.topology_union(&b);
        assert_eq!(a.active_voxel_count(), 4096 + 1);
        assert_eq!(a.get_value(&Coord::new(10, 10, 10)), 0.0);
    }
}
