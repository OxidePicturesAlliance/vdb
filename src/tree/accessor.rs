//! Value accessors: cached random access into a tree (port of `openvdb/tree/ValueAccessor.h`).
//!
//! Accessors remember the most recently visited leaf node and top-level internal node, so
//! spatially coherent access patterns skip most of the tree traversal.

use super::leaf::LeafNode;
use super::node::{Lookup, Node};
use super::tree::TreeBase;
use crate::math::Coord;
use crate::types::Value;
use std::cell::Cell;
use std::marker::PhantomData;

/// Read-only cached accessor.
///
/// Besides the last leaf and top-level node, it remembers the last tile (or background
/// region) a lookup ended in, so reads inside large tiles don't descend the tree again.
pub struct ValueAccessor<'a, C: Node> {
    tree: &'a TreeBase<C>,
    leaf_key: Cell<Coord>,
    leaf: Cell<Option<&'a LeafNode<C::Value>>>,
    node_key: Cell<Coord>,
    node: Cell<Option<&'a C>>,
    tile: Cell<CachedTile<C::Value>>,
}

/// The tile cached by a [`ValueAccessor`]: the region `key .. key + dim`, where `mask` is
/// `!(dim - 1)`.
#[derive(Clone, Copy)]
struct CachedTile<T> {
    key: Coord,
    mask: i32,
    value: T,
    active: bool,
}

impl<T: Copy> CachedTile<T> {
    #[inline]
    fn hit(&self, xyz: &Coord) -> bool {
        xyz.align_mask(self.mask) == self.key
    }
}

impl<'a, C: Node> Clone for ValueAccessor<'a, C> {
    fn clone(&self) -> Self {
        ValueAccessor {
            tree: self.tree,
            leaf_key: self.leaf_key.clone(),
            leaf: self.leaf.clone(),
            node_key: self.node_key.clone(),
            node: self.node.clone(),
            tile: self.tile.clone(),
        }
    }
}

impl<'a, C: Node> ValueAccessor<'a, C> {
    pub fn new(tree: &'a TreeBase<C>) -> Self {
        ValueAccessor {
            tree,
            leaf_key: Cell::new(Coord::max_coord()),
            leaf: Cell::new(None),
            node_key: Cell::new(Coord::max_coord()),
            node: Cell::new(None),
            tile: Cell::new(Self::no_tile()),
        }
    }
    #[inline]
    fn no_tile() -> CachedTile<C::Value> {
        // Every coordinate masks to the origin, which never equals the key: no hits.
        CachedTile { key: Coord::max_coord(), mask: 0, value: C::Value::zero(), active: false }
    }
    pub fn tree(&self) -> &'a TreeBase<C> {
        self.tree
    }
    /// Drop all cached nodes.
    pub fn clear(&self) {
        self.leaf.set(None);
        self.leaf_key.set(Coord::max_coord());
        self.node.set(None);
        self.tile.set(Self::no_tile());
    }

    #[inline]
    fn cached_leaf(&self, xyz: &Coord) -> Option<&'a LeafNode<C::Value>> {
        if xyz.align_down(8) == self.leaf_key.get() {
            self.leaf.get()
        } else {
            None
        }
    }
    #[inline]
    fn top_node(&self, xyz: &Coord) -> Option<&'a C> {
        let key = xyz.align_down(C::DIM);
        if key == self.node_key.get() {
            if let Some(n) = self.node.get() {
                return Some(n);
            }
        }
        let n = self.tree.root().probe_child(xyz)?;
        self.node_key.set(key);
        self.node.set(Some(n));
        Some(n)
    }
    #[inline]
    fn find_leaf(&self, xyz: &Coord) -> Option<&'a LeafNode<C::Value>> {
        if let Some(l) = self.cached_leaf(xyz) {
            return Some(l);
        }
        let l = self.top_node(xyz)?.probe_leaf(xyz)?;
        self.leaf_key.set(l.origin());
        self.leaf.set(Some(l));
        Some(l)
    }

    /// Find the leaf containing `xyz`, or the value and state of the tile covering it,
    /// with a single descent, caching whichever is found.
    #[inline]
    fn lookup(&self, xyz: &Coord) -> Result<&'a LeafNode<C::Value>, (C::Value, bool)> {
        if let Some(l) = self.cached_leaf(xyz) {
            return Ok(l);
        }
        self.lookup_slow(xyz)
    }
    #[inline]
    fn lookup_slow(&self, xyz: &Coord) -> Result<&'a LeafNode<C::Value>, (C::Value, bool)> {
        let t = self.tile.get();
        if t.hit(xyz) {
            return Err((t.value, t.active));
        }
        let (value, active, dim) = match self.top_node(xyz) {
            Some(n) => match n.lookup(xyz) {
                Lookup::Leaf(l) => {
                    self.leaf_key.set(l.origin());
                    self.leaf.set(Some(l));
                    return Ok(l);
                }
                Lookup::Tile { value, active, dim } => (*value, active, dim),
            },
            None => {
                let (v, on) = self.tree.root().probe_value(xyz);
                (v, on, C::DIM)
            }
        };
        let mask = !(dim - 1);
        self.tile.set(CachedTile { key: xyz.align_mask(mask), mask, value, active });
        Err((value, active))
    }

    /// The value at `xyz`.
    #[inline]
    pub fn get_value(&self, xyz: &Coord) -> C::Value {
        match self.lookup(xyz) {
            Ok(l) => *l.get_value(xyz),
            Err((v, _)) => v,
        }
    }
    #[inline]
    pub fn is_value_on(&self, xyz: &Coord) -> bool {
        match self.lookup(xyz) {
            Ok(l) => l.is_value_on(xyz),
            Err((_, on)) => on,
        }
    }
    #[inline]
    pub fn probe_value(&self, xyz: &Coord) -> (C::Value, bool) {
        match self.lookup(xyz) {
            Ok(l) => l.probe_value(xyz),
            Err(t) => t,
        }
    }
    pub fn probe_leaf(&self, xyz: &Coord) -> Option<&'a LeafNode<C::Value>> {
        self.find_leaf(xyz)
    }
    /// The top-level node (child of the root) containing `xyz`, if any (cached).
    #[inline]
    pub fn probe_top_node(&self, xyz: &Coord) -> Option<&'a C> {
        self.top_node(xyz)
    }
    /// Depth of the value at `xyz` (see [`TreeBase::get_value_depth`]).
    pub fn get_value_depth(&self, xyz: &Coord) -> i32 {
        self.tree.get_value_depth(xyz)
    }
}

/// Read/write cached accessor.
///
/// Holds an exclusive borrow of the tree. Cached node pointers stay valid because every node
/// is individually boxed and is only freed by operations that also reset the cache.
pub struct ValueAccessorMut<'a, C: Node> {
    tree: *mut TreeBase<C>,
    leaf_key: Coord,
    leaf: *mut LeafNode<C::Value>,
    node_key: Coord,
    node: *mut C,
    _marker: PhantomData<&'a mut TreeBase<C>>,
}

// SAFETY: the accessor is semantically an `&'a mut TreeBase<C>`.
unsafe impl<'a, C: Node> Send for ValueAccessorMut<'a, C> {}

impl<'a, C: Node> ValueAccessorMut<'a, C> {
    pub fn new(tree: &'a mut TreeBase<C>) -> Self {
        ValueAccessorMut {
            tree: tree as *mut _,
            leaf_key: Coord::max_coord(),
            leaf: std::ptr::null_mut(),
            node_key: Coord::max_coord(),
            node: std::ptr::null_mut(),
            _marker: PhantomData,
        }
    }
    /// Shared access to the underlying tree.
    pub fn tree(&self) -> &TreeBase<C> {
        // SAFETY: we hold the unique borrow for 'a.
        unsafe { &*self.tree }
    }
    /// Exclusive access to the underlying tree. Clears the cache, since the caller may
    /// restructure the tree.
    pub fn tree_mut(&mut self) -> &mut TreeBase<C> {
        self.clear();
        // SAFETY: we hold the unique borrow for 'a.
        unsafe { &mut *self.tree }
    }
    pub fn clear(&mut self) {
        self.leaf = std::ptr::null_mut();
        self.node = std::ptr::null_mut();
    }

    #[inline]
    fn root(&self) -> &super::root::RootNode<C> {
        unsafe { (*self.tree).root() }
    }
    #[inline]
    fn root_mut(&mut self) -> &mut super::root::RootNode<C> {
        unsafe { (*self.tree).root_mut() }
    }

    #[inline]
    fn cached_leaf(&self, xyz: &Coord) -> Option<&LeafNode<C::Value>> {
        if !self.leaf.is_null() && xyz.align_down(8) == self.leaf_key {
            // SAFETY: pointer refers to a boxed leaf owned by the exclusively borrowed tree.
            Some(unsafe { &*self.leaf })
        } else {
            None
        }
    }
    #[inline]
    fn leaf_hit(&self, xyz: &Coord) -> bool {
        !self.leaf.is_null() && xyz.align_down(8) == self.leaf_key
    }
    #[inline]
    fn cached_leaf_mut(&mut self, xyz: &Coord) -> Option<&mut LeafNode<C::Value>> {
        if !self.leaf.is_null() && xyz.align_down(8) == self.leaf_key {
            // SAFETY: as above, and `&mut self` guarantees exclusivity.
            Some(unsafe { &mut *self.leaf })
        } else {
            None
        }
    }
    #[inline]
    fn top_node(&mut self, xyz: &Coord) -> Option<&mut C> {
        let key = xyz.align_down(C::DIM);
        if !self.node.is_null() && key == self.node_key {
            return Some(unsafe { &mut *self.node });
        }
        let n: *mut C = self.root_mut().probe_child_mut(xyz)?;
        self.node_key = key;
        self.node = n;
        Some(unsafe { &mut *n })
    }
    #[inline]
    fn top_node_ref(&self, xyz: &Coord) -> Option<&C> {
        let key = xyz.align_down(C::DIM);
        if !self.node.is_null() && key == self.node_key {
            return Some(unsafe { &*self.node });
        }
        self.root().probe_child(xyz)
    }
    #[inline]
    fn touch_top_node(&mut self, xyz: &Coord) -> &mut C {
        let key = xyz.align_down(C::DIM);
        if !self.node.is_null() && key == self.node_key {
            return unsafe { &mut *self.node };
        }
        let n: *mut C = self.root_mut().touch_child(xyz);
        self.node_key = key;
        self.node = n;
        unsafe { &mut *n }
    }

    #[inline]
    pub fn get_value(&self, xyz: &Coord) -> C::Value {
        if let Some(l) = self.cached_leaf(xyz) {
            return *l.get_value(xyz);
        }
        match self.top_node_ref(xyz) {
            Some(n) => *n.get_value(xyz),
            None => *self.root().get_value(xyz),
        }
    }
    #[inline]
    pub fn is_value_on(&self, xyz: &Coord) -> bool {
        if let Some(l) = self.cached_leaf(xyz) {
            return l.is_value_on(xyz);
        }
        match self.top_node_ref(xyz) {
            Some(n) => n.is_value_on(xyz),
            None => self.root().is_value_on(xyz),
        }
    }
    #[inline]
    pub fn probe_value(&self, xyz: &Coord) -> (C::Value, bool) {
        if let Some(l) = self.cached_leaf(xyz) {
            return l.probe_value(xyz);
        }
        match self.top_node_ref(xyz) {
            Some(n) => n.probe_value(xyz),
            None => self.root().probe_value(xyz),
        }
    }

    /// Return the leaf containing `xyz`, creating it if necessary.
    #[inline]
    pub fn touch_leaf(&mut self, xyz: &Coord) -> &mut LeafNode<C::Value> {
        if self.leaf_hit(xyz) {
            return unsafe { &mut *self.leaf };
        }
        let l: *mut LeafNode<C::Value> = self.touch_top_node(xyz).touch_leaf(xyz);
        self.leaf_key = xyz.align_down(8);
        self.leaf = l;
        unsafe { &mut *l }
    }
    /// Return the leaf containing `xyz`, if it exists.
    #[inline]
    pub fn probe_leaf(&mut self, xyz: &Coord) -> Option<&mut LeafNode<C::Value>> {
        if self.leaf_hit(xyz) {
            return Some(unsafe { &mut *self.leaf });
        }
        let l: *mut LeafNode<C::Value> = self.top_node(xyz)?.probe_leaf_mut(xyz)?;
        self.leaf_key = xyz.align_down(8);
        self.leaf = l;
        Some(unsafe { &mut *l })
    }

    #[inline]
    pub fn set_value(&mut self, xyz: &Coord, value: C::Value) {
        self.set_value_on(xyz, value)
    }
    #[inline]
    pub fn set_value_on(&mut self, xyz: &Coord, value: C::Value) {
        if let Some(l) = self.cached_leaf_mut(xyz) {
            return l.set_value_on(xyz, value);
        }
        self.touch_leaf(xyz).set_value_on(xyz, value);
    }
    #[inline]
    pub fn set_value_off(&mut self, xyz: &Coord, value: C::Value) {
        if let Some(l) = self.cached_leaf_mut(xyz) {
            return l.set_value_off(xyz, value);
        }
        let (v, on) = self.probe_value(xyz);
        if !on && v == value {
            return;
        }
        self.touch_leaf(xyz).set_value_off(xyz, value);
    }
    #[inline]
    pub fn set_value_only(&mut self, xyz: &Coord, value: C::Value) {
        if let Some(l) = self.cached_leaf_mut(xyz) {
            return l.set_value_only(xyz, value);
        }
        if self.get_value(xyz) == value {
            return;
        }
        self.touch_leaf(xyz).set_value_only(xyz, value);
    }
    #[inline]
    pub fn set_active_state(&mut self, xyz: &Coord, on: bool) {
        if let Some(l) = self.cached_leaf_mut(xyz) {
            return l.set_active_state(xyz, on);
        }
        if self.is_value_on(xyz) == on {
            return;
        }
        self.touch_leaf(xyz).set_active_state(xyz, on);
    }
    #[inline]
    pub fn set_value_on_only(&mut self, xyz: &Coord) {
        self.set_active_state(xyz, true)
    }
    #[inline]
    pub fn modify_value<F: FnOnce(&mut C::Value)>(&mut self, xyz: &Coord, f: F) {
        self.touch_leaf(xyz).modify_value(xyz, f)
    }
    #[inline]
    pub fn modify_value_and_active_state<F: FnOnce(&mut C::Value, &mut bool)>(&mut self, xyz: &Coord, f: F) {
        self.touch_leaf(xyz).modify_value_and_active_state(xyz, f)
    }
    /// Add a tile at `level` (see [`TreeBase::add_tile`]). Clears the cache.
    pub fn add_tile(&mut self, level: u32, xyz: &Coord, value: C::Value, active: bool) {
        self.clear();
        self.root_mut().add_tile(level, xyz, value, active);
    }
    /// Insert a leaf node. Clears the cache.
    pub fn add_leaf(&mut self, leaf: Box<LeafNode<C::Value>>) {
        self.clear();
        self.root_mut().add_leaf(leaf);
    }
}

#[cfg(test)]
mod tests {
    use crate::math::Coord;
    use crate::tree::FloatTree;

    #[test]
    fn accessors() {
        let mut tree = FloatTree::new(0.5);
        {
            let mut acc = tree.accessor_mut();
            for i in 0..100 {
                acc.set_value(&Coord::new(i, i * 2, -i), i as f32);
            }
            assert_eq!(acc.get_value(&Coord::new(3, 6, -3)), 3.0);
            acc.set_value_off(&Coord::new(3, 6, -3), 9.0);
            assert!(!acc.is_value_on(&Coord::new(3, 6, -3)));
            acc.modify_value(&Coord::new(4, 8, -4), |v| *v += 1.0);
            acc.set_active_state(&Coord::new(10000, 0, 0), false);
        }
        assert_eq!(tree.active_voxel_count(), 99);
        let acc = tree.accessor();
        assert_eq!(acc.get_value(&Coord::new(4, 8, -4)), 5.0);
        assert_eq!(acc.get_value(&Coord::new(3, 6, -3)), 9.0);
        assert_eq!(acc.get_value(&Coord::new(1000, 1000, 1000)), 0.5);
        assert!(acc.is_value_on(&Coord::new(99, 198, -99)));
        assert_eq!(tree.leaf_count(), tree.leaves().count());
    }
}
