//! Ports of the `Tree`/`RootNode`/`InternalNode`/`LeafNode` member functions that the tools
//! depend on but that the core tree module does not provide with OpenVDB's exact semantics:
//!
//! * [`merge`] / [`Tree::merge`](crate::tree::TreeBase) with the three [`MergePolicy`] modes
//!   (`Tree::merge`),
//! * [`combine_extended`] (`Tree::combineExtended`, used by the `comp*` compositing tools),
//! * [`topology_union`], [`topology_intersection`], [`topology_difference`] and
//!   [`topology_copy`] (`Tree::topologyUnion` etc. and the `TopologyCopy` constructors),
//! * [`reset_background`] (`Node::resetBackground`, which also maps `-old` to `-new`),
//! * helpers to collect the internal nodes of the standard tree.

use crate::math::{Coord, CoordBBox};
use crate::tools::value_traits::ToolValue;
use crate::tree::{Internal1, Internal2, InternalNode, LeafNode, Node, RootEntry, Tree, TreeBase};
use crate::types::Value;
use rayon::prelude::*;
use std::collections::BTreeSet;

/// How [`merge`] combines the nodes of two trees (OpenVDB's `MergePolicy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergePolicy {
    /// Active values of the other tree replace inactive values of this tree
    /// (`MERGE_ACTIVE_STATES`).
    #[default]
    ActiveStates,
    /// Nodes of the other tree are transferred where this tree only has tiles
    /// (`MERGE_NODES`).
    Nodes,
    /// Combination of both (`MERGE_ACTIVE_STATES_AND_NODES`).
    ActiveStatesAndNodes,
}

/// Arguments of a combine operation (OpenVDB's `CombineArgs`).
///
/// The operation reads `a`, `b` and their active states and writes `result`.
/// `result_active` is initialized to `a_active || b_active` and may be overridden.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CombineArgs<T> {
    pub a: T,
    pub b: T,
    pub a_active: bool,
    pub b_active: bool,
    pub result: T,
    pub result_active: bool,
}

impl<T: Value> CombineArgs<T> {
    fn new(a: T, a_active: bool, b: T, b_active: bool) -> Self {
        CombineArgs { a, b, a_active, b_active, result: a, result_active: a_active || b_active }
    }
}

/// Apply `op` to `(a, b)` (or to `(b, a)` if `swapped`) and return the result and its state.
#[inline]
fn apply_op<T: Value, F: FnMut(&mut CombineArgs<T>)>(
    op: &mut F,
    a: T,
    a_on: bool,
    b: T,
    b_on: bool,
    swapped: bool,
) -> (T, bool) {
    let mut args = if swapped { CombineArgs::new(b, b_on, a, a_on) } else { CombineArgs::new(a, a_on, b, b_on) };
    op(&mut args);
    (args.result, args.result_active)
}

/// Node-level operations used by the tools, implemented for leaf and internal nodes.
pub trait NodeOps<T: ToolValue>: Node<Value = T> {
    /// `Node::resetBackground`: replace inactive values approximately equal to `old`
    /// (or `-old`) with `new` (or `-new`).
    fn reset_background_cpp(&mut self, old: &T, new: &T);
    /// `Node::merge<Policy>(other, background, otherBackground)`.
    fn merge_node(&mut self, other: &mut Self, policy: MergePolicy, background: &T, other_background: &T);
    /// `Node::merge<Policy>(tileValue, tileActive)` (only affects `ActiveStatesAndNodes`).
    fn merge_tile(&mut self, policy: MergePolicy, value: &T, active: bool);
    /// `Node::combine(other, op)`.
    fn combine_node<F: FnMut(&mut CombineArgs<T>)>(&mut self, other: &mut Self, op: &mut F);
    /// `Node::combine(value, active, op)`; if `swapped`, the constant is the A operand.
    fn combine_value<F: FnMut(&mut CombineArgs<T>)>(&mut self, value: &T, active: bool, op: &mut F, swapped: bool);
    /// Negate every value (`Node::negate`).
    fn negate_values(&mut self);
    /// The value at offset 0, descending into children (`getFirstValue`).
    fn first_value(&self) -> T;
    /// The value at the last offset, descending into children (`getLastValue`).
    fn last_value(&self) -> T;
}

impl<T: ToolValue> NodeOps<T> for LeafNode<T> {
    fn reset_background_cpp(&mut self, old: &T, new: &T) {
        let neg_old = old.negative();
        let neg_new = new.negative();
        let (buf, mask) = self.buffer_and_mask_mut();
        for n in mask.iter_off() {
            if buf[n].is_approx_equal(old) {
                buf[n] = *new;
            } else if buf[n].is_approx_equal(&neg_old) {
                buf[n] = neg_new;
            }
        }
    }
    fn merge_node(&mut self, other: &mut Self, policy: MergePolicy, _bg: &T, _obg: &T) {
        if policy == MergePolicy::Nodes {
            return;
        }
        for n in other.value_mask().iter_on() {
            if !self.is_value_on_at(n) {
                self.set_value_on_at(n, *other.get_value_at(n));
            }
        }
    }
    fn merge_tile(&mut self, policy: MergePolicy, value: &T, active: bool) {
        if policy != MergePolicy::ActiveStatesAndNodes || !active {
            return;
        }
        let (buf, mask) = self.buffer_and_mask_mut();
        for n in 0..LeafNode::<T>::SIZE {
            if mask.is_off(n) {
                buf[n] = *value;
                mask.set_on(n);
            }
        }
    }
    fn combine_node<F: FnMut(&mut CombineArgs<T>)>(&mut self, other: &mut Self, op: &mut F) {
        for n in 0..LeafNode::<T>::SIZE {
            let (r, on) = apply_op(
                op,
                *self.get_value_at(n),
                self.is_value_on_at(n),
                *other.get_value_at(n),
                other.is_value_on_at(n),
                false,
            );
            self.set_value_only_at(n, r);
            self.set_active_state_at(n, on);
        }
    }
    fn combine_value<F: FnMut(&mut CombineArgs<T>)>(&mut self, value: &T, active: bool, op: &mut F, swapped: bool) {
        for n in 0..LeafNode::<T>::SIZE {
            let (r, on) = apply_op(op, *self.get_value_at(n), self.is_value_on_at(n), *value, active, swapped);
            self.set_value_only_at(n, r);
            self.set_active_state_at(n, on);
        }
    }
    fn negate_values(&mut self) {
        for v in self.buffer_mut() {
            *v = v.negative();
        }
    }
    fn first_value(&self) -> T {
        *self.get_value_at(0)
    }
    fn last_value(&self) -> T {
        *self.get_value_at(LeafNode::<T>::SIZE - 1)
    }
}

impl<T: ToolValue, C: NodeOps<T>, const L: u32> NodeOps<T> for InternalNode<C, L> {
    fn reset_background_cpp(&mut self, old: &T, new: &T) {
        let neg_old = old.negative();
        let neg_new = new.negative();
        let children: Vec<&mut C> = self.children_mut().collect();
        children.into_par_iter().with_min_len(if C::LEVEL == 0 { 64 } else { 1 }).for_each(|c| {
            c.reset_background_cpp(old, new);
        });
        for n in 0..Self::NUM_VALUES {
            if self.is_child(n) {
                continue;
            } else if !self.value_mask().is_on(n) {
                let v = self.values()[n];
                if v.is_approx_equal(old) {
                    self.values_mut()[n] = *new;
                } else if v.is_approx_equal(&neg_old) {
                    self.values_mut()[n] = neg_new;
                }
            }
        }
    }
    fn merge_node(&mut self, other: &mut Self, policy: MergePolicy, bg: &T, obg: &T) {
        match policy {
            MergePolicy::ActiveStates => {
                for n in 0..Self::NUM_VALUES {
                    if !other.is_child(n) {
                        continue;
                    }
                    if self.is_child(n) {
                        let oc = other.child_mut(n).unwrap();
                        self.child_mut(n).unwrap().merge_node(oc, policy, bg, obg);
                    } else if !self.value_mask().is_on(n) {
                        let ov = other.values()[n];
                        let mut child = other.set_tile(n, ov, false).unwrap();
                        child.reset_background_cpp(obg, bg);
                        self.set_child(n, child);
                    }
                }
                for n in 0..Self::NUM_VALUES {
                    if other.is_child(n) || !other.value_mask().is_on(n) {
                        continue;
                    }
                    if !self.value_mask().is_on(n) {
                        let v = other.values()[n];
                        self.set_tile(n, v, true);
                    }
                }
            }
            MergePolicy::Nodes => {
                for n in 0..Self::NUM_VALUES {
                    if !other.is_child(n) {
                        continue;
                    }
                    if self.is_child(n) {
                        let oc = other.child_mut(n).unwrap();
                        self.child_mut(n).unwrap().merge_node(oc, policy, bg, obg);
                    } else {
                        let ov = other.values()[n];
                        let mut child = other.set_tile(n, ov, false).unwrap();
                        child.reset_background_cpp(obg, bg);
                        self.set_child(n, child);
                    }
                }
            }
            MergePolicy::ActiveStatesAndNodes => {
                for n in 0..Self::NUM_VALUES {
                    if !other.is_child(n) {
                        continue;
                    }
                    if self.is_child(n) {
                        let oc = other.child_mut(n).unwrap();
                        self.child_mut(n).unwrap().merge_node(oc, policy, bg, obg);
                    } else {
                        let ov = other.values()[n];
                        let mut child = other.set_tile(n, ov, false).unwrap();
                        child.reset_background_cpp(obg, bg);
                        if self.value_mask().is_on(n) {
                            let tv = self.values()[n];
                            child.merge_tile(policy, &tv, true);
                        }
                        self.set_child(n, child);
                    }
                }
                for n in 0..Self::NUM_VALUES {
                    if other.is_child(n) || !other.value_mask().is_on(n) {
                        continue;
                    }
                    let v = other.values()[n];
                    if self.is_child(n) {
                        self.child_mut(n).unwrap().merge_tile(policy, &v, true);
                    } else if !self.value_mask().is_on(n) {
                        self.set_tile(n, v, true);
                    }
                }
            }
        }
    }
    fn merge_tile(&mut self, policy: MergePolicy, value: &T, active: bool) {
        if policy != MergePolicy::ActiveStatesAndNodes || !active {
            return;
        }
        for n in 0..Self::NUM_VALUES {
            if let Some(c) = self.child_mut(n) {
                c.merge_tile(policy, value, true);
            } else if !self.value_mask().is_on(n) {
                self.set_tile(n, *value, true);
            }
        }
    }
    fn combine_node<F: FnMut(&mut CombineArgs<T>)>(&mut self, other: &mut Self, op: &mut F) {
        for n in 0..Self::NUM_VALUES {
            match (self.is_child(n), other.is_child(n)) {
                (false, false) => {
                    let (r, on) = apply_op(
                        op,
                        self.values()[n],
                        self.value_mask().is_on(n),
                        other.values()[n],
                        other.value_mask().is_on(n),
                        false,
                    );
                    self.set_tile(n, r, on);
                }
                (true, false) => {
                    let (ov, oon) = (other.values()[n], other.value_mask().is_on(n));
                    self.child_mut(n).unwrap().combine_value(&ov, oon, op, false);
                }
                (false, true) => {
                    let (tv, ton) = (self.values()[n], self.value_mask().is_on(n));
                    let mut child = other.set_tile(n, T::zero(), false).unwrap();
                    child.combine_value(&tv, ton, op, true);
                    self.set_child(n, child);
                }
                (true, true) => {
                    let oc = other.child_mut(n).unwrap();
                    self.child_mut(n).unwrap().combine_node(oc, op);
                }
            }
        }
    }
    fn combine_value<F: FnMut(&mut CombineArgs<T>)>(&mut self, value: &T, active: bool, op: &mut F, swapped: bool) {
        for n in 0..Self::NUM_VALUES {
            if let Some(c) = self.child_mut(n) {
                c.combine_value(value, active, op, swapped);
            } else {
                let (r, on) = apply_op(op, self.values()[n], self.value_mask().is_on(n), *value, active, swapped);
                self.set_tile(n, r, on);
            }
        }
    }
    fn negate_values(&mut self) {
        for n in 0..Self::NUM_VALUES {
            if let Some(c) = self.child_mut(n) {
                c.negate_values();
            } else {
                let v = self.values()[n];
                self.values_mut()[n] = v.negative();
            }
        }
    }
    fn first_value(&self) -> T {
        match self.child(0) {
            Some(c) => c.first_value(),
            None => self.values()[0],
        }
    }
    fn last_value(&self) -> T {
        let n = Self::NUM_VALUES - 1;
        match self.child(n) {
            Some(c) => c.last_value(),
            None => self.values()[n],
        }
    }
}

/// Topology operations between nodes of the same configuration but possibly different
/// value types (`topologyUnion`, `topologyIntersection`, `topologyDifference` and the
/// `TopologyCopy` constructors).
pub trait TopologyOps<O: Node>: Node {
    /// Copy the topology of `other`; every value is set to `background`.
    fn topology_copy(other: &O, background: Self::Value) -> Self;
    /// Copy the topology of `other`; inactive values are `off`, active values `on`.
    fn topology_copy2(other: &O, off: Self::Value, on: Self::Value) -> Self;
    /// `topologyUnion(other, preserveTiles)`.
    fn topology_union_node(&mut self, other: &O, preserve_tiles: bool);
    /// `topologyIntersection(other, background)`.
    fn topology_intersection_node(&mut self, other: &O, background: &Self::Value);
    /// `topologyDifference(other, background)`.
    fn topology_difference_node(&mut self, other: &O, background: &Self::Value);
}

impl<T: Value, U: Value> TopologyOps<LeafNode<U>> for LeafNode<T> {
    fn topology_copy(other: &LeafNode<U>, background: T) -> Self {
        LeafNode::from_parts(other.origin(), other.value_mask().clone(), vec![background; LeafNode::<T>::SIZE])
    }
    fn topology_copy2(other: &LeafNode<U>, off: T, on: T) -> Self {
        let buf = (0..LeafNode::<T>::SIZE).map(|n| if other.is_value_on_at(n) { on } else { off }).collect();
        LeafNode::from_parts(other.origin(), other.value_mask().clone(), buf)
    }
    fn topology_union_node(&mut self, other: &LeafNode<U>, _preserve: bool) {
        self.value_mask_mut().union_with(other.value_mask());
    }
    fn topology_intersection_node(&mut self, other: &LeafNode<U>, _bg: &T) {
        self.value_mask_mut().intersect_with(other.value_mask());
    }
    fn topology_difference_node(&mut self, other: &LeafNode<U>, _bg: &T) {
        self.value_mask_mut().difference_with(other.value_mask());
    }
}

impl<C: Node, D: Node, const L: u32> TopologyOps<InternalNode<D, L>> for InternalNode<C, L>
where
    C: TopologyOps<D>,
{
    fn topology_copy(other: &InternalNode<D, L>, background: C::Value) -> Self {
        let n = Self::NUM_VALUES;
        let children = (0..n).map(|i| other.child(i).map(|c| Box::new(C::topology_copy(c, background)))).collect();
        InternalNode::from_parts(
            other.origin(),
            other.child_mask().clone(),
            other.value_mask().clone(),
            vec![background; n],
            children,
        )
    }
    fn topology_copy2(other: &InternalNode<D, L>, off: C::Value, on: C::Value) -> Self {
        let n = Self::NUM_VALUES;
        let children = (0..n).map(|i| other.child(i).map(|c| Box::new(C::topology_copy2(c, off, on)))).collect();
        let values = (0..n).map(|i| if other.value_mask().is_on(i) { on } else { off }).collect();
        InternalNode::from_parts(
            other.origin(),
            other.child_mask().clone(),
            other.value_mask().clone(),
            values,
            children,
        )
    }
    fn topology_union_node(&mut self, other: &InternalNode<D, L>, preserve: bool) {
        for i in 0..Self::NUM_VALUES {
            if let Some(oc) = other.child(i) {
                if let Some(c) = self.child_mut(i) {
                    c.topology_union_node(oc, preserve);
                } else {
                    let on = self.value_mask().is_on(i);
                    if !preserve || !on {
                        let mut child = C::topology_copy(oc, self.values()[i]);
                        if on {
                            child.set_values_on_off(true);
                        }
                        self.set_child(i, Box::new(child));
                    }
                }
            } else if other.value_mask().is_on(i) {
                if let Some(c) = self.child_mut(i) {
                    c.set_values_on_off(true);
                } else {
                    self.value_mask_mut().set_on(i);
                }
            }
        }
    }
    fn topology_intersection_node(&mut self, other: &InternalNode<D, L>, bg: &C::Value) {
        for i in 0..Self::NUM_VALUES {
            let s_child = other.child(i);
            let s_on = other.value_mask().is_on(i);
            if self.is_child(i) {
                if let Some(oc) = s_child {
                    self.child_mut(i).unwrap().topology_intersection_node(oc, bg);
                } else if !s_on {
                    self.set_tile(i, *bg, false);
                }
            } else if self.value_mask().is_on(i) {
                if let Some(oc) = s_child {
                    let child = C::topology_copy(oc, self.values()[i]);
                    self.set_child(i, Box::new(child));
                } else if !s_on {
                    self.value_mask_mut().set_off(i);
                }
            }
        }
    }
    fn topology_difference_node(&mut self, other: &InternalNode<D, L>, bg: &C::Value) {
        for i in 0..Self::NUM_VALUES {
            let s_child = other.child(i);
            let s_on = other.value_mask().is_on(i);
            if self.is_child(i) {
                if let Some(oc) = s_child {
                    self.child_mut(i).unwrap().topology_difference_node(oc, bg);
                } else if s_on {
                    self.set_tile(i, *bg, false);
                }
            } else if self.value_mask().is_on(i) {
                if let Some(oc) = s_child {
                    let mut child = C::new(oc.origin(), self.values()[i], true);
                    child.topology_difference_node(oc, bg);
                    self.set_child(i, Box::new(child));
                } else if s_on {
                    self.value_mask_mut().set_off(i);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Tree-level operations
// ---------------------------------------------------------------------------------------

/// `Tree::merge(other, policy)`: transfer nodes and/or active values from `other` into
/// `tree`. `other` is left empty.
pub fn merge<T: ToolValue, C: NodeOps<T>>(tree: &mut TreeBase<C>, other: &mut TreeBase<C>, policy: MergePolicy) {
    let bg = *tree.background();
    let obg = *other.background();
    let other_table = std::mem::take(other.root_mut().table_mut());
    let table = tree.root_mut().table_mut();
    for (key, oe) in other_table {
        match oe {
            RootEntry::Child(mut oc) => {
                match table.get_mut(&key) {
                    None => {
                        oc.reset_background_cpp(&obg, &bg);
                        table.insert(key, RootEntry::Child(oc));
                    }
                    Some(RootEntry::Tile { value, active }) => {
                        let (tv, ton) = (*value, *active);
                        match policy {
                            MergePolicy::ActiveStates => {
                                if !ton {
                                    oc.reset_background_cpp(&obg, &bg);
                                    table.insert(key, RootEntry::Child(oc));
                                }
                            }
                            MergePolicy::Nodes => {
                                oc.reset_background_cpp(&obg, &bg);
                                table.insert(key, RootEntry::Child(oc));
                            }
                            MergePolicy::ActiveStatesAndNodes => {
                                oc.reset_background_cpp(&obg, &bg);
                                if ton {
                                    oc.merge_tile(policy, &tv, true);
                                }
                                table.insert(key, RootEntry::Child(oc));
                            }
                        }
                    }
                    // OpenVDB passes the two backgrounds in swapped order here, so nodes
                    // transferred below the root have inactive values equal to *this*
                    // tree's background replaced by the other tree's background. This
                    // quirk is reproduced for identical results.
                    Some(RootEntry::Child(c)) => c.merge_node(&mut oc, policy, &obg, &bg),
                }
            }
            RootEntry::Tile { value, active: true } => match policy {
                MergePolicy::ActiveStates => match table.get(&key) {
                    Some(RootEntry::Tile { active: true, .. }) => {}
                    _ => {
                        table.insert(key, RootEntry::Tile { value, active: true });
                    }
                },
                MergePolicy::Nodes => {}
                MergePolicy::ActiveStatesAndNodes => match table.get_mut(&key) {
                    None => {
                        table.insert(key, RootEntry::Tile { value, active: true });
                    }
                    Some(RootEntry::Tile { active: false, .. }) => {
                        table.insert(key, RootEntry::Tile { value, active: true });
                    }
                    Some(RootEntry::Child(c)) => c.merge_tile(policy, &value, true),
                    Some(RootEntry::Tile { .. }) => {}
                },
            },
            RootEntry::Tile { .. } => {}
        }
    }
}

impl<T: ToolValue> Tree<T> {
    /// Merge `other` into this tree using the given policy (OpenVDB's `Tree::merge`).
    /// `other` is left empty. See [`merge`].
    pub fn merge(&mut self, other: &mut Tree<T>, policy: MergePolicy) {
        merge(self, other, policy)
    }
}

/// `Tree::combineExtended(other, op, prune)`: combine every value of `tree` with the
/// corresponding value of `other` (which is left empty). Missing root entries are treated
/// as inactive background tiles, and the backgrounds are combined too.
pub fn combine_extended<T: ToolValue, C: NodeOps<T>, F: FnMut(&mut CombineArgs<T>)>(
    tree: &mut TreeBase<C>,
    other: &mut TreeBase<C>,
    mut op: F,
    prune: bool,
) {
    let bg = *tree.background();
    let obg = *other.background();
    let mut keys: BTreeSet<Coord> = tree.root().table().keys().copied().collect();
    keys.extend(other.root().table().keys().copied());
    let mut other_table = std::mem::take(other.root_mut().table_mut());
    {
        let table = tree.root_mut().table_mut();
        for key in keys {
            let e = table.remove(&key).unwrap_or(RootEntry::Tile { value: bg, active: false });
            let oe = other_table.remove(&key).unwrap_or(RootEntry::Tile { value: obg, active: false });
            let mut result = match (e, oe) {
                (RootEntry::Tile { value: a, active: aon }, RootEntry::Tile { value: b, active: bon }) => {
                    let (r, on) = apply_op(&mut op, a, aon, b, bon, false);
                    RootEntry::Tile { value: r, active: on }
                }
                (RootEntry::Child(mut c), RootEntry::Tile { value: b, active: bon }) => {
                    c.combine_value(&b, bon, &mut op, false);
                    RootEntry::Child(c)
                }
                (RootEntry::Tile { value: a, active: aon }, RootEntry::Child(mut oc)) => {
                    oc.combine_value(&a, aon, &mut op, true);
                    RootEntry::Child(oc)
                }
                (RootEntry::Child(mut c), RootEntry::Child(mut oc)) => {
                    c.combine_node(&mut oc, &mut op);
                    RootEntry::Child(c)
                }
            };
            if prune {
                if let RootEntry::Child(c) = &mut result {
                    c.prune(&T::zero());
                }
            }
            table.insert(key, result);
        }
    }
    let (r, _) = apply_op(&mut op, bg, false, obg, false, false);
    tree.root_mut().set_background_unchecked(r);
}

/// `Tree::topologyUnion(other, preserveTiles)`: activate every value of `tree` that is
/// active in `other`, with OpenVDB's node-level semantics (active tiles of `tree` are kept
/// if `preserve_tiles` is true).
pub fn topology_union<C, D>(tree: &mut TreeBase<C>, other: &TreeBase<D>, preserve_tiles: bool)
where
    C: TopologyOps<D>,
    D: Node,
{
    let bg = *tree.background();
    let table = tree.root_mut().table_mut();
    for (key, oe) in other.root().table() {
        match oe {
            RootEntry::Child(oc) => match table.get_mut(key) {
                None => {
                    table.insert(*key, RootEntry::Child(Box::new(C::topology_copy(oc, bg))));
                }
                Some(RootEntry::Child(c)) => c.topology_union_node(oc, preserve_tiles),
                Some(RootEntry::Tile { value, active }) => {
                    if !preserve_tiles || !*active {
                        let mut child = C::topology_copy(oc, *value);
                        if *active {
                            child.set_values_on_off(true);
                        }
                        table.insert(*key, RootEntry::Child(Box::new(child)));
                    }
                }
            },
            RootEntry::Tile { active: true, .. } => match table.get_mut(key) {
                None => {
                    table.insert(*key, RootEntry::Tile { value: bg, active: true });
                }
                Some(RootEntry::Child(c)) => c.set_values_on_off(true),
                Some(RootEntry::Tile { active, .. }) => *active = true,
            },
            RootEntry::Tile { .. } => {}
        }
    }
}

/// `Tree::topologyIntersection(other)`: deactivate every value of `tree` that is inactive
/// in `other`.
pub fn topology_intersection<C, D>(tree: &mut TreeBase<C>, other: &TreeBase<D>)
where
    C: TopologyOps<D>,
    D: Node,
{
    let bg = *tree.background();
    let otable = other.root().table();
    let table = tree.root_mut().table_mut();
    let mut erase = Vec::new();
    for (key, e) in table.iter_mut() {
        let oe = otable.get(key);
        let other_off = matches!(oe, None | Some(RootEntry::Tile { active: false, .. }));
        match e {
            RootEntry::Child(c) => {
                if other_off {
                    erase.push(*key);
                } else if let Some(RootEntry::Child(oc)) = oe {
                    c.topology_intersection_node(oc, &bg);
                }
            }
            RootEntry::Tile { value, active: true } => {
                if other_off {
                    *e = RootEntry::Tile { value: *value, active: false };
                } else if let Some(RootEntry::Child(oc)) = oe {
                    *e = RootEntry::Child(Box::new(C::topology_copy(oc, *value)));
                }
            }
            _ => {}
        }
    }
    for k in erase {
        table.remove(&k);
    }
}

/// `Tree::topologyDifference(other)`: deactivate every value of `tree` that is active in
/// `other`.
pub fn topology_difference<C, D>(tree: &mut TreeBase<C>, other: &TreeBase<D>)
where
    C: TopologyOps<D>,
    D: Node,
{
    let bg = *tree.background();
    let table = tree.root_mut().table_mut();
    for (key, oe) in other.root().table() {
        match oe {
            RootEntry::Child(oc) => match table.get_mut(key) {
                Some(RootEntry::Child(c)) => c.topology_difference_node(oc, &bg),
                Some(RootEntry::Tile { value, active: true }) => {
                    let mut child = C::new(*key, *value, true);
                    child.topology_difference_node(oc, &bg);
                    table.insert(*key, RootEntry::Child(Box::new(child)));
                }
                _ => {}
            },
            RootEntry::Tile { active: true, .. } => match table.get_mut(key) {
                Some(RootEntry::Child(_)) => {
                    table.remove(key);
                }
                Some(RootEntry::Tile { active, .. }) => *active = false,
                None => {}
            },
            RootEntry::Tile { .. } => {}
        }
    }
}

/// The topology-copy constructor `Tree(other, background, TopologyCopy())`: a tree with the
/// same topology as `other` whose values all equal `background`.
pub fn topology_copy<C, D>(other: &TreeBase<D>, background: C::Value) -> TreeBase<C>
where
    C: TopologyOps<D>,
    D: Node,
{
    let mut tree = TreeBase::<C>::new(background);
    let table = tree.root_mut().table_mut();
    for (key, oe) in other.root().table() {
        let e = match oe {
            RootEntry::Child(oc) => RootEntry::Child(Box::new(C::topology_copy(oc, background))),
            RootEntry::Tile { active, .. } => RootEntry::Tile { value: background, active: *active },
        };
        table.insert(*key, e);
    }
    tree
}

/// The topology-copy constructor `Tree(other, inactiveValue, activeValue, TopologyCopy())`:
/// active values are set to `active_value`, inactive ones (and the background) to
/// `inactive_value`.
pub fn topology_copy_with_values<C, D>(
    other: &TreeBase<D>,
    inactive_value: C::Value,
    active_value: C::Value,
) -> TreeBase<C>
where
    C: TopologyOps<D>,
    D: Node,
{
    let mut tree = TreeBase::<C>::new(inactive_value);
    let table = tree.root_mut().table_mut();
    for (key, oe) in other.root().table() {
        let e = match oe {
            RootEntry::Child(oc) => RootEntry::Child(Box::new(C::topology_copy2(oc, inactive_value, active_value))),
            RootEntry::Tile { active, .. } => {
                RootEntry::Tile { value: if *active { active_value } else { inactive_value }, active: *active }
            }
        };
        table.insert(*key, e);
    }
    tree
}

/// Change the background of `tree` like OpenVDB's `RootNode::setBackground(bg, true)`:
/// inactive values approximately equal to the old background (or its negation) are
/// replaced by the new background (or its negation).
pub fn reset_background<T: ToolValue, C: NodeOps<T>>(tree: &mut TreeBase<C>, background: T) {
    let old = *tree.background();
    if old.is_approx_equal(&background) && old == background {
        return;
    }
    let neg_old = old.negative();
    for e in tree.root_mut().table_mut().values_mut() {
        match e {
            RootEntry::Child(c) => c.reset_background_cpp(&old, &background),
            RootEntry::Tile { value, active: false } => {
                if value.is_approx_equal(&old) {
                    *value = background;
                } else if value.is_approx_equal(&neg_old) {
                    *value = background.negative();
                }
            }
            _ => {}
        }
    }
    tree.root_mut().set_background_unchecked(background);
}

/// True if the tree has any active tiles (`Tree::hasActiveTiles`).
pub fn has_active_tiles<C: Node>(tree: &TreeBase<C>) -> bool {
    tree.active_tile_count() > 0
}

/// Remove all leaf nodes from the tree (replacing each by an inactive tile of `value`)
/// and return them (`Tree::stealNodes`).
pub fn steal_leaves<T: Value>(tree: &mut Tree<T>, value: T) -> Vec<Box<LeafNode<T>>> {
    let mut out = Vec::new();
    for n2 in tree.root_mut().children_mut() {
        for n1 in n2.children_mut() {
            for i in 0..Internal1::<T>::NUM_VALUES {
                if n1.is_child(i) {
                    out.push(n1.set_tile(i, value, false).unwrap());
                }
            }
        }
    }
    out
}

/// References to all level-1 internal nodes (parents of leaves), in tree order.
pub fn internal1_nodes<T: Value>(tree: &Tree<T>) -> Vec<&Internal1<T>> {
    tree.root().children().flat_map(|n2| n2.children().map(|(_, n1)| n1)).collect()
}

/// Mutable references to all level-1 internal nodes, in tree order.
pub fn internal1_nodes_mut<T: Value>(tree: &mut Tree<T>) -> Vec<&mut Internal1<T>> {
    tree.root_mut().children_mut().flat_map(|n2| n2.children_mut()).collect()
}

/// Mutable references to all level-2 internal nodes (children of the root), in tree order.
pub fn internal2_nodes_mut<T: Value>(tree: &mut Tree<T>) -> Vec<&mut Internal2<T>> {
    tree.root_mut().children_mut().collect()
}

/// Visit every tile of the tree (root and internal) except root-level background tiles, and
/// call `f(bbox, value, active, level)`; equivalent to OpenVDB's `ValueAllIter` restricted to
/// non-leaf depths.
pub fn for_each_tile<T: Value>(tree: &Tree<T>, mut f: impl FnMut(CoordBBox, T, bool, u32)) {
    tree.visit_tiles(&mut |b, v, on, level| f(b, *v, on, level));
}

/// An active value visited by [`visit_values_on_ordered`].
#[derive(Clone, Copy, Debug)]
pub enum ActiveValue<T> {
    /// An active leaf voxel.
    Voxel(Coord, T),
    /// An active tile covering `bbox`.
    Tile(CoordBBox, T),
}

/// Visit every active value (voxels and tiles) in the order of OpenVDB's
/// `Tree::cbeginValueOn()` iterator: depth first, with tiles and child nodes interleaved in
/// slot order at every level.
pub fn visit_values_on_ordered<T: Value>(tree: &Tree<T>, f: &mut dyn FnMut(ActiveValue<T>)) {
    fn visit_internal<C: Node, const L: u32>(
        node: &InternalNode<C, L>,
        f: &mut dyn FnMut(ActiveValue<C::Value>),
        child: &mut dyn FnMut(&C, &mut dyn FnMut(ActiveValue<C::Value>)),
    ) {
        for n in 0..(1usize << (3 * L)) {
            if let Some(c) = node.child(n) {
                child(c, f);
            } else if node.value_mask().is_on(n) {
                let o = node.offset_to_global_coord(n);
                f(ActiveValue::Tile(CoordBBox::create_cube(o, C::DIM), node.values()[n]));
            }
        }
    }
    for (key, e) in tree.root().table() {
        match e {
            RootEntry::Tile { value, active } => {
                if *active {
                    f(ActiveValue::Tile(CoordBBox::create_cube(*key, Internal2::<T>::DIM), *value));
                }
            }
            RootEntry::Child(n2) => {
                visit_internal(n2.as_ref(), f, &mut |n1: &Internal1<T>, f| {
                    visit_internal(n1, f, &mut |leaf: &LeafNode<T>, f| {
                        for (pos, v) in leaf.iter_on() {
                            f(ActiveValue::Voxel(leaf.offset_to_global_coord(pos), *v));
                        }
                    });
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{BoolTree, FloatTree};

    #[test]
    fn merge_policies() {
        let mut a = FloatTree::new(0.0);
        let mut b = FloatTree::new(0.0);
        a.set_value(&Coord::new(0, 0, 0), 1.0);
        b.set_value(&Coord::new(0, 0, 0), 2.0);
        b.set_value(&Coord::new(1, 0, 0), 3.0);
        b.set_value(&Coord::new(1000, 0, 0), 4.0);
        a.merge(&mut b, MergePolicy::ActiveStates);
        assert_eq!(a.get_value(&Coord::new(0, 0, 0)), 1.0);
        assert_eq!(a.get_value(&Coord::new(1, 0, 0)), 3.0);
        assert_eq!(a.get_value(&Coord::new(1000, 0, 0)), 4.0);
        assert_eq!(a.active_voxel_count(), 3);
        assert!(b.is_empty());

        let mut c = FloatTree::new(0.0);
        let mut d = FloatTree::new(0.0);
        c.set_value(&Coord::new(0, 0, 0), 1.0);
        d.set_value(&Coord::new(1, 0, 0), 3.0);
        c.merge(&mut d, MergePolicy::Nodes);
        assert_eq!(c.active_voxel_count(), 1);
    }

    #[test]
    fn combine_and_topology() {
        let mut a = FloatTree::new(1.0);
        let mut b = FloatTree::new(2.0);
        a.set_value(&Coord::new(0, 0, 0), 5.0);
        b.set_value(&Coord::new(0, 0, 1), 7.0);
        combine_extended(&mut a, &mut b, |args| args.result = args.a + args.b, false);
        assert_eq!(*a.background(), 3.0);
        assert_eq!(a.get_value(&Coord::new(0, 0, 0)), 7.0);
        assert_eq!(a.get_value(&Coord::new(0, 0, 1)), 8.0);
        assert_eq!(a.active_voxel_count(), 2);

        let mut m = BoolTree::new(false);
        m.fill(&CoordBBox::from_coords(0, 0, 0, 15, 15, 15), true, true);
        let mut f = FloatTree::new(0.0);
        f.set_value(&Coord::new(-5, 0, 0), 1.0);
        topology_union(&mut f, &m, false);
        assert_eq!(f.active_voxel_count(), 4097);
        topology_intersection(&mut f, &m);
        assert_eq!(f.active_voxel_count(), 4096);
        let mut small = BoolTree::new(false);
        small.set_value(&Coord::new(1, 1, 1), true);
        topology_difference(&mut f, &small);
        assert_eq!(f.active_voxel_count(), 4095);
        let copy: FloatTree = topology_copy(&m, 2.0);
        assert_eq!(copy.active_voxel_count(), 4096);
        assert_eq!(copy.get_value(&Coord::new(3, 3, 3)), 2.0);
    }
}
