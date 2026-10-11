//! Level-set CSG and per-voxel compositing (port of `openvdb/tools/Composite.h` and the
//! CSG merge operators of `openvdb/tools/Merge.h`).
//!
//! * [`csg_union`], [`csg_intersection`], [`csg_difference`] modify the A level set in
//!   place and leave B empty (B's nodes are stolen where possible),
//! * [`csg_union_copy`], [`csg_intersection_copy`], [`csg_difference_copy`] return a new
//!   level set and leave both inputs untouched,
//! * [`comp_max`], [`comp_min`], [`comp_sum`], [`comp_mul`], [`comp_div`] combine every value
//!   of A with the corresponding value of B (B is left empty),
//! * [`comp_replace`] copies the active values of B into A,
//! * [`comp_active_leaf_voxels`] transfers the leaves of a source tree into a destination
//!   tree, combining overlapping active voxels.

use crate::math::{Vec3d, Vec3f, Vec3i};
use crate::tools::prune::level_set_prune;
use crate::tools::signed_flood_fill::signed_flood_fill_min_level;
use crate::tools::tree_ops::{combine_extended, internal1_nodes, merge, topology_union, MergePolicy, NodeOps};
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue, TreeAdapter};
use crate::tree::{Internal2, InternalNode, LeafNode, Node, RootEntry, Tree};
use crate::types::Value;
use std::collections::{BTreeSet, HashMap};

// ---------------------------------------------------------------------------------------
// In-place CSG (Merge.h: CsgUnionOrIntersectionOp, CsgDifferenceOp)
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum CsgMode {
    Union,
    Intersection,
    Difference,
}

/// Background values and options shared by the CSG node operators.
struct CsgCtx<T> {
    mode: CsgMode,
    background: T,
    other_background: T,
    prune_cancelled: bool,
}

impl<T: ToolValue> CsgCtx<T> {
    fn union(&self) -> bool {
        self.mode == CsgMode::Union
    }
    fn inside_background(&self) -> T {
        if self.union() {
            self.background.negative()
        } else {
            self.background
        }
    }
    /// For union an "inside" value is negative; for intersection it is positive.
    fn is_valid(&self, v: &T) -> bool {
        if self.union() {
            v.is_negative()
        } else {
            T::zero().cwise_lt(v)
        }
    }
    fn is_invalid(&self, v: &T) -> bool {
        if self.union() {
            T::zero().cwise_lt(v)
        } else {
            v.is_negative()
        }
    }
}

/// Recursive top-down CSG merge of a node of A with the corresponding node of B.
trait CsgNode<T: ToolValue>: NodeOps<T> {
    fn csg(&mut self, other: &mut Self, ctx: &CsgCtx<T>);
}

impl<T: ToolValue> CsgNode<T> for LeafNode<T> {
    fn csg(&mut self, other: &mut Self, ctx: &CsgCtx<T>) {
        match ctx.mode {
            CsgMode::Union | CsgMode::Intersection => {
                let union = ctx.union();
                let mut all_neg_equal = true;
                for i in 0..LeafNode::<T>::SIZE {
                    let new_value = *other.get_value_at(i);
                    let old_value = *self.get_value_at(i);
                    all_neg_equal &= old_value == new_value.negative();
                    let do_merge = if union { new_value.cwise_lt(&old_value) } else { new_value.cwise_gt(&old_value) };
                    if do_merge {
                        self.set_value_only_at(i, new_value);
                        self.set_active_state_at(i, other.is_value_on_at(i));
                    }
                }
                if ctx.prune_cancelled && all_neg_equal {
                    let v = if union { ctx.background.negative() } else { ctx.background };
                    self.fill_all(v, false);
                }
            }
            CsgMode::Difference => {
                let mut all_equal = true;
                for i in 0..LeafNode::<T>::SIZE {
                    let a = *self.get_value_at(i);
                    let b = *other.get_value_at(i);
                    all_equal &= a == b;
                    let b = b.negative();
                    if a.cwise_lt(&b) {
                        self.set_value_only_at(i, b);
                        self.set_active_state_at(i, other.is_value_on_at(i));
                    }
                }
                if ctx.prune_cancelled && all_equal {
                    self.fill_all(ctx.background, false);
                }
            }
        }
    }
}

impl<T: ToolValue, C: CsgNode<T>, const L: u32> CsgNode<T> for InternalNode<C, L> {
    fn csg(&mut self, other: &mut Self, ctx: &CsgCtx<T>) {
        let n_values = Self::NUM_VALUES;
        let mut recurse = false;
        match ctx.mode {
            CsgMode::Union | CsgMode::Intersection => {
                let inside_bg = ctx.inside_background();
                let outside_bg = inside_bg.negative();
                let mut valid = vec![false; n_values];
                let mut invalid = vec![false; n_values];
                for n in 0..n_values {
                    if !self.is_child(n) {
                        let v = self.values()[n];
                        if ctx.is_valid(&v) {
                            valid[n] = true;
                        } else if ctx.is_invalid(&v) {
                            invalid[n] = true;
                        }
                    }
                }
                for n in 0..n_values {
                    if other.is_child(n) || valid[n] {
                        continue;
                    }
                    if ctx.is_valid(&other.values()[n]) {
                        self.set_tile(n, inside_bg, other.value_mask().is_on(n));
                        valid[n] = true;
                    }
                }
                for n in 0..n_values {
                    if !other.is_child(n) {
                        continue;
                    }
                    if valid[n] {
                        other.set_tile(n, outside_bg, false);
                    } else if invalid[n] {
                        let mut child = other.set_tile(n, ctx.other_background, false).unwrap();
                        child.reset_background_cpp(&ctx.other_background, &ctx.background);
                        self.set_child(n, child);
                        invalid[n] = false;
                    } else {
                        recurse = true;
                    }
                }
            }
            CsgMode::Difference => {
                let inside: Vec<bool> =
                    (0..n_values).map(|n| !self.is_child(n) && self.values()[n].is_negative()).collect();
                for n in 0..n_values {
                    if other.is_child(n) {
                        continue;
                    }
                    if other.values()[n].is_negative() && (inside[n] || self.is_child(n)) {
                        self.set_tile(n, ctx.background, other.value_mask().is_on(n));
                    }
                }
                for n in 0..n_values {
                    if !other.is_child(n) {
                        continue;
                    }
                    if inside[n] {
                        let mut child = other.set_tile(n, ctx.other_background, false).unwrap();
                        child.reset_background_cpp(&ctx.other_background, &ctx.background);
                        child.negate_values();
                        self.set_child(n, child);
                    } else if self.is_child(n) {
                        recurse = true;
                    }
                }
            }
        }
        if recurse {
            for n in 0..n_values {
                if let (Some(c), Some(oc)) = (self.child_mut(n), other.child_mut(n)) {
                    c.csg(oc, ctx);
                }
            }
        }
    }
}

const ACTIVE_TILE: u8 = 0x1;
const INSIDE_TILE: u8 = 0x2;
const OUTSIDE_TILE: u8 = 0x4;
const CHILD: u8 = 0x4;

/// Root-level CSG merge followed by recursion into conflicting branches.
fn csg_in_place<T: ToolValue>(a: &mut Tree<T>, b: &mut Tree<T>, mode: CsgMode, prune_cancelled: bool) {
    let ctx = CsgCtx { mode, background: *a.background(), other_background: *b.background(), prune_cancelled };
    let recurse = match mode {
        CsgMode::Union | CsgMode::Intersection => csg_root_union_intersection(a, b, &ctx),
        CsgMode::Difference => csg_root_difference(a, b, &ctx),
    };
    if recurse {
        let keys: Vec<_> = a.root().table().keys().copied().collect();
        for key in keys {
            let oc = match b.root_mut().table_mut().get_mut(&key) {
                Some(RootEntry::Child(oc)) => oc,
                _ => continue,
            };
            if let Some(RootEntry::Child(c)) = a.root_mut().table_mut().get_mut(&key) {
                c.csg(oc, &ctx);
            }
        }
    }
    b.clear();
}

fn csg_root_union_intersection<T: ToolValue>(a: &mut Tree<T>, b: &mut Tree<T>, ctx: &CsgCtx<T>) -> bool {
    let union = ctx.union();
    let (inside_state, outside_state) = if union { (INSIDE_TILE, OUTSIDE_TILE) } else { (OUTSIDE_TILE, INSIDE_TILE) };
    let inside_bg = ctx.inside_background();
    let outside_bg = inside_bg.negative();
    let tile_flag = |v: &T, on: bool| -> u8 {
        let mut f = 0;
        if v.is_negative() {
            f |= INSIDE_TILE;
        } else if T::zero().cwise_lt(v) {
            f |= OUTSIDE_TILE;
        }
        if on {
            f |= ACTIVE_TILE;
        }
        f
    };
    let root = a.root_mut();
    root.erase_background_tiles();
    if !union {
        let btable = b.root().table();
        let to_delete: Vec<_> = root.table().keys().filter(|k| !btable.contains_key(k)).copied().collect();
        for k in to_delete {
            root.table_mut().remove(&k);
        }
    }
    let mut tiles: HashMap<crate::math::Coord, u8> = HashMap::new();
    for (k, v, on) in root.tiles() {
        tiles.insert(k, tile_flag(v, on));
    }
    for (k, v, on) in b.root().tiles() {
        match tiles.get_mut(&k) {
            None => {
                tiles.insert(k, tile_flag(v, on));
            }
            Some(flag) => {
                if *flag & outside_state != 0 {
                    let nf = tile_flag(v, on);
                    if nf & inside_state != 0 {
                        *flag = nf;
                    }
                }
            }
        }
    }
    let mut sorted: Vec<_> = tiles.iter().map(|(k, f)| (*k, *f)).collect();
    sorted.sort_by_key(|(k, _)| *k);
    for &(k, flag) in &sorted {
        if flag & inside_state != 0 && (union || root.table().contains_key(&k)) {
            root.add_root_tile(&k, inside_bg, flag & ACTIVE_TILE != 0);
        }
    }
    let mut children: BTreeSet<_> = root.table().iter().filter(|(_, e)| e.is_child()).map(|(k, _)| *k).collect();
    let mut recurse = false;
    let bkeys: Vec<_> = b.root().table().iter().filter(|(_, e)| e.is_child()).map(|(k, _)| *k).collect();
    for k in bkeys {
        if !union && !root.table().contains_key(&k) {
            continue;
        }
        if children.contains(&k) {
            recurse = true;
            continue;
        }
        if tiles.get(&k) == Some(&inside_state) {
            continue;
        }
        let entry = b.root_mut().table_mut().insert(k, RootEntry::Tile { value: ctx.other_background, active: false });
        if let Some(RootEntry::Child(mut child)) = entry {
            child.reset_background_cpp(&ctx.other_background, &ctx.background);
            root.add_child(child);
        }
        children.insert(k);
    }
    for &(k, flag) in &sorted {
        if flag & outside_state != 0 && !children.contains(&k) && (union || root.table().contains_key(&k)) {
            root.add_root_tile(&k, outside_bg, flag & ACTIVE_TILE != 0);
        }
    }
    root.erase_background_tiles();
    recurse
}

fn csg_root_difference<T: ToolValue>(a: &mut Tree<T>, b: &mut Tree<T>, ctx: &CsgCtx<T>) -> bool {
    let root = a.root_mut();
    root.erase_background_tiles();
    let mut flags: HashMap<crate::math::Coord, u8> = HashMap::new();
    for (k, e) in root.table() {
        match e {
            RootEntry::Tile { value, active } => {
                if value.is_negative() {
                    flags.insert(*k, INSIDE_TILE | if *active { ACTIVE_TILE } else { 0 });
                }
            }
            RootEntry::Child(_) => {
                flags.insert(*k, CHILD);
            }
        }
    }
    let mut recurse = false;
    let btiles: Vec<_> = b.root().tiles().map(|(k, v, on)| (k, *v, on)).collect();
    for (k, v, on) in btiles {
        if v.is_negative() && flags.contains_key(&k) {
            root.add_root_tile(&k, ctx.background, on);
        }
    }
    let bkeys: Vec<_> = b.root().table().iter().filter(|(_, e)| e.is_child()).map(|(k, _)| *k).collect();
    for k in bkeys {
        if let Some(&flag) = flags.get(&k) {
            if flag & CHILD != 0 {
                recurse = true;
            } else if flag & INSIDE_TILE != 0 {
                let entry =
                    b.root_mut().table_mut().insert(k, RootEntry::Tile { value: ctx.other_background, active: false });
                if let Some(RootEntry::Child(mut child)) = entry {
                    child.reset_background_cpp(&ctx.other_background, &ctx.background);
                    child.negate_values();
                    root.add_child(child);
                }
            }
        }
    }
    root.erase_background_tiles();
    recurse
}

fn validate_level_set<T: ToolValue>(tree: &Tree<T>, name: &str) -> ToolResult<()> {
    let bg = *tree.background();
    if !T::zero().cwise_lt(&bg) {
        return Err(ToolError::Value(format!("expected grid {name} outside value > 0, got {bg:?}")));
    }
    if !bg.negative().is_negative() {
        return Err(ToolError::Value(format!("expected grid {name} inside value < 0, got {:?}", bg.negative())));
    }
    Ok(())
}

fn csg<T: ToolValue, G: TreeAdapter<T>>(
    a: &mut G,
    b: &mut G,
    mode: CsgMode,
    prune: bool,
    prune_cancelled_tiles: bool,
) -> ToolResult<()> {
    validate_level_set(a.tree_ref(), "A")?;
    validate_level_set(b.tree_ref(), "B")?;
    csg_in_place(a.tree_mut_ref(), b.tree_mut_ref(), mode, prune && prune_cancelled_tiles);
    if prune {
        let bg = *a.tree_ref().background();
        level_set_prune(a.tree_mut_ref(), bg, bg.negative(), true);
    }
    Ok(())
}

/// Replace the A level set with the union of A and B (`tools::csgUnion`). B is left empty.
///
/// If `prune` is true, the result is pruned with `pruneLevelSet`; if
/// `prune_cancelled_tiles` is also true, overlapping leaves whose values cancel are
/// collapsed. Returns an error if either background is not positive.
pub fn csg_union<T: ToolValue, G: TreeAdapter<T>>(
    a: &mut G,
    b: &mut G,
    prune: bool,
    prune_cancelled_tiles: bool,
) -> ToolResult<()> {
    csg(a, b, CsgMode::Union, prune, prune_cancelled_tiles)
}

/// Replace the A level set with the intersection of A and B (`tools::csgIntersection`).
/// B is left empty.
pub fn csg_intersection<T: ToolValue, G: TreeAdapter<T>>(
    a: &mut G,
    b: &mut G,
    prune: bool,
    prune_cancelled_tiles: bool,
) -> ToolResult<()> {
    csg(a, b, CsgMode::Intersection, prune, prune_cancelled_tiles)
}

/// Replace the A level set with the difference A − B (`tools::csgDifference`).
/// B is left empty.
pub fn csg_difference<T: ToolValue, G: TreeAdapter<T>>(
    a: &mut G,
    b: &mut G,
    prune: bool,
    prune_cancelled_tiles: bool,
) -> ToolResult<()> {
    csg(a, b, CsgMode::Difference, prune, prune_cancelled_tiles)
}

// ---------------------------------------------------------------------------------------
// CSG copies (Composite.h: BuildPrimarySegment / BuildSecondarySegment)
// ---------------------------------------------------------------------------------------

/// Combine overlapping leaves voxel by voxel.
fn combine_leaves<T: ToolValue>(lhs: &LeafNode<T>, rhs: &LeafNode<T>, mode: CsgMode, out: &mut LeafNode<T>) {
    for pos in 0..LeafNode::<T>::SIZE {
        let l = *lhs.get_value_at(pos);
        let (r, r_on) = (*rhs.get_value_at(pos), rhs.is_value_on_at(pos));
        let (from_rhs, rv) = match mode {
            CsgMode::Intersection => (l.cwise_lt(&r), r),
            CsgMode::Difference => {
                let rv = r.negative();
                (l.cwise_lt(&rv), rv)
            }
            CsgMode::Union => (l.cwise_gt(&r), r),
        };
        if from_rhs {
            out.set_value_only_at(pos, rv);
            out.set_active_state_at(pos, r_on);
        } else {
            out.set_value_only_at(pos, l);
            out.set_active_state_at(pos, lhs.is_value_on_at(pos));
        }
    }
}

fn build_primary<T: ToolValue>(lhs: &Tree<T>, rhs: &Tree<T>, mode: CsgMode) -> Tree<T> {
    let mut out = Tree::new(*lhs.background());
    let rhs_acc = rhs.accessor();
    let keep = |v: T| match mode {
        CsgMode::Intersection => v.is_negative(),
        _ => !v.is_negative(),
    };
    let mut leaf_list: Vec<&LeafNode<T>> = Vec::new();
    for n1 in internal1_nodes(lhs) {
        let ijk = n1.origin();
        let rhs_has =
            rhs.root().probe_child(&ijk).and_then(|n2| n2.child(Internal2::<T>::coord_to_offset(&ijk))).is_some();
        if rhs_has {
            leaf_list.extend(n1.children().map(|(_, l)| l));
        } else if keep(rhs_acc.get_value(&ijk)) {
            for (_, l) in n1.children() {
                out.add_leaf(Box::new(l.clone()));
            }
        }
    }
    for lhs_leaf in leaf_list {
        let ijk = lhs_leaf.origin();
        if let Some(rhs_leaf) = rhs_acc.probe_leaf(&ijk) {
            let o = out.touch_leaf(&ijk);
            combine_leaves(lhs_leaf, rhs_leaf, mode, o);
        } else if keep(rhs_acc.get_value(&ijk)) {
            out.add_leaf(Box::new(lhs_leaf.clone()));
        }
    }
    out
}

fn build_secondary<T: ToolValue>(lhs: &Tree<T>, rhs: &Tree<T>, mode: CsgMode) -> Tree<T> {
    let mut out = Tree::new(*lhs.background());
    let lhs_acc = lhs.accessor();
    let keep = |v: T| match mode {
        CsgMode::Union => !v.is_negative(),
        _ => v.is_negative(),
    };
    let add = |out: &mut Tree<T>, l: &LeafNode<T>| {
        let mut leaf = Box::new(l.clone());
        if mode == CsgMode::Difference {
            leaf.negate_values();
        }
        out.add_leaf(leaf);
    };
    let mut leaf_list: Vec<&LeafNode<T>> = Vec::new();
    for n1 in internal1_nodes(rhs) {
        let ijk = n1.origin();
        let lhs_has =
            lhs.root().probe_child(&ijk).and_then(|n2| n2.child(Internal2::<T>::coord_to_offset(&ijk))).is_some();
        if lhs_has {
            leaf_list.extend(n1.children().map(|(_, l)| l));
        } else if keep(lhs_acc.get_value(&ijk)) {
            for (_, l) in n1.children() {
                add(&mut out, l);
            }
        }
    }
    for rhs_leaf in leaf_list {
        let ijk = rhs_leaf.origin();
        if lhs_acc.probe_leaf(&ijk).is_none() && keep(lhs_acc.get_value(&ijk)) {
            add(&mut out, rhs_leaf);
        }
    }
    out
}

fn csg_copy<T: ToolValue>(lhs: &Tree<T>, rhs: &Tree<T>, mode: CsgMode) -> Tree<T> {
    let (mut primary, mut secondary) =
        rayon::join(|| build_primary(lhs, rhs, mode), || build_secondary(lhs, rhs, mode));
    merge(&mut primary, &mut secondary, MergePolicy::ActiveStates);
    // Leaf signs are set during segment construction; flood-fill the tiles only.
    let _ = signed_flood_fill_min_level(&mut primary, true, 1);
    primary
}

/// Return the union of the A and B level sets as a new level set with A's transform and
/// metadata (`tools::csgUnionCopy`). Neither input is modified.
pub fn csg_union_copy<T: ToolValue, G: TreeAdapter<T, Rebind<T> = G>>(a: &G, b: &G) -> G {
    a.rebind(csg_copy(a.tree_ref(), b.tree_ref(), CsgMode::Union))
}

/// Return the intersection of the A and B level sets as a new level set
/// (`tools::csgIntersectionCopy`).
pub fn csg_intersection_copy<T: ToolValue, G: TreeAdapter<T, Rebind<T> = G>>(a: &G, b: &G) -> G {
    a.rebind(csg_copy(a.tree_ref(), b.tree_ref(), CsgMode::Intersection))
}

/// Return the difference A − B of two level sets as a new level set
/// (`tools::csgDifferenceCopy`).
pub fn csg_difference_copy<T: ToolValue, G: TreeAdapter<T, Rebind<T> = G>>(a: &G, b: &G) -> G {
    a.rebind(csg_copy(a.tree_ref(), b.tree_ref(), CsgMode::Difference))
}

// ---------------------------------------------------------------------------------------
// Per-voxel compositing
// ---------------------------------------------------------------------------------------

/// Value types supported by the `comp*` operations (`composite::min/max/divide` and the
/// arithmetic operators).
pub trait CompositeValue: ToolValue {
    /// `composite::max`: `max` for scalars; for vectors the one with the larger magnitude
    /// (ties broken lexicographically).
    fn comp_max(a: Self, b: Self) -> Self;
    /// `composite::min`, the counterpart of [`comp_max`](Self::comp_max).
    fn comp_min(a: Self, b: Self) -> Self;
    /// `a + b` (logical or for `bool`).
    fn comp_add(a: Self, b: Self) -> Self;
    /// `a * b` (logical and for `bool`).
    fn comp_mul(a: Self, b: Self) -> Self;
    /// `composite::divide`: integer division by zero saturates; `bool` returns `a`.
    fn comp_div(a: Self, b: Self) -> Self;
}

macro_rules! impl_comp_float {
    ($($t:ty),*) => {$(
        impl CompositeValue for $t {
            #[inline] fn comp_max(a: Self, b: Self) -> Self { if a < b { b } else { a } }
            #[inline] fn comp_min(a: Self, b: Self) -> Self { if b < a { b } else { a } }
            #[inline] fn comp_add(a: Self, b: Self) -> Self { a + b }
            #[inline] fn comp_mul(a: Self, b: Self) -> Self { a * b }
            #[inline] fn comp_div(a: Self, b: Self) -> Self { a / b }
        }
    )*};
}
impl_comp_float!(f32, f64);

macro_rules! impl_comp_int {
    ($($t:ty),*) => {$(
        impl CompositeValue for $t {
            #[inline] fn comp_max(a: Self, b: Self) -> Self { if a < b { b } else { a } }
            #[inline] fn comp_min(a: Self, b: Self) -> Self { if b < a { b } else { a } }
            #[inline] fn comp_add(a: Self, b: Self) -> Self { a.wrapping_add(b) }
            #[inline] fn comp_mul(a: Self, b: Self) -> Self { a.wrapping_mul(b) }
            #[inline]
            fn comp_div(a: Self, b: Self) -> Self {
                if b != 0 {
                    return a.wrapping_div(b);
                }
                if a == 0 {
                    return 0;
                }
                if a > 0 { <$t>::MAX } else { <$t>::MAX.wrapping_neg() }
            }
        }
    )*};
}
impl_comp_int!(i32, i64);

impl CompositeValue for bool {
    fn comp_max(a: Self, b: Self) -> Self {
        a | b
    }
    fn comp_min(a: Self, b: Self) -> Self {
        a & b
    }
    fn comp_add(a: Self, b: Self) -> Self {
        a | b
    }
    fn comp_mul(a: Self, b: Self) -> Self {
        a && b
    }
    fn comp_div(a: Self, _b: Self) -> Self {
        a
    }
}

macro_rules! impl_comp_vec {
    ($($t:ty => $len2:expr),*) => {$(
        impl CompositeValue for $t {
            fn comp_max(a: Self, b: Self) -> Self {
                let (am, bm) = ($len2(&a), $len2(&b));
                if am < bm { b } else if bm < am { a } else if a.cwise_gt(&b) { a } else { b }
            }
            fn comp_min(a: Self, b: Self) -> Self {
                let (am, bm) = ($len2(&a), $len2(&b));
                if am < bm { a } else if bm < am { b } else if a.cwise_lt(&b) { a } else { b }
            }
            fn comp_add(a: Self, b: Self) -> Self { a + b }
            fn comp_mul(a: Self, b: Self) -> Self { a * b }
            fn comp_div(a: Self, b: Self) -> Self { a / b }
        }
    )*};
}
impl_comp_vec!(Vec3f => |v: &Vec3f| v.length_sqr(), Vec3d => |v: &Vec3d| v.length_sqr());

impl CompositeValue for Vec3i {
    fn comp_max(a: Self, b: Self) -> Self {
        let (am, bm) = (a.length_sqr(), b.length_sqr());
        if am < bm {
            b
        } else if bm < am {
            a
        } else if a.cwise_gt(&b) {
            a
        } else {
            b
        }
    }
    fn comp_min(a: Self, b: Self) -> Self {
        let (am, bm) = (a.length_sqr(), b.length_sqr());
        if am < bm {
            a
        } else if bm < am {
            b
        } else if a.cwise_lt(&b) {
            a
        } else {
            b
        }
    }
    fn comp_add(a: Self, b: Self) -> Self {
        a.zip_map(b, |x, y| x.wrapping_add(y))
    }
    fn comp_mul(a: Self, b: Self) -> Self {
        a.zip_map(b, |x, y| x.wrapping_mul(y))
    }
    fn comp_div(a: Self, b: Self) -> Self {
        a.zip_map(b, i32::comp_div)
    }
}

fn comp_op<T: CompositeValue, G: TreeAdapter<T>>(a: &mut G, b: &mut G, f: fn(T, T) -> T) {
    combine_extended(a.tree_mut_ref(), b.tree_mut_ref(), |args| args.result = f(args.a, args.b), false);
}

/// Set A to the per-voxel maximum of A and B (`tools::compMax`). B is left empty.
/// The result is active wherever either input was.
pub fn comp_max<T: CompositeValue, G: TreeAdapter<T>>(a: &mut G, b: &mut G) {
    comp_op(a, b, T::comp_max)
}

/// Set A to the per-voxel minimum of A and B (`tools::compMin`). B is left empty.
pub fn comp_min<T: CompositeValue, G: TreeAdapter<T>>(a: &mut G, b: &mut G) {
    comp_op(a, b, T::comp_min)
}

/// Set A to the per-voxel sum A + B (`tools::compSum`). B is left empty.
pub fn comp_sum<T: CompositeValue, G: TreeAdapter<T>>(a: &mut G, b: &mut G) {
    comp_op(a, b, T::comp_add)
}

/// Set A to the per-voxel product A · B (`tools::compMul`). B is left empty.
pub fn comp_mul<T: CompositeValue, G: TreeAdapter<T>>(a: &mut G, b: &mut G) {
    comp_op(a, b, T::comp_mul)
}

/// Set A to the per-voxel quotient A / B (`tools::compDiv`). B is left empty.
pub fn comp_div<T: CompositeValue, G: TreeAdapter<T>>(a: &mut G, b: &mut G) {
    comp_op(a, b, T::comp_div)
}

/// Copy the active values of B into A (`tools::compReplace`): A's active topology becomes
/// the union of both, and A takes B's value wherever B is active.
pub fn comp_replace<T: ToolValue, G: TreeAdapter<T>>(a: &mut G, b: &G) {
    let at = a.tree_mut_ref();
    let bt = b.tree_ref();
    topology_union(at, bt, false);
    let mut tiles = Vec::new();
    bt.visit_tiles(&mut |bbox, v, on, _| {
        if on {
            tiles.push((bbox, *v));
        }
    });
    for (bbox, v) in tiles {
        at.fill(&bbox, v, true);
    }
    let mut acc = at.accessor_mut();
    for (c, v) in bt.active_voxels() {
        acc.set_value(&c, v);
    }
}

/// Transfer the leaves of `src` into `dst` (`tools::compActiveLeafVoxels`): leaves without
/// a counterpart in `dst` are moved as they are; for overlapping leaves the active masks
/// are unioned and `op(dst_value, src_value)` is applied to every active source voxel.
/// `src` is left empty. Tiles are ignored.
pub fn comp_active_leaf_voxels<T: Value, F: Fn(&mut T, &T) + Sync>(src: &mut Tree<T>, dst: &mut Tree<T>, op: F) {
    use rayon::prelude::*;
    let bg = *src.background();
    let leaves = crate::tools::tree_ops::steal_leaves(src, bg);
    src.clear();
    let mut overlapping = Vec::new();
    for leaf in leaves {
        if dst.probe_leaf(&leaf.origin()).is_some() {
            overlapping.push(leaf);
        } else {
            dst.add_leaf(leaf);
        }
    }
    let mut pairs: Vec<(&mut LeafNode<T>, Box<LeafNode<T>>)> = Vec::new();
    {
        let mut by_origin: HashMap<crate::math::Coord, Box<LeafNode<T>>> =
            overlapping.into_iter().map(|l| (l.origin(), l)).collect();
        for d in dst.leaves_mut() {
            if let Some(s) = by_origin.remove(&d.origin()) {
                pairs.push((d, s));
            }
        }
    }
    pairs.par_iter_mut().for_each(|(d, s)| {
        d.value_mask_mut().union_with(s.value_mask());
        let buf = d.buffer_mut();
        for (n, v) in s.iter_on() {
            op(&mut buf[n], v);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;
    use crate::tools::level_set_sphere::create_level_set_sphere;
    use crate::tree::FloatTree;

    #[test]
    fn csg_spheres() {
        let a = create_level_set_sphere::<f32>(1.0, Vec3f::new(0.0, 0.0, 0.0), 0.05, 3.0, true).unwrap();
        let b = create_level_set_sphere::<f32>(1.0, Vec3f::new(0.5, 0.0, 0.0), 0.05, 3.0, true).unwrap();
        for (mode, f) in [
            (0, csg_union_copy::<f32, crate::grid::Grid<f32>> as fn(&_, &_) -> _),
            (1, csg_intersection_copy),
            (2, csg_difference_copy),
        ] {
            let copy = f(&a, &b);
            let mut ia = a.clone();
            let mut ib = b.clone();
            match mode {
                0 => csg_union(&mut ia, &mut ib, true, false).unwrap(),
                1 => csg_intersection(&mut ia, &mut ib, true, false).unwrap(),
                _ => csg_difference(&mut ia, &mut ib, true, false).unwrap(),
            }
            assert!(ib.empty());
            assert_eq!(ia.active_voxel_count(), copy.active_voxel_count(), "mode {mode}");
            for c in [Coord::new(0, 0, 0), Coord::new(10, 0, 0), Coord::new(-19, 0, 0), Coord::new(25, 3, 0)] {
                assert_eq!(ia.get_value(&c), copy.get_value(&c), "mode {mode} at {c:?}");
            }
        }
    }

    #[test]
    fn comp_ops() {
        let mut a = FloatTree::new(0.0);
        let mut b = FloatTree::new(0.0);
        a.set_value(&Coord::new(0, 0, 0), 2.0);
        b.set_value(&Coord::new(0, 0, 0), 3.0);
        b.set_value(&Coord::new(5, 0, 0), -1.0);
        let (mut a2, mut b2) = (a.clone(), b.clone());
        comp_max(&mut a, &mut b);
        assert_eq!(a.get_value(&Coord::new(0, 0, 0)), 3.0);
        assert_eq!(a.get_value(&Coord::new(5, 0, 0)), 0.0);
        assert!(a.is_value_on(&Coord::new(5, 0, 0)));
        comp_mul(&mut a2, &mut b2);
        assert_eq!(a2.get_value(&Coord::new(0, 0, 0)), 6.0);
        assert_eq!(i32::comp_div(5, 0), i32::MAX);
        assert_eq!(i32::comp_div(-5, 0), -i32::MAX);
        let v = Vec3f::comp_max(Vec3f::new(1.0, 0.0, 0.0), Vec3f::new(0.0, 2.0, 0.0));
        assert_eq!(v, Vec3f::new(0.0, 2.0, 0.0));

        let mut c = FloatTree::new(0.0);
        let mut d = FloatTree::new(0.0);
        c.set_value(&Coord::new(1, 1, 1), 1.0);
        d.set_value(&Coord::new(1, 1, 2), 4.0);
        d.fill(&crate::math::CoordBBox::from_coords(0, 0, 128, 127, 127, 255), 7.0, true);
        comp_replace(&mut c, &d);
        assert_eq!(c.get_value(&Coord::new(1, 1, 2)), 4.0);
        assert_eq!(c.get_value(&Coord::new(1, 1, 200)), 7.0);
        assert_eq!(c.active_voxel_count(), 2 + 128 * 128 * 128);

        let mut s = FloatTree::new(0.0);
        let mut t = FloatTree::new(0.0);
        s.set_value(&Coord::new(0, 0, 0), 1.0);
        s.set_value(&Coord::new(100, 0, 0), 2.0);
        t.set_value(&Coord::new(0, 0, 0), 5.0);
        t.set_value(&Coord::new(0, 0, 1), 6.0);
        comp_active_leaf_voxels(&mut s, &mut t, |d, s| *d += *s);
        assert_eq!(t.get_value(&Coord::new(0, 0, 0)), 6.0);
        assert_eq!(t.get_value(&Coord::new(100, 0, 0)), 2.0);
        assert_eq!(t.active_voxel_count(), 3);
        assert!(s.is_empty());
    }
}
