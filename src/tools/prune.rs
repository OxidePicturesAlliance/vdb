//! Tree pruning (port of `openvdb/tools/Prune.h`).
//!
//! * [`prune`] replaces nodes whose values are all within a tolerance (and whose active
//!   states agree) with tiles holding the node's *median* value,
//! * [`prune_tiles`] does the same but leaves leaf nodes untouched,
//! * [`prune_inactive`] / [`prune_inactive_with_value`] replace nodes without active values
//!   by inactive tiles,
//! * [`prune_level_set`] replaces inactive nodes of a narrow-band level set by inside or
//!   outside tiles depending on the sign of their first value.

use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::{Internal1, Internal2, InternalNode, LeafNode, Node, RootEntry, Tree};
use rayon::prelude::*;

/// Node-level helpers for tolerance pruning.
trait PruneNode<T: ToolValue>: Node<Value = T> {
    /// The constness test of `TolerancePruneOp`: returns the median value and the shared
    /// active state if the node is constant within `tol`.
    fn constant_median(&self, tol: &T) -> Option<(T, bool)>;
}

/// `isConstant(minValue, maxValue, state, tolerance)` over a slice of values.
fn min_max_constant<T: ToolValue>(values: &[T], tol: &T) -> bool {
    let mut min = values[0];
    let mut max = values[0];
    for v in &values[1..] {
        if v.cwise_lt(&min) {
            if max.difference(*v).cwise_gt(tol) {
                return false;
            }
            min = *v;
        } else if v.cwise_gt(&max) {
            if v.difference(min).cwise_gt(tol) {
                return false;
            }
            max = *v;
        }
    }
    true
}

impl<T: ToolValue> PruneNode<T> for LeafNode<T> {
    fn constant_median(&self, tol: &T) -> Option<(T, bool)> {
        let state = self.value_mask().is_constant()?;
        if T::IS_BOOL {
            let first = self.buffer()[0];
            return self.buffer().iter().all(|v| v.approx_eq(&first, tol)).then_some((first, state));
        }
        if !min_max_constant(self.buffer(), tol) {
            return None;
        }
        let mut tmp = self.buffer().to_vec();
        Some((T::median(&mut tmp), state))
    }
}

impl<T: ToolValue, C: Node<Value = T>, const L: u32> PruneNode<T> for InternalNode<C, L> {
    fn constant_median(&self, tol: &T) -> Option<(T, bool)> {
        if !self.child_mask().is_all_off() {
            return None;
        }
        let state = self.value_mask().is_constant()?;
        if T::IS_BOOL {
            let first = self.values()[0];
            return self.values().iter().all(|v| v.approx_eq(&first, tol)).then_some((first, state));
        }
        if !min_max_constant(self.values(), tol) {
            return None;
        }
        let mut tmp = self.values().to_vec();
        Some((T::median(&mut tmp), state))
    }
}

/// Replace each child of `node` for which `test` returns a tile by that tile.
fn prune_children<C: Node, const L: u32>(node: &mut InternalNode<C, L>, test: impl Fn(&C) -> Option<(C::Value, bool)>) {
    for n in 0..InternalNode::<C, L>::NUM_VALUES {
        if let Some(c) = node.child(n) {
            if let Some((v, s)) = test(c) {
                node.set_tile(n, v, s);
            }
        }
    }
}

/// Apply the pruning test bottom-up: level-1 nodes (optionally), level-2 nodes, then the
/// root, finally erasing inactive background tiles from the root.
fn prune_bottom_up<T: ToolValue>(
    tree: &mut Tree<T>,
    include_leaves: bool,
    threaded: bool,
    leaf_test: impl Fn(&LeafNode<T>) -> Option<(T, bool)> + Sync,
    n1_test: impl Fn(&Internal1<T>) -> Option<(T, bool)> + Sync,
    n2_test: impl Fn(&Internal2<T>) -> Option<(T, bool)> + Sync,
) {
    if include_leaves {
        let mut nodes = crate::tools::tree_ops::internal1_nodes_mut(tree);
        if threaded {
            nodes.par_iter_mut().for_each(|n| prune_children(n, &leaf_test));
        } else {
            nodes.iter_mut().for_each(|n| prune_children(n, &leaf_test));
        }
    }
    {
        let mut nodes = crate::tools::tree_ops::internal2_nodes_mut(tree);
        if threaded {
            nodes.par_iter_mut().for_each(|n| prune_children(n, &n1_test));
        } else {
            nodes.iter_mut().for_each(|n| prune_children(n, &n1_test));
        }
    }
    let root = tree.root_mut();
    for e in root.table_mut().values_mut() {
        if let RootEntry::Child(c) = e {
            if let Some((value, active)) = n2_test(c) {
                *e = RootEntry::Tile { value, active };
            }
        }
    }
    root.erase_background_tiles();
}

/// Reduce the memory footprint of `tree` by replacing nodes whose values are all within
/// `tolerance` of each other (and whose active states agree) with tiles holding the
/// node's median value (`tools::prune`).
pub fn prune<T: ToolValue>(tree: &mut Tree<T>, tolerance: T, threaded: bool) {
    prune_bottom_up(
        tree,
        true,
        threaded,
        |l| l.constant_median(&tolerance),
        |n| n.constant_median(&tolerance),
        |n| n.constant_median(&tolerance),
    );
}

/// Like [`prune`] but never prunes leaf nodes, only internal nodes that consist entirely
/// of tiles (`tools::pruneTiles`).
pub fn prune_tiles<T: ToolValue>(tree: &mut Tree<T>, tolerance: T, threaded: bool) {
    prune_bottom_up(
        tree,
        false,
        threaded,
        |_| None,
        |n| n.constant_median(&tolerance),
        |n| n.constant_median(&tolerance),
    );
}

/// Replace nodes that contain no active values by inactive tiles of the background value
/// (`tools::pruneInactive`).
pub fn prune_inactive<T: ToolValue>(tree: &mut Tree<T>, threaded: bool) {
    let bg = *tree.background();
    prune_inactive_with_value(tree, bg, threaded);
}

/// Replace nodes that contain no active values by inactive tiles of `value`
/// (`tools::pruneInactiveWithValue`).
pub fn prune_inactive_with_value<T: ToolValue>(tree: &mut Tree<T>, value: T, threaded: bool) {
    prune_bottom_up(
        tree,
        true,
        threaded,
        |l| l.is_inactive().then_some((value, false)),
        |n| n.is_inactive().then_some((value, false)),
        |n| n.is_inactive().then_some((value, false)),
    );
}

/// Replace inactive nodes of a narrow-band level set by inactive tiles whose value is
/// `-background` if the node's first value is negative and `background` otherwise
/// (`tools::pruneLevelSet`).
pub fn prune_level_set<T: ToolValue>(tree: &mut Tree<T>, threaded: bool) -> ToolResult<()> {
    let outside = *tree.background();
    if outside.is_negative() {
        return Err(ToolError::Value("LevelSetPruneOp: the background value cannot be negative!".into()));
    }
    level_set_prune(tree, outside, outside.negative(), threaded);
    Ok(())
}

/// [`prune_level_set`] with explicit outside (non-negative) and inside (negative) tile values.
pub fn prune_level_set_with_values<T: ToolValue>(
    tree: &mut Tree<T>,
    outside: T,
    inside: T,
    threaded: bool,
) -> ToolResult<()> {
    if outside.is_negative() {
        return Err(ToolError::Value("LevelSetPruneOp: the outside value cannot be negative!".into()));
    }
    if !inside.is_negative() {
        return Err(ToolError::Value("LevelSetPruneOp: the inside value must be negative!".into()));
    }
    level_set_prune(tree, outside, inside, threaded);
    Ok(())
}

pub(crate) fn level_set_prune<T: ToolValue>(tree: &mut Tree<T>, outside: T, inside: T, threaded: bool) {
    use crate::tools::tree_ops::NodeOps;
    let tile = |first: T| if first.is_negative() { inside } else { outside };
    prune_bottom_up(
        tree,
        true,
        threaded,
        |l| l.is_inactive().then(|| (tile(l.first_value()), false)),
        |n| n.is_inactive().then(|| (tile(n.first_value()), false)),
        |n| n.is_inactive().then(|| (tile(n.first_value()), false)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, CoordBBox};
    use crate::tree::FloatTree;

    #[test]
    fn prune_uses_median() {
        let mut tree = FloatTree::new(0.0);
        let bbox = CoordBBox::from_coords(0, 0, 0, 7, 7, 7);
        for (i, c) in bbox.iter().enumerate() {
            tree.set_value(&c, 1.0 + (i % 3) as f32 * 0.01);
        }
        prune(&mut tree, 0.05, true);
        assert_eq!(tree.leaf_count(), 0);
        assert_eq!(tree.active_voxel_count(), 512);
        // 171 values of 1.0, 171 of 1.01 and 170 of 1.02: the median (index 255) is 1.01.
        assert_eq!(tree.get_value(&Coord::new(3, 3, 3)), 1.01);
    }

    #[test]
    fn prune_inactive_and_level_set() {
        let mut tree = FloatTree::new(2.0);
        tree.set_value_off(&Coord::new(0, 0, 0), -2.0);
        tree.set_value_off(&Coord::new(100, 0, 0), 2.0);
        tree.set_value_on(&Coord::new(200, 0, 0), 1.0);
        let mut ls = tree.clone();
        prune_inactive(&mut tree, false);
        assert_eq!(tree.leaf_count(), 1);
        prune_level_set(&mut ls, true).unwrap();
        assert_eq!(ls.leaf_count(), 1);
        assert_eq!(ls.get_value(&Coord::new(0, 0, 0)), -2.0);
        assert_eq!(ls.get_value(&Coord::new(1, 1, 1)), -2.0);
        assert!(prune_level_set_with_values(&mut ls, 1.0, 1.0, true).is_err());
    }
}
