//! Efficient replacement of the background value of a tree (port of
//! `openvdb/tools/ChangeBackground.h`).

use crate::tools::tree_ops::{internal1_nodes_mut, internal2_nodes_mut};
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::{InternalNode, Node, RootEntry, Tree};
use rayon::prelude::*;

/// Apply `set` to every inactive voxel of every leaf and to the tiles of every internal
/// node selected by `tile_filter(node, slot)`, and to the root tiles selected by
/// `root_filter(active)`.
fn apply_to_values<T: ToolValue>(
    tree: &mut Tree<T>,
    threaded: bool,
    set: impl Fn(&mut T) + Sync,
    include_active_tiles: bool,
) {
    for e in tree.root_mut().table_mut().values_mut() {
        if let RootEntry::Tile { value, active: false } = e {
            set(value);
        }
    }
    fn fix_node<T: ToolValue, C: Node<Value = T>, const L: u32>(
        node: &mut InternalNode<C, L>,
        set: &(impl Fn(&mut T) + Sync),
        include_active: bool,
    ) {
        for n in 0..InternalNode::<C, L>::NUM_VALUES {
            if node.is_child(n) || (!include_active && node.value_mask().is_on(n)) {
                continue;
            }
            set(&mut node.values_mut()[n]);
        }
    }
    {
        let mut n2 = internal2_nodes_mut(tree);
        if threaded {
            n2.par_iter_mut().for_each(|n| fix_node(*n, &set, include_active_tiles));
        } else {
            n2.iter_mut().for_each(|n| fix_node(*n, &set, include_active_tiles));
        }
    }
    {
        let mut n1 = internal1_nodes_mut(tree);
        if threaded {
            n1.par_iter_mut().for_each(|n| fix_node(*n, &set, include_active_tiles));
        } else {
            n1.iter_mut().for_each(|n| fix_node(*n, &set, include_active_tiles));
        }
    }
    let leaf_op = |leaf: &mut crate::tree::LeafNode<T>| {
        let (buf, mask) = leaf.buffer_and_mask_mut();
        for n in mask.iter_off() {
            set(&mut buf[n]);
        }
    };
    if threaded {
        tree.par_for_each_leaf_mut(leaf_op);
    } else {
        tree.leaves_mut().for_each(leaf_op);
    }
}

/// Replace the background value of `tree`, along with every inactive value that is
/// approximately equal to the old background (and their negations), with `background`
/// (`tools::changeBackground`).
pub fn change_background<T: ToolValue>(tree: &mut Tree<T>, background: T, threaded: bool) {
    let old = *tree.background();
    let neg_old = old.negative();
    let neg_new = background.negative();
    apply_to_values(
        tree,
        threaded,
        |v| {
            if v.is_approx_equal(&old) {
                *v = background;
            } else if v.is_approx_equal(&neg_old) {
                *v = neg_new;
            }
        },
        false,
    );
    tree.root_mut().set_background_unchecked(background);
}

/// Replace the background of a narrow-band level set by `half_width` (outside) and
/// `-half_width` (inside): every inactive value (and every tile) becomes `±half_width`
/// according to its sign (`tools::changeLevelSetBackground`).
pub fn change_level_set_background<T: ToolValue>(tree: &mut Tree<T>, half_width: T, threaded: bool) -> ToolResult<()> {
    change_asymmetric_level_set_background(tree, half_width, half_width.negative(), threaded)
}

/// Like [`change_level_set_background`] with distinct outside (non-negative) and inside
/// (negative) values (`tools::changeAsymmetricLevelSetBackground`).
pub fn change_asymmetric_level_set_background<T: ToolValue>(
    tree: &mut Tree<T>,
    outside: T,
    inside: T,
    threaded: bool,
) -> ToolResult<()> {
    if outside.is_negative() {
        return Err(ToolError::Value("ChangeLevelSetBackgroundOp: the outside value cannot be negative!".into()));
    }
    if !inside.is_negative() {
        return Err(ToolError::Value("ChangeLevelSetBackgroundOp: the inside value must be negative!".into()));
    }
    set_level_set_background(tree, outside, inside, threaded);
    Ok(())
}

/// [`change_asymmetric_level_set_background`] without the sign checks on `outside` and
/// `inside`, for callers that have already validated them (or deliberately shrink a band).
pub(crate) fn set_level_set_background<T: ToolValue>(tree: &mut Tree<T>, outside: T, inside: T, threaded: bool) {
    apply_to_values(tree, threaded, |v| *v = if v.is_negative() { inside } else { outside }, true);
    tree.root_mut().set_background_unchecked(outside);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, CoordBBox};
    use crate::tree::FloatTree;

    #[test]
    fn change_backgrounds() {
        let mut tree = FloatTree::new(1.0);
        tree.set_value_off(&Coord::new(0, 0, 0), -1.0);
        tree.set_value_on(&Coord::new(0, 0, 1), 1.0);
        tree.fill(&CoordBBox::from_coords(128, 0, 0, 255, 127, 127), -1.0, false);
        change_background(&mut tree, 5.0, true);
        assert_eq!(*tree.background(), 5.0);
        assert_eq!(tree.get_value(&Coord::new(0, 0, 0)), -5.0);
        assert_eq!(tree.get_value(&Coord::new(0, 0, 1)), 1.0);
        assert_eq!(tree.get_value(&Coord::new(0, 0, 2)), 5.0);
        assert_eq!(tree.get_value(&Coord::new(130, 0, 0)), -5.0);
        assert_eq!(tree.get_value(&Coord::new(-1000, 0, 0)), 5.0);

        change_level_set_background(&mut tree, 2.0, false).unwrap();
        assert_eq!(tree.get_value(&Coord::new(0, 0, 0)), -2.0);
        assert_eq!(tree.get_value(&Coord::new(0, 0, 1)), 1.0);
        assert_eq!(tree.get_value(&Coord::new(130, 0, 0)), -2.0);
        assert!(change_asymmetric_level_set_background(&mut tree, 1.0, 1.0, false).is_err());
    }
}
