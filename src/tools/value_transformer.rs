//! Apply functors to the values of a tree (port of `openvdb/tools/ValueTransformer.h`).
//!
//! * [`foreach`] modifies values (and active states) in place,
//! * [`transform_values`] populates another tree from the values of an input tree,
//! * [`accumulate`] reduces over values with a user-supplied join,
//! * [`set_value_on_min`], [`set_value_on_max`], [`set_value_on_sum`] and
//!   [`set_value_on_mult`] combine a value into a single voxel and activate it.

use crate::math::{Coord, CoordBBox};
use crate::tools::composite::CompositeValue;
use crate::tools::statistics::ValueIterKind;
use crate::tools::tree_ops::{merge, MergePolicy};
use crate::tools::value_traits::ToolValue;
use crate::tree::{Internal2, LeafNode, Node, Tree, ValueAccessorMut};
use crate::types::Value;
use rayon::prelude::*;

/// A mutable view of one tree value (a voxel or a tile) visited by [`foreach`].
pub struct ValueRef<'a, T> {
    /// Region covered by the value (a single voxel for leaf values).
    pub bbox: CoordBBox,
    /// 0 for voxels, the node level for tiles.
    pub level: u32,
    /// The value, which may be modified.
    pub value: &'a mut T,
    /// The active state, which may be modified.
    pub active: &'a mut bool,
}

impl<T> ValueRef<'_, T> {
    /// True for a voxel stored in a leaf node (`isVoxelValue`).
    pub fn is_voxel(&self) -> bool {
        self.level == 0
    }
    /// The coordinate of the voxel, or the origin of the tile.
    pub fn coord(&self) -> Coord {
        self.bbox.min
    }
    /// Number of voxels covered by the value.
    pub fn voxel_count(&self) -> u64 {
        self.bbox.volume()
    }
}

/// An immutable view of one tree value visited by [`transform_values`] and [`accumulate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueItem<T> {
    pub bbox: CoordBBox,
    pub level: u32,
    pub value: T,
    pub active: bool,
}

impl<T> ValueItem<T> {
    /// True for a voxel stored in a leaf node.
    pub fn is_voxel(&self) -> bool {
        self.level == 0
    }
    /// The coordinate of the voxel, or the origin of the tile.
    pub fn coord(&self) -> Coord {
        self.bbox.min
    }
}

/// Apply `op` to every value selected by `kind` (`tools::foreach`). Tiles are visited
/// first, serially; leaf voxels are visited in parallel if `threaded`.
pub fn foreach<T: Value, F: Fn(&mut ValueRef<'_, T>) + Sync>(
    tree: &mut Tree<T>,
    kind: ValueIterKind,
    threaded: bool,
    op: F,
) {
    tree.visit_tiles_mut(&mut |bbox, v, on, level| {
        if kind.accepts(*on) {
            op(&mut ValueRef { bbox, level, value: v, active: on });
        }
    });
    let leaf_op = |leaf: &mut LeafNode<T>| {
        let origin = leaf.origin();
        let (buf, mask) = leaf.buffer_and_mask_mut();
        for (n, value) in buf.iter_mut().enumerate() {
            let mut on = mask.is_on(n);
            if !kind.accepts(on) {
                continue;
            }
            let c = origin + LeafNode::<T>::offset_to_local_coord(n);
            op(&mut ValueRef { bbox: CoordBBox::new(c, c), level: 0, value, active: &mut on });
            mask.set(n, on);
        }
    };
    if threaded {
        tree.par_for_each_leaf_mut(leaf_op);
    } else {
        tree.leaves_mut().for_each(leaf_op);
    }
}

/// Collect the tile items selected by `kind`.
fn tile_items<T: Value>(tree: &Tree<T>, kind: ValueIterKind) -> Vec<ValueItem<T>> {
    let mut items = Vec::new();
    tree.visit_tiles(&mut |bbox, v, on, level| {
        if kind.accepts(on) {
            items.push(ValueItem { bbox, level, value: *v, active: on });
        }
    });
    items
}

fn leaf_items<T: Value>(leaf: &LeafNode<T>, kind: ValueIterKind) -> impl Iterator<Item = ValueItem<T>> + '_ {
    leaf.iter_all().filter(move |(_, _, on)| kind.accepts(*on)).map(move |(n, v, on)| {
        let c = leaf.offset_to_global_coord(n);
        ValueItem { bbox: CoordBBox::new(c, c), level: 0, value: *v, active: on }
    })
}

/// Populate `output` by applying `op(item, output_accessor)` to every value of `input`
/// selected by `kind` (`tools::transformValues`). In threaded mode, each thread writes into
/// its own tree (with a zero background), and the partial results are merged into
/// `output` with the given policy.
pub fn transform_values<T, U, F>(
    input: &Tree<T>,
    kind: ValueIterKind,
    output: &mut Tree<U>,
    op: F,
    threaded: bool,
    policy: MergePolicy,
) where
    T: Value,
    U: ToolValue,
    F: Fn(&ValueItem<T>, &mut ValueAccessorMut<'_, Internal2<U>>) + Sync,
{
    {
        let mut acc = output.accessor_mut();
        for item in tile_items(input, kind) {
            op(&item, &mut acc);
        }
    }
    let leaves = input.leaf_refs();
    if !threaded {
        let mut acc = output.accessor_mut();
        for leaf in leaves {
            for item in leaf_items(leaf, kind) {
                op(&item, &mut acc);
            }
        }
        return;
    }
    let chunk = (leaves.len() / (4 * rayon::current_num_threads()).max(1)).max(1);
    let parts: Vec<Tree<U>> = leaves
        .par_chunks(chunk)
        .map(|ls| {
            let mut t = Tree::new(U::zero());
            {
                let mut acc = t.accessor_mut();
                for leaf in ls {
                    for item in leaf_items(leaf, kind) {
                        op(&item, &mut acc);
                    }
                }
            }
            t
        })
        .collect();
    for mut p in parts {
        merge(output, &mut p, policy);
    }
}

/// Reduce over the values of `tree` selected by `kind` (`tools::accumulate`): each worker
/// starts from a clone of `init`, calls `op(acc, item)` for its values, and partial results
/// are combined with `join(acc, other)`.
pub fn accumulate<T, A, F, J>(tree: &Tree<T>, kind: ValueIterKind, threaded: bool, init: A, op: F, join: J) -> A
where
    T: Value,
    A: Clone + Send + Sync,
    F: Fn(&mut A, &ValueItem<T>) + Sync,
    J: Fn(&mut A, A),
{
    let mut acc = init.clone();
    for item in tile_items(tree, kind) {
        op(&mut acc, &item);
    }
    let leaves = tree.leaf_refs();
    let leaf_op = |leaf: &&LeafNode<T>| {
        let mut a = init.clone();
        for item in leaf_items(leaf, kind) {
            op(&mut a, &item);
        }
        a
    };
    let parts: Vec<A> =
        if threaded { leaves.par_iter().map(leaf_op).collect() } else { leaves.iter().map(leaf_op).collect() };
    for p in parts {
        join(&mut acc, p);
    }
    acc
}

/// Set the voxel at `xyz` to the minimum of its value and `value`, and activate it
/// (`tools::setValueOnMin`).
pub fn set_value_on_min<T: ToolValue>(tree: &mut Tree<T>, xyz: &Coord, value: T) {
    tree.modify_value(xyz, |v| {
        if value.cwise_lt(v) {
            *v = value
        }
    });
}

/// Set the voxel at `xyz` to the maximum of its value and `value`, and activate it
/// (`tools::setValueOnMax`).
pub fn set_value_on_max<T: ToolValue>(tree: &mut Tree<T>, xyz: &Coord, value: T) {
    tree.modify_value(xyz, |v| {
        if value.cwise_gt(v) {
            *v = value
        }
    });
}

/// Add `value` to the voxel at `xyz` (logical or for `bool`) and activate it
/// (`tools::setValueOnSum`).
pub fn set_value_on_sum<T: CompositeValue>(tree: &mut Tree<T>, xyz: &Coord, value: T) {
    tree.modify_value(xyz, |v| *v = T::comp_add(*v, value));
}

/// Multiply the voxel at `xyz` by `value` (logical and for `bool`) and activate it
/// (`tools::setValueOnMult`).
pub fn set_value_on_mult<T: CompositeValue>(tree: &mut Tree<T>, xyz: &Coord, value: T) {
    tree.modify_value(xyz, |v| *v = T::comp_mul(*v, value));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::FloatTree;

    #[test]
    fn transform_and_foreach() {
        let mut tree = FloatTree::new(0.0);
        tree.fill(&CoordBBox::from_coords(0, 0, 0, 7, 7, 7), 2.0, true);
        tree.set_value(&Coord::new(20, 0, 0), 3.0);
        foreach(&mut tree, ValueIterKind::On, true, |v| {
            *v.value *= 2.0;
            if *v.value > 5.0 {
                *v.active = false;
            }
        });
        assert_eq!(tree.get_value(&Coord::new(1, 1, 1)), 4.0);
        assert!(!tree.is_value_on(&Coord::new(20, 0, 0)));

        let mut out = crate::tree::Tree::<f64>::new(0.0);
        transform_values(
            &tree,
            ValueIterKind::On,
            &mut out,
            |item, acc| {
                if item.is_voxel() {
                    acc.set_value(&item.coord(), item.value as f64 + 0.5);
                } else {
                    acc.tree_mut().fill(&item.bbox, item.value as f64 + 0.5, true);
                }
            },
            true,
            MergePolicy::ActiveStatesAndNodes,
        );
        assert_eq!(out.get_value(&Coord::new(2, 2, 2)), 4.5);
        assert_eq!(out.active_voxel_count(), 512);

        let sum = accumulate(&tree, ValueIterKind::All, true, 0u64, |a, item| *a += item.bbox.volume(), |a, b| *a += b);
        assert!(sum >= 1024);

        let mut t = FloatTree::new(1.0);
        set_value_on_min(&mut t, &Coord::new(0, 0, 0), 0.5);
        set_value_on_max(&mut t, &Coord::new(0, 0, 1), 0.5);
        set_value_on_sum(&mut t, &Coord::new(0, 0, 2), 0.5);
        set_value_on_mult(&mut t, &Coord::new(0, 0, 3), 0.5);
        assert_eq!(t.get_value(&Coord::new(0, 0, 0)), 0.5);
        assert_eq!(t.get_value(&Coord::new(0, 0, 1)), 1.0);
        assert_eq!(t.get_value(&Coord::new(0, 0, 2)), 1.5);
        assert_eq!(t.get_value(&Coord::new(0, 0, 3)), 0.5);
        assert_eq!(t.active_voxel_count(), 4);
    }
}
