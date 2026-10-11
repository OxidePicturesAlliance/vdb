//! Propagate the signs of distance values from the active voxels of a narrow band into
//! its inactive voxels and tiles (port of `openvdb/tools/SignedFloodFill.h`).
//!
//! The sign of each inactive value is inferred from the nearest active value along a
//! scanline (in the z direction first, then y and x), working bottom-up from the leaves
//! to the root. At the root level, the gaps between children that lie on the same
//! z-scanline and are both "inside" at their facing ends are filled with inside tiles.

use crate::math::Coord;
use crate::tools::tree_ops::{internal1_nodes_mut, internal2_nodes_mut, NodeOps};
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::{InternalNode, LeafNode, Node, RootEntry, Tree};
use rayon::prelude::*;

/// Flood-fill a leaf: inactive voxels take the sign of the preceding active voxel along
/// z (or y, or x at the start of a row).
fn fill_leaf<T: ToolValue>(leaf: &mut LeafNode<T>, outside: T, inside: T) {
    let (buffer, mask) = leaf.buffer_and_mask_mut();
    let Some(first) = mask.find_first_on() else {
        let v = if buffer[0].is_negative() { inside } else { outside };
        buffer.iter_mut().for_each(|b| *b = v);
        return;
    };
    let mut x_inside = buffer[first].is_negative();
    for x in 0..8usize {
        let x00 = x << 6;
        if mask.is_on(x00) {
            x_inside = buffer[x00].is_negative();
        }
        let mut y_inside = x_inside;
        for y in 0..8usize {
            let xy0 = x00 + (y << 3);
            if mask.is_on(xy0) {
                y_inside = buffer[xy0].is_negative();
            }
            let mut z_inside = y_inside;
            for z in 0..8usize {
                let xyz = xy0 + z;
                if mask.is_on(xyz) {
                    z_inside = buffer[xyz].is_negative();
                } else {
                    buffer[xyz] = if z_inside { inside } else { outside };
                }
            }
        }
    }
}

/// Flood-fill the tiles of an internal node using the first/last values of its children.
fn fill_internal<T: ToolValue, C: NodeOps<T>, const L: u32>(node: &mut InternalNode<C, L>, outside: T, inside: T) {
    let dim = 1usize << L;
    let first = (0..InternalNode::<C, L>::NUM_VALUES).find(|&n| node.is_child(n));
    let Some(first) = first else {
        let v = if node.values()[0].is_negative() { inside } else { outside };
        node.values_mut().iter_mut().for_each(|t| *t = v);
        return;
    };
    let mut x_inside = node.child(first).unwrap().first_value().is_negative();
    for x in 0..dim {
        let x00 = x << (2 * L);
        if let Some(c) = node.child(x00) {
            x_inside = c.last_value().is_negative();
        }
        let mut y_inside = x_inside;
        for y in 0..dim {
            let xy0 = x00 + (y << L);
            if let Some(c) = node.child(xy0) {
                y_inside = c.last_value().is_negative();
            }
            let mut z_inside = y_inside;
            for z in 0..dim {
                let xyz = xy0 + z;
                if let Some(c) = node.child(xyz) {
                    z_inside = c.last_value().is_negative();
                } else {
                    node.values_mut()[xyz] = if z_inside { inside } else { outside };
                }
            }
        }
    }
}

/// Core flood fill with explicit values (the sign of `outside`/`inside` is forced).
fn do_flood_fill<T: ToolValue>(tree: &mut Tree<T>, outside: T, inside: T, threaded: bool, min_level: u32) {
    let outside = outside.abs_value();
    let inside = inside.abs_value().negative();
    if min_level == 0 {
        if threaded {
            tree.par_for_each_leaf_mut(|l| fill_leaf(l, outside, inside));
        } else {
            tree.leaves_mut().for_each(|l| fill_leaf(l, outside, inside));
        }
    }
    if min_level <= 1 {
        let mut nodes = internal1_nodes_mut(tree);
        if threaded {
            nodes.par_iter_mut().for_each(|n| fill_internal(*n, outside, inside));
        } else {
            nodes.iter_mut().for_each(|n| fill_internal(*n, outside, inside));
        }
    }
    if min_level <= 2 {
        let mut nodes = internal2_nodes_mut(tree);
        if threaded {
            nodes.par_iter_mut().for_each(|n| fill_internal(*n, outside, inside));
        } else {
            nodes.iter_mut().for_each(|n| fill_internal(*n, outside, inside));
        }
    }
    if min_level <= 3 {
        fill_root(tree, outside, inside);
    }
}

fn fill_root<T: ToolValue>(tree: &mut Tree<T>, outside: T, inside: T) {
    let dim = <crate::tree::Internal2<T> as Node>::DIM;
    let root = tree.root_mut();
    let children: Vec<(Coord, T, T)> = root
        .table()
        .iter()
        .filter_map(|(k, e)| match e {
            RootEntry::Child(c) => Some((*k, c.first_value(), c.last_value())),
            _ => None,
        })
        .collect();
    for w in children.windows(2) {
        let (a, _, a_last) = w[0];
        let (b, b_first, _) = w[1];
        let d = b - a;
        if d[0] != 0 || d[1] != 0 || d[2] == dim {
            continue;
        }
        if !(a_last.is_negative()) || !(b_first.is_negative()) {
            continue;
        }
        let mut c = a + Coord::new(0, 0, dim);
        while c[2] != b[2] {
            root.add_root_tile(&c, inside, false);
            c[2] += dim;
        }
    }
    root.set_background_unchecked(outside);
}

/// Set the values of all inactive voxels and tiles of a narrow-band level set (or other
/// signed field) to `±|background|` depending on their inferred sign (`tools::signedFloodFill`).
///
/// Returns an error for unsigned value types.
pub fn signed_flood_fill<T: ToolValue>(tree: &mut Tree<T>, threaded: bool) -> ToolResult<()> {
    signed_flood_fill_min_level(tree, threaded, 0)
}

/// [`signed_flood_fill`] restricted to nodes at or above `min_level`
/// (0 = leaves, 1 and 2 = internal nodes, 3 = root).
pub fn signed_flood_fill_min_level<T: ToolValue>(tree: &mut Tree<T>, threaded: bool, min_level: u32) -> ToolResult<()> {
    let bg = *tree.background();
    signed_flood_fill_with_values_min_level(tree, bg, bg.negative(), threaded, min_level)
}

/// Like [`signed_flood_fill`], but inactive values are set to `|outside|` or `-|inside|`
/// (`tools::signedFloodFillWithValues`).
pub fn signed_flood_fill_with_values<T: ToolValue>(
    tree: &mut Tree<T>,
    outside: T,
    inside: T,
    threaded: bool,
) -> ToolResult<()> {
    signed_flood_fill_with_values_min_level(tree, outside, inside, threaded, 0)
}

/// [`signed_flood_fill_with_values`] restricted to nodes at or above `min_level`.
pub fn signed_flood_fill_with_values_min_level<T: ToolValue>(
    tree: &mut Tree<T>,
    outside: T,
    inside: T,
    threaded: bool,
    min_level: u32,
) -> ToolResult<()> {
    if !is_signed::<T>() {
        return Err(ToolError::Type("signedFloodFill is supported only for signed value grids".into()));
    }
    do_flood_fill(tree, outside, inside, threaded, min_level);
    Ok(())
}

/// True if `T` is a signed value type (OpenVDB requires `std::is_signed<ValueType>`).
fn is_signed<T: ToolValue>() -> bool {
    use std::any::TypeId;
    let id = TypeId::of::<T>();
    !(id == TypeId::of::<u8>()
        || id == TypeId::of::<u16>()
        || id == TypeId::of::<u32>()
        || id == TypeId::of::<u64>()
        || id == TypeId::of::<bool>()
        || id == TypeId::of::<crate::math::Vec3f>()
        || id == TypeId::of::<crate::math::Vec3d>()
        || id == TypeId::of::<crate::math::Vec3i>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;
    use crate::tree::FloatTree;

    #[test]
    fn flood_fill_sphere_like() {
        // A thin shell of active voxels around the origin: x^2+y^2+z^2 ~ 20^2.
        let mut tree = FloatTree::new(3.0);
        let r = 20.0f32;
        for x in -25..=25 {
            for y in -25..=25 {
                for z in -25..=25 {
                    let d = ((x * x + y * y + z * z) as f32).sqrt() - r;
                    if d.abs() < 2.0 {
                        tree.set_value(&Coord::new(x, y, z), d);
                    }
                }
            }
        }
        signed_flood_fill(&mut tree, true).unwrap();
        assert_eq!(tree.get_value(&Coord::new(0, 0, 0)), -3.0);
        assert_eq!(tree.get_value(&Coord::new(5, -3, 2)), -3.0);
        assert_eq!(tree.get_value(&Coord::new(30, 0, 0)), 3.0);
        assert_eq!(tree.get_value(&Coord::new(1000, 0, 0)), 3.0);
        let mut u = crate::tree::Tree::<u32>::new(1);
        assert!(signed_flood_fill(&mut u, true).is_err());
    }
}
