//! Fast queries for active values inside a bounding box (port of
//! `openvdb/tools/FindActiveValues.h`).

use crate::math::CoordBBox;
use crate::tools::count::count_active_voxels_in_bbox;
use crate::tree::{Internal1, Internal2, InternalNode, LeafNode, Node, RootEntry, Tree};
use crate::types::Value;

/// An active tile intersecting a query box (OpenVDB's `TileData`). The bounding box is
/// trimmed to the query box; `level` is 1 for 8³ tiles, 2 for 128³ tiles and 3 for root
/// tiles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileData<T> {
    pub bbox: CoordBBox,
    pub value: T,
    pub level: u32,
    pub state: bool,
}

fn leaf_any_on<T: Value>(leaf: &LeafNode<T>, bbox: &CoordBBox) -> bool {
    let lb = leaf.node_bbox();
    if bbox.is_inside_bbox(&lb) {
        return !leaf.is_empty();
    }
    if leaf.is_dense() {
        return lb.has_overlap(bbox);
    }
    leaf.iter_on_coords().any(|(c, _)| bbox.is_inside(&c))
}

/// Slots of `node` whose child/tile regions overlap `bbox`.
fn overlapping_slots<C: Node, const L: u32>(node: &InternalNode<C, L>, bbox: &CoordBBox) -> Vec<usize> {
    let b = node.node_bbox().intersection(bbox);
    if b.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let lo = (b.min - node.origin()).align_down(C::DIM);
    let hi = (b.max - node.origin()).align_down(C::DIM);
    let mut x = lo[0];
    while x <= hi[0] {
        let mut y = lo[1];
        while y <= hi[1] {
            let mut z = lo[2];
            while z <= hi[2] {
                out.push(InternalNode::<C, L>::coord_to_offset(&(node.origin() + crate::math::Coord::new(x, y, z))));
                z += C::DIM;
            }
            y += C::DIM;
        }
        x += C::DIM;
    }
    out
}

fn n1_any<T: Value>(node: &Internal1<T>, bbox: &CoordBBox, tiles: bool, voxels: bool) -> bool {
    overlapping_slots(node, bbox).into_iter().any(|n| match node.child(n) {
        Some(l) => voxels && leaf_any_on(l, bbox),
        None => tiles && node.value_mask().is_on(n),
    })
}

fn n2_any<T: Value>(node: &Internal2<T>, bbox: &CoordBBox, tiles: bool, voxels: bool) -> bool {
    overlapping_slots(node, bbox).into_iter().any(|n| match node.child(n) {
        Some(c) => n1_any(c, bbox, tiles, voxels),
        None => tiles && node.value_mask().is_on(n),
    })
}

fn root_any<T: Value>(tree: &Tree<T>, bbox: &CoordBBox, tiles: bool, voxels: bool) -> bool {
    tree.root().table().iter().any(|(k, e)| {
        let nb = CoordBBox::create_cube(*k, <Internal2<T> as Node>::DIM);
        if !nb.has_overlap(bbox) {
            return false;
        }
        match e {
            RootEntry::Tile { active, .. } => tiles && *active,
            RootEntry::Child(c) => n2_any(c, bbox, tiles, voxels),
        }
    })
}

/// True if `bbox` intersects any active value (voxel or tile) of the tree
/// (`tools::anyActiveValues`).
pub fn any_active_values<T: Value>(tree: &Tree<T>, bbox: &CoordBBox) -> bool {
    root_any(tree, bbox, true, true)
}

/// True if `bbox` intersects any active voxel stored in a leaf node, ignoring tiles
/// (`tools::anyActiveVoxels`).
pub fn any_active_voxels<T: Value>(tree: &Tree<T>, bbox: &CoordBBox) -> bool {
    root_any(tree, bbox, false, true)
}

/// True if `bbox` intersects any active tile, ignoring leaf voxels (`tools::anyActiveTiles`).
pub fn any_active_tiles<T: Value>(tree: &Tree<T>, bbox: &CoordBBox) -> bool {
    root_any(tree, bbox, true, false)
}

/// True if `bbox` intersects no active values (`tools::noActiveValues`).
pub fn no_active_values<T: Value>(tree: &Tree<T>, bbox: &CoordBBox) -> bool {
    !any_active_values(tree, bbox)
}

/// Number of active values (voxels, including those of active tiles) inside `bbox`
/// (`tools::countActiveValues`).
pub fn count_active_values<T: Value>(tree: &Tree<T>, bbox: &CoordBBox) -> u64 {
    count_active_voxels_in_bbox(tree, bbox)
}

fn tiles_of_node<C: Node, const L: u32>(
    node: &InternalNode<C, L>,
    bbox: &CoordBBox,
    level: u32,
    out: &mut Vec<TileData<C::Value>>,
) {
    for n in overlapping_slots(node, bbox) {
        if node.child(n).is_none() && node.value_mask().is_on(n) {
            let tb = CoordBBox::create_cube(node.offset_to_global_coord(n), C::DIM).intersection(bbox);
            out.push(TileData { bbox: tb, value: node.values()[n], level, state: true });
        }
    }
}

/// All intersections between active tiles and `bbox`, with bounding boxes trimmed to `bbox`
/// (`tools::activeTiles`).
pub fn active_tiles<T: Value>(tree: &Tree<T>, bbox: &CoordBBox) -> Vec<TileData<T>> {
    let mut tiles = Vec::new();
    let dim = <Internal2<T> as Node>::DIM;
    for (k, e) in tree.root().table() {
        if let RootEntry::Tile { value, active: true } = e {
            let tb = CoordBBox::create_cube(*k, dim);
            if !tb.has_overlap(bbox) {
                continue;
            }
            if tb.is_inside_bbox(bbox) {
                tiles.push(TileData { bbox: *bbox, value: *value, level: 3, state: true });
                return tiles;
            }
            tiles.push(TileData { bbox: tb.intersection(bbox), value: *value, level: 3, state: true });
        }
    }
    for (k, e) in tree.root().table() {
        if let RootEntry::Child(n2) = e {
            let nb = CoordBBox::create_cube(*k, dim);
            if !nb.has_overlap(bbox) {
                continue;
            }
            for n in overlapping_slots(n2.as_ref(), bbox) {
                if let Some(n1) = n2.child(n) {
                    tiles_of_node(n1, bbox, 1, &mut tiles);
                }
            }
            tiles_of_node(n2.as_ref(), bbox, 2, &mut tiles);
            if nb.is_inside_bbox(bbox) {
                return tiles;
            }
        }
    }
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;
    use crate::tree::FloatTree;

    #[test]
    fn queries() {
        let mut tree = FloatTree::new(0.0);
        tree.set_value(&Coord::new(5, 5, 5), 1.0);
        tree.fill(&CoordBBox::from_coords(128, 0, 0, 255, 127, 127), 2.0, true);
        tree.fill(&CoordBBox::from_coords(256, 0, 0, 263, 7, 7), 3.0, true);
        let b = CoordBBox::from_coords(0, 0, 0, 4, 4, 4);
        assert!(no_active_values(&tree, &b));
        let b = CoordBBox::from_coords(0, 0, 0, 5, 5, 5);
        assert!(any_active_values(&tree, &b));
        assert!(any_active_voxels(&tree, &b));
        assert!(!any_active_tiles(&tree, &b));
        let b = CoordBBox::from_coords(100, 0, 0, 300, 10, 10);
        assert!(any_active_tiles(&tree, &b));
        assert!(!any_active_voxels(&tree, &b));
        let tiles = active_tiles(&tree, &b);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].level, 1);
        assert_eq!(tiles[0].bbox, CoordBBox::from_coords(256, 0, 0, 263, 7, 7));
        assert_eq!(tiles[1].level, 2);
        assert_eq!(tiles[1].bbox, CoordBBox::from_coords(128, 0, 0, 255, 10, 10));
        assert_eq!(count_active_values(&tree, &b), 128 * 11 * 11 + 512);
    }
}
