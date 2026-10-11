//! Counting voxels, tiles and memory, and finding extreme values (port of
//! `openvdb/tools/Count.h`).

use crate::math::CoordBBox;
use crate::tools::value_traits::ToolValue;
use crate::tree::{LeafNode, Node, Tree};
use crate::types::Value;
use rayon::prelude::*;

/// Total number of active voxels, including those covered by active tiles
/// (`tools::countActiveVoxels`).
pub fn count_active_voxels<T: Value>(tree: &Tree<T>) -> u64 {
    let mut count = 0u64;
    tree.visit_tiles(&mut |b, _, on, _| {
        if on {
            count += b.volume();
        }
    });
    count + count_active_leaf_voxels(tree)
}

fn leaf_count_in_bbox<T: Value>(leaf: &LeafNode<T>, bbox: &CoordBBox) -> u64 {
    let lb = leaf.node_bbox();
    if bbox.is_inside_bbox(&lb) {
        leaf.on_voxel_count()
    } else if !lb.has_overlap(bbox) {
        0
    } else if leaf.is_dense() {
        lb.intersection(bbox).volume()
    } else {
        leaf.iter_on_coords().filter(|(c, _)| bbox.is_inside(c)).count() as u64
    }
}

/// Number of active voxels (including those of active tiles) inside `bbox`
/// (`tools::countActiveVoxels(tree, bbox)`).
pub fn count_active_voxels_in_bbox<T: Value>(tree: &Tree<T>, bbox: &CoordBBox) -> u64 {
    if bbox.is_empty() {
        return 0;
    }
    let mut count = 0u64;
    tree.visit_tiles(&mut |b, _, on, _| {
        if on && b.has_overlap(bbox) {
            count += b.intersection(bbox).volume();
        }
    });
    count + count_active_leaf_voxels_in_bbox(tree, bbox)
}

/// Number of active voxels stored in leaf nodes (`tools::countActiveLeafVoxels`).
pub fn count_active_leaf_voxels<T: Value>(tree: &Tree<T>) -> u64 {
    tree.leaf_refs().par_iter().map(|l| l.on_voxel_count()).sum()
}

/// Number of active leaf voxels inside `bbox` (`tools::countActiveLeafVoxels(tree, bbox)`).
pub fn count_active_leaf_voxels_in_bbox<T: Value>(tree: &Tree<T>, bbox: &CoordBBox) -> u64 {
    if bbox.is_empty() {
        return 0;
    }
    tree.leaf_refs().par_iter().map(|l| leaf_count_in_bbox(l, bbox)).sum()
}

/// Number of inactive voxels, excluding background tiles of the root
/// (`tools::countInactiveVoxels`).
pub fn count_inactive_voxels<T: ToolValue>(tree: &Tree<T>) -> u64 {
    let bg = *tree.background();
    let mut count = 0u64;
    let root_level = crate::tree::RootNode::<crate::tree::Internal2<T>>::LEVEL;
    tree.visit_tiles(&mut |b, v, on, level| {
        if !on && !(level == root_level && v.is_approx_equal(&bg)) {
            count += b.volume();
        }
    });
    count + count_inactive_leaf_voxels(tree)
}

/// Number of inactive voxels stored in leaf nodes (`tools::countInactiveLeafVoxels`).
pub fn count_inactive_leaf_voxels<T: Value>(tree: &Tree<T>) -> u64 {
    tree.leaf_refs().par_iter().map(|l| l.off_voxel_count()).sum()
}

/// Number of active tiles at all levels (`tools::countActiveTiles`).
pub fn count_active_tiles<T: Value>(tree: &Tree<T>) -> u64 {
    let mut count = 0u64;
    tree.visit_tiles(&mut |_, _, on, _| count += on as u64);
    count
}

/// Approximate memory footprint of the tree in bytes (`tools::memUsage`).
pub fn mem_usage<T: Value>(tree: &Tree<T>) -> u64 {
    tree.mem_usage() as u64
}

pub use crate::math::stats::MinMax;

/// The minimum and maximum active values (voxels and tiles) of the tree, compared with
/// `math::cwiseLessThan` (lexicographically for vectors). Both are zero for a tree without
/// active values (`tools::minMax`).
pub fn min_max<T: ToolValue>(tree: &Tree<T>) -> MinMax<T> {
    let add = |acc: &mut Option<MinMax<T>>, v: T| match acc {
        None => *acc = Some(MinMax::with_bounds(v, v)),
        Some(m) => m.add_by(v, |a, b| a.cwise_lt(b)),
    };
    let mut acc: Option<MinMax<T>> = None;
    tree.visit_tiles(&mut |_, v, on, _| {
        if on {
            add(&mut acc, *v)
        }
    });
    let leaf_results: Vec<Option<MinMax<T>>> = tree
        .leaf_refs()
        .par_iter()
        .map(|l| {
            let mut a = None;
            for (_, v) in l.iter_on() {
                add(&mut a, *v);
            }
            a
        })
        .collect();
    for r in leaf_results.into_iter().flatten() {
        add(&mut acc, r.min);
        add(&mut acc, r.max);
    }
    acc.unwrap_or(MinMax::with_bounds(T::zero(), T::zero()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;
    use crate::tree::FloatTree;

    #[test]
    fn counts() {
        let mut tree = FloatTree::new(0.0);
        tree.fill(&CoordBBox::from_coords(0, 0, 0, 127, 127, 127), 2.0, true);
        tree.set_value(&Coord::new(-1, -1, -1), -3.0);
        tree.set_value_off(&Coord::new(-2, -1, -1), 5.0);
        assert_eq!(count_active_voxels(&tree), 128 * 128 * 128 + 1);
        assert_eq!(count_active_leaf_voxels(&tree), 1);
        assert_eq!(count_active_tiles(&tree), 1);
        assert_eq!(count_inactive_leaf_voxels(&tree), 511);
        let b = CoordBBox::from_coords(-1, -1, -1, 9, 9, 9);
        assert_eq!(count_active_voxels_in_bbox(&tree, &b), 1000 + 1);
        assert_eq!(count_active_leaf_voxels_in_bbox(&tree, &b), 1);
        let mm = min_max(&tree);
        assert_eq!((mm.min, mm.max), (-3.0, 2.0));
        assert_eq!(min_max(&FloatTree::new(1.0)).max, 0.0);
        assert!(count_inactive_voxels(&tree) > 511);
        assert!(mem_usage(&tree) > 0);
    }
}
