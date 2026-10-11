//! Topological dilation and erosion of active values (port of `openvdb/tools/Morphology.h`).
//!
//! Dilation and erosion operate directly on the 64-bit words of the leaf active masks (and
//! those of their neighbor leaves), with face (6), face-edge (18) or face-edge-vertex (26)
//! connectivity. Voxel values are never changed; newly activated voxels keep the value
//! they had (the value of the tile or background they were created from).

use crate::math::Coord;
use crate::tools::prune::prune;
use crate::tools::tree_ops::{has_active_tiles, topology_intersection, topology_union};
use crate::tools::value_traits::ToolValue;
use crate::tree::{LeafNode, Node, Tree};
use rayon::prelude::*;

/// Voxel neighborhoods used by dilation and erosion (OpenVDB's `NearestNeighbors`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NearestNeighbors {
    /// The 6 face-adjacent neighbors (`NN_FACE`).
    #[default]
    Face,
    /// Face- and edge-adjacent neighbors, 18 in total (`NN_FACE_EDGE`).
    FaceEdge,
    /// Face-, edge- and vertex-adjacent neighbors, 26 in total (`NN_FACE_EDGE_VERTEX`).
    FaceEdgeVertex,
}

/// How active tiles are treated by [`dilate_active_values`] and [`erode_active_values`]
/// (OpenVDB's `TilePolicy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TilePolicy {
    /// Only voxels are dilated; active tiles still count as active neighbors when eroding
    /// but are never eroded themselves (`IGNORE_TILES`).
    IgnoreTiles,
    /// Active tiles are voxelized first and left voxelized (`EXPAND_TILES`).
    ExpandTiles,
    /// Active tiles contribute like active voxels but are preserved where possible, and
    /// constant nodes are pruned afterwards (`PRESERVE_TILES`).
    #[default]
    PreserveTiles,
}

/// The 8 words of a leaf's active mask: word `x` holds the 8×8 `(y, z)` slab at that x,
/// byte `y` of a word holds the z row and bit `z` of that byte the voxel.
type Words = [u64; 8];

static EMPTY: Words = [0; 8];
static FULL: Words = [!0; 8];

/// Bit `z = 0` / `z = 7` of every z row.
const Z_LO: u64 = 0x0101_0101_0101_0101;
const Z_HI: u64 = 0x8080_8080_8080_8080;
/// Byte `y = 0` / `y = 7` of a word.
const Y_LO: u64 = 0xFF;
const Y_HI: u64 = 0xFF << 56;

/// Union (dilation, `DILATE = true`) or intersection (erosion) of three words.
#[inline(always)]
fn combine<const DILATE: bool>(a: u64, b: u64, c: u64) -> u64 {
    if DILATE {
        a | b | c
    } else {
        a & b & c
    }
}

/// `c` combined with its copies shifted by ±1 voxel along z; `m` / `p` are the masks of the
/// −z / +z neighbor leaves.
#[inline(always)]
fn along_z<const DILATE: bool>(c: &Words, m: &Words, p: &Words) -> Words {
    std::array::from_fn(|i| {
        let below = ((c[i] << 1) & !Z_LO) | ((m[i] >> 7) & Z_LO); // voxel at z − 1
        let above = ((c[i] >> 1) & !Z_HI) | ((p[i] << 7) & Z_HI); // voxel at z + 1
        combine::<DILATE>(c[i], below, above)
    })
}

/// As [`along_z`], along y.
#[inline(always)]
fn along_y<const DILATE: bool>(c: &Words, m: &Words, p: &Words) -> Words {
    std::array::from_fn(|i| combine::<DILATE>(c[i], (c[i] << 8) | (m[i] >> 56), (c[i] >> 8) | (p[i] << 56)))
}

/// As [`along_z`], along x.
#[inline(always)]
fn along_x<const DILATE: bool>(c: &Words, m: &Words, p: &Words) -> Words {
    std::array::from_fn(|i| {
        let below = if i == 0 { m[7] } else { c[i - 1] };
        let above = if i == 7 { p[0] } else { c[i + 1] };
        combine::<DILATE>(c[i], below, above)
    })
}

#[inline(always)]
fn combine_words<const DILATE: bool>(a: &Words, b: &Words, c: &Words) -> Words {
    std::array::from_fn(|i| combine::<DILATE>(a[i], b[i], c[i]))
}

/// Index of the neighbor at offset `(dx, dy, dz)` (each in −1..=1) in a 3×3×3 neighborhood.
const fn nb(dx: i32, dy: i32, dz: i32) -> usize {
    ((dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)) as usize
}

/// The neighbor offsets (excluding the center) used by `nn`.
fn stencil(nn: NearestNeighbors) -> impl Iterator<Item = [i32; 3]> {
    let max_nonzero = match nn {
        NearestNeighbors::Face => 1,
        NearestNeighbors::FaceEdge => 2,
        NearestNeighbors::FaceEdgeVertex => 3,
    };
    (0..27).map(|i| [i / 9 - 1, (i / 3) % 3 - 1, i % 3 - 1]).filter(move |d| {
        let k = d.iter().filter(|&&c| c != 0).count();
        k > 0 && k <= max_nonzero
    })
}

/// Dilate (`DILATE = true`) or erode the center leaf mask of a neighborhood `n`
/// (indexed by [`nb`]; only the entries in [`stencil`]`(nn)` are read).
#[inline(always)]
fn filter<const DILATE: bool>(nn: NearestNeighbors, n: &[&Words; 27]) -> Words {
    let at = |dx, dy, dz| n[nb(dx, dy, dz)];
    let z_col = |dx, dy| along_z::<DILATE>(at(dx, dy, 0), at(dx, dy, -1), at(dx, dy, 1));
    match nn {
        NearestNeighbors::Face => {
            let c = at(0, 0, 0);
            let z = along_z::<DILATE>(c, at(0, 0, -1), at(0, 0, 1));
            let y = along_y::<DILATE>(c, at(0, -1, 0), at(0, 1, 0));
            let x = along_x::<DILATE>(c, at(-1, 0, 0), at(1, 0, 0));
            combine_words::<DILATE>(&z, &y, &x)
        }
        NearestNeighbors::FaceEdge => {
            // Union (intersection) of the 3×3 squares in the xy, xz and yz planes.
            let z0 = z_col(0, 0);
            let yz = along_y::<DILATE>(&z0, &z_col(0, -1), &z_col(0, 1));
            let xz = along_x::<DILATE>(&z0, &z_col(-1, 0), &z_col(1, 0));
            let y_row = |dx| along_y::<DILATE>(at(dx, 0, 0), at(dx, -1, 0), at(dx, 1, 0));
            let xy = along_x::<DILATE>(&y_row(0), &y_row(-1), &y_row(1));
            combine_words::<DILATE>(&yz, &xz, &xy)
        }
        NearestNeighbors::FaceEdgeVertex => {
            let y_slab = |dx| along_y::<DILATE>(&z_col(dx, 0), &z_col(dx, -1), &z_col(dx, 1));
            along_x::<DILATE>(&y_slab(0), &y_slab(-1), &y_slab(1))
        }
    }
}

fn leaf_words<T: crate::types::Value>(leaf: &LeafNode<T>) -> &Words {
    leaf.value_mask().words().try_into().expect("leaf masks have 8 words")
}

/// Whether `w` has active voxels on the face, edge or corner shared with the neighbor leaf at
/// offset `d` (i.e. whether dilating `w` spills into that neighbor).
fn touches(w: &Words, d: [i32; 3]) -> bool {
    let words: &[u64] = match d[0] {
        -1 => &w[..1],
        1 => &w[7..],
        _ => &w[..],
    };
    let y = match d[1] {
        -1 => Y_LO,
        1 => Y_HI,
        _ => !0,
    };
    let z = match d[2] {
        -1 => Z_LO,
        1 => Z_HI,
        _ => !0,
    };
    words.iter().fold(0, |a, w| a | w) & y & z != 0
}

/// Compute the dilated (`DILATE = true`) or eroded mask of every leaf of `tree`, in leaf
/// order, from the current masks. Missing neighbors count as empty, or for erosion as full
/// when they lie in an active tile.
fn filter_leaves<T: ToolValue, const DILATE: bool>(tree: &Tree<T>, nn: NearestNeighbors, threaded: bool) -> Vec<Words> {
    let offsets: Vec<[i32; 3]> = stencil(nn).collect();
    let one = |acc: &mut crate::tree::ValueAccessor<'_, _>, leaf: &LeafNode<T>| -> Words {
        let center = leaf_words(leaf);
        if !DILATE && center.iter().all(|w| *w == 0) {
            return EMPTY;
        }
        let origin = leaf.origin();
        let mut n: [&Words; 27] = [&EMPTY; 27];
        n[nb(0, 0, 0)] = center;
        for &d in &offsets {
            if !DILATE && !touches(center, d) {
                continue; // the neighbor only affects border voxels that are inactive anyway
            }
            let o = origin.offset_by(8 * d[0], 8 * d[1], 8 * d[2]);
            if let Some(l) = acc.probe_leaf(&o) {
                n[nb(d[0], d[1], d[2])] = leaf_words(l);
            } else if !DILATE && acc.is_value_on(&o) {
                n[nb(d[0], d[1], d[2])] = &FULL;
            }
        }
        filter::<DILATE>(nn, &n)
    };
    let leaves = tree.leaf_refs();
    if threaded {
        leaves.par_iter().map_init(|| tree.accessor(), |acc, l| one(acc, l)).collect()
    } else {
        let mut acc = tree.accessor();
        leaves.iter().map(|l| one(&mut acc, l)).collect()
    }
}

/// Replace the masks of all leaves of `tree` (in leaf order) with `masks`.
fn store_masks<T: ToolValue>(tree: &mut Tree<T>, masks: Vec<Words>, threaded: bool) {
    let store = |(leaf, w): (&mut LeafNode<T>, &Words)| leaf.value_mask_mut().words_mut().copy_from_slice(w);
    if threaded {
        tree.leaf_refs_mut().into_par_iter().zip(masks.par_iter()).for_each(store);
    } else {
        tree.leaves_mut().zip(masks.iter()).for_each(store);
    }
}

/// Topologically dilate the active voxels of all leaf nodes `iterations` times
/// (`Morphology::dilateVoxels`). Active tiles neither dilate nor receive dilation (they are
/// already active); new leaves are created as needed from the surrounding tile values.
pub fn dilate_voxels<T: ToolValue>(tree: &mut Tree<T>, iterations: usize, nn: NearestNeighbors, threaded: bool) {
    let offsets: Vec<[i32; 3]> = stencil(nn).collect();
    for _ in 0..iterations {
        if tree.leaf_count() == 0 {
            return;
        }
        // Create the (empty) leaves that receive voxels from their neighbors.
        let missing: Vec<Coord> = {
            let t = &*tree;
            let one = |acc: &mut crate::tree::ValueAccessor<'_, _>, leaf: &LeafNode<T>| -> Vec<Coord> {
                let w = leaf_words(leaf);
                let mut out = Vec::new();
                if w.iter().all(|w| *w == 0) {
                    return out;
                }
                for &d in &offsets {
                    if touches(w, d) {
                        let o = leaf.origin().offset_by(8 * d[0], 8 * d[1], 8 * d[2]);
                        if acc.probe_leaf(&o).is_none() {
                            out.push(o);
                        }
                    }
                }
                out
            };
            let leaves = t.leaf_refs();
            let mut missing: Vec<Coord> = if threaded {
                leaves.par_iter().map_init(|| t.accessor(), |acc, l| one(acc, l)).flatten().collect()
            } else {
                let mut acc = t.accessor();
                leaves.iter().flat_map(|l| one(&mut acc, l)).collect()
            };
            missing.sort_unstable();
            missing.dedup();
            missing
        };
        let mut accessor = tree.accessor_mut();
        for o in missing {
            if !accessor.is_value_on(&o) {
                accessor.touch_leaf(&o); // otherwise inside an active tile
            }
        }
        let masks = filter_leaves::<T, true>(tree, nn, threaded);
        store_masks(tree, masks, threaded);
    }
}

/// Topologically erode the active voxels of all leaf nodes `iterations` times
/// (`Morphology::erodeVoxels`): a voxel stays active only if all of its neighbors are
/// active. Active tiles count as active neighbors and are never eroded; inactive tiles and
/// the background count as inactive.
pub fn erode_voxels<T: ToolValue>(tree: &mut Tree<T>, iterations: usize, nn: NearestNeighbors, threaded: bool) {
    if tree.leaf_count() == 0 {
        return;
    }
    for _ in 0..iterations {
        let masks = filter_leaves::<T, false>(tree, nn, threaded);
        store_masks(tree, masks, threaded);
    }
}

/// Topologically dilate all active values (voxels and, depending on `mode`, tiles) of a
/// tree (`tools::dilateActiveValues`). Values are unchanged.
pub fn dilate_active_values<T: ToolValue>(
    tree: &mut Tree<T>,
    iterations: i32,
    nn: NearestNeighbors,
    mode: TilePolicy,
    threaded: bool,
) {
    if iterations <= 0 {
        return;
    }
    let iters = iterations as usize;
    match mode {
        TilePolicy::IgnoreTiles => dilate_voxels(tree, iters, nn, threaded),
        TilePolicy::ExpandTiles => {
            tree.voxelize_active_tiles();
            dilate_voxels(tree, iters, nn, threaded);
        }
        TilePolicy::PreserveTiles => {
            let mut topology = Tree::<bool>::new(false);
            topology_union(&mut topology, tree, false);
            topology.voxelize_active_tiles();
            dilate_voxels(&mut topology, iters, nn, threaded);
            prune(&mut topology, false, threaded);
            topology_union(tree, &topology, true);
            prune(tree, T::zero(), threaded);
        }
    }
}

/// Topologically erode all active values of a tree (`tools::erodeActiveValues`).
/// Values are unchanged.
pub fn erode_active_values<T: ToolValue>(
    tree: &mut Tree<T>,
    iterations: i32,
    nn: NearestNeighbors,
    mode: TilePolicy,
    threaded: bool,
) {
    if iterations <= 0 {
        return;
    }
    let iters = iterations as usize;
    match mode {
        TilePolicy::PreserveTiles => {
            let mut topology = Tree::<bool>::new(false);
            topology_union(&mut topology, tree, false);
            topology.voxelize_active_tiles();
            erode_voxels(&mut topology, iters, nn, threaded);
            prune(&mut topology, false, threaded);
            topology_intersection(tree, &topology);
        }
        TilePolicy::ExpandTiles => {
            if has_active_tiles(tree) {
                tree.voxelize_active_tiles();
            }
            erode_voxels(tree, iters, nn, threaded);
        }
        TilePolicy::IgnoreTiles => erode_voxels(tree, iters, nn, threaded),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::CoordBBox;
    use crate::tree::FloatTree;

    #[test]
    fn dilate_single_voxel() {
        for (nn, count, leaves) in [
            (NearestNeighbors::Face, 7, 4),
            (NearestNeighbors::FaceEdge, 19, 7),
            (NearestNeighbors::FaceEdgeVertex, 27, 8),
        ] {
            let mut tree = FloatTree::new(0.0);
            tree.set_value(&Coord::new(7, 7, 7), 1.0);
            dilate_active_values(&mut tree, 1, nn, TilePolicy::IgnoreTiles, true);
            assert_eq!(tree.active_voxel_count(), count);
            assert_eq!(tree.leaf_count(), leaves);
            erode_active_values(&mut tree, 1, nn, TilePolicy::IgnoreTiles, true);
            assert_eq!(tree.active_voxel_count(), 1);
        }
        let mut tree = FloatTree::new(0.0);
        tree.set_value(&Coord::new(0, 0, 0), 1.0);
        dilate_active_values(&mut tree, 3, NearestNeighbors::Face, TilePolicy::PreserveTiles, false);
        // Octahedron of radius 3: 1 + 6 + 18 + 38 = 63 voxels.
        assert_eq!(tree.active_voxel_count(), 63);
    }

    /// Compare against a voxel-by-voxel evaluation on a dense random block of voxels spanning leaf borders.
    #[test]
    fn matches_brute_force() {
        use std::collections::HashSet;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut tree = FloatTree::new(0.0);
        for x in -20..20 {
            for y in -20..20 {
                for z in -20..20 {
                    if next() % 8 != 0 {
                        tree.set_value(&Coord::new(x, y, z), 1.0);
                    }
                }
            }
        }
        for nn in [NearestNeighbors::Face, NearestNeighbors::FaceEdge, NearestNeighbors::FaceEdgeVertex] {
            let max_nonzero = match nn {
                NearestNeighbors::Face => 1,
                NearestNeighbors::FaceEdge => 2,
                NearestNeighbors::FaceEdgeVertex => 3,
            };
            let offsets: Vec<Coord> = (-1..=1)
                .flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| Coord::new(x, y, z))))
                .filter(|d| [d.x(), d.y(), d.z()].iter().filter(|&&c| c != 0).count() <= max_nonzero)
                .collect();
            let active: HashSet<Coord> = tree.active_voxels().map(|(c, _)| c).collect();
            let dilated: HashSet<Coord> = active.iter().flat_map(|c| offsets.iter().map(move |d| *c + *d)).collect();
            let eroded: HashSet<Coord> =
                active.iter().filter(|c| offsets.iter().all(|d| active.contains(&(**c + *d)))).copied().collect();
            assert!(!eroded.is_empty());
            for threaded in [false, true] {
                let mut d = tree.clone();
                dilate_voxels(&mut d, 1, nn, threaded);
                assert_eq!(d.active_voxels().map(|(c, _)| c).collect::<HashSet<_>>(), dilated, "dilate {nn:?}");
                let mut e = tree.clone();
                erode_voxels(&mut e, 1, nn, threaded);
                assert_eq!(e.active_voxels().map(|(c, _)| c).collect::<HashSet<_>>(), eroded, "erode {nn:?}");
            }
        }
    }

    #[test]
    fn tiles() {
        let mut tree = FloatTree::new(0.0);
        tree.fill(&CoordBBox::from_coords(0, 0, 0, 127, 127, 127), 1.0, true);
        let mut t2 = tree.clone();
        dilate_active_values(&mut tree, 1, NearestNeighbors::Face, TilePolicy::PreserveTiles, true);
        assert_eq!(tree.active_voxel_count(), 128 * 128 * 128 + 6 * 128 * 128);
        assert_eq!(tree.get_value(&Coord::new(5, 5, 5)), 1.0);
        erode_active_values(&mut t2, 2, NearestNeighbors::Face, TilePolicy::PreserveTiles, true);
        assert_eq!(t2.active_voxel_count(), 124 * 124 * 124);
        let mut t3 = FloatTree::new(0.0);
        t3.fill(&CoordBBox::from_coords(0, 0, 0, 127, 127, 127), 1.0, true);
        erode_active_values(&mut t3, 2, NearestNeighbors::FaceEdgeVertex, TilePolicy::IgnoreTiles, true);
        assert_eq!(t3.active_voxel_count(), 128 * 128 * 128);
    }
}
