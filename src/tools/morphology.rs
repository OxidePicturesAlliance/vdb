//! Topological dilation and erosion of active values (port of `openvdb/tools/Morphology.h`).
//!
//! Dilation and erosion operate on the active masks of leaf nodes, 8×8×8 voxels at a time
//! using 64-bit word operations, with face (6), face-edge (18) or face-edge-vertex (26)
//! connectivity. Voxel values are never changed; newly activated voxels keep the value
//! they had (the value of the tile or background they were created from).

use crate::math::Coord;
use crate::tools::prune::prune;
use crate::tools::tree_ops::{has_active_tiles, topology_intersection, topology_union};
use crate::tools::value_traits::ToolValue;
use crate::tree::{LeafNode, Node, Tree};
use rayon::prelude::*;
use std::collections::HashMap;

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

/// The 8 words (one per x slab; byte y holds the z row) of a leaf's active mask.
type Words = [u64; 8];

/// A 24³ bit volume (3×3×3 leaves), stored as z rows of 24 bits indexed by `[x][y]`.
#[derive(Clone)]
struct Local {
    rows: Box<[[u32; 24]; 24]>,
}

const ROW_MASK: u32 = 0x00FF_FFFF;

impl Local {
    fn new() -> Self {
        Local { rows: Box::new([[0u32; 24]; 24]) }
    }
    /// Store the words of the leaf at block offset `(bx, by, bz)` (each in 0..3).
    fn set_block(&mut self, b: [usize; 3], w: &Words) {
        for x in 0..8 {
            for y in 0..8 {
                let byte = ((w[x] >> (8 * y)) & 0xFF) as u32;
                let r = &mut self.rows[8 * b[0] + x][8 * b[1] + y];
                *r = (*r & !(0xFF << (8 * b[2]))) | (byte << (8 * b[2]));
            }
        }
    }
    fn fill_block(&mut self, b: [usize; 3]) {
        self.set_block(b, &[!0u64; 8]);
    }
    fn block(&self, b: [usize; 3]) -> Words {
        let mut w = [0u64; 8];
        for (x, wx) in w.iter_mut().enumerate() {
            for y in 0..8 {
                let byte = ((self.rows[8 * b[0] + x][8 * b[1] + y] >> (8 * b[2])) & 0xFF) as u64;
                *wx |= byte << (8 * y);
            }
        }
        w
    }
    fn is_block_empty(&self, b: [usize; 3]) -> bool {
        (0..8).all(|x| (0..8).all(|y| (self.rows[8 * b[0] + x][8 * b[1] + y] >> (8 * b[2])) & 0xFF == 0))
    }
    fn map_rows(&self, f: impl Fn(u32) -> u32) -> Local {
        let mut out = self.clone();
        for row in out.rows.iter_mut() {
            for r in row.iter_mut() {
                *r = f(*r) & ROW_MASK;
            }
        }
        out
    }
    /// Combine each row with its neighbors along y (`axis == 1`) or x (`axis == 0`).
    fn combine_axis(&self, axis: usize, f: impl Fn(u32, u32, u32) -> u32) -> Local {
        let mut out = Local::new();
        for x in 0..24 {
            for y in 0..24 {
                let get = |dx: i32, dy: i32| -> u32 {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if (0..24).contains(&xx) && (0..24).contains(&yy) {
                        self.rows[xx as usize][yy as usize]
                    } else {
                        0
                    }
                };
                let (m, p) = if axis == 0 { (get(-1, 0), get(1, 0)) } else { (get(0, -1), get(0, 1)) };
                out.rows[x][y] = f(m, self.rows[x][y], p) & ROW_MASK;
            }
        }
        out
    }
    fn or(&self, o: &Local) -> Local {
        let mut out = self.clone();
        for x in 0..24 {
            for y in 0..24 {
                out.rows[x][y] |= o.rows[x][y];
            }
        }
        out
    }
    fn and(&self, o: &Local) -> Local {
        let mut out = self.clone();
        for x in 0..24 {
            for y in 0..24 {
                out.rows[x][y] &= o.rows[x][y];
            }
        }
        out
    }

    // Dilation (max filters).
    fn box_z(&self) -> Local {
        self.map_rows(|r| r | (r << 1) | (r >> 1))
    }
    fn box_y(&self) -> Local {
        self.combine_axis(1, |m, c, p| m | c | p)
    }
    fn box_x(&self) -> Local {
        self.combine_axis(0, |m, c, p| m | c | p)
    }
    fn dilate(&self, nn: NearestNeighbors) -> Local {
        match nn {
            NearestNeighbors::Face => {
                let z = self.box_z();
                let y = self.combine_axis(1, |m, _, p| m | p);
                let x = self.combine_axis(0, |m, _, p| m | p);
                z.or(&y).or(&x)
            }
            NearestNeighbors::FaceEdge => {
                let xy = self.box_y().box_x();
                let xz = self.box_z().box_x();
                let yz = self.box_z().box_y();
                xy.or(&xz).or(&yz)
            }
            NearestNeighbors::FaceEdgeVertex => self.box_z().box_y().box_x(),
        }
    }

    // Erosion (min filters).
    fn min_z(&self) -> Local {
        self.map_rows(|r| r & (r << 1) & (r >> 1))
    }
    fn min_y(&self) -> Local {
        self.combine_axis(1, |m, c, p| m & c & p)
    }
    fn min_x(&self) -> Local {
        self.combine_axis(0, |m, c, p| m & c & p)
    }
    fn erode(&self, nn: NearestNeighbors) -> Local {
        match nn {
            NearestNeighbors::Face => {
                let z = self.min_z();
                let y = self.combine_axis(1, |m, c, p| m & c & p);
                let x = self.combine_axis(0, |m, c, p| m & c & p);
                z.and(&y).and(&x)
            }
            NearestNeighbors::FaceEdge => {
                let xy = self.min_y().min_x();
                let xz = self.min_z().min_x();
                let yz = self.min_z().min_y();
                xy.and(&xz).and(&yz)
            }
            NearestNeighbors::FaceEdgeVertex => self.min_z().min_y().min_x(),
        }
    }
}

fn leaf_words<T: crate::types::Value>(leaf: &LeafNode<T>) -> Words {
    let mut w = [0u64; 8];
    w.copy_from_slice(leaf.value_mask().words());
    w
}

const BLOCKS: [[i32; 3]; 27] = {
    let mut b = [[0i32; 3]; 27];
    let mut i = 0;
    while i < 27 {
        b[i] = [(i / 9) as i32 - 1, ((i / 3) % 3) as i32 - 1, (i % 3) as i32 - 1];
        i += 1;
    }
    b
};

/// Topologically dilate the active voxels of all leaf nodes `iterations` times
/// (`Morphology::dilateVoxels`). Active tiles neither dilate nor receive dilation (they are
/// already active); new leaves are created as needed from the surrounding tile values.
pub fn dilate_voxels<T: ToolValue>(tree: &mut Tree<T>, iterations: usize, nn: NearestNeighbors, threaded: bool) {
    for _ in 0..iterations {
        let sources: Vec<(Coord, Words)> = tree.leaves().map(|l| (l.origin(), leaf_words(l))).collect();
        if sources.is_empty() {
            return;
        }
        let dilate_one = |&(origin, words): &(Coord, Words)| -> Vec<(Coord, Words)> {
            if words.iter().all(|w| *w == 0) {
                return Vec::new();
            }
            let mut local = Local::new();
            local.set_block([1, 1, 1], &words);
            let d = local.dilate(nn);
            let mut out = Vec::with_capacity(27);
            for b in BLOCKS {
                let bi = [(b[0] + 1) as usize, (b[1] + 1) as usize, (b[2] + 1) as usize];
                if !d.is_block_empty(bi) {
                    out.push((origin.offset_by(8 * b[0], 8 * b[1], 8 * b[2]), d.block(bi)));
                }
            }
            out
        };
        let parts: Vec<Vec<(Coord, Words)>> = if threaded {
            sources.par_iter().map(dilate_one).collect()
        } else {
            sources.iter().map(dilate_one).collect()
        };
        let mut acc: HashMap<Coord, Words> = HashMap::new();
        for (o, w) in parts.into_iter().flatten() {
            let e = acc.entry(o).or_insert([0u64; 8]);
            for i in 0..8 {
                e[i] |= w[i];
            }
        }
        let mut targets: Vec<(Coord, Words)> = acc.into_iter().collect();
        targets.sort_by_key(|(o, _)| *o);
        let mut accessor = tree.accessor_mut();
        for (o, w) in targets {
            if accessor.probe_leaf(&o).is_none() && accessor.is_value_on(&o) {
                continue; // inside an active tile
            }
            let leaf = accessor.touch_leaf(&o);
            for (dst, src) in leaf.value_mask_mut().words_mut().iter_mut().zip(w.iter()) {
                *dst |= *src;
            }
        }
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
        let new_masks: Vec<(Coord, Words)> = {
            let t = &*tree;
            let erode_one = |leaf: &LeafNode<T>| -> (Coord, Words) {
                let origin = leaf.origin();
                let words = leaf_words(leaf);
                if words.iter().all(|w| *w == 0) {
                    return (origin, words);
                }
                let acc = t.accessor();
                let mut local = Local::new();
                for b in BLOCKS {
                    let bi = [(b[0] + 1) as usize, (b[1] + 1) as usize, (b[2] + 1) as usize];
                    if b == [0, 0, 0] {
                        local.set_block(bi, &words);
                        continue;
                    }
                    let o = origin.offset_by(8 * b[0], 8 * b[1], 8 * b[2]);
                    if let Some(n) = acc.probe_leaf(&o) {
                        local.set_block(bi, &leaf_words(n));
                    } else if acc.is_value_on(&o) {
                        local.fill_block(bi);
                    }
                }
                (origin, local.erode(nn).block([1, 1, 1]))
            };
            let leaves = tree.leaf_refs();
            if threaded {
                leaves.par_iter().map(|l| erode_one(l)).collect()
            } else {
                leaves.iter().map(|l| erode_one(l)).collect()
            }
        };
        for (leaf, (_, w)) in tree.leaves_mut().zip(new_masks) {
            leaf.value_mask_mut().words_mut().copy_from_slice(&w);
        }
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
