//! Crate-internal helpers shared by the filtering, level-set and advection tools.
//!
//! These cover the small parts of OpenVDB's `LeafManager` (auxiliary per-leaf buffers),
//! `Morphology.h` (face-neighbour dilation/erosion of voxels), `ChangeBackground.h` and
//! `Prune.h` that the ported tools rely on.

use super::interpolation::AlphaMask;
use crate::grid::Grid;
use crate::math::Coord;
use crate::tree::{InternalNode, LeafNode, Node, Tree};
use crate::types::{ArithValue, RealValue, Value};
use rayon::prelude::*;

/// The six face-neighbour offsets.
pub(crate) const FACE_NEIGHBORS: [[i32; 3]; 6] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];

/// Copies of every leaf buffer, in leaf iteration order (OpenVDB's auxiliary leaf buffers).
pub(crate) struct AuxBuffers<T> {
    bufs: Vec<Vec<Vec<T>>>,
}

impl<T: Value> AuxBuffers<T> {
    /// `count` copies of the current leaf buffers of `tree`.
    pub(crate) fn new(tree: &Tree<T>, count: usize) -> Self {
        let snap: Vec<Vec<T>> = tree.leaf_refs().par_iter().map(|l| l.buffer().to_vec()).collect();
        let mut bufs = Vec::with_capacity(count);
        for _ in 1..count {
            bufs.push(snap.clone());
        }
        if count > 0 {
            bufs.push(snap);
        }
        AuxBuffers { bufs }
    }
    /// Buffer `k` (1-based, like OpenVDB's buffer indices) of leaf `leaf`.
    #[inline]
    pub(crate) fn get(&self, k: usize, leaf: usize) -> &[T] {
        &self.bufs[k - 1][leaf]
    }
    /// Replace auxiliary buffer `k` (1-based).
    pub(crate) fn set(&mut self, k: usize, data: Vec<Vec<T>>) {
        self.bufs[k - 1] = data;
    }
    /// Swap the tree's leaf buffers with auxiliary buffer `k` (`LeafManager::swapLeafBuffer`).
    pub(crate) fn swap_with_tree(&mut self, tree: &mut Tree<T>, k: usize) {
        let leaves = tree.leaf_refs_mut();
        leaves
            .into_par_iter()
            .zip(self.bufs[k - 1].par_iter_mut())
            .for_each(|(l, b)| l.buffer_mut().swap_with_slice(b));
    }
}

/// For every leaf of `tree` (in parallel), start from a copy of `init(leaf_index, leaf)` and
/// let `f` modify it; return the per-leaf results in leaf order. `state` creates per-thread
/// scratch data such as stencils.
pub(crate) fn par_leaf_compute<T, St, I, S, F>(tree: &Tree<T>, state: S, init: I, f: F) -> Vec<Vec<T>>
where
    T: Value,
    S: Fn() -> St + Sync + Send,
    I: Fn(usize, &LeafNode<T>) -> Vec<T> + Sync + Send,
    F: Fn(&mut St, usize, &LeafNode<T>, &mut [T]) + Sync + Send,
{
    let leaves = tree.leaf_refs();
    leaves
        .par_iter()
        .enumerate()
        .map_init(state, |st, (i, leaf)| {
            let mut out = init(i, leaf);
            f(st, i, leaf, &mut out);
            out
        })
        .collect()
}

/// Mask parameters for [`masked_voxel_pass`]: the mask grid, the `[min, max]` range remapped to
/// `[0, 1]`, and whether the mask is inverted.
pub(crate) type MaskSpec<'g, M> = Option<(&'g Grid<M>, f64, f64, bool)>;

/// Compute new values for all active voxels of `grid` in parallel: `kernel(state, xyz, value)`
/// gives the filtered value, which is blended as `b * value + a * filtered` with the weights of
/// the optional alpha mask. Returns new per-leaf buffers (inactive voxels unchanged).
pub(crate) fn masked_voxel_pass<'g, T, M, St, S, K>(
    grid: &'g Grid<T>,
    mask: MaskSpec<'g, M>,
    state: S,
    kernel: K,
) -> Vec<Vec<T>>
where
    T: ArithValue,
    M: RealValue + ArithValue,
    S: Fn(&'g Grid<T>) -> St + Sync + Send,
    K: Fn(&mut St, &Coord, T) -> T + Sync + Send,
{
    par_leaf_compute(
        grid.tree(),
        || (state(grid), mask.map(|(m, lo, hi, inv)| AlphaMask::new(grid.transform(), m, lo, hi, inv))),
        |_, leaf| leaf.buffer().to_vec(),
        |(st, alpha), _, leaf: &LeafNode<T>, out| {
            let origin = leaf.origin();
            for n in leaf.value_mask().iter_on() {
                let xyz = origin + LeafNode::<T>::offset_to_local_coord(n);
                let v = out[n];
                match alpha {
                    Some(am) => {
                        if let Some((a, b)) = am.weights(&xyz) {
                            out[n] = v.scale(b) + kernel(st, &xyz, v).scale(a);
                        }
                    }
                    None => out[n] = kernel(st, &xyz, v),
                }
            }
        },
    )
}

/// Copy per-leaf buffers (in leaf order) into the tree's leaves.
pub(crate) fn write_leaf_buffers<T: Value>(tree: &mut Tree<T>, bufs: &[Vec<T>]) {
    tree.leaf_refs_mut().into_par_iter().zip(bufs.par_iter()).for_each(|(l, b)| l.buffer_mut().copy_from_slice(b));
}

/// Activate the six face neighbours of every active voxel, creating leaves as needed
/// (`dilateActiveValues(tree, 1, NN_FACE, IGNORE_TILES)`). Newly activated voxels keep their
/// previous (inactive) values.
pub(crate) fn dilate_face<T: Value>(tree: &mut Tree<T>, iterations: usize) {
    dilate_neighbors(tree, iterations, &FACE_NEIGHBORS)
}

/// The 18 face and edge neighbour offsets.
pub(crate) fn face_edge_neighbors() -> Vec<[i32; 3]> {
    neighbor_offsets(2)
}

/// The 26 face, edge and vertex neighbour offsets.
pub(crate) fn face_edge_vertex_neighbors() -> Vec<[i32; 3]> {
    neighbor_offsets(3)
}

fn neighbor_offsets(max_nonzero: usize) -> Vec<[i32; 3]> {
    let mut v = Vec::new();
    for i in -1..=1 {
        for j in -1..=1 {
            for k in -1..=1 {
                let nz = [i, j, k].iter().filter(|&&x| x != 0).count();
                if nz > 0 && nz <= max_nonzero {
                    v.push([i, j, k]);
                }
            }
        }
    }
    v
}

/// Activate the given neighbours of every active voxel, `iterations` times
/// (`dilateActiveValues(tree, n, nn, IGNORE_TILES)`).
pub(crate) fn dilate_neighbors<T: Value>(tree: &mut Tree<T>, iterations: usize, offsets: &[[i32; 3]]) {
    for _ in 0..iterations {
        let new: Vec<Coord> = tree
            .leaf_refs()
            .par_iter()
            .flat_map_iter(|leaf| {
                let mut out = Vec::new();
                let origin = leaf.origin();
                for n in leaf.value_mask().iter_on() {
                    let c = origin + LeafNode::<T>::offset_to_local_coord(n);
                    let local = LeafNode::<T>::offset_to_local_coord(n);
                    for o in offsets {
                        let l = local.offset_by(o[0], o[1], o[2]);
                        let inside = (0..3).all(|a| (0..8).contains(&l[a]));
                        if inside {
                            if !leaf.is_value_on_at(LeafNode::<T>::coord_to_offset(&l)) {
                                out.push(c.offset_by(o[0], o[1], o[2]));
                            }
                        } else {
                            out.push(c.offset_by(o[0], o[1], o[2]));
                        }
                    }
                }
                out
            })
            .collect();
        let mut acc = tree.accessor_mut();
        for c in new {
            if !acc.is_value_on(&c) {
                acc.touch_leaf(&c).set_active_state(&c, true);
            }
        }
    }
}

/// Deactivate active voxels that have an inactive face neighbour, `iterations` times
/// (`erodeActiveValues(tree, n, NN_FACE, IGNORE_TILES)`). Values are left unchanged.
pub(crate) fn erode_face<T: Value>(tree: &mut Tree<T>, iterations: usize) {
    for _ in 0..iterations {
        let off: Vec<Coord> = {
            let t: &Tree<T> = tree;
            t.leaf_refs()
                .par_iter()
                .map_init(
                    || t.accessor(),
                    |acc, leaf| {
                        let mut out = Vec::new();
                        for (c, _) in leaf.iter_on_coords() {
                            if FACE_NEIGHBORS.iter().any(|o| !acc.is_value_on(&c.offset_by(o[0], o[1], o[2]))) {
                                out.push(c);
                            }
                        }
                        out
                    },
                )
                .flatten()
                .collect()
        };
        let mut acc = tree.accessor_mut();
        for c in off {
            acc.set_active_state(&c, false);
        }
    }
}

/// Set every inactive value to `-bg` if negative and `bg` otherwise, and the background to
/// `bg` (`changeLevelSetBackground`).
pub(crate) fn change_level_set_background<T: RealValue>(tree: &mut Tree<T>, bg: T) {
    let z = <T as Value>::zero();
    tree.par_for_each_leaf_mut(|leaf| {
        let (buf, mask) = leaf.buffer_and_mask_mut();
        for (n, v) in buf.iter_mut().enumerate() {
            if !mask.is_on(n) {
                *v = if *v < z { -bg } else { bg };
            }
        }
    });
    tree.visit_tiles_mut(&mut |_, v, on, _| {
        if !*on {
            *v = if *v < z { -bg } else { bg };
        }
    });
    tree.root_mut().set_background_unchecked(bg);
}

/// Set inactive values to `inside` if negative and `outside` otherwise, and the background to
/// `outside` (`changeAsymmetricLevelSetBackground`).
pub(crate) fn change_asymmetric_level_set_background<T: RealValue>(tree: &mut Tree<T>, outside: T, inside: T) {
    let z = <T as Value>::zero();
    tree.par_for_each_leaf_mut(|leaf| {
        let (buf, mask) = leaf.buffer_and_mask_mut();
        for (n, v) in buf.iter_mut().enumerate() {
            if !mask.is_on(n) {
                *v = if *v < z { inside } else { outside };
            }
        }
    });
    tree.visit_tiles_mut(&mut |_, v, on, _| {
        if !*on {
            *v = if *v < z { inside } else { outside };
        }
    });
    tree.root_mut().set_background_unchecked(outside);
}

/// Replace leaves without active voxels by inactive tiles of `±background` (by the sign of the
/// leaf's first value) and prune the resulting constant nodes (`pruneLevelSet`).
pub(crate) fn prune_level_set<T: RealValue>(tree: &mut Tree<T>) {
    let bg = *tree.background();
    let z = <T as Value>::zero();
    let empty: Vec<(Coord, bool)> =
        tree.leaves().filter(|l| l.is_empty()).map(|l| (l.origin(), l.buffer()[0] < z)).collect();
    for (origin, inside) in empty {
        tree.add_tile(1, &origin, if inside { -bg } else { bg }, false);
    }
    // Merge uniform inactive tiles into larger tiles; constant active leaves cannot occur in a
    // narrow band, so a zero-tolerance prune only affects inactive regions.
    tree.prune(z);
}

/// A tree with the active topology of `src` (leaves and active tiles) and every value set to
/// `bg` (OpenVDB's `TopologyCopy` constructor, restricted to active topology).
pub(crate) fn topology_copy<T: Value, U: Value>(src: &Tree<T>, bg: U) -> Tree<U> {
    let mut out = Tree::new(bg);
    let mut tiles = Vec::new();
    src.visit_tiles(&mut |b, _, on, level| {
        if on {
            tiles.push((b, level));
        }
    });
    for (b, level) in tiles {
        out.add_tile(level, &b.min, bg, true);
    }
    for leaf in src.leaves() {
        let l = out.touch_leaf(&leaf.origin());
        l.fill_all(bg, false);
        *l.value_mask_mut() = leaf.value_mask().clone();
    }
    out
}

/// A boolean mask of the active voxels of `tree` (tiles ignored).
pub(crate) fn voxel_mask<T: Value>(tree: &Tree<T>) -> Tree<bool> {
    let mut out = Tree::new(false);
    for leaf in tree.leaves() {
        let l = out.touch_leaf(&leaf.origin());
        *l.value_mask_mut() = leaf.value_mask().clone();
    }
    out
}

/// Remove from `a` the active voxels of `b` (`topologyDifference`, voxels only).
pub(crate) fn mask_difference(a: &mut Tree<bool>, b: &Tree<bool>) {
    a.par_for_each_leaf_mut(|leaf| {
        if let Some(other) = b.probe_leaf(&leaf.origin()) {
            leaf.value_mask_mut().difference_with(other.value_mask());
        }
    });
}

/// The voxel size of a grid with uniform voxels, or `None`.
pub(crate) fn uniform_voxel_size<T: Value>(grid: &Grid<T>) -> Option<f64> {
    let vs = grid.voxel_size();
    let tol = 1e-6 * vs[0].abs();
    ((vs[0] - vs[1]).abs() <= tol && (vs[0] - vs[2]).abs() <= tol && grid.transform().has_uniform_scale())
        .then_some(vs[0])
}

/// First and last values of a node (`getFirstValue` / `getLastValue`), descending into
/// children where needed.
pub(crate) trait EndValues {
    type V: Value;
    fn first_value(&self) -> Self::V;
    fn last_value(&self) -> Self::V;
}

impl<T: Value> EndValues for LeafNode<T> {
    type V = T;
    fn first_value(&self) -> T {
        self.buffer()[0]
    }
    fn last_value(&self) -> T {
        self.buffer()[LeafNode::<T>::SIZE - 1]
    }
}

impl<C: Node + EndValues<V = <C as Node>::Value>, const L: u32> EndValues for InternalNode<C, L> {
    type V = C::Value;
    fn first_value(&self) -> C::Value {
        match self.child(0) {
            Some(c) => c.first_value(),
            None => self.values()[0],
        }
    }
    fn last_value(&self) -> C::Value {
        let n = (1usize << (3 * L)) - 1;
        match self.child(n) {
            Some(c) => c.last_value(),
            None => self.values()[n],
        }
    }
}

/// Signed flood fill of the tiles of one internal node (`SignedFloodFillOp::operator()(NodeT&)`).
fn flood_fill_internal<T, C, const L: u32>(node: &mut InternalNode<C, L>, inside: T, outside: T)
where
    T: RealValue,
    C: Node<Value = T> + EndValues<V = T>,
{
    let z = <T as Value>::zero();
    let dim = 1usize << L;
    let num = 1usize << (3 * L);
    let first = node.child_mask().find_first_on();
    let mut new_vals: Vec<(usize, T)> = Vec::new();
    match first {
        Some(first) => {
            let mut x_in = node.child(first).unwrap().first_value() < z;
            for x in 0..dim {
                let x00 = x << (2 * L);
                if let Some(c) = node.child(x00) {
                    x_in = c.last_value() < z;
                }
                let mut y_in = x_in;
                for y in 0..dim {
                    let xy0 = x00 + (y << L);
                    if let Some(c) = node.child(xy0) {
                        y_in = c.last_value() < z;
                    }
                    let mut z_in = y_in;
                    for zz in 0..dim {
                        let xyz = xy0 + zz;
                        match node.child(xyz) {
                            Some(c) => z_in = c.last_value() < z,
                            None => new_vals.push((xyz, if z_in { inside } else { outside })),
                        }
                    }
                }
            }
        }
        None => {
            let v = if node.values()[0] < z { inside } else { outside };
            new_vals.extend((0..num).map(|i| (i, v)));
        }
    }
    let vals = node.values_mut();
    for (i, v) in new_vals {
        vals[i] = v;
    }
}

/// Propagate the sign of the active values of a level set to its inactive voxels and tiles,
/// setting them to `±|background|` (`tools::signedFloodFill`).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // mirrors the C++ NaN-tolerant comparison
pub(crate) fn signed_flood_fill<T: RealValue>(tree: &mut Tree<T>) {
    let outside = tree.background().abs();
    let inside = -outside;
    let z = <T as Value>::zero();
    tree.par_for_each_leaf_mut(|leaf| {
        let (buf, mask) = leaf.buffer_and_mask_mut();
        match mask.find_first_on() {
            Some(first) => {
                let mut x_in = buf[first] < z;
                for x in 0..8usize {
                    let x00 = x << 6;
                    if mask.is_on(x00) {
                        x_in = buf[x00] < z;
                    }
                    let mut y_in = x_in;
                    for y in 0..8usize {
                        let xy0 = x00 + (y << 3);
                        if mask.is_on(xy0) {
                            y_in = buf[xy0] < z;
                        }
                        let mut z_in = y_in;
                        for zz in 0..8usize {
                            let xyz = xy0 + zz;
                            if mask.is_on(xyz) {
                                z_in = buf[xyz] < z;
                            } else {
                                buf[xyz] = if z_in { inside } else { outside };
                            }
                        }
                    }
                }
            }
            None => {
                let v = if buf[0] < z { inside } else { outside };
                buf.iter_mut().for_each(|b| *b = v);
            }
        }
    });
    for n2 in tree.root_mut().children_mut() {
        for n1 in n2.children_mut() {
            flood_fill_internal(n1, inside, outside);
        }
    }
    for n2 in tree.root_mut().children_mut() {
        flood_fill_internal(n2, inside, outside);
    }
    // Root: fill the gaps between inside children along z scanlines.
    let dim = <crate::tree::Internal2<T> as Node>::DIM;
    let keys: Vec<(Coord, T, T)> =
        tree.root().children().map(|c| (c.origin(), c.first_value(), c.last_value())).collect();
    for w in keys.windows(2) {
        let (a, _, a_last) = w[0];
        let (b, b_first, _) = w[1];
        let d = b - a;
        if d[0] != 0 || d[1] != 0 || d[2] == dim {
            continue;
        }
        if !(a_last < z) || !(b_first < z) {
            continue;
        }
        let mut c = a.offset_by(0, 0, dim);
        while c[2] != b[2] {
            tree.root_mut().add_root_tile(&c, inside, false);
            c[2] += dim;
        }
    }
    tree.root_mut().set_background_unchecked(outside);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::CoordBBox;

    #[test]
    fn dilate_erode_roundtrip() {
        let mut tree = Tree::<f32>::new(1.0);
        tree.set_value(&Coord::new(7, 7, 7), 0.0);
        dilate_face(&mut tree, 1);
        assert_eq!(tree.active_voxel_count(), 7);
        assert!(tree.is_value_on(&Coord::new(8, 7, 7)));
        assert_eq!(tree.get_value(&Coord::new(8, 7, 7)), 1.0);
        dilate_face(&mut tree, 1);
        assert_eq!(tree.active_voxel_count(), 25);
        erode_face(&mut tree, 2);
        assert_eq!(tree.active_voxel_count(), 1);
    }

    #[test]
    fn background_and_prune() {
        let mut tree = Tree::<f32>::new(2.0);
        tree.fill(&CoordBBox::from_coords(0, 0, 0, 15, 15, 15), -2.0, false);
        tree.voxelize_active_tiles();
        tree.set_value_off(&Coord::new(20, 0, 0), 2.0);
        tree.set_value_off(&Coord::new(-20, 0, 0), -2.0);
        tree.set_value(&Coord::new(40, 0, 0), 0.5);
        change_level_set_background(&mut tree, 3.0);
        assert_eq!(tree.get_value(&Coord::new(-20, 0, 0)), -3.0);
        assert_eq!(tree.get_value(&Coord::new(20, 0, 0)), 3.0);
        assert_eq!(tree.get_value(&Coord::new(3, 3, 3)), -3.0);
        assert_eq!(*tree.background(), 3.0);
        prune_level_set(&mut tree);
        assert_eq!(tree.leaf_count(), 1);
        // The sign of a pruned leaf is taken from its first voxel.
        assert_eq!(tree.get_value(&Coord::new(-20, 0, 0)), 3.0);
        assert_eq!(tree.get_value(&Coord::new(3, 3, 3)), -3.0);
        assert_eq!(tree.get_value(&Coord::new(40, 0, 0)), 0.5);
        let topo = topology_copy(&tree, 0i32);
        assert_eq!(topo.active_voxel_count(), 1);
    }
}
