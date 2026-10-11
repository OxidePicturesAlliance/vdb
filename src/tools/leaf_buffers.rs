//! Crate-internal helpers emulating the auxiliary-buffer parts of OpenVDB's
//! `tree::LeafManager`, shared by the filtering, level-set and advection tools: per-leaf
//! buffer copies, parallel per-leaf computation and masked voxel passes.

use super::interpolation::AlphaMask;
use crate::grid::Grid;
use crate::math::Coord;
use crate::tree::{LeafNode, Node, Tree};
use crate::types::{ArithValue, RealValue, Value};
use rayon::prelude::*;

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
