//! Utilities for level sets and signed distance fields (port of
//! `openvdb/tools/LevelSetUtil.h`):
//!
//! * [`sdf_to_fog_volume`] converts a level set into a fog volume,
//! * [`sdf_interior_mask`] masks the interior of an SDF,
//! * [`extract_enclosed_region`] masks the interior plus enclosed cavities,
//! * [`extract_isosurface_mask`] masks the voxels that straddle an isosurface,
//! * [`extract_active_voxel_segment_masks`], [`segment_active_voxels`] and [`segment_sdf`]
//!   split a volume into its connected components.
//!
//! `distanceFieldToSDF` (which relies on the mesh-to-volume pipeline) is not ported.

use crate::grid::{Grid, GridClass};
use crate::math::{Coord, CoordBBox};
use crate::tools::prune::prune_inactive;
use crate::tools::signed_flood_fill::{signed_flood_fill, signed_flood_fill_with_values};
use crate::tools::tree_ops::{
    has_active_tiles, internal1_nodes, internal1_nodes_mut, internal2_nodes_mut, steal_leaves, topology_copy,
    topology_intersection,
};
use crate::tools::value_traits::{ToolValue, TreeAdapter};
use crate::tree::{Internal1, LeafNode, Node, Tree};
use crate::types::{RealValue, Value};
use crate::util::NodeMask;
use rayon::prelude::*;
use std::collections::HashMap;

type BoolTree = Tree<bool>;

// ---------------------------------------------------------------------------------------
// sdfToFogVolume
// ---------------------------------------------------------------------------------------

/// Convert a level set (or other SDF) into a fog volume, in place (`tools::sdfToFogVolume`).
///
/// The negative interior part of the narrow band becomes a linear ramp from 0 (at the
/// surface) to 1 (at `-cutoff_distance`, clamped to the most negative value of the SDF),
/// the inactive interior becomes active with value 1, and the exterior becomes inactive
/// with value 0. Pass `T::max_value()` for OpenVDB's default (the full interior band).
pub fn sdf_to_fog_volume<T: RealValue + ToolValue>(grid: &mut Grid<T>, cutoff_distance: T) {
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let old = grid.tree_mut();
    let background = *old.background();

    // Steal all leaves (leaving background tiles behind) and find the minimum SDF value.
    let leaves = steal_leaves(old, background);
    let mut min_sdf = T::max_value();
    for n1 in internal1_nodes(old) {
        for (_, v, _) in n1.tiles() {
            if *v < min_sdf {
                min_sdf = *v;
            }
        }
    }
    if min_sdf > zero {
        for l in &leaves {
            for v in l.buffer() {
                if *v < min_sdf {
                    min_sdf = *v;
                }
            }
        }
    }
    let mut cutoff = -cutoff_distance.abs();
    if min_sdf > cutoff {
        cutoff = min_sdf;
    }
    let weight = one / cutoff;

    // Transform the voxel values, dropping leaves that become empty.
    let fog_leaves: Vec<Box<LeafNode<T>>> = leaves
        .into_par_iter()
        .filter_map(|mut leaf| {
            leaf.set_values_on_off(false);
            let (buf, mask) = leaf.buffer_and_mask_mut();
            for (i, v) in buf.iter_mut().enumerate() {
                *v = if *v > zero { zero } else { *v * weight };
                if *v > zero {
                    mask.set_on(i);
                }
            }
            (!leaf.is_empty()).then_some(leaf)
        })
        .collect();

    let mut new_tree = Tree::new(zero);
    for leaf in fog_leaves {
        new_tree.add_leaf(leaf);
    }
    let old = &*old;
    {
        let acc = old.accessor();
        let mut n1s = internal1_nodes_mut(&mut new_tree);
        n1s.par_iter_mut().for_each(|n1| {
            let acc = old.accessor();
            for n in 0..Internal1::<T>::NUM_VALUES {
                if !n1.is_child(n) && acc.get_value(&n1.offset_to_global_coord(n)) < zero {
                    n1.set_tile(n, one, true);
                }
            }
        });
        for n2 in internal2_nodes_mut(&mut new_tree) {
            for n in 0..crate::tree::Internal2::<T>::NUM_VALUES {
                if !n2.is_child(n) && acc.get_value(&n2.offset_to_global_coord(n)) < zero {
                    n2.set_tile(n, one, true);
                }
            }
        }
    }
    for (k, v, _) in old.root().tiles() {
        if *v < zero {
            new_tree.add_tile(3, &k, one, true);
        }
    }
    grid.set_tree(new_tree);
    grid.set_grid_class(GridClass::FogVolume);
}

// ---------------------------------------------------------------------------------------
// Interior masks
// ---------------------------------------------------------------------------------------

/// `computeInteriorMask` with an already clamped isovalue.
fn compute_interior_mask<T: ToolValue>(tree: &Tree<T>, iso: T) -> BoolTree {
    let leaves: Vec<Box<LeafNode<bool>>> = tree
        .leaf_refs()
        .par_iter()
        .filter_map(|l| {
            let mut m = LeafNode::new(l.origin(), false, false);
            for (i, v) in l.buffer().iter().enumerate() {
                if v.cwise_lt(&iso) {
                    m.set_value_on_at(i, true);
                }
            }
            (!m.is_empty()).then(|| Box::new(m))
        })
        .collect();
    let mut mask = BoolTree::new(false);
    for l in leaves {
        mask.add_leaf(l);
    }
    let acc = tree.accessor();
    let mut n1s = internal1_nodes_mut(&mut mask);
    n1s.par_iter_mut().for_each(|n1| {
        let acc = tree.accessor();
        for n in 0..Internal1::<bool>::NUM_VALUES {
            if !n1.is_child(n) && acc.get_value(&n1.offset_to_global_coord(n)).cwise_lt(&iso) {
                n1.set_tile(n, true, true);
            }
        }
    });
    for n2 in internal2_nodes_mut(&mut mask) {
        for n in 0..crate::tree::Internal2::<bool>::NUM_VALUES {
            if !n2.is_child(n) && acc.get_value(&n2.offset_to_global_coord(n)).cwise_lt(&iso) {
                n2.set_tile(n, true, true);
            }
        }
    }
    mask
}

/// Return a boolean mask whose active (and `true`) values are the voxels of `volume` whose
/// value is below `isovalue` (`tools::sdfInteriorMask`). The isovalue is clamped to just
/// below the background value.
pub fn sdf_interior_mask<T: RealValue + ToolValue, G: TreeAdapter<T>>(volume: &G, isovalue: T) -> G::Rebind<bool> {
    let tree = volume.tree_ref();
    let clamp = *tree.background() - T::tolerance();
    let iso = if clamp < isovalue { clamp } else { isovalue };
    volume.rebind_transform(compute_interior_mask(tree, iso))
}

// ---------------------------------------------------------------------------------------
// Enclosed regions
// ---------------------------------------------------------------------------------------

/// Leaf connectivity used by [`trace_exterior_boundaries`]: for each leaf, the index of the
/// next leaf in the -/+ x, y, z directions (skipping empty space within the leaf bounds).
struct Connectivity {
    /// `[axis][0 = prev, 1 = next]`.
    offsets: Vec<[[Option<usize>; 2]; 3]>,
}

impl Connectivity {
    fn new<T: Value>(origins: &[Coord], tree: &Tree<T>) -> Self {
        let index: HashMap<Coord, usize> = origins.iter().enumerate().map(|(i, o)| (*o, i)).collect();
        let mut bbox = CoordBBox::empty();
        for l in tree.leaves() {
            if !l.is_empty() {
                bbox.expand_bbox(&l.node_bbox());
            }
        }
        let offsets = origins
            .par_iter()
            .map(|o| {
                let mut r = [[None; 2]; 3];
                for (axis, ra) in r.iter_mut().enumerate() {
                    for (dir, step) in [-8, 8].into_iter().enumerate() {
                        let mut s = Coord::origin();
                        s[axis] = step;
                        let mut ijk = *o + s;
                        while bbox.is_inside(&ijk) {
                            if let Some(&i) = index.get(&ijk) {
                                ra[dir] = Some(i);
                                break;
                            }
                            ijk += s;
                        }
                    }
                }
                r
            })
            .collect();
        Connectivity { offsets }
    }
}

/// Port of `tools::traceExteriorBoundaries` for the `char` region masks used by
/// [`extract_enclosed_region`]: starting from the outer faces of the leaf configuration,
/// flip the sign of every exterior voxel (value 1) that can be reached without crossing
/// a boundary voxel (value 0). Negative voxels also act as sources.
fn trace_exterior_boundaries(tree: &mut Tree<i8>) {
    let origins: Vec<Coord> = tree.leaves().map(|l| l.origin()).collect();
    if origins.is_empty() {
        return;
    }
    let conn = Connectivity::new(&origins, tree);
    let mut data: Vec<Vec<i8>> = tree.leaves().map(|l| l.buffer().to_vec()).collect();
    let stride = [64usize, 8, 1];
    let mut queue: Vec<(usize, usize)> = Vec::new();

    // Seeds: existing negative voxels, and the first voxel of every voxel line entering
    // the leaf configuration from outside along each axis (both directions).
    for (n, d) in data.iter().enumerate() {
        for (pos, v) in d.iter().enumerate() {
            if *v < 0 {
                queue.push((n, pos));
            }
        }
    }
    for n in 0..data.len() {
        for axis in 0..3 {
            for dir in 0..2 {
                if conn.offsets[n][axis][dir].is_some() {
                    continue;
                }
                let (a_axis, b_axis) = match axis {
                    0 => (1, 2),
                    1 => (0, 2),
                    _ => (0, 1),
                };
                for a in 0..8 {
                    for b in 0..8 {
                        let mut l = [0usize; 3];
                        l[a_axis] = a;
                        l[b_axis] = b;
                        l[axis] = if dir == 0 { 0 } else { 7 };
                        let pos = l[0] * 64 + l[1] * 8 + l[2];
                        if data[n][pos] > 0 {
                            data[n][pos] = -data[n][pos];
                            queue.push((n, pos));
                        }
                    }
                }
            }
        }
    }
    // Flood fill through exterior voxels.
    while let Some((n, pos)) = queue.pop() {
        let l = [pos / 64, (pos / 8) % 8, pos % 8];
        for axis in 0..3 {
            for dir in 0..2 {
                let (nn, npos) = if dir == 0 {
                    if l[axis] > 0 {
                        (n, pos - stride[axis])
                    } else if let Some(p) = conn.offsets[n][axis][0] {
                        (p, pos + 7 * stride[axis])
                    } else {
                        continue;
                    }
                } else if l[axis] < 7 {
                    (n, pos + stride[axis])
                } else if let Some(p) = conn.offsets[n][axis][1] {
                    (p, pos - 7 * stride[axis])
                } else {
                    continue;
                };
                if data[nn][npos] > 0 {
                    data[nn][npos] = -data[nn][npos];
                    queue.push((nn, npos));
                }
            }
        }
    }
    for (leaf, d) in tree.leaves_mut().zip(data) {
        leaf.buffer_mut().copy_from_slice(&d);
    }
}

/// Boundary voxels of the fill mask (port of `FillMaskBoundary`): active mask voxels with
/// an inactive face neighbor whose distance is above the isovalue.
fn fill_mask_boundary<T: ToolValue>(
    tree: &Tree<T>,
    iso: T,
    fill_mask: &BoolTree,
    node: &LeafNode<bool>,
) -> Option<Box<LeafNode<bool>>> {
    let mask_acc = fill_mask.accessor();
    let dist_acc = tree.accessor();
    let origin = node.origin();
    let dense = node.is_dense();
    if dense {
        let dense_neighbors = [(-1, 0, 0), (8, 0, 0), (0, -1, 0), (0, 8, 0), (0, 0, -1), (0, 0, 8)]
            .iter()
            .filter(|&&(x, y, z)| mask_acc.probe_leaf(&origin.offset_by(x, y, z)).is_some_and(|l| l.is_dense()))
            .count();
        if dense_neighbors == 6 {
            return None;
        }
    }
    let mut vm = [0u8; 512];
    let on = |p: usize| node.is_value_on_at(p);
    let dist_node = dist_acc.probe_leaf(&origin);
    let mut early = false;
    if !dense {
        if let Some(dn) = dist_node {
            // Internal neighbors in the positive and negative directions.
            for (stride, positive) in [(1usize, true), (8, true), (64, true), (1, false), (8, false), (64, false)] {
                for pos in 0..512usize {
                    let l = [pos / 64, (pos / 8) % 8, pos % 8];
                    let axis = match stride {
                        1 => 2,
                        8 => 1,
                        _ => 0,
                    };
                    if (positive && l[axis] == 7) || (!positive && l[axis] == 0) {
                        continue;
                    }
                    if vm[pos] != 0 || !on(pos) {
                        continue;
                    }
                    let np = if positive { pos + stride } else { pos - stride };
                    if !on(np) && iso.cwise_lt(dn.get_value_at(np)) {
                        vm[pos] = 1;
                    }
                }
            }
        } else if iso.cwise_lt(&dist_acc.get_value(&origin)) {
            'outer: for (stride, positive) in
                [(1usize, true), (8, true), (64, true), (1, false), (8, false), (64, false)]
            {
                let axis = match stride {
                    1 => 2,
                    8 => 1,
                    _ => 0,
                };
                for pos in 0..512usize {
                    let l = [pos / 64, (pos / 8) % 8, pos % 8];
                    if (positive && l[axis] == 7) || (!positive && l[axis] == 0) {
                        continue;
                    }
                    let np = if positive { pos + stride } else { pos - stride };
                    if on(pos) && !on(np) {
                        vm[pos] = 1;
                        early = true;
                        break 'outer;
                    }
                }
                if early {
                    break;
                }
            }
        }
    }
    if !early {
        for axis in 0..3 {
            for up in [true, false] {
                let face = if up { 7 } else { 0 };
                for a in 0..8usize {
                    for b in 0..8usize {
                        let mut l = [0usize; 3];
                        l[axis] = face;
                        let (ia, ib) = match axis {
                            0 => (1, 2),
                            1 => (0, 2),
                            _ => (0, 1),
                        };
                        l[ia] = a;
                        l[ib] = b;
                        let pos = l[0] * 64 + l[1] * 8 + l[2];
                        if vm[pos] == 0 && on(pos) {
                            let mut d = Coord::origin();
                            d[axis] = if up { 1 } else { -1 };
                            let nijk = origin + Coord::new(l[0] as i32, l[1] as i32, l[2] as i32) + d;
                            if !mask_acc.is_value_on(&nijk) && iso.cwise_lt(&dist_acc.get_value(&nijk)) {
                                vm[pos] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    if vm.contains(&1) {
        let mut leaf = LeafNode::new(origin, false, false);
        for (i, v) in vm.iter().enumerate() {
            if *v == 1 {
                leaf.set_value_on_at(i, true);
            }
        }
        Some(Box::new(leaf))
    } else {
        None
    }
}

/// `computeEnclosedRegionMask`: a `char` tree that is +1 in the exterior and -1 in the
/// interior and enclosed regions.
fn compute_enclosed_region_mask<T: ToolValue>(tree: &Tree<T>, iso: T, fill_mask: Option<&BoolTree>) -> Tree<i8> {
    let leaves: Vec<Box<LeafNode<i8>>> = tree
        .leaf_refs()
        .par_iter()
        .filter_map(|l| {
            let mut m = LeafNode::new(l.origin(), 1i8, false);
            for (pos, v) in l.iter_on() {
                m.set_value_on_at(pos, if v.difference(iso).is_negative() { 0 } else { 1 });
            }
            (!m.is_empty()).then(|| Box::new(m))
        })
        .collect();
    let mut mask = Tree::<i8>::new(1);
    let primary: Vec<Coord> = leaves.iter().map(|l| l.origin()).collect();
    for l in leaves {
        mask.add_leaf(l);
    }
    let mut extra: Vec<Coord> = Vec::new();
    if let Some(fill) = fill_mask {
        let fill_leaves = fill.leaf_refs();
        let boundary: Vec<Option<Box<LeafNode<bool>>>> =
            fill_leaves.par_iter().map(|n| fill_mask_boundary(tree, iso, fill, n)).collect();
        for b in boundary.into_iter().flatten() {
            let o = b.origin();
            if mask.probe_leaf(&o).is_none() {
                extra.push(o);
            }
            let leaf = mask.touch_leaf(&o);
            for (pos, _) in b.iter_on() {
                if *leaf.get_value_at(pos) != 0 {
                    leaf.set_value_only_at(pos, -1);
                }
            }
        }
    }
    trace_exterior_boundaries(&mut mask);
    let flip = |leaf: &mut LeafNode<i8>| {
        for v in leaf.buffer_mut() {
            *v = if *v < 0 { 1 } else { -1 };
        }
    };
    for o in primary.iter().chain(extra.iter()) {
        if let Some(l) = mask.probe_leaf_mut(o) {
            flip(l);
        }
    }
    let _ = signed_flood_fill(&mut mask, true);
    mask
}

/// Return a boolean mask of the interior of `volume` together with the regions it
/// encloses (`tools::extractEnclosedRegion`), e.g. the walls and the cavity of a capped
/// bottle. If `fill_mask` is given, enclosed cavities that it does not completely fill are
/// ignored.
pub fn extract_enclosed_region<T: ToolValue, G: TreeAdapter<T>>(
    volume: &G,
    isovalue: T,
    fill_mask: Option<&BoolTree>,
) -> G::Rebind<bool> {
    let region = compute_enclosed_region_mask(volume.tree_ref(), isovalue, fill_mask);
    // The region mask has background 1 and no tolerance, so the clamped isovalue is 0.
    volume.rebind_transform(compute_interior_mask(&region, 0i8))
}

// ---------------------------------------------------------------------------------------
// Isosurface mask
// ---------------------------------------------------------------------------------------

/// Return a boolean mask of the active voxels of `volume` that have a face neighbor on the
/// other side of `isovalue` (`tools::extractIsosurfaceMask`).
pub fn extract_isosurface_mask<T: ToolValue, G: TreeAdapter<T>>(volume: &G, isovalue: T) -> G::Rebind<bool> {
    volume.rebind_transform(isosurface_mask(volume.tree_ref(), isovalue))
}

fn isosurface_mask<T: ToolValue>(tree: &Tree<T>, iso: T) -> BoolTree {
    let leaves: Vec<Box<LeafNode<bool>>> = tree
        .leaf_refs()
        .par_iter()
        .filter_map(|leaf| {
            let acc = tree.accessor();
            let mut m = LeafNode::new(leaf.origin(), false, false);
            let mut any = false;
            for (pos, v) in leaf.iter_on() {
                let under = v.cwise_lt(&iso);
                let ijk = leaf.offset_to_global_coord(pos);
                let change = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
                    .iter()
                    .any(|&(x, y, z)| under != acc.get_value(&ijk.offset_by(x, y, z)).cwise_lt(&iso));
                if change {
                    m.set_value_on_at(pos, true);
                    any = true;
                }
            }
            any.then(|| Box::new(m))
        })
        .collect();
    let mut mask = BoolTree::new(false);
    for l in leaves {
        mask.add_leaf(l);
    }
    mask
}

// ---------------------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------------------

/// Split a leaf mask into face-connected components (`nodeMaskSegmentation`).
fn node_mask_segments(mask: &NodeMask<3>) -> Vec<NodeMask<3>> {
    let mut remaining = mask.clone();
    let mut segments = Vec::new();
    while let Some(first) = remaining.find_first_on() {
        let mut seg = NodeMask::<3>::new();
        let mut stack = vec![first];
        remaining.set_off(first);
        while let Some(pos) = stack.pop() {
            if seg.is_on(pos) {
                continue;
            }
            seg.set_on(pos);
            let l = [pos >> 6, (pos >> 3) & 7, pos & 7];
            let mut try_push = |np: usize| {
                if remaining.is_on(np) {
                    remaining.set_off(np);
                    stack.push(np);
                }
            };
            if l[2] != 0 {
                try_push(pos - 1);
            }
            if l[2] != 7 {
                try_push(pos + 1);
            }
            if l[1] != 0 {
                try_push(pos - 8);
            }
            if l[1] != 7 {
                try_push(pos + 8);
            }
            if l[0] != 0 {
                try_push(pos - 64);
            }
            if l[0] != 7 {
                try_push(pos + 64);
            }
        }
        segments.push(seg);
    }
    segments
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Face-connected components of the active voxels of `tree`, as boolean mask trees sorted
/// by decreasing active voxel count.
fn active_voxel_segment_masks<T: Value>(tree: &Tree<T>) -> Vec<BoolTree> {
    let mut topology: BoolTree = topology_copy(tree, false);
    prune_inactive(&mut topology, true);
    if has_active_tiles(&topology) {
        topology.voxelize_active_tiles();
    }
    let leaves: Vec<(Coord, NodeMask<3>)> = topology.leaves().map(|l| (l.origin(), l.value_mask().clone())).collect();
    if leaves.is_empty() {
        return Vec::new();
    }
    let segs: Vec<Vec<NodeMask<3>>> = leaves.par_iter().map(|(_, m)| node_mask_segments(m)).collect();
    let mut base = Vec::with_capacity(segs.len());
    let mut total = 0usize;
    for s in &segs {
        base.push(total);
        total += s.len();
    }
    let index: HashMap<Coord, usize> = leaves.iter().enumerate().map(|(i, (o, _))| (*o, i)).collect();
    let seg_of = |leaf: usize, pos: usize| -> Option<usize> {
        segs[leaf].iter().position(|m| m.is_on(pos)).map(|s| base[leaf] + s)
    };
    // Connect segments across leaf faces.
    let mut parent: Vec<usize> = (0..total).collect();
    let mut links: Vec<Vec<usize>> = vec![Vec::new(); total];
    for (n, (origin, mask)) in leaves.iter().enumerate() {
        for (axis, stride) in [(0usize, 64usize), (1, 8), (2, 1)] {
            for (dir, delta) in [(0, -8i32), (1, 8)] {
                let mut d = Coord::origin();
                d[axis] = delta;
                let Some(&nb) = index.get(&(*origin + d)) else { continue };
                for pos in mask.iter_on() {
                    let l = [pos >> 6, (pos >> 3) & 7, pos & 7];
                    let npos = if dir == 0 {
                        if l[axis] != 0 {
                            continue;
                        }
                        pos + 7 * stride
                    } else {
                        if l[axis] != 7 {
                            continue;
                        }
                        pos - 7 * stride
                    };
                    if leaves[nb].1.is_on(npos) {
                        let a = seg_of(n, pos).unwrap();
                        let b = seg_of(nb, npos).unwrap();
                        if !links[a].contains(&b) {
                            links[a].push(b);
                        }
                    }
                }
            }
        }
    }
    for (a, ls) in links.iter().enumerate() {
        for &b in ls {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    // Group discovery order of OpenVDB: start with the first segment, then repeatedly with
    // the *last* unvisited segment.
    let roots: Vec<usize> = (0..total).map(|i| find(&mut parent, i)).collect();
    let mut group_order: Vec<usize> = Vec::new();
    let mut visited_root: HashMap<usize, usize> = HashMap::new();
    let mut next = Some(0usize);
    while let Some(s) = next {
        let r = roots[s];
        visited_root.insert(r, group_order.len());
        group_order.push(r);
        next = (0..total).rev().find(|&i| !visited_root.contains_key(&roots[i]));
    }
    let mut masks: Vec<BoolTree> = if group_order.len() == 1 {
        let mut m: BoolTree = topology_copy(tree, false);
        prune_inactive(&mut m, true);
        if has_active_tiles(&m) {
            m.voxelize_active_tiles();
        }
        vec![m]
    } else {
        let mut ms: Vec<BoolTree> = (0..group_order.len()).map(|_| BoolTree::new(false)).collect();
        for (n, (origin, _)) in leaves.iter().enumerate() {
            for (s, seg) in segs[n].iter().enumerate() {
                let g = visited_root[&roots[base[n] + s]];
                let leaf = ms[g].touch_leaf(origin);
                leaf.value_mask_mut().union_with(seg);
            }
        }
        ms
    };
    masks.sort_by_key(|m| std::cmp::Reverse(m.active_voxel_count()));
    masks
}

/// Return a boolean mask for each face-connected component of the active voxels of
/// `volume`, sorted by decreasing active voxel count
/// (`tools::extractActiveVoxelSegmentMasks`). Active tiles are voxelized first.
pub fn extract_active_voxel_segment_masks<T: Value, G: TreeAdapter<T>>(volume: &G) -> Vec<G::Rebind<bool>> {
    active_voxel_segment_masks(volume.tree_ref()).into_iter().map(|m| volume.rebind_transform(m)).collect()
}

/// `MaskedCopy`: copy the values of `input` at the active voxels of `mask`.
fn masked_copy<T: Value>(input: &Tree<T>, mask: &BoolTree) -> Tree<T> {
    let mut out = Tree::new(*input.background());
    let acc = input.accessor();
    for mleaf in mask.leaves() {
        if mleaf.is_empty() {
            continue;
        }
        let ijk = mleaf.origin();
        if let Some(ileaf) = acc.probe_leaf(&ijk) {
            let o = out.touch_leaf(&ijk);
            for (pos, _) in mleaf.iter_on() {
                o.set_value_on_at(pos, *ileaf.get_value_at(pos));
            }
        } else {
            let depth = acc.get_value_depth(&ijk);
            if depth >= 0 {
                out.add_tile(3 - depth as u32, &ijk, acc.get_value(&ijk), true);
            }
        }
    }
    out
}

/// Split `volume` into its face-connected components of active voxels (active tiles are
/// supported), sorted by decreasing active voxel count (`tools::segmentActiveVoxels`).
/// A volume without active voxels yields a single empty copy.
pub fn segment_active_voxels<T: Value, G: TreeAdapter<T>>(volume: &G) -> Vec<G::Rebind<T>> {
    let input = volume.tree_ref();
    let masks = active_voxel_segment_masks(input);
    let segments: Vec<Tree<T>> = if masks.is_empty() {
        vec![Tree::new(*input.background())]
    } else if masks.len() == 1 {
        let mut seg = input.clone();
        if seg.leaf_count() != input.leaf_count() {
            topology_intersection(&mut seg, &masks[0]);
        }
        vec![seg]
    } else {
        masks.par_iter().map(|m| masked_copy(input, m)).collect()
    };
    segments.into_iter().map(|s| volume.rebind(s)).collect()
}

/// `ExpandNarrowbandMask`: grow `mask` into active voxels of `dist` whose absolute distance
/// increases away from the mask.
fn expand_narrowband_mask<T: ToolValue>(dist: &Tree<T>, mask: &mut BoolTree) {
    let dacc = dist.accessor();
    let mut stack: Vec<Coord> = mask.active_voxels().map(|(c, _)| c).collect();
    let mut macc = mask.accessor_mut();
    while let Some(p) = stack.pop() {
        let Some(pleaf) = dacc.probe_leaf(&p) else { continue };
        let dp = pleaf.get_value(&p).abs_value();
        for (x, y, z) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
            let q = p.offset_by(x, y, z);
            if macc.is_value_on(&q) {
                continue;
            }
            let Some(qleaf) = dacc.probe_leaf(&q) else { continue };
            if qleaf.is_value_on(&q) && qleaf.get_value(&q).abs_value().cwise_gt(&dp) {
                macc.set_value_on(&q, false);
                stack.push(q);
            }
        }
    }
}

/// Split a signed distance field into its disjoint surfaces, sorted by decreasing number
/// of surface-intersecting voxels (`tools::segmentSDF`). Each segment contains the narrow
/// band around one connected component of the zero crossing, with its interior
/// flood-filled.
pub fn segment_sdf<T: RealValue + ToolValue, G: TreeAdapter<T>>(volume: &G) -> Vec<G::Rebind<T>> {
    let input = volume.tree_ref();
    let mask = isosurface_mask(input, T::from_f64(0.0));
    let mut masks = active_voxel_segment_masks(&mask);
    let segments: Vec<Tree<T>> = if masks.is_empty() {
        vec![Tree::new(*input.background())]
    } else {
        masks.par_iter_mut().for_each(|m| expand_narrowband_mask(input, m));
        let mut segs: Vec<Tree<T>> = masks.par_iter().map(|m| masked_copy(input, m)).collect();
        // FloodFillSign: the interior value is the most negative value of the input.
        let mut min_sdf = T::max_value();
        let n1s = internal1_nodes(input);
        for n1 in &n1s {
            for (_, v, _) in n1.tiles() {
                if *v < min_sdf {
                    min_sdf = *v;
                }
            }
        }
        if min_sdf > T::from_f64(0.0) {
            for l in input.leaves() {
                for v in l.buffer() {
                    if *v < min_sdf {
                        min_sdf = *v;
                    }
                }
            }
        }
        let interior = -min_sdf.abs();
        let exterior = input.background().abs();
        segs.par_iter_mut().for_each(|s| {
            let _ = signed_flood_fill_with_values(s, exterior, interior, false);
        });
        segs
    };
    segments.into_iter().map(|s| volume.rebind(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3f;
    use crate::tools::composite::csg_union;
    use crate::tools::level_set_sphere::create_level_set_sphere;

    fn two_spheres() -> Grid<f32> {
        let mut a = create_level_set_sphere::<f32>(1.0, Vec3f::new(0.0, 0.0, 0.0), 0.1, 3.0, true).unwrap();
        let mut b = create_level_set_sphere::<f32>(1.0, Vec3f::new(3.0, 0.0, 0.0), 0.1, 3.0, true).unwrap();
        csg_union(&mut a, &mut b, true, false).unwrap();
        a
    }

    #[test]
    fn interior_mask_and_fog() {
        let g = create_level_set_sphere::<f32>(1.0, Vec3f::new(0.0, 0.0, 0.0), 0.1, 3.0, true).unwrap();
        let mask = sdf_interior_mask(&g, 0.0);
        assert!(mask.get_value(&Coord::new(0, 0, 0)));
        assert!(mask.tree().is_value_on(&Coord::new(5, 0, 0)));
        assert!(!mask.tree().is_value_on(&Coord::new(10, 0, 0)));
        let enclosed = extract_enclosed_region(&g, 0.0, None);
        assert_eq!(enclosed.active_voxel_count(), mask.active_voxel_count());
        let mut fog = g.clone();
        sdf_to_fog_volume(&mut fog, f32::MAX);
        assert_eq!(fog.grid_class(), GridClass::FogVolume);
        assert_eq!(fog.get_value(&Coord::new(0, 0, 0)), 1.0);
        assert_eq!(fog.get_value(&Coord::new(11, 0, 0)), 0.0);
        let v = fog.get_value(&Coord::new(9, 0, 0));
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn segmentation() {
        let g = two_spheres();
        let masks = extract_active_voxel_segment_masks(&g);
        assert_eq!(masks.len(), 2);
        let segs = segment_active_voxels(&g);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].active_voxel_count() + segs[1].active_voxel_count(), g.active_voxel_count());
        let sdf_segs = segment_sdf(&g);
        assert_eq!(sdf_segs.len(), 2);
        for s in &sdf_segs {
            assert!(s.get_value(&Coord::new(0, 0, 0)) < 0.0 || s.get_value(&Coord::new(30, 0, 0)) < 0.0);
        }
        let iso = extract_isosurface_mask(&g, 0.0);
        assert!(iso.active_voxel_count() > 0);
    }
}
