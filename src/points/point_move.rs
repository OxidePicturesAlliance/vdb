//! Moving points with a deformer (port of `openvdb/points/PointMove.h`).
//!
//! [`move_points`] applies a [`PointDeformer`] to every point accepted by a filter (points
//! that are rejected, or that lie in inactive voxels, are deleted), and rebuilds the tree so
//! that every point is stored in the voxel and leaf containing its new position, carrying all
//! of its attributes along. Point order is deterministic: within a voxel, points arriving
//! from other leaves come first (ordered by source leaf and index), then the points that
//! stayed in the same leaf.

use super::attribute_array::AttributeHandle;
use super::attribute_set::AttributeSet;
use super::index_filter::IndexFilter;
use super::index_iterator::{IndexState, VoxelSelection};
use super::point_data_grid::{PointDataGrid, PointDataLeaf, PointDataLeafNode, PointDataTree, PointLeaf, PointLeafMut};
use super::point_mask::{convert_points_to_scalar_transformed, PointDeformer};
use super::{Error, Result};
use crate::math::{Coord, Transform, Vec3d, Vec3f};
use crate::tree::LeafNode;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Move the points of `grid` with `deformer`, keeping the grid's transform
/// (`movePoints(points, deformer, filter)`).
pub fn move_points<D: PointDeformer, F: IndexFilter>(grid: &mut PointDataGrid, deformer: &D, filter: &F) -> Result<()> {
    let t = grid.transform().clone();
    move_points_to_transform(grid, &t, deformer, filter)
}

/// One point moving into a target leaf.
#[derive(Clone, Copy)]
struct Move {
    source_leaf: usize,
    source_index: u32,
    voxel: usize,
}

/// Move the points of `grid` with `deformer` and re-bin them into a grid with `transform`
/// (`movePoints(points, transform, deformer, filter)`).
pub fn move_points_to_transform<D: PointDeformer, F: IndexFilter>(
    grid: &mut PointDataGrid,
    transform: &Transform,
    deformer: &D,
    filter: &F,
) -> Result<()> {
    let Some(first) = grid.tree().leaves().next() else { return Ok(()) };
    let existing = first.attributes.clone();

    // Target topology and per-voxel counts.
    let counts = convert_points_to_scalar_transformed::<u32, F, D>(grid, transform, filter, deformer);
    let mut target = PointDataTree::new();
    for leaf in counts.tree().leaves() {
        let mut offsets = leaf.clone();
        let mut sum = 0u32;
        for v in offsets.buffer_mut().iter_mut() {
            sum += *v;
            *v = sum;
        }
        let set = AttributeSet::like(&existing, sum)?;
        target.add_leaf(PointDataLeafNode::from_parts(offsets, set));
    }
    let target_index: HashMap<Coord, usize> = target.leaves().enumerate().map(|(i, l)| (l.origin(), i)).collect();

    // Compute new positions (written back into the source "P" arrays as voxel-relative
    // positions in the target index space) and the move maps.
    let source_transform = grid.transform().clone();
    let all = filter.state() == IndexState::All;
    let per_source: Vec<Result<Vec<(usize, Move)>>> = grid
        .tree_mut()
        .leaves_mut()
        .into_par_iter()
        .enumerate()
        .map(|(idx, mut leaf)| {
            let mut deformer = deformer.clone();
            deformer.reset(&PointDataLeaf { leaf: &*leaf.leaf, attributes: &*leaf.attributes }, idx);
            let entries: Vec<(Coord, u32)> = if all {
                leaf.index_on().with_coords().collect()
            } else {
                leaf.index_filtered(VoxelSelection::On, filter).with_coords().collect()
            };
            let mut out = Vec::with_capacity(entries.len());
            let p = leaf.attribute_array_mut("P")?;
            let mut moved = Vec::with_capacity(entries.len());
            {
                let h = AttributeHandle::<Vec3f>::new(p)?;
                for &(ijk, i) in &entries {
                    let q = h.get(i);
                    let mut pos = Vec3d::new(q[0] as f64, q[1] as f64, q[2] as f64) + ijk.as_vec3d();
                    if D::INDEX_SPACE {
                        deformer.apply(&mut pos, i, &ijk);
                    }
                    let mut ws = source_transform.index_to_world(pos);
                    if !D::INDEX_SPACE {
                        deformer.apply(&mut ws, i, &ijk);
                    }
                    let is = transform.world_to_index(ws);
                    let voxel = Coord::round(is);
                    let local = is - voxel.as_vec3d();
                    moved.push((i, voxel, Vec3f::new(local[0] as f32, local[1] as f32, local[2] as f32)));
                }
            }
            for &(i, voxel, local) in &moved {
                p.set(i, local);
                let Some(&t) = target_index.get(&voxel.align_down(8)) else { continue };
                out.push((t, Move { source_leaf: idx, source_index: i, voxel: LeafNode::<u32>::coord_to_offset(&voxel) }));
            }
            Ok(out)
        })
        .collect();

    // Distribute the moves to the target leaves: moves between leaves (sorted by source
    // leaf and index) before moves within a leaf (in iteration order).
    let target_origins: Vec<Coord> = target.leaves().map(|l| l.origin()).collect();
    let source_origins: Vec<Coord> = grid.tree().leaves().map(|l| l.origin()).collect();
    let mut global: Vec<Vec<Move>> = vec![Vec::new(); target_origins.len()];
    let mut local: Vec<Vec<Move>> = vec![Vec::new(); target_origins.len()];
    for r in per_source {
        for (t, m) in r? {
            if source_origins[m.source_leaf] == target_origins[t] {
                local[t].push(m);
            } else {
                global[t].push(m);
            }
        }
    }
    global.par_iter_mut().for_each(|g| g.sort_by_key(|m| (m.source_leaf, m.source_index)));

    // Copy every attribute.
    let source_leaves = grid.tree().leaf_vec();
    let results: Vec<Result<()>> = target
        .leaves_mut()
        .into_par_iter()
        .enumerate()
        .map(|(t, mut leaf)| {
            let moves: Vec<Move> = global[t].iter().chain(local[t].iter()).copied().collect();
            if moves.is_empty() {
                return Ok(());
            }
            // Target index of each move.
            let mut counters = [0u32; 512];
            let buf = leaf.leaf.buffer();
            let targets: Vec<u32> = moves
                .iter()
                .map(|m| {
                    let base = if m.voxel == 0 { 0 } else { buf[m.voxel - 1] };
                    let k = base + counters[m.voxel];
                    counters[m.voxel] += 1;
                    k
                })
                .collect();
            let nattr = leaf.attributes.size();
            for pos in 0..nattr {
                let dst = leaf.attribute_array_at_mut(pos)?;
                let mut start = 0;
                while start < moves.len() {
                    let src_leaf = moves[start].source_leaf;
                    let mut end = start;
                    while end < moves.len() && moves[end].source_leaf == src_leaf {
                        end += 1;
                    }
                    let src = source_leaves[src_leaf]
                        .attributes
                        .get(pos)
                        .ok_or_else(|| Error::Lookup("source leaf lacks an attribute".into()))?;
                    let pairs = (start..end).map(|k| (moves[k].source_index, targets[k]));
                    dst.copy_values(src, pairs, false)?;
                    start = end;
                }
            }
            Ok(())
        })
        .collect();
    results.into_iter().collect::<Result<()>>()?;
    drop(source_leaves);
    grid.set_tree(target);
    grid.set_transform(transform.clone());
    Ok(())
}

/// Per-leaf cache of deformed world-space positions.
#[derive(Clone, Debug, Default)]
struct LeafCache {
    vec_data: Vec<Vec3d>,
    map_data: HashMap<u32, Vec3d>,
}

/// A deformer that replays positions computed once by another deformer (`CachedDeformer`),
/// e.g. to move points with an expensive deformer to several transforms.
#[derive(Clone, Debug, Default)]
pub struct CachedDeformer {
    cache: Arc<Vec<LeafCache>>,
    current: Option<usize>,
}

impl CachedDeformer {
    /// An empty cache; call [`evaluate`](Self::evaluate) before use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cache the world-space positions of the points of `grid` accepted by `filter` after
    /// applying `deformer`.
    pub fn evaluate<D: PointDeformer, F: IndexFilter>(&mut self, grid: &PointDataGrid, deformer: &D, filter: &F) {
        let transform = grid.transform();
        let leaves = grid.tree().leaf_vec();
        let cache: Vec<LeafCache> = leaves
            .par_iter()
            .enumerate()
            .map(|(idx, leaf)| {
                let mut c = LeafCache::default();
                let total = leaf.point_count();
                if total == 0 {
                    return c;
                }
                let mut d = deformer.clone();
                d.reset(leaf, idx);
                let Ok(h) = leaf.attribute_handle::<Vec3f>("P") else { return c };
                let use_vec = filter.state() == IndexState::All && leaf.on_point_count() == total;
                if use_vec {
                    c.vec_data = vec![Vec3d::zero(); total as usize];
                }
                for (ijk, i) in leaf.index_filtered(VoxelSelection::On, filter).with_coords() {
                    let q = h.get(i);
                    let mut pos = Vec3d::new(q[0] as f64, q[1] as f64, q[2] as f64) + ijk.as_vec3d();
                    if D::INDEX_SPACE {
                        d.apply(&mut pos, i, &ijk);
                        pos = transform.index_to_world(pos);
                    } else {
                        pos = transform.index_to_world(pos);
                        d.apply(&mut pos, i, &ijk);
                    }
                    if use_vec {
                        c.vec_data[i as usize] = pos;
                    } else {
                        c.map_data.insert(i, pos);
                    }
                }
                c
            })
            .collect();
        self.cache = Arc::new(cache);
        self.current = None;
    }
}

impl PointDeformer for CachedDeformer {
    fn reset(&mut self, _: &PointDataLeaf<'_>, leaf_index: usize) {
        assert!(leaf_index < self.cache.len(), "leaf index out of range of the cached deformer (not evaluated?)");
        self.current = Some(leaf_index);
    }
    fn apply(&self, pos: &mut Vec3d, index: u32, _: &Coord) {
        let Some(c) = self.current.map(|i| &self.cache[i]) else { return };
        if !c.map_data.is_empty() {
            if let Some(p) = c.map_data.get(&index) {
                *pos = *p;
            }
        } else if let Some(p) = c.vec_data.get(index as usize) {
            *pos = *p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::{
        append_typed_attribute, create_point_data_grid_from_index, point_count, populate_attribute, Codec, NullFilter,
    };
    use crate::tools::point_index_grid::create_point_index_grid;

    #[derive(Clone)]
    struct Offset(Vec3d);
    impl PointDeformer for Offset {
        fn reset(&mut self, _: &PointDataLeaf<'_>, _: usize) {}
        fn apply(&self, pos: &mut Vec3d, _: u32, _: &Coord) {
            *pos += self.0;
        }
    }

    fn grid() -> PointDataGrid {
        let pos: Vec<Vec3f> = (0..500).map(|i| Vec3f::new((i % 25) as f32 * 0.8, (i / 25) as f32 * 0.6, 1.0)).collect();
        let t = Transform::linear(0.5);
        let index = create_point_index_grid(&pos, &t);
        let mut grid = create_point_data_grid_from_index(&index, &pos, Codec::Null, &t, None).unwrap();
        let ids: Vec<i32> = (0..500).collect();
        append_typed_attribute(grid.tree_mut(), "id", Codec::Null, 0i32, 1, None, false, false).unwrap();
        populate_attribute(grid.tree_mut(), index.tree(), "id", &ids, 1).unwrap();
        grid
    }

    fn positions_by_id(grid: &PointDataGrid) -> Vec<Vec3d> {
        let mut out = vec![Vec3d::zero(); point_count(grid.tree()) as usize];
        for leaf in grid.tree().leaves() {
            leaf.validate_offsets().unwrap();
            let p = leaf.attribute_handle::<Vec3f>("P").unwrap();
            let id = leaf.attribute_handle::<i32>("id").unwrap();
            for (ijk, i) in leaf.index_all().with_coords() {
                let q = p.get(i);
                out[id.get(i) as usize] =
                    grid.transform().index_to_world(ijk.as_vec3d() + Vec3d::new(q[0] as f64, q[1] as f64, q[2] as f64));
            }
        }
        out
    }

    #[test]
    fn move_by_offset() {
        let mut g = grid();
        let before = positions_by_id(&g);
        let d = Offset(Vec3d::new(3.3, -7.25, 0.4));
        move_points(&mut g, &d, &NullFilter).unwrap();
        assert_eq!(point_count(g.tree()), 500);
        let after = positions_by_id(&g);
        for (a, b) in before.iter().zip(&after) {
            assert!((*a + d.0 - *b).length() < 1e-5);
        }
        // Every point lies in its voxel.
        for leaf in g.tree().leaves() {
            let p = leaf.attribute_handle::<Vec3f>("P").unwrap();
            assert!(leaf.index_all().all(|i| p.get(i).abs().max_component() <= 0.5 + 1e-6));
        }
        // Re-binning into a coarser transform keeps the world positions.
        let coarse = Transform::linear(2.0);
        move_points_to_transform(&mut g, &coarse, &crate::points::NullDeformer, &NullFilter).unwrap();
        let rebinned = positions_by_id(&g);
        for (a, b) in after.iter().zip(&rebinned) {
            assert!((*a - *b).length() < 1e-5);
        }
        // Cached deformer reproduces the offset deformer.
        let mut cached = CachedDeformer::new();
        cached.evaluate(&g, &d, &NullFilter);
        move_points(&mut g, &cached, &NullFilter).unwrap();
        let moved = positions_by_id(&g);
        for (a, b) in rebinned.iter().zip(&moved) {
            assert!((*a + d.0 - *b).length() < 1e-4);
        }
    }
}
