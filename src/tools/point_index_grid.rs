//! Space-partitioning acceleration structure for points (port of
//! `openvdb/tools/PointIndexGrid.h`).
//!
//! A [`PointIndexGrid`] partitions points into voxels using a cell-centered transform. Its
//! topology is a regular `Tree<u32>` whose active voxels contain points and whose leaf
//! values are the end offsets of each voxel's run of indices in the leaf's index array
//! (exactly the layout of OpenVDB's `PointIndexLeafNode`). The per-leaf index arrays are
//! stored alongside the tree.
//!
//! [`PointIndexIterator`] provides index-space voxel/bbox queries and world-space radial and
//! bounding box queries with the same visiting order as OpenVDB.

use super::point_partitioner::{PointArray, PointPartitioner};
use crate::math::{Coord, CoordBBox, Transform, Vec3d};
use crate::tree::{LeafNode, Node, Tree};
use rayon::prelude::*;
use std::collections::HashMap;

/// A leaf of a point index tree: voxel offsets (from the tree) plus the index array.
#[derive(Clone, Copy)]
pub struct PointIndexLeaf<'a> {
    leaf: &'a LeafNode<u32>,
    indices: &'a [u32],
}

impl<'a> PointIndexLeaf<'a> {
    /// Origin of the leaf.
    pub fn origin(&self) -> Coord {
        self.leaf.origin()
    }
    /// The underlying offset leaf (values are end offsets into [`indices`](Self::indices)).
    pub fn leaf(&self) -> &'a LeafNode<u32> {
        self.leaf
    }
    /// All point indices stored in this leaf, grouped by voxel.
    pub fn indices(&self) -> &'a [u32] {
        self.indices
    }
    /// The point indices of the voxel at `offset`, or `None` if the voxel is inactive.
    pub fn get_indices_at(&self, offset: usize) -> Option<&'a [u32]> {
        if self.leaf.is_value_on_at(offset) {
            let b = if offset == 0 { 0 } else { self.leaf.buffer()[offset - 1] } as usize;
            let e = self.leaf.buffer()[offset] as usize;
            Some(&self.indices[b..e])
        } else {
            None
        }
    }
    /// The point indices of the voxel containing `ijk`.
    pub fn get_indices(&self, ijk: &Coord) -> Option<&'a [u32]> {
        self.get_indices_at(LeafNode::<u32>::coord_to_offset(ijk))
    }
    /// True if no voxel in `bbox` (clipped to this leaf) contains points.
    pub fn is_empty(&self, bbox: &CoordBBox) -> bool {
        let buf = self.leaf.buffer();
        let z_stride = (bbox.max[2] - bbox.min[2]) as usize;
        for x in bbox.min[0]..=bbox.max[0] {
            let x_pos = ((x & 7) as usize) << 6;
            for y in bbox.min[1]..=bbox.max[1] {
                let pos = x_pos + (((y & 7) as usize) << 3) + (bbox.min[2] & 7) as usize;
                let begin = if pos == 0 { 0 } else { buf[pos - 1] };
                if buf[pos + z_stride] > begin {
                    return false;
                }
            }
        }
        true
    }
}

/// The point index tree: topology and offsets in a `Tree<u32>`, plus per-leaf index arrays.
#[derive(Clone, Default)]
pub struct PointIndexTree {
    tree: Tree<u32>,
    indices: HashMap<Coord, Vec<u32>>,
}

impl PointIndexTree {
    /// The offset/topology tree.
    pub fn tree(&self) -> &Tree<u32> {
        &self.tree
    }
    /// The leaf containing `ijk`, if any.
    pub fn probe_leaf(&self, ijk: &Coord) -> Option<PointIndexLeaf<'_>> {
        let leaf = self.tree.probe_leaf(ijk)?;
        let indices = self.indices.get(&leaf.origin()).map(|v| v.as_slice()).unwrap_or(&[]);
        Some(PointIndexLeaf { leaf, indices })
    }
    /// Iterate over all leaves, in tree order.
    pub fn leaves(&self) -> impl Iterator<Item = PointIndexLeaf<'_>> + '_ {
        self.tree.leaves().map(move |leaf| PointIndexLeaf {
            leaf,
            indices: self.indices.get(&leaf.origin()).map(|v| v.as_slice()).unwrap_or(&[]),
        })
    }
    /// Number of leaves.
    pub fn leaf_count(&self) -> usize {
        self.tree.leaf_count()
    }
    /// Number of voxels containing points.
    pub fn active_voxel_count(&self) -> u64 {
        self.tree.active_voxel_count()
    }
    /// Total number of indexed points.
    pub fn point_count(&self) -> usize {
        self.indices.values().map(|v| v.len()).sum()
    }
}

/// A point index grid (OpenVDB's `PointIndexGrid`): a [`PointIndexTree`] and a transform.
#[derive(Clone, Default)]
pub struct PointIndexGrid {
    tree: PointIndexTree,
    transform: Transform,
}

impl PointIndexGrid {
    pub fn tree(&self) -> &PointIndexTree {
        &self.tree
    }
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
}

/// Partition points into a point index grid with the given transform
/// (`tools::createPointIndexGrid`).
pub fn create_point_index_grid<P: PointArray + ?Sized>(points: &P, xform: &Transform) -> PointIndexGrid {
    let mut grid = PointIndexGrid { tree: PointIndexTree::default(), transform: xform.clone() };
    if points.size() == 0 {
        return grid;
    }
    let partitioner = PointPartitioner::<3>::create(points, xform, false, true, true);
    let voxel_offsets = partitioner.voxel_offsets().unwrap();
    let leaves: Vec<(Box<LeafNode<u32>>, Vec<u32>)> = (0..partitioner.size())
        .into_par_iter()
        .map(|n| {
            let mut node = LeafNode::<u32>::new(partitioner.origin(n), 0, false);
            let indices = partitioner.indices(n);
            let offsets: Vec<u32> = indices.iter().map(|&i| voxel_offsets[i as usize]).collect();
            let mut histogram = [0u32; 512];
            for &o in &offsets {
                histogram[o as usize] += 1;
            }
            let mut count = 0u32;
            for (i, h) in histogram.iter_mut().enumerate() {
                if *h > 0 {
                    let start = count;
                    count += *h;
                    *h = start;
                    node.value_mask_mut().set_on(i);
                }
                node.buffer_mut()[i] = count;
            }
            let mut ordered = vec![0u32; indices.len()];
            for (k, &o) in offsets.iter().enumerate() {
                ordered[histogram[o as usize] as usize] = indices[k];
                histogram[o as usize] += 1;
            }
            (Box::new(node), ordered)
        })
        .collect();
    for (node, idx) in leaves {
        grid.tree.indices.insert(node.origin(), idx);
        grid.tree.tree.add_leaf(node);
    }
    grid
}

/// Partition points into a point index grid with a uniform voxel size.
pub fn create_point_index_grid_with_voxel_size<P: PointArray + ?Sized>(points: &P, voxel_size: f64) -> PointIndexGrid {
    create_point_index_grid(points, &Transform::linear(voxel_size))
}

/// True if `grid` is a valid partition of `points` (`tools::isValidPartition`).
pub fn is_valid_partition<P: PointArray + ?Sized>(points: &P, grid: &PointIndexGrid) -> bool {
    if points.size() != grid.tree.point_count() {
        return false;
    }
    let leaves: Vec<PointIndexLeaf<'_>> = grid.tree.leaves().collect();
    leaves.par_iter().all(|leaf| {
        for pos in leaf.leaf.value_mask().iter_on() {
            let c = leaf.leaf.offset_to_global_coord(pos);
            for &i in leaf.get_indices_at(pos).unwrap() {
                if c != grid.transform.world_to_index_cell_centered(points.get_pos(i as usize)) {
                    return false;
                }
            }
        }
        true
    })
}

/// Return `grid` if it is a valid partition of `points`, otherwise a new grid
/// (`tools::getValidPointIndexGrid`).
pub fn get_valid_point_index_grid<P: PointArray + ?Sized>(points: &P, grid: PointIndexGrid) -> PointIndexGrid {
    if is_valid_partition(points, &grid) {
        grid
    } else {
        create_point_index_grid(points, &grid.transform)
    }
}

// ---------------------------------------------------------------------------------------
// Searching
// ---------------------------------------------------------------------------------------

fn construct_exclusive_regions(bbox: &CoordBBox, ibox: &CoordBBox) -> Vec<CoordBBox> {
    let mut regions = Vec::with_capacity(6);
    let mut cmin = ibox.min;
    let mut cmax = ibox.max;
    let mut r = *bbox;
    r.max[2] = cmin[2];
    regions.push(r);
    let mut r = *bbox;
    r.min[2] = cmax[2];
    regions.push(r);
    cmax[2] -= 1;
    cmin[2] += 1;
    let mut last = *bbox;
    last.min[2] = cmin[2];
    last.max[2] = cmax[2];
    last.max[0] = cmin[0];
    regions.push(last);
    last.min[0] = cmax[0];
    last.max[0] = bbox.max[0];
    regions.push(last);
    cmax[0] -= 1;
    cmin[0] += 1;
    last.min[0] = cmin[0];
    last.max[0] = cmax[0];
    last.max[1] = cmin[1];
    regions.push(last);
    last.min[1] = cmax[1];
    last.max[1] = bbox.max[1];
    regions.push(last);
    regions
}

fn point_index_search_voxels<'a>(ranges: &mut Vec<&'a [u32]>, leaf: &PointIndexLeaf<'a>, min: &Coord, max: &Coord) {
    let buf = leaf.leaf.buffer();
    let z_stride = (max[2] - min[2]) as usize;
    for x in min[0]..=max[0] {
        let x_pos = ((x & 7) as usize) << 6;
        for y in min[1]..=max[1] {
            let pos = x_pos + (((y & 7) as usize) << 3) + (min[2] & 7) as usize;
            let begin = if pos == 0 { 0 } else { buf[pos - 1] } as usize;
            let end = buf[pos + z_stride] as usize;
            if end > begin {
                ranges.push(&leaf.indices[begin..end]);
            }
        }
    }
}

fn point_index_search<'a>(ranges: &mut Vec<&'a [u32]>, tree: &'a PointIndexTree, bbox: &CoordBBox) {
    let leaf_min = bbox.min.align_down(8);
    let leaf_max = bbox.max.align_down(8);
    let mut x = leaf_min[0];
    while x <= leaf_max[0] {
        let mut y = leaf_min[1];
        while y <= leaf_max[1] {
            let mut z = leaf_min[2];
            while z <= leaf_max[2] {
                let ijk = Coord::new(x, y, z);
                if let Some(leaf) = tree.probe_leaf(&ijk) {
                    let ijk_max = ijk.offset(7);
                    let a = bbox.min.max_elem(&ijk);
                    let b = bbox.max.min_elem(&ijk_max);
                    if a != ijk || b != ijk_max {
                        point_index_search_voxels(ranges, &leaf, &a, &b);
                    } else if !leaf.indices.is_empty() {
                        ranges.push(leaf.indices);
                    }
                }
                z += 8;
            }
            y += 8;
        }
        x += 8;
    }
}

trait RangeFilter<'a> {
    fn filter_leaf_node(&mut self, leaf: &PointIndexLeaf<'a>);
    fn filter_voxel(&mut self, ijk: &Coord, indices: &'a [u32]);
}

fn filtered_point_index_search<'a, F: RangeFilter<'a>>(filter: &mut F, tree: &'a PointIndexTree, bbox: &CoordBBox) {
    let leaf_min = bbox.min.align_down(8);
    let leaf_max = bbox.max.align_down(8);
    let mut x = leaf_min[0];
    while x <= leaf_max[0] {
        let mut y = leaf_min[1];
        while y <= leaf_max[1] {
            let mut z = leaf_min[2];
            while z <= leaf_max[2] {
                let ijk = Coord::new(x, y, z);
                if let Some(leaf) = tree.probe_leaf(&ijk) {
                    let ijk_max = ijk.offset(7);
                    let a = bbox.min.max_elem(&ijk);
                    let b = bbox.max.min_elem(&ijk_max);
                    if a != ijk || b != ijk_max {
                        let buf = leaf.leaf.buffer();
                        for xx in a[0]..=b[0] {
                            let xp = ((xx & 7) as usize) << 6;
                            for yy in a[1]..=b[1] {
                                let yp = xp + (((yy & 7) as usize) << 3);
                                for zz in a[2]..=b[2] {
                                    let pos = yp + (zz & 7) as usize;
                                    let bo = if pos == 0 { 0 } else { buf[pos - 1] } as usize;
                                    let eo = buf[pos] as usize;
                                    if eo > bo {
                                        filter.filter_voxel(&Coord::new(xx, yy, zz), &leaf.indices[bo..eo]);
                                    }
                                }
                            }
                        }
                    } else {
                        filter.filter_leaf_node(&leaf);
                    }
                }
                z += 8;
            }
            y += 8;
        }
        x += 8;
    }
}

struct BBoxFilter<'a, 'p, P: PointArray + ?Sized> {
    indices: Vec<u32>,
    region_min: Vec3d,
    region_max: Vec3d,
    points: &'p P,
    xform: &'p Transform,
    _m: std::marker::PhantomData<&'a ()>,
}

impl<'a, 'p, P: PointArray + ?Sized> RangeFilter<'a> for BBoxFilter<'a, 'p, P> {
    fn filter_leaf_node(&mut self, leaf: &PointIndexLeaf<'a>) {
        for pos in leaf.leaf.value_mask().iter_on() {
            let c = leaf.leaf.offset_to_global_coord(pos);
            self.filter_voxel(&c, leaf.get_indices_at(pos).unwrap());
        }
    }
    fn filter_voxel(&mut self, _ijk: &Coord, indices: &'a [u32]) {
        for &i in indices {
            let p = self.xform.world_to_index(self.points.get_pos(i as usize));
            if (0..3).all(|k| p[k] >= self.region_min[k] && p[k] <= self.region_max[k]) {
                self.indices.push(i);
            }
        }
    }
}

struct RadialRangeFilter<'a, 'p, P: PointArray + ?Sized> {
    ranges: Vec<&'a [u32]>,
    indices: Vec<u32>,
    center: Vec3d,
    ws_center: Vec3d,
    voxel_dist1: f64,
    voxel_dist2: f64,
    leaf_dist1: f64,
    leaf_dist2: f64,
    ws_radius_sqr: f64,
    points: &'p P,
    subvoxel_accuracy: bool,
}

impl<'a, 'p, P: PointArray + ?Sized> RadialRangeFilter<'a, 'p, P> {
    fn new(center: Vec3d, radius: f64, points: &'p P, xform: &Transform, leaf_dim: f64, subvoxel: bool) -> Self {
        let voxel_radius = 3f64.sqrt() * 0.5;
        let mut voxel_dist1 = voxel_radius + radius;
        voxel_dist1 *= voxel_dist1;
        let mut voxel_dist2 = 0.0;
        if radius > voxel_radius {
            voxel_dist2 = radius - voxel_radius;
            voxel_dist2 *= voxel_dist2;
        }
        let leaf_radius = leaf_dim * 3f64.sqrt() * 0.5;
        let mut leaf_dist1 = leaf_radius + radius;
        leaf_dist1 *= leaf_dist1;
        let mut leaf_dist2 = 0.0;
        if radius > leaf_radius {
            leaf_dist2 = radius - leaf_radius;
            leaf_dist2 *= leaf_dist2;
        }
        let mut ws_radius_sqr = radius * xform.voxel_size()[0];
        ws_radius_sqr *= ws_radius_sqr;
        RadialRangeFilter {
            ranges: Vec::new(),
            indices: Vec::new(),
            center,
            ws_center: xform.index_to_world(center),
            voxel_dist1,
            voxel_dist2,
            leaf_dist1,
            leaf_dist2,
            ws_radius_sqr,
            points,
            subvoxel_accuracy: subvoxel,
        }
    }
}

impl<'a, 'p, P: PointArray + ?Sized> RangeFilter<'a> for RadialRangeFilter<'a, 'p, P> {
    fn filter_leaf_node(&mut self, leaf: &PointIndexLeaf<'a>) {
        let o = leaf.origin().as_vec3d();
        let v = o + Vec3d::splat(7.0 * 0.5) - self.center;
        let dist = v.length_sqr();
        if dist > self.leaf_dist1 {
            return;
        }
        if self.leaf_dist2 > 0.0 && dist < self.leaf_dist2 {
            self.ranges.push(leaf.indices);
            return;
        }
        for pos in leaf.leaf.value_mask().iter_on() {
            let c = leaf.leaf.offset_to_global_coord(pos);
            self.filter_voxel(&c, leaf.get_indices_at(pos).unwrap());
        }
    }
    fn filter_voxel(&mut self, ijk: &Coord, indices: &'a [u32]) {
        let v = self.center - ijk.as_vec3d();
        let dist = v.length_sqr();
        if dist > self.voxel_dist1 {
            return;
        }
        if !self.subvoxel_accuracy || (self.voxel_dist2 > 0.0 && dist < self.voxel_dist2) {
            self.ranges.push(indices);
            return;
        }
        for &i in indices {
            let d = self.ws_center - self.points.get_pos(i as usize);
            if d.length_sqr() < self.ws_radius_sqr {
                self.indices.push(i);
            }
        }
    }
}

/// Iterates over the point indices found by a spatial query on a [`PointIndexTree`]
/// (port of `PointIndexIterator`). Indices from whole voxel/leaf ranges come first, followed
/// by individually filtered indices, in the same order as OpenVDB.
#[derive(Clone, Default)]
pub struct PointIndexIterator<'a> {
    ranges: Vec<&'a [u32]>,
    filtered: Vec<u32>,
    range: usize,
    pos: usize,
}

impl<'a> PointIndexIterator<'a> {
    /// An empty iterator.
    pub fn new() -> Self {
        Self::default()
    }
    /// Iterator over the points in voxel `ijk`.
    pub fn from_coord(ijk: &Coord, tree: &'a PointIndexTree) -> Self {
        let mut it = Self::new();
        it.search_and_update_coord(ijk, tree);
        it
    }
    /// Iterator over the points in the index-space box `bbox`.
    pub fn from_bbox(bbox: &CoordBBox, tree: &'a PointIndexTree) -> Self {
        let mut it = Self::new();
        it.search_and_update_bbox(bbox, tree);
        it
    }

    fn clear(&mut self) {
        self.ranges.clear();
        self.filtered.clear();
        self.range = 0;
        self.pos = 0;
    }

    /// Reset to iterate over the points in voxel `ijk`.
    pub fn search_and_update_coord(&mut self, ijk: &Coord, tree: &'a PointIndexTree) {
        self.clear();
        if let Some(leaf) = tree.probe_leaf(ijk) {
            if let Some(r) = leaf.get_indices(ijk) {
                self.ranges.push(r);
            }
        }
        self.reset();
    }

    /// Reset to iterate over the points in the index-space box `bbox`.
    pub fn search_and_update_bbox(&mut self, bbox: &CoordBBox, tree: &'a PointIndexTree) {
        self.clear();
        point_index_search(&mut self.ranges, tree, bbox);
        self.reset();
    }

    /// Reset to iterate over the points inside the (fractional) index-space box
    /// `[bbox_min, bbox_max]`.
    pub fn search_and_update_index_bbox<P: PointArray + ?Sized>(
        &mut self,
        bbox_min: Vec3d,
        bbox_max: Vec3d,
        tree: &'a PointIndexTree,
        points: &P,
        xform: &Transform,
    ) {
        self.clear();
        let region = CoordBBox::new(Coord::round(bbox_min), Coord::round(bbox_max));
        let dim = region.dim();
        let min_extent = dim[0].min(dim[1].min(dim[2]));
        let regions = if min_extent > 2 {
            let mut ibox = region;
            ibox.expand(-1);
            point_index_search(&mut self.ranges, tree, &ibox);
            ibox.expand(1);
            construct_exclusive_regions(&region, &ibox)
        } else {
            vec![region]
        };
        let mut filter = BBoxFilter {
            indices: Vec::new(),
            region_min: bbox_min,
            region_max: bbox_max,
            points,
            xform,
            _m: std::marker::PhantomData,
        };
        for r in &regions {
            filtered_point_index_search(&mut filter, tree, r);
        }
        self.filtered = filter.indices;
        self.reset();
    }

    /// Reset to iterate over the points within `radius` (index units) of the index-space
    /// position `center`. Without `subvoxel_accuracy`, whole voxels that intersect the
    /// sphere are returned.
    pub fn search_and_update_radial<P: PointArray + ?Sized>(
        &mut self,
        center: Vec3d,
        radius: f64,
        tree: &'a PointIndexTree,
        points: &P,
        xform: &Transform,
        subvoxel_accuracy: bool,
    ) {
        self.clear();
        let mut bbox =
            CoordBBox::new(Coord::round(center - Vec3d::splat(radius)), Coord::round(center + Vec3d::splat(radius)));
        bbox.expand(1);
        let i_radius = radius * (1.0 / 3f64.sqrt());
        let regions = if i_radius > 2.0 {
            let mut ibox = CoordBBox::new(
                Coord::round(center - Vec3d::splat(i_radius)),
                Coord::round(center + Vec3d::splat(i_radius)),
            );
            ibox.expand(-1);
            point_index_search(&mut self.ranges, tree, &ibox);
            ibox.expand(1);
            construct_exclusive_regions(&bbox, &ibox)
        } else {
            vec![bbox]
        };
        let mut filter = RadialRangeFilter::new(center, radius, points, xform, 8.0, subvoxel_accuracy);
        for r in &regions {
            filtered_point_index_search(&mut filter, tree, r);
        }
        self.ranges.extend(filter.ranges);
        self.filtered = filter.indices;
        self.reset();
    }

    /// World-space bounding box query.
    pub fn world_space_search_and_update_bbox<P: PointArray + ?Sized>(
        &mut self,
        bbox_min: Vec3d,
        bbox_max: Vec3d,
        tree: &'a PointIndexTree,
        points: &P,
        xform: &Transform,
    ) {
        self.search_and_update_index_bbox(
            xform.world_to_index(bbox_min),
            xform.world_to_index(bbox_max),
            tree,
            points,
            xform,
        )
    }

    /// World-space radial query.
    pub fn world_space_search_and_update_radial<P: PointArray + ?Sized>(
        &mut self,
        center: Vec3d,
        radius: f64,
        tree: &'a PointIndexTree,
        points: &P,
        xform: &Transform,
        subvoxel_accuracy: bool,
    ) {
        self.search_and_update_radial(
            xform.world_to_index(center),
            radius / xform.voxel_size()[0],
            tree,
            points,
            xform,
            subvoxel_accuracy,
        )
    }

    /// Restart the iteration.
    pub fn reset(&mut self) {
        self.range = 0;
        self.pos = 0;
        self.skip_empty();
    }

    fn skip_empty(&mut self) {
        while self.range < self.ranges.len() && self.pos >= self.ranges[self.range].len() {
            self.range += 1;
            self.pos = 0;
        }
    }

    /// True if the iterator points at a valid index.
    pub fn test(&self) -> bool {
        if self.range < self.ranges.len() {
            true
        } else {
            self.pos < self.filtered.len()
        }
    }

    /// The current index.
    pub fn get(&self) -> u32 {
        if self.range < self.ranges.len() {
            self.ranges[self.range][self.pos]
        } else {
            self.filtered[self.pos]
        }
    }

    /// Advance to the next index.
    pub fn increment(&mut self) {
        self.pos += 1;
        if self.range < self.ranges.len() {
            self.skip_empty();
        }
    }

    /// Advance and return true if the iterator is still valid.
    pub fn next_index(&mut self) -> bool {
        if !self.test() {
            return false;
        }
        self.increment();
        self.test()
    }

    /// Total number of indices.
    pub fn size(&self) -> usize {
        self.ranges.iter().map(|r| r.len()).sum::<usize>() + self.filtered.len()
    }

    /// All indices as a vector (in iteration order).
    pub fn to_vec(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.ranges.iter().flat_map(|r| r.iter().copied()).collect();
        v.extend_from_slice(&self.filtered);
        v
    }
}

impl<'a> Iterator for PointIndexIterator<'a> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if !self.test() {
            return None;
        }
        let v = self.get();
        self.increment();
        Some(v)
    }
}

/// Selectively extracts and filters point data (port of `PointIndexFilter`).
pub struct PointIndexFilter<'a, P: PointArray + ?Sized> {
    points: &'a P,
    tree: &'a PointIndexTree,
    xform: Transform,
    inv_voxel_size: f64,
}

impl<'a, P: PointArray + ?Sized> PointIndexFilter<'a, P> {
    pub fn new(points: &'a P, tree: &'a PointIndexTree, xform: &Transform) -> Self {
        PointIndexFilter { points, tree, xform: xform.clone(), inv_voxel_size: 1.0 / xform.voxel_size()[0] }
    }

    /// Call `op(dist_sqr, index)` for every point within `radius` (world units) of `center`.
    pub fn search_and_apply<F: FnMut(f64, u32)>(&self, center: Vec3d, radius: f64, mut op: F) {
        let mut it = PointIndexIterator::new();
        if radius * self.inv_voxel_size < 8.0 {
            it.search_and_update_bbox(
                &CoordBBox::new(
                    self.xform.world_to_index_cell_centered(center - Vec3d::splat(radius)),
                    self.xform.world_to_index_cell_centered(center + Vec3d::splat(radius)),
                ),
                self.tree,
            );
        } else {
            it.world_space_search_and_update_radial(center, radius, self.tree, self.points, &self.xform, false);
        }
        let r2 = radius * radius;
        for i in it {
            let d = (self.points.get_pos(i as usize) - center).length_sqr();
            if d < r2 {
                op(d, i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3f;

    fn random_points(n: usize, seed: u64, scale: f32) -> Vec<Vec3f> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        (0..n).map(|_| Vec3f::new(next() * scale, next() * scale, next() * scale)).collect()
    }

    #[test]
    fn grid_partition_and_queries() {
        let pts = random_points(5000, 7, 20.0);
        let xform = Transform::linear(0.7);
        let grid = create_point_index_grid(&pts, &xform);
        assert_eq!(grid.tree().point_count(), pts.len());
        assert!(is_valid_partition(&pts, &grid));

        // Index-space bbox query matches brute force.
        let bbox = CoordBBox::new(Coord::new(-5, -3, -10), Coord::new(7, 4, 2));
        let mut found = PointIndexIterator::from_bbox(&bbox, grid.tree()).to_vec();
        found.sort_unstable();
        let mut expect: Vec<u32> = (0..pts.len() as u32)
            .filter(|&i| bbox.is_inside(&xform.world_to_index_cell_centered(pts.get_pos(i as usize))))
            .collect();
        expect.sort_unstable();
        assert_eq!(found, expect);

        // World-space radial query.
        for (center, radius) in [(Vec3d::new(1.0, 2.0, -3.0), 5.5), (Vec3d::new(-4.0, 0.0, 6.0), 1.2)] {
            let mut it = PointIndexIterator::new();
            it.world_space_search_and_update_radial(center, radius, grid.tree(), &pts, &xform, true);
            let mut found = it.to_vec();
            found.sort_unstable();
            let expect: Vec<u32> = (0..pts.len() as u32)
                .filter(|&i| (pts.get_pos(i as usize) - center).length_sqr() < radius * radius)
                .collect();
            assert_eq!(found, expect);
        }

        // World-space bbox query.
        let (bmin, bmax) = (Vec3d::new(-3.3, -1.0, 0.5), Vec3d::new(6.1, 7.2, 9.9));
        let mut it = PointIndexIterator::new();
        it.world_space_search_and_update_bbox(bmin, bmax, grid.tree(), &pts, &xform);
        let mut found = it.to_vec();
        found.sort_unstable();
        let expect: Vec<u32> = (0..pts.len() as u32)
            .filter(|&i| {
                let p = xform.world_to_index(pts.get_pos(i as usize));
                let (a, b) = (xform.world_to_index(bmin), xform.world_to_index(bmax));
                (0..3).all(|k| p[k] >= a[k] && p[k] <= b[k])
            })
            .collect();
        assert_eq!(found, expect);

        // Filter.
        let filter = PointIndexFilter::new(&pts, grid.tree(), &xform);
        let mut count = 0;
        filter.search_and_apply(Vec3d::new(0.0, 0.0, 0.0), 3.0, |d, _| {
            assert!(d < 9.0);
            count += 1;
        });
        let expect = pts.iter().filter(|p| (p.length() as f64) < 3.0).count();
        assert!((count as i64 - expect as i64).abs() <= 1);
    }
}
