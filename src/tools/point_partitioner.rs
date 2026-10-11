//! Spatially partitions points into buckets (port of `openvdb/tools/PointPartitioner.h`).
//!
//! Points are bucketed by the index-space cell (cell- or node-centered) they fall into;
//! each bucket covers a cube of `2^LOG2DIM` voxels. Buckets are ordered like OpenVDB's
//! (by 32-bucket bins in lexicographic coordinate order, then by bucket offset within the
//! bin), and the point indices within a bucket are in increasing order (or grouped by voxel
//! when `voxel_order` is requested). Non-finite positions are ignored.

use crate::math::{Coord, CoordBBox, Transform, Vec3d, Vec3f};
use rayon::prelude::*;

/// Random access to point positions (OpenVDB's `PointArray` concept: `size()` and
/// `getPos(n, pos)`).
pub trait PointArray: Sync {
    /// Number of points.
    fn size(&self) -> usize;
    /// World-space position of point `n`.
    fn get_pos(&self, n: usize) -> Vec3d;
}

impl PointArray for [Vec3f] {
    fn size(&self) -> usize {
        self.len()
    }
    fn get_pos(&self, n: usize) -> Vec3d {
        let p = self[n];
        Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64)
    }
}
impl PointArray for [Vec3d] {
    fn size(&self) -> usize {
        self.len()
    }
    fn get_pos(&self, n: usize) -> Vec3d {
        self[n]
    }
}
impl PointArray for Vec<Vec3f> {
    fn size(&self) -> usize {
        self.len()
    }
    fn get_pos(&self, n: usize) -> Vec3d {
        self.as_slice().get_pos(n)
    }
}
impl PointArray for Vec<Vec3d> {
    fn size(&self) -> usize {
        self.len()
    }
    fn get_pos(&self, n: usize) -> Vec3d {
        self[n]
    }
}
impl<P: PointArray + ?Sized> PointArray for &P {
    fn size(&self) -> usize {
        (**self).size()
    }
    fn get_pos(&self, n: usize) -> Vec3d {
        (**self).get_pos(n)
    }
}

const BIN_LOG2DIM: u32 = 5;

/// Partitions points into spatial buckets of `2^LOG2DIM` voxels per axis
/// (OpenVDB's `PointPartitioner<uint32_t, LOG2DIM>`).
#[derive(Clone, Debug, Default)]
pub struct PointPartitioner<const LOG2DIM: u32 = 3> {
    point_indices: Vec<u32>,
    voxel_offsets: Option<Vec<u32>>,
    page_offsets: Vec<u32>,
    page_coordinates: Vec<Coord>,
    using_cell_centered_transform: bool,
}

/// The standard partitioner with 8³ buckets.
pub type UInt32PointPartitioner = PointPartitioner<3>;

impl<const LOG2DIM: u32> PointPartitioner<LOG2DIM> {
    /// An empty partitioner.
    pub fn new() -> Self {
        PointPartitioner { using_cell_centered_transform: true, ..Default::default() }
    }

    /// Partition `points` into spatial buckets.
    ///
    /// * `voxel_order` — order the indices within each bucket by voxel;
    /// * `record_voxel_offsets` — store the in-bucket voxel offset of every point;
    /// * `cell_centered_transform` — use cell-centered (rounding) instead of node-centered
    ///   (flooring) world-to-index conversion.
    pub fn construct<P: PointArray + ?Sized>(
        &mut self,
        points: &P,
        xform: &Transform,
        voxel_order: bool,
        record_voxel_offsets: bool,
        cell_centered_transform: bool,
    ) {
        self.using_cell_centered_transform = cell_centered_transform;
        let log2dim2 = 2 * LOG2DIM;
        let bucket_mask: i32 = (1 << LOG2DIM) - 1;
        let bin_mask: i32 = (1 << (LOG2DIM + BIN_LOG2DIM)) - 1;
        let inv_bin_mask = !bin_mask;
        let n = points.size();

        // (bin coord, bucket offset, point index, voxel offset); None for non-finite points.
        let keyed: Vec<Option<(Coord, u32, u32, u32)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let pos = points.get_pos(i);
                if !(pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite()) {
                    return None;
                }
                let ijk = if cell_centered_transform {
                    xform.world_to_index_cell_centered(pos)
                } else {
                    xform.world_to_index_node_centered(pos)
                };
                let loc = Coord::new(ijk[0] & bucket_mask, ijk[1] & bucket_mask, ijk[2] & bucket_mask);
                let voxel_offset = ((loc[0] << log2dim2) + (loc[1] << LOG2DIM) + loc[2]) as u32;
                let bin = Coord::new(ijk[0] & inv_bin_mask, ijk[1] & inv_bin_mask, ijk[2] & inv_bin_mask);
                let b = Coord::new(
                    (ijk[0] & bin_mask) >> LOG2DIM,
                    (ijk[1] & bin_mask) >> LOG2DIM,
                    (ijk[2] & bin_mask) >> LOG2DIM,
                );
                let bucket = ((b[0] << (2 * BIN_LOG2DIM)) + (b[1] << BIN_LOG2DIM) + b[2]) as u32;
                Some((bin, bucket, i as u32, voxel_offset))
            })
            .collect();

        self.voxel_offsets =
            if record_voxel_offsets { Some(keyed.iter().map(|k| k.map(|k| k.3).unwrap_or(0)).collect()) } else { None };

        let mut entries: Vec<(Coord, u32, u32, u32)> = keyed.into_iter().flatten().collect();
        entries.par_sort_unstable_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));

        self.point_indices = Vec::with_capacity(entries.len());
        self.page_offsets = Vec::new();
        self.page_coordinates = Vec::new();
        let mut start = 0usize;
        while start < entries.len() {
            let (bin, bucket, _, _) = entries[start];
            let mut end = start + 1;
            while end < entries.len() && entries[end].0 == bin && entries[end].1 == bucket {
                end += 1;
            }
            self.page_offsets.push(start as u32);
            let bx = (bucket >> (2 * BIN_LOG2DIM)) as i32;
            let rem = bucket - ((bx as u32) << (2 * BIN_LOG2DIM));
            let by = (rem >> BIN_LOG2DIM) as i32;
            let bz = (rem - ((by as u32) << BIN_LOG2DIM)) as i32;
            self.page_coordinates.push(Coord::new(bx << LOG2DIM, by << LOG2DIM, bz << LOG2DIM) + bin);
            let page = &entries[start..end];
            if voxel_order {
                let mut sorted: Vec<(u32, u32)> = page.iter().map(|e| (e.3, e.2)).collect();
                sorted.sort_by_key(|e| e.0); // stable: keeps increasing point order per voxel
                self.point_indices.extend(sorted.iter().map(|e| e.1));
            } else {
                self.point_indices.extend(page.iter().map(|e| e.2));
            }
            start = end;
        }
        self.page_offsets.push(self.point_indices.len() as u32);
        if self.page_coordinates.is_empty() {
            self.page_offsets.clear();
        }
    }

    /// Create a partitioner for `points` (see [`construct`](Self::construct)).
    pub fn create<P: PointArray + ?Sized>(
        points: &P,
        xform: &Transform,
        voxel_order: bool,
        record_voxel_offsets: bool,
        cell_centered_transform: bool,
    ) -> Self {
        let mut p = Self::new();
        p.construct(points, xform, voxel_order, record_voxel_offsets, cell_centered_transform);
        p
    }

    /// Number of buckets (pages).
    pub fn size(&self) -> usize {
        self.page_coordinates.len()
    }
    /// True if there are no buckets.
    pub fn is_empty(&self) -> bool {
        self.page_coordinates.is_empty()
    }
    /// Remove all data.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
    /// The point indices of bucket `n`.
    pub fn indices(&self, n: usize) -> &[u32] {
        &self.point_indices[self.page_offsets[n] as usize..self.page_offsets[n + 1] as usize]
    }
    /// Index-space bounding box of bucket `n`.
    pub fn get_bbox(&self, n: usize) -> CoordBBox {
        CoordBBox::create_cube(self.page_coordinates[n], 1 << LOG2DIM)
    }
    /// Origin of bucket `n`.
    pub fn origin(&self, n: usize) -> Coord {
        self.page_coordinates[n]
    }
    /// Per-point voxel offsets within their bucket (only if recorded).
    pub fn voxel_offsets(&self) -> Option<&[u32]> {
        self.voxel_offsets.as_deref()
    }
    /// True if the partitioning used a cell-centered transform.
    pub fn using_cell_centered_transform(&self) -> bool {
        self.using_cell_centered_transform
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_basic() {
        let pts = vec![
            Vec3f::new(0.1, 0.2, 0.3),
            Vec3f::new(9.0, 0.0, 0.0),
            Vec3f::new(1.0, 1.0, 1.0),
            Vec3f::new(-1.0, 0.0, 0.0),
            Vec3f::new(f32::NAN, 0.0, 0.0),
            Vec3f::new(300.0, 0.0, 0.0),
        ];
        let p = UInt32PointPartitioner::create(&pts, &Transform::linear(1.0), false, true, true);
        assert_eq!(p.size(), 4);
        let total: usize = (0..p.size()).map(|n| p.indices(n).len()).sum();
        assert_eq!(total, 5);
        // The bucket containing the origin has points 0 and 2 in increasing order.
        let n = (0..p.size()).find(|&n| p.origin(n) == Coord::new(0, 0, 0)).unwrap();
        assert_eq!(p.indices(n), &[0, 2]);
        for n in 0..p.size() {
            for &i in p.indices(n) {
                let ijk = Transform::linear(1.0).world_to_index_cell_centered(pts.get_pos(i as usize));
                assert!(p.get_bbox(n).is_inside(&ijk));
            }
        }
        assert_eq!(p.voxel_offsets().unwrap()[2], (1 << 6) + (1 << 3) + 1);
    }
}
