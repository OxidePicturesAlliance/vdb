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

#[cfg(test)]
thread_local!(static COUNTING_SORT_USED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });

/// A raw pointer that can be shared between the threads writing disjoint slots through it.
struct SyncPtr(*mut u32);
unsafe impl Sync for SyncPtr {}

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
        let bucket_mask: i32 = (1 << LOG2DIM) - 1;
        let n = points.size();

        // Index coordinates; None for non-finite points.
        let ijks: Vec<Option<Coord>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let pos = points.get_pos(i);
                if !(pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite()) {
                    return None;
                }
                Some(if cell_centered_transform {
                    xform.world_to_index_cell_centered(pos)
                } else {
                    xform.world_to_index_node_centered(pos)
                })
            })
            .collect();

        self.voxel_offsets = if record_voxel_offsets {
            Some(ijks.par_iter().map(|c| c.map(|c| Self::voxel_offset(&c, bucket_mask)).unwrap_or(0)).collect())
        } else {
            None
        };

        // Bins are numbered densely, in lexicographic order, within the bounding box of the
        // occupied ones. That box is usually small, so pages can be found with a counting
        // sort on (bin number, bucket) instead of a comparison sort.
        let bin_shift = LOG2DIM + BIN_LOG2DIM;
        let (lo, hi) = ijks
            .par_iter()
            .with_min_len(1 << 14)
            .fold(
                || (Coord::max_coord(), Coord::min_coord()),
                |(lo, hi), c| match c {
                    Some(c) => (lo.min_elem(c), hi.max_elem(c)),
                    None => (lo, hi),
                },
            )
            .reduce(|| (Coord::max_coord(), Coord::min_coord()), |a, b| (a.0.min_elem(&b.0), a.1.max_elem(&b.1)));
        let bin_lo = [lo[0] >> bin_shift, lo[1] >> bin_shift, lo[2] >> bin_shift];
        let bin_dims: [usize; 3] =
            std::array::from_fn(|a| if lo[a] > hi[a] { 0 } else { ((hi[a] >> bin_shift) - bin_lo[a]) as usize + 1 });
        let bin_count = bin_dims[0].saturating_mul(bin_dims[1]).saturating_mul(bin_dims[2]);

        let chunks = rayon::current_num_threads().max(1);
        let keys_per_chunk = bin_count.saturating_mul(1 << (3 * BIN_LOG2DIM));
        if bin_count >= 1 << (32 - 3 * BIN_LOG2DIM) || keys_per_chunk.saturating_mul(chunks) > (n * 8).max(1 << 20) {
            return self.construct_by_sort(&ijks, voxel_order);
        }
        #[cfg(test)]
        COUNTING_SORT_USED.with(|c| c.set(true));
        let bin_number = |c: &Coord| {
            let b = |a: usize| ((c[a] >> bin_shift) - bin_lo[a]) as usize;
            (b(0) * bin_dims[1] + b(1)) * bin_dims[2] + b(2)
        };
        let bin_origin = |k: usize| {
            let (x, rest) = (k / (bin_dims[1] * bin_dims[2]), k % (bin_dims[1] * bin_dims[2]));
            let (y, z) = (rest / bin_dims[2], rest % bin_dims[2]);
            let o = |a: usize, v: usize| (bin_lo[a] + v as i32) << bin_shift;
            Coord::new(o(0, x), o(1, y), o(2, z))
        };

        // Page key of every point: bin number, then bucket offset within the bin.
        let keys: Vec<u32> = ijks
            .par_iter()
            .map(|c| match c {
                None => u32::MAX,
                Some(c) => ((bin_number(c) << (3 * BIN_LOG2DIM)) as u32) | Self::bucket_offset(c),
            })
            .collect();

        // Stable counting sort: per-chunk histograms, then each chunk scatters its points.
        let chunk_len = n.div_ceil(chunks).max(1);
        let mut offsets: Vec<Vec<u32>> = keys
            .par_chunks(chunk_len)
            .map(|ks| {
                let mut h = vec![0u32; keys_per_chunk];
                for &k in ks {
                    if k != u32::MAX {
                        h[k as usize] += 1;
                    }
                }
                h
            })
            .collect();
        self.page_offsets = Vec::new();
        self.page_coordinates = Vec::new();
        let mut total = 0u32;
        for k in 0..keys_per_chunk {
            let start = total;
            for h in offsets.iter_mut() {
                let c = h[k];
                h[k] = total;
                total += c;
            }
            if total > start {
                self.page_offsets.push(start);
                let bin = bin_origin(k >> (3 * BIN_LOG2DIM));
                self.page_coordinates.push(Self::bucket_origin(bin, (k & ((1 << (3 * BIN_LOG2DIM)) - 1)) as u32));
            }
        }
        let mut indices = vec![0u32; total as usize];
        {
            let out = SyncPtr(indices.as_mut_ptr());
            keys.par_chunks(chunk_len).zip(offsets.par_iter_mut()).enumerate().for_each(|(c, (ks, offs))| {
                let out = &out;
                for (j, &k) in ks.iter().enumerate() {
                    if k != u32::MAX {
                        let slot = &mut offs[k as usize];
                        // SAFETY: the offsets give every point of every chunk its own slot
                        // in `indices`, so the writes are disjoint and in bounds.
                        unsafe { *out.0.add(*slot as usize) = (c * chunk_len + j) as u32 };
                        *slot += 1;
                    }
                }
            });
        }
        self.point_indices = indices;
        self.page_offsets.push(total);
        if self.page_coordinates.is_empty() {
            self.page_offsets.clear();
        }
        if voxel_order {
            self.order_pages_by_voxel(&ijks, bucket_mask);
        }
    }

    /// Fallback for points spread over very many bins: a comparison sort.
    fn construct_by_sort(&mut self, ijks: &[Option<Coord>], voxel_order: bool) {
        let bucket_mask: i32 = (1 << LOG2DIM) - 1;
        let inv_bin_mask = !((1 << (LOG2DIM + BIN_LOG2DIM)) - 1);
        let mut entries: Vec<(Coord, u32, u32)> = ijks
            .par_iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (c.align_mask(inv_bin_mask), Self::bucket_offset(&c), i as u32)))
            .collect();
        entries.par_sort_unstable();

        self.point_indices = Vec::with_capacity(entries.len());
        self.page_offsets = Vec::new();
        self.page_coordinates = Vec::new();
        let mut start = 0usize;
        while start < entries.len() {
            let (bin, bucket, _) = entries[start];
            let mut end = start + 1;
            while end < entries.len() && entries[end].0 == bin && entries[end].1 == bucket {
                end += 1;
            }
            self.page_offsets.push(start as u32);
            self.page_coordinates.push(Self::bucket_origin(bin, bucket));
            self.point_indices.extend(entries[start..end].iter().map(|e| e.2));
            start = end;
        }
        self.page_offsets.push(self.point_indices.len() as u32);
        if self.page_coordinates.is_empty() {
            self.page_offsets.clear();
        }
        if voxel_order {
            self.order_pages_by_voxel(ijks, bucket_mask);
        }
    }

    /// Group the indices of every page by voxel (stably, so increasing within a voxel).
    fn order_pages_by_voxel(&mut self, ijks: &[Option<Coord>], bucket_mask: i32) {
        let offsets = &self.page_offsets;
        let mut pages: Vec<&mut [u32]> = Vec::with_capacity(self.page_coordinates.len());
        let mut rest = &mut self.point_indices[..];
        for w in offsets.windows(2) {
            let (page, tail) = rest.split_at_mut((w[1] - w[0]) as usize);
            pages.push(page);
            rest = tail;
        }
        pages.into_par_iter().for_each(|page| {
            page.sort_by_key(|&i| Self::voxel_offset(&ijks[i as usize].unwrap(), bucket_mask));
        });
    }

    #[inline]
    fn voxel_offset(ijk: &Coord, bucket_mask: i32) -> u32 {
        (((ijk[0] & bucket_mask) << (2 * LOG2DIM)) + ((ijk[1] & bucket_mask) << LOG2DIM) + (ijk[2] & bucket_mask))
            as u32
    }
    /// Offset of the bucket containing `ijk` within its bin.
    #[inline]
    fn bucket_offset(ijk: &Coord) -> u32 {
        let bin_mask: i32 = (1 << (LOG2DIM + BIN_LOG2DIM)) - 1;
        let b = |v: i32| ((v & bin_mask) >> LOG2DIM) as u32;
        (b(ijk[0]) << (2 * BIN_LOG2DIM)) | (b(ijk[1]) << BIN_LOG2DIM) | b(ijk[2])
    }
    /// Origin of the bucket at offset `bucket` within the bin at `bin`.
    #[inline]
    fn bucket_origin(bin: Coord, bucket: u32) -> Coord {
        let m = (1u32 << BIN_LOG2DIM) - 1;
        let (bx, by, bz) = (bucket >> (2 * BIN_LOG2DIM), (bucket >> BIN_LOG2DIM) & m, bucket & m);
        Coord::new((bx as i32) << LOG2DIM, (by as i32) << LOG2DIM, (bz as i32) << LOG2DIM) + bin
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

    #[test]
    fn counting_sort_matches_comparison_sort() {
        let mut seed = 3u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        // Spans four bins (few enough for the counting sort), with some NaN points.
        let pts: Vec<Vec3d> = (0..50_000)
            .map(|i| {
                if i % 997 == 0 {
                    Vec3d::new(f64::NAN, 0.0, 0.0)
                } else {
                    Vec3d::new(rnd() * 400.0 - 200.0, rnd() * 400.0 - 200.0, rnd() * 200.0)
                }
            })
            .collect();
        let xform = Transform::linear(1.0);
        for voxel_order in [false, true] {
            for cell_centered in [false, true] {
                COUNTING_SORT_USED.with(|c| c.set(false));
                let a = UInt32PointPartitioner::create(&pts, &xform, voxel_order, true, cell_centered);
                assert!(COUNTING_SORT_USED.with(|c| c.get()));
                let ijks: Vec<Option<Coord>> = pts
                    .iter()
                    .map(|p| {
                        p[0].is_finite().then(|| {
                            if cell_centered {
                                xform.world_to_index_cell_centered(*p)
                            } else {
                                xform.world_to_index_node_centered(*p)
                            }
                        })
                    })
                    .collect();
                let mut b = UInt32PointPartitioner::new();
                b.construct_by_sort(&ijks, voxel_order);
                assert!(a.size() > 1000);
                assert_eq!(a.page_offsets, b.page_offsets);
                assert_eq!(a.page_coordinates, b.page_coordinates);
                assert_eq!(a.point_indices, b.point_indices);
            }
        }
    }
}
