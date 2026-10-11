//! Iteration over the point indices of a leaf (port of `openvdb/points/IndexIterator.h`).
//!
//! The voxel values of a point data leaf are end offsets into its attribute arrays: the
//! points of voxel `n` have indices `value(n-1)..value(n)` (with `value(-1) = 0`). An
//! [`IndexIter`] walks those ranges for all voxels, active voxels, inactive voxels or a
//! single voxel, skipping points rejected by a [`BoundFilter`].

use super::index_filter::{BoundFilter, NullBound};
use crate::math::Coord;
use crate::tree::LeafNode;

/// Early-exit hint of a filter (`index::State`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexState {
    /// Points must be tested individually.
    Partial,
    /// No point passes the filter.
    None,
    /// Every point passes the filter.
    All,
}

/// Which voxels an [`IndexIter`] visits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelSelection {
    /// All 512 voxels (`beginIndexAll`).
    All,
    /// Active voxels (`beginIndexOn`).
    On,
    /// Inactive voxels (`beginIndexOff`).
    Off,
    /// One voxel, by linear offset (`beginIndexVoxel`).
    Voxel(usize),
    /// Nothing (a filter with state [`IndexState::None`]).
    Empty,
}

/// Iterator over the point indices of a leaf (OpenVDB's `IndexIter`), yielding `u32`
/// indices into the leaf's attribute arrays.
pub struct IndexIter<'a, F: BoundFilter = NullBound> {
    leaf: &'a LeafNode<u32>,
    selection: VoxelSelection,
    /// Current voxel offset (512 when exhausted).
    voxel: usize,
    item: u32,
    end: u32,
    filter: F,
}

impl<'a, F: BoundFilter> IndexIter<'a, F> {
    /// Create an iterator over the selected voxels of `leaf`.
    pub fn new(leaf: &'a LeafNode<u32>, selection: VoxelSelection, filter: F) -> Self {
        let mut it = IndexIter { leaf, selection, voxel: 0, item: 0, end: 0, filter };
        match selection {
            VoxelSelection::Empty => it.voxel = 512,
            VoxelSelection::Voxel(n) => {
                it.voxel = n;
                it.item = if n == 0 { 0 } else { leaf.buffer()[n - 1] };
                it.end = leaf.buffer()[n];
            }
            _ => {
                it.voxel = usize::MAX;
                it.advance_voxel();
            }
        }
        it
    }

    fn selected(&self, n: usize) -> bool {
        match self.selection {
            VoxelSelection::All => true,
            VoxelSelection::On => self.leaf.is_value_on_at(n),
            VoxelSelection::Off => !self.leaf.is_value_on_at(n),
            _ => false,
        }
    }

    /// Move to the next selected voxel (after the current one) with at least one point.
    fn advance_voxel(&mut self) {
        let buf = self.leaf.buffer();
        if matches!(self.selection, VoxelSelection::Voxel(_) | VoxelSelection::Empty) {
            self.voxel = 512;
            return;
        }
        let mut n = self.voxel.wrapping_add(1);
        while n < 512 {
            if self.selected(n) {
                let start = if n == 0 { 0 } else { buf[n - 1] };
                if buf[n] > start {
                    self.voxel = n;
                    self.item = start;
                    self.end = buf[n];
                    return;
                }
            }
            n += 1;
        }
        self.voxel = 512;
    }

    /// The next accepted `(voxel offset, point index)`.
    pub fn next_entry(&mut self) -> Option<(usize, u32)> {
        loop {
            if self.voxel >= 512 {
                return None;
            }
            if self.item >= self.end {
                self.advance_voxel();
                continue;
            }
            let idx = self.item;
            self.item += 1;
            let n = self.voxel;
            let ijk = self.leaf.offset_to_global_coord(n);
            if self.filter.valid(idx, &ijk, self.leaf.is_value_on_at(n)) {
                return Some((n, idx));
            }
        }
    }

    /// Adapt into an iterator over `(voxel coordinate, point index)`.
    pub fn with_coords(self) -> IndexCoordIter<'a, F> {
        IndexCoordIter { inner: self }
    }

    /// Number of remaining accepted indices (`iterCount`).
    pub fn count_remaining(self) -> u64 {
        let mut it = self;
        let mut n = 0u64;
        while it.next_entry().is_some() {
            n += 1;
        }
        n
    }
}

impl<F: BoundFilter> Iterator for IndexIter<'_, F> {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        self.next_entry().map(|(_, i)| i)
    }
}

/// Iterator over `(voxel coordinate, point index)` (see [`IndexIter::with_coords`]).
pub struct IndexCoordIter<'a, F: BoundFilter> {
    inner: IndexIter<'a, F>,
}

impl<F: BoundFilter> Iterator for IndexCoordIter<'_, F> {
    type Item = (Coord, u32);
    #[inline]
    fn next(&mut self) -> Option<(Coord, u32)> {
        let (n, i) = self.inner.next_entry()?;
        Some((self.inner.leaf.offset_to_global_coord(n), i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::index_filter::NullBound;
    use crate::tree::Node;
    use crate::util::NodeMask;

    #[test]
    fn index_ranges() {
        let mut buf = vec![0u32; 512];
        // voxel 1 has 2 points (inactive), voxel 3 has 3 points (active).
        for (i, v) in buf.iter_mut().enumerate() {
            *v = if i >= 3 {
                5
            } else if i >= 1 {
                2
            } else {
                0
            };
        }
        let mut mask = NodeMask::<3>::new();
        mask.set_on(3);
        let leaf = LeafNode::from_parts(Coord::new(8, 0, 0), mask, buf);
        let all: Vec<u32> = IndexIter::new(&leaf, VoxelSelection::All, NullBound).collect();
        assert_eq!(all, vec![0, 1, 2, 3, 4]);
        let on: Vec<u32> = IndexIter::new(&leaf, VoxelSelection::On, NullBound).collect();
        assert_eq!(on, vec![2, 3, 4]);
        let off: Vec<u32> = IndexIter::new(&leaf, VoxelSelection::Off, NullBound).collect();
        assert_eq!(off, vec![0, 1]);
        let v: Vec<(Coord, u32)> = IndexIter::new(&leaf, VoxelSelection::Voxel(1), NullBound).with_coords().collect();
        assert_eq!(v, vec![(Coord::new(8, 0, 1), 0), (Coord::new(8, 0, 1), 1)]);
        assert_eq!(IndexIter::new(&leaf, VoxelSelection::Empty, NullBound).count(), 0);
        assert_eq!(leaf.origin(), Coord::new(8, 0, 0));
    }
}
