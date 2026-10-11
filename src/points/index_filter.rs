//! Point index filters (port of `openvdb/points/IndexFilter.h` and the `GroupFilter` of
//! `AttributeGroup.h`).
//!
//! A filter is an [`IndexFilter`]: it reports an early-exit [`IndexState`] and, for each
//! leaf, binds to the leaf's data ([`IndexFilter::bind`]) to produce a [`BoundFilter`] that
//! accepts or rejects individual points.
//!
//! Random choices use the same engines and distributions as OpenVDB built with GCC
//! (`std::mt19937`, libstdc++'s `uniform_int_distribution`, `uniform_real_distribution` and
//! `std::shuffle`), so [`RandomLeafFilter`] and [`AttributeHashFilter`] select the same
//! points as the C++ library for the same seed.

use super::attribute_array::AttributeHandle;
use super::attribute_group::{GroupHandle, GroupIndex};
use super::attribute_set::{AttributeSet, Descriptor};
use super::index_iterator::IndexState;
use super::point_data_grid::{PointDataTree, PointLeaf};
use super::Result;
use crate::grid::Grid;
use crate::math::{BBoxd, Coord, Transform, Vec3d, Vec3f};
use crate::tools::interpolation::{BoxSampler, Sampler};
use crate::tools::point_scatter::{generate_canonical, uniform_int, Mt19937, RandomGenerator};
use crate::tree::{LeafNode, Node};
use crate::types::Value;
use std::collections::HashMap;

/// A filter bound to one leaf: decides for each point whether it is accepted.
pub trait BoundFilter {
    /// True if point `index`, in the voxel at `ijk` (active if `active`), is accepted.
    fn valid(&mut self, index: u32, ijk: &Coord, active: bool) -> bool;
}

/// A point filter (OpenVDB's filter concept: `state()`, `state(leaf)`, `reset(leaf)`,
/// `valid(iter)`).
pub trait IndexFilter: Sync {
    /// The per-leaf filter.
    type Bound<'a>: BoundFilter
    where
        Self: 'a;
    /// Global early-exit state.
    fn state(&self) -> IndexState {
        IndexState::Partial
    }
    /// Early-exit state for one leaf.
    fn leaf_state(&self, leaf: &LeafNode<u32>, attributes: &AttributeSet) -> IndexState {
        let _ = (leaf, attributes);
        IndexState::Partial
    }
    /// Bind the filter to a leaf's voxel offsets and attributes.
    fn bind<'a>(&'a self, leaf: &'a LeafNode<u32>, attributes: &'a AttributeSet) -> Self::Bound<'a>;
}

/// Accepts every point (`NullFilter`).
#[derive(Clone, Copy, Debug, Default)]
pub struct NullFilter;

/// The bound form of [`NullFilter`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NullBound;

impl BoundFilter for NullBound {
    #[inline]
    fn valid(&mut self, _: u32, _: &Coord, _: bool) -> bool {
        true
    }
}

impl IndexFilter for NullFilter {
    type Bound<'a> = NullBound;
    fn state(&self) -> IndexState {
        IndexState::All
    }
    fn leaf_state(&self, _: &LeafNode<u32>, _: &AttributeSet) -> IndexState {
        IndexState::All
    }
    fn bind<'a>(&'a self, _: &'a LeafNode<u32>, _: &'a AttributeSet) -> NullBound {
        NullBound
    }
}

/// Accepts points in active (`ON == true`) or inactive voxels (`ValueMaskFilter`).
#[derive(Clone, Copy, Debug, Default)]
pub struct ValueMaskFilter<const ON: bool>;
/// Accepts points in active voxels.
pub type ActiveFilter = ValueMaskFilter<true>;
/// Accepts points in inactive voxels.
pub type InactiveFilter = ValueMaskFilter<false>;

impl<const ON: bool> BoundFilter for ValueMaskFilter<ON> {
    #[inline]
    fn valid(&mut self, _: u32, _: &Coord, active: bool) -> bool {
        active == ON
    }
}

impl<const ON: bool> IndexFilter for ValueMaskFilter<ON> {
    type Bound<'a> = ValueMaskFilter<ON>;
    fn leaf_state(&self, leaf: &LeafNode<u32>, _: &AttributeSet) -> IndexState {
        if leaf.is_dense() {
            if ON {
                IndexState::All
            } else {
                IndexState::None
            }
        } else if leaf.is_empty() {
            if ON {
                IndexState::None
            } else {
                IndexState::All
            }
        } else {
            IndexState::Partial
        }
    }
    fn bind<'a>(&'a self, _: &'a LeafNode<u32>, _: &'a AttributeSet) -> Self {
        *self
    }
}

/// Accepts the members of one group (`GroupFilter`).
#[derive(Clone, Debug)]
pub struct GroupFilter {
    index: GroupIndex,
}

impl GroupFilter {
    /// A filter for group `name` of the given descriptor; fails if the group does not exist.
    pub fn new(name: &str, descriptor: &Descriptor) -> Result<Self> {
        Ok(GroupFilter { index: descriptor.group_index(name)? })
    }
    /// A filter for a group index.
    pub fn from_index(index: GroupIndex) -> Self {
        GroupFilter { index }
    }
}

/// The bound form of [`GroupFilter`] (rejects everything if the leaf lacks the group array).
pub struct GroupBound<'a> {
    handle: Option<GroupHandle<'a>>,
}

impl BoundFilter for GroupBound<'_> {
    #[inline]
    fn valid(&mut self, index: u32, _: &Coord, _: bool) -> bool {
        self.handle.as_ref().is_some_and(|h| h.get(index))
    }
}

fn group_handle(attributes: &AttributeSet, index: GroupIndex) -> Option<GroupHandle<'_>> {
    attributes.get(index.0).and_then(|a| GroupHandle::new(a, index.1).ok())
}

impl IndexFilter for GroupFilter {
    type Bound<'a> = GroupBound<'a>;
    fn bind<'a>(&'a self, _: &'a LeafNode<u32>, attributes: &'a AttributeSet) -> GroupBound<'a> {
        GroupBound { handle: group_handle(attributes, self.index) }
    }
}

/// Accepts points in any of the include groups (or all points if there are none) that are in
/// none of the exclude groups (`MultiGroupFilter`).
#[derive(Clone, Debug, Default)]
pub struct MultiGroupFilter {
    include: Vec<GroupIndex>,
    exclude: Vec<GroupIndex>,
}

impl MultiGroupFilter {
    /// A filter from group names; names that do not exist in `descriptor` are ignored.
    pub fn new<S: AsRef<str>>(include: &[S], exclude: &[S], descriptor: &Descriptor) -> Self {
        let idx = |names: &[S]| -> Vec<GroupIndex> {
            names.iter().filter_map(|n| descriptor.group_index(n.as_ref()).ok()).collect()
        };
        MultiGroupFilter { include: idx(include), exclude: idx(exclude) }
    }
    /// A filter from group indices.
    pub fn from_indices(include: Vec<GroupIndex>, exclude: Vec<GroupIndex>) -> Self {
        MultiGroupFilter { include, exclude }
    }
}

/// The bound form of [`MultiGroupFilter`].
pub struct MultiGroupBound<'a> {
    include: Vec<Option<GroupHandle<'a>>>,
    exclude: Vec<Option<GroupHandle<'a>>>,
}

impl BoundFilter for MultiGroupBound<'_> {
    fn valid(&mut self, index: u32, _: &Coord, _: bool) -> bool {
        let inc = self.include.is_empty() || self.include.iter().any(|h| h.as_ref().is_some_and(|h| h.get(index)));
        inc && !self.exclude.iter().any(|h| h.as_ref().is_some_and(|h| h.get(index)))
    }
}

impl IndexFilter for MultiGroupFilter {
    type Bound<'a> = MultiGroupBound<'a>;
    fn state(&self) -> IndexState {
        if self.include.is_empty() && self.exclude.is_empty() {
            IndexState::All
        } else {
            IndexState::Partial
        }
    }
    fn bind<'a>(&'a self, _: &'a LeafNode<u32>, attributes: &'a AttributeSet) -> MultiGroupBound<'a> {
        MultiGroupBound {
            include: self.include.iter().map(|&i| group_handle(attributes, i)).collect(),
            exclude: self.exclude.iter().map(|&i| group_handle(attributes, i)).collect(),
        }
    }
}

/// libstdc++'s `std::shuffle` with a 32-bit generator.
pub(crate) fn std_shuffle<T, G: RandomGenerator>(v: &mut [T], g: &mut G) {
    let n = v.len() as u64;
    if n == 0 {
        return;
    }
    let urngrange = u32::MAX as u64;
    if urngrange / n >= n {
        let mut i = 1usize;
        if n % 2 == 0 {
            let j = uniform_int(g, 0, 1) as usize;
            v.swap(i, j);
            i += 1;
        }
        while i < v.len() {
            let swap_range = i as u64 + 1;
            let x = uniform_int(g, 0, swap_range * (swap_range + 1) - 1);
            let (a, b) = (x / (swap_range + 1), x % (swap_range + 1));
            v.swap(i, a as usize);
            i += 1;
            v.swap(i, b as usize);
            i += 1;
        }
        return;
    }
    for i in 1..v.len() {
        let j = uniform_int(g, 0, i as u64) as usize;
        v.swap(i, j);
    }
}

/// A sorted random subset of `n` of the indices `0..m` (`generateRandomSubset`).
pub fn generate_random_subset(seed: u32, n: i32, m: i32) -> Vec<i32> {
    if n <= 0 {
        return Vec::new();
    }
    let mut values: Vec<i32> = (0..m.max(0)).collect();
    if n >= m {
        return values;
    }
    let mut g = Mt19937::new(seed);
    std_shuffle(&mut values, &mut g);
    values.truncate(n as usize);
    values.sort_unstable();
    values
}

/// Accepts a random subset of points per leaf so that about `target_points` points are
/// accepted in total (`RandomLeafFilter<PointDataTree, std::mt19937>`).
#[derive(Clone, Debug)]
pub struct RandomLeafFilter {
    leaf_map: HashMap<Coord, (u32, i32)>,
}

impl RandomLeafFilter {
    /// Distribute `target_points` over the leaves of `tree` proportionally to their point
    /// counts.
    pub fn new(tree: &PointDataTree, target_points: u64, seed: u32) -> Self {
        let leaves: Vec<_> = tree.leaves().collect();
        let current: u64 = leaves.iter().map(|l| l.point_count()).sum();
        let factor = if target_points > current { 1.0f32 } else { target_points as f32 / current as f32 };
        let mut g = Mt19937::new(seed);
        let mut leaf_map = HashMap::new();
        let mut total_float = 0.0f32;
        let mut total = 0i32;
        let count = leaves.len();
        for (k, leaf) in leaves.iter().enumerate() {
            let s = uniform_int(&mut g, 0, (u32::MAX - 1) as u64) as u32;
            if k + 1 == count {
                leaf_map.insert(leaf.origin(), (s, target_points as i32 - total));
                break;
            }
            total_float += factor * leaf.point_count() as f32;
            let lp = total_float.floor() as i32;
            total_float -= lp as f32;
            total += lp;
            leaf_map.insert(leaf.origin(), (s, lp));
        }
        RandomLeafFilter { leaf_map }
    }
}

/// The bound form of [`RandomLeafFilter`].
pub struct RandomLeafBound {
    indices: Vec<i32>,
    next: usize,
}

impl BoundFilter for RandomLeafBound {
    fn valid(&mut self, index: u32, _: &Coord, _: bool) -> bool {
        let index = index as i32;
        while self.next < self.indices.len() && self.indices[self.next] < index {
            self.next += 1;
        }
        self.next < self.indices.len() && self.indices[self.next] == index
    }
}

impl IndexFilter for RandomLeafFilter {
    type Bound<'a> = RandomLeafBound;
    fn bind<'a>(&'a self, leaf: &'a LeafNode<u32>, _: &'a AttributeSet) -> RandomLeafBound {
        let Some(&(seed, count)) = self.leaf_map.get(&leaf.origin()) else {
            return RandomLeafBound { indices: Vec::new(), next: 0 };
        };
        let total = leaf.buffer()[511] as i32;
        let count = count.min(total);
        RandomLeafBound { indices: generate_random_subset(seed, count, total), next: 0 }
    }
}

/// Deterministically accepts about `percentage` percent of the points by hashing an integer
/// id attribute (`AttributeHashFilter<std::mt19937, IntType>`).
#[derive(Clone, Debug)]
pub struct AttributeHashFilter {
    index: usize,
    factor: f64,
    seed: u32,
}

impl AttributeHashFilter {
    /// Filter on the `int32` (or `int64`) attribute at position `index`.
    pub fn new(index: usize, percentage: f64, seed: u32) -> Self {
        AttributeHashFilter { index, factor: percentage / 100.0, seed }
    }
}

/// The bound form of [`AttributeHashFilter`].
pub struct AttributeHashBound<'a> {
    i32_handle: Option<AttributeHandle<'a, i32>>,
    i64_handle: Option<AttributeHandle<'a, i64>>,
    factor: f64,
    seed: u32,
}

impl BoundFilter for AttributeHashBound<'_> {
    fn valid(&mut self, index: u32, _: &Coord, _: bool) -> bool {
        let id = if let Some(h) = &self.i32_handle {
            h.get(index) as u32
        } else if let Some(h) = &self.i64_handle {
            h.get(index) as u32
        } else {
            return false;
        };
        let mut g = Mt19937::new(self.seed.wrapping_add(id));
        generate_canonical(&mut g) < self.factor
    }
}

impl IndexFilter for AttributeHashFilter {
    type Bound<'a> = AttributeHashBound<'a>;
    fn bind<'a>(&'a self, _: &'a LeafNode<u32>, attributes: &'a AttributeSet) -> AttributeHashBound<'a> {
        let array = attributes.get(self.index);
        AttributeHashBound {
            i32_handle: array.and_then(|a| AttributeHandle::new(a).ok()),
            i64_handle: array.and_then(|a| AttributeHandle::new(a).ok()),
            factor: self.factor,
            seed: self.seed,
        }
    }
}

/// Bound filters reading the `"P"` attribute.
fn position_handle(attributes: &AttributeSet) -> Option<AttributeHandle<'_, Vec3f>> {
    attributes.get_by_name("P").and_then(|a| AttributeHandle::new(a).ok())
}

/// Accepts points whose world-space position lies inside a box (`BBoxFilter`).
#[derive(Clone, Debug)]
pub struct BBoxFilter {
    bbox: BBoxd,
}

impl BBoxFilter {
    /// A filter for the world-space box `bbox_ws` of points in a grid with `transform`.
    pub fn new(transform: &Transform, bbox_ws: &BBoxd) -> Self {
        BBoxFilter { bbox: transform.world_to_index_bboxd(bbox_ws) }
    }
}

/// The bound form of [`BBoxFilter`].
pub struct BBoxBound<'a> {
    bbox: BBoxd,
    handle: Option<AttributeHandle<'a, Vec3f>>,
}

impl BoundFilter for BBoxBound<'_> {
    fn valid(&mut self, index: u32, ijk: &Coord, _: bool) -> bool {
        let Some(h) = &self.handle else { return false };
        let p = h.get(index);
        // OpenVDB computes the index-space position in single precision.
        let pos = Vec3f::new(p[0] + ijk[0] as f32, p[1] + ijk[1] as f32, p[2] + ijk[2] as f32);
        self.bbox.is_inside(&Vec3d::new(pos[0] as f64, pos[1] as f64, pos[2] as f64))
    }
}

impl IndexFilter for BBoxFilter {
    type Bound<'a> = BBoxBound<'a>;
    fn state(&self) -> IndexState {
        if self.bbox.is_empty() {
            IndexState::None
        } else {
            IndexState::Partial
        }
    }
    fn bind<'a>(&'a self, _: &'a LeafNode<u32>, attributes: &'a AttributeSet) -> BBoxBound<'a> {
        BBoxBound { bbox: self.bbox, handle: position_handle(attributes) }
    }
}

/// Accepts points where a level set, sampled trilinearly, lies strictly between `min` and
/// `max` (or outside that range if `min > max`) (`LevelSetFilter`).
pub struct LevelSetFilter<'g, T: Value> {
    grid: &'g Grid<T>,
    transform: Transform,
    min: T,
    max: T,
}

impl<'g, T: Value> LevelSetFilter<'g, T> {
    /// A filter for points in a grid with `transform`, testing the level set `grid`.
    pub fn new(grid: &'g Grid<T>, transform: &Transform, min: T, max: T) -> Self {
        LevelSetFilter { grid, transform: transform.clone(), min, max }
    }
}

/// The bound form of [`LevelSetFilter`].
pub struct LevelSetBound<'a, T: Value> {
    filter: &'a LevelSetFilter<'a, T>,
    handle: Option<AttributeHandle<'a, Vec3f>>,
}

impl<T: Value + crate::types::ArithValue + PartialOrd> BoundFilter for LevelSetBound<'_, T> {
    fn valid(&mut self, index: u32, ijk: &Coord, _: bool) -> bool {
        let Some(h) = &self.handle else { return false };
        let p = h.get(index);
        let pos = Vec3f::new(p[0] + ijk[0] as f32, p[1] + ijk[1] as f32, p[2] + ijk[2] as f32);
        let ws = self.filter.transform.index_to_world(Vec3d::new(pos[0] as f64, pos[1] as f64, pos[2] as f64));
        let ws = Vec3f::new(ws[0] as f32, ws[1] as f32, ws[2] as f32);
        let is = self.filter.grid.transform().world_to_index(Vec3d::new(ws[0] as f64, ws[1] as f64, ws[2] as f64));
        let is = Vec3d::new(is[0] as f32 as f64, is[1] as f32 as f64, is[2] as f32 as f64);
        let value = BoxSampler::sample(self.filter.grid.tree(), is);
        let (min, max) = (self.filter.min, self.filter.max);
        if min > max {
            value < max || value > min
        } else {
            value < max && value > min
        }
    }
}

impl<'g, T: Value + crate::types::ArithValue + PartialOrd> IndexFilter for LevelSetFilter<'g, T> {
    type Bound<'a>
        = LevelSetBound<'a, T>
    where
        Self: 'a;
    fn bind<'a>(&'a self, _: &'a LeafNode<u32>, attributes: &'a AttributeSet) -> LevelSetBound<'a, T> {
        LevelSetBound { filter: self, handle: position_handle(attributes) }
    }
}

/// Combines two filters with a logical and (`AND == true`) or or (`BinaryFilter`).
#[derive(Clone, Debug)]
pub struct BinaryFilter<A, B, const AND: bool = true> {
    a: A,
    b: B,
}

impl<A, B, const AND: bool> BinaryFilter<A, B, AND> {
    /// Combine two filters.
    pub fn new(a: A, b: B) -> Self {
        BinaryFilter { a, b }
    }
    fn combine(s1: IndexState, s2: IndexState) -> IndexState {
        use IndexState::*;
        if AND {
            if s1 == None || s2 == None {
                return None;
            }
        } else if s1 == None && s2 == None {
            return None;
        }
        if s1 == All && s2 == All {
            All
        } else {
            Partial
        }
    }
}

/// The bound form of [`BinaryFilter`].
pub struct BinaryBound<A, B, const AND: bool> {
    a: A,
    b: B,
}

impl<A: BoundFilter, B: BoundFilter, const AND: bool> BoundFilter for BinaryBound<A, B, AND> {
    fn valid(&mut self, index: u32, ijk: &Coord, active: bool) -> bool {
        if AND {
            self.a.valid(index, ijk, active) && self.b.valid(index, ijk, active)
        } else {
            self.a.valid(index, ijk, active) || self.b.valid(index, ijk, active)
        }
    }
}

impl<A: IndexFilter, B: IndexFilter, const AND: bool> IndexFilter for BinaryFilter<A, B, AND> {
    type Bound<'a>
        = BinaryBound<A::Bound<'a>, B::Bound<'a>, AND>
    where
        Self: 'a;
    fn state(&self) -> IndexState {
        Self::combine(self.a.state(), self.b.state())
    }
    fn leaf_state(&self, leaf: &LeafNode<u32>, attributes: &AttributeSet) -> IndexState {
        Self::combine(self.a.leaf_state(leaf, attributes), self.b.leaf_state(leaf, attributes))
    }
    fn bind<'a>(&'a self, leaf: &'a LeafNode<u32>, attributes: &'a AttributeSet) -> Self::Bound<'a> {
        BinaryBound { a: self.a.bind(leaf, attributes), b: self.b.bind(leaf, attributes) }
    }
}

impl<F: IndexFilter + ?Sized> IndexFilter for &F {
    type Bound<'a>
        = F::Bound<'a>
    where
        Self: 'a;
    fn state(&self) -> IndexState {
        (**self).state()
    }
    fn leaf_state(&self, leaf: &LeafNode<u32>, attributes: &AttributeSet) -> IndexState {
        (**self).leaf_state(leaf, attributes)
    }
    fn bind<'a>(&'a self, leaf: &'a LeafNode<u32>, attributes: &'a AttributeSet) -> Self::Bound<'a> {
        (**self).bind(leaf, attributes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_subset() {
        let s = generate_random_subset(0, 3, 10);
        assert_eq!(s.len(), 3);
        assert!(s.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(generate_random_subset(1, 20, 5), vec![0, 1, 2, 3, 4]);
        assert!(generate_random_subset(1, 0, 5).is_empty());
        // Deterministic for a given seed.
        assert_eq!(generate_random_subset(42, 5, 100), generate_random_subset(42, 5, 100));
    }
}
