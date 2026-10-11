//! Integer grid coordinates and bounding boxes (port of `openvdb/math/Coord.h`).

use super::vec3::{Vec3, Vec3d, Vec3i};
use std::fmt;
use std::ops::{Add, AddAssign, BitAnd, Index, IndexMut, Neg, Sub, SubAssign};

/// Signed (x, y, z) 32-bit integer coordinates in index space.
///
/// Ordering is lexicographic (x, then y, then z), matching `openvdb::math::Coord::operator<`,
/// which determines the order of root-node tables and therefore of serialized data.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[repr(C)]
pub struct Coord(pub [i32; 3]);

impl Coord {
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Coord([x, y, z])
    }
    #[inline]
    pub const fn splat(v: i32) -> Self {
        Coord([v, v, v])
    }
    #[inline]
    pub const fn origin() -> Self {
        Coord([0, 0, 0])
    }
    #[inline]
    pub const fn min_coord() -> Self {
        Coord([i32::MIN; 3])
    }
    #[inline]
    pub const fn max_coord() -> Self {
        Coord([i32::MAX; 3])
    }
    #[inline]
    pub const fn x(&self) -> i32 {
        self.0[0]
    }
    #[inline]
    pub const fn y(&self) -> i32 {
        self.0[1]
    }
    #[inline]
    pub const fn z(&self) -> i32 {
        self.0[2]
    }
    /// Round each component of a floating-point position to the nearest integer.
    #[inline]
    pub fn round(xyz: Vec3d) -> Self {
        Coord([(xyz[0] + 0.5).floor() as i32, (xyz[1] + 0.5).floor() as i32, (xyz[2] + 0.5).floor() as i32])
    }
    /// Floor each component of a floating-point position.
    #[inline]
    pub fn floor(xyz: Vec3d) -> Self {
        Coord([xyz[0].floor() as i32, xyz[1].floor() as i32, xyz[2].floor() as i32])
    }
    /// Ceil each component of a floating-point position.
    #[inline]
    pub fn ceil(xyz: Vec3d) -> Self {
        Coord([xyz[0].ceil() as i32, xyz[1].ceil() as i32, xyz[2].ceil() as i32])
    }
    #[inline]
    pub fn offset_by(&self, dx: i32, dy: i32, dz: i32) -> Self {
        Coord([self.0[0].wrapping_add(dx), self.0[1].wrapping_add(dy), self.0[2].wrapping_add(dz)])
    }
    #[inline]
    pub fn offset(&self, n: i32) -> Self {
        self.offset_by(n, n, n)
    }
    /// Component-wise minimum.
    #[inline]
    pub fn min_elem(&self, o: &Coord) -> Coord {
        Coord([self.0[0].min(o.0[0]), self.0[1].min(o.0[1]), self.0[2].min(o.0[2])])
    }
    /// Component-wise maximum.
    #[inline]
    pub fn max_elem(&self, o: &Coord) -> Coord {
        Coord([self.0[0].max(o.0[0]), self.0[1].max(o.0[1]), self.0[2].max(o.0[2])])
    }
    #[inline]
    pub fn min_component(&self) -> i32 {
        self.0[0].min(self.0[1]).min(self.0[2])
    }
    #[inline]
    pub fn max_component(&self) -> i32 {
        self.0[0].max(self.0[1]).max(self.0[2])
    }
    /// Index of the smallest component (ties resolved toward the lower index).
    #[inline]
    pub fn min_index(&self) -> usize {
        Vec3i::from(self.0).min_index()
    }
    /// Index of the largest component.
    #[inline]
    pub fn max_index(&self) -> usize {
        Vec3i::from(self.0).max_index()
    }
    /// Bitwise-and every component with `mask` (`align_down(dim)` is `align_mask(!(dim - 1))`).
    #[inline]
    pub fn align_mask(&self, mask: i32) -> Coord {
        Coord([self.0[0] & mask, self.0[1] & mask, self.0[2] & mask])
    }
    /// Mask off the low bits: `self & !(dim - 1)` component-wise, where `dim` is a power of 2.
    #[inline]
    pub fn align_down(&self, dim: i32) -> Coord {
        let m = !(dim - 1);
        Coord([self.0[0] & m, self.0[1] & m, self.0[2] & m])
    }
    #[inline]
    pub fn as_vec3d(&self) -> Vec3d {
        Vec3d::new(self.0[0] as f64, self.0[1] as f64, self.0[2] as f64)
    }
    #[inline]
    pub fn as_vec3i(&self) -> Vec3i {
        Vec3(self.0)
    }
    /// True if every component of `self` is less than the corresponding one of `o`.
    #[inline]
    pub fn all_lt(&self, o: &Coord) -> bool {
        self.0[0] < o.0[0] && self.0[1] < o.0[1] && self.0[2] < o.0[2]
    }
    /// True if every component of `self` is less than or equal to the corresponding one of `o`.
    #[inline]
    pub fn all_le(&self, o: &Coord) -> bool {
        self.0[0] <= o.0[0] && self.0[1] <= o.0[1] && self.0[2] <= o.0[2]
    }
    /// A hash suitable for spatial hash tables (same as OpenVDB's `Coord::hash<Log2N>`).
    #[inline]
    pub fn spatial_hash(&self, log2n: u32) -> usize {
        let h = ((self.0[0] as u32).wrapping_mul(73856093))
            ^ ((self.0[1] as u32).wrapping_mul(19349669))
            ^ ((self.0[2] as u32).wrapping_mul(83492791));
        (h & ((1u32 << log2n) - 1)) as usize
    }
}

impl Index<usize> for Coord {
    type Output = i32;
    #[inline]
    fn index(&self, i: usize) -> &i32 {
        &self.0[i]
    }
}
impl IndexMut<usize> for Coord {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut i32 {
        &mut self.0[i]
    }
}
impl Add for Coord {
    type Output = Coord;
    #[inline]
    fn add(self, o: Coord) -> Coord {
        self.offset_by(o.0[0], o.0[1], o.0[2])
    }
}
impl Sub for Coord {
    type Output = Coord;
    #[inline]
    fn sub(self, o: Coord) -> Coord {
        Coord([self.0[0].wrapping_sub(o.0[0]), self.0[1].wrapping_sub(o.0[1]), self.0[2].wrapping_sub(o.0[2])])
    }
}
impl AddAssign for Coord {
    #[inline]
    fn add_assign(&mut self, o: Coord) {
        *self = *self + o;
    }
}
impl SubAssign for Coord {
    #[inline]
    fn sub_assign(&mut self, o: Coord) {
        *self = *self - o;
    }
}
impl Neg for Coord {
    type Output = Coord;
    #[inline]
    fn neg(self) -> Coord {
        Coord([-self.0[0], -self.0[1], -self.0[2]])
    }
}
impl BitAnd<i32> for Coord {
    type Output = Coord;
    #[inline]
    fn bitand(self, m: i32) -> Coord {
        Coord([self.0[0] & m, self.0[1] & m, self.0[2] & m])
    }
}
impl From<[i32; 3]> for Coord {
    fn from(a: [i32; 3]) -> Self {
        Coord(a)
    }
}
impl From<Vec3i> for Coord {
    fn from(v: Vec3i) -> Self {
        Coord(v.0)
    }
}
impl fmt::Debug for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.0[0], self.0[1], self.0[2])
    }
}
impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// An axis-aligned bounding box of integer coordinates, with *inclusive* min and max corners.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoordBBox {
    pub min: Coord,
    pub max: Coord,
}

impl Default for CoordBBox {
    /// The default box is empty (min = i32::MAX, max = i32::MIN).
    fn default() -> Self {
        Self::empty()
    }
}

impl CoordBBox {
    #[inline]
    pub const fn new(min: Coord, max: Coord) -> Self {
        CoordBBox { min, max }
    }
    /// An empty bounding box (inverted extremes).
    #[inline]
    pub const fn empty() -> Self {
        CoordBBox { min: Coord::max_coord(), max: Coord::min_coord() }
    }
    /// The largest representable bounding box.
    #[inline]
    pub const fn inf() -> Self {
        CoordBBox { min: Coord::min_coord(), max: Coord::max_coord() }
    }
    /// A cube with the given min corner and edge length (in voxels).
    #[inline]
    pub fn create_cube(min: Coord, dim: i32) -> Self {
        CoordBBox { min, max: min.offset(dim - 1) }
    }
    #[inline]
    pub fn from_coords(xmin: i32, ymin: i32, zmin: i32, xmax: i32, ymax: i32, zmax: i32) -> Self {
        CoordBBox { min: Coord::new(xmin, ymin, zmin), max: Coord::new(xmax, ymax, zmax) }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.0[0] > self.max.0[0] || self.min.0[1] > self.max.0[1] || self.min.0[2] > self.max.0[2]
    }
    #[inline]
    pub fn has_volume(&self) -> bool {
        !self.is_empty()
    }
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::empty();
    }
    /// Edge lengths (`max - min + 1`).
    #[inline]
    pub fn dim(&self) -> Coord {
        if self.is_empty() {
            Coord::origin()
        } else {
            self.max.offset(1) - self.min
        }
    }
    /// Number of voxels in the box.
    #[inline]
    pub fn volume(&self) -> u64 {
        let d = self.dim();
        d.0[0] as u64 * d.0[1] as u64 * d.0[2] as u64
    }
    #[inline]
    pub fn is_inside(&self, xyz: &Coord) -> bool {
        self.min.all_le(xyz) && xyz.all_le(&self.max)
    }
    /// True if `b` is entirely inside this box.
    #[inline]
    pub fn is_inside_bbox(&self, b: &CoordBBox) -> bool {
        self.min.all_le(&b.min) && b.max.all_le(&self.max)
    }
    #[inline]
    pub fn has_overlap(&self, b: &CoordBBox) -> bool {
        !(self.max.0[0] < b.min.0[0]
            || self.max.0[1] < b.min.0[1]
            || self.max.0[2] < b.min.0[2]
            || self.min.0[0] > b.max.0[0]
            || self.min.0[1] > b.max.0[1]
            || self.min.0[2] > b.max.0[2])
    }
    /// Expand to include a coordinate.
    #[inline]
    pub fn expand_coord(&mut self, xyz: Coord) {
        self.min = self.min.min_elem(&xyz);
        self.max = self.max.max_elem(&xyz);
    }
    /// Expand to include another box.
    #[inline]
    pub fn expand_bbox(&mut self, b: &CoordBBox) {
        self.min = self.min.min_elem(&b.min);
        self.max = self.max.max_elem(&b.max);
    }
    /// Expand to include a cube with the given min corner and edge length.
    #[inline]
    pub fn expand_cube(&mut self, min: Coord, dim: i32) {
        self.min = self.min.min_elem(&min);
        self.max = self.max.max_elem(&min.offset(dim - 1));
    }
    /// Pad the box by `n` voxels on each side (negative values shrink it).
    #[inline]
    pub fn expand(&mut self, n: i32) {
        self.min = self.min.offset(-n);
        self.max = self.max.offset(n);
    }
    #[inline]
    pub fn expanded(&self, n: i32) -> Self {
        let mut b = *self;
        b.expand(n);
        b
    }
    /// Intersect with another box.
    #[inline]
    pub fn intersect(&mut self, b: &CoordBBox) {
        self.min = self.min.max_elem(&b.min);
        self.max = self.max.min_elem(&b.max);
    }
    #[inline]
    pub fn intersection(&self, b: &CoordBBox) -> Self {
        let mut r = *self;
        r.intersect(b);
        r
    }
    #[inline]
    pub fn translate(&mut self, t: Coord) {
        self.min += t;
        self.max += t;
    }
    #[inline]
    pub fn get_center(&self) -> Vec3d {
        (self.min.as_vec3d() + self.max.as_vec3d()) * 0.5
    }
    /// Index of the longest axis.
    #[inline]
    pub fn max_extent(&self) -> usize {
        self.dim().max_index()
    }
    /// Return the eight corners of the box.
    pub fn corners(&self) -> [Coord; 8] {
        let (a, b) = (self.min, self.max);
        [
            Coord::new(a[0], a[1], a[2]),
            Coord::new(a[0], a[1], b[2]),
            Coord::new(a[0], b[1], a[2]),
            Coord::new(a[0], b[1], b[2]),
            Coord::new(b[0], a[1], a[2]),
            Coord::new(b[0], a[1], b[2]),
            Coord::new(b[0], b[1], a[2]),
            Coord::new(b[0], b[1], b[2]),
        ]
    }
    /// Iterate over every coordinate in the box in z-fastest (x, y, z) order.
    pub fn iter(&self) -> CoordBBoxIter {
        CoordBBoxIter { bbox: *self, cur: self.min, done: self.is_empty() }
    }
    /// Split the box into two halves along its longest axis, if it has at least two voxels on it.
    pub fn split(&self) -> Option<(CoordBBox, CoordBBox)> {
        let n = self.max_extent();
        if self.max[n] <= self.min[n] {
            return None;
        }
        let mid = self.min[n] + (self.max[n] - self.min[n]) / 2;
        let mut a = *self;
        let mut b = *self;
        a.max[n] = mid;
        b.min[n] = mid + 1;
        Some((a, b))
    }
}

impl fmt::Debug for CoordBBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} -> {:?}", self.min, self.max)
    }
}

impl IntoIterator for CoordBBox {
    type Item = Coord;
    type IntoIter = CoordBBoxIter;
    fn into_iter(self) -> CoordBBoxIter {
        self.iter()
    }
}

/// Iterator over the coordinates of a [`CoordBBox`] (z varies fastest).
#[derive(Clone)]
pub struct CoordBBoxIter {
    bbox: CoordBBox,
    cur: Coord,
    done: bool,
}

impl Iterator for CoordBBoxIter {
    type Item = Coord;
    #[inline]
    fn next(&mut self) -> Option<Coord> {
        if self.done {
            return None;
        }
        let r = self.cur;
        if self.cur[2] < self.bbox.max[2] {
            self.cur[2] += 1;
        } else {
            self.cur[2] = self.bbox.min[2];
            if self.cur[1] < self.bbox.max[1] {
                self.cur[1] += 1;
            } else {
                self.cur[1] = self.bbox.min[1];
                if self.cur[0] < self.bbox.max[0] {
                    self.cur[0] += 1;
                } else {
                    self.done = true;
                }
            }
        }
        Some(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coord_ordering_is_lexicographic() {
        assert!(Coord::new(0, 5, 5) < Coord::new(1, 0, 0));
        assert!(Coord::new(1, 0, 5) < Coord::new(1, 1, 0));
    }
    #[test]
    fn bbox_iter_and_volume() {
        let b = CoordBBox::from_coords(0, 0, 0, 1, 2, 3);
        assert_eq!(b.volume(), 24);
        assert_eq!(b.iter().count(), 24);
        assert_eq!(b.iter().next(), Some(Coord::new(0, 0, 0)));
        assert_eq!(b.iter().nth(1), Some(Coord::new(0, 0, 1)));
        assert!(CoordBBox::empty().is_empty());
        assert_eq!(CoordBBox::empty().iter().count(), 0);
    }
    #[test]
    fn bbox_split() {
        let b = CoordBBox::from_coords(0, 0, 0, 7, 3, 3);
        let (l, r) = b.split().unwrap();
        assert_eq!(l.volume() + r.volume(), b.volume());
    }
}
