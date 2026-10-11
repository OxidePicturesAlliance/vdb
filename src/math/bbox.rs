//! Axis-aligned floating-point bounding boxes (port of `openvdb/math/BBox.h`).
//!
//! Integer index-space boxes are [`CoordBBox`](super::CoordBBox); this type covers
//! OpenVDB's `BBox<Vec3T>` for floating-point `Vec3T` (e.g. `BBoxd`).

use super::maps::MapBase;
use super::math_util::{is_approx_equal, max_index, min_index, FloatTolerance};
use super::vec3::{Vec3, Vec3d};
use std::fmt;

/// An axis-aligned bounding box with floating-point corners `min` and `max`.
#[derive(Clone, Copy)]
pub struct BBox<T> {
    min: Vec3<T>,
    max: Vec3<T>,
}

/// Double-precision bounding box (OpenVDB's `BBoxd`).
pub type BBoxd = BBox<f64>;
/// Single-precision bounding box.
pub type BBoxf = BBox<f32>;

impl<T: FloatTolerance> Default for BBox<T> {
    /// An empty box: `min = +MAX`, `max = -MAX`.
    fn default() -> Self {
        BBox { min: Vec3::splat(T::max_value()), max: Vec3::splat(-T::max_value()) }
    }
}

impl<T: FloatTolerance> BBox<T> {
    /// A box with the given corners (not sorted).
    pub fn new(min: Vec3<T>, max: Vec3<T>) -> Self {
        BBox { min, max }
    }
    /// A box with the given corners, sorted component-wise if `sorted` is false.
    pub fn new_sorted(min: Vec3<T>, max: Vec3<T>, sorted: bool) -> Self {
        let mut b = BBox { min, max };
        if !sorted {
            b.sort();
        }
        b
    }
    /// A cube with lower corner `min` and edge length `length`.
    pub fn from_min_length(min: Vec3<T>, length: T) -> Self {
        BBox { min, max: min + Vec3::splat(length) }
    }
    /// A box from `[xmin, ymin, zmin, xmax, ymax, zmax]`, sorted if `sorted` is false.
    pub fn from_array(xyz: &[T; 6], sorted: bool) -> Self {
        Self::new_sorted(Vec3::new(xyz[0], xyz[1], xyz[2]), Vec3::new(xyz[3], xyz[4], xyz[5]), sorted)
    }
    /// An empty box (same as `default()`).
    pub fn empty() -> Self {
        Self::default()
    }
    /// Sort the corners so that `min <= max` component-wise.
    pub fn sort(&mut self) {
        let (a, b) = (self.min, self.max);
        self.min = a.min_elem(&b);
        self.max = a.max_elem(&b);
    }
    /// The lower corner.
    #[inline]
    pub fn min(&self) -> Vec3<T> {
        self.min
    }
    /// The upper corner.
    #[inline]
    pub fn max(&self) -> Vec3<T> {
        self.max
    }
    /// Mutable lower corner.
    #[inline]
    pub fn min_mut(&mut self) -> &mut Vec3<T> {
        &mut self.min
    }
    /// Mutable upper corner.
    #[inline]
    pub fn max_mut(&mut self) -> &mut Vec3<T> {
        &mut self.max
    }
    /// True if the box has no volume (`min >= max` along some axis).
    pub fn is_empty(&self) -> bool {
        (0..3).any(|i| self.min[i] >= self.max[i])
    }
    /// True if the box has a positive volume.
    pub fn has_volume(&self) -> bool {
        !self.is_empty()
    }
    /// True if `min < max + tolerance` on every axis.
    pub fn is_sorted(&self) -> bool {
        let t = T::tolerance();
        (0..3).all(|i| self.min[i] < self.max[i] + t)
    }
    /// The center of the box.
    pub fn get_center(&self) -> Vec3<T> {
        (self.min + self.max) * T::from(0.5).unwrap()
    }
    /// `max - min`.
    pub fn extents(&self) -> Vec3<T> {
        self.max - self.min
    }
    /// Index of the longest axis (ties resolve to the largest index).
    pub fn max_extent(&self) -> usize {
        max_index(&self.extents())
    }
    /// Index of the shortest axis (ties resolve to the largest index).
    pub fn min_extent(&self) -> usize {
        min_index(&self.extents())
    }
    /// The volume of the box.
    pub fn volume(&self) -> T {
        let e = self.extents();
        e[0] * e[1] * e[2]
    }
    /// True if the point is inside the box (with the default tolerance).
    pub fn is_inside(&self, xyz: &Vec3<T>) -> bool {
        let t = T::tolerance();
        (0..3).all(|i| xyz[i] > self.min[i] - t && xyz[i] < self.max[i] + t)
    }
    /// True if `b` lies strictly inside this box (with the default tolerance).
    pub fn is_inside_bbox(&self, b: &BBox<T>) -> bool {
        let t = T::tolerance();
        (0..3).all(|i| b.min[i] - t > self.min[i] && b.max[i] + t < self.max[i])
    }
    /// True if the boxes overlap (with the default tolerance).
    pub fn has_overlap(&self, b: &BBox<T>) -> bool {
        let t = T::tolerance();
        (0..3).all(|i| self.max[i] > b.min[i] - t && self.min[i] < b.max[i] + t)
    }
    /// Same as [`has_overlap`](Self::has_overlap).
    pub fn intersects(&self, b: &BBox<T>) -> bool {
        self.has_overlap(b)
    }
    /// Pad the box by `|padding|` on every side.
    pub fn expand_by(&mut self, padding: T) {
        let d = padding.abs();
        self.min -= Vec3::splat(d);
        self.max += Vec3::splat(d);
    }
    /// Expand to include a point.
    pub fn expand_point(&mut self, xyz: &Vec3<T>) {
        self.min = self.min.min_elem(xyz);
        self.max = self.max.max_elem(xyz);
    }
    /// Expand to include another box.
    pub fn expand_bbox(&mut self, b: &BBox<T>) {
        self.min = self.min.min_elem(&b.min);
        self.max = self.max.max_elem(&b.max);
    }
    /// Expand to include the cube with lower corner `min` and edge length `length`.
    pub fn expand_cube(&mut self, min: &Vec3<T>, length: T) {
        self.min = self.min.min_elem(min);
        self.max = self.max.max_elem(&(*min + Vec3::splat(length)));
    }
    /// Translate by `t`.
    pub fn translate(&mut self, t: &Vec3<T>) {
        self.min += *t;
        self.max += *t;
    }
    fn map_corners(&self, f: impl Fn(Vec3d) -> Vec3d) -> Self {
        let to_d = |v: Vec3<T>| v.map(|c| c.to_f64().unwrap());
        let (lo, hi) = (to_d(self.min), to_d(self.max));
        let mut out = BBox::default();
        for &(x, y, z) in &[(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)] {
            let p = Vec3d::new(
                if x == 0 { lo[0] } else { hi[0] },
                if y == 0 { lo[1] } else { hi[1] },
                if z == 0 { lo[2] } else { hi[2] },
            );
            out.expand_point(&f(p).map(|c| T::from(c).unwrap()));
        }
        out
    }
    /// The bounding box of the images of this box's eight corners under `map`.
    pub fn apply_map(&self, map: &MapBase) -> Self {
        self.map_corners(|p| map.apply_map(p))
    }
    /// The bounding box of the pre-images of this box's eight corners under `map`.
    pub fn apply_inverse_map(&self, map: &MapBase) -> Self {
        self.map_corners(|p| map.apply_inverse_map(p))
    }
}

impl<T: FloatTolerance> PartialEq for BBox<T> {
    /// Approximate equality of the corners with the default tolerance (`BBox::operator==`).
    fn eq(&self, o: &Self) -> bool {
        (0..3).all(|i| is_approx_equal(self.min[i], o.min[i]) && is_approx_equal(self.max[i], o.max[i]))
    }
}

impl<T: fmt::Debug> fmt::Debug for BBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} -> {:?}", self.min, self.max)
    }
}
impl<T: fmt::Display> fmt::Display for BBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.min, self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let mut b = BBoxd::default();
        assert!(b.is_empty());
        b.expand_point(&Vec3d::new(1.0, 2.0, 3.0));
        b.expand_point(&Vec3d::new(-1.0, 0.0, 5.0));
        assert_eq!(b.min(), Vec3d::new(-1.0, 0.0, 3.0));
        assert_eq!(b.max(), Vec3d::new(1.0, 2.0, 5.0));
        assert_eq!(b.volume(), 8.0);
        assert_eq!(b.get_center(), Vec3d::new(0.0, 1.0, 4.0));
        assert!(b.is_inside(&Vec3d::new(0.0, 1.0, 4.0)));
        assert!(!b.is_inside(&Vec3d::new(0.0, 3.0, 4.0)));
        assert_eq!(b.max_extent(), 2);
        let c = BBoxd::new_sorted(Vec3d::new(2.0, 2.0, 2.0), Vec3d::new(0.5, 0.5, 4.5), false);
        assert!(c.is_sorted());
        assert!(b.has_overlap(&c));
        assert!(!b.is_inside_bbox(&c));
        let mut d = c;
        d.expand_by(-0.25);
        assert_eq!(d.min(), Vec3d::new(0.25, 0.25, 1.75));
        d.translate(&Vec3d::splat(1.0));
        assert_eq!(d.max(), Vec3d::new(3.25, 3.25, 5.75));
        assert_eq!(BBoxd::from_min_length(Vec3d::zero(), 2.0).volume(), 8.0);
    }

    #[test]
    fn apply_map() {
        let b = BBoxd::new(Vec3d::zero(), Vec3d::splat(1.0));
        let m = MapBase::uniform_scale(2.0).post_translate(Vec3d::new(1.0, 0.0, 0.0));
        let w = b.apply_map(&m);
        assert_eq!(w, BBoxd::new(Vec3d::new(1.0, 0.0, 0.0), Vec3d::new(3.0, 2.0, 2.0)));
        assert_eq!(w.apply_inverse_map(&m), b);
    }
}
