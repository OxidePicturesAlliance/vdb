//! Rays and ray marching (port of `openvdb/math/Ray.h` and the `DDA` class of
//! `openvdb/math/DDA.h`).
//!
//! The hierarchical tree walkers of `DDA.h` (`LevelSetHDDA`, `VolumeHDDA`) depend on tree
//! accessors and belong with the ray-intersection tools; the voxel-level [`Dda`] here is the
//! building block they use.

use super::bbox::BBox;
use super::coord::Coord;
use super::maps::MapBase;
use super::math_util::{is_approx_zero, is_rel_or_approx_equal, min_index, FloatTolerance};
use super::transform::Transform;
use super::vec3::{Vec3, Vec3d};
use std::fmt;

/// A closed time interval `[t0, t1]` along a ray (`Ray::TimeSpan`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct TimeSpan<T> {
    /// Start time.
    pub t0: T,
    /// End time.
    pub t1: T,
}

impl<T: FloatTolerance> TimeSpan<T> {
    /// A span from `t0` to `t1`.
    pub fn new(t0: T, t1: T) -> Self {
        TimeSpan { t0, t1 }
    }
    /// Set both times.
    pub fn set(&mut self, t0: T, t1: T) {
        self.t0 = t0;
        self.t1 = t1;
    }
    /// The times as a tuple.
    pub fn get(&self) -> (T, T) {
        (self.t0, self.t1)
    }
    /// True if `t1 - t0 > eps` (OpenVDB's default `eps` is `math::Delta<T>`; see
    /// [`is_valid`](Self::is_valid)).
    pub fn valid(&self, eps: T) -> bool {
        (self.t1 - self.t0) > eps
    }
    /// [`valid`](Self::valid) with the default epsilon `math::Delta<T>`.
    pub fn is_valid(&self) -> bool {
        self.valid(T::delta())
    }
    /// The midpoint.
    pub fn mid(&self) -> T {
        T::from(0.5).unwrap() * (self.t0 + self.t1)
    }
    /// Multiply both times by `s > 0`.
    pub fn scale(&mut self, s: T) {
        debug_assert!(s > T::zero());
        self.t0 = self.t0 * s;
        self.t1 = self.t1 * s;
    }
    /// True if `t0 <= t <= t1`.
    pub fn test(&self, t: T) -> bool {
        t >= self.t0 && t <= self.t1
    }
}

/// A ray `eye + t * dir` restricted to a time span.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray<T = f64> {
    eye: Vec3<T>,
    dir: Vec3<T>,
    inv_dir: Vec3<T>,
    time_span: TimeSpan<T>,
}

/// Double-precision ray.
pub type Rayd = Ray<f64>;
/// Single-precision ray.
pub type Rayf = Ray<f32>;

impl<T: FloatTolerance> Default for Ray<T> {
    /// Eye at the origin, direction +x, times `[Delta, MAX]`.
    fn default() -> Self {
        Self::new(Vec3::zero(), Vec3::new(T::one(), T::zero(), T::zero()))
    }
}

impl<T: FloatTolerance> Ray<T> {
    /// A ray with the default time span `[math::Delta<T>, T::MAX]`.
    pub fn new(eye: Vec3<T>, direction: Vec3<T>) -> Self {
        Self::with_times(eye, direction, T::delta(), T::max_value())
    }
    /// A ray with the given time span.
    pub fn with_times(eye: Vec3<T>, direction: Vec3<T>, t0: T, t1: T) -> Self {
        Ray { eye, dir: direction, inv_dir: direction.map(|d| T::one() / d), time_span: TimeSpan { t0, t1 } }
    }
    /// Set the origin.
    pub fn set_eye(&mut self, eye: Vec3<T>) {
        self.eye = eye;
    }
    /// Set the direction (and its cached reciprocal).
    pub fn set_dir(&mut self, dir: Vec3<T>) {
        self.dir = dir;
        self.inv_dir = dir.map(|d| T::one() / d);
    }
    /// Set the start time.
    pub fn set_min_time(&mut self, t0: T) {
        debug_assert!(t0 > T::zero());
        self.time_span.t0 = t0;
    }
    /// Set the end time.
    pub fn set_max_time(&mut self, t1: T) {
        debug_assert!(t1 > T::zero());
        self.time_span.t1 = t1;
    }
    /// Set both times.
    pub fn set_times(&mut self, t0: T, t1: T) {
        debug_assert!(t0 > T::zero() && t1 > T::zero());
        self.time_span.set(t0, t1);
    }
    /// Reset the times to the default `[Delta, MAX]`.
    pub fn reset_times(&mut self) {
        self.time_span.set(T::delta(), T::max_value());
    }
    /// Scale both times.
    pub fn scale_times(&mut self, scale: T) {
        self.time_span.scale(scale);
    }
    /// Reset eye, direction and times.
    pub fn reset(&mut self, eye: Vec3<T>, direction: Vec3<T>, t0: T, t1: T) {
        self.set_eye(eye);
        self.set_dir(direction);
        self.set_times(t0, t1);
    }
    /// The origin.
    pub fn eye(&self) -> Vec3<T> {
        self.eye
    }
    /// The direction.
    pub fn dir(&self) -> Vec3<T> {
        self.dir
    }
    /// The component-wise reciprocal of the direction.
    pub fn inv_dir(&self) -> Vec3<T> {
        self.inv_dir
    }
    /// The start time.
    pub fn t0(&self) -> T {
        self.time_span.t0
    }
    /// The end time.
    pub fn t1(&self) -> T {
        self.time_span.t1
    }
    /// The time span.
    pub fn time_span(&self) -> TimeSpan<T> {
        self.time_span
    }
    /// The position at `time` (`operator()`).
    pub fn at(&self, time: T) -> Vec3<T> {
        self.eye + self.dir * time
    }
    /// The position at the start time.
    pub fn start(&self) -> Vec3<T> {
        self.at(self.time_span.t0)
    }
    /// The position at the end time.
    pub fn end(&self) -> Vec3<T> {
        self.at(self.time_span.t1)
    }
    /// The position at the middle of the time span.
    pub fn mid(&self) -> Vec3<T> {
        self.at(self.time_span.mid())
    }
    /// True if the time span is longer than `eps` (OpenVDB's default is `Delta<float>` = 1e-5).
    pub fn valid(&self, eps: T) -> bool {
        self.time_span.valid(eps)
    }
    /// [`valid`](Self::valid) with OpenVDB's default epsilon `math::Delta<float>` (1e-5).
    pub fn is_valid(&self) -> bool {
        self.valid(T::from(super::DELTA_F32).unwrap())
    }
    /// True if `time` lies in the time span.
    pub fn test(&self, time: T) -> bool {
        self.time_span.test(time)
    }

    fn to_d(v: Vec3<T>) -> Vec3d {
        v.map(|c| c.to_f64().unwrap())
    }
    fn from_d(v: Vec3d) -> Vec3<T> {
        v.map(|c| T::from(c).unwrap())
    }
    /// The ray transformed by a linear map; the direction is renormalized and the times scaled
    /// accordingly. The direction must be unit length.
    pub fn apply_map(&self, map: &MapBase) -> Self {
        debug_assert!(map.is_linear());
        debug_assert!(is_rel_or_approx_equal(self.dir.length(), T::one(), T::tolerance(), T::delta()));
        let eye = Self::from_d(map.apply_map(Self::to_d(self.eye)));
        let dir = Self::from_d(map.apply_jacobian(Self::to_d(self.dir)));
        let length = dir.length();
        Self::with_times(eye, dir / length, length * self.time_span.t0, length * self.time_span.t1)
    }
    /// The ray transformed by the inverse of a linear map. The direction must be unit length.
    pub fn apply_inverse_map(&self, map: &MapBase) -> Self {
        debug_assert!(map.is_linear());
        debug_assert!(is_rel_or_approx_equal(self.dir.length(), T::one(), T::tolerance(), T::delta()));
        let eye = Self::from_d(map.apply_inverse_map(Self::to_d(self.eye)));
        let dir = Self::from_d(map.apply_inverse_jacobian(Self::to_d(self.dir)));
        let length = dir.length();
        Self::with_times(eye, dir / length, length * self.time_span.t0, length * self.time_span.t1)
    }
    /// Index-space ray to world space (OpenVDB's `indexToWorld(grid)`, given the grid's transform).
    pub fn index_to_world(&self, xform: &Transform) -> Self {
        self.apply_map(xform.map())
    }
    /// World-space ray to index space (OpenVDB's `worldToIndex(grid)`, given the grid's transform).
    pub fn world_to_index(&self, xform: &Transform) -> Self {
        self.apply_inverse_map(xform.map())
    }
    /// Intersection times `(t0, t1)` with a sphere, clamped to the ray's time span; `None` if the
    /// ray misses.
    pub fn intersects_sphere_times(&self, center: Vec3<T>, radius: T) -> Option<(T, T)> {
        let origin = self.eye - center;
        let two = T::one() + T::one();
        let four = two + two;
        let a = self.dir.length_sqr();
        let b = two * self.dir.dot(&origin);
        let c = origin.length_sqr() - radius * radius;
        let d = b * b - four * a * c;
        if d < T::zero() {
            return None;
        }
        let q = T::from(-0.5).unwrap() * if b < T::zero() { b + d.sqrt() } else { b - d.sqrt() };
        let mut t0 = q / a;
        let mut t1 = c / q;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        if t0 < self.time_span.t0 {
            t0 = self.time_span.t0;
        }
        if t1 > self.time_span.t1 {
            t1 = self.time_span.t1;
        }
        if t0 <= t1 {
            Some((t0, t1))
        } else {
            None
        }
    }
    /// True if the ray hits the sphere within its time span.
    pub fn intersects_sphere(&self, center: Vec3<T>, radius: T) -> bool {
        self.intersects_sphere_times(center, radius).is_some()
    }
    /// Clip the time span to the sphere; returns `true` on a hit (else leaves the ray unchanged).
    pub fn clip_sphere(&mut self, center: Vec3<T>, radius: T) -> bool {
        match self.intersects_sphere_times(center, radius) {
            Some((t0, t1)) => {
                self.time_span.set(t0, t1);
                true
            }
            None => false,
        }
    }
    /// Intersection times with an axis-aligned box given by its corners, clamped to the
    /// time span (slab test).
    pub fn intersects_box_times(&self, min: Vec3<T>, max: Vec3<T>) -> Option<(T, T)> {
        let (mut t0, mut t1) = self.time_span.get();
        for i in 0..3 {
            let mut a = (min[i] - self.eye[i]) * self.inv_dir[i];
            let mut b = (max[i] - self.eye[i]) * self.inv_dir[i];
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            if a > t0 {
                t0 = a;
            }
            if b < t1 {
                t1 = b;
            }
            if t0 > t1 {
                return None;
            }
        }
        Some((t0, t1))
    }
    /// Intersection times with a floating-point bounding box.
    pub fn intersects_bbox_times(&self, bbox: &BBox<T>) -> Option<(T, T)> {
        self.intersects_box_times(bbox.min(), bbox.max())
    }
    /// True if the ray hits the box within its time span.
    pub fn intersects_bbox(&self, bbox: &BBox<T>) -> bool {
        self.intersects_bbox_times(bbox).is_some()
    }
    /// Intersection times with an integer coordinate box (corners taken as given).
    pub fn intersects_coord_bbox_times(&self, bbox: &super::CoordBBox) -> Option<(T, T)> {
        let cv = |c: Coord| Vec3::new(T::from(c[0]).unwrap(), T::from(c[1]).unwrap(), T::from(c[2]).unwrap());
        self.intersects_box_times(cv(bbox.min), cv(bbox.max))
    }
    /// Clip the time span to the box; returns `true` on a hit.
    pub fn clip_bbox(&mut self, bbox: &BBox<T>) -> bool {
        match self.intersects_bbox_times(bbox) {
            Some((t0, t1)) => {
                self.time_span.set(t0, t1);
                true
            }
            None => false,
        }
    }
    /// Clip the time span to an integer coordinate box; returns `true` on a hit.
    pub fn clip_coord_bbox(&mut self, bbox: &super::CoordBBox) -> bool {
        match self.intersects_coord_bbox_times(bbox) {
            Some((t0, t1)) => {
                self.time_span.set(t0, t1);
                true
            }
            None => false,
        }
    }
    /// Intersection time with the plane `{x : x·normal = distance}`, if the ray is not parallel
    /// to it and the hit lies within the time span.
    pub fn intersects_plane(&self, normal: Vec3<T>, distance: T) -> Option<T> {
        let cos_angle = self.dir.dot(&normal);
        if is_approx_zero(cos_angle) {
            return None;
        }
        let t = (distance - self.eye.dot(&normal)) / cos_angle;
        if self.test(t) {
            Some(t)
        } else {
            None
        }
    }
    /// Intersection time with the plane through `point` with the given `normal`.
    pub fn intersects_plane_point(&self, normal: Vec3<T>, point: Vec3<T>) -> Option<T> {
        self.intersects_plane(normal, point.dot(&normal))
    }
}

impl<T: fmt::Display> fmt::Display for Ray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "eye={} dir={} 1/dir={} t0={} t1={}",
            self.eye, self.dir, self.inv_dir, self.time_span.t0, self.time_span.t1
        )
    }
}

/// A digital differential analyzer that steps a ray through the cells of a uniform grid of
/// `2^log2_dim`-sized blocks of voxels (`math::DDA<RayT, Log2Dim>`, with the template level
/// turned into a const generic).
#[derive(Clone, Copy, Debug)]
pub struct Dda<T, const LOG2_DIM: u32 = 0> {
    t0: T,
    t1: T,
    voxel: Coord,
    step: Coord,
    delta: Vec3<T>,
    next: Vec3<T>,
}

impl<T: FloatTolerance, const LOG2_DIM: u32> Dda<T, LOG2_DIM> {
    /// The block size, `1 << LOG2_DIM`.
    pub const DIM: i32 = 1 << LOG2_DIM;

    /// A DDA over the ray's full time span.
    pub fn new(ray: &Ray<T>) -> Self {
        Self::with_times(ray, ray.t0(), ray.t1())
    }
    /// A DDA starting at `start_time` and ending at the ray's end time.
    pub fn with_start(ray: &Ray<T>, start_time: T) -> Self {
        Self::with_times(ray, start_time, ray.t1())
    }
    /// A DDA over `[start_time, max_time]`.
    pub fn with_times(ray: &Ray<T>, start_time: T, max_time: T) -> Self {
        let mut d = Dda {
            t0: T::zero(),
            t1: T::zero(),
            voxel: Coord::origin(),
            step: Coord::origin(),
            delta: Vec3::zero(),
            next: Vec3::zero(),
        };
        d.init(ray, start_time, max_time);
        d
    }
    /// Re-initialize for a new ray and time interval.
    pub fn init(&mut self, ray: &Ray<T>, start_time: T, max_time: T) {
        debug_assert!(start_time <= max_time);
        let dim = Self::DIM;
        self.t0 = start_time;
        self.t1 = max_time;
        let pos = ray.at(self.t0);
        let (dir, inv) = (ray.dir(), ray.inv_dir());
        let p = pos.map(|c| c.to_f64().unwrap());
        self.voxel = Coord::floor(p).align_down(dim);
        let tf = |i: i32| T::from(i).unwrap();
        for axis in 0..3 {
            if dir[axis] == T::zero() {
                self.step[axis] = 0;
                self.next[axis] = T::max_value();
                self.delta[axis] = T::max_value();
            } else if inv[axis] > T::zero() {
                self.step[axis] = dim;
                self.next[axis] = self.t0 + (tf(self.voxel[axis]) + tf(dim) - pos[axis]) * inv[axis];
                self.delta[axis] = tf(self.step[axis]) * inv[axis];
            } else {
                self.step[axis] = -dim;
                self.next[axis] = self.t0 + (tf(self.voxel[axis]) - pos[axis]) * inv[axis];
                self.delta[axis] = tf(self.step[axis]) * inv[axis];
            }
        }
    }
    /// Re-initialize over the ray's full time span.
    pub fn init_ray(&mut self, ray: &Ray<T>) {
        self.init(ray, ray.t0(), ray.t1());
    }
    /// Step to the next cell; returns `false` once the current time exceeds the maximum time.
    pub fn step(&mut self) -> bool {
        let axis = min_index(&self.next);
        self.t0 = self.next[axis];
        self.next[axis] = self.next[axis] + self.delta[axis];
        self.voxel[axis] += self.step[axis];
        self.t0 <= self.t1
    }
    /// The lower corner of the current cell.
    pub fn voxel(&self) -> Coord {
        self.voxel
    }
    /// The time at which the ray entered the current cell.
    pub fn time(&self) -> T {
        self.t0
    }
    /// The maximum time.
    pub fn max_time(&self) -> T {
        self.t1
    }
    /// The time at which the ray will leave the current cell (clamped to the maximum time).
    pub fn next(&self) -> T {
        let mut m = self.t1;
        for i in 0..3 {
            if self.next[i] < m {
                m = self.next[i];
            }
        }
        m
    }
}

impl<T: fmt::Display + FloatTolerance, const LOG2_DIM: u32> fmt::Display for Dda<T, LOG2_DIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dim={} time={} next()={} voxel={}", 1 << LOG2_DIM, self.t0, self.next(), self.voxel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Axis, BBoxd, CoordBBox};

    #[test]
    fn intersections() {
        let ray = Rayd::new(Vec3d::new(-5.0, 0.5, 0.5), Vec3d::new(1.0, 0.0, 0.0));
        let (t0, t1) = ray.intersects_bbox_times(&BBoxd::new(Vec3d::zero(), Vec3d::splat(1.0))).unwrap();
        assert_eq!((t0, t1), (5.0, 6.0));
        assert!(!ray.intersects_bbox(&BBoxd::new(Vec3d::new(0.0, 2.0, 0.0), Vec3d::splat(3.0))));
        let (s0, s1) = ray.intersects_sphere_times(Vec3d::new(0.0, 0.5, 0.5), 2.0).unwrap();
        assert!((s0 - 3.0).abs() < 1e-12 && (s1 - 7.0).abs() < 1e-12);
        assert!(!ray.intersects_sphere(Vec3d::new(0.0, 5.0, 0.0), 1.0));
        let t = ray.intersects_plane(Vec3d::new(1.0, 0.0, 0.0), 2.0).unwrap();
        assert_eq!(t, 7.0);
        assert!(ray.intersects_plane(Vec3d::new(0.0, 1.0, 0.0), 2.0).is_none());
        let mut r2 = ray;
        assert!(r2.clip_coord_bbox(&CoordBBox::from_coords(0, 0, 0, 3, 3, 3)));
        assert_eq!((r2.t0(), r2.t1()), (5.0, 8.0));
        assert_eq!(r2.start(), Vec3d::new(0.0, 0.5, 0.5));
        assert!(r2.valid(1e-5));
    }

    #[test]
    fn map_application() {
        let mut xform = Transform::linear(0.5);
        xform.post_rotate(0.3, Axis::Z);
        xform.post_translate(Vec3d::new(1.0, 2.0, 3.0));
        let ray = Rayd::with_times(Vec3d::new(1.0, 1.0, 1.0), Vec3d::new(0.0, 0.6, 0.8), 1.0, 4.0);
        let w = ray.index_to_world(&xform);
        assert!((w.t0() - 0.5).abs() < 1e-12 && (w.t1() - 2.0).abs() < 1e-12);
        assert!(w.start().is_approx_equal(&xform.index_to_world(ray.start()), 1e-12));
        assert!(w.end().is_approx_equal(&xform.index_to_world(ray.end()), 1e-12));
        let back = w.world_to_index(&xform);
        assert!(back.eye().is_approx_equal(&ray.eye(), 1e-12));
        assert!(back.dir().is_approx_equal(&ray.dir(), 1e-12));
    }

    #[test]
    fn dda_steps_voxels() {
        let ray = Rayd::with_times(Vec3d::new(0.5, 0.5, 0.5), Vec3d::new(1.0, 0.0, 0.0), 0.0, 3.25);
        let mut dda: Dda<f64> = Dda::new(&ray);
        let mut voxels = vec![dda.voxel()];
        while dda.step() {
            voxels.push(dda.voxel());
        }
        assert_eq!(voxels, vec![Coord::new(0, 0, 0), Coord::new(1, 0, 0), Coord::new(2, 0, 0), Coord::new(3, 0, 0)]);

        // Blocks of 8 voxels along a diagonal ray.
        let d = Vec3d::new(1.0, 1.0, 0.0).unit();
        let ray = Rayd::with_times(Vec3d::new(-1.0, 0.5, 0.5), d, 0.0, 20.0);
        let mut dda: Dda<f64, 3> = Dda::new(&ray);
        assert_eq!(dda.voxel(), Coord::new(-8, 0, 0));
        let mut cells = vec![dda.voxel()];
        let mut last = dda.time();
        while dda.step() {
            assert!(dda.time() >= last);
            last = dda.time();
            cells.push(dda.voxel());
        }
        for c in &cells {
            assert_eq!(c[0] % 8, 0);
            assert_eq!(c[1] % 8, 0);
        }
        assert_eq!(cells[1], Coord::new(0, 0, 0));
        assert!(cells.contains(&Coord::new(8, 8, 0)));
    }
}
