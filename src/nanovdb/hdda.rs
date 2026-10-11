//! Rays, (hierarchical) digital differential analyzers and ray marching in index space
//! (port of `nanovdb/math/Ray.h` and `nanovdb/math/HDDA.h`).

use super::build_types::BuildType;
use super::data::Map;
use super::sampler::SampleReal;
use super::tree::ReadAccessor;
use crate::math::{Coord, CoordBBox, Vec3d};
use num_traits::Float;

/// Real types usable by [`Ray`] (`f32`, `f64`).
pub trait RayReal: Float + SampleReal + std::fmt::Debug {
    /// `Delta<T>::value()`: 1e-5 for `f32`, 1e-9 for `f64`.
    fn delta() -> Self;
}
impl RayReal for f32 {
    fn delta() -> Self {
        1e-5
    }
}
impl RayReal for f64 {
    fn delta() -> Self {
        1e-9
    }
}

/// A ray `eye + t * dir` restricted to the time span `[t0, t1]` (`nanovdb::math::Ray`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray<T: RayReal> {
    eye: [T; 3],
    dir: [T; 3],
    inv_dir: [T; 3],
    t0: T,
    t1: T,
}

impl<T: RayReal> Ray<T> {
    /// A ray with the given eye, direction and time span.
    pub fn new(eye: [T; 3], dir: [T; 3], t0: T, t1: T) -> Self {
        let one = T::one();
        Ray { eye, dir, inv_dir: [one / dir[0], one / dir[1], one / dir[2]], t0, t1 }
    }
    /// A ray with the default time span `[Delta, Maximum]`.
    pub fn with_dir(eye: [T; 3], dir: [T; 3]) -> Self {
        Self::new(eye, dir, T::delta(), T::max_value())
    }
    /// The eye (origin).
    pub fn eye(&self) -> [T; 3] {
        self.eye
    }
    /// The direction.
    pub fn dir(&self) -> [T; 3] {
        self.dir
    }
    /// Component-wise inverse of the direction.
    pub fn inv_dir(&self) -> [T; 3] {
        self.inv_dir
    }
    /// Start of the time span.
    pub fn t0(&self) -> T {
        self.t0
    }
    /// End of the time span.
    pub fn t1(&self) -> T {
        self.t1
    }
    /// Set the time span.
    pub fn set_times(&mut self, t0: T, t1: T) {
        self.t0 = t0;
        self.t1 = t1;
    }
    /// Position at time `t` (fused multiply-add, as in C++).
    #[inline]
    pub fn at(&self, t: T) -> [T; 3] {
        [t.mul_add(self.dir[0], self.eye[0]), t.mul_add(self.dir[1], self.eye[1]), t.mul_add(self.dir[2], self.eye[2])]
    }
    /// Position at `t0`.
    pub fn start(&self) -> [T; 3] {
        self.at(self.t0)
    }
    /// Position at `t1`.
    pub fn end(&self) -> [T; 3] {
        self.at(self.t1)
    }
    /// Intersection times of the ray with an index-space box (treating voxel `max` as
    /// extending to `max + 1`), clipped to the time span (`Ray::intersects(CoordBBox)`).
    pub fn intersects_bbox(&self, bbox: &CoordBBox) -> Option<(T, T)> {
        let (mut t0, mut t1) = (self.t0, self.t1);
        for i in 0..3 {
            let mut a = T::from_double(bbox.min[i] as f64);
            let mut b = T::from_double((bbox.max[i] as i64 + 1) as f64);
            if a >= b {
                return None;
            }
            a = (a - self.eye[i]) * self.inv_dir[i];
            b = (b - self.eye[i]) * self.inv_dir[i];
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
    /// Clip the time span to an index-space box; false if the ray misses it.
    pub fn clip(&mut self, bbox: &CoordBBox) -> bool {
        match self.intersects_bbox(bbox) {
            Some((t0, t1)) => {
                self.t0 = t0;
                self.t1 = t1;
                true
            }
            None => false,
        }
    }
    /// Intersection times with a sphere, clipped to the time span.
    pub fn intersects_sphere(&self, center: [T; 3], radius: T) -> Option<(T, T)> {
        let o = [self.eye[0] - center[0], self.eye[1] - center[1], self.eye[2] - center[2]];
        let d = self.dir;
        let a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let two = T::from_double(2.0);
        let b = two * (d[0] * o[0] + d[1] * o[1] + d[2] * o[2]);
        let c = o[0] * o[0] + o[1] * o[1] + o[2] * o[2] - radius * radius;
        let disc = b * b - T::from_double(4.0) * a * c;
        if disc < T::zero_val() {
            return None;
        }
        let q = T::from_double(-0.5) * if b < T::zero_val() { b + disc.sqrt() } else { b - disc.sqrt() };
        let (mut t0, mut t1) = (q / a, c / q);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        if t0 < self.t0 {
            t0 = self.t0;
        }
        if t1 > self.t1 {
            t1 = self.t1;
        }
        (t0 <= t1).then_some((t0, t1))
    }
}

impl Ray<f64> {
    /// Transform a world-space ray to index space (`Ray::applyInverseMap`); times are
    /// rescaled so that positions are preserved.
    pub fn world_to_index(&self, map: &Map) -> Ray<f64> {
        let eye = map.apply_inverse_map(Vec3d::new(self.eye[0], self.eye[1], self.eye[2]));
        let dir = map.apply_inverse_jacobian(Vec3d::new(self.dir[0], self.dir[1], self.dir[2]));
        let len = dir.length();
        let inv = 1.0 / len;
        Ray::new(eye.0, [dir[0] * inv, dir[1] * inv, dir[2] * inv], len * self.t0, len * self.t1)
    }
}

/// `RoundDown<Coord>(xyz)`.
#[inline]
pub fn round_down<T: RayReal>(p: [T; 3]) -> Coord {
    Coord::new(p[0].floor().to_double() as i32, p[1].floor().to_double() as i32, p[2].floor().to_double() as i32)
}

#[inline]
fn min_index<T: PartialOrd>(v: &[T; 3]) -> usize {
    if v[0] < v[1] && v[0] < v[2] {
        0
    } else if v[1] < v[2] {
        1
    } else {
        2
    }
}

#[inline]
fn mask_coord(c: Coord, dim: i32) -> Coord {
    let m = !(dim - 1);
    Coord::new(c[0] & m, c[1] & m, c[2] & m)
}

/// Hierarchical DDA: steps along a ray through cells whose size (`dim`) can change at
/// every step (`nanovdb::math::HDDA`).
#[derive(Clone, Copy, Debug)]
pub struct Hdda<T: RayReal> {
    dim: i32,
    t0: T,
    t1: T,
    voxel: Coord,
    step: [i32; 3],
    delta: [T; 3],
    next: [T; 3],
}

impl<T: RayReal> Hdda<T> {
    /// Start at the ray's `t0` with cells of size `dim`.
    pub fn new(ray: &Ray<T>, dim: i32) -> Self {
        let mut h = Hdda {
            dim,
            t0: T::zero_val(),
            t1: T::zero_val(),
            voxel: Coord::origin(),
            step: [0; 3],
            delta: [T::zero_val(); 3],
            next: [T::zero_val(); 3],
        };
        h.init(ray, ray.t0(), ray.t1(), dim);
        h
    }
    /// Restart on `[start_time, max_time]` with cells of size `dim`.
    pub fn init(&mut self, ray: &Ray<T>, start_time: T, max_time: T, dim: i32) {
        self.dim = dim;
        self.t0 = start_time;
        self.t1 = max_time;
        let pos = ray.at(self.t0);
        let (dir, inv) = (ray.dir(), ray.inv_dir());
        self.voxel = mask_coord(round_down(pos), dim);
        for axis in 0..3 {
            if dir[axis] == T::zero_val() {
                self.next[axis] = T::max_value();
                self.step[axis] = 0;
            } else if inv[axis] > T::zero_val() {
                self.step[axis] = 1;
                self.next[axis] = self.t0 + (T::from_double((self.voxel[axis] + dim) as f64) - pos[axis]) * inv[axis];
                self.delta[axis] = inv[axis];
            } else {
                self.step[axis] = -1;
                self.next[axis] = self.t0 + (T::from_double(self.voxel[axis] as f64) - pos[axis]) * inv[axis];
                self.delta[axis] = -inv[axis];
            }
        }
    }
    /// Change the cell size; returns false if it is unchanged.
    pub fn update(&mut self, ray: &Ray<T>, dim: i32) -> bool {
        if self.dim == dim {
            return false;
        }
        let vmax = mask_coord(self.voxel + Coord::splat(self.dim - 1), dim);
        let vmin = mask_coord(self.voxel, dim);
        self.dim = dim;
        let pos = ray.at(self.t0);
        let inv = ray.inv_dir();
        let v = mask_coord(round_down(pos), dim);
        self.voxel =
            Coord::new(v[0].min(vmax[0]).max(vmin[0]), v[1].min(vmax[1]).max(vmin[1]), v[2].min(vmax[2]).max(vmin[2]));
        for axis in 0..3 {
            if self.step[axis] == 0 {
                continue;
            }
            self.next[axis] = self.t0 + (T::from_double(self.voxel[axis] as f64) - pos[axis]) * inv[axis];
            if self.step[axis] > 0 {
                self.next[axis] += T::from_double(dim as f64) * inv[axis];
            }
        }
        true
    }
    /// The current cell size.
    pub fn dim(&self) -> i32 {
        self.dim
    }
    /// Step to the next cell; false once past the end of the time span.
    pub fn step(&mut self) -> bool {
        let axis = min_index(&self.next);
        if self.next[axis] <= self.t0 {
            let fix = self.t0 - T::from_double(0.999999f32 as f64) * self.next[axis] + T::from_double(1.0e-6f32 as f64);
            self.next[axis] += fix;
        }
        self.t0 = self.next[axis];
        self.next[axis] += T::from_double(self.dim as f64) * self.delta[axis];
        self.voxel[axis] += self.dim * self.step[axis];
        self.t0 <= self.t1
    }
    /// The current cell (its minimum corner).
    pub fn voxel(&self) -> Coord {
        self.voxel
    }
    /// The entry time of the current cell.
    pub fn time(&self) -> T {
        self.t0
    }
    /// The end of the time span.
    pub fn max_time(&self) -> T {
        self.t1
    }
    /// The exit time of the current cell.
    pub fn next(&self) -> T {
        self.t1.min(self.next[0].min(self.next[1].min(self.next[2])))
    }
}

/// Fixed-size DDA (`nanovdb::math::DDA<RayT, Coord, DIM>`).
#[derive(Clone, Copy, Debug)]
pub struct Dda<T: RayReal> {
    dim: i32,
    hdda: Hdda<T>,
}

impl<T: RayReal> Dda<T> {
    /// Start at the ray's `t0` with cells of size `dim`.
    pub fn new(ray: &Ray<T>, dim: i32) -> Self {
        Dda { dim, hdda: Hdda::new(ray, dim) }
    }
    /// Step to the next cell; false once past the end of the time span.
    pub fn step(&mut self) -> bool {
        let h = &mut self.hdda;
        let axis = min_index(&h.next);
        if h.next[axis] <= h.t0 {
            let fix = h.t0 - T::from_double(0.999999f32 as f64) * h.next[axis] + T::from_double(1.0e-6f32 as f64);
            h.next[axis] += fix;
        }
        h.t0 = h.next[axis];
        let d = h.delta[axis];
        h.next[axis] += d;
        h.voxel[axis] += self.dim * h.step[axis];
        h.t0 <= h.t1
    }
    /// The current cell.
    pub fn voxel(&self) -> Coord {
        self.hdda.voxel
    }
    /// The entry time of the current cell.
    pub fn time(&self) -> T {
        self.hdda.t0
    }
}

/// Find the first sign change of a level set along an index-space ray using hierarchical
/// DDA (`nanovdb::math::zeroCrossing`). Returns the voxel, its value and the hit time.
pub fn zero_crossing<B, T>(ray: &mut Ray<T>, acc: &ReadAccessor<'_, B>) -> Option<(Coord, B::Value, T)>
where
    B: BuildType,
    B::Value: SampleReal,
    T: RayReal,
{
    iso_crossing(ray, acc, B::Value::zero_val())
}

/// Find the first crossing of the iso value `iso` along an index-space ray
/// (`nanovdb::math::isoCrossing`).
pub fn iso_crossing<B, T>(ray: &mut Ray<T>, acc: &ReadAccessor<'_, B>, iso: B::Value) -> Option<(Coord, B::Value, T)>
where
    B: BuildType,
    B::Value: SampleReal,
    T: RayReal,
{
    if !ray.clip(&acc.root().bbox()) || ray.t1().to_double() > 1e20 {
        return None;
    }
    let delta = T::from_double(1.0001f32 as f64);
    let zero = B::Value::zero_val();
    let mut ijk = round_down(ray.start());
    let mut hdda = Hdda::new(ray, acc.get_dim(ijk) as i32);
    let v0 = acc.get_value(ijk) - iso;
    while hdda.step() {
        ijk = round_down(ray.at(hdda.time() + delta));
        hdda.update(ray, acc.get_dim(ijk) as i32);
        if hdda.dim() > 1 || !acc.is_active(ijk) {
            continue;
        }
        while hdda.step() && acc.is_active(hdda.voxel()) {
            let v = acc.get_value(hdda.voxel()) - iso;
            if v * v0 < zero {
                return Some((hdda.voxel(), v, hdda.time()));
            }
        }
    }
    None
}

/// Find the first active value along an index-space ray (`nanovdb::math::firstActive`).
/// Returns the voxel and the time.
pub fn first_active<B: BuildType, T: RayReal>(ray: &mut Ray<T>, acc: &ReadAccessor<'_, B>) -> Option<(Coord, T)> {
    if !ray.clip(&acc.root().bbox()) || ray.t1().to_double() > 1e20 {
        return None;
    }
    let delta = T::from_double(1.0001f32 as f64);
    let mut t = ray.t0();
    let mut ijk = round_down(ray.start());
    let mut hdda = Hdda::new(ray, acc.get_dim(ijk) as i32);
    while !acc.is_active(ijk) {
        if !hdda.step() {
            return None;
        }
        t = hdda.time() + delta;
        ijk = round_down(ray.at(t));
        hdda.update(ray, acc.get_dim(ijk) as i32);
    }
    Some((ijk, t))
}
