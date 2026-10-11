//! Ray intersection with level sets and generic volumes (port of
//! `openvdb/tools/RayIntersector.h`, with the parts of `math/Ray.h` and `math/DDA.h` it needs).
//!
//! * [`LevelSetRayIntersector`] finds the first zero crossing of a narrow-band level set along a
//!   ray, using a hierarchical DDA over the tree's nodes followed by a linear search with
//!   trilinear interpolation (OpenVDB's `LinearSearchImpl`).
//! * [`VolumeRayIntersector`] marches a ray through the active topology of any grid and
//!   returns the time spans during which the ray is inside active values.
//!
//! [`Ray`] and [`Dda`] are minimal ports of `math::Ray<double>` and `math::DDA`, kept in this
//! module until the full math versions are available.

use crate::grid::{Grid, GridClass};
use crate::math::stencils::BoxStencil;
use crate::math::{Coord, CoordBBox, Transform, Vec3d};
use crate::tree::{Internal2, Tree, ValueAccessor};
use crate::types::{RealValue, Value};

/// `math::Delta<double>`.
const DELTA_D: f64 = 1e-9;
/// `math::Delta<float>`, the default epsilon of `Ray::valid`.
const DELTA_F: f64 = 1e-5;

/// An interval of ray parameters (`Ray::TimeSpan`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeSpan {
    /// Start time.
    pub t0: f64,
    /// End time.
    pub t1: f64,
}

impl TimeSpan {
    /// A span from `t0` to `t1`.
    pub fn new(t0: f64, t1: f64) -> Self {
        TimeSpan { t0, t1 }
    }
    /// True if the span is longer than `math::Delta<double>`.
    pub fn valid(&self) -> bool {
        self.valid_eps(DELTA_D)
    }
    /// True if the span is longer than `eps`.
    pub fn valid_eps(&self, eps: f64) -> bool {
        (self.t1 - self.t0) > eps
    }
    /// The mid-point.
    pub fn mid(&self) -> f64 {
        0.5 * (self.t0 + self.t1)
    }
    /// True if `t` lies in the span.
    pub fn test(&self, t: f64) -> bool {
        t >= self.t0 && t <= self.t1
    }
}

/// A ray `eye + t * dir` with `t` restricted to `[t0, t1]` (`math::Ray<double>`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    eye: Vec3d,
    dir: Vec3d,
    inv_dir: Vec3d,
    span: TimeSpan,
}

impl Default for Ray {
    fn default() -> Self {
        Ray::new(Vec3d::zero(), Vec3d::new(1.0, 0.0, 0.0))
    }
}

impl Ray {
    /// A ray from `eye` along `dir` with the default time span `[Delta, f64::MAX]`.
    pub fn new(eye: Vec3d, dir: Vec3d) -> Self {
        Self::with_times(eye, dir, DELTA_D, f64::MAX)
    }
    /// A ray with an explicit time span.
    pub fn with_times(eye: Vec3d, dir: Vec3d, t0: f64, t1: f64) -> Self {
        Ray { eye, dir, inv_dir: dir.map(|d| 1.0 / d), span: TimeSpan::new(t0, t1) }
    }
    /// The origin.
    pub fn eye(&self) -> Vec3d {
        self.eye
    }
    /// The direction.
    pub fn dir(&self) -> Vec3d {
        self.dir
    }
    /// The component-wise inverse of the direction.
    pub fn inv_dir(&self) -> Vec3d {
        self.inv_dir
    }
    /// The start time.
    pub fn t0(&self) -> f64 {
        self.span.t0
    }
    /// The end time.
    pub fn t1(&self) -> f64 {
        self.span.t1
    }
    /// The time span.
    pub fn span(&self) -> TimeSpan {
        self.span
    }
    /// Set the origin.
    pub fn set_eye(&mut self, eye: Vec3d) {
        self.eye = eye;
    }
    /// Set the direction.
    pub fn set_dir(&mut self, dir: Vec3d) {
        self.dir = dir;
        self.inv_dir = dir.map(|d| 1.0 / d);
    }
    /// Set the time span.
    pub fn set_times(&mut self, t0: f64, t1: f64) {
        self.span = TimeSpan::new(t0, t1);
    }
    /// Set the start time.
    pub fn set_min_time(&mut self, t0: f64) {
        self.span.t0 = t0;
    }
    /// Set the end time.
    pub fn set_max_time(&mut self, t1: f64) {
        self.span.t1 = t1;
    }
    /// The point at parameter `t`.
    #[inline]
    pub fn at(&self, t: f64) -> Vec3d {
        self.eye + self.dir * t
    }
    /// The point at the start time.
    pub fn start(&self) -> Vec3d {
        self.at(self.span.t0)
    }
    /// The point at the end time.
    pub fn end(&self) -> Vec3d {
        self.at(self.span.t1)
    }
    /// True if the span is longer than `math::Delta<float>`.
    pub fn valid(&self) -> bool {
        self.span.valid_eps(DELTA_F)
    }
    /// True if `t` lies in the time span.
    pub fn test(&self, t: f64) -> bool {
        self.span.test(t)
    }
    /// Transform a world-space ray (with unit direction) to the index space of `xform`; the
    /// time span is rescaled so that it covers the same points.
    pub fn world_to_index(&self, xform: &Transform) -> Ray {
        let eye = xform.world_to_index(self.eye);
        let dir = xform.world_to_index(self.eye + self.dir) - eye;
        let len = dir.length();
        Ray::with_times(eye, dir / len, len * self.span.t0, len * self.span.t1)
    }
    /// Transform an index-space ray (with unit direction) to world space.
    pub fn index_to_world(&self, xform: &Transform) -> Ray {
        let eye = xform.index_to_world(self.eye);
        let dir = xform.index_to_world(self.eye + self.dir) - eye;
        let len = dir.length();
        Ray::with_times(eye, dir / len, len * self.span.t0, len * self.span.t1)
    }
    /// Intersect with a sphere: returns the clipped entry and exit times.
    pub fn intersects_sphere(&self, center: Vec3d, radius: f64) -> Option<(f64, f64)> {
        let origin = self.eye - center;
        let a = self.dir.length_sqr();
        let b = 2.0 * self.dir.dot(&origin);
        let c = origin.length_sqr() - radius * radius;
        let d = b * b - 4.0 * a * c;
        if d < 0.0 {
            return None;
        }
        let q = -0.5 * if b < 0.0 { b + d.sqrt() } else { b - d.sqrt() };
        let (mut t0, mut t1) = (q / a, c / q);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t0 = t0.max(self.span.t0);
        t1 = t1.min(self.span.t1);
        (t0 <= t1).then_some((t0, t1))
    }
    /// Intersect with an axis-aligned box given by its corners: returns the clipped entry and
    /// exit times.
    pub fn intersects_box(&self, min: Vec3d, max: Vec3d) -> Option<(f64, f64)> {
        let (mut t0, mut t1) = (self.span.t0, self.span.t1);
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
    /// Intersect with an index-space bounding box.
    pub fn intersects_bbox(&self, bbox: &CoordBBox) -> Option<(f64, f64)> {
        self.intersects_box(bbox.min.as_vec3d(), bbox.max.as_vec3d())
    }
    /// Clip the time span to `bbox`; returns false (leaving the ray unchanged) on a miss.
    pub fn clip_bbox(&mut self, bbox: &CoordBBox) -> bool {
        match self.intersects_bbox(bbox) {
            Some((t0, t1)) => {
                self.span = TimeSpan::new(t0, t1);
                true
            }
            None => false,
        }
    }
    /// Clip the time span to a sphere; returns false on a miss.
    pub fn clip_sphere(&mut self, center: Vec3d, radius: f64) -> bool {
        match self.intersects_sphere(center, radius) {
            Some((t0, t1)) => {
                self.span = TimeSpan::new(t0, t1);
                true
            }
            None => false,
        }
    }
    /// Intersect with the plane `{x : x·normal = distance}`.
    pub fn intersects_plane(&self, normal: Vec3d, distance: f64) -> Option<f64> {
        let cos = self.dir.dot(&normal);
        if cos.abs() <= DELTA_D {
            return None;
        }
        let t = (distance - self.eye.dot(&normal)) / cos;
        self.test(t).then_some(t)
    }
}

/// A 3D digital differential analyzer stepping through the cells of size `2^log2dim` pierced by
/// a ray (`math::DDA`).
#[derive(Clone, Copy, Debug)]
pub struct Dda {
    t0: f64,
    t1: f64,
    voxel: Coord,
    step: Coord,
    delta: Vec3d,
    next: Vec3d,
}

impl Dda {
    /// Start at the ray's start time and stop at its end time.
    pub fn new(ray: &Ray, log2dim: u32) -> Self {
        Self::with_times(ray, log2dim, ray.t0(), ray.t1())
    }
    /// Start at `start` and stop at `max_time`.
    pub fn with_times(ray: &Ray, log2dim: u32, start: f64, max_time: f64) -> Self {
        let dim = 1i32 << log2dim;
        let pos = ray.at(start);
        let (dir, inv) = (ray.dir(), ray.inv_dir());
        let voxel = Coord::floor(pos) & !(dim - 1);
        let mut d =
            Dda { t0: start, t1: max_time, voxel, step: Coord::origin(), delta: Vec3d::zero(), next: Vec3d::zero() };
        for axis in 0..3 {
            if dir[axis] == 0.0 {
                d.step[axis] = 0;
                d.next[axis] = f64::MAX;
                d.delta[axis] = f64::MAX;
            } else if inv[axis] > 0.0 {
                d.step[axis] = dim;
                d.next[axis] = start + (voxel[axis] as f64 + dim as f64 - pos[axis]) * inv[axis];
                d.delta[axis] = dim as f64 * inv[axis];
            } else {
                d.step[axis] = -dim;
                d.next[axis] = start + (voxel[axis] as f64 - pos[axis]) * inv[axis];
                d.delta[axis] = -dim as f64 * inv[axis];
            }
        }
        d
    }
    /// Step to the next cell; returns false once past the end time.
    #[inline]
    pub fn step(&mut self) -> bool {
        let a = self.next.min_index();
        self.t0 = self.next[a];
        self.next[a] += self.delta[a];
        self.voxel[a] += self.step[a];
        self.t0 <= self.t1
    }
    /// The current cell (its minimum corner).
    #[inline]
    pub fn voxel(&self) -> Coord {
        self.voxel
    }
    /// The time at which the ray entered the current cell.
    #[inline]
    pub fn time(&self) -> f64 {
        self.t0
    }
    /// The end time.
    #[inline]
    pub fn max_time(&self) -> f64 {
        self.t1
    }
    /// The time at which the ray leaves the current cell (clamped to the end time).
    #[inline]
    pub fn next(&self) -> f64 {
        self.t1.min(self.next[0]).min(self.next[1]).min(self.next[2])
    }
}

/// log2 of the extent of the nodes at `level` in the standard tree (leaf = 0).
const fn node_total(level: i32) -> u32 {
    match level {
        0 => 3,
        1 => 7,
        _ => 12,
    }
}

/// True if a node of the given level (0 = leaf, 1, 2) exists at `ijk`.
#[inline]
fn has_node<T: Value>(tree: &Tree<T>, level: i32, ijk: &Coord) -> bool {
    match level {
        0 => tree.probe_leaf(ijk).is_some(),
        1 => tree
            .root()
            .probe_child(ijk)
            .map(|n2: &Internal2<T>| n2.child(Internal2::<T>::coord_to_offset(ijk)).is_some())
            .unwrap_or(false),
        _ => tree.root().probe_child(ijk).is_some(),
    }
}

/// Errors from constructing a ray intersector.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RayIntersectorError {
    /// The grid's voxels are not cubes.
    #[error("ray intersectors only support uniform voxels")]
    NonUniformVoxels,
    /// The grid is not a level set.
    #[error("LevelSetRayIntersector only supports level sets")]
    NotALevelSet,
    /// The grid is empty.
    #[error("ray intersectors do not support empty grids")]
    EmptyGrid,
    /// The iso-value is outside the narrow band.
    #[error("the iso-value must be inside the narrow band")]
    IsoValueOutsideBand,
}

fn check_uniform<T: Value>(grid: &Grid<T>) -> Result<(), RayIntersectorError> {
    super::internal::uniform_voxel_size(grid).map(|_| ()).ok_or(RayIntersectorError::NonUniformVoxels)
}

/// Linear search for the zero crossing of the trilinearly interpolated level set along an
/// index-space ray (`tools::LinearSearchImpl`).
struct LinearSearch<'a, T: RealValue> {
    ray: Ray,
    stencil: BoxStencil<'a, T>,
    time: f64,
    v: [T; 2],
    t: [f64; 2],
    iso: T,
    min_value: T,
    max_value: T,
    iterations: usize,
}

impl<'a, T: RealValue> LinearSearch<'a, T> {
    #[inline]
    fn interp_value(&mut self, time: f64) -> T {
        let pos = self.ray.at(time);
        self.stencil.move_to_pos(pos);
        self.stencil.interpolation(pos) - self.iso
    }
    #[inline]
    fn interp_time(&self) -> f64 {
        let (v0, v1) = (self.v[0].to_f64(), self.v[1].to_f64());
        self.t[0] + (self.t[1] - self.t[0]) * v0 / (v0 - v1)
    }
    fn init(&mut self, t0: f64) {
        self.t[0] = t0;
        self.v[0] = self.interp_value(t0);
    }
    /// Test voxel `ijk`, entered by the ray and left at `time`.
    fn test(&mut self, ijk: &Coord, time: f64) -> bool {
        let (v, on) = self.stencil.accessor().probe_value(ijk);
        if on && v > self.min_value && v < self.max_value {
            self.t[1] = time;
            self.v[1] = self.interp_value(time);
            if self.v[0] * self.v[1] <= <T as Value>::zero() {
                self.time = self.interp_time();
                for _ in 0..self.iterations {
                    let vv = self.interp_value(self.time);
                    let m = if self.v[0] * vv <= <T as Value>::zero() { 1 } else { 0 };
                    self.v[m] = vv;
                    self.t[m] = self.time;
                    self.time = self.interp_time();
                }
                return true;
            }
            self.t[0] = self.t[1];
            self.v[0] = self.v[1];
        }
        false
    }

    /// Hierarchical DDA (`math::LevelSetHDDA<TreeT, level>::test`).
    fn hdda(&mut self, tree: &Tree<T>, level: i32) -> bool {
        if level < 0 {
            let mut dda = Dda::new(&self.ray, 0);
            self.init(dda.time());
            loop {
                if self.test(&dda.voxel(), dda.next()) {
                    return true;
                }
                if !dda.step() {
                    return false;
                }
            }
        }
        let mut dda = Dda::new(&self.ray, node_total(level));
        loop {
            if has_node(tree, level, &dda.voxel()) {
                self.ray.set_times(dda.time(), dda.next());
                if self.hdda(tree, level - 1) {
                    return true;
                }
            }
            if !dda.step() {
                return false;
            }
        }
    }
}

/// The result of a successful level-set ray intersection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelSetHit {
    /// Ray parameter of the hit (index-space time for `intersects_is`, world time for
    /// `intersects_ws`).
    pub time: f64,
    /// Hit position (index space for `intersects_is`, world space for `intersects_ws`).
    pub position: Vec3d,
}

/// First-hit ray intersection with a narrow-band level set (`tools::LevelSetRayIntersector`).
///
/// Methods take `&mut self` because the intersector caches tree nodes; use one intersector per
/// thread (it is cheap to clone).
#[derive(Clone)]
pub struct LevelSetRayIntersector<'a, T: RealValue> {
    grid: &'a Grid<T>,
    iso: T,
    bbox: CoordBBox,
    iterations: usize,
    stencil: BoxStencil<'a, T>,
}

impl<'a, T: RealValue> LevelSetRayIntersector<'a, T> {
    /// An intersector for the zero level set of `grid`.
    pub fn new(grid: &'a Grid<T>) -> Result<Self, RayIntersectorError> {
        Self::with_iso_value(grid, <T as Value>::zero())
    }
    /// An intersector for the `iso` level set of `grid`.
    pub fn with_iso_value(grid: &'a Grid<T>, iso: T) -> Result<Self, RayIntersectorError> {
        check_uniform(grid)?;
        if grid.grid_class() != GridClass::LevelSet {
            return Err(RayIntersectorError::NotALevelSet);
        }
        if grid.empty() {
            return Err(RayIntersectorError::EmptyGrid);
        }
        let bg = *grid.background();
        if iso <= -bg || iso >= bg {
            return Err(RayIntersectorError::IsoValueOutsideBand);
        }
        let mut bbox = CoordBBox::empty();
        grid.tree().root().eval_active_bbox(&mut bbox, false);
        Ok(LevelSetRayIntersector { grid, iso, bbox, iterations: 0, stencil: BoxStencil::new(grid) })
    }
    /// The iso-value.
    pub fn iso_value(&self) -> T {
        self.iso
    }
    /// Number of secant refinement iterations after the linear search (0 by default, like
    /// OpenVDB's `LinearSearchImpl<GridT, 0>`).
    pub fn set_iterations(&mut self, n: usize) {
        self.iterations = n;
    }

    fn search(&mut self, iray: Ray) -> Option<(Ray, f64)> {
        let mut ray = iray;
        if !ray.clip_bbox(&self.bbox) {
            return None;
        }
        let dx2 = T::from_f64(2.0 * self.grid.voxel_size()[0]);
        let mut s = LinearSearch {
            ray,
            stencil: self.stencil.clone(),
            time: 0.0,
            v: [<T as Value>::zero(); 2],
            t: [0.0; 2],
            iso: self.iso,
            min_value: self.iso - dx2,
            max_value: self.iso + dx2,
            iterations: self.iterations,
        };
        let hit = s.hdda(self.grid.tree(), 2);
        self.stencil = s.stencil;
        hit.then_some((s.ray, s.time))
    }

    /// Intersect an index-space ray; returns the index-space time and position of the hit.
    pub fn intersects_is(&mut self, iray: &Ray) -> Option<LevelSetHit> {
        let (ray, t) = self.search(*iray)?;
        Some(LevelSetHit { time: t, position: ray.at(t) })
    }

    /// Intersect a world-space ray (with unit direction); returns the world-space time and
    /// position of the hit.
    pub fn intersects_ws(&mut self, wray: &Ray) -> Option<LevelSetHit> {
        let xform = self.grid.transform();
        let (ray, t) = self.search(wray.world_to_index(xform))?;
        let pos = xform.index_to_world(ray.at(t));
        Some(LevelSetHit { time: t * world_scale(xform, ray.dir()), position: pos })
    }

    /// Intersect a world-space ray; returns the hit and the unit surface normal there.
    pub fn intersects_ws_normal(&mut self, wray: &Ray) -> Option<(LevelSetHit, Vec3d)> {
        let xform = self.grid.transform();
        let (ray, t) = self.search(wray.world_to_index(xform))?;
        let ipos = ray.at(t);
        self.stencil.move_to_pos(ipos);
        let mut n: Vec3d = self.stencil.gradient(ipos).map(|x| x.to_f64());
        n.normalize();
        Some((LevelSetHit { time: t * world_scale(xform, ray.dir()), position: xform.index_to_world(ipos) }, n))
    }

    /// True if the world-space ray hits the level set.
    pub fn hits_ws(&mut self, wray: &Ray) -> bool {
        self.search(wray.world_to_index(self.grid.transform())).is_some()
    }
}

/// Length of the world-space image of an index-space direction (`applyJacobian(dir).length()`).
fn world_scale(xform: &Transform, dir: Vec3d) -> f64 {
    (xform.index_to_world(dir) - xform.index_to_world(Vec3d::zero())).length()
}

/// Marches rays through the active topology of a grid (`tools::VolumeRayIntersector`).
pub struct VolumeRayIntersector<'a, T: Value> {
    grid: &'a Grid<T>,
    tree: Tree<bool>,
    ray: Ray,
    tmax: f64,
    bbox: CoordBBox,
}

impl<'a, T: Value> VolumeRayIntersector<'a, T> {
    /// An intersector for the active values of `grid`, optionally dilated by `dilation`
    /// voxels.
    pub fn new(grid: &'a Grid<T>, dilation: usize) -> Result<Self, RayIntersectorError> {
        check_uniform(grid)?;
        if grid.empty() {
            return Err(RayIntersectorError::EmptyGrid);
        }
        let mut tree = super::internal::topology_copy(grid.tree(), false);
        super::internal::dilate_face(&mut tree, dilation);
        let mut bbox = CoordBBox::empty();
        tree.root().eval_active_bbox(&mut bbox, false);
        bbox.max = bbox.max.offset(1);
        Ok(VolumeRayIntersector { grid, tree, ray: Ray::default(), tmax: 0.0, bbox })
    }
    /// The index-space bounding box of the (padded) active topology.
    pub fn bbox(&self) -> CoordBBox {
        self.bbox
    }
    /// Start marching along an index-space ray; returns false if it misses the bounding box.
    pub fn set_index_ray(&mut self, iray: &Ray) -> bool {
        self.ray = *iray;
        let hit = self.ray.clip_bbox(&self.bbox);
        if hit {
            self.tmax = self.ray.t1();
        }
        hit
    }
    /// Start marching along a world-space ray (with unit direction).
    pub fn set_world_ray(&mut self, wray: &Ray) -> bool {
        self.set_index_ray(&wray.world_to_index(self.grid.transform()))
    }
    /// The next index-space time span during which the ray is inside active values, or `None`
    /// once the ray has left the volume.
    pub fn march(&mut self) -> Option<TimeSpan> {
        let mut t = TimeSpan::new(-1.0, -1.0);
        if self.ray.valid() {
            let tree = &self.tree;
            vhdda_march(tree, &mut self.ray, &mut t, 2);
        }
        if t.t1 > 0.0 {
            self.ray.set_times(t.t1 + DELTA_D, self.tmax);
        }
        t.valid().then_some(t)
    }
    /// All remaining time spans along the ray (`hits`).
    pub fn hits(&mut self) -> Vec<TimeSpan> {
        let mut out = Vec::new();
        let mut t = TimeSpan::new(-1.0, -1.0);
        vhdda_hits(&self.tree, &mut self.ray, &mut out, &mut t, 2);
        if t.valid() {
            out.push(t);
        }
        out
    }
    /// Index-space position at `time`.
    pub fn index_pos(&self, time: f64) -> Vec3d {
        self.ray.at(time)
    }
    /// World-space position at index-space `time`.
    pub fn world_pos(&self, time: f64) -> Vec3d {
        self.grid.index_to_world(self.ray.at(time))
    }
    /// Convert an index-space time to a world-space time.
    pub fn world_time(&self, time: f64) -> f64 {
        time * world_scale(self.grid.transform(), self.ray.dir())
    }
    /// The grid.
    pub fn grid(&self) -> &Grid<T> {
        self.grid
    }
    /// The (dilated) topology being marched.
    pub fn tree(&self) -> &Tree<bool> {
        &self.tree
    }
}

/// Classification of a DDA cell during a volume march.
#[inline]
fn vhdda_cell(tree: &Tree<bool>, acc: &ValueAccessor<'_, Internal2<bool>>, level: i32, ijk: &Coord) -> (bool, bool) {
    // (has child node at this level, is an active value/leaf)
    if level == 0 {
        (false, tree.probe_leaf(ijk).is_some() || acc.is_value_on(ijk))
    } else {
        let child = has_node(tree, level, ijk);
        (child, !child && acc.is_value_on(ijk))
    }
}

/// `math::VolumeHDDA::march`.
fn vhdda_march(tree: &Tree<bool>, ray: &mut Ray, t: &mut TimeSpan, level: i32) -> bool {
    let acc = tree.accessor();
    let mut dda = Dda::new(ray, node_total(level));
    loop {
        let (child, active) = vhdda_cell(tree, &acc, level, &dda.voxel());
        if child {
            ray.set_times(dda.time(), dda.next());
            if vhdda_march(tree, ray, t, level - 1) {
                return true;
            }
        } else if active {
            if t.t0 < 0.0 {
                t.t0 = dda.time();
            }
        } else if t.t0 >= 0.0 {
            t.t1 = dda.time();
            if t.valid() {
                return true;
            }
            *t = TimeSpan::new(-1.0, -1.0);
        }
        if !dda.step() {
            break;
        }
    }
    if t.t0 >= 0.0 {
        t.t1 = dda.max_time();
    }
    false
}

/// `math::VolumeHDDA::hits`.
fn vhdda_hits(tree: &Tree<bool>, ray: &mut Ray, out: &mut Vec<TimeSpan>, t: &mut TimeSpan, level: i32) {
    let acc = tree.accessor();
    let mut dda = Dda::new(ray, node_total(level));
    loop {
        let (child, active) = vhdda_cell(tree, &acc, level, &dda.voxel());
        if child {
            ray.set_times(dda.time(), dda.next());
            vhdda_hits(tree, ray, out, t, level - 1);
        } else if active {
            if t.t0 < 0.0 {
                t.t0 = dda.time();
            }
        } else if t.t0 >= 0.0 {
            t.t1 = dda.time();
            if t.valid() {
                out.push(*t);
            }
            *t = TimeSpan::new(-1.0, -1.0);
        }
        if !dda.step() {
            break;
        }
    }
    if t.t0 >= 0.0 {
        t.t1 = dda.max_time();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::test_util::{assert_close, make_sphere};

    #[test]
    fn ray_basics() {
        let r = Ray::new(Vec3d::new(-5.0, 0.0, 0.0), Vec3d::new(1.0, 0.0, 0.0));
        let (t0, t1) = r.intersects_sphere(Vec3d::zero(), 1.0).unwrap();
        assert_close(t0, 4.0, 1e-12);
        assert_close(t1, 6.0, 1e-12);
        let (b0, b1) = r.intersects_box(Vec3d::splat(-2.0), Vec3d::splat(2.0)).unwrap();
        assert_close(b0, 3.0, 1e-12);
        assert_close(b1, 7.0, 1e-12);
        assert_eq!(r.intersects_plane(Vec3d::new(1.0, 0.0, 0.0), 2.0), Some(7.0));
        let xf = Transform::linear(0.5);
        let ir = r.world_to_index(&xf);
        assert_close(ir.t0(), 2e-9, 1e-15);
        assert!((ir.eye() - Vec3d::new(-10.0, 0.0, 0.0)).length() < 1e-12);
        // DDA visits consecutive voxels.
        let ray = Ray::with_times(Vec3d::new(0.5, 0.5, 0.5), Vec3d::new(1.0, 0.0, 0.0), 0.0, 3.0);
        let mut dda = Dda::new(&ray, 0);
        let mut seen = vec![dda.voxel()];
        while dda.step() {
            seen.push(dda.voxel());
        }
        assert_eq!(seen, vec![Coord::new(0, 0, 0), Coord::new(1, 0, 0), Coord::new(2, 0, 0), Coord::new(3, 0, 0)]);
    }

    /// Reference values from `scratchpad/cppref/rays.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let grid = make_sphere::<f32>(1.0, Vec3d::new(0.1, -0.05, 0.02), 0.05, 3.0);
        let mut lsi = LevelSetRayIntersector::new(&grid).unwrap();
        let rays = [
            Ray::new(Vec3d::new(-3.0, 0.0, 0.0), Vec3d::new(1.0, 0.0, 0.0)),
            Ray::new(Vec3d::new(0.3, 4.0, -0.2), Vec3d::new(0.0, -1.0, 0.0)),
            Ray::new(Vec3d::new(2.0, 2.0, 2.0), Vec3d::new(-1.0, -1.0, -1.0).unit()),
            Ray::new(Vec3d::new(0.0, 0.0, -5.0), Vec3d::new(0.6, 0.0, 0.8)),
        ];
        let expect: [Option<([f64; 3], f64, [f64; 3])>; 4] = [
            Some(([-0.898548839895, 0.0, 0.0], 2.1014511601, [-0.997188399614, 0.0747687283374, 0.00499328924262])),
            Some(([0.3, 0.904767847965, -0.2], 3.09523215203, [0.223413584449, 0.95488227232, -0.19566863901])),
            Some((
                [0.597420681903, 0.597420681903, 0.597420681903],
                2.42933864059,
                [0.49304651991, 0.651000642632, 0.577151013599],
            )),
            None,
        ];
        let iexpect =
            [Some(([-17.9709767979, 0.0, 0.0], 42.0290232021)), Some(([6.0, 18.0953569593, -4.0], 61.9046430407))];
        for (r, e) in rays.iter().zip(iexpect.iter()) {
            let h = lsi.intersects_is(&r.world_to_index(grid.transform())).unwrap();
            let (p, t) = e.unwrap();
            for i in 0..3 {
                assert_close(h.position[i], p[i], 1e-4);
            }
            assert_close(h.time, t, 1e-4);
        }
        for (r, e) in rays.iter().zip(expect.iter()) {
            let hit = lsi.intersects_ws_normal(r);
            match (hit, e) {
                (None, None) => {}
                (Some((h, n)), Some((p, t, en))) => {
                    for i in 0..3 {
                        assert_close(h.position[i], p[i], 1e-5);
                        assert_close(n[i], en[i], 1e-5);
                    }
                    assert_close(h.time, *t, 1e-5);
                }
                _ => panic!("hit mismatch for {r:?}: {hit:?}"),
            }
        }
        // Volume intersector: spans along the x axis through the narrow band shell.
        let mut vri = VolumeRayIntersector::new(&grid, 0).unwrap();
        let vexpect: [Option<(f64, f64)>; 4] =
            [Some((36.0, 92.0)), Some((56.0, 104.0)), Some((41.5692193817, 96.9948452239)), None];
        for (r, e) in rays.iter().zip(vexpect.iter()) {
            if !vri.set_world_ray(r) {
                assert!(e.is_none());
                continue;
            }
            let mut spans = Vec::new();
            while let Some(s) = vri.march() {
                spans.push((s.t0, s.t1));
            }
            let (t0, t1) = e.unwrap();
            assert_eq!(spans.len(), 1);
            assert_close(spans[0].0, t0, 1e-8);
            assert_close(spans[0].1, t1, 1e-8);
        }
        let mut vri2 = VolumeRayIntersector::new(&grid, 2).unwrap();
        vri2.set_world_ray(&rays[1]);
        let hits = vri2.hits();
        assert_eq!(hits.len(), 1);
        assert_close(hits[0].t0, 56.0, 1e-8);
        assert_close(hits[0].t1, 112.0, 1e-8);
    }

    #[test]
    fn hits_sphere_surface() {
        let grid = make_sphere::<f64>(1.0, Vec3d::zero(), 0.05, 3.0);
        let mut lsi = LevelSetRayIntersector::new(&grid).unwrap();
        for dir in [Vec3d::new(1.0, 0.0, 0.0), Vec3d::new(0.3, -0.2, 0.9).unit(), Vec3d::new(-0.5, 0.7, 0.1).unit()] {
            let eye = -dir * 4.0;
            let (h, n) = lsi.intersects_ws_normal(&Ray::new(eye, dir)).unwrap();
            assert_close(h.position.length(), 1.0, 2e-3);
            assert_close(h.time, 3.0, 2e-3);
            // Trilinear gradients are only first-order accurate (as in OpenVDB).
            assert!((n + dir).length() < 0.1, "{n:?}");
        }
        assert!(lsi.intersects_ws(&Ray::new(Vec3d::new(0.0, 2.0, -4.0), Vec3d::new(0.0, 0.0, 1.0))).is_none());
    }
}
