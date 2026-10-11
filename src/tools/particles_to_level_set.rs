//! Rasterize particles into a narrow-band level set or a boolean mask
//! (port of `openvdb/tools/ParticlesToLevelSet.h`).
//!
//! Particles are rasterized as spheres (with per-particle or constant radius) or as trails
//! of spheres along their velocity, optionally transferring a per-particle attribute (the
//! attribute of the closest particle is stored in a separate grid). Particles whose radius
//! in voxel units is below [`ParticlesToLevelSet::rmin`] (default 1.5, the Nyquist limit)
//! or above [`ParticlesToLevelSet::rmax`] are ignored and counted.
//!
//! Differences from OpenVDB: level sets are rasterized in parallel over ranges of particle
//! buckets, but the per-range results are merged so that the output is exactly that of
//! processing the buckets sequentially in partitioner order (OpenVDB with
//! `setGrainSize(0)`), which makes it fully deterministic. Masks are rasterized sequentially. When attributes are transferred into a non-empty level set, the final
//! CSG union is a voxel-wise minimum.

use super::point_partitioner::{PointArray, PointPartitioner};
use super::prune::level_set_prune;
use super::signed_flood_fill::signed_flood_fill_with_values;
use super::tree_ops::steal_leaves;
use crate::grid::Grid;
use crate::math::{Coord, CoordBBox, Vec3d};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::{RealValue, Value};
use rayon::prelude::*;

/// A list of particles (OpenVDB's `ParticleList` concept): positions (via [`PointArray`]),
/// radii and optionally velocities, all in world units.
pub trait ParticleList: PointArray {
    /// Position and radius of particle `n`.
    fn get_pos_rad(&self, n: usize) -> (Vec3d, f64);
    /// Position, radius and velocity of particle `n` (needed for trails).
    fn get_pos_rad_vel(&self, n: usize) -> (Vec3d, f64, Vec3d) {
        let (p, r) = self.get_pos_rad(n);
        (p, r, Vec3d::zero())
    }
}

/// Per-particle attributes for attribute transfer (OpenVDB's `getAtt`).
pub trait ParticleAttributes<A> {
    fn get_att(&self, n: usize) -> A;
}

/// Output value types: `f32`/`f64` level sets, or `bool` masks.
pub trait P2lsValue: Value + PartialOrd {
    /// True for boolean mask output.
    const IS_MASK: bool;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    /// The "on" value of masks (unused for level sets).
    fn on() -> Self;
    fn negate(self) -> Self;
}
impl P2lsValue for f32 {
    const IS_MASK: bool = false;
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn on() -> Self {
        1.0
    }
    fn negate(self) -> Self {
        -self
    }
}
impl P2lsValue for f64 {
    const IS_MASK: bool = false;
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn on() -> Self {
        1.0
    }
    fn negate(self) -> Self {
        -self
    }
}
impl P2lsValue for bool {
    const IS_MASK: bool = true;
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
    fn to_f64(self) -> f64 {
        self as i32 as f64
    }
    fn on() -> Self {
        true
    }
    fn negate(self) -> Self {
        self
    }
}

/// A distance (or mask) value combined with a particle attribute
/// (`p2ls_internal::BlindData`). Comparisons only consider the visible value.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlindData<T, A> {
    pub visible: T,
    pub blind: A,
}
impl<T: PartialEq, A> PartialEq for BlindData<T, A> {
    fn eq(&self, o: &Self) -> bool {
        self.visible == o.visible
    }
}
impl<T: Value, A: Value> Value for BlindData<T, A> {
    fn approx_eq(&self, other: &Self, tolerance: &Self) -> bool {
        self.visible.approx_eq(&other.visible, &tolerance.visible)
    }
}

/// The attribute type used when no attribute is transferred.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoAttribute;
impl Value for NoAttribute {
    fn approx_eq(&self, _: &Self, _: &Self) -> bool {
        true
    }
}

/// Values that can be rasterized: plain output values, or [`BlindData`] with attributes.
trait RasterValue<T: P2lsValue, A>: Value {
    fn visible(&self) -> T;
    fn make(v: T, a: A) -> Self;
    fn negated(self) -> Self;
}
impl<T: P2lsValue> RasterValue<T, ()> for T {
    fn visible(&self) -> T {
        *self
    }
    fn make(v: T, _a: ()) -> Self {
        v
    }
    fn negated(self) -> Self {
        self.negate()
    }
}
impl<T: P2lsValue, A: Value> RasterValue<T, A> for BlindData<T, A> {
    fn visible(&self) -> T {
        self.visible
    }
    fn make(v: T, a: A) -> Self {
        BlindData { visible: v, blind: a }
    }
    fn negated(self) -> Self {
        BlindData { visible: self.visible.negate(), blind: self.blind }
    }
}

struct RasterParams {
    dx: f64,
    half_width: f64,
    rmin: f64,
    rmax: f64,
}

fn make_sphere_sdf<T: P2lsValue, A: Copy, V: RasterValue<T, A>>(
    tree: &mut Tree<V>,
    p: Vec3d,
    r: f64,
    att: A,
    prm: &RasterParams,
) {
    let dx = prm.dx;
    let w = prm.half_width;
    let max = r + w;
    let max2 = max * max;
    let min2 = {
        let m = (r - w).max(0.0);
        m * m
    };
    let lo = Coord::new((p[0] - max).floor() as i32, (p[1] - max).floor() as i32, (p[2] - max).floor() as i32);
    let hi = Coord::new((p[0] + max).ceil() as i32, (p[1] + max).ceil() as i32, (p[2] + max).ceil() as i32);
    let inside = tree.background().negated();
    let zero = T::from_f64(0.0);
    let mut acc = tree.accessor_mut();
    for x in lo[0]..=hi[0] {
        let x2 = (x as f64 - p[0]) * (x as f64 - p[0]);
        for y in lo[1]..=hi[1] {
            let x2y2 = x2 + (y as f64 - p[1]) * (y as f64 - p[1]);
            if x2y2 >= max2 {
                continue;
            }
            for z in lo[2]..=hi[2] {
                let x2y2z2 = x2y2 + (z as f64 - p[2]) * (z as f64 - p[2]);
                if x2y2z2 >= max2 {
                    continue;
                }
                let c = Coord::new(x, y, z);
                if let Some(leaf) = acc.probe_leaf(&c) {
                    // Fast path: work on the (cached) leaf directly.
                    let n = LeafNode::<V>::coord_to_offset(&c);
                    let (v, on) = (*leaf.get_value_at(n), leaf.is_value_on_at(n));
                    if !on && v.visible() < zero {
                        continue;
                    }
                    if x2y2z2 <= min2 {
                        leaf.set_value_off_at(n, inside);
                    } else {
                        let d = V::make(T::from_f64(dx * (x2y2z2.sqrt() - r)), att);
                        if d.visible() < v.visible() {
                            leaf.set_value_on_at(n, d);
                        }
                    }
                    continue;
                }
                let (v, on) = acc.probe_value(&c);
                if !on && v.visible() < zero {
                    continue;
                }
                if x2y2z2 <= min2 {
                    acc.set_value_off(&c, inside);
                    continue;
                }
                let d = V::make(T::from_f64(dx * (x2y2z2.sqrt() - r)), att);
                if d.visible() < v.visible() {
                    acc.set_value(&c, d);
                }
            }
        }
    }
}

/// Merge `b`, rasterized from particles that come after those rasterized into `a`, into
/// `a`. The result is the same as rasterizing all of them into `a` in order: a voxel inside
/// a particle in either tree stays inside, otherwise the smaller distance wins and ties go
/// to `a`. (`b` must have been rasterized into an empty tree.)
fn merge_sdf<T: P2lsValue, A: Copy, V: RasterValue<T, A>>(a: &mut Tree<V>, mut b: Tree<V>) {
    let zero = T::from_f64(0.0);
    let bg = *a.background();
    let is_inside = |v: &V, on: bool| !on && v.visible() < zero;
    let mut acc = a.accessor_mut();
    for leaf in steal_leaves(&mut b, bg) {
        let origin = leaf.origin();
        if acc.probe_leaf(&origin).is_none() {
            let (v, on) = acc.probe_value(&origin);
            if !on && v == bg {
                // Untouched region of `a`: take the leaf as is.
                acc.add_leaf(leaf);
                continue;
            }
            // A tile: only densify it if some voxel changes, as the sequential order would.
            let changes = !is_inside(&v, on)
                && (0..LeafNode::<V>::SIZE).any(|n| {
                    let (bv, bon) = (*leaf.get_value_at(n), leaf.is_value_on_at(n));
                    is_inside(&bv, bon) || (bon && bv.visible() < v.visible())
                });
            if !changes {
                continue;
            }
        }
        let dst = acc.touch_leaf(&origin);
        for n in 0..LeafNode::<V>::SIZE {
            let (bv, bon) = (*leaf.get_value_at(n), leaf.is_value_on_at(n));
            if !bon && !is_inside(&bv, bon) {
                continue; // not touched in `b`
            }
            let (av, aon) = (*dst.get_value_at(n), dst.is_value_on_at(n));
            if is_inside(&av, aon) {
                continue;
            }
            if is_inside(&bv, bon) {
                dst.set_value_off_at(n, bv);
            } else if bv.visible() < av.visible() {
                dst.set_value_on_at(n, bv);
            }
        }
    }
}

fn make_sphere_mask<T: P2lsValue, A: Copy, V: RasterValue<T, A>>(tree: &mut Tree<V>, p: Vec3d, r: f64, att: A) {
    let r_sq = r * r;
    let in_w = r / 6f64.sqrt();
    let fl = |v: f64| v.floor() as i32;
    let ce = |v: f64| v.ceil() as i32;
    let out_lo = Coord::new(fl(p[0] - r), fl(p[1] - r), fl(p[2] - r));
    let out_hi = Coord::new(ce(p[0] + r), ce(p[1] + r), ce(p[2] + r));
    let in_lo = Coord::new(ce(p[0] - in_w), ce(p[1] - in_w), ce(p[2] - in_w));
    let in_hi = Coord::new(fl(p[0] + in_w), fl(p[1] + in_w), fl(p[2] + in_w));
    let padding = [
        CoordBBox::from_coords(out_lo[0], out_lo[1], out_lo[2], in_lo[0] - 1, out_hi[1], out_hi[2]),
        CoordBBox::from_coords(in_hi[0] + 1, out_lo[1], out_lo[2], out_hi[0], out_hi[1], out_hi[2]),
        CoordBBox::from_coords(out_lo[0], out_lo[1], out_lo[2], out_hi[0], in_lo[1] - 1, out_hi[2]),
        CoordBBox::from_coords(out_lo[0], in_hi[1] + 1, out_lo[2], out_hi[0], out_hi[1], out_hi[2]),
        CoordBBox::from_coords(out_lo[0], out_lo[1], out_lo[2], out_hi[0], out_hi[1], in_lo[2] - 1),
        CoordBBox::from_coords(out_lo[0], out_lo[1], in_hi[2] + 1, out_hi[0], out_hi[1], out_hi[2]),
    ];
    let on_value = V::make(T::on(), att);
    let inner = CoordBBox::new(in_lo, in_hi);
    if !inner.is_empty() {
        tree.sparse_fill(&inner, on_value, true);
    }
    let mut acc = tree.accessor_mut();
    for b in padding.iter() {
        for x in b.min[0]..=b.max[0] {
            let x2 = (x as f64 - p[0]) * (x as f64 - p[0]);
            for y in b.min[1]..=b.max[1] {
                let x2y2 = x2 + (y as f64 - p[1]) * (y as f64 - p[1]);
                for z in b.min[2]..=b.max[2] {
                    let x2y2z2 = x2y2 + (z as f64 - p[2]) * (z as f64 - p[2]);
                    if x2y2z2 < r_sq {
                        acc.set_value(&Coord::new(x, y, z), on_value);
                    }
                }
            }
        }
    }
}

#[inline]
fn make_sphere<T: P2lsValue, A: Copy, V: RasterValue<T, A>>(
    tree: &mut Tree<V>,
    p: Vec3d,
    r: f64,
    att: A,
    prm: &RasterParams,
) {
    if T::IS_MASK {
        make_sphere_mask::<T, A, V>(tree, p, r, att)
    } else {
        make_sphere_sdf::<T, A, V>(tree, p, r, att, prm)
    }
}

enum Mode {
    Spheres,
    FixedSpheres(f64),
    Trails(f64),
}

/// Returns (min count, max count) of ignored particles.
///
/// Level sets are rasterized in parallel: each task rasterizes a contiguous range of
/// partitioner buckets into its own tree, and the trees are merged in bucket order with
/// [`merge_sdf`], which gives the same result as the sequential order. Masks are rasterized
/// sequentially.
fn raster<T, A, V, P, F>(
    tree: &mut Tree<V>,
    xform: &crate::math::Transform,
    particles: &P,
    att: F,
    prm: &RasterParams,
    mode: Mode,
) -> (usize, usize)
where
    T: P2lsValue,
    A: Copy,
    V: RasterValue<T, A>,
    P: ParticleList + ?Sized,
    F: Fn(usize) -> A + Sync,
{
    let partitioner = PointPartitioner::<3>::create(particles, xform, false, false, true);
    let buckets = partitioner.size();
    let threads = rayon::current_num_threads();
    let tasks = if T::IS_MASK || threads == 1 { 1 } else { (threads * 4).min(buckets / 16).max(1) };
    if tasks == 1 {
        return raster_buckets::<T, A, V, P, F>(tree, xform, particles, &att, prm, &mode, &partitioner, 0..buckets);
    }
    let bg = *tree.background();
    let chunk = buckets.div_ceil(tasks);
    let (merged, min_count, max_count) = (0..tasks)
        .into_par_iter()
        .map(|t| {
            let mut sub = Tree::<V>::new(bg);
            let range = (t * chunk).min(buckets)..((t + 1) * chunk).min(buckets);
            let (mi, ma) =
                raster_buckets::<T, A, V, P, F>(&mut sub, xform, particles, &att, prm, &mode, &partitioner, range);
            (sub, mi, ma)
        })
        .reduce_with(|(mut a, ami, ama), (b, bmi, bma)| {
            merge_sdf::<T, A, V>(&mut a, b);
            (a, ami + bmi, ama + bma)
        })
        .unwrap();
    merge_sdf::<T, A, V>(tree, merged);
    (min_count, max_count)
}

#[allow(clippy::too_many_arguments)]
fn raster_buckets<T, A, V, P, F>(
    tree: &mut Tree<V>,
    xform: &crate::math::Transform,
    particles: &P,
    att: &F,
    prm: &RasterParams,
    mode: &Mode,
    partitioner: &PointPartitioner<3>,
    buckets: std::ops::Range<usize>,
) -> (usize, usize)
where
    T: P2lsValue,
    A: Copy,
    V: RasterValue<T, A>,
    P: ParticleList + ?Sized,
    F: Fn(usize) -> A,
{
    let inv_dx = 1.0 / prm.dx;
    let (mut min_count, mut max_count) = (0usize, 0usize);
    let ignore = |r: f64, min_count: &mut usize, max_count: &mut usize| {
        if r < prm.rmin {
            *min_count += 1;
            true
        } else if r > prm.rmax {
            *max_count += 1;
            true
        } else {
            false
        }
    };
    let origin = xform.world_to_index(Vec3d::zero());
    for n in buckets {
        for &id in partitioner.indices(n) {
            let id = id as usize;
            match *mode {
                Mode::Spheres => {
                    let (pos, rad) = particles.get_pos_rad(id);
                    let r = inv_dx * rad;
                    if ignore(r, &mut min_count, &mut max_count) {
                        continue;
                    }
                    let p = xform.world_to_index(pos);
                    make_sphere::<T, A, V>(tree, p, r, att(id), prm);
                }
                Mode::FixedSpheres(r) => {
                    let a = att(id);
                    let p = xform.world_to_index(particles.get_pos(id));
                    make_sphere::<T, A, V>(tree, p, r, a, prm);
                }
                Mode::Trails(delta) => {
                    let (pos, rad, vel) = particles.get_pos_rad_vel(id);
                    let r0 = inv_dx * rad;
                    if ignore(r0, &mut min_count, &mut max_count) {
                        continue;
                    }
                    let a = att(id);
                    let p0 = xform.world_to_index(pos);
                    let v = xform.world_to_index(vel) - origin;
                    let speed = v.length();
                    let inv_speed = 1.0 / speed;
                    let nrml = -v * inv_speed;
                    let mut p = p0;
                    let mut r = r0;
                    let mut d = 0.0;
                    while d <= speed {
                        make_sphere::<T, A, V>(tree, p, r, a, prm);
                        p += nrml * (0.5 * delta * r);
                        d = (p - p0).length();
                        r = r0 - (r0 - prm.rmin) * d * inv_speed;
                    }
                }
            }
        }
    }
    (min_count, max_count)
}

/// Rasterizes particles into a level set (`f32`/`f64` grid) or a mask (`bool` grid),
/// optionally transferring an attribute of type `A` (port of `ParticlesToLevelSet`).
pub struct ParticlesToLevelSet<'a, T: P2lsValue, A: Value = NoAttribute> {
    grid: &'a mut Grid<T>,
    blind: Option<Tree<BlindData<T, A>>>,
    att_grid: Option<Grid<A>>,
    dx: f64,
    half_width: f64,
    rmin: f64,
    rmax: f64,
    min_count: usize,
    max_count: usize,
}

impl<'a, T: P2lsValue> ParticlesToLevelSet<'a, T, NoAttribute> {
    /// Rasterize directly into `grid` (no attribute transfer). For level sets, the grid's
    /// background defines the narrow band half width.
    pub fn new(grid: &'a mut Grid<T>) -> Self {
        Self::make(grid, false)
    }
}

impl<'a, T: P2lsValue, A: Value> ParticlesToLevelSet<'a, T, A> {
    /// Rasterize with transfer of a particle attribute of type `A` (see
    /// [`attribute_grid`](Self::attribute_grid) after [`finalize`](Self::finalize)).
    pub fn with_attributes(grid: &'a mut Grid<T>) -> Self {
        Self::make(grid, true)
    }

    fn make(grid: &'a mut Grid<T>, attributes: bool) -> Self {
        let dx = grid.voxel_size()[0];
        let half_width = grid.background().to_f64() / dx;
        let blind = if attributes {
            Some(Tree::new(BlindData { visible: *grid.background(), blind: A::zero() }))
        } else {
            None
        };
        ParticlesToLevelSet {
            grid,
            blind,
            att_grid: None,
            dx,
            half_width,
            rmin: 1.5,
            rmax: 100.0,
            min_count: 0,
            max_count: 0,
        }
    }

    pub fn voxel_size(&self) -> f64 {
        self.dx
    }
    pub fn half_width(&self) -> f64 {
        self.half_width
    }
    /// Smallest radius (in voxel units) of particles that are rasterized.
    pub fn rmin(&self) -> f64 {
        self.rmin
    }
    pub fn set_rmin(&mut self, rmin: f64) {
        self.rmin = rmin.max(0.0);
    }
    /// Largest radius (in voxel units) of particles that are rasterized.
    pub fn rmax(&self) -> f64 {
        self.rmax
    }
    pub fn set_rmax(&mut self, rmax: f64) {
        self.rmax = self.rmin.max(rmax);
    }
    /// True if any particles were ignored because of their radius.
    pub fn ignored_particles(&self) -> bool {
        self.min_count > 0 || self.max_count > 0
    }
    /// Number of particles ignored for being too small.
    pub fn min_count(&self) -> usize {
        self.min_count
    }
    /// Number of particles ignored for being too large.
    pub fn max_count(&self) -> usize {
        self.max_count
    }

    fn params(&self) -> RasterParams {
        RasterParams { dx: self.dx, half_width: self.half_width, rmin: self.rmin, rmax: self.rmax }
    }

    fn run<P, F>(&mut self, particles: &P, att: F, mode: Mode)
    where
        P: ParticleList + ?Sized,
        F: Fn(usize) -> A + Sync,
    {
        let prm = self.params();
        let xform = self.grid.transform().clone();
        let (mi, ma) = match self.blind.as_mut() {
            Some(b) => raster::<T, A, BlindData<T, A>, P, _>(b, &xform, particles, att, &prm, mode),
            None => raster::<T, (), T, P, _>(self.grid.tree_mut(), &xform, particles, |_| (), &prm, mode),
        };
        self.min_count = mi;
        self.max_count = ma;
    }

    /// Rasterize every particle as a sphere with its own radius.
    pub fn rasterize_spheres<P: ParticleList + ?Sized>(&mut self, particles: &P) {
        self.run(particles, |_| A::zero(), Mode::Spheres)
    }
    /// Rasterize every particle as a sphere with the given world-space radius.
    pub fn rasterize_spheres_with_radius<P: ParticleList + ?Sized>(&mut self, particles: &P, radius: f64) {
        let r = radius / self.dx;
        if r < self.rmin || r > self.rmax {
            self.min_count = if r < self.rmin { particles.size() } else { 0 };
            self.max_count = if r > self.rmax { particles.size() } else { 0 };
            return;
        }
        self.run(particles, |_| A::zero(), Mode::FixedSpheres(r))
    }
    /// Rasterize every particle as a trail of spheres along its velocity; `delta` controls
    /// the distance between instances (in units of the local radius).
    pub fn rasterize_trails<P: ParticleList + ?Sized>(&mut self, particles: &P, delta: f64) {
        self.run(particles, |_| A::zero(), Mode::Trails(delta))
    }

    /// [`rasterize_spheres`](Self::rasterize_spheres) with attribute transfer.
    pub fn rasterize_spheres_att<P: ParticleList + ParticleAttributes<A> + ?Sized>(&mut self, particles: &P) {
        self.run(particles, |n| particles.get_att(n), Mode::Spheres)
    }
    /// [`rasterize_spheres_with_radius`](Self::rasterize_spheres_with_radius) with attribute
    /// transfer.
    pub fn rasterize_spheres_with_radius_att<P: ParticleList + ParticleAttributes<A> + ?Sized>(
        &mut self,
        particles: &P,
        radius: f64,
    ) {
        let r = radius / self.dx;
        if r < self.rmin || r > self.rmax {
            self.min_count = if r < self.rmin { particles.size() } else { 0 };
            self.max_count = if r > self.rmax { particles.size() } else { 0 };
            return;
        }
        self.run(particles, |n| particles.get_att(n), Mode::FixedSpheres(r))
    }
    /// [`rasterize_trails`](Self::rasterize_trails) with attribute transfer.
    pub fn rasterize_trails_att<P: ParticleList + ParticleAttributes<A> + ?Sized>(
        &mut self,
        particles: &P,
        delta: f64,
    ) {
        self.run(particles, |n| particles.get_att(n), Mode::Trails(delta))
    }

    /// The attribute grid produced by [`finalize`](Self::finalize) (attribute transfer only).
    pub fn attribute_grid(&self) -> Option<&Grid<A>> {
        self.att_grid.as_ref()
    }
    /// Take the attribute grid.
    pub fn take_attribute_grid(&mut self) -> Option<Grid<A>> {
        self.att_grid.take()
    }

    /// Finish rasterization: optionally prune, and with attribute transfer split the internal
    /// data into the output grid and the attribute grid.
    pub fn finalize(&mut self, prune: bool) {
        let mut blind = match self.blind.take() {
            None => {
                if prune {
                    if T::IS_MASK {
                        self.grid.tree_mut().prune(T::zero());
                    } else {
                        let bg = *self.grid.background();
                        prune_level_set_generic(self.grid.tree_mut(), bg);
                    }
                }
                return;
            }
            Some(b) => b,
        };
        if prune {
            blind.prune(BlindData { visible: T::zero(), blind: A::zero() });
        }
        let bg = *blind.background();
        let mut att_tree = Tree::<A>::new(bg.blind);
        let mut sdf_tree = if T::IS_MASK { Tree::<T>::new(T::zero()) } else { Tree::<T>::new(bg.visible) };
        // Tiles (only present after pruning).
        let mut tiles = Vec::new();
        blind.visit_tiles(&mut |b, v, on, level| tiles.push((b, *v, on, level)));
        for (b, v, on, level) in tiles {
            let sv = if T::IS_MASK {
                if on {
                    T::on()
                } else {
                    T::zero()
                }
            } else {
                bg.visible
            };
            sdf_tree.add_tile(level, &b.min, sv, on);
            let av = if T::IS_MASK && on { v.blind } else { bg.blind };
            att_tree.add_tile(level, &b.min, av, on);
        }
        for leaf in steal_leaves(&mut blind, bg) {
            let origin = leaf.origin();
            let mut sl = crate::tree::LeafNode::<T>::new(origin, sdf_tree.background().clone(), false);
            let mut al = crate::tree::LeafNode::<A>::new(origin, bg.blind, false);
            *sl.value_mask_mut() = leaf.value_mask().clone();
            *al.value_mask_mut() = leaf.value_mask().clone();
            if T::IS_MASK {
                for n in leaf.value_mask().iter_on() {
                    sl.buffer_mut()[n] = T::on();
                    al.buffer_mut()[n] = leaf.buffer()[n].blind;
                }
            } else if leaf.value_mask().is_all_off() {
                for n in 0..512 {
                    sl.buffer_mut()[n] = leaf.buffer()[n].visible;
                    al.buffer_mut()[n] = leaf.buffer()[n].blind;
                }
            } else {
                for n in leaf.value_mask().iter_on() {
                    sl.buffer_mut()[n] = leaf.buffer()[n].visible;
                    al.buffer_mut()[n] = leaf.buffer()[n].blind;
                }
            }
            sdf_tree.add_leaf(Box::new(sl));
            att_tree.add_leaf(Box::new(al));
        }
        if !T::IS_MASK {
            signed_flood_fill_generic(&mut sdf_tree);
        }
        self.att_grid = Some(Grid::from_tree(att_tree).with_transform(self.grid.transform().clone()));
        if self.grid.tree().is_empty() {
            self.grid.set_tree(sdf_tree);
        } else if T::IS_MASK {
            self.grid.tree_mut().topology_union(&sdf_tree);
            self.grid.tree_mut().prune(T::zero());
        } else {
            let dst = self.grid.tree_mut();
            for leaf in sdf_tree.leaves() {
                let mine = dst.touch_leaf(&leaf.origin());
                for n in 0..512 {
                    let v = leaf.buffer()[n];
                    if leaf.is_value_on_at(n) && (!mine.is_value_on_at(n) || v < mine.buffer()[n]) {
                        mine.set_value_on_at(n, v);
                    }
                }
            }
            let bg = *dst.background();
            prune_level_set_generic(dst, bg);
        }
    }
}

fn prune_level_set_generic<T: P2lsValue>(tree: &mut Tree<T>, bg: T) {
    // Only reached for floating-point outputs.
    if let Some(t) = (tree as &mut dyn std::any::Any).downcast_mut::<Tree<f32>>() {
        let b = bg.to_f64() as f32;
        level_set_prune(t, b, -b, true);
    } else if let Some(t) = (tree as &mut dyn std::any::Any).downcast_mut::<Tree<f64>>() {
        let b = bg.to_f64();
        level_set_prune(t, b, -b, true);
    }
}

fn signed_flood_fill_generic<T: P2lsValue>(tree: &mut Tree<T>) {
    if let Some(t) = (tree as &mut dyn std::any::Any).downcast_mut::<Tree<f32>>() {
        let b = t.background().abs();
        let _ = signed_flood_fill_with_values(t, b, -b, true);
    } else if let Some(t) = (tree as &mut dyn std::any::Any).downcast_mut::<Tree<f64>>() {
        let b = t.background().abs();
        let _ = signed_flood_fill_with_values(t, b, -b, true);
    }
}

fn prune_ls<T: RealValue + P2lsValue>(grid: &mut Grid<T>) {
    let bg = *grid.background();
    level_set_prune(grid.tree_mut(), bg, -bg, true);
}

/// Populate a level set grid with spheres given by the particle positions and radii
/// (`tools::particlesToSdf`). The grid background defines the narrow band.
pub fn particles_to_sdf<T: RealValue + P2lsValue, P: ParticleList + ?Sized>(particles: &P, grid: &mut Grid<T>) {
    ParticlesToLevelSet::new(grid).rasterize_spheres(particles);
    prune_ls(grid);
}

/// Populate a level set grid with spheres of a constant world-space radius
/// (`tools::particlesToSdf` with radius).
pub fn particles_to_sdf_with_radius<T: RealValue + P2lsValue, P: ParticleList + ?Sized>(
    particles: &P,
    grid: &mut Grid<T>,
    radius: f64,
) {
    ParticlesToLevelSet::new(grid).rasterize_spheres_with_radius(particles, radius);
    prune_ls(grid);
}

/// Populate a level set grid with trails of spheres along the particle velocities
/// (`tools::particleTrailsToSdf`).
pub fn particle_trails_to_sdf<T: RealValue + P2lsValue, P: ParticleList + ?Sized>(
    particles: &P,
    grid: &mut Grid<T>,
    delta: f64,
) {
    ParticlesToLevelSet::new(grid).rasterize_trails(particles, delta);
    prune_ls(grid);
}

/// Activate the voxels of a boolean grid covered by particle spheres (`tools::particlesToMask`).
pub fn particles_to_mask<P: ParticleList + ?Sized>(particles: &P, grid: &mut Grid<bool>) {
    ParticlesToLevelSet::new(grid).rasterize_spheres(particles);
    grid.tree_mut().prune(false);
}

/// Mask version with a constant world-space radius.
pub fn particles_to_mask_with_radius<P: ParticleList + ?Sized>(particles: &P, grid: &mut Grid<bool>, radius: f64) {
    ParticlesToLevelSet::new(grid).rasterize_spheres_with_radius(particles, radius);
    grid.tree_mut().prune(false);
}

/// Mask version of [`particle_trails_to_sdf`] (`tools::particleTrailsToMask`).
pub fn particle_trails_to_mask<P: ParticleList + ?Sized>(particles: &P, grid: &mut Grid<bool>, delta: f64) {
    ParticlesToLevelSet::new(grid).rasterize_trails(particles, delta);
    grid.tree_mut().prune(false);
}

/// A simple particle list: positions, radii and velocities (world space).
#[derive(Clone, Debug, Default)]
pub struct SimpleParticleList {
    pub positions: Vec<Vec3d>,
    pub radii: Vec<f64>,
    pub velocities: Vec<Vec3d>,
}

impl PointArray for SimpleParticleList {
    fn size(&self) -> usize {
        self.positions.len()
    }
    fn get_pos(&self, n: usize) -> Vec3d {
        self.positions[n]
    }
}
impl ParticleList for SimpleParticleList {
    fn get_pos_rad(&self, n: usize) -> (Vec3d, f64) {
        (self.positions[n], self.radii[n])
    }
    fn get_pos_rad_vel(&self, n: usize) -> (Vec3d, f64, Vec3d) {
        (self.positions[n], self.radii[n], self.velocities.get(n).copied().unwrap_or_else(Vec3d::zero))
    }
}
impl ParticleAttributes<i32> for SimpleParticleList {
    fn get_att(&self, n: usize) -> i32 {
        n as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridClass;
    use crate::math::Transform;

    #[test]
    fn spheres_to_sdf() {
        let mut grid = Grid::<f32>::new(0.3).with_transform(Transform::linear(0.1));
        grid.set_grid_class(GridClass::LevelSet);
        let pl = SimpleParticleList {
            positions: vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(1.0, 0.0, 0.0), Vec3d::new(5.0, 5.0, 5.0)],
            radii: vec![0.5, 0.4, 0.01],
            velocities: vec![],
        };
        let mut p2ls = ParticlesToLevelSet::new(&mut grid);
        p2ls.rasterize_spheres(&pl);
        assert_eq!(p2ls.min_count(), 1);
        p2ls.finalize(true);
        assert!(grid.active_voxel_count() > 0);
        // Union of the two spheres: at x = 0.7 the second sphere (center 1.0, radius 0.4) wins.
        let v = grid.get_value(&Coord::new(7, 0, 0));
        assert!((v + 0.1).abs() < 1e-5, "{v}");
        let v = grid.get_value(&Coord::new(0, 6, 0));
        assert!((v - 0.1).abs() < 1e-5, "{v}");
        assert!(grid.get_value(&Coord::new(0, 0, 0)) < 0.0);
    }

    #[test]
    fn attribute_transfer_and_mask() {
        let pl = SimpleParticleList {
            positions: vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(1.0, 0.0, 0.0)],
            radii: vec![0.3, 0.3],
            velocities: vec![Vec3d::new(0.0, 1.0, 0.0), Vec3d::new(0.0, 0.0, 1.0)],
        };
        let mut grid = Grid::<f32>::new(0.3).with_transform(Transform::linear(0.1));
        let mut p2ls = ParticlesToLevelSet::<f32, i32>::with_attributes(&mut grid);
        p2ls.rasterize_spheres_att(&pl);
        p2ls.finalize(false);
        let att = p2ls.take_attribute_grid().unwrap();
        assert_eq!(att.get_value(&Coord::new(10, 2, 0)), 1);
        assert_eq!(att.get_value(&Coord::new(0, 2, 0)), 0);
        assert!(grid.get_value(&Coord::new(0, 0, 0)) < 0.0);

        let mut mask = Grid::<bool>::new(false).with_transform(Transform::linear(0.1));
        particle_trails_to_mask(&pl, &mut mask, 1.0);
        // Trails extend opposite to the velocity.
        assert!(mask.get_value(&Coord::new(0, -5, 0)));
        assert!(mask.get_value(&Coord::new(10, 0, 0)));
        assert!(!mask.get_value(&Coord::new(5, 0, 0)));
    }

    /// Rasterize with the given number of threads, into a grid with existing content, with
    /// and without attribute transfer. Returns (plain level set, level set, attributes).
    fn rasterize_with_threads(pl: &SimpleParticleList, threads: usize, trails: bool) -> [Tree<f32>; 2] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let new_grid = || {
            let mut grid = Grid::<f32>::new(0.3).with_transform(Transform::linear(0.1));
            // Existing content, including "inside" values above -background and an inactive
            // positive value: the merge must treat it like the sequential order does.
            grid.tree_mut().fill(&CoordBBox::from_coords(-25, -25, -25, -5, -5, -5), -0.05, false);
            grid.tree_mut().set_value(&Coord::new(5, 5, 5), 0.05);
            grid.tree_mut().set_value_off(&Coord::new(6, 5, 5), 0.02);
            grid
        };
        pool.install(|| {
            let mut plain = new_grid();
            let mut p2ls = ParticlesToLevelSet::new(&mut plain);
            if trails {
                p2ls.rasterize_trails(pl, 1.0);
            } else {
                p2ls.rasterize_spheres(pl);
            }
            p2ls.finalize(false);
            [plain.tree().clone(), Tree::new(0.0)]
        })
    }

    fn rasterize_att_with_threads(pl: &SimpleParticleList, threads: usize, trails: bool) -> (Tree<f32>, Tree<i32>) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let mut grid = Grid::<f32>::new(0.3).with_transform(Transform::linear(0.1));
            let mut p2ls = ParticlesToLevelSet::<f32, i32>::with_attributes(&mut grid);
            if trails {
                p2ls.rasterize_trails_att(pl, 1.0);
            } else {
                p2ls.rasterize_spheres_att(pl);
            }
            p2ls.finalize(false);
            let att = p2ls.take_attribute_grid().unwrap();
            (grid.tree().clone(), att.tree().clone())
        })
    }

    fn assert_same_tree<T: Value>(a: &Tree<T>, b: &Tree<T>) {
        assert_eq!(a.leaf_count(), b.leaf_count());
        for (la, lb) in a.leaves().zip(b.leaves()) {
            assert_eq!(la.origin(), lb.origin());
            assert_eq!(la.value_mask(), lb.value_mask(), "mask at {:?}", la.origin());
            assert_eq!(la.buffer(), lb.buffer(), "values at {:?}", la.origin());
        }
        let tiles = |t: &Tree<T>| {
            let mut v = Vec::new();
            t.visit_tiles(&mut |b, x, on, l| v.push((b, *x, on, l)));
            v
        };
        assert_eq!(tiles(a), tiles(b));
    }

    #[test]
    fn parallel_matches_sequential() {
        // Overlapping particles of varied radii, so merging has inside voxels, ties and
        // distances to resolve across task boundaries.
        let mut pl = SimpleParticleList::default();
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..3000 {
            pl.positions.push(Vec3d::new(rnd() * 4.0 - 2.0, rnd() * 4.0 - 2.0, rnd() * 4.0 - 2.0));
            pl.radii.push(0.1 + rnd() * 0.4);
            pl.velocities.push(Vec3d::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5));
        }
        // Mirror-image pairs are exactly equidistant from the voxels on the plane x = 0, so
        // their distances tie (with different attributes) across buckets and tasks.
        for i in 0..40 {
            let (y, z, r) = (rnd() * 4.0 - 2.0, rnd() * 4.0 - 2.0, 0.2 + rnd() * 0.3);
            for x in [r * 0.8, -r * 0.8] {
                pl.positions.push(Vec3d::new(x, y, z));
                pl.radii.push(r);
                pl.velocities.push(Vec3d::new(0.0, 0.1 * i as f64, 0.0));
            }
        }
        for trails in [false, true] {
            let [p1, _] = rasterize_with_threads(&pl, 1, trails);
            let [p4, _] = rasterize_with_threads(&pl, 4, trails);
            assert!(p1.active_voxel_count() > 10_000);
            assert_same_tree(&p1, &p4);
            let (g1, a1) = rasterize_att_with_threads(&pl, 1, trails);
            let (g4, a4) = rasterize_att_with_threads(&pl, 4, trails);
            assert_same_tree(&g1, &g4);
            assert_same_tree(&a1, &a4);
        }
    }
}
