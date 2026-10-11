//! Scatter points in the active voxels of a grid (port of `openvdb/tools/PointScatter.h`).
//!
//! The random number machinery reproduces `std::mt19937`, libstdc++'s
//! `std::uniform_real_distribution<double>` and `std::uniform_int_distribution<uint64_t>`
//! bit for bit, so the scatterers produce exactly the same points as OpenVDB built with GCC
//! for the same seed (verified against the installed C++ library).
//!
//! * [`UniformPointScatter`] — a fixed total point count (or density) distributed uniformly
//!   over the active voxels;
//! * [`DenseUniformPointScatter`] — a fixed number of points per active voxel;
//! * [`NonUniformPointScatter`] — a point density proportional to the voxel values.

use super::mesh_support::{visit_values_on_ordered, ActiveValue};
use crate::grid::Grid;
use crate::math::{Vec3d, Vec3f};
use crate::types::Value;

/// A 32-bit uniform random bit generator (`min() == 0`, `max() == u32::MAX`).
pub trait RandomGenerator: Clone {
    /// The next 32 random bits.
    fn next_u32(&mut self) -> u32;
}

macro_rules! mersenne_twister {
    ($(#[$doc:meta])* $name:ident, n = $n:expr, m = $m:expr, r = $r:expr, a = $a:expr, u = $u:expr,
     d = $d:expr, s = $s:expr, b = $b:expr, t = $t:expr, c = $c:expr, l = $l:expr, f = $f:expr) => {
        $(#[$doc])*
        #[derive(Clone)]
        pub struct $name {
            mt: [u32; $n],
            index: usize,
        }

        impl $name {
            /// The default seed of the standard library engines.
            pub const DEFAULT_SEED: u32 = 5489;

            /// A generator initialized with `seed` (as the C++ engine constructor).
            pub fn new(seed: u32) -> Self {
                let mut mt = [0u32; $n];
                mt[0] = seed;
                for i in 1..$n {
                    mt[i] = ($f as u32).wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
                }
                $name { mt, index: $n }
            }

            fn twist(&mut self) {
                const UPPER: u32 = !0u32 << $r;
                const LOWER: u32 = !UPPER;
                for i in 0..$n {
                    let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % $n] & LOWER);
                    let mut v = self.mt[(i + $m) % $n] ^ (y >> 1);
                    if y & 1 != 0 {
                        v ^= $a;
                    }
                    self.mt[i] = v;
                }
                self.index = 0;
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new(Self::DEFAULT_SEED)
            }
        }

        impl RandomGenerator for $name {
            fn next_u32(&mut self) -> u32 {
                if self.index >= $n {
                    self.twist();
                }
                let mut y = self.mt[self.index];
                self.index += 1;
                y ^= (y >> $u) & $d;
                y ^= (y << $s) & $b;
                y ^= (y << $t) & $c;
                y ^= y >> $l;
                y
            }
        }
    };
}

mersenne_twister!(
    /// The Mersenne twister `std::mt19937`.
    Mt19937, n = 624, m = 397, r = 31, a = 0x9908_b0dfu32, u = 11, d = 0xffff_ffffu32, s = 7,
    b = 0x9d2c_5680u32, t = 15, c = 0xefc6_0000u32, l = 18, f = 1812433253u32
);

mersenne_twister!(
    /// The Mersenne twister `boost::mt11213b` (as used by OpenVDB's `fillWithSpheres`).
    Mt11213b, n = 351, m = 175, r = 19, a = 0xccab_8ee7u32, u = 11, d = 0xffff_ffffu32, s = 7,
    b = 0x31b6_ab00u32, t = 15, c = 0xffe5_0000u32, l = 17, f = 1812433253u32
);

/// `std::generate_canonical<double, 53>` for a 32-bit generator (libstdc++).
pub fn generate_canonical<G: RandomGenerator>(g: &mut G) -> f64 {
    let r = 4294967296.0f64;
    let mut sum = 0.0f64;
    let mut tmp = 1.0f64;
    for _ in 0..2 {
        sum += g.next_u32() as f64 * tmp;
        tmp *= r;
    }
    let ret = sum / tmp;
    if ret >= 1.0 {
        1.0f64 - f64::EPSILON / 2.0
    } else {
        ret
    }
}

/// `std::uniform_int_distribution<uint64_t>(lo, hi)` for a 32-bit generator (libstdc++).
pub fn uniform_int<G: RandomGenerator>(g: &mut G, lo: u64, hi: u64) -> u64 {
    let urngrange: u64 = u32::MAX as u64;
    let urange = hi - lo;
    let ret = if urngrange > urange {
        let range = (urange + 1) as u32;
        let mut product = g.next_u32() as u64 * range as u64;
        let mut low = product as u32;
        if low < range {
            let threshold = range.wrapping_neg() % range;
            while low < threshold {
                product = g.next_u32() as u64 * range as u64;
                low = product as u32;
            }
        }
        product >> 32
    } else if urngrange < urange {
        loop {
            let uerngrange = urngrange + 1;
            let tmp = uerngrange.wrapping_mul(uniform_int(g, 0, urange / uerngrange));
            let ret = tmp.wrapping_add(g.next_u32() as u64);
            if !(ret > urange || ret < tmp) {
                break ret;
            }
        }
    } else {
        g.next_u32() as u64
    };
    ret + lo
}

/// Uniform random numbers in `[0, 1)` (`math::Rand01<double, Engine>`). Like OpenVDB, it
/// holds its own copy of the engine.
#[derive(Clone)]
pub struct Rand01<G: RandomGenerator> {
    engine: G,
}

impl<G: RandomGenerator> Rand01<G> {
    pub fn new(engine: &G) -> Self {
        Rand01 { engine: engine.clone() }
    }
    /// The internal engine.
    pub fn engine(&self) -> &G {
        &self.engine
    }
    /// The next random number in `[0, 1)`.
    pub fn next(&mut self) -> f64 {
        generate_canonical(&mut self.engine)
    }
}

/// Uniform random integers in `[imin, imax]` (`math::RandInt<uint64_t, Engine>`).
#[derive(Clone)]
pub struct RandInt<G: RandomGenerator> {
    engine: G,
    lo: u64,
    hi: u64,
}

impl<G: RandomGenerator> RandInt<G> {
    pub fn new(engine: &G, imin: u64, imax: u64) -> Self {
        RandInt { engine: engine.clone(), lo: imin.min(imax), hi: imin.max(imax) }
    }
    pub fn next(&mut self) -> u64 {
        uniform_int(&mut self.engine, self.lo, self.hi)
    }
}

/// Receives scattered world-space points (OpenVDB's `PointAccessorType` with `add()`).
pub trait PointAccessor {
    fn add(&mut self, pos: Vec3d);
}
impl PointAccessor for Vec<Vec3d> {
    fn add(&mut self, pos: Vec3d) {
        self.push(pos)
    }
}
impl PointAccessor for Vec<Vec3f> {
    fn add(&mut self, pos: Vec3d) {
        self.push(Vec3f::new(pos[0] as f32, pos[1] as f32, pos[2] as f32))
    }
}

struct BaseScatter<'a, P: PointAccessor, G: RandomGenerator> {
    points: &'a mut P,
    point_count: u64,
    voxel_count: u64,
    spread: f64,
    rand01: Rand01<G>,
}

impl<'a, P: PointAccessor, G: RandomGenerator> BaseScatter<'a, P, G> {
    fn new(points: &'a mut P, rng: &G, spread: f64) -> Self {
        BaseScatter { points, point_count: 0, voxel_count: 0, spread: spread.clamp(0.0, 1.0), rand01: Rand01::new(rng) }
    }
    #[inline]
    fn get_rand01(&mut self) -> f64 {
        self.rand01.next()
    }
    #[inline]
    fn get_rand(&mut self) -> f64 {
        0.5 + self.spread * (self.rand01.next() - 0.5)
    }
    fn add_point<T: Value>(&mut self, grid: &Grid<T>, dmin: Vec3d) {
        // GCC evaluates the constructor arguments of OpenVDB's `Vec3R(x, y, z)` right to
        // left; the random numbers are drawn in the same order to reproduce its output.
        let z = dmin[2] + self.get_rand();
        let y = dmin[1] + self.get_rand();
        let x = dmin[0] + self.get_rand();
        self.points.add(grid.index_to_world(Vec3d::new(x, y, z)));
        self.point_count += 1;
    }
    fn add_point_sized<T: Value>(&mut self, grid: &Grid<T>, dmin: Vec3d, size: [i32; 3]) {
        let z = dmin[2] + size[2] as f64 * self.get_rand();
        let y = dmin[1] + size[1] as f64 * self.get_rand();
        let x = dmin[0] + size[0] as f64 * self.get_rand();
        self.points.add(grid.index_to_world(Vec3d::new(x, y, z)));
        self.point_count += 1;
    }
}

const OFFSET: f64 = 0.5;

fn tile_min(b: &crate::math::CoordBBox) -> Vec3d {
    b.min.as_vec3d() - Vec3d::splat(OFFSET)
}

/// Uniformly scatters a fixed total number of points (or a fixed density) over the active
/// voxels and tiles of a grid (port of `UniformPointScatter`).
pub struct UniformPointScatter<'a, P: PointAccessor, G: RandomGenerator> {
    base: BaseScatter<'a, P, G>,
    target_point_count: u64,
    points_per_volume: f32,
}

impl<'a, P: PointAccessor, G: RandomGenerator> UniformPointScatter<'a, P, G> {
    /// Scatter `point_count` points in total.
    pub fn with_count(points: &'a mut P, point_count: u64, rng: &G, spread: f64) -> Self {
        UniformPointScatter {
            base: BaseScatter::new(points, rng, spread),
            target_point_count: point_count,
            points_per_volume: 0.0,
        }
    }
    /// Scatter `points_per_volume` points per unit of world-space volume.
    pub fn with_density(points: &'a mut P, points_per_volume: f32, rng: &G, spread: f64) -> Self {
        UniformPointScatter { base: BaseScatter::new(points, rng, spread), target_point_count: 0, points_per_volume }
    }

    /// Scatter points into the active voxels of `grid`; returns false if nothing was done.
    pub fn run<T: Value>(&mut self, grid: &Grid<T>) -> bool {
        self.base.voxel_count = grid.active_voxel_count();
        if self.base.voxel_count == 0 {
            return false;
        }
        let voxel_volume = grid.transform().voxel_volume();
        if self.points_per_volume > 0.0 {
            self.target_point_count =
                (self.points_per_volume as f64 * voxel_volume * self.base.voxel_count as f64) as u64;
        } else if self.target_point_count > 0 {
            self.points_per_volume =
                self.target_point_count as f32 / (voxel_volume * self.base.voxel_count as f64) as f32;
        } else {
            return false;
        }
        let mut rand = RandInt::new(self.base.rand01.engine(), 0, self.base.voxel_count - 1);
        let mut ids: Vec<u64> = (0..self.target_point_count).map(|_| rand.next()).collect();
        ids.sort_unstable();

        let mut values = Vec::new();
        visit_values_on_ordered(grid.tree(), &mut |v| values.push(v));
        let count_of = |v: &ActiveValue<T>| match v {
            ActiveValue::Voxel(..) => 1u64,
            ActiveValue::Tile(b, _) => b.volume(),
        };
        let mut vi = 0usize;
        let mut n = count_of(&values[0]);
        for &voxel_id in &ids {
            while n <= voxel_id {
                vi += 1;
                n += count_of(&values[vi]);
            }
            match values[vi] {
                ActiveValue::Voxel(c, _) => self.base.add_point(grid, c.as_vec3d() - Vec3d::splat(OFFSET)),
                ActiveValue::Tile(b, _) => {
                    let d = b.dim();
                    self.base.add_point_sized(grid, tile_min(&b), [d[0], d[1], d[2]])
                }
            }
        }
        true
    }

    pub fn get_point_count(&self) -> u64 {
        self.base.point_count
    }
    pub fn get_voxel_count(&self) -> u64 {
        self.base.voxel_count
    }
    pub fn get_points_per_volume(&self) -> f32 {
        self.points_per_volume
    }
    pub fn get_target_point_count(&self) -> u64 {
        self.target_point_count
    }
}

/// Scatters a fixed (possibly fractional) number of points per active voxel
/// (port of `DenseUniformPointScatter`).
pub struct DenseUniformPointScatter<'a, P: PointAccessor, G: RandomGenerator> {
    base: BaseScatter<'a, P, G>,
    points_per_voxel: f32,
}

impl<'a, P: PointAccessor, G: RandomGenerator> DenseUniformPointScatter<'a, P, G> {
    pub fn new(points: &'a mut P, points_per_voxel: f32, rng: &G, spread: f64) -> Self {
        DenseUniformPointScatter { base: BaseScatter::new(points, rng, spread), points_per_voxel }
    }

    /// Scatter points into the active voxels of `grid`; returns false if nothing was done.
    pub fn run<T: Value>(&mut self, grid: &Grid<T>) -> bool {
        if (self.points_per_voxel as f64) < 1.0e-6 {
            return false;
        }
        self.base.voxel_count = grid.active_voxel_count();
        if self.base.voxel_count == 0 {
            return false;
        }
        let ppv = self.points_per_voxel.floor() as i32;
        let delta = (self.points_per_voxel - ppv as f32) as f64;
        let fractional = delta.abs() > 1.0e-6;
        let mut values = Vec::new();
        visit_values_on_ordered(grid.tree(), &mut |v| values.push(v));
        for v in values {
            match v {
                ActiveValue::Voxel(c, _) => {
                    let dmin = c.as_vec3d() - Vec3d::splat(OFFSET);
                    for _ in 0..ppv {
                        self.base.add_point(grid, dmin);
                    }
                    if fractional && self.base.get_rand01() < delta {
                        self.base.add_point(grid, dmin);
                    }
                }
                ActiveValue::Tile(b, _) => {
                    let d = b.dim();
                    let size = [d[0], d[1], d[2]];
                    let dmin = tile_min(&b);
                    let dd = (self.points_per_voxel * b.volume() as f32) as f64;
                    let m = dd.floor() as i32;
                    for _ in 0..m {
                        self.base.add_point_sized(grid, dmin, size);
                    }
                    if self.base.get_rand01() < dd - m as f64 {
                        self.base.add_point_sized(grid, dmin, size);
                    }
                }
            }
        }
        true
    }

    pub fn get_point_count(&self) -> u64 {
        self.base.point_count
    }
    pub fn get_voxel_count(&self) -> u64 {
        self.base.voxel_count
    }
    pub fn get_points_per_voxel(&self) -> f32 {
        self.points_per_voxel
    }
}

/// Scatters points with a density proportional to the voxel values
/// (port of `NonUniformPointScatter`).
pub struct NonUniformPointScatter<'a, P: PointAccessor, G: RandomGenerator> {
    base: BaseScatter<'a, P, G>,
    points_per_volume: f32,
}

impl<'a, P: PointAccessor, G: RandomGenerator> NonUniformPointScatter<'a, P, G> {
    /// `points_per_volume` is a multiplier for the local point density given by the values.
    pub fn new(points: &'a mut P, points_per_volume: f32, rng: &G, spread: f64) -> Self {
        NonUniformPointScatter { base: BaseScatter::new(points, rng, spread), points_per_volume }
    }

    /// Scatter points into the active voxels of `grid`; returns false if nothing was done.
    pub fn run<T: Value + num_traits::ToPrimitive>(&mut self, grid: &Grid<T>) -> bool {
        if self.points_per_volume <= 0.0 {
            return false;
        }
        self.base.voxel_count = grid.active_voxel_count();
        if self.base.voxel_count == 0 {
            return false;
        }
        let dim = grid.voxel_size();
        let volume_per_voxel = dim[0] * dim[1] * dim[2];
        let points_per_voxel = self.points_per_volume as f64 * volume_per_voxel;
        let mut values = Vec::new();
        visit_values_on_ordered(grid.tree(), &mut |v| values.push(v));
        for v in values {
            match v {
                ActiveValue::Voxel(c, val) => {
                    let d = val.to_f64().unwrap_or(0.0) * points_per_voxel;
                    let n = d as i32;
                    let dmin = c.as_vec3d() - Vec3d::splat(OFFSET);
                    for _ in 0..n {
                        self.base.add_point(grid, dmin);
                    }
                    if self.base.get_rand01() < d - n as f64 {
                        self.base.add_point(grid, dmin);
                    }
                }
                ActiveValue::Tile(b, val) => {
                    let d = val.to_f64().unwrap_or(0.0) * points_per_voxel * b.volume() as f64;
                    let n = d as i32;
                    let dd = b.dim();
                    let size = [dd[0], dd[1], dd[2]];
                    let dmin = tile_min(&b);
                    for _ in 0..n {
                        self.base.add_point_sized(grid, dmin, size);
                    }
                    if self.base.get_rand01() < d - n as f64 {
                        self.base.add_point_sized(grid, dmin, size);
                    }
                }
            }
        }
        true
    }

    pub fn get_point_count(&self) -> u64 {
        self.base.point_count
    }
    pub fn get_voxel_count(&self) -> u64 {
        self.base.voxel_count
    }
    pub fn get_point_per_volume(&self) -> f32 {
        self.points_per_volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, CoordBBox};

    #[test]
    fn mt19937_reference() {
        // The 10000th output of a default-constructed std::mt19937 is 4123659995.
        let mut g = Mt19937::default();
        let mut v = 0;
        for _ in 0..10000 {
            v = g.next_u32();
        }
        assert_eq!(v, 4123659995);
    }

    #[test]
    fn scatter_counts() {
        let mut grid = Grid::<f32>::new(0.0);
        grid.tree_mut().fill(&CoordBBox::new(Coord::new(0, 0, 0), Coord::new(15, 15, 15)), 1.0, true);
        grid.tree_mut().set_value(&Coord::new(-3, 2, 1), 0.5);
        let rng = Mt19937::default();
        let mut pts: Vec<Vec3d> = Vec::new();
        let mut s = UniformPointScatter::with_count(&mut pts, 1000, &rng, 1.0);
        assert!(s.run(&grid));
        assert_eq!(s.get_point_count(), 1000);
        assert_eq!(pts.len(), 1000);
        for p in &pts {
            assert!(p[0] >= -3.5 && p[0] <= 15.5);
        }
        let mut pts2: Vec<Vec3d> = Vec::new();
        let mut d = DenseUniformPointScatter::new(&mut pts2, 2.0, &rng, 1.0);
        assert!(d.run(&grid));
        assert_eq!(pts2.len() as u64, 2 * grid.active_voxel_count());
        let mut pts3: Vec<Vec3d> = Vec::new();
        let mut nu = NonUniformPointScatter::new(&mut pts3, 3.0, &rng, 1.0);
        assert!(nu.run(&grid));
        // 4096 voxels with density 3 plus one voxel with density 1.5.
        assert!(pts3.len() == 4096 * 3 + 1 || pts3.len() == 4096 * 3 + 2);
    }
}
