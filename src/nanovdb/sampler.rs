//! Interpolation of voxel values at fractional index-space positions (port of
//! `nanovdb/math/SampleFromVoxels.h`): nearest neighbor (order 0), trilinear (1),
//! triquadratic (2) and tricubic (3), each with an optional stencil cache.
//!
//! Positions are double precision; scalar values are combined with the same mix of single
//! and double precision arithmetic as the C++ samplers with `Vec3d` positions.

use super::build_types::BuildType;
use super::tree::{NanoTree, ReadAccessor};
use crate::math::{Coord, Vec3d};
use std::cell::{Cell, RefCell};

/// Anything that returns voxel values: a [`ReadAccessor`] or a [`NanoTree`].
pub trait VoxelSource {
    /// The voxel value type.
    type Value: Copy;
    /// The value at `ijk`.
    fn get_value(&self, ijk: Coord) -> Self::Value;
}

impl<B: BuildType> VoxelSource for ReadAccessor<'_, B> {
    type Value = B::Value;
    #[inline]
    fn get_value(&self, ijk: Coord) -> B::Value {
        ReadAccessor::get_value(self, ijk)
    }
}

impl<B: BuildType> VoxelSource for NanoTree<'_, B> {
    type Value = B::Value;
    #[inline]
    fn get_value(&self, ijk: Coord) -> B::Value {
        NanoTree::get_value(self, ijk)
    }
}

/// Floating-point voxel values that can be interpolated (`f32`, `f64`).
pub trait SampleReal:
    Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::AddAssign
{
    /// Conversion from a double.
    fn from_double(v: f64) -> Self;
    /// Conversion to a double.
    fn to_double(self) -> f64;
    /// Zero.
    fn zero_val() -> Self {
        Self::from_double(0.0)
    }
}

impl SampleReal for f32 {
    #[inline]
    fn from_double(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_double(self) -> f64 {
        self as f64
    }
}

impl SampleReal for f64 {
    #[inline]
    fn from_double(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_double(self) -> f64 {
        self
    }
}

/// Split a position into its integer cell and fractional offset (`Floor<CoordT>(xyz)`).
#[inline]
pub fn floor_split(xyz: Vec3d) -> (Coord, Vec3d) {
    let f = [xyz[0].floor(), xyz[1].floor(), xyz[2].floor()];
    (Coord::new(f[0] as i32, f[1] as i32, f[2] as i32), Vec3d::new(xyz[0] - f[0], xyz[1] - f[1], xyz[2] - f[2]))
}

/// The nearest voxel of a position (`Round<CoordT>` for double precision).
#[inline]
pub fn round_coord(xyz: Vec3d) -> Coord {
    Coord::new((xyz[0] + 0.5).floor() as i32, (xyz[1] + 0.5).floor() as i32, (xyz[2] + 0.5).floor() as i32)
}

/// Common interface of the samplers.
pub trait Sampler {
    /// The interpolated value type.
    type Value;
    /// Interpolation order (0 to 3).
    const ORDER: u32;
    /// Interpolate at the index-space position `xyz`.
    fn sample(&self, xyz: Vec3d) -> Self::Value;
    /// The value of the voxel `ijk` (no interpolation).
    fn sample_voxel(&self, ijk: Coord) -> Self::Value;
}

// ------------------------------------------------------------------------------------------
// Order 0
// ------------------------------------------------------------------------------------------

/// Nearest-neighbor sampler (`SampleFromVoxels<T, 0, UseCache>`), for any value type.
pub struct NearestSampler<'s, S: VoxelSource> {
    src: &'s S,
    cache: Option<Cell<(Coord, Option<S::Value>)>>,
}

impl<'s, S: VoxelSource> NearestSampler<'s, S> {
    /// A sampler; `use_cache` remembers the last voxel looked up.
    pub fn new(src: &'s S, use_cache: bool) -> Self {
        NearestSampler { src, cache: use_cache.then(|| Cell::new((Coord::splat(i32::MAX), None))) }
    }
}

impl<S: VoxelSource> Sampler for NearestSampler<'_, S> {
    type Value = S::Value;
    const ORDER: u32 = 0;
    fn sample(&self, xyz: Vec3d) -> S::Value {
        self.sample_voxel(round_coord(xyz))
    }
    fn sample_voxel(&self, ijk: Coord) -> S::Value {
        match &self.cache {
            None => self.src.get_value(ijk),
            Some(c) => {
                if let (pos, Some(v)) = c.get() {
                    if pos == ijk {
                        return v;
                    }
                }
                let v = self.src.get_value(ijk);
                c.set((ijk, Some(v)));
                v
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// Order 1
// ------------------------------------------------------------------------------------------

/// Fetch the 2x2x2 stencil at `ijk` (`TrilinearSampler::stencil`).
pub fn trilinear_stencil<S: VoxelSource>(src: &S, ijk: Coord) -> [[[S::Value; 2]; 2]; 2] {
    let g = |dx: i32, dy: i32, dz: i32| src.get_value(Coord::new(ijk[0] + dx, ijk[1] + dy, ijk[2] + dz));
    [[[g(0, 0, 0), g(0, 0, 1)], [g(0, 1, 0), g(0, 1, 1)]], [[g(1, 0, 0), g(1, 0, 1)], [g(1, 1, 0), g(1, 1, 1)]]]
}

#[inline]
fn lerp<T: SampleReal>(a: T, b: T, w: f64) -> T {
    a + T::from_double(w) * (b - a)
}

/// Trilinear interpolation of a stencil at fractional offset `uvw`.
pub fn trilinear_sample<T: SampleReal>(uvw: Vec3d, v: &[[[T; 2]; 2]; 2]) -> T {
    lerp(
        lerp(lerp(v[0][0][0], v[0][0][1], uvw[2]), lerp(v[0][1][0], v[0][1][1], uvw[2]), uvw[1]),
        lerp(lerp(v[1][0][0], v[1][0][1], uvw[2]), lerp(v[1][1][0], v[1][1][1], uvw[2]), uvw[1]),
        uvw[0],
    )
}

/// Gradient (in index space) of the trilinear interpolant of a stencil.
pub fn trilinear_gradient<T: SampleReal>(uvw: Vec3d, v: &[[[T; 2]; 2]; 2]) -> [T; 3] {
    let mut d = [v[0][0][1] - v[0][0][0], v[0][1][1] - v[0][1][0], v[1][0][1] - v[1][0][0], v[1][1][1] - v[1][1][0]];
    let gz = lerp(lerp(d[0], d[1], uvw[1]), lerp(d[2], d[3], uvw[1]), uvw[0]);
    let w = T::from_double(uvw[2]);
    d[0] = v[0][0][0] + d[0] * w;
    d[1] = v[0][1][0] + d[1] * w;
    d[2] = v[1][0][0] + d[2] * w;
    d[3] = v[1][1][0] + d[3] * w;
    let gx = lerp(d[2], d[3], uvw[1]) - lerp(d[0], d[1], uvw[1]);
    let gy = lerp(d[1] - d[0], d[3] - d[2], uvw[0]);
    [gx, gy, gz]
}

/// True if the stencil values change sign.
pub fn trilinear_zero_crossing<T: SampleReal>(v: &[[[T; 2]; 2]; 2]) -> bool {
    let z = T::zero_val();
    let less = v[0][0][0] < z;
    v.iter().flatten().flatten().any(|x| (*x < z) != less)
}

/// A cached stencil: the cell it belongs to and its values.
type StencilCache<V, const N: usize> = Option<RefCell<(Coord, [[[V; N]; N]; N])>>;

/// Trilinear sampler (`SampleFromVoxels<T, 1, UseCache>`).
pub struct TrilinearSampler<'s, S: VoxelSource> {
    src: &'s S,
    cache: StencilCache<S::Value, 2>,
}

impl<'s, S: VoxelSource> TrilinearSampler<'s, S>
where
    S::Value: SampleReal,
{
    /// A sampler; `use_cache` keeps the last stencil.
    pub fn new(src: &'s S, use_cache: bool) -> Self {
        let z = S::Value::zero_val();
        TrilinearSampler { src, cache: use_cache.then(|| RefCell::new((Coord::splat(i32::MAX), [[[z; 2]; 2]; 2]))) }
    }
    fn stencil(&self, ijk: Coord) -> [[[S::Value; 2]; 2]; 2] {
        match &self.cache {
            None => trilinear_stencil(self.src, ijk),
            Some(c) => {
                let mut c = c.borrow_mut();
                if c.0 != ijk {
                    *c = (ijk, trilinear_stencil(self.src, ijk));
                }
                c.1
            }
        }
    }
    /// Index-space gradient at `xyz`.
    pub fn gradient(&self, xyz: Vec3d) -> [S::Value; 3] {
        let (ijk, uvw) = floor_split(xyz);
        trilinear_gradient(uvw, &self.stencil(ijk))
    }
    /// True if the values around `xyz` change sign.
    pub fn zero_crossing(&self, xyz: Vec3d) -> bool {
        let (ijk, _) = floor_split(xyz);
        trilinear_zero_crossing(&self.stencil(ijk))
    }
}

impl<S: VoxelSource> Sampler for TrilinearSampler<'_, S>
where
    S::Value: SampleReal,
{
    type Value = S::Value;
    const ORDER: u32 = 1;
    fn sample(&self, xyz: Vec3d) -> S::Value {
        let (ijk, uvw) = floor_split(xyz);
        if self.cache.is_some() {
            return trilinear_sample(uvw, &self.stencil(ijk));
        }
        // Uncached evaluation order of the C++ implementation.
        let s = self.src;
        let mut c = ijk;
        let vz = s.get_value(c);
        c[2] += 1;
        let vz1 = s.get_value(c);
        let vy = lerp(vz, vz1, uvw[2]);
        c[1] += 1;
        let vz1 = s.get_value(c);
        c[2] -= 1;
        let vz = s.get_value(c);
        let vy1 = lerp(vz, vz1, uvw[2]);
        let vx = lerp(vy, vy1, uvw[1]);
        c[0] += 1;
        let vz = s.get_value(c);
        c[2] += 1;
        let vz1 = s.get_value(c);
        let vy1 = lerp(vz, vz1, uvw[2]);
        c[1] -= 1;
        let vz1 = s.get_value(c);
        c[2] -= 1;
        let vz = s.get_value(c);
        let vy = lerp(vz, vz1, uvw[2]);
        let vx1 = lerp(vy, vy1, uvw[1]);
        lerp(vx, vx1, uvw[0])
    }
    fn sample_voxel(&self, ijk: Coord) -> S::Value {
        if let Some(c) = &self.cache {
            let c = c.borrow();
            if c.0 == ijk {
                return c.1[0][0][0];
            }
        }
        self.src.get_value(ijk)
    }
}

// ------------------------------------------------------------------------------------------
// Order 2
// ------------------------------------------------------------------------------------------

/// Fetch the 3x3x3 stencil centered at `ijk`.
pub fn triquadratic_stencil<S: VoxelSource>(src: &S, ijk: Coord) -> [[[S::Value; 3]; 3]; 3] {
    let g = |dx: i32, dy: i32, dz: i32| src.get_value(Coord::new(ijk[0] + dx - 1, ijk[1] + dy - 1, ijk[2] + dz - 1));
    let row = |dx: i32, dy: i32| [g(dx, dy, 0), g(dx, dy, 1), g(dx, dy, 2)];
    let plane = |dx: i32| [row(dx, 0), row(dx, 1), row(dx, 2)];
    [plane(0), plane(1), plane(2)]
}

#[inline]
fn quad_kernel<T: SampleReal>(v: &[T; 3], w: f64) -> T {
    let h = T::from_double(0.5);
    T::from_double(
        w * (w * (h * (v[0] + v[2]) - v[1]).to_double() + (h * (v[2] - v[0])).to_double()) + v[1].to_double(),
    )
}

/// Triquadratic interpolation of a stencil at fractional offset `uvw`.
pub fn triquadratic_sample<T: SampleReal>(uvw: Vec3d, v: &[[[T; 3]; 3]; 3]) -> T {
    let mut vx = [T::zero_val(); 3];
    for dx in 0..3 {
        let mut vy = [T::zero_val(); 3];
        for dy in 0..3 {
            vy[dy] = quad_kernel(&v[dx][dy], uvw[2]);
        }
        vx[dx] = quad_kernel(&vy, uvw[1]);
    }
    quad_kernel(&vx, uvw[0])
}

/// Triquadratic sampler (`SampleFromVoxels<T, 2, UseCache>`).
pub struct TriquadraticSampler<'s, S: VoxelSource> {
    src: &'s S,
    cache: StencilCache<S::Value, 3>,
}

impl<'s, S: VoxelSource> TriquadraticSampler<'s, S>
where
    S::Value: SampleReal,
{
    /// A sampler; `use_cache` keeps the last stencil.
    pub fn new(src: &'s S, use_cache: bool) -> Self {
        let z = S::Value::zero_val();
        TriquadraticSampler { src, cache: use_cache.then(|| RefCell::new((Coord::splat(i32::MAX), [[[z; 3]; 3]; 3]))) }
    }
    fn stencil(&self, ijk: Coord) -> [[[S::Value; 3]; 3]; 3] {
        match &self.cache {
            None => triquadratic_stencil(self.src, ijk),
            Some(c) => {
                let mut c = c.borrow_mut();
                if c.0 != ijk {
                    *c = (ijk, triquadratic_stencil(self.src, ijk));
                }
                c.1
            }
        }
    }
    /// True if the values around `xyz` change sign.
    pub fn zero_crossing(&self, xyz: Vec3d) -> bool {
        let (ijk, _) = floor_split(xyz);
        let v = self.stencil(ijk);
        let z = S::Value::zero_val();
        let less = v[0][0][0] < z;
        v.iter().flatten().flatten().any(|x| (*x < z) != less)
    }
}

impl<S: VoxelSource> Sampler for TriquadraticSampler<'_, S>
where
    S::Value: SampleReal,
{
    type Value = S::Value;
    const ORDER: u32 = 2;
    fn sample(&self, xyz: Vec3d) -> S::Value {
        let (ijk, uvw) = floor_split(xyz);
        triquadratic_sample(uvw, &self.stencil(ijk))
    }
    fn sample_voxel(&self, ijk: Coord) -> S::Value {
        if let Some(c) = &self.cache {
            let c = c.borrow();
            if c.0 == ijk {
                return c.1[1][1][1];
            }
        }
        self.src.get_value(ijk)
    }
}

// ------------------------------------------------------------------------------------------
// Order 3
// ------------------------------------------------------------------------------------------

/// The 64x64 matrix mapping corner values and derivatives to tricubic coefficients.
#[rustfmt::skip]
static TRICUBIC_A: [[i8; 64]; 64] = include!("tricubic_matrix.in");

/// The 64 tricubic coefficients of the cell at `ijk` (`TricubicSampler::stencil`).
pub fn tricubic_stencil<S: VoxelSource>(src: &S, ijk: Coord) -> [S::Value; 64]
where
    S::Value: SampleReal,
{
    let mut c = [S::Value::zero_val(); 64];
    for i in -1..3 {
        for j in -1..3 {
            for k in -1..3 {
                c[(((i + 1) << 4) + ((j + 1) << 2) + k + 1) as usize] =
                    src.get_value(Coord::new(ijk[0] + i, ijk[1] + j, ijk[2] + k));
            }
        }
    }
    let f = |i: i32, j: i32, k: i32| c[(((i + 1) << 4) + ((j + 1) << 2) + k + 1) as usize];
    let half = S::Value::from_double(0.5);
    let quarter = S::Value::from_double(0.25);
    let eighth = S::Value::from_double(0.125);
    let x: [S::Value; 64] = [
        f(0, 0, 0),
        f(1, 0, 0),
        f(0, 1, 0),
        f(1, 1, 0),
        f(0, 0, 1),
        f(1, 0, 1),
        f(0, 1, 1),
        f(1, 1, 1),
        half * (f(1, 0, 0) - f(-1, 0, 0)),
        half * (f(2, 0, 0) - f(0, 0, 0)),
        half * (f(1, 1, 0) - f(-1, 1, 0)),
        half * (f(2, 1, 0) - f(0, 1, 0)),
        half * (f(1, 0, 1) - f(-1, 0, 1)),
        half * (f(2, 0, 1) - f(0, 0, 1)),
        half * (f(1, 1, 1) - f(-1, 1, 1)),
        half * (f(2, 1, 1) - f(0, 1, 1)),
        half * (f(0, 1, 0) - f(0, -1, 0)),
        half * (f(1, 1, 0) - f(1, -1, 0)),
        half * (f(0, 2, 0) - f(0, 0, 0)),
        half * (f(1, 2, 0) - f(1, 0, 0)),
        half * (f(0, 1, 1) - f(0, -1, 1)),
        half * (f(1, 1, 1) - f(1, -1, 1)),
        half * (f(0, 2, 1) - f(0, 0, 1)),
        half * (f(1, 2, 1) - f(1, 0, 1)),
        half * (f(0, 0, 1) - f(0, 0, -1)),
        half * (f(1, 0, 1) - f(1, 0, -1)),
        half * (f(0, 1, 1) - f(0, 1, -1)),
        half * (f(1, 1, 1) - f(1, 1, -1)),
        half * (f(0, 0, 2) - f(0, 0, 0)),
        half * (f(1, 0, 2) - f(1, 0, 0)),
        half * (f(0, 1, 2) - f(0, 1, 0)),
        half * (f(1, 1, 2) - f(1, 1, 0)),
        quarter * (f(1, 1, 0) - f(-1, 1, 0) - f(1, -1, 0) + f(-1, -1, 0)),
        quarter * (f(2, 1, 0) - f(0, 1, 0) - f(2, -1, 0) + f(0, -1, 0)),
        quarter * (f(1, 2, 0) - f(-1, 2, 0) - f(1, 0, 0) + f(-1, 0, 0)),
        quarter * (f(2, 2, 0) - f(0, 2, 0) - f(2, 0, 0) + f(0, 0, 0)),
        quarter * (f(1, 1, 1) - f(-1, 1, 1) - f(1, -1, 1) + f(-1, -1, 1)),
        quarter * (f(2, 1, 1) - f(0, 1, 1) - f(2, -1, 1) + f(0, -1, 1)),
        quarter * (f(1, 2, 1) - f(-1, 2, 1) - f(1, 0, 1) + f(-1, 0, 1)),
        quarter * (f(2, 2, 1) - f(0, 2, 1) - f(2, 0, 1) + f(0, 0, 1)),
        quarter * (f(1, 0, 1) - f(-1, 0, 1) - f(1, 0, -1) + f(-1, 0, -1)),
        quarter * (f(2, 0, 1) - f(0, 0, 1) - f(2, 0, -1) + f(0, 0, -1)),
        quarter * (f(1, 1, 1) - f(-1, 1, 1) - f(1, 1, -1) + f(-1, 1, -1)),
        quarter * (f(2, 1, 1) - f(0, 1, 1) - f(2, 1, -1) + f(0, 1, -1)),
        quarter * (f(1, 0, 2) - f(-1, 0, 2) - f(1, 0, 0) + f(-1, 0, 0)),
        quarter * (f(2, 0, 2) - f(0, 0, 2) - f(2, 0, 0) + f(0, 0, 0)),
        quarter * (f(1, 1, 2) - f(-1, 1, 2) - f(1, 1, 0) + f(-1, 1, 0)),
        quarter * (f(2, 1, 2) - f(0, 1, 2) - f(2, 1, 0) + f(0, 1, 0)),
        quarter * (f(0, 1, 1) - f(0, -1, 1) - f(0, 1, -1) + f(0, -1, -1)),
        quarter * (f(1, 1, 1) - f(1, -1, 1) - f(1, 1, -1) + f(1, -1, -1)),
        quarter * (f(0, 2, 1) - f(0, 0, 1) - f(0, 2, -1) + f(0, 0, -1)),
        quarter * (f(1, 2, 1) - f(1, 0, 1) - f(1, 2, -1) + f(1, 0, -1)),
        quarter * (f(0, 1, 2) - f(0, -1, 2) - f(0, 1, 0) + f(0, -1, 0)),
        quarter * (f(1, 1, 2) - f(1, -1, 2) - f(1, 1, 0) + f(1, -1, 0)),
        quarter * (f(0, 2, 2) - f(0, 0, 2) - f(0, 2, 0) + f(0, 0, 0)),
        quarter * (f(1, 2, 2) - f(1, 0, 2) - f(1, 2, 0) + f(1, 0, 0)),
        eighth
            * (f(1, 1, 1) - f(-1, 1, 1) - f(1, -1, 1) + f(-1, -1, 1) - f(1, 1, -1) + f(-1, 1, -1) + f(1, -1, -1)
                - f(-1, -1, -1)),
        eighth
            * (f(2, 1, 1) - f(0, 1, 1) - f(2, -1, 1) + f(0, -1, 1) - f(2, 1, -1) + f(0, 1, -1) + f(2, -1, -1)
                - f(0, -1, -1)),
        eighth
            * (f(1, 2, 1) - f(-1, 2, 1) - f(1, 0, 1) + f(-1, 0, 1) - f(1, 2, -1) + f(-1, 2, -1) + f(1, 0, -1)
                - f(-1, 0, -1)),
        eighth
            * (f(2, 2, 1) - f(0, 2, 1) - f(2, 0, 1) + f(0, 0, 1) - f(2, 2, -1) + f(0, 2, -1) + f(2, 0, -1)
                - f(0, 0, -1)),
        eighth
            * (f(1, 1, 2) - f(-1, 1, 2) - f(1, -1, 2) + f(-1, -1, 2) - f(1, 1, 0) + f(-1, 1, 0) + f(1, -1, 0)
                - f(-1, -1, 0)),
        eighth
            * (f(2, 1, 2) - f(0, 1, 2) - f(2, -1, 2) + f(0, -1, 2) - f(2, 1, 0) + f(0, 1, 0) + f(2, -1, 0)
                - f(0, -1, 0)),
        eighth
            * (f(1, 2, 2) - f(-1, 2, 2) - f(1, 0, 2) + f(-1, 0, 2) - f(1, 2, 0) + f(-1, 2, 0) + f(1, 0, 0)
                - f(-1, 0, 0)),
        eighth
            * (f(2, 2, 2) - f(0, 2, 2) - f(2, 0, 2) + f(0, 0, 2) - f(2, 2, 0) + f(0, 2, 0) + f(2, 0, 0) - f(0, 0, 0)),
    ];
    let mut out = [S::Value::zero_val(); 64];
    for (i, row) in TRICUBIC_A.iter().enumerate() {
        let a = |j: usize| S::Value::from_double(row[j] as f64) * x[j];
        let mut acc = S::Value::zero_val();
        for j in (0..64).step_by(4) {
            acc += a(j) + a(j + 1) + a(j + 2) + a(j + 3);
        }
        out[i] = acc;
    }
    out
}

/// Evaluate the tricubic polynomial with coefficients `c` at fractional offset `uvw`.
pub fn tricubic_sample<T: SampleReal>(uvw: Vec3d, c: &[T; 64]) -> T {
    let (mut z_pow, mut sum) = (T::from_double(1.0), T::zero_val());
    let mut n = 0;
    for _ in 0..4 {
        let mut y_pow = T::from_double(1.0);
        for _ in 0..4 {
            let x = uvw[0];
            let poly =
                c[n].to_double() + x * (c[n + 1].to_double() + x * (c[n + 2].to_double() + x * c[n + 3].to_double()));
            sum = T::from_double(sum.to_double() + (y_pow * z_pow).to_double() * poly);
            y_pow = T::from_double(y_pow.to_double() * uvw[1]);
            n += 4;
        }
        z_pow = T::from_double(z_pow.to_double() * uvw[2]);
    }
    sum
}

/// Tricubic sampler (`SampleFromVoxels<T, 3, UseCache>`).
pub struct TricubicSampler<'s, S: VoxelSource> {
    src: &'s S,
    cache: Option<RefCell<(Coord, [S::Value; 64])>>,
}

impl<'s, S: VoxelSource> TricubicSampler<'s, S>
where
    S::Value: SampleReal,
{
    /// A sampler; `use_cache` keeps the coefficients of the last cell.
    pub fn new(src: &'s S, use_cache: bool) -> Self {
        TricubicSampler {
            src,
            cache: use_cache.then(|| RefCell::new((Coord::splat(i32::MAX), [S::Value::zero_val(); 64]))),
        }
    }
}

impl<S: VoxelSource> Sampler for TricubicSampler<'_, S>
where
    S::Value: SampleReal,
{
    type Value = S::Value;
    const ORDER: u32 = 3;
    fn sample(&self, xyz: Vec3d) -> S::Value {
        let (ijk, uvw) = floor_split(xyz);
        match &self.cache {
            None => tricubic_sample(uvw, &tricubic_stencil(self.src, ijk)),
            Some(c) => {
                let mut c = c.borrow_mut();
                if c.0 != ijk {
                    *c = (ijk, tricubic_stencil(self.src, ijk));
                }
                tricubic_sample(uvw, &c.1)
            }
        }
    }
    fn sample_voxel(&self, ijk: Coord) -> S::Value {
        self.src.get_value(ijk)
    }
}
