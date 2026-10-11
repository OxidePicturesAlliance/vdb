//! Samplers that reconstruct values at fractional voxel positions (port of
//! `openvdb/tools/Interpolation.h`).
//!
//! The [`Sampler`] trait describes an index-space interpolation kernel: [`PointSampler`]
//! (nearest neighbour), [`BoxSampler`] (trilinear), [`QuadraticSampler`] (triquadratic) and the
//! staggered variants for MAC-grid vector data. Samplers read through any
//! [`ValueSource`] — trees, grids or (faster) cached accessors.
//!
//! [`GridSampler`] couples a sampler with a transform to sample in world space,
//! [`DualGridSampler`] samples one grid at the voxel centers of another, and [`AlphaMask`]
//! derives smooth blending weights from a mask grid.
//!
//! ```
//! use vdb::prelude::*;
//! use vdb::tree::FloatTree;
//! use vdb::tools::interpolation::{BoxSampler, Sampler};
//! let mut tree = FloatTree::new(0.0);
//! tree.set_value(&Coord::new(1, 0, 0), 2.0);
//! let v = BoxSampler::sample(&tree, Vec3d::new(0.5, 0.0, 0.0));
//! assert_eq!(v, 1.0);
//! ```

use crate::grid::Grid;
use crate::math::{Coord, Transform, Vec3, Vec3d};
use crate::tree::{Internal2, ValueAccessor, ValueSource};
use crate::types::{ArithValue, RealValue, Value};
use std::marker::PhantomData;

/// An index-space interpolation kernel for values of type `T`.
pub trait Sampler<T: Value> {
    /// Short name of the sampler.
    const NAME: &'static str;
    /// Radius of the kernel in voxels, not including the center voxel.
    const RADIUS: i32;
    /// Whether downscaling by more than a factor of two should use successive halvings.
    const MIPMAP: bool;
    /// Whether sampling exactly at a grid point returns that grid point's value.
    const CONSISTENT: bool;
    /// Whether the sampler is designed for staggered (MAC) vector grids.
    const STAGGERED: bool;
    /// Polynomial order of the kernel.
    const ORDER: usize;

    /// Reconstruct the value at the fractional index-space position `xyz`.
    fn sample<S: ValueSource<Value = T> + ?Sized>(src: &S, xyz: Vec3d) -> T;
    /// Reconstruct the value at `xyz`; also return true if any of the contributing values is
    /// active.
    fn probe_sample<S: ValueSource<Value = T> + ?Sized>(src: &S, xyz: Vec3d) -> (T, bool);
}

#[inline]
fn floor_coord(v: Vec3d) -> Coord {
    Coord::floor(v)
}

/// Round half away from zero, like C's `round()`.
#[inline]
fn round_coord(v: Vec3d) -> Coord {
    Coord::new(v[0].round() as i32, v[1].round() as i32, v[2].round() as i32)
}

/// Nearest-neighbour sampling (`tools::PointSampler`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PointSampler;

impl<T: Value> Sampler<T> for PointSampler {
    const NAME: &'static str = "point";
    const RADIUS: i32 = 0;
    const MIPMAP: bool = false;
    const CONSISTENT: bool = true;
    const STAGGERED: bool = false;
    const ORDER: usize = 0;
    #[inline]
    fn sample<S: ValueSource<Value = T> + ?Sized>(src: &S, xyz: Vec3d) -> T {
        src.get_value(&round_coord(xyz))
    }
    #[inline]
    fn probe_sample<S: ValueSource<Value = T> + ?Sized>(src: &S, xyz: Vec3d) -> (T, bool) {
        src.probe_value(&round_coord(xyz))
    }
}

/// Trilinear interpolation (`tools::BoxSampler`).
#[derive(Clone, Copy, Debug, Default)]
pub struct BoxSampler;

impl BoxSampler {
    /// Read the eight corner values of the cell whose lower corner is `ijk`, indexed
    /// `[dx][dy][dz]`.
    #[inline]
    pub fn get_values<S: ValueSource + ?Sized>(src: &S, ijk: Coord) -> [[[S::Value; 2]; 2]; 2] {
        src.get_cell_values(&ijk)
    }
    /// Like [`get_values`](Self::get_values), also returning whether any corner is active.
    #[inline]
    pub fn probe_values<S: ValueSource + ?Sized>(src: &S, ijk: Coord) -> ([[[S::Value; 2]; 2]; 2], bool) {
        src.probe_cell_values(&ijk)
    }
    /// The smallest and largest of the eight corner values.
    pub fn extrema<T: Value + PartialOrd>(data: &[[[T; 2]; 2]; 2]) -> (T, T) {
        let mut lo = data[0][0][0];
        let mut hi = lo;
        for v in data.iter().flatten().flatten() {
            if *v < lo {
                lo = *v;
            }
            if *v > hi {
                hi = *v;
            }
        }
        (lo, hi)
    }
    /// Trilinear interpolation of the corner values at the unit-cell coordinates `uvw`.
    #[inline]
    pub fn trilinear_interpolation<T: ArithValue>(data: &[[[T; 2]; 2]; 2], uvw: Vec3d) -> T {
        #[inline(always)]
        fn lerp<T: ArithValue>(a: T, b: T, w: f64) -> T {
            a + (b - a).scale(w)
        }
        lerp(
            lerp(lerp(data[0][0][0], data[0][0][1], uvw[2]), lerp(data[0][1][0], data[0][1][1], uvw[2]), uvw[1]),
            lerp(lerp(data[1][0][0], data[1][0][1], uvw[2]), lerp(data[1][1][0], data[1][1][1], uvw[2]), uvw[1]),
            uvw[0],
        )
    }
}

impl<T: ArithValue> Sampler<T> for BoxSampler {
    const NAME: &'static str = "box";
    const RADIUS: i32 = 1;
    const MIPMAP: bool = true;
    const CONSISTENT: bool = true;
    const STAGGERED: bool = false;
    const ORDER: usize = 1;
    #[inline]
    fn sample<S: ValueSource<Value = T> + ?Sized>(src: &S, xyz: Vec3d) -> T {
        let ijk = floor_coord(xyz);
        let data = Self::get_values(src, ijk);
        Self::trilinear_interpolation(&data, xyz - ijk.as_vec3d())
    }
    #[inline]
    fn probe_sample<S: ValueSource<Value = T> + ?Sized>(src: &S, xyz: Vec3d) -> (T, bool) {
        let ijk = floor_coord(xyz);
        let (data, active) = Self::probe_values(src, ijk);
        (Self::trilinear_interpolation(&data, xyz - ijk.as_vec3d()), active)
    }
}

/// Triquadratic interpolation (`tools::QuadraticSampler`).
#[derive(Clone, Copy, Debug, Default)]
pub struct QuadraticSampler;

impl QuadraticSampler {
    /// Triquadratic interpolation of a 3³ block of values (indexed `[dx][dy][dz]`, centered on
    /// `[1][1][1]`) at the unit-cell coordinates `uvw`.
    pub fn triquadratic_interpolation<T: ArithValue>(data: &[[[T; 3]; 3]; 3], uvw: Vec3d) -> T {
        #[inline(always)]
        fn interp<T: ArithValue>(v: &[T; 3], w: f64) -> T {
            let a = (v[0] + v[2]).scale(0.5) - v[1];
            let b = (v[2] - v[0]).scale(0.5);
            (a.scale(w) + b).scale(w) + v[1]
        }
        let mut vx = [T::zero(); 3];
        for (dx, out) in vx.iter_mut().enumerate() {
            let mut vy = [T::zero(); 3];
            for (dy, o) in vy.iter_mut().enumerate() {
                *o = interp(&data[dx][dy], uvw[2]);
            }
            *out = interp(&vy, uvw[1]);
        }
        interp(&vx, uvw[0])
    }
}

impl<T: ArithValue> Sampler<T> for QuadraticSampler {
    const NAME: &'static str = "quadratic";
    const RADIUS: i32 = 1;
    const MIPMAP: bool = true;
    const CONSISTENT: bool = false;
    const STAGGERED: bool = false;
    const ORDER: usize = 2;
    fn sample<S: ValueSource<Value = T> + ?Sized>(src: &S, xyz: Vec3d) -> T {
        Self::probe_sample(src, xyz).0
    }
    fn probe_sample<S: ValueSource<Value = T> + ?Sized>(src: &S, xyz: Vec3d) -> (T, bool) {
        let ijk = floor_coord(xyz);
        let lo = ijk.offset(-1);
        let mut data = [[[T::zero(); 3]; 3]; 3];
        let mut active = false;
        for (dx, px) in data.iter_mut().enumerate() {
            for (dy, py) in px.iter_mut().enumerate() {
                for (dz, v) in py.iter_mut().enumerate() {
                    let (val, on) = src.probe_value(&lo.offset_by(dx as i32, dy as i32, dz as i32));
                    *v = val;
                    active |= on;
                }
            }
        }
        (Self::triquadratic_interpolation(&data, xyz - ijk.as_vec3d()), active)
    }
}

macro_rules! staggered {
    ($(#[$doc:meta])* $name:ident, $base:ident, $nm:expr, $r:expr, $mip:expr, $ord:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;
        impl<T> Sampler<Vec3<T>> for $name
        where
            Vec3<T>: Value,
            $base: Sampler<Vec3<T>>,
            T: Copy,
        {
            const NAME: &'static str = $nm;
            const RADIUS: i32 = $r;
            const MIPMAP: bool = $mip;
            const CONSISTENT: bool = false;
            const STAGGERED: bool = true;
            const ORDER: usize = $ord;
            fn sample<S: ValueSource<Value = Vec3<T>> + ?Sized>(src: &S, xyz: Vec3d) -> Vec3<T> {
                let x = $base::sample(src, xyz + Vec3d::new(0.5, 0.0, 0.0));
                let y = $base::sample(src, xyz + Vec3d::new(0.0, 0.5, 0.0));
                let z = $base::sample(src, xyz + Vec3d::new(0.0, 0.0, 0.5));
                Vec3::new(x[0], y[1], z[2])
            }
            fn probe_sample<S: ValueSource<Value = Vec3<T>> + ?Sized>(src: &S, xyz: Vec3d) -> (Vec3<T>, bool) {
                let (x, ax) = $base::probe_sample(src, xyz + Vec3d::new(0.5, 0.0, 0.0));
                let (y, ay) = $base::probe_sample(src, xyz + Vec3d::new(0.0, 0.5, 0.0));
                let (z, az) = $base::probe_sample(src, xyz + Vec3d::new(0.0, 0.0, 0.5));
                (Vec3::new(x[0], y[1], z[2]), ax || ay || az)
            }
        }
    };
}

staggered!(
    /// Nearest-neighbour sampling of staggered (MAC) vector grids (`tools::StaggeredPointSampler`).
    StaggeredPointSampler,
    PointSampler,
    "point",
    0,
    false,
    0
);
staggered!(
    /// Trilinear sampling of staggered (MAC) vector grids (`tools::StaggeredBoxSampler`).
    StaggeredBoxSampler,
    BoxSampler,
    "box",
    1,
    true,
    1
);
staggered!(
    /// Triquadratic sampling of staggered (MAC) vector grids (`tools::StaggeredQuadraticSampler`).
    StaggeredQuadraticSampler,
    QuadraticSampler,
    "quadratic",
    1,
    true,
    2
);

/// Samples a tree or accessor in index or world space (`tools::GridSampler`).
///
/// When built from an accessor the sampler caches nodes and is not thread-safe; create one
/// per thread.
pub struct GridSampler<'t, S, Sm> {
    source: S,
    transform: &'t Transform,
    _sampler: PhantomData<Sm>,
}

impl<'t, S: ValueSource, Sm: Sampler<S::Value>> GridSampler<'t, S, Sm> {
    /// Sample `source` (a tree, grid or accessor) whose index space is mapped to world space by
    /// `transform`.
    pub fn new(source: S, transform: &'t Transform) -> Self {
        GridSampler { source, transform, _sampler: PhantomData }
    }
    /// The transform.
    pub fn transform(&self) -> &Transform {
        self.transform
    }
    /// The value source.
    pub fn source(&self) -> &S {
        &self.source
    }
    /// The value of the voxel at `ijk`.
    #[inline]
    pub fn is_sample_coord(&self, ijk: &Coord) -> S::Value {
        self.source.get_value(ijk)
    }
    /// The interpolated value at the index-space position `xyz`.
    #[inline]
    pub fn is_sample(&self, xyz: Vec3d) -> S::Value {
        Sm::sample(&self.source, xyz)
    }
    /// The interpolated value at the world-space position `xyz`.
    #[inline]
    pub fn ws_sample(&self, xyz: Vec3d) -> S::Value {
        Sm::sample(&self.source, self.transform.world_to_index(xyz))
    }
    /// Interpolated value at index position `(x, y, z)` (`sampleVoxel`).
    #[inline]
    pub fn sample_voxel(&self, x: f64, y: f64, z: f64) -> S::Value {
        self.is_sample(Vec3d::new(x, y, z))
    }
}

impl<'a, T: Value, Sm: Sampler<T>> GridSampler<'a, ValueAccessor<'a, Internal2<T>>, Sm> {
    /// A sampler reading `grid` through a cached accessor.
    pub fn from_grid(grid: &'a Grid<T>) -> Self {
        GridSampler::new(grid.accessor(), grid.transform())
    }
}

/// Samples a source grid at the voxel centers of a target grid with a different transform
/// (`tools::DualGridSampler`).
pub struct DualGridSampler<'t, S, Sm> {
    source: S,
    source_xform: &'t Transform,
    target_xform: &'t Transform,
    aligned: bool,
    _sampler: PhantomData<Sm>,
}

impl<'t, S: ValueSource, Sm: Sampler<S::Value>> DualGridSampler<'t, S, Sm> {
    /// Sample `source` (with transform `source_xform`) at target index coordinates mapped by
    /// `target_xform`.
    pub fn new(source: S, source_xform: &'t Transform, target_xform: &'t Transform) -> Self {
        let aligned = source_xform == target_xform;
        DualGridSampler { source, source_xform, target_xform, aligned, _sampler: PhantomData }
    }
    /// The source value at target voxel `ijk`.
    #[inline]
    pub fn sample(&self, ijk: &Coord) -> S::Value {
        if self.aligned {
            return self.source.get_value(ijk);
        }
        let world = self.target_xform.index_to_world(ijk.as_vec3d());
        Sm::sample(&self.source, self.source_xform.world_to_index(world))
    }
    /// True if both transforms are identical (no interpolation needed).
    pub fn is_aligned(&self) -> bool {
        self.aligned
    }
}

/// Smoothly maps the values of a mask grid to blending weights (`tools::AlphaMask`).
pub struct AlphaMask<'a, M: Value> {
    sampler: DualGridSampler<'a, ValueAccessor<'a, Internal2<M>>, BoxSampler>,
    min: f64,
    inv_norm: f64,
    invert: bool,
}

impl<'a, M: RealValue + ArithValue> AlphaMask<'a, M> {
    /// Weights for voxels of a grid with transform `target`, from `mask` values remapped from
    /// `[min, max]` to `[0, 1]` with a smooth step. If `invert`, the weights are swapped.
    pub fn new(target: &'a Transform, mask: &'a Grid<M>, min: f64, max: f64, invert: bool) -> Self {
        assert!(min < max, "AlphaMask: invalid mask range");
        AlphaMask {
            sampler: DualGridSampler::new(mask.accessor(), mask.transform(), target),
            min,
            inv_norm: 1.0 / (max - min),
            invert,
        }
    }
    /// The weights `(a, b)` with `a + b = 1` at voxel `xyz`, or `None` if `a` is zero.
    #[inline]
    pub fn weights(&self, xyz: &Coord) -> Option<(f64, f64)> {
        let mut a = smooth_unit_step((self.sampler.sample(xyz).to_f64() - self.min) * self.inv_norm);
        let mut b = 1.0 - a;
        if self.invert {
            std::mem::swap(&mut a, &mut b);
        }
        (a > 0.0).then_some((a, b))
    }
}

/// OpenVDB's `math::SmoothUnitStep`: 0 for `x <= 0`, 1 for `x >= 1`, `(3 - 2x)x²` in between.
pub use crate::math::math_util::smooth_unit_step;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::CoordBBox;
    use crate::tools::test_util::assert_close;
    use crate::tree::{FloatTree, Vec3STree};

    fn test_tree() -> FloatTree {
        // Values f(i,j,k) = sin(0.3 i) + 0.2 j*k - 0.1 k, active for i <= 2.
        let mut tree = FloatTree::new(0.0);
        for c in CoordBBox::from_coords(-4, -4, -4, 4, 4, 4) {
            let v = (0.3 * c[0] as f64).sin() + 0.2 * (c[1] * c[2]) as f64 - 0.1 * c[2] as f64;
            if c[0] <= 2 {
                tree.set_value_on(&c, v as f32);
            } else {
                tree.set_value_off(&c, v as f32);
            }
        }
        tree
    }

    /// Reference values from the C++ library (`scratchpad/cppref/interp.cpp`).
    #[test]
    fn matches_openvdb_reference() {
        let tree = test_tree();
        let acc = tree.accessor();
        let pts = [
            Vec3d::new(0.3, -1.7, 2.25),
            Vec3d::new(-2.5, 0.5, -0.5),
            Vec3d::new(2.6, 3.1, -3.9),
            Vec3d::new(1.0, 2.0, 3.0),
        ];
        let expect: [[f64; 3]; 4] = [
            [-1.0, -0.901343942, -0.901343942],
            [-0.883326888, -0.673984706, -0.682731271],
            [-1.21667314, -1.33214688, -1.41150868],
            [1.19552016, 1.19552016, 1.19552016],
        ];
        let actives = [[true, true, true], [true, true, true], [false, true, true], [true, true, true]];
        for ((p, e), act) in pts.iter().zip(expect.iter()).zip(actives.iter()) {
            let (pv, pa) = PointSampler::probe_sample(&acc, *p);
            let (bv, ba) = BoxSampler::probe_sample(&acc, *p);
            let (qv, qa) = QuadraticSampler::probe_sample(&tree, *p);
            assert_close(pv as f64, e[0], 1e-6);
            assert_close(bv as f64, e[1], 1e-6);
            assert_close(qv as f64, e[2], 1e-5);
            assert_eq!([pa, ba, qa], *act);
        }
    }

    #[test]
    fn staggered_and_grid_samplers() {
        let mut vt = Vec3STree::new(Vec3::zero());
        for c in CoordBBox::from_coords(-2, -2, -2, 3, 3, 3) {
            vt.set_value(&c, Vec3::new(c[0] as f32, 10.0 * c[1] as f32, 100.0 * c[2] as f32));
        }
        // Linear fields are reproduced exactly; staggering shifts each component by half a voxel.
        let p = Vec3d::new(0.25, 0.5, -0.75);
        let v = StaggeredBoxSampler::sample(&vt, p);
        assert!((v - Vec3::new(0.75, 10.0, -25.0)).length() < 1e-4, "{v:?}");
        let v = BoxSampler::sample(&vt, p);
        assert!((v - Vec3::new(0.25, 5.0, -75.0)).length() < 1e-4);
        let sp = StaggeredPointSampler::sample(&vt, p);
        assert_eq!(sp, Vec3::new(1.0, 10.0, -0.0));

        let grid = Grid::from_tree(test_tree()).with_transform(Transform::linear(0.5));
        let gs: GridSampler<_, BoxSampler> = GridSampler::from_grid(&grid);
        let w = Vec3d::new(0.15, -0.85, 1.125);
        assert_close(gs.ws_sample(w) as f64, gs.is_sample(Vec3d::new(0.3, -1.7, 2.25)) as f64, 1e-6);
        assert_close(gs.is_sample(Vec3d::new(0.3, -1.7, 2.25)) as f64, -0.901343942, 1e-6);

        let target = Transform::linear(0.25);
        let dual: DualGridSampler<_, BoxSampler> = DualGridSampler::new(grid.accessor(), grid.transform(), &target);
        assert!(!dual.is_aligned());
        assert_close(
            dual.sample(&Coord::new(2, 2, 2)) as f64,
            grid.tree().get_value(&Coord::new(1, 1, 1)) as f64,
            1e-6,
        );
    }

    #[test]
    fn alpha_mask() {
        let mut mask = Grid::<f32>::new(0.0);
        mask.set_value(&Coord::new(0, 0, 0), 0.5);
        mask.set_value(&Coord::new(1, 0, 0), 1.0);
        let xf = Transform::identity();
        let am = AlphaMask::new(&xf, &mask, 0.0, 1.0, false);
        assert_eq!(am.weights(&Coord::new(5, 5, 5)), None);
        assert_eq!(am.weights(&Coord::new(0, 0, 0)), Some((0.5, 0.5)));
        assert_eq!(am.weights(&Coord::new(1, 0, 0)), Some((1.0, 0.0)));
        let inv = AlphaMask::new(&xf, &mask, 0.0, 1.0, true);
        assert_eq!(inv.weights(&Coord::new(5, 5, 5)), Some((1.0, 0.0)));
    }
}
