//! Finite difference stencils that cache the values around a voxel (port of
//! `openvdb/math/Stencils.h`).
//!
//! Every stencil wraps a [`BaseStencil`], which owns a cached [`ValueAccessor`] into a grid and
//! a buffer of the values at a fixed set of offsets from the current center voxel. The buffer
//! layout (the order of the offsets) matches OpenVDB's, so `get_value(pos)` returns the same
//! element as the C++ `getValue(pos)`. Values at arbitrary offsets within the stencil are read
//! with [`StencilValues::value_at`], which also lets the finite difference functions of
//! [`super::finite_difference`] operate directly on stencils.
//!
//! The specialised stencils dereference to [`BaseStencil`] and add their own operators
//! (gradients, Laplacians, curvature, zero-crossing tests, ...).

use super::finite_difference::{godunovs_norm_sqrd, weno5, StencilValues};
use super::operators::MapJacobian;
use super::{Coord, Vec3, Vec3d};
use crate::grid::Grid;
use crate::tree::{Internal2, ValueAccessor};
use crate::types::{ArithValue, RealValue, Value};
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

/// The stencil layouts defined by OpenVDB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilKind {
    /// Center plus the six face neighbors (`SevenPointStencil`).
    SevenPoint,
    /// The eight corners of the cell `[i, i+1] x [j, j+1] x [k, k+1]` (`BoxStencil`).
    Box,
    /// The 19-point second-order dense stencil (`SecondOrderDenseStencil`).
    SecondOrderDense,
    /// Center plus two neighbors along each axis direction (`ThirteenPointStencil`).
    ThirteenPoint,
    /// The 61-point fourth-order dense stencil (`FourthOrderDenseStencil`).
    FourthOrderDense,
    /// Center plus three neighbors along each axis direction (`NineteenPointStencil`).
    NineteenPoint,
    /// The 127-point sixth-order dense stencil (`SixthOrderDenseStencil`).
    SixthOrderDense,
    /// The seven-point stencil with `GradStencil`'s layout.
    Grad,
    /// The 19-point stencil with `WenoStencil`'s layout.
    Weno,
    /// The 19-point stencil with `CurvatureStencil`'s layout.
    Curvature,
    /// A dense cube of the given half-width (`DenseStencil`).
    Dense(i32),
}

const SEVEN_PT: [[i32; 3]; 7] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]];
const BOX_PT: [[i32; 3]; 8] = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]];
const DENSE_PT: [[i32; 3]; 19] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-1, -1, 0],
    [0, -1, -1],
    [-1, 0, -1],
    [1, -1, 0],
    [0, 1, -1],
    [-1, 0, 1],
    [-1, 1, 0],
    [0, -1, 1],
    [1, 0, -1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
];
const THIRTEEN_PT: [[i32; 3]; 13] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [-2, 0, 0],
    [0, -2, 0],
    [0, 0, -2],
];
const NINETEEN_PT: [[i32; 3]; 19] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [-2, 0, 0],
    [0, -2, 0],
    [0, 0, -2],
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, 3],
    [-3, 0, 0],
    [0, -3, 0],
    [0, 0, -3],
];
const GRAD_PT: [[i32; 3]; 7] = [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];
const WENO_PT: [[i32; 3]; 19] = [
    [0, 0, 0],
    [-3, 0, 0],
    [-2, 0, 0],
    [-1, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0],
    [0, -3, 0],
    [0, -2, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 2, 0],
    [0, 3, 0],
    [0, 0, -3],
    [0, 0, -2],
    [0, 0, -1],
    [0, 0, 1],
    [0, 0, 2],
    [0, 0, 3],
];
const CURV_PT: [[i32; 3]; 19] = [
    [0, 0, 0],
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 1, 0],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [1, 0, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
    [0, 1, 1],
];

/// Layout of the 4th/6th-order dense stencils: the three axis-aligned planes through the
/// center, in OpenVDB's order.
fn plane_stencil(r: i32) -> Vec<[i32; 3]> {
    let mut v = vec![[0, 0, 0]];
    for j in (-r..=r).rev() {
        for i in -r..=r {
            if i != 0 || j != 0 {
                v.push([i, j, 0]);
            }
        }
    }
    for k in (-r..=r).rev().filter(|&k| k != 0) {
        for i in -r..=r {
            v.push([i, 0, k]);
        }
    }
    for k in (-r..=r).rev().filter(|&k| k != 0) {
        for j in (-r..=r).filter(|&j| j != 0) {
            v.push([0, j, k]);
        }
    }
    v
}

fn dense_cube(r: i32) -> Vec<[i32; 3]> {
    let mut v = Vec::with_capacity(((2 * r + 1) as usize).pow(3));
    for i in -r..=r {
        for j in -r..=r {
            for k in -r..=r {
                v.push([i, j, k]);
            }
        }
    }
    v
}

impl StencilKind {
    /// The offsets of the stencil points, in buffer order.
    pub fn offsets(&self) -> Cow<'static, [[i32; 3]]> {
        match *self {
            StencilKind::SevenPoint => Cow::Borrowed(&SEVEN_PT),
            StencilKind::Box => Cow::Borrowed(&BOX_PT),
            StencilKind::SecondOrderDense => Cow::Borrowed(&DENSE_PT),
            StencilKind::ThirteenPoint => Cow::Borrowed(&THIRTEEN_PT),
            StencilKind::FourthOrderDense => Cow::Owned(plane_stencil(2)),
            StencilKind::NineteenPoint => Cow::Borrowed(&NINETEEN_PT),
            StencilKind::SixthOrderDense => Cow::Owned(plane_stencil(3)),
            StencilKind::Grad => Cow::Borrowed(&GRAD_PT),
            StencilKind::Weno => Cow::Borrowed(&WENO_PT),
            StencilKind::Curvature => Cow::Borrowed(&CURV_PT),
            StencilKind::Dense(r) => {
                assert!(r > 0, "DenseStencil requires a positive half-width");
                Cow::Owned(dense_cube(r))
            }
        }
    }
}

const NO_ENTRY: u16 = u16::MAX;

/// The common part of all stencils (`math::BaseStencil`): a cached accessor and a buffer of
/// the values around the current center voxel.
pub struct BaseStencil<'a, T: Value> {
    grid: &'a Grid<T>,
    acc: ValueAccessor<'a, Internal2<T>>,
    offsets: Cow<'static, [[i32; 3]]>,
    lut: Vec<u16>,
    radius: i32,
    side: i32,
    center_pos: usize,
    values: Vec<T>,
    center: Coord,
}

impl<'a, T: Value> Clone for BaseStencil<'a, T> {
    fn clone(&self) -> Self {
        BaseStencil {
            grid: self.grid,
            acc: ValueAccessor::new(self.grid.tree()),
            offsets: self.offsets.clone(),
            lut: self.lut.clone(),
            radius: self.radius,
            side: self.side,
            center_pos: self.center_pos,
            values: self.values.clone(),
            center: self.center,
        }
    }
}

impl<'a, T: Value> BaseStencil<'a, T> {
    /// A stencil of the given layout over `grid`.
    pub fn new(grid: &'a Grid<T>, kind: StencilKind) -> Self {
        let offsets = kind.offsets();
        let radius = offsets.iter().flat_map(|o| o.iter().map(|c| c.abs())).max().unwrap_or(0);
        let side = 2 * radius + 1;
        let mut lut = vec![NO_ENTRY; (side * side * side) as usize];
        let mut center_pos = 0;
        for (n, o) in offsets.iter().enumerate() {
            let idx = (((o[0] + radius) * side + (o[1] + radius)) * side + (o[2] + radius)) as usize;
            lut[idx] = n as u16;
            if *o == [0, 0, 0] {
                center_pos = n;
            }
        }
        let size = offsets.len();
        BaseStencil {
            grid,
            acc: ValueAccessor::new(grid.tree()),
            offsets,
            lut,
            radius,
            side,
            center_pos,
            values: vec![*grid.background(); size],
            center: Coord::max_coord(),
        }
    }

    /// Fill the buffer with the values around `ijk`.
    #[inline]
    pub fn move_to(&mut self, ijk: &Coord) {
        self.center = *ijk;
        for (v, o) in self.values.iter_mut().zip(self.offsets.iter()) {
            *v = self.acc.get_value(&ijk.offset_by(o[0], o[1], o[2]));
        }
    }

    /// Fill the buffer around `ijk`, using the already known `center_value` for the center.
    #[inline]
    pub fn move_to_with_center(&mut self, ijk: &Coord, center_value: T) {
        self.center = *ijk;
        let cp = self.center_pos;
        for (n, (v, o)) in self.values.iter_mut().zip(self.offsets.iter()).enumerate() {
            *v = if n == cp { center_value } else { self.acc.get_value(&ijk.offset_by(o[0], o[1], o[2])) };
        }
    }

    /// Move to the voxel containing the fractional index position `xyz` (floor), but only
    /// refill the buffer if that voxel differs from the current center.
    #[inline]
    pub fn move_to_pos(&mut self, xyz: Vec3d) {
        let ijk = Coord::floor(xyz);
        if ijk != self.center {
            self.move_to(&ijk);
        }
    }

    /// The value at linear buffer position `pos` (OpenVDB's `getValue(pos)`).
    #[inline]
    pub fn get_value(&self, pos: usize) -> T {
        self.values[pos]
    }
    /// The buffer position of the offset `(i, j, k)`, if it belongs to the stencil.
    #[inline]
    pub fn pos(&self, i: i32, j: i32, k: i32) -> Option<usize> {
        let r = self.radius;
        if i.abs() > r || j.abs() > r || k.abs() > r {
            return None;
        }
        let e = self.lut[(((i + r) * self.side + (j + r)) * self.side + (k + r)) as usize];
        (e != NO_ENTRY).then_some(e as usize)
    }
    /// The value at offset `(i, j, k)` from the center.
    ///
    /// # Panics
    /// If the offset is not part of the stencil.
    #[inline]
    pub fn value(&self, i: i32, j: i32, k: i32) -> T {
        match self.pos(i, j, k) {
            Some(p) => self.values[p],
            None => panic!("offset ({i}, {j}, {k}) is not part of this stencil"),
        }
    }
    /// Overwrite the cached value at offset `(i, j, k)`.
    #[inline]
    pub fn set_value(&mut self, i: i32, j: i32, k: i32, v: T) {
        let p = self.pos(i, j, k).expect("offset is not part of this stencil");
        self.values[p] = v;
    }
    /// Number of points in the stencil.
    #[inline]
    pub fn size(&self) -> usize {
        self.values.len()
    }
    /// The buffered values, in OpenVDB's layout order.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }
    /// The offsets of the stencil points, in buffer order.
    pub fn offsets(&self) -> &[[i32; 3]] {
        &self.offsets
    }
    /// The center coordinate.
    #[inline]
    pub fn center_coord(&self) -> Coord {
        self.center
    }
    /// The value at the center.
    #[inline]
    pub fn center_value(&self) -> T {
        self.values[self.center_pos]
    }
    /// The grid the stencil reads from.
    pub fn grid(&self) -> &'a Grid<T> {
        self.grid
    }
    /// The stencil's accessor.
    pub fn accessor(&self) -> &ValueAccessor<'a, Internal2<T>> {
        &self.acc
    }
}

impl<T: Value + PartialOrd> BaseStencil<'_, T> {
    /// The median of the buffered values (for even sizes, the lower median, like OpenVDB's
    /// `nth_element` at `(size - 1) / 2`).
    pub fn median(&self) -> T {
        let mut tmp = self.values.clone();
        let mid = (tmp.len() - 1) >> 1;
        tmp.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        tmp[mid]
    }
    /// The smallest buffered value.
    pub fn min(&self) -> T {
        self.values.iter().copied().fold(self.values[0], |a, b| if b < a { b } else { a })
    }
    /// The largest buffered value.
    pub fn max(&self) -> T {
        self.values.iter().copied().fold(self.values[0], |a, b| if b > a { b } else { a })
    }
    /// True if the center value and any face neighbor lie on different sides of `iso`
    /// (requires the face neighbors to be part of the stencil).
    pub fn intersects(&self, iso: T) -> bool {
        self.intersection_mask(iso) != 0
    }
    /// Six-bit mask of iso-surface crossings between the center and its face neighbors, in the
    /// order (-x, +x, -y, +y, -z, +z) (`intersectionMask`).
    pub fn intersection_mask(&self, iso: T) -> u8 {
        let less = self.center_value() < iso;
        let nbrs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)];
        let mut mask = 0u8;
        for (bit, &(i, j, k)) in nbrs.iter().enumerate() {
            if less ^ (self.value(i, j, k) < iso) {
                mask |= 1 << bit;
            }
        }
        mask
    }
}

impl<T: ArithValue> BaseStencil<'_, T> {
    /// The mean of the buffered values.
    pub fn mean(&self) -> T {
        let sum = self.values.iter().copied().fold(<T as Value>::zero(), |a, b| a + b);
        sum.scale(1.0 / self.values.len() as f64)
    }
}

impl<T: Value> StencilValues for BaseStencil<'_, T> {
    type Value = T;
    #[inline]
    fn value_at(&self, i: i32, j: i32, k: i32) -> T {
        self.value(i, j, k)
    }
    #[inline]
    fn center_value(&self) -> T {
        BaseStencil::center_value(self)
    }
    #[inline]
    fn center_coord(&self) -> Coord {
        self.center
    }
}

macro_rules! simple_stencil {
    ($(#[$doc:meta])* $name:ident, $kind:expr) => {
        $(#[$doc])*
        #[derive(Clone)]
        pub struct $name<'a, T: Value>(BaseStencil<'a, T>);
        impl<'a, T: Value> $name<'a, T> {
            /// A stencil over `grid`.
            pub fn new(grid: &'a Grid<T>) -> Self {
                $name(BaseStencil::new(grid, $kind))
            }
        }
        impl<'a, T: Value> Deref for $name<'a, T> {
            type Target = BaseStencil<'a, T>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl<'a, T: Value> DerefMut for $name<'a, T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
        impl<T: Value> StencilValues for $name<'_, T> {
            type Value = T;
            #[inline]
            fn value_at(&self, i: i32, j: i32, k: i32) -> T {
                self.0.value(i, j, k)
            }
            #[inline]
            fn center_value(&self) -> T {
                self.0.center_value()
            }
            #[inline]
            fn center_coord(&self) -> Coord {
                self.0.center_coord()
            }
        }
    };
}

simple_stencil!(
    /// Center plus its six face neighbors (`math::SevenPointStencil`).
    SevenPointStencil,
    StencilKind::SevenPoint
);
simple_stencil!(
    /// The eight corners of a voxel cell, used for trilinear interpolation (`math::BoxStencil`).
    BoxStencil,
    StencilKind::Box
);
simple_stencil!(
    /// 19-point stencil for second-order mixed derivatives (`math::SecondOrderDenseStencil`).
    SecondOrderDenseStencil,
    StencilKind::SecondOrderDense
);
simple_stencil!(
    /// Center plus two neighbors in each axis direction (`math::ThirteenPointStencil`).
    ThirteenPointStencil,
    StencilKind::ThirteenPoint
);
simple_stencil!(
    /// 61-point stencil for fourth-order mixed derivatives (`math::FourthOrderDenseStencil`).
    FourthOrderDenseStencil,
    StencilKind::FourthOrderDense
);
simple_stencil!(
    /// Center plus three neighbors in each axis direction (`math::NineteenPointStencil`).
    NineteenPointStencil,
    StencilKind::NineteenPoint
);
simple_stencil!(
    /// 127-point stencil for sixth-order mixed derivatives (`math::SixthOrderDenseStencil`).
    SixthOrderDenseStencil,
    StencilKind::SixthOrderDense
);

impl<T: RealValue> BoxStencil<'_, T> {
    /// True if any corner lies on a different side of `iso` than the first corner.
    pub fn intersects_box(&self, iso: T) -> bool {
        let less = self.values()[0] < iso;
        self.values()[1..].iter().any(|&v| less ^ (v < iso))
    }
    /// Trilinear interpolation at the index-space position `xyz`, which must lie in the cell
    /// whose lower corner is the current center (see [`BaseStencil::move_to_pos`]).
    pub fn interpolation(&self, xyz: Vec3d) -> T {
        let c = self.center_coord();
        let u = T::from_f64(xyz[0] - c[0] as f64);
        let v = T::from_f64(xyz[1] - c[1] as f64);
        let w = T::from_f64(xyz[2] - c[2] as f64);
        let g = |i, j, k| self.value(i, j, k);
        let mut vv = g(0, 0, 0);
        let mut a = vv + (g(0, 0, 1) - vv) * w;
        vv = g(0, 1, 0);
        let mut b = vv + (g(0, 1, 1) - vv) * w;
        let cc = a + (b - a) * v;
        vv = g(1, 0, 0);
        a = vv + (g(1, 0, 1) - vv) * w;
        vv = g(1, 1, 0);
        b = vv + (g(1, 1, 1) - vv) * w;
        let d = a + (b - a) * v;
        cc + (d - cc) * u
    }
    /// World-space gradient of the trilinear interpolant at index-space position `xyz`.
    pub fn gradient(&self, xyz: Vec3d) -> Vec3<T> {
        let c = self.center_coord();
        let u = T::from_f64(xyz[0] - c[0] as f64);
        let v = T::from_f64(xyz[1] - c[1] as f64);
        let w = T::from_f64(xyz[2] - c[2] as f64);
        let g = |i, j, k| self.value(i, j, k);
        let mut d =
            [g(0, 0, 1) - g(0, 0, 0), g(0, 1, 1) - g(0, 1, 0), g(1, 0, 1) - g(1, 0, 0), g(1, 1, 1) - g(1, 1, 0)];
        let mut a = d[0] + (d[1] - d[0]) * v;
        let mut b = d[2] + (d[3] - d[2]) * v;
        let mut grad = Vec3::new(<T as Value>::zero(), <T as Value>::zero(), a + (b - a) * u);
        d[0] = g(0, 0, 0) + d[0] * w;
        d[1] = g(0, 1, 0) + d[1] * w;
        d[2] = g(1, 0, 0) + d[2] * w;
        d[3] = g(1, 1, 0) + d[3] * w;
        a = d[0] + (d[1] - d[0]) * v;
        b = d[2] + (d[3] - d[2]) * v;
        grad[0] = b - a;
        a = d[1] - d[0];
        b = d[3] - d[2];
        grad[1] = a + (b - a) * u;
        let jac = MapJacobian::new(self.grid().transform());
        jac.apply_ijt(grad.map(|x| x.to_f64()), xyz).map(T::from_f64)
    }
}

/// Seven-point stencil supporting central-difference and first-order upwind gradients,
/// Laplacian, closest-point transform and zero-crossing test (`math::GradStencil`).
#[derive(Clone)]
pub struct GradStencil<'a, T: Value> {
    base: BaseStencil<'a, T>,
    inv_2dx: T,
    inv_dx2: T,
}

impl<'a, T: Value> Deref for GradStencil<'a, T> {
    type Target = BaseStencil<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
impl<'a, T: Value> DerefMut for GradStencil<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
impl<T: Value> StencilValues for GradStencil<'_, T> {
    type Value = T;
    #[inline]
    fn value_at(&self, i: i32, j: i32, k: i32) -> T {
        self.base.value(i, j, k)
    }
    #[inline]
    fn center_value(&self) -> T {
        self.base.center_value()
    }
    #[inline]
    fn center_coord(&self) -> Coord {
        self.base.center_coord()
    }
}

impl<'a, T: RealValue> GradStencil<'a, T> {
    /// A stencil using the grid's voxel size (x component).
    pub fn new(grid: &'a Grid<T>) -> Self {
        Self::with_dx(grid, grid.voxel_size()[0])
    }
    /// A stencil with an explicit voxel size.
    pub fn with_dx(grid: &'a Grid<T>, dx: f64) -> Self {
        let inv_2dx = T::from_f64(0.5 / dx);
        let inv_dx2 = T::from_f64(4.0 * inv_2dx.to_f64() * inv_2dx.to_f64());
        GradStencil { base: BaseStencil::new(grid, StencilKind::Grad), inv_2dx, inv_dx2 }
    }
    /// `|∇φ|²` using first-order upwinding and Godunov's scheme.
    pub fn norm_sq_grad(&self) -> T {
        let v = self.base.values();
        self.inv_dx2
            * godunovs_norm_sqrd(
                v[0] > <T as Value>::zero(),
                v[0] - v[1],
                v[2] - v[0],
                v[0] - v[3],
                v[4] - v[0],
                v[0] - v[5],
                v[6] - v[0],
            )
    }
    /// Central-difference gradient (world units).
    pub fn gradient(&self) -> Vec3<T> {
        let v = self.base.values();
        Vec3::new(v[2] - v[1], v[4] - v[3], v[6] - v[5]) * self.inv_2dx
    }
    /// First-order upwind gradient with respect to the direction `vel`.
    pub fn gradient_upwind(&self, vel: Vec3<T>) -> Vec3<T> {
        let v = self.base.values();
        let z = <T as Value>::zero();
        Vec3::new(
            if vel[0] > z { v[0] - v[1] } else { v[2] - v[0] },
            if vel[1] > z { v[0] - v[3] } else { v[4] - v[0] },
            if vel[2] > z { v[0] - v[5] } else { v[6] - v[0] },
        ) * (T::from_f64(2.0) * self.inv_2dx)
    }
    /// Central-difference Laplacian.
    pub fn laplacian(&self) -> T {
        let v = self.base.values();
        self.inv_dx2 * (v[1] + v[2] + v[3] + v[4] + v[5] + v[6] - T::from_f64(6.0) * v[0])
    }
    /// True if the center value's sign differs from that of any face neighbor.
    pub fn zero_crossing(&self) -> bool {
        zero_crossing(self.base.values(), [1, 2, 3, 4, 5, 6])
    }
    /// Closest point on the zero level set, in index space (assumes a level set in world
    /// units with a uniform-scale transform).
    pub fn cpt(&self) -> Vec3<T> {
        let ijk = self.base.center_coord();
        let v = self.base.values();
        let d = T::from_f64(v[0].to_f64() * 0.5 * self.inv_dx2.to_f64());
        Vec3::new(
            T::from_f64(ijk[0] as f64) - d * (v[2] - v[1]),
            T::from_f64(ijk[1] as f64) - d * (v[4] - v[3]),
            T::from_f64(ijk[2] as f64) - d * (v[6] - v[5]),
        )
    }
}

#[inline]
fn zero_crossing<T: RealValue>(v: &[T], n: [usize; 6]) -> bool {
    let z = <T as Value>::zero();
    if v[0] > z {
        n.iter().any(|&i| v[i] < z)
    } else {
        n.iter().any(|&i| v[i] > z)
    }
}

/// 19-point stencil supporting fifth-order WENO upwinding, central differencing, Laplacian
/// and zero-crossing test (`math::WenoStencil`).
#[derive(Clone)]
pub struct WenoStencil<'a, T: Value> {
    base: BaseStencil<'a, T>,
    inv_2dx: T,
    inv_dx2: T,
    dx2: f32,
}

impl<'a, T: Value> Deref for WenoStencil<'a, T> {
    type Target = BaseStencil<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
impl<'a, T: Value> DerefMut for WenoStencil<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
impl<T: Value> StencilValues for WenoStencil<'_, T> {
    type Value = T;
    #[inline]
    fn value_at(&self, i: i32, j: i32, k: i32) -> T {
        self.base.value(i, j, k)
    }
    #[inline]
    fn center_value(&self) -> T {
        self.base.center_value()
    }
    #[inline]
    fn center_coord(&self) -> Coord {
        self.base.center_coord()
    }
}

impl<'a, T: RealValue> WenoStencil<'a, T> {
    /// A stencil using the grid's voxel size (x component).
    pub fn new(grid: &'a Grid<T>) -> Self {
        Self::with_dx(grid, grid.voxel_size()[0])
    }
    /// A stencil with an explicit voxel size.
    pub fn with_dx(grid: &'a Grid<T>, dx: f64) -> Self {
        let dx2 = T::from_f64(dx * dx);
        WenoStencil {
            base: BaseStencil::new(grid, StencilKind::Weno),
            inv_2dx: T::from_f64(0.5 / dx),
            inv_dx2: T::from_f64(1.0 / dx2.to_f64()),
            dx2: dx2.to_f64() as f32,
        }
    }
    #[inline]
    fn upwind(&self, axis: usize) -> (T, T) {
        weno_upwind(self.base.values(), axis, self.dx2)
    }
    /// `|∇φ|²` using WENO5 upwinding and Godunov's scheme, relative to `iso`.
    pub fn norm_sq_grad(&self, iso: T) -> T {
        weno_norm_sq_grad_values(self.base.values(), self.dx2, self.inv_dx2, iso)
    }
    /// Fifth-order upwind gradient with respect to the direction `vel`.
    pub fn gradient_upwind(&self, vel: Vec3<T>) -> Vec3<T> {
        let z = <T as Value>::zero();
        let mut g = Vec3::splat(z);
        for a in 0..3 {
            let (m, p) = self.upwind(a);
            g[a] = if vel[a] > z { m } else { p };
        }
        g * (T::from_f64(2.0) * self.inv_2dx)
    }
    /// Central-difference gradient (world units).
    pub fn gradient(&self) -> Vec3<T> {
        let v = self.base.values();
        Vec3::new(v[4] - v[3], v[10] - v[9], v[16] - v[15]) * self.inv_2dx
    }
    /// Central-difference Laplacian.
    pub fn laplacian(&self) -> T {
        let v = self.base.values();
        self.inv_dx2 * (v[3] + v[4] + v[9] + v[10] + v[15] + v[16] - T::from_f64(6.0) * v[0])
    }
    /// True if the center value's sign differs from that of any face neighbor.
    pub fn zero_crossing(&self) -> bool {
        weno_zero_crossing(self.base.values())
    }
}

/// The 19 values of a [`WenoStencil`] (center, then offsets -3, -2, -1, 1, 2, 3 along x, y
/// and z) read from any value source, e.g. an [`AccessorAt`](super::finite_difference::AccessorAt).
pub fn weno_values<S: StencilValues + ?Sized>(s: &S) -> [S::Value; 19] {
    WENO_PT.map(|o| s.value_at(o[0], o[1], o[2]))
}

/// WENO5 backward and forward one-sided derivatives along `axis` from the 19 values of a
/// [`WenoStencil`] (index-space differences).
#[inline]
fn weno_upwind<T: RealValue>(v: &[T], axis: usize, dx2: f32) -> (T, T) {
    let o = 6 * axis;
    let (m3, m2, m1, p1, p2, p3) = (v[o + 1], v[o + 2], v[o + 3], v[o + 4], v[o + 5], v[o + 6]);
    let c = v[0];
    let dm = weno5(m2 - m3, m1 - m2, c - m1, p1 - c, p2 - p1, dx2);
    let dp = weno5(p3 - p2, p2 - p1, p1 - c, c - m1, m1 - m2, dx2);
    (dm, dp)
}

/// `WenoStencil::normSqGrad` from the 19 stencil values (see [`weno_values`]), with `dx2` the
/// squared voxel size and `inv_dx2` its inverse.
pub fn weno_norm_sq_grad_values<T: RealValue>(v: &[T], dx2: f32, inv_dx2: T, iso: T) -> T {
    let (xm, xp) = weno_upwind(v, 0, dx2);
    let (ym, yp) = weno_upwind(v, 1, dx2);
    let (zm, zp) = weno_upwind(v, 2, dx2);
    T::from_f64(
        inv_dx2.to_f64()
            * godunovs_norm_sqrd(
                v[0] > iso,
                xm.to_f64(),
                xp.to_f64(),
                ym.to_f64(),
                yp.to_f64(),
                zm.to_f64(),
                zp.to_f64(),
            ),
    )
}

/// `WenoStencil::zeroCrossing` from the 19 stencil values: true if the center value's sign
/// differs from that of any face neighbor.
pub fn weno_zero_crossing<T: RealValue>(v: &[T]) -> bool {
    zero_crossing(v, [3, 4, 9, 10, 15, 16])
}

/// 19-point stencil supporting mean, Gaussian and principal curvatures
/// (`math::CurvatureStencil`).
#[derive(Clone)]
pub struct CurvatureStencil<'a, T: Value> {
    base: BaseStencil<'a, T>,
    inv_2dx: T,
    inv_dx2: T,
}

impl<'a, T: Value> Deref for CurvatureStencil<'a, T> {
    type Target = BaseStencil<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
impl<'a, T: Value> DerefMut for CurvatureStencil<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
impl<T: Value> StencilValues for CurvatureStencil<'_, T> {
    type Value = T;
    #[inline]
    fn value_at(&self, i: i32, j: i32, k: i32) -> T {
        self.base.value(i, j, k)
    }
    #[inline]
    fn center_value(&self) -> T {
        self.base.center_value()
    }
    #[inline]
    fn center_coord(&self) -> Coord {
        self.base.center_coord()
    }
}

/// Unscaled first and second central differences around the stencil center.
struct CurvTerms {
    dx: f64,
    dy: f64,
    dz: f64,
    dxx: f64,
    dyy: f64,
    dzz: f64,
    dxy: f64,
    dxz: f64,
    dyz: f64,
}

impl<'a, T: RealValue> CurvatureStencil<'a, T> {
    /// A stencil using the grid's voxel size (x component).
    pub fn new(grid: &'a Grid<T>) -> Self {
        Self::with_dx(grid, grid.voxel_size()[0])
    }
    /// A stencil with an explicit voxel size.
    pub fn with_dx(grid: &'a Grid<T>, dx: f64) -> Self {
        let inv_2dx = T::from_f64(0.5 / dx);
        let inv_dx2 = T::from_f64(4.0 * inv_2dx.to_f64() * inv_2dx.to_f64());
        CurvatureStencil { base: BaseStencil::new(grid, StencilKind::Curvature), inv_2dx, inv_dx2 }
    }

    fn terms(&self) -> Option<(CurvTerms, f64)> {
        let v = self.base.values();
        let two = T::from_f64(2.0);
        let dx = 0.5 * (v[2] - v[1]).to_f64();
        let dy = 0.5 * (v[4] - v[3]).to_f64();
        let dz = 0.5 * (v[6] - v[5]).to_f64();
        let ng2 = dx * dx + dy * dy + dz * dz;
        // OpenVDB compares against `math::Tolerance<Real>` (double precision).
        if ng2 <= crate::math::TOLERANCE_F64 {
            return None;
        }
        let t = CurvTerms {
            dx,
            dy,
            dz,
            dxx: (v[2] - two * v[0] + v[1]).to_f64(),
            dyy: (v[4] - two * v[0] + v[3]).to_f64(),
            dzz: (v[6] - two * v[0] + v[5]).to_f64(),
            dxy: 0.25 * (v[10] - v[8] + v[7] - v[9]).to_f64(),
            dxz: 0.25 * (v[14] - v[12] + v[11] - v[13]).to_f64(),
            dyz: 0.25 * (v[18] - v[16] + v[15] - v[17]).to_f64(),
        };
        Some((t, ng2.sqrt()))
    }

    fn alpha_mean(t: &CurvTerms) -> f64 {
        let (dx2, dy2, dz2) = (t.dx * t.dx, t.dy * t.dy, t.dz * t.dz);
        dx2 * (t.dyy + t.dzz) + dy2 * (t.dxx + t.dzz) + dz2 * (t.dxx + t.dyy)
            - 2.0 * (t.dx * (t.dy * t.dxy + t.dz * t.dxz) + t.dy * t.dz * t.dyz)
    }

    fn alpha_gauss(t: &CurvTerms) -> f64 {
        let (dx2, dy2, dz2) = (t.dx * t.dx, t.dy * t.dy, t.dz * t.dz);
        dx2 * (t.dyy * t.dzz - t.dyz * t.dyz)
            + dy2 * (t.dxx * t.dzz - t.dxz * t.dxz)
            + dz2 * (t.dxx * t.dyy - t.dxy * t.dxy)
            + 2.0
                * (t.dy * t.dz * (t.dxy * t.dxz - t.dyz * t.dxx)
                    + t.dx * t.dz * (t.dxy * t.dyz - t.dxz * t.dyy)
                    + t.dx * t.dy * (t.dxz * t.dyz - t.dxy * t.dzz))
    }

    /// Mean curvature at the center.
    pub fn mean_curvature(&self) -> T {
        match self.terms() {
            Some((t, ng)) => T::from_f64(Self::alpha_mean(&t) * self.inv_2dx.to_f64() / (ng * ng * ng)),
            None => <T as Value>::zero(),
        }
    }
    /// Gaussian curvature at the center.
    pub fn gaussian_curvature(&self) -> T {
        match self.terms() {
            Some((t, ng)) => T::from_f64(Self::alpha_gauss(&t) * self.inv_dx2.to_f64() / ng.powi(4)),
            None => <T as Value>::zero(),
        }
    }
    /// Mean and Gaussian curvature at the center.
    pub fn curvatures(&self) -> (T, T) {
        match self.terms() {
            Some((t, ng)) => (
                T::from_f64(Self::alpha_mean(&t) * self.inv_2dx.to_f64() / (ng * ng * ng)),
                T::from_f64(Self::alpha_gauss(&t) * self.inv_dx2.to_f64() / ng.powi(4)),
            ),
            None => (<T as Value>::zero(), <T as Value>::zero()),
        }
    }
    /// Mean curvature times `|∇φ|` (useful for mean-curvature flow).
    pub fn mean_curvature_norm_grad(&self) -> T {
        match self.terms() {
            Some((t, ng)) => T::from_f64(Self::alpha_mean(&t) * self.inv_dx2.to_f64() / (2.0 * ng * ng)),
            None => <T as Value>::zero(),
        }
    }
    /// Gaussian curvature times `|∇φ|`.
    pub fn gaussian_curvature_norm_grad(&self) -> T {
        match self.terms() {
            Some((t, ng)) => T::from_f64(
                2.0 * Self::alpha_gauss(&t) * self.inv_2dx.to_f64() * self.inv_dx2.to_f64() / (ng * ng * ng),
            ),
            None => <T as Value>::zero(),
        }
    }
    /// Mean and Gaussian curvature times `|∇φ|`.
    pub fn curvatures_norm_grad(&self) -> (T, T) {
        match self.terms() {
            Some((t, ng)) => (
                T::from_f64(Self::alpha_mean(&t) * self.inv_dx2.to_f64() / (2.0 * ng * ng)),
                T::from_f64(
                    2.0 * Self::alpha_gauss(&t) * self.inv_2dx.to_f64() * self.inv_dx2.to_f64() / (ng * ng * ng),
                ),
            ),
            None => (<T as Value>::zero(), <T as Value>::zero()),
        }
    }
    /// The (minimum, maximum) principal curvatures at the center.
    pub fn principal_curvatures(&self) -> (T, T) {
        match self.terms() {
            Some((t, ng)) => {
                let mean = Self::alpha_mean(&t) * self.inv_2dx.to_f64() / (ng * ng * ng);
                let tmp = (mean * mean - Self::alpha_gauss(&t) * self.inv_dx2.to_f64() / ng.powi(4)).sqrt();
                (T::from_f64(mean - tmp), T::from_f64(mean + tmp))
            }
            None => (<T as Value>::zero(), <T as Value>::zero()),
        }
    }
    /// Central-difference Laplacian.
    pub fn laplacian(&self) -> T {
        let v = self.base.values();
        self.inv_dx2 * (v[1] + v[2] + v[3] + v[4] + v[5] + v[6] - T::from_f64(6.0) * v[0])
    }
    /// Central-difference gradient (world units).
    pub fn gradient(&self) -> Vec3<T> {
        let v = self.base.values();
        Vec3::new(v[2] - v[1], v[4] - v[3], v[6] - v[5]) * self.inv_2dx
    }
}

/// A dense cube of values of a given half-width (`math::DenseStencil`), typically used for
/// median filtering. Values are stored in x-major order; the center is in the middle.
#[derive(Clone)]
pub struct DenseStencil<'a, T: Value> {
    base: BaseStencil<'a, T>,
    half_width: i32,
}

impl<'a, T: Value> DenseStencil<'a, T> {
    /// A dense stencil of the given half-width (> 0).
    pub fn new(grid: &'a Grid<T>, half_width: i32) -> Self {
        DenseStencil { base: BaseStencil::new(grid, StencilKind::Dense(half_width)), half_width }
    }
    /// The half-width.
    pub fn half_width(&self) -> i32 {
        self.half_width
    }
}

impl<'a, T: Value> Deref for DenseStencil<'a, T> {
    type Target = BaseStencil<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
impl<'a, T: Value> DerefMut for DenseStencil<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
impl<T: Value> StencilValues for DenseStencil<'_, T> {
    type Value = T;
    #[inline]
    fn value_at(&self, i: i32, j: i32, k: i32) -> T {
        self.base.value(i, j, k)
    }
    #[inline]
    fn center_value(&self) -> T {
        self.base.center_value()
    }
    #[inline]
    fn center_coord(&self) -> Coord {
        self.base.center_coord()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::finite_difference::{d1, d2_cross, DDScheme, DScheme};
    use crate::math::Transform;
    use crate::tree::FloatTree;

    fn quadratic_grid() -> Grid<f32> {
        // f(x,y,z) = x^2 + 2y^2 - z^2 + xy, voxel size 0.5
        let mut tree = FloatTree::new(0.0);
        let mut acc = tree.accessor_mut();
        for c in crate::math::CoordBBox::from_coords(-6, -6, -6, 6, 6, 6) {
            let (x, y, z) = (c[0] as f32 * 0.5, c[1] as f32 * 0.5, c[2] as f32 * 0.5);
            acc.set_value(&c, x * x + 2.0 * y * y - z * z + x * y);
        }
        drop(acc);
        Grid::from_tree(tree).with_transform(Transform::linear(0.5))
    }

    #[test]
    fn layouts() {
        assert_eq!(StencilKind::FourthOrderDense.offsets().len(), 61);
        assert_eq!(StencilKind::SixthOrderDense.offsets().len(), 127);
        assert_eq!(StencilKind::FourthOrderDense.offsets()[25], [-2, 0, 2]);
        assert_eq!(StencilKind::FourthOrderDense.offsets()[45], [0, -2, 2]);
        assert_eq!(StencilKind::SixthOrderDense.offsets()[49], [-3, 0, 3]);
        assert_eq!(StencilKind::SixthOrderDense.offsets()[126], [0, 3, -3]);
        assert_eq!(StencilKind::Dense(1).offsets().len(), 27);
    }

    #[test]
    fn grad_stencil_ops() {
        let g = quadratic_grid();
        let mut s = GradStencil::new(&g);
        s.move_to(&Coord::new(2, -2, 4)); // world (1, -1, 2)
        let grad = s.gradient();
        // df/dx = 2x + y = 1, df/dy = 4y + x = -3, df/dz = -2z = -4
        assert!((grad - Vec3::new(1.0, -3.0, -4.0)).length() < 1e-5);
        assert!((s.laplacian() - 4.0).abs() < 1e-4); // 2 + 4 - 2
        let mut d = SecondOrderDenseStencil::new(&g);
        d.move_to(&Coord::new(2, -2, 4));
        assert!((d2_cross(DDScheme::CdSecond, 0, 1, &*d) / 0.25 - 1.0).abs() < 1e-5);
        let mut n = NineteenPointStencil::new(&g);
        n.move_to(&Coord::new(0, 0, 0));
        assert!((d1(DScheme::Cd6th, 1, &*n)).abs() < 1e-6);
        assert_eq!(n.center_value(), 0.0);
    }

    #[test]
    fn box_interpolation_and_median() {
        let g = quadratic_grid();
        let mut b = BoxStencil::new(&g);
        let p = Vec3d::new(1.25, 0.5, -0.75);
        b.move_to_pos(p);
        assert_eq!(b.center_coord(), Coord::new(1, 0, -1));
        let f = |x: f64, y: f64, z: f64| (x * x + 2.0 * y * y - z * z + x * y) * 0.25;
        // Trilinear interpolation of corner values.
        let (u, v, w) = (0.25, 0.5, 0.25);
        let mut expect = 0.0;
        for (i, wu) in [(1.0, 1.0 - u), (2.0, u)] {
            for (j, wv) in [(0.0, 1.0 - v), (1.0, v)] {
                for (k, ww) in [(-1.0, 1.0 - w), (0.0, w)] {
                    expect += wu * wv * ww * f(i, j, k);
                }
            }
        }
        assert!((b.interpolation(p) as f64 - expect).abs() < 1e-5);
        let mut d = DenseStencil::new(&g, 1);
        d.move_to(&Coord::new(0, 0, 0));
        assert_eq!(d.size(), 27);
        assert_eq!(d.center_value(), 0.0);
        let mut vals: Vec<f32> = d.values().to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(d.median(), vals[13]);
        assert_eq!(d.min(), vals[0]);
        assert_eq!(d.max(), vals[26]);
    }

    #[test]
    fn curvature_of_sphere() {
        // Exact SDF of a sphere of radius 10 (voxel size 1): mean curvature 1/r.
        let mut tree = FloatTree::new(3.0);
        let mut acc = tree.accessor_mut();
        for c in crate::math::CoordBBox::from_coords(7, -3, -3, 13, 3, 3) {
            acc.set_value(&c, (c.as_vec3d().length() - 10.0) as f32);
        }
        drop(acc);
        let g = Grid::from_tree(tree);
        let mut s = CurvatureStencil::new(&g);
        s.move_to(&Coord::new(10, 0, 0));
        assert!((s.mean_curvature() - 0.1).abs() < 2e-3, "{}", s.mean_curvature());
        assert!((s.gaussian_curvature() - 0.01).abs() < 1e-3);
        let (k1, k2) = s.principal_curvatures();
        assert!((k1 - 0.1).abs() < 5e-3 && (k2 - 0.1).abs() < 5e-3);
        let mut w = WenoStencil::new(&g);
        w.move_to(&Coord::new(10, 0, 0));
        assert!((w.norm_sq_grad(0.0) - 1.0).abs() < 1e-2);
        assert!(w.zero_crossing());
        w.move_to(&Coord::new(12, 0, 0));
        assert!(!w.zero_crossing());
        assert_eq!(w.intersection_mask(0.0) != 0, w.intersects(0.0));
    }
}
