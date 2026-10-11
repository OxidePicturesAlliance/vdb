//! Finite difference schemes (port of `openvdb/math/FiniteDifference.h`).
//!
//! The C++ header defines `D1<Scheme>`, `D1Vec<Scheme>` and `D2<Scheme>` template structs whose
//! `inX/inY/inZ` methods read values either from an accessor at a coordinate or from a stencil.
//! Here both cases are unified by the [`StencilValues`] trait, which yields the value at an
//! integer offset from a center point:
//!
//! * [`AccessorAt`] adapts any [`ValueSource`] (tree, grid, accessor) and a center coordinate;
//! * the stencils of [`super::stencils`] implement it directly from their cached buffers.
//!
//! The scheme is selected at run time through the [`DScheme`] and [`DDScheme`] enums.
//!
//! ```
//! use vdb::math::finite_difference::{d1, DScheme, AccessorAt};
//! use vdb::prelude::*;
//! use vdb::tree::FloatTree;
//! let mut tree = FloatTree::new(0.0);
//! for i in -3..=3 {
//!     tree.set_value(&Coord::new(i, 0, 0), (i * i) as f32);
//! }
//! let at = AccessorAt::new(&tree, Coord::new(1, 0, 0));
//! assert_eq!(d1(DScheme::Cd2nd, 0, &at), 2.0);
//! ```

use super::{Coord, Vec3};
use crate::tree::ValueSource;
use crate::types::RealValue;
use num_traits::Float;
use std::fmt;

/// First-derivative finite difference schemes (`math::DScheme`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DScheme {
    /// Center difference, 2nd order, not divided by two (`CD_2NDT`).
    Cd2ndT,
    /// Center difference, 2nd order (`CD_2ND`).
    Cd2nd,
    /// Center difference, 4th order (`CD_4TH`).
    Cd4th,
    /// Center difference, 6th order (`CD_6TH`).
    Cd6th,
    /// Forward difference, 1st order (`FD_1ST`).
    Fd1st,
    /// Forward difference, 2nd order (`FD_2ND`).
    Fd2nd,
    /// Forward difference, 3rd order (`FD_3RD`).
    Fd3rd,
    /// Backward difference, 1st order (`BD_1ST`).
    Bd1st,
    /// Backward difference, 2nd order (`BD_2ND`).
    Bd2nd,
    /// Backward difference, 3rd order (`BD_3RD`).
    Bd3rd,
    /// Forward WENO5 difference (`FD_WENO5`).
    FdWeno5,
    /// Backward WENO5 difference (`BD_WENO5`).
    BdWeno5,
    /// Forward Hamilton-Jacobi WENO5 difference (`FD_HJWENO5`).
    FdHjWeno5,
    /// Backward Hamilton-Jacobi WENO5 difference (`BD_HJWENO5`).
    BdHjWeno5,
}

impl DScheme {
    /// All schemes, in OpenVDB's enum order.
    pub const ALL: [DScheme; 14] = [
        DScheme::Cd2ndT,
        DScheme::Cd2nd,
        DScheme::Cd4th,
        DScheme::Cd6th,
        DScheme::Fd1st,
        DScheme::Fd2nd,
        DScheme::Fd3rd,
        DScheme::Bd1st,
        DScheme::Bd2nd,
        DScheme::Bd3rd,
        DScheme::FdWeno5,
        DScheme::BdWeno5,
        DScheme::FdHjWeno5,
        DScheme::BdHjWeno5,
    ];
    /// OpenVDB's short name (`dsSchemeToString`).
    pub fn as_str(&self) -> &'static str {
        match self {
            DScheme::Cd2ndT => "cd_2ndt",
            DScheme::Cd2nd => "cd_2nd",
            DScheme::Cd4th => "cd_4th",
            DScheme::Cd6th => "cd_6th",
            DScheme::Fd1st => "fd_1st",
            DScheme::Fd2nd => "fd_2nd",
            DScheme::Fd3rd => "fd_3rd",
            DScheme::Bd1st => "bd_1st",
            DScheme::Bd2nd => "bd_2nd",
            DScheme::Bd3rd => "bd_3rd",
            DScheme::FdWeno5 => "fd_weno5",
            DScheme::BdWeno5 => "bd_weno5",
            DScheme::FdHjWeno5 => "fd_hjweno5",
            DScheme::BdHjWeno5 => "bd_hjweno5",
        }
    }
    /// Parse a short name (case-insensitive, surrounding whitespace ignored).
    pub fn from_name(s: &str) -> Option<DScheme> {
        let s = s.trim().to_lowercase();
        Self::ALL.iter().copied().find(|d| d.as_str() == s)
    }
    /// OpenVDB's UI name (`dsSchemeToMenuName`).
    pub fn menu_name(&self) -> &'static str {
        match self {
            DScheme::Cd2ndT => "Twice 2nd-order center difference",
            DScheme::Cd2nd => "2nd-order center difference",
            DScheme::Cd4th => "4th-order center difference",
            DScheme::Cd6th => "6th-order center difference",
            DScheme::Fd1st => "1st-order forward difference",
            DScheme::Fd2nd => "2nd-order forward difference",
            DScheme::Fd3rd => "3rd-order forward difference",
            DScheme::Bd1st => "1st-order backward difference",
            DScheme::Bd2nd => "2nd-order backward difference",
            DScheme::Bd3rd => "3rd-order backward difference",
            DScheme::FdWeno5 => "5th-order WENO forward difference",
            DScheme::BdWeno5 => "5th-order WENO backward difference",
            DScheme::FdHjWeno5 => "5th-order HJ-WENO forward difference",
            DScheme::BdHjWeno5 => "5th-order HJ-WENO backward difference",
        }
    }
    /// Number of voxels the scheme reads on each side of the center.
    pub fn radius(&self) -> i32 {
        match self {
            DScheme::Cd2ndT | DScheme::Cd2nd | DScheme::Fd1st | DScheme::Bd1st => 1,
            DScheme::Cd4th | DScheme::Fd2nd | DScheme::Bd2nd => 2,
            _ => 3,
        }
    }
}

impl fmt::Display for DScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Second-derivative finite difference schemes (`math::DDScheme`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DDScheme {
    /// Center difference, 2nd order (`CD_SECOND`).
    CdSecond,
    /// Center difference, 4th order (`CD_FOURTH`).
    CdFourth,
    /// Center difference, 6th order (`CD_SIXTH`).
    CdSixth,
}

impl DDScheme {
    /// Number of voxels the scheme reads on each side of the center.
    pub fn radius(&self) -> i32 {
        match self {
            DDScheme::CdSecond => 1,
            DDScheme::CdFourth => 2,
            DDScheme::CdSixth => 3,
        }
    }
}

/// Upwind (biased) gradient schemes (`math::BiasedGradientScheme`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiasedGradientScheme {
    /// Uses `FD_1ST` and `BD_1ST`.
    FirstBias,
    /// Uses `FD_2ND` and `BD_2ND`.
    SecondBias,
    /// Uses `FD_3RD` and `BD_3RD`.
    ThirdBias,
    /// Uses `FD_WENO5` and `BD_WENO5`.
    Weno5Bias,
    /// Uses `FD_HJWENO5` and `BD_HJWENO5`.
    HjWeno5Bias,
}

impl BiasedGradientScheme {
    /// All schemes, in OpenVDB's enum order.
    pub const ALL: [BiasedGradientScheme; 5] = [
        BiasedGradientScheme::FirstBias,
        BiasedGradientScheme::SecondBias,
        BiasedGradientScheme::ThirdBias,
        BiasedGradientScheme::Weno5Bias,
        BiasedGradientScheme::HjWeno5Bias,
    ];
    /// The forward difference scheme (`BIAS_SCHEME<bgs>::FD`).
    pub fn forward(&self) -> DScheme {
        match self {
            BiasedGradientScheme::FirstBias => DScheme::Fd1st,
            BiasedGradientScheme::SecondBias => DScheme::Fd2nd,
            BiasedGradientScheme::ThirdBias => DScheme::Fd3rd,
            BiasedGradientScheme::Weno5Bias => DScheme::FdWeno5,
            BiasedGradientScheme::HjWeno5Bias => DScheme::FdHjWeno5,
        }
    }
    /// The backward difference scheme (`BIAS_SCHEME<bgs>::BD`).
    pub fn backward(&self) -> DScheme {
        match self {
            BiasedGradientScheme::FirstBias => DScheme::Bd1st,
            BiasedGradientScheme::SecondBias => DScheme::Bd2nd,
            BiasedGradientScheme::ThirdBias => DScheme::Bd3rd,
            BiasedGradientScheme::Weno5Bias => DScheme::BdWeno5,
            BiasedGradientScheme::HjWeno5Bias => DScheme::BdHjWeno5,
        }
    }
    /// The stencil required by the scheme (`BIAS_SCHEME<bgs>::ISStencil`).
    pub fn stencil_kind(&self) -> super::stencils::StencilKind {
        use super::stencils::StencilKind;
        match self {
            BiasedGradientScheme::FirstBias => StencilKind::SevenPoint,
            BiasedGradientScheme::SecondBias => StencilKind::ThirteenPoint,
            _ => StencilKind::NineteenPoint,
        }
    }
    /// OpenVDB's short name (`biasedGradientSchemeToString`).
    pub fn as_str(&self) -> &'static str {
        match self {
            BiasedGradientScheme::FirstBias => "first_bias",
            BiasedGradientScheme::SecondBias => "second_bias",
            BiasedGradientScheme::ThirdBias => "third_bias",
            BiasedGradientScheme::Weno5Bias => "weno5_bias",
            BiasedGradientScheme::HjWeno5Bias => "hjweno5_bias",
        }
    }
    /// Parse a short name.
    pub fn from_name(s: &str) -> Option<Self> {
        let s = s.trim().to_lowercase();
        Self::ALL.iter().copied().find(|d| d.as_str() == s)
    }
    /// OpenVDB's UI name.
    pub fn menu_name(&self) -> &'static str {
        match self {
            BiasedGradientScheme::FirstBias => "1st-order biased gradient",
            BiasedGradientScheme::SecondBias => "2nd-order biased gradient",
            BiasedGradientScheme::ThirdBias => "3rd-order biased gradient",
            BiasedGradientScheme::Weno5Bias => "5th-order WENO biased gradient",
            BiasedGradientScheme::HjWeno5Bias => "5th-order HJ-WENO biased gradient",
        }
    }
}

/// Explicit time integration schemes (`math::TemporalIntegrationScheme`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TemporalIntegrationScheme {
    /// Forward Euler.
    TvdRk1,
    /// 2nd-order TVD Runge-Kutta.
    TvdRk2,
    /// 3rd-order TVD Runge-Kutta.
    TvdRk3,
}

impl TemporalIntegrationScheme {
    /// All schemes.
    pub const ALL: [TemporalIntegrationScheme; 3] =
        [TemporalIntegrationScheme::TvdRk1, TemporalIntegrationScheme::TvdRk2, TemporalIntegrationScheme::TvdRk3];
    /// OpenVDB's short name.
    pub fn as_str(&self) -> &'static str {
        match self {
            TemporalIntegrationScheme::TvdRk1 => "tvd_rk1",
            TemporalIntegrationScheme::TvdRk2 => "tvd_rk2",
            TemporalIntegrationScheme::TvdRk3 => "tvd_rk3",
        }
    }
    /// Parse a short name.
    pub fn from_name(s: &str) -> Option<Self> {
        let s = s.trim().to_lowercase();
        Self::ALL.iter().copied().find(|d| d.as_str() == s)
    }
    /// OpenVDB's UI name.
    pub fn menu_name(&self) -> &'static str {
        match self {
            TemporalIntegrationScheme::TvdRk1 => "Forward Euler",
            TemporalIntegrationScheme::TvdRk2 => "2nd-order Runge-Kutta",
            TemporalIntegrationScheme::TvdRk3 => "3rd-order Runge-Kutta",
        }
    }
    /// Number of auxiliary buffers needed (one for RK1/RK2, two for RK3).
    pub fn aux_buffers(&self) -> usize {
        match self {
            TemporalIntegrationScheme::TvdRk3 => 2,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Value sources

/// Values at integer offsets from a center point: implemented by stencils and by
/// [`AccessorAt`].
pub trait StencilValues {
    /// The value type.
    type Value: Copy;
    /// The value at offset `(i, j, k)` from the center.
    fn value_at(&self, i: i32, j: i32, k: i32) -> Self::Value;
    /// The value at the center.
    #[inline]
    fn center_value(&self) -> Self::Value {
        self.value_at(0, 0, 0)
    }
    /// The index-space coordinate of the center (used by map-aware operators).
    #[inline]
    fn center_coord(&self) -> Coord {
        Coord::origin()
    }
}

/// Adapts a [`ValueSource`] and a center coordinate to [`StencilValues`] (the "random access"
/// variants of the C++ operators).
pub struct AccessorAt<'a, S: ?Sized> {
    /// The value source.
    pub source: &'a S,
    /// The center coordinate.
    pub ijk: Coord,
}

impl<'a, S: ?Sized> AccessorAt<'a, S> {
    /// Center `source` at `ijk`.
    #[inline]
    pub fn new(source: &'a S, ijk: Coord) -> Self {
        AccessorAt { source, ijk }
    }
}

impl<S: ValueSource + ?Sized> StencilValues for AccessorAt<'_, S> {
    type Value = S::Value;
    #[inline]
    fn value_at(&self, i: i32, j: i32, k: i32) -> S::Value {
        self.source.get_value(&self.ijk.offset_by(i, j, k))
    }
    #[inline]
    fn center_coord(&self) -> Coord {
        self.ijk
    }
}

/// Views one component of a vector-valued [`StencilValues`] (used by `D1Vec`).
pub struct Component<'a, S: ?Sized> {
    source: &'a S,
    n: usize,
}

impl<'a, S: ?Sized> Component<'a, S> {
    /// View component `n` (0, 1 or 2) of `source`.
    #[inline]
    pub fn new(source: &'a S, n: usize) -> Self {
        Component { source, n }
    }
}

impl<S, T> StencilValues for Component<'_, S>
where
    S: StencilValues<Value = Vec3<T>> + ?Sized,
    T: Copy,
{
    type Value = T;
    #[inline]
    fn value_at(&self, i: i32, j: i32, k: i32) -> T {
        self.source.value_at(i, j, k).0[self.n]
    }
    #[inline]
    fn center_coord(&self) -> Coord {
        self.source.center_coord()
    }
}

#[inline]
fn axis_offset(axis: usize, o: i32) -> (i32, i32, i32) {
    match axis {
        0 => (o, 0, 0),
        1 => (0, o, 0),
        _ => (0, 0, o),
    }
}

#[inline]
fn plane_offset(a: usize, b: usize, oa: i32, ob: i32) -> (i32, i32, i32) {
    let mut o = [0i32; 3];
    o[a] = oa;
    o[b] = ob;
    (o[0], o[1], o[2])
}

#[inline(always)]
fn c<T: RealValue>(v: f64) -> T {
    T::from_f64(v)
}

// ---------------------------------------------------------------------------------------------
// WENO and Godunov

/// Fifth-order WENO interpolation of the derivative from five consecutive one-sided differences
/// (`math::WENO5`). `scale2` is the squared length scale used to scale the epsilon (0.01 by
/// default in OpenVDB).
#[inline]
pub fn weno5<T: RealValue>(v1: T, v2: T, v3: T, v4: T, v5: T, scale2: f32) -> T {
    const C: f64 = 13.0 / 12.0;
    let eps = 1.0e-6 * scale2 as f64;
    let two = c::<T>(2.0);
    let four = c::<T>(4.0);
    let sq = |x: f64| x * x;
    let a1 = 0.1
        / sq(C * sq((v1 - two * v2 + v3).to_f64()) + 0.25 * sq((v1 - four * v2).to_f64() + 3.0 * v3.to_f64()) + eps);
    let a2 = 0.6 / sq(C * sq((v2 - two * v3 + v4).to_f64()) + 0.25 * sq((v2 - v4).to_f64()) + eps);
    let a3 = 0.3
        / sq(C * sq((v3 - two * v4 + v5).to_f64())
            + 0.25 * sq(3.0 * v3.to_f64() - (four * v4).to_f64() + v5.to_f64())
            + eps);
    let (v1, v2, v3, v4, v5) = (v1.to_f64(), v2.to_f64(), v3.to_f64(), v4.to_f64(), v5.to_f64());
    let num =
        a1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + a2 * (5.0 * v3 - v2 + 2.0 * v4) + a3 * (2.0 * v3 + 5.0 * v4 - v5);
    T::from_f64(T::from_f64(num).to_f64() / (6.0 * (a1 + a2 + a3)))
}

/// [`weno5`] with OpenVDB's default `scale2 = 0.01`.
#[inline]
pub fn weno5_default<T: RealValue>(v1: T, v2: T, v3: T, v4: T, v5: T) -> T {
    weno5(v1, v2, v3, v4, v5, 0.01)
}

/// Godunov's upwind approximation of `|∇φ|²` from one-sided differences
/// (`math::GodunovsNormSqrd`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn godunovs_norm_sqrd<T: Float>(is_outside: bool, dxm: T, dxp: T, dym: T, dyp: T, dzm: T, dzp: T) -> T {
    let z = T::zero();
    let p2 = |x: T| x * x;
    if is_outside {
        p2(dxm.max(z)).max(p2(dxp.min(z))) + p2(dym.max(z)).max(p2(dyp.min(z))) + p2(dzm.max(z)).max(p2(dzp.min(z)))
    } else {
        p2(dxm.min(z)).max(p2(dxp.max(z))) + p2(dym.min(z)).max(p2(dyp.max(z))) + p2(dzm.min(z)).max(p2(dzp.max(z)))
    }
}

/// [`godunovs_norm_sqrd`] taking the backward (`m`) and forward (`p`) gradients as vectors.
#[inline]
pub fn godunovs_norm_sqrd_vec<T: Float>(is_outside: bool, m: Vec3<T>, p: Vec3<T>) -> T {
    godunovs_norm_sqrd(is_outside, m[0], p[0], m[1], p[1], m[2], p[2])
}

// ---------------------------------------------------------------------------------------------
// First derivatives

#[inline]
fn fd2<T: RealValue>(xp2: T, xp1: T, xp0: T) -> T {
    c::<T>(2.0) * xp1 - (c::<T>(0.5) * xp2 + c::<T>(1.5) * xp0)
}

#[inline]
fn fd3<T: RealValue>(xp3: T, xp2: T, xp1: T, xp0: T) -> T {
    T::from_f64(xp3.to_f64() / 3.0 - 1.5 * xp2.to_f64() + 3.0 * xp1.to_f64() - 11.0 * xp0.to_f64() / 6.0)
}

#[inline]
fn fd_weno5<T: RealValue>(xp3: T, xp2: T, xp1: T, xp0: T, xm1: T, xm2: T) -> T {
    weno5_default(xp3, xp2, xp1, xp0, xm1) - weno5_default(xp2, xp1, xp0, xm1, xm2)
}

#[inline]
fn fd_hjweno5<T: RealValue>(xp3: T, xp2: T, xp1: T, xp0: T, xm1: T, xm2: T) -> T {
    weno5_default(xp3 - xp2, xp2 - xp1, xp1 - xp0, xp0 - xm1, xm1 - xm2)
}

/// One-dimensional first difference along a line of values `v(offset)`.
#[inline]
pub fn d1_line<T: RealValue>(scheme: DScheme, v: impl Fn(i32) -> T) -> T {
    match scheme {
        DScheme::Cd2ndT => v(1) - v(-1),
        DScheme::Cd2nd => (v(1) - v(-1)) * c::<T>(0.5),
        DScheme::Cd4th => c::<T>(2.0 / 3.0) * (v(1) - v(-1)) + c::<T>(1.0 / 12.0) * (v(-2) - v(2)),
        DScheme::Cd6th => {
            c::<T>(3.0 / 4.0) * (v(1) - v(-1)) - c::<T>(0.15) * (v(2) - v(-2)) + c::<T>(1.0 / 60.0) * (v(3) - v(-3))
        }
        DScheme::Fd1st => v(1) - v(0),
        DScheme::Fd2nd => fd2(v(2), v(1), v(0)),
        DScheme::Fd3rd => fd3(v(3), v(2), v(1), v(0)),
        DScheme::Bd1st => -(v(-1) - v(0)),
        DScheme::Bd2nd => -fd2(v(-2), v(-1), v(0)),
        DScheme::Bd3rd => -fd3(v(-3), v(-2), v(-1), v(0)),
        DScheme::FdWeno5 => fd_weno5(v(3), v(2), v(1), v(0), v(-1), v(-2)),
        DScheme::BdWeno5 => -fd_weno5(v(-3), v(-2), v(-1), v(0), v(1), v(2)),
        DScheme::FdHjWeno5 => fd_hjweno5(v(3), v(2), v(1), v(0), v(-1), v(-2)),
        DScheme::BdHjWeno5 => -fd_hjweno5(v(-3), v(-2), v(-1), v(0), v(1), v(2)),
    }
}

/// First derivative along `axis` (0 = x, 1 = y, 2 = z) in index space, unscaled by the voxel
/// size (`D1<scheme>::inX/inY/inZ`).
#[inline]
pub fn d1<S>(scheme: DScheme, axis: usize, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    d1_line(scheme, |o| {
        let (i, j, k) = axis_offset(axis, o);
        s.value_at(i, j, k)
    })
}

/// `D1<scheme>::inX`.
#[inline]
pub fn d1_x<S: StencilValues + ?Sized>(scheme: DScheme, s: &S) -> S::Value
where
    S::Value: RealValue,
{
    d1(scheme, 0, s)
}
/// `D1<scheme>::inY`.
#[inline]
pub fn d1_y<S: StencilValues + ?Sized>(scheme: DScheme, s: &S) -> S::Value
where
    S::Value: RealValue,
{
    d1(scheme, 1, s)
}
/// `D1<scheme>::inZ`.
#[inline]
pub fn d1_z<S: StencilValues + ?Sized>(scheme: DScheme, s: &S) -> S::Value
where
    S::Value: RealValue,
{
    d1(scheme, 2, s)
}

/// First derivative of component `n` of a vector field along `axis` (`D1Vec<scheme>`).
#[inline]
pub fn d1_vec<S, T>(scheme: DScheme, axis: usize, s: &S, n: usize) -> T
where
    S: StencilValues<Value = Vec3<T>> + ?Sized,
    T: RealValue,
{
    d1(scheme, axis, &Component::new(s, n))
}

// ---------------------------------------------------------------------------------------------
// Second derivatives

/// Second derivative along `axis`, unscaled (`D2<scheme>::inX/inY/inZ`).
#[inline]
pub fn d2<S>(scheme: DDScheme, axis: usize, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    let v = |o: i32| {
        let (i, j, k) = axis_offset(axis, o);
        s.value_at(i, j, k)
    };
    type T<S> = <S as StencilValues>::Value;
    match scheme {
        DDScheme::CdSecond => v(1) + v(-1) - c::<T<S>>(2.0) * v(0),
        DDScheme::CdFourth => {
            c::<T<S>>(-1.0 / 12.0) * (v(2) + v(-2)) + c::<T<S>>(4.0 / 3.0) * (v(1) + v(-1)) - c::<T<S>>(2.5) * v(0)
        }
        DDScheme::CdSixth => {
            c::<T<S>>(1.0 / 90.0) * (v(3) + v(-3)) - c::<T<S>>(3.0 / 20.0) * (v(2) + v(-2))
                + c::<T<S>>(1.5) * (v(1) + v(-1))
                - c::<T<S>>(49.0 / 18.0) * v(0)
        }
    }
}

/// Mixed second derivative along axes `a` and `b`, unscaled (`D2<scheme>::inXandY` etc.).
#[inline]
pub fn d2_cross<S>(scheme: DDScheme, a: usize, b: usize, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    type T<S> = <S as StencilValues>::Value;
    let v = |oa: i32, ob: i32| {
        let (i, j, k) = plane_offset(a, b, oa, ob);
        s.value_at(i, j, k)
    };
    // (f(x+p, y+q) - f(x-p, y+q) - f(x+p, y-q) + f(x-p, y-q))
    let q = |p: i32, qq: i32| v(p, qq) - v(-p, qq) - v(p, -qq) + v(-p, -qq);
    match scheme {
        DDScheme::CdSecond => c::<T<S>>(0.25) * (v(1, 1) + v(-1, -1) - v(1, -1) - v(-1, 1)),
        DDScheme::CdFourth => {
            let tmp = |k: i32| c::<T<S>>(2.0 / 3.0) * q(1, k) - c::<T<S>>(1.0 / 12.0) * q(2, k);
            c::<T<S>>(2.0 / 3.0) * tmp(1) - c::<T<S>>(1.0 / 12.0) * tmp(2)
        }
        DDScheme::CdSixth => {
            let tmp = |k: i32| c::<T<S>>(0.75) * q(1, k) - c::<T<S>>(0.15) * q(2, k) + c::<T<S>>(1.0 / 60.0) * q(3, k);
            c::<T<S>>(0.75) * tmp(1) - c::<T<S>>(0.15) * tmp(2) + c::<T<S>>(1.0 / 60.0) * tmp(3)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A stencil that evaluates an analytic function of the offset (unit spacing).
    pub(crate) struct Analytic<F: Fn(f64, f64, f64) -> f64>(pub F);
    impl<F: Fn(f64, f64, f64) -> f64> StencilValues for Analytic<F> {
        type Value = f64;
        fn value_at(&self, i: i32, j: i32, k: i32) -> f64 {
            (self.0)(i as f64, j as f64, k as f64)
        }
    }

    #[test]
    fn exactness_on_polynomials() {
        // Cubic in x: exact for 3rd/4th/6th-order schemes; derivative at 0 is 2.
        let s = Analytic(|x: f64, _, _| 0.5 * x * x * x - x * x + 2.0 * x + 1.0);
        for sch in [DScheme::Cd4th, DScheme::Cd6th, DScheme::Fd3rd, DScheme::Bd3rd] {
            assert!((d1(sch, 0, &s) - 2.0).abs() < 1e-12, "{sch}");
        }
        let q = Analytic(|_, y: f64, _| 3.0 * y * y + y);
        for sch in [DScheme::Cd2nd, DScheme::Fd2nd, DScheme::Bd2nd, DScheme::Cd4th] {
            assert!((d1(sch, 1, &q) - 1.0).abs() < 1e-12, "{sch}");
        }
        assert!((d1(DScheme::Cd2ndT, 1, &q) - 2.0).abs() < 1e-12);
        let l = Analytic(|_, _, z: f64| 4.0 * z - 1.0);
        for sch in DScheme::ALL {
            let expect = if sch == DScheme::Cd2ndT { 8.0 } else { 4.0 };
            assert!((d1(sch, 2, &l) - expect).abs() < 1e-9, "{sch}");
        }
        // Second derivatives.
        let p = Analytic(|x: f64, y: f64, z: f64| x * x * x * x / 12.0 + x * y * 2.0 + z * z);
        for sch in [DDScheme::CdFourth, DDScheme::CdSixth] {
            assert!((d2(sch, 0, &p) - 0.0).abs() < 1e-12);
        }
        for sch in [DDScheme::CdSecond, DDScheme::CdFourth, DDScheme::CdSixth] {
            assert!((d2(sch, 2, &p) - 2.0).abs() < 1e-12);
            assert!((d2_cross(sch, 0, 1, &p) - 2.0).abs() < 1e-12);
            assert!((d2_cross(sch, 1, 2, &p)).abs() < 1e-12);
        }
    }

    #[test]
    fn names() {
        for s in DScheme::ALL {
            assert_eq!(DScheme::from_name(&format!(" {} ", s.as_str().to_uppercase())), Some(s));
        }
        assert_eq!(BiasedGradientScheme::from_name("hjweno5_bias"), Some(BiasedGradientScheme::HjWeno5Bias));
        assert_eq!(TemporalIntegrationScheme::from_name("tvd_rk3"), Some(TemporalIntegrationScheme::TvdRk3));
        assert_eq!(DScheme::from_name("nope"), None);
    }

    #[test]
    fn godunov() {
        assert_eq!(godunovs_norm_sqrd(true, 1.0, 2.0, -1.0, -3.0, 0.5, 0.5), 1.0 + 9.0 + 0.25);
        assert_eq!(godunovs_norm_sqrd(false, 1.0, 2.0, -1.0, -3.0, 0.5, 0.5), 4.0 + 1.0 + 0.25);
    }
}
