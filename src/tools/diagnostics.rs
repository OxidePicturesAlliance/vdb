//! Diagnostics of grid values (port of `openvdb/tools/Diagnostics.h`):
//!
//! * [`check_level_set`] and [`check_fog_volume`] validate the conventions of level sets
//!   and fog volumes and return an empty string on success,
//! * [`Diagnose`] runs individual checks ([`CheckNan`], [`CheckInf`], [`CheckFinite`],
//!   [`CheckMagnitude`], [`CheckRange`], [`CheckMin`], [`CheckMax`], [`CheckNormGrad`],
//!   [`CheckEikonal`], [`CheckDivergence`]) and optionally records failing values in a
//!   boolean mask grid,
//! * [`unique_inactive_values`] collects the distinct inactive values of a grid.

use crate::grid::{Grid, GridClass};
use crate::math::{Coord, Vec3, Vec3d, Vec3f, Vec3i};
use crate::tools::level_set_measure::{first_bias_norm_sq_grad, weno_norm_sq_grad, weno_values, weno_zero_crossing};
use crate::tools::statistics::ValueIterKind;
use crate::tools::tree_ops::{has_active_tiles, merge, MergePolicy};
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::{Internal2, LeafNode, Node, Tree, ValueAccessor};
use crate::types::{RealValue, Value};
use rayon::prelude::*;

/// Format a number like C++ `std::ostream` with its default precision (`%g` with 6
/// significant digits).
pub fn format_g(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if !v.is_finite() {
        return if v.is_nan() {
            "nan".into()
        } else if v > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let s = format!("{:.5e}", v);
    let (mantissa, e) = s.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap();
    if (-5..6).contains(&e) {
        let decimals = (5 - e).max(0) as usize;
        let f = format!("{:.*}", decimals, v);
        if f.contains('.') {
            f.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            f
        }
    } else {
        let m = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!("{}e{}{:02}", m, if e < 0 { '-' } else { '+' }, e.abs())
    }
}

/// Access to the scalar components of a value (`VecTraits::ElementType`).
pub trait DiagValue: Value {
    /// Number of scalar components.
    const SIZE: usize;
    /// Component `i` as a double.
    fn element(&self, i: usize) -> f64;
}

macro_rules! impl_diag_scalar {
    ($($t:ty),*) => {$(
        impl DiagValue for $t {
            const SIZE: usize = 1;
            #[inline] fn element(&self, _i: usize) -> f64 { *self as f64 }
        }
    )*};
}
impl_diag_scalar!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
impl DiagValue for bool {
    const SIZE: usize = 1;
    fn element(&self, _i: usize) -> f64 {
        *self as u8 as f64
    }
}
macro_rules! impl_diag_vec {
    ($($t:ty),*) => {$(
        impl DiagValue for $t {
            const SIZE: usize = 3;
            #[inline] fn element(&self, i: usize) -> f64 { self.0[i] as f64 }
        }
    )*};
}
impl_diag_vec!(Vec3f, Vec3d, Vec3i);

/// A test applied to grid values; it returns true for values that *fail* the test.
pub trait DiagnosticCheck<T: Value>: Sync {
    /// The values visited by the check (`TreeIterT`).
    fn iter_kind(&self) -> ValueIterKind;
    /// Test a value (used for the background and for tiles).
    fn check_value(&self, v: &T) -> bool;
    /// Test a tile value.
    fn check_tile(&self, v: &T) -> bool {
        self.check_value(v)
    }
    /// Test a voxel value at `ijk`.
    fn check_voxel(&self, _acc: &ValueAccessor<'_, Internal2<T>>, _ijk: &Coord, v: &T) -> bool {
        self.check_value(v)
    }
    /// Description of a failure, e.g. `"not finite"`.
    fn description(&self) -> String;
}

fn any_element<T: DiagValue>(v: &T, f: impl Fn(f64) -> bool) -> bool {
    (0..T::SIZE).any(|i| f(v.element(i)))
}

/// Fails for NaN values (`CheckNan`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckNan {
    pub kind: ValueIterKind,
}
impl<T: DiagValue> DiagnosticCheck<T> for CheckNan {
    fn iter_kind(&self) -> ValueIterKind {
        self.kind
    }
    fn check_value(&self, v: &T) -> bool {
        any_element(v, |x| x.is_nan())
    }
    fn description(&self) -> String {
        "NaN".into()
    }
}

/// Fails for infinite values (`CheckInf`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckInf {
    pub kind: ValueIterKind,
}
impl<T: DiagValue> DiagnosticCheck<T> for CheckInf {
    fn iter_kind(&self) -> ValueIterKind {
        self.kind
    }
    fn check_value(&self, v: &T) -> bool {
        any_element(v, |x| x.is_infinite())
    }
    fn description(&self) -> String {
        "infinite".into()
    }
}

/// Fails for NaN or infinite values (`CheckFinite`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckFinite {
    pub kind: ValueIterKind,
}
impl<T: DiagValue> DiagnosticCheck<T> for CheckFinite {
    fn iter_kind(&self) -> ValueIterKind {
        self.kind
    }
    fn check_value(&self, v: &T) -> bool {
        any_element(v, |x| !x.is_finite())
    }
    fn description(&self) -> String {
        "not finite".into()
    }
}

/// Fails for values whose magnitude differs from `abs_val` by more than `tol_val`
/// (`CheckMagnitude`, by default on inactive values).
#[derive(Clone, Copy, Debug)]
pub struct CheckMagnitude {
    pub kind: ValueIterKind,
    pub abs_val: f64,
    pub tol_val: f64,
}
impl CheckMagnitude {
    /// A check of inactive values against `±a` with tolerance `t`.
    pub fn new(a: f64, t: f64) -> Self {
        CheckMagnitude { kind: ValueIterKind::Off, abs_val: a.abs(), tol_val: t.abs() }
    }
}
impl<T: DiagValue> DiagnosticCheck<T> for CheckMagnitude {
    fn iter_kind(&self) -> ValueIterKind {
        self.kind
    }
    fn check_value(&self, v: &T) -> bool {
        any_element(v, |x| (x.abs() - self.abs_val).abs() > self.tol_val)
    }
    fn description(&self) -> String {
        format!("not equal to +/-{} with a tolerance of {}", format_g(self.abs_val), format_g(self.tol_val))
    }
}

/// Fails for values outside `[min, max]` (`CheckRange`; inclusivity is configurable).
#[derive(Clone, Copy, Debug)]
pub struct CheckRange {
    pub kind: ValueIterKind,
    pub min: f64,
    pub max: f64,
    pub min_inclusive: bool,
    pub max_inclusive: bool,
}
impl CheckRange {
    /// A check of active values against the closed range `[min, max]`.
    pub fn new(min: f64, max: f64) -> ToolResult<Self> {
        if min > max {
            return Err(ToolError::Value("CheckRange: Invalid range (min > max)".into()));
        }
        Ok(CheckRange { kind: ValueIterKind::On, min, max, min_inclusive: true, max_inclusive: true })
    }
}
impl<T: DiagValue> DiagnosticCheck<T> for CheckRange {
    fn iter_kind(&self) -> ValueIterKind {
        self.kind
    }
    fn check_value(&self, v: &T) -> bool {
        any_element(v, |x| {
            (if self.min_inclusive { x < self.min } else { x <= self.min })
                || (if self.max_inclusive { x > self.max } else { x >= self.max })
        })
    }
    fn description(&self) -> String {
        format!(
            "outside the value range {}{},{}{}",
            if self.min_inclusive { "[" } else { "]" },
            format_g(self.min),
            format_g(self.max),
            if self.max_inclusive { "]" } else { "[" }
        )
    }
}

/// Fails for values smaller than `min` (`CheckMin`).
#[derive(Clone, Copy, Debug)]
pub struct CheckMin {
    pub kind: ValueIterKind,
    pub min: f64,
}
impl<T: DiagValue> DiagnosticCheck<T> for CheckMin {
    fn iter_kind(&self) -> ValueIterKind {
        self.kind
    }
    fn check_value(&self, v: &T) -> bool {
        any_element(v, |x| x < self.min)
    }
    fn description(&self) -> String {
        format!("smaller than {}", format_g(self.min))
    }
}

/// Fails for values larger than `max` (`CheckMax`).
#[derive(Clone, Copy, Debug)]
pub struct CheckMax {
    pub kind: ValueIterKind,
    pub max: f64,
}
impl<T: DiagValue> DiagnosticCheck<T> for CheckMax {
    fn iter_kind(&self) -> ValueIterKind {
        self.kind
    }
    fn check_value(&self, v: &T) -> bool {
        any_element(v, |x| x > self.max)
    }
    fn description(&self) -> String {
        format!("larger than {}", format_g(self.max))
    }
}

/// Fails where the norm of the first-order upwind gradient is outside `[min, max]`
/// (`CheckNormGrad`, active voxels only; tiles are tested as zero).
#[derive(Clone, Copy, Debug)]
pub struct CheckNormGrad {
    inv_dx2: f64,
    min2: f64,
    max2: f64,
}
impl CheckNormGrad {
    pub fn new<T: Value>(grid: &Grid<T>, min: f64, max: f64) -> ToolResult<Self> {
        if !grid.transform().has_uniform_scale() {
            return Err(ToolError::Value("CheckNormGrad: The transform must have uniform scale".into()));
        }
        if min > max {
            return Err(ToolError::Value("CheckNormGrad: Invalid range (min > max)".into()));
        }
        let dx = grid.voxel_size()[0];
        Ok(CheckNormGrad { inv_dx2: 1.0 / (dx * dx), min2: min * min, max2: max * max })
    }
}
impl<T: RealValue + ToolValue> DiagnosticCheck<T> for CheckNormGrad {
    fn iter_kind(&self) -> ValueIterKind {
        ValueIterKind::On
    }
    fn check_value(&self, v: &T) -> bool {
        let x = (*v).to_f64();
        x < self.min2 || x > self.max2
    }
    fn check_tile(&self, _v: &T) -> bool {
        self.check_value(&T::from_f64(0.0))
    }
    fn check_voxel(&self, acc: &ValueAccessor<'_, Internal2<T>>, ijk: &Coord, v: &T) -> bool {
        let n = first_bias_norm_sq_grad(acc, ijk, *v).to_f64() * self.inv_dx2;
        n < self.min2 || n > self.max2
    }
    fn description(&self) -> String {
        format!("outside the range of NormGrad [{},{}]", format_g(self.min2.sqrt()), format_g(self.max2.sqrt()))
    }
}

/// Fails at zero-crossing voxels where the squared WENO gradient norm is outside
/// `[min, max]` (`CheckEikonal`, active voxels only).
#[derive(Clone, Copy, Debug)]
pub struct CheckEikonal {
    dx: f64,
    min: f64,
    max: f64,
}
impl CheckEikonal {
    pub fn new<T: Value>(grid: &Grid<T>, min: f64, max: f64) -> ToolResult<Self> {
        if !grid.transform().has_uniform_scale() {
            return Err(ToolError::Value("CheckEikonal: The transform must have uniform scale".into()));
        }
        if min > max {
            return Err(ToolError::Value("CheckEikonal: Invalid range (min > max)".into()));
        }
        Ok(CheckEikonal { dx: grid.voxel_size()[0], min, max })
    }
}
impl<T: RealValue + ToolValue> DiagnosticCheck<T> for CheckEikonal {
    fn iter_kind(&self) -> ValueIterKind {
        ValueIterKind::On
    }
    fn check_value(&self, v: &T) -> bool {
        let x = (*v).to_f64();
        x < self.min || x > self.max
    }
    fn check_tile(&self, _v: &T) -> bool {
        self.check_value(&T::from_f64(0.0))
    }
    fn check_voxel(&self, acc: &ValueAccessor<'_, Internal2<T>>, ijk: &Coord, v: &T) -> bool {
        let s = weno_values(acc, ijk, *v);
        if !weno_zero_crossing(&s) {
            return false;
        }
        let n = weno_norm_sq_grad(&s, self.dx, T::from_f64(0.0)).to_f64();
        n < self.min || n > self.max
    }
    fn description(&self) -> String {
        format!("outside the range of NormGrad [{},{}]", format_g(self.min), format_g(self.max))
    }
}

/// Fails where the central-difference divergence of a vector grid is outside
/// `[min, max]` (`CheckDivergence`, active voxels only).
#[derive(Clone, Copy, Debug)]
pub struct CheckDivergence {
    inv_dx: f64,
    min: f64,
    max: f64,
}
impl CheckDivergence {
    pub fn new<T: Value>(grid: &Grid<T>, min: f64, max: f64) -> ToolResult<Self> {
        if !grid.transform().has_uniform_scale() {
            return Err(ToolError::Value("CheckDivergence: The transform must have uniform scale".into()));
        }
        if min > max {
            return Err(ToolError::Value("CheckDivergence: Invalid range (min > max)".into()));
        }
        Ok(CheckDivergence { inv_dx: 1.0 / grid.voxel_size()[0], min, max })
    }
}
impl<E: RealValue> DiagnosticCheck<Vec3<E>> for CheckDivergence
where
    Vec3<E>: DiagValue,
{
    fn iter_kind(&self) -> ValueIterKind {
        ValueIterKind::On
    }
    fn check_value(&self, v: &Vec3<E>) -> bool {
        any_element(v, |x| x < self.min || x > self.max)
    }
    fn check_tile(&self, _v: &Vec3<E>) -> bool {
        0.0 < self.min || 0.0 > self.max
    }
    fn check_voxel(&self, acc: &ValueAccessor<'_, Internal2<Vec3<E>>>, ijk: &Coord, _v: &Vec3<E>) -> bool {
        let d = |axis: usize| {
            let mut o = [0i32; 3];
            o[axis] = 1;
            let p = acc.get_value(&ijk.offset_by(o[0], o[1], o[2]))[axis];
            let m = acc.get_value(&ijk.offset_by(-o[0], -o[1], -o[2]))[axis];
            (p - m).to_f64() * 0.5
        };
        let div = self.inv_dx * (d(0) + d(1) + d(2));
        div < self.min || div > self.max
    }
    fn description(&self) -> String {
        format!("outside the range of divergence [{},{}]", format_g(self.min), format_g(self.max))
    }
}

/// Runs checks on a grid and accumulates failures (`tools::Diagnose`).
pub struct Diagnose<'a, T: Value> {
    grid: &'a Grid<T>,
    mask: Grid<bool>,
    count: u64,
}

impl<'a, T: Value> Diagnose<'a, T> {
    pub fn new(grid: &'a Grid<T>) -> Self {
        Diagnose { grid, mask: Grid::new(false).with_transform(grid.transform().clone()), count: 0 }
    }

    /// Run `check` on the background, the tiles and/or the voxels of the grid, returning a
    /// description of the failures (empty if none). If `update_mask` is true, failing
    /// tiles and voxels are activated in [`mask`](Self::mask).
    pub fn check<C: DiagnosticCheck<T>>(
        &mut self,
        check: &C,
        update_mask: bool,
        check_voxels: bool,
        check_tiles: bool,
        check_background: bool,
    ) -> String {
        let mut out = String::new();
        let tree = self.grid.tree();
        let kind = check.iter_kind();
        if check_background && check.check_value(tree.background()) {
            self.count += 1;
            out.push_str(&format!("Background is {}\n", check.description()));
        }
        if check_tiles {
            let mut failing = Vec::new();
            tree.visit_tiles(&mut |bbox, v, on, _| {
                if kind.accepts(on) && check.check_tile(v) {
                    failing.push(bbox);
                }
            });
            let m = failing.len() as u64;
            self.count += m;
            if update_mask {
                for b in failing {
                    self.mask.tree_mut().fill(&b, true, true);
                }
            }
            if m > 0 {
                out.push_str(&format!("{} tile{}{}\n", m, if m == 1 { " is " } else { "s are " }, check.description()));
            }
        }
        if check_voxels {
            let results: Vec<(u64, Option<Box<LeafNode<bool>>>)> = tree
                .leaf_refs()
                .par_iter()
                .map(|leaf| {
                    let acc = tree.accessor();
                    let mut n = 0u64;
                    let mut mleaf: Option<Box<LeafNode<bool>>> = None;
                    for (pos, v, on) in leaf.iter_all() {
                        if !kind.accepts(on) {
                            continue;
                        }
                        let ijk = leaf.offset_to_global_coord(pos);
                        if check.check_voxel(&acc, &ijk, v) {
                            n += 1;
                            if update_mask {
                                mleaf
                                    .get_or_insert_with(|| Box::new(LeafNode::new(leaf.origin(), false, false)))
                                    .set_value_on_at(pos, true);
                            }
                        }
                    }
                    (n, mleaf)
                })
                .collect();
            let m: u64 = results.iter().map(|r| r.0).sum();
            self.count += m;
            if update_mask {
                let mut t = Tree::<bool>::new(false);
                for (_, l) in results {
                    if let Some(l) = l {
                        t.add_leaf(l);
                    }
                }
                merge(self.mask.tree_mut(), &mut t, MergePolicy::ActiveStatesAndNodes);
            }
            if m > 0 {
                out.push_str(&format!(
                    "{} voxel{}{}\n",
                    m,
                    if m == 1 { " is " } else { "s are " },
                    check.description()
                ));
            }
        }
        out
    }
    /// Mask of the failing values recorded so far.
    pub fn mask(&self) -> &Grid<bool> {
        &self.mask
    }
    /// Number of active voxels in the mask.
    pub fn value_count(&self) -> u64 {
        self.mask.active_voxel_count()
    }
    /// Number of failures (background, tiles and voxels) recorded so far.
    pub fn failure_count(&self) -> u64 {
        self.count
    }
    pub fn grid(&self) -> &Grid<T> {
        self.grid
    }
    pub fn clear(&mut self) {
        self.mask = Grid::new(false).with_transform(self.grid.transform().clone());
        self.count = 0;
    }
}

/// Level-set validation (`tools::CheckLevelSet`).
pub struct CheckLevelSet<'a, T: RealValue + ToolValue + DiagValue> {
    diagnose: Diagnose<'a, T>,
}

impl<'a, T: RealValue + ToolValue + DiagValue> CheckLevelSet<'a, T> {
    pub fn new(grid: &'a Grid<T>) -> Self {
        CheckLevelSet { diagnose: Diagnose::new(grid) }
    }
    pub fn diagnose(&self) -> &Diagnose<'a, T> {
        &self.diagnose
    }
    pub fn check_value_type(&self) -> String {
        String::new()
    }
    pub fn check_class_type(&self) -> String {
        if self.diagnose.grid().grid_class() == GridClass::LevelSet {
            String::new()
        } else {
            "Class type is not \"GRID_LEVEL_SET\"\n".into()
        }
    }
    pub fn check_transform(&self) -> String {
        if self.diagnose.grid().transform().has_uniform_scale() {
            String::new()
        } else {
            "Does not have uniform voxels\n".into()
        }
    }
    pub fn check_background(&self, half_width: f64) -> String {
        let g = self.diagnose.grid();
        let bg = (*g.background()).to_f64();
        let w = bg / g.voxel_size()[0];
        if w < half_width * (1.0 - 1e-6) {
            format!("The background value ({}) is less than {} voxel units\n", format_g(bg), format_g(half_width))
        } else {
            String::new()
        }
    }
    pub fn check_tiles(&self) -> String {
        if has_active_tiles(self.diagnose.grid().tree()) {
            "Has active tile values\n".into()
        } else {
            String::new()
        }
    }
    pub fn check_finite(&mut self, update_mask: bool) -> String {
        self.diagnose.check(&CheckFinite { kind: ValueIterKind::All }, update_mask, true, true, true)
    }
    pub fn check_range(&mut self, update_mask: bool) -> String {
        let bg = (*self.diagnose.grid().background()).to_f64();
        let c = CheckRange { kind: ValueIterKind::On, min: -bg, max: bg, min_inclusive: true, max_inclusive: true };
        self.diagnose.check(&c, update_mask, true, false, false)
    }
    pub fn check_inactive_values(&mut self, update_mask: bool) -> String {
        let bg = (*self.diagnose.grid().background()).to_f64();
        let c = CheckMagnitude::new(bg, T::tolerance().to_f64());
        self.diagnose.check(&c, update_mask, true, true, false)
    }
    pub fn check_eikonal(&mut self, update_mask: bool, min: f64, max: f64) -> String {
        match CheckEikonal::new(self.diagnose.grid(), min, max) {
            Ok(c) => self.diagnose.check(&c, update_mask, true, false, false),
            Err(e) => e.to_string(),
        }
    }
    /// Run the first `n` (up to 9) checks, stopping at the first failure.
    pub fn check(&mut self, n: usize, update_mask: bool) -> String {
        let mut s = self.check_value_type();
        if s.is_empty() && n > 1 {
            s = self.check_class_type();
        }
        if s.is_empty() && n > 2 {
            s = self.check_transform();
        }
        if s.is_empty() && n > 3 {
            s = self.check_background(crate::tools::level_set_sphere::LEVEL_SET_HALF_WIDTH);
        }
        if s.is_empty() && n > 4 {
            s = self.check_tiles();
        }
        if s.is_empty() && n > 5 {
            s = self.check_finite(update_mask);
        }
        if s.is_empty() && n > 6 {
            s = self.check_range(update_mask);
        }
        if s.is_empty() && n > 7 {
            s = self.check_inactive_values(update_mask);
        }
        if s.is_empty() && n > 8 {
            s = self.check_eikonal(update_mask, 0.5, 1.5);
        }
        s
    }
}

/// Validate a level set; returns an empty string if the first `n` (default 9) checks pass,
/// otherwise a description of the first failure (`tools::checkLevelSet`).
pub fn check_level_set<T: RealValue + ToolValue + DiagValue>(grid: &Grid<T>, n: usize) -> String {
    CheckLevelSet::new(grid).check(n, false)
}

/// Fog-volume validation (`tools::CheckFogVolume`).
pub struct CheckFogVolume<'a, T: RealValue + ToolValue + DiagValue> {
    diagnose: Diagnose<'a, T>,
}

impl<'a, T: RealValue + ToolValue + DiagValue> CheckFogVolume<'a, T> {
    pub fn new(grid: &'a Grid<T>) -> Self {
        CheckFogVolume { diagnose: Diagnose::new(grid) }
    }
    pub fn diagnose(&self) -> &Diagnose<'a, T> {
        &self.diagnose
    }
    pub fn check_value_type(&self) -> String {
        String::new()
    }
    pub fn check_class_type(&self) -> String {
        if self.diagnose.grid().grid_class() == GridClass::FogVolume {
            String::new()
        } else {
            // (sic) OpenVDB reports the level-set class name here.
            "Class type is not \"GRID_LEVEL_SET\"".into()
        }
    }
    pub fn check_background(&self) -> String {
        let bg = (*self.diagnose.grid().background()).to_f64();
        if bg.abs() > T::tolerance().to_f64() {
            format!("The background value ({}) is not zero", format_g(bg))
        } else {
            String::new()
        }
    }
    pub fn check_finite(&mut self, update_mask: bool) -> String {
        self.diagnose.check(&CheckFinite { kind: ValueIterKind::All }, update_mask, true, true, true)
    }
    pub fn check_inactive_values(&mut self, update_mask: bool) -> String {
        let c = CheckMagnitude::new(0.0, T::tolerance().to_f64());
        self.diagnose.check(&c, update_mask, true, true, true)
    }
    pub fn check_range(&mut self, update_mask: bool) -> String {
        let c = CheckRange { kind: ValueIterKind::On, min: 0.0, max: 1.0, min_inclusive: true, max_inclusive: true };
        self.diagnose.check(&c, update_mask, true, true, false)
    }
    /// Run the first `n` (up to 6) checks, stopping at the first failure.
    pub fn check(&mut self, n: usize, update_mask: bool) -> String {
        let mut s = self.check_value_type();
        if s.is_empty() && n > 1 {
            s = self.check_class_type();
        }
        if s.is_empty() && n > 2 {
            s = self.check_background();
        }
        if s.is_empty() && n > 3 {
            s = self.check_finite(update_mask);
        }
        if s.is_empty() && n > 4 {
            s = self.check_inactive_values(update_mask);
        }
        if s.is_empty() && n > 5 {
            s = self.check_range(update_mask);
        }
        s
    }
}

/// Validate a fog volume; returns an empty string if the first `n` (default 6) checks
/// pass (`tools::checkFogVolume`).
pub fn check_fog_volume<T: RealValue + ToolValue + DiagValue>(grid: &Grid<T>, n: usize) -> String {
    CheckFogVolume::new(grid).check(n, false)
}

/// Collect the distinct inactive values (voxels and tiles) of `grid`, sorted; returns
/// `None` if there are more than `num_values` of them (`tools::uniqueInactiveValues`).
pub fn unique_inactive_values<T: ToolValue>(grid: &Grid<T>, num_values: usize) -> Option<Vec<T>> {
    let tree = grid.tree();
    let mut values: Vec<T> = Vec::new();
    let add = |v: T, values: &mut Vec<T>| {
        if let Err(i) = values.binary_search_by(|a| {
            if a.cwise_lt(&v) {
                std::cmp::Ordering::Less
            } else if v.cwise_lt(a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            values.insert(i, v);
        }
    };
    for leaf in tree.leaves() {
        for (_, v) in leaf.iter_off() {
            add(*v, &mut values);
        }
        if values.len() > num_values {
            return None;
        }
    }
    tree.visit_tiles(&mut |_, v, on, _| {
        if !on {
            add(*v, &mut values);
        }
    });
    (values.len() <= num_values).then_some(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::level_set_sphere::create_level_set_sphere;
    use crate::tools::level_set_util::sdf_to_fog_volume;

    #[test]
    fn level_set_and_fog_checks() {
        let g = create_level_set_sphere::<f32>(1.0, Vec3f::splat(0.0), 0.1, 3.0, true).unwrap();
        assert_eq!(check_level_set(&g, 9), "");
        let mut bad = g.clone();
        bad.set_value(&Coord::new(0, 0, 0), f32::NAN);
        assert_eq!(check_level_set(&bad, 9), "1 voxel is not finite\n");
        let mut narrow = g.clone();
        crate::tools::change_background::change_level_set_background(narrow.tree_mut(), 0.1, true).unwrap();
        assert_eq!(check_level_set(&narrow, 9), "The background value (0.1) is less than 3 voxel units\n");

        let mut fog = g.clone();
        sdf_to_fog_volume(&mut fog, f32::MAX);
        assert_eq!(check_fog_volume(&fog, 6), "");
        assert_eq!(check_fog_volume(&g, 6), "Class type is not \"GRID_LEVEL_SET\"");
        let vals = unique_inactive_values(&g, 2).unwrap();
        assert_eq!(vals, vec![-0.3, 0.3]);

        let mut d = Diagnose::new(&bad);
        let s = d.check(&CheckNan::default(), true, true, true, true);
        assert_eq!(s, "1 voxel is NaN\n");
        assert_eq!(d.value_count(), 1);
        assert_eq!(format_g(0.123456789), "0.123457");
        assert_eq!(format_g(1e-7), "1e-07");
        assert_eq!(format_g(-3.0), "-3");
    }
}
