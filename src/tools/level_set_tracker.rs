//! Narrow-band tracking and re-normalization of level sets (port of
//! `openvdb/tools/LevelSetTracker.h`).
//!
//! [`LevelSetTracker`] keeps a narrow-band level set a valid signed distance field while it is
//! being deformed: [`normalize`](LevelSetTracker::normalize) solves the re-initialization PDE
//! `∂φ/∂t + sign(φ)(|∇φ| - 1) = 0` with upwind spatial differences and TVD Runge-Kutta time
//! integration, [`track`](LevelSetTracker::track) dilates the band by one voxel, re-normalizes
//! and trims voxels that left the band, and [`resize`](LevelSetTracker::resize) changes the
//! band's half-width.

use super::internal::{
    change_level_set_background, dilate_face, erode_face, mask_difference, par_leaf_compute, prune_level_set,
    uniform_voxel_size, voxel_mask, AuxBuffers,
};
use crate::grid::{Grid, GridClass};
use crate::math::finite_difference::{BiasedGradientScheme, TemporalIntegrationScheme};
use crate::math::operators::is_gradient_norm_sqrd;
use crate::math::stencils::{BaseStencil, StencilKind};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::{RealValue, Value};

/// OpenVDB's default narrow-band half-width, in voxels (`LEVEL_SET_HALF_WIDTH`).
pub const LEVEL_SET_HALF_WIDTH: f64 = 3.0;

/// How [`LevelSetTracker::prune`] treats voxels that have left the narrow band
/// (`lstrack::TrimMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrimMode {
    /// Leave out-of-band voxels intact.
    None,
    /// Set out-of-band interior voxels to the (negative) background value.
    Interior,
    /// Set out-of-band exterior voxels to the background value.
    Exterior,
    /// Set all out-of-band voxels to the background value.
    #[default]
    All,
}

/// Numerical parameters of a [`LevelSetTracker`] (`LevelSetTracker::State`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackerState {
    /// Upwind scheme used for `|∇φ|`.
    pub spatial_scheme: BiasedGradientScheme,
    /// Time integration scheme.
    pub temporal_scheme: TemporalIntegrationScheme,
    /// Number of normalization iterations per call to `normalize`.
    pub norm_count: usize,
    /// Parallel grain size (0 disables threading in OpenVDB; kept for API parity).
    pub grain_size: usize,
}

impl Default for TrackerState {
    fn default() -> Self {
        TrackerState {
            spatial_scheme: BiasedGradientScheme::HjWeno5Bias,
            temporal_scheme: TemporalIntegrationScheme::TvdRk1,
            norm_count: LEVEL_SET_HALF_WIDTH as usize,
            grain_size: 1,
        }
    }
}

/// Errors raised when a grid cannot be processed as a level set.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LevelSetError {
    /// The transform does not have uniform voxels.
    #[error("the transform must have uniform scale for level set tools to function")]
    NonUniformVoxels,
    /// The grid class is not `GridClass::LevelSet`.
    #[error("expected a level set grid, got a grid of class \"{0}\"")]
    NotALevelSet(&'static str),
    /// The map type is not supported.
    #[error("unsupported transform: {0}")]
    UnsupportedTransform(&'static str),
}

/// Check that `grid` is a level set with uniform voxels and return its voxel size.
pub(crate) fn check_level_set<T: Value>(grid: &Grid<T>) -> Result<f64, LevelSetError> {
    let dx = uniform_voxel_size(grid).ok_or(LevelSetError::NonUniformVoxels)?;
    if grid.grid_class() != GridClass::LevelSet {
        return Err(LevelSetError::NotALevelSet(grid.grid_class().as_str()));
    }
    Ok(dx)
}

/// One Euler stage `result = alpha * phi + (1 - alpha) * eval(stencil)` over the active voxels
/// (or the voxels of `mask`), reading `phi` from auxiliary buffer `phi_idx` (0 = the tree) and
/// writing into auxiliary buffer `res_idx`.
#[allow(clippy::too_many_arguments)]
fn euler_stage<T, E>(
    grid: &Grid<T>,
    aux: &mut AuxBuffers<T>,
    nom: i32,
    den: i32,
    phi_idx: usize,
    res_idx: usize,
    kind: StencilKind,
    mask: Option<&Tree<bool>>,
    eval: &E,
) where
    T: RealValue,
    E: Fn(&BaseStencil<'_, T>, usize, usize) -> Option<T> + Sync,
{
    let alpha = T::from_f64(nom as f64) / T::from_f64(den as f64);
    let beta = T::from_f64(1.0) - alpha;
    let aux_ref: &AuxBuffers<T> = aux;
    let bufs = par_leaf_compute(
        grid.tree(),
        || BaseStencil::new(grid, kind),
        |i, _| aux_ref.get(res_idx, i).to_vec(),
        |st, i, leaf: &LeafNode<T>, out| {
            let phi: &[T] = if phi_idx == 0 { leaf.buffer() } else { aux_ref.get(phi_idx, i) };
            let origin = leaf.origin();
            let mut visit = |n: usize, st: &mut BaseStencil<'_, T>| {
                let xyz = origin + LeafNode::<T>::offset_to_local_coord(n);
                st.move_to_with_center(&xyz, leaf.buffer()[n]);
                if let Some(v) = eval(st, i, n) {
                    out[n] = if nom != 0 { alpha * phi[n] + beta * v } else { v };
                }
            };
            match mask {
                None => leaf.value_mask().iter_on().for_each(|n| visit(n, st)),
                Some(m) => {
                    if let Some(ml) = m.probe_leaf(&origin) {
                        ml.value_mask().iter_on().for_each(|n| visit(n, st));
                    }
                }
            }
        },
    );
    aux.set(res_idx, bufs);
}

/// Advance the active voxels of `grid` by one TVD Runge-Kutta step, where `eval(stencil,
/// leaf_index, voxel_offset)` returns the forward-Euler update of the voxel at the stencil's
/// center (the stencil already holds the current values around it), or `None` to leave the
/// voxel's result buffer untouched.
pub(crate) fn tvd_rk_step<T, E>(
    grid: &mut Grid<T>,
    temporal: TemporalIntegrationScheme,
    kind: StencilKind,
    mask: Option<&Tree<bool>>,
    eval: E,
) where
    T: RealValue,
    E: Fn(&BaseStencil<'_, T>, usize, usize) -> Option<T> + Sync,
{
    let mut aux = AuxBuffers::new(grid.tree(), temporal.aux_buffers());
    let stages: &[(i32, i32, usize, usize)] = match temporal {
        TemporalIntegrationScheme::TvdRk1 => &[(0, 1, 0, 1)],
        TemporalIntegrationScheme::TvdRk2 => &[(0, 1, 0, 1), (1, 2, 1, 1)],
        TemporalIntegrationScheme::TvdRk3 => &[(0, 1, 0, 1), (3, 4, 1, 2), (1, 3, 1, 2)],
    };
    for &(nom, den, phi, res) in stages {
        euler_stage(grid, &mut aux, nom, den, phi, res, kind, mask, &eval);
        aux.swap_with_tree(grid.tree_mut(), res);
    }
}

/// Maintains a narrow-band level set (`tools::LevelSetTracker`).
pub struct LevelSetTracker<'a, T: RealValue> {
    grid: &'a mut Grid<T>,
    dx: T,
    state: TrackerState,
    trim: TrimMode,
}

impl<'a, T: RealValue> LevelSetTracker<'a, T> {
    /// A tracker for `grid`, which must be a level set (grid class) with uniform voxels.
    pub fn new(grid: &'a mut Grid<T>) -> Result<Self, LevelSetError> {
        let dx = check_level_set(grid)?;
        Ok(LevelSetTracker { grid, dx: T::from_f64(dx), state: TrackerState::default(), trim: TrimMode::All })
    }
    /// The level set grid.
    pub fn grid(&self) -> &Grid<T> {
        self.grid
    }
    /// Mutable access to the level set grid.
    pub fn grid_mut(&mut self) -> &mut Grid<T> {
        self.grid
    }
    /// The voxel size.
    pub fn voxel_size(&self) -> T {
        self.dx
    }
    /// The narrow-band half-width in voxels (background / voxel size).
    pub fn half_width(&self) -> T {
        *self.grid.background() / self.dx
    }
    /// The numerical parameters.
    pub fn state(&self) -> TrackerState {
        self.state
    }
    /// Set the numerical parameters.
    pub fn set_state(&mut self, s: TrackerState) {
        self.state = s;
    }
    /// The spatial scheme.
    pub fn spatial_scheme(&self) -> BiasedGradientScheme {
        self.state.spatial_scheme
    }
    /// Set the spatial scheme.
    pub fn set_spatial_scheme(&mut self, s: BiasedGradientScheme) {
        self.state.spatial_scheme = s;
    }
    /// The temporal scheme.
    pub fn temporal_scheme(&self) -> TemporalIntegrationScheme {
        self.state.temporal_scheme
    }
    /// Set the temporal scheme.
    pub fn set_temporal_scheme(&mut self, s: TemporalIntegrationScheme) {
        self.state.temporal_scheme = s;
    }
    /// Number of normalization iterations.
    pub fn norm_count(&self) -> usize {
        self.state.norm_count
    }
    /// Set the number of normalization iterations.
    pub fn set_norm_count(&mut self, n: usize) {
        self.state.norm_count = n;
    }
    /// The parallel grain size.
    pub fn grain_size(&self) -> usize {
        self.state.grain_size
    }
    /// Set the parallel grain size.
    pub fn set_grain_size(&mut self, g: usize) {
        self.state.grain_size = g;
    }
    /// The trimming mode used by [`prune`](Self::prune).
    pub fn trimming(&self) -> TrimMode {
        self.trim
    }
    /// Set the trimming mode.
    pub fn set_trimming(&mut self, mode: TrimMode) {
        self.trim = mode;
    }

    /// Re-normalize the active voxels to a signed distance field.
    pub fn normalize(&mut self) {
        self.normalize_impl(None)
    }
    /// Re-normalize only the voxels that are active in `mask`.
    pub fn normalize_masked(&mut self, mask: &Tree<bool>) {
        self.normalize_impl(Some(mask))
    }

    fn normalize_impl(&mut self, mask: Option<&Tree<bool>>) {
        let spatial = self.state.spatial_scheme;
        let temporal = self.state.temporal_scheme;
        let cfl = match temporal {
            TemporalIntegrationScheme::TvdRk1 => 0.3f32,
            TemporalIntegrationScheme::TvdRk2 => 0.9f32,
            TemporalIntegrationScheme::TvdRk3 => 1.0f32,
        };
        let dt = self.dx * T::from_f64(cfl as f64);
        let inv_dx = T::from_f64((1.0f32 / self.dx.to_f64() as f32) as f64);
        let one = T::from_f64(1.0);
        let tol = T::tolerance();
        for _ in 0..self.state.norm_count {
            tvd_rk_step(self.grid, temporal, spatial.stencil_kind(), mask, |st, _, _| {
                let norm_sq = is_gradient_norm_sqrd(spatial, st);
                let phi0 = st.center_value();
                let v = phi0 / ((phi0 * phi0 + norm_sq).sqrt() + tol);
                Some(phi0 - dt * v * (norm_sq.sqrt() * inv_dx - one))
            });
        }
    }

    /// Trim voxels outside the narrow band (according to [`trimming`](Self::trimming)) and
    /// prune the tree.
    pub fn prune(&mut self) {
        let gamma = *self.grid.background();
        let mode = self.trim;
        if mode != TrimMode::None {
            self.grid.tree_mut().par_for_each_leaf_mut(|leaf| {
                let on: Vec<usize> = leaf.value_mask().iter_on().collect();
                for n in on {
                    let v = *leaf.get_value_at(n);
                    let interior = matches!(mode, TrimMode::Interior | TrimMode::All);
                    let exterior = matches!(mode, TrimMode::Exterior | TrimMode::All);
                    if interior && v <= -gamma {
                        leaf.set_value_off_at(n, -gamma);
                    } else if exterior && v >= gamma {
                        leaf.set_value_off_at(n, gamma);
                    }
                }
            });
        }
        prune_level_set(self.grid.tree_mut());
    }

    /// Dilate the band by one voxel, re-normalize and prune (one step of narrow-band tracking).
    pub fn track(&mut self) {
        dilate_face(self.grid.tree_mut(), 1);
        self.normalize();
        self.prune();
    }

    /// Grow the narrow band by `iterations` voxels.
    pub fn dilate(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let mask0 = if self.state.norm_count > 0 { Some(voxel_mask(self.grid.tree())) } else { None };
            dilate_face(self.grid.tree_mut(), 1);
            let bg = self.dx + *self.grid.background();
            change_level_set_background(self.grid.tree_mut(), bg);
            if let Some(mask0) = mask0 {
                let mut mask = voxel_mask(self.grid.tree());
                mask_difference(&mut mask, &mask0);
                self.normalize_masked(&mask);
            }
        }
    }

    /// Shrink the narrow band by `iterations` voxels.
    pub fn erode(&mut self, iterations: usize) {
        erode_face(self.grid.tree_mut(), iterations);
        prune_level_set(self.grid.tree_mut());
        let bg = *self.grid.background() - T::from_f64(iterations as f64) * self.dx;
        change_level_set_background(self.grid.tree_mut(), bg);
    }

    /// Change the half-width of the narrow band (in voxels); returns true if it changed.
    pub fn resize(&mut self, half_width: usize) -> bool {
        let w_old = self.half_width().to_f64().floor() as i64;
        let w_new = half_width as i64;
        if w_old < w_new {
            self.dilate((w_new - w_old) as usize);
        } else if w_old > w_new {
            self.erode((w_old - w_new) as usize);
        }
        w_old != w_new
    }
}

/// Statistics of `|∇φ|` over the active voxels (used by tests): returns `(min, max, mean)` of
/// the WENO upwind gradient norm.
#[cfg(test)]
pub(crate) fn grad_norm_stats<T: RealValue>(grid: &Grid<T>) -> (f64, f64, f64) {
    use crate::math::operators::{gradient_norm_sqrd, MapJacobian};
    use rayon::prelude::*;
    let map = MapJacobian::new(grid.transform());
    let vals: Vec<f64> = grid
        .tree()
        .leaf_refs()
        .par_iter()
        .map_init(
            || BaseStencil::new(grid, StencilKind::NineteenPoint),
            |st, leaf| {
                let mut out = Vec::new();
                for (c, _) in leaf.iter_on_coords() {
                    st.move_to(&c);
                    out.push(gradient_norm_sqrd(&map, BiasedGradientScheme::HjWeno5Bias, st).to_f64().sqrt());
                }
                out
            },
        )
        .flatten()
        .collect();
    let n = vals.len().max(1) as f64;
    let min = vals.iter().copied().fold(f64::MAX, f64::min);
    let max = vals.iter().copied().fold(f64::MIN, f64::max);
    (min, max, vals.iter().sum::<f64>() / n)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::math::{Coord, Vec3d};
    use crate::tools::test_util::{assert_close, make_sphere};

    #[test]
    fn rejects_non_level_sets() {
        let mut g = Grid::<f32>::new(1.0);
        assert!(matches!(LevelSetTracker::new(&mut g), Err(LevelSetError::NotALevelSet(_))));
    }

    #[test]
    fn normalize_restores_distance() {
        // Scale a sphere SDF by 2: normalization should bring |∇φ| back towards 1.
        let mut grid = make_sphere::<f32>(1.0, Vec3d::zero(), 0.05, 3.0);
        grid.tree_mut().par_for_each_leaf_mut(|l| l.for_each_on_mut(|_, v| *v *= 2.0));
        let q = Coord::new(22, 0, 0); // true distance 0.1, scaled to 0.2
        assert_close(grid.get_value(&q) as f64, 0.2, 1e-6);
        let mut t = LevelSetTracker::new(&mut grid).unwrap();
        t.set_norm_count(10);
        t.normalize();
        assert!(t.grid().get_value(&Coord::new(20, 0, 0)).abs() < 0.01);
        let v = t.grid().get_value(&q) as f64;
        assert!((v - 0.1).abs() < 0.03, "{v}");
    }

    pub(crate) const REF_PTS: [Coord; 4] =
        [Coord::new(20, 3, -2), Coord::new(-5, 17, 6), Coord::new(9, -12, 11), Coord::new(0, 0, 21)];

    /// The input of the C++ reference runs: a sphere SDF scaled by 1.5 and offset by 0.01.
    pub(crate) fn distorted_sphere() -> Grid<f32> {
        let mut grid = make_sphere::<f32>(1.0, Vec3d::new(0.1, -0.05, 0.02), 0.05, 3.0);
        grid.tree_mut().par_for_each_leaf_mut(|l| l.for_each_on_mut(|_, v| *v = *v * 1.5 + 0.01));
        grid
    }

    #[track_caller]
    pub(crate) fn check_ref(grid: &Grid<f32>, expect: [f64; 4], count: u64, tol: f64) {
        for (p, e) in REF_PTS.iter().zip(expect.iter()) {
            assert_close(grid.get_value(p) as f64, *e, tol);
        }
        assert_eq!(grid.active_voxel_count(), count);
    }

    /// Reference values from `scratchpad/cppref/tracker.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        use BiasedGradientScheme::*;
        use TemporalIntegrationScheme::*;
        let cases = [
            (HjWeno5Bias, TvdRk1, [-0.0796735734, 0.0140926754, -0.150000006, 0.0524771996]),
            (FirstBias, TvdRk3, [-0.0664629042, 0.0128010102, -0.150000006, 0.0453793071]),
            (SecondBias, TvdRk2, [-0.0679191798, 0.0131199267, -0.150000006, 0.0470633209]),
            (ThirdBias, TvdRk1, [-0.0791552961, 0.0143878888, -0.150000006, 0.0524079055]),
            (Weno5Bias, TvdRk2, [-0.064359352, 0.0119629763, -0.150000006, 0.0461023003]),
        ];
        for (spatial, temporal, expect) in cases {
            let mut grid = distorted_sphere();
            let mut t = LevelSetTracker::new(&mut grid).unwrap();
            t.set_spatial_scheme(spatial);
            t.set_temporal_scheme(temporal);
            t.normalize();
            check_ref(&grid, expect, 30372, 1e-6);
        }
        let mut grid = distorted_sphere();
        LevelSetTracker::new(&mut grid).unwrap().track();
        check_ref(&grid, [-0.0796735585, 0.0140926745, -0.150000006, 0.0524771288], 22569, 1e-6);
        let mut grid = distorted_sphere();
        LevelSetTracker::new(&mut grid).unwrap().resize(5);
        check_ref(&grid, [-0.0954032838, 0.0181528442, -0.176679477, 0.0640752912], 47624, 1e-6);
        let mut grid = distorted_sphere();
        LevelSetTracker::new(&mut grid).unwrap().resize(2);
        check_ref(&grid, [-0.0954032838, 0.0181528442, -0.100000009, 0.0640752912], 21876, 1e-6);
    }

    #[test]
    fn track_and_resize() {
        let mut grid = make_sphere::<f32>(1.0, Vec3d::zero(), 0.05, 3.0);
        let n0 = grid.active_voxel_count();
        let mut t = LevelSetTracker::new(&mut grid).unwrap();
        assert_close(t.half_width() as f64, 3.0, 1e-5);
        assert!(t.resize(5));
        assert_close(t.half_width() as f64, 5.0, 1e-5);
        let n1 = t.grid().active_voxel_count();
        assert!(n1 > n0 * 3 / 2, "{n0} {n1}");
        // Dilated voxels are normalized distances.
        let v = t.grid().get_value(&Coord::new(24, 0, 0));
        assert_close(v as f64, 0.2, 0.01);
        assert!(t.resize(2));
        assert_close(t.half_width() as f64, 2.0, 1e-5);
        assert!(t.grid().active_voxel_count() < n0);
        t.track();
        let (_, _, mean) = grad_norm_stats(t.grid());
        assert_close(mean, 1.0, 0.02);
    }
}
