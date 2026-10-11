//! Sparse fast sweeping: signed distance fields from fog volumes or level sets, extension of
//! fields off an iso-surface, and dilation of SDFs (port of `openvdb/tools/FastSweeping.h`).
//!
//! The Eikonal equation `|∇φ| = 1` is solved on the active voxels by Gauss-Seidel sweeps along
//! the eight diagonal directions, processing voxels on planes of constant `±i ± j ± k` (which
//! are mutually independent) in parallel, exactly like OpenVDB.
//!
//! * [`fog_to_sdf`] / [`sdf_to_sdf`] — (re)compute signed distances to an iso-surface.
//! * [`fog_to_ext`] / [`sdf_to_ext`] (and the `_and_ext` variants) — extend a field defined on
//!   the iso-surface by a functor along the distance field's characteristics.
//! * [`dilate_sdf`] — grow the narrow band of an SDF.
//! * [`mask_sdf`] — extend an SDF into the active region of a mask.
//!
//! Unlike OpenVDB 10, the extension grid inherits the transform of the input grid and the
//! extension functor is called with world-space positions (OpenVDB uses an identity transform
//! for the extension grid, so for unit voxels at the origin the results coincide).

use super::change_background::set_level_set_background;
use super::morphology::{dilate_active_values, TilePolicy};
use crate::grid::{Grid, GridClass};
use crate::math::stencils::GradStencil;
use crate::math::{Coord, Vec3, Vec3d};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::{RealValue, Value};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Which side of the iso-surface is swept (`FastSweepingDomain`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FastSweepingDomain {
    /// Sweep on both sides.
    #[default]
    SweepAll,
    /// Only sweep where the value is greater than the iso-value.
    SweepGreaterThanIsoValue,
    /// Only sweep where the value is less than the iso-value.
    SweepLessThanIsoValue,
}

/// Neighbourhood used for dilation (`NearestNeighbors`).
pub use super::morphology::NearestNeighbors;

/// Values that can be extended by fast sweeping: floating-point scalars and vectors are
/// interpolated, integers take the value of the closest neighbour.
pub trait ExtValue: Value {
    /// True for integer types (nearest-neighbour extension).
    const NEAREST: bool;
    /// `a * x + b * y`.
    fn lin2(a: f64, x: Self, b: f64, y: Self) -> Self;
    /// `a * x + b * y + c * z`.
    fn lin3(a: f64, x: Self, b: f64, y: Self, c: f64, z: Self) -> Self;
    /// `s * x`.
    fn scaled(s: f64, x: Self) -> Self;
}

macro_rules! ext_float {
    ($($t:ty),*) => {$(
        impl ExtValue for $t {
            const NEAREST: bool = false;
            fn lin2(a: f64, x: Self, b: f64, y: Self) -> Self { (a * x as f64 + b * y as f64) as $t }
            fn lin3(a: f64, x: Self, b: f64, y: Self, c: f64, z: Self) -> Self {
                (a * x as f64 + b * y as f64 + c * z as f64) as $t
            }
            fn scaled(s: f64, x: Self) -> Self { (s * x as f64) as $t }
        }
        impl ExtValue for Vec3<$t> {
            const NEAREST: bool = false;
            fn lin2(a: f64, x: Self, b: f64, y: Self) -> Self {
                Vec3::new(<$t>::lin2(a, x[0], b, y[0]), <$t>::lin2(a, x[1], b, y[1]), <$t>::lin2(a, x[2], b, y[2]))
            }
            fn lin3(a: f64, x: Self, b: f64, y: Self, c: f64, z: Self) -> Self {
                Vec3::new(
                    <$t>::lin3(a, x[0], b, y[0], c, z[0]),
                    <$t>::lin3(a, x[1], b, y[1], c, z[1]),
                    <$t>::lin3(a, x[2], b, y[2], c, z[2]),
                )
            }
            fn scaled(s: f64, x: Self) -> Self { x.map(|v| <$t>::scaled(s, v)) }
        }
    )*};
}
ext_float!(f32, f64);

impl ExtValue for i32 {
    const NEAREST: bool = true;
    fn lin2(_: f64, x: Self, _: f64, _: Self) -> Self {
        x
    }
    fn lin3(_: f64, x: Self, _: f64, _: Self, _: f64, _: Self) -> Self {
        x
    }
    fn scaled(_: f64, x: Self) -> Self {
        x
    }
}

/// The six face-neighbour offsets, in the order used by OpenVDB's sweeping kernels.
const OFFSETS: [[i32; 3]; 6] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];

/// One slice of voxels with equal diagonal key: `(leaf index, voxel offsets)`.
type Slice = Vec<(usize, Vec<u16>)>;

/// The sparse fast sweeping solver (`tools::FastSweeping`).
pub struct FastSweeping<T: RealValue, E: ExtValue = T> {
    sdf: Option<Grid<T>>,
    ext: Option<Grid<E>>,
    ext_input: Option<Grid<E>>,
    sweep_mask: Tree<bool>,
    leaf_origins: Vec<Coord>,
    sweeping_voxels: usize,
    boundary_voxels: usize,
    mode: FastSweepingDomain,
    is_input_sdf: bool,
}

impl<T: RealValue, E: ExtValue> Default for FastSweeping<T, E> {
    fn default() -> Self {
        FastSweeping {
            sdf: None,
            ext: None,
            ext_input: None,
            sweep_mask: Tree::new(false),
            leaf_origins: Vec::new(),
            sweeping_voxels: 0,
            boundary_voxels: 0,
            mode: FastSweepingDomain::SweepAll,
            is_input_sdf: true,
        }
    }
}

/// Errors raised by [`FastSweeping`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FastSweepingError {
    /// `sweep` was called before an `init_*` method.
    #[error("FastSweeping::sweep called before initialization")]
    NotInitialized,
    /// There is no iso-surface in the active region.
    #[error("FastSweeping: no boundary voxels found")]
    NoBoundaryVoxels,
    /// There are no voxels to update.
    #[error("FastSweeping: no computing voxels found")]
    NoSweepingVoxels,
    /// A SDF input has active tiles.
    #[error("FastSweeping: a SDF should not have active tiles")]
    SdfWithActiveTiles,
    /// One-sided extension needs an input extension grid with the same transform.
    #[error("FastSweeping: one-sided extension requires an aligned input extension grid")]
    MissingExtensionGrid,
    /// The mask is not aligned with the grid.
    #[error("FastSweeping: mask not aligned with the grid")]
    MisalignedMask,
}

impl<T: RealValue, E: ExtValue> FastSweeping<T, E> {
    /// An uninitialized solver.
    pub fn new() -> Self {
        Self::default()
    }
    /// The signed distance grid (after initialization).
    pub fn sdf_grid(&self) -> Option<&Grid<T>> {
        self.sdf.as_ref()
    }
    /// The extension grid (after `init_ext`).
    pub fn ext_grid(&self) -> Option<&Grid<E>> {
        self.ext.as_ref()
    }
    /// Take the signed distance grid out of the solver.
    pub fn take_sdf_grid(&mut self) -> Option<Grid<T>> {
        self.sdf.take()
    }
    /// Take the extension grid out of the solver.
    pub fn take_ext_grid(&mut self) -> Option<Grid<E>> {
        self.ext.take()
    }
    /// Number of voxels updated by the sweeps.
    pub fn sweeping_voxel_count(&self) -> usize {
        self.sweeping_voxels
    }
    /// Number of voxels next to the iso-surface (fixed boundary conditions).
    pub fn boundary_voxel_count(&self) -> usize {
        self.boundary_voxels
    }
    /// True if there is something to sweep.
    pub fn is_valid(&self) -> bool {
        self.sweeping_voxels > 0 && self.boundary_voxels > 0
    }
    /// The sweep direction.
    pub fn sweep_direction(&self) -> FastSweepingDomain {
        self.mode
    }
    /// True if the input was a SDF (as opposed to a fog volume).
    pub fn is_input_sdf(&self) -> bool {
        self.is_input_sdf
    }
    /// Reset the solver.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    fn compute_sweep_mask_leaf_origins(&mut self) {
        self.sweep_mask.prune_inactive();
        self.sweep_mask.voxelize_active_tiles();
        self.leaf_origins = self.sweep_mask.leaves().map(|l| l.origin()).collect();
        self.sweeping_voxels = self.sweep_mask.active_voxel_count() as usize;
        self.boundary_voxels = 0;
        if let Some(sdf) = &self.sdf {
            self.boundary_voxels = (sdf.active_voxel_count() as usize).saturating_sub(self.sweeping_voxels);
        }
    }

    /// Distance initialization of the voxels next to the iso-surface (shared by `InitSdf` and
    /// `InitExt`). Returns per-leaf new SDF buffers, boundary voxel coordinates and the
    /// optional extension values `(leaf index, offset, value)`.
    #[allow(clippy::type_complexity)]
    fn init_boundary<F>(&self, iso: T, op: Option<&F>) -> (Vec<Vec<T>>, Vec<Coord>, Vec<(usize, usize, E)>)
    where
        F: Fn(Vec3d) -> E + Sync,
    {
        let sdf = self.sdf.as_ref().expect("initialized");
        let max = T::max_value();
        let sign = if self.is_input_sdf { T::from_f64(1.0) } else { T::from_f64(-1.0) };
        let above = sign * max;
        let h = sign * T::from_f64(sdf.voxel_size()[0]);
        let xform = sdf.transform().clone();
        let z = <T as Value>::zero();
        let results: Vec<(Vec<T>, Vec<Coord>, Vec<(usize, usize, E)>)> = sdf
            .tree()
            .leaf_refs()
            .par_iter()
            .enumerate()
            .map_init(
                || GradStencil::new(sdf),
                |st, (li, leaf): (usize, &&LeafNode<T>)| {
                    let mut buf = leaf.buffer().to_vec();
                    let mut boundary = Vec::new();
                    let mut ext = Vec::new();
                    let origin = leaf.origin();
                    for (n, slot) in buf.iter_mut().enumerate() {
                        let value = *leaf.get_value_at(n);
                        let is_above = value > iso;
                        if !leaf.is_value_on_at(n) {
                            *slot = if is_above { above } else { -above };
                            continue;
                        }
                        let ijk = origin + LeafNode::<T>::offset_to_local_coord(n);
                        st.move_to_with_center(&ijk, value);
                        let mask = st.intersection_mask(iso);
                        if mask == 0 {
                            *slot = if is_above { above } else { -above };
                            continue;
                        }
                        boundary.push(ijk);
                        let delta = value - iso;
                        if delta.abs() <= T::tolerance() {
                            *slot = z;
                            if let Some(op) = op {
                                ext.push((li, n, op(xform.index_to_world(ijk.as_vec3d()))));
                            }
                            continue;
                        }
                        let vals = st.values();
                        let mut sum1 = z;
                        let mut sum2: Option<E> = None;
                        let mut min_d = max;
                        let mut nn_idx = 0usize;
                        let mut i = 0usize;
                        while i < 6 {
                            let mut d = max;
                            if mask & (1 << i) != 0 {
                                d = (delta / (value - vals[i + 1])).abs();
                                nn_idx = i;
                            }
                            i += 1;
                            if mask & (1 << i) != 0 {
                                let d2 = (delta / (value - vals[i + 1])).abs();
                                if d2 < d {
                                    d = d2;
                                    nn_idx = i;
                                }
                            }
                            i += 1;
                            if d < max {
                                let w = T::from_f64(1.0) / (d * d);
                                sum1 += w;
                                if let Some(op) = op {
                                    let o = OFFSETS[nn_idx];
                                    let p = Vec3d::new(
                                        (T::from_f64(ijk[0] as f64) + d * T::from_f64(o[0] as f64)).to_f64(),
                                        (T::from_f64(ijk[1] as f64) + d * T::from_f64(o[1] as f64)).to_f64(),
                                        (T::from_f64(ijk[2] as f64) + d * T::from_f64(o[2] as f64)).to_f64(),
                                    );
                                    let e = op(xform.index_to_world(p));
                                    sum2 = Some(if E::NEAREST {
                                        if d < min_d {
                                            e
                                        } else {
                                            sum2.unwrap_or(E::zero())
                                        }
                                    } else {
                                        match sum2 {
                                            Some(s) => E::lin2(1.0, s, w.to_f64(), e),
                                            None => E::scaled(w.to_f64(), e),
                                        }
                                    });
                                }
                                if d < min_d {
                                    min_d = d;
                                }
                            }
                        }
                        if op.is_some() {
                            let s2 = sum2.unwrap_or(E::zero());
                            let v = if E::NEAREST { s2 } else { E::scaled((T::from_f64(1.0) / sum1).to_f64(), s2) };
                            ext.push((li, n, v));
                        }
                        let r = h / sum1.sqrt();
                        *slot = if is_above { r } else { -r };
                    }
                    (buf, boundary, ext)
                },
            )
            .collect();
        let mut bufs = Vec::with_capacity(results.len());
        let mut boundary = Vec::new();
        let mut ext = Vec::new();
        for (b, bd, e) in results {
            bufs.push(b);
            boundary.extend(bd);
            ext.extend(e);
        }
        (bufs, boundary, ext)
    }

    fn finish_init(&mut self, iso: T, bufs: Vec<Vec<T>>, boundary: Vec<Coord>, has_active_tiles: bool) {
        let max = T::max_value();
        let sign = if self.is_input_sdf { T::from_f64(1.0) } else { T::from_f64(-1.0) };
        let above = sign * max;
        let sdf = self.sdf.as_mut().expect("initialized");
        super::leaf_buffers::write_leaf_buffers(sdf.tree_mut(), &bufs);
        {
            let mut acc = self.sweep_mask.accessor_mut();
            for c in boundary {
                acc.set_active_state(&c, false);
            }
        }
        sdf.tree_mut().visit_tiles_mut(&mut |_, v, _, _| *v = if *v > iso { above } else { -above });
        sdf.tree_mut().root_mut().set_background_unchecked(max);
        if has_active_tiles {
            sdf.tree_mut().voxelize_active_tiles();
        }
    }

    /// Initialize from a fog volume (`is_input_sdf = false`) or a SDF, with the distance
    /// measured to the `iso` surface. Returns [`is_valid`](Self::is_valid).
    pub fn init_sdf(&mut self, grid: &Grid<T>, iso: T, is_input_sdf: bool) -> Result<bool, FastSweepingError> {
        self.clear();
        let has_active_tiles = grid.tree().active_tile_count() > 0;
        if is_input_sdf && has_active_tiles {
            return Err(FastSweepingError::SdfWithActiveTiles);
        }
        self.sdf = Some(grid.clone());
        self.is_input_sdf = is_input_sdf;
        self.sweep_mask = Tree::new(false);
        self.sweep_mask.topology_union(grid.tree());
        let (bufs, boundary, _) = self.init_boundary::<fn(Vec3d) -> E>(iso, None);
        self.finish_init(iso, bufs, boundary, has_active_tiles);
        self.compute_sweep_mask_leaf_origins();
        Ok(self.is_valid())
    }

    /// Initialize an extension: `op` gives the field value at world positions on the iso-surface;
    /// `background` is the extension grid's background. For one-sided modes, `ext_grid` provides
    /// the values on the other side.
    #[allow(clippy::too_many_arguments)]
    pub fn init_ext<F>(
        &mut self,
        grid: &Grid<T>,
        op: &F,
        background: E,
        iso: T,
        is_input_sdf: bool,
        mode: FastSweepingDomain,
        ext_grid: Option<&Grid<E>>,
    ) -> Result<bool, FastSweepingError>
    where
        F: Fn(Vec3d) -> E + Sync,
    {
        if mode != FastSweepingDomain::SweepAll {
            match ext_grid {
                Some(e) if e.transform() == grid.transform() => {}
                _ => return Err(FastSweepingError::MissingExtensionGrid),
            }
        }
        self.clear();
        let has_active_tiles = grid.tree().active_tile_count() > 0;
        if is_input_sdf && has_active_tiles {
            return Err(FastSweepingError::SdfWithActiveTiles);
        }
        self.sdf = Some(grid.clone());
        self.mode = mode;
        self.is_input_sdf = is_input_sdf;
        if mode != FastSweepingDomain::SweepAll {
            self.ext_input = ext_grid.cloned();
        }
        let mut ext = Grid::new(background).with_transform(grid.transform().clone());
        ext.tree_mut().topology_union(grid.tree());
        self.sweep_mask = Tree::new(false);
        self.sweep_mask.topology_union(grid.tree());
        let (bufs, boundary, ext_vals) = self.init_boundary(iso, Some(op));
        {
            let origins: Vec<Coord> = grid.tree().leaves().map(|l| l.origin()).collect();
            let mut acc = ext.accessor_mut();
            for (li, n, v) in ext_vals {
                let c = origins[li] + LeafNode::<E>::offset_to_local_coord(n);
                acc.set_value_only(&c, v);
            }
        }
        self.finish_init(iso, bufs, boundary, has_active_tiles);
        if has_active_tiles {
            ext.tree_mut().voxelize_active_tiles();
        }
        self.ext = Some(ext);
        self.compute_sweep_mask_leaf_origins();
        Ok(self.is_valid())
    }

    /// Initialize the dilation of a SDF by `dilation` voxels.
    pub fn init_dilate(
        &mut self,
        grid: &Grid<T>,
        dilation: usize,
        nn: NearestNeighbors,
        mode: FastSweepingDomain,
    ) -> Result<bool, FastSweepingError> {
        self.clear();
        let mut sdf = grid.clone();
        self.mode = mode;
        let background = *grid.background();
        let unknown = T::max_value();
        set_level_set_background(sdf.tree_mut(), unknown, -unknown, true);
        dilate_active_values(sdf.tree_mut(), dilation as i32, nn, TilePolicy::IgnoreTiles, true);
        self.sweep_mask = Tree::new(false);
        self.sweep_mask.topology_union(sdf.tree());
        let input = grid.tree();
        let mut mask_off = Vec::new();
        let mut deactivate = Vec::new();
        let mut set_vals = Vec::new();
        for leaf in sdf.tree().leaves() {
            for (ijk, value) in leaf.iter_on_coords() {
                if value.abs() < background {
                    mask_off.push(ijk);
                    continue;
                }
                let swept = match mode {
                    FastSweepingDomain::SweepAll => true,
                    FastSweepingDomain::SweepGreaterThanIsoValue => value > <T as Value>::zero(),
                    FastSweepingDomain::SweepLessThanIsoValue => value < <T as Value>::zero(),
                };
                if swept {
                    set_vals.push((ijk, if value > <T as Value>::zero() { unknown } else { -unknown }));
                } else {
                    mask_off.push(ijk);
                    let (iv, on) = input.probe_value(&ijk);
                    if on {
                        set_vals.push((ijk, iv));
                    } else {
                        deactivate.push(ijk);
                    }
                }
            }
        }
        {
            let mut acc = sdf.accessor_mut();
            for (c, v) in set_vals {
                acc.set_value_only(&c, v);
            }
            for c in deactivate {
                acc.set_active_state(&c, false);
            }
        }
        {
            let mut acc = self.sweep_mask.accessor_mut();
            for c in mask_off {
                acc.set_active_state(&c, false);
            }
        }
        self.sdf = Some(sdf);
        self.compute_sweep_mask_leaf_origins();
        Ok(self.is_valid())
    }

    /// Initialize the extension of a SDF into the active voxels of `mask` (for a level-set
    /// mask, into its interior).
    pub fn init_mask<M: Value + PartialOrd>(
        &mut self,
        grid: &Grid<T>,
        mask: &Grid<M>,
        ignore_active_tiles: bool,
    ) -> Result<bool, FastSweepingError> {
        self.clear();
        if grid.transform() != mask.transform() {
            return Err(FastSweepingError::MisalignedMask);
        }
        let mut sdf = grid.clone();
        let unknown = T::max_value();
        set_level_set_background(sdf.tree_mut(), unknown, -unknown, true);
        // The topology to sweep into.
        let mut topo = Tree::new(false);
        if mask.grid_class() == GridClass::LevelSet {
            // Interior of the mask level set: all voxels (active or not) with negative values.
            let z = M::zero();
            for leaf in mask.tree().leaves() {
                let l = topo.touch_leaf(&leaf.origin());
                for (n, v, _) in leaf.iter_all() {
                    if *v < z {
                        l.set_value_on_at(n, true);
                    }
                }
            }
            mask.tree().visit_tiles(&mut |b, v, _, level| {
                if *v < z {
                    topo.add_tile(level, &b.min, true, true);
                }
            });
            topo.voxelize_active_tiles();
        } else {
            topo.topology_union(mask.tree());
            if !ignore_active_tiles {
                topo.voxelize_active_tiles();
            }
        }
        sdf.tree_mut().topology_union(&topo);
        self.sweep_mask = Tree::new(false);
        self.sweep_mask.topology_union(sdf.tree());
        let mut off = Vec::new();
        for leaf in self.sweep_mask.leaves() {
            for n in leaf.value_mask().iter_on() {
                let c = leaf.origin() + LeafNode::<bool>::offset_to_local_coord(n);
                if sdf.tree().get_value(&c).abs() < unknown {
                    off.push(c);
                }
            }
        }
        {
            let mut acc = self.sweep_mask.accessor_mut();
            for c in off {
                acc.set_active_state(&c, false);
            }
        }
        self.sdf = Some(sdf);
        self.compute_sweep_mask_leaf_origins();
        Ok(self.is_valid())
    }

    fn voxel_slices(&self, hash: impl Fn(&Coord) -> i64 + Sync) -> BTreeMap<i64, Slice> {
        let per_leaf: Vec<Vec<(i64, usize, u16)>> = self
            .sweep_mask
            .leaf_refs()
            .par_iter()
            .enumerate()
            .map(|(li, leaf)| {
                let key0 = hash(&leaf.origin());
                leaf.value_mask()
                    .iter_on()
                    .map(|n| (key0 + hash(&LeafNode::<bool>::offset_to_local_coord(n)), li, n as u16))
                    .collect()
            })
            .collect();
        let mut map: BTreeMap<i64, Slice> = BTreeMap::new();
        for items in per_leaf {
            for (key, li, n) in items {
                let slice = map.entry(key).or_default();
                match slice.last_mut() {
                    Some((l, v)) if *l == li => v.push(n),
                    _ => slice.push((li, vec![n])),
                }
            }
        }
        map
    }

    /// Run `n_iter` iterations of the eight sweeps; if `finalize`, replace the remaining
    /// unknown values and set an asymmetric background from the extrema of the result.
    pub fn sweep(&mut self, n_iter: usize, finalize: bool) -> Result<(), FastSweepingError> {
        if self.sdf.is_none() {
            return Err(FastSweepingError::NotInitialized);
        }
        if self.ext.is_some() && self.mode != FastSweepingDomain::SweepAll && self.ext_input.is_none() {
            return Err(FastSweepingError::MissingExtensionGrid);
        }
        if self.boundary_voxels == 0 {
            return Err(FastSweepingError::NoBoundaryVoxels);
        }
        if self.sweeping_voxels == 0 {
            return Err(FastSweepingError::NoSweepingVoxels);
        }
        let hashes: [fn(&Coord) -> i64; 4] = [
            |a| a[0] as i64 + a[1] as i64 + a[2] as i64,
            |a| a[0] as i64 + a[1] as i64 - a[2] as i64,
            |a| a[0] as i64 - a[1] as i64 + a[2] as i64,
            |a| a[0] as i64 - a[1] as i64 - a[2] as i64,
        ];
        let slices: Vec<BTreeMap<i64, Slice>> = hashes.iter().map(|h| self.voxel_slices(h)).collect();
        for _ in 0..n_iter {
            for s in &slices {
                for slice in s.values() {
                    self.sweep_slice(slice);
                }
                for slice in s.values().rev() {
                    self.sweep_slice(slice);
                }
            }
        }
        if finalize {
            self.finalize();
        }
        Ok(())
    }

    fn sweep_slice(&mut self, slice: &Slice) {
        let sdf = self.sdf.as_ref().expect("initialized");
        let h = T::from_f64(sdf.voxel_size()[0]);
        let sqrt2h = T::from_f64(2.0).sqrt() * h;
        let mode = self.mode;
        let is_input_sdf = self.is_input_sdf;
        let origins = &self.leaf_origins;
        let ext = self.ext.as_ref();
        let ext_in = self.ext_input.as_ref();
        let z = <T as Value>::zero();
        let one = T::from_f64(1.0);
        let thousand = T::from_f64(1000.0);
        let compute = |acc: &crate::tree::ValueAccessor<'_, crate::tree::Internal2<T>>,
                       eacc: Option<&crate::tree::ValueAccessor<'_, crate::tree::Internal2<E>>>,
                       iacc: Option<&crate::tree::ValueAccessor<'_, crate::tree::Internal2<E>>>,
                       ijk: Coord|
         -> Option<(T, Option<E>)> {
            let nn = |i: usize| -> (T, usize) {
                let o = OFFSETS[i];
                (acc.get_value(&ijk.offset_by(o[0], o[1], o[2])).abs(), i)
            };
            let min2 = |a: (T, usize), b: (T, usize)| if b.0 < a.0 { b } else { a };
            let mut d1 = min2(nn(0), nn(1));
            let mut d2 = min2(nn(2), nn(3));
            let mut d3 = min2(nn(4), nn(5));
            if !(d1.0 < thousand || d2.0 < thousand || d3.0 < thousand) {
                return None;
            }
            let value = acc.get_value(&ijk);
            let sign = if value >= z { one } else { -one };
            let abs_v = value.abs();
            if d2.0 < d1.0 {
                std::mem::swap(&mut d1, &mut d2);
            }
            if d3.0 < d2.0 {
                std::mem::swap(&mut d2, &mut d3);
            }
            if d2.0 < d1.0 {
                std::mem::swap(&mut d1, &mut d2);
            }
            let at = |d: (T, usize)| {
                let o = OFFSETS[d.1];
                ijk.offset_by(o[0], o[1], o[2])
            };
            let choose = |new_value: T, ext_val: E| -> E {
                let inp = || iacc.map(|a| a.get_value(&ijk)).unwrap_or(ext_val);
                match mode {
                    FastSweepingDomain::SweepAll => ext_val,
                    FastSweepingDomain::SweepGreaterThanIsoValue => {
                        let keep = if is_input_sdf { new_value >= z } else { new_value <= z };
                        if keep {
                            ext_val
                        } else {
                            inp()
                        }
                    }
                    FastSweepingDomain::SweepLessThanIsoValue => {
                        let keep = if is_input_sdf { new_value <= z } else { new_value >= z };
                        if keep {
                            ext_val
                        } else {
                            inp()
                        }
                    }
                }
            };
            // One neighbour.
            let update = d1.0 + h;
            if update <= d2.0 {
                if update < abs_v {
                    let nv = sign * update;
                    let e = eacc.map(|a| choose(nv, a.get_value(&at(d1))));
                    return Some((nv, e));
                }
                return None;
            }
            // Two neighbours.
            if d2.0 <= sqrt2h + d1.0 {
                let dd = T::from_f64(2.0) * h * h - (d1.0 - d2.0) * (d1.0 - d2.0);
                let update = T::from_f64(0.5) * (d1.0 + d2.0 + dd.sqrt());
                if update > d2.0 && update <= d3.0 {
                    if update < abs_v {
                        let nv = sign * update;
                        let e = eacc.map(|a| {
                            let (a1, a2) = (d1.0 - update, d2.0 - update);
                            let w = one / (a1 + a2);
                            let (v1, v2) = (a.get_value(&at(d1)), a.get_value(&at(d2)));
                            let ev = if E::NEAREST {
                                if a1 < a2 {
                                    v1
                                } else {
                                    v2
                                }
                            } else {
                                E::lin2((w * a1).to_f64(), v1, (w * a2).to_f64(), v2)
                            };
                            choose(nv, ev)
                        });
                        return Some((nv, e));
                    }
                    return None;
                }
            }
            // Three neighbours.
            let d123 = d1.0 + d2.0 + d3.0;
            let dd = d123 * d123 - T::from_f64(3.0) * (d1.0 * d1.0 + d2.0 * d2.0 + d3.0 * d3.0 - h * h);
            if dd >= z {
                let update = T::from_f64(1.0 / 3.0) * (d123 + dd.sqrt());
                if update < abs_v {
                    let nv = sign * update;
                    let e = eacc.map(|a| {
                        let (a1, a2, a3) = (d1.0 - update, d2.0 - update, d3.0 - update);
                        let w = one / (a1 + a2 + a3);
                        let (v1, v2, v3) = (a.get_value(&at(d1)), a.get_value(&at(d2)), a.get_value(&at(d3)));
                        let ev = if E::NEAREST {
                            let m = [a1, a2, a3];
                            let mut idx = 0;
                            for k in 1..3 {
                                if m[k] < m[idx] {
                                    idx = k;
                                }
                            }
                            [v1, v2, v3][idx]
                        } else {
                            E::lin3((w * a1).to_f64(), v1, (w * a2).to_f64(), v2, (w * a3).to_f64(), v3)
                        };
                        choose(nv, ev)
                    });
                    return Some((nv, e));
                }
            }
            None
        };
        let run = |entries: &[(usize, Vec<u16>)]| -> Vec<(Coord, T, Option<E>)> {
            let acc = sdf.tree().accessor();
            let eacc = ext.map(|e| e.tree().accessor());
            let iacc = ext_in.map(|e| e.tree().accessor());
            let mut out = Vec::new();
            for (li, offs) in entries {
                let origin = origins[*li];
                for &n in offs {
                    let ijk = origin + LeafNode::<bool>::offset_to_local_coord(n as usize);
                    if let Some((v, e)) = compute(&acc, eacc.as_ref(), iacc.as_ref(), ijk) {
                        out.push((ijk, v, e));
                    }
                }
            }
            out
        };
        let updates: Vec<(Coord, T, Option<E>)> =
            if slice.len() > 8 { slice.par_chunks(4).flat_map_iter(run).collect() } else { run(slice) };
        if updates.is_empty() {
            return;
        }
        {
            let sdf = self.sdf.as_mut().expect("initialized");
            let mut acc = sdf.accessor_mut();
            for (c, v, _) in &updates {
                acc.set_value_only(c, *v);
            }
        }
        if let Some(ext) = self.ext.as_mut() {
            let mut acc = ext.accessor_mut();
            for (c, _, e) in updates {
                if let Some(e) = e {
                    acc.set_value_on(&c, e);
                }
            }
        }
    }

    fn finalize(&mut self) {
        let sdf = self.sdf.as_mut().expect("initialized");
        let max = T::max_value();
        let (mut lo, mut hi) = (max, -max);
        for leaf in sdf.tree().leaves() {
            for (_, &v) in leaf.iter_on() {
                if v < lo && v != -max {
                    lo = v;
                }
                if v > hi && v != max {
                    hi = v;
                }
            }
        }
        // OpenVDB always runs this pass (its "max exists" flag starts out true).
        let fix = |v: &mut T| {
            if *v == -max {
                *v = lo;
            }
            if *v == max {
                *v = hi;
            }
        };
        sdf.tree_mut().visit_tiles_mut(&mut |_, v, _, _| fix(v));
        sdf.tree_mut().par_for_each_leaf_mut(|leaf| leaf.for_each_on_mut(|_, v| fix(v)));
        set_level_set_background(sdf.tree_mut(), hi, lo, true);
    }
}

/// Signed distance field of the `iso` surface of a fog volume (density above `iso` is inside).
pub fn fog_to_sdf<T: RealValue>(fog: &Grid<T>, iso: T, n_iter: usize) -> Grid<T> {
    let mut fs: FastSweeping<T, f32> = FastSweeping::new();
    if let Ok(true) = fs.init_sdf(fog, iso, false) {
        let _ = fs.sweep(n_iter, true);
    }
    fs.take_sdf_grid().unwrap_or_else(|| fog.clone())
}

/// Re-computed signed distance field of the `iso` surface of a SDF.
pub fn sdf_to_sdf<T: RealValue>(sdf: &Grid<T>, iso: T, n_iter: usize) -> Grid<T> {
    let mut fs: FastSweeping<T, f32> = FastSweeping::new();
    if let Ok(true) = fs.init_sdf(sdf, iso, true) {
        let _ = fs.sweep(n_iter, true);
    }
    fs.take_sdf_grid().unwrap_or_else(|| sdf.clone())
}

#[allow(clippy::too_many_arguments)]
fn ext_impl<T: RealValue, E: ExtValue, F: Fn(Vec3d) -> E + Sync>(
    grid: &Grid<T>,
    op: &F,
    background: E,
    iso: T,
    n_iter: usize,
    mode: FastSweepingDomain,
    ext_grid: Option<&Grid<E>>,
    is_sdf: bool,
) -> Result<(Grid<T>, Grid<E>), FastSweepingError> {
    let mut fs: FastSweeping<T, E> = FastSweeping::new();
    if fs.init_ext(grid, op, background, iso, is_sdf, mode, ext_grid)? {
        fs.sweep(n_iter, true)?;
    }
    Ok((fs.take_sdf_grid().expect("initialized"), fs.take_ext_grid().expect("initialized")))
}

/// Extend the field `op` (evaluated on the `iso` surface of a fog volume) off the surface.
pub fn fog_to_ext<T: RealValue, E: ExtValue, F: Fn(Vec3d) -> E + Sync>(
    fog: &Grid<T>,
    op: &F,
    background: E,
    iso: T,
    n_iter: usize,
    mode: FastSweepingDomain,
    ext_grid: Option<&Grid<E>>,
) -> Result<Grid<E>, FastSweepingError> {
    Ok(ext_impl(fog, op, background, iso, n_iter, mode, ext_grid, false)?.1)
}

/// Extend the field `op` (evaluated on the `iso` surface of a SDF) off the surface.
pub fn sdf_to_ext<T: RealValue, E: ExtValue, F: Fn(Vec3d) -> E + Sync>(
    sdf: &Grid<T>,
    op: &F,
    background: E,
    iso: T,
    n_iter: usize,
    mode: FastSweepingDomain,
    ext_grid: Option<&Grid<E>>,
) -> Result<Grid<E>, FastSweepingError> {
    Ok(ext_impl(sdf, op, background, iso, n_iter, mode, ext_grid, true)?.1)
}

/// [`fog_to_sdf`] and [`fog_to_ext`] in one pass.
pub fn fog_to_sdf_and_ext<T: RealValue, E: ExtValue, F: Fn(Vec3d) -> E + Sync>(
    fog: &Grid<T>,
    op: &F,
    background: E,
    iso: T,
    n_iter: usize,
    mode: FastSweepingDomain,
    ext_grid: Option<&Grid<E>>,
) -> Result<(Grid<T>, Grid<E>), FastSweepingError> {
    ext_impl(fog, op, background, iso, n_iter, mode, ext_grid, false)
}

/// [`sdf_to_sdf`] and [`sdf_to_ext`] in one pass.
pub fn sdf_to_sdf_and_ext<T: RealValue, E: ExtValue, F: Fn(Vec3d) -> E + Sync>(
    sdf: &Grid<T>,
    op: &F,
    background: E,
    iso: T,
    n_iter: usize,
    mode: FastSweepingDomain,
    ext_grid: Option<&Grid<E>>,
) -> Result<(Grid<T>, Grid<E>), FastSweepingError> {
    ext_impl(sdf, op, background, iso, n_iter, mode, ext_grid, true)
}

/// Dilate the narrow band of a SDF by `dilation` voxels.
pub fn dilate_sdf<T: RealValue>(
    sdf: &Grid<T>,
    dilation: usize,
    nn: NearestNeighbors,
    n_iter: usize,
    mode: FastSweepingDomain,
) -> Grid<T> {
    let mut fs: FastSweeping<T, f32> = FastSweeping::new();
    if let Ok(true) = fs.init_dilate(sdf, dilation, nn, mode) {
        let _ = fs.sweep(n_iter, true);
    }
    fs.take_sdf_grid().unwrap_or_else(|| sdf.clone())
}

/// Extend a SDF into the active region (or the interior, for a level set) of `mask`.
pub fn mask_sdf<T: RealValue, M: Value + PartialOrd>(
    sdf: &Grid<T>,
    mask: &Grid<M>,
    ignore_active_tiles: bool,
    n_iter: usize,
) -> Result<Grid<T>, FastSweepingError> {
    let mut fs: FastSweeping<T, f32> = FastSweeping::new();
    if fs.init_mask(sdf, mask, ignore_active_tiles)? {
        fs.sweep(n_iter, true)?;
    }
    Ok(fs.take_sdf_grid().expect("initialized"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{CoordBBox, Transform};
    use crate::tools::test_util::{assert_close, make_sphere};

    fn fog_ball() -> Grid<f32> {
        let mut g = Grid::<f32>::new(0.0);
        for c in CoordBBox::from_coords(-14, -14, -14, 14, 14, 14) {
            let r = c.as_vec3d().length();
            if r < 12.0 {
                g.tree_mut().set_value(&c, (1.0 - r / 12.0) as f32);
            }
        }
        g
    }

    /// Reference values from `scratchpad/cppref/sweep.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let fog = fog_ball();
        let sdf = fog_to_sdf(&fog, 0.5f32, 1);
        let pts = [Coord::new(0, 0, 0), Coord::new(6, 0, 0), Coord::new(3, 4, -2), Coord::new(10, 2, 1)];
        let expect = [-5.19240665, 0.0, -0.78900969, 4.33128262];
        for (p, e) in pts.iter().zip(expect.iter()) {
            assert_close(sdf.get_value(p) as f64, *e, 1e-5);
        }
        assert_close(*sdf.background() as f64, 6.36866522, 1e-5);
        assert_eq!(sdf.active_voxel_count(), 7123);

        let ls = make_sphere::<f32>(6.0, Vec3d::new(0.3, -0.2, 0.1), 1.0, 3.0);
        let ext = sdf_to_ext(
            &ls,
            &|p: Vec3d| p[0] as f32 * 2.0 + p[2] as f32,
            0.0f32,
            0.0,
            1,
            FastSweepingDomain::SweepAll,
            None,
        )
        .unwrap();
        let expect = [12.5916185, 1.0, -5.94066191, 5.13673258];
        let epts = [Coord::new(6, 0, 0), Coord::new(0, 6, 1), Coord::new(-4, -4, 2), Coord::new(4, 3, -3)];
        for (p, e) in epts.iter().zip(expect.iter()) {
            assert_close(ext.get_value(p) as f64, *e, 1e-5);
        }
        assert_eq!(ext.active_voxel_count(), 2942);
        let dil = dilate_sdf(&ls, 3, NearestNeighbors::Face, 1, FastSweepingDomain::SweepAll);
        let dpts = [Coord::new(11, 0, 0), Coord::new(0, 0, 0), Coord::new(5, 5, 5), Coord::new(-9, 1, 0)];
        let expect = [4.70287323, -5.0583806, 2.55219269, 3.38363886];
        for (p, e) in dpts.iter().zip(expect.iter()) {
            assert_close(dil.get_value(p) as f64, *e, 1e-5);
        }
        assert_close(*dil.background() as f64, 5.98645782, 1e-5);
        assert_eq!(dil.active_voxel_count(), 6174);
        let re = sdf_to_sdf(&ls, 0.0f32, 1);
        let expect = [-0.29580918, 0.27593255, 0.0493059829, -0.19147633];
        for (p, e) in epts.iter().zip(expect.iter()) {
            assert_close(re.get_value(p) as f64, *e, 1e-5);
        }
        assert_close(*re.background() as f64, 3.31000662, 1e-5);
        assert_eq!(re.active_voxel_count(), ls.active_voxel_count());
    }

    #[test]
    fn fog_sdf_distances_are_metric() {
        let mut fog = fog_ball();
        fog.set_transform(Transform::linear(0.5));
        let sdf = fog_to_sdf(&fog, 0.5f32, 2);
        // The 0.5 iso-surface is the sphere of radius 6 voxels = 3 world units.
        let v = sdf.get_value(&Coord::new(10, 0, 0)) as f64;
        assert_close(v, 2.0, 0.1);
        let v = sdf.get_value(&Coord::new(0, 0, 0)) as f64;
        // First-order sweeping underestimates far from the surface; matches OpenVDB (-5.1924 voxels).
        assert_close(v, -5.19240665 * 0.5, 1e-4);
    }
}
