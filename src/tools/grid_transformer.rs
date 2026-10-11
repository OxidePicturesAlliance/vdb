//! Resampling of grids under affine and general transformations (port of
//! `openvdb/tools/GridTransformer.h`).
//!
//! * [`GridResampler`] resamples an input grid into an output grid through any [`Transformer`]
//!   (a map from input index space to output index space).
//! * [`GridTransformer`] applies an affine matrix (or a pivot/scale/rotate/translate
//!   decomposition) with mip-mapping for strong down-scaling.
//! * [`resample_to_match`] resamples a grid into the index space of another grid.
//!
//! Level sets are resampled with their active bounding box as clip region and then pruned and
//! sign-flood-filled. (OpenVDB's `resampleToMatch` rebuilds level sets through a mesh when the
//! transforms differ; that path depends on the meshing tools and is not used here.)

use super::internal::{prune_level_set, signed_flood_fill};
use super::interpolation::Sampler;
use crate::grid::{Grid, GridClass};
use crate::math::{Axis, Coord, CoordBBox, Mat3d, Mat4d, Transform, Vec3, Vec3d};
use crate::tree::{Node, Tree, ValueSource};
use crate::types::Value;
use rayon::prelude::*;

/// Value types that can be resampled. Level-set post-processing (pruning and signed flood
/// filling) is applied for floating-point scalars.
pub trait ResampleValue: Value {
    /// Restore the narrow band of a resampled level set (no-op for non-scalar types).
    fn finish_level_set(_tree: &mut Tree<Self>) {}
}
impl ResampleValue for f32 {
    fn finish_level_set(tree: &mut Tree<Self>) {
        prune_level_set(tree);
        signed_flood_fill(tree);
    }
}
impl ResampleValue for f64 {
    fn finish_level_set(tree: &mut Tree<Self>) {
        prune_level_set(tree);
        signed_flood_fill(tree);
    }
}
macro_rules! resample_plain {
    ($($t:ty),*) => {$( impl ResampleValue for $t {} )*};
}
resample_plain!(bool, i32, i64, Vec3<f32>, Vec3<f64>, Vec3<i32>);

/// A mapping from input index space to output index space.
pub trait Transformer: Sync {
    /// True if the mapping is affine (enables incremental evaluation).
    fn is_affine(&self) -> bool;
    /// Map an input index-space position to output index space.
    fn transform(&self, p: Vec3d) -> Vec3d;
    /// Map an output index-space position to input index space.
    fn inv_transform(&self, p: Vec3d) -> Vec3d;
}

/// An affine matrix and its inverse (`GridTransformer::MatrixTransform`).
#[derive(Clone, Copy, Debug)]
pub struct MatrixTransform {
    /// The matrix (row-vector convention).
    pub mat: Mat4d,
    /// Its inverse.
    pub inv_mat: Mat4d,
}

impl MatrixTransform {
    /// Wrap a matrix.
    pub fn new(mat: Mat4d) -> Self {
        MatrixTransform { mat, inv_mat: invert_affine(&mat).expect("singular transform matrix") }
    }
}

impl Transformer for MatrixTransform {
    fn is_affine(&self) -> bool {
        self.mat.is_affine()
    }
    fn transform(&self, p: Vec3d) -> Vec3d {
        self.mat.transform_h(p)
    }
    fn inv_transform(&self, p: Vec3d) -> Vec3d {
        self.inv_mat.transform_h(p)
    }
}

/// The mapping from the index space of transform A to that of transform B (`ABTransform`).
#[derive(Clone, Copy, Debug)]
pub struct ABTransform<'a> {
    a: &'a Transform,
    b: &'a Transform,
    affine: bool,
    identity: bool,
}

impl<'a> ABTransform<'a> {
    /// The mapping A-index → world → B-index.
    pub fn new(a: &'a Transform, b: &'a Transform) -> Self {
        let affine = a.is_linear() && b.is_linear();
        ABTransform { a, b, affine, identity: affine && a == b }
    }
    /// True if both transforms are identical linear maps.
    pub fn is_identity(&self) -> bool {
        self.identity
    }
    /// Transform A.
    pub fn a(&self) -> &Transform {
        self.a
    }
    /// Transform B.
    pub fn b(&self) -> &Transform {
        self.b
    }
}

impl Transformer for ABTransform<'_> {
    fn is_affine(&self) -> bool {
        self.affine
    }
    fn transform(&self, p: Vec3d) -> Vec3d {
        self.b.world_to_index(self.a.index_to_world(p))
    }
    fn inv_transform(&self, p: Vec3d) -> Vec3d {
        self.a.world_to_index(self.b.index_to_world(p))
    }
}

/// The affine matrix of a linear transform, reconstructed from its action on the unit vectors.
pub fn affine_matrix(t: &Transform) -> Mat4d {
    let o = t.index_to_world(Vec3d::zero());
    let mut m = Mat4d::identity();
    for i in 0..3 {
        let mut e = Vec3d::zero();
        e[i] = 1.0;
        let r = t.index_to_world(e) - o;
        for j in 0..3 {
            m.0[i][j] = r[j];
        }
    }
    for j in 0..3 {
        m.0[3][j] = o[j];
    }
    m
}

/// Invert an affine matrix with the same floating-point operation order as OpenVDB's
/// `Mat4::inverse` (block inversion via the 3×3 cofactors), so that resampling positions match
/// the C++ library bit for bit; falls back to Gauss-Jordan elimination otherwise.
#[allow(clippy::needless_range_loop)]
pub fn invert_affine(m: &Mat4d) -> Option<Mat4d> {
    let a = &m.0;
    let affine = a[0][3] == 0.0 && a[1][3] == 0.0 && a[2][3] == 0.0 && a[3][3] == 1.0;
    let m0011 = a[0][0] * a[1][1];
    let m0012 = a[0][0] * a[1][2];
    let m0110 = a[0][1] * a[1][0];
    let m0210 = a[0][2] * a[1][0];
    let m0120 = a[0][1] * a[2][0];
    let m0220 = a[0][2] * a[2][0];
    let det_a =
        m0011 * a[2][2] - m0012 * a[2][1] - m0110 * a[2][2] + m0210 * a[2][1] + m0120 * a[1][2] - m0220 * a[1][1];
    if !affine || det_a.abs() <= 1e-8 {
        return m.inverse(0.0);
    }
    let d = 1.0 / det_a;
    let mut inv = Mat4d::identity();
    let r = &mut inv.0;
    r[0][0] = d * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
    r[0][1] = d * (-a[0][1] * a[2][2] + a[0][2] * a[2][1]);
    r[0][2] = d * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);
    r[1][0] = d * (-a[1][0] * a[2][2] + a[1][2] * a[2][0]);
    r[1][1] = d * (a[0][0] * a[2][2] - m0220);
    r[1][2] = d * (m0210 - m0012);
    r[2][0] = d * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    r[2][1] = d * (m0120 - a[0][0] * a[2][1]);
    r[2][2] = d * (m0011 - m0110);
    for j in 0..3 {
        r[3][j] = -(a[3][0] * r[0][j] + a[3][1] * r[1][j] + a[3][2] * r[2][j]);
    }
    Some(inv)
}

/// Resamples grids through arbitrary [`Transformer`]s (`tools::GridResampler`).
#[derive(Clone, Copy, Debug)]
pub struct GridResampler {
    threaded: bool,
    transform_tiles: bool,
}

impl Default for GridResampler {
    fn default() -> Self {
        GridResampler { threaded: true, transform_tiles: true }
    }
}

/// A write produced while transforming a box: `(coord, value, active)`.
type Write<T> = (Coord, T, bool);

impl GridResampler {
    /// A threaded resampler that also transforms tiles.
    pub fn new() -> Self {
        Self::default()
    }
    /// Enable or disable multi-threading.
    pub fn set_threaded(&mut self, b: bool) {
        self.threaded = b;
    }
    /// True if multi-threaded.
    pub fn threaded(&self) -> bool {
        self.threaded
    }
    /// Enable or disable the transformation of tiles (ignored for level sets).
    pub fn set_transform_tiles(&mut self, b: bool) {
        self.transform_tiles = b;
    }
    /// True if tiles are transformed.
    pub fn transform_tiles(&self) -> bool {
        self.transform_tiles
    }

    /// Resample `input` into `output` through `xform` (input index → output index); the
    /// output's background is replaced by the input's.
    pub fn transform_grid<S, T, X>(&self, xform: &X, input: &Grid<T>, output: &mut Grid<T>)
    where
        S: Sampler<T> + Sync,
        T: ResampleValue,
        X: Transformer,
    {
        output.tree_mut().set_background(*input.background());
        self.apply_transform::<S, T, X>(xform, input, output);
    }

    pub(crate) fn apply_transform<S, T, X>(&self, xform: &X, input: &Grid<T>, output: &mut Grid<T>)
    where
        S: Sampler<T> + Sync,
        T: ResampleValue,
        X: Transformer,
    {
        let in_tree = input.tree();
        let level_set = input.grid_class() == GridClass::LevelSet;
        if !level_set && self.transform_tiles {
            let out_bg = *output.background();
            let mut tiles = Vec::new();
            in_tree.visit_tiles(&mut |bbox, v, on, _| {
                if on || !v.approx_eq(&out_bg, &T::zero()) {
                    tiles.push((bbox, *v, on));
                }
            });
            let jobs: Vec<(CoordBBox, Option<(T, bool)>)> =
                tiles.into_iter().map(|(b, v, on)| (b, if S::RADIUS > 0 { Some((v, on)) } else { None })).collect();
            self.run_jobs::<S, T, X>(xform, in_tree, output.tree_mut(), &jobs);
        }
        let clip = if level_set { input.eval_active_voxel_bbox() } else { CoordBBox::empty() };
        let jobs: Vec<(CoordBBox, Option<(T, bool)>)> = in_tree
            .leaves()
            .filter_map(|leaf| {
                let o = leaf.origin();
                let mut b = CoordBBox::new(o, o.offset(8));
                if !clip.is_empty() {
                    b = CoordBBox::new(b.min.max_elem(&clip.min), b.max.min_elem(&clip.max));
                }
                (!b.is_empty()).then_some((b, None))
            })
            .collect();
        self.run_jobs::<S, T, X>(xform, in_tree, output.tree_mut(), &jobs);
        if level_set {
            T::finish_level_set(output.tree_mut());
        }
    }

    /// Transform every box of `jobs` (in order); writes are computed in parallel batches and
    /// applied serially so that the result matches a serial evaluation.
    fn run_jobs<S, T, X>(
        &self,
        xform: &X,
        in_tree: &Tree<T>,
        out: &mut Tree<T>,
        jobs: &[(CoordBBox, Option<(T, bool)>)],
    ) where
        S: Sampler<T> + Sync,
        T: ResampleValue,
        X: Transformer,
    {
        const BATCH: usize = 64;
        for chunk in jobs.chunks(BATCH) {
            let compute = |(bbox, tile): &(CoordBBox, Option<(T, bool)>)| {
                let acc = in_tree.accessor();
                transform_bbox::<S, T, X, _>(xform, bbox, &acc, *tile)
            };
            let writes: Vec<Vec<Write<T>>> = if self.threaded {
                chunk.par_iter().map(compute).collect()
            } else {
                chunk.iter().map(compute).collect()
            };
            let mut acc = out.accessor_mut();
            for w in writes {
                for (c, v, on) in w {
                    if on {
                        acc.set_value_on(&c, v);
                    } else if !acc.is_value_on(&c) {
                        acc.set_value_off(&c, v);
                    }
                }
            }
        }
    }
}

/// Sample every output voxel whose pre-image may intersect `bbox` (`transformBBox`).
fn transform_bbox<S, T, X, A>(xform: &X, bbox: &CoordBBox, acc: &A, tile: Option<(T, bool)>) -> Vec<Write<T>>
where
    S: Sampler<T>,
    T: Value,
    X: Transformer,
    A: ValueSource<Value = T>,
{
    let in_min = bbox.min.as_vec3d();
    let in_max = bbox.max.as_vec3d() + Vec3d::splat(1.0);
    let (a, b) = (xform.transform(in_min), xform.transform(in_max));
    let mut out_min = a.min_elem(&b);
    let mut out_max = a.max_elem(&b);
    for i in 0..8 {
        let corner = Vec3d::new(
            if i & 1 != 0 { in_max[0] } else { in_min[0] },
            if i & 2 != 0 { in_max[1] } else { in_min[1] },
            if i & 4 != 0 { in_max[2] } else { in_min[2] },
        );
        let c = xform.transform(corner);
        out_min = out_min.min_elem(&c);
        out_max = out_max.max_elem(&c);
    }
    let r = S::RADIUS;
    let lo = Coord::new(out_min[0].floor() as i32 - r, out_min[1].floor() as i32 - r, out_min[2].floor() as i32 - r);
    let hi = Coord::new(out_max[0].ceil() as i32 + r, out_max[1].ceil() as i32 + r, out_max[2].ceil() as i32 + r);
    // Tile shortcut: inside the tile's box shrunk by the kernel radius, the value is constant.
    let tile_box = tile.map(|(v, on)| {
        let mn = bbox.min.as_vec3d() + Vec3d::splat(r as f64);
        let mx = bbox.max.as_vec3d() - Vec3d::splat(r as f64);
        (mn, mx, v, on)
    });
    let sample = |p: Vec3d| -> (T, bool) {
        if let Some((mn, mx, v, on)) = tile_box {
            let empty = (0..3).any(|i| mn[i] > mx[i]);
            if !empty && (0..3).all(|i| p[i] >= mn[i] && p[i] <= mx[i]) {
                return (v, on);
            }
        }
        S::probe_sample(acc, p)
    };
    let mut out = Vec::new();
    if !xform.is_affine() {
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let (v, on) = sample(xform.inv_transform(Vec3d::new(x as f64, y as f64, z as f64)));
                    out.push((Coord::new(x, y, z), v, on));
                }
            }
        }
    } else {
        let t = xform.inv_transform(Vec3d::zero());
        let dx = xform.inv_transform(Vec3d::new(1.0, 0.0, 0.0)) - t;
        let dy = xform.inv_transform(Vec3d::new(0.0, 1.0, 0.0)) - t;
        let dz = xform.inv_transform(Vec3d::new(0.0, 0.0, 1.0)) - t;
        let mut start_x = xform.inv_transform(lo.as_vec3d());
        for x in lo[0]..=hi[0] {
            let mut start_y = start_x;
            for y in lo[1]..=hi[1] {
                let mut p = start_y;
                for z in lo[2]..=hi[2] {
                    let (v, on) = sample(p);
                    out.push((Coord::new(x, y, z), v, on));
                    p += dz;
                }
                start_y += dy;
            }
            start_x += dx;
        }
    }
    out
}

/// Errors for invalid transform specifications.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GridTransformerError {
    /// The transform order string is not a permutation of "rst".
    #[error("invalid transform order ({0})")]
    InvalidTransformOrder(String),
    /// The rotation order string is not a permutation of "xyz".
    #[error("invalid rotation order ({0})")]
    InvalidRotationOrder(String),
}

/// Resample a grid under an affine transformation (`tools::GridTransformer`).
#[derive(Clone, Debug)]
pub struct GridTransformer {
    resampler: GridResampler,
    pivot: Vec3d,
    mip_levels: [i32; 3],
    transform: Mat4d,
    pre_scale: Mat4d,
    post_scale: Mat4d,
}

impl GridTransformer {
    /// A transformer for the matrix `xform` (input index space → output index space). If the
    /// matrix decomposes into scale, rotation and translation, down-scaling uses mip-mapping.
    pub fn from_matrix(xform: Mat4d) -> Self {
        let mut t = GridTransformer {
            resampler: GridResampler::default(),
            pivot: Vec3d::zero(),
            mip_levels: [0; 3],
            transform: xform,
            pre_scale: Mat4d::identity(),
            post_scale: Mat4d::identity(),
        };
        if let Some((scale, rotate, translate)) = decompose(&xform) {
            t.init(Vec3d::zero(), scale, rotate, translate, "rst", "zyx").expect("valid orders");
        }
        t
    }
    /// A transformer that scales, rotates (radians) and translates about `pivot`, composing the
    /// operations in `xform_order` (a permutation of "rst", default "tsr") and the rotations in
    /// `rot_order` (a permutation of "xyz", default "zyx").
    pub fn new(
        pivot: Vec3d,
        scale: Vec3d,
        rotate: Vec3d,
        translate: Vec3d,
        xform_order: &str,
        rot_order: &str,
    ) -> Result<Self, GridTransformerError> {
        let mut t = GridTransformer {
            resampler: GridResampler::default(),
            pivot: Vec3d::zero(),
            mip_levels: [0; 3],
            transform: Mat4d::identity(),
            pre_scale: Mat4d::identity(),
            post_scale: Mat4d::identity(),
        };
        t.init(pivot, scale, rotate, translate, xform_order, rot_order)?;
        Ok(t)
    }
    /// The overall transformation matrix.
    pub fn transform(&self) -> &Mat4d {
        &self.transform
    }
    /// The underlying resampler settings.
    pub fn resampler_mut(&mut self) -> &mut GridResampler {
        &mut self.resampler
    }

    fn init(
        &mut self,
        pivot: Vec3d,
        scale: Vec3d,
        rotate: Vec3d,
        translate: Vec3d,
        xform_order: &str,
        rot_order: &str,
    ) -> Result<(), GridTransformerError> {
        let xo: Vec<char> = xform_order.chars().collect();
        let ro: Vec<char> = rot_order.chars().collect();
        if xo.len() != 3 {
            return Err(GridTransformerError::InvalidTransformOrder(xform_order.into()));
        }
        if ro.len() != 3 {
            return Err(GridTransformerError::InvalidRotationOrder(rot_order.into()));
        }
        self.pivot = pivot;
        let mut scale_rem = scale;
        for i in 0..3 {
            let s = scale[i].abs();
            if s < 0.5 {
                self.mip_levels[i] = (-s.ln() / 2f64.ln()).floor() as i32;
                scale_rem[i] = scale[i] * (1i64 << self.mip_levels[i]) as f64;
            }
        }
        self.transform = Mat4d::identity();
        self.pre_scale = Mat4d::identity();
        self.post_scale = Mat4d::identity();
        let mut use_post = true;
        let (mut rpos, mut spos, mut tpos) = (3, 3, 3);
        for ix in (0..3).rev() {
            match xo[ix] {
                'r' => {
                    rpos = ix;
                    self.transform.pre_translate(pivot);
                    rem(self, use_post).pre_translate(pivot);
                    let (mut xp, mut yp, mut zp) = (3, 3, 3);
                    for ir in (0..3).rev() {
                        let (axis, angle) = match ro[ir] {
                            'x' => {
                                xp = ir;
                                (Axis::X, rotate[0])
                            }
                            'y' => {
                                yp = ir;
                                (Axis::Y, rotate[1])
                            }
                            'z' => {
                                zp = ir;
                                (Axis::Z, rotate[2])
                            }
                            _ => continue,
                        };
                        self.transform.pre_rotate(axis, angle);
                        rem(self, use_post).pre_rotate(axis, angle);
                    }
                    if xp > 2 || yp > 2 || zp > 2 {
                        return Err(GridTransformerError::InvalidRotationOrder(rot_order.into()));
                    }
                    self.transform.pre_translate(-pivot);
                    rem(self, use_post).pre_translate(-pivot);
                }
                's' => {
                    spos = ix;
                    self.transform.pre_translate(pivot);
                    self.transform.pre_scale(scale);
                    self.transform.pre_translate(-pivot);
                    let r = rem(self, use_post);
                    r.pre_translate(pivot);
                    r.pre_scale(scale_rem);
                    r.pre_translate(-pivot);
                    use_post = false;
                }
                't' => {
                    tpos = ix;
                    self.transform.pre_translate(translate);
                    rem(self, use_post).pre_translate(translate);
                }
                _ => {}
            }
        }
        if tpos > 2 || rpos > 2 || spos > 2 {
            return Err(GridTransformerError::InvalidTransformOrder(xform_order.into()));
        }
        Ok(())
    }

    /// Resample `input` into `output` (whose transform is left unchanged).
    pub fn transform_grid<S, T>(&self, input: &Grid<T>, output: &mut Grid<T>)
    where
        S: Sampler<T> + Sync,
        T: ResampleValue,
    {
        output.tree_mut().set_background(*input.background());
        if !S::MIPMAP || self.mip_levels == [0; 3] {
            let xform = MatrixTransform::new(self.transform);
            self.resampler.apply_transform::<S, T, _>(&xform, input, output);
            return;
        }
        let bg = *input.background();
        let mut first_pass = true;
        let mut temp = Grid::new(bg);
        if !self.pre_scale.is_approx_equal(&Mat4d::identity(), 1e-8) {
            first_pass = false;
            self.resampler.apply_transform::<S, T, _>(&MatrixTransform::new(self.pre_scale), input, &mut temp);
        }
        let mut count = self.mip_levels;
        while count != [0; 3] {
            let mut m = Mat4d::identity();
            m.set_translation(self.pivot);
            m.pre_scale(Vec3d::new(
                if count[0] != 0 { 0.5 } else { 1.0 },
                if count[1] != 0 { 0.5 } else { 1.0 },
                if count[2] != 0 { 0.5 } else { 1.0 },
            ));
            m.pre_translate(-self.pivot);
            let xform = MatrixTransform::new(m);
            if first_pass {
                first_pass = false;
                self.resampler.apply_transform::<S, T, _>(&xform, input, &mut temp);
            } else {
                let mut dest = Grid::new(bg);
                self.resampler.apply_transform::<S, T, _>(&xform, &temp, &mut dest);
                temp = dest;
            }
            count = count.map(|c| (c - 1).max(0));
        }
        if !self.post_scale.is_approx_equal(&Mat4d::identity(), 1e-8) {
            self.resampler.apply_transform::<S, T, _>(&MatrixTransform::new(self.post_scale), &temp, output);
        } else {
            output.set_tree(temp.into_tree());
        }
    }
}

fn rem(t: &mut GridTransformer, post: bool) -> &mut Mat4d {
    if post {
        &mut t.post_scale
    } else {
        &mut t.pre_scale
    }
}

/// XYZ Euler angles of a rotation matrix (`math::eulerAngles(mat, XYZ_ROTATION)`).
fn euler_xyz(m: &Mat3d) -> Vec3d {
    let eps = 1e-8;
    let m = &m.0;
    let (phi, theta, psi);
    if (m[2][0] - 1.0).abs() <= eps {
        theta = std::f64::consts::FRAC_PI_2;
        phi = 0.5 * m[1][2].atan2(m[1][1]);
        psi = phi;
    } else if (m[2][0] + 1.0).abs() <= eps {
        theta = -std::f64::consts::FRAC_PI_2;
        phi = 0.5 * m[1][2].atan2(m[1][1]);
        psi = -phi;
    } else {
        psi = (-m[1][0]).atan2(m[0][0]);
        phi = (-m[2][1]).atan2(m[2][2]);
        theta = m[2][0].atan2((m[2][1] * m[2][1] + m[2][2] * m[2][2]).sqrt());
    }
    Vec3d::new(phi, theta, psi)
}

/// Decompose an affine matrix into scale, XYZ rotation and translation
/// (`local_util::decompose`); `None` if no exact decomposition exists.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // mirrors the C++ NaN-tolerant comparison
fn decompose(m: &Mat4d) -> Option<(Vec3d, Vec3d, Vec3d)> {
    if !m.is_affine() {
        return None;
    }
    let translate = m.get_translation();
    let xform = m.get_mat3();
    let us = Vec3d::new(
        xform.transform(Vec3d::new(1.0, 0.0, 0.0)).length(),
        xform.transform(Vec3d::new(0.0, 1.0, 0.0)).length(),
        xform.transform(Vec3d::new(0.0, 0.0, 1.0)).length(),
    );
    let uniform = us.is_approx_equal(&Vec3d::splat(us[0]), 1e-7);
    let mut result = None;
    let mut min_angle = f64::MAX;
    for n in 0..8 {
        let ss = Vec3d::new(
            if n & 1 != 0 { -us[0] } else { us[0] },
            if n & 2 != 0 { -us[1] } else { us[1] },
            if n & 4 != 0 { -us[2] } else { us[2] },
        );
        let sm = Mat3d::from_diagonal(ss);
        let mat = xform * sm.inverse(0.0)?;
        if mat.det() < 0.0 {
            continue;
        }
        let a = euler_xyz(&mat);
        let rebuild =
            Mat3d::rotation(Axis::Z, a[2]) * Mat3d::rotation(Axis::Y, a[1]) * Mat3d::rotation(Axis::X, a[0]) * sm;
        if xform.is_approx_equal(&rebuild, 1e-8) {
            let max_angle = a[0].abs().max(a[1].abs()).max(a[2].abs());
            if !(min_angle < max_angle) {
                min_angle = max_angle;
                let has_rotation = !a.is_approx_equal(&Vec3d::zero(), 1e-7);
                result = Some((ss, a, translate));
                if uniform || !has_rotation {
                    break;
                }
            }
        }
    }
    result
}

/// Resample `input` into the index space of `output` (`tools::resampleToMatch`). If the
/// transforms are identical (and the sampler is consistent) the tree is copied.
pub fn resample_to_match<S, T>(input: &Grid<T>, output: &mut Grid<T>)
where
    S: Sampler<T> + Sync,
    T: ResampleValue,
{
    if input.grid_class() == GridClass::LevelSet && input.transform() == output.transform() {
        output.set_tree(input.tree().clone());
        return;
    }
    let ab = ABTransform::new(input.transform(), output.transform());
    if S::CONSISTENT && ab.is_identity() {
        output.set_tree(input.tree().clone());
    } else if ab.is_affine() {
        let a = affine_matrix(input.transform());
        let b = affine_matrix(output.transform());
        let mat = a * invert_affine(&b).expect("singular transform");
        GridTransformer::from_matrix(mat).transform_grid::<S, T>(input, output);
    } else {
        let xin = input.transform().clone();
        let xout = output.transform().clone();
        GridResampler::new().transform_grid::<S, T, _>(&ABTransform::new(&xin, &xout), input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::interpolation::{BoxSampler, PointSampler, QuadraticSampler};
    use crate::tools::test_util::{assert_close, make_sphere};

    fn fog() -> Grid<f32> {
        let mut g = Grid::<f32>::new(0.0);
        for c in CoordBBox::from_coords(-10, -10, -10, 10, 10, 10) {
            let r = c.as_vec3d().length();
            if r < 9.0 {
                g.tree_mut().set_value(&c, (1.0 - r / 9.0) as f32);
            }
        }
        g.tree_mut().fill(&CoordBBox::from_coords(16, 16, 16, 31, 31, 31), 0.5, true);
        g
    }

    /// Reference values from `scratchpad/cppref/xform.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let input = fog();
        let pts = [Coord::new(0, 0, 0), Coord::new(3, -2, 1), Coord::new(-4, 4, 2), Coord::new(14, 12, 13)];
        // Rotate + scale + translate.
        let gt = GridTransformer::new(
            Vec3d::zero(),
            Vec3d::new(1.5, 1.5, 1.5),
            Vec3d::new(0.3, 0.0, 0.5),
            Vec3d::new(2.0, 0.0, -1.0),
            "tsr",
            "zyx",
        )
        .unwrap();
        let mut out = Grid::<f32>::new(0.0);
        gt.transform_grid::<BoxSampler, f32>(&input, &mut out);
        let expect = [0.751547992, 0.674592316, 0.43317759, 0.0];
        for (p, e) in pts.iter().zip(expect.iter()) {
            assert_close(out.get_value(p) as f64, *e, 1e-5);
        }
        assert_eq!(out.active_voxel_count(), 29286);
        // Strong down-scaling (mip-mapping).
        let gt = GridTransformer::from_matrix(Mat4d::scale(Vec3d::splat(0.2)));
        let mut out = Grid::<f32>::new(0.0);
        gt.transform_grid::<QuadraticSampler, f32>(&input, &mut out);
        let mip_pts = [Coord::new(1, 0, 0), Coord::new(0, 1, 1), Coord::new(4, 4, 4), Coord::new(5, 5, 6)];
        let expect = [0.444444478, 0.236170754, 0.5, 0.3125];
        for (p, e) in mip_pts.iter().zip(expect.iter()) {
            assert_close(out.get_value(p) as f64, *e, 1e-5);
        }
        assert_close(out.get_value(&Coord::new(0, 0, 0)) as f64, 1.0, 1e-6);
        assert_eq!(out.active_voxel_count(), 340);
        // resampleToMatch between voxel sizes.
        let mut src = input.clone();
        src.set_transform(Transform::linear(0.1));
        let mut dst = Grid::<f32>::new(0.0).with_transform(Transform::linear(0.07));
        resample_to_match::<PointSampler, f32>(&src, &mut dst);
        let expect = [1.0, 0.727834463, 0.515677869, 0.0];
        for (p, e) in pts.iter().zip(expect.iter()) {
            assert_close(dst.get_value(p) as f64, *e, 1e-5);
        }
        assert_eq!(dst.active_voxel_count(), 20636);
    }

    #[test]
    fn level_set_resample_keeps_signs() {
        let ls = make_sphere::<f32>(1.0, Vec3d::zero(), 0.1, 3.0);
        let mut out = Grid::<f32>::new(0.3).with_transform(Transform::linear(0.07));
        out.set_grid_class(GridClass::LevelSet);
        resample_to_match::<BoxSampler, f32>(&ls, &mut out);
        assert_close(out.get_value(&Coord::new(0, 0, 0)) as f64, -0.3, 1e-6);
        assert_close(out.get_value(&Coord::new(40, 0, 0)) as f64, 0.3, 1e-6);
        let v = out.get_value(&Coord::new(14, 0, 0)) as f64; // world x = 0.98
        assert_close(v, -0.02, 5e-3);
    }

    #[test]
    fn decomposition() {
        let mut m = Mat4d::scale(Vec3d::new(2.0, 2.0, 2.0));
        m.post_rotate(Axis::Z, 0.4);
        m.post_translate(Vec3d::new(1.0, 2.0, 3.0));
        let (s, r, t) = decompose(&m).unwrap();
        assert!(s.is_approx_equal(&Vec3d::splat(2.0), 1e-9));
        assert!(r.is_approx_equal(&Vec3d::new(0.0, 0.0, 0.4), 1e-9));
        assert!(t.is_approx_equal(&Vec3d::new(1.0, 2.0, 3.0), 1e-9));
        let gt = GridTransformer::from_matrix(m);
        assert!(gt.transform().is_approx_equal(&m, 1e-9));
    }
}
