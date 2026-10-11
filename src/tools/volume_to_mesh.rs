//! Extract polygonal surfaces from scalar volumes (port of `openvdb/tools/VolumeToMesh.h`).
//!
//! The mesher is a dual-contouring style scheme: one point is generated per edge group of
//! every cell that straddles the isosurface, and one quad per sign-changing voxel edge.
//! With a non-zero adaptivity, coherent 2³, 4³ and 8³ blocks of cells inside each leaf
//! node are collapsed into a single point and the resulting degenerate quads are emitted as
//! triangles. All lookup tables are those of OpenVDB ([`super::volume_to_mesh_tables`]).
//!
//! Supported input value types are `f32`, `f64` and `bool` (see [`MeshValue`]).

use super::volume_to_mesh_tables::{ADAPTABLE, AMBIGUOUS_FACE, EDGE_GROUP_TABLE};
use crate::grid::{Grid, GridClass};
use crate::math::{Coord, CoordBBox, Transform, Vec3d, Vec3f};
use crate::tree::{Internal2, LeafNode, Node, Tree, ValueAccessor};
use crate::types::Value;
use crate::util::NodeMask;
use crate::util::INVALID_IDX;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// Polygon flag: the polygon is coincident with the reference surface.
pub const POLYFLAG_EXTERIOR: u8 = 0x1;
/// Polygon flag: the polygon is in proximity to a fracture seam line.
pub const POLYFLAG_FRACTURE_SEAM: u8 = 0x2;
/// Polygon flag: the polygon was produced by subdividing a nonplanar seam line quad.
pub const POLYFLAG_SUBDIVIDED: u8 = 0x4;

pub(crate) const SIGNS: i16 = 0xFF;
pub(crate) const EDGES: i16 = 0xE00;
pub(crate) const INSIDE: i16 = 0x100;
const XEDGE: i16 = 0x200;
const YEDGE: i16 = 0x400;
const ZEDGE: i16 = 0x800;
const SEAM: i16 = 0x1000;

const MASK_FIRST_10_BITS: u32 = 0x0000_03FF;
const MASK_DIRTY_BIT: u32 = 0x8000_0000;
const MASK_INVALID_BIT: u32 = 0x4000_0000;

/// Scalar value types that can be meshed.
pub trait MeshValue: Value {
    /// True for boolean grids (whose inside test ignores the isovalue).
    const IS_BOOL: bool;
    /// OpenVDB's `isInsideValue(value, iso)`.
    fn is_inside(self, iso: Self) -> bool;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    /// `D1<CD_2ND>::difference(xp1, xm1)` converted to `f32`.
    fn central_diff(xp1: Self, xm1: Self) -> f32;
}
impl MeshValue for f32 {
    const IS_BOOL: bool = false;
    #[inline]
    fn is_inside(self, iso: Self) -> bool {
        self < iso
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn central_diff(a: Self, b: Self) -> f32 {
        (a - b) * 0.5
    }
}
impl MeshValue for f64 {
    const IS_BOOL: bool = false;
    #[inline]
    fn is_inside(self, iso: Self) -> bool {
        self < iso
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn central_diff(a: Self, b: Self) -> f32 {
        ((a - b) * 0.5) as f32
    }
}
impl MeshValue for bool {
    const IS_BOOL: bool = true;
    #[inline]
    fn is_inside(self, _iso: Self) -> bool {
        self
    }
    #[inline]
    fn to_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
    #[inline]
    fn central_diff(a: Self, _b: Self) -> f32 {
        if a {
            1.0
        } else {
            0.0
        }
    }
}

/// A collection of quads and triangles with per-polygon flags (port of `PolygonPool`).
#[derive(Clone, Debug, Default)]
pub struct PolygonPool {
    pub quads: Vec<[u32; 4]>,
    pub triangles: Vec<[u32; 3]>,
    pub quad_flags: Vec<u8>,
    pub triangle_flags: Vec<u8>,
}

impl PolygonPool {
    pub fn num_quads(&self) -> usize {
        self.quads.len()
    }
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }
    pub fn quad(&self, n: usize) -> &[u32; 4] {
        &self.quads[n]
    }
    pub fn triangle(&self, n: usize) -> &[u32; 3] {
        &self.triangles[n]
    }
}

type Acc<'a, T> = ValueAccessor<'a, Internal2<T>>;

#[inline]
fn coord_and(c: &Coord, mask: i32) -> Coord {
    Coord::new(c[0] & mask, c[1] & mask, c[2] & mask)
}

// ---------------------------------------------------------------------------------------
// Feature point
// ---------------------------------------------------------------------------------------

type M3 = [[f64; 3]; 3];

fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

fn jacobi_pivot(i: usize, j: usize, s: &mut M3, d: &mut [f64; 3], q: &mut M3) {
    let tol = crate::math::TOLERANCE_F64;
    let sij = s[i][j];
    let sjj_minus_sii = d[j] - d[i];
    let tan_theta = if sjj_minus_sii.abs() * (10.0 * tol) > sij.abs() {
        sij / sjj_minus_sii
    } else {
        let cot2 = 0.5 * sjj_minus_sii / sij;
        if cot2 < 0.0 {
            -1.0 / ((1.0 + cot2 * cot2).sqrt() - cot2)
        } else {
            1.0 / ((1.0 + cot2 * cot2).sqrt() + cot2)
        }
    };
    let c = 1.0 / (1.0 + tan_theta * tan_theta).sqrt();
    let sn = c * tan_theta;
    let z = tan_theta * sij;
    s[i][j] = 0.0;
    d[i] -= z;
    d[j] += z;
    for k in 0..i {
        let t = s[k][i];
        s[k][i] = c * t - sn * s[k][j];
        s[k][j] = sn * t + c * s[k][j];
    }
    for k in (i + 1)..j {
        let t = s[i][k];
        s[i][k] = c * t - sn * s[k][j];
        s[k][j] = sn * t + c * s[k][j];
    }
    for k in (j + 1)..3 {
        let t = s[i][k];
        s[i][k] = c * t - sn * s[j][k];
        s[j][k] = sn * t + c * s[j][k];
    }
    for k in 0..3 {
        let t = q[k][i];
        q[k][i] = c * t - sn * q[k][j];
        q[k][j] = sn * t + c * q[k][j];
    }
}

/// `math::diagonalizeSymmetricMatrix` (Jacobi iterations with largest-pivot selection).
fn diagonalize_symmetric_matrix(input: &M3, max_iterations: u32) -> (M3, [f64; 3], bool) {
    let tol = crate::math::TOLERANCE_F64;
    let mut q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut s = *input;
    let mut d = [s[0][0], s[1][1], s[2][2]];
    let mut iterations = 0u32;
    loop {
        let mut er = 0.0;
        for i in 0..3 {
            for j in (i + 1)..3 {
                er += s[i][j].abs();
            }
        }
        if er.abs() < tol {
            return (q, d, true);
        }
        iterations += 1;
        let mut max_el = 0.0;
        let (mut ip, mut jp) = (0, 0);
        for i in 0..3 {
            for j in (i + 1)..3 {
                if d[i].abs() * (10.0 * tol) > s[i][j].abs() {
                    s[i][j] = 0.0;
                }
                if s[i][j].abs() > max_el {
                    max_el = s[i][j].abs();
                    ip = i;
                    jp = j;
                }
            }
        }
        jacobi_pivot(ip, jp, &mut s, &mut d, &mut q);
        if iterations >= max_iterations {
            return (q, d, false);
        }
    }
}

/// Given a set of tangent elements (`points` with corresponding `normals`), return the
/// intersection point of all tangent elements (`tools::findFeaturePoint`, after "Feature
/// Sensitive Surface Extraction from Volume Data", Kobbelt et al. 2001).
pub fn find_feature_point(points: &[Vec3d], normals: &[Vec3d]) -> Vec3d {
    let mut avg = Vec3d::zero();
    if points.is_empty() {
        return avg;
    }
    for p in points {
        avg += *p;
    }
    avg /= points.len() as f64;
    let (mut m00, mut m01, mut m02, mut m11, mut m12, mut m22) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut rhs = Vec3d::zero();
    for (p, n) in points.iter().zip(normals) {
        m00 += n[0] * n[0];
        m11 += n[1] * n[1];
        m22 += n[2] * n[2];
        m01 += n[0] * n[1];
        m02 += n[0] * n[2];
        m12 += n[1] * n[2];
        rhs += *n * n.dot(&(*p - avg));
    }
    let a = [[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]];
    let (ev, eval, _) = diagonalize_symmetric_matrix(&a, 300);
    let mut dm = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let tolerance = eval[0].abs().max(eval[1].abs()).max(eval[2].abs()) * 0.01;
    let mut clamped = 0;
    for i in 0..3 {
        if eval[i].abs() < tolerance {
            dm[i][i] = 0.0;
            clamped += 1;
        } else {
            dm[i][i] = 1.0 / eval[i];
        }
    }
    if clamped < 3 {
        let mut evt = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                evt[i][j] = ev[j][i];
            }
        }
        let p = mat_mul(&mat_mul(&ev, &dm), &evt);
        let r = Vec3d::new(
            p[0][0] * rhs[0] + p[0][1] * rhs[1] + p[0][2] * rhs[2],
            p[1][0] * rhs[0] + p[1][1] * rhs[1] + p[1][2] * rhs[2],
            p[2][0] * rhs[0] + p[2][1] * rhs[1] + p[2][2] * rhs[2],
        );
        return avg + r;
    }
    avg
}

// ---------------------------------------------------------------------------------------
// Cell utilities
// ---------------------------------------------------------------------------------------

fn is_planar_quad(p0: &Vec3d, p1: &Vec3d, p2: &Vec3d, p3: &Vec3d, epsilon: f64) -> bool {
    let mut normal = (*p2 - *p0).cross(&(*p1 - *p3));
    normal.normalize_eps(1.0e-7);
    let centroid = *p0 + *p1 + *p2 + *p3;
    let d = centroid.dot(&normal) * 0.25;
    for p in [p0, p1, p2, p3] {
        if (p.dot(&normal) - d).abs() > epsilon {
            return false;
        }
    }
    true
}

fn pack_point(v: &Vec3d) -> u32 {
    let mut data = 0u32;
    data |= ((v[0] * 1023.0) as u32 & MASK_FIRST_10_BITS) << 20;
    data |= ((v[1] * 1023.0) as u32 & MASK_FIRST_10_BITS) << 10;
    data |= (v[2] * 1023.0) as u32 & MASK_FIRST_10_BITS;
    data
}

fn unpack_point(mut data: u32) -> Vec3d {
    let z = (data & MASK_FIRST_10_BITS) as f64 * 0.0009775171;
    data >>= 10;
    let y = (data & MASK_FIRST_10_BITS) as f64 * 0.0009775171;
    data >>= 10;
    let x = (data & MASK_FIRST_10_BITS) as f64 * 0.0009775171;
    Vec3d::new(x, y, z)
}

#[inline]
fn is_internal_leaf_coord(ijk: &Coord) -> bool {
    ijk[0] < 7 && ijk[1] < 7 && ijk[2] < 7
}

#[inline]
fn cell_values_acc<T: MeshValue>(acc: &Acc<'_, T>, mut ijk: Coord) -> [T; 8] {
    let mut v = [T::default(); 8];
    v[0] = acc.get_value(&ijk);
    ijk[0] += 1;
    v[1] = acc.get_value(&ijk);
    ijk[2] += 1;
    v[2] = acc.get_value(&ijk);
    ijk[0] -= 1;
    v[3] = acc.get_value(&ijk);
    ijk[2] -= 1;
    ijk[1] += 1;
    v[4] = acc.get_value(&ijk);
    ijk[0] += 1;
    v[5] = acc.get_value(&ijk);
    ijk[2] += 1;
    v[6] = acc.get_value(&ijk);
    ijk[0] -= 1;
    v[7] = acc.get_value(&ijk);
    v
}

#[inline]
fn cell_values_leaf<T: MeshValue>(leaf: &LeafNode<T>, offset: usize) -> [T; 8] {
    let b = leaf.buffer();
    [
        b[offset],
        b[offset + 64],
        b[offset + 64 + 1],
        b[offset + 1],
        b[offset + 8],
        b[offset + 64 + 8],
        b[offset + 64 + 8 + 1],
        b[offset + 8 + 1],
    ]
}

#[inline]
fn cell_values<T: MeshValue>(leaf: Option<&LeafNode<T>>, acc: &Acc<'_, T>, offset: usize, origin: &Coord) -> [T; 8] {
    let local = LeafNode::<T>::offset_to_local_coord(offset);
    match leaf {
        Some(l) if is_internal_leaf_coord(&local) => cell_values_leaf(l, offset),
        _ => cell_values_acc(acc, *origin + local),
    }
}

#[inline]
fn to_f64_array<T: MeshValue>(v: &[T; 8]) -> [f64; 8] {
    v.map(|x| x.to_f64())
}

#[inline]
fn compute_sign_flags<T: MeshValue>(v: &[T; 8], iso: T) -> u8 {
    let mut s = 0u8;
    for (i, x) in v.iter().enumerate() {
        if x.is_inside(iso) {
            s |= 1 << i;
        }
    }
    s
}

#[inline]
fn eval_cell_signs<T: MeshValue>(acc: &Acc<'_, T>, ijk: &Coord, iso: T) -> u8 {
    compute_sign_flags(&cell_values_acc(acc, *ijk), iso)
}

fn correct_cell_signs<T: MeshValue>(signs: &mut u8, face: u8, acc: &Acc<'_, T>, mut ijk: Coord, iso: T) {
    let (axis, delta, expected) = match face {
        1 => (2, -1, 3),
        2 => (0, 1, 4),
        3 => (2, 1, 1),
        4 => (0, -1, 2),
        5 => (1, -1, 6),
        6 => (1, 1, 5),
        _ => return,
    };
    ijk[axis] += delta;
    if AMBIGUOUS_FACE[eval_cell_signs(acc, &ijk, iso) as usize] == expected {
        *signs = !*signs;
    }
}

fn is_non_manifold<T: MeshValue>(acc: &Acc<'_, T>, ijk: &Coord, iso: T, dim: i32) -> bool {
    let h = dim >> 1;
    let inside = |c: Coord| acc.get_value(&c).is_inside(iso);
    let (i, j, k) = (ijk[0], ijk[1], ijk[2]);
    let p = [
        inside(Coord::new(i, j, k)),
        inside(Coord::new(i + dim, j, k)),
        inside(Coord::new(i + dim, j, k + dim)),
        inside(Coord::new(i, j, k + dim)),
        inside(Coord::new(i, j + dim, k)),
        inside(Coord::new(i + dim, j + dim, k)),
        inside(Coord::new(i + dim, j + dim, k + dim)),
        inside(Coord::new(i, j + dim, k + dim)),
    ];
    let mut signs = 0usize;
    for (n, &b) in p.iter().enumerate() {
        if b {
            signs |= 1 << n;
        }
    }
    if !ADAPTABLE[signs] {
        return true;
    }
    let (ip, ipp) = (i + h, i + dim);
    let (jp, jpp) = (j + h, j + dim);
    let (kp, kpp) = (k + h, k + dim);
    // edges
    let edges: [(Coord, usize, usize); 12] = [
        (Coord::new(ip, j, k), 0, 1),
        (Coord::new(ipp, j, kp), 1, 2),
        (Coord::new(ip, j, kpp), 2, 3),
        (Coord::new(i, j, kp), 0, 3),
        (Coord::new(ip, jpp, k), 4, 5),
        (Coord::new(ipp, jpp, kp), 5, 6),
        (Coord::new(ip, jpp, kpp), 6, 7),
        (Coord::new(i, jpp, kp), 7, 4),
        (Coord::new(i, jp, k), 0, 4),
        (Coord::new(ipp, jp, k), 1, 5),
        (Coord::new(ipp, jp, kpp), 2, 6),
        (Coord::new(i, jp, kpp), 3, 7),
    ];
    for (c, a, b) in edges.iter() {
        let m = inside(*c);
        if p[*a] != m && p[*b] != m {
            return true;
        }
    }
    let faces: [(Coord, [usize; 4]); 6] = [
        (Coord::new(ip, jp, k), [0, 1, 4, 5]),
        (Coord::new(ipp, jp, kp), [1, 2, 5, 6]),
        (Coord::new(ip, jp, kpp), [2, 3, 6, 7]),
        (Coord::new(i, jp, kp), [0, 3, 4, 7]),
        (Coord::new(ip, j, kp), [0, 1, 2, 3]),
        (Coord::new(ip, jpp, kp), [4, 5, 6, 7]),
    ];
    for (c, idx) in faces.iter() {
        let m = inside(*c);
        if idx.iter().all(|&n| p[n] != m) {
            return true;
        }
    }
    let m = inside(Coord::new(ip, jp, kp));
    p.iter().all(|&b| b != m)
}

#[inline]
fn eval_zero_crossing(v0: f64, v1: f64, iso: f64) -> f64 {
    (iso - v0) / (v1 - v0)
}

/// Edge descriptors: for edge `e` (1..=12), the corner pair and the fixed coordinates
/// (`None` marks the interpolated axis).
const EDGE_DEFS: [(usize, usize, [Option<f64>; 3]); 12] = [
    (0, 1, [None, Some(0.0), Some(0.0)]),
    (1, 2, [Some(1.0), Some(0.0), None]),
    (3, 2, [None, Some(0.0), Some(1.0)]),
    (0, 3, [Some(0.0), Some(0.0), None]),
    (4, 5, [None, Some(1.0), Some(0.0)]),
    (5, 6, [Some(1.0), Some(1.0), None]),
    (7, 6, [None, Some(1.0), Some(1.0)]),
    (4, 7, [Some(0.0), Some(1.0), None]),
    (0, 4, [Some(0.0), None, Some(0.0)]),
    (1, 5, [Some(1.0), None, Some(0.0)]),
    (2, 6, [Some(1.0), None, Some(1.0)]),
    (3, 7, [Some(0.0), None, Some(1.0)]),
];

#[inline]
fn edge_sample(values: &[f64; 8], e: usize, iso: f64) -> Vec3d {
    let (a, b, fixed) = EDGE_DEFS[e];
    let mut p = Vec3d::zero();
    for axis in 0..3 {
        p[axis] = match fixed[axis] {
            Some(v) => v,
            None => eval_zero_crossing(values[a], values[b], iso),
        };
    }
    p
}

/// Average cell point for a given edge group (`computePoint`). The accumulation order is
/// the same as in OpenVDB (component-wise additions in edge order, zero terms skipped).
fn compute_point(values: &[f64; 8], signs: u8, edge_group: u8, iso: f64) -> Vec3d {
    let mut avg = Vec3d::zero();
    let mut samples = 0;
    let row = &EDGE_GROUP_TABLE[signs as usize];
    for e in 0..12 {
        if row[e + 1] == edge_group {
            add_edge_sample(&mut avg, values, e, iso);
            samples += 1;
        }
    }
    if samples > 1 {
        let w = 1.0 / samples as f64;
        avg *= w;
    }
    avg
}

/// Adds the coordinates of edge sample `e` to `avg` exactly as OpenVDB does: only the
/// non-zero fixed coordinates (1.0) and the interpolated coordinate are added.
#[inline]
fn add_edge_sample(avg: &mut Vec3d, values: &[f64; 8], e: usize, iso: f64) {
    let (a, b, fixed) = EDGE_DEFS[e];
    for axis in 0..3 {
        match fixed[axis] {
            Some(v) => {
                if v != 0.0 {
                    avg[axis] += v;
                }
            }
            None => avg[axis] += eval_zero_crossing(values[a], values[b], iso),
        }
    }
}

fn compute_masked_point(values: &[f64; 8], signs: u8, signs_mask: u8, edge_group: u8, iso: f64) -> (Vec3d, i32) {
    let mut avg = Vec3d::zero();
    let mut samples = 0;
    let row = &EDGE_GROUP_TABLE[signs as usize];
    let mrow = &EDGE_GROUP_TABLE[signs_mask as usize];
    for e in 0..12 {
        if row[e + 1] == edge_group && mrow[e + 1] == 0 {
            add_edge_sample(&mut avg, values, e, iso);
            samples += 1;
        }
    }
    if samples > 1 {
        let w = 1.0 / samples as f64;
        avg *= w;
    }
    (avg, samples)
}

fn compute_weighted_point(p: &Vec3d, values: &[f64; 8], signs: u8, edge_group: u8, iso: f64) -> Vec3d {
    let row = &EDGE_GROUP_TABLE[signs as usize];
    let samples: Vec<Vec3d> =
        (0..12).filter(|&e| row[e + 1] == edge_group).map(|e| edge_sample(values, e, iso)).collect();
    if samples.len() == 1 {
        return samples[0];
    }
    let mut weights: Vec<f64> = samples.iter().map(|s| (*s - *p).length_sqr()).collect();
    let mut min_w = weights[0];
    let mut max_w = weights[0];
    for w in &weights[1..] {
        min_w = min_w.min(*w);
        max_w = max_w.max(*w);
    }
    let offset = max_w + min_w * 0.1;
    for w in weights.iter_mut() {
        *w = offset - *w;
    }
    let sum: f64 = weights.iter().fold(0.0, |a, b| a + b);
    let mut avg = Vec3d::zero();
    for (s, w) in samples.iter().zip(&weights) {
        avg += *s * (*w / sum);
    }
    avg
}

fn compute_cell_points(points: &mut [Vec3d; 4], values: &[f64; 8], signs: u8, iso: f64) -> usize {
    let n = EDGE_GROUP_TABLE[signs as usize][0] as usize;
    for (i, p) in points.iter_mut().enumerate().take(n) {
        *p = compute_point(values, signs, (i + 1) as u8, iso);
    }
    n
}

fn match_edge_group(group: u8, lhs: u8, rhs: u8) -> i32 {
    for i in 1..=12 {
        if EDGE_GROUP_TABLE[lhs as usize][i] == group && EDGE_GROUP_TABLE[rhs as usize][i] != 0 {
            return EDGE_GROUP_TABLE[rhs as usize][i] as i32;
        }
    }
    -1
}

#[allow(clippy::too_many_arguments)]
fn compute_cell_points_ref(
    points: &mut [Vec3d; 4],
    weighted: &mut [bool; 4],
    lhs_values: &[f64; 8],
    rhs_values: &[f64; 8],
    lhs_signs: u8,
    rhs_signs: u8,
    iso: f64,
    point_idx: usize,
    seam_points: &[u32],
) -> usize {
    let n = EDGE_GROUP_TABLE[lhs_signs as usize][0] as usize;
    for offset in 0..n {
        let id = match_edge_group((offset + 1) as u8, lhs_signs, rhs_signs);
        if id != -1 {
            let e = id as u8;
            let q = seam_points[point_idx + (id - 1) as usize];
            if (q & MASK_DIRTY_BIT) != 0 && (q & MASK_INVALID_BIT) == 0 {
                let p = unpack_point(q);
                points[offset] = compute_weighted_point(&p, rhs_values, rhs_signs, e, iso);
                weighted[offset] = true;
            } else {
                points[offset] = compute_point(rhs_values, rhs_signs, e, iso);
                weighted[offset] = false;
            }
        } else {
            points[offset] = compute_point(lhs_values, lhs_signs, (offset + 1) as u8, iso);
            weighted[offset] = false;
        }
    }
    n
}

// ---------------------------------------------------------------------------------------
// Surface intersecting voxels
// ---------------------------------------------------------------------------------------

struct VoxelOffsets {
    min_x: Vec<usize>,
    max_x: Vec<usize>,
    min_y: Vec<usize>,
    max_y: Vec<usize>,
    min_z: Vec<usize>,
    max_z: Vec<usize>,
    internal_x: Vec<usize>,
    internal_y: Vec<usize>,
    internal_z: Vec<usize>,
}

impl VoxelOffsets {
    fn new() -> Self {
        let mut o = VoxelOffsets {
            min_x: vec![],
            max_x: vec![],
            min_y: vec![],
            max_y: vec![],
            min_z: vec![],
            max_z: vec![],
            internal_x: vec![],
            internal_y: vec![],
            internal_z: vec![],
        };
        for x in 0..8usize {
            for y in 0..8usize {
                for z in 0..8usize {
                    let p = (x << 6) + (y << 3) + z;
                    if x < 7 {
                        o.internal_x.push(p);
                    }
                    if y < 7 {
                        o.internal_y.push(p);
                    }
                    if z < 7 {
                        o.internal_z.push(p);
                    }
                }
            }
        }
        for y in 0..8usize {
            for z in 0..8usize {
                o.min_x.push((y << 3) + z);
                o.max_x.push((7 << 6) + (y << 3) + z);
            }
        }
        // Note: OpenVDB's min/max Y face lists skip z == 7; reproduced for identical output.
        for x in 0..8usize {
            for z in 0..7usize {
                o.min_y.push((x << 6) + z);
                o.max_y.push((x << 6) + (7 << 3) + z);
            }
        }
        for x in 0..8usize {
            for y in 0..8usize {
                o.min_z.push((x << 6) + (y << 3));
                o.max_z.push((x << 6) + (y << 3) + 7);
            }
        }
        o
    }
}

#[inline]
fn edge_set(out: &mut Vec<Coord>, axis: usize, mut ijk: Coord) {
    out.push(ijk);
    match axis {
        0 => {
            ijk[1] -= 1;
            out.push(ijk);
            ijk[2] -= 1;
            out.push(ijk);
            ijk[1] += 1;
            out.push(ijk);
        }
        1 => {
            ijk[2] -= 1;
            out.push(ijk);
            ijk[0] -= 1;
            out.push(ijk);
            ijk[2] += 1;
            out.push(ijk);
        }
        _ => {
            ijk[1] -= 1;
            out.push(ijk);
            ijk[0] -= 1;
            out.push(ijk);
            ijk[1] += 1;
            out.push(ijk);
        }
    }
}

fn identify_leaf_edges<T: MeshValue>(
    leaf: &LeafNode<T>,
    acc: &Acc<'_, T>,
    offs: &VoxelOffsets,
    iso: T,
    out: &mut Vec<Coord>,
) {
    let buf = leaf.buffer();
    let mask = leaf.value_mask();
    // internal edges
    for axis in 0..3 {
        let (nvo, list) = match axis {
            0 => (64, &offs.internal_x),
            1 => (8, &offs.internal_y),
            _ => (1, &offs.internal_z),
        };
        for &pos in list {
            let active = mask.is_on(pos) || mask.is_on(pos + nvo);
            if active && buf[pos].is_inside(iso) != buf[pos + nvo].is_inside(iso) {
                edge_set(out, axis, leaf.offset_to_global_coord(pos));
            }
        }
    }
    // external edges (+ faces)
    for axis in 0..3 {
        let (lhs, rhs) = match axis {
            0 => (&offs.max_x, &offs.min_x),
            1 => (&offs.max_y, &offs.min_y),
            _ => (&offs.max_z, &offs.min_z),
        };
        let mut ijk = leaf.origin();
        ijk[axis] += 8;
        if let Some(rn) = acc.probe_leaf(&ijk) {
            let rb = rn.buffer();
            for (n, &pos) in lhs.iter().enumerate() {
                let r = rhs[n];
                let active = mask.is_on(pos) || rn.value_mask().is_on(r);
                if active && buf[pos].is_inside(iso) != rb[r].is_inside(iso) {
                    edge_set(out, axis, leaf.offset_to_global_coord(pos));
                }
            }
        } else {
            let (value, on) = acc.probe_value(&ijk);
            if !on {
                let inside = value.is_inside(iso);
                for &pos in lhs.iter() {
                    if mask.is_on(pos) && inside != buf[pos].is_inside(iso) {
                        edge_set(out, axis, leaf.offset_to_global_coord(pos));
                    }
                }
            }
        }
    }
    // external edges (- faces), only when the neighbour is an inactive tile
    for axis in 0..3 {
        let mut ijk = leaf.origin();
        ijk[axis] -= 1;
        if acc.probe_leaf(&ijk).is_some() {
            continue;
        }
        let (value, on) = acc.probe_value(&ijk);
        if on {
            continue;
        }
        let list = match axis {
            0 => &offs.internal_x,
            1 => &offs.internal_y,
            _ => &offs.internal_z,
        };
        let inside = value.is_inside(iso);
        for &pos in list {
            if mask.is_on(pos) && inside != buf[pos].is_inside(iso) {
                let mut c = leaf.offset_to_global_coord(pos);
                c[axis] -= 1;
                edge_set(out, axis, c);
            }
        }
    }
}

fn mask_active_tile_borders<T: MeshValue>(tree: &Tree<T>, iso: T, mask: &mut Tree<bool>) {
    let mut tiles = Vec::new();
    tree.visit_tiles(&mut |bbox, _, on, _| {
        if on {
            tiles.push(bbox);
        }
    });
    if tiles.is_empty() {
        return;
    }
    let acc = tree.accessor();
    let mut regions = Vec::new();
    for t in tiles {
        let dimm = t.max[0] - t.min[0];
        let bbox = CoordBBox::new(t.min, t.min.offset(dimm));
        let inside = acc.get_value(&bbox.min).is_inside(iso);
        let depth = acc.get_value_depth(&bbox.min);
        for axis in 0..3 {
            // positive side
            let ijk = bbox.max;
            let mut nijk = ijk;
            nijk[axis] += 1;
            let mut process = true;
            if depth >= acc.get_value_depth(&nijk) {
                process = inside != acc.get_value(&nijk).is_inside(iso);
            }
            if process {
                let mut r = bbox.expanded(1);
                r.min[axis] = ijk[axis];
                r.max[axis] = ijk[axis];
                regions.push(r);
            }
            // negative side
            let mut ijk = bbox.min;
            ijk[axis] -= 1;
            let mut process = true;
            if depth >= acc.get_value_depth(&ijk) {
                let (v, on) = acc.probe_value(&ijk);
                process = !on && inside != v.is_inside(iso);
            }
            if process {
                let mut r = bbox.expanded(1);
                r.min[axis] = ijk[axis];
                r.max[axis] = ijk[axis];
                regions.push(r);
            }
        }
    }
    for r in regions {
        mask.fill(&r, false, true);
    }
}

pub(crate) fn identify_surface_intersecting_voxels<T: MeshValue>(tree: &Tree<T>, iso: T) -> Tree<bool> {
    let offs = VoxelOffsets::new();
    let coords: Vec<Vec<Coord>> = tree
        .leaf_refs()
        .par_iter()
        .map_init(
            || tree.accessor(),
            |acc, leaf| {
                let mut out = Vec::new();
                identify_leaf_edges(leaf, acc, &offs, iso, &mut out);
                out
            },
        )
        .collect();
    let mut mask = Tree::<bool>::new(false);
    {
        let mut macc = mask.accessor_mut();
        for c in coords.iter().flatten() {
            macc.set_active_state(c, true);
        }
    }
    mask_active_tile_borders(tree, iso, &mut mask);
    mask
}

fn apply_surface_mask<T: MeshValue>(
    intersection: &mut Tree<bool>,
    border: &mut Tree<bool>,
    grid: &Grid<T>,
    mask_grid: &Grid<bool>,
    invert: bool,
    iso: T,
) {
    let transform = grid.transform();
    let mask_tree = mask_grid.tree();
    let mask_xform = mask_grid.transform();
    let matching = transform == mask_xform;

    // MaskSurface
    intersection.par_for_each_leaf_mut(|node| {
        let macc = mask_tree.accessor();
        if matching {
            if let Some(mn) = macc.probe_leaf(&node.origin()) {
                let on: Vec<usize> = node.value_mask().iter_on().collect();
                for pos in on {
                    if mn.is_value_on_at(pos) == invert {
                        node.buffer_mut()[pos] = true;
                    }
                }
            } else if macc.is_value_on(&node.origin()) == invert {
                let on: Vec<usize> = node.value_mask().iter_on().collect();
                for pos in on {
                    node.buffer_mut()[pos] = true;
                }
            }
        } else {
            let on: Vec<usize> = node.value_mask().iter_on().collect();
            for pos in on {
                let c = node.offset_to_global_coord(pos);
                let ijk = mask_xform.world_to_index_cell_centered(transform.index_to_world_coord(c));
                if macc.is_value_on(&ijk) == invert {
                    node.buffer_mut()[pos] = true;
                }
            }
        }
    });

    // MaskBorderVoxels
    {
        let it = &*intersection;
        let borders: Vec<Vec<Coord>> = it
            .leaf_refs()
            .par_iter()
            .map_init(
                || it.accessor(),
                |macc, node| {
                    let mut out = Vec::new();
                    for (pos, &lhs) in node.iter_on() {
                        let ijk = node.offset_to_global_coord(pos);
                        let nbrs = [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0)];
                        let edge = nbrs.iter().any(|&(dx, dy, dz)| {
                            let (r, on) = macc.probe_value(&ijk.offset_by(dx, dy, dz));
                            on && lhs != r
                        });
                        if edge {
                            out.push(ijk);
                        }
                    }
                    out
                },
            )
            .collect();
        let mut bacc = border.accessor_mut();
        for c in borders.iter().flatten() {
            bacc.set_value(c, true);
        }
    }

    // MaskIntersectingVoxels
    let tmp_coords: Vec<Vec<Coord>> = {
        let tree = grid.tree();
        intersection
            .leaf_refs()
            .par_iter()
            .map_init(
                || tree.accessor(),
                |iacc, node| {
                    let mut out = Vec::new();
                    for (pos, &v) in node.iter_on() {
                        if v {
                            continue;
                        }
                        let ijk = node.offset_to_global_coord(pos);
                        let inside = iacc.get_value(&ijk).is_inside(iso);
                        if inside != iacc.get_value(&ijk.offset_by(1, 0, 0)).is_inside(iso) {
                            edge_set(&mut out, 0, ijk);
                        }
                        if inside != iacc.get_value(&ijk.offset_by(0, 1, 0)).is_inside(iso) {
                            edge_set(&mut out, 1, ijk);
                        }
                        if inside != iacc.get_value(&ijk.offset_by(0, 0, 1)).is_inside(iso) {
                            edge_set(&mut out, 2, ijk);
                        }
                    }
                    out
                },
            )
            .collect()
    };
    let mut tmp = Tree::<bool>::new(false);
    {
        let mut tacc = tmp.accessor_mut();
        for c in tmp_coords.iter().flatten() {
            tacc.set_active_state(c, true);
        }
    }
    // SyncMaskValues
    {
        let it = &*intersection;
        tmp.par_for_each_leaf_mut(|node| {
            if let Some(mn) = it.probe_leaf(&node.origin()) {
                let on: Vec<usize> = node.value_mask().iter_on().collect();
                for pos in on {
                    if mn.buffer()[pos] {
                        node.buffer_mut()[pos] = true;
                    }
                }
            }
        });
    }
    *intersection = tmp;
}

// ---------------------------------------------------------------------------------------
// Auxiliary data
// ---------------------------------------------------------------------------------------

pub(crate) fn compute_auxiliary_data<T: MeshValue>(
    intersection: &Tree<bool>,
    tree: &Tree<T>,
    iso: T,
) -> (Tree<i16>, Tree<u32>) {
    let leaves: Vec<Option<Box<LeafNode<i16>>>> = intersection
        .leaf_refs()
        .par_iter()
        .map_init(
            || tree.accessor(),
            |acc, mask_node| {
                let origin = mask_node.origin();
                let leaf = acc.probe_leaf(&origin);
                let mut signs_node = LeafNode::<i16>::new(origin, 0, false);
                let mut updated = false;
                for (pos, &mval) in mask_node.iter_on() {
                    let vals = cell_values(leaf, acc, pos, &origin);
                    let mut sign_flags = compute_sign_flags(&vals, iso);
                    if sign_flags != 0 && sign_flags != 0xFF {
                        let inside = sign_flags & 0x1 != 0;
                        let mut edge_flags: i16 = if inside { INSIDE } else { 0 };
                        if !mval {
                            if inside != ((sign_flags & 0x02) != 0) {
                                edge_flags |= XEDGE;
                            }
                            if inside != ((sign_flags & 0x10) != 0) {
                                edge_flags |= YEDGE;
                            }
                            if inside != ((sign_flags & 0x08) != 0) {
                                edge_flags |= ZEDGE;
                            }
                        }
                        let amb = AMBIGUOUS_FACE[sign_flags as usize];
                        if amb != 0 {
                            let ijk = origin + LeafNode::<bool>::offset_to_local_coord(pos);
                            correct_cell_signs(&mut sign_flags, amb, acc, ijk, iso);
                        }
                        edge_flags |= sign_flags as i16;
                        signs_node.set_value_on_at(pos, edge_flags);
                        updated = true;
                    }
                }
                if updated {
                    Some(Box::new(signs_node))
                } else {
                    None
                }
            },
        )
        .collect();
    let mut sign_tree = Tree::<i16>::new(0);
    let mut index_tree = Tree::<u32>::new(u32::MAX);
    for l in leaves.into_iter().flatten() {
        let mut idx = LeafNode::<u32>::new(l.origin(), u32::MAX, false);
        idx.value_mask_mut().union_with(l.value_mask());
        let on: Vec<usize> = idx.value_mask().iter_on().collect();
        for pos in on {
            idx.buffer_mut()[pos] = 0;
        }
        index_tree.add_leaf(Box::new(idx));
        sign_tree.add_leaf(l);
    }
    (sign_tree, index_tree)
}

pub(crate) fn leaf_point_count(node: &LeafNode<i16>) -> u32 {
    node.buffer().iter().map(|&v| EDGE_GROUP_TABLE[(SIGNS & v) as usize][0] as u32).sum()
}

// ---------------------------------------------------------------------------------------
// Seam lines (reference meshing)
// ---------------------------------------------------------------------------------------

fn mark_seam_line_data(sign_tree: &mut Tree<i16>, ref_sign_tree: &Tree<i16>) {
    // SetSeamLineFlags
    sign_tree.par_for_each_leaf_mut(|node| {
        if let Some(rn) = ref_sign_tree.probe_leaf(&node.origin()) {
            let on: Vec<usize> = node.value_mask().iter_on().collect();
            for pos in on {
                let rhs = (rn.buffer()[pos] & SIGNS) as u8;
                if EDGE_GROUP_TABLE[rhs as usize][0] > 0 {
                    let v = node.buffer()[pos];
                    let lhs = (v & SIGNS) as u8;
                    if rhs != lhs {
                        node.buffer_mut()[pos] = v | SEAM;
                    }
                }
            }
        }
    });
    // MaskSeamLineVoxels
    let seam: Vec<Vec<usize>> = {
        let st = &*sign_tree;
        st.leaf_refs()
            .par_iter()
            .map_init(
                || st.accessor(),
                |sacc, node| {
                    let mut out = Vec::new();
                    let is_seam = |c: &Coord| sacc.get_value(c) & SEAM != 0;
                    for (pos, &flags) in node.iter_on() {
                        if (flags & SEAM) == 0 && (flags & EDGES) != 0 {
                            let mut ijk = node.offset_to_global_coord(pos);
                            let mut s = false;
                            if flags & XEDGE != 0 {
                                ijk[1] -= 1;
                                s = is_seam(&ijk);
                                ijk[2] -= 1;
                                s = s || is_seam(&ijk);
                                ijk[1] += 1;
                                s = s || is_seam(&ijk);
                                ijk[2] += 1;
                            }
                            if !s && flags & YEDGE != 0 {
                                ijk[2] -= 1;
                                s = s || is_seam(&ijk);
                                ijk[0] -= 1;
                                s = s || is_seam(&ijk);
                                ijk[2] += 1;
                                s = s || is_seam(&ijk);
                                ijk[0] += 1;
                            }
                            if !s && flags & ZEDGE != 0 {
                                ijk[1] -= 1;
                                s = s || is_seam(&ijk);
                                ijk[0] -= 1;
                                s = s || is_seam(&ijk);
                                ijk[1] += 1;
                                s = s || is_seam(&ijk);
                                ijk[0] += 1;
                            }
                            if s {
                                out.push(pos);
                            }
                        }
                    }
                    out
                },
            )
            .collect()
    };
    // TransferSeamLineFlags
    for (node, s) in sign_tree.leaves_mut().zip(seam) {
        for pos in s {
            node.buffer_mut()[pos] |= SEAM;
        }
    }
}

// ---------------------------------------------------------------------------------------
// Adaptive merging
// ---------------------------------------------------------------------------------------

struct MergeParams<'a, T: MeshValue> {
    input: &'a Tree<T>,
    input_xform: &'a Transform,
    point_index: &'a Tree<u32>,
    iso: T,
    surface_adaptivity: f32,
    internal_adaptivity: f32,
    invert_surface_orientation: bool,
    spatial: Option<(&'a Tree<f32>, &'a Transform)>,
    mask: Option<&'a Tree<bool>>,
    ref_sign: Option<&'a Tree<i16>>,
}

fn is_mergeable(grad: &LeafNode<Vec3f>, start: &Coord, dim: i32, adaptivity: f32) -> bool {
    if (adaptivity as f64) < 1e-6 {
        return false;
    }
    let mut norms = Vec::new();
    for x in start[0]..start[0] + dim {
        for y in start[1]..start[1] + dim {
            for z in start[2]..start[2] + dim {
                let p = LeafNode::<Vec3f>::coord_to_offset(&Coord::new(x, y, z));
                if grad.is_value_on_at(p) {
                    norms.push(grad.buffer()[p]);
                }
            }
        }
    }
    for ni in &norms {
        for nj in &norms {
            if (1.0 - ni.dot(nj) as f64) > adaptivity as f64 {
                return false;
            }
        }
    }
    true
}

fn merge_voxel_regions<T: MeshValue>(
    p: &MergeParams<'_, T>,
    sign_node: &LeafNode<i16>,
    index_node: &mut LeafNode<u32>,
) {
    let input_acc = p.input.accessor();
    let pidx_acc = p.point_index.accessor();
    let origin = index_node.origin();
    let end = origin.offset(8);
    let mut mask = NodeMask::<3>::new();
    let set_mask = |mask: &mut NodeMask<3>, c: &Coord| mask.set_on(LeafNode::<u32>::coord_to_offset(c));
    let mask_on = |mask: &NodeMask<3>, c: &Coord| mask.is_on(LeafNode::<u32>::coord_to_offset(c));

    if let Some(mt) = p.mask {
        if let Some(ml) = mt.probe_leaf(&origin) {
            for n in ml.value_mask().iter_on() {
                set_mask(&mut mask, &coord_and(&ml.offset_to_global_coord(n), !1));
            }
        }
    }

    let mut adaptivity = match p.ref_sign {
        Some(r) if r.probe_leaf(&origin).is_none() => p.internal_adaptivity,
        _ => p.surface_adaptivity,
    };
    let mut use_gradients = adaptivity < 1.0;
    let mut adaptivity_leaf = [adaptivity; 512];
    if let Some((stree, sxform)) = p.spatial {
        use_gradients = false;
        let sacc = stree.accessor();
        for (offset, a) in adaptivity_leaf.iter_mut().enumerate() {
            let ijk = origin + LeafNode::<f32>::offset_to_local_coord(offset);
            let ijk = sxform.world_to_index_cell_centered(p.input_xform.index_to_world_coord(ijk));
            let weight = sacc.get_value(&ijk);
            let v = weight * adaptivity;
            if v < 1.0 {
                use_gradients = true;
            }
            *a = v;
        }
    }

    // Mask off ambiguous voxels.
    for (pos, &flags) in sign_node.iter_on() {
        let signs = (SIGNS & flags) as usize;
        let c = sign_node.offset_to_global_coord(pos);
        if (flags & SEAM) != 0 || !ADAPTABLE[signs] || EDGE_GROUP_TABLE[signs][0] > 1 {
            set_mask(&mut mask, &coord_and(&c, !1));
        } else if (flags & EDGES) != 0 {
            let mut mr = false;
            let mut ijk = c;
            if !pidx_acc.is_value_on(&ijk) {
                mr = true;
            }
            if !mr && (flags & XEDGE) != 0 {
                ijk[1] -= 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[2] -= 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[1] += 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[2] += 1;
            }
            if !mr && (flags & YEDGE) != 0 {
                ijk[2] -= 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[0] -= 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[2] += 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[0] += 1;
            }
            if !mr && (flags & ZEDGE) != 0 {
                ijk[1] -= 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[0] -= 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[1] += 1;
                if !mr && !pidx_acc.is_value_on(&ijk) {
                    mr = true;
                }
                ijk[0] += 1;
            }
            let _ = ijk;
            if mr {
                set_mask(&mut mask, &coord_and(&c, !1));
            }
        }
    }

    // Mask off topologically ambiguous 2x2x2 voxel sub-blocks.
    let mut dim = 2;
    let mut x = origin[0];
    while x < end[0] {
        let mut y = origin[1];
        while y < end[1] {
            let mut z = origin[2];
            while z < end[2] {
                let ijk = Coord::new(x, y, z);
                if !mask_on(&mask, &ijk) && is_non_manifold(&input_acc, &ijk, p.iso, dim) {
                    set_mask(&mut mask, &ijk);
                }
                z += dim;
            }
            y += dim;
        }
        x += dim;
    }

    // Gradients.
    let invert_grad = p.invert_surface_orientation || T::IS_BOOL;
    let mut gradient = LeafNode::<Vec3f>::new(Coord::origin(), Vec3f::zero(), false);
    if use_gradients {
        for (pos, _) in sign_node.iter_on() {
            let ijk = sign_node.offset_to_global_coord(pos);
            if !mask_on(&mask, &coord_and(&ijk, !1)) {
                let g = |dx: i32, dy: i32, dz: i32| {
                    T::central_diff(
                        input_acc.get_value(&ijk.offset_by(dx, dy, dz)),
                        input_acc.get_value(&ijk.offset_by(-dx, -dy, -dz)),
                    )
                };
                let mut dir = Vec3f::new(g(1, 0, 0), g(0, 1, 0), g(0, 0, 1));
                dir.normalize_eps(1.0e-7);
                if invert_grad {
                    dir = -dir;
                }
                gradient.set_value_on_at(pos, dir);
            }
        }
    }

    // Merge regions.
    let mut region_id: u32 = 1;
    while dim <= 8 {
        let coord_mask = !((dim << 1) - 1);
        let mut x = origin[0];
        while x < end[0] {
            let mut y = origin[1];
            while y < end[1] {
                let mut z = origin[2];
                while z < end[2] {
                    let ijk = Coord::new(x, y, z);
                    adaptivity = adaptivity_leaf[LeafNode::<f32>::coord_to_offset(&ijk)];
                    if mask_on(&mask, &ijk)
                        || is_non_manifold(&input_acc, &ijk, p.iso, dim)
                        || (use_gradients && !is_mergeable(&gradient, &ijk, dim, adaptivity))
                    {
                        set_mask(&mut mask, &coord_and(&ijk, coord_mask));
                    } else {
                        for xx in x..x + dim {
                            for yy in y..y + dim {
                                for zz in z..z + dim {
                                    let o = LeafNode::<u32>::coord_to_offset(&Coord::new(xx, yy, zz));
                                    index_node.buffer_mut()[o] = region_id;
                                }
                            }
                        }
                        region_id += 1;
                    }
                    z += dim;
                }
                y += dim;
            }
            x += dim;
        }
        dim <<= 1;
    }
}

// ---------------------------------------------------------------------------------------
// Points and polygons
// ---------------------------------------------------------------------------------------

pub(crate) struct RefData<'a, T: MeshValue> {
    input: &'a Tree<T>,
    point_index: &'a Tree<u32>,
    sign: &'a Tree<i16>,
    seam_points: &'a [u32],
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_points_leaf<T: MeshValue>(
    index_node: &mut LeafNode<u32>,
    sign_node: &LeafNode<i16>,
    input: &Tree<T>,
    mut point_offset: u32,
    xform: &Transform,
    iso: f64,
    refdata: Option<&RefData<'_, T>>,
    points: &mut Vec<Vec3f>,
    flags_out: &mut Vec<u8>,
) {
    let input_acc = input.accessor();
    let origin = index_node.origin();
    let input_node = input_acc.probe_leaf(&origin);
    let ref_acc = refdata.map(|r| r.input.accessor());
    let (ref_input_node, ref_pidx_node, ref_sign_node) = match refdata {
        Some(r) => (r.input.probe_leaf(&origin), r.point_index.probe_leaf(&origin), r.sign.probe_leaf(&origin)),
        None => (None, None, None),
    };
    let mut regions: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut pts = [Vec3d::zero(); 4];
    let mut weighted = [false; 4];
    let on: Vec<usize> = index_node.value_mask().iter_on().collect();
    for offset in on {
        let id = index_node.buffer()[offset];
        if id != 0 {
            if id != INVALID_IDX {
                regions.entry(id).or_default().push(offset);
            }
            continue;
        }
        index_node.buffer_mut()[offset] = point_offset;
        let flags = sign_node.buffer()[offset];
        let signs = (SIGNS & flags) as u8;
        let mut ref_signs = 0u8;
        if (flags & SEAM) != 0 && ref_pidx_node.is_some() {
            if let Some(rs) = ref_sign_node {
                if rs.is_value_on_at(offset) {
                    ref_signs = (SIGNS & rs.buffer()[offset]) as u8;
                }
            }
        }
        let local = LeafNode::<u32>::offset_to_local_coord(offset);
        let inclusive = input_node.is_some() && is_internal_leaf_coord(&local);
        let ijk = origin + local;
        let values = if inclusive {
            to_f64_array(&cell_values_leaf(input_node.unwrap(), offset))
        } else {
            to_f64_array(&cell_values_acc(&input_acc, ijk))
        };
        let (count, weightcount) = if ref_signs == 0 {
            (compute_cell_points(&mut pts, &values, signs, iso), 0)
        } else {
            let r = refdata.unwrap();
            let ref_values = match ref_input_node {
                Some(rn) if inclusive => to_f64_array(&cell_values_leaf(rn, offset)),
                _ => to_f64_array(&cell_values_acc(ref_acc.as_ref().unwrap(), ijk)),
            };
            let pidx = ref_pidx_node.unwrap().buffer()[offset] as usize;
            let c = compute_cell_points_ref(
                &mut pts,
                &mut weighted,
                &values,
                &ref_values,
                signs,
                ref_signs,
                iso,
                pidx,
                r.seam_points,
            );
            (c, c)
        };
        let xyz = ijk.as_vec3d();
        for (i, p) in pts.iter().enumerate().take(count) {
            if !p.is_finite() {
                panic!(
                    "VolumeToMesh encountered NaNs or infs in the input VDB! Hint: Check the input and \
                     consider using the \"Diagnostics\" tool to detect and resolve the NaNs."
                );
            }
            let w = xform.index_to_world(*p + xyz);
            points.push(Vec3f::new(w[0] as f32, w[1] as f32, w[2] as f32));
            if refdata.is_some() {
                flags_out.push(if weightcount > 0 && weighted[i] { 1 } else { 0 });
            }
            point_offset += 1;
        }
    }
    for (_, voxels) in regions {
        let mut avg = Vec3d::zero();
        for &offset in &voxels {
            let local = LeafNode::<u32>::offset_to_local_coord(offset);
            let inclusive = input_node.is_some() && is_internal_leaf_coord(&local);
            let ijk = origin + local;
            index_node.buffer_mut()[offset] = point_offset;
            let signs = (SIGNS & sign_node.buffer()[offset]) as u8;
            let values = if inclusive {
                to_f64_array(&cell_values_leaf(input_node.unwrap(), offset))
            } else {
                to_f64_array(&cell_values_acc(&input_acc, ijk))
            };
            compute_cell_points(&mut pts, &values, signs, iso);
            avg[0] += ijk[0] as f64 + pts[0][0];
            avg[1] += ijk[1] as f64 + pts[0][1];
            avg[2] += ijk[2] as f64 + pts[0][2];
        }
        if voxels.len() > 1 {
            let w = 1.0 / voxels.len() as f64;
            avg *= w;
        }
        let w = xform.index_to_world(avg);
        points.push(Vec3f::new(w[0] as f32, w[1] as f32, w[2] as f32));
        if refdata.is_some() {
            flags_out.push(0);
        }
        point_offset += 1;
    }
}

trait PrimBuilder {
    fn new(upper_bound: usize) -> Self;
    fn add_prim(&mut self, verts: [u32; 4], reverse: bool, flags: u8);
    fn done(self) -> PolygonPool;
}

struct UniformPrimBuilder {
    pool: PolygonPool,
}
impl PrimBuilder for UniformPrimBuilder {
    fn new(n: usize) -> Self {
        UniformPrimBuilder {
            pool: PolygonPool {
                quads: Vec::with_capacity(n),
                triangles: vec![],
                quad_flags: Vec::with_capacity(n),
                triangle_flags: vec![],
            },
        }
    }
    fn add_prim(&mut self, v: [u32; 4], reverse: bool, flags: u8) {
        self.pool.quads.push(if reverse { [v[3], v[2], v[1], v[0]] } else { v });
        self.pool.quad_flags.push(flags);
    }
    fn done(self) -> PolygonPool {
        self.pool
    }
}

struct AdaptivePrimBuilder {
    pool: PolygonPool,
}
impl AdaptivePrimBuilder {
    fn add_triangle(&mut self, v0: u32, v1: u32, v2: u32, reverse: bool, flags: u8) {
        self.pool.triangles.push(if reverse { [v2, v1, v0] } else { [v0, v1, v2] });
        self.pool.triangle_flags.push(flags);
    }
}
impl PrimBuilder for AdaptivePrimBuilder {
    fn new(_n: usize) -> Self {
        AdaptivePrimBuilder { pool: PolygonPool::default() }
    }
    fn add_prim(&mut self, v: [u32; 4], reverse: bool, flags: u8) {
        if v[0] != v[1] && v[0] != v[2] && v[0] != v[3] && v[1] != v[2] && v[1] != v[3] && v[2] != v[3] {
            self.pool.quads.push(if reverse { [v[3], v[2], v[1], v[0]] } else { v });
            self.pool.quad_flags.push(flags);
        } else if v[0] == v[3] && v[1] != v[2] && v[1] != v[0] && v[2] != v[0] {
            self.add_triangle(v[0], v[1], v[2], reverse, flags);
        } else if v[1] == v[2] && v[0] != v[3] && v[0] != v[1] && v[3] != v[1] {
            self.add_triangle(v[0], v[1], v[3], reverse, flags);
        } else if v[0] == v[1] && v[2] != v[3] && v[2] != v[0] && v[3] != v[0] {
            self.add_triangle(v[0], v[2], v[3], reverse, flags);
        } else if v[2] == v[3] && v[0] != v[1] && v[0] != v[2] && v[1] != v[2] {
            self.add_triangle(v[0], v[1], v[2], reverse, flags);
        }
    }
    fn done(self) -> PolygonPool {
        self.pool
    }
}

#[allow(clippy::too_many_arguments)]
fn construct_polygons<B: PrimBuilder>(
    invert: bool,
    flags: i16,
    ref_flags: i16,
    offsets: [i32; 3],
    ijk: &Coord,
    sign_acc: &Acc<'_, i16>,
    idx_acc: &Acc<'_, u32>,
    mesher: &mut B,
) {
    let (v0, active) = idx_acc.probe_value(ijk);
    if !active || v0 == INVALID_IDX {
        return;
    }
    let tag0: u8 = if (flags & SEAM) != 0 { POLYFLAG_FRACTURE_SEAM } else { 0 };
    let tags = [tag0, tag0 | POLYFLAG_EXTERIOR];
    let mut is_inside = (flags & INSIDE) != 0;
    if invert {
        is_inside = !is_inside;
    }
    let mut coord = *ijk;
    let mut quad = [0u32; 4];
    let step = |coord: &Coord, slot: &mut u32, edge: usize| -> bool {
        let (v, on) = idx_acc.probe_value(coord);
        let cell = (SIGNS & sign_acc.get_value(coord)) as usize;
        let add = if EDGE_GROUP_TABLE[cell][0] > 1 { EDGE_GROUP_TABLE[cell][edge] as i32 - 1 } else { 0 };
        *slot = v.wrapping_add(add as u32);
        on
    };
    if (flags & XEDGE) != 0 {
        quad[0] = v0.wrapping_add(offsets[0] as u32);
        coord[1] -= 1;
        let mut av = step(&coord, &mut quad[1], 5);
        coord[2] -= 1;
        av = av && step(&coord, &mut quad[2], 7);
        coord[1] += 1;
        av = av && step(&coord, &mut quad[3], 3);
        if av {
            mesher.add_prim(quad, is_inside, tags[((ref_flags & XEDGE) != 0) as usize]);
        }
        coord[2] += 1;
    }
    if (flags & YEDGE) != 0 {
        quad[0] = v0.wrapping_add(offsets[1] as u32);
        coord[2] -= 1;
        let mut av = step(&coord, &mut quad[1], 12);
        coord[0] -= 1;
        av = av && step(&coord, &mut quad[2], 11);
        coord[2] += 1;
        av = av && step(&coord, &mut quad[3], 10);
        if av {
            mesher.add_prim(quad, is_inside, tags[((ref_flags & YEDGE) != 0) as usize]);
        }
        coord[0] += 1;
    }
    if (flags & ZEDGE) != 0 {
        quad[0] = v0.wrapping_add(offsets[2] as u32);
        coord[1] -= 1;
        let mut av = step(&coord, &mut quad[1], 8);
        coord[0] -= 1;
        av = av && step(&coord, &mut quad[2], 6);
        coord[1] += 1;
        av = av && step(&coord, &mut quad[3], 2);
        if av {
            mesher.add_prim(quad, !is_inside, tags[((ref_flags & ZEDGE) != 0) as usize]);
        }
    }
}

fn compute_polygons<B: PrimBuilder>(
    sign_tree: &Tree<i16>,
    index_tree: &Tree<u32>,
    ref_sign: Option<&Tree<i16>>,
    invert: bool,
) -> Vec<PolygonPool> {
    sign_tree
        .leaf_refs()
        .par_iter()
        .map_init(
            || (sign_tree.accessor(), index_tree.accessor()),
            |(sacc, iacc), node| {
                let mut edge_count = 0usize;
                for (_, &v) in node.iter_on() {
                    edge_count +=
                        ((v & XEDGE) != 0) as usize + ((v & YEDGE) != 0) as usize + ((v & ZEDGE) != 0) as usize;
                }
                if edge_count == 0 {
                    return PolygonPool::default();
                }
                let mut mesher = B::new(edge_count);
                let ref_leaf = ref_sign.and_then(|r| r.probe_leaf(&node.origin()));
                for (pos, &flags) in node.iter_on() {
                    if (flags & 0xE00) == 0 {
                        continue;
                    }
                    let ref_flags = ref_leaf.map(|l| l.buffer()[pos]).unwrap_or(0);
                    let cell = (SIGNS & flags) as usize;
                    let offsets = if EDGE_GROUP_TABLE[cell][0] > 1 {
                        [
                            EDGE_GROUP_TABLE[cell][1] as i32 - 1,
                            EDGE_GROUP_TABLE[cell][9] as i32 - 1,
                            EDGE_GROUP_TABLE[cell][4] as i32 - 1,
                        ]
                    } else {
                        [0, 0, 0]
                    };
                    let ijk = node.offset_to_global_coord(pos);
                    construct_polygons(invert, flags, ref_flags, offsets, &ijk, sacc, iacc, &mut mesher);
                }
                mesher.done()
            },
        )
        .collect()
}

fn relax_disoriented_triangles<T: MeshValue>(
    invert: bool,
    tree: &Tree<T>,
    xform: &Transform,
    pools: &[PolygonPool],
    points: &mut [Vec3f],
) {
    let n = points.len();
    let invert_grad = invert || T::IS_BOOL;
    let masks: Vec<Vec<u32>> = {
        let pts = &*points;
        pools
            .par_iter()
            .map_init(
                || tree.accessor(),
                |acc, pool| {
                    let mut out = Vec::new();
                    for v in &pool.triangles {
                        let (v0, v1, v2) = (pts[v[0] as usize], pts[v[1] as usize], pts[v[2] as usize]);
                        let mut normal = (v2 - v0).cross(&(v1 - v0));
                        normal.normalize_eps(1.0e-7);
                        let centroid = (v0 + v1 + v2) * (1.0f32 / 3.0f32);
                        let ijk = xform.world_to_index_cell_centered(Vec3d::new(
                            centroid[0] as f64,
                            centroid[1] as f64,
                            centroid[2] as f64,
                        ));
                        let g = |dx: i32, dy: i32, dz: i32| {
                            T::central_diff(
                                acc.get_value(&ijk.offset_by(dx, dy, dz)),
                                acc.get_value(&ijk.offset_by(-dx, -dy, -dz)),
                            )
                        };
                        let mut dir = Vec3f::new(g(1, 0, 0), g(0, 1, 0), g(0, 0, 1));
                        dir.normalize_eps(1.0e-7);
                        if invert_grad {
                            dir = -dir;
                        }
                        if dir.dot(&normal) < -0.5 {
                            out.extend_from_slice(&[v[0], v[1], v[2]]);
                        }
                    }
                    out
                },
            )
            .collect()
    };
    let mut point_mask = vec![0u8; n];
    for i in masks.iter().flatten() {
        point_mask[*i as usize] = 1;
    }
    let mut updates = vec![0u8; n];
    let mut new_points = vec![Vec3f::zero(); n];
    for pool in pools {
        for v in &pool.quads {
            for k in 0..4 {
                let pi = v[k] as usize;
                if point_mask[pi] == 1 {
                    new_points[pi] +=
                        points[v[0] as usize] + points[v[1] as usize] + points[v[2] as usize] + points[v[3] as usize];
                    updates[pi] = updates[pi].wrapping_add(4);
                }
            }
        }
        for v in &pool.triangles {
            for k in 0..3 {
                let pi = v[k] as usize;
                if point_mask[pi] == 1 {
                    new_points[pi] += points[v[0] as usize] + points[v[1] as usize] + points[v[2] as usize];
                    updates[pi] = updates[pi].wrapping_add(3);
                }
            }
        }
    }
    for i in 0..n {
        if updates[i] > 0 {
            let w = 1.0 / updates[i] as f64;
            points[i] = new_points[i] * (w as f32);
        }
    }
}

fn subdivide_nonplanar_seam_line_quads(pools: &mut [PolygonPool], points: &mut Vec<Vec3f>, point_flags: &mut Vec<u8>) {
    let to_d = |p: &Vec3f| Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64);
    let mut counts = Vec::with_capacity(pools.len());
    for pool in pools.iter_mut() {
        let mut count = 0u32;
        for i in 0..pool.quads.len() {
            let flags = pool.quad_flags[i];
            if (flags & POLYFLAG_FRACTURE_SEAM) != 0 && (flags & POLYFLAG_EXTERIOR) == 0 {
                let q = pool.quads[i];
                let edge = q.iter().any(|&v| point_flags[v as usize] != 0);
                if !edge {
                    continue;
                }
                let p: Vec<Vec3d> = q.iter().map(|&v| to_d(&points[v as usize])).collect();
                if !is_planar_quad(&p[0], &p[1], &p[2], &p[3], 1e-6f32 as f64) {
                    pool.quad_flags[i] |= POLYFLAG_SUBDIVIDED;
                    count += 1;
                }
            }
        }
        counts.push(count);
    }
    let point_count = points.len();
    let mut centroids = Vec::new();
    for (pool, &count) in pools.iter_mut().zip(&counts) {
        if count == 0 {
            continue;
        }
        let mut tris = Vec::new();
        let mut tflags = Vec::new();
        let mut quads = Vec::new();
        let mut qflags = Vec::new();
        for i in 0..pool.quads.len() {
            let f = pool.quad_flags[i];
            let q = pool.quads[i];
            if (f & POLYFLAG_SUBDIVIDED) == 0 {
                continue;
            }
            let new_idx = (point_count + centroids.len()) as u32;
            centroids.push(
                (points[q[0] as usize] + points[q[1] as usize] + points[q[2] as usize] + points[q[3] as usize])
                    * 0.25f32,
            );
            tris.push([q[0], new_idx, q[3]]);
            tris.push([q[0], q[1], new_idx]);
            tris.push([q[1], q[2], new_idx]);
            tris.push([q[2], q[3], new_idx]);
            tflags.extend_from_slice(&[f; 4]);
        }
        tris.extend_from_slice(&pool.triangles);
        tflags.extend_from_slice(&pool.triangle_flags);
        for i in 0..pool.quads.len() {
            if (pool.quad_flags[i] & POLYFLAG_SUBDIVIDED) == 0 {
                quads.push(pool.quads[i]);
                qflags.push(pool.quad_flags[i]);
            }
        }
        *pool = PolygonPool { quads, triangles: tris, quad_flags: qflags, triangle_flags: tflags };
    }
    if !centroids.is_empty() {
        points.extend(centroids);
        point_flags.resize(points.len(), 0);
    }
}

fn revise_seam_line_flags(pools: &mut [PolygonPool], point_flags: &[u8]) {
    for pool in pools.iter_mut() {
        for i in 0..pool.quads.len() {
            if (pool.quad_flags[i] & POLYFLAG_FRACTURE_SEAM) != 0
                && !pool.quads[i].iter().any(|&v| point_flags[v as usize] != 0)
            {
                pool.quad_flags[i] &= !POLYFLAG_FRACTURE_SEAM;
            }
        }
        for i in 0..pool.triangles.len() {
            if (pool.triangle_flags[i] & POLYFLAG_FRACTURE_SEAM) != 0
                && !pool.triangles[i].iter().any(|&v| point_flags[v as usize] != 0)
            {
                pool.triangle_flags[i] &= !POLYFLAG_FRACTURE_SEAM;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// VolumeToMesh
// ---------------------------------------------------------------------------------------

/// Mesher for any scalar grid that has a continuous isosurface (port of `VolumeToMesh`).
pub struct VolumeToMesh<'a, T: MeshValue> {
    points: Vec<Vec3f>,
    polygons: Vec<PolygonPool>,
    point_flags: Vec<u8>,
    isovalue: f64,
    prim_adaptivity: f64,
    sec_adaptivity: f64,
    relax_disoriented_triangles: bool,
    ref_grid: Option<&'a Grid<T>>,
    surface_mask: Option<(&'a Grid<bool>, bool)>,
    adaptivity_grid: Option<&'a Grid<f32>>,
    adaptivity_mask: Option<&'a Tree<bool>>,
    ref_sign_tree: Option<Tree<i16>>,
    ref_idx_tree: Option<Tree<u32>>,
    seam_point_list_size: usize,
    quantized_seam_points: Vec<u32>,
}

impl<'a, T: MeshValue> VolumeToMesh<'a, T> {
    /// * `isovalue` — which isosurface to mesh;
    /// * `adaptivity` — adaptivity threshold in `[0, 1]`;
    /// * `relax_disoriented_triangles` — relax disoriented triangles during adaptive meshing.
    pub fn new(isovalue: f64, adaptivity: f64, relax_disoriented_triangles: bool) -> Self {
        VolumeToMesh {
            points: Vec::new(),
            polygons: Vec::new(),
            point_flags: Vec::new(),
            isovalue,
            prim_adaptivity: adaptivity,
            sec_adaptivity: 0.0,
            relax_disoriented_triangles,
            ref_grid: None,
            surface_mask: None,
            adaptivity_grid: None,
            adaptivity_mask: None,
            ref_sign_tree: None,
            ref_idx_tree: None,
            seam_point_list_size: 0,
            quantized_seam_points: Vec::new(),
        }
    }

    /// The generated world-space points.
    pub fn point_list(&self) -> &[Vec3f] {
        &self.points
    }
    /// Number of generated points.
    pub fn point_list_size(&self) -> usize {
        self.points.len()
    }
    /// One polygon pool per sign-flag leaf node.
    pub fn polygon_pool_list(&self) -> &[PolygonPool] {
        &self.polygons
    }
    pub fn polygon_pool_list_mut(&mut self) -> &mut [PolygonPool] {
        &mut self.polygons
    }
    /// Number of polygon pools.
    pub fn polygon_pool_list_size(&self) -> usize {
        self.polygons.len()
    }
    /// Per-point seam line flags (only filled for reference based meshing).
    pub fn point_flags(&self) -> &[u8] {
        &self.point_flags
    }

    /// When surfacing fractured SDF fragments, the original unfractured SDF grid can be used
    /// to eliminate seam lines and tag polygons that are coincident with the reference surface
    /// with [`POLYFLAG_EXTERIOR`] and polygons in proximity to the seam lines with
    /// [`POLYFLAG_FRACTURE_SEAM`]. `sec_adaptivity` is used in regions that do not exist in
    /// the reference grid.
    pub fn set_ref_grid(&mut self, grid: &'a Grid<T>, sec_adaptivity: f64) {
        self.ref_grid = Some(grid);
        self.sec_adaptivity = sec_adaptivity;
        self.ref_sign_tree = None;
        self.ref_idx_tree = None;
        self.seam_point_list_size = 0;
        self.quantized_seam_points.clear();
    }
    /// A boolean grid whose active topology defines the region to mesh (or its complement
    /// when `invert_mask` is true).
    pub fn set_surface_mask(&mut self, mask: &'a Grid<bool>, invert_mask: bool) {
        self.surface_mask = Some((mask, invert_mask));
    }
    /// A scalar grid used as a spatial multiplier for the adaptivity threshold.
    pub fn set_spatial_adaptivity(&mut self, grid: &'a Grid<f32>) {
        self.adaptivity_grid = Some(grid);
    }
    /// A boolean tree whose active topology defines the adaptivity mask (masked voxels are
    /// not collapsed).
    pub fn set_adaptivity_mask(&mut self, tree: &'a Tree<bool>) {
        self.adaptivity_mask = Some(tree);
    }

    /// Mesh `input_grid` (OpenVDB's `VolumeToMesh::operator()`).
    pub fn run(&mut self, input_grid: &Grid<T>) {
        self.points.clear();
        self.polygons.clear();
        self.point_flags.clear();

        let transform = input_grid.transform();
        let isovalue = T::from_f64(self.isovalue);
        let adaptivity_threshold = self.prim_adaptivity as f32;
        let adaptive = self.prim_adaptivity > 1e-7 || self.sec_adaptivity > 1e-7;
        let invert = !T::IS_BOOL && input_grid.grid_class() != GridClass::LevelSet;
        let input_tree = input_grid.tree();

        let mut adaptivity_mask = Tree::<bool>::new(false);
        if let Some(m) = self.adaptivity_mask {
            adaptivity_mask.topology_union(m);
        }

        let mut intersection = identify_surface_intersecting_voxels(input_tree, isovalue);
        if let Some((mask, inv)) = self.surface_mask {
            apply_surface_mask(&mut intersection, &mut adaptivity_mask, input_grid, mask, inv, isovalue);
        }
        if intersection.is_empty() {
            return;
        }
        let (mut sign_tree, mut index_tree) = compute_auxiliary_data(&intersection, input_tree, isovalue);
        drop(intersection);

        // Optional reference data.
        let mut reference = false;
        if let Some(ref_grid) = self.ref_grid {
            let ref_tree = ref_grid.tree();
            if self.ref_sign_tree.is_none() && self.ref_idx_tree.is_none() {
                let ri = identify_surface_intersecting_voxels(ref_tree, isovalue);
                let (rs, rx) = compute_auxiliary_data(&ri, ref_tree, isovalue);
                self.ref_sign_tree = Some(rs);
                self.ref_idx_tree = Some(rx);
            }
            if let (Some(rs), Some(rx)) = (self.ref_sign_tree.as_ref(), self.ref_idx_tree.as_mut()) {
                mark_seam_line_data(&mut sign_tree, rs);
                if self.seam_point_list_size == 0 {
                    let counts: Vec<u32> = rs.leaf_refs().par_iter().map(|l| leaf_point_count(l)).collect();
                    let mut offsets = Vec::with_capacity(counts.len());
                    let mut total = 0u32;
                    for c in &counts {
                        offsets.push(total);
                        total += *c;
                    }
                    self.seam_point_list_size = total as usize;
                    if total != 0 {
                        self.quantized_seam_points = vec![0u32; total as usize];
                        let sign_leaves = rs.leaf_refs();
                        for ((idx_node, sign_node), &off) in rx.leaves_mut().zip(sign_leaves).zip(&offsets) {
                            let mut po = off;
                            let on: Vec<usize> = idx_node.value_mask().iter_on().collect();
                            for pos in on {
                                idx_node.buffer_mut()[pos] = po;
                                po += EDGE_GROUP_TABLE[(SIGNS & sign_node.buffer()[pos]) as usize][0] as u32;
                            }
                        }
                    }
                }
                if self.seam_point_list_size != 0 {
                    let rx = &*rx;
                    let iacc = input_tree.accessor();
                    let iso = isovalue.to_f64();
                    for node in sign_tree.leaves() {
                        let origin = node.origin();
                        let (rsn, rpn) = match (rs.probe_leaf(&origin), rx.probe_leaf(&origin)) {
                            (Some(a), Some(b)) => (a, b),
                            _ => continue,
                        };
                        let input_node = iacc.probe_leaf(&origin);
                        for (pos, &flags) in node.iter_on() {
                            if (flags & SEAM) != 0 && rsn.is_value_on_at(pos) {
                                let lhs = (SIGNS & flags) as u8;
                                let rhs = (SIGNS & rsn.buffer()[pos]) as u8;
                                let values = to_f64_array(&cell_values(input_node, &iacc, pos, &origin));
                                for i in 1..=(EDGE_GROUP_TABLE[lhs as usize][0]) {
                                    let id = match_edge_group(i, lhs, rhs);
                                    if id != -1 {
                                        let data = &mut self.quantized_seam_points
                                            [rpn.buffer()[pos] as usize + (id - 1) as usize];
                                        if (*data & MASK_DIRTY_BIT) == 0 {
                                            let (p, samples) = compute_masked_point(&values, lhs, rhs, i, iso);
                                            *data = if samples > 0 { pack_point(&p) } else { MASK_INVALID_BIT };
                                            *data |= MASK_DIRTY_BIT;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                reference = true;
            }
        }

        // Adapt and count unique points.
        let counts: Vec<u32> = if adaptive {
            {
                let spatial = self.adaptivity_grid.map(|g| (g.tree(), g.transform()));
                let mask_ref = if adaptivity_mask.is_empty() { None } else { Some(&adaptivity_mask) };
                let ref_sign = if reference { self.ref_sign_tree.as_ref() } else { None };
                let snapshot = index_tree.clone();
                let params = MergeParams {
                    input: input_tree,
                    input_xform: transform,
                    point_index: &snapshot,
                    iso: isovalue,
                    surface_adaptivity: adaptivity_threshold,
                    internal_adaptivity: if reference { self.sec_adaptivity as f32 } else { adaptivity_threshold },
                    invert_surface_orientation: invert,
                    spatial,
                    mask: mask_ref,
                    ref_sign,
                };
                let sign_leaves = sign_tree.leaf_refs();
                index_tree
                    .leaf_refs_mut()
                    .into_par_iter()
                    .zip(sign_leaves.into_par_iter())
                    .for_each(|(idx, sign)| merge_voxel_regions(&params, sign, idx));
            }
            index_tree
                .leaves()
                .zip(sign_tree.leaves())
                .collect::<Vec<_>>()
                .par_iter()
                .map(|(idx, sign)| {
                    let mut count = 0usize;
                    let mut unique = BTreeSet::new();
                    for (pos, &id) in idx.iter_on() {
                        if id == 0 {
                            count += EDGE_GROUP_TABLE[(SIGNS & sign.buffer()[pos]) as usize][0] as usize;
                        } else if id != INVALID_IDX {
                            unique.insert(id);
                        }
                    }
                    (count + unique.len()) as u32
                })
                .collect()
        } else {
            sign_tree.leaf_refs().par_iter().map(|l| leaf_point_count(l)).collect()
        };

        let mut offsets = Vec::with_capacity(counts.len());
        let mut total = 0u32;
        for c in &counts {
            offsets.push(total);
            total += *c;
        }

        // Compute points.
        let ref_data = if reference {
            Some(RefData {
                input: self.ref_grid.unwrap().tree(),
                point_index: self.ref_idx_tree.as_ref().unwrap(),
                sign: self.ref_sign_tree.as_ref().unwrap(),
                seam_points: &self.quantized_seam_points,
            })
        } else {
            None
        };
        let iso = self.isovalue;
        let per_leaf: Vec<(Vec<Vec3f>, Vec<u8>)> = {
            let sign_leaves = sign_tree.leaf_refs();
            index_tree
                .leaf_refs_mut()
                .into_par_iter()
                .zip(sign_leaves.into_par_iter())
                .zip(offsets.par_iter())
                .map(|((idx, sign), &off)| {
                    let mut pts = Vec::new();
                    let mut fl = Vec::new();
                    compute_points_leaf(
                        idx,
                        sign,
                        input_tree,
                        off,
                        transform,
                        iso,
                        ref_data.as_ref(),
                        &mut pts,
                        &mut fl,
                    );
                    (pts, fl)
                })
                .collect()
        };
        let mut points = Vec::with_capacity(total as usize);
        let mut point_flags = Vec::new();
        for (p, f) in per_leaf {
            points.extend(p);
            point_flags.extend(f);
        }
        debug_assert_eq!(points.len(), total as usize);

        // Compute polygons.
        let ref_sign = if reference { self.ref_sign_tree.as_ref() } else { None };
        let mut polygons = if adaptive {
            compute_polygons::<AdaptivePrimBuilder>(&sign_tree, &index_tree, ref_sign, invert)
        } else {
            compute_polygons::<UniformPrimBuilder>(&sign_tree, &index_tree, ref_sign, invert)
        };
        drop(sign_tree);
        drop(index_tree);

        if adaptive && self.relax_disoriented_triangles {
            relax_disoriented_triangles(invert, input_tree, transform, &polygons, &mut points);
        }
        if reference {
            subdivide_nonplanar_seam_line_quads(&mut polygons, &mut points, &mut point_flags);
            revise_seam_line_flags(&mut polygons, &point_flags);
        }
        self.points = points;
        self.polygons = polygons;
        self.point_flags = point_flags;
    }
}

/// Uniformly mesh a scalar grid that has a continuous isosurface (`tools::volumeToMesh`
/// without adaptivity). Returns world-space points and quads.
pub fn volume_to_mesh<T: MeshValue>(grid: &Grid<T>, isovalue: f64) -> (Vec<Vec3f>, Vec<[u32; 4]>) {
    let (p, _, q) = volume_to_mesh_adaptive(grid, isovalue, 0.0, true);
    (p, q)
}

/// Adaptively mesh a scalar grid that has a continuous isosurface (`tools::volumeToMesh`).
/// Returns world-space points, triangles and quads. Triangles are only created for areas of
/// the mesh which hit the adaptivity threshold and can't be represented as quads.
pub fn volume_to_mesh_adaptive<T: MeshValue>(
    grid: &Grid<T>,
    isovalue: f64,
    adaptivity: f64,
    relax_disoriented_triangles: bool,
) -> (Vec<Vec3f>, Vec<[u32; 3]>, Vec<[u32; 4]>) {
    let mut mesher = VolumeToMesh::new(isovalue, adaptivity, relax_disoriented_triangles);
    mesher.run(grid);
    let mut tris = Vec::new();
    let mut quads = Vec::new();
    for pool in mesher.polygon_pool_list() {
        quads.extend_from_slice(&pool.quads);
        tris.extend_from_slice(&pool.triangles);
    }
    (std::mem::take(&mut mesher.points), tris, quads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_grid(radius: f32, voxel: f32, hw: f32) -> Grid<f32> {
        crate::tools::test_util::make_sphere(radius as f64, Vec3d::zero(), voxel as f64, hw as f64)
    }

    #[test]
    fn sphere_mesh_is_closed() {
        let g = sphere_grid(5.0, 0.5, 3.0);
        let (points, quads) = volume_to_mesh(&g, 0.0);
        assert!(!points.is_empty());
        assert!(!quads.is_empty());
        for p in &points {
            let r = p.length();
            assert!((r - 5.0).abs() < 0.3, "{r}");
        }
        // Every edge of a closed quad mesh is shared by exactly two quads.
        let mut edges = std::collections::HashMap::new();
        for q in &quads {
            for k in 0..4 {
                let (a, b) = (q[k], q[(k + 1) % 4]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2));
        // Euler characteristic of a sphere.
        let chi = points.len() as i64 - edges.len() as i64 + quads.len() as i64;
        assert_eq!(chi, 2);
    }

    #[test]
    fn adaptive_reduces_polygons() {
        let g = sphere_grid(5.0, 0.25, 3.0);
        let (p0, _, q0) = volume_to_mesh_adaptive(&g, 0.0, 0.0, true);
        let (p1, t1, q1) = volume_to_mesh_adaptive(&g, 0.0, 0.5, true);
        assert!(p1.len() < p0.len());
        assert!(q1.len() + t1.len() < q0.len());
    }

    #[test]
    fn bool_grid_mesh() {
        let mut g = Grid::<bool>::new(false);
        for i in -6..=6 {
            for j in -6..=6 {
                for k in -6..=6 {
                    if i * i + j * j + k * k <= 25 {
                        g.tree_mut().set_value(&Coord::new(i, j, k), true);
                    }
                }
            }
        }
        let (points, tris, quads) = volume_to_mesh_adaptive(&g, 0.5, 0.0, true);
        assert!(tris.is_empty() && !quads.is_empty());
        let mut edges = std::collections::HashMap::new();
        for q in &quads {
            for k in 0..4 {
                let (a, b) = (q[k], q[(k + 1) % 4]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2));
        assert_eq!(points.len() as i64 - edges.len() as i64 + quads.len() as i64, 2);
    }

    #[test]
    fn feature_point() {
        let pts = [Vec3d::new(1.0, 0.0, 0.0), Vec3d::new(0.0, 2.0, 0.0), Vec3d::new(0.0, 0.0, 3.0)];
        let nrm = [Vec3d::new(1.0, 0.0, 0.0), Vec3d::new(0.0, 1.0, 0.0), Vec3d::new(0.0, 0.0, 1.0)];
        let p = find_feature_point(&pts, &nrm);
        assert!((p - Vec3d::new(1.0, 2.0, 3.0)).length() < 1e-9);
    }
}
