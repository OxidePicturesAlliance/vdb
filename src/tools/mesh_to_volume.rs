//! Convert polygonal meshes made of quads and/or triangles into signed or unsigned distance
//! field volumes (port of `openvdb/tools/MeshToVolume.h`).
//!
//! The signed distance field conversion requires a closed surface but not necessarily a
//! manifold one. It supports self intersections and degenerate faces and is independent of
//! the polygon orientation. The algorithm is the one of OpenVDB:
//!
//! 1. every polygon is voxelized into a thin band of squared distances and closest-polygon
//!    indices (large polygons are recursively subdivided as in OpenVDB);
//! 2. exterior voxels are classified by sweeping scan lines through the leaf nodes and
//!    flood filling across leaf faces ([`trace_exterior_boundaries`]), or by a user supplied
//!    interior test ([`evaluate_interior_test`]);
//! 3. voxels created by self intersecting portions of the mesh are removed;
//! 4. the narrow band is grown layer by layer to the requested exterior / interior widths;
//! 5. distances are renormalized and the band is trimmed.
//!
//! Differences from OpenVDB: there is no interrupter support. The `mesh_to_volume` overload
//! without interrupter in OpenVDB silently drops the interior test; here
//! [`mesh_to_volume_with_interior_test`] honours it.

/// `util::INVALID_IDX`: marks the unused fourth index of triangles stored as 4-tuples.
pub use super::mesh_support::INVALID_IDX;
use super::mesh_support::{
    closest_point_on_triangle_to_point, normalize_vec, prune_level_set, signed_flood_fill_with_values, take_leaves,
    COORD_OFFSETS,
};
use crate::grid::{Grid, GridClass};
use crate::math::{Coord, CoordBBox, Transform, Vec3, Vec3d, Vec3f};
use crate::tree::{Internal2, LeafNode, Node, Tree, ValueAccessor};
use crate::types::{RealValue, Value};
use crate::util::NodeMask;
use rayon::prelude::*;
use std::collections::HashMap;

/// Mesh to volume conversion flags (OpenVDB's `MeshToVolumeFlags`).
pub mod flags {
    /// Compute an unsigned distance field (does not require a closed mesh).
    pub const UNSIGNED_DISTANCE_FIELD: u32 = 0x1;
    /// Disable the cleanup step that removes voxels created by self intersecting portions
    /// of the mesh.
    pub const DISABLE_INTERSECTING_VOXEL_REMOVAL: u32 = 0x2;
    /// Disable the distance renormalization step that smooths out bumps caused by self
    /// intersecting or overlapping portions of the mesh.
    pub const DISABLE_RENORMALIZATION: u32 = 0x4;
    /// Disable the cleanup step that removes active voxels that exceed the narrow band
    /// limits (only relevant for small limits).
    pub const DISABLE_NARROW_BAND_TRIMMING: u32 = 0x8;
}
pub use flags::*;

/// Strategy used by [`evaluate_interior_test`] (OpenVDB's `InteriorTestStrategy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InteriorTestStrategy {
    /// Evaluate the interior test at every voxel.
    #[default]
    EvalEveryVoxel,
    /// Evaluate the interior test at least once per tile and flood fill within the tile.
    EvalEveryTile,
}

/// Access to the polygons of a mesh, with points in local grid index space
/// (OpenVDB's `MeshDataAdapter` concept).
pub trait MeshDataAdapter: Sync {
    /// Total number of polygons.
    fn polygon_count(&self) -> usize;
    /// Total number of points.
    fn point_count(&self) -> usize;
    /// Vertex count (3 or 4) of polygon `n`.
    fn vertex_count(&self, n: usize) -> usize;
    /// Index-space position of vertex `v` of polygon `n`.
    fn index_space_point(&self, n: usize, v: usize) -> Vec3d;
}

/// A point type usable by [`QuadAndTriangleDataAdapter`].
pub trait PointLike: Sync {
    fn to_vec3d(&self) -> Vec3d;
}
impl PointLike for Vec3f {
    fn to_vec3d(&self) -> Vec3d {
        Vec3d::new(self[0] as f64, self[1] as f64, self[2] as f64)
    }
}
impl PointLike for Vec3d {
    fn to_vec3d(&self) -> Vec3d {
        *self
    }
}
impl PointLike for [f32; 3] {
    fn to_vec3d(&self) -> Vec3d {
        Vec3d::new(self[0] as f64, self[1] as f64, self[2] as f64)
    }
}

/// A polygon type usable by [`QuadAndTriangleDataAdapter`]: a 3- or 4-tuple of point
/// indices. A 4-tuple whose last index is [`INVALID_IDX`] is a triangle.
pub trait PolygonLike: Sync {
    /// Tuple size (3 or 4).
    const SIZE: usize;
    fn vertex(&self, i: usize) -> u32;
}
impl PolygonLike for [u32; 3] {
    const SIZE: usize = 3;
    fn vertex(&self, i: usize) -> u32 {
        self[i]
    }
}
impl PolygonLike for [u32; 4] {
    const SIZE: usize = 4;
    fn vertex(&self, i: usize) -> u32 {
        self[i]
    }
}
impl PolygonLike for Vec3<u32> {
    const SIZE: usize = 3;
    fn vertex(&self, i: usize) -> u32 {
        self[i]
    }
}

/// Contiguous quad and triangle data adapter (port of `QuadAndTriangleDataAdapter`).
/// Points are assumed to be in local grid index space.
pub struct QuadAndTriangleDataAdapter<'a, P: PointLike, Q: PolygonLike> {
    points: &'a [P],
    polygons: &'a [Q],
}

impl<'a, P: PointLike, Q: PolygonLike> QuadAndTriangleDataAdapter<'a, P, Q> {
    pub fn new(points: &'a [P], polygons: &'a [Q]) -> Self {
        QuadAndTriangleDataAdapter { points, polygons }
    }
}

impl<'a, P: PointLike, Q: PolygonLike> MeshDataAdapter for QuadAndTriangleDataAdapter<'a, P, Q> {
    fn polygon_count(&self) -> usize {
        self.polygons.len()
    }
    fn point_count(&self) -> usize {
        self.points.len()
    }
    fn vertex_count(&self, n: usize) -> usize {
        if Q::SIZE == 3 || self.polygons[n].vertex(3) == INVALID_IDX {
            3
        } else {
            4
        }
    }
    fn index_space_point(&self, n: usize, v: usize) -> Vec3d {
        self.points[self.polygons[n].vertex(v) as usize].to_vec3d()
    }
}

type Acc<'a, T> = ValueAccessor<'a, Internal2<T>>;

/// `mesh_to_volume_internal::Tolerance`.
fn tol_epsilon<T: RealValue>() -> T {
    T::from_f64(1e-7)
}

#[inline]
fn t<T: RealValue>(v: f64) -> T {
    T::from_f64(v)
}

// ---------------------------------------------------------------------------------------
// Voxelization
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Triangle {
    a: Vec3d,
    b: Vec3d,
    c: Vec3d,
    index: i32,
}

struct VoxelizationData<T: RealValue> {
    dist: Tree<T>,
    index: Tree<i32>,
    prim_id: Tree<u8>,
    prim_count: u8,
}

const MAX_PRIM_ID: u8 = 100;
const POLYGON_LIMIT: usize = 1000;

impl<T: RealValue> VoxelizationData<T> {
    fn new() -> Self {
        VoxelizationData {
            dist: Tree::new(T::max_value()),
            index: Tree::new(INVALID_IDX as i32),
            prim_id: Tree::new(MAX_PRIM_ID),
            prim_count: 0,
        }
    }
    fn new_prim_id(&mut self) -> u8 {
        if self.prim_count == MAX_PRIM_ID || self.prim_id.leaf_count() > 1000 {
            self.prim_count = 0;
            self.prim_id.clear();
        }
        let id = self.prim_count;
        self.prim_count += 1;
        id
    }
}

fn eval_subdivision_count(p: &Triangle) -> i32 {
    let ext = |i: usize| {
        let (a, b, c) = (p.a[i], p.b[i], p.c[i]);
        a.max(b.max(c)) - a.min(b.min(c))
    };
    (ext(0).max(ext(1).max(ext(2))) / (8.0 * 2.0)) as i32
}

fn update_distance<T: RealValue>(ijk: &Coord, prim: &Triangle, data: &mut VoxelizationData<T>) -> bool {
    let center = ijk.as_vec3d();
    let (cp, _) = closest_point_on_triangle_to_point(&prim.a, &prim.c, &prim.b, &center);
    let dist: T = t((center - cp).length_sqr());
    if dist.is_nan() {
        return false;
    }
    let old = data.dist.get_value(ijk);
    if dist < old {
        data.dist.set_value(ijk, dist);
        data.index.set_value(ijk, prim.index);
    } else if dist == old {
        let cur = data.index.get_value(ijk);
        data.index.set_value_only(ijk, prim.index.min(cur));
    }
    !(dist > t(0.75))
}

fn voxelize_triangle<T: RealValue>(prim: &Triangle, data: &mut VoxelizationData<T>) {
    let mut coords: Vec<Coord> = Vec::new();
    let ijk = Coord::floor(prim.a);
    coords.push(ijk);
    update_distance(&ijk, prim, data);
    let prim_id = data.new_prim_id();
    data.prim_id.set_value_only(&ijk, prim_id);
    while let Some(ijk) = coords.pop() {
        for off in COORD_OFFSETS.iter() {
            let n = ijk + *off;
            if prim_id != data.prim_id.get_value(&n) {
                data.prim_id.set_value_only(&n, prim_id);
                if update_distance(&n, prim, data) {
                    coords.push(n);
                }
            }
        }
    }
}

fn subdivide_and_voxelize<T: RealValue>(
    main: &Triangle,
    data: &mut VoxelizationData<T>,
    subdivision_count: i32,
    polygon_count: usize,
) {
    let sub = subdivision_count - 1;
    let count = polygon_count * 4;
    let ac = (main.a + main.c) * 0.5;
    let bc = (main.b + main.c) * 0.5;
    let ab = (main.a + main.b) * 0.5;
    let prims = [
        Triangle { a: main.a, b: ab, c: ac, index: main.index },
        Triangle { a: ab, b: bc, c: ac, index: main.index },
        Triangle { a: ab, b: main.b, c: bc, index: main.index },
        Triangle { a: ac, b: bc, c: main.c, index: main.index },
    ];
    for p in prims.iter() {
        if sub <= 0 || count >= POLYGON_LIMIT {
            voxelize_triangle(p, data);
        } else {
            subdivide_and_voxelize(p, data, sub, count);
        }
    }
}

fn eval_triangle<T: RealValue>(prim: &Triangle, data: &mut VoxelizationData<T>, polygon_count: usize) {
    let sub = if polygon_count < POLYGON_LIMIT { eval_subdivision_count(prim) } else { 0 };
    if sub <= 0 {
        voxelize_triangle(prim, data);
    } else {
        subdivide_and_voxelize(prim, data, sub, polygon_count);
    }
}

fn voxelize_polygon<T: RealValue, M: MeshDataAdapter>(mesh: &M, n: usize, data: &mut VoxelizationData<T>) {
    let nv = mesh.vertex_count(n);
    if nv == 3 || nv == 4 {
        let mut prim = Triangle {
            a: mesh.index_space_point(n, 0),
            b: mesh.index_space_point(n, 1),
            c: mesh.index_space_point(n, 2),
            index: n as i32,
        };
        eval_triangle(&prim, data, mesh.polygon_count());
        if nv == 4 {
            prim.b = mesh.index_space_point(n, 3);
            eval_triangle(&prim, data, mesh.polygon_count());
        }
    }
}

/// Combine the distance/index trees of one voxelization batch into the output trees
/// (`combineData`): unique leaves are moved, overlapping voxels take the smaller distance
/// (ties are resolved toward the smaller polygon index).
fn combine_data<T: RealValue>(dist: &mut Tree<T>, index: &mut Tree<i32>, rdist: Tree<T>, rindex: Tree<i32>) {
    let rdist_leaves = take_leaves(rdist);
    let mut rindex_leaves: HashMap<Coord, Box<LeafNode<i32>>> =
        take_leaves(rindex).into_iter().map(|l| (l.origin(), l)).collect();
    for rd in rdist_leaves {
        let origin = rd.origin();
        let ri = rindex_leaves.remove(&origin);
        if dist.probe_leaf(&origin).is_none() {
            dist.add_leaf(rd);
            if let Some(ri) = ri {
                index.add_leaf(ri);
            }
            continue;
        }
        let ri = match ri {
            Some(ri) => ri,
            None => continue,
        };
        let ld = dist.probe_leaf_mut(&origin).unwrap();
        let li = index.touch_leaf(&origin);
        for n in 0..512 {
            let ridx = ri.buffer()[n];
            if ridx != INVALID_IDX as i32 {
                let lv = ld.buffer()[n];
                let rv = rd.buffer()[n];
                if rv < lv {
                    ld.set_value_on_at(n, rv);
                    li.set_value_on_at(n, ridx);
                } else if rv == lv {
                    let m = li.buffer()[n].min(ridx);
                    li.set_value_on_at(n, m);
                }
            }
        }
    }
    // Index leaves without a distance counterpart (should not happen, kept for safety).
    for (_, ri) in rindex_leaves {
        if index.probe_leaf(&ri.origin()).is_none() {
            index.add_leaf(ri);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Exterior sign tracing
// ---------------------------------------------------------------------------------------

const INVALID_OFFSET: usize = usize::MAX;

struct Connectivity {
    /// next/prev offsets for x, y, z: [nextX, prevX, nextY, prevY, nextZ, prevZ]
    offsets: [Vec<usize>; 6],
}

fn build_connectivity(origins: &[Coord], bbox: &CoordBBox) -> Connectivity {
    let map: HashMap<Coord, usize> = origins.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let find = |start: Coord, step: Coord| -> usize {
        let mut ijk = start + step;
        while bbox.is_inside(&ijk) {
            if let Some(&i) = map.get(&ijk.align_down(8)) {
                return i;
            }
            ijk += step;
        }
        INVALID_OFFSET
    };
    let steps = [
        Coord::new(8, 0, 0),
        Coord::new(-8, 0, 0),
        Coord::new(0, 8, 0),
        Coord::new(0, -8, 0),
        Coord::new(0, 0, 8),
        Coord::new(0, 0, -8),
    ];
    let offsets = steps.map(|s| origins.par_iter().map(|o| find(*o, s)).collect::<Vec<_>>());
    Connectivity { offsets }
}

fn trace_voxel_line<T: RealValue>(data: &mut [T], mut pos: i32, step: i32) -> bool {
    let mut is_outside = true;
    for _ in 0..8 {
        let dist = &mut data[pos as usize];
        if *dist < <T as Value>::zero() {
            is_outside = true;
        } else {
            if !(*dist > t(0.75)) {
                is_outside = false;
            }
            if is_outside {
                *dist = -*dist;
            }
        }
        pos += step;
    }
    is_outside
}

fn scan_fill<T: RealValue>(data: &mut [T]) -> bool {
    let mut updated_node = false;
    let mut updated = true;
    let zero = <T as Value>::zero();
    while updated {
        updated = false;
        for pos in 0..512usize {
            let dist = data[pos];
            if !(dist < zero) && dist > t(0.75) {
                let (x, y, z) = (pos >> 6, (pos >> 3) & 7, pos & 7);
                let flip = (z != 0 && data[pos - 1] < zero)
                    || (z != 7 && data[pos + 1] < zero)
                    || (y != 0 && data[pos - 8] < zero)
                    || (y != 7 && data[pos + 8] < zero)
                    || (x != 0 && data[pos - 64] < zero)
                    || (x != 7 && data[pos + 64] < zero);
                if flip {
                    updated = true;
                    data[pos] = -dist;
                }
            }
        }
        updated_node |= updated;
    }
    updated_node
}

/// Traces the exterior voxel boundary of closed objects in `tree`: exterior voxels are
/// marked with a negative sign; voxels with a value below `0.75` are left unchanged and act
/// as the boundary layer. Does not propagate sign information into tile regions.
pub fn trace_exterior_boundaries<T: RealValue>(tree: &mut Tree<T>) {
    let bbox = {
        let mut b = CoordBBox::empty();
        for l in tree.leaves() {
            b.expand_bbox(&l.node_bbox());
        }
        b
    };
    let mut nodes: Vec<&mut LeafNode<T>> = tree.leaf_refs_mut();
    if nodes.is_empty() {
        return;
    }
    let origins: Vec<Coord> = nodes.iter().map(|n| n.origin()).collect();
    let conn = build_connectivity(&origins, &bbox);
    let num = nodes.len();

    // Sweeps along z, y and x.
    for axis in [2usize, 1, 0] {
        let (idx_a, idx_b, step, next, prev) = match axis {
            2 => (0usize, 1usize, 1i32, &conn.offsets[4], &conn.offsets[5]),
            1 => (0, 2, 8, &conn.offsets[2], &conn.offsets[3]),
            _ => (1, 2, 64, &conn.offsets[0], &conn.offsets[1]),
        };
        let prev_of_axis = prev;
        let starts: Vec<usize> = (0..num).filter(|&n| prev_of_axis[n] == INVALID_OFFSET).collect();
        for &start in &starts {
            let mut last = start;
            for a in 0..8 {
                for b in 0..8 {
                    let mut ijk = [0i32; 3];
                    ijk[idx_a] = a;
                    ijk[idx_b] = b;
                    let mut pos = ((ijk[0] << 6) | (ijk[1] << 3) | ijk[2]) as i32;
                    let mut offset = start;
                    while offset != INVALID_OFFSET && trace_voxel_line(nodes[offset].buffer_mut(), pos, step) {
                        last = offset;
                        offset = next[offset];
                    }
                    offset = last;
                    while offset != INVALID_OFFSET {
                        last = offset;
                        offset = next[offset];
                    }
                    offset = last;
                    pos += step * 7;
                    while offset != INVALID_OFFSET && trace_voxel_line(nodes[offset].buffer_mut(), pos, -step) {
                        offset = prev[offset];
                    }
                }
            }
        }
    }

    let mut changed_a = vec![true; num];
    let mut changed_b = vec![false; num];
    let mut changed_voxels = vec![false; num * 512];

    loop {
        nodes.par_iter_mut().zip(changed_a.par_iter()).for_each(|(n, &c)| {
            if c {
                scan_fill(n.buffer_mut());
            }
        });

        {
            let view: &[&mut LeafNode<T>] = &nodes;
            let ca = &changed_a;
            changed_voxels.par_chunks_mut(512).zip(changed_b.par_iter_mut()).enumerate().for_each(|(n, (mask, nb))| {
                let lhs = view[n].buffer();
                let mut changed = false;
                // (offset table index, lhs face offset, rhs face offset, iteration)
                let faces: [(usize, bool); 6] = [(5, true), (4, false), (3, true), (2, false), (1, true), (0, false)];
                for (k, (tbl, first)) in faces.iter().enumerate() {
                    let off = conn.offsets[*tbl][n];
                    if off == INVALID_OFFSET || !ca[off] {
                        continue;
                    }
                    let rhs = view[off].buffer();
                    let axis = 2 - k / 2; // z, z, y, y, x, x
                    let last = match axis {
                        2 => 7usize,
                        1 => 8 * 7,
                        _ => 64 * 7,
                    };
                    let (lo, ro) = if *first { (0, last) } else { (last, 0) };
                    for u in 0..8usize {
                        for v in 0..8usize {
                            let pos = match axis {
                                2 => (u << 6) + (v << 3),
                                1 => (u << 6) + v,
                                _ => (u << 3) + v,
                            };
                            if lhs[pos + lo] > t(0.75) && rhs[pos + ro] < <T as Value>::zero() {
                                changed = true;
                                mask[pos + lo] = true;
                            }
                        }
                    }
                }
                *nb = changed;
            });
        }

        std::mem::swap(&mut changed_a, &mut changed_b);
        let updated = changed_a.iter().any(|&c| c);
        if updated {
            nodes.par_iter_mut().zip(changed_voxels.par_chunks_mut(512)).zip(changed_a.par_iter()).for_each(
                |((node, mask), &c)| {
                    if c {
                        let data = node.buffer_mut();
                        for pos in 0..512 {
                            if mask[pos] {
                                data[pos] = -data[pos];
                                mask[pos] = false;
                            }
                        }
                    }
                },
            );
        } else {
            break;
        }
    }
}

// ---------------------------------------------------------------------------------------
// Interior test
// ---------------------------------------------------------------------------------------

fn flood_fill_leaf_node<T: RealValue, F: Fn(&Coord) -> bool>(leaf: &mut LeafNode<T>, test: &F) {
    const NOT_VISITED: u8 = 0;
    const POSITIVE: u8 = 1;
    const NEGATIVE: u8 = 2;
    const NOT_ASSIGNED: u8 = 3;
    let mut state = [NOT_VISITED; 512];
    let mut stack: Vec<(usize, u8)> = Vec::with_capacity(512);
    let thr: T = t(0.75);
    for offset in 0..512usize {
        let value = leaf.buffer()[offset];
        if value.abs() <= thr {
            state[offset] = NOT_ASSIGNED;
        } else if state[offset] == NOT_VISITED {
            let coord = leaf.offset_to_global_coord(offset);
            let s = if test(&coord) { POSITIVE } else { NEGATIVE };
            stack.push((offset, s));
            state[offset] = s;
            while let Some((off, s)) = stack.pop() {
                if s == NEGATIVE {
                    let v = leaf.buffer()[off];
                    leaf.buffer_mut()[off] = -v;
                }
                for dim in (0..3i32).rev() {
                    for i in [-1i32, 1] {
                        let dim_idx = ((off as i32) >> (dim * 3)) % 8;
                        let neigh = off as i32 + (1 << (dim * 3)) * i;
                        if 0 < dim_idx && dim_idx < 7 && state[neigh as usize] == NOT_VISITED {
                            let neigh = neigh as usize;
                            if leaf.buffer()[neigh].abs() <= thr {
                                state[neigh] = NOT_ASSIGNED;
                            } else {
                                stack.push((neigh, s));
                                state[neigh] = s;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Sets the sign of the voxel values of `tree` based on `interior_test` (a `Coord -> bool`
/// predicate that is true inside the mesh). Inside is set to positive and outside to
/// negative, which is the reverse of the usual level set convention but matches the internal
/// convention of [`mesh_to_volume`] at the point where this is evaluated.
pub fn evaluate_interior_test<T: RealValue, F>(tree: &mut Tree<T>, interior_test: F, strategy: InteriorTestStrategy)
where
    F: Fn(&Coord) -> bool + Sync,
{
    match strategy {
        InteriorTestStrategy::EvalEveryVoxel => tree.par_for_each_leaf_mut(|leaf| {
            for n in 0..512 {
                let c = leaf.offset_to_global_coord(n);
                if !interior_test(&c) {
                    let v = leaf.buffer()[n];
                    leaf.buffer_mut()[n] = -v;
                }
            }
        }),
        InteriorTestStrategy::EvalEveryTile => {
            tree.par_for_each_leaf_mut(|leaf| flood_fill_leaf_node(leaf, &interior_test))
        }
    }
    tree.visit_tiles_mut(&mut |bbox, v, _, _| {
        if !interior_test(&bbox.min) {
            *v = -*v;
        }
    });
}

// ---------------------------------------------------------------------------------------
// Intersecting voxel sign correction and cleanup
// ---------------------------------------------------------------------------------------

fn closest_point<M: MeshDataAdapter>(mesh: &M, center: &Vec3d, poly_idx: i32) -> Vec3d {
    let polygon = poly_idx as usize;
    let a = mesh.index_space_point(polygon, 0);
    let mut b = mesh.index_space_point(polygon, 1);
    let c = mesh.index_space_point(polygon, 2);
    let (mut cp, _) = closest_point_on_triangle_to_point(&a, &c, &b, center);
    if mesh.vertex_count(polygon) == 4 {
        b = mesh.index_space_point(polygon, 3);
        let (c2, _) = closest_point_on_triangle_to_point(&a, &b, &c, center);
        if (*center - c2).length_sqr() < (*center - cp).length_sqr() {
            cp = c2;
        }
    }
    cp
}

/// Returns the offsets of the voxels whose sign must be flipped (`ComputeIntersectingVoxelSign`).
fn compute_intersecting_voxel_sign<T: RealValue, M: MeshDataAdapter>(
    leaf: &LeafNode<T>,
    dist_acc: &Acc<'_, T>,
    idx_acc: &Acc<'_, i32>,
    mesh: &M,
) -> Vec<usize> {
    let mut flips = Vec::new();
    let data = leaf.buffer();
    let idx_leaf = match idx_acc.probe_leaf(&leaf.origin()) {
        Some(l) => l,
        None => return flips,
    };
    let idx_data = idx_leaf.buffer();
    let node_min = leaf.origin();
    let node_max = node_min.offset(7);
    let mut points = vec![Vec3d::zero(); 1024];
    let mut computed = [false; 512];
    let zero = <T as Value>::zero();
    for pos in leaf.value_mask().iter_on() {
        let dist = data[pos];
        if dist < zero || dist > t(0.75) {
            continue;
        }
        let ijk = leaf.offset_to_global_coord(pos);
        let xyz = ijk.as_vec3d();
        let bmin = ijk.offset(-1).max_elem(&node_min);
        let bmax = ijk.offset(1).min_elem(&node_max);
        let bbox = CoordBBox::new(bmin, bmax);
        let mut flip = false;
        let mut x = bmin[0];
        while x <= bmax[0] && !flip {
            let x_pos = ((x & 7) as usize) << 6;
            let mut y = bmin[1];
            while y <= bmax[1] && !flip {
                let y_pos = x_pos + (((y & 7) as usize) << 3);
                for z in bmin[2]..=bmax[2] {
                    let npos = y_pos + (z & 7) as usize;
                    let poly = idx_data[npos];
                    if poly == INVALID_IDX as i32 || !(data[npos] < t(-0.75)) {
                        continue;
                    }
                    let pi = npos * 2;
                    if !computed[npos] {
                        computed[npos] = true;
                        let nxyz = Vec3d::new(x as f64, y as f64, z as f64);
                        let p = closest_point(mesh, &nxyz, poly);
                        points[pi] = p;
                        let mut dir = nxyz - p;
                        normalize_vec(&mut dir);
                        points[pi + 1] = dir;
                    }
                    let mut dir1 = xyz - points[pi];
                    normalize_vec(&mut dir1);
                    if points[pi + 1].dot(&dir1) > 0.0 {
                        flip = true;
                        break;
                    }
                }
                y += 1;
            }
            x += 1;
        }
        if flip {
            flips.push(pos);
        } else {
            for off in COORD_OFFSETS.iter() {
                let nijk = ijk + *off;
                if bbox.is_inside(&nijk) {
                    continue;
                }
                let (nval, on) = dist_acc.probe_value(&nijk);
                if on && nval < t(-0.75) {
                    let nxyz = nijk.as_vec3d();
                    let cp = closest_point(mesh, &nxyz, idx_acc.get_value(&nijk));
                    let mut dir1 = xyz - cp;
                    normalize_vec(&mut dir1);
                    let mut dir2 = nxyz - cp;
                    normalize_vec(&mut dir2);
                    if dir2.dot(&dir1) > 0.0 {
                        flips.push(pos);
                        break;
                    }
                }
            }
        }
    }
    flips
}

/// True if any of the 26 neighbours of voxel `pos` of `leaf` satisfies `pred`.
#[inline]
fn any_neighbour<T: Value>(leaf: &LeafNode<T>, acc: &Acc<'_, T>, pos: usize, pred: impl Fn(T) -> bool) -> bool {
    let local = LeafNode::<T>::offset_to_local_coord(pos);
    let data = leaf.buffer();
    for off in COORD_OFFSETS.iter() {
        let n = local + *off;
        let v = if (0..8).contains(&n[0]) && (0..8).contains(&n[1]) && (0..8).contains(&n[2]) {
            data[((n[0] << 6) | (n[1] << 3) | n[2]) as usize]
        } else {
            acc.get_value(&(leaf.origin() + n))
        };
        if pred(v) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------------------
// Narrow band expansion
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Fragment {
    idx: i32,
    x: i32,
    y: i32,
    z: i32,
}

struct ExpandContext<'m, T: RealValue, M: MeshDataAdapter> {
    mesh: &'m M,
    exterior: T,
    interior: T,
    voxel_size: T,
}

enum ExpandResult<T: Value> {
    None,
    New(Box<LeafNode<T>>, Box<LeafNode<i32>>),
    Updated(Box<LeafNode<T>>, Box<LeafNode<i32>>),
}

impl<'m, T: RealValue, M: MeshDataAdapter> ExpandContext<'m, T, M> {
    fn gather_fragments(&self, bbox: &CoordBBox, dist_acc: &Acc<'_, T>, idx_acc: &Acc<'_, i32>) -> Vec<Fragment> {
        let mut fragments = Vec::new();
        let node_min = bbox.min.align_down(8);
        let node_max = bbox.max.align_down(8);
        let mut x = node_min[0];
        while x <= node_max[0] {
            let mut y = node_min[1];
            while y <= node_max[1] {
                let mut z = node_min[2];
                while z <= node_max[2] {
                    let ijk = Coord::new(x, y, z);
                    if let Some(dl) = dist_acc.probe_leaf(&ijk) {
                        let il = idx_acc.probe_leaf(&ijk).expect("index leaf");
                        let rmin = bbox.min.max_elem(&ijk);
                        let rmax = bbox.max.min_elem(&ijk.offset(7));
                        for xx in rmin[0]..=rmax[0] {
                            let xp = ((xx & 7) as usize) << 6;
                            for yy in rmin[1]..=rmax[1] {
                                let yp = xp + (((yy & 7) as usize) << 3);
                                for zz in rmin[2]..=rmax[2] {
                                    let p = yp + (zz & 7) as usize;
                                    if dl.value_mask().is_on(p) {
                                        fragments.push(Fragment { idx: il.buffer()[p], x: xx, y: yy, z: zz });
                                    }
                                }
                            }
                        }
                    }
                    z += 8;
                }
                y += 8;
            }
            x += 8;
        }
        fragments.sort_by_key(|f| f.idx);
        fragments
    }

    fn compute_distance(&self, ijk: &Coord, manhattan_limit: i32, fragments: &[Fragment]) -> (T, i32) {
        let center = ijk.as_vec3d();
        let mut dist = f64::MAX;
        let mut last_idx = INVALID_IDX as i32;
        let mut closest = 0i32;
        for f in fragments {
            if last_idx == f.idx {
                continue;
            }
            let manhattan = (f.x - ijk[0]).abs() + (f.y - ijk[1]).abs() + (f.z - ijk[2]).abs();
            if manhattan > manhattan_limit {
                continue;
            }
            last_idx = f.idx;
            let polygon = last_idx as usize;
            let a = self.mesh.index_space_point(polygon, 0);
            let mut b = self.mesh.index_space_point(polygon, 1);
            let c = self.mesh.index_space_point(polygon, 2);
            let (cp, _) = closest_point_on_triangle_to_point(&a, &c, &b, &center);
            let mut prim_dist = (center - cp).length_sqr();
            if self.mesh.vertex_count(polygon) == 4 {
                b = self.mesh.index_space_point(polygon, 3);
                let (cp2, _) = closest_point_on_triangle_to_point(&a, &b, &c, &center);
                let tmp = (center - cp2).length_sqr();
                if tmp < prim_dist {
                    prim_dist = tmp;
                }
            }
            if prim_dist < dist {
                dist = prim_dist;
                closest = last_idx;
            }
        }
        (T::from_f64(dist.sqrt()) * self.voxel_size, closest)
    }

    fn update_voxel(
        &self,
        ijk: &Coord,
        limit: i32,
        fragments: &[Fragment],
        dist_leaf: &mut LeafNode<T>,
        idx_leaf: &mut LeafNode<i32>,
        updated: Option<&mut bool>,
    ) -> bool {
        let (distance, closest) = self.compute_distance(ijk, limit, fragments);
        let pos = LeafNode::<T>::coord_to_offset(ijk);
        let inside = dist_leaf.buffer()[pos] < <T as Value>::zero();
        let mut activate = false;
        if !inside && distance < self.exterior {
            if let Some(u) = updated {
                *u = true;
            }
            activate = (distance + self.voxel_size) < self.exterior;
            dist_leaf.buffer_mut()[pos] = distance;
            idx_leaf.set_value_on_at(pos, closest);
        } else if inside && distance < self.interior {
            if let Some(u) = updated {
                *u = true;
            }
            activate = (distance + self.voxel_size) < self.interior;
            dist_leaf.buffer_mut()[pos] = -distance;
            idx_leaf.set_value_on_at(pos, closest);
        }
        activate
    }

    fn expand_leaf(
        &self,
        mask_leaf: &LeafNode<bool>,
        dist_acc: &Acc<'_, T>,
        idx_acc: &Acc<'_, i32>,
        new_mask: &mut Vec<Coord>,
    ) -> ExpandResult<T> {
        if mask_leaf.value_mask().is_all_off() {
            return ExpandResult::None;
        }
        let origin = mask_leaf.origin();
        let (mut dist_leaf, mut idx_leaf, using_new) = match (dist_acc.probe_leaf(&origin), idx_acc.probe_leaf(&origin))
        {
            (Some(d), Some(i)) => (Box::new(d.clone()), Box::new(i.clone()), false),
            _ => {
                let bg = dist_acc.get_value(&origin);
                (
                    Box::new(LeafNode::new(origin, bg, false)),
                    Box::new(LeafNode::new(origin, idx_acc.get_value(&origin), false)),
                    true,
                )
            }
        };

        let mut bbox = CoordBBox::empty();
        for n in mask_leaf.value_mask().iter_on() {
            bbox.expand_coord(mask_leaf.offset_to_global_coord(n));
        }
        bbox.expand(1);
        let fragments = self.gather_fragments(&bbox, dist_acc, idx_acc);

        let node_bbox = mask_leaf.node_bbox();
        let mut mask = NodeMask::<3>::new();
        let mut updated = false;
        for n in mask_leaf.value_mask().iter_on() {
            let ijk = mask_leaf.offset_to_global_coord(n);
            if self.update_voxel(&ijk, 5, &fragments, &mut dist_leaf, &mut idx_leaf, Some(&mut updated)) {
                for (i, off) in COORD_OFFSETS.iter().enumerate() {
                    let nijk = ijk + *off;
                    if node_bbox.is_inside(&nijk) {
                        mask.set_on(LeafNode::<bool>::coord_to_offset(&nijk));
                    } else if i < 6 {
                        new_mask.push(nijk);
                    }
                }
            }
        }
        if !updated {
            return ExpandResult::None;
        }
        mask.difference_with(idx_leaf.value_mask());
        for pos in mask.iter_on() {
            let ijk = origin + LeafNode::<T>::offset_to_local_coord(pos);
            if self.update_voxel(&ijk, 6, &fragments, &mut dist_leaf, &mut idx_leaf, None) {
                for off in COORD_OFFSETS.iter().take(6) {
                    new_mask.push(ijk + *off);
                }
            }
        }
        if using_new {
            ExpandResult::New(dist_leaf, idx_leaf)
        } else {
            ExpandResult::Updated(dist_leaf, idx_leaf)
        }
    }
}

// ---------------------------------------------------------------------------------------
// Main entry points
// ---------------------------------------------------------------------------------------

/// Convert a polygonal mesh into a signed or unsigned distance field volume
/// (port of `tools::meshToVolume`).
///
/// * `mesh` — polygon access with points in grid index space;
/// * `transform` — the output grid transform;
/// * `exterior_band_width`, `interior_band_width` — narrow band widths in voxel units (use
///   `f32::MAX` for the interior width to fill the interior with distance values);
/// * `flags` — combination of the [`flags`] constants;
/// * `polygon_index_grid` — optional output grid receiving the closest-polygon index of
///   every voxel in the narrow band.
pub fn mesh_to_volume<T: RealValue, M: MeshDataAdapter>(
    mesh: &M,
    transform: &Transform,
    exterior_band_width: f32,
    interior_band_width: f32,
    flags: u32,
    polygon_index_grid: Option<&mut Grid<i32>>,
) -> Grid<T> {
    mesh_to_volume_impl::<T, M, fn(&Coord) -> bool>(
        mesh,
        transform,
        exterior_band_width,
        interior_band_width,
        flags,
        polygon_index_grid,
        None,
        InteriorTestStrategy::EvalEveryVoxel,
    )
}

/// [`mesh_to_volume`] with a user supplied interior test (`Coord -> bool`, true inside),
/// used instead of the exterior boundary tracing to classify voxels.
#[allow(clippy::too_many_arguments)]
pub fn mesh_to_volume_with_interior_test<T: RealValue, M: MeshDataAdapter, F>(
    mesh: &M,
    transform: &Transform,
    exterior_band_width: f32,
    interior_band_width: f32,
    flags: u32,
    polygon_index_grid: Option<&mut Grid<i32>>,
    interior_test: F,
    strategy: InteriorTestStrategy,
) -> Grid<T>
where
    F: Fn(&Coord) -> bool + Sync,
{
    mesh_to_volume_impl(
        mesh,
        transform,
        exterior_band_width,
        interior_band_width,
        flags,
        polygon_index_grid,
        Some(interior_test),
        strategy,
    )
}

#[allow(clippy::too_many_arguments)]
fn mesh_to_volume_impl<T: RealValue, M: MeshDataAdapter, F>(
    mesh: &M,
    transform: &Transform,
    exterior_band_width: f32,
    interior_band_width: f32,
    flags: u32,
    polygon_index_grid: Option<&mut Grid<i32>>,
    interior_test: Option<F>,
    strategy: InteriorTestStrategy,
) -> Grid<T>
where
    F: Fn(&Coord) -> bool + Sync,
{
    let mut dist_grid = Grid::<T>::new(T::max_value()).with_transform(transform.clone());

    let mut exterior_width: T = t(exterior_band_width as f64);
    let mut interior_width: T = t(interior_band_width as f64);
    if !exterior_width.is_finite() || interior_width.is_nan() {
        return dist_grid;
    }
    let voxel_size: T = t(transform.voxel_size()[0]);
    if !voxel_size.is_finite() || voxel_size == <T as Value>::zero() {
        return dist_grid;
    }
    exterior_width *= voxel_size;
    if interior_width < T::max_value() {
        interior_width *= voxel_size;
    }

    let signed = flags & UNSIGNED_DISTANCE_FIELD == 0;
    let remove_intersecting = flags & DISABLE_INTERSECTING_VOXEL_REMOVAL == 0;
    let renormalize = flags & DISABLE_RENORMALIZATION == 0;
    let trim = flags & DISABLE_NARROW_BAND_TRIMMING == 0;

    if signed {
        dist_grid.set_grid_class(GridClass::LevelSet);
    } else {
        dist_grid.set_grid_class(GridClass::Unknown);
        interior_width = <T as Value>::zero();
    }

    let mut dist_tree = Tree::<T>::new(T::max_value());
    let mut index_tree = Tree::<i32>::new(INVALID_IDX as i32);

    // Voxelize the mesh.
    {
        let n = mesh.polygon_count();
        let batches: Vec<VoxelizationData<T>> = (0..n)
            .into_par_iter()
            .fold(VoxelizationData::<T>::new, |mut d, i| {
                voxelize_polygon(mesh, i, &mut d);
                d
            })
            .collect();
        for b in batches {
            combine_data(&mut dist_tree, &mut index_tree, b.dist, b.index);
        }
    }

    // Classify interior and exterior regions.
    if signed {
        let every_voxel = interior_test.is_some() && strategy == InteriorTestStrategy::EvalEveryVoxel;
        match &interior_test {
            None => trace_exterior_boundaries(&mut dist_tree),
            Some(f) => evaluate_interior_test(&mut dist_tree, f, strategy),
        }
        if !every_voxel {
            let flips: Vec<Vec<usize>> = {
                let dt = &dist_tree;
                let it = &index_tree;
                dt.leaf_refs()
                    .par_iter()
                    .map_init(
                        || (dt.accessor(), it.accessor()),
                        |(da, ia), leaf| compute_intersecting_voxel_sign(leaf, da, ia, mesh),
                    )
                    .collect()
            };
            for (leaf, f) in dist_tree.leaves_mut().zip(flips) {
                for pos in f {
                    let v = leaf.buffer()[pos];
                    leaf.buffer_mut()[pos] = -v;
                }
            }

            if remove_intersecting {
                // ValidateIntersectingVoxels
                let pushes: Vec<Vec<usize>> = {
                    let dt = &dist_tree;
                    dt.leaf_refs()
                        .par_iter()
                        .map_init(
                            || dt.accessor(),
                            |acc, leaf| {
                                let mut out = Vec::new();
                                for pos in leaf.value_mask().iter_on() {
                                    let d = leaf.buffer()[pos];
                                    if d < <T as Value>::zero() || d > t(0.75) {
                                        continue;
                                    }
                                    if !any_neighbour(leaf, acc, pos, |v| v < <T as Value>::zero()) {
                                        out.push(pos);
                                    }
                                }
                                out
                            },
                        )
                        .collect()
                };
                let push_val = t::<T>(0.75) + tol_epsilon::<T>();
                for (leaf, p) in dist_tree.leaves_mut().zip(pushes) {
                    for pos in p {
                        leaf.buffer_mut()[pos] = push_val;
                    }
                }
                // RemoveSelfIntersectingSurface
                let offs: Vec<(Coord, Vec<usize>)> = {
                    let dt = &dist_tree;
                    dt.leaf_refs()
                        .par_iter()
                        .map_init(
                            || dt.accessor(),
                            |acc, leaf| {
                                let mut out = Vec::new();
                                for pos in leaf.value_mask().iter_on() {
                                    if !(leaf.buffer()[pos] > t(0.75)) {
                                        continue;
                                    }
                                    if !any_neighbour(leaf, acc, pos, |v| !(v > t(0.75))) {
                                        out.push(pos);
                                    }
                                }
                                (leaf.origin(), out)
                            },
                        )
                        .collect()
                };
                for (leaf, (_, p)) in dist_tree.leaves_mut().zip(offs.iter()) {
                    for &pos in p {
                        leaf.set_active_state_at(pos, false);
                    }
                }
                for (origin, p) in offs.iter() {
                    if p.is_empty() {
                        continue;
                    }
                    if let Some(il) = index_tree.probe_leaf_mut(origin) {
                        for &pos in p {
                            il.set_active_state_at(pos, false);
                        }
                    }
                }
                dist_tree.prune_inactive();
                index_tree.prune_inactive();
            }
        }
    }

    if dist_tree.active_voxel_count() == 0 {
        dist_tree.clear();
        dist_tree.root_mut().set_background(exterior_width, false);
        dist_grid.set_tree(dist_tree);
        finish_index_grid(polygon_index_grid, Tree::new(INVALID_IDX as i32), transform);
        return dist_grid;
    }

    // Transform values (sqrt, world space scaling and sign flip for SDFs).
    {
        let udf = !signed;
        let w = [-voxel_size, voxel_size];
        dist_tree.par_for_each_leaf_mut(|leaf| {
            leaf.for_each_on_mut(|_, v| {
                let i = (udf || *v < <T as Value>::zero()) as usize;
                *v = w[i] * v.abs().sqrt();
            })
        });
    }

    if signed {
        dist_tree.root_mut().set_background(exterior_width, false);
        signed_flood_fill_with_values(&mut dist_tree, exterior_width, -interior_width);
    } else {
        dist_tree.set_background(exterior_width);
    }

    // Expand the narrow band.
    let min_band_width = voxel_size * t(2.0);
    if interior_width > min_band_width || exterior_width > min_band_width {
        let mut mask_tree = Tree::<bool>::new(false);
        {
            let dt = &dist_tree;
            let coords: Vec<Vec<Coord>> = dt
                .leaf_refs()
                .par_iter()
                .map_init(
                    || dt.accessor(),
                    |acc, leaf| {
                        let mut out = Vec::new();
                        for pos in leaf.value_mask().iter_on() {
                            let ijk = leaf.offset_to_global_coord(pos);
                            for off in COORD_OFFSETS.iter().take(6) {
                                let n = ijk + *off;
                                let on = if leaf.node_bbox().is_inside(&n) {
                                    leaf.is_value_on_at(LeafNode::<T>::coord_to_offset(&n))
                                } else {
                                    acc.is_value_on(&n)
                                };
                                if !on {
                                    out.push(n);
                                }
                            }
                        }
                        out
                    },
                )
                .collect();
            let mut macc = mask_tree.accessor_mut();
            for c in coords.iter().flatten() {
                macc.set_value_on_only(c);
            }
        }

        let mut max_iterations = u32::MAX;
        let estimated = 2.0 * ((exterior_width.max(interior_width) - min_band_width) / voxel_size).to_f64().ceil();
        if estimated < max_iterations as f64 {
            max_iterations = estimated as u32;
        }

        let ctx = ExpandContext { mesh, exterior: exterior_width, interior: interior_width, voxel_size };
        let mut count = 0u32;
        loop {
            if mask_tree.leaf_count() == 0 {
                break;
            }
            // DiffLeafNodeMask
            {
                let dt = &dist_tree;
                mask_tree.par_for_each_leaf_mut(|ml| {
                    if let Some(dl) = dt.probe_leaf(&ml.origin()) {
                        ml.value_mask_mut().difference_with(dl.value_mask());
                    }
                });
            }
            // ExpandNarrowband
            let results: Vec<(ExpandResult<T>, Vec<Coord>)> = {
                let dt = &dist_tree;
                let it = &index_tree;
                mask_tree
                    .leaf_refs()
                    .par_iter()
                    .map_init(
                        || (dt.accessor(), it.accessor()),
                        |(da, ia), ml| {
                            let mut nm = Vec::new();
                            let r = ctx.expand_leaf(ml, da, ia, &mut nm);
                            (r, nm)
                        },
                    )
                    .collect()
            };
            let mut new_mask = Tree::<bool>::new(false);
            {
                let mut macc = new_mask.accessor_mut();
                for (_, nm) in results.iter() {
                    for c in nm {
                        macc.set_value_on_only(c);
                    }
                }
            }
            for (r, _) in results {
                match r {
                    ExpandResult::None => {}
                    ExpandResult::Updated(mut d, i) => {
                        d.value_mask_mut().union_with(i.value_mask());
                        dist_tree.add_leaf(d);
                        index_tree.add_leaf(i);
                    }
                    ExpandResult::New(mut d, i) => {
                        d.value_mask_mut().union_with(i.value_mask());
                        dist_tree.add_leaf(d);
                        index_tree.add_leaf(i);
                    }
                }
            }
            mask_tree = new_mask;
            count += 1;
            if count >= max_iterations {
                break;
            }
        }
    }

    // Renormalize distances.
    if signed && renormalize {
        let offset: T = t(0.8 * voxel_size.to_f64());
        dist_tree.par_for_each_leaf_mut(|leaf| leaf.for_each_on_mut(|_, v| *v -= offset));
        let buffers: Vec<Vec<(usize, T)>> = {
            let dt = &dist_tree;
            let dx = voxel_size;
            let inv_dx = T::one() / voxel_size;
            dt.leaf_refs()
                .par_iter()
                .map_init(
                    || dt.accessor(),
                    |acc, leaf| {
                        let mut out = Vec::new();
                        for (pos, &phi0) in leaf.iter_on() {
                            let ijk = leaf.offset_to_global_coord(pos);
                            let up = Vec3([
                                acc.get_value(&ijk.offset_by(1, 0, 0)) - phi0,
                                acc.get_value(&ijk.offset_by(0, 1, 0)) - phi0,
                                acc.get_value(&ijk.offset_by(0, 0, 1)) - phi0,
                            ]);
                            let down = Vec3([
                                phi0 - acc.get_value(&ijk.offset_by(-1, 0, 0)),
                                phi0 - acc.get_value(&ijk.offset_by(0, -1, 0)),
                                phi0 - acc.get_value(&ijk.offset_by(0, 0, -1)),
                            ]);
                            let norm_sq = godunovs_norm_sqrd(phi0 > <T as Value>::zero(), &down, &up);
                            let diff = norm_sq.sqrt() * inv_dx - T::one();
                            let s = phi0 / (phi0 * phi0 + norm_sq).sqrt();
                            out.push((pos, phi0 - dx * s * diff));
                        }
                        out
                    },
                )
                .collect()
        };
        for (leaf, b) in dist_tree.leaves_mut().zip(buffers) {
            let data = leaf.buffer_mut();
            for (pos, v) in b {
                if v < data[pos] {
                    data[pos] = v;
                }
            }
        }
        let back = offset - tol_epsilon::<T>();
        dist_tree.par_for_each_leaf_mut(|leaf| leaf.for_each_on_mut(|_, v| *v += back));
    }

    // Remove active voxels that exceed the narrow band limits.
    if trim && interior_width.min(exterior_width) < voxel_size * t(4.0) {
        let ex = exterior_width;
        let inw = if signed { interior_width } else { exterior_width };
        let in_val = -inw;
        dist_tree.par_for_each_leaf_mut(|leaf| {
            let (buf, mask) = leaf.buffer_and_mask_mut();
            let on: Vec<usize> = mask.iter_on().collect();
            for pos in on {
                let val = buf[pos];
                let inside = val < <T as Value>::zero();
                if inside && !(val > in_val) {
                    buf[pos] = in_val;
                    mask.set_off(pos);
                } else if !inside && !(val < ex) {
                    buf[pos] = ex;
                    mask.set_off(pos);
                }
            }
        });
        prune_level_set(&mut dist_tree, exterior_width, if signed { -interior_width } else { -exterior_width });
    }

    dist_grid.set_tree(dist_tree);
    finish_index_grid(polygon_index_grid, index_tree, transform);
    dist_grid
}

fn finish_index_grid(out: Option<&mut Grid<i32>>, tree: Tree<i32>, transform: &Transform) {
    if let Some(g) = out {
        g.set_tree(tree);
        g.set_transform(transform.clone());
    }
}

/// `math::GodunovsNormSqrd`.
fn godunovs_norm_sqrd<T: RealValue>(is_outside: bool, m: &Vec3<T>, p: &Vec3<T>) -> T {
    let zero = <T as Value>::zero();
    let sq = |v: T| v * v;
    let mut len2 = zero;
    for i in 0..3 {
        len2 += if is_outside {
            sq(m[i].max(zero)).max(sq(p[i].min(zero)))
        } else {
            sq(m[i].min(zero)).max(sq(p[i].max(zero)))
        };
    }
    len2
}

fn do_mesh_conversion<T: RealValue>(
    xform: &Transform,
    points: &[Vec3f],
    triangles: &[[u32; 3]],
    quads: &[[u32; 4]],
    ex_band_width: f32,
    in_band_width: f32,
    unsigned: bool,
) -> Grid<T> {
    if points.is_empty() {
        return Grid::<T>::new(t(ex_band_width as f64));
    }
    let index_points: Vec<Vec3f> = points
        .par_iter()
        .map(|p| {
            let q = xform.world_to_index(Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64));
            Vec3f::new(q[0] as f32, q[1] as f32, q[2] as f32)
        })
        .collect();
    let fl = if unsigned { UNSIGNED_DISTANCE_FIELD } else { 0 };
    if quads.is_empty() {
        let mesh = QuadAndTriangleDataAdapter::new(&index_points, triangles);
        mesh_to_volume(&mesh, xform, ex_band_width, in_band_width, fl, None)
    } else if triangles.is_empty() {
        let mesh = QuadAndTriangleDataAdapter::new(&index_points, quads);
        mesh_to_volume(&mesh, xform, ex_band_width, in_band_width, fl, None)
    } else {
        let prims: Vec<[u32; 4]> =
            triangles.iter().map(|t| [t[0], t[1], t[2], INVALID_IDX]).chain(quads.iter().copied()).collect();
        let mesh = QuadAndTriangleDataAdapter::new(&index_points, &prims);
        mesh_to_volume(&mesh, xform, ex_band_width, in_band_width, fl, None)
    }
}

/// Convert a triangle and/or quad mesh (world-space points) to a narrow-band level set
/// with the given half width in voxel units (`tools::meshToLevelSet`).
pub fn mesh_to_level_set<T: RealValue>(
    xform: &Transform,
    points: &[Vec3f],
    triangles: &[[u32; 3]],
    quads: &[[u32; 4]],
    half_width: f32,
) -> Grid<T> {
    do_mesh_conversion(xform, points, triangles, quads, half_width, half_width, false)
}

/// Convert a triangle and quad mesh to a signed distance field with an asymmetrical narrow
/// band (`tools::meshToSignedDistanceField`).
pub fn mesh_to_signed_distance_field<T: RealValue>(
    xform: &Transform,
    points: &[Vec3f],
    triangles: &[[u32; 3]],
    quads: &[[u32; 4]],
    ex_band_width: f32,
    in_band_width: f32,
) -> Grid<T> {
    do_mesh_conversion(xform, points, triangles, quads, ex_band_width, in_band_width, false)
}

/// Convert a triangle and quad mesh to an unsigned distance field
/// (`tools::meshToUnsignedDistanceField`); does not require a closed surface.
pub fn mesh_to_unsigned_distance_field<T: RealValue>(
    xform: &Transform,
    points: &[Vec3f],
    triangles: &[[u32; 3]],
    quads: &[[u32; 4]],
    band_width: f32,
) -> Grid<T> {
    do_mesh_conversion(xform, points, triangles, quads, band_width, band_width, true)
}

/// Return a narrow-band level set of an axis-aligned box given by its world-space corners
/// (`tools::createLevelSetBox`).
pub fn create_level_set_box<T: RealValue>(
    bbox_min: Vec3d,
    bbox_max: Vec3d,
    xform: &Transform,
    half_width: f64,
) -> Grid<T> {
    let pmin = xform.world_to_index(bbox_min).map(|v| v as f32);
    let pmax = xform.world_to_index(bbox_max).map(|v| v as f32);
    let points = [
        Vec3f::new(pmin[0], pmin[1], pmin[2]),
        Vec3f::new(pmin[0], pmin[1], pmax[2]),
        Vec3f::new(pmax[0], pmin[1], pmax[2]),
        Vec3f::new(pmax[0], pmin[1], pmin[2]),
        Vec3f::new(pmin[0], pmax[1], pmin[2]),
        Vec3f::new(pmin[0], pmax[1], pmax[2]),
        Vec3f::new(pmax[0], pmax[1], pmax[2]),
        Vec3f::new(pmax[0], pmax[1], pmin[2]),
    ];
    let faces: [[u32; 4]; 6] = [[0, 1, 2, 3], [7, 6, 5, 4], [4, 5, 1, 0], [6, 7, 3, 2], [0, 3, 7, 4], [1, 5, 6, 2]];
    let mesh = QuadAndTriangleDataAdapter::new(&points, &faces);
    mesh_to_volume(&mesh, xform, half_width as f32, half_width as f32, 0, None)
}

// ---------------------------------------------------------------------------------------
// MeshToVoxelEdgeData
// ---------------------------------------------------------------------------------------

/// Per-voxel edge intersection data (`MeshToVoxelEdgeData::EdgeData`): for each of the
/// three voxel edges starting at the voxel (in +x, +y, +z), the distance to the closest
/// intersection with the mesh and the intersected primitive.
#[derive(Clone, Copy, Debug)]
pub struct EdgeData {
    pub x_dist: f32,
    pub y_dist: f32,
    pub z_dist: f32,
    pub x_prim: u32,
    pub y_prim: u32,
    pub z_prim: u32,
}

impl EdgeData {
    pub fn new(dist: f32) -> Self {
        EdgeData {
            x_dist: dist,
            y_dist: dist,
            z_dist: dist,
            x_prim: INVALID_IDX,
            y_prim: INVALID_IDX,
            z_prim: INVALID_IDX,
        }
    }
}
impl Default for EdgeData {
    fn default() -> Self {
        EdgeData::new(1.0)
    }
}
impl PartialEq for EdgeData {
    fn eq(&self, o: &Self) -> bool {
        self.x_prim == o.x_prim && self.y_prim == o.y_prim && self.z_prim == o.z_prim
    }
}
impl Value for EdgeData {
    fn approx_eq(&self, other: &Self, _tolerance: &Self) -> bool {
        self == other
    }
}

/// Extracts and stores voxel edge intersection data from a mesh
/// (port of `MeshToVoxelEdgeData`).
#[derive(Default)]
pub struct MeshToVoxelEdgeData {
    tree: Tree<EdgeData>,
}

fn ray_triangle_intersection(origin: &Vec3d, dir: &Vec3d, a: &Vec3d, b: &Vec3d, c: &Vec3d) -> Option<f64> {
    let e1 = *b - *a;
    let e2 = *c - *a;
    let s1 = dir.cross(&e2);
    let divisor = s1.dot(&e1);
    if !(divisor.abs() > 0.0) {
        return None;
    }
    let inv = 1.0 / divisor;
    let d = *origin - *a;
    let b1 = d.dot(&s1) * inv;
    if !(0.0..=1.0).contains(&b1) {
        return None;
    }
    let s2 = d.cross(&e1);
    let b2 = dir.dot(&s2) * inv;
    if b2 < 0.0 || (b1 + b2) > 1.0 {
        return None;
    }
    let t = e2.dot(&s2) * inv;
    if t < 0.0 {
        None
    } else {
        Some(t)
    }
}

struct GenEdgeData<'a> {
    tree: Tree<EdgeData>,
    last_prim: Tree<i32>,
    points: &'a [Vec3f],
}

impl<'a> GenEdgeData<'a> {
    fn eval_primitive(&mut self, ijk: &Coord, a: &Vec3d, b: &Vec3d, c: &Vec3d, d: Option<&Vec3d>, index: i32) -> bool {
        let org = ijk.as_vec3d();
        let mut intersecting = false;
        let (mut ed, _) = self.tree.probe_value(ijk);
        let (cp, _) = closest_point_on_triangle_to_point(a, c, b, &org);
        let mut dist = (org - cp).length_sqr();
        let dirs = [Vec3d::new(1.0, 0.0, 0.0), Vec3d::new(0.0, 1.0, 0.0), Vec3d::new(0.0, 0.0, 1.0)];
        let mut test = |p: &Vec3d, q: &Vec3d, r: &Vec3d, ed: &mut EdgeData| {
            for (i, dir) in dirs.iter().enumerate() {
                if let Some(tt) = ray_triangle_intersection(&org, dir, p, q, r) {
                    let (dd, pp) = match i {
                        0 => (&mut ed.x_dist, &mut ed.x_prim),
                        1 => (&mut ed.y_dist, &mut ed.y_prim),
                        _ => (&mut ed.z_dist, &mut ed.z_prim),
                    };
                    if tt < *dd as f64 {
                        *dd = tt as f32;
                        *pp = index as u32;
                        intersecting = true;
                    }
                }
            }
        };
        test(a, c, b, &mut ed);
        if let Some(d) = d {
            let (cp2, _) = closest_point_on_triangle_to_point(a, d, c, &org);
            let second = (org - cp2).length_sqr();
            if second < dist {
                dist = second;
            }
            test(a, d, c, &mut ed);
        }
        if intersecting {
            self.tree.set_value(ijk, ed);
        }
        dist < 0.86602540378443861
    }

    fn voxelize(&mut self, a: &Vec3d, b: &Vec3d, c: &Vec3d, d: Option<&Vec3d>, index: i32) {
        let mut coords = vec![Coord::floor(*a)];
        let start = coords[0];
        self.eval_primitive(&start, a, b, c, d, index);
        while let Some(ijk) = coords.pop() {
            for off in COORD_OFFSETS.iter() {
                let n = ijk + *off;
                if index != self.last_prim.get_value(&n) {
                    self.last_prim.set_value(&n, index);
                    if self.eval_primitive(&n, a, b, c, d, index) {
                        coords.push(n);
                    }
                }
            }
        }
    }

    fn join(&mut self, rhs: GenEdgeData<'a>) {
        for leaf in take_leaves(rhs.tree) {
            let origin = leaf.origin();
            match self.tree.probe_leaf_mut(&origin) {
                None => self.tree.add_leaf(leaf),
                Some(lhs) => {
                    for (n, rv) in leaf.iter_on() {
                        if !lhs.is_value_on_at(n) {
                            lhs.set_value_on_at(n, *rv);
                        } else {
                            let lv = &mut lhs.buffer_mut()[n];
                            if rv.x_dist < lv.x_dist {
                                lv.x_dist = rv.x_dist;
                                lv.x_prim = rv.x_prim;
                            }
                            if rv.y_dist < lv.y_dist {
                                lv.y_dist = rv.y_dist;
                                lv.y_prim = rv.y_prim;
                            }
                            if rv.z_dist < lv.z_dist {
                                lv.z_dist = rv.z_dist;
                                lv.z_prim = rv.z_prim;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl MeshToVoxelEdgeData {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract voxel edge data (the closest intersection point and corresponding primitive
    /// index) from the given mesh. Points are in grid index space; polygons are quads, or
    /// triangles with the fourth index set to [`INVALID_IDX`].
    pub fn convert(&mut self, points: &[Vec3f], polygons: &[[u32; 4]]) {
        let to_d = |p: &Vec3f| Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64);
        let new_gen =
            || GenEdgeData { tree: Tree::new(EdgeData::default()), last_prim: Tree::new(INVALID_IDX as i32), points };
        let result = (0..polygons.len())
            .into_par_iter()
            .fold(new_gen, |mut g, n| {
                let v = &polygons[n];
                let a = to_d(&g.points[v[0] as usize]);
                let b = to_d(&g.points[v[1] as usize]);
                let c = to_d(&g.points[v[2] as usize]);
                if v[3] != INVALID_IDX {
                    let d = to_d(&g.points[v[3] as usize]);
                    g.voxelize(&a, &b, &c, Some(&d), n as i32);
                } else {
                    g.voxelize(&a, &b, &c, None, n as i32);
                }
                g
            })
            .reduce(new_gen, |mut l, r| {
                l.join(r);
                l
            });
        self.tree = result.tree;
    }

    /// The internal edge data tree.
    pub fn tree(&self) -> &Tree<EdgeData> {
        &self.tree
    }

    /// Return the intersection points (in index space) and primitive indices of the edges of
    /// the voxel cell starting at `ijk` (`MeshToVoxelEdgeData::getEdgeData`).
    pub fn get_edge_data(&self, ijk: &Coord, points: &mut Vec<Vec3d>, primitives: &mut Vec<u32>) {
        let acc = self.tree.accessor();
        let mut coord = *ijk;
        let push = |c: &Coord, axis: usize, data: &EdgeData, points: &mut Vec<Vec3d>, prims: &mut Vec<u32>| {
            let (prim, dist) = match axis {
                0 => (data.x_prim, data.x_dist),
                1 => (data.y_prim, data.y_dist),
                _ => (data.z_prim, data.z_dist),
            };
            if prim != INVALID_IDX {
                let mut p = c.as_vec3d();
                p[axis] += dist as f64;
                points.push(p);
                prims.push(prim);
            }
        };
        let visit = |c: &Coord, axes: &[usize], points: &mut Vec<Vec3d>, prims: &mut Vec<u32>| {
            let (data, on) = acc.probe_value(c);
            if on {
                for &a in axes {
                    push(c, a, &data, points, prims);
                }
            }
        };
        visit(&coord, &[0, 1, 2], points, primitives);
        coord[0] += 1;
        visit(&coord, &[1, 2], points, primitives);
        coord[2] += 1;
        visit(&coord, &[1], points, primitives);
        coord[0] -= 1;
        visit(&coord, &[0, 1], points, primitives);
        coord[1] += 1;
        visit(&coord, &[0], points, primitives);
        coord[2] -= 1;
        visit(&coord, &[0, 2], points, primitives);
        coord[0] += 1;
        visit(&coord, &[2], points, primitives);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn cube_mesh(min: f32, max: f32) -> (Vec<Vec3f>, Vec<[u32; 4]>) {
        let points = vec![
            Vec3f::new(min, min, min),
            Vec3f::new(min, min, max),
            Vec3f::new(max, min, max),
            Vec3f::new(max, min, min),
            Vec3f::new(min, max, min),
            Vec3f::new(min, max, max),
            Vec3f::new(max, max, max),
            Vec3f::new(max, max, min),
        ];
        let faces = vec![[0, 1, 2, 3], [7, 6, 5, 4], [4, 5, 1, 0], [6, 7, 3, 2], [0, 3, 7, 4], [1, 5, 6, 2]];
        (points, faces)
    }

    #[test]
    fn cube_level_set_signs() {
        let (points, quads) = cube_mesh(-5.0, 5.0);
        let xform = Transform::linear(1.0);
        let grid: Grid<f32> = mesh_to_level_set(&xform, &points, &[], &quads, 3.0);
        assert_eq!(grid.grid_class(), GridClass::LevelSet);
        assert!(grid.active_voxel_count() > 0);
        assert!(grid.get_value(&Coord::new(0, 0, 0)) < 0.0);
        assert!(grid.get_value(&Coord::new(20, 0, 0)) > 0.0);
        // Distance to the face at x = 5 from voxel (7, 0, 0) is 2.
        let v = grid.get_value(&Coord::new(7, 0, 0));
        assert!((v - 2.0).abs() < 0.2, "{v}");
        let v = grid.get_value(&Coord::new(3, 0, 0));
        assert!((v + 2.0).abs() < 0.2, "{v}");
    }

    #[test]
    fn polygon_index_grid() {
        let (points, quads) = cube_mesh(-5.0, 5.0);
        let mesh = QuadAndTriangleDataAdapter::new(&points, &quads);
        let mut index = Grid::<i32>::new(INVALID_IDX as i32);
        let g: Grid<f32> = mesh_to_volume(&mesh, &Transform::linear(1.0), 3.0, 3.0, 0, Some(&mut index));
        assert!(index.active_voxel_count() >= g.active_voxel_count());
        for (c, i) in index.tree().active_voxels() {
            assert!((0..6).contains(&i));
            if c == Coord::new(7, 0, 0) {
                assert_eq!(quads[i as usize], [6, 7, 3, 2]); // the +x face
            }
        }
    }

    #[test]
    fn unsigned_field() {
        let points = vec![Vec3f::new(0.0, 0.0, 0.0), Vec3f::new(10.0, 0.0, 0.0), Vec3f::new(0.0, 10.0, 0.0)];
        let tris = vec![[0u32, 1, 2]];
        let grid: Grid<f32> = mesh_to_unsigned_distance_field(&Transform::linear(1.0), &points, &tris, &[], 3.0);
        assert!(grid.active_voxel_count() > 0);
        for (_, v) in grid.tree().active_voxels() {
            assert!(v >= 0.0);
        }
        assert!((grid.get_value(&Coord::new(2, 2, 2)) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn edge_data() {
        let points = vec![Vec3f::new(0.5, -5.0, -5.0), Vec3f::new(0.5, 5.0, -5.0), Vec3f::new(0.5, 0.0, 5.0)];
        let mut ed = MeshToVoxelEdgeData::new();
        ed.convert(&points, &[[0, 1, 2, INVALID_IDX]]);
        let mut pts = Vec::new();
        let mut prims = Vec::new();
        ed.get_edge_data(&Coord::new(0, 0, 0), &mut pts, &mut prims);
        assert!(!pts.is_empty());
        assert!(pts.iter().any(|p| (p[0] - 0.5).abs() < 1e-6));
    }
}
