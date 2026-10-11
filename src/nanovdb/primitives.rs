//! Narrow-band level sets and fog volumes of simple shapes (port of the sphere, torus and
//! box generators of `nanovdb/tools/CreatePrimitives.h`).
//!
//! The source grids are built voxel by voxel exactly as the C++ `initSphere`, `initTorus`
//! and `initBox` do (single-precision distances, leap-frogging outside the band), then
//! sign-flood-filled with NanoVDB's own `sdfToLevelSet`, so that the resulting NanoVDB
//! grids are byte-for-byte identical to `nanovdb::tools::createLevelSetSphere<float>` etc.

use super::create::{BuildFrom, CreateNanoGrid};
use super::data::Map;
use super::handle::GridHandle;
use crate::grid::{FloatGrid, GridClass};
use crate::math::{Coord, Transform, Vec3d};
use crate::tree::{Internal1, Internal2, LeafNode, Node, RootEntry};

fn new_grid(voxel_size: f64, half_width: f64, origin: Vec3d, name: &str) -> FloatGrid {
    let mut g = FloatGrid::new((half_width * voxel_size) as f32);
    g.set_name(name);
    let mut t = Transform::linear(voxel_size);
    t.post_translate(origin);
    g.set_transform(t);
    g
}

fn center_in_voxels(center: Vec3d, origin: Vec3d, voxel_size: f64) -> [f32; 3] {
    let v = voxel_size as f32;
    [(center[0] - origin[0]) as f32 / v, (center[1] - origin[1]) as f32 / v, (center[2] - origin[2]) as f32 / v]
}

/// Leap-frog step outside the narrow band (`m += Floor(d - halfWidth)`).
#[inline]
fn leap(d: f32, half_width: f64) -> i32 {
    1 + (d as f64 - half_width).floor() as i32
}

/// The source grid of a level-set sphere (`initSphere` followed by `sdfToLevelSet`).
pub fn level_set_sphere_grid(
    radius: f64,
    center: Vec3d,
    voxel_size: f64,
    half_width: f64,
    origin: Vec3d,
    name: &str,
) -> FloatGrid {
    assert!(radius > 0.0 && voxel_size > 0.0 && half_width > 0.0, "sphere: invalid arguments");
    let mut grid = new_grid(voxel_size, half_width, origin, name);
    let r0 = (radius / voxel_size) as f32;
    let rmax = r0 + half_width as f32;
    if r0 >= 1.5 {
        let c = center_in_voxels(center, origin, voxel_size);
        let (imin, imax) = ((c[0] - rmax).floor() as i32, (c[0] + rmax).ceil() as i32);
        let (jmin, jmax) = ((c[1] - rmax).floor() as i32, (c[1] + rmax).ceil() as i32);
        let (kmin, kmax) = ((c[2] - rmax).floor() as i32, (c[2] + rmax).ceil() as i32);
        let dx = voxel_size as f32;
        let tree = grid.tree_mut();
        for i in imin..=imax {
            let x2 = (i as f32 - c[0]) * (i as f32 - c[0]);
            for j in jmin..=jmax {
                let x2y2 = (j as f32 - c[1]) * (j as f32 - c[1]) + x2;
                let mut k = kmin;
                while k <= kmax {
                    let v = (x2y2 + (k as f32 - c[2]) * (k as f32 - c[2])).sqrt() - r0;
                    let d = v.abs();
                    let mut m = 1;
                    if (d as f64) < half_width {
                        tree.set_value(&Coord::new(i, j, k), dx * v);
                    } else {
                        m = leap(d, half_width);
                    }
                    k += m;
                }
            }
        }
    }
    sdf_to_level_set(&mut grid);
    grid
}

/// The source grid of a level-set torus in the xz-plane (`initTorus` + `sdfToLevelSet`).
pub fn level_set_torus_grid(
    major_radius: f64,
    minor_radius: f64,
    center: Vec3d,
    voxel_size: f64,
    half_width: f64,
    origin: Vec3d,
    name: &str,
) -> FloatGrid {
    assert!(minor_radius > 0.0 && major_radius > minor_radius, "torus: invalid radii");
    assert!(voxel_size > 0.0 && half_width > 0.0, "torus: invalid arguments");
    let mut grid = new_grid(voxel_size, half_width, origin, name);
    let r1 = (major_radius / voxel_size) as f32;
    let r2 = (minor_radius / voxel_size) as f32;
    let rmax1 = r1 + r2 + half_width as f32;
    let rmax2 = r2 + half_width as f32;
    if r2 >= 1.5 {
        let c = center_in_voxels(center, origin, voxel_size);
        let (imin, imax) = ((c[0] - rmax1).floor() as i32, (c[0] + rmax1).ceil() as i32);
        let (jmin, jmax) = ((c[1] - rmax2).floor() as i32, (c[1] + rmax2).ceil() as i32);
        let (kmin, kmax) = ((c[2] - rmax1).floor() as i32, (c[2] + rmax1).ceil() as i32);
        let dx = voxel_size as f32;
        let tree = grid.tree_mut();
        for i in imin..=imax {
            let x2 = (i as f32 - c[0]) * (i as f32 - c[0]);
            for k in kmin..=kmax {
                let a = ((k as f32 - c[2]) * (k as f32 - c[2]) + x2).sqrt() - r1;
                let x2z2 = a * a;
                let mut j = jmin;
                while j <= jmax {
                    let v = (x2z2 + (j as f32 - c[1]) * (j as f32 - c[1])).sqrt() - r2;
                    let d = v.abs();
                    let mut m = 1;
                    if (d as f64) < half_width {
                        tree.set_value(&Coord::new(i, j, k), dx * v);
                    } else {
                        m = leap(d, half_width);
                    }
                    j += m;
                }
            }
        }
    }
    sdf_to_level_set(&mut grid);
    grid
}

/// The source grid of a level-set box (`initBox` + `sdfToLevelSet`).
#[allow(clippy::too_many_arguments)]
pub fn level_set_box_grid(
    width: f64,
    height: f64,
    depth: f64,
    center: Vec3d,
    voxel_size: f64,
    half_width: f64,
    origin: Vec3d,
    name: &str,
) -> FloatGrid {
    assert!(width > 0.0 && height > 0.0 && depth > 0.0, "box: invalid size");
    assert!(voxel_size > 0.0 && half_width > 0.0, "box: invalid arguments");
    let mut grid = new_grid(voxel_size, half_width, origin, name);
    let two_dx = (2.0 * voxel_size as f32) as f64;
    let r = [(width / two_dx) as f32, (height / two_dx) as f32, (depth / two_dx) as f32];
    let rmin = r[0].min(r[1]).min(r[2]);
    if rmin >= 1.5 {
        let c = center_in_voxels(center, origin, voxel_size);
        let hw = half_width as f32;
        let lo: Vec<i32> = (0..3).map(|i| (c[i] - r[i] - hw).floor() as i32).collect();
        let hi: Vec<i32> = (0..3).map(|i| (c[i] + r[i] + hw).ceil() as i32).collect();
        let pos = |x: f32| if x > 0.0 { x } else { 0.0 };
        let neg = |x: f32| if x < 0.0 { x } else { 0.0 };
        let max = |a: f32, b: f32| if a > b { a } else { b };
        let dx = voxel_size as f32;
        let tree = grid.tree_mut();
        for p0 in lo[0]..=hi[0] {
            let q1 = (p0 as f32 - c[0]).abs() - r[0];
            let x2 = pos(q1) * pos(q1);
            for p1 in lo[1]..=hi[1] {
                let q2 = (p1 as f32 - c[1]).abs() - r[1];
                let q0 = max(q1, q2);
                let x2y2 = x2 + pos(q2) * pos(q2);
                let mut p2 = lo[2];
                while p2 <= hi[2] {
                    let q3 = (p2 as f32 - c[2]).abs() - r[2];
                    let v = (x2y2 + pos(q3) * pos(q3)).sqrt() + neg(max(q0, q3));
                    let d = v.abs();
                    let mut m = 1;
                    if (d as f64) < half_width {
                        tree.set_value(&Coord::new(p0, p1, p2), dx * v);
                    } else {
                        m = leap(d, half_width);
                    }
                    p2 += m;
                }
            }
        }
    }
    sdf_to_level_set(&mut grid);
    grid
}

// ------------------------------------------------------------------------------------------
// sdfToLevelSet / levelSetToFog (tools::build)
// ------------------------------------------------------------------------------------------

fn leaf_fill(leaf: &mut LeafNode<f32>, outside: f32) {
    let first = match leaf.value_mask().find_first_on() {
        Some(f) => f,
        None => return,
    };
    let (buf, mask) = leaf.buffer_and_mask_mut();
    let mut x_inside = buf[first] < 0.0;
    for x in 0..8 {
        let x00 = x << 6;
        if mask.is_on(x00) {
            x_inside = buf[x00] < 0.0;
        }
        let mut y_inside = x_inside;
        for y in 0..8 {
            let xy0 = x00 + (y << 3);
            if mask.is_on(xy0) {
                y_inside = buf[xy0] < 0.0;
            }
            let mut z_inside = y_inside;
            for z in 0..8 {
                let xyz = xy0 + z;
                if mask.is_on(xyz) {
                    z_inside = buf[xyz] < 0.0;
                } else {
                    buf[xyz] = if z_inside { -outside } else { outside };
                }
            }
        }
    }
}

trait FirstLast {
    fn first_value(&self) -> f32;
    fn last_value(&self) -> f32;
}
impl FirstLast for LeafNode<f32> {
    fn first_value(&self) -> f32 {
        self.buffer()[0]
    }
    fn last_value(&self) -> f32 {
        self.buffer()[511]
    }
}
impl<C: Node<Value = f32> + FirstLast, const L: u32> FirstLast for crate::tree::InternalNode<C, L> {
    fn first_value(&self) -> f32 {
        self.child(0).map(|c| c.first_value()).unwrap_or(self.values()[0])
    }
    fn last_value(&self) -> f32 {
        let n = (1usize << (3 * L)) - 1;
        self.child(n).map(|c| c.last_value()).unwrap_or(self.values()[n])
    }
}

fn internal_fill<C: Node<Value = f32> + FirstLast, const L: u32>(
    node: &mut crate::tree::InternalNode<C, L>,
    outside: f32,
) {
    let first = match node.child_mask().find_first_on() {
        Some(f) => f,
        None => return,
    };
    let dim = 1usize << L;
    let mut writes = Vec::new();
    let mut x_inside = node.child(first).unwrap().first_value() < 0.0;
    for x in 0..dim {
        let x00 = x << (2 * L);
        if let Some(c) = node.child(x00) {
            x_inside = c.last_value() < 0.0;
        }
        let mut y_inside = x_inside;
        for y in 0..dim {
            let xy0 = x00 + (y << L);
            if let Some(c) = node.child(xy0) {
                y_inside = c.last_value() < 0.0;
            }
            let mut z_inside = y_inside;
            for z in 0..dim {
                let xyz = xy0 + z;
                match node.child(xyz) {
                    Some(c) => z_inside = c.last_value() < 0.0,
                    None => writes.push((xyz, if z_inside { -outside } else { outside })),
                }
            }
        }
    }
    let values = node.values_mut();
    for (n, v) in writes {
        values[n] = v;
    }
}

/// Sign-flood-fill a narrow-band distance field with NanoVDB's bottom-up algorithm and mark
/// it as a level set (`nanovdb::tools::build::sdfToLevelSet`). Inactive values become
/// `-background` inside and `background` outside; inactive root tiles are inserted between
/// interior top-level nodes on the same z-scanline.
pub fn sdf_to_level_set(grid: &mut FloatGrid) {
    grid.set_grid_class(GridClass::LevelSet);
    let outside = *grid.background();
    let root = grid.tree_mut().root_mut();
    for upper in root.children_mut() {
        for lower in upper.children_mut() {
            for leaf in lower.children_mut() {
                leaf_fill(leaf, outside);
            }
        }
    }
    for upper in root.children_mut() {
        for lower in upper.children_mut() {
            internal_fill::<LeafNode<f32>, 4>(lower, outside);
        }
    }
    for upper in root.children_mut() {
        internal_fill::<Internal1<f32>, 5>(upper, outside);
    }
    // Root: z-scanline fill between consecutive interior children.
    let keys: Vec<(Coord, f32, f32)> =
        root.table().iter().filter_map(|(k, e)| e.child().map(|c| (*k, c.first_value(), c.last_value()))).collect();
    let dim = Internal2::<f32>::DIM;
    for w in keys.windows(2) {
        let (a, b) = (w[0], w[1]);
        let d = b.0 - a.0;
        if d[0] != 0 || d[1] != 0 || d[2] == dim {
            continue;
        }
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN values are not inside
        if !(a.2 < 0.0) || !(b.1 < 0.0) {
            continue;
        }
        let mut c = a.0 + Coord::new(0, 0, dim);
        while c[2] != b.0[2] {
            root.add_root_tile(&c, -outside, false);
            c[2] += dim;
        }
    }
}

/// Convert a level set into a fog volume (`nanovdb::tools::build::levelSetToFog` without
/// rebuilding): values outside become inactive zeros, values inside the band become
/// `v / -background` and the interior becomes active ones; nodes left without active
/// values are pruned.
pub fn level_set_to_fog(grid: &mut FloatGrid) {
    grid.set_grid_class(GridClass::FogVolume);
    let d = -*grid.background();
    let w = 1.0f32 / d;
    let op = |v: &mut f32| -> bool {
        if *v > 0.0 {
            *v = 0.0;
            return false;
        }
        *v = if *v > d { *v * w } else { 1.0 };
        true
    };
    let root = grid.tree_mut().root_mut();
    for upper in root.children_mut() {
        for lower in upper.children_mut() {
            for leaf in lower.children_mut() {
                let (buf, mask) = leaf.buffer_and_mask_mut();
                for (i, v) in buf.iter_mut().enumerate() {
                    let on = op(v);
                    mask.set(i, on);
                }
            }
        }
    }
    for upper in root.children_mut() {
        for lower in upper.children_mut() {
            prune_or_op::<LeafNode<f32>, 4>(lower, &op, |leaf| leaf.value_mask().is_all_off());
        }
    }
    for upper in root.children_mut() {
        prune_or_op::<Internal1<f32>, 5>(upper, &op, |c| c.child_mask().is_all_off() && c.value_mask().is_all_off());
    }
    let keys: Vec<Coord> = root.table().keys().copied().collect();
    for k in keys {
        let replace = match root.table_mut().get_mut(&k).unwrap() {
            RootEntry::Tile { value, active } => {
                *active = op(value);
                None
            }
            RootEntry::Child(c) => {
                (c.child_mask().is_all_off() && c.value_mask().is_all_off()).then(|| c.first_value())
            }
        };
        if let Some(v) = replace {
            root.add_root_tile(&k, v, false);
        }
    }
}

fn prune_or_op<C: Node<Value = f32> + FirstLast, const L: u32>(
    node: &mut crate::tree::InternalNode<C, L>,
    op: &dyn Fn(&mut f32) -> bool,
    empty: impl Fn(&C) -> bool,
) {
    for n in 0..(1usize << (3 * L)) {
        if let Some(c) = node.child(n) {
            if empty(c) {
                let v = c.first_value();
                let active = node.value_mask().is_on(n);
                node.set_tile(n, v, active);
            }
        } else {
            let mut v = node.values()[n];
            let on = op(&mut v);
            node.values_mut()[n] = v;
            node.value_mask_mut().set(n, on);
        }
    }
}

// ------------------------------------------------------------------------------------------
// NanoVDB handles
// ------------------------------------------------------------------------------------------

/// A converter for a primitive's source grid using the exact map of the C++ builder
/// (`Map::set(voxelSize, origin)`), to choose the build type, statistics, checksum or
/// dithering.
pub fn primitive_converter(grid: &FloatGrid, voxel_size: f64, origin: Vec3d) -> CreateNanoGrid<'_, f32> {
    CreateNanoGrid::new(grid).map(Map::from_scale_translation(voxel_size, origin))
}

/// A NanoVDB level-set sphere (`nanovdb::tools::createLevelSetSphere<B>`), with default
/// statistics and checksum. `B` is `f32` or one of the fixed-rate quantized types.
pub fn create_level_set_sphere<B: BuildFrom<f32>>(
    radius: f64,
    center: Vec3d,
    voxel_size: f64,
    half_width: f64,
    origin: Vec3d,
    name: &str,
) -> GridHandle {
    let g = level_set_sphere_grid(radius, center, voxel_size, half_width, origin, name);
    primitive_converter(&g, voxel_size, origin).get_handle::<B>()
}

/// A NanoVDB level-set torus (`nanovdb::tools::createLevelSetTorus<B>`).
#[allow(clippy::too_many_arguments)]
pub fn create_level_set_torus<B: BuildFrom<f32>>(
    major_radius: f64,
    minor_radius: f64,
    center: Vec3d,
    voxel_size: f64,
    half_width: f64,
    origin: Vec3d,
    name: &str,
) -> GridHandle {
    let g = level_set_torus_grid(major_radius, minor_radius, center, voxel_size, half_width, origin, name);
    primitive_converter(&g, voxel_size, origin).get_handle::<B>()
}

/// A NanoVDB level-set box (`nanovdb::tools::createLevelSetBox<B>`).
#[allow(clippy::too_many_arguments)]
pub fn create_level_set_box<B: BuildFrom<f32>>(
    width: f64,
    height: f64,
    depth: f64,
    center: Vec3d,
    voxel_size: f64,
    half_width: f64,
    origin: Vec3d,
    name: &str,
) -> GridHandle {
    let g = level_set_box_grid(width, height, depth, center, voxel_size, half_width, origin, name);
    primitive_converter(&g, voxel_size, origin).get_handle::<B>()
}

/// A NanoVDB fog-volume sphere (`nanovdb::tools::createFogVolumeSphere<B>`).
pub fn create_fog_volume_sphere<B: BuildFrom<f32>>(
    radius: f64,
    center: Vec3d,
    voxel_size: f64,
    half_width: f64,
    origin: Vec3d,
    name: &str,
) -> GridHandle {
    let mut g = level_set_sphere_grid(radius, center, voxel_size, half_width, origin, name);
    level_set_to_fog(&mut g);
    primitive_converter(&g, voxel_size, origin).get_handle::<B>()
}
