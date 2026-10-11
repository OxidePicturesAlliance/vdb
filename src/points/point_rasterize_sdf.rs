//! Rasterizing points into narrow-band level sets (port of
//! `openvdb/points/PointRasterizeSDF.h`).
//!
//! * [`rasterize_spheres`] / [`rasterize_spheres_varying`]: the union of spheres around the
//!   points (fixed radius, or a per-point radius attribute);
//! * [`rasterize_smooth_spheres`] / [`rasterize_smooth_spheres_varying`]: the smooth surface
//!   of Zhu and Bridson ("Animating Sand as a Fluid", 2005): weighted averages of nearby point
//!   positions and radii within a search radius.
//!
//! The result is a level set (`GridClass::LevelSet`) with the requested transform (the
//! points' transform by default) and a background of `voxel size × halfband`.

use super::attribute_array::AttributeHandle;
use super::index_filter::{BoundFilter, IndexFilter};
use super::point_data_grid::{PointDataGrid, PointDataLeaf, PointLeaf};
use super::point_statistics::eval_min_max_trees;
use super::{Error, Result};
use crate::grid::{Grid, GridClass};
use crate::math::{BBoxd, Coord, CoordBBox, Transform, Vec3d, Vec3f};
use crate::tree::{LeafNode, Node, Tree};
use crate::util::NodeMask;
use rayon::prelude::*;

/// Default level set half width in voxels (`LEVEL_SET_HALF_WIDTH`).
pub const LEVEL_SET_HALF_WIDTH: f64 = 3.0;

/// Radius of a point, in index space of the surface, with its narrow band.
#[derive(Clone, Copy)]
struct BandRadius {
    r: f64,
    max: f64,
    min_sq: f64,
    max_sq: f64,
}

impl BandRadius {
    fn new(r: f64, halfband: f64) -> Self {
        let min = (r - halfband).max(0.0);
        let max = r + halfband;
        BandRadius { r, max, min_sq: min * min, max_sq: max * max }
    }
    /// The same computation in single precision (OpenVDB's `FixedBandRadius<float>`).
    fn new_f32(r: f32, halfband: f32) -> Self {
        let min = (r - halfband).max(0.0);
        let max = r + halfband;
        BandRadius { r: r as f64, max: max as f64, min_sq: (min * min) as f64, max_sq: (max * max) as f64 }
    }
}

/// Fixed or per-point radii.
#[derive(Clone, Copy)]
enum RadiusSource {
    Fixed(BandRadius),
    /// Attribute position, index-space scale, half band.
    Varying(usize, f32, f32),
}

/// Bounding box of the active voxels of a leaf (the node box if all are active).
fn active_bounds<T: crate::types::Value>(leaf: &LeafNode<T>) -> Option<CoordBBox> {
    let mask = leaf.value_mask();
    match mask.is_constant() {
        Some(false) => None,
        Some(true) => Some(leaf.node_bbox()),
        None => Some(leaf.active_bbox()),
    }
}

/// `SurfaceMaskOp::fill`: activate (in `mask`) the surface voxels within `dist` of the
/// active voxels of a point leaf.
fn fill_mask(mask: &mut Tree<bool>, leaf: &LeafNode<u32>, points: &Transform, surface: &Transform, dist: i32) {
    let Some(b) = active_bounds(leaf) else { return };
    let mut b = surface.world_to_index_cell_centered_bbox(&points.index_coord_bbox_to_world(&b));
    b.expand(dist);
    mask.sparse_fill(&b, true, true);
}

/// Build the narrow-band topology of the surface (`initFixedSdf` / `initVariableSdf`):
/// voxels within `max_dist(leaf)` of the points are active, voxels within `min_dist(leaf)`
/// (if positive) are interior (inactive, negative background).
fn init_sdf(
    points: &PointDataGrid,
    transform: &Transform,
    background: f32,
    dists: impl Fn(&PointDataLeaf<'_>) -> (i32, i32) + Sync,
) -> Grid<f32> {
    let leaves = points.tree().leaf_vec();
    let (mut mask, maskoff) = leaves
        .par_iter()
        .fold(
            || (Tree::<bool>::new(false), Tree::<bool>::new(false)),
            |(mut m, mut off), leaf| {
                let (min, max) = dists(leaf);
                fill_mask(&mut m, leaf.leaf, points.transform(), transform, max);
                if min > 0 {
                    fill_mask(&mut off, leaf.leaf, points.transform(), transform, min);
                }
                (m, off)
            },
        )
        .reduce(
            || (Tree::<bool>::new(false), Tree::<bool>::new(false)),
            |(mut a, mut aoff), (b, boff)| {
                a.topology_union(&b);
                aoff.topology_union(&boff);
                (a, aoff)
            },
        );
    let mut surface = Grid::<f32>::new(background).with_transform(transform.clone());
    surface.set_grid_class(GridClass::LevelSet);
    let has_interior = maskoff.leaf_count() > 0 || maskoff.active_tile_count() > 0;
    if has_interior {
        // mask -= maskoff (whole tiles first, then voxels).
        let off_acc = maskoff.accessor();
        mask.visit_tiles_mut(&mut |bbox, _, active, _| {
            if *active && crate::tools::count::count_active_voxels_in_bbox(&maskoff, &bbox) == bbox.volume() {
                *active = false;
            }
        });
        mask.voxelize_active_tiles();
        mask.par_for_each_leaf_mut(|leaf| {
            let acc = maskoff.accessor();
            let on: Vec<usize> = leaf.value_mask().iter_on().collect();
            for n in on {
                if acc.is_value_on(&leaf.offset_to_global_coord(n)) {
                    leaf.set_active_state_at(n, false);
                }
            }
        });
        mask.prune_inactive();
        surface.tree_mut().topology_union(&mask);
        // Interior values are negative.
        surface.tree_mut().visit_tiles_mut(&mut |bbox, v, active, _| {
            if !*active && off_acc.is_value_on(&bbox.min) {
                *v = -*v;
            }
        });
        surface.tree_mut().par_for_each_leaf_mut(|leaf| {
            let acc = maskoff.accessor();
            let off: Vec<usize> = leaf.value_mask().iter_off().collect();
            for n in off {
                if acc.is_value_on(&leaf.offset_to_global_coord(n)) {
                    let v = *leaf.get_value_at(n);
                    leaf.set_value_only_at(n, -v);
                }
            }
        });
    } else {
        surface.tree_mut().topology_union(&mask);
    }
    surface.tree_mut().voxelize_active_tiles();
    surface
}

/// Visit the points (accepted by `filter`) of the point leaves overlapping the surface
/// region `search` (in surface index space), as OpenVDB's `RasterizePoints` does: `start`
/// is called for each point leaf (returning false skips it), `f` for each point and `end`
/// after each point leaf (returning false stops the traversal).
#[allow(clippy::too_many_arguments)]
fn visit_points<'p, F: IndexFilter, S>(
    points: &'p PointDataGrid,
    surface: &Transform,
    search: CoordBBox,
    filter: &F,
    state: &mut S,
    mut start: impl FnMut(&mut S, &PointDataLeaf<'p>) -> bool,
    mut f: impl FnMut(&mut S, Coord, u32),
    mut end: impl FnMut(&mut S) -> bool,
) {
    let bbox = BBoxd::new(search.min.as_vec3d(), search.max.as_vec3d());
    let search = points.transform().world_to_index_cell_centered_bbox(&surface.index_to_world_bboxd(&bbox));
    let min = search.min.align_down(8);
    let max = search.max;
    let tree = points.tree();
    let mut x = min[0];
    while x <= max[0] {
        let mut y = min[1];
        while y <= max[1] {
            let mut z = min[2];
            while z <= max[2] {
                let origin = Coord::new(x, y, z);
                let pbox = CoordBBox::create_cube(origin, 8).intersection(&search);
                if let (false, Some(pl)) = (pbox.is_empty(), tree.probe_leaf(&origin)) {
                    if start(state, &pl) {
                        let mut bound = filter.bind(pl.leaf, pl.attributes);
                        for c in pbox.iter() {
                            let n = LeafNode::<u32>::coord_to_offset(&c);
                            let active = pl.leaf.is_value_on_at(n);
                            for id in pl.voxel_range(n) {
                                if bound.valid(id, &c, active) {
                                    f(state, c, id);
                                }
                            }
                        }
                        if !end(state) {
                            return;
                        }
                    }
                }
                z += 8;
            }
            y += 8;
        }
        x += 8;
    }
}

/// Per-surface-leaf state of the rasterizers.
struct LeafState<'p, 'l> {
    buf: &'l mut [f32],
    mask: &'l mut NodeMask<3>,
    position: Option<AttributeHandle<'p, Vec3f>>,
    radii: Option<AttributeHandle<'p, f32>>,
}

impl<'p> LeafState<'p, '_> {
    fn start(&mut self, pl: &PointDataLeaf<'p>, radius: RadiusSource) -> bool {
        self.position = pl.attributes.get_by_name("P").and_then(|a| AttributeHandle::new(a).ok());
        if let RadiusSource::Varying(pos, _, _) = radius {
            self.radii = pl.attributes.get(pos).and_then(|a| AttributeHandle::new(a).ok());
        }
        self.position.is_some()
    }
}

/// Union-of-spheres rasterization of one surface leaf (`SphericalTransfer`).
fn rasterize_spherical_leaf<F: IndexFilter>(
    leaf: &mut LeafNode<f32>,
    points: &PointDataGrid,
    surface: &Transform,
    background: f32,
    width: i32,
    radius: RadiusSource,
    filter: &F,
) {
    let Some(bounds) = active_bounds(leaf) else { return };
    let dx = surface.voxel_size()[0];
    let source = points.transform();
    let (buf, mask) = leaf.buffer_and_mask_mut();
    let mut state = LeafState { buf, mask, position: None, radii: None };
    visit_points(
        points,
        surface,
        bounds.expanded(width),
        filter,
        &mut state,
        |s, pl| s.start(pl, radius),
        |s, ijk, id| {
            let Some(p) = &s.position else { return };
            let q = p.get(id);
            let pos = surface.world_to_index(
                source.index_to_world(ijk.as_vec3d() + Vec3d::new(q[0] as f64, q[1] as f64, q[2] as f64)),
            );
            let r = match radius {
                RadiusSource::Fixed(r) => r,
                RadiusSource::Varying(_, scale, hb) => match &s.radii {
                    Some(h) => BandRadius::new_f32(h.get(id) * scale, hb),
                    None => return,
                },
            };
            let b = CoordBBox::new(Coord::floor(pos - r.max), Coord::ceil(pos + r.max)).intersection(&bounds);
            if b.is_empty() {
                return;
            }
            let min2 = if r.min_sq == 0.0 { -1.0 } else { r.min_sq };
            for x in b.min[0]..=b.max[0] {
                let x2 = (x as f64 - pos[0]).powi(2);
                let i = ((x & 7) as usize) << 6;
                for y in b.min[1]..=b.max[1] {
                    let x2y2 = x2 + (y as f64 - pos[1]).powi(2);
                    let ij = i + (((y & 7) as usize) << 3);
                    for z in b.min[2]..=b.max[2] {
                        let offset = ij + (z & 7) as usize;
                        if !s.mask.is_on(offset) {
                            continue;
                        }
                        let d2 = x2y2 + (z as f64 - pos[2]).powi(2);
                        if d2 >= r.max_sq {
                            continue;
                        }
                        if d2 <= min2 {
                            s.buf[offset] = -background;
                            s.mask.set_off(offset);
                            continue;
                        }
                        let d = (dx * (d2.sqrt() - r.r)) as f32;
                        if d < s.buf[offset] {
                            s.buf[offset] = d;
                        }
                    }
                }
            }
        },
        // Stop once every voxel of the surface leaf is inside the level set.
        |s| !s.mask.is_all_off(),
    );
    // finalize: voxels that were not reached are deactivated.
    let on: Vec<usize> = state.mask.iter_on().collect();
    for n in on {
        if state.buf[n] == background {
            state.mask.set_off(n);
        }
    }
}

/// Zhu-Bridson rasterization of one surface leaf (`AveragePositionTransfer`).
#[allow(clippy::too_many_arguments)]
fn rasterize_average_leaf<F: IndexFilter>(
    leaf: &mut LeafNode<f32>,
    points: &PointDataGrid,
    surface: &Transform,
    background: f32,
    width: i32,
    search: f64,
    radius: RadiusSource,
    filter: &F,
) {
    let Some(bounds) = active_bounds(leaf) else { return };
    let origin = leaf.origin();
    let dx = surface.voxel_size()[0];
    let source = points.transform();
    let search_sq = search * search;
    let inv_sq = 1.0 / search_sq;
    let mut weights_p = vec![Vec3d::zero(); 512];
    let mut weights_r = vec![0f64; 512];
    let (buf, mask) = leaf.buffer_and_mask_mut();
    let on: Vec<usize> = mask.iter_on().collect();
    for &n in &on {
        buf[n] = 0.0;
    }
    let mut state = LeafState { buf, mask, position: None, radii: None };
    visit_points(
        points,
        surface,
        bounds.expanded(width),
        filter,
        &mut state,
        |s, pl| s.start(pl, radius),
        |s, ijk, id| {
            let Some(p) = &s.position else { return };
            let q = p.get(id);
            let pws = source.index_to_world(ijk.as_vec3d() + Vec3d::new(q[0] as f64, q[1] as f64, q[2] as f64));
            let pos = surface.world_to_index(pws);
            let rad = match radius {
                RadiusSource::Fixed(r) => r.r,
                RadiusSource::Varying(_, scale, _) => match &s.radii {
                    Some(h) => (h.get(id) * scale) as f64,
                    None => return,
                },
            };
            let b = CoordBBox::new(Coord::floor(pos - search), Coord::ceil(pos + search)).intersection(&bounds);
            if b.is_empty() {
                return;
            }
            for x in b.min[0]..=b.max[0] {
                let x2 = (x as f64 - pos[0]).powi(2);
                let i = ((x & 7) as usize) << 6;
                for y in b.min[1]..=b.max[1] {
                    let x2y2 = x2 + (y as f64 - pos[1]).powi(2);
                    let ij = i + (((y & 7) as usize) << 3);
                    for z in b.min[2]..=b.max[2] {
                        let d2 = x2y2 + (z as f64 - pos[2]).powi(2);
                        if d2 >= search_sq {
                            continue;
                        }
                        let offset = ij + (z & 7) as usize;
                        if !s.mask.is_on(offset) {
                            continue;
                        }
                        let w = (1.0 - d2 * inv_sq).powi(3);
                        s.buf[offset] = (s.buf[offset] as f64 + w) as f32;
                        weights_p[offset] += pws * w;
                        weights_r[offset] += rad * w;
                    }
                }
            }
        },
        |_| true,
    );
    for n in on {
        let w = state.buf[n];
        if w == 0.0 {
            state.mask.set_off(n);
            state.buf[n] = background;
        } else {
            let ijk = origin + LeafNode::<f32>::offset_to_local_coord(n);
            let ws = surface.index_to_world(ijk.as_vec3d());
            let wi = 1.0 / w as f64;
            let p = weights_p[n] * wi - ws;
            let r = weights_r[n] * (wi * dx);
            let mut v = (p.length() - r) as f32;
            if v.abs() >= background {
                v = background.copysign(v);
                state.mask.set_off(n);
            }
            state.buf[n] = v;
        }
    }
}

fn check_points(points: &PointDataGrid) -> Result<()> {
    if let Some(d) = points.tree().descriptor() {
        if d.position("P").is_none() {
            return Err(Error::Lookup("failed to find position attribute".into()));
        }
    }
    Ok(())
}

/// Voxel offset between the points' and the surface's voxel sizes (`SurfaceMaskOp`).
fn voxel_offset(points: &Transform, surface: &Transform) -> i32 {
    (points.voxel_size()[0] / surface.voxel_size()[0]).ceil() as i32
}

/// Rasterize spheres of a fixed world-space `radius` around the points into a level set
/// (`rasterizeSpheres(points, radius, halfband, transform, filter)`).
pub fn rasterize_spheres<F: IndexFilter>(
    points: &PointDataGrid,
    radius: f64,
    halfband: f64,
    transform: Option<&Transform>,
    filter: &F,
) -> Result<Grid<f32>> {
    check_points(points)?;
    let transform = transform.cloned().unwrap_or_else(|| points.transform().clone());
    let vs = transform.voxel_size()[0];
    let background = (vs * halfband) as f32;
    let rad = BandRadius::new(radius / vs, halfband);
    let min_band = (rad.r - halfband).max(0.0);
    let max_band = rad.max;
    let width = max_band.ceil() as i32;
    let off = voxel_offset(points.transform(), &transform);
    let min_dist = (((2.0 * min_band) / 3f64.sqrt()) / 2.0).floor() as i32 - off;
    let min_dist = min_dist.max(0);
    let max_dist = max_band.ceil() as i32 + off;
    let mut surface = init_sdf(points, &transform, background, |_| (min_dist, max_dist));
    surface.tree_mut().par_for_each_leaf_mut(|leaf| {
        rasterize_spherical_leaf(leaf, points, &transform, background, width, RadiusSource::Fixed(rad), filter)
    });
    prune_level_set(&mut surface);
    Ok(surface)
}

/// Rasterize spheres whose radii are the `float` attribute `radius` times `scale` (world
/// units) (`rasterizeSpheres(points, radius, scale, halfband, transform, filter)`).
pub fn rasterize_spheres_varying<F: IndexFilter>(
    points: &PointDataGrid,
    radius: &str,
    scale: f64,
    halfband: f64,
    transform: Option<&Transform>,
    filter: &F,
) -> Result<Grid<f32>> {
    check_points(points)?;
    let transform = transform.cloned().unwrap_or_else(|| points.transform().clone());
    let vs = transform.voxel_size()[0];
    let background = (vs * halfband) as f32;
    let is_scale = (scale / vs) as f32;
    let off = voxel_offset(points.transform(), &transform);
    let Some(ridx) = points.tree().descriptor().map(|d| d.position(radius)) else {
        let mut g = Grid::<f32>::new(background).with_transform(transform);
        g.set_grid_class(GridClass::LevelSet);
        return Ok(g);
    };
    let ridx = ridx.ok_or_else(|| Error::Lookup(format!("failed to find radius attribute \"{}\"", radius)))?;
    let (max, min_tree, max_tree) = match eval_min_max_trees::<f32, F>(points.tree(), radius, filter) {
        Some(((_, max), a, b)) => (max, a, b),
        None => (0.0, Tree::new(0.0), Tree::new(0.0)),
    };
    let (min_acc, max_acc) = (&min_tree, &max_tree);
    let surface_dists = |leaf: &PointDataLeaf<'_>| {
        let mx = max_acc.get_value(&leaf.origin());
        let mn = min_acc.get_value(&leaf.origin());
        let max_d = ((mx * is_scale) as f64 + halfband).ceil() as i32 + off;
        let v = ((mn * is_scale) as f64 - halfband).max(0.0);
        let v = (v - off as f64).max(0.0);
        let min_d = (((2.0 * v) / 3f64.sqrt()) / 2.0) as i32;
        (min_d, max_d)
    };
    let mut surface = init_sdf(points, &transform, background, surface_dists);
    let width = ((max * is_scale) as f64 + halfband).ceil() as i32;
    let rad = RadiusSource::Varying(ridx, is_scale, halfband as f32);
    surface.tree_mut().par_for_each_leaf_mut(|leaf| {
        rasterize_spherical_leaf(leaf, points, &transform, background, width, rad, filter)
    });
    prune_level_set(&mut surface);
    Ok(surface)
}

/// Zhu-Bridson smooth spheres of a fixed world-space `radius`, averaging the points within
/// `search_radius` (`rasterizeSmoothSpheres(points, radius, searchRadius, halfband, ...)`).
pub fn rasterize_smooth_spheres<F: IndexFilter>(
    points: &PointDataGrid,
    radius: f64,
    search_radius: f64,
    halfband: f64,
    transform: Option<&Transform>,
    filter: &F,
) -> Result<Grid<f32>> {
    check_points(points)?;
    let transform = transform.cloned().unwrap_or_else(|| points.transform().clone());
    let vs = transform.voxel_size()[0];
    let background = (vs * halfband) as f32;
    let search = search_radius / vs;
    let bands = BandRadius::new(radius / vs, halfband);
    let off = voxel_offset(points.transform(), &transform);
    let max_dist = bands.max.ceil() as i32 + off;
    let mut surface = init_sdf(points, &transform, background, |_| (0, max_dist));
    let width = search.ceil() as i32;
    let rad = RadiusSource::Fixed(BandRadius::new(radius / vs, 0.0));
    surface.tree_mut().par_for_each_leaf_mut(|leaf| {
        rasterize_average_leaf(leaf, points, &transform, background, width, search, rad, filter)
    });
    surface.tree_mut().prune_inactive();
    Ok(surface)
}

/// Zhu-Bridson smooth spheres with per-point radii (the `float` attribute `radius` times
/// `radius_scale`) (`rasterizeSmoothSpheres(points, radius, radiusScale, searchRadius, ...)`).
pub fn rasterize_smooth_spheres_varying<F: IndexFilter>(
    points: &PointDataGrid,
    radius: &str,
    radius_scale: f64,
    search_radius: f64,
    halfband: f64,
    transform: Option<&Transform>,
    filter: &F,
) -> Result<Grid<f32>> {
    check_points(points)?;
    let transform = transform.cloned().unwrap_or_else(|| points.transform().clone());
    let vs = transform.voxel_size()[0];
    let background = (vs * halfband) as f32;
    let search = search_radius / vs;
    let is_scale = (radius_scale / vs) as f32;
    let off = voxel_offset(points.transform(), &transform);
    let Some(ridx) = points.tree().descriptor().map(|d| d.position(radius)) else {
        let mut g = Grid::<f32>::new(background).with_transform(transform);
        g.set_grid_class(GridClass::LevelSet);
        return Ok(g);
    };
    let ridx = ridx.ok_or_else(|| Error::Lookup(format!("failed to find radius attribute \"{}\"", radius)))?;
    let (min_tree, max_tree) = match eval_min_max_trees::<f32, F>(points.tree(), radius, filter) {
        Some((_, a, b)) => (a, b),
        None => (Tree::new(0.0), Tree::new(0.0)),
    };
    let surface_dists = |leaf: &PointDataLeaf<'_>| {
        let mx = max_tree.get_value(&leaf.origin());
        let mn = min_tree.get_value(&leaf.origin());
        let max_d = ((mx * is_scale) as f64 + halfband).ceil() as i32 + off;
        let v = ((mn * is_scale) as f64 - halfband).max(0.0);
        let v = (v - off as f64).max(0.0);
        let min_d = (((2.0 * v) / 3f64.sqrt()) / 2.0) as i32;
        (min_d, max_d)
    };
    let mut surface = init_sdf(points, &transform, background, surface_dists);
    let width = search.ceil() as i32;
    let rad = RadiusSource::Varying(ridx, is_scale, 0.0);
    surface.tree_mut().par_for_each_leaf_mut(|leaf| {
        rasterize_average_leaf(leaf, points, &transform, background, width, search, rad, filter)
    });
    surface.tree_mut().prune_inactive();
    Ok(surface)
}

/// `tools::pruneLevelSet`: replace inactive-only nodes with ±background tiles.
fn prune_level_set(surface: &mut Grid<f32>) {
    let _ = crate::tools::prune::prune_level_set(surface.tree_mut(), true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::{append_typed_attribute, create_point_data_grid, Codec, NullFilter};

    #[test]
    fn single_sphere() {
        let pos = vec![Vec3f::new(0.0, 0.0, 0.0)];
        let points = create_point_data_grid(&pos, Codec::Null, &Transform::linear(0.1)).unwrap();
        let sdf = rasterize_spheres(&points, 1.0, 3.0, None, &NullFilter).unwrap();
        assert_eq!(sdf.grid_class(), GridClass::LevelSet);
        let bg = *sdf.background();
        assert!((bg - 0.3).abs() < 1e-6);
        // On the surface, the distance is ~0; inside negative, outside positive.
        let v = sdf.get_value(&Coord::new(10, 0, 0));
        assert!(v.abs() < 1e-5, "{}", v);
        assert!((sdf.get_value(&Coord::new(12, 0, 0)) - 0.2).abs() < 1e-5);
        assert!((sdf.get_value(&Coord::new(0, 9, 0)) + 0.1).abs() < 1e-5);
        assert!(sdf.get_value(&Coord::new(0, 0, 0)) < 0.0);
        assert!(sdf.get_value(&Coord::new(30, 0, 0)) > 0.0);
        assert!(sdf.active_voxel_count() > 0);

        // Varying radius with a uniform radius attribute gives the same field.
        let mut p2 = points.clone();
        append_typed_attribute(p2.tree_mut(), "pscale", Codec::Null, 0.5f32, 1, None, false, false).unwrap();
        let sdf2 = rasterize_spheres_varying(&p2, "pscale", 2.0, 3.0, None, &NullFilter).unwrap();
        for c in [Coord::new(10, 0, 0), Coord::new(12, 0, 0), Coord::new(0, 9, 0), Coord::new(7, 7, 0)] {
            assert!((sdf.get_value(&c) - sdf2.get_value(&c)).abs() < 1e-5);
        }

        // Smooth spheres of a single point: distance to the sphere of the point radius.
        let smooth = rasterize_smooth_spheres(&points, 1.0, 2.0, 3.0, None, &NullFilter).unwrap();
        assert!(smooth.get_value(&Coord::new(10, 0, 0)).abs() < 1e-5);
        assert!((smooth.get_value(&Coord::new(11, 0, 0)) - 0.1).abs() < 1e-5);
        let smooth2 = rasterize_smooth_spheres_varying(&p2, "pscale", 2.0, 2.0, 3.0, None, &NullFilter).unwrap();
        assert!((smooth2.get_value(&Coord::new(11, 0, 0)) - 0.1).abs() < 1e-5);
    }
}
