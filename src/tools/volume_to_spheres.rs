//! Fill a closed level set or fog volume with adaptively sized spheres, and find closest
//! surface points (port of `openvdb/tools/VolumeToSpheres.h`).
//!
//! The private dependencies of the OpenVDB implementation (`sdfInteriorMask`, face-adjacent
//! erosion with `IGNORE_TILES`, `levelSetRebuild` and the `boost::mt11213b` generator) are
//! ported locally so the output matches OpenVDB's.

use super::mesh_support::INVALID_IDX;
use super::mesh_to_volume::{mesh_to_volume, QuadAndTriangleDataAdapter, DISABLE_RENORMALIZATION};
use super::point_scatter::{Mt11213b, UniformPointScatter};
use super::volume_to_mesh::{
    compute_auxiliary_data, compute_points_leaf, identify_surface_intersecting_voxels, leaf_point_count, MeshValue,
    VolumeToMesh, EDGES, INSIDE,
};
use crate::grid::{Grid, GridClass};
use crate::math::{Coord, Transform, Vec3d, Vec3f};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::RealValue;
use crate::util::NodeMask;
use rayon::prelude::*;

/// Finds the closest surface point (on the isosurface of a volume) for arbitrary query
/// points (port of `ClosestSurfacePoint`).
pub struct ClosestSurfacePoint {
    leaf_bounding_spheres: Vec<[f64; 4]>,
    node_bounding_spheres: Vec<[f64; 4]>,
    leaf_ranges: Vec<(usize, usize)>,
    /// Per leaf: the surface point indices of its active voxels, in voxel order.
    leaf_points: Vec<Vec<u32>>,
    surface_points: Vec<Vec3f>,
    max_node_leafs: usize,
    idx_tree: Tree<u32>,
    sign_tree: Tree<i16>,
}

impl ClosestSurfacePoint {
    /// Extract the surface point cloud of `grid` at `isovalue`.
    pub fn create<T: RealValue + MeshValue>(grid: &Grid<T>, isovalue: f32) -> Option<Self> {
        let tree = grid.tree();
        let transform = grid.transform();
        let iso = <T as RealValue>::from_f64(isovalue as f64);
        let mask = identify_surface_intersecting_voxels(tree, iso);
        let (sign_tree, mut idx_tree) = compute_auxiliary_data(&mask, tree, iso);

        let counts: Vec<u32> = sign_tree.leaf_refs().par_iter().map(|l| leaf_point_count(l)).collect();
        let mut offsets = Vec::with_capacity(counts.len());
        let mut total = 0u32;
        for c in &counts {
            offsets.push(total);
            total += *c;
        }
        let per_leaf: Vec<Vec<Vec3f>> = {
            let sign_leaves = sign_tree.leaf_refs();
            idx_tree
                .leaf_refs_mut()
                .into_par_iter()
                .zip(sign_leaves.into_par_iter())
                .zip(offsets.par_iter())
                .map(|((idx, sign), &off)| {
                    let mut pts = Vec::new();
                    let mut fl = Vec::new();
                    compute_points_leaf(idx, sign, tree, off, transform, isovalue as f64, None, &mut pts, &mut fl);
                    pts
                })
                .collect()
        };
        let surface_points: Vec<Vec3f> = per_leaf.into_iter().flatten().collect();

        // Leaves grouped by their parent (level-1) internal node.
        let mut leaf_ranges = Vec::new();
        let mut leaf_points: Vec<Vec<u32>> = Vec::new();
        let mut max_node_leafs = 0;
        for n2 in idx_tree.root().children() {
            for (_, n1) in n2.children() {
                let first = leaf_points.len();
                for (_, leaf) in n1.children() {
                    leaf_points.push(leaf.iter_on().map(|(_, &v)| v).collect());
                }
                max_node_leafs = max_node_leafs.max(leaf_points.len() - first);
                leaf_ranges.push((first, leaf_points.len()));
            }
        }

        let leaf_bounding_spheres: Vec<[f64; 4]> = leaf_points
            .par_iter()
            .map(|idx| {
                let mut avg = Vec3f::zero();
                for &i in idx {
                    avg += surface_points[i as usize];
                }
                if idx.len() > 1 {
                    avg *= (1.0 / idx.len() as f64) as f32;
                }
                let mut max_dist = 0.0f32;
                for &i in idx {
                    let d = (surface_points[i as usize] - avg).length_sqr();
                    if d > max_dist {
                        max_dist = d;
                    }
                }
                [avg[0] as f64, avg[1] as f64, avg[2] as f64, max_dist as f64 * 2.0]
            })
            .collect();

        let node_bounding_spheres: Vec<[f64; 4]> = leaf_ranges
            .par_iter()
            .map(|&(a, b)| {
                let mut avg = Vec3d::zero();
                let count = b as i32 - a as i32;
                for s in &leaf_bounding_spheres[a..b] {
                    avg[0] += s[0];
                    avg[1] += s[1];
                    avg[2] += s[2];
                }
                if count > 1 {
                    avg *= ((1.0 / count as f64) as f32) as f64;
                }
                let mut max_dist = 0.0f64;
                for s in &leaf_bounding_spheres[a..b] {
                    let pos = Vec3d::new(s[0], s[1], s[2]);
                    let d = (pos - avg).length_sqr() + s[3];
                    if d > max_dist {
                        max_dist = d;
                    }
                }
                [avg[0], avg[1], avg[2], max_dist * 2.0]
            })
            .collect();

        Some(ClosestSurfacePoint {
            leaf_bounding_spheres,
            node_bounding_spheres,
            leaf_ranges,
            leaf_points,
            surface_points,
            max_node_leafs,
            idx_tree,
            sign_tree,
        })
    }

    /// The point index tree (surface point index of every surface-intersecting voxel).
    pub fn index_tree(&self) -> &Tree<u32> {
        &self.idx_tree
    }
    /// The sign flag tree of the surface-intersecting voxels.
    pub fn sign_tree(&self) -> &Tree<i16> {
        &self.sign_tree
    }
    /// The extracted surface points (world space).
    pub fn surface_points(&self) -> &[Vec3f] {
        &self.surface_points
    }

    fn eval_leaf(&self, center: Vec3f, leaf: usize, dist: &mut f32, closest: &mut usize) {
        for &i in &self.leaf_points[leaf] {
            let d = (self.surface_points[i as usize] - center).length_sqr();
            if d < *dist {
                *dist = d;
                *closest = i as usize;
            }
        }
    }

    fn eval_node(&self, pos: &Vec3d, node: usize, dist: &mut f32, closest: &mut usize, leaf_dists: &mut [f32]) {
        if node >= self.leaf_ranges.len() {
            return;
        }
        let (a, b) = self.leaf_ranges[node];
        let mut min_dist = *dist;
        let mut min_idx = 0usize;
        let mut updated = false;
        for (n, i) in (a..b).enumerate() {
            let s = &self.leaf_bounding_spheres[i];
            let c = Vec3d::new(s[0], s[1], s[2]);
            let d = (0.0f64).max((*pos - c).length_sqr() - s[3]) as f32;
            leaf_dists[n] = d;
            if d < min_dist {
                min_dist = d;
                min_idx = i;
                updated = true;
            }
        }
        if !updated {
            return;
        }
        let center = Vec3f::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
        self.eval_leaf(center, min_idx, dist, closest);
        for (n, i) in (a..b).enumerate() {
            if leaf_dists[n] < *dist && i != min_idx {
                self.eval_leaf(center, i, dist, closest);
            }
        }
    }

    fn search_impl(&self, points: &mut [Vec3d], transform_points: bool) -> Vec<f32> {
        let mut distances = vec![f32::INFINITY; points.len()];
        points.par_chunks_mut(1024).zip(distances.par_chunks_mut(1024)).for_each(|(pts, dists)| {
            let mut leaf_dists = vec![0.0f32; self.max_node_leafs];
            let mut node_dists = vec![0.0f32; self.leaf_ranges.len()];
            let mut closest = 0usize;
            for (pos, dist) in pts.iter_mut().zip(dists.iter_mut()) {
                let mut min_dist = *dist;
                let mut min_idx = 0usize;
                for (i, s) in self.node_bounding_spheres.iter().enumerate() {
                    let c = Vec3d::new(s[0], s[1], s[2]);
                    let d = (0.0f64).max((*pos - c).length_sqr() - s[3]) as f32;
                    node_dists[i] = d;
                    if d < min_dist {
                        min_dist = d;
                        min_idx = i;
                    }
                }
                self.eval_node(pos, min_idx, dist, &mut closest, &mut leaf_dists);
                for i in 0..node_dists.len() {
                    if node_dists[i] < *dist && i != min_idx {
                        self.eval_node(pos, i, dist, &mut closest, &mut leaf_dists);
                    }
                }
                *dist = dist.sqrt();
                if transform_points {
                    let p = self.surface_points[closest];
                    *pos = Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64);
                }
            }
        });
        distances
    }

    /// Compute the distance from each input point to its closest surface point.
    pub fn search(&self, points: &[Vec3d]) -> Vec<f32> {
        let mut pts = points.to_vec();
        self.search_impl(&mut pts, false)
    }

    /// Overwrite each input point with its closest surface point and return the distances.
    pub fn search_and_replace(&self, points: &mut [Vec3d]) -> Vec<f32> {
        self.search_impl(points, true)
    }
}

// ---------------------------------------------------------------------------------------
// fillWithSpheres
// ---------------------------------------------------------------------------------------

/// `level_set_util_internal::computeInteriorMask`.
fn sdf_interior_mask<T: RealValue>(grid: &Grid<T>, iso: T) -> Grid<bool> {
    let tree = grid.tree();
    let iso = iso.min(*tree.background() - T::tolerance());
    let leaves: Vec<Option<Box<LeafNode<bool>>>> = tree
        .leaf_refs()
        .par_iter()
        .map(|leaf| {
            let mut m = LeafNode::<bool>::new(leaf.origin(), false, false);
            for (i, v) in leaf.buffer().iter().enumerate() {
                if *v < iso {
                    m.set_value_on_at(i, true);
                }
            }
            if m.on_voxel_count() > 0 {
                Some(Box::new(m))
            } else {
                None
            }
        })
        .collect();
    let mut mask = Tree::<bool>::new(false);
    for l in leaves.into_iter().flatten() {
        mask.add_leaf(l);
    }
    let acc = tree.accessor();
    for n2 in mask.root_mut().children_mut() {
        for n1 in n2.children_mut() {
            for slot in 0..4096 {
                if !n1.is_child(slot) && acc.get_value(&n1.offset_to_global_coord(slot)) < iso {
                    n1.set_tile(slot, true, true);
                }
            }
        }
        for slot in 0..32768 {
            if !n2.is_child(slot) && acc.get_value(&n2.offset_to_global_coord(slot)) < iso {
                n2.set_tile(slot, true, true);
            }
        }
    }
    Grid::from_tree(mask).with_transform(grid.transform().clone())
}

/// One iteration of face-adjacent erosion of the leaf voxels (tiles are ignored but count
/// as active neighbours), as `erodeActiveValues(tree, 1, NN_FACE, IGNORE_TILES)`.
fn erode_face_once(tree: &mut Tree<bool>) {
    let new_masks: Vec<Option<NodeMask<3>>> = {
        let t = &*tree;
        t.leaf_refs()
            .par_iter()
            .map(|leaf| {
                if leaf.value_mask().is_all_off() {
                    return None;
                }
                let neighbour_on = |c: &Coord| -> bool {
                    match t.probe_leaf(c) {
                        Some(l) => l.is_value_on(c),
                        None => t.is_value_on(c),
                    }
                };
                let mut m = leaf.value_mask().clone();
                for pos in leaf.value_mask().iter_on() {
                    let local = LeafNode::<bool>::offset_to_local_coord(pos);
                    let mut keep = true;
                    for (axis, d) in [(0usize, -1i32), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)] {
                        let mut n = local;
                        n[axis] += d;
                        let on = if (0..8).contains(&n[axis]) {
                            leaf.value_mask().is_on(LeafNode::<bool>::coord_to_offset(&n))
                        } else {
                            neighbour_on(&(leaf.origin() + n))
                        };
                        if !on {
                            keep = false;
                            break;
                        }
                    }
                    if !keep {
                        m.set_off(pos);
                    }
                }
                Some(m)
            })
            .collect()
    };
    for (leaf, m) in tree.leaves_mut().zip(new_masks) {
        if let Some(m) = m {
            *leaf.value_mask_mut() = m;
        }
    }
}

/// `tools::levelSetRebuild` (volume to mesh to volume) into the given transform, without
/// renormalization.
fn level_set_rebuild<T: RealValue + MeshValue>(
    grid: &Grid<T>,
    iso: f32,
    ex: f32,
    inw: f32,
    xform: &Transform,
) -> Grid<T> {
    let mut mesher = VolumeToMesh::<T>::new(iso as f64, 0.0, true);
    mesher.run(grid);
    let points: Vec<Vec3f> = mesher
        .point_list()
        .par_iter()
        .map(|p| {
            let q = xform.world_to_index(Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64));
            Vec3f::new(q[0] as f32, q[1] as f32, q[2] as f32)
        })
        .collect();
    let mut prims: Vec<[u32; 4]> = Vec::new();
    for pool in mesher.polygon_pool_list() {
        prims.extend_from_slice(&pool.quads);
        prims.extend(pool.triangles.iter().map(|t| [t[0], t[1], t[2], INVALID_IDX]));
    }
    let mesh = QuadAndTriangleDataAdapter::new(&points, &prims);
    mesh_to_volume(&mesh, xform, ex, inw, DISABLE_RENORMALIZATION, None)
}

/// Fill a closed level set or fog volume with adaptively sized spheres
/// (`tools::fillWithSpheres`). Returns spheres as `[x, y, z, radius]` in world space.
///
/// * `sphere_count` — minimum and maximum number of spheres;
/// * `overlapping` — allow spheres to overlap;
/// * `min_radius`, `max_radius` — radius limits in voxel units;
/// * `isovalue` — the isosurface defining the volume;
/// * `instance_count` — number of candidate sphere centers to consider.
#[allow(clippy::too_many_arguments)]
pub fn fill_with_spheres<T: RealValue + MeshValue>(
    grid: &Grid<T>,
    sphere_count: (i32, i32),
    overlapping: bool,
    min_radius: f32,
    max_radius: f32,
    isovalue: f32,
    instance_count: i32,
) -> Vec<[f32; 4]> {
    let mut spheres = Vec::new();
    if grid.tree().is_empty() {
        return spheres;
    }
    let (min_sphere_count, max_sphere_count) = sphere_count;
    if min_sphere_count > max_sphere_count || max_sphere_count < 1 {
        return spheres;
    }
    let mut isovalue = isovalue;
    match grid.grid_class() {
        GridClass::LevelSet => {
            let clamp = *grid.background() - <T as RealValue>::from_f64(crate::math::TOLERANCE_F32 as f64);
            isovalue = isovalue.min(<T as RealValue>::to_f64(clamp) as f32);
        }
        GridClass::FogVolume => {
            isovalue = isovalue.clamp(crate::math::TOLERANCE_F32, 1.0);
        }
        _ => {}
    }

    let mut resampled: Option<Grid<T>> = None;
    let mut num_voxels = grid.active_voxel_count();
    if num_voxels < 10000 {
        let scale = 1.0 / (2.0 * 10000.0 / num_voxels as f64).cbrt();
        let mut xf = grid.transform().clone();
        xf.pre_scale(Vec3d::splat(scale));
        let g = level_set_rebuild(grid, isovalue, 3.0, 3.0, &xf);
        let n = g.active_voxel_count();
        if n > num_voxels {
            num_voxels = n;
            resampled = Some(g);
        }
    }
    let grid = resampled.as_ref().unwrap_or(grid);

    let add_narrow_band_points = num_voxels < 10000;
    let instances = instance_count.max(max_sphere_count);
    let rng = Mt11213b::new(0);
    let transform = grid.transform().clone();

    let mut instance_points: Vec<Vec3d> = Vec::new();
    {
        let mut interior = if grid.grid_class() == GridClass::LevelSet {
            sdf_interior_mask(grid, <T as RealValue>::from_f64(isovalue as f64))
        } else {
            let mut m = Grid::<bool>::new(false).with_transform(transform.clone());
            m.tree_mut().topology_union(grid.tree());
            m
        };
        if !add_narrow_band_points || min_sphere_count <= 0 {
            erode_face_once(interior.tree_mut());
            interior.tree_mut().prune_inactive();
        } else {
            let copy = interior.tree().clone();
            erode_face_once(interior.tree_mut());
            interior.tree_mut().prune_inactive();
            if interior.tree().is_empty() {
                interior.set_tree(copy);
            }
        }
        let scatter_count = if add_narrow_band_points { instances / 2 } else { instances } as u64;
        UniformPointScatter::with_count(&mut instance_points, scatter_count, &rng, 1.0).run(&interior);
    }

    let csp = match ClosestSurfacePoint::create(grid, isovalue) {
        Some(c) => c,
        None => return spheres,
    };

    if instance_points.len() < instances as usize {
        'outer: for leaf in csp.sign_tree().leaves() {
            for (pos, &flags) in leaf.iter_on() {
                if (flags & EDGES) == 0 && (flags & INSIDE) != 0 {
                    instance_points.push(transform.index_to_world_coord(leaf.offset_to_global_coord(pos)));
                }
                if instance_points.len() == instances as usize {
                    break 'outer;
                }
            }
        }
    }

    let mut instance_radius = csp.search(&instance_points);
    let mut largest_radius = 0.0f32;
    let mut largest_idx = 0usize;
    for (n, &r) in instance_radius.iter().enumerate() {
        if r > largest_radius {
            largest_radius = r;
            largest_idx = n;
        }
    }
    let mut mask = vec![0u8; instance_points.len()];
    let min_radius = (min_radius as f64 * transform.voxel_size()[0]) as f32;
    let max_radius = (max_radius as f64 * transform.voxel_size()[0]) as f32;

    let count = (max_sphere_count as usize).min(instance_points.len());
    for s in 0..count {
        largest_radius = max_radius.min(largest_radius);
        if (s as i32) >= min_sphere_count && largest_radius < min_radius {
            break;
        }
        let p = instance_points[largest_idx];
        let sphere = [p[0] as f32, p[1] as f32, p[2] as f32, largest_radius];
        spheres.push(sphere);
        mask[largest_idx] = 1;

        // UpdatePoints
        let mut radius = 0.0f32;
        let mut index = 0usize;
        for n in 0..instance_points.len() {
            if mask[n] != 0 {
                continue;
            }
            let q = instance_points[n];
            let pos = Vec3f::new(q[0] as f32 - sphere[0], q[1] as f32 - sphere[1], q[2] as f32 - sphere[2]);
            let dist = pos.length();
            if dist < sphere[3] {
                mask[n] = 1;
                continue;
            }
            if !overlapping {
                instance_radius[n] = instance_radius[n].min(dist - sphere[3]);
            }
            if instance_radius[n] > radius {
                radius = instance_radius[n];
                index = n;
            }
        }
        largest_radius = radius;
        largest_idx = index;
    }
    spheres
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::mesh_support::signed_flood_fill_with_values;

    fn sphere_grid(radius: f32, voxel: f32, hw: f32) -> Grid<f32> {
        let mut g = Grid::<f32>::new(voxel * hw).with_transform(Transform::linear(voxel as f64));
        g.set_grid_class(GridClass::LevelSet);
        let r0 = radius / voxel;
        let n = (r0 + hw).ceil() as i32 + 1;
        let tree = g.tree_mut();
        for i in -n..=n {
            for j in -n..=n {
                for k in -n..=n {
                    let d = ((i * i + j * j + k * k) as f32).sqrt() - r0;
                    if d.abs() < hw {
                        tree.set_value(&Coord::new(i, j, k), d * voxel);
                    }
                }
            }
        }
        signed_flood_fill_with_values(tree, voxel * hw, -voxel * hw);
        g
    }

    #[test]
    fn closest_surface_point() {
        let g = sphere_grid(3.0, 0.1, 3.0);
        let csp = ClosestSurfacePoint::create(&g, 0.0).unwrap();
        let mut pts = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(5.0, 0.0, 0.0), Vec3d::new(1.0, 1.0, 1.0)];
        let d = csp.search_and_replace(&mut pts);
        assert!((d[0] - 3.0).abs() < 0.05, "{:?}", d);
        assert!((d[1] - 2.0).abs() < 0.05);
        assert!((d[2] - (3.0 - 3f32.sqrt())).abs() < 0.05);
        assert!((pts[1] - Vec3d::new(3.0, 0.0, 0.0)).length() < 0.1);
    }

    #[test]
    fn spheres_inside_sphere() {
        let g = sphere_grid(3.0, 0.1, 3.0);
        let spheres = fill_with_spheres(&g, (1, 10), false, 1.0, f32::MAX, 0.0, 2000);
        assert!(!spheres.is_empty() && spheres.len() <= 10);
        // The largest sphere is (nearly) the whole volume.
        assert!(spheres[0][3] > 2.5 && spheres[0][3] < 3.05, "{:?}", spheres[0]);
        for s in &spheres {
            let c = Vec3d::new(s[0] as f64, s[1] as f64, s[2] as f64);
            assert!(c.length() + s[3] as f64 <= 3.1);
        }
    }
}
