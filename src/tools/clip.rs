//! Clip a grid against a world-space box or a mask grid (port of `openvdb/tools/Clip.h`).
//!
//! The result is a new grid with the input's transform and metadata that keeps (or, with
//! `keep_interior == false`, removes) the voxels inside the clipping region. For level
//! sets, the interior (negative) region counts as part of the grid when computing the
//! clipped topology. Clipping against a frustum is not ported (it requires the non-linear
//! frustum map).

use crate::grid::{Grid, GridClass};
use crate::math::{Coord, CoordBBox, Vec3d};
use crate::tools::prune::prune;
use crate::tools::tree_ops::{topology_difference, topology_intersection, topology_union};
use crate::tools::value_traits::ToolValue;
use crate::tree::{LeafNode, Node, Tree};
use crate::types::Value;

type MaskTree = Tree<bool>;

/// `clip_internal::doClip`: clip `grid` against an index-space mask in its own index space.
fn do_clip<T: ToolValue>(grid: &Grid<T>, clip_mask: &MaskTree, keep_interior: bool) -> Grid<T> {
    let tree = grid.tree();
    let mut grid_mask = MaskTree::new(false);
    topology_union(&mut grid_mask, tree, false);
    if grid.grid_class() == GridClass::LevelSet {
        let acc = tree.accessor();
        for leaf in grid_mask.leaves_mut() {
            if let Some(r) = acc.probe_leaf(&leaf.origin()) {
                for pos in 0..LeafNode::<bool>::SIZE {
                    if !leaf.is_value_on_at(pos) {
                        leaf.set_active_state_at(pos, r.get_value_at(pos).is_negative());
                    }
                }
            }
        }
        grid_mask.visit_tiles_mut(&mut |bbox, _, on, _| {
            *on = acc.get_value(&bbox.min).is_negative();
        });
    }
    if keep_interior {
        topology_intersection(&mut grid_mask, clip_mask);
    } else {
        topology_difference(&mut grid_mask, clip_mask);
    }

    let mut out = grid.copy_with_new_tree();
    {
        let out_tree = out.tree_mut();
        let racc = tree.accessor();
        for mleaf in grid_mask.leaves() {
            let ijk = mleaf.origin();
            let nleaf = out_tree.touch_leaf(&ijk);
            if let Some(r) = racc.probe_leaf(&ijk) {
                for (pos, _) in mleaf.iter_on() {
                    nleaf.set_value_only_at(pos, *r.get_value_at(pos));
                    nleaf.set_active_state_at(pos, r.is_value_on_at(pos));
                }
            } else {
                let (v, on) = racc.probe_value(&ijk);
                for (pos, _) in mleaf.iter_on() {
                    nleaf.set_value_only_at(pos, v);
                    nleaf.set_active_state_at(pos, on);
                }
            }
        }
        let macc = grid_mask.accessor();
        out_tree.visit_tiles_mut(&mut |bbox, value, on, _| {
            if macc.is_value_on(&bbox.min) {
                let (v, active) = racc.probe_value(&bbox.min);
                *value = v;
                if !active {
                    *on = false;
                }
            }
        });
    }
    out
}

/// Clip `grid` against the world-space axis-aligned box `[min, max]` (`tools::clip`):
/// the box is transformed into index space (taking the bounds of its eight corners), and
/// voxels inside it are kept if `keep_interior` is true, removed otherwise.
pub fn clip<T: ToolValue>(grid: &Grid<T>, min: Vec3d, max: Vec3d, keep_interior: bool) -> Grid<T> {
    let corners = [
        min,
        Vec3d::new(max[0], min[1], min[2]),
        Vec3d::new(max[0], max[1], min[2]),
        Vec3d::new(min[0], max[1], min[2]),
        Vec3d::new(min[0], min[1], max[2]),
        Vec3d::new(max[0], min[1], max[2]),
        max,
        Vec3d::new(min[0], max[1], max[2]),
    ];
    let mut lo = grid.transform().world_to_index(corners[0]);
    let mut hi = lo;
    for c in &corners[1..] {
        let p = grid.transform().world_to_index(*c);
        lo = lo.min_elem(&p);
        hi = hi.max_elem(&p);
    }
    let region = CoordBBox::new(Coord::floor(lo), Coord::floor(hi));
    let mut clip_mask = MaskTree::new(false);
    clip_mask.fill(&region, true, true);
    do_clip(grid, &clip_mask, keep_interior)
}

/// Clip `grid` against the active topology of `mask` (`tools::clip(grid, maskGrid)`).
///
/// If the transforms differ, the mask is resampled into `grid`'s index space by sampling it
/// at each voxel center (nearest lower voxel, like OpenVDB's `BoolSampler`).
pub fn clip_with_mask<T: ToolValue, U: Value>(grid: &Grid<T>, mask: &Grid<U>, keep_interior: bool) -> Grid<T> {
    let mut topology = MaskTree::new(false);
    topology_union(&mut topology, mask.tree(), false);
    if !grid.transform().is_approx_equal(mask.transform(), 1e-12) {
        topology = resample_mask(&topology, mask, grid);
    }
    do_clip(grid, &topology, keep_interior)
}

/// Point-sample a mask tree (in `src`'s index space) into `dst`'s index space.
fn resample_mask<U: Value, T: Value>(mask: &MaskTree, src: &Grid<U>, dst: &Grid<T>) -> MaskTree {
    let mut out = MaskTree::new(false);
    let bbox = mask.eval_active_voxel_bbox();
    if bbox.is_empty() {
        return out;
    }
    // Bounds of the source active region in destination index space.
    let mut lo = Vec3d::splat(f64::MAX);
    let mut hi = Vec3d::splat(f64::MIN);
    let expanded = CoordBBox::new(bbox.min, bbox.max.offset(1));
    for c in expanded.corners() {
        let p = dst.transform().world_to_index(src.transform().index_to_world_coord(c));
        lo = lo.min_elem(&p);
        hi = hi.max_elem(&p);
    }
    let region = CoordBBox::new(Coord::floor(lo).offset(-1), Coord::ceil(hi).offset(1));
    {
        let macc = mask.accessor();
        let mut oacc = out.accessor_mut();
        for ijk in region {
            let w = dst.transform().index_to_world_coord(ijk);
            let p = src.transform().world_to_index(w);
            if macc.is_value_on(&Coord::floor(p)) {
                oacc.set_value_on(&ijk, true);
            }
        }
    }
    prune(&mut out, false, true);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Transform, Vec3f};
    use crate::tools::level_set_sphere::create_level_set_sphere;

    #[test]
    fn clip_box_and_mask() {
        let mut g = Grid::<f32>::new(0.0);
        g.tree_mut().fill(&CoordBBox::from_coords(-20, -20, -20, 20, 20, 20), 1.0, true);
        let c = clip(&g, Vec3d::splat(-5.0), Vec3d::splat(5.0), true);
        assert_eq!(c.active_voxel_count(), 11 * 11 * 11);
        let d = clip(&g, Vec3d::splat(-5.0), Vec3d::splat(5.0), false);
        // Like OpenVDB, active tiles of the input are not carried over (only leaf voxels are).
        assert_eq!(d.active_voxel_count(), 38918);
        assert_eq!(d.tree().leaf_count(), 160);
        assert_eq!(c.get_value(&Coord::new(5, 5, 5)), 1.0);

        let mut m = Grid::<bool>::new(false);
        m.tree_mut().fill(&CoordBBox::from_coords(0, 0, 0, 9, 9, 9), true, true);
        let e = clip_with_mask(&g, &m, true);
        assert_eq!(e.active_voxel_count(), 488);
        let mut m2 = Grid::<bool>::new(false).with_transform(Transform::linear(2.0));
        m2.tree_mut().fill(&CoordBBox::from_coords(0, 0, 0, 4, 4, 4), true, true);
        let f = clip_with_mask(&g, &m2, true);
        assert_eq!(f.active_voxel_count(), 488);

        let ls = create_level_set_sphere::<f32>(1.0, Vec3f::splat(0.0), 0.1, 3.0, true).unwrap();
        let half = clip(&ls, Vec3d::new(0.0, -5.0, -5.0), Vec3d::splat(5.0), true);
        assert_eq!(half.get_value(&Coord::new(-5, 0, 0)), *ls.background());
        assert_eq!(half.get_value(&Coord::new(5, 0, 0)), ls.get_value(&Coord::new(5, 0, 0)));
        // Reference values from OpenVDB 10.
        assert_eq!((half.active_voxel_count(), half.tree().leaf_count()), (4021, 28));
        assert_eq!(half.get_value(&Coord::new(2, 0, 0)), -0.3);
        let half2 = clip(&ls, Vec3d::new(0.03, -0.52, -0.27), Vec3d::splat(5.0), false);
        assert_eq!((half2.active_voxel_count(), half2.tree().leaf_count()), (5555, 49));
        let sum: f64 = half2.tree().active_voxels().map(|(_, v)| v as f64).sum();
        assert!((sum - 342.054).abs() < 1e-2);
    }
}
