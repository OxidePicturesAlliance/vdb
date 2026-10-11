//! Convert the active topology of any grid into a narrow-band level set (port of
//! `openvdb/tools/TopologyToLevelSet.h`).
//!
//! Renormalization and smoothing use [`LevelSetTracker`] and [`LevelSetFilter`] with the
//! first-order upwind scheme, exactly like OpenVDB.

use crate::grid::{Grid, GridClass};
use crate::math::finite_difference::BiasedGradientScheme;
use crate::tools::level_set_filter::LevelSetFilter;
use crate::tools::level_set_tracker::LevelSetTracker;
use crate::tools::morphology::{dilate_active_values, erode_active_values, NearestNeighbors, TilePolicy};
use crate::tools::prune::{level_set_prune, prune_inactive};
use crate::tools::tree_ops::{topology_copy, topology_copy_with_values, topology_difference};
use crate::tools::value_traits::{ToolError, ToolResult};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::Value;

fn tracker_error(e: crate::tools::level_set_tracker::LevelSetError) -> ToolError {
    ToolError::Value(e.to_string())
}

/// `ttls_internal::normalizeLevelSet`: `norm_count` first-order renormalization steps, then
/// trimming and pruning of the narrow band.
fn normalize_level_set(grid: &mut Grid<f32>, norm_count: i32) -> ToolResult<()> {
    let mut tracker = LevelSetTracker::new(grid).map_err(tracker_error)?;
    tracker.set_spatial_scheme(BiasedGradientScheme::FirstBias);
    tracker.set_norm_count(norm_count.max(0) as usize);
    tracker.normalize();
    tracker.prune();
    Ok(())
}

/// `ttls_internal::smoothLevelSet`: `iterations` mean-value filter passes on a copy, which is
/// then offset by half a voxel and min-combined with the original before renormalization.
fn smooth_level_set(grid: &mut Grid<f32>, iterations: i32, half_width: i32) -> ToolResult<()> {
    let mut filtered = grid.clone();
    {
        let mut filter = LevelSetFilter::new(&mut filtered).map_err(tracker_error)?;
        filter.set_spatial_scheme(BiasedGradientScheme::FirstBias);
        for _ in 0..iterations {
            filter.mean(1);
        }
    }
    let offset = -(0.5 * grid.voxel_size()[0] as f32);
    let f: &Tree<f32> = filtered.tree();
    grid.tree_mut().par_for_each_leaf_mut(|leaf: &mut LeafNode<f32>| {
        if let Some(r) = f.probe_leaf(&leaf.origin()) {
            let (buf, mask) = leaf.buffer_and_mask_mut();
            for pos in mask.iter_on() {
                buf[pos] = buf[pos].min(offset + *r.get_value_at(pos));
            }
        }
    });
    normalize_level_set(grid, half_width)
}

/// Return a narrow-band level set (with `half_width` voxels on each side) whose zero
/// crossing follows the boundary of the active voxels of `grid`
/// (`tools::topologyToLevelSet`).
///
/// The topology is first closed morphologically (`closing_steps` dilations followed by as
/// many erosions, plus `dilation` extra dilations), then the level set is built from the
/// implicit zero crossing and renormalized. `smoothing_steps` mean-value filter passes
/// may optionally be applied. Returns an error for non-uniform voxels.
pub fn topology_to_level_set<T: Value>(
    grid: &Grid<T>,
    half_width: i32,
    closing_steps: i32,
    dilation: i32,
    smoothing_steps: i32,
) -> ToolResult<Grid<f32>> {
    let half_width = half_width.max(1);
    let closing_steps = closing_steps.max(0);
    let dilation = dilation.max(0);
    if !grid.transform().has_uniform_scale() {
        return Err(ToolError::Value("Non-uniform voxels are not supported!".into()));
    }
    let mut mask: Tree<bool> = topology_copy(grid.tree(), false);
    dilate_active_values(&mut mask, closing_steps + dilation, NearestNeighbors::Face, TilePolicy::PreserveTiles, true);
    erode_active_values(&mut mask, closing_steps, NearestNeighbors::Face, TilePolicy::PreserveTiles, true);
    prune_inactive(&mut mask, true);

    let dx = grid.voxel_size()[0] as f32;
    let background = dx * half_width as f32;
    let mut ls: Tree<f32> = topology_copy_with_values(&mask, background, -background);
    rayon::join(
        || {
            erode_active_values(&mut mask, half_width, NearestNeighbors::Face, TilePolicy::PreserveTiles, true);
            prune_inactive(&mut mask, true);
        },
        || dilate_active_values(&mut ls, half_width, NearestNeighbors::Face, TilePolicy::PreserveTiles, true),
    );
    topology_difference(&mut ls, &mask);
    ls.voxelize_active_tiles();
    level_set_prune(&mut ls, background, -background, true);

    let mut out = Grid::from_tree(ls).with_transform(grid.transform().clone());
    out.set_grid_class(GridClass::LevelSet);
    normalize_level_set(&mut out, 3 * half_width)?;
    if smoothing_steps > 0 {
        smooth_level_set(&mut out, smoothing_steps, half_width)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, CoordBBox};

    #[test]
    fn cube_topology() {
        let mut g = Grid::<bool>::new(false);
        g.tree_mut().fill(&CoordBBox::from_coords(0, 0, 0, 19, 19, 19), true, true);
        let ls = topology_to_level_set(&g, 3, 1, 0, 0).unwrap();
        assert_eq!(ls.grid_class(), GridClass::LevelSet);
        assert!(ls.get_value(&Coord::new(10, 10, 10)) < 0.0);
        assert!(ls.get_value(&Coord::new(30, 10, 10)) > 0.0);
        let v = ls.get_value(&Coord::new(20, 10, 10));
        assert!(v.abs() <= 1.0, "{v}");
        let smooth = topology_to_level_set(&g, 3, 1, 0, 2).unwrap();
        assert!(smooth.get_value(&Coord::new(10, 10, 10)) < 0.0);
    }
}
