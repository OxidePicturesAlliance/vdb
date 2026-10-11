//! Convert the active topology of any grid into a narrow-band level set (port of
//! `openvdb/tools/TopologyToLevelSet.h`).
//!
//! The PDE-based renormalization (`LevelSetFilter::normalize` with first-order upwinding
//! and forward-Euler time integration), the trimming of the narrow band and the box
//! filter used for optional smoothing are implemented privately in this module, following
//! `LevelSetTracker.h` and `LevelSetFilter.h`.

use crate::grid::{Grid, GridClass};
use crate::tools::level_set_measure::first_bias_norm_sq_grad;
use crate::tools::morphology::{dilate_active_values, erode_active_values, NearestNeighbors, TilePolicy};
use crate::tools::prune::{level_set_prune, prune_inactive};
use crate::tools::tree_ops::{topology_copy, topology_copy_with_values, topology_difference};
use crate::tools::value_traits::{ToolError, ToolResult};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::Value;
use rayon::prelude::*;

/// One forward-Euler renormalization step of all active voxels (`Normalizer::euler01`
/// with `FIRST_BIAS`).
fn normalize_step(tree: &mut Tree<f32>, dx: f32) {
    let dt = dx * 0.3;
    let inv_dx = 1.0 / dx;
    let results: Vec<Vec<f32>> = {
        let t = &*tree;
        t.leaf_refs()
            .par_iter()
            .map(|leaf| {
                let acc = t.accessor();
                let mut out = leaf.buffer().to_vec();
                for (pos, phi0) in leaf.iter_on() {
                    let ijk = leaf.offset_to_global_coord(pos);
                    let phi0 = *phi0;
                    let nsg = first_bias_norm_sq_grad(&acc, &ijk, phi0);
                    let mut v = phi0 / ((phi0 * phi0 + nsg).sqrt() + crate::math::TOLERANCE_F32);
                    v = phi0 - dt * v * (nsg.sqrt() * inv_dx - 1.0);
                    out[pos] = v;
                }
                out
            })
            .collect()
    };
    for (leaf, r) in tree.leaves_mut().zip(results) {
        leaf.buffer_mut().copy_from_slice(&r);
    }
}

/// `LevelSetTracker::prune`: deactivate voxels outside the band and prune the tree.
fn trim_and_prune(tree: &mut Tree<f32>) {
    let gamma = *tree.background();
    tree.par_for_each_leaf_mut(|leaf: &mut LeafNode<f32>| {
        for pos in 0..LeafNode::<f32>::SIZE {
            if !leaf.is_value_on_at(pos) {
                continue;
            }
            let v = *leaf.get_value_at(pos);
            if v <= -gamma {
                leaf.set_value_off_at(pos, -gamma);
            } else if v >= gamma {
                leaf.set_value_off_at(pos, gamma);
            }
        }
    });
    level_set_prune(tree, gamma, -gamma, true);
}

/// `ttls_internal::normalizeLevelSet`.
fn normalize_level_set(tree: &mut Tree<f32>, dx: f32, norm_count: i32) {
    for _ in 0..norm_count {
        normalize_step(tree, dx);
    }
    trim_and_prune(tree);
}

/// One pass of a 3-tap box filter along `axis` over the active voxels (`Filter::Avg`).
fn box_pass(tree: &mut Tree<f32>, axis: usize) {
    let results: Vec<Vec<f32>> = {
        let t = &*tree;
        t.leaf_refs()
            .par_iter()
            .map(|leaf| {
                let acc = t.accessor();
                let mut out = leaf.buffer().to_vec();
                for (pos, _) in leaf.iter_on() {
                    let mut ijk = leaf.offset_to_global_coord(pos);
                    let mut sum = 0.0f32;
                    let c = ijk[axis];
                    for i in c - 1..=c + 1 {
                        ijk[axis] = i;
                        sum += acc.get_value(&ijk);
                    }
                    out[pos] = sum * (1.0 / 3.0);
                }
                out
            })
            .collect()
    };
    for (leaf, r) in tree.leaves_mut().zip(results) {
        leaf.buffer_mut().copy_from_slice(&r);
    }
}

/// `ttls_internal::smoothLevelSet`.
fn smooth_level_set(tree: &mut Tree<f32>, dx: f32, iterations: i32, half_width: i32) {
    let mut filtered = tree.clone();
    for _ in 0..iterations {
        // LevelSetFilter::mean(1): box filter (axes in OpenVDB's order) then track().
        box_pass(&mut filtered, 0);
        box_pass(&mut filtered, 2);
        box_pass(&mut filtered, 1);
        dilate_active_values(&mut filtered, 1, NearestNeighbors::Face, TilePolicy::IgnoreTiles, true);
        normalize_level_set(&mut filtered, dx, crate::tools::level_set_sphere::LEVEL_SET_HALF_WIDTH as i32);
    }
    let offset = -(0.5 * dx);
    let f = &filtered;
    tree.par_for_each_leaf_mut(|leaf: &mut LeafNode<f32>| {
        if let Some(r) = f.probe_leaf(&leaf.origin()) {
            let (buf, mask) = leaf.buffer_and_mask_mut();
            for pos in mask.iter_on() {
                buf[pos] = buf[pos].min(offset + *r.get_value_at(pos));
            }
        }
    });
    normalize_level_set(tree, dx, half_width);
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

    normalize_level_set(&mut ls, dx, 3 * half_width);
    if smoothing_steps > 0 {
        smooth_level_set(&mut ls, dx, smoothing_steps, half_width);
    }
    let mut out = Grid::from_tree(ls).with_transform(grid.transform().clone());
    out.set_grid_class(GridClass::LevelSet);
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
