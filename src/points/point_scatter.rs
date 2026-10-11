//! Scattering points into point data grids (port of `openvdb/points/PointScatter.h`).
//!
//! The scatterers fill the active voxels of a grid with random points, stored in a
//! [`PointDataGrid`] with the grid's transform. They use `std::mt19937` and libstdc++'s
//! distributions bit for bit (see [`crate::tools::point_scatter`]), so for the same seed they
//! generate the same points as OpenVDB built with GCC.

use super::attribute_array::{AttributeType, AttributeWriteHandle};
use super::attribute_set::Descriptor;
use super::codec::Codec;
use super::point_data_grid::{PointDataGrid, PointDataTree, PointLeafMut};
use super::{Error, Result};
use crate::grid::Grid;
use crate::math::Vec3f;
use crate::tools::point_scatter::{uniform_int, Mt19937, RandomGenerator};
use crate::types::Value;
use std::sync::Arc;

/// `math::Rand01<float, std::mt19937>`: libstdc++'s `generate_canonical<float, 24>`.
struct Rand01f {
    g: Mt19937,
}

impl Rand01f {
    fn new(seed: u32) -> Self {
        Rand01f { g: Mt19937::new(seed) }
    }
    fn next(&mut self) -> f32 {
        let sum = self.g.next_u32() as f32;
        let r = sum / 4294967296.0f32;
        if r >= 1.0 {
            1.0f32 - f32::EPSILON / 2.0
        } else {
            r
        }
    }
}

/// A point data tree with the active topology of `grid` (tiles voxelized) and zero offsets
/// (`initialisePointTopology()`).
fn initialise_topology<T: Value>(grid: &Grid<T>) -> PointDataGrid {
    let mut tree = crate::tree::Tree::<u32>::new(0);
    tree.topology_union(grid.tree());
    tree.voxelize_active_tiles();
    PointDataGrid::from_tree(PointDataTree::from_topology(tree), grid.transform().clone())
}

fn position_descriptor(codec: Codec) -> Result<Arc<Descriptor>> {
    let ty = AttributeType::of::<Vec3f>(codec);
    if !ty.is_registered() {
        return Err(Error::Type(format!("codec {} is not supported for positions", codec)));
    }
    Ok(Arc::new(Descriptor::with_position(ty)?))
}

/// Initialize a leaf's attributes for `count` points with random positions
/// (`generatePositions()`).
fn generate_positions(
    leaf: &mut impl PointLeafMut,
    descriptor: &Arc<Descriptor>,
    count: u32,
    spread: f32,
    rand01: &mut Rand01f,
) -> Result<()> {
    leaf.initialize_attributes(descriptor.clone(), count)?;
    let array = leaf.attribute_array_at_mut(0)?;
    array.expand(false);
    let mut h = AttributeWriteHandle::<Vec3f>::with_expand(array, false)?;
    for i in 0..count {
        let x = spread * (rand01.next() - 0.5);
        let y = spread * (rand01.next() - 0.5);
        let z = spread * (rand01.next() - 0.5);
        h.set(i, Vec3f::new(x, y, z));
    }
    Ok(())
}

fn check_spread(spread: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&spread) {
        return Err(Error::Value("spread must be between 0 and 1".into()));
    }
    Ok(())
}

/// Scatter `count` points uniformly over the active voxels of `grid`
/// (`uniformPointScatter()`). Positions use `codec`; `spread` (0..1) scales the jitter
/// around voxel centers.
pub fn uniform_point_scatter<T: Value>(
    grid: &Grid<T>,
    count: u64,
    seed: u32,
    spread: f32,
    codec: Codec,
) -> Result<PointDataGrid> {
    check_spread(spread)?;
    let mut points = initialise_topology(grid);
    if points.tree().leaf_count() == 0 {
        return Ok(points);
    }
    let voxel_count = points.tree().offsets().active_leaf_voxel_count();
    let points_per_volume = count as f64 / voxel_count as f64;
    let points_per_voxel = points_per_volume.floor() as u32;
    let remainder = count - points_per_voxel as u64 * voxel_count;
    if remainder == 0 {
        return dense_uniform_point_scatter(grid, points_per_voxel as f32, seed, spread, codec);
    }
    // Random voxel ids for the remaining points.
    let mut values: Vec<u64> = {
        let mut g = Mt19937::new(seed);
        (0..remainder).map(|_| uniform_int(&mut g, 0, voxel_count - 1)).collect()
    };
    values.sort_unstable();
    let fractional_only = points_per_voxel == 0;
    {
        let mut leaves = points.tree_mut().leaves_mut();
        let mut lower_offset = 0u64;
        let mut lo = 0usize;
        for leaf in leaves.iter_mut() {
            let n_on = leaf.leaf.on_voxel_count();
            let upper_offset = lower_offset + n_on;
            let on: Vec<usize> = leaf.leaf.value_mask().iter_on().collect();
            let mut added = !fractional_only;
            while lo < values.len() && values[lo] < lower_offset {
                lo += 1;
            }
            while lo < values.len() && values[lo] < upper_offset {
                let pos = on[(values[lo] - lower_offset) as usize];
                let b = leaf.leaf.buffer_mut();
                b[pos] += 1;
                added = true;
                lo += 1;
            }
            if !added {
                leaf.leaf.value_mask_mut().set_all_off();
            }
            lower_offset = upper_offset;
        }
    }
    if fractional_only {
        points.tree_mut().retain_leaves(|l| !l.leaf.is_empty());
    }
    let descriptor = position_descriptor(codec)?;
    let mut rand01 = Rand01f::new(seed);
    for mut leaf in points.tree_mut().leaves_mut() {
        let mut offset = 0u32;
        for n in 0..512 {
            if leaf.leaf.is_value_on_at(n) {
                let value = (points_per_volume + leaf.leaf.buffer()[n] as f64) as u32;
                if value == 0 {
                    leaf.leaf.set_active_state_at(n, false);
                } else {
                    offset += value;
                }
            }
            leaf.set_offset_only(n, offset);
        }
        generate_positions(&mut leaf, &descriptor, offset, spread, &mut rand01)?;
    }
    Ok(points)
}

/// Scatter `points_per_voxel` points in each active voxel of `grid`; a fractional part is
/// realized randomly (`denseUniformPointScatter()`).
pub fn dense_uniform_point_scatter<T: Value>(
    grid: &Grid<T>,
    points_per_voxel: f32,
    seed: u32,
    spread: f32,
    codec: Codec,
) -> Result<PointDataGrid> {
    if points_per_voxel < 0.0 {
        return Err(Error::Value("points per voxel must not be less than zero".into()));
    }
    check_spread(spread)?;
    let mut points = initialise_topology(grid);
    if points.tree().leaf_count() == 0 {
        return Ok(points);
    }
    let ppv_int = points_per_voxel.floor() as u32;
    let delta = (points_per_voxel - ppv_int as f32) as f64;
    let fractional = delta.abs() > 1.0e-6;
    let fractional_only = ppv_int == 0;
    let descriptor = position_descriptor(codec)?;
    let mut rand01 = Rand01f::new(seed);
    for mut leaf in points.tree_mut().leaves_mut() {
        let mut offset = 0u32;
        for n in 0..512 {
            if leaf.leaf.is_value_on_at(n) {
                offset += ppv_int;
                if fractional && (rand01.next() as f64) < delta {
                    offset += 1;
                } else if fractional_only {
                    leaf.leaf.set_active_state_at(n, false);
                }
            }
            leaf.set_offset_only(n, offset);
        }
        if offset != 0 {
            generate_positions(&mut leaf, &descriptor, offset, spread, &mut rand01)?;
        }
    }
    if fractional_only {
        points.tree_mut().retain_leaves(|l| !l.leaf.is_empty());
    }
    Ok(points)
}

/// Scatter points with a density of `points_per_voxel` times the voxel value in each active
/// voxel of a scalar grid (`nonUniformPointScatter()`).
pub fn non_uniform_point_scatter<T: Value + num_traits::ToPrimitive>(
    grid: &Grid<T>,
    points_per_voxel: f32,
    seed: u32,
    spread: f32,
    codec: Codec,
) -> Result<PointDataGrid> {
    if points_per_voxel < 0.0 {
        return Err(Error::Value("points per voxel must not be less than zero".into()));
    }
    check_spread(spread)?;
    let mut points = initialise_topology(grid);
    if points.tree().leaf_count() == 0 {
        return Ok(points);
    }
    let descriptor = position_descriptor(codec)?;
    let mut rand01 = Rand01f::new(seed);
    let acc = grid.accessor();
    for mut leaf in points.tree_mut().leaves_mut() {
        let mut offset = 0u32;
        for n in 0..512 {
            if leaf.leaf.is_value_on_at(n) {
                let ijk = leaf.leaf.offset_to_global_coord(n);
                let v = acc.get_value(&ijk).to_f64().unwrap_or(0.0);
                let f = (v * points_per_voxel as f64).max(0.0);
                let mut count = f as i32;
                if (rand01.next() as f64) < f - count as f64 {
                    count += 1;
                } else if count == 0 {
                    leaf.leaf.set_active_state_at(n, false);
                }
                offset += count as u32;
            }
            leaf.set_offset_only(n, offset);
        }
        if offset != 0 {
            generate_positions(&mut leaf, &descriptor, offset, spread, &mut rand01)?;
        }
    }
    points.tree_mut().retain_leaves(|l| !l.leaf.is_empty());
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::CoordBBox;
    use crate::points::{point_count, PointLeaf};

    fn mask() -> Grid<f32> {
        let mut g = Grid::<f32>::new(0.0);
        g.tree_mut().fill(&CoordBBox::from_coords(0, 0, 0, 9, 9, 9), 0.5, true);
        g
    }

    #[test]
    fn scatter_counts() {
        let g = mask();
        let p = uniform_point_scatter(&g, 2500, 3, 1.0, Codec::Null).unwrap();
        assert_eq!(point_count(p.tree()), 2500);
        let p = uniform_point_scatter(&g, 300, 3, 0.5, Codec::FIXED_POINT_16).unwrap();
        assert_eq!(point_count(p.tree()), 300);
        for leaf in p.tree().leaves() {
            leaf.validate_offsets().unwrap();
            let h = leaf.attribute_handle::<Vec3f>("P").unwrap();
            assert!(leaf.index_all().all(|i| h.get(i).abs().max_component() <= 0.25 + 1e-4));
        }
        let p = dense_uniform_point_scatter(&g, 2.0, 1, 1.0, Codec::Null).unwrap();
        assert_eq!(point_count(p.tree()), 2000);
        let p = dense_uniform_point_scatter(&g, 0.5, 1, 1.0, Codec::Null).unwrap();
        let n = point_count(p.tree());
        assert!(n > 350 && n < 650, "{}", n);
        let p = non_uniform_point_scatter(&g, 4.0, 1, 1.0, Codec::Null).unwrap();
        assert_eq!(point_count(p.tree()), 2000);
        assert!(uniform_point_scatter(&g, 10, 0, 2.0, Codec::Null).is_err());
    }
}
