//! Conversion between point arrays and point data grids (port of
//! `openvdb/points/PointConversion.h`).
//!
//! * [`create_point_data_grid`] partitions world-space positions into voxels (through a
//!   [`PointIndexGrid`]) and stores them, relative to their voxel centers, in a `"P"`
//!   attribute with the requested codec;
//! * [`populate_attribute`] and [`populate_string_attribute`] fill attributes from arrays
//!   indexed like the input positions;
//! * the `convert_point_data_grid_*` functions extract positions, attributes and groups
//!   back into flat arrays (in leaf order, using [`point_offsets`](super::point_offsets));
//! * [`compute_voxel_size`] estimates a voxel size giving a target number of points per
//!   voxel.

use super::attribute_array::{AttributeHandle, AttributeType, AttributeWriteHandle};
use super::attribute_group::GroupIndex;
use super::attribute_set::Descriptor;
use super::attribute_string::{StringAttributeHandle, StringAttributeWriteHandle, StringMetaInserter};
use super::codec::{AttributeValue, Codec};
use super::index_filter::IndexFilter;
use super::index_iterator::VoxelSelection;
use super::point_data_grid::{PointDataGrid, PointDataLeaf, PointDataTree, PointLeaf, PointLeafMut};
use super::{Error, Result};
use crate::math::{Mat4d, Transform, Vec3d, Vec3f};
use crate::metadata::MetaValue;
use crate::tools::point_index_grid::{create_point_index_grid, PointIndexGrid, PointIndexTree};
use crate::tools::point_partitioner::PointArray;
use crate::tools::points_to_mask::create_point_mask;
use rayon::prelude::*;
use std::sync::Arc;

/// Create a point data grid from world-space positions, storing them with `codec`
/// (`createPointDataGrid<Codec>(positions, xform)`).
pub fn create_point_data_grid<P: PointArray + ?Sized>(
    positions: &P,
    codec: Codec,
    transform: &Transform,
) -> Result<PointDataGrid> {
    let index = create_point_index_grid(positions, transform);
    create_point_data_grid_from_index(&index, positions, codec, transform, None)
}

/// Create a point data grid from positions already partitioned by a point index grid
/// (`createPointDataGrid<Codec>(pointIndexGrid, positions, xform, positionDefaultValue)`).
pub fn create_point_data_grid_from_index<P: PointArray + ?Sized>(
    index_grid: &PointIndexGrid,
    positions: &P,
    codec: Codec,
    transform: &Transform,
    position_default: Option<MetaValue>,
) -> Result<PointDataGrid> {
    let position_type = AttributeType::of::<Vec3f>(codec);
    if !position_type.is_registered() {
        return Err(Error::Type(format!("codec {} is not supported for positions", codec)));
    }
    let mut descriptor = Descriptor::with_position(position_type)?;
    if let Some(v) = position_default {
        descriptor.set_default_value("P", v)?;
    }
    let descriptor = Arc::new(descriptor);
    let index_tree = index_grid.tree();
    let mut tree = PointDataTree::from_topology(index_tree.tree().clone());
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let index_leaf = index_tree
                .probe_leaf(&leaf.origin())
                .ok_or_else(|| Error::Lookup("missing point index leaf".into()))?;
            leaf.initialize_attributes(descriptor.clone(), index_leaf.indices().len() as u32)?;
            let mut handle = AttributeWriteHandle::<Vec3f>::new(leaf.attribute_array_at_mut(0)?)?;
            let mut i = 0u32;
            for n in index_leaf.leaf().value_mask().iter_on() {
                let center = index_leaf.leaf().offset_to_global_coord(n).as_vec3d();
                for &p in index_leaf.get_indices_at(n).unwrap_or(&[]) {
                    let ps = transform.world_to_index(positions.get_pos(p as usize)) - center;
                    handle.set(i, Vec3f::new(ps[0] as f32, ps[1] as f32, ps[2] as f32));
                    i += 1;
                }
            }
            Ok(())
        })
        .collect();
    results.into_iter().collect::<Result<()>>()?;
    Ok(PointDataGrid::from_tree(tree, transform.clone()))
}

/// Fill attribute `name` from `data` (`stride` values per point, indexed like the positions
/// the point index tree was built from) (`populateAttribute()`).
pub fn populate_attribute<T: AttributeValue>(
    tree: &mut PointDataTree,
    index_tree: &PointIndexTree,
    name: &str,
    data: &[T],
    stride: u32,
) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    let pos =
        descriptor.position(name).ok_or_else(|| Error::Key(format!("attribute not found to populate - {}", name)))?;
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let Some(index_leaf) = index_tree.probe_leaf(&leaf.origin()) else { return Ok(()) };
            let mut h = AttributeWriteHandle::<T>::new(leaf.attribute_array_at_mut(pos)?)?;
            for (k, &i) in index_leaf.indices().iter().enumerate() {
                for m in 0..stride {
                    let v = *data
                        .get(i as usize * stride as usize + m as usize)
                        .ok_or_else(|| Error::Index("attribute data shorter than the point count".into()))?;
                    h.set_strided(k as u32, m, v);
                }
            }
            h.compact();
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Fill the string attribute `name` from `data`, inserting the strings into the descriptor
/// metadata first (`populateAttribute()` with `Name` values).
pub fn populate_string_attribute(
    tree: &mut PointDataTree,
    index_tree: &PointIndexTree,
    name: &str,
    data: &[impl AsRef<str> + Sync],
    stride: u32,
) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    let pos =
        descriptor.position(name).ok_or_else(|| Error::Key(format!("attribute not found to populate - {}", name)))?;
    let mut d = descriptor.clone();
    {
        let mut ins = StringMetaInserter::new(d.metadata_mut());
        for s in data {
            if !s.as_ref().is_empty() {
                ins.insert(s.as_ref(), 0);
            }
        }
    }
    let d = Arc::new(d);
    for mut leaf in tree.leaves_mut() {
        leaf.attribute_set_mut().reset_descriptor(d.clone(), true)?;
    }
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let Some(index_leaf) = index_tree.probe_leaf(&leaf.origin()) else { return Ok(()) };
            let (_, set) = leaf.parts_mut();
            let meta = set.descriptor().metadata().clone();
            let array = set.get_mut(pos).ok_or_else(|| Error::Lookup("missing string attribute".into()))?;
            let mut h = StringAttributeWriteHandle::new(array, &meta, true)?;
            for (k, &i) in index_leaf.indices().iter().enumerate() {
                for m in 0..stride {
                    let s = data
                        .get(i as usize * stride as usize + m as usize)
                        .ok_or_else(|| Error::Index("attribute data shorter than the point count".into()))?;
                    h.set_strided(k as u32, m, s.as_ref())?;
                }
            }
            h.compact();
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Split `out` into the per-leaf ranges given by cumulative `offsets` (times `stride`),
/// starting at `start`.
fn split_by_offsets<'a, T>(out: &'a mut [T], offsets: &[u64], start: u64, stride: usize) -> Vec<&'a mut [T]> {
    let mut chunks = Vec::with_capacity(offsets.len());
    let len = out.len();
    let mut rest = &mut out[(start as usize * stride).min(len)..];
    let mut prev = 0u64;
    for &o in offsets {
        let n = ((o - prev) as usize * stride).min(rest.len());
        let (a, b) = std::mem::take(&mut rest).split_at_mut(n);
        chunks.push(a);
        rest = b;
        prev = o;
    }
    chunks
}

/// Write the world-space positions of the points accepted by `filter` (in active voxels)
/// into `out`, leaf by leaf at the given cumulative `point_offsets` plus `start_offset`
/// (`convertPointDataGridPosition()`).
pub fn convert_point_data_grid_position<F: IndexFilter>(
    out: &mut [Vec3d],
    grid: &PointDataGrid,
    point_offsets: &[u64],
    start_offset: u64,
    filter: &F,
) -> Result<()> {
    let leaves = grid.tree().leaf_vec();
    let chunks = split_by_offsets(out, point_offsets, start_offset, 1);
    let transform = grid.transform();
    leaves.par_iter().zip(chunks).try_for_each(|(leaf, chunk)| -> Result<()> {
        let h = leaf.attribute_handle::<Vec3f>("P")?;
        for ((ijk, i), dst) in leaf.index_filtered(VoxelSelection::On, filter).with_coords().zip(chunk.iter_mut()) {
            let p = h.get(i);
            *dst = transform.index_to_world(Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64) + ijk.as_vec3d());
        }
        Ok(())
    })
}

/// Write the values of the attribute at `index` of the points accepted by `filter` into
/// `out` (`stride` values per point) (`convertPointDataGridAttribute()`).
pub fn convert_point_data_grid_attribute<T: AttributeValue, F: IndexFilter>(
    out: &mut [T],
    tree: &PointDataTree,
    point_offsets: &[u64],
    start_offset: u64,
    index: usize,
    stride: u32,
    filter: &F,
) -> Result<()> {
    let leaves = tree.leaf_vec();
    let chunks = split_by_offsets(out, point_offsets, start_offset, stride as usize);
    leaves.par_iter().zip(chunks).try_for_each(|(leaf, chunk)| -> Result<()> {
        let h = AttributeHandle::<T>::new(leaf.attribute_array_at(index)?)?;
        let mut k = 0usize;
        for i in leaf.index_filtered(VoxelSelection::On, filter) {
            for m in 0..stride {
                if let Some(dst) = chunk.get_mut(k) {
                    *dst = h.get_strided(i, m);
                }
                k += 1;
            }
        }
        Ok(())
    })
}

/// Write the strings of the string attribute at `index` into `out`.
pub fn convert_point_data_grid_string_attribute<F: IndexFilter>(
    out: &mut [String],
    tree: &PointDataTree,
    point_offsets: &[u64],
    start_offset: u64,
    index: usize,
    stride: u32,
    filter: &F,
) -> Result<()> {
    let leaves = tree.leaf_vec();
    let chunks = split_by_offsets(out, point_offsets, start_offset, stride as usize);
    leaves.par_iter().zip(chunks).try_for_each(|(leaf, chunk)| -> Result<()> {
        let h = StringAttributeHandle::new(leaf.attribute_array_at(index)?, leaf.attributes.descriptor().metadata())?;
        let mut k = 0usize;
        for i in leaf.index_filtered(VoxelSelection::On, filter) {
            for m in 0..stride {
                if let Some(dst) = chunk.get_mut(k) {
                    *dst = h.get_strided(i, m)?;
                }
                k += 1;
            }
        }
        Ok(())
    })
}

/// Write the membership of the points accepted by `filter` in a group into `out`
/// (`convertPointDataGridGroup()`).
pub fn convert_point_data_grid_group<F: IndexFilter>(
    out: &mut [bool],
    tree: &PointDataTree,
    point_offsets: &[u64],
    start_offset: u64,
    group: GroupIndex,
    filter: &F,
) -> Result<()> {
    let leaves = tree.leaf_vec();
    let chunks = split_by_offsets(out, point_offsets, start_offset, 1);
    leaves.par_iter().zip(chunks).try_for_each(|(leaf, chunk): (&PointDataLeaf<'_>, &mut [bool])| -> Result<()> {
        let h = leaf.group_handle(group)?;
        for (i, dst) in leaf.index_filtered(VoxelSelection::On, filter).zip(chunk.iter_mut()) {
            *dst = h.get(i);
        }
        Ok(())
    })
}

/// Estimate a voxel size such that the points occupy voxels with about `points_per_voxel`
/// points each (`computeVoxelSize()`). With a non-identity `transform`, its rotation and
/// shear are kept. The result is truncated to `decimal_places` decimal places.
pub fn compute_voxel_size<P: PointArray + ?Sized>(
    positions: &P,
    points_per_voxel: u32,
    transform: &Mat4d,
    decimal_places: u32,
) -> Result<f32> {
    if points_per_voxel == 0 {
        return Err(Error::Value("points per voxel cannot be zero".into()));
    }
    let mut voxel_size = 0.1f32;
    let num_points = positions.size();
    if num_points <= 1 {
        return Ok(voxel_size);
    }
    let target_voxel_count = (num_points / points_per_voxel as usize).max(1);

    // Bounds of the positions in the transform's (unit-scaled) frame.
    let inverse = transform.inverse(0.0).unwrap_or_else(Mat4d::identity);
    let inverse = unit_mat4(&inverse);
    let (min, max) = (0..num_points)
        .into_par_iter()
        .map(|n| {
            let p = inverse.transform(positions.get_pos(n));
            (p, p)
        })
        .reduce(|| (Vec3d::splat(f64::MAX), Vec3d::splat(-f64::MAX)), |a, b| (a.0.min_elem(&b.0), a.1.max_elem(&b.1)));
    if min == max {
        return Ok(voxel_size);
    }
    let ext = max - min;
    let mut volume = ext[0] * ext[1] * ext[2];
    if volume.abs() <= 1e-15 {
        let mut e = [ext[0], ext[1], ext[2]];
        e.sort_by(|a, b| b.partial_cmp(a).unwrap());
        volume = if e[1].abs() <= 1e-15 { e[0] * e[0] * e[0] } else { e[0] * e[1] * e[1] };
    }
    let voxel_size_from_volume = |volume: f64, count: usize, vs: &mut f32| -> bool {
        const MIN_VOXEL_VOLUME: f64 = 3e-15;
        let max_voxel_volume = f32::MAX as f64;
        let mut voxel_volume = volume / count as f64;
        let mut valid = true;
        if voxel_volume < MIN_VOXEL_VOLUME {
            voxel_volume = MIN_VOXEL_VOLUME;
            valid = false;
        } else if voxel_volume > max_voxel_volume {
            voxel_volume = max_voxel_volume;
            valid = false;
        }
        *vs = voxel_volume.powf(1.0 / 3.0) as f32;
        valid
    };
    let mut previous_volume = volume;
    if !voxel_size_from_volume(volume, target_voxel_count, &mut voxel_size) {
        return Ok(voxel_size);
    }
    let mut previous_voxel_count = 0u64;
    let mut voxel_count = 1u64;
    let is_identity = transform.eq_default(&Mat4d::identity());
    while voxel_count > previous_voxel_count {
        let new_transform = if !is_identity {
            let mut m = *transform;
            let scale = mat4_scale(&m);
            m.pre_scale(Vec3d::splat(voxel_size as f64).zip_map(scale, |a, b| a / b));
            Transform::from_matrix(m)
        } else {
            Transform::linear(voxel_size as f64)
        };
        let mask = create_point_mask(positions, &new_transform);
        previous_voxel_count = voxel_count;
        voxel_count = mask.active_voxel_count();
        volume = (voxel_size.powi(3) * voxel_count as f32) as f64;
        if volume >= previous_volume {
            break;
        }
        previous_volume = volume;
        let previous_voxel_size = voxel_size;
        if !voxel_size_from_volume(volume, target_voxel_count, &mut voxel_size) {
            voxel_size = previous_voxel_size;
            break;
        }
        if voxel_size / previous_voxel_size > 0.9 {
            break;
        }
    }
    // Truncate for readability, trying more decimal places if the result would be zero.
    let mut truncated = voxel_size;
    for i in decimal_places..11 {
        truncated = crate::math::math_util::truncate(voxel_size as f64, i) as f32;
        if truncated != 0.0 {
            break;
        }
    }
    Ok(truncated)
}

/// `math::unit(Mat4d)`: normalize the rows of the 3x3 part.
fn unit_mat4(m: &Mat4d) -> Mat4d {
    let mut r = *m;
    for i in 0..3 {
        let len = (m.0[i][0] * m.0[i][0] + m.0[i][1] * m.0[i][1] + m.0[i][2] * m.0[i][2]).sqrt();
        if len > 0.0 {
            for j in 0..3 {
                r.0[i][j] = m.0[i][j] / len;
            }
        }
    }
    r
}

/// `math::getScale(Mat4d)`: the lengths of the first three rows.
fn mat4_scale(m: &Mat4d) -> Vec3d {
    Vec3d::new(
        Vec3d::new(m.0[0][0], m.0[0][1], m.0[0][2]).length(),
        Vec3d::new(m.0[1][0], m.0[1][1], m.0[1][2]).length(),
        Vec3d::new(m.0[2][0], m.0[2][1], m.0[2][2]).length(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::{point_count, point_offsets, NullFilter};

    fn positions() -> Vec<Vec3f> {
        (0..200).map(|i| Vec3f::new((i % 7) as f32 * 0.3, (i % 11) as f32 * -0.2, (i / 13) as f32 * 0.17)).collect()
    }

    #[test]
    fn create_and_extract() {
        let pos = positions();
        let t = Transform::linear(0.25);
        for codec in [Codec::Null, Codec::FIXED_POINT_16, Codec::FIXED_POINT_8, Codec::Truncate] {
            let grid = create_point_data_grid(&pos, codec, &t).unwrap();
            assert_eq!(point_count(grid.tree()), pos.len() as u64);
            let (offsets, total) = point_offsets(grid.tree(), &NullFilter);
            let mut out = vec![Vec3d::zero(); total as usize];
            convert_point_data_grid_position(&mut out, &grid, &offsets, 0, &NullFilter).unwrap();
            let tol = match codec {
                Codec::FIXED_POINT_8 => 0.25 / 255.0 + 1e-6,
                Codec::Truncate => 2e-3,
                _ => 1e-5,
            };
            // Every input position has an output position within tolerance.
            for p in &pos {
                let p = Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64);
                assert!(out.iter().any(|q| (*q - p).abs().max_component() <= tol), "{:?} {}", p, codec);
            }
        }
    }

    #[test]
    fn populate_and_convert_attributes() {
        let pos = positions();
        let t = Transform::linear(0.5);
        let index = create_point_index_grid(&pos, &t);
        let mut grid = create_point_data_grid_from_index(&index, &pos, Codec::Null, &t, None).unwrap();
        let ids: Vec<i32> = (0..pos.len() as i32).collect();
        let names: Vec<String> = (0..pos.len()).map(|i| format!("n{}", i % 3)).collect();
        crate::points::append_attribute(
            grid.tree_mut(),
            "id",
            AttributeType::of::<i32>(Codec::Null),
            1,
            true,
            None,
            false,
            false,
        )
        .unwrap();
        crate::points::append_attribute(grid.tree_mut(), "name", AttributeType::string(), 1, true, None, false, false)
            .unwrap();
        populate_attribute(grid.tree_mut(), index.tree(), "id", &ids, 1).unwrap();
        populate_string_attribute(grid.tree_mut(), index.tree(), "name", &names, 1).unwrap();
        let (offsets, total) = point_offsets(grid.tree(), &NullFilter);
        let mut pos_out = vec![Vec3d::zero(); total as usize];
        convert_point_data_grid_position(&mut pos_out, &grid, &offsets, 0, &NullFilter).unwrap();
        let mut id_out = vec![0i32; total as usize];
        convert_point_data_grid_attribute(&mut id_out, grid.tree(), &offsets, 0, 1, 1, &NullFilter).unwrap();
        let mut name_out = vec![String::new(); total as usize];
        convert_point_data_grid_string_attribute(&mut name_out, grid.tree(), &offsets, 0, 2, 1, &NullFilter).unwrap();
        for k in 0..total as usize {
            let i = id_out[k] as usize;
            let p = pos[i];
            assert!((pos_out[k] - Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64)).length() < 1e-5);
            assert_eq!(name_out[k], names[i]);
        }
    }

    #[test]
    fn voxel_size() {
        let pos = positions();
        let vs = compute_voxel_size(&pos, 8, &Mat4d::identity(), 5).unwrap();
        assert!(vs > 0.0 && vs < 2.0, "{}", vs);
        assert_eq!(compute_voxel_size(&pos[..1], 8, &Mat4d::identity(), 5).unwrap(), 0.1);
        assert!(compute_voxel_size(&pos, 0, &Mat4d::identity(), 5).is_err());
    }
}
