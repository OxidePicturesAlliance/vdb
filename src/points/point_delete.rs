//! Deleting points (port of `openvdb/points/PointDelete.h`).

use super::attribute_set::AttributeSet;
use super::index_filter::{IndexFilter, MultiGroupFilter};
use super::index_iterator::VoxelSelection;
use super::point_data_grid::{PointDataTree, PointLeaf, PointLeafMut};
use super::point_group::drop_groups;
use super::{Error, Result};
use rayon::prelude::*;

/// Keep only the points accepted by `filter` in every leaf (`DeleteByFilterOp`); offsets
/// and active states are rebuilt from the remaining points. Leaves are not removed.
pub fn keep_points_by_filter<F: IndexFilter>(tree: &mut PointDataTree, filter: &F) -> Result<()> {
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let new_size = leaf.index_all_filtered(filter).count_remaining();
            if new_size == 0 {
                return leaf.clear_attributes(true);
            }
            if new_size == leaf.point_count() {
                return Ok(());
            }
            let mut mapping: Vec<(u32, u32)> = Vec::with_capacity(new_size as usize);
            let mut offsets = Vec::with_capacity(512);
            let mut target = 0u32;
            for n in 0..512 {
                for i in leaf.index_filtered(VoxelSelection::Voxel(n), filter) {
                    mapping.push((i, target));
                    target += 1;
                }
                offsets.push(target);
            }
            let existing = leaf.attribute_set().clone();
            let mut set = AttributeSet::like(&existing, new_size as u32)?;
            {
                let (_, mut new_arrays) = set.arrays_mut();
                for (dst, src) in new_arrays.iter_mut().zip(existing.arrays()) {
                    if !dst.has_constant_stride() || !src.has_constant_stride() {
                        return Err(Error::Value(
                            "transfer of attribute values for dynamic arrays not currently supported".into(),
                        ));
                    }
                    if dst.stride() != src.stride() {
                        return Err(Error::Lookup("cannot transfer attribute values with mismatching strides".into()));
                    }
                    dst.copy_values(src, mapping.iter().copied(), true)?;
                }
            }
            leaf.replace_attribute_set(set, false)?;
            leaf.set_offsets(&offsets, true)
        })
        .collect();
    results.into_iter().collect()
}

/// Delete the points in any of `groups` (or, with `invert`, the points in none of them);
/// leaves left without active voxels are removed, and unless `invert`, the groups are
/// dropped if `drop` (`deleteFromGroups()`).
pub fn delete_from_groups(tree: &mut PointDataTree, groups: &[&str], invert: bool, drop: bool) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    let available: Vec<&str> = groups.iter().copied().filter(|g| descriptor.has_group(g)).collect();
    if available.is_empty() {
        return Ok(());
    }
    let none: [&str; 0] = [];
    let filter = if invert {
        MultiGroupFilter::new(groups, &none, descriptor)
    } else {
        MultiGroupFilter::new(&none, groups, descriptor)
    };
    keep_points_by_filter(tree, &filter)?;
    tree.retain_leaves(|l| !l.leaf.is_empty());
    if drop && !invert {
        drop_groups(tree, &available)?;
    }
    Ok(())
}

/// Delete the points in `group` (`deleteFromGroup()`).
pub fn delete_from_group(tree: &mut PointDataTree, group: &str, invert: bool, drop: bool) -> Result<()> {
    delete_from_groups(tree, &[group], invert, drop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Transform, Vec3f};
    use crate::points::{
        append_group, append_typed_attribute, create_point_data_grid_from_index, group_point_count, point_count,
        populate_attribute, set_group, Codec,
    };
    use crate::tools::point_index_grid::create_point_index_grid;

    #[test]
    fn delete_groups() {
        let pos: Vec<Vec3f> = (0..300).map(|i| Vec3f::new((i % 30) as f32 * 0.9, (i / 30) as f32 * 3.1, 0.0)).collect();
        let t = Transform::linear(1.0);
        let index = create_point_index_grid(&pos, &t);
        let mut grid = create_point_data_grid_from_index(&index, &pos, Codec::Null, &t, None).unwrap();
        let tree = grid.tree_mut();
        let ids: Vec<i32> = (0..300).collect();
        append_typed_attribute(tree, "id", Codec::Null, 0i32, 1, None, false, false).unwrap();
        populate_attribute(tree, index.tree(), "id", &ids, 1).unwrap();
        append_group(tree, "low").unwrap();
        let low: Vec<i16> = (0..300).map(|i| (i < 100) as i16).collect();
        set_group(tree, index.tree(), &low, "low", false).unwrap();
        append_group(tree, "keep").unwrap();
        let keep: Vec<i16> = (0..300).map(|i| (i % 3 == 0) as i16).collect();
        set_group(tree, index.tree(), &keep, "keep", false).unwrap();

        delete_from_group(tree, "low", false, true).unwrap();
        assert_eq!(point_count(tree), 200);
        assert!(!tree.descriptor().unwrap().has_group("low"));
        for leaf in tree.leaves() {
            leaf.validate_offsets().unwrap();
            let id = leaf.attribute_handle::<i32>("id").unwrap();
            assert!(leaf.index_all().all(|i| id.get(i) >= 100));
        }
        // Keep only the "keep" group.
        delete_from_group(tree, "keep", true, true).unwrap();
        assert_eq!(point_count(tree), 66);
        assert_eq!(group_point_count(tree, "keep"), 66);
        assert!(tree.leaves().all(|l| !l.leaf.is_empty()));
    }
}
