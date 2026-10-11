//! Point group management (port of `openvdb/points/PointGroup.h`).
//!
//! Groups are bits of `uint8` group arrays (named `"__group"`, `"__group0"`, ...). Appending
//! a group reuses an unused bit or adds a new group array; dropping groups compacts the
//! remaining groups into as few arrays as possible.

use super::attribute_array::AttributeType;
use super::attribute_set::Descriptor;
use super::index_filter::{IndexFilter, RandomLeafFilter};
use super::index_iterator::VoxelSelection;
use super::point_attribute::drop_attributes;
use super::point_count::point_count;
use super::point_data_grid::{PointDataTree, PointLeaf, PointLeafMut};
use super::{Error, Result};
use crate::tools::point_index_grid::PointIndexTree;
use rayon::prelude::*;
use std::sync::Arc;

/// Remove from `groups` the names that are not groups of `descriptor`
/// (`deleteMissingPointGroups()`).
pub fn delete_missing_point_groups(groups: &mut Vec<String>, descriptor: &Descriptor) {
    groups.retain(|g| descriptor.has_group(g));
}

/// Give every leaf the descriptor `d`.
fn reset_descriptors(tree: &mut PointDataTree, d: &Arc<Descriptor>) {
    for mut leaf in tree.leaves_mut() {
        let _ = leaf.attribute_set_mut().reset_descriptor(d.clone(), true);
    }
}

/// Add group `name` (no-op if it exists) (`appendGroup()`). New members must be set with
/// [`set_group`] or similar.
pub fn append_group(tree: &mut PointDataTree, name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::Key("cannot use an empty group name as a key".into()));
    }
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    if descriptor.has_group(name) {
        return Ok(());
    }
    let has_unused = descriptor.unused_groups() > 0;
    let mut d = descriptor.clone();
    if !has_unused {
        let group_name = d.unique_name("__group");
        let new = Arc::new(d.duplicate_append(&group_name, AttributeType::group())?);
        let pos = new.find(&group_name);
        let results: Vec<Result<()>> = tree
            .leaves_mut()
            .into_par_iter()
            .map(|mut leaf| {
                let set = leaf.attribute_set_mut();
                let expected = set.descriptor().clone();
                set.append_attribute_with(&expected, &new, pos, 1, true, None)
            })
            .collect();
        results.into_iter().collect::<Result<()>>()?;
        d = (*new).clone();
    }
    let offset = d.unused_group_offset(None).ok_or_else(|| Error::Value("no group offsets available".into()))?;
    d.set_group(name, offset, false)?;
    reset_descriptors(tree, &Arc::new(d));
    if has_unused {
        set_group_all(tree, name, false)?;
    }
    Ok(())
}

/// Add several groups (`appendGroups()`).
pub fn append_groups(tree: &mut PointDataTree, names: &[&str]) -> Result<()> {
    for n in names {
        append_group(tree, n)?;
    }
    Ok(())
}

/// Remove group `name`, compacting the group arrays if `compact` (`dropGroup()`).
pub fn drop_group(tree: &mut PointDataTree, name: &str, compact: bool) -> Result<()> {
    if name.is_empty() {
        return Err(Error::Key("cannot use an empty group name as a key".into()));
    }
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    let mut d = descriptor.clone();
    d.drop_group(name);
    reset_descriptors(tree, &Arc::new(d));
    if compact {
        compact_groups(tree)?;
    }
    Ok(())
}

/// Remove several groups, then compact (`dropGroups(tree, groups)`).
pub fn drop_groups(tree: &mut PointDataTree, names: &[&str]) -> Result<()> {
    for n in names {
        drop_group(tree, n, false)?;
    }
    compact_groups(tree)
}

/// Remove all groups and group arrays (`dropGroups(tree)`).
pub fn drop_all_groups(tree: &mut PointDataTree) -> Result<()> {
    let Some(first) = tree.leaves().next() else { return Ok(()) };
    let indices = first.attributes.group_attribute_indices();
    let mut d = first.attributes.descriptor().clone();
    d.clear_groups();
    reset_descriptors(tree, &Arc::new(d));
    drop_attributes(tree, &indices)
}

/// Move groups into the lowest bits and drop group arrays that become unused
/// (`compactGroups()`).
pub fn compact_groups(tree: &mut PointDataTree) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    if !descriptor.can_compact_groups() {
        return Ok(());
    }
    let mut d = descriptor.clone();
    while let Some((name, source, target)) = d.requires_group_move() {
        let si = d.group_index_of_offset(source)?;
        let ti = d.group_index_of_offset(target)?;
        tree.par_for_each_leaf_mut(|leaf| {
            let n = leaf.point_count() as u32;
            let values: Vec<bool> = match leaf.group_handle(si) {
                Ok(h) => (0..n).map(|i| h.get(i)).collect(),
                Err(_) => return,
            };
            if let Ok(mut w) = leaf.group_write_handle(ti) {
                for (i, v) in values.into_iter().enumerate() {
                    w.set(i as u32, v);
                }
            }
        });
        d.set_group(&name, target, false)?;
    }
    let d = Arc::new(d);
    reset_descriptors(tree, &d);
    let first = tree.leaves().next().unwrap();
    let indices = first.attributes.group_attribute_indices();
    let to_drop = d.unused_groups() / super::GROUP_BITS;
    let drop: Vec<usize> = indices[indices.len().saturating_sub(to_drop)..].to_vec();
    drop_attributes(tree, &drop)
}

/// Set group membership from a per-point array indexed like the positions the point index
/// tree was built from (`setGroup(tree, indexTree, membership, group, remove)`). Unless
/// `remove`, only members are set; with `remove`, non-members are also cleared.
pub fn set_group(
    tree: &mut PointDataTree,
    index_tree: &PointIndexTree,
    membership: &[i16],
    name: &str,
    remove: bool,
) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    if !descriptor.has_group(name) {
        return Err(Error::Lookup("group must exist on tree before defining membership".into()));
    }
    let max = index_tree.leaves().filter_map(|l| l.indices().iter().max().copied()).max();
    if let Some(m) = max {
        if membership.len() <= m as usize {
            return Err(Error::Index(
                "group membership vector size must be larger than the maximum index within the index tree".into(),
            ));
        }
    }
    let index = descriptor.group_index(name)?;
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let Some(il) = index_tree.probe_leaf(&leaf.origin()) else { return Ok(()) };
            let mut g = leaf.group_write_handle(index)?;
            for (k, &i) in il.indices().iter().enumerate() {
                let m = membership[i as usize];
                if remove {
                    g.set(k as u32, m != 0);
                } else if m == 1 {
                    g.set(k as u32, true);
                }
            }
            g.compact();
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Add every point to (or remove every point from) group `name` (`setGroup(tree, group,
/// member)`).
pub fn set_group_all(tree: &mut PointDataTree, name: &str, member: bool) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    if !descriptor.has_group(name) {
        return Err(Error::Lookup("group must exist on tree before defining membership".into()));
    }
    let index = descriptor.group_index(name)?;
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            leaf.group_write_handle(index)?.collapse(member);
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Add the points accepted by `filter` to group `name` (`setGroupByFilter()`).
pub fn set_group_by_filter<F: IndexFilter>(tree: &mut PointDataTree, name: &str, filter: &F) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    if !descriptor.has_group(name) {
        return Err(Error::Lookup("group must exist on tree before defining membership".into()));
    }
    let index = descriptor.group_index(name)?;
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let members: Vec<u32> = leaf.index_filtered(VoxelSelection::All, filter).collect();
            let mut g = leaf.group_write_handle(index)?;
            for i in members {
                g.set(i, true);
            }
            g.compact();
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Add about `target_points` random points to group `name` (`setGroupByRandomTarget()`).
pub fn set_group_by_random_target(tree: &mut PointDataTree, name: &str, target_points: u64, seed: u32) -> Result<()> {
    let filter = RandomLeafFilter::new(tree, target_points, seed);
    set_group_by_filter(tree, name, &filter)
}

/// Add about `percentage` percent of the points, chosen randomly, to group `name`
/// (`setGroupByRandomPercentage()`).
pub fn set_group_by_random_percentage(tree: &mut PointDataTree, name: &str, percentage: f32, seed: u32) -> Result<()> {
    let current = point_count(tree) as i32;
    let target = ((percentage * current as f32) / 100.0).round() as i32;
    let filter = RandomLeafFilter::new(tree, target.max(0) as u64, seed);
    set_group_by_filter(tree, name, &filter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Transform, Vec3f};
    use crate::points::{create_point_data_grid_from_index, group_point_count, Codec};
    use crate::tools::point_index_grid::create_point_index_grid;

    #[test]
    fn groups_lifecycle() {
        let pos: Vec<Vec3f> = (0..100).map(|i| Vec3f::new(i as f32 * 0.21, (i % 9) as f32, 0.0)).collect();
        let t = Transform::linear(1.0);
        let index = create_point_index_grid(&pos, &t);
        let mut grid = create_point_data_grid_from_index(&index, &pos, Codec::Null, &t, None).unwrap();
        let tree = grid.tree_mut();
        for k in 0..10 {
            append_group(tree, &format!("g{}", k)).unwrap();
        }
        assert_eq!(tree.descriptor().unwrap().count(&AttributeType::group()), 2);
        let even: Vec<i16> = (0..100).map(|i| (i % 2 == 0) as i16).collect();
        set_group(tree, index.tree(), &even, "g9", false).unwrap();
        assert_eq!(group_point_count(tree, "g9"), 50);
        set_group_all(tree, "g3", true).unwrap();
        assert_eq!(group_point_count(tree, "g3"), 100);
        drop_groups(tree, &["g0", "g1", "g2", "g4"]).unwrap();
        // 6 groups left fit in one array; g9 was moved into the first array.
        assert_eq!(tree.descriptor().unwrap().count(&AttributeType::group()), 1);
        assert_eq!(group_point_count(tree, "g9"), 50);
        assert_eq!(group_point_count(tree, "g3"), 100);
        assert_eq!(group_point_count(tree, "g5"), 0);
        append_group(tree, "r").unwrap();
        set_group_by_random_target(tree, "r", 30, 1).unwrap();
        assert_eq!(group_point_count(tree, "r"), 30);
        drop_all_groups(tree).unwrap();
        assert_eq!(tree.descriptor().unwrap().size(), 1);
        assert!(tree.descriptor().unwrap().group_map().is_empty());
    }
}
