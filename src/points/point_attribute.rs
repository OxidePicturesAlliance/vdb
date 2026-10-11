//! Adding, removing and renaming attributes of a point data tree (port of
//! `openvdb/points/PointAttribute.h`).
//!
//! All leaves share one descriptor; these functions build the new descriptor once and
//! apply the change to every leaf in parallel.

use super::attribute_array::{AttributeType, AttributeWriteHandle};
use super::attribute_string::{StringAttributeWriteHandle, StringMetaInserter};
use super::codec::{AttributeValue, Codec};
use super::point_data_grid::{PointDataTree, PointLeafMut};
use super::{Error, Result};
use crate::metadata::MetaValue;
use rayon::prelude::*;
use std::sync::Arc;

/// Append an attribute of the given type to every leaf (`appendAttribute(tree, name, type,
/// ...)`). The new arrays are uniform, set to `default` (stored as the attribute's default
/// value) or zero.
#[allow(clippy::too_many_arguments)]
pub fn append_attribute(
    tree: &mut PointDataTree,
    name: &str,
    ty: AttributeType,
    stride_or_total: u32,
    constant_stride: bool,
    default: Option<MetaValue>,
    hidden: bool,
    transient: bool,
) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    if descriptor.position(name).is_some() {
        return Err(Error::Key(format!("cannot append an attribute with a non-unique name - {}", name)));
    }
    let mut new = descriptor.duplicate_append(name, ty)?;
    if let Some(v) = &default {
        new.set_default_value(name, v.clone())?;
    }
    let pos = new.find(name);
    let new = Arc::new(new);
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let set = leaf.attribute_set_mut();
            let expected = set.descriptor().clone();
            set.append_attribute_with(&expected, &new, pos, stride_or_total, constant_stride, default.as_ref())?;
            let a = set.get_mut(pos).unwrap();
            a.set_hidden(hidden);
            a.set_transient(transient);
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Append an attribute of `T` with `codec` whose values are all `uniform`
/// (`appendAttribute<T, Codec>(tree, name, uniformValue, ...)`).
#[allow(clippy::too_many_arguments)]
pub fn append_typed_attribute<T: AttributeValue>(
    tree: &mut PointDataTree,
    name: &str,
    codec: Codec,
    uniform: T,
    stride: u32,
    default: Option<T>,
    hidden: bool,
    transient: bool,
) -> Result<()> {
    let default_meta = match default {
        Some(d) => Some(d.to_meta().ok_or_else(|| Error::Type(format!("{} has no metadata type", T::TYPE_NAME)))?),
        None => None,
    };
    append_attribute(tree, name, AttributeType::of::<T>(codec), stride, true, default_meta, hidden, transient)?;
    if uniform != default.unwrap_or_default() {
        collapse_attribute(tree, name, uniform)?;
    }
    Ok(())
}

/// Append a string attribute whose values are all `uniform`.
pub fn append_string_attribute(tree: &mut PointDataTree, name: &str, uniform: &str, stride: u32) -> Result<()> {
    append_attribute(tree, name, AttributeType::string(), stride, true, None, false, false)?;
    if !uniform.is_empty() {
        collapse_string_attribute(tree, name, uniform)?;
    }
    Ok(())
}

/// Set every value of attribute `name` to `uniform` (`collapseAttribute()`).
pub fn collapse_attribute<T: AttributeValue>(tree: &mut PointDataTree, name: &str, uniform: T) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    let pos =
        descriptor.position(name).ok_or_else(|| Error::Key("cannot find attribute name in point data tree".into()))?;
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let mut h = AttributeWriteHandle::<T>::with_expand(leaf.attribute_array_at_mut(pos)?, false)?;
            h.collapse_to(uniform);
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Set every value of the string attribute `name` to `uniform`, inserting the string into
/// the descriptor metadata.
pub fn collapse_string_attribute(tree: &mut PointDataTree, name: &str, uniform: &str) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    let pos =
        descriptor.position(name).ok_or_else(|| Error::Key("cannot find attribute name in point data tree".into()))?;
    let mut d = descriptor.clone();
    if !uniform.is_empty() {
        StringMetaInserter::new(d.metadata_mut()).insert(uniform, 0);
    }
    let d = Arc::new(d);
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let set = leaf.attribute_set_mut();
            set.reset_descriptor(d.clone(), true)?;
            let array = set.get_mut(pos).unwrap();
            StringAttributeWriteHandle::new(array, d.metadata(), false)?.collapse(uniform)
        })
        .collect();
    results.into_iter().collect()
}

/// Drop the attributes at `indices` (`dropAttributes(tree, indices)`); the position
/// attribute cannot be dropped.
pub fn drop_attributes(tree: &mut PointDataTree, indices: &[usize]) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    if let Some(p) = descriptor.position("P") {
        if indices.contains(&p) {
            return Err(Error::Key("cannot drop mandatory position attribute".into()));
        }
    }
    let new = Arc::new(descriptor.duplicate_drop(indices)?);
    let results: Vec<Result<()>> = tree
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let set = leaf.attribute_set_mut();
            let expected = set.descriptor().clone();
            set.drop_attributes_with(indices, &expected, &new)
        })
        .collect();
    results.into_iter().collect()
}

/// Drop the attributes with the given names (`dropAttributes(tree, names)`).
pub fn drop_attributes_by_name(tree: &mut PointDataTree, names: &[&str]) -> Result<()> {
    let Some(descriptor) = tree.descriptor() else { return Ok(()) };
    let mut indices = Vec::new();
    for n in names {
        indices.push(
            descriptor
                .position(n)
                .ok_or_else(|| Error::Key(format!("cannot drop an attribute that does not exist - {}", n)))?,
        );
    }
    drop_attributes(tree, &indices)
}

/// Drop one attribute by name (`dropAttribute()`).
pub fn drop_attribute(tree: &mut PointDataTree, name: &str) -> Result<()> {
    drop_attributes_by_name(tree, &[name])
}

/// Rename attributes (`renameAttributes()`); group arrays cannot be renamed.
pub fn rename_attributes(tree: &mut PointDataTree, old_names: &[&str], new_names: &[&str]) -> Result<()> {
    if old_names.len() != new_names.len() {
        return Err(Error::Value("mismatching sizes of name vectors, cannot rename attributes".into()));
    }
    let Some(first) = tree.leaves().next() else { return Ok(()) };
    let descriptor = first.attributes.descriptor().clone();
    let mut new = descriptor.clone();
    for (old, newn) in old_names.iter().zip(new_names) {
        let pos =
            descriptor.position(old).ok_or_else(|| Error::Key(format!("cannot find requested attribute - {}", old)))?;
        if descriptor.position(newn).is_some() {
            return Err(Error::Key(format!("cannot rename attribute as new name already exists - {}", newn)));
        }
        if first.attributes.get(pos).is_some_and(|a| a.is_group()) {
            return Err(Error::Key(format!("cannot rename group attribute - {}", old)));
        }
        new.rename(old, newn)?;
    }
    let new = Arc::new(new);
    for mut leaf in tree.leaves_mut() {
        let set = leaf.attribute_set_mut();
        let expected = set.descriptor().clone();
        set.rename_attributes(&expected, &new)?;
    }
    Ok(())
}

/// Rename one attribute (`renameAttribute()`).
pub fn rename_attribute(tree: &mut PointDataTree, old_name: &str, new_name: &str) -> Result<()> {
    rename_attributes(tree, &[old_name], &[new_name])
}

/// Collapse every attribute array whose values are all equal (`compactAttributes()`).
pub fn compact_attributes(tree: &mut PointDataTree) {
    tree.par_for_each_leaf_mut(|leaf| leaf.compact_attributes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Transform, Vec3f};
    use crate::points::{create_point_data_grid, PointLeaf};

    #[test]
    fn append_drop_rename() {
        let pos: Vec<Vec3f> = (0..50).map(|i| Vec3f::new(i as f32 * 0.37, 0.0, (i % 5) as f32)).collect();
        let mut grid = create_point_data_grid(&pos, Codec::Null, &Transform::linear(1.0)).unwrap();
        let tree = grid.tree_mut();
        append_typed_attribute(tree, "density", Codec::Null, 2.0f32, 1, Some(1.0), false, false).unwrap();
        assert!(append_typed_attribute(tree, "density", Codec::Null, 2.0f32, 1, None, false, false).is_err());
        for leaf in tree.leaves() {
            let a = leaf.attribute_array("density").unwrap();
            assert!(a.is_uniform());
            assert_eq!(a.get::<f32>(0), 2.0);
        }
        assert_eq!(tree.descriptor().unwrap().default_value("density"), Some(&MetaValue::Float(1.0)));
        append_string_attribute(tree, "name", "hello", 1).unwrap();
        let leaf = tree.leaves().next().unwrap();
        let h = crate::points::StringAttributeHandle::new(
            leaf.attribute_array("name").unwrap(),
            leaf.attributes.descriptor().metadata(),
        )
        .unwrap();
        assert_eq!(h.get(0).unwrap(), "hello");
        rename_attribute(tree, "density", "rho").unwrap();
        assert!(tree.descriptor().unwrap().position("rho").is_some());
        assert!(tree.descriptor().unwrap().has_default_value("rho"));
        assert!(drop_attribute(tree, "P").is_err());
        drop_attribute(tree, "rho").unwrap();
        assert_eq!(tree.descriptor().unwrap().names(), vec!["P", "name"]);
        assert!(tree.leaves().all(|l| l.attributes.size() == 2));
        compact_attributes(tree);
    }
}
