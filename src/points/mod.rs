//! Point data grids: points with arbitrary attributes stored in a VDB tree (port of
//! `openvdb/points`).
//!
//! * [`AttributeArray`] — per-point values of one type with an optional compression
//!   [`Codec`] (null, truncate to half, 8/16-bit fixed point, quantized unit vectors), plus
//!   typed [`AttributeHandle`]/[`AttributeWriteHandle`] access, group arrays
//!   ([`GroupHandle`]) and string arrays ([`StringAttributeHandle`]).
//! * [`AttributeSet`] and [`Descriptor`] — the attributes of a leaf, their names, types,
//!   groups and metadata.
//! * [`PointDataTree`]/[`PointDataGrid`] — a tree whose leaf voxel values are end offsets into
//!   per-leaf attribute arrays (OpenVDB's `Tree_ptdataidx32_5_4_3`), readable and writable
//!   in `.vdb` files compatible with OpenVDB (see [`crate::io`]).
//! * Index iteration with filters ([`IndexIter`], [`IndexFilter`]).
//!
//! ```
//! use vdb::math::{Transform, Vec3f};
//! use vdb::points::{self, Codec, PointLeaf};
//!
//! let positions = vec![Vec3f::new(0.0, 1.0, 0.0), Vec3f::new(1.5, 3.5, 1.0), Vec3f::new(-1.0, 6.0, -2.0)];
//! let transform = Transform::linear(0.5);
//! let grid = points::create_point_data_grid(&positions, Codec::FIXED_POINT_16, &transform).unwrap();
//! assert_eq!(points::point_count(grid.tree()), 3);
//! let ws = grid.world_positions().unwrap();
//! assert!(ws.iter().any(|p| (p[1] - 3.5).abs() < 1e-3));
//! ```

pub mod attribute_array;
pub mod attribute_group;
pub mod attribute_set;
pub mod attribute_string;
pub mod codec;
pub mod index_filter;
pub mod index_iterator;
pub mod point_advect;
pub mod point_attribute;
pub mod point_conversion;
pub mod point_count;
pub mod point_data_grid;
pub mod point_delete;
pub mod point_group;
pub mod point_index;
pub mod point_mask;
pub mod point_move;
pub mod point_rasterize_sdf;
pub mod point_rasterize_trilinear;
pub mod point_sample;
pub mod point_scatter;
pub mod point_statistics;

pub use attribute_array::{AttributeArray, AttributeHandle, AttributeType, AttributeWriteHandle};
pub use attribute_group::{GroupHandle, GroupIndex, GroupType, GroupWriteHandle, GROUP_BITS};
pub use attribute_set::{AttributeSet, Descriptor, INVALID_POS};
pub use attribute_string::{StringAttributeHandle, StringAttributeWriteHandle, StringMetaInserter};
pub use codec::{AttributeValue, Codec, FixedPointRange};
pub use index_filter::{
    ActiveFilter, AttributeHashFilter, BBoxFilter, BinaryFilter, BoundFilter, GroupFilter, InactiveFilter, IndexFilter,
    LevelSetFilter, MultiGroupFilter, NullFilter, RandomLeafFilter,
};
pub use index_iterator::{IndexIter, IndexState, VoxelSelection};
pub use point_advect::{advect_points, AdvectionDeformer};
pub use point_attribute::{
    append_attribute, append_string_attribute, append_typed_attribute, collapse_attribute, collapse_string_attribute,
    compact_attributes, drop_attribute, drop_attributes, drop_attributes_by_name, rename_attribute, rename_attributes,
};
pub use point_conversion::{
    compute_voxel_size, convert_point_data_grid_attribute, convert_point_data_grid_group,
    convert_point_data_grid_position, convert_point_data_grid_string_attribute, create_point_data_grid,
    create_point_data_grid_from_index, populate_attribute, populate_string_attribute,
};
pub use point_count::{
    active_point_count, group_point_count, inactive_point_count, point_count, point_count_filtered, point_count_grid,
    point_count_grid_transformed, point_offsets,
};
pub use point_data_grid::{
    PointDataGrid, PointDataLeaf, PointDataLeafMut, PointDataLeafNode, PointDataTree, PointLeaf, PointLeafMut,
};
pub use point_delete::{delete_from_group, delete_from_groups, keep_points_by_filter};
pub use point_group::{
    append_group, append_groups, compact_groups, delete_missing_point_groups, drop_all_groups, drop_group, drop_groups,
    set_group, set_group_all, set_group_by_filter, set_group_by_random_percentage, set_group_by_random_target,
};
pub use point_mask::{
    convert_points_to_mask, convert_points_to_mask_deformed, convert_points_to_mask_transformed,
    convert_points_to_mask_tree, CountValue, NullDeformer, PointDeformer,
};
pub use point_move::{move_points, move_points_to_transform, CachedDeformer};
pub use point_rasterize_sdf::{
    rasterize_smooth_spheres, rasterize_smooth_spheres_varying, rasterize_spheres, rasterize_spheres_varying,
    LEVEL_SET_HALF_WIDTH,
};
pub use point_rasterize_trilinear::{rasterize_trilinear, rasterize_trilinear_staggered, TrilinearSource};
pub use point_sample::{
    box_sample, point_sample, quadratic_sample, sample_grid, sample_grid_auto, SampleSource, SampleTarget, SampledValue,
};
pub use point_scatter::{dense_uniform_point_scatter, non_uniform_point_scatter, uniform_point_scatter};
pub use point_statistics::{
    accumulate, accumulate_tree, eval_average, eval_average_tree, eval_min_max, eval_min_max_trees, StatValue,
};

use thiserror::Error;

/// Errors of point data operations (the exceptions thrown by OpenVDB's points module).
#[derive(Debug, Error, Clone, PartialEq)]
pub enum Error {
    /// A value or argument is invalid (`ValueError`).
    #[error("value error: {0}")]
    Value(String),
    /// A name or index could not be found (`LookupError`).
    #[error("lookup error: {0}")]
    Lookup(String),
    /// A key is invalid or duplicated (`KeyError`).
    #[error("key error: {0}")]
    Key(String),
    /// A type does not match (`TypeError`).
    #[error("type error: {0}")]
    Type(String),
    /// An index is out of range (`IndexError`).
    #[error("index error: {0}")]
    Index(String),
}

/// Result alias for point operations.
pub type Result<T> = std::result::Result<T, Error>;
