//! Grids: a tree plus a transform and metadata (port of `openvdb/Grid.h`).

use crate::math::{Coord, CoordBBox, Transform, Vec3d, Vec3f, Vec3i};
use crate::metadata::{MetaMap, MetaValue};
use crate::tree::{Tree, ValueAccessor, ValueAccessorMut};
use crate::types::{TypeName, Value};
use std::fmt;

/// Reserved metadata names (`GridBase::META_*` in OpenVDB).
pub mod meta_names {
    pub const GRID_CLASS: &str = "class";
    pub const GRID_CREATOR: &str = "creator";
    pub const GRID_NAME: &str = "name";
    pub const SAVE_HALF_FLOAT: &str = "is_saved_as_half_float";
    pub const IS_LOCAL_SPACE: &str = "is_local_space";
    pub const VECTOR_TYPE: &str = "vector_type";
    pub const FILE_BBOX_MIN: &str = "file_bbox_min";
    pub const FILE_BBOX_MAX: &str = "file_bbox_max";
    pub const FILE_COMPRESSION: &str = "file_compression";
    pub const FILE_MEM_BYTES: &str = "file_mem_bytes";
    pub const FILE_VOXEL_COUNT: &str = "file_voxel_count";
    pub const FILE_DELAYED_LOAD: &str = "file_delayed_load";
}

/// Semantic class of a grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GridClass {
    #[default]
    Unknown,
    LevelSet,
    FogVolume,
    Staggered,
}

impl GridClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            GridClass::Unknown => "unknown",
            GridClass::LevelSet => "level set",
            GridClass::FogVolume => "fog volume",
            GridClass::Staggered => "staggered",
        }
    }
    pub fn from_str(s: &str) -> Self {
        match s {
            "level set" => GridClass::LevelSet,
            "fog volume" => GridClass::FogVolume,
            "staggered" => GridClass::Staggered,
            _ => GridClass::Unknown,
        }
    }
}

/// How vector values transform under a change of coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VecType {
    #[default]
    Invariant,
    Covariant,
    CovariantNormalize,
    ContravariantRelative,
    ContravariantAbsolute,
}

impl VecType {
    pub fn as_str(&self) -> &'static str {
        match self {
            VecType::Invariant => "invariant",
            VecType::Covariant => "covariant",
            VecType::CovariantNormalize => "covariant normalize",
            VecType::ContravariantRelative => "contravariant relative",
            VecType::ContravariantAbsolute => "contravariant absolute",
        }
    }
    pub fn from_str(s: &str) -> Self {
        match s {
            "covariant" => VecType::Covariant,
            "covariant normalize" => VecType::CovariantNormalize,
            "contravariant relative" => VecType::ContravariantRelative,
            "contravariant absolute" => VecType::ContravariantAbsolute,
            _ => VecType::Invariant,
        }
    }
}

/// A grid: a sparse [`Tree`] of values, a [`Transform`] to world space, and metadata.
#[derive(Clone)]
pub struct Grid<T: Value> {
    tree: Tree<T>,
    transform: Transform,
    meta: MetaMap,
}

pub type FloatGrid = Grid<f32>;
pub type DoubleGrid = Grid<f64>;
pub type Int32Grid = Grid<i32>;
pub type Int64Grid = Grid<i64>;
pub type BoolGrid = Grid<bool>;
pub type Vec3SGrid = Grid<Vec3f>;
pub type Vec3DGrid = Grid<Vec3d>;
pub type Vec3IGrid = Grid<Vec3i>;

impl<T: Value> Default for Grid<T> {
    fn default() -> Self {
        Self::new(T::zero())
    }
}

impl<T: Value> Grid<T> {
    /// An empty grid with the given background value and an identity transform.
    pub fn new(background: T) -> Self {
        Grid { tree: Tree::new(background), transform: Transform::identity(), meta: MetaMap::new() }
    }
    /// Wrap an existing tree.
    pub fn from_tree(tree: Tree<T>) -> Self {
        Grid { tree, transform: Transform::identity(), meta: MetaMap::new() }
    }
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }
    pub fn with_name(mut self, name: &str) -> Self {
        self.set_name(name);
        self
    }

    pub fn tree(&self) -> &Tree<T> {
        &self.tree
    }
    pub fn tree_mut(&mut self) -> &mut Tree<T> {
        &mut self.tree
    }
    pub fn set_tree(&mut self, tree: Tree<T>) {
        self.tree = tree;
    }
    pub fn into_tree(self) -> Tree<T> {
        self.tree
    }
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
    pub fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
    pub fn set_transform(&mut self, t: Transform) {
        self.transform = t;
    }
    pub fn metadata(&self) -> &MetaMap {
        &self.meta
    }
    pub fn metadata_mut(&mut self) -> &mut MetaMap {
        &mut self.meta
    }
    pub fn insert_meta(&mut self, name: &str, value: impl Into<MetaValue>) {
        self.meta.insert(name, value);
    }

    /// A copy of this grid with the same transform and metadata but an empty tree.
    pub fn copy_with_new_tree(&self) -> Self {
        Grid { tree: Tree::new(*self.tree.background()), transform: self.transform.clone(), meta: self.meta.clone() }
    }
    /// A grid with a different tree but this grid's transform and metadata.
    pub fn copy_replacing_tree<U: Value>(&self, tree: Tree<U>) -> Grid<U> {
        Grid { tree, transform: self.transform.clone(), meta: self.meta.clone() }
    }

    pub fn name(&self) -> &str {
        self.meta.get_str(meta_names::GRID_NAME).unwrap_or("")
    }
    pub fn set_name(&mut self, name: &str) {
        self.meta.insert(meta_names::GRID_NAME, name);
    }
    pub fn creator(&self) -> &str {
        self.meta.get_str(meta_names::GRID_CREATOR).unwrap_or("")
    }
    pub fn set_creator(&mut self, creator: &str) {
        self.meta.insert(meta_names::GRID_CREATOR, creator);
    }
    pub fn grid_class(&self) -> GridClass {
        self.meta.get_str(meta_names::GRID_CLASS).map(GridClass::from_str).unwrap_or_default()
    }
    pub fn set_grid_class(&mut self, c: GridClass) {
        self.meta.insert(meta_names::GRID_CLASS, c.as_str());
    }
    pub fn vector_type(&self) -> VecType {
        self.meta.get_str(meta_names::VECTOR_TYPE).map(VecType::from_str).unwrap_or_default()
    }
    pub fn set_vector_type(&mut self, v: VecType) {
        self.meta.insert(meta_names::VECTOR_TYPE, v.as_str());
    }
    pub fn save_float_as_half(&self) -> bool {
        self.meta.get(meta_names::SAVE_HALF_FLOAT).and_then(|m| m.as_bool()).unwrap_or(false)
    }
    pub fn set_save_float_as_half(&mut self, on: bool) {
        self.meta.insert(meta_names::SAVE_HALF_FLOAT, on);
    }
    pub fn is_in_world_space(&self) -> bool {
        !self.meta.get(meta_names::IS_LOCAL_SPACE).and_then(|m| m.as_bool()).unwrap_or(false)
    }
    pub fn set_is_in_world_space(&mut self, world: bool) {
        self.meta.insert(meta_names::IS_LOCAL_SPACE, !world);
    }

    pub fn background(&self) -> &T {
        self.tree.background()
    }
    pub fn voxel_size(&self) -> Vec3d {
        self.transform.voxel_size()
    }
    pub fn index_to_world(&self, xyz: Vec3d) -> Vec3d {
        self.transform.index_to_world(xyz)
    }
    pub fn world_to_index(&self, xyz: Vec3d) -> Vec3d {
        self.transform.world_to_index(xyz)
    }
    pub fn get_value(&self, xyz: &Coord) -> T {
        self.tree.get_value(xyz)
    }
    pub fn set_value(&mut self, xyz: &Coord, v: T) {
        self.tree.set_value(xyz, v)
    }
    pub fn accessor(&self) -> ValueAccessor<'_, crate::tree::Internal2<T>> {
        self.tree.accessor()
    }
    pub fn accessor_mut(&mut self) -> ValueAccessorMut<'_, crate::tree::Internal2<T>> {
        self.tree.accessor_mut()
    }
    pub fn active_voxel_count(&self) -> u64 {
        self.tree.active_voxel_count()
    }
    pub fn eval_active_voxel_bbox(&self) -> CoordBBox {
        self.tree.eval_active_voxel_bbox()
    }
    pub fn mem_usage(&self) -> usize {
        self.tree.mem_usage()
    }
    pub fn empty(&self) -> bool {
        self.tree.is_empty()
    }
    pub fn prune(&mut self, tolerance: T) {
        self.tree.prune(tolerance)
    }
    pub fn clip(&mut self, bbox: &CoordBBox) {
        self.tree.clip(bbox)
    }
    /// Clip to a world-space axis-aligned box.
    pub fn clip_world(&mut self, min: Vec3d, max: Vec3d) {
        let a = self.transform.world_to_index(min);
        let b = self.transform.world_to_index(max);
        let lo = a.min_elem(&b);
        let hi = a.max_elem(&b);
        self.tree.clip(&CoordBBox::new(Coord::ceil(lo), Coord::floor(hi)));
    }
    /// Store the active bounding box and voxel count in the metadata, as OpenVDB does
    /// before writing a file.
    pub fn add_stats_metadata(&mut self) {
        let bbox = self.eval_active_voxel_bbox();
        let (min, max) = if bbox.is_empty() { (Coord::origin(), Coord::origin()) } else { (bbox.min, bbox.max) };
        self.meta.insert(meta_names::FILE_BBOX_MIN, min.as_vec3i());
        self.meta.insert(meta_names::FILE_BBOX_MAX, max.as_vec3i());
        self.meta.insert(meta_names::FILE_MEM_BYTES, self.mem_usage() as i64);
        self.meta.insert(meta_names::FILE_VOXEL_COUNT, self.active_voxel_count() as i64);
    }
}

impl<T: Value + TypeName> Grid<T> {
    /// OpenVDB grid type name, e.g. `"Tree_float_5_4_3"`.
    pub fn grid_type() -> String {
        format!("Tree_{}_5_4_3", T::type_name())
    }
    pub fn value_type() -> &'static str {
        T::type_name()
    }
}

impl<T: Value> fmt::Debug for Grid<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Grid")
            .field("name", &self.name())
            .field("background", self.background())
            .field("active_voxels", &self.active_voxel_count())
            .field("transform", &self.transform)
            .finish()
    }
}

/// A grid of any supported value type, as read from a file.
#[derive(Clone, Debug)]
pub enum AnyGrid {
    Float(FloatGrid),
    Double(DoubleGrid),
    Int32(Int32Grid),
    Int64(Int64Grid),
    Bool(BoolGrid),
    Vec3S(Vec3SGrid),
    Vec3D(Vec3DGrid),
    Vec3I(Vec3IGrid),
    /// A topology-only grid (OpenVDB's `MaskGrid`, tree type `"Tree_mask_5_4_3"`),
    /// represented as a [`BoolGrid`] whose voxel values equal their active states.
    Mask(BoolGrid),
    /// A point data grid (OpenVDB's `PointDataGrid`, tree type `"Tree_ptdataidx32_5_4_3"`).
    PointData(crate::points::PointDataGrid),
    /// A point index grid (OpenVDB's `PointIndexGrid`, tree type `"Tree_ptidx32_5_4_3"`).
    PointIndex(crate::tools::point_index_grid::PointIndexGrid),
}

/// Run an expression on the typed grid inside an [`AnyGrid`].
#[macro_export]
macro_rules! with_any_grid {
    ($any:expr, $g:ident => $body:expr) => {
        match $any {
            $crate::grid::AnyGrid::Float($g) => $body,
            $crate::grid::AnyGrid::Double($g) => $body,
            $crate::grid::AnyGrid::Int32($g) => $body,
            $crate::grid::AnyGrid::Int64($g) => $body,
            $crate::grid::AnyGrid::Bool($g) => $body,
            $crate::grid::AnyGrid::Vec3S($g) => $body,
            $crate::grid::AnyGrid::Vec3D($g) => $body,
            $crate::grid::AnyGrid::Vec3I($g) => $body,
            $crate::grid::AnyGrid::Mask($g) => $body,
            $crate::grid::AnyGrid::PointData($g) => $body,
            $crate::grid::AnyGrid::PointIndex($g) => $body,
        }
    };
}

impl AnyGrid {
    pub fn name(&self) -> &str {
        with_any_grid!(self, g => g.name())
    }
    pub fn set_name(&mut self, name: &str) {
        with_any_grid!(self, g => g.set_name(name))
    }
    pub fn transform(&self) -> &Transform {
        with_any_grid!(self, g => g.transform())
    }
    pub fn metadata(&self) -> &MetaMap {
        with_any_grid!(self, g => g.metadata())
    }
    pub fn metadata_mut(&mut self) -> &mut MetaMap {
        with_any_grid!(self, g => g.metadata_mut())
    }
    pub fn active_voxel_count(&self) -> u64 {
        with_any_grid!(self, g => g.active_voxel_count())
    }
    pub fn eval_active_voxel_bbox(&self) -> CoordBBox {
        with_any_grid!(self, g => g.eval_active_voxel_bbox())
    }
    pub fn grid_class(&self) -> GridClass {
        with_any_grid!(self, g => g.grid_class())
    }
    /// OpenVDB grid type name, e.g. `"Tree_float_5_4_3"`.
    pub fn grid_type(&self) -> String {
        match self {
            AnyGrid::Float(_) => FloatGrid::grid_type(),
            AnyGrid::Double(_) => DoubleGrid::grid_type(),
            AnyGrid::Int32(_) => Int32Grid::grid_type(),
            AnyGrid::Int64(_) => Int64Grid::grid_type(),
            AnyGrid::Bool(_) => BoolGrid::grid_type(),
            AnyGrid::Vec3S(_) => Vec3SGrid::grid_type(),
            AnyGrid::Vec3D(_) => Vec3DGrid::grid_type(),
            AnyGrid::Vec3I(_) => Vec3IGrid::grid_type(),
            AnyGrid::Mask(_) => "Tree_mask_5_4_3".to_string(),
            AnyGrid::PointData(_) => crate::points::PointDataGrid::grid_type(),
            AnyGrid::PointIndex(_) => crate::tools::point_index_grid::PointIndexGrid::grid_type(),
        }
    }
    /// The point data grid, if this is an [`AnyGrid::PointData`].
    pub fn as_point_data(&self) -> Option<&crate::points::PointDataGrid> {
        if let AnyGrid::PointData(g) = self {
            Some(g)
        } else {
            None
        }
    }
    /// The point data grid, mutably, if this is an [`AnyGrid::PointData`].
    pub fn as_point_data_mut(&mut self) -> Option<&mut crate::points::PointDataGrid> {
        if let AnyGrid::PointData(g) = self {
            Some(g)
        } else {
            None
        }
    }
    /// Convert into the point data grid, if this is an [`AnyGrid::PointData`].
    pub fn into_point_data(self) -> Option<crate::points::PointDataGrid> {
        if let AnyGrid::PointData(g) = self {
            Some(g)
        } else {
            None
        }
    }
    /// The mask grid, if this is an [`AnyGrid::Mask`].
    pub fn as_mask(&self) -> Option<&BoolGrid> {
        if let AnyGrid::Mask(g) = self {
            Some(g)
        } else {
            None
        }
    }
    /// The mask grid, mutably, if this is an [`AnyGrid::Mask`].
    pub fn as_mask_mut(&mut self) -> Option<&mut BoolGrid> {
        if let AnyGrid::Mask(g) = self {
            Some(g)
        } else {
            None
        }
    }
    /// Convert into the mask grid, if this is an [`AnyGrid::Mask`].
    pub fn into_mask(self) -> Option<BoolGrid> {
        if let AnyGrid::Mask(g) = self {
            Some(g)
        } else {
            None
        }
    }
}

macro_rules! any_grid_conv {
    ($($v:ident, $t:ty, $as:ident, $as_mut:ident, $into:ident);* $(;)?) => {
        impl AnyGrid {$(
            pub fn $as(&self) -> Option<&Grid<$t>> {
                if let AnyGrid::$v(g) = self { Some(g) } else { None }
            }
            pub fn $as_mut(&mut self) -> Option<&mut Grid<$t>> {
                if let AnyGrid::$v(g) = self { Some(g) } else { None }
            }
            pub fn $into(self) -> Option<Grid<$t>> {
                if let AnyGrid::$v(g) = self { Some(g) } else { None }
            }
        )*}
        $(impl From<Grid<$t>> for AnyGrid {
            fn from(g: Grid<$t>) -> Self { AnyGrid::$v(g) }
        })*
    };
}
any_grid_conv!(
    Float, f32, as_float, as_float_mut, into_float;
    Double, f64, as_double, as_double_mut, into_double;
    Int32, i32, as_int32, as_int32_mut, into_int32;
    Int64, i64, as_int64, as_int64_mut, into_int64;
    Bool, bool, as_bool, as_bool_mut, into_bool;
    Vec3S, Vec3f, as_vec3s, as_vec3s_mut, into_vec3s;
    Vec3D, Vec3d, as_vec3d, as_vec3d_mut, into_vec3d;
    Vec3I, Vec3i, as_vec3i, as_vec3i_mut, into_vec3i;
);

impl From<crate::points::PointDataGrid> for AnyGrid {
    fn from(g: crate::points::PointDataGrid) -> Self {
        AnyGrid::PointData(g)
    }
}

impl From<crate::tools::point_index_grid::PointIndexGrid> for AnyGrid {
    fn from(g: crate::tools::point_index_grid::PointIndexGrid) -> Self {
        AnyGrid::PointIndex(g)
    }
}

impl AnyGrid {
    /// The point index grid, if this is an [`AnyGrid::PointIndex`].
    pub fn as_point_index(&self) -> Option<&crate::tools::point_index_grid::PointIndexGrid> {
        if let AnyGrid::PointIndex(g) = self {
            Some(g)
        } else {
            None
        }
    }
    /// Convert into the point index grid, if this is an [`AnyGrid::PointIndex`].
    pub fn into_point_index(self) -> Option<crate::tools::point_index_grid::PointIndexGrid> {
        if let AnyGrid::PointIndex(g) = self {
            Some(g)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grid_meta() {
        let mut g = FloatGrid::new(3.0).with_name("density");
        g.set_grid_class(GridClass::FogVolume);
        assert_eq!(g.name(), "density");
        assert_eq!(g.grid_class(), GridClass::FogVolume);
        assert_eq!(FloatGrid::grid_type(), "Tree_float_5_4_3");
        assert_eq!(Vec3SGrid::grid_type(), "Tree_vec3s_5_4_3");
        g.set_value(&Coord::new(1, 2, 3), 1.0);
        let any: AnyGrid = g.into();
        assert_eq!(any.name(), "density");
        assert_eq!(any.active_voxel_count(), 1);
        assert!(any.as_float().is_some());
    }
}
