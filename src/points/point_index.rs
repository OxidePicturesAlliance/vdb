//! Grid-like accessors of [`PointIndexGrid`] (OpenVDB's `PointIndexGrid`, tree type
//! `"Tree_ptidx32_5_4_3"`), so that point index grids can be stored in an
//! [`AnyGrid`](crate::AnyGrid) and written to and read from `.vdb` files.

use crate::grid::{meta_names, GridClass};
use crate::math::CoordBBox;
use crate::tools::point_index_grid::{PointIndexGrid, PointIndexTree};

impl PointIndexGrid {
    /// OpenVDB grid type name.
    pub const GRID_TYPE: &'static str = "Tree_ptidx32_5_4_3";

    /// OpenVDB grid type name, `"Tree_ptidx32_5_4_3"`.
    pub fn grid_type() -> String {
        Self::GRID_TYPE.to_string()
    }
    /// The grid name.
    pub fn name(&self) -> &str {
        self.metadata().get_str(meta_names::GRID_NAME).unwrap_or("")
    }
    /// Set the grid name.
    pub fn set_name(&mut self, name: &str) {
        self.metadata_mut().insert(meta_names::GRID_NAME, name);
    }
    /// Builder-style [`set_name`](Self::set_name).
    pub fn with_name(mut self, name: &str) -> Self {
        self.set_name(name);
        self
    }
    /// The grid class.
    pub fn grid_class(&self) -> GridClass {
        self.metadata().get_str(meta_names::GRID_CLASS).map(GridClass::from_str).unwrap_or_default()
    }
    /// True if floating-point values are saved as half floats (irrelevant for indices).
    pub fn save_float_as_half(&self) -> bool {
        self.metadata().get(meta_names::SAVE_HALF_FLOAT).and_then(|m| m.as_bool()).unwrap_or(false)
    }
    /// Set the half-float flag.
    pub fn set_save_float_as_half(&mut self, on: bool) {
        self.metadata_mut().insert(meta_names::SAVE_HALF_FLOAT, on);
    }
    /// The background value (0).
    pub fn background(&self) -> &u32 {
        self.tree().tree().background()
    }
    /// Number of voxels containing points.
    pub fn active_voxel_count(&self) -> u64 {
        self.tree().active_voxel_count()
    }
    /// Bounding box of the voxels containing points.
    pub fn eval_active_voxel_bbox(&self) -> CoordBBox {
        self.tree().tree().eval_active_voxel_bbox()
    }
    /// Approximate memory usage in bytes.
    pub fn mem_usage(&self) -> usize {
        self.tree().mem_usage()
    }
}

impl std::fmt::Debug for PointIndexGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointIndexGrid")
            .field("name", &self.name())
            .field("points", &self.tree().point_count())
            .field("transform", self.transform())
            .finish()
    }
}

impl PointIndexTree {
    /// The background value (0).
    pub fn background(&self) -> &u32 {
        self.tree().background()
    }
    /// Number of nodes per level (leaf to root).
    pub fn node_count(&self) -> Vec<usize> {
        self.tree().node_count()
    }
    /// Number of active tiles.
    pub fn active_tile_count(&self) -> u64 {
        self.tree().active_tile_count()
    }
    /// Number of inactive voxels.
    pub fn inactive_voxel_count(&self) -> u64 {
        self.tree().inactive_voxel_count()
    }
    /// Approximate memory usage in bytes.
    pub fn mem_usage(&self) -> usize {
        self.tree().mem_usage() + self.indices_map().values().map(|v| v.len() * 4).sum::<usize>()
    }
}
