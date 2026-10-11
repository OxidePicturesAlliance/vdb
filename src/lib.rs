//! # vdb — a Rust port of OpenVDB
//!
//! OpenVDB is a library for efficiently storing and manipulating sparse volumetric data
//! on three-dimensional grids. This crate ports its core:
//!
//! * [`math`] — vectors, coordinates, bounding boxes, matrices, maps and transforms
//! * [`tree`] — the hierarchical sparse VDB tree (root, internal and leaf nodes) and
//!   cached value accessors
//! * [`grid`] — grids (tree + transform + metadata)
//! * [`metadata`] — typed metadata maps
//!
//! ```
//! use vdb::prelude::*;
//!
//! let mut grid = FloatGrid::new(0.0);
//! grid.set_name("density");
//! let mut acc = grid.accessor_mut();
//! acc.set_value(&Coord::new(1, 2, 3), 1.5);
//! assert_eq!(acc.get_value(&Coord::new(1, 2, 3)), 1.5);
//! drop(acc);
//! assert_eq!(grid.active_voxel_count(), 1);
//! ```
//!
//! Building a level set and writing it to a `.vdb` file that C++ OpenVDB can read:
//!
//! ```
//! use vdb::prelude::*;
//! use vdb::tools::level_set_sphere::create_level_set_sphere;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let sphere: FloatGrid = create_level_set_sphere(1.0, Vec3f::zero(), 0.05, 3.0, true)?;
//! let path = std::env::temp_dir().join("vdb_doctest_sphere.vdb");
//! vdb::io::write_file(&path, &[sphere.clone().into()], &MetaMap::new())?;
//! let (_meta, grids) = vdb::io::read_file(&path)?;
//! assert_eq!(grids[0].active_voxel_count(), sphere.active_voxel_count());
//! # std::fs::remove_file(&path).ok();
//! # Ok(())
//! # }
//! ```

pub mod grid;
pub mod io;
pub mod math;
pub mod metadata;
pub mod nanovdb;
pub mod points;
pub mod tools;
pub mod tree;
pub mod types;
pub mod util;

pub use grid::{
    AnyGrid, BoolGrid, DoubleGrid, FloatGrid, Grid, GridClass, Int32Grid, Int64Grid, Vec3DGrid, Vec3IGrid, Vec3SGrid,
    VecType,
};
pub use math::{Coord, CoordBBox, Transform, Vec3d, Vec3f, Vec3i};
pub use metadata::{MetaMap, MetaValue};
pub use tree::{Node, Tree};
pub use types::Value;

/// Commonly used items.
pub mod prelude {
    pub use crate::grid::*;
    pub use crate::math::{Coord, CoordBBox, Mat3d, Mat4d, Transform, Vec3, Vec3d, Vec3f, Vec3i};
    pub use crate::metadata::{MetaMap, MetaValue};
    pub use crate::tree::{LeafNode, Node, Tree, TreeBase};
    pub use crate::types::Value;
}
