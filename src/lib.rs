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
//! # Cargo features
//!
//! All features are enabled by default.
//!
//! * `read` — read `.vdb` files and streams ([`io`]), and `.nvdb` files with `nanovdb`.
//! * `write` — write `.vdb` files and streams, `.nvdb` files with `nanovdb`, and images
//!   ([`tools::ray_tracer::Film::save_ppm`]). This feature gates every function that creates
//!   or writes files and every encoder of the file formats: a build without it (for example
//!   `default-features = false, features = ["read"]`) cannot write to disk.
//! * `nanovdb` — the [`nanovdb`] module.
//!
//! The [`io`] module exists when `read` or `write` is enabled; the zip and LZ4 dependencies
//! are only pulled in by these two features.
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
//! # #[cfg(all(feature = "read", feature = "write"))]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let sphere: FloatGrid = create_level_set_sphere(1.0, Vec3f::zero(), 0.05, 3.0, true)?;
//! let path = std::env::temp_dir().join("vdb_doctest_sphere.vdb");
//! vdb::io::write_file(&path, &[sphere.clone().into()], &MetaMap::new())?;
//! let (_meta, grids) = vdb::io::read_file(&path)?;
//! assert_eq!(grids[0].active_voxel_count(), sphere.active_voxel_count());
//! # std::fs::remove_file(&path).ok();
//! # Ok(())
//! # }
//! # #[cfg(not(all(feature = "read", feature = "write")))]
//! # fn main() {}
//! ```

// Deliberate style choices in a numerical port that mirrors OpenVDB's loops and types:
// index loops over small fixed-size matrices and tables, `!(a < b)` to keep NaN semantics,
// `drop(accessor)` to end a borrow of the tree, and OpenVDB-style `next`/`from_str` names.
#![allow(
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::should_implement_trait,
    // `as_chunks` and `is_multiple_of` need Rust 1.88 / 1.90, newer than the MSRV (1.85).
    clippy::chunks_exact_to_as_chunks,
    clippy::manual_is_multiple_of,
    unknown_lints
)]

pub mod grid;
#[cfg(any(feature = "read", feature = "write"))]
pub mod io;
pub mod math;
pub mod metadata;
#[cfg(feature = "nanovdb")]
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
