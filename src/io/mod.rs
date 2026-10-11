//! File I/O: reading and writing `.vdb` files compatible with OpenVDB
//! (port of `openvdb/io`).
//!
//! # Supported data
//!
//! * File format versions 220 to 225 (OpenVDB 3.0 and later); files are written as
//!   version 224 with library version 10.0, readable by OpenVDB 4 and later.
//! * Grid types `Tree_{float,double,int32,int64,bool,vec3s,vec3d,vec3i,mask}_5_4_3`,
//!   loaded as the corresponding [`AnyGrid`] variant (mask grids become
//!   [`AnyGrid::Mask`]). Point data grids (`Tree_ptdataidx32_5_4_3`, loaded as
//!   [`AnyGrid::PointData`], with OpenVDB's multi-pass leaf buffers and paged attribute
//!   streams) and point index grids (`Tree_ptidx32_5_4_3`, [`AnyGrid::PointIndex`]) are
//!   supported too. Grids of other types are listed in the descriptors but skipped when
//!   loading.
//! * Compression: none, zip, active-value masks and Blosc (implemented in pure Rust,
//!   see [`blosc`]), plus 16-bit half-float storage of floating-point grids.
//! * Instanced grids (grids sharing another grid's tree) are resolved on read; the writer
//!   never produces instances.
//! * Transforms with linear maps; frustum maps are reported as unsupported.
//!
//! Delayed loading and memory mapping are not implemented: trees are read in full.
//!
//! # Features
//!
//! The module exists when the `read` or `write` feature is enabled. Readers ([`File`],
//! [`Stream`], [`read_file`], ...) require `read`; writers ([`write_file`], [`write_to`],
//! [`write_stream`], [`WriteOptions`], ...) and every encoder they use require `write`. A
//! build with only `read` contains no code that writes VDB data or creates files. This
//! module contains no `unsafe` code.
//!
//! # Example
//!
//! ```
//! # #[cfg(all(feature = "read", feature = "write"))] {
//! use vdb::prelude::*;
//! use vdb::io;
//!
//! let mut grid = FloatGrid::new(0.0).with_name("density");
//! grid.set_value(&Coord::new(1, 2, 3), 0.5);
//! let path = std::env::temp_dir().join("vdb_io_doc_example.vdb");
//! io::write_file(&path, &[grid.into()], &MetaMap::new())?;
//!
//! let (meta, grids) = io::read_file(&path)?;
//! assert!(meta.is_empty());
//! let density = grids[0].as_float().unwrap();
//! assert_eq!(density.get_value(&Coord::new(1, 2, 3)), 0.5);
//! # std::fs::remove_file(&path).ok();
//! # }
//! # Ok::<(), vdb::io::Error>(())
//! ```

#![forbid(unsafe_code)]

mod archive;
pub mod blosc;
mod bytes;
mod compression;
mod error;
mod file;
mod metadata_io;
mod points_io;
mod transform_io;
mod tree_io;
mod value_io;

pub use archive::{
    add_suffix, name_as_string, string_as_unique_name, strip_suffix, FileHeader, GridDescriptor, MAGIC,
    MAX_FILE_VERSION, MIN_FILE_VERSION,
};
pub use compression::Compression;
pub use error::{Error, Result};
#[cfg(feature = "write")]
pub use file::write_to;
#[cfg(feature = "read")]
pub use file::{File, GridMetadata, Stream};
pub use value_io::IoValue;

use crate::grid::AnyGrid;
use crate::metadata::MetaMap;
use std::path::Path;

/// The file format version written by this library (`OPENVDB_FILE_VERSION_MULTIPASS_IO`).
pub const FILE_VERSION: u32 = 224;
/// Library major version recorded in written files.
pub const LIBRARY_MAJOR_VERSION: u32 = 10;
/// Library minor version recorded in written files.
pub const LIBRARY_MINOR_VERSION: u32 = 0;

/// Options for writing files and streams.
#[cfg(feature = "write")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteOptions {
    /// Compression flags. Zip is never applied to level sets and fog volumes (as in
    /// OpenVDB); the other flags apply to all grids.
    pub compression: Compression,
    /// Add `file_bbox_min`, `file_bbox_max`, `file_mem_bytes`, `file_voxel_count` and
    /// `file_compression` metadata to each grid, as OpenVDB does by default.
    pub write_grid_stats: bool,
}

#[cfg(feature = "write")]
impl Default for WriteOptions {
    /// Zip and active-mask compression, with grid statistics.
    fn default() -> Self {
        WriteOptions { compression: Compression::ZIP | Compression::ACTIVE_MASK, write_grid_stats: true }
    }
}

#[cfg(feature = "write")]
impl WriteOptions {
    /// Default options with the given compression flags.
    pub fn with_compression(compression: Compression) -> Self {
        WriteOptions { compression, ..Default::default() }
    }
}

/// Read the file-level metadata and all grids of supported types from a file.
#[cfg(feature = "read")]
pub fn read_file(path: impl AsRef<Path>) -> Result<(MetaMap, Vec<AnyGrid>)> {
    let file = File::open(path)?;
    let grids = file.read_grids()?;
    Ok((file.metadata().clone(), grids))
}

/// Read one grid by name (or unique name) from a file.
#[cfg(feature = "read")]
pub fn read_grid(path: impl AsRef<Path>, name: &str) -> Result<AnyGrid> {
    File::open(path)?.read_grid(name)
}

/// Read the descriptors, metadata and transforms of all grids in a file, without trees.
#[cfg(feature = "read")]
pub fn read_grid_metadata(path: impl AsRef<Path>) -> Result<Vec<GridMetadata>> {
    File::open(path)?.read_all_grid_metadata()
}

/// Write grids and file-level metadata to a file with default options
/// (zip + active-mask compression, grid statistics).
#[cfg(feature = "write")]
pub fn write_file(path: impl AsRef<Path>, grids: &[AnyGrid], metadata: &MetaMap) -> Result<()> {
    write_file_with_options(path, grids, metadata, &WriteOptions::default())
}

/// Write grids and file-level metadata to a file.
#[cfg(feature = "write")]
pub fn write_file_with_options(
    path: impl AsRef<Path>,
    grids: &[AnyGrid],
    metadata: &MetaMap,
    options: &WriteOptions,
) -> Result<()> {
    let f = std::fs::File::create(path)?;
    write_to(std::io::BufWriter::new(f), grids, metadata, options)
}

/// Read a complete archive sequentially from a reader (see [`Stream`]); returns the
/// file-level metadata and the grids of supported types.
#[cfg(feature = "read")]
pub fn read_stream<R: std::io::Read>(reader: R) -> Result<(MetaMap, Vec<AnyGrid>)> {
    let s = Stream::read(reader)?;
    let meta = s.metadata.clone();
    Ok((meta, s.into_grids()))
}

/// Write grids and file-level metadata to a writer as a stream (without grid offsets).
#[cfg(feature = "write")]
pub fn write_stream<W: std::io::Write>(
    writer: W,
    grids: &[AnyGrid],
    metadata: &MetaMap,
    options: &WriteOptions,
) -> Result<()> {
    file::write_stream_to(writer, grids, metadata, options)
}
