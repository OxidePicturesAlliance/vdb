//! Error type for `.vdb` file I/O.

use thiserror::Error;

/// Errors produced while reading or writing `.vdb` files and streams.
#[derive(Debug, Error)]
pub enum Error {
    /// An error from the underlying reader or writer.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The input does not start with the VDB magic number.
    #[error("not a VDB file (bad magic number)")]
    NotVdb,
    /// The file format version is outside the supported range.
    #[error("unsupported VDB file format version {0} (supported: 220 to 225)")]
    UnsupportedVersion(u32),
    /// The input is truncated or structurally invalid.
    #[error("corrupt VDB data: {0}")]
    Corrupt(String),
    /// A requested grid does not exist in the file.
    #[error("no grid named \"{0}\"")]
    GridNotFound(String),
    /// The grid's tree type is not supported by this library.
    #[error("unsupported grid type \"{0}\"")]
    UnsupportedGridType(String),
    /// A feature of the file is not supported (e.g. a non-linear transform map).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Zip (zlib) compression or decompression failed.
    #[error("zip error: {0}")]
    Zip(String),
    /// Blosc compression or decompression failed.
    #[error("blosc error: {0}")]
    Blosc(String),
}

/// Result alias for I/O operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "read")]
impl Error {
    pub(crate) fn corrupt(msg: impl Into<String>) -> Self {
        Error::Corrupt(msg.into())
    }
}
