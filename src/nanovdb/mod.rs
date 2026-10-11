//! NanoVDB: a linearized, pointer-free, read-only representation of VDB grids
//! (port of the core of `nanovdb/NanoVDB.h` and its companion tools).
//!
//! A NanoVDB grid is a single contiguous, 32-byte aligned byte buffer whose layout is
//! byte-for-byte identical to the C++ `nanovdb::Grid` memory layout (ABI version
//! [`MAJOR_VERSION`]`.`[`MINOR_VERSION`]`.`[`PATCH_VERSION`]):
//!
//! ```text
//! [GridData][TreeData][RootData][ROOT TILES...][Upper nodes...][Lower nodes...][Leaves...]
//! [GridBlindMetaData...][blind data...]
//! ```
//!
//! Child "pointers" are signed byte offsets relative to the parent node, so a buffer can be
//! copied, memory mapped, written to disk or uploaded to a GPU as-is. This module provides:
//!
//! * [`GridHandle`] — owns an aligned buffer holding one or more grids,
//! * zero-copy read-only views: [`NanoGrid`], [`NanoTree`], [`NanoRoot`], [`NanoUpper`],
//!   [`NanoLower`], [`NanoLeaf`] and a caching [`ReadAccessor`],
//! * conversion from the crate's [`Grid`](crate::grid::Grid) ([`create_nano_grid`] and the
//!   configurable [`CreateNanoGrid`], including the quantized [`Fp4`], [`Fp8`], [`Fp16`] and
//!   [`FpN`] encodings and the [`ValueIndex`]/[`ValueOnIndex`] index grids) and back
//!   ([`nano_to_openvdb`]),
//! * statistics ([`update_grid_stats`]) and CRC32 checksums ([`update_checksum`]),
//! * `.nvdb` file I/O with the NONE, ZIP and BLOSC codecs ([`io`]),
//! * interpolation ([`sampler`]), ray marching ([`hdda`]) and level-set primitives
//!   ([`primitives`]).
//!
//! ```
//! use vdb::prelude::*;
//! use vdb::nanovdb::{create_nano_grid, NanoGrid};
//!
//! let mut grid = FloatGrid::new(0.0);
//! grid.set_value(&Coord::new(1, 2, 3), 1.5);
//! let handle = create_nano_grid::<f32, f32>(&grid);
//! let nano = handle.grid::<f32>(0).unwrap();
//! assert_eq!(nano.tree().get_value(Coord::new(1, 2, 3)), 1.5);
//! assert_eq!(nano.active_voxel_count(), 1);
//! ```

mod build_types;
mod checksum;
mod create;
mod data;
mod dither;
mod grid;
mod handle;
pub mod hdda;
pub mod io;
pub mod primitives;
pub mod sampler;
mod stats;
mod to_openvdb;
mod tree;

pub use build_types::{
    BuildType, Fp16, Fp4, Fp8, FpN, IndexBuildType, NanoPod, StatsValue, ValueIndex, ValueMask, ValueOnIndex,
};
pub use checksum::{crc32, eval_checksum, update_checksum, update_grid_count, validate_checksum};
pub use create::{create_nano_grid, AbsDiff, BuildFrom, CreateNanoGrid, Oracle, RelDiff};
pub use data::{Checksum, GridBlindMetaData, InternalLayout, LeafLayout, Map, MaskView, NodeLayout, Version};
pub use grid::{BlindType, GridHeader, NanoGrid};
pub use handle::{AlignedBuffer, GridHandle};
pub use stats::{update_grid_stats, StatsMode};
pub use to_openvdb::{index_to_openvdb, nano_to_openvdb, ToOpenVdb};
pub use tree::{
    NanoInternal, NanoLeaf, NanoLower, NanoRoot, NanoTree, NanoUpper, ReadAccessor, RootTile, ValueLocation,
};

/// Alignment, in bytes, of every NanoVDB data structure (`NANOVDB_DATA_ALIGNMENT`).
pub const DATA_ALIGNMENT: usize = 32;
/// Legacy magic number used by both grids and files before v32.6 ("NanoVDB0").
pub const MAGIC_NUMB: u64 = 0x304244566f6e614e;
/// Magic number of a serialized grid ("NanoVDB1").
pub const MAGIC_GRID: u64 = 0x314244566f6e614e;
/// Magic number of a NanoVDB file ("NanoVDB2").
pub const MAGIC_FILE: u64 = 0x324244566f6e614e;
/// Mask that removes the trailing number from a magic number.
pub const MAGIC_MASK: u64 = 0x00FFFFFFFFFFFFFF;
/// Major version: reflects changes to the ABI and hence the file format.
pub const MAJOR_VERSION: u32 = 32;
/// Minor version: reflects changes to the API but not the ABI.
pub const MINOR_VERSION: u32 = 9;
/// Patch version: reflects changes that affect neither ABI nor API.
pub const PATCH_VERSION: u32 = 2;

/// Size in bytes of `GridData`.
pub const GRID_DATA_SIZE: usize = 672;
/// Size in bytes of `TreeData`.
pub const TREE_DATA_SIZE: usize = 64;
/// Size in bytes of `GridBlindMetaData`.
pub const BLIND_META_DATA_SIZE: usize = 288;
/// Maximum length (including the NUL terminator) of a grid or blind-data name stored inline.
pub const MAX_NAME_SIZE: usize = 256;

/// Errors raised when interpreting or producing NanoVDB buffers.
#[derive(Debug, thiserror::Error)]
pub enum NanoError {
    /// The buffer does not hold a valid NanoVDB grid.
    #[error("invalid NanoVDB grid: {0}")]
    InvalidGrid(String),
    /// The grid's value type differs from the requested one.
    #[error("grid type mismatch: expected {expected:?}, found {found:?}")]
    TypeMismatch {
        /// The requested type.
        expected: GridType,
        /// The type stored in the buffer.
        found: GridType,
    },
    /// An unsupported feature was requested.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// An I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A malformed file.
    #[error("invalid NanoVDB file: {0}")]
    InvalidFile(String),
}

/// Result type of fallible NanoVDB operations.
pub type Result<T> = std::result::Result<T, NanoError>;

macro_rules! u32_enum {
    ($(#[$m:meta])* $name:ident { $($(#[$vm:meta])* $v:ident = $n:expr),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum $name { $($(#[$vm])* $v = $n),* }
        impl $name {
            /// Decode a raw value (unknown values map to the first variant).
            pub fn from_u32(v: u32) -> Self {
                match v { $($n => $name::$v,)* _ => Self::from_u32_fallback() }
            }
            /// The raw value.
            #[inline]
            pub fn to_u32(self) -> u32 { self as u32 }
        }
    };
}

u32_enum!(
    /// Value types supported by NanoVDB (`nanovdb::GridType`).
    GridType {
        /// Unknown value type.
        Unknown = 0,
        /// `float`.
        Float = 1,
        /// `double`.
        Double = 2,
        /// `int16_t`.
        Int16 = 3,
        /// `int32_t`.
        Int32 = 4,
        /// `int64_t`.
        Int64 = 5,
        /// `Vec3f`.
        Vec3f = 6,
        /// `Vec3d`.
        Vec3d = 7,
        /// Active state only.
        Mask = 8,
        /// IEEE 754 half.
        Half = 9,
        /// `uint32_t`.
        UInt32 = 10,
        /// Boolean values stored as bits.
        Boolean = 11,
        /// Packed RGBA.
        RGBA8 = 12,
        /// 4-bit quantized floats.
        Fp4 = 13,
        /// 8-bit quantized floats.
        Fp8 = 14,
        /// 16-bit quantized floats.
        Fp16 = 15,
        /// Variable bit-width quantized floats.
        FpN = 16,
        /// `Vec4f`.
        Vec4f = 17,
        /// `Vec4d`.
        Vec4d = 18,
        /// Index into an external array of active and inactive values.
        Index = 19,
        /// Index into an external array of active values.
        OnIndex = 20,
        /// Point indices.
        PointIndex = 23,
        /// 8-bit quantized vectors (blind data only).
        Vec3u8 = 24,
        /// 16-bit quantized vectors (blind data only).
        Vec3u16 = 25,
        /// `uint8_t`.
        UInt8 = 26,
        /// One past the last valid type.
        End = 27,
    }
);

impl GridType {
    fn from_u32_fallback() -> Self {
        GridType::Unknown
    }
    /// The C++ name of the type (`nanovdb::toStr(GridType)`).
    pub fn as_str(&self) -> &'static str {
        use GridType::*;
        match self {
            Unknown => "Unknown",
            Float => "float",
            Double => "double",
            Int16 => "int16",
            Int32 => "int32",
            Int64 => "int64",
            Vec3f => "Vec3f",
            Vec3d => "Vec3d",
            Mask => "Mask",
            Half => "Half",
            UInt32 => "uint32",
            Boolean => "bool",
            RGBA8 => "RGBA8",
            Fp4 => "Float4",
            Fp8 => "Float8",
            Fp16 => "Float16",
            FpN => "FloatN",
            Vec4f => "Vec4f",
            Vec4d => "Vec4d",
            Index => "Index",
            OnIndex => "OnIndex",
            PointIndex => "PointIndex",
            Vec3u8 => "Vec3u8",
            Vec3u16 => "Vec3u16",
            UInt8 => "uint8",
            End => "End",
        }
    }
}

u32_enum!(
    /// Grid classes (`nanovdb::GridClass`, a superset of OpenVDB's).
    GridClass {
        /// Unknown class.
        Unknown = 0,
        /// Narrow-band level set.
        LevelSet = 1,
        /// Fog volume.
        FogVolume = 2,
        /// Staggered MAC grid.
        Staggered = 3,
        /// Point index grid.
        PointIndex = 4,
        /// Point data grid.
        PointData = 5,
        /// Active states only.
        Topology = 6,
        /// Volume of geometric cubes.
        VoxelVolume = 7,
        /// Grid whose values are offsets into an external array.
        IndexGrid = 8,
        /// Index grid for learnable tensor features.
        TensorGrid = 9,
        /// Voxels point to lists of primitive ids.
        VoxelBVH = 10,
        /// One past the last valid class.
        End = 11,
    }
);

impl GridClass {
    fn from_u32_fallback() -> Self {
        GridClass::Unknown
    }
    /// Short C++ name of the class (`nanovdb::toStr(GridClass)`).
    pub fn as_str(&self) -> &'static str {
        use GridClass::*;
        match self {
            Unknown => "?",
            LevelSet => "SDF",
            FogVolume => "FOG",
            Staggered => "MAC",
            PointIndex => "PNTIDX",
            PointData => "PNTDAT",
            Topology => "TOPO",
            VoxelVolume => "VOX",
            IndexGrid => "INDEX",
            TensorGrid => "TENSOR",
            VoxelBVH => "VOXBVH",
            End => "END",
        }
    }
}

/// Grid flags stored in `GridData::mFlags` (`nanovdb::GridFlags`).
pub mod grid_flags {
    /// The grid name is longer than 255 characters and stored as blind data.
    pub const HAS_LONG_GRID_NAME: u32 = 1 << 0;
    /// Nodes contain bounding boxes of active values.
    pub const HAS_BBOX: u32 = 1 << 1;
    /// Nodes contain min/max of active values.
    pub const HAS_MIN_MAX: u32 = 1 << 2;
    /// Nodes contain averages of active values.
    pub const HAS_AVERAGE: u32 = 1 << 3;
    /// Nodes contain standard deviations of active values.
    pub const HAS_STD_DEVIATION: u32 = 1 << 4;
    /// Nodes are arranged breadth-first in memory.
    pub const IS_BREADTH_FIRST: u32 = 1 << 5;
}

u32_enum!(
    /// Modes for computing a grid checksum (`nanovdb::CheckMode`).
    CheckMode {
        /// No checksum (all 64 bits set).
        Disable = 0,
        /// CRC32 of the grid and tree headers only (fast); the default.
        Partial = 1,
        /// CRC32 of the headers and of all nodes and blind data.
        Full = 2,
    }
);

impl CheckMode {
    fn from_u32_fallback() -> Self {
        CheckMode::Disable
    }
}

#[allow(clippy::derivable_impls)]
impl Default for CheckMode {
    fn default() -> Self {
        CheckMode::Partial
    }
}

u32_enum!(
    /// Classes of blind data (`nanovdb::GridBlindDataClass`).
    GridBlindDataClass {
        /// Unknown.
        Unknown = 0,
        /// Indices, e.g. into other arrays.
        IndexArray = 1,
        /// Attributes, typically of points.
        AttributeArray = 2,
        /// A grid name longer than 255 characters.
        GridName = 3,
        /// A channel of values, typically used by index grids.
        ChannelArray = 4,
    }
);

impl GridBlindDataClass {
    fn from_u32_fallback() -> Self {
        GridBlindDataClass::Unknown
    }
}

u32_enum!(
    /// Semantics of blind data (`nanovdb::GridBlindDataSemantic`).
    GridBlindDataSemantic {
        /// Unknown.
        Unknown = 0,
        /// Point positions.
        PointPosition = 1,
        /// Point colors.
        PointColor = 2,
        /// Point normals.
        PointNormal = 3,
        /// Point radii.
        PointRadius = 4,
        /// Point velocities.
        PointVelocity = 5,
        /// Point ids.
        PointId = 6,
        /// World-space coordinates.
        WorldCoords = 7,
        /// Index-space coordinates.
        GridCoords = 8,
        /// Voxel-space coordinates.
        VoxelCoords = 9,
        /// Level-set values.
        LevelSet = 10,
        /// Fog-volume values.
        FogVolume = 11,
        /// Staggered values.
        Staggered = 12,
        /// Point opacities.
        PointOpacity = 13,
        /// Point rotations.
        PointQuat = 14,
        /// Point scales.
        PointScale = 15,
        /// Spherical harmonics, DC component.
        PointSH0 = 16,
        /// Spherical harmonics, AC components.
        PointSHN = 17,
        /// Line ids.
        LineId = 18,
        /// Triangle ids.
        TriangleId = 19,
        /// Gaussian ids.
        GaussianId = 20,
        /// Begin/end index pairs.
        Range = 21,
        /// Voxel BVH values.
        VoxelBVH = 22,
    }
);

impl GridBlindDataSemantic {
    fn from_u32_fallback() -> Self {
        GridBlindDataSemantic::Unknown
    }
    /// Map a grid class to the semantic used for its values (`nanovdb::toSemantic`).
    pub fn from_grid_class(c: GridClass) -> Self {
        match c {
            GridClass::PointIndex => GridBlindDataSemantic::PointId,
            GridClass::LevelSet => GridBlindDataSemantic::LevelSet,
            GridClass::FogVolume => GridBlindDataSemantic::FogVolume,
            GridClass::Staggered => GridBlindDataSemantic::Staggered,
            GridClass::VoxelBVH => GridBlindDataSemantic::VoxelBVH,
            _ => GridBlindDataSemantic::Unknown,
        }
    }
    /// Map a semantic to a grid class (`nanovdb::toGridClass`).
    pub fn to_grid_class(self) -> GridClass {
        use GridBlindDataSemantic::*;
        match self {
            PointPosition | PointColor | PointNormal | PointRadius | PointVelocity => GridClass::PointData,
            PointId => GridClass::PointIndex,
            LevelSet => GridClass::LevelSet,
            FogVolume => GridClass::FogVolume,
            Staggered => GridClass::Staggered,
            VoxelBVH => GridClass::VoxelBVH,
            _ => GridClass::Unknown,
        }
    }
}

/// Classification of the first 64 bits of a buffer (`nanovdb::MagicType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagicType {
    /// None of the cases below.
    Unknown,
    /// An OpenVDB file.
    OpenVDB,
    /// Legacy NanoVDB magic (grid or file).
    NanoVDB,
    /// A serialized NanoVDB grid.
    NanoGrid,
    /// A NanoVDB file.
    NanoFile,
}

/// Classify a 64-bit magic number (`nanovdb::toMagic`).
pub fn to_magic(magic: u64) -> MagicType {
    match magic {
        MAGIC_NUMB => MagicType::NanoVDB,
        MAGIC_GRID => MagicType::NanoGrid,
        MAGIC_FILE => MagicType::NanoFile,
        m if (m & 0xFFFF_FFFF) == 0x56444220 => MagicType::OpenVDB,
        _ => MagicType::Unknown,
    }
}

#[cfg(test)]
mod tests;
