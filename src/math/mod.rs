//! Math primitives: vectors, coordinates, bounding boxes, matrices, maps and transforms.

pub mod conj_gradient;
pub mod coord;
pub mod finite_difference;
pub mod maps;
pub mod mat;
pub mod operators;
pub mod stencils;
pub mod transform;
pub mod vec3;

pub use coord::{Coord, CoordBBox, CoordBBoxIter};
pub use maps::MapBase;
pub use mat::{Axis, Mat3, Mat3d, Mat3s, Mat4, Mat4d, Mat4s};
pub use operators::MapJacobian;
pub use transform::Transform;
pub use vec3::{Vec3, Vec3d, Vec3f, Vec3h, Vec3i, Vec3s, Vec3ui};

/// Tolerance used by OpenVDB's `math::Tolerance<float>`.
pub const TOLERANCE_F32: f32 = 1e-8;
/// Tolerance used by OpenVDB's `math::Tolerance<double>`.
pub const TOLERANCE_F64: f64 = 1e-15;
/// `math::Delta<float>`.
pub const DELTA_F32: f32 = 1e-5;
/// `math::Delta<double>`.
pub const DELTA_F64: f64 = 1e-9;
