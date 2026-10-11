//! Math primitives: vectors, quaternions, coordinates, bounding boxes, matrices, maps,
//! transforms, rays and statistics.

pub mod bbox;
pub mod conj_gradient;
pub mod coord;
pub mod finite_difference;
pub mod maps;
pub mod mat;
pub mod math_util;
pub mod operators;
pub mod quat;
pub mod ray;
pub mod stats;
pub mod stencils;
pub mod transform;
pub mod vec2;
pub mod vec3;
pub mod vec4;

pub use bbox::{BBox, BBoxd, BBoxf};
pub use coord::{Coord, CoordBBox, CoordBBoxIter};
pub use maps::{
    approx_inverse, simplify, AffineMap, ArithmeticError, MapBase, NonlinearFrustumMap, ScaleMap, ScaleTranslateMap,
    TranslationMap, UniformScaleMap, UniformScaleTranslateMap, UnitaryMap,
};
pub use mat::{diagonalize_symmetric_matrix, polar_decomposition, Axis, Mat3, Mat3d, Mat3s, Mat4, Mat4d, Mat4s};
pub use math_util::FloatTolerance;
pub use operators::MapJacobian;
pub use quat::{slerp, slerp_mat3, Quat, Quatd, Quats};
pub use ray::{Dda, Ray, Rayd, Rayf, TimeSpan};
pub use stats::{Extrema, Histogram, HistogramError, MinMax, Stats};
pub use transform::{calculate_bounds, Transform};
pub use vec2::{Vec2, Vec2d, Vec2f, Vec2h, Vec2i, Vec2s, Vec2ui};
pub use vec3::{Vec3, Vec3d, Vec3f, Vec3h, Vec3i, Vec3s, Vec3ui};
pub use vec4::{Vec4, Vec4d, Vec4f, Vec4h, Vec4i, Vec4s, Vec4ui};

/// Tolerance used by OpenVDB's `math::Tolerance<float>`.
pub const TOLERANCE_F32: f32 = 1e-8;
/// Tolerance used by OpenVDB's `math::Tolerance<double>`.
pub const TOLERANCE_F64: f64 = 1e-15;
/// `math::Delta<float>`.
pub const DELTA_F32: f32 = 1e-5;
/// `math::Delta<double>`.
pub const DELTA_F64: f64 = 1e-9;
