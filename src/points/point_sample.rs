//! Sampling grids onto point attributes (port of `openvdb/points/PointSample.h`).
//!
//! [`point_sample`], [`box_sample`] and [`quadratic_sample`] interpolate a grid at the
//! position of every point (accepted by a filter, in active voxels) with nearest-neighbour,
//! trilinear or triquadratic interpolation (staggered variants for staggered vector grids)
//! and store the result in an attribute, created with the grid's value type if needed.
//! Values are converted as OpenVDB does: floating-point to integer conversions round to
//! the nearest integer, scalars broadcast to vectors.

use super::attribute_array::{AttributeHandle, AttributeType, AttributeWriteHandle};
use super::codec::{AttributeValue, Codec};
use super::index_filter::IndexFilter;
use super::point_attribute::append_attribute;
use super::point_data_grid::{PointDataGrid, PointLeaf, PointLeafMut};
use super::{Error, Result};
use crate::grid::{Grid, GridClass};
use crate::math::{Vec3, Vec3d, Vec3f, Vec3i};
use crate::tools::interpolation::{
    BoxSampler, PointSampler, QuadraticSampler, Sampler, StaggeredBoxSampler, StaggeredPointSampler,
    StaggeredQuadraticSampler,
};
use crate::tree::ValueSource;
use crate::types::Value;
use rayon::prelude::*;

/// A sampled value in a type-independent form, used for the conversion to the attribute type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampledValue {
    /// A floating-point scalar.
    Float(f64),
    /// An integer scalar.
    Int(i64),
    /// A boolean.
    Bool(bool),
    /// A floating-point vector.
    Vec3(Vec3d),
    /// An integer vector.
    Vec3Int(Vec3i),
}

/// Grid value types that can be sampled onto points.
pub trait SampleSource: Value {
    /// Interpolate `src` at the index-space position `xyz` with the given order
    /// (0 = nearest, 1 = trilinear, 2 = triquadratic), staggered if `staggered`.
    fn sample_at<A: ValueSource<Value = Self>>(src: &A, xyz: Vec3d, order: usize, staggered: bool) -> Self;
    /// Convert to a [`SampledValue`].
    fn to_sampled(self) -> SampledValue;
}

macro_rules! impl_source_scalar {
    ($($t:ty => $v:ident as $c:ty),*) => {$(
        impl SampleSource for $t {
            fn sample_at<A: ValueSource<Value = Self>>(src: &A, xyz: Vec3d, order: usize, _: bool) -> Self {
                match order {
                    0 => PointSampler::sample(src, xyz),
                    1 => BoxSampler::sample(src, xyz),
                    _ => QuadraticSampler::sample(src, xyz),
                }
            }
            fn to_sampled(self) -> SampledValue {
                SampledValue::$v(self as $c)
            }
        }
    )*};
}
impl_source_scalar!(f32 => Float as f64, f64 => Float as f64, i32 => Int as i64, i64 => Int as i64);

impl SampleSource for bool {
    fn sample_at<A: ValueSource<Value = Self>>(src: &A, xyz: Vec3d, _: usize, _: bool) -> Self {
        // Bool grids can only be sampled with the point sampler.
        PointSampler::sample(src, xyz)
    }
    fn to_sampled(self) -> SampledValue {
        SampledValue::Bool(self)
    }
}

macro_rules! impl_source_vec {
    ($($t:ty => $conv:expr),*) => {$(
        impl SampleSource for $t {
            fn sample_at<A: ValueSource<Value = Self>>(src: &A, xyz: Vec3d, order: usize, staggered: bool) -> Self {
                match (order, staggered) {
                    (0, false) => PointSampler::sample(src, xyz),
                    (0, true) => StaggeredPointSampler::sample(src, xyz),
                    (1, false) => BoxSampler::sample(src, xyz),
                    (1, true) => StaggeredBoxSampler::sample(src, xyz),
                    (_, false) => QuadraticSampler::sample(src, xyz),
                    (_, true) => StaggeredQuadraticSampler::sample(src, xyz),
                }
            }
            fn to_sampled(self) -> SampledValue {
                $conv(self)
            }
        }
    )*};
}
impl_source_vec!(
    Vec3f => |v: Vec3f| SampledValue::Vec3(v.map(|c| c as f64)),
    Vec3d => SampledValue::Vec3,
    Vec3i => SampledValue::Vec3Int
);

/// Attribute value types that sampled values can be stored in.
pub trait SampleTarget: AttributeValue {
    /// Convert a sampled value, or `None` if the conversion is not supported.
    fn from_sampled(s: SampledValue) -> Option<Self>;
}

macro_rules! impl_target_float {
    ($($t:ty),*) => {$(
        impl SampleTarget for $t {
            fn from_sampled(s: SampledValue) -> Option<Self> {
                Some(match s {
                    SampledValue::Float(x) => x as $t,
                    SampledValue::Int(i) => i as $t,
                    SampledValue::Bool(b) => b as i32 as $t,
                    _ => return None,
                })
            }
        }
    )*};
}
impl_target_float!(f32, f64);

macro_rules! impl_target_int {
    ($($t:ty),*) => {$(
        impl SampleTarget for $t {
            fn from_sampled(s: SampledValue) -> Option<Self> {
                Some(match s {
                    // math::Round: round half up
                    SampledValue::Float(x) => (x + 0.5).floor() as $t,
                    SampledValue::Int(i) => i as $t,
                    SampledValue::Bool(b) => b as $t,
                    _ => return None,
                })
            }
        }
    )*};
}
impl_target_int!(i8, i16, i32, i64);

impl SampleTarget for bool {
    fn from_sampled(s: SampledValue) -> Option<Self> {
        Some(match s {
            SampledValue::Float(x) => x != 0.0,
            SampledValue::Int(i) => i != 0,
            SampledValue::Bool(b) => b,
            _ => return None,
        })
    }
}

macro_rules! impl_target_vec {
    ($($s:ty),*) => {$(
        impl SampleTarget for Vec3<$s> {
            fn from_sampled(s: SampledValue) -> Option<Self> {
                Some(match s {
                    SampledValue::Float(x) => Vec3::splat(x as $s),
                    SampledValue::Int(i) => Vec3::splat(i as $s),
                    SampledValue::Bool(b) => Vec3::splat(b as i32 as $s),
                    SampledValue::Vec3(v) => v.map(|c| c as $s),
                    SampledValue::Vec3Int(v) => v.map(|c| c as $s),
                })
            }
        }
    )*};
}
impl_target_vec!(f32, f64, i32);

/// Sample `source` onto attribute `attribute` (the grid's name if empty) of the points
/// accepted by `filter`, storing values of type `T` (`sampleGrid<..., TargetValueT>()`). The
/// attribute is created if needed; an existing attribute must have value type `T`.
pub fn sample_grid<S: SampleSource, T: SampleTarget, F: IndexFilter>(
    order: usize,
    points: &mut PointDataGrid,
    source: &Grid<S>,
    attribute: &str,
    filter: &F,
) -> Result<()> {
    let attribute = if attribute.is_empty() { source.name().to_string() } else { attribute.to_string() };
    if attribute == "P" {
        return Err(Error::Value("cannot sample onto the \"P\" attribute".into()));
    }
    let Some(descriptor) = points.tree().descriptor() else { return Ok(()) };
    match descriptor.position(&attribute) {
        None => append_attribute(
            points.tree_mut(),
            &attribute,
            AttributeType::of::<T>(Codec::Null),
            1,
            true,
            None,
            false,
            false,
        )?,
        Some(pos) => {
            if descriptor.value_type(pos) != T::TYPE_NAME {
                return Err(Error::Type(format!(
                    "requested attribute type {} for sampling does not match existing attribute type {}",
                    T::TYPE_NAME,
                    descriptor.value_type(pos)
                )));
            }
        }
    }
    sample_into::<S, T, F>(order, points, source, &attribute, filter)
}

/// Sample `source` onto attribute `attribute`, using the value type of the existing attribute
/// (any of `vec3s`, `vec3d`, `vec3i`, `int8`..`int64`, `float`, `double`, `bool`) or creating
/// one of the grid's value type (`sampleGrid()` with `DummySampleType`).
pub fn sample_grid_auto<S: SampleSource + SampleTarget, F: IndexFilter>(
    order: usize,
    points: &mut PointDataGrid,
    source: &Grid<S>,
    attribute: &str,
    filter: &F,
) -> Result<()> {
    let name = if attribute.is_empty() { source.name().to_string() } else { attribute.to_string() };
    let Some(descriptor) = points.tree().descriptor() else { return Ok(()) };
    let Some(pos) = descriptor.position(&name) else {
        return sample_grid::<S, S, F>(order, points, source, &name, filter);
    };
    match descriptor.value_type(pos) {
        "vec3s" => sample_grid::<S, Vec3f, F>(order, points, source, &name, filter),
        "vec3d" => sample_grid::<S, Vec3d, F>(order, points, source, &name, filter),
        "vec3i" => sample_grid::<S, Vec3i, F>(order, points, source, &name, filter),
        "int8" => sample_grid::<S, i8, F>(order, points, source, &name, filter),
        "int16" => sample_grid::<S, i16, F>(order, points, source, &name, filter),
        "int32" => sample_grid::<S, i32, F>(order, points, source, &name, filter),
        "int64" => sample_grid::<S, i64, F>(order, points, source, &name, filter),
        "float" => sample_grid::<S, f32, F>(order, points, source, &name, filter),
        "double" => sample_grid::<S, f64, F>(order, points, source, &name, filter),
        "bool" => sample_grid::<S, bool, F>(order, points, source, &name, filter),
        t => Err(Error::Type(format!("cannot sample attribute of type - {}", t))),
    }
}

fn sample_into<S: SampleSource, T: SampleTarget, F: IndexFilter>(
    order: usize,
    points: &mut PointDataGrid,
    source: &Grid<S>,
    attribute: &str,
    filter: &F,
) -> Result<()> {
    let staggered = source.grid_class() == GridClass::Staggered;
    let aligned = source.transform() == points.transform();
    let (source_xform, points_xform) = (source.transform().clone(), points.transform().clone());
    // Fail early on unsupported conversions.
    if T::from_sampled(S::default().to_sampled()).is_none() {
        return Err(Error::Type(format!("cannot sample a {:?} grid on to a {} attribute", S::default(), T::TYPE_NAME)));
    }
    let results: Vec<Result<()>> = points
        .tree_mut()
        .leaves_mut()
        .into_par_iter()
        .map(|mut leaf| {
            let accessor = source.accessor();
            let positions: Vec<(u32, Vec3d)> = {
                let h = AttributeHandle::<Vec3f>::new(leaf.attribute_array("P")?)?;
                leaf.index_on_filtered(filter)
                    .with_coords()
                    .map(|(ijk, i)| {
                        let p = h.get(i);
                        let is = Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64) + ijk.as_vec3d();
                        let is = if aligned { is } else { source_xform.world_to_index(points_xform.index_to_world(is)) };
                        (i, is)
                    })
                    .collect()
            };
            let mut target = AttributeWriteHandle::<T>::new(leaf.attribute_array_mut(attribute)?)?;
            for (i, is) in positions {
                let v = S::sample_at(&accessor, is, order, staggered);
                target.set(i, T::from_sampled(v.to_sampled()).unwrap_or_default());
            }
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Nearest-neighbour sampling of `source` onto attribute `attribute` (`pointSample()`).
pub fn point_sample<S: SampleSource + SampleTarget, F: IndexFilter>(
    points: &mut PointDataGrid,
    source: &Grid<S>,
    attribute: &str,
    filter: &F,
) -> Result<()> {
    sample_grid_auto(0, points, source, attribute, filter)
}

/// Trilinear sampling of `source` onto attribute `attribute` (`boxSample()`).
pub fn box_sample<S: SampleSource + SampleTarget, F: IndexFilter>(
    points: &mut PointDataGrid,
    source: &Grid<S>,
    attribute: &str,
    filter: &F,
) -> Result<()> {
    sample_grid_auto(1, points, source, attribute, filter)
}

/// Triquadratic sampling of `source` onto attribute `attribute` (`quadraticSample()`).
pub fn quadratic_sample<S: SampleSource + SampleTarget, F: IndexFilter>(
    points: &mut PointDataGrid,
    source: &Grid<S>,
    attribute: &str,
    filter: &F,
) -> Result<()> {
    sample_grid_auto(2, points, source, attribute, filter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, CoordBBox, Transform};
    use crate::points::{create_point_data_grid, NullFilter};

    #[test]
    fn sample_linear_field() {
        // f(x, y, z) = x + 2y in index space of a grid with voxel size 0.5.
        let mut field = Grid::<f32>::new(0.0).with_name("f").with_transform(Transform::linear(0.5));
        for c in CoordBBox::from_coords(-4, -4, -4, 24, 24, 24).iter() {
            field.set_value(&c, (c[0] + 2 * c[1]) as f32);
        }
        let pos: Vec<Vec3f> = (0..50).map(|i| Vec3f::new(i as f32 * 0.17, (i % 7) as f32 * 0.3, 1.0)).collect();
        let mut points = create_point_data_grid(&pos, Codec::Null, &Transform::linear(0.25)).unwrap();
        box_sample(&mut points, &field, "", &NullFilter).unwrap();
        point_sample(&mut points, &field, "nearest", &NullFilter).unwrap();
        // Integer target with rounding.
        sample_grid::<f32, i32, _>(1, &mut points, &field, "fi", &NullFilter).unwrap();
        assert!(sample_grid::<f32, Vec3f, _>(1, &mut points, &field, "f", &NullFilter).is_err());
        for leaf in points.tree().leaves() {
            let p = leaf.attribute_handle::<Vec3f>("P").unwrap();
            let f = leaf.attribute_handle::<f32>("f").unwrap();
            let fi = leaf.attribute_handle::<i32>("fi").unwrap();
            for (ijk, i) in leaf.index_on().with_coords() {
                let q = p.get(i);
                let ws = points
                    .transform()
                    .index_to_world(ijk.as_vec3d() + Vec3d::new(q[0] as f64, q[1] as f64, q[2] as f64));
                let is = field.transform().world_to_index(ws);
                let expected = is[0] + 2.0 * is[1];
                assert!((f.get(i) as f64 - expected).abs() < 1e-4);
                assert_eq!(fi.get(i), (expected + 0.5).floor() as i32);
            }
        }
        assert!(sample_grid::<f32, f32, _>(1, &mut points, &field, "P", &NullFilter).is_err());
        let _ = Coord::origin();
    }
}
