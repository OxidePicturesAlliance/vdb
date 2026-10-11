//! Statistics of point attributes (port of `openvdb/points/PointStatistics.h`).
//!
//! As in OpenVDB, with a filter whose state is [`IndexState::All`] every point of every
//! leaf contributes; otherwise only accepted points in active voxels do. Vector values are
//! compared component-wise.
//!
//! Unlike OpenVDB 10, a uniform array contributes one value per point to
//! [`accumulate`] and its uniform value to [`eval_average`] (OpenVDB counts a uniform array's
//! value once in `accumulate` and divides it by the point count in `evalAverage`).

use super::attribute_array::AttributeHandle;
use super::codec::AttributeValue;
use super::index_filter::IndexFilter;
use super::index_iterator::IndexState;
use super::point_data_grid::{PointDataLeaf, PointDataTree, PointLeaf};
use crate::math::{Vec3, Vec3d, Vec3f, Vec3i};
use crate::tree::Tree;
use crate::types::Value;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// Attribute value types supported by the statistics functions.
pub trait StatValue: AttributeValue + Value {
    /// The averaging type (`ConvertElementType<T, double>`).
    type Avg: Value
        + Add<Output = Self::Avg>
        + Sub<Output = Self::Avg>
        + Mul<f64, Output = Self::Avg>
        + Div<f64, Output = Self::Avg>;
    /// The accumulation type (`PromoteType<T>::Highest`).
    type Sum: Value + AddAssign + Mul<Self::SumScalar, Output = Self::Sum>;
    /// Scalar of the accumulation type.
    type SumScalar: Copy;
    /// Convert to the averaging type.
    fn to_avg(self) -> Self::Avg;
    /// Convert to the accumulation type.
    fn to_sum(self) -> Self::Sum;
    /// A count as a scalar of the accumulation type.
    fn count_scalar(n: u64) -> Self::SumScalar;
    /// Component-wise minimum.
    fn min_value(self, o: Self) -> Self;
    /// Component-wise maximum.
    fn max_value(self, o: Self) -> Self;
}

macro_rules! impl_stat_scalar {
    ($($t:ty => $sum:ty),*) => {$(
        impl StatValue for $t {
            type Avg = f64;
            type Sum = $sum;
            type SumScalar = $sum;
            fn to_avg(self) -> f64 { self as f64 }
            fn to_sum(self) -> $sum { self as $sum }
            fn count_scalar(n: u64) -> $sum { n as $sum }
            fn min_value(self, o: Self) -> Self { if o < self { o } else { self } }
            fn max_value(self, o: Self) -> Self { if o > self { o } else { self } }
        }
    )*};
}
impl_stat_scalar!(f32 => f64, f64 => f64, i32 => i64, i64 => i64);

macro_rules! impl_stat_vec {
    ($($t:ty, $s:ty => $sum:ty),*) => {$(
        impl StatValue for $t {
            type Avg = Vec3d;
            type Sum = Vec3<$sum>;
            type SumScalar = $sum;
            fn to_avg(self) -> Vec3d { self.map(|c| c as f64) }
            fn to_sum(self) -> Vec3<$sum> { self.map(|c| c as $sum) }
            fn count_scalar(n: u64) -> $sum { n as $sum }
            fn min_value(self, o: Self) -> Self { self.zip_map(o, |a, b| if b < a { b } else { a }) }
            fn max_value(self, o: Self) -> Self { self.zip_map(o, |a, b| if b > a { b } else { a }) }
        }
    )*};
}
impl_stat_vec!(Vec3f, f32 => f64, Vec3d, f64 => f64, Vec3i, i32 => i64);

/// The values of the attribute at `pos` contributing to the statistics in a leaf, visited
/// by `f`; returns false if the leaf has no such attribute.
fn visit_values<T: StatValue, F: IndexFilter>(
    leaf: &PointDataLeaf<'_>,
    pos: usize,
    filter: &F,
    mut f: impl FnMut(T, u64),
) -> bool {
    let Some(array) = leaf.attributes.get(pos) else { return false };
    let Ok(h) = AttributeHandle::<T>::new(array) else { return false };
    if leaf.point_count() == 0 {
        return true;
    }
    if filter.state() == IndexState::All {
        if h.is_uniform() {
            f(h.get(0), h.size() as u64);
        } else {
            for i in 0..h.size() {
                f(h.get(i), 1);
            }
        }
    } else {
        for i in leaf.index_on_filtered(filter) {
            f(h.get(i), 1);
        }
    }
    true
}

fn attribute_position(tree: &PointDataTree, attribute: &str) -> Option<usize> {
    tree.descriptor()?.position(attribute)
}

/// Per-leaf extents of an attribute.
fn leaf_extents<T: StatValue, F: IndexFilter>(tree: &PointDataTree, pos: usize, filter: &F) -> Vec<Option<(T, T)>> {
    tree.leaf_vec()
        .par_iter()
        .map(|leaf| {
            let mut ext: Option<(T, T)> = None;
            visit_values::<T, F>(leaf, pos, filter, |v, _| {
                ext = Some(match ext {
                    None => (v, v),
                    Some((a, b)) => (a.min_value(v), b.max_value(v)),
                });
            });
            ext
        })
        .collect()
}

/// Minimum and maximum (component-wise for vectors) of `attribute` over the points
/// accepted by `filter` (`evalMinMax()`); `None` if there are no such points.
pub fn eval_min_max<T: StatValue, F: IndexFilter>(tree: &PointDataTree, attribute: &str, filter: &F) -> Option<(T, T)> {
    let pos = attribute_position(tree, attribute)?;
    combine_extents(&leaf_extents::<T, F>(tree, pos, filter))
}

fn combine_extents<T: StatValue>(exts: &[Option<(T, T)>]) -> Option<(T, T)> {
    exts.iter().flatten().fold(None, |acc, &(a, b)| match acc {
        None => Some((a, b)),
        Some((x, y)) => Some((x.min_value(a), y.max_value(b))),
    })
}

/// Like [`eval_min_max`], also returning trees with each leaf's minimum and maximum as
/// active leaf-sized tiles.
pub fn eval_min_max_trees<T: StatValue, F: IndexFilter>(
    tree: &PointDataTree,
    attribute: &str,
    filter: &F,
) -> Option<((T, T), Tree<T>, Tree<T>)> {
    let pos = attribute_position(tree, attribute)?;
    let exts = leaf_extents::<T, F>(tree, pos, filter);
    let result = combine_extents(&exts)?;
    let mut min_tree = Tree::new(T::default());
    let mut max_tree = Tree::new(T::default());
    for (leaf, e) in tree.leaves().zip(&exts) {
        if let Some((a, b)) = e {
            min_tree.add_tile(1, &leaf.origin(), *a, true);
            max_tree.add_tile(1, &leaf.origin(), *b, true);
        }
    }
    Some((result, min_tree, max_tree))
}

/// Running average (OpenVDB's `Sample`).
#[derive(Clone, Copy)]
struct Sample<A> {
    avg: A,
    size: u64,
}

impl<A: Copy + Add<Output = A> + Sub<Output = A> + Mul<f64, Output = A> + Div<f64, Output = A>> Sample<A> {
    fn add(&mut self, v: A) {
        self.size += 1;
        let delta = v - self.avg;
        self.avg = self.avg + delta / self.size as f64;
    }
    fn merge(&mut self, o: &Sample<A>) {
        let denom = 1.0 / (self.size + o.size) as f64;
        let delta = o.avg - self.avg;
        self.avg = self.avg + delta * (denom * o.size as f64);
        self.size += o.size;
    }
}

fn leaf_averages<T: StatValue, F: IndexFilter>(
    tree: &PointDataTree,
    pos: usize,
    filter: &F,
) -> Vec<Option<Sample<T::Avg>>> {
    tree.leaf_vec()
        .par_iter()
        .map(|leaf| {
            let mut s: Option<Sample<T::Avg>> = None;
            visit_values::<T, F>(leaf, pos, filter, |v, n| match &mut s {
                None => s = Some(Sample { avg: v.to_avg(), size: n }),
                Some(s) => {
                    if n == 1 {
                        s.add(v.to_avg())
                    } else {
                        s.merge(&Sample { avg: v.to_avg(), size: n })
                    }
                }
            });
            s
        })
        .collect()
}

fn combine_averages<A>(samples: &[Option<Sample<A>>]) -> Option<A>
where
    A: Copy + Add<Output = A> + Sub<Output = A> + Mul<f64, Output = A> + Div<f64, Output = A>,
{
    let mut it = samples.iter().flatten();
    let mut s = *it.next()?;
    for o in it {
        s.merge(o);
    }
    Some(s.avg)
}

/// Average of `attribute` (in double precision) over the points accepted by `filter`
/// (`evalAverage()`); `None` if there are no such points.
pub fn eval_average<T: StatValue, F: IndexFilter>(tree: &PointDataTree, attribute: &str, filter: &F) -> Option<T::Avg> {
    let pos = attribute_position(tree, attribute)?;
    combine_averages(&leaf_averages::<T, F>(tree, pos, filter))
}

/// Like [`eval_average`], also returning a tree with each leaf's average as an active
/// leaf-sized tile.
pub fn eval_average_tree<T: StatValue, F: IndexFilter>(
    tree: &PointDataTree,
    attribute: &str,
    filter: &F,
) -> Option<(T::Avg, Tree<T::Avg>)> {
    let pos = attribute_position(tree, attribute)?;
    let samples = leaf_averages::<T, F>(tree, pos, filter);
    let avg = combine_averages(&samples)?;
    let mut out = Tree::new(<T::Avg as Default>::default());
    for (leaf, s) in tree.leaves().zip(&samples) {
        if let Some(s) = s {
            out.add_tile(1, &leaf.origin(), s.avg, true);
        }
    }
    Some((avg, out))
}

fn leaf_sums<T: StatValue, F: IndexFilter>(tree: &PointDataTree, pos: usize, filter: &F) -> Vec<Option<T::Sum>> {
    tree.leaf_vec()
        .par_iter()
        .map(|leaf| {
            let mut total: Option<T::Sum> = None;
            visit_values::<T, F>(leaf, pos, filter, |v, n| {
                let x = v.to_sum() * T::count_scalar(n);
                match &mut total {
                    None => total = Some(x),
                    Some(t) => *t += x,
                }
            });
            total
        })
        .collect()
}

/// Sum of `attribute` (in the widest type: `f64`, `i64` or their vectors) over the points
/// accepted by `filter` (`accumulate()`); `None` if there are no such points.
pub fn accumulate<T: StatValue, F: IndexFilter>(tree: &PointDataTree, attribute: &str, filter: &F) -> Option<T::Sum> {
    let pos = attribute_position(tree, attribute)?;
    let sums = leaf_sums::<T, F>(tree, pos, filter);
    let mut it = sums.into_iter().flatten();
    let mut total = it.next()?;
    for s in it {
        total += s;
    }
    Some(total)
}

/// Like [`accumulate`], also returning a tree with each leaf's sum as an active leaf-sized
/// tile.
pub fn accumulate_tree<T: StatValue, F: IndexFilter>(
    tree: &PointDataTree,
    attribute: &str,
    filter: &F,
) -> Option<(T::Sum, Tree<T::Sum>)> {
    let pos = attribute_position(tree, attribute)?;
    let sums = leaf_sums::<T, F>(tree, pos, filter);
    let mut out = Tree::new(<T::Sum as Default>::default());
    let mut total: Option<T::Sum> = None;
    for (leaf, s) in tree.leaves().zip(&sums) {
        if let Some(s) = s {
            out.add_tile(1, &leaf.origin(), *s, true);
            match &mut total {
                None => total = Some(*s),
                Some(t) => *t += *s,
            }
        }
    }
    Some((total?, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Transform;
    use crate::points::{
        append_typed_attribute, create_point_data_grid_from_index, populate_attribute, Codec, NullFilter,
    };
    use crate::tools::point_index_grid::create_point_index_grid;

    #[test]
    fn stats() {
        let pos: Vec<Vec3f> = (0..100).map(|i| Vec3f::new(i as f32 * 0.7, (i % 3) as f32 * 9.0, 0.0)).collect();
        let t = Transform::linear(1.0);
        let index = create_point_index_grid(&pos, &t);
        let mut grid = create_point_data_grid_from_index(&index, &pos, Codec::Null, &t, None).unwrap();
        let tree = grid.tree_mut();
        let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
        append_typed_attribute(tree, "f", Codec::Null, 0.0f32, 1, None, false, false).unwrap();
        populate_attribute(tree, index.tree(), "f", &vals, 1).unwrap();
        append_typed_attribute(tree, "u", Codec::Null, 2i32, 1, None, false, false).unwrap();
        let vecs: Vec<Vec3f> = (0..100).map(|i| Vec3f::new(i as f32, -(i as f32), 1.0)).collect();
        append_typed_attribute(tree, "v", Codec::Null, Vec3f::zero(), 1, None, false, false).unwrap();
        populate_attribute(tree, index.tree(), "v", &vecs, 1).unwrap();

        assert_eq!(eval_min_max::<f32, _>(tree, "f", &NullFilter), Some((0.0, 99.0)));
        assert!((eval_average::<f32, _>(tree, "f", &NullFilter).unwrap() - 49.5).abs() < 1e-9);
        assert_eq!(accumulate::<f32, _>(tree, "f", &NullFilter), Some(4950.0));
        assert_eq!(accumulate::<i32, _>(tree, "u", &NullFilter), Some(200));
        assert_eq!(eval_average::<i32, _>(tree, "u", &NullFilter), Some(2.0));
        let (mn, mx) = eval_min_max::<Vec3f, _>(tree, "v", &NullFilter).unwrap();
        assert_eq!(mn, Vec3f::new(0.0, -99.0, 1.0));
        assert_eq!(mx, Vec3f::new(99.0, 0.0, 1.0));
        assert_eq!(accumulate::<Vec3f, _>(tree, "v", &NullFilter), Some(Vec3d::new(4950.0, -4950.0, 100.0)));
        assert!(eval_min_max::<f32, _>(tree, "missing", &NullFilter).is_none());
        let (_, min_tree, _) = eval_min_max_trees::<f32, _>(tree, "f", &NullFilter).unwrap();
        assert_eq!(min_tree.active_tile_count(), tree.leaf_count() as u64);
        let (_, sum_tree) = accumulate_tree::<i32, _>(tree, "u", &NullFilter).unwrap();
        assert_eq!(sum_tree.active_tile_count(), tree.leaf_count() as u64);
        let (_, avg_tree) = eval_average_tree::<f32, _>(tree, "f", &NullFilter).unwrap();
        assert_eq!(avg_tree.active_tile_count(), tree.leaf_count() as u64);
        let _ = Vec3i::zero();
    }
}
