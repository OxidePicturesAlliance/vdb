//! Value-type helpers shared by the tools (ports of the generic `math::` helpers that the
//! OpenVDB tools rely on: `math::negative`, `math::isNegative`, `math::isApproxEqual`,
//! `math::cwiseLessThan`, ...), plus the error type returned by fallible tools.

use crate::grid::Grid;
use crate::math::math_util::is_rel_or_approx_equal;
use crate::math::Vec3;
use crate::tree::Tree;
use crate::types::Value;
use std::cmp::Ordering;

/// Errors raised by the tools (OpenVDB throws `ValueError`, `TypeError` or `RuntimeError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolError {
    /// An argument has an invalid value (OpenVDB's `ValueError`).
    #[error("value error: {0}")]
    Value(String),
    /// The operation is not supported for the value type (OpenVDB's `TypeError`).
    #[error("type error: {0}")]
    Type(String),
    /// The operation failed at run time (OpenVDB's `RuntimeError`).
    #[error("runtime error: {0}")]
    Runtime(String),
}

/// Result type used by fallible tools.
pub type ToolResult<T> = Result<T, ToolError>;

/// Uniform access to the tree of a [`Grid`] or a bare [`Tree`] (OpenVDB's `TreeAdapter`),
/// so that tools can accept either.
pub trait TreeAdapter<T: Value> {
    /// The same kind of container with a different value type (`ValueConverter<U>::Type`).
    type Rebind<U: Value>: TreeAdapter<U>;
    /// The underlying tree.
    fn tree_ref(&self) -> &Tree<T>;
    /// The underlying tree, mutably.
    fn tree_mut_ref(&mut self) -> &mut Tree<T>;
    /// Wrap `tree` like `self`: for grids, the result has `self`'s transform and metadata.
    fn rebind<U: Value>(&self, tree: Tree<U>) -> Self::Rebind<U>;
    /// Wrap `tree` like `self`, copying only the transform (not the metadata) of grids.
    fn rebind_transform<U: Value>(&self, tree: Tree<U>) -> Self::Rebind<U>;
}

impl<T: Value> TreeAdapter<T> for Tree<T> {
    type Rebind<U: Value> = Tree<U>;
    fn tree_ref(&self) -> &Tree<T> {
        self
    }
    fn tree_mut_ref(&mut self) -> &mut Tree<T> {
        self
    }
    fn rebind<U: Value>(&self, tree: Tree<U>) -> Tree<U> {
        tree
    }
    fn rebind_transform<U: Value>(&self, tree: Tree<U>) -> Tree<U> {
        tree
    }
}

impl<T: Value> TreeAdapter<T> for Grid<T> {
    type Rebind<U: Value> = Grid<U>;
    fn tree_ref(&self) -> &Tree<T> {
        self.tree()
    }
    fn tree_mut_ref(&mut self) -> &mut Tree<T> {
        self.tree_mut()
    }
    fn rebind<U: Value>(&self, tree: Tree<U>) -> Grid<U> {
        self.copy_replacing_tree(tree)
    }
    fn rebind_transform<U: Value>(&self, tree: Tree<U>) -> Grid<U> {
        Grid::from_tree(tree).with_transform(self.transform().clone())
    }
}

/// Extra operations on tree values needed by the tools.
///
/// Comparisons follow OpenVDB's `math::cwiseLessThan`/`cwiseGreaterThan`, which for
/// vectors compare components lexicographically.
pub trait ToolValue: Value {
    /// OpenVDB's `math::negative` (logical not for `bool`).
    fn negative(self) -> Self;
    /// `math::cwiseLessThan(self, other)`.
    fn cwise_lt(&self, other: &Self) -> bool;
    /// `math::cwiseGreaterThan(self, other)`.
    fn cwise_gt(&self, other: &Self) -> bool;
    /// `math::isNegative(self)` (always false for `bool`).
    fn is_negative(&self) -> bool {
        self.cwise_lt(&Self::zero())
    }
    /// `math::isApproxEqual(self, other)` with the default tolerance
    /// (`math::Tolerance<T>` for scalars, `Vec3::eq` for vectors).
    fn is_approx_equal(&self, other: &Self) -> bool;
    /// `self - other` (inequality for `bool`), as used by tolerance tests.
    fn difference(self, other: Self) -> Self;
    /// `math::Abs` (identity for unsigned and `bool`).
    fn abs_value(self) -> Self;
    /// A total order consistent with `cwise_lt` (used for medians and sorting).
    fn total_order(&self, other: &Self) -> Ordering;
    /// The scalar used by statistics: the value itself, or the length of a vector.
    fn stat_value(&self) -> f64;
    /// True for `bool`, whose nodes are pruned without computing a median.
    const IS_BOOL: bool = false;

    /// The median of `values` as computed by `std::nth_element` with midpoint
    /// `(len - 1) / 2` (the slice is reordered).
    fn median(values: &mut [Self]) -> Self {
        let mid = (values.len() - 1) >> 1;
        values.select_nth_unstable_by(mid, |a, b| a.total_order(b));
        values[mid]
    }
}

macro_rules! impl_tool_float {
    ($($t:ty => $tol:expr),*) => {$(
        impl ToolValue for $t {
            #[inline] fn negative(self) -> Self { -self }
            #[inline] fn cwise_lt(&self, o: &Self) -> bool { self < o }
            #[inline] fn cwise_gt(&self, o: &Self) -> bool { self > o }
            #[inline] fn is_approx_equal(&self, o: &Self) -> bool { !((self - o).abs() > $tol) }
            #[inline] fn difference(self, o: Self) -> Self { self - o }
            #[inline] fn abs_value(self) -> Self { self.abs() }
            #[inline] fn total_order(&self, o: &Self) -> Ordering { self.total_cmp(o) }
            #[inline] fn stat_value(&self) -> f64 { *self as f64 }
        }
    )*};
}
impl_tool_float!(f32 => crate::math::TOLERANCE_F32, f64 => crate::math::TOLERANCE_F64);

macro_rules! impl_tool_signed {
    ($($t:ty),*) => {$(
        impl ToolValue for $t {
            #[inline] fn negative(self) -> Self { self.wrapping_neg() }
            #[inline] fn cwise_lt(&self, o: &Self) -> bool { self < o }
            #[inline] fn cwise_gt(&self, o: &Self) -> bool { self > o }
            #[inline] fn is_approx_equal(&self, o: &Self) -> bool { self == o }
            #[inline] fn difference(self, o: Self) -> Self { self.wrapping_sub(o) }
            #[inline] fn abs_value(self) -> Self { self.wrapping_abs() }
            #[inline] fn total_order(&self, o: &Self) -> Ordering { self.cmp(o) }
            #[inline] fn stat_value(&self) -> f64 { *self as f64 }
        }
    )*};
}
impl_tool_signed!(i8, i16, i32, i64);

macro_rules! impl_tool_unsigned {
    ($($t:ty),*) => {$(
        impl ToolValue for $t {
            #[inline] fn negative(self) -> Self { self.wrapping_neg() }
            #[inline] fn cwise_lt(&self, o: &Self) -> bool { self < o }
            #[inline] fn cwise_gt(&self, o: &Self) -> bool { self > o }
            #[inline] fn is_approx_equal(&self, o: &Self) -> bool { self == o }
            #[inline] fn difference(self, o: Self) -> Self { self.wrapping_sub(o) }
            #[inline] fn abs_value(self) -> Self { self }
            #[inline] fn total_order(&self, o: &Self) -> Ordering { self.cmp(o) }
            #[inline] fn stat_value(&self) -> f64 { *self as f64 }
        }
    )*};
}
impl_tool_unsigned!(u8, u16, u32, u64);

impl ToolValue for bool {
    #[inline]
    fn negative(self) -> Self {
        !self
    }
    #[inline]
    fn cwise_lt(&self, o: &Self) -> bool {
        !*self & *o
    }
    #[inline]
    fn cwise_gt(&self, o: &Self) -> bool {
        *self & !*o
    }
    #[inline]
    fn is_negative(&self) -> bool {
        false
    }
    #[inline]
    fn is_approx_equal(&self, o: &Self) -> bool {
        self == o
    }
    #[inline]
    fn difference(self, o: Self) -> Self {
        self != o
    }
    #[inline]
    fn abs_value(self) -> Self {
        self
    }
    #[inline]
    fn total_order(&self, o: &Self) -> Ordering {
        self.cmp(o)
    }
    #[inline]
    fn stat_value(&self) -> f64 {
        *self as u8 as f64
    }
    const IS_BOOL: bool = true;
    fn median(values: &mut [Self]) -> Self {
        values[0]
    }
}

/// Lexicographic comparison of vector components (`math::cwiseLessThan` for `Vec3`).
#[inline]
fn lex_lt<T: PartialOrd + Copy>(a: &Vec3<T>, b: &Vec3<T>) -> bool {
    for i in 0..2 {
        if a.0[i] != b.0[i] {
            return a.0[i] < b.0[i];
        }
    }
    a.0[2] < b.0[2]
}
#[inline]
fn lex_gt<T: PartialOrd + Copy>(a: &Vec3<T>, b: &Vec3<T>) -> bool {
    for i in 0..2 {
        if a.0[i] != b.0[i] {
            return a.0[i] > b.0[i];
        }
    }
    a.0[2] > b.0[2]
}

macro_rules! impl_tool_vec_float {
    ($($t:ty),*) => {$(
        impl ToolValue for Vec3<$t> {
            #[inline] fn negative(self) -> Self { -self }
            #[inline] fn cwise_lt(&self, o: &Self) -> bool { lex_lt(self, o) }
            #[inline] fn cwise_gt(&self, o: &Self) -> bool { lex_gt(self, o) }
            #[inline]
            fn is_approx_equal(&self, o: &Self) -> bool {
                (0..3).all(|i| is_rel_or_approx_equal(self.0[i] as f64, o.0[i] as f64, 1.0e-7, 1.0e-7))
            }
            #[inline] fn difference(self, o: Self) -> Self { self - o }
            #[inline] fn abs_value(self) -> Self { self.map(|v| v.abs()) }
            #[inline]
            fn total_order(&self, o: &Self) -> Ordering {
                for i in 0..3 {
                    match self.0[i].total_cmp(&o.0[i]) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                Ordering::Equal
            }
            #[inline] fn stat_value(&self) -> f64 { (self.0[0] as f64).hypot(self.0[1] as f64).hypot(self.0[2] as f64) }
        }
    )*};
}
impl_tool_vec_float!(f32, f64);

impl ToolValue for Vec3<i32> {
    #[inline]
    fn negative(self) -> Self {
        self.map(|v| v.wrapping_neg())
    }
    #[inline]
    fn cwise_lt(&self, o: &Self) -> bool {
        lex_lt(self, o)
    }
    #[inline]
    fn cwise_gt(&self, o: &Self) -> bool {
        lex_gt(self, o)
    }
    #[inline]
    fn is_approx_equal(&self, o: &Self) -> bool {
        self == o
    }
    #[inline]
    fn difference(self, o: Self) -> Self {
        self.zip_map(o, |a, b| a.wrapping_sub(b))
    }
    #[inline]
    fn abs_value(self) -> Self {
        self.map(|v| v.wrapping_abs())
    }
    #[inline]
    fn total_order(&self, o: &Self) -> Ordering {
        self.0.cmp(&o.0)
    }
    #[inline]
    fn stat_value(&self) -> f64 {
        let s: f64 = self.0.iter().map(|&v| (v as f64) * (v as f64)).sum();
        s.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3f;

    #[test]
    fn median_and_compare() {
        let mut v = vec![5.0f32, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(f32::median(&mut v), 3.0);
        let mut w = vec![4, 1, 3, 2];
        assert_eq!(i32::median(&mut w), 2);
        assert!(Vec3f::new(0.0, 1.0, 5.0).cwise_lt(&Vec3f::new(0.0, 2.0, 0.0)));
        assert!(!true.is_negative());
        assert!((-1.0f64).is_negative());
        assert!(1.0f32.is_approx_equal(&(1.0 + 1e-9)));
        assert!(!true.negative());
    }
}
