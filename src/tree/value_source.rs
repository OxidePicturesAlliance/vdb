//! A minimal read-only interface over anything that can be queried for voxel values.
//!
//! OpenVDB's tools are templated on "tree or accessor" types that provide `getValue()` and
//! `probeValue()`. [`ValueSource`] captures that duck-typed interface so samplers, finite
//! difference operators and stencils can work on trees, grids and cached accessors alike.

use super::accessor::{ValueAccessor, ValueAccessorMut};
use super::node::Node;
use super::tree::TreeBase;
use crate::grid::Grid;
use crate::math::Coord;
use crate::types::Value;

/// Random read access to voxel values.
pub trait ValueSource {
    /// The stored value type.
    type Value: Value;
    /// The value at `xyz`.
    fn get_value(&self, xyz: &Coord) -> Self::Value;
    /// The value at `xyz` and whether it is active.
    fn probe_value(&self, xyz: &Coord) -> (Self::Value, bool);
    /// True if the value at `xyz` is active.
    fn is_value_on(&self, xyz: &Coord) -> bool {
        self.probe_value(xyz).1
    }
}

impl<C: Node> ValueSource for TreeBase<C> {
    type Value = C::Value;
    #[inline]
    fn get_value(&self, xyz: &Coord) -> C::Value {
        TreeBase::get_value(self, xyz)
    }
    #[inline]
    fn probe_value(&self, xyz: &Coord) -> (C::Value, bool) {
        TreeBase::probe_value(self, xyz)
    }
    #[inline]
    fn is_value_on(&self, xyz: &Coord) -> bool {
        TreeBase::is_value_on(self, xyz)
    }
}

impl<C: Node> ValueSource for ValueAccessor<'_, C> {
    type Value = C::Value;
    #[inline]
    fn get_value(&self, xyz: &Coord) -> C::Value {
        ValueAccessor::get_value(self, xyz)
    }
    #[inline]
    fn probe_value(&self, xyz: &Coord) -> (C::Value, bool) {
        ValueAccessor::probe_value(self, xyz)
    }
    #[inline]
    fn is_value_on(&self, xyz: &Coord) -> bool {
        ValueAccessor::is_value_on(self, xyz)
    }
}

impl<C: Node> ValueSource for ValueAccessorMut<'_, C> {
    type Value = C::Value;
    #[inline]
    fn get_value(&self, xyz: &Coord) -> C::Value {
        ValueAccessorMut::get_value(self, xyz)
    }
    #[inline]
    fn probe_value(&self, xyz: &Coord) -> (C::Value, bool) {
        ValueAccessorMut::probe_value(self, xyz)
    }
    #[inline]
    fn is_value_on(&self, xyz: &Coord) -> bool {
        ValueAccessorMut::is_value_on(self, xyz)
    }
}

impl<T: Value> ValueSource for Grid<T> {
    type Value = T;
    #[inline]
    fn get_value(&self, xyz: &Coord) -> T {
        self.tree().get_value(xyz)
    }
    #[inline]
    fn probe_value(&self, xyz: &Coord) -> (T, bool) {
        self.tree().probe_value(xyz)
    }
}

impl<S: ValueSource + ?Sized> ValueSource for &S {
    type Value = S::Value;
    #[inline]
    fn get_value(&self, xyz: &Coord) -> S::Value {
        (**self).get_value(xyz)
    }
    #[inline]
    fn probe_value(&self, xyz: &Coord) -> (S::Value, bool) {
        (**self).probe_value(xyz)
    }
    #[inline]
    fn is_value_on(&self, xyz: &Coord) -> bool {
        (**self).is_value_on(xyz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::FloatTree;

    fn read<S: ValueSource<Value = f32>>(s: &S, c: Coord) -> (f32, bool) {
        s.probe_value(&c)
    }

    #[test]
    fn sources_agree() {
        let mut tree = FloatTree::new(2.0);
        tree.set_value(&Coord::new(1, 2, 3), 5.0);
        let c = Coord::new(1, 2, 3);
        assert_eq!(read(&tree, c), (5.0, true));
        assert_eq!(read(&tree.accessor(), c), (5.0, true));
        assert_eq!(read(&&tree, Coord::new(0, 0, 0)), (2.0, false));
        let grid = Grid::from_tree(tree);
        assert_eq!(read(&grid, c), (5.0, true));
    }
}
