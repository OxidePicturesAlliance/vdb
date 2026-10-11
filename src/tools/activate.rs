//! Activate or deactivate values by value (port of `openvdb/tools/Activate.h`).

use crate::tools::value_traits::{ToolValue, TreeAdapter};
use crate::tree::{LeafNode, Tree};

fn matches<T: ToolValue>(v: &T, value: &T, tolerance: &T) -> bool {
    if *tolerance == T::zero() {
        v == value
    } else {
        v.approx_eq(value, tolerance)
    }
}

fn set_state<T: ToolValue>(tree: &mut Tree<T>, value: T, tolerance: T, on: bool, threaded: bool) {
    tree.visit_tiles_mut(&mut |_, v, active, _| {
        if *active != on && matches(v, &value, &tolerance) {
            *active = on;
        }
    });
    let op = |leaf: &mut LeafNode<T>| {
        let (buf, mask) = leaf.buffer_and_mask_mut();
        for n in 0..LeafNode::<T>::SIZE {
            if mask.is_on(n) != on && matches(&buf[n], &value, &tolerance) {
                mask.set(n, on);
            }
        }
    };
    if threaded {
        tree.par_for_each_leaf_mut(op);
    } else {
        tree.leaves_mut().for_each(op);
    }
}

/// Mark as active every inactive value (voxel or tile) equal to `value`, or within
/// `tolerance` of it if the tolerance is non-zero (`tools::activate`).
pub fn activate<T: ToolValue, G: TreeAdapter<T>>(grid: &mut G, value: T, tolerance: T, threaded: bool) {
    set_state(grid.tree_mut_ref(), value, tolerance, true, threaded);
}

/// Mark as inactive every active value (voxel or tile) equal to `value`, or within
/// `tolerance` of it if the tolerance is non-zero (`tools::deactivate`).
pub fn deactivate<T: ToolValue, G: TreeAdapter<T>>(grid: &mut G, value: T, tolerance: T, threaded: bool) {
    set_state(grid.tree_mut_ref(), value, tolerance, false, threaded);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, CoordBBox};
    use crate::tree::FloatTree;

    #[test]
    fn activate_deactivate() {
        let mut tree = FloatTree::new(0.0);
        tree.set_value_off(&Coord::new(0, 0, 0), 1.0);
        tree.set_value_off(&Coord::new(0, 0, 1), 1.001);
        tree.fill(&CoordBBox::from_coords(128, 0, 0, 255, 127, 127), 1.0, false);
        activate(&mut tree, 1.0, 0.0, true);
        assert_eq!(tree.active_voxel_count(), 1 + 128 * 128 * 128);
        activate(&mut tree, 1.0, 0.01, false);
        assert_eq!(tree.active_voxel_count(), 2 + 128 * 128 * 128);
        deactivate(&mut tree, 1.0, 0.0, true);
        assert_eq!(tree.active_voxel_count(), 1);
    }
}
