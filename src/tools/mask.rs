//! Boolean masks of grid interiors (port of `openvdb/tools/Mask.h`).

use crate::grid::{Grid, GridClass};
use crate::math::{Vec3d, Vec3f, Vec3i};
use crate::tools::level_set_util::sdf_interior_mask;
use crate::tools::tree_ops::topology_union;
use crate::types::Value;

/// Value types accepted by [`interior_mask`].
pub trait InteriorMaskValue: Value {
    /// The mask of a grid of this value type.
    fn interior_mask(grid: &Grid<Self>, isovalue: f64) -> Grid<bool>;
}

/// The mask of the active topology of any grid.
fn topology_mask<T: Value>(grid: &Grid<T>) -> Grid<bool> {
    let mut mask = Grid::<bool>::new(false).with_transform(grid.transform().clone());
    topology_union(mask.tree_mut(), grid.tree(), false);
    mask
}

macro_rules! impl_interior_real {
    ($($t:ty),*) => {$(
        impl InteriorMaskValue for $t {
            fn interior_mask(grid: &Grid<Self>, isovalue: f64) -> Grid<bool> {
                if grid.grid_class() == GridClass::LevelSet {
                    sdf_interior_mask(grid, isovalue as $t)
                } else {
                    topology_mask(grid)
                }
            }
        }
    )*};
}
impl_interior_real!(f32, f64);

macro_rules! impl_interior_topology {
    ($($t:ty),*) => {$(
        impl InteriorMaskValue for $t {
            fn interior_mask(grid: &Grid<Self>, _isovalue: f64) -> Grid<bool> {
                topology_mask(grid)
            }
        }
    )*};
}
impl_interior_topology!(i8, i16, i32, i64, u8, u16, u32, u64, Vec3f, Vec3d, Vec3i);

impl InteriorMaskValue for bool {
    fn interior_mask(grid: &Grid<bool>, _isovalue: f64) -> Grid<bool> {
        grid.clone()
    }
}

/// Return a boolean grid whose active topology matches the active topology of `grid` or,
/// for a floating-point level set, its interior (values below `isovalue`)
/// (`tools::interiorMask`). A boolean input grid is simply copied.
pub fn interior_mask<T: InteriorMaskValue>(grid: &Grid<T>, isovalue: f64) -> Grid<bool> {
    T::interior_mask(grid, isovalue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Coord, Vec3f};
    use crate::tools::level_set_sphere::create_level_set_sphere;

    #[test]
    fn masks() {
        let ls = create_level_set_sphere::<f32>(1.0, Vec3f::splat(0.0), 0.1, 3.0, true).unwrap();
        let m = interior_mask(&ls, 0.0);
        assert!(m.tree().is_value_on(&Coord::new(0, 0, 0)));
        assert!(!m.tree().is_value_on(&Coord::new(11, 0, 0)));
        let mut v = Grid::<i32>::new(0);
        v.set_value(&Coord::new(1, 2, 3), 5);
        let mv = interior_mask(&v, 0.0);
        assert_eq!(mv.active_voxel_count(), 1);
    }
}
