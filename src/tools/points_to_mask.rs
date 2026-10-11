//! Activate the voxels of a grid that contain points (port of `openvdb/tools/PointsToMask.h`).
//!
//! OpenVDB's `MaskGrid` (value type `ValueMask`) is represented by a `Grid<bool>` whose
//! active voxels hold `true`.

use super::point_partitioner::PointArray;
use crate::grid::Grid;
use crate::math::{Coord, Transform};
use crate::types::Value;
use rayon::prelude::*;

/// Makes every voxel of a grid that contains at least one point active
/// (port of `PointsToMask`).
pub struct PointsToMask<'a, T: Value> {
    grid: &'a mut Grid<T>,
}

impl<'a, T: Value> PointsToMask<'a, T> {
    pub fn new(grid: &'a mut Grid<T>) -> Self {
        PointsToMask { grid }
    }

    /// Activate the voxels (cell-centered) containing the given world-space points. Voxel
    /// values are left unchanged. `grain_size == 0` disables threading.
    pub fn add_points<P: PointArray + ?Sized>(&mut self, points: &P, grain_size: usize) {
        let xform = self.grid.transform().clone();
        let coords: Vec<Coord> = if grain_size > 0 {
            (0..points.size())
                .into_par_iter()
                .with_min_len(grain_size)
                .map(|i| xform.world_to_index_cell_centered(points.get_pos(i)))
                .collect()
        } else {
            (0..points.size()).map(|i| xform.world_to_index_cell_centered(points.get_pos(i))).collect()
        };
        let mut acc = self.grid.tree_mut().accessor_mut();
        for c in &coords {
            acc.set_active_state(c, true);
        }
    }
}

/// Activate the voxels of `grid` that contain points (`tools::maskPoints`).
pub fn mask_points<P: PointArray + ?Sized, T: Value>(points: &P, grid: &mut Grid<T>) {
    PointsToMask::new(grid).add_points(points, 1024);
}

/// Return a mask grid (with the given transform) whose active voxels contain points
/// (`tools::createPointMask`). Active voxels are set to `true`.
pub fn create_point_mask<P: PointArray + ?Sized>(points: &P, xform: &Transform) -> Grid<bool> {
    let mut grid = Grid::<bool>::new(false).with_transform(xform.clone());
    let coords: Vec<Coord> =
        (0..points.size()).into_par_iter().map(|i| xform.world_to_index_cell_centered(points.get_pos(i))).collect();
    let mut acc = grid.tree_mut().accessor_mut();
    for c in &coords {
        acc.set_value_on(c, true);
    }
    drop(acc);
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3f;

    #[test]
    fn mask() {
        let pts = vec![Vec3f::new(0.1, 0.2, 0.3), Vec3f::new(0.4, 0.0, 0.0), Vec3f::new(10.0, 0.0, 0.0)];
        let g = create_point_mask(&pts, &Transform::linear(1.0));
        assert_eq!(g.active_voxel_count(), 2);
        assert!(g.get_value(&Coord::new(10, 0, 0)));
        let mut f = Grid::<f32>::new(0.0);
        mask_points(&pts, &mut f);
        assert_eq!(f.active_voxel_count(), 2);
    }
}
