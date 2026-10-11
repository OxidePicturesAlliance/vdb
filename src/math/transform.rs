//! Index-space to world-space transforms (port of `openvdb/math/Transform.h`).
//!
//! This module currently models every transform as a general affine map; the specialised
//! map types of OpenVDB (scale, translation, frustum, ...) live in [`super::maps`].

use super::maps::MapBase;
use super::mat::{Axis, Mat4d};
use super::vec3::Vec3d;
use super::{Coord, CoordBBox};
use std::fmt;

/// Mapping between index space (voxel coordinates) and world space.
#[derive(Clone, PartialEq)]
pub struct Transform {
    map: MapBase,
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transform {
    /// Wrap a map.
    pub fn from_map(map: MapBase) -> Self {
        Transform { map }
    }
    /// The identity transform (voxel size 1, no offset).
    pub fn identity() -> Self {
        Self::linear(1.0)
    }
    /// A uniform-scale linear transform with the given voxel size.
    pub fn linear(voxel_size: f64) -> Self {
        Transform { map: MapBase::uniform_scale(voxel_size) }
    }
    /// A linear transform from an affine matrix (row-vector convention).
    pub fn from_matrix(m: Mat4d) -> Self {
        Transform { map: MapBase::from_matrix(m) }
    }
    pub fn map(&self) -> &MapBase {
        &self.map
    }
    pub fn map_mut(&mut self) -> &mut MapBase {
        &mut self.map
    }
    /// True if the transform is linear (affine).
    pub fn is_linear(&self) -> bool {
        self.map.is_linear()
    }
    /// True if voxels are cubes and the map has no shear.
    pub fn has_uniform_scale(&self) -> bool {
        self.map.has_uniform_scale()
    }
    /// True if the map is the identity.
    pub fn is_identity(&self) -> bool {
        self.map.is_identity()
    }
    /// The OpenVDB type name of the underlying map (e.g. `"UniformScaleMap"`).
    pub fn map_type(&self) -> &'static str {
        self.map.type_name()
    }

    /// World-space size of a voxel at the origin.
    pub fn voxel_size(&self) -> Vec3d {
        self.map.voxel_size()
    }
    /// World-space size of the voxel at the given world position (differs from
    /// [`voxel_size`](Self::voxel_size) only for non-linear maps).
    pub fn voxel_size_at(&self, xyz: Vec3d) -> Vec3d {
        self.map.voxel_size_at(xyz)
    }
    /// World-space volume of a voxel at the origin.
    pub fn voxel_volume(&self) -> f64 {
        self.voxel_size().product()
    }

    pub fn index_to_world(&self, xyz: Vec3d) -> Vec3d {
        self.map.apply_map(xyz)
    }
    pub fn index_to_world_coord(&self, ijk: Coord) -> Vec3d {
        self.index_to_world(ijk.as_vec3d())
    }
    pub fn world_to_index(&self, xyz: Vec3d) -> Vec3d {
        self.map.apply_inverse_map(xyz)
    }
    /// World position to the voxel whose center is nearest.
    pub fn world_to_index_cell_centered(&self, xyz: Vec3d) -> Coord {
        Coord::round(self.world_to_index(xyz))
    }
    /// World position to the voxel whose lower corner is nearest below (floor).
    pub fn world_to_index_node_centered(&self, xyz: Vec3d) -> Coord {
        Coord::floor(self.world_to_index(xyz))
    }

    /// World-space axis-aligned bounds of an index-space box (inclusive voxel centers).
    pub fn index_to_world_bbox(&self, bbox: &CoordBBox) -> (Vec3d, Vec3d) {
        let mut lo = Vec3d::splat(f64::MAX);
        let mut hi = Vec3d::splat(f64::MIN);
        for c in bbox.corners() {
            let w = self.index_to_world_coord(c);
            lo = lo.min_elem(&w);
            hi = hi.max_elem(&w);
        }
        (lo, hi)
    }

    // Linear transform editing, mirroring OpenVDB's pre/post operations.
    pub fn pre_rotate(&mut self, angle: f64, axis: Axis) {
        self.map = self.map.pre_rotate(angle, axis);
    }
    pub fn pre_translate(&mut self, t: Vec3d) {
        self.map = self.map.pre_translate(t);
    }
    pub fn pre_scale(&mut self, s: Vec3d) {
        self.map = self.map.pre_scale(s);
    }
    pub fn post_rotate(&mut self, angle: f64, axis: Axis) {
        self.map = self.map.post_rotate(angle, axis);
    }
    pub fn post_translate(&mut self, t: Vec3d) {
        self.map = self.map.post_translate(t);
    }
    pub fn post_scale(&mut self, s: Vec3d) {
        self.map = self.map.post_scale(s);
    }
    pub fn pre_mult(&mut self, m: &Mat4d) {
        self.map = self.map.pre_mult(m);
    }
    pub fn post_mult(&mut self, m: &Mat4d) {
        self.map = self.map.post_mult(m);
    }

    /// Approximate equality of the underlying maps.
    pub fn is_approx_equal(&self, other: &Transform, tol: f64) -> bool {
        self.map.is_approx_equal(&other.map, tol)
    }
}

impl fmt::Debug for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transform({:?})", self.map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linear_roundtrip() {
        let mut t = Transform::linear(0.5);
        t.post_translate(Vec3d::new(1.0, 2.0, 3.0));
        let w = t.index_to_world(Vec3d::new(2.0, 2.0, 2.0));
        assert!(w.is_approx_equal(&Vec3d::new(2.0, 3.0, 4.0), 1e-12));
        let i = t.world_to_index(w);
        assert!(i.is_approx_equal(&Vec3d::new(2.0, 2.0, 2.0), 1e-12));
        assert_eq!(t.world_to_index_cell_centered(Vec3d::new(2.2, 3.2, 4.2)), Coord::new(2, 2, 2));
        assert!(t.voxel_size().is_approx_equal(&Vec3d::splat(0.5), 1e-12));
    }
}
