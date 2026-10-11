//! Index-space to world-space transforms (port of `openvdb/math/Transform.h` and
//! `Transform.cc`).
//!
//! A [`Transform`] wraps a [`MapBase`]. Editing operations (`pre_scale`, `post_translate`, ...)
//! replace the map with the simplest map type that represents the result, exactly as OpenVDB
//! does, so e.g. `Transform::linear(0.5)` has a `UniformScaleMap` and becomes a
//! `UniformScaleTranslateMap` after a `post_translate`.

use super::bbox::BBoxd;
use super::maps::{read_name, simplify, write_name, MapBase, NonlinearFrustumMap};
use super::mat::{Axis, Mat3d, Mat4d};
use super::vec3::Vec3d;
use super::{Coord, CoordBBox};
use std::fmt;
use std::io::{self, Read, Write};

/// Mapping between index space (voxel coordinates) and world space.
#[derive(Clone)]
pub struct Transform {
    map: MapBase,
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transform {
    /// Wrap a map as is (like `Transform::read`, no simplification).
    pub fn from_map(map: MapBase) -> Self {
        Transform { map }
    }
    /// Wrap a map, converting linear maps (other than `UniformScaleMap`) to their simplest
    /// equivalent type, like OpenVDB's `Transform(MapBase::Ptr)` constructor.
    pub fn from_map_simplified(map: MapBase) -> Self {
        if !matches!(map, MapBase::UniformScale(_)) && map.is_linear() {
            Transform { map: simplify(map.get_affine_map()) }
        } else {
            Transform { map }
        }
    }
    /// The identity transform (voxel size 1, no offset), with a `UniformScaleMap`.
    pub fn identity() -> Self {
        Self::linear(1.0)
    }
    /// A uniform-scale linear transform with the given voxel size
    /// (`Transform::createLinearTransform(voxelSize)`).
    pub fn linear(voxel_size: f64) -> Self {
        Transform { map: MapBase::uniform_scale(voxel_size) }
    }
    /// Same as [`linear`](Self::linear) (`createLinearTransform`).
    pub fn create_linear_transform(voxel_size: f64) -> Self {
        Self::linear(voxel_size)
    }
    /// A linear transform from an affine matrix (row-vector convention), simplified to the
    /// simplest map type (`Transform::createLinearTransform(Mat4)`).
    ///
    /// # Panics
    /// If the matrix is not affine or is nearly singular.
    pub fn from_matrix(m: Mat4d) -> Self {
        Self::from_map_simplified(MapBase::from_matrix(m))
    }
    /// Same as [`from_matrix`](Self::from_matrix).
    pub fn create_linear_transform_from_matrix(m: Mat4d) -> Self {
        Self::from_matrix(m)
    }
    /// A frustum transform: the index-space `bbox` is mapped onto a frustum with the given
    /// `taper` (near/far plane width ratio) and `depth` (in near-plane widths), whose near plane
    /// is `voxel_size * bbox x-extent` wide (`Transform::createFrustumTransform`).
    ///
    /// # Panics
    /// If the box is degenerate.
    pub fn create_frustum_transform(bbox: &BBoxd, taper: f64, depth: f64, voxel_size: f64) -> Self {
        let f = MapBase::NonlinearFrustum(NonlinearFrustumMap::new(*bbox, taper, depth));
        Self::from_map_simplified(f.pre_scale(Vec3d::splat(voxel_size)))
    }
    /// The map.
    pub fn map(&self) -> &MapBase {
        &self.map
    }
    /// The map, mutably.
    pub fn map_mut(&mut self) -> &mut MapBase {
        &mut self.map
    }
    /// The map (`baseMap`).
    pub fn base_map(&self) -> &MapBase {
        &self.map
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
    /// World-space size of the voxel at the given index-space position (differs from
    /// [`voxel_size`](Self::voxel_size) only for non-linear maps).
    pub fn voxel_size_at(&self, xyz: Vec3d) -> Vec3d {
        self.map.voxel_size_at(xyz)
    }
    /// World-space volume of a voxel: the determinant of the map's Jacobian (negative for
    /// reflections), as in OpenVDB.
    pub fn voxel_volume(&self) -> f64 {
        self.map.determinant()
    }
    /// World-space volume of the voxel at the given index-space position.
    pub fn voxel_volume_at(&self, xyz: Vec3d) -> f64 {
        self.map.determinant_at(xyz)
    }

    /// Index space to world space.
    pub fn index_to_world(&self, xyz: Vec3d) -> Vec3d {
        self.map.apply_map(xyz)
    }
    /// Index coordinate to world space.
    pub fn index_to_world_coord(&self, ijk: Coord) -> Vec3d {
        self.index_to_world(ijk.as_vec3d())
    }
    /// World space to index space.
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

    /// World-space axis-aligned bounds of an index-space box (inclusive voxel centers), as a
    /// `(min, max)` pair.
    pub fn index_to_world_bbox(&self, bbox: &CoordBBox) -> (Vec3d, Vec3d) {
        let b = self.index_to_world_bboxd(&BBoxd::new(bbox.min.as_vec3d(), bbox.max.as_vec3d()));
        (b.min(), b.max())
    }
    /// World-space bounds of an index-space coordinate box (`indexToWorld(CoordBBox)`).
    pub fn index_coord_bbox_to_world(&self, bbox: &CoordBBox) -> BBoxd {
        self.index_to_world_bboxd(&BBoxd::new(bbox.min.as_vec3d(), bbox.max.as_vec3d()))
    }
    /// World-space bounds of the images of the eight corners of an index-space box
    /// (`indexToWorld(BBoxd)`).
    pub fn index_to_world_bboxd(&self, bbox: &BBoxd) -> BBoxd {
        let (lo, hi) = bounds_of_corners(bbox.min(), bbox.max(), |p| self.index_to_world(p));
        BBoxd::new(lo, hi)
    }
    /// Index-space bounds of the pre-images of the eight corners of a world-space box
    /// (`worldToIndex(BBoxd)`).
    pub fn world_to_index_bboxd(&self, bbox: &BBoxd) -> BBoxd {
        let (lo, hi) = calculate_bounds(self, bbox.min(), bbox.max());
        BBoxd::new(lo, hi)
    }
    /// Index-space coordinate bounds (rounded) of a world-space box
    /// (`worldToIndexCellCentered(BBoxd)`).
    pub fn world_to_index_cell_centered_bbox(&self, bbox: &BBoxd) -> CoordBBox {
        let (lo, hi) = calculate_bounds(self, bbox.min(), bbox.max());
        CoordBBox::new(Coord::round(lo), Coord::round(hi))
    }
    /// Index-space coordinate bounds (floored) of a world-space box
    /// (`worldToIndexNodeCentered(BBoxd)`).
    pub fn world_to_index_node_centered_bbox(&self, bbox: &BBoxd) -> CoordBBox {
        let (lo, hi) = calculate_bounds(self, bbox.min(), bbox.max());
        CoordBBox::new(Coord::floor(lo), Coord::floor(hi))
    }

    /// Prepend a rotation by `angle` radians about `axis`.
    pub fn pre_rotate(&mut self, angle: f64, axis: Axis) {
        self.map = self.map.pre_rotate(angle, axis);
    }
    /// Prepend a translation.
    pub fn pre_translate(&mut self, t: Vec3d) {
        self.map = self.map.pre_translate(t);
    }
    /// Prepend a scale.
    pub fn pre_scale(&mut self, s: Vec3d) {
        self.map = self.map.pre_scale(s);
    }
    /// Prepend a uniform scale.
    pub fn pre_scale_uniform(&mut self, s: f64) {
        self.pre_scale(Vec3d::splat(s));
    }
    /// Prepend a shear along `axis0` by a fraction of `axis1`.
    pub fn pre_shear(&mut self, shear: f64, axis0: Axis, axis1: Axis) {
        self.map = self.map.pre_shear(shear, axis0, axis1);
    }
    /// Prepend a matrix (`M = m * M`).
    pub fn pre_mult(&mut self, m: &Mat4d) {
        self.map = self.map.pre_mult(m);
    }
    /// Prepend a 3x3 matrix.
    pub fn pre_mult_mat3(&mut self, m: &Mat3d) {
        self.pre_mult(&Mat4d::from_mat3_translation(m, Vec3d::zero()));
    }
    /// Append a rotation by `angle` radians about `axis`.
    pub fn post_rotate(&mut self, angle: f64, axis: Axis) {
        self.map = self.map.post_rotate(angle, axis);
    }
    /// Append a translation.
    pub fn post_translate(&mut self, t: Vec3d) {
        self.map = self.map.post_translate(t);
    }
    /// Append a scale.
    pub fn post_scale(&mut self, s: Vec3d) {
        self.map = self.map.post_scale(s);
    }
    /// Append a uniform scale.
    pub fn post_scale_uniform(&mut self, s: f64) {
        self.post_scale(Vec3d::splat(s));
    }
    /// Append a shear along `axis0` by a fraction of `axis1`.
    pub fn post_shear(&mut self, shear: f64, axis0: Axis, axis1: Axis) {
        self.map = self.map.post_shear(shear, axis0, axis1);
    }
    /// Append a matrix (`M = M * m`).
    pub fn post_mult(&mut self, m: &Mat4d) {
        self.map = self.map.post_mult(m);
    }
    /// Append a 3x3 matrix.
    pub fn post_mult_mat3(&mut self, m: &Mat3d) {
        self.post_mult(&Mat4d::from_mat3_translation(m, Vec3d::zero()));
    }

    /// Approximate equality of the underlying maps.
    pub fn is_approx_equal(&self, other: &Transform, tol: f64) -> bool {
        self.map.is_approx_equal(&other.map, tol)
    }

    /// Read a transform as written by OpenVDB's `Transform::write`: the map's type name (a
    /// `u32` length followed by the bytes) and then the map's data.
    pub fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let name = read_name(r)?;
        Ok(Transform { map: MapBase::read_typed(&name, r)? })
    }
    /// Write the transform as OpenVDB's `Transform::write` does.
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_name(w, self.map.type_name())?;
        self.map.write(w)
    }
}

impl PartialEq for Transform {
    /// OpenVDB's `Transform::operator==`: equal voxel sizes, then either the same map type with
    /// equal maps, or two linear maps with approximately equal matrices.
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (self.voxel_size(), other.voxel_size());
        if !(0..3).all(|i| super::math_util::is_approx_equal_tol(a[i], b[i], 1.0e-7)) {
            return false;
        }
        if self.map_type() == other.map_type() {
            return self.map.is_equal(&other.map);
        }
        if self.is_linear() && other.is_linear() {
            return self.map.get_affine_map() == other.map.get_affine_map();
        }
        self.map.is_equal(&other.map)
    }
}

/// The component-wise bounds of `f` applied to the eight corners of the box `[lo, hi]`, visited
/// in OpenVDB's corner order.
fn bounds_of_corners(lo: Vec3d, hi: Vec3d, f: impl Fn(Vec3d) -> Vec3d) -> (Vec3d, Vec3d) {
    let corners = [
        lo,
        Vec3d::new(hi[0], lo[1], lo[2]),
        Vec3d::new(hi[0], hi[1], lo[2]),
        Vec3d::new(lo[0], hi[1], lo[2]),
        Vec3d::new(lo[0], lo[1], hi[2]),
        Vec3d::new(hi[0], lo[1], hi[2]),
        hi,
        Vec3d::new(lo[0], hi[1], hi[2]),
    ];
    let first = f(corners[0]);
    let (mut min, mut max) = (first, first);
    for c in &corners[1..] {
        let p = f(*c);
        for j in 0..3 {
            min[j] = min[j].min(p[j]);
            max[j] = max[j].max(p[j]);
        }
    }
    (min, max)
}

/// Index-space bounds `(min, max)` of the pre-images of the eight corners of the world-space box
/// `[min_ws, max_ws]` (`math::calculateBounds`).
pub fn calculate_bounds(t: &Transform, min_ws: Vec3d, max_ws: Vec3d) -> (Vec3d, Vec3d) {
    bounds_of_corners(min_ws, max_ws, |p| t.world_to_index(p))
}

impl fmt::Debug for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transform({:?})", self.map)
    }
}

impl fmt::Display for Transform {
    /// A summary in the spirit of `Transform::print`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dim = self.voxel_size();
        if (0..3).all(|i| super::math_util::is_approx_equal_tol(dim[i], dim[0], 1.0e-7)) {
            writeln!(f, "voxel size: {:.3}", dim[0])?;
        } else {
            writeln!(f, "voxel dimensions: [{:.3}, {:.3}, {:.3}]", dim[0], dim[1], dim[2])?;
        }
        let m = self.map.affine_matrix();
        if let MapBase::NonlinearFrustum(fr) = &self.map {
            writeln!(f, "frustum: taper {} depth {} bounds {:?}", fr.get_taper(), fr.get_depth(), fr.get_bbox())?;
            writeln!(f, "linear:")?;
        } else {
            writeln!(f, "index to world:")?;
        }
        for row in &m.0 {
            writeln!(f, "   [{:.3}, {:.3}, {:.3}, {:.3}]", row[0], row[1], row[2], row[3])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(t: &Transform) -> String {
        let mut buf = Vec::new();
        t.write(&mut buf).unwrap();
        buf.iter().map(|b| format!("{b:02x}")).collect()
    }

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

    /// A step of a transform-editing sequence checked against OpenVDB 10: expected map type,
    /// serialized bytes (`Transform::write`), voxel size, image of (1.5, -2.25, 3) and
    /// voxel volume.
    struct Expect {
        ty: &'static str,
        bytes: &'static str,
        vs: [f64; 3],
        w: [f64; 3],
        det: f64,
        uniform: bool,
    }

    fn check(label: &str, t: &Transform, e: &Expect) {
        assert_eq!(t.map_type(), e.ty, "{label}");
        assert_eq!(hex(t), e.bytes, "{label}");
        assert_eq!(t.voxel_size(), Vec3d::from(e.vs), "{label}");
        assert_eq!(t.index_to_world(Vec3d::new(1.5, -2.25, 3.0)), Vec3d::from(e.w), "{label}");
        assert_eq!(t.voxel_volume(), e.det, "{label}");
        assert_eq!(t.has_uniform_scale(), e.uniform, "{label}");
        let back = Transform::read(&mut &hex_to_bytes(e.bytes)[..]).unwrap();
        assert_eq!(back.map_type(), e.ty);
        assert_eq!(hex(&back), e.bytes);
        assert!(back == *t, "{label}");
    }

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    const T1: &str = "0f000000556e69666f726d5363616c654d6170000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000004000000000000000400000000000000040000000000000104000000000000010400000000000001040000000000000f03f000000000000f03f000000000000f03f";
    const T2: &str = "18000000556e69666f726d5363616c655472616e736c6174654d6170000000000000f03f00000000000000400000000000000840000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000e03f000000000000004000000000000000400000000000000040000000000000104000000000000010400000000000001040000000000000f03f000000000000f03f000000000000f03f";
    const T3: &str = "110000005363616c655472616e736c6174654d6170000000000000f03f00000000000000400000000000000840000000000000e03f000000000000f03f000000000000f83f000000000000e03f000000000000f03f000000000000f83f0000000000000040000000000000f03f555555555555e53f0000000000001040000000000000f03f1cc7711cc771dc3f000000000000f03f000000000000e03f555555555555d53f";
    const T4: &str = "09000000416666696e654d6170ba092fd41d92de3f000000000000000033baba95cde9c2bf00000000000000000000000000000000000000000000f03f000000000000000000000000000000004c179860b45edc3f00000000000000004c47235f96edf63f00000000000000008390631a6978fd3f000000000000004006f06bac5c900440000000000000f03f";
    const T5: &str = "09000000416666696e654d6170ba092fd41d92de3f000000000000000033baba95cde9c2bf00000000000000000000000000000000000000000000f03f000000000000000000000000000000004c179860b45edc3f00000000000000004c47235f96edf63f0000000000000000523a7eb0416b0240000000000000044008ec86d773b40940000000000000f03f";
    const T6: &str = "09000000416666696e654d6170825d9cabefaddc3f000000000000000033baba95cde9c2bf00000000000000000000000000000000000000000000f03f00000000000000000000000000000000c4c1f3ef2f5be73f00000000000000004c47235f96edf63f0000000000000000543699db588f0740000000000000044008ec86d773b40940000000000000f03f";

    #[test]
    fn editing_sequence_matches_openvdb() {
        let mut t = Transform::create_linear_transform(0.5);
        let e1 = Expect {
            ty: "UniformScaleMap",
            bytes: T1,
            vs: [0.5; 3],
            w: [0.75, -1.125, 1.5],
            det: 0.125,
            uniform: true,
        };
        check("T1", &t, &e1);
        t.post_translate(Vec3d::new(1.0, 2.0, 3.0));
        let e2 = Expect {
            ty: "UniformScaleTranslateMap",
            bytes: T2,
            vs: [0.5; 3],
            w: [1.75, 0.875, 4.5],
            det: 0.125,
            uniform: true,
        };
        check("T2", &t, &e2);
        t.pre_scale(Vec3d::new(1.0, 2.0, 3.0));
        let e3 = Expect {
            ty: "ScaleTranslateMap",
            bytes: T3,
            vs: [0.5, 1.0, 1.5],
            w: [1.75, -0.25, 7.5],
            det: 0.75,
            uniform: false,
        };
        check("T3", &t, &e3);
        t.post_rotate(0.3, Axis::Y);
        let e4 = Expect {
            ty: "AffineMap",
            bytes: T4,
            vs: [0.5, 1.0, 1.5],
            w: [3.8882404059298574, -0.25, 6.6478633067847017],
            det: 0.75,
            uniform: false,
        };
        check("T4", &t, &e4);
        t.pre_translate(Vec3d::splat(0.5));
        let e5 = Expect {
            ty: "AffineMap",
            bytes: T5,
            vs: [0.49999999999999983, 1.0, 1.5],
            w: [4.3487146832072634, 0.25, 7.2904856219635716],
            det: 0.75,
            uniform: false,
        };
        check("T5", &t, &e5);
        t.post_shear(0.2, Axis::X, Axis::Z);
        let e6 = Expect {
            ty: "AffineMap",
            bytes: T6,
            vs: [0.47184870271698293, 1.0, 1.6081757415465205],
            w: [5.8068118075999777, 0.25, 7.2904856219635716],
            det: 0.75,
            uniform: false,
        };
        check("T6", &t, &e6);
    }

    const U1: &str = "18000000556e69666f726d5363616c655472616e736c6174654d6170000000000000004000000000000000400000000000000040000000000000004000000000000000400000000000000040000000000000004000000000000000400000000000000040000000000000e03f000000000000e03f000000000000e03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f";
    const U2: &str = "18000000556e69666f726d5363616c655472616e736c6174654d6170000000000000184000000000000018400000000000001840000000000000184000000000000018400000000000001840000000000000184000000000000018400000000000001840555555555555c53f555555555555c53f555555555555c53f1cc7711cc7719c3f1cc7711cc7719c3f1cc7711cc7719c3f555555555555b53f555555555555b53f555555555555b53f";
    const U3: &str = "110000005363616c655472616e736c6174654d6170000000000000184000000000000028400000000000001840000000000000184000000000000028400000000000001840000000000000184000000000000028400000000000001840555555555555c53f555555555555b53f555555555555c53f1cc7711cc7719c3f1cc7711cc7717c3f1cc7711cc7719c3f555555555555b53f555555555555a53f555555555555b53f";
    const V1: &str = "18000000556e69666f726d5363616c655472616e736c6174654d6170000000000000f03f00000000000000400000000000000840000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000e03f000000000000e03f000000000000e03f";
    const V2: &str = "18000000556e69666f726d5363616c655472616e736c6174654d6170000000000000f03f00000000000000400000000000000840000000000000004000000000000000400000000000000040000000000000004000000000000000400000000000000040000000000000e03f000000000000e03f000000000000e03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f";
    const V3: &str = "110000005363616c655472616e736c6174654d6170000000000000f03f00000000000010400000000000002240000000000000f03f00000000000000400000000000000840000000000000f03f00000000000000400000000000000840000000000000f03f000000000000e03f555555555555d53f000000000000f03f000000000000d03f1cc7711cc771bc3f000000000000e03f000000000000d03f555555555555c53f";
    const V4: &str = "09000000416666696e654d6170000000000000f03f0000000000000000000000000000000000000000000000000000000000000000f8a2dbe75479ed3f6b672be93aecd83f000000000000000000000000000000006b672be93aecd8bff8a2dbe75479ed3f0000000000000000000000000000000000000000000000000000000000000000000000000000f03f";
    const V5: &str = "0f000000556e69666f726d5363616c654d6170000000000000004000000000000000400000000000000040000000000000004000000000000000400000000000000040000000000000e03f000000000000e03f000000000000e03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f000000000000d03f";
    const V6: &str = "18000000556e69666f726d5363616c655472616e736c6174654d6170000000000000084000000000000010400000000000001440000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000e03f000000000000e03f000000000000e03f";

    #[test]
    fn map_type_conversions_match_openvdb() {
        use crate::math::maps::{ScaleMap, TranslationMap, UnitaryMap};
        let mut t = Transform::linear(2.0);
        t.pre_translate(Vec3d::splat(1.0));
        check(
            "U1",
            &t,
            &Expect {
                ty: "UniformScaleTranslateMap",
                bytes: U1,
                vs: [2.0; 3],
                w: [5.0, -2.5, 8.0],
                det: 8.0,
                uniform: true,
            },
        );
        t.post_scale(Vec3d::splat(3.0));
        check(
            "U2",
            &t,
            &Expect {
                ty: "UniformScaleTranslateMap",
                bytes: U2,
                vs: [6.0; 3],
                w: [15.0, -7.5, 24.0],
                det: 216.0,
                uniform: true,
            },
        );
        t.post_scale(Vec3d::new(1.0, 2.0, 1.0));
        let e3 = Expect {
            ty: "ScaleTranslateMap",
            bytes: U3,
            vs: [6.0, 12.0, 6.0],
            w: [15.0, -15.0, 24.0],
            det: 432.0,
            uniform: false,
        };
        check("U3", &t, &e3);
        t.pre_rotate(0.0, Axis::Z);
        check("U4", &t, &e3);

        let mut t =
            Transform::from_map_simplified(MapBase::Translation(TranslationMap::new(Vec3d::new(1.0, 2.0, 3.0))));
        check(
            "V1",
            &t,
            &Expect {
                ty: "UniformScaleTranslateMap",
                bytes: V1,
                vs: [1.0; 3],
                w: [2.5, -0.25, 6.0],
                det: 1.0,
                uniform: true,
            },
        );
        t.pre_scale_uniform(2.0);
        check(
            "V2",
            &t,
            &Expect {
                ty: "UniformScaleTranslateMap",
                bytes: V2,
                vs: [2.0; 3],
                w: [4.0, -2.5, 9.0],
                det: 8.0,
                uniform: true,
            },
        );
        let mut t2 =
            Transform::from_map_simplified(MapBase::Translation(TranslationMap::new(Vec3d::new(1.0, 2.0, 3.0))));
        t2.post_scale(Vec3d::new(1.0, 2.0, 3.0));
        check(
            "V3",
            &t2,
            &Expect {
                ty: "ScaleTranslateMap",
                bytes: V3,
                vs: [1.0, 2.0, 3.0],
                w: [2.5, -0.5, 18.0],
                det: 6.0,
                uniform: false,
            },
        );
        let t3 = Transform::from_map_simplified(MapBase::Unitary(UnitaryMap::from_axis(Axis::X, 0.4)));
        check(
            "V4",
            &t3,
            &Expect {
                ty: "AffineMap",
                bytes: V4,
                vs: [1.0; 3],
                w: [1.5, -3.2406422634324432, 1.8869917118141917],
                det: 1.0,
                uniform: true,
            },
        );
        let t4 = Transform::from_map_simplified(MapBase::Scale(ScaleMap::new(Vec3d::splat(2.0))));
        check(
            "V5",
            &t4,
            &Expect { ty: "UniformScaleMap", bytes: V5, vs: [2.0; 3], w: [3.0, -4.5, 6.0], det: 8.0, uniform: true },
        );
        let t5 = Transform::from_matrix(Mat4d::translation(Vec3d::new(3.0, 4.0, 5.0)));
        check(
            "V6",
            &t5,
            &Expect {
                ty: "UniformScaleTranslateMap",
                bytes: V6,
                vs: [1.0; 3],
                w: [4.5, 1.75, 8.0],
                det: 1.0,
                uniform: true,
            },
        );
    }

    const F1: &str = "130000004e6f6e6c696e6561724672757374756d4d6170000000000000000000000000000000000000000000000000000000000000244000000000000034400000000000003e40000000000000e03f000000000000004009000000416666696e654d6170000000000000d03f0000000000000000000000000000000000000000000000000000000000000000000000000000d03f0000000000000000000000000000000000000000000000000000000000000000000000000000d03f0000000000000000000000000000000000000000000000000000000000000000000000000000f03f";
    const F2: &str = "130000004e6f6e6c696e6561724672757374756d4d6170000000000000000000000000000000000000000000000000000000000000244000000000000034400000000000003e40000000000000e03f000000000000004009000000416666696e654d6170000000000000d03f0000000000000000000000000000000000000000000000000000000000000000000000000000d03f0000000000000000000000000000000000000000000000000000000000000000000000000000d03f0000000000000000000000000000f03f00000000000000400000000000000840000000000000f03f";
    const F3: &str = "130000004e6f6e6c696e6561724672757374756d4d6170000000000000000000000000000000000000000000000000000000000000244000000000000034400000000000003e40000000000000e03f000000000000004009000000416666696e654d6170000000000000d03f00000000000000000000000000000000000000000000000000000000000000007a627795b45ccf3fbcd23d23ff6da93f00000000000000000000000000000000bcd23d23ff6da9bf7a627795b45ccf3f0000000000000000000000000000f03f00000000000000400000000000000840000000000000f03f";

    #[test]
    fn frustum_transform_matches_openvdb() {
        let bbox = BBoxd::new(Vec3d::zero(), Vec3d::new(10.0, 20.0, 30.0));
        let mut t = Transform::create_frustum_transform(&bbox, 0.5, 2.0, 0.25);
        assert_eq!(t.map_type(), "NonlinearFrustumMap");
        assert!(!t.is_linear());
        assert!(!t.is_identity());
        assert!(!t.has_uniform_scale());
        assert_eq!(hex(&t), F1);
        assert_eq!(t.voxel_size(), Vec3d::new(0.025000000000000001, 0.025000000000000001, 0.016666666666666666));
        assert_eq!(t.voxel_volume(), 0.015625);
        t.post_translate(Vec3d::new(1.0, 2.0, 3.0));
        assert_eq!(hex(&t), F2);
        t.pre_rotate(0.2, Axis::X);
        assert_eq!(hex(&t), F3);
        let back = Transform::read(&mut &hex_to_bytes(F3)[..]).unwrap();
        assert!(back == t);
        assert_eq!(hex(&back), F3);
        // pre/post multiplication by matrices is applied to the frustum's second map.
        let mut u = t.clone();
        u.post_mult(&Mat4d::identity());
        assert!(u == t);
    }

    #[test]
    fn bbox_helpers_match_openvdb() {
        let mut t = Transform::linear(0.5);
        t.post_rotate(0.3, Axis::Z);
        let b = t.index_coord_bbox_to_world(&CoordBBox::from_coords(0, 0, 0, 10, 20, 30));
        assert!(b.min().is_approx_equal(&Vec3d::new(-2.9552020666133956, 0.0, 0.0), 1e-14));
        assert!(b.max().is_approx_equal(&Vec3d::new(4.7766824456280297, 11.030965924562757, 15.0), 1e-14));
        assert_eq!(t.world_to_index_cell_centered_bbox(&b), CoordBBox::from_coords(-6, -3, 0, 16, 23, 30));
        let (lo, hi) = t.index_to_world_bbox(&CoordBBox::from_coords(0, 0, 0, 10, 20, 30));
        assert_eq!((lo, hi), (b.min(), b.max()));
        let ib = t.world_to_index_bboxd(&b);
        assert!(ib.is_inside(&Vec3d::new(5.0, 5.0, 5.0)));
        let nb = t.world_to_index_node_centered_bbox(&b);
        assert!(nb.min.all_le(&Coord::new(0, 0, 0)));
    }

    #[test]
    fn equality_and_mults() {
        let a = Transform::linear(0.5);
        let b = Transform::from_matrix(Mat4d::scale(Vec3d::splat(0.5)));
        assert_eq!(b.map_type(), "UniformScaleMap");
        assert!(a == b);
        let c = Transform::from_map(MapBase::from_matrix(Mat4d::scale(Vec3d::splat(0.5))));
        assert_eq!(c.map_type(), "AffineMap");
        assert!(a == c);
        assert!(a != Transform::linear(0.25));
        let mut d = Transform::linear(1.0);
        d.pre_mult_mat3(&Mat3d::from_diagonal(Vec3d::new(1.0, 2.0, 3.0)));
        assert_eq!(d.map_type(), "ScaleMap");
        d.post_mult(&Mat4d::translation(Vec3d::splat(1.0)));
        assert_eq!(d.map_type(), "ScaleTranslateMap");
        assert!(Transform::identity().is_identity());
        assert!(Transform::default().is_identity());
        assert!(!format!("{}", d).is_empty());
    }
}
