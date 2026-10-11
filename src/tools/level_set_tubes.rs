//! Narrow-band level sets of capsules, tapered capsules and tube complexes
//! (port of `openvdb/tools/LevelSetTubes.h`).
//!
//! The signed distance functions are exactly those of OpenVDB's `CapsuleVoxelizer` and
//! `TaperedCapsuleVoxelizer` (evaluated in the grid's value precision). OpenVDB rasterizes
//! them with its `ConvexVoxelizer`, which scan-converts the dilated shape column by column;
//! this port instead visits the shape's bounding box with a distance-based leapfrog along
//! z, which touches exactly the voxels whose distance lies inside the narrow band. Inactive
//! values are then set by a signed flood fill. The active voxels and their values match
//! OpenVDB's (verified against the C++ implementation); the tile layout of the interior may
//! differ.

use super::mesh_support::{prune_level_set, signed_flood_fill_with_values};
use crate::grid::{Grid, GridClass};
use crate::math::{Coord, CoordBBox, Transform, Vec3};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::{RealValue, Value};
use rayon::prelude::*;
use std::collections::HashMap;

/// How radii given to [`create_level_set_tube_complex_with_radii`] are interpreted
/// (OpenVDB's `TubeRadiiPolicy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TubeRadiiPolicy {
    /// Per-vertex radii if there are as many radii as vertices, else per-segment radii.
    #[default]
    Automatic,
    /// One radius per vertex (tapered capsules).
    VertexRadii,
    /// One radius per segment (capsules).
    SegmentRadii,
}

/// A convex primitive with a signed distance function in index (voxel) units.
pub(crate) trait SdfShape<T: RealValue>: Sync {
    /// Signed distance (in voxel units) at the index-space position `p`.
    fn sd(&self, p: &Vec3<T>) -> T;
    /// Index-space bounding box of the shape dilated by the narrow band.
    fn bbox(&self) -> CoordBBox;
}

#[inline]
fn t<T: RealValue>(v: f64) -> T {
    T::from_f64(v)
}

fn vec_t<T: RealValue, S: RealValue>(v: &Vec3<S>) -> Vec3<T> {
    Vec3([T::from_f64(v[0].to_f64()), T::from_f64(v[1].to_f64()), T::from_f64(v[2].to_f64())])
}

fn bbox_of(min: [f64; 3], max: [f64; 3]) -> CoordBBox {
    CoordBBox::new(
        Coord::new(min[0].floor() as i32 - 1, min[1].floor() as i32 - 1, min[2].floor() as i32 - 1),
        Coord::new(max[0].ceil() as i32 + 1, max[1].ceil() as i32 + 1, max[2].ceil() as i32 + 1),
    )
}

/// The capsule of `lvlset::CapsuleVoxelizer`.
pub(crate) struct Capsule<T> {
    pt1: Vec3<T>,
    pt2: Vec3<T>,
    v: Vec3<T>,
    vlen_sqr: T,
    rad: T,
    orad: T,
}

impl<T: RealValue> Capsule<T> {
    pub(crate) fn new<S: RealValue>(pt1: &Vec3<S>, pt2: &Vec3<S>, r: S, vx: T, hw: T) -> Self {
        let (a, b) = if pt1[0] <= pt2[0] { (pt1, pt2) } else { (pt2, pt1) };
        let pt1 = vec_t::<T, S>(a) / vx;
        let pt2 = vec_t::<T, S>(b) / vx;
        let rad = T::from_f64(r.to_f64()) / vx;
        let v = pt2 - pt1;
        Capsule { pt1, pt2, v, vlen_sqr: v.length_sqr(), rad, orad: rad + hw }
    }
}

impl<T: RealValue> SdfShape<T> for Capsule<T> {
    #[inline]
    fn sd(&self, p: &Vec3<T>) -> T {
        let w = *p - self.pt1;
        let dot = w.dot(&self.v);
        if dot <= T::tolerance() {
            return w.length() - self.rad;
        }
        if dot >= self.vlen_sqr {
            return (*p - self.pt2).length() - self.rad;
        }
        let tt = w.dot(&self.v) / self.vlen_sqr;
        (w - self.v * tt).length() - self.rad
    }
    fn bbox(&self) -> CoordBBox {
        let o = self.orad.to_f64();
        let mut mn = [0.0; 3];
        let mut mx = [0.0; 3];
        for i in 0..3 {
            mn[i] = self.pt1[i].to_f64().min(self.pt2[i].to_f64()) - o;
            mx[i] = self.pt1[i].to_f64().max(self.pt2[i].to_f64()) + o;
        }
        bbox_of(mn, mx)
    }
}

/// The tapered capsule (round cone) of `lvlset::TaperedCapsuleVoxelizer`.
pub(crate) struct TaperedCapsule<T> {
    pt1: Vec3<T>,
    pt2: Vec3<T>,
    v: Vec3<T>,
    vlen_sqr: T,
    inv_vlen_sqr: T,
    rad1: T,
    rad2: T,
    orad1: T,
    orad2: T,
    rdiff: T,
    rdiff2: T,
    a2: T,
}

impl<T: RealValue> TaperedCapsule<T> {
    /// Returns `None` if the dilated shape is empty.
    pub(crate) fn new<S: RealValue>(pt1: &Vec3<S>, pt2: &Vec3<S>, r1: S, r2: S, vx: T, hw: T) -> Option<Self> {
        let (p1, p2, rad1, rad2) = if r2 <= r1 {
            (
                vec_t::<T, S>(pt1) / vx,
                vec_t::<T, S>(pt2) / vx,
                T::from_f64(r1.to_f64()) / vx,
                T::from_f64(r2.to_f64()) / vx,
            )
        } else {
            (
                vec_t::<T, S>(pt2) / vx,
                vec_t::<T, S>(pt1) / vx,
                T::from_f64(r2.to_f64()) / vx,
                T::from_f64(r1.to_f64()) / vx,
            )
        };
        let orad1 = rad1 + hw;
        let orad2 = rad2 + hw;
        if orad1 <= <T as Value>::zero() {
            return None;
        }
        if orad2 < <T as Value>::zero() {
            let tt = if r2 <= r1 { r2 / (r2 - r1) } else { r1 / (r1 - r2) };
            let pt_0 = if r2 <= r1 { *pt1 } else { *pt2 };
            let r_0 = if r2 <= r1 { r1 } else { r2 };
            let pt_mid = *pt1 * tt + *pt2 * (S::one() - tt);
            return Self::new(&pt_0, &pt_mid, r_0, <S as Value>::zero(), vx, hw);
        }
        let v = p2 - p1;
        let vlen_sqr = v.length_sqr();
        let inv_vlen_sqr = if vlen_sqr != <T as Value>::zero() { T::one() / vlen_sqr } else { T::one() };
        let rdiff = rad1 - rad2;
        let rdiff2 = rdiff * rdiff;
        Some(TaperedCapsule {
            pt1: p1,
            pt2: p2,
            v,
            vlen_sqr,
            inv_vlen_sqr,
            rad1,
            rad2,
            orad1,
            orad2,
            rdiff,
            rdiff2,
            a2: vlen_sqr - rdiff2,
        })
    }
}

#[inline]
fn sign<T: RealValue>(v: T) -> T {
    if v > <T as Value>::zero() {
        T::one()
    } else if v < <T as Value>::zero() {
        -T::one()
    } else {
        <T as Value>::zero()
    }
}

impl<T: RealValue> SdfShape<T> for TaperedCapsule<T> {
    #[inline]
    fn sd(&self, p: &Vec3<T>) -> T {
        let w = *p - self.pt1;
        let y = w.dot(&self.v);
        let z = y - self.vlen_sqr;
        let x2 = (w * self.vlen_sqr - self.v * y).length_sqr();
        let y2 = y * y * self.vlen_sqr;
        let z2 = z * z * self.vlen_sqr;
        let k = self.rdiff2 * x2;
        if sign(z) * self.a2 * z2 >= k {
            return (x2 + z2).sqrt() * self.inv_vlen_sqr - self.rad2;
        }
        if sign(y) * self.a2 * y2 <= k {
            return (x2 + y2).sqrt() * self.inv_vlen_sqr - self.rad1;
        }
        ((x2 * self.a2 * self.inv_vlen_sqr).sqrt() + y * self.rdiff) * self.inv_vlen_sqr - self.rad1
    }
    fn bbox(&self) -> CoordBBox {
        let (o1, o2) = (self.orad1.to_f64(), self.orad2.to_f64());
        let mut mn = [0.0; 3];
        let mut mx = [0.0; 3];
        for i in 0..3 {
            mn[i] = (self.pt1[i].to_f64() - o1).min(self.pt2[i].to_f64() - o2);
            mx[i] = (self.pt1[i].to_f64() + o1).max(self.pt2[i].to_f64() + o2);
        }
        bbox_of(mn, mx)
    }
}

/// A shape given as a boxed trait object.
pub(crate) type DynShape<T> = Box<dyn SdfShape<T> + Send>;

/// Rasterize the narrow band of a single shape into `tree` (min-combining with existing
/// active values).
fn rasterize_shape<T: RealValue>(tree: &mut Tree<T>, shape: &dyn SdfShape<T>, vox: T, hw: T) {
    let bg = *tree.background();
    let neg_bg = -bg;
    let b = shape.bbox();
    let hw64 = hw.to_f64();
    let mut acc = tree.accessor_mut();
    for x in b.min[0]..=b.max[0] {
        for y in b.min[1]..=b.max[1] {
            let mut z = b.min[2];
            while z <= b.max[2] {
                let p = Vec3([t::<T>(x as f64), t::<T>(y as f64), t::<T>(z as f64)]);
                let sd = shape.sd(&p);
                let dist = vox * sd;
                if dist > neg_bg && dist < bg {
                    let c = Coord::new(x, y, z);
                    let (val, on) = acc.probe_value(&c);
                    if !on || dist < val {
                        acc.set_value_on(&c, dist);
                    }
                }
                let a = sd.to_f64().abs() - hw64;
                z += if a > 1.5 { (a - 0.5).floor() as i32 } else { 1 };
            }
        }
    }
}

/// Min-merge the active voxels of `rhs` into `lhs`.
fn merge_min<T: RealValue>(lhs: &mut Tree<T>, rhs: Tree<T>) {
    for leaf in super::mesh_support::take_leaves(rhs) {
        match lhs.probe_leaf_mut(&leaf.origin()) {
            None => lhs.add_leaf(leaf),
            Some(mine) => {
                for (n, &v) in leaf.iter_on() {
                    if !mine.is_value_on_at(n) || v < mine.buffer()[n] {
                        mine.set_value_on_at(n, v);
                    }
                }
            }
        }
    }
}

/// Rasterize the union of `shapes` into a new narrow-band level set tree.
pub(crate) fn rasterize_union<T: RealValue>(shapes: &[DynShape<T>], vox: T, hw: T, bg: T, prune: bool) -> Tree<T> {
    let trees: Vec<Tree<T>> = shapes
        .par_iter()
        .fold(
            || Tree::<T>::new(bg),
            |mut tree, s| {
                rasterize_shape(&mut tree, s.as_ref(), vox, hw);
                tree
            },
        )
        .collect();
    let mut tree = Tree::<T>::new(bg);
    for t in trees {
        merge_min(&mut tree, t);
    }
    if shapes.len() > 1 {
        // Voxels deep inside any shape are interior (OpenVDB marks them with inactive -bg).
        let mut index: HashMap<Coord, Vec<usize>> = HashMap::new();
        for (i, s) in shapes.iter().enumerate() {
            let b = s.bbox();
            let (lo, hi) = (b.min.align_down(8), b.max.align_down(8));
            let mut x = lo[0];
            while x <= hi[0] {
                let mut y = lo[1];
                while y <= hi[1] {
                    let mut z = lo[2];
                    while z <= hi[2] {
                        index.entry(Coord::new(x, y, z)).or_default().push(i);
                        z += 8;
                    }
                    y += 8;
                }
                x += 8;
            }
        }
        let neg_bg = -bg;
        tree.par_for_each_leaf_mut(|leaf: &mut LeafNode<T>| {
            let cands = match index.get(&leaf.origin()) {
                Some(c) => c,
                None => return,
            };
            let on: Vec<usize> = leaf.value_mask().iter_on().collect();
            for n in on {
                let c = leaf.offset_to_global_coord(n);
                let p = Vec3([t::<T>(c[0] as f64), t::<T>(c[1] as f64), t::<T>(c[2] as f64)]);
                for &i in cands {
                    if vox * shapes[i].sd(&p) <= neg_bg {
                        leaf.set_value_off_at(n, neg_bg);
                        break;
                    }
                }
            }
        });
    }
    signed_flood_fill_with_values(&mut tree, bg, -bg);
    if prune {
        prune_level_set(&mut tree, bg, -bg);
    }
    tree
}

/// `createLevelSet<GridType>(voxelSize, halfWidth)` plus the voxelizer constants
/// `(voxel size, half width in voxels, background)`.
pub(crate) fn level_set_params<T: RealValue>(voxel_size: f32, half_width: f32) -> (Grid<T>, T, T, T) {
    assert!(voxel_size > 0.0, "voxel size must be positive");
    assert!(half_width > 0.0, "half-width must be positive");
    let bg = T::from_f64((voxel_size * half_width) as f64);
    let mut grid = Grid::<T>::new(bg).with_transform(Transform::linear(voxel_size as f64));
    grid.set_grid_class(GridClass::LevelSet);
    let vs = grid.voxel_size()[0];
    let vox = T::from_f64(vs);
    let hw = T::from_f64(bg.to_f64() / vs);
    (grid, vox, hw, bg)
}

/// Narrow-band level set of a capsule with end points `pt1`, `pt2` and the given radius
/// (world units) (`tools::createLevelSetCapsule`).
///
/// # Panics
/// If `voxel_size` or `half_width` is not positive.
pub fn create_level_set_capsule<T: RealValue, S: RealValue>(
    pt1: &Vec3<S>,
    pt2: &Vec3<S>,
    radius: S,
    voxel_size: f32,
    half_width: f32,
) -> Grid<T> {
    let (mut grid, vox, hw, bg) = level_set_params::<T>(voxel_size, half_width);
    let shapes: Vec<DynShape<T>> = vec![Box::new(Capsule::new(pt1, pt2, radius, vox, hw))];
    grid.set_tree(rasterize_union(&shapes, vox, hw, bg, false));
    grid
}

fn tapered_or_degenerate<T: RealValue, S: RealValue>(
    pt1: &Vec3<S>,
    pt2: &Vec3<S>,
    r1: S,
    r2: S,
    voxel_size: S,
    vox: T,
    hw: T,
) -> Option<DynShape<T>> {
    let d = r1 - r2;
    if (*pt1 - *pt2).length_sqr() <= d * d {
        Some(if r1 >= r2 {
            Box::new(Capsule::new(pt1, pt1, r1, vox, hw))
        } else {
            Box::new(Capsule::new(pt2, pt2, r2, vox, hw))
        })
    } else if (r1 - r2).abs() < S::from_f64(0.001) * voxel_size {
        Some(Box::new(Capsule::new(pt1, pt2, r1, vox, hw)))
    } else {
        TaperedCapsule::new(pt1, pt2, r1, r2, vox, hw).map(|s| Box::new(s) as DynShape<T>)
    }
}

/// Narrow-band level set of a tapered capsule with radii `radius1` at `pt1` and `radius2`
/// at `pt2` (`tools::createLevelSetTaperedCapsule`). Degenerate inputs produce a ball or a
/// capsule, as in OpenVDB.
pub fn create_level_set_tapered_capsule<T: RealValue, S: RealValue>(
    pt1: &Vec3<S>,
    pt2: &Vec3<S>,
    radius1: S,
    radius2: S,
    voxel_size: f32,
    half_width: f32,
) -> Grid<T> {
    let (mut grid, vox, hw, bg) = level_set_params::<T>(voxel_size, half_width);
    let shapes: Vec<DynShape<T>> =
        tapered_or_degenerate(pt1, pt2, radius1, radius2, S::from_f64(voxel_size as f64), vox, hw)
            .into_iter()
            .collect();
    grid.set_tree(rasterize_union(&shapes, vox, hw, bg, false));
    grid
}

/// Narrow-band level set of a tube complex: the union of capsules of constant `radius`
/// around the given segments (`tools::createLevelSetTubeComplex`).
pub fn create_level_set_tube_complex<T: RealValue, S: RealValue>(
    vertices: &[Vec3<S>],
    segments: &[[u32; 2]],
    radius: S,
    voxel_size: f32,
    half_width: f32,
) -> Grid<T> {
    let (mut grid, vox, hw, bg) = level_set_params::<T>(voxel_size, half_width);
    let shapes: Vec<DynShape<T>> = segments
        .iter()
        .map(|s| {
            assert!((s[0] as usize) < vertices.len() && (s[1] as usize) < vertices.len(), "out of bounds index");
            Box::new(Capsule::new(&vertices[s[0] as usize], &vertices[s[1] as usize], radius, vox, hw)) as DynShape<T>
        })
        .collect();
    grid.set_tree(rasterize_union(&shapes, vox, hw, bg, true));
    grid
}

/// Tube complex with per-vertex radii (tapered capsules) or per-segment radii (capsules)
/// (`tools::createLevelSetTubeComplex` with radii).
///
/// # Panics
/// If the number of radii does not match the policy.
pub fn create_level_set_tube_complex_with_radii<T: RealValue, S: RealValue>(
    vertices: &[Vec3<S>],
    segments: &[[u32; 2]],
    radii: &[S],
    voxel_size: f32,
    half_width: f32,
    policy: TubeRadiiPolicy,
) -> Grid<T> {
    let policy = match policy {
        TubeRadiiPolicy::Automatic => {
            assert!(
                vertices.len() == radii.len() || segments.len() == radii.len(),
                "createLevelSetTubeComplex requires equal number of vertices and radii, or segments and radii"
            );
            if vertices.len() == radii.len() {
                TubeRadiiPolicy::VertexRadii
            } else {
                TubeRadiiPolicy::SegmentRadii
            }
        }
        p => p,
    };
    let (mut grid, vox, hw, bg) = level_set_params::<T>(voxel_size, half_width);
    let vs = S::from_f64(voxel_size as f64);
    let shapes: Vec<DynShape<T>> = match policy {
        TubeRadiiPolicy::VertexRadii => {
            assert_eq!(vertices.len(), radii.len(), "per-vertex radii policy requires one radius per vertex");
            segments
                .iter()
                .filter_map(|s| {
                    let (i, j) = (s[0] as usize, s[1] as usize);
                    tapered_or_degenerate(&vertices[i], &vertices[j], radii[i], radii[j], vs, vox, hw)
                })
                .collect()
        }
        _ => {
            assert_eq!(segments.len(), radii.len(), "per-segment radii policy requires one radius per segment");
            segments
                .iter()
                .zip(radii)
                .map(|(s, &r)| {
                    Box::new(Capsule::new(&vertices[s[0] as usize], &vertices[s[1] as usize], r, vox, hw))
                        as DynShape<T>
                })
                .collect()
        }
    };
    grid.set_tree(rasterize_union(&shapes, vox, hw, bg, true));
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3f;

    #[test]
    fn capsule_values() {
        let g: Grid<f32> =
            create_level_set_capsule(&Vec3f::new(-1.0, 0.0, 0.0), &Vec3f::new(1.0, 0.0, 0.0), 0.5, 0.1, 3.0);
        assert_eq!(g.grid_class(), GridClass::LevelSet);
        // On the axis, 0.7 above the segment: distance 0.2.
        assert!((g.get_value(&Coord::new(0, 7, 0)) - 0.2).abs() < 1e-5);
        // Beyond the end cap.
        assert!((g.get_value(&Coord::new(17, 0, 0)) - 0.2).abs() < 1e-5);
        assert!(g.get_value(&Coord::new(0, 0, 0)) < 0.0);
        assert!(g.get_value(&Coord::new(0, 20, 0)) > 0.0);
        for (c, v) in g.tree().active_voxels() {
            let p = Vec3f::new(c[0] as f32 * 0.1, c[1] as f32 * 0.1, c[2] as f32 * 0.1);
            let q = Vec3f::new(p[0].clamp(-1.0, 1.0), 0.0, 0.0);
            assert!(((p - q).length() - 0.5 - v).abs() < 1e-4);
        }
    }

    #[test]
    fn tapered_and_complex() {
        let g: Grid<f32> = create_level_set_tapered_capsule(
            &Vec3f::new(0.0, 0.0, 0.0),
            &Vec3f::new(2.0, 0.0, 0.0),
            0.8,
            0.3,
            0.1,
            3.0,
        );
        assert!((g.get_value(&Coord::new(-10, 0, 0)) - 0.2).abs() < 1e-4);
        assert!((g.get_value(&Coord::new(25, 0, 0)) - 0.2).abs() < 1e-4);
        let verts = vec![Vec3f::new(0.0, 0.0, 0.0), Vec3f::new(1.0, 0.0, 0.0), Vec3f::new(1.0, 1.0, 0.0)];
        let segs = vec![[0, 1], [1, 2]];
        let c: Grid<f32> = create_level_set_tube_complex(&verts, &segs, 0.2, 0.05, 3.0);
        assert!(c.get_value(&Coord::new(20, 10, 0)) < 0.0);
        assert!(c.get_value(&Coord::new(10, 10, 0)) > 0.0);
        assert!((c.get_value(&Coord::new(10, 0, 6)) - 0.1).abs() < 1e-4);
        let r: Grid<f32> = create_level_set_tube_complex_with_radii(
            &verts,
            &segs,
            &[0.1f32, 0.3, 0.1],
            0.05,
            3.0,
            TubeRadiiPolicy::Automatic,
        );
        assert!(r.get_value(&Coord::new(20, 0, 0)) < 0.0);
    }
}
