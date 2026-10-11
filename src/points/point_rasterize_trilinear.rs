//! Trilinear rasterization of point attributes into grids (port of
//! `openvdb/points/PointRasterizeTrilinear.h`).
//!
//! Each point splats its attribute value onto the 8 surrounding voxel centers (cell
//! centered) or, for the staggered variant, the 3 × 8 surrounding face centers of a MAC
//! grid, with trilinear weights; every voxel then holds the weighted average. The result
//! has the points' index space; its topology is the (filtered) active voxels of the points
//! dilated by one voxel.

use super::attribute_array::AttributeHandle;
use super::codec::AttributeValue;
use super::index_filter::{BoundFilter, IndexFilter};
use super::index_iterator::IndexState;
use super::point_data_grid::{PointDataTree, PointLeaf};
use super::point_mask::convert_points_to_mask_tree;
use super::{Error, Result};
use crate::math::{Coord, CoordBBox, Vec3, Vec3f};
use crate::tree::{LeafNode, Node, Tree};
use crate::types::Value;
use std::ops::{Add, Div, Mul};

/// Attribute types that can be rasterized trilinearly: floating-point scalars and vectors.
pub trait TrilinearSource: AttributeValue + Value + Add<Output = Self> {
    /// The scalar element type.
    type Elem: num_traits::Float + Value + Send + Sync + Mul<Vec3<Self::Elem>, Output = Vec3<Self::Elem>>;
    /// `self * w`.
    fn scale(self, w: Self::Elem) -> Self;
    /// `self / w`.
    fn div(self, w: Self::Elem) -> Self;
    /// The staggered contribution `self * r` (component-wise for vectors).
    fn stagger(self, r: Vec3<Self::Elem>) -> Vec3<Self::Elem>;
}

macro_rules! impl_trilinear_scalar {
    ($($t:ty),*) => {$(
        impl TrilinearSource for $t {
            type Elem = $t;
            fn scale(self, w: $t) -> Self { self * w }
            fn div(self, w: $t) -> Self { self / w }
            fn stagger(self, r: Vec3<$t>) -> Vec3<$t> { r * self }
        }
    )*};
}
impl_trilinear_scalar!(f32, f64);

macro_rules! impl_trilinear_vec {
    ($($e:ty),*) => {$(
        impl TrilinearSource for Vec3<$e> {
            type Elem = $e;
            fn scale(self, w: $e) -> Self { self * w }
            fn div(self, w: $e) -> Self { self / w }
            fn stagger(self, r: Vec3<$e>) -> Vec3<$e> { self.mul_elem(&r) }
        }
    )*};
}
impl_trilinear_vec!(f32, f64);

/// Numeric conversion to the element type.
#[inline]
fn cast<E: num_traits::Float>(x: impl num_traits::ToPrimitive) -> E {
    <E as num_traits::NumCast>::from(x).unwrap()
}

/// The tent kernel `max(0, 1 - |x|)`.
#[inline]
fn tent<E: num_traits::Float>(x: E) -> E {
    let a = x.abs();
    if a < E::one() {
        E::one() - a
    } else {
        E::zero()
    }
}

/// The target topology: (filtered) active point voxels dilated by one voxel in all 26
/// directions (`dilateActiveValues(NN_FACE_EDGE_VERTEX, EXPAND_TILES)`).
fn target_topology<V: Value, F: IndexFilter>(points: &PointDataTree, filter: &F) -> Tree<V> {
    let mut seed = Tree::<V>::new(V::default());
    if filter.state() == IndexState::All {
        seed.topology_union(points.offsets());
    } else {
        seed.topology_union(&convert_points_to_mask_tree(points, filter));
    }
    seed.voxelize_active_tiles();
    let mut out = Tree::<V>::new(V::default());
    for leaf in seed.leaves() {
        for n in leaf.value_mask().iter_on() {
            let c = leaf.offset_to_global_coord(n);
            out.activate_region(&CoordBBox::new(c.offset(-1), c.offset(1)));
        }
    }
    out.voxelize_active_tiles();
    out
}

/// Visit the points (accepted by `filter`) in the voxels of `search`, calling `f(ijk, P,
/// leaf, index)` with the voxel-space position.
fn for_each_nearby_point<'p, F: IndexFilter>(
    points: &'p PointDataTree,
    search: CoordBBox,
    filter: &F,
    mut f: impl FnMut(&super::point_data_grid::PointDataLeaf<'p>, Coord, u32),
) {
    let min = search.min.align_down(8);
    let max = search.max;
    let mut x = min[0];
    while x <= max[0] {
        let mut y = min[1];
        while y <= max[1] {
            let mut z = min[2];
            while z <= max[2] {
                let origin = Coord::new(x, y, z);
                let pbox = CoordBBox::create_cube(origin, 8).intersection(&search);
                if let (false, Some(pl)) = (pbox.is_empty(), points.probe_leaf(&origin)) {
                    let mut bound = filter.bind(pl.leaf, pl.attributes);
                    for c in pbox.iter() {
                        let n = LeafNode::<u32>::coord_to_offset(&c);
                        let active = pl.leaf.is_value_on_at(n);
                        for id in pl.voxel_range(n) {
                            if bound.valid(id, &c, active) {
                                f(&pl, c, id);
                            }
                        }
                    }
                }
                z += 8;
            }
            y += 8;
        }
        x += 8;
    }
}

/// Position and source handles of the current point leaf.
type Handles<'p, T> = Option<(Coord, AttributeHandle<'p, Vec3f>, AttributeHandle<'p, T>)>;

/// The handles of point leaf `pl`, rebinding the cache when the leaf changes.
fn handles<'c, 'p, T: TrilinearSource>(
    cache: &'c mut Handles<'p, T>,
    pl: &super::point_data_grid::PointDataLeaf<'p>,
    pidx: usize,
    sidx: usize,
) -> Option<&'c (Coord, AttributeHandle<'p, Vec3f>, AttributeHandle<'p, T>)> {
    if cache.as_ref().map(|c| c.0) != Some(pl.origin()) {
        *cache = match (pl.attributes.get(pidx), pl.attributes.get(sidx)) {
            (Some(pa), Some(sa)) => match (AttributeHandle::new(pa), AttributeHandle::new(sa)) {
                (Ok(a), Ok(b)) => Some((pl.origin(), a, b)),
                _ => None,
            },
            _ => None,
        };
    }
    cache.as_ref()
}

fn indices(points: &PointDataTree, attribute: &str) -> Result<Option<(usize, usize)>> {
    let Some(d) = points.descriptor() else { return Ok(None) };
    let p = d.position("P").ok_or_else(|| Error::Lookup("failed to find position attribute".into()))?;
    let s = d.position(attribute).ok_or_else(|| Error::Lookup("failed to find source attribute".into()))?;
    Ok(Some((p, s)))
}

/// Cell-centered trilinear rasterization of `attribute` (`rasterizeTrilinear<false, T>()`).
pub fn rasterize_trilinear<T: TrilinearSource, F: IndexFilter>(
    points: &PointDataTree,
    attribute: &str,
    filter: &F,
) -> Result<Tree<T>> {
    let Some((pidx, sidx)) = indices(points, attribute)? else { return Ok(Tree::new(T::default())) };
    let mut tree = target_topology::<T, F>(points, filter);
    let zero = <T::Elem as num_traits::Zero>::zero();
    tree.par_for_each_leaf_mut(|leaf| {
        let bounds = match leaf.value_mask().is_constant() {
            Some(false) => return,
            Some(true) => leaf.node_bbox(),
            None => leaf.active_bbox(),
        };
        let mut weights = [zero; 512];
        let (buf, mask) = leaf.buffer_and_mask_mut();
        let mut cache: Handles<'_, T> = None;
        for_each_nearby_point(points, bounds.expanded(1), filter, |pl, ijk, id| {
            let Some((_, ph, sh)) = handles(&mut cache, pl, pidx, sidx) else { return };
            let p = ph.get(id);
            let s = sh.get(id);
            let mut b = CoordBBox::new(ijk, ijk);
            for k in 0..3 {
                if p[k] < 0.0 {
                    b.min.0[k] -= 1;
                } else {
                    b.max.0[k] += 1;
                }
            }
            let b = b.intersection(&bounds);
            if b.is_empty() {
                return;
            }
            let pe = p.map(cast::<T::Elem>);
            for c in b.iter() {
                let offset = LeafNode::<T>::coord_to_offset(&c);
                if !mask.is_on(offset) {
                    continue;
                }
                let d = c - ijk;
                let w = tent(pe[0] - cast::<T::Elem>(d[0]))
                    * tent(pe[1] - cast::<T::Elem>(d[1]))
                    * tent(pe[2] - cast::<T::Elem>(d[2]));
                buf[offset] = buf[offset] + s.scale(w);
                weights[offset] = weights[offset] + w;
            }
        });
        for n in mask.iter_on() {
            if weights[n] != zero {
                buf[n] = buf[n].div(weights[n]);
            }
        }
    });
    Ok(tree)
}

/// Staggered (MAC grid) trilinear rasterization of `attribute`; scalar attributes produce a
/// vector grid (`rasterizeTrilinear<true, T>()`).
pub fn rasterize_trilinear_staggered<T: TrilinearSource, F: IndexFilter>(
    points: &PointDataTree,
    attribute: &str,
    filter: &F,
) -> Result<Tree<Vec3<T::Elem>>>
where
    Vec3<T::Elem>: Value + Add<Output = Vec3<T::Elem>> + Div<Output = Vec3<T::Elem>>,
{
    let Some((pidx, sidx)) = indices(points, attribute)? else {
        return Ok(Tree::new(Vec3::splat(<T::Elem as num_traits::Zero>::zero())));
    };
    let mut tree = target_topology::<Vec3<T::Elem>, F>(points, filter);
    let zero = <T::Elem as num_traits::Zero>::zero();
    let half = cast::<T::Elem>(0.5);
    tree.par_for_each_leaf_mut(|leaf| {
        let bounds = match leaf.value_mask().is_constant() {
            Some(false) => return,
            Some(true) => leaf.node_bbox(),
            None => leaf.active_bbox(),
        };
        let mut weights = [Vec3::splat(zero); 512];
        let (buf, mask) = leaf.buffer_and_mask_mut();
        let mut cache: Handles<'_, T> = None;
        for_each_nearby_point(points, bounds.expanded(1), filter, |pl, ijk, id| {
            let Some((_, ph, sh)) = handles(&mut cache, pl, pidx, sidx) else { return };
            let b = CoordBBox::new(ijk.offset(-1), ijk.offset(1)).intersection(&bounds);
            if b.is_empty() {
                return;
            }
            let p = ph.get(id).map(cast::<T::Elem>);
            let s = sh.get(id);
            for c in b.iter() {
                let offset = LeafNode::<u32>::coord_to_offset(&c);
                if !mask.is_on(offset) {
                    continue;
                }
                let d = c - ijk;
                let (x, y, z) = (cast::<T::Elem>(d[0]), cast::<T::Elem>(d[1]), cast::<T::Elem>(d[2]));
                let center = Vec3::new(tent(p[0] - x), tent(p[1] - y), tent(p[2] - z));
                let mac = Vec3::new(tent(p[0] - (x - half)), tent(p[1] - (y - half)), tent(p[2] - (z - half)));
                let r = Vec3::new(
                    mac[0] * center[1] * center[2],
                    mac[1] * center[0] * center[2],
                    mac[2] * center[0] * center[1],
                );
                buf[offset] += s.stagger(r);
                weights[offset] += r;
            }
        });
        for n in mask.iter_on() {
            let w = weights[n];
            let mut v = buf[n];
            for k in 0..3 {
                if w[k] != zero {
                    v.0[k] = v.0[k] / w[k];
                }
            }
            buf[n] = v;
        }
    });
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Transform;
    use crate::points::{append_typed_attribute, create_point_data_grid, Codec, NullFilter};

    #[test]
    fn trilinear_constant_field() {
        let pos: Vec<Vec3f> =
            (0..400).map(|i| Vec3f::new((i % 20) as f32 * 0.13, (i / 20) as f32 * 0.11, 0.3)).collect();
        let mut grid = create_point_data_grid(&pos, Codec::Null, &Transform::linear(0.1)).unwrap();
        append_typed_attribute(grid.tree_mut(), "v", Codec::Null, 2.5f32, 1, None, false, false).unwrap();
        append_typed_attribute(grid.tree_mut(), "c", Codec::Null, Vec3f::new(1.0, 2.0, 3.0), 1, None, false, false)
            .unwrap();
        let t = rasterize_trilinear::<f32, _>(grid.tree(), "v", &NullFilter).unwrap();
        assert!(t.active_voxel_count() > grid.tree().active_voxel_count());
        // A constant attribute rasterizes to the same constant wherever any weight landed.
        for (_, v) in t.active_voxels() {
            assert!(v == 0.0 || (v - 2.5).abs() < 1e-5, "{}", v);
        }
        let s = rasterize_trilinear_staggered::<Vec3f, _>(grid.tree(), "c", &NullFilter).unwrap();
        let mut hits = 0;
        for (_, v) in s.active_voxels() {
            for (k, e) in [1.0f32, 2.0, 3.0].iter().enumerate() {
                assert!(v[k] == 0.0 || (v[k] - e).abs() < 1e-5, "{:?}", v);
            }
            hits += (v[0] != 0.0) as u32;
        }
        assert!(hits > 0);
        let s2 = rasterize_trilinear_staggered::<f32, _>(grid.tree(), "v", &NullFilter).unwrap();
        assert_eq!(s2.active_voxel_count(), s.active_voxel_count());
        assert!(rasterize_trilinear::<f32, _>(grid.tree(), "missing", &NullFilter).is_err());
    }
}
