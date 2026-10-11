//! Small shared helpers used by the mesh, particle and point tools.
//!
//! These are private ports of a few pieces of OpenVDB that the mesh/point tools depend on
//! (`math/Proximity`, `util::COORD_OFFSETS`, `tools::signedFloodFillWithValues`,
//! `tools::pruneLevelSet`, `Tree::merge`), kept local to avoid coupling with other modules.

use crate::math::{Coord, CoordBBox, Vec3, Vec3d};
use crate::tree::{Internal1, Internal2, LeafNode, Node, RootEntry, Tree};
use crate::types::{RealValue, Value};
use num_traits::Float;

/// `util::INVALID_IDX` (the maximum 32-bit unsigned value).
pub const INVALID_IDX: u32 = u32::MAX;

/// OpenVDB's `util::COORD_OFFSETS`: the 26 neighbour offsets (6 face, 12 edge, 8 corner).
pub const COORD_OFFSETS: [Coord; 26] = [
    Coord::new(1, 0, 0),
    Coord::new(-1, 0, 0),
    Coord::new(0, 1, 0),
    Coord::new(0, -1, 0),
    Coord::new(0, 0, 1),
    Coord::new(0, 0, -1),
    Coord::new(1, 0, -1),
    Coord::new(-1, 0, -1),
    Coord::new(1, 0, 1),
    Coord::new(-1, 0, 1),
    Coord::new(1, 1, 0),
    Coord::new(-1, 1, 0),
    Coord::new(1, -1, 0),
    Coord::new(-1, -1, 0),
    Coord::new(0, -1, 1),
    Coord::new(0, -1, -1),
    Coord::new(0, 1, 1),
    Coord::new(0, 1, -1),
    Coord::new(-1, -1, -1),
    Coord::new(-1, -1, 1),
    Coord::new(1, -1, 1),
    Coord::new(1, -1, -1),
    Coord::new(-1, 1, -1),
    Coord::new(-1, 1, 1),
    Coord::new(1, 1, 1),
    Coord::new(1, 1, -1),
];

/// `math::isRelOrApproxEqual(a, b, absTol, relTol)`.
#[inline]
pub fn is_rel_or_approx_equal(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
    if !((a - b).abs() > abs_tol) {
        return true;
    }
    let rel = if b.abs() > a.abs() { ((a - b) / b).abs() } else { ((a - b) / a).abs() };
    rel <= rel_tol
}

/// `math::isApproxEqual(Vec3d, Vec3d)` (i.e. `Vec3::eq` with a tolerance of 1e-7).
#[inline]
pub fn vec_approx_eq(a: &Vec3d, b: &Vec3d) -> bool {
    (0..3).all(|i| is_rel_or_approx_equal(a[i], b[i], 1e-7, 1e-7))
}

/// `Vec3<T>::normalize(eps = 1e-7)`: returns false (leaving the vector untouched) if the
/// length is within `eps` of zero; otherwise multiplies by the reciprocal length.
#[inline]
pub fn normalize_vec<T: Float>(v: &mut Vec3<T>) -> bool {
    let d = v.length();
    let eps = T::from(1.0e-7).unwrap();
    if !((d - T::zero()).abs() > eps) {
        return false;
    }
    let inv = T::one() / d;
    *v = Vec3([v.0[0] * inv, v.0[1] * inv, v.0[2] * inv]);
    true
}

/// `math::closestPointOnSegmentToPoint`; returns `(point, t)`.
pub fn closest_point_on_segment_to_point(a: &Vec3d, b: &Vec3d, p: &Vec3d) -> (Vec3d, f64) {
    let ab = *b - *a;
    let mut t = (*p - *a).dot(&ab);
    if t <= 0.0 {
        (*a, 0.0)
    } else {
        let denom = ab.dot(&ab);
        if t >= denom {
            (*b, 1.0)
        } else {
            t /= denom;
            (*a + ab * t, t)
        }
    }
}

/// `math::closestPointOnTriangleToPoint`; returns `(point, uvw)` where `uvw` are the
/// barycentric coordinates of the closest point.
pub fn closest_point_on_triangle_to_point(a: &Vec3d, b: &Vec3d, c: &Vec3d, p: &Vec3d) -> (Vec3d, Vec3d) {
    let mut uvw = Vec3d::zero();
    let (a, b, c, p) = (*a, *b, *c, *p);

    if vec_approx_eq(&a, &b) && vec_approx_eq(&a, &c) {
        uvw[0] = 1.0;
        return (a, uvw);
    }

    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);

    if vec_approx_eq(&a, &b) {
        let (cp, t) = closest_point_on_segment_to_point(&a, &c, &p);
        uvw[0] = 1.0 - t;
        uvw[2] = t;
        return (cp, uvw);
    } else if vec_approx_eq(&a, &c) || vec_approx_eq(&b, &c) {
        let (cp, t) = closest_point_on_segment_to_point(&a, &b, &p);
        uvw[0] = 1.0 - t;
        uvw[1] = t;
        return (cp, uvw);
    }

    if d1 <= 0.0 && d2 <= 0.0 {
        uvw[0] = 1.0;
        return (a, uvw);
    }

    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        uvw[1] = 1.0;
        return (b, uvw);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        uvw[1] = d1 / (d1 - d3);
        uvw[0] = 1.0 - uvw[1];
        return (a + ab * uvw[1], uvw);
    }

    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        uvw[2] = 1.0;
        return (c, uvw);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        uvw[2] = d2 / (d2 - d6);
        uvw[0] = 1.0 - uvw[2];
        return (a + ac * uvw[2], uvw);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        uvw[2] = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        uvw[1] = 1.0 - uvw[2];
        return (b + (c - b) * uvw[2], uvw);
    }

    let denom = 1.0 / (va + vb + vc);
    uvw[2] = vc * denom;
    uvw[1] = vb * denom;
    uvw[0] = 1.0 - uvw[1] - uvw[2];
    (a + ab * uvw[1] + ac * uvw[2], uvw)
}

/// First value of a node (the value at its origin).
#[inline]
fn first_value<N: Node>(node: &N) -> N::Value {
    *node.get_value(&node.origin())
}

/// Last value of a node (the value at its maximum corner).
#[inline]
fn last_value<N: Node>(node: &N) -> N::Value {
    *node.get_value(&node.origin().offset(N::DIM - 1))
}

fn flood_fill_leaf<T: RealValue>(leaf: &mut LeafNode<T>, outside: T, inside: T) {
    let first = leaf.value_mask().find_first_on();
    let (buffer, mask) = leaf.buffer_and_mask_mut();
    match first {
        Some(first) => {
            let mut x_inside = buffer[first] < <T as Value>::zero();
            for x in 0..8usize {
                let x00 = x << 6;
                if mask.is_on(x00) {
                    x_inside = buffer[x00] < <T as Value>::zero();
                }
                let mut y_inside = x_inside;
                for y in 0..8usize {
                    let xy0 = x00 + (y << 3);
                    if mask.is_on(xy0) {
                        y_inside = buffer[xy0] < <T as Value>::zero();
                    }
                    let mut z_inside = y_inside;
                    for z in 0..8usize {
                        let xyz = xy0 + z;
                        if mask.is_on(xyz) {
                            z_inside = buffer[xyz] < <T as Value>::zero();
                        } else {
                            buffer[xyz] = if z_inside { inside } else { outside };
                        }
                    }
                }
            }
        }
        None => {
            let v = if buffer[0] < <T as Value>::zero() { inside } else { outside };
            buffer.iter_mut().for_each(|b| *b = v);
        }
    }
}

fn flood_fill_internal<C: Node, const L: u32>(
    node: &mut crate::tree::InternalNode<C, L>,
    outside: C::Value,
    inside: C::Value,
) where
    C::Value: RealValue,
{
    let dim = 1usize << L;
    let first = node.child_mask().find_first_on();
    match first {
        Some(first) => {
            let mut x_inside = first_value(node.child(first).unwrap()) < <C::Value as Value>::zero();
            for x in 0..dim {
                let x00 = x << (2 * L);
                if node.is_child(x00) {
                    x_inside = last_value(node.child(x00).unwrap()) < <C::Value as Value>::zero();
                }
                let mut y_inside = x_inside;
                for y in 0..dim {
                    let xy0 = x00 + (y << L);
                    if node.is_child(xy0) {
                        y_inside = last_value(node.child(xy0).unwrap()) < <C::Value as Value>::zero();
                    }
                    let mut z_inside = y_inside;
                    for z in 0..dim {
                        let xyz = xy0 + z;
                        if node.is_child(xyz) {
                            z_inside = last_value(node.child(xyz).unwrap()) < <C::Value as Value>::zero();
                        } else {
                            node.values_mut()[xyz] = if z_inside { inside } else { outside };
                        }
                    }
                }
            }
        }
        None => {
            let v = if node.values()[0] < <C::Value as Value>::zero() { inside } else { outside };
            node.values_mut().iter_mut().for_each(|t| *t = v);
        }
    }
}

/// Port of `tools::signedFloodFillWithValues`: sets every inactive voxel and tile of a
/// narrow-band level set to `|outside|` or `-|inside|` according to the sign of the nearest
/// active value along scan lines, and sets the background to `|outside|`.
pub fn signed_flood_fill_with_values<T: RealValue>(tree: &mut Tree<T>, outside: T, inside: T) {
    use rayon::prelude::*;
    let outside = outside.abs();
    let inside = -inside.abs();
    tree.par_for_each_leaf_mut(|leaf| flood_fill_leaf(leaf, outside, inside));
    {
        let mut n1: Vec<&mut Internal1<T>> = Vec::new();
        for c2 in tree.root_mut().children_mut() {
            for c1 in c2.children_mut() {
                n1.push(c1);
            }
        }
        n1.into_par_iter().for_each(|n| flood_fill_internal(n, outside, inside));
    }
    {
        let n2: Vec<&mut Internal2<T>> = tree.root_mut().children_mut().collect();
        n2.into_par_iter().for_each(|n| flood_fill_internal(n, outside, inside));
    }
    // Root: insert inside tiles between z-adjacent inside children.
    let root = tree.root_mut();
    let dim = Internal2::<T>::DIM;
    let keys: Vec<(Coord, T, T)> =
        root.table().iter().filter_map(|(k, e)| e.child().map(|c| (*k, first_value(c), last_value(c)))).collect();
    let mut tiles = Vec::new();
    for w in keys.windows(2) {
        let (a, _, a_last) = w[0];
        let (b, b_first, _) = w[1];
        let d = b - a;
        if d[0] != 0 || d[1] != 0 || d[2] == dim {
            continue;
        }
        if !(a_last < <T as Value>::zero()) || !(b_first < <T as Value>::zero()) {
            continue;
        }
        let mut c = a + Coord::new(0, 0, dim);
        while c[2] != b[2] {
            tiles.push(c);
            c[2] += dim;
        }
    }
    for c in tiles {
        root.add_root_tile(&c, inside, false);
    }
    root.set_background(outside, false);
}

/// Port of `tools::pruneLevelSet(tree, outside, inside)`: replaces every node whose values
/// are all inactive by an inactive tile of `inside` (if its first value is negative) or
/// `outside`, then erases background tiles at the root.
pub fn prune_level_set<T: RealValue>(tree: &mut Tree<T>, outside: T, inside: T) {
    use rayon::prelude::*;
    let tile_value = |v: T| if v < <T as Value>::zero() { inside } else { outside };
    {
        let mut n1: Vec<&mut Internal1<T>> = Vec::new();
        for c2 in tree.root_mut().children_mut() {
            for c1 in c2.children_mut() {
                n1.push(c1);
            }
        }
        n1.into_par_iter().for_each(|node| {
            let idx: Vec<usize> = node.children().filter(|(_, c)| c.is_inactive()).map(|(i, _)| i).collect();
            for i in idx {
                let v = tile_value(first_value(node.child(i).unwrap()));
                node.set_tile(i, v, false);
            }
        });
    }
    {
        let n2: Vec<&mut Internal2<T>> = tree.root_mut().children_mut().collect();
        n2.into_par_iter().for_each(|node| {
            let idx: Vec<usize> = node.children().filter(|(_, c)| c.is_inactive()).map(|(i, _)| i).collect();
            for i in idx {
                let v = tile_value(first_value(node.child(i).unwrap()));
                node.set_tile(i, v, false);
            }
        });
    }
    let root = tree.root_mut();
    for e in root.table_mut().values_mut() {
        if let RootEntry::Child(c) = e {
            if c.is_inactive() {
                let v = tile_value(first_value(c.as_ref()));
                *e = RootEntry::Tile { value: v, active: false };
            }
        }
    }
    root.erase_background_tiles();
}

/// Move all leaves out of a tree (consuming it), in index order.
pub fn take_leaves<T: Value>(tree: Tree<T>) -> Vec<Box<LeafNode<T>>> {
    let mut out = Vec::new();
    let mut tree = tree;
    for c2 in tree.root_mut().children_mut() {
        for c1 in c2.children_mut() {
            let idx: Vec<usize> = c1.children().map(|(i, _)| i).collect();
            for i in idx {
                if let Some(l) = c1.set_tile(i, <T as Value>::zero(), false) {
                    out.push(l);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_point_triangle() {
        let a = Vec3d::new(0.0, 0.0, 0.0);
        let b = Vec3d::new(1.0, 0.0, 0.0);
        let c = Vec3d::new(0.0, 1.0, 0.0);
        let (p, uvw) = closest_point_on_triangle_to_point(&a, &b, &c, &Vec3d::new(0.25, 0.25, 1.0));
        assert!((p - Vec3d::new(0.25, 0.25, 0.0)).length() < 1e-12);
        assert!((uvw.sum() - 1.0).abs() < 1e-12);
        let (p, _) = closest_point_on_triangle_to_point(&a, &b, &c, &Vec3d::new(-1.0, -1.0, 0.0));
        assert_eq!(p, a);
        let (p, _) = closest_point_on_triangle_to_point(&a, &b, &c, &Vec3d::new(2.0, 2.0, 0.0));
        assert!((p - Vec3d::new(0.5, 0.5, 0.0)).length() < 1e-12);
    }

    #[test]
    fn flood_fill_and_prune() {
        let mut tree = Tree::<f32>::new(3.0);
        // A shell of active voxels around the origin.
        for x in -20..=20 {
            for y in -20..=20 {
                for z in -20..=20 {
                    let r = ((x * x + y * y + z * z) as f32).sqrt() - 15.0;
                    if r.abs() < 2.0 {
                        tree.set_value(&Coord::new(x, y, z), r);
                    }
                }
            }
        }
        signed_flood_fill_with_values(&mut tree, 3.0, -3.0);
        assert!(tree.get_value(&Coord::new(0, 0, 0)) < 0.0);
        assert!(tree.get_value(&Coord::new(40, 0, 0)) > 0.0);
        assert_eq!(tree.get_value(&Coord::new(1, 1, 1)), -3.0);
        prune_level_set(&mut tree, 3.0, -3.0);
        assert!(tree.get_value(&Coord::new(0, 0, 0)) < 0.0);
    }
}

/// An active value visited by [`visit_values_on_ordered`].
#[derive(Clone, Copy, Debug)]
pub enum ActiveValue<T> {
    /// An active leaf voxel.
    Voxel(Coord, T),
    /// An active tile covering `bbox`.
    Tile(CoordBBox, T),
}

/// Visit every active value (voxels and tiles) in the order of OpenVDB's
/// `Tree::cbeginValueOn()` iterator: depth first, with tiles and child nodes interleaved in
/// slot order at every level.
pub fn visit_values_on_ordered<T: Value>(tree: &Tree<T>, f: &mut dyn FnMut(ActiveValue<T>)) {
    fn visit_internal<C: Node, const L: u32>(
        node: &crate::tree::InternalNode<C, L>,
        f: &mut dyn FnMut(ActiveValue<C::Value>),
        child: &mut dyn FnMut(&C, &mut dyn FnMut(ActiveValue<C::Value>)),
    ) {
        for n in 0..(1usize << (3 * L)) {
            if let Some(c) = node.child(n) {
                child(c, f);
            } else if node.value_mask().is_on(n) {
                let o = node.offset_to_global_coord(n);
                f(ActiveValue::Tile(CoordBBox::create_cube(o, C::DIM), node.values()[n]));
            }
        }
    }
    for (key, e) in tree.root().table() {
        match e {
            RootEntry::Tile { value, active } => {
                if *active {
                    f(ActiveValue::Tile(CoordBBox::create_cube(*key, Internal2::<T>::DIM), *value));
                }
            }
            RootEntry::Child(n2) => {
                visit_internal(n2.as_ref(), f, &mut |n1: &Internal1<T>, f| {
                    visit_internal(n1, f, &mut |leaf: &LeafNode<T>, f| {
                        for (pos, v) in leaf.iter_on() {
                            f(ActiveValue::Voxel(leaf.offset_to_global_coord(pos), *v));
                        }
                    });
                });
            }
        }
    }
}
