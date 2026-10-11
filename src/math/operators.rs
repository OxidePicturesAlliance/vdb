//! Differential operators in index space and world space (port of `openvdb/math/Operators.h`).
//!
//! The `is_*` functions are the index-space operators (`ISGradient`, `ISLaplacian`, ...); they
//! read values through [`StencilValues`], so they work both on stencils and on any
//! [`ValueSource`](crate::tree::ValueSource) wrapped in an
//! [`AccessorAt`](super::finite_difference::AccessorAt).
//!
//! The map-aware operators (`gradient`, `laplacian`, `divergence`, ...) additionally take a
//! [`MapJacobian`], built once from a grid's [`Transform`], that converts index-space
//! derivatives to world space. For linear transforms the inverse Jacobian is precomputed (and
//! scale-only maps use a fast diagonal path, equivalent to OpenVDB's `ScaleMap` and
//! `UniformScaleMap` specialisations); for non-linear transforms it is evaluated numerically at
//! the stencil center.

use super::finite_difference::{
    d1, d1_vec, d2, d2_cross, godunovs_norm_sqrd_vec, BiasedGradientScheme, DDScheme, DScheme, StencilValues,
};
use super::{Mat3d, Transform, Vec3, Vec3d};
use crate::types::{RealValue, Value};

/// The inverse Jacobian of a transform's index-to-world map, used to convert index-space
/// derivatives to world space (`MapBase::applyIJT` / `applyIJC`).
#[derive(Clone, Debug)]
pub struct MapJacobian {
    transform: Transform,
    linear: bool,
    /// `J^{-T}` in OpenVDB's row-vector convention (`AffineMap::mJacobianInv`).
    jinv_t: Mat3d,
    /// Per-axis inverse scale if the linear map has a diagonal Jacobian.
    inv_scale: Option<Vec3d>,
}

fn jacobian_at(t: &Transform, ijk: Vec3d) -> Mat3d {
    let mut m = Mat3d::zero();
    if t.is_linear() {
        let o = t.index_to_world(Vec3d::zero());
        for i in 0..3 {
            let mut e = Vec3d::zero();
            e[i] = 1.0;
            m.set_row(i, t.index_to_world(e) - o);
        }
    } else {
        let h = 1e-3;
        for i in 0..3 {
            let mut e = Vec3d::zero();
            e[i] = h;
            m.set_row(i, (t.index_to_world(ijk + e) - t.index_to_world(ijk - e)) / (2.0 * h));
        }
    }
    m
}

fn inv_transpose(j: &Mat3d) -> Mat3d {
    j.inverse(0.0).expect("MapJacobian: singular Jacobian").transpose()
}

impl MapJacobian {
    /// Analyse `transform`.
    pub fn new(transform: &Transform) -> Self {
        let linear = transform.is_linear();
        let j = jacobian_at(transform, Vec3d::zero());
        let jinv_t = inv_transpose(&j);
        let scale = j.0[0][0].abs().max(j.0[1][1].abs()).max(j.0[2][2].abs()).max(1e-300);
        let diag = (0..3).all(|r| (0..3).all(|c| r == c || j.0[r][c].abs() <= 1e-12 * scale));
        let inv_scale =
            if linear && diag { Some(Vec3d::new(1.0 / j.0[0][0], 1.0 / j.0[1][1], 1.0 / j.0[2][2])) } else { None };
        MapJacobian { transform: transform.clone(), linear, jinv_t, inv_scale }
    }
    /// The transform this Jacobian was built from.
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
    /// True if the map is linear.
    pub fn is_linear(&self) -> bool {
        self.linear
    }
    /// The per-axis inverse voxel size if the map is a (possibly non-uniform) scale plus
    /// translation.
    pub fn inv_scale(&self) -> Option<Vec3d> {
        self.inv_scale
    }
    /// The inverse voxel size if the map is a uniform scale plus translation.
    pub fn uniform_inv_scale(&self) -> Option<f64> {
        let s = self.inv_scale?;
        let tol = 1e-12 * s[0].abs();
        ((s[0] - s[1]).abs() <= tol && (s[0] - s[2]).abs() <= tol).then_some(s[0])
    }
    #[inline]
    fn jinv_t_at(&self, ijk: Vec3d) -> Mat3d {
        if self.linear {
            self.jinv_t
        } else {
            inv_transpose(&jacobian_at(&self.transform, ijk))
        }
    }
    /// Apply the transpose of the inverse Jacobian at index position `ijk` to an index-space
    /// gradient, giving the world-space gradient (`applyIJT`).
    #[inline]
    pub fn apply_ijt(&self, v: Vec3d, ijk: Vec3d) -> Vec3d {
        if let Some(s) = self.inv_scale {
            return Vec3d::new(v[0] * s[0], v[1] * s[1], v[2] * s[2]);
        }
        self.jinv_t_at(ijk).transform(v)
    }
    /// Convert an index-space Hessian to world space (`applyIJC`; the map's own curvature is
    /// neglected, which is exact for linear maps).
    #[inline]
    pub fn apply_ijc(&self, m: &Mat3d, ijk: Vec3d) -> Mat3d {
        let k = self.jinv_t_at(ijk);
        k.transpose() * *m * k
    }
    /// Map an index-space position to world space.
    #[inline]
    pub fn apply_map(&self, ijk: Vec3d) -> Vec3d {
        self.transform.index_to_world(ijk)
    }
    /// Map a world-space position to index space.
    #[inline]
    pub fn apply_inverse_map(&self, xyz: Vec3d) -> Vec3d {
        self.transform.world_to_index(xyz)
    }
}

#[inline]
fn to_d<T: RealValue>(v: Vec3<T>) -> Vec3d {
    v.map(|x| x.to_f64())
}
#[inline]
fn from_d<T: RealValue>(v: Vec3d) -> Vec3<T> {
    v.map(T::from_f64)
}

// ---------------------------------------------------------------------------------------------
// Index-space operators

/// Index-space gradient (`ISGradient<scheme>`).
#[inline]
pub fn is_gradient<S>(scheme: DScheme, s: &S) -> Vec3<S::Value>
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    Vec3::new(d1(scheme, 0, s), d1(scheme, 1, s), d1(scheme, 2, s))
}

/// Index-space upwind gradient with respect to the velocity `v` (`ISGradientBiased`): the
/// forward difference is used where `v < 0`, the backward difference elsewhere.
#[inline]
pub fn is_gradient_biased<S, B>(scheme: BiasedGradientScheme, s: &S, v: Vec3<B>) -> Vec3<S::Value>
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
    B: RealValue,
{
    let (fd, bd) = (scheme.forward(), scheme.backward());
    let z = <B as Value>::zero();
    Vec3::new(
        if v[0] < z { d1(fd, 0, s) } else { d1(bd, 0, s) },
        if v[1] < z { d1(fd, 1, s) } else { d1(bd, 1, s) },
        if v[2] < z { d1(fd, 2, s) } else { d1(bd, 2, s) },
    )
}

/// Index-space `|∇φ|²` by Godunov's upwind scheme (`ISGradientNormSqrd`).
#[inline]
pub fn is_gradient_norm_sqrd<S>(scheme: BiasedGradientScheme, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    let up = is_gradient(scheme.forward(), s);
    let down = is_gradient(scheme.backward(), s);
    godunovs_norm_sqrd_vec(s.center_value() > <S::Value as Value>::zero(), down, up)
}

/// Index-space Laplacian (`ISLaplacian<scheme>`).
#[inline]
pub fn is_laplacian<S>(scheme: DDScheme, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    type T<S> = <S as StencilValues>::Value;
    let ring = |r: i32| {
        s.value_at(r, 0, 0)
            + s.value_at(-r, 0, 0)
            + s.value_at(0, r, 0)
            + s.value_at(0, -r, 0)
            + s.value_at(0, 0, r)
            + s.value_at(0, 0, -r)
    };
    let c0 = s.center_value();
    match scheme {
        DDScheme::CdSecond => ring(1) - T::<S>::from_f64(6.0) * c0,
        DDScheme::CdFourth => {
            T::<S>::from_f64((-1.0 / 12.0) * ring(2).to_f64() + (4.0 / 3.0) * ring(1).to_f64() - 7.5 * c0.to_f64())
        }
        DDScheme::CdSixth => T::<S>::from_f64(
            (1.0 / 90.0) * ring(3).to_f64() - (3.0 / 20.0) * ring(2).to_f64() + 1.5 * ring(1).to_f64()
                - (3.0 * 49.0 / 18.0) * c0.to_f64(),
        ),
    }
}

/// Index-space divergence of a vector field (`ISDivergence<scheme>`).
#[inline]
pub fn is_divergence<S, T>(scheme: DScheme, s: &S) -> T
where
    S: StencilValues<Value = Vec3<T>> + ?Sized,
    T: RealValue,
{
    d1_vec(scheme, 0, s, 0) + d1_vec(scheme, 1, s, 1) + d1_vec(scheme, 2, s, 2)
}

/// Index-space curl of a vector field (`ISCurl<scheme>`).
#[inline]
pub fn is_curl<S, T>(scheme: DScheme, s: &S) -> Vec3<T>
where
    S: StencilValues<Value = Vec3<T>> + ?Sized,
    T: RealValue,
{
    Vec3::new(
        d1_vec(scheme, 1, s, 2) - d1_vec(scheme, 2, s, 1),
        d1_vec(scheme, 2, s, 0) - d1_vec(scheme, 0, s, 2),
        d1_vec(scheme, 0, s, 1) - d1_vec(scheme, 1, s, 0),
    )
}

/// Index-space mean curvature terms (`ISMeanCurvature`): returns `(alpha, beta)` such that the
/// mean curvature is `alpha / (2 beta³)` and the mean curvature times `|∇φ|` is
/// `alpha / (2 beta²)`, or `None` where the gradient vanishes.
#[inline]
pub fn is_mean_curvature<S>(scheme2: DDScheme, scheme1: DScheme, s: &S) -> Option<(S::Value, S::Value)>
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    type T<S> = <S as StencilValues>::Value;
    let (dx, dy, dz) = (d1(scheme1, 0, s), d1(scheme1, 1, s), d1(scheme1, 2, s));
    let (dx2, dy2, dz2) = (dx * dx, dy * dy, dz * dz);
    let norm_grad = dx2 + dy2 + dz2;
    if norm_grad <= T::<S>::tolerance() {
        return None;
    }
    let dxx = d2(scheme2, 0, s);
    let dyy = d2(scheme2, 1, s);
    let dzz = d2(scheme2, 2, s);
    let dxy = d2_cross(scheme2, 0, 1, s);
    let dyz = d2_cross(scheme2, 1, 2, s);
    let dxz = d2_cross(scheme2, 0, 2, s);
    let two = T::<S>::from_f64(2.0);
    let alpha =
        dx2 * (dyy + dzz) + dy2 * (dxx + dzz) + dz2 * (dxx + dyy) - two * (dx * (dy * dxy + dz * dxz) + dy * dz * dyz);
    let beta = T::<S>::from_f64(norm_grad.to_f64().sqrt());
    Some((alpha, beta))
}

fn is_hessian<S>(scheme2: DDScheme, s: &S) -> Mat3d
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    let dxx = d2(scheme2, 0, s).to_f64();
    let dyy = d2(scheme2, 1, s).to_f64();
    let dzz = d2(scheme2, 2, s).to_f64();
    let dxy = d2_cross(scheme2, 0, 1, s).to_f64();
    let dyz = d2_cross(scheme2, 1, 2, s).to_f64();
    let dxz = d2_cross(scheme2, 0, 2, s).to_f64();
    Mat3d::from_rows(Vec3d::new(dxx, dxy, dxz), Vec3d::new(dxy, dyy, dyz), Vec3d::new(dxz, dyz, dzz))
}

// ---------------------------------------------------------------------------------------------
// World-space operators

/// World-space gradient (`Gradient<MapType, scheme>`).
#[inline]
pub fn gradient<S>(map: &MapJacobian, scheme: DScheme, s: &S) -> Vec3<S::Value>
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    if scheme == DScheme::Cd2nd {
        if let Some(inv) = map.inv_scale {
            // ScaleMap/UniformScaleMap specialisation: CD_2NDT times 1/(2 dx).
            let g = is_gradient(DScheme::Cd2ndT, s);
            return Vec3::new(
                g[0] * S::Value::from_f64(0.5 * inv[0]),
                g[1] * S::Value::from_f64(0.5 * inv[1]),
                g[2] * S::Value::from_f64(0.5 * inv[2]),
            );
        }
    }
    let g = is_gradient(scheme, s);
    from_d(map.apply_ijt(to_d(g), s.center_coord().as_vec3d()))
}

/// World-space upwind gradient with respect to the velocity `v` (`GradientBiased`).
#[inline]
pub fn gradient_biased<S>(map: &MapJacobian, scheme: BiasedGradientScheme, s: &S, v: Vec3<S::Value>) -> Vec3<S::Value>
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    let g = is_gradient_biased(scheme, s, v);
    from_d(map.apply_ijt(to_d(g), s.center_coord().as_vec3d()))
}

/// World-space `|∇φ|²` by Godunov's scheme (`GradientNormSqrd`).
#[inline]
pub fn gradient_norm_sqrd<S>(map: &MapJacobian, scheme: BiasedGradientScheme, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    if let Some(inv) = map.uniform_inv_scale() {
        return S::Value::from_f64(inv * inv) * is_gradient_norm_sqrd(scheme, s);
    }
    let up = gradient(map, scheme.forward(), s);
    let down = gradient(map, scheme.backward(), s);
    godunovs_norm_sqrd_vec(s.center_value() > <S::Value as Value>::zero(), down, up)
}

/// World-space divergence of a vector field (`Divergence<MapType, scheme>`).
#[inline]
pub fn divergence<S, T>(map: &MapJacobian, scheme: DScheme, s: &S) -> T
where
    S: StencilValues<Value = Vec3<T>> + ?Sized,
    T: RealValue,
{
    if let Some(inv) = map.inv_scale {
        if scheme == DScheme::Cd2nd {
            return T::from_f64(
                d1_vec(DScheme::Cd2ndT, 0, s, 0).to_f64() * 0.5 * inv[0]
                    + d1_vec(DScheme::Cd2ndT, 1, s, 1).to_f64() * 0.5 * inv[1]
                    + d1_vec(DScheme::Cd2ndT, 2, s, 2).to_f64() * 0.5 * inv[2],
            );
        }
        return T::from_f64(
            d1_vec(scheme, 0, s, 0).to_f64() * inv[0]
                + d1_vec(scheme, 1, s, 1).to_f64() * inv[1]
                + d1_vec(scheme, 2, s, 2).to_f64() * inv[2],
        );
    }
    let ijk = s.center_coord().as_vec3d();
    let mut div = <T as Value>::zero();
    for i in 0..3 {
        let v = Vec3d::new(
            d1_vec(scheme, 0, s, i).to_f64(),
            d1_vec(scheme, 1, s, i).to_f64(),
            d1_vec(scheme, 2, s, i).to_f64(),
        );
        div += T::from_f64(map.apply_ijt(v, ijk)[i]);
    }
    div
}

/// World-space curl of a vector field (`Curl<MapType, scheme>`).
#[inline]
pub fn curl<S, T>(map: &MapJacobian, scheme: DScheme, s: &S) -> Vec3<T>
where
    S: StencilValues<Value = Vec3<T>> + ?Sized,
    T: RealValue,
{
    if let Some(inv) = map.uniform_inv_scale() {
        if scheme == DScheme::Cd2nd {
            return is_curl(DScheme::Cd2ndT, s) * T::from_f64(0.5 * inv);
        }
        return is_curl(scheme, s) * T::from_f64(inv);
    }
    let ijk = s.center_coord().as_vec3d();
    let mut mat = [Vec3d::zero(); 3];
    for (i, m) in mat.iter_mut().enumerate() {
        let v = Vec3d::new(
            d1_vec(scheme, 0, s, i).to_f64(),
            d1_vec(scheme, 1, s, i).to_f64(),
            d1_vec(scheme, 2, s, i).to_f64(),
        );
        *m = map.apply_ijt(v, ijk);
    }
    from_d(Vec3d::new(mat[2][1] - mat[1][2], mat[0][2] - mat[2][0], mat[1][0] - mat[0][1]))
}

/// World-space Laplacian (`Laplacian<MapType, scheme>`).
#[inline]
pub fn laplacian<S>(map: &MapJacobian, scheme: DDScheme, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    type T<S> = <S as StencilValues>::Value;
    if let Some(inv) = map.uniform_inv_scale() {
        return is_laplacian(scheme, s) * T::<S>::from_f64(inv * inv);
    }
    if let Some(inv) = map.inv_scale {
        return T::<S>::from_f64(
            d2(scheme, 0, s).to_f64() * inv[0] * inv[0]
                + d2(scheme, 1, s).to_f64() * inv[1] * inv[1]
                + d2(scheme, 2, s).to_f64() * inv[2] * inv[2],
        );
    }
    let h = map.apply_ijc(&is_hessian(scheme, s), s.center_coord().as_vec3d());
    T::<S>::from_f64(h.trace())
}

/// World-space mean curvature terms `(alpha, beta)` (`MeanCurvature::compute`).
fn mean_curvature_terms<S>(map: &MapJacobian, scheme2: DDScheme, scheme1: DScheme, s: &S) -> Option<(f64, f64)>
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    if let Some(inv) = map.uniform_inv_scale() {
        // UniformScaleMap specialisation: index-space terms rescaled.
        let (alpha, beta) = is_mean_curvature(scheme2, scheme1, s)?;
        return Some((alpha.to_f64() * inv * inv * inv * inv, beta.to_f64() * inv));
    }
    let ijk = s.center_coord().as_vec3d();
    let d1_is = to_d(is_gradient(scheme1, s));
    let d1 = map.apply_ijt(d1_is, ijk);
    let (dx2, dy2, dz2) = (d1[0] * d1[0], d1[1] * d1[1], d1[2] * d1[2]);
    let norm_grad = dx2 + dy2 + dz2;
    if norm_grad <= crate::math::TOLERANCE_F64 {
        return None;
    }
    let h = map.apply_ijc(&is_hessian(scheme2, s), ijk);
    let m = &h.0;
    let alpha = dx2 * (m[1][1] + m[2][2]) + dy2 * (m[0][0] + m[2][2]) + dz2 * (m[0][0] + m[1][1])
        - 2.0 * (d1[0] * (d1[1] * m[0][1] + d1[2] * m[0][2]) + d1[1] * d1[2] * m[1][2]);
    Some((alpha, norm_grad.sqrt()))
}

/// World-space mean curvature (`MeanCurvature<MapType, scheme2, scheme1>::result`).
#[inline]
pub fn mean_curvature<S>(map: &MapJacobian, scheme2: DDScheme, scheme1: DScheme, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    match mean_curvature_terms(map, scheme2, scheme1, s) {
        Some((a, b)) => S::Value::from_f64(a / (2.0 * b * b * b)),
        None => <S::Value as Value>::zero(),
    }
}

/// World-space mean curvature times `|∇φ|` (`MeanCurvature::normGrad`).
#[inline]
pub fn mean_curvature_norm_grad<S>(map: &MapJacobian, scheme2: DDScheme, scheme1: DScheme, s: &S) -> S::Value
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    match mean_curvature_terms(map, scheme2, scheme1, s) {
        Some((a, b)) => S::Value::from_f64(a / (2.0 * b * b)),
        None => <S::Value as Value>::zero(),
    }
}

/// Closest point on the zero level set, in index space (`CPT<MapType, scheme>`), assuming the
/// values are signed distances in world units.
#[inline]
pub fn cpt<S>(map: &MapJacobian, scheme: DScheme, s: &S) -> Vec3<S::Value>
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    from_d(map.apply_inverse_map(to_d(cpt_range(map, scheme, s))))
}

/// Closest point on the zero level set, in world space (`CPT_RANGE<MapType, scheme>`).
#[inline]
pub fn cpt_range<S>(map: &MapJacobian, scheme: DScheme, s: &S) -> Vec3<S::Value>
where
    S: StencilValues + ?Sized,
    S::Value: RealValue,
{
    let d = s.center_value();
    let from_surface = gradient(map, scheme, s) * d;
    from_d(map.apply_map(s.center_coord().as_vec3d()) - to_d(from_surface))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::finite_difference::AccessorAt;
    use crate::math::{Coord, CoordBBox, Mat4d};
    use crate::tree::{FloatTree, Vec3STree};

    fn sphere_tree(r: f64, t: &Transform, bbox: CoordBBox) -> FloatTree {
        let mut tree = FloatTree::new(3.0);
        let mut acc = tree.accessor_mut();
        for c in bbox {
            let w = t.index_to_world(c.as_vec3d());
            acc.set_value(&c, (w.length() - r) as f32);
        }
        drop(acc);
        tree
    }

    #[test]
    fn world_gradient_and_laplacian_of_sphere() {
        let mut t = Transform::linear(0.1);
        t.post_translate(Vec3d::new(0.05, -0.1, 0.2));
        let bbox = CoordBBox::from_coords(5, -6, -10, 25, 6, 6);
        let tree = sphere_tree(1.0, &t, bbox);
        let map = MapJacobian::new(&t);
        assert!(map.uniform_inv_scale().is_some());
        let ijk = Coord::new(10, 1, -2);
        let w = t.index_to_world(ijk.as_vec3d());
        let at = AccessorAt::new(&tree, ijk);
        for sch in [DScheme::Cd2nd, DScheme::Cd4th, DScheme::Cd6th, DScheme::FdWeno5] {
            let g = gradient(&map, sch, &at);
            assert!((to_d(g) - w.unit()).length() < 2e-2, "{sch}: {g:?}");
        }
        // Laplacian of |x| - r is 2/|x|.
        let lap = laplacian(&map, DDScheme::CdFourth, &at) as f64;
        assert!((lap - 2.0 / w.length()).abs() < 0.05, "{lap}");
        let mc = mean_curvature(&map, DDScheme::CdSecond, DScheme::Cd2nd, &at) as f64;
        assert!((mc - 1.0 / w.length()).abs() < 0.05, "{mc}");
        // Closest point lies on the sphere.
        let cp = to_d(cpt_range(&map, DScheme::Cd2nd, &at));
        assert!((cp.length() - 1.0).abs() < 1e-2);
        let cpi = to_d(cpt(&map, DScheme::Cd2nd, &at));
        assert!((t.index_to_world(cpi) - cp).length() < 1e-5);
        assert!((gradient_norm_sqrd(&map, BiasedGradientScheme::HjWeno5Bias, &at) - 1.0).abs() < 2e-2);
    }

    #[test]
    fn general_affine_matches_uniform() {
        // A rotated + scaled map: the world-space gradient must not depend on the rotation.
        let mut m = Mat4d::scale(Vec3d::splat(0.1));
        m.post_rotate(crate::math::Axis::Z, 0.3);
        m.post_rotate(crate::math::Axis::X, -0.7);
        let t = Transform::from_matrix(m);
        let map = MapJacobian::new(&t);
        assert!(map.inv_scale().is_none());
        let bbox = CoordBBox::from_coords(-14, -14, -14, 14, 14, 14);
        let tree = sphere_tree(0.8, &t, bbox);
        let ijk = Coord::new(5, 4, -3);
        let at = AccessorAt::new(&tree, ijk);
        let w = t.index_to_world(ijk.as_vec3d());
        let g = to_d(gradient(&map, DScheme::Cd4th, &at));
        assert!((g - w.unit()).length() < 1e-2, "{g:?} vs {:?}", w.unit());
        let lap = laplacian(&map, DDScheme::CdFourth, &at) as f64;
        assert!((lap - 2.0 / w.length()).abs() < 0.05, "{lap}");
        let mc = mean_curvature(&map, DDScheme::CdFourth, DScheme::Cd4th, &at) as f64;
        assert!((mc - 1.0 / w.length()).abs() < 0.05, "{mc}");
    }

    #[test]
    fn divergence_and_curl() {
        // v = (x*y, y*z, z*x) + rotation (−y, x, 0): div = y + z + x, curl = (-y, -z, -x) + (0,0,2)
        let t = Transform::linear(0.5);
        let mut tree = Vec3STree::new(Vec3::zero());
        let mut acc = tree.accessor_mut();
        for c in CoordBBox::from_coords(-4, -4, -4, 4, 4, 4) {
            let p = t.index_to_world(c.as_vec3d());
            let v = Vec3d::new(p[0] * p[1] - p[1], p[1] * p[2] + p[0], p[2] * p[0]);
            acc.set_value(&c, v.map(|x| x as f32));
        }
        drop(acc);
        let map = MapJacobian::new(&t);
        let ijk = Coord::new(1, 2, -1);
        let p = t.index_to_world(ijk.as_vec3d());
        let at = AccessorAt::new(&tree, ijk);
        let div = divergence(&map, DScheme::Cd2nd, &at) as f64;
        assert!((div - (p[0] + p[1] + p[2])).abs() < 1e-5);
        let cu = to_d(curl(&map, DScheme::Cd2nd, &at));
        assert!((cu - Vec3d::new(-p[1], -p[2], 2.0 - p[0])).length() < 1e-5, "{cu:?}");
        let isd = is_divergence(DScheme::Cd4th, &at) as f64;
        assert!((isd * 2.0 - (p[0] + p[1] + p[2])).abs() < 1e-5);
    }

    /// Values produced by the C++ OpenVDB 10 library (`scratchpad/cppref/fd.cpp`) on the same
    /// narrow-band sphere.
    #[test]
    fn matches_openvdb_reference() {
        use crate::math::finite_difference::{d1, d2, d2_cross};
        use crate::math::stencils::{CurvatureStencil, GradStencil, WenoStencil};
        use crate::tools::test_util::{assert_close, make_sphere};
        let grid = make_sphere::<f32>(1.0, Vec3d::new(0.1, -0.05, 0.02), 0.05, 4.0);
        let map = MapJacobian::new(grid.transform());
        let acc = grid.accessor();
        struct Ref {
            p: Coord,
            v: f64,
            d1x: [f64; 14],
            d1y: [f64; 3],
            d2: [f64; 6],
            grad: [f64; 6],
            lap: [f64; 2],
            mc: [f64; 2],
            gns: [f64; 3],
            gbias: [f64; 3],
            cpt: [f64; 3],
            weno: [f64; 5],
            curv: [f64; 6],
            gradst: [f64; 4],
            zc: bool,
        }
        let refs = [
            Ref {
                p: Coord::new(20, 3, -2),
                v: -0.0702688545,
                d1x: [
                    0.0967933461,
                    0.0483966731,
                    0.048401136,
                    0.0481582507,
                    0.0484815091,
                    0.0484086908,
                    0.0484021045,
                    0.048311837,
                    0.0484114438,
                    0.0435425155,
                    0.048401095,
                    0.0481505506,
                    0.0484011061,
                    0.0484012775,
                ],
                d1y: [0.0107558081, 0.0107557625, -0.00645348243],
                d2: [0.000169679523, 0.00256451964, 0.00264413655, -0.000559052452, 0.000335937366, 7.46608057e-05],
                grad: [0.967933476, 0.214819789, -0.128886476, 0.968022704, 0.21511361, -0.129068166],
                lap: [2.15007067, 2.15115333],
                mc: [1.07504153, 1.07557237],
                gns: [1.02212071, 1.0000006, 1.00000095],
                gbias: [0.870850325, 0.215051934, -0.128981054],
                cpt: [1.06801558, 0.165095136, -0.109056704],
                weno: [1.00000107, 0.968022227, 0.215115458, -0.129070178, 2.15007067],
                curv: [1.07504439, 1.15572047, 1.07485867, 1.1555208, 1.07487714, 1.07521176],
                gradst: [1.02212071, 2.15007067, 21.3603115, 3.30190277],
                zc: false,
            },
            Ref {
                p: Coord::new(-5, 17, 6),
                v: 0.00543522928,
                d1x: [
                    -0.0347730182,
                    -0.0173865091,
                    -0.0174052976,
                    -0.0174053963,
                    -0.0162941888,
                    -0.017441269,
                    -0.0174079333,
                    -0.0184788294,
                    -0.0174444821,
                    -0.0174046438,
                    -0.0174053982,
                    -0.0174053926,
                    -0.0174053982,
                    -0.0174053945,
                ],
                d1y: [0.0447567664, 0.0447567329, 0.0139243165],
                d2: [0.00218464248, 0.000494142063, 0.0022936454, 0.000773463398, 0.000241035916, -0.000619838771],
                grad: [-0.34773019, 0.894914389, 0.278169185, -0.348105967, 0.895136237, 0.278484464],
                lap: [1.98899209, 1.98918355],
                mc: [0.994444489, 0.994589925],
                gns: [0.963383794, 0.999999762, 0.999998868],
                gbias: [-0.348092884, 0.895178616, 0.278555334],
                cpt: [-0.248110011, 0.845135927, 0.298488081],
                weno: [0.999998689, -0.348107964, 0.895134091, 0.278486222, 1.98899508],
                curv: [0.994444311, 0.988919377, 0.994029641, 0.988507032, 0.994131923, 0.994756699],
                gradst: [0.963383794, 1.98899508, -4.96220016, 16.9027195],
                zc: true,
            },
            Ref {
                p: Coord::new(9, -12, 11),
                v: -0.159821451,
                d1x: [
                    0.0415969491,
                    0.0207984746,
                    0.0208287165,
                    0.0206548553,
                    0.0220279247,
                    0.0208901167,
                    0.0208295416,
                    0.0195690244,
                    0.0208882987,
                    0.0173506513,
                    0.0208288878,
                    0.0207531452,
                    0.0208288915,
                    0.0208276249,
                ],
                d1y: [-0.0327296481, -0.0329887271, 0.0336398818],
                d2: [0.00245890021, -9.98973846e-05, -0.000559717417, -0.0021998249, 0.000600011321, -0.0030292396],
                grad: [0.415969491, -0.653960407, 0.630145967, 0.416574329, -0.690629959, 0.662549675],
                lap: [2.38165855, 1.02570462],
                mc: [-0.394017488, -0.374285579],
                gns: [1.06431186, 1.01328886, 0.999966621],
                gbias: [0.347013026, -0.973316908, 0.899146676],
                cpt: [0.516480863, -0.704516888, 0.650710821],
                weno: [0.999966681, 0.416577816, -0.654608965, 0.630794227, 2.38163471],
                curv: [-0.39401713, -0.445523351, -0.393579274, -0.445028275, -1.16911256, 0.381078273],
                gradst: [1.06431186, 2.38163471, 10.3296165, -14.0903378],
                zc: false,
            },
        ];
        let rel = |a: f64, b: f64, t: f64| assert_close(a, b, t * (1.0 + b.abs()));
        for r in &refs {
            let at = AccessorAt::new(&acc, r.p);
            assert_close(at.center_value() as f64, r.v, 1e-8);
            for (sch, &e) in DScheme::ALL.iter().zip(r.d1x.iter()) {
                rel(d1(*sch, 0, &at) as f64, e, 2e-6);
            }
            rel(d1(DScheme::FdHjWeno5, 1, &at) as f64, r.d1y[0], 2e-6);
            rel(d1(DScheme::BdWeno5, 1, &at) as f64, r.d1y[1], 2e-6);
            rel(d1(DScheme::Cd6th, 2, &at) as f64, r.d1y[2], 2e-6);
            let d2v = [
                d2(DDScheme::CdSecond, 0, &at),
                d2(DDScheme::CdFourth, 1, &at),
                d2(DDScheme::CdSixth, 2, &at),
                d2_cross(DDScheme::CdSecond, 0, 1, &at),
                d2_cross(DDScheme::CdFourth, 0, 2, &at),
                d2_cross(DDScheme::CdSixth, 1, 2, &at),
            ];
            for (a, e) in d2v.iter().zip(r.d2.iter()) {
                assert_close(*a as f64, *e, 2e-8);
            }
            let g2 = gradient(&map, DScheme::Cd2nd, &at);
            let g4 = gradient(&map, DScheme::Cd4th, &at);
            for i in 0..3 {
                rel(g2[i] as f64, r.grad[i], 1e-6);
                rel(g4[i] as f64, r.grad[3 + i], 1e-6);
            }
            rel(laplacian(&map, DDScheme::CdSecond, &at) as f64, r.lap[0], 1e-5);
            rel(laplacian(&map, DDScheme::CdFourth, &at) as f64, r.lap[1], 1e-5);
            rel(mean_curvature(&map, DDScheme::CdSecond, DScheme::Cd2nd, &at) as f64, r.mc[0], 1e-5);
            rel(mean_curvature_norm_grad(&map, DDScheme::CdFourth, DScheme::Cd4th, &at) as f64, r.mc[1], 1e-5);
            rel(gradient_norm_sqrd(&map, BiasedGradientScheme::FirstBias, &at) as f64, r.gns[0], 1e-5);
            rel(gradient_norm_sqrd(&map, BiasedGradientScheme::Weno5Bias, &at) as f64, r.gns[1], 1e-5);
            rel(gradient_norm_sqrd(&map, BiasedGradientScheme::HjWeno5Bias, &at) as f64, r.gns[2], 1e-5);
            let gb = gradient_biased(&map, BiasedGradientScheme::ThirdBias, &at, Vec3::new(1.0f32, -1.0, 1.0));
            let cp = cpt_range(&map, DScheme::Cd2nd, &at);
            for i in 0..3 {
                rel(gb[i] as f64, r.gbias[i], 1e-5);
                rel(cp[i] as f64, r.cpt[i], 1e-6);
            }
            let mut ws = WenoStencil::new(&grid);
            ws.move_to(&r.p);
            let wg = ws.gradient_upwind(Vec3::new(-1.0, 1.0, 1.0));
            rel(ws.norm_sq_grad(0.0) as f64, r.weno[0], 1e-5);
            for i in 0..3 {
                rel(wg[i] as f64, r.weno[1 + i], 1e-5);
            }
            rel(ws.laplacian() as f64, r.weno[4], 1e-5);
            let mut cs = CurvatureStencil::new(&grid);
            cs.move_to(&r.p);
            let (k1, k2) = cs.principal_curvatures();
            let cv = [
                cs.mean_curvature(),
                cs.gaussian_curvature(),
                cs.mean_curvature_norm_grad(),
                cs.gaussian_curvature_norm_grad(),
                k1,
                k2,
            ];
            for (a, e) in cv.iter().zip(r.curv.iter()) {
                rel(*a as f64, *e, 1e-5);
            }
            let mut gs = GradStencil::new(&grid);
            gs.move_to(&r.p);
            let c2 = gs.cpt();
            rel(gs.norm_sq_grad() as f64, r.gradst[0], 1e-5);
            rel(gs.laplacian() as f64, r.gradst[1], 1e-5);
            rel(c2[0] as f64, r.gradst[2], 1e-5);
            rel(c2[1] as f64, r.gradst[3], 1e-5);
            assert_eq!(gs.zero_crossing(), r.zc);
        }
    }

    /// General affine map, compared with the C++ `Gradient/Laplacian/MeanCurvature<AffineMap>`.
    #[test]
    fn affine_matches_openvdb_reference() {
        use crate::tools::test_util::assert_close;
        let mut m = Mat4d::scale(Vec3d::splat(0.05));
        m.post_rotate(crate::math::Axis::Z, 0.3);
        m.post_rotate(crate::math::Axis::X, -0.7);
        let t = Transform::from_matrix(m);
        let mut tree = FloatTree::new(1.0);
        {
            let mut acc = tree.accessor_mut();
            for c in CoordBBox::from_coords(-8, -8, -8, 8, 8, 8) {
                let c = c.offset_by(5, 4, -3);
                acc.set_value(&c, (t.index_to_world(c.as_vec3d()).length() - 0.8) as f32);
            }
        }
        let map = MapJacobian::new(&t);
        let at = AccessorAt::new(&tree, Coord::new(5, 4, -3));
        let g = gradient(&map, DScheme::Cd4th, &at);
        let expect = [0.508399844, 0.299879342, -0.807138026];
        for i in 0..3 {
            assert_close(g[i] as f64, expect[i], 1e-6);
        }
        assert_close(laplacian(&map, DDScheme::CdFourth, &at) as f64, 5.65786362, 1e-4);
        assert_close(mean_curvature(&map, DDScheme::CdFourth, DScheme::Cd4th, &at) as f64, 2.82923603, 1e-4);
    }
}
