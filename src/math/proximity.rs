//! Closest-point queries on simple primitives (port of `openvdb/math/Proximity.h`).

use super::vec3::Vec3d;

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

    if a.eq_eps(&b, 1e-7) && a.eq_eps(&c, 1e-7) {
        uvw[0] = 1.0;
        return (a, uvw);
    }

    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);

    if a.eq_eps(&b, 1e-7) {
        let (cp, t) = closest_point_on_segment_to_point(&a, &c, &p);
        uvw[0] = 1.0 - t;
        uvw[2] = t;
        return (cp, uvw);
    } else if a.eq_eps(&c, 1e-7) || b.eq_eps(&c, 1e-7) {
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
}
