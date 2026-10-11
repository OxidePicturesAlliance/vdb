//! Transform serialization (port of `Transform::read/write` and the linear maps'
//! `read`/`write` in `openvdb/math/Maps.h`).
//!
//! This module is self-contained so that it can later be replaced by the map types'
//! own serialization. Every linear map is converted to and from its affine matrix
//! (row-vector convention, translation in the last row):
//!
//! | map type | payload (all `f64`, little-endian) |
//! |----------|------------------------------------|
//! | `AffineMap`, `UnitaryMap` | 4x4 matrix, row-major |
//! | `ScaleMap`, `UniformScaleMap` | scale, voxel size, inverse scale, inverse scale², inverse scale / 2 (3 each) |
//! | `ScaleTranslateMap`, `UniformScaleTranslateMap` | translation (3), then the `ScaleMap` payload |
//! | `TranslationMap` | translation (3) |
//!
//! `NonlinearFrustumMap` is not supported.

use super::bytes::{Reader, WriteLe};
use super::error::{Error, Result};
use crate::math::{Mat4d, Transform, Vec3d};
use std::io::Read;

/// Read a transform: the map's type name followed by its payload.
#[cfg(test)]
pub(crate) fn read_transform<R: Read>(r: &mut Reader<R>) -> Result<Transform> {
    let type_name = r.read_string()?;
    read_map(r, &type_name).map(Transform::from_matrix)
}

/// Read the payload of a linear map of the given type and return its affine matrix.
pub(crate) fn read_map<R: Read>(r: &mut Reader<R>, type_name: &str) -> Result<Mat4d> {
    let vec3 = |r: &mut Reader<R>| -> Result<Vec3d> {
        let a = r.read_f64s::<3>()?;
        Ok(Vec3d::new(a[0], a[1], a[2]))
    };
    match type_name {
        "AffineMap" | "UnitaryMap" => {
            let v = r.read_f64s::<16>()?;
            let mut m = Mat4d::identity();
            for i in 0..4 {
                for j in 0..4 {
                    m.0[i][j] = v[4 * i + j];
                }
            }
            Ok(m)
        }
        "ScaleMap" | "UniformScaleMap" => {
            let scale = vec3(r)?;
            r.read_f64s::<12>()?; // derived quantities
            Ok(Mat4d::scale(scale))
        }
        "ScaleTranslateMap" | "UniformScaleTranslateMap" => {
            let t = vec3(r)?;
            let scale = vec3(r)?;
            r.read_f64s::<12>()?;
            let mut m = Mat4d::scale(scale);
            m.set_translation(t);
            Ok(m)
        }
        "TranslationMap" => Ok(Mat4d::translation(vec3(r)?)),
        "NonlinearFrustumMap" => Err(Error::Unsupported("NonlinearFrustumMap transforms".into())),
        other => Err(Error::Unsupported(format!("map type \"{}\"", other))),
    }
}

/// The OpenVDB map type that [`write_transform`] uses for a matrix.
pub(crate) fn map_type_for(m: &Mat4d) -> &'static str {
    let diag = m.0[0][0];
    let off_diag_zero = (0..3).all(|i| (0..3).all(|j| i == j || m.0[i][j] == 0.0));
    let projective_ok = m.0[0][3] == 0.0 && m.0[1][3] == 0.0 && m.0[2][3] == 0.0 && m.0[3][3] == 1.0;
    let uniform = off_diag_zero && projective_ok && m.0[1][1] == diag && m.0[2][2] == diag && diag != 0.0;
    if !uniform {
        return "AffineMap";
    }
    if m.get_translation() == Vec3d::zero() {
        "UniformScaleMap"
    } else {
        "UniformScaleTranslateMap"
    }
}

fn put_vec3(out: &mut Vec<u8>, v: Vec3d) {
    for i in 0..3 {
        out.put_f64(v[i]);
    }
}

/// Write the payload of a `ScaleMap` with the given scale.
fn put_scale_payload(out: &mut Vec<u8>, s: Vec3d) {
    let inv = Vec3d::new(1.0 / s[0], 1.0 / s[1], 1.0 / s[2]);
    put_vec3(out, s);
    put_vec3(out, Vec3d::new(s[0].abs(), s[1].abs(), s[2].abs()));
    put_vec3(out, inv);
    put_vec3(out, Vec3d::new(inv[0] * inv[0], inv[1] * inv[1], inv[2] * inv[2]));
    put_vec3(out, Vec3d::new(inv[0] * 0.5, inv[1] * 0.5, inv[2] * 0.5));
}

/// Write a transform. Pure uniform scales (with or without translation) are written as
/// `UniformScaleMap` / `UniformScaleTranslateMap`, everything else as `AffineMap`.
pub(crate) fn write_transform(out: &mut Vec<u8>, t: &Transform) -> Result<()> {
    if !t.is_linear() {
        return Err(Error::Unsupported("writing non-linear transforms".into()));
    }
    let m = t.map().affine_matrix();
    let kind = map_type_for(&m);
    out.put_string(kind);
    match kind {
        "UniformScaleMap" => put_scale_payload(out, Vec3d::splat(m.0[0][0])),
        "UniformScaleTranslateMap" => {
            put_vec3(out, m.get_translation());
            put_scale_payload(out, Vec3d::splat(m.0[0][0]));
        }
        _ => {
            for row in &m.0 {
                for v in row {
                    out.put_f64(*v);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Axis;

    fn roundtrip(t: &Transform) -> (String, Transform) {
        let mut out = Vec::new();
        write_transform(&mut out, t).unwrap();
        let mut r = Reader::new(&out[..]);
        let name = Reader::new(&out[..]).read_string().unwrap();
        let back = read_transform(&mut r).unwrap();
        assert_eq!(r.pos() as usize, out.len());
        (name, back)
    }

    #[test]
    fn map_type_selection() {
        let (n, b) = roundtrip(&Transform::linear(0.5));
        assert_eq!(n, "UniformScaleMap");
        assert_eq!(b.map().affine_matrix(), Transform::linear(0.5).map().affine_matrix());

        let mut t = Transform::linear(0.25);
        t.post_translate(Vec3d::new(1.0, 2.0, 3.0));
        let (n, b) = roundtrip(&t);
        assert_eq!(n, "UniformScaleTranslateMap");
        assert_eq!(b.map().affine_matrix(), t.map().affine_matrix());

        let mut t = Transform::linear(1.0);
        t.pre_rotate(0.3, Axis::Z);
        t.post_scale(Vec3d::new(1.0, 2.0, 3.0));
        let (n, b) = roundtrip(&t);
        assert_eq!(n, "AffineMap");
        assert_eq!(b.map().affine_matrix(), t.map().affine_matrix());
    }

    fn encode(name: &str, vals: &[f64]) -> Vec<u8> {
        let mut out = Vec::new();
        out.put_string(name);
        for v in vals {
            out.put_f64(*v);
        }
        out
    }

    #[test]
    fn read_all_linear_maps() {
        // TranslationMap
        let b = encode("TranslationMap", &[1.0, 2.0, 3.0]);
        let t = read_transform(&mut Reader::new(&b[..])).unwrap();
        assert!(t.index_to_world(Vec3d::zero()).is_approx_equal(&Vec3d::new(1.0, 2.0, 3.0), 0.0));

        // ScaleMap (non-uniform)
        let s = [2.0, 3.0, 4.0];
        let mut vals = s.to_vec();
        vals.extend_from_slice(&s);
        vals.extend_from_slice(&[0.5, 1.0 / 3.0, 0.25, 0.25, 1.0 / 9.0, 1.0 / 16.0, 0.25, 1.0 / 6.0, 0.125]);
        let b = encode("ScaleMap", &vals);
        let t = read_transform(&mut Reader::new(&b[..])).unwrap();
        assert!(t.voxel_size().is_approx_equal(&Vec3d::new(2.0, 3.0, 4.0), 0.0));

        // ScaleTranslateMap
        let mut v2 = vec![-1.0, 0.0, 1.0];
        v2.extend_from_slice(&vals);
        let b = encode("ScaleTranslateMap", &v2);
        let t = read_transform(&mut Reader::new(&b[..])).unwrap();
        assert!(t.index_to_world(Vec3d::new(1.0, 1.0, 1.0)).is_approx_equal(&Vec3d::new(1.0, 3.0, 5.0), 0.0));

        // UnitaryMap (stores an affine map)
        let rot = Mat4d::rotation(Axis::X, 0.5);
        let flat: Vec<f64> = rot.0.iter().flatten().copied().collect();
        let b = encode("UnitaryMap", &flat);
        let t = read_transform(&mut Reader::new(&b[..])).unwrap();
        assert_eq!(t.map().affine_matrix(), rot);

        // Frustum maps are reported as unsupported.
        let b = encode("NonlinearFrustumMap", &[0.0; 8]);
        assert!(matches!(read_transform(&mut Reader::new(&b[..])), Err(Error::Unsupported(_))));
    }
}
