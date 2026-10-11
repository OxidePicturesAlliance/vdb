//! Integration tests for `.vdb` file I/O.
//!
//! The reference files in tests/data were written by C++ OpenVDB 10.0.1 with the generator in
//! tests/data/cpp/vdb_ref.cpp; `<name>.dump.txt` holds `vdb_ref dump <name>.vdb`, and the
//! expected values below were printed by `vdb_ref probe`.

#![allow(clippy::approx_constant, clippy::excessive_precision)]

mod common;

use common::*;
use vdb::io::{self, Compression, File, WriteOptions};
use vdb::math::{Axis, Mat4d, Vec3d, Vec3f, Vec3i};
use vdb::metadata::{MetaMap, MetaValue};
use vdb::prelude::*;
use vdb::tree::RootEntry;
use vdb::AnyGrid;

const REFERENCE_FILES: &[&str] = &[
    "sphere",
    "sphere_half",
    "sphere_blosc",
    "fog_tiles",
    "types_zip",
    "types_zip_only",
    "types_mask",
    "types_blosc",
    "types_none",
    "points",
    "stream",
];

fn assert_same_dump(actual: &str, expected: &str, what: &str) {
    if actual == expected {
        return;
    }
    for (i, (a, e)) in actual.lines().zip(expected.lines()).enumerate() {
        if a != e {
            panic!("{}: first difference at line {}:\n  actual:   {}\n  expected: {}", what, i + 1, a, e);
        }
    }
    panic!("{}: dumps differ in length ({} vs {} lines)", what, actual.lines().count(), expected.lines().count());
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-7 * b.abs().max(1.0)
}

// ---------------------------------------------------------------------------
// Reading C++-written files

/// Every reference file reads back exactly (values, inactive values, tiles, metadata,
/// transforms) as the C++ library sees it.
#[test]
fn reference_dumps_match_cpp() {
    for name in REFERENCE_FILES {
        let expected = std::fs::read_to_string(data(&format!("{}.dump.txt", name))).unwrap();
        let actual = dump_file(&data(&format!("{}.vdb", name)));
        assert_same_dump(&actual, &expected, &format!("{}.vdb", name));
    }
}

#[test]
fn sphere_level_set() {
    let file = File::open(data("sphere.vdb")).unwrap();
    assert_eq!(file.header().file_version, 224);
    assert_eq!(file.header().library_version, (10, 0));
    assert!(file.header().has_grid_offsets);
    assert_eq!(file.version(), "10.0/224");
    assert_eq!(file.grid_names(), vec!["sphere"]);

    let meta = file.metadata();
    assert_eq!(meta.get_str("creator_tool"), Some("vdb_ref"));
    assert_eq!(meta.get("answer"), Some(&MetaValue::Int32(42)));
    assert_eq!(meta.get("big"), Some(&MetaValue::Int64(1 << 40)));
    assert_eq!(meta.get("pi_f"), Some(&MetaValue::Float(3.14159)));
    assert_eq!(meta.get("e_d"), Some(&MetaValue::Double(2.718281828459045)));
    assert_eq!(meta.get("flag"), Some(&MetaValue::Bool(true)));
    assert_eq!(meta.get("origin"), Some(&MetaValue::Vec3d(Vec3d::new(1.5, -2.0, 0.25))));
    assert_eq!(meta.get("res"), Some(&MetaValue::Vec3i(Vec3i::new(640, 480, 1))));
    assert_eq!(meta.get("dir"), Some(&MetaValue::Vec3s(Vec3f::new(0.0, 1.0, -0.5))));
    let mut xform = Mat4d::identity();
    xform.set_translation(Vec3d::new(1.0, 2.0, 3.0));
    assert_eq!(meta.get("xform"), Some(&MetaValue::Mat4d(xform)));

    let gm = file.read_grid_metadata("sphere").unwrap();
    assert_eq!(gm.compression, Compression::ACTIVE_MASK); // zip is not used for level sets
    assert_eq!(gm.map_type, "UniformScaleMap");
    assert_eq!(gm.metadata.get("file_voxel_count"), Some(&MetaValue::Int64(7700)));
    assert_eq!(gm.metadata.get("file_bbox_min"), Some(&MetaValue::Vec3i(Vec3i::new(-9, -16, -12))));

    let grid = file.read_grid("sphere").unwrap();
    let g = grid.as_float().unwrap();
    assert_eq!(g.name(), "sphere");
    assert_eq!(g.creator(), "vdb_ref");
    assert_eq!(g.grid_class(), GridClass::LevelSet);
    assert_eq!(*g.background(), 1.5);
    assert!(g.voxel_size().is_approx_equal(&Vec3d::splat(0.5), 0.0));
    assert_eq!(g.active_voxel_count(), 7700);
    assert_eq!(g.tree().leaf_count(), 44);
    assert_eq!(g.eval_active_voxel_bbox(), CoordBBox::from_coords(-9, -16, -12, 15, 8, 13));

    let probe = |x, y, z| {
        let c = Coord::new(x, y, z);
        (g.tree().get_value(&c), g.tree().is_value_on(&c))
    };
    assert_eq!(probe(3, -4, 0), (-1.5, false)); // deep inside
    assert_eq!(probe(3, -4, 10), (-0.25, true));
    assert_eq!(probe(100, 100, 100), (1.5, false)); // far outside
    for (x, y, z) in [(13, -4, 0), (3, 6, 1), (-7, -4, 0)] {
        let (v, on) = probe(x, y, z);
        assert!(on && approx(v, 0.00624608994), "{} at {:?}", v, (x, y, z));
    }
}

#[test]
fn half_float_grids() {
    let file = File::open(data("sphere_half.vdb")).unwrap();
    for d in file.descriptors() {
        assert!(d.save_float_as_half, "{}", d.grid_name);
        assert!(!d.grid_type.ends_with("_HalfFloat"));
    }
    let s = file.read_grid("sphere_half").unwrap().into_float().unwrap();
    assert!(s.save_float_as_half());
    let v = s.tree().get_value(&Coord::new(13, -4, 0));
    assert!(approx(v, 0.00624465942), "{}", v);
    assert_eq!(v, half::f16::from_f32(0.00624608994).to_f32());

    let mc = file.read_grid("maskcomp_half").unwrap().into_float().unwrap();
    assert_eq!(mc.tree().get_value(&Coord::new(33, 0, 1)), 7.0);
    assert!(!mc.tree().is_value_on(&Coord::new(33, 0, 1)));

    let vel = file.read_grid("vel_half").unwrap().into_vec3s().unwrap();
    assert_eq!(*vel.background(), Vec3f::new(0.5, 0.0, -0.5));
    assert_eq!(vel.active_voxel_count(), 300);
}

#[test]
fn blosc_level_set_matches_uncompressed() {
    let a = io::read_grid(data("sphere.vdb"), "sphere").unwrap().into_float().unwrap();
    let b = io::read_grid(data("sphere_blosc.vdb"), "sphere_blosc").unwrap().into_float().unwrap();
    assert_eq!(
        File::open(data("sphere_blosc.vdb")).unwrap().read_grid_metadata("sphere_blosc").unwrap().compression,
        Compression::BLOSC | Compression::ACTIVE_MASK
    );
    assert!(a.tree() == b.tree());
    let (mut da, mut db) = (String::new(), String::new());
    dump_grid(&mut da, &AnyGrid::Float(a));
    dump_grid(&mut db, &AnyGrid::Float(b));
    assert_eq!(da, db);
}

#[test]
fn fog_volume_with_tiles() {
    let grid = io::read_grid(data("fog_tiles.vdb"), "fog").unwrap().into_float().unwrap();
    assert_eq!(grid.grid_class(), GridClass::FogVolume);
    assert_eq!(grid.active_voxel_count(), 137448161226);
    assert_eq!(grid.tree().active_tile_count(), 9764);
    assert_eq!(grid.tree().leaf_count(), 62);
    assert_eq!(grid.eval_active_voxel_bbox(), CoordBBox::from_coords(-300, 0, 0, 12287, 4095, 4095));
    assert!(grid.voxel_size().is_approx_equal(&Vec3d::splat(0.1), 1e-15));
    assert!(grid.index_to_world(Vec3d::zero()).is_approx_equal(&Vec3d::new(10.0, -5.0, 2.5), 1e-12));

    // Root-level tiles.
    let table = grid.tree().root().table();
    for x in [4096, 8192] {
        assert!(
            matches!(table.get(&Coord::new(x, 0, 0)), Some(RootEntry::Tile { value, active: true }) if *value == 1.0)
        );
    }
    assert!(matches!(
        table.get(&Coord::new(-4096, -4096, -4096)),
        Some(RootEntry::Tile { value, active: false }) if *value == 0.25
    ));

    let t = grid.tree();
    let probe = |x, y, z| (t.get_value(&Coord::new(x, y, z)), t.is_value_on(&Coord::new(x, y, z)));
    assert_eq!(probe(5000, 100, 100), (1.0, true));
    assert_eq!(probe(-5, -5, -5), (0.25, false));
    assert_eq!(probe(-100, 100, 100), (0.5, true)); // internal tile
    assert!(t.get_value_depth(&Coord::new(-100, 100, 100)) < 3);
    assert_eq!(probe(-295, 3, 3), (0.75, true)); // partial leaf
    let (v, on) = probe(1, 1, 0);
    assert!(on && approx(v, 0.360666513));
    assert_eq!(probe(2, 1, 0), (0.0, false));
    assert_eq!(probe(50000, 0, 0), (0.0, false));
}

fn check_typed_grids(file_name: &str) {
    let file = File::open(data(file_name)).unwrap();
    let names = file.grid_names();
    assert_eq!(
        names,
        vec![
            "ints",
            "ints_instance",
            "doubles",
            "longs",
            "flags",
            "mask",
            "velocity",
            "vel_d",
            "indices",
            "maskcomp",
            "dup",
            "dup",
            "empty"
        ],
        "{}",
        file_name
    );
    let descs = file.descriptors();
    assert_eq!(descs[1].instance_parent, "ints");
    assert!(descs[1].is_instance());
    assert_eq!(descs[10].unique_name, "dup\x1e0");
    assert_eq!(descs[11].unique_name, "dup\x1e1");
    assert_eq!(descs[11].display_name(), "dup[1]");
    assert_eq!(descs[5].grid_type, "Tree_mask_5_4_3");

    let grids = file.read_grids().unwrap();
    assert_eq!(grids.len(), 13);
    let get = |name: &str| file.read_grid(name).unwrap();

    let ints = get("ints").into_int32().unwrap();
    assert_eq!(*ints.background(), -1);
    assert_eq!(ints.metadata().get_str("note"), Some("integers"));
    assert_eq!(ints.metadata().get("ivec2"), Some(&MetaValue::Vec2i([-3, 4])));
    assert_eq!(ints.metadata().get("scale2"), Some(&MetaValue::Vec2s([0.5, 2.0])));
    assert_eq!(ints.metadata().get("dvec4"), Some(&MetaValue::Vec4d([1.0, 2.0, 3.0, 4.0])));
    assert_eq!(ints.active_voxel_count(), 2097452);
    assert_eq!(
        (ints.get_value(&Coord::new(12, 300, 50)), ints.tree().is_value_on(&Coord::new(12, 300, 50))),
        (-2993, true)
    );
    assert_eq!(
        (ints.get_value(&Coord::new(11, 300, 50)), ints.tree().is_value_on(&Coord::new(11, 300, 50))),
        (5, false)
    );
    assert_eq!(ints.get_value(&Coord::new(1050, 1100, 1001)), 77);

    let inst = get("ints_instance").into_int32().unwrap();
    assert!(inst.tree() == ints.tree());
    assert!(inst.voxel_size().is_approx_equal(&Vec3d::splat(2.0), 0.0));

    let dbl = get("doubles").into_double().unwrap();
    assert_eq!(dbl.get_value(&Coord::new(3, 597, 6)), 0.25);
    assert_eq!(dbl.transform().map_type(), "AffineMap");
    let mut m = Mat4d::identity();
    m.pre_scale(Vec3d::splat(0.5));
    m.post_rotate(Axis::Y, 0.3);
    m.post_translate(Vec3d::new(1.0, 2.0, 3.0));
    assert!(dbl.transform().map().affine_matrix().is_approx_equal(&m, 1e-15));

    let longs = get("longs").into_int64().unwrap();
    assert_eq!(longs.get_value(&Coord::new(30, 7, -10)), 1099511627786);
    assert_eq!(*longs.background(), -5);

    let flags = get("flags").into_bool().unwrap();
    let f = |x, y, z| (flags.get_value(&Coord::new(x, y, z)), flags.tree().is_value_on(&Coord::new(x, y, z)));
    assert_eq!(f(0, 0, 0), (true, true));
    assert_eq!(f(1, 1, 1), (false, true));
    assert_eq!(f(2, 2, 2), (true, false));
    assert_eq!(f(300, 300, 300), (true, true));
    assert_eq!(flags.active_voxel_count(), 2097419);

    let mask_any = get("mask");
    assert_eq!(mask_any.grid_type(), "Tree_mask_5_4_3");
    let mask = mask_any.into_mask().unwrap();
    assert_eq!(mask.active_voxel_count(), 4194404);
    assert!(mask.tree().is_value_on(&Coord::new(5, 1, 2)) && mask.get_value(&Coord::new(5, 1, 2)));
    assert!(mask.tree().is_value_on(&Coord::new(600, 600, 600)));
    assert!(!mask.tree().is_value_on(&Coord::new(6, 1, 2)) && !mask.get_value(&Coord::new(6, 1, 2)));

    let vel = get("velocity").into_vec3s().unwrap();
    assert_eq!(vel.vector_type(), VecType::ContravariantRelative);
    let v = vel.get_value(&Coord::new(10, 3, 10));
    assert_eq!((v[0], v[1]), (10.0, -5.0));
    assert!(approx(v[2], 1.0 / 11.0));
    assert!(vel.index_to_world(Vec3d::new(1.0, 1.0, 1.0)).is_approx_equal(&Vec3d::new(-0.5, 1.0, 6.0), 1e-15));

    let vel_d = get("vel_d").into_vec3d().unwrap();
    assert_eq!(vel_d.get_value(&Coord::new(90, 10, 3)), Vec3d::new(1.0, 2.0, 3.0));
    assert_eq!(*vel_d.background(), Vec3d::new(1.0, 2.0, 3.0));
    assert!(vel_d.index_to_world(Vec3d::zero()).is_approx_equal(&Vec3d::new(0.5, 0.25, -0.125), 0.0));

    let idx = get("indices").into_vec3i().unwrap();
    assert_eq!(idx.get_value(&Coord::new(7, 14, 21)), Vec3i::new(7, -7, 49));
    assert!(idx.voxel_size().is_approx_equal(&Vec3d::new(1.0, 2.0, 3.0), 0.0));

    let mc = get("maskcomp").into_float().unwrap();
    for ((x, y, z), expected) in [
        ((8, 0, 0), -2.0),
        ((16, 0, 0), 5.0),
        ((24, 0, 1), -2.0),
        ((48, 0, 4), 4.0),
        ((130, 3, 3), 9.0),
        ((400, 3, 3), 2.0),
        ((3, 0, 520), 6.0),
    ] {
        let c = Coord::new(x, y, z);
        assert_eq!(mc.get_value(&c), expected, "{} at {:?}", file_name, c);
        assert!(!mc.tree().is_value_on(&c));
    }

    let dup1 = get("dup[1]").into_float().unwrap();
    assert_eq!(dup1.get_value(&Coord::new(4, 5, 6)), 2.0);
    let dup0 = get("dup").into_float().unwrap();
    assert_eq!(dup0.get_value(&Coord::new(1, 2, 3)), 1.0);

    let empty = get("empty").into_float().unwrap();
    assert!(empty.empty());
    assert_eq!(*empty.background(), 3.5);
    assert_eq!(empty.metadata().get("file_bbox_min"), Some(&MetaValue::Vec3i(Vec3i::splat(i32::MAX))));
}

#[test]
fn typed_grids_all_compressions() {
    for f in ["types_zip.vdb", "types_zip_only.vdb", "types_mask.vdb", "types_blosc.vdb"] {
        check_typed_grids(f);
    }
    let file = File::open(data("types_zip_only.vdb")).unwrap();
    assert_eq!(file.read_grid_metadata("ints").unwrap().compression, Compression::ZIP);
    let none = File::open(data("types_none.vdb")).unwrap();
    assert_eq!(none.grid_names(), vec!["flags", "maskcomp"]);
    assert_eq!(none.read_grid_metadata("flags").unwrap().compression, Compression::NONE);
    let a = none.read_grid("maskcomp").unwrap();
    let b = io::read_grid(data("types_zip.vdb"), "maskcomp").unwrap();
    let (mut da, mut db) = (String::new(), String::new());
    dump_grid(&mut da, &a);
    dump_grid(&mut db, &b);
    assert_eq!(da, db);
}

#[test]
fn point_data_grids_are_loaded() {
    let file = File::open(data("points.vdb")).unwrap();
    assert_eq!(file.descriptors().len(), 2);
    let pts = &file.descriptors()[0];
    assert_eq!(pts.grid_type, "Tree_ptdataidx32_5_4_3");
    assert_eq!(pts.value_type(), Some("ptdataidx32"));
    assert!(pts.is_supported());
    let gm = file.read_grid_metadata("points").unwrap();
    assert_eq!(gm.metadata.get("file_voxel_count"), Some(&MetaValue::Int64(4)));
    let points = file.read_grid("points").unwrap();
    assert_eq!(vdb::points::point_count(points.as_point_data().unwrap().tree()), 4);

    let (_, grids) = io::read_file(data("points.vdb")).unwrap();
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[1].name(), "after_points");
    assert_eq!(grids[1].as_float().unwrap().get_value(&Coord::new(3, 2, 1)), 42.0);
    let s = io::Stream::read(std::fs::File::open(data("points.vdb")).unwrap()).unwrap();
    assert!(s.grids[0].as_ref().unwrap().as_point_data().is_some());
}

#[test]
fn unsupported_grid_types_are_skipped() {
    // Write two grids, then rename the first one's type to an unknown (same-length) type.
    let mut a = FloatGrid::new(0.0).with_name("unknown");
    a.set_value(&Coord::new(1, 1, 1), 1.0);
    let mut b = FloatGrid::new(0.0).with_name("after");
    b.set_value(&Coord::new(3, 2, 1), 42.0);
    let mut bytes = Vec::new();
    io::write_to(&mut bytes, &[a.into(), b.into()], &MetaMap::new(), &WriteOptions::default()).unwrap();
    let pat = b"Tree_float_5_4_3";
    let pos = bytes.windows(pat.len()).position(|w| w == pat).unwrap();
    bytes[pos..pos + pat.len()].copy_from_slice(b"Tree_fl0at_5_4_3");
    let path = scratch_dir("unsupported").join("unsupported.vdb");
    std::fs::write(&path, &bytes).unwrap();

    let file = File::open(&path).unwrap();
    let d = &file.descriptors()[0];
    assert_eq!(d.value_type(), Some("fl0at"));
    assert!(!d.is_supported());
    // Metadata of unsupported grids can still be read.
    let gm = file.read_grid_metadata("unknown").unwrap();
    assert_eq!(gm.metadata.get("file_voxel_count"), Some(&MetaValue::Int64(1)));
    assert!(matches!(file.read_grid("unknown"), Err(io::Error::UnsupportedGridType(_))));
    let (_, grids) = io::read_file(&path).unwrap();
    assert_eq!(grids.len(), 1);
    assert_eq!(grids[0].as_float().unwrap().get_value(&Coord::new(3, 2, 1)), 42.0);
    // Sequential reading skips the grid using its end offset.
    let s = io::Stream::read(&bytes[..]).unwrap();
    assert!(s.grids[0].is_none());
    assert_eq!(s.grids[1].as_ref().unwrap().name(), "after");
}

#[test]
fn stream_written_file() {
    let file = File::open(data("stream.vdb")).unwrap();
    assert!(!file.header().has_grid_offsets);
    assert_eq!(file.grid_names(), vec!["a", "b"]);
    assert_eq!(file.metadata().get("streamed"), Some(&MetaValue::Bool(true)));
    assert_eq!(file.read_grid("a").unwrap().as_float().unwrap().get_value(&Coord::new(7, 7, 7)), 7.0);

    let (meta, grids) = io::read_stream(std::fs::File::open(data("stream.vdb")).unwrap()).unwrap();
    assert_eq!(meta.get("streamed"), Some(&MetaValue::Bool(true)));
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[1].as_int32().unwrap().get_value(&Coord::new(-5, 5, -5)), 12345);
    assert_eq!(grids[1].as_int32().unwrap().active_voxel_count(), 1);
}

#[test]
fn metadata_only_reading() {
    let metas = io::read_grid_metadata(data("types_blosc.vdb")).unwrap();
    assert_eq!(metas.len(), 13);
    let vel = metas.iter().find(|m| m.descriptor.grid_name == "velocity").unwrap();
    assert_eq!(vel.map_type, "ScaleTranslateMap");
    assert_eq!(vel.compression, Compression::BLOSC | Compression::ACTIVE_MASK);
    assert_eq!(vel.metadata.get_str("vector_type"), Some("contravariant relative"));
    let t = vel.transform.as_ref().unwrap();
    assert!(t.voxel_size().is_approx_equal(&Vec3d::new(0.5, 1.0, 2.0), 0.0));
    let idx = metas.iter().find(|m| m.descriptor.grid_name == "indices").unwrap();
    assert_eq!(idx.map_type, "ScaleMap");
    let ints = metas.iter().find(|m| m.descriptor.grid_name == "ints").unwrap();
    assert_eq!(ints.metadata.get("file_bbox_max"), Some(&MetaValue::Vec3i(Vec3i::splat(1127))));
    assert!(
        ints.descriptor.grid_pos < ints.descriptor.block_pos && ints.descriptor.block_pos < ints.descriptor.end_pos
    );
}

#[test]
fn blosc_reference_vectors() {
    for name in
        ["lz4_2048", "lz4_64", "lz4_multiblock", "lz4_odd", "lz4_noshuf_ts1", "blosclz_2048", "blosclz_multiblock"]
    {
        let frame = std::fs::read(data(&format!("blosc/{}.blosc", name))).unwrap();
        let raw = std::fs::read(data(&format!("blosc/{}.raw", name))).unwrap();
        assert_eq!(io::blosc::decompress(&frame).unwrap(), raw, "{}", name);
        assert_eq!(io::blosc::decompress(&io::blosc::compress(&raw, 4)).unwrap(), raw);
    }
}

#[test]
fn errors() {
    let dir = scratch_dir("errors");
    let p = dir.join("garbage.vdb");
    std::fs::write(&p, b"this is not a vdb file at all").unwrap();
    assert!(matches!(File::open(&p), Err(io::Error::NotVdb)));
    assert!(matches!(File::open(dir.join("missing.vdb")), Err(io::Error::Io(_))));

    let mut bytes = std::fs::read(data("types_zip.vdb")).unwrap();
    assert!(matches!(io::read_grid(data("types_zip.vdb"), "nope"), Err(io::Error::GridNotFound(_))));

    let mut v = bytes.clone();
    v[8..12].copy_from_slice(&219u32.to_le_bytes());
    std::fs::write(&p, &v).unwrap();
    assert!(matches!(File::open(&p), Err(io::Error::UnsupportedVersion(219))));

    bytes.truncate(bytes.len() - 1000);
    std::fs::write(&p, &bytes).unwrap();
    // Either the descriptors or the grid data are cut off.
    assert!(File::open(&p).and_then(|f| f.read_grids()).is_err());
    assert!(io::read_stream(&bytes[..]).is_err());
}

/// Formats 222 and 223 differ from 224 only for point data grids, so patching the version
/// number of a reference file yields a valid older file.
#[test]
fn file_versions_222_and_223() {
    let dir = scratch_dir("versions");
    let bytes = std::fs::read(data("types_zip.vdb")).unwrap();
    let expected = std::fs::read_to_string(data("types_zip.dump.txt")).unwrap();
    for v in [222u32, 223] {
        let mut b = bytes.clone();
        b[8..12].copy_from_slice(&v.to_le_bytes());
        let p = dir.join(format!("v{}.vdb", v));
        std::fs::write(&p, &b).unwrap();
        let exp = expected.replacen("file_version 224", &format!("file_version {}", v), 1);
        assert_same_dump(&dump_file(&p), &exp, &format!("version {}", v));
    }
}

/// A hand-built file in format 221 (file-level compression flag, no per-node metadata,
/// internal nodes storing only their non-child values, leaf buffers with origin and buffer
/// count), with "zip" chunks stored uncompressed.
#[test]
fn file_version_221() {
    fn put<T: Copy>(b: &mut Vec<u8>, v: &[T], f: impl Fn(T) -> Vec<u8>) {
        for x in v {
            b.extend(f(*x));
        }
    }
    let s = |b: &mut Vec<u8>, t: &str| {
        b.extend((t.len() as u32).to_le_bytes());
        b.extend(t.as_bytes());
    };
    let zip_chunk = |b: &mut Vec<u8>, raw: &[u8]| {
        b.extend((-(raw.len() as i64)).to_le_bytes());
        b.extend(raw);
    };
    let mut b = Vec::new();
    b.extend(0x56444220i64.to_le_bytes());
    b.extend(221u32.to_le_bytes());
    b.extend(3u32.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.push(1); // has grid offsets
    b.push(1); // zip compressed
    b.extend(b"00000000-0000-0000-0000-000000000000");
    b.extend(0u32.to_le_bytes()); // file metadata
    b.extend(1i32.to_le_bytes()); // grid count
    s(&mut b, "old");
    s(&mut b, "Tree_float_5_4_3");
    s(&mut b, "");
    let offsets_at = b.len();
    b.extend([0u8; 24]);
    let grid_pos = b.len();
    b.extend(0u32.to_le_bytes()); // grid metadata
    s(&mut b, "UniformScaleMap");
    put(&mut b, &[0.5f64, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0], |x| {
        x.to_le_bytes().to_vec()
    });
    b.extend(1i32.to_le_bytes()); // buffer count
    b.extend(7.0f32.to_le_bytes()); // background
    b.extend(0u32.to_le_bytes()); // tiles
    b.extend(1u32.to_le_bytes()); // children
    put(&mut b, &[0i32, 0, 0], |x| x.to_le_bytes().to_vec());
    // Internal node (32^3): child 0, tile 1 active with value 3, all other tiles inactive 7.
    let mut mask = vec![0u64; 512];
    mask[0] = 1;
    put(&mut b, &mask, |x| x.to_le_bytes().to_vec());
    mask[0] = 2;
    put(&mut b, &mask, |x| x.to_le_bytes().to_vec());
    let vals: Vec<u8> = (1..32768).flat_map(|i| if i == 1 { 3.0f32 } else { 7.0 }.to_le_bytes()).collect();
    zip_chunk(&mut b, &vals);
    // Internal node (16^3): child 0, all tiles inactive 7.
    let mut mask = vec![0u64; 64];
    mask[0] = 1;
    put(&mut b, &mask, |x| x.to_le_bytes().to_vec());
    put(&mut b, &[0u64; 64], |x| x.to_le_bytes().to_vec());
    let vals: Vec<u8> = (1..4096).flat_map(|_| 7.0f32.to_le_bytes()).collect();
    zip_chunk(&mut b, &vals);
    // Leaf topology: voxels 0 and 511 active.
    let mut leaf_mask = [0u64; 8];
    leaf_mask[0] = 1;
    leaf_mask[7] = 1 << 63;
    put(&mut b, &leaf_mask, |x| x.to_le_bytes().to_vec());
    let block_pos = b.len();
    // Leaf buffers: mask, origin, buffer count, values.
    put(&mut b, &leaf_mask, |x| x.to_le_bytes().to_vec());
    put(&mut b, &[0i32, 0, 0], |x| x.to_le_bytes().to_vec());
    b.push(1);
    let vals: Vec<u8> = (0..512).flat_map(|i| (i as f32).to_le_bytes()).collect();
    zip_chunk(&mut b, &vals);
    let end_pos = b.len();
    let offs: Vec<u8> =
        [grid_pos as i64, block_pos as i64, end_pos as i64].iter().flat_map(|x| x.to_le_bytes()).collect();
    b[offsets_at..offsets_at + 24].copy_from_slice(&offs);

    let p = scratch_dir("v221").join("v221.vdb");
    std::fs::write(&p, &b).unwrap();
    let file = File::open(&p).unwrap();
    assert_eq!(file.header().file_version, 221);
    assert_eq!(file.header().compression, Compression::ZIP);
    let g = file.read_grid("old").unwrap().into_float().unwrap();
    assert_eq!(g.name(), "old");
    assert!(g.voxel_size().is_approx_equal(&Vec3d::splat(0.5), 0.0));
    assert_eq!(*g.background(), 7.0);
    assert_eq!(g.get_value(&Coord::new(0, 0, 0)), 0.0);
    assert_eq!(g.get_value(&Coord::new(7, 7, 7)), 511.0);
    assert_eq!(g.get_value(&Coord::new(1, 0, 0)), 64.0);
    assert!(g.tree().is_value_on(&Coord::new(7, 7, 7)) && !g.tree().is_value_on(&Coord::new(1, 0, 0)));
    assert_eq!(g.get_value(&Coord::new(0, 0, 200)), 3.0); // the active 128^3 tile
    assert!(g.tree().is_value_on(&Coord::new(0, 0, 200)));
    assert_eq!(g.get_value(&Coord::new(0, 0, 50)), 7.0);
    assert_eq!(g.active_voxel_count(), 2 + 128 * 128 * 128);
}

// ---------------------------------------------------------------------------
// Writing

/// A set of grids covering all value types, tiles at every level, inactive values,
/// metadata of every type and several transforms.
fn sample_grids() -> Vec<AnyGrid> {
    let mut out = Vec::new();

    let mut f = FloatGrid::new(2.0).with_name("floats");
    f.set_grid_class(GridClass::Unknown);
    for i in 0..300 {
        f.set_value(&Coord::new(i % 17, i / 17, -i % 5), i as f32 * 0.5);
        f.tree_mut().set_value_off(&Coord::new(i % 17, i / 17, 10), if i % 3 == 0 { -2.0 } else { 7.0 });
    }
    f.tree_mut().fill(&CoordBBox::from_coords(-512, 0, 0, -385, 127, 127), 3.0, true);
    f.tree_mut().fill(&CoordBBox::from_coords(-64, -64, -64, -57, -57, -57), 4.0, false);
    f.tree_mut().root_mut().add_root_tile(&Coord::new(8192, 0, 0), 5.0, true);
    f.tree_mut().root_mut().add_root_tile(&Coord::new(-8192, 0, 0), -5.0, false);
    f.insert_meta("b", true);
    f.insert_meta("i", 7i32);
    f.insert_meta("l", -7i64);
    f.insert_meta("d", 0.125f64);
    f.insert_meta("s", "text");
    f.insert_meta("v3d", Vec3d::new(1.0, 2.0, 3.0));
    f.insert_meta("v4s", MetaValue::Vec4s([1.0, 2.0, 3.0, 4.0]));
    f.insert_meta("m", Mat4d::identity());
    let mut t = Transform::linear(0.5);
    t.post_translate(Vec3d::new(1.0, -2.0, 3.0));
    f.set_transform(t);
    out.push(f.into());

    let mut ls = FloatGrid::new(0.3).with_name("levelset");
    ls.set_grid_class(GridClass::LevelSet);
    for c in CoordBBox::from_coords(-5, -5, -5, 5, 5, 5) {
        let d = ((c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) as f32).sqrt() * 0.1 - 0.3;
        if d.abs() < 0.3 {
            ls.set_value(&c, d);
        } else {
            ls.tree_mut().set_value_off(&c, if d < 0.0 { -0.3 } else { 0.3 });
        }
    }
    out.push(ls.into());

    let mut d = DoubleGrid::new(-1.0).with_name("doubles");
    for i in 0..100 {
        d.set_value(&Coord::new(i, 2 * i, -3 * i), 1.0 / (i as f64 + 1.0));
    }
    let mut t = Transform::identity();
    t.pre_rotate(0.4, Axis::Z);
    t.post_scale(Vec3d::new(1.0, 2.0, 0.5));
    d.set_transform(t);
    out.push(d.into());

    let mut i32g = Int32Grid::new(9).with_name("ints");
    for i in 0..200 {
        i32g.set_value(&Coord::new(-i, i, i % 9), i * 1000 - 77);
    }
    i32g.tree_mut().fill(&CoordBBox::from_coords(128, 128, 128, 255, 255, 383), 1, true);
    out.push(i32g.into());

    let mut i64g = Int64Grid::new(0).with_name("longs");
    i64g.set_value(&Coord::new(1, 1, 1), i64::MAX);
    i64g.set_value(&Coord::new(-1, -1, -1), i64::MIN);
    out.push(i64g.into());

    let mut b = BoolGrid::new(false).with_name("bools");
    for i in 0..150 {
        let c = Coord::new(i % 11, i % 7, i);
        match i % 3 {
            0 => b.set_value(&c, true),
            1 => b.set_value(&c, false),
            _ => b.tree_mut().set_value_off(&c, true),
        }
    }
    b.tree_mut().fill(&CoordBBox::from_coords(0, 0, 4096, 4095, 4095, 8191), true, true);
    out.push(b.into());

    let mut m = BoolGrid::new(false).with_name("mask");
    for i in 0..120 {
        m.set_value(&Coord::new(3 * i, -i, i % 13), true);
    }
    m.tree_mut().fill(&CoordBBox::from_coords(1024, 1024, 1024, 1151, 1151, 1151), true, true);
    out.push(AnyGrid::Mask(m));

    let mut v = Vec3SGrid::new(Vec3f::new(0.0, 1.0, 0.0)).with_name("vel");
    v.set_vector_type(VecType::Covariant);
    for i in 0..100 {
        v.set_value(&Coord::new(i, i % 5, 0), Vec3f::new(i as f32, -0.25, 1.0 / (i as f32 + 1.0)));
    }
    out.push(v.into());

    let mut vd = Vec3DGrid::new(Vec3d::zero()).with_name("vel_d");
    vd.set_value(&Coord::new(5, 6, 7), Vec3d::new(0.1, 0.2, 0.3));
    out.push(vd.into());

    let mut vi = Vec3IGrid::new(Vec3i::new(1, 1, 1)).with_name("vec_i");
    vi.set_value(&Coord::new(-5, 6, -7), Vec3i::new(-1, 0, 1));
    out.push(vi.into());

    let mut dup = FloatGrid::new(0.0).with_name("floats");
    dup.set_value(&Coord::new(0, 0, 0), 1.0);
    out.push(dup.into());

    out.push(FloatGrid::new(1.0).into()); // unnamed and empty
    out
}

fn file_meta() -> MetaMap {
    let mut m = MetaMap::new();
    m.insert("author", "vdb-rs");
    m.insert("count", 3i32);
    m.insert("scale", Vec3d::new(1.0, 2.0, 3.0));
    m
}

fn dump_any(g: &AnyGrid) -> String {
    let mut s = String::new();
    dump_grid(&mut s, g);
    s
}

/// Metadata without the statistics added when writing.
fn without_stats(m: &MetaMap) -> MetaMap {
    let mut m = m.clone();
    for k in ["file_bbox_min", "file_bbox_max", "file_mem_bytes", "file_voxel_count", "file_compression"] {
        m.remove(k);
    }
    m
}

fn all_options() -> Vec<WriteOptions> {
    [
        Compression::NONE,
        Compression::ZIP,
        Compression::ACTIVE_MASK,
        Compression::ZIP | Compression::ACTIVE_MASK,
        Compression::BLOSC,
        Compression::BLOSC | Compression::ACTIVE_MASK,
    ]
    .into_iter()
    .map(WriteOptions::with_compression)
    .collect()
}

#[test]
fn roundtrip_all_types_and_compressions() {
    let dir = scratch_dir("roundtrip");
    let grids = sample_grids();
    for (k, opts) in all_options().iter().enumerate() {
        let path = dir.join(format!("rt_{}.vdb", k));
        io::write_file_with_options(&path, &grids, &file_meta(), opts).unwrap();
        let (meta, back) = io::read_file(&path).unwrap();
        assert_eq!(meta, file_meta());
        assert_eq!(back.len(), grids.len());
        for (a, b) in grids.iter().zip(&back) {
            assert_eq!(a.grid_type(), b.grid_type());
            assert_eq!(dump_any(a), dump_any(b), "{} with {}", a.name(), opts.compression);
            assert_eq!(without_stats(a.metadata()), without_stats(b.metadata()));
        }
        // Statistics metadata, as OpenVDB writes it.
        let ints = &back[3];
        let bbox = grids[3].eval_active_voxel_bbox();
        assert_eq!(ints.metadata().get("file_bbox_min"), Some(&MetaValue::Vec3i(bbox.min.as_vec3i())));
        assert_eq!(ints.metadata().get("file_bbox_max"), Some(&MetaValue::Vec3i(bbox.max.as_vec3i())));
        assert_eq!(
            ints.metadata().get("file_voxel_count"),
            Some(&MetaValue::Int64(grids[3].active_voxel_count() as i64))
        );
        let file = File::open(&path).unwrap();
        let ls = file.read_grid_metadata("levelset").unwrap();
        assert_eq!(ls.compression, opts.compression.without(Compression::ZIP));
        assert_eq!(ls.metadata.get_str("file_compression"), Some(ls.compression.to_string().as_str()));
        let names: Vec<_> = file.descriptors().iter().map(|d| d.unique_name.clone()).collect();
        assert_eq!(names[0], "floats\x1e0");
        assert_eq!(names[10], "floats\x1e1");
        assert_eq!(names[11], "\x1e0");
        assert_eq!(file.read_grid("floats[1]").unwrap().active_voxel_count(), 1);
    }
}

#[test]
fn roundtrip_half_floats() {
    let dir = scratch_dir("half");
    let mut grids = sample_grids();
    for g in grids.iter_mut() {
        vdb::with_any_grid!(g, g => g.set_save_float_as_half(true));
    }
    for opts in all_options() {
        let path = dir.join("half.vdb");
        io::write_file_with_options(&path, &grids, &MetaMap::new(), &opts).unwrap();
        let file = File::open(&path).unwrap();
        assert!(file.descriptors().iter().all(|d| d.save_float_as_half));
        let back = file.read_grids().unwrap();
        for (a, b) in grids.iter().zip(&back) {
            match (a, b) {
                (AnyGrid::Float(a), AnyGrid::Float(b)) => {
                    assert!(b.save_float_as_half());
                    for leaf in a.tree().leaves() {
                        for (n, v, _) in leaf.iter_all() {
                            let c = leaf.offset_to_global_coord(n);
                            let expected = half::f16::from_f32(*v).to_f32();
                            assert_eq!(b.get_value(&c), expected, "{} at {:?}", a.name(), c);
                        }
                    }
                    assert_eq!(b.background(), &half::f16::from_f32(*a.background()).to_f32());
                }
                (AnyGrid::Vec3S(a), AnyGrid::Vec3S(b)) => {
                    let v = a.get_value(&Coord::new(2, 2, 0));
                    let w = b.get_value(&Coord::new(2, 2, 0));
                    for i in 0..3 {
                        assert_eq!(w[i], half::f16::from_f32(v[i]).to_f32());
                    }
                }
                (AnyGrid::Double(a), AnyGrid::Double(b)) => {
                    let c = Coord::new(2, 4, -6);
                    assert_eq!(b.get_value(&c), half::f16::from_f64(a.get_value(&c)).to_f64());
                }
                (AnyGrid::Vec3D(a), AnyGrid::Vec3D(b)) => {
                    let c = Coord::new(5, 6, 7);
                    let (v, w) = (a.get_value(&c), b.get_value(&c));
                    for i in 0..3 {
                        assert_eq!(w[i], half::f16::from_f64(v[i]).to_f64());
                    }
                }
                // Integer, bool and mask grids are unaffected by half-float storage.
                _ => assert_eq!(dump_any(a), dump_any(b)),
            }
        }
    }
}

#[test]
fn roundtrip_stream() {
    let grids = sample_grids();
    let mut buf = Vec::new();
    io::write_stream(&mut buf, &grids, &file_meta(), &WriteOptions::default()).unwrap();
    let s = io::Stream::read(&buf[..]).unwrap();
    assert!(!s.header.has_grid_offsets);
    assert!(s.descriptors.iter().all(|d| d.grid_pos == 0 && d.end_pos == 0));
    assert_eq!(s.metadata, file_meta());
    let back = s.into_grids();
    for (a, b) in grids.iter().zip(&back) {
        assert_eq!(dump_any(a), dump_any(b));
    }
    // A stream saved to disk is readable as a file.
    let path = scratch_dir("stream").join("stream.vdb");
    std::fs::write(&path, &buf).unwrap();
    let (_, back) = io::read_file(&path).unwrap();
    assert_eq!(back.len(), grids.len());

    // The file layout can also be written to any writer.
    let mut buf2 = Vec::new();
    io::write_to(&mut buf2, &grids, &file_meta(), &WriteOptions::default()).unwrap();
    let s = io::Stream::read(&buf2[..]).unwrap();
    assert!(s.header.has_grid_offsets);
    assert_eq!(s.grids.len(), grids.len());
}

/// Re-writing a reference file without adding statistics reproduces its content exactly.
#[test]
fn rewrite_reference_files() {
    let dir = scratch_dir("rewrite");
    for name in REFERENCE_FILES.iter().filter(|n| **n != "points") {
        let src = data(&format!("{}.vdb", name));
        let file = File::open(&src).unwrap();
        let grids = file.read_grids().unwrap();
        let compression = file.read_grid_metadata(&file.descriptors()[0].unique_name).unwrap().compression;
        let opts = WriteOptions { compression, write_grid_stats: false };
        let out = dir.join(format!("{}.vdb", name));
        io::write_file_with_options(&out, &grids, file.metadata(), &opts).unwrap();
        let expected = std::fs::read_to_string(data(&format!("{}.dump.txt", name))).unwrap();
        assert_same_dump(&dump_file(&out), &expected, &format!("rewritten {}", name));
    }
}

#[test]
fn write_empty_file() {
    let path = scratch_dir("empty").join("empty.vdb");
    io::write_file(&path, &[], &MetaMap::new()).unwrap();
    let (meta, grids) = io::read_file(&path).unwrap();
    assert!(meta.is_empty() && grids.is_empty());
}

// ---------------------------------------------------------------------------
// C++ interoperability

/// Write files with every option and check that the C++ library reads exactly what this
/// crate reads. Requires the `vdb_ref` tool (see tests/data/cpp/README.md): set `VDB_REF`
/// or build it as tests/data/cpp/vdb_ref, then run `cargo test -- --ignored`.
#[test]
#[ignore]
fn cpp_reads_rust_files() {
    let bin = vdb_ref_binary().expect("vdb_ref not found: set VDB_REF or build tests/data/cpp/vdb_ref");
    let dir = scratch_dir("cpp_interop");
    let mut half = sample_grids();
    for g in half.iter_mut() {
        vdb::with_any_grid!(g, g => g.set_save_float_as_half(true));
    }
    let mut cases = Vec::new();
    for (k, opts) in all_options().iter().enumerate() {
        cases.push((format!("rust_{}.vdb", k), sample_grids(), *opts));
        cases.push((format!("rust_half_{}.vdb", k), half.clone(), *opts));
    }
    for (name, grids, opts) in cases {
        let path = dir.join(&name);
        io::write_file_with_options(&path, &grids, &file_meta(), &opts).unwrap();
        let out = std::process::Command::new(&bin).arg("dump").arg(&path).output().unwrap();
        assert!(out.status.success(), "{}: {}", name, String::from_utf8_lossy(&out.stderr));
        let cpp = String::from_utf8(out.stdout).unwrap();
        assert_same_dump(&dump_file(&path), &cpp, &name);
    }
    // A stream (no grid offsets) saved to disk.
    let path = dir.join("rust_stream.vdb");
    let mut f = std::fs::File::create(&path).unwrap();
    io::write_stream(&mut f, &sample_grids(), &file_meta(), &WriteOptions::default()).unwrap();
    drop(f);
    let out = std::process::Command::new(&bin).arg("dump").arg(&path).output().unwrap();
    assert!(out.status.success(), "stream: {}", String::from_utf8_lossy(&out.stderr));
    assert_same_dump(&dump_file(&path), &String::from_utf8(out.stdout).unwrap(), "rust_stream.vdb");
}
