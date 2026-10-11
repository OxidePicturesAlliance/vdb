//! Tests of the NanoVDB port against reference files written by C++ NanoVDB 32.9.2
//! (`tests/data/cpp/nanovdb_ref.cpp`, see `tests/data/cpp/README.md`).

use vdb::grid::{BoolGrid, DoubleGrid, FloatGrid, Grid, GridClass as VdbGridClass, Int32Grid, Int64Grid};
use vdb::math::{Coord, CoordBBox, Transform, Vec3d, Vec3f};
use vdb::nanovdb::io::{self, Codec};
use vdb::nanovdb::{
    self, create_nano_grid, nano_to_openvdb, AbsDiff, BuildType, CheckMode, CreateNanoGrid, Fp16, Fp4, Fp8, FpN,
    GridHandle, GridType, StatsMode, ValueIndex, ValueMask, ValueOnIndex,
};
use vdb::types::Value;

fn data(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/nanovdb").join(name)
}

// ---------------------------------------------------------------------------------------
// Source grids, mirroring nanovdb_ref.cpp
// ---------------------------------------------------------------------------------------

fn make_grid<T: Value>(name: &str, bg: T, val: impl Fn(i32) -> T) -> Grid<T> {
    let mut g = Grid::new(bg);
    g.set_name(name);
    let mut t = Transform::linear(0.5);
    t.post_translate(Vec3d::new(1.0, -2.0, 3.0));
    g.set_transform(t);
    let tree = g.tree_mut();
    tree.set_value(&Coord::new(0, 0, 0), val(0));
    tree.set_value(&Coord::new(1, 2, 3), val(1));
    tree.set_value(&Coord::new(-5, 7, 100), val(2));
    tree.set_value(&Coord::new(1000, -2000, 3000), val(3));
    tree.set_value_off(&Coord::new(10, 10, 10), val(4));
    tree.fill(&CoordBBox::new(Coord::splat(-256), Coord::splat(-129)), val(5), true);
    tree.fill(&CoordBBox::new(Coord::splat(64), Coord::splat(71)), val(6), true);
    tree.add_tile(3, &Coord::new(8192, 0, 0), val(7), true);
    tree.add_tile(3, &Coord::new(-8192, 4096, 0), val(8), false);
    for i in 0..20 {
        tree.set_value(&Coord::new(20 + i, 30, 40 + (i % 3)), val(9 + i));
    }
    g
}

fn float_grid() -> FloatGrid {
    let mut g = make_grid("float_grid", 2.0f32, |i| 0.25 * i as f32 - 1.0);
    g.set_grid_class(VdbGridClass::FogVolume);
    g
}
fn double_grid() -> DoubleGrid {
    make_grid("double_grid", -1.5f64, |i| 0.125 * i as f64 - 2.0)
}
fn int32_grid() -> Int32Grid {
    make_grid("int32_grid", 7i32, |i| i * 3 - 10)
}
fn int64_grid() -> Int64Grid {
    make_grid("int64_grid", -9i64, |i| i as i64 * 1000000007 - 5)
}
fn vec3f_grid() -> Grid<Vec3f> {
    make_grid("vec3f_grid", Vec3f::new(0.5, 0.0, -1.0), |i| Vec3f::new(i as f32, -0.5 * i as f32, 2.0))
}
fn vec3d_grid() -> Grid<Vec3d> {
    make_grid("vec3d_grid", Vec3d::new(0.0, 1.0, 0.0), |i| Vec3d::new(0.5 * i as f64, 3.0, -(i as f64)))
}
fn bool_grid() -> BoolGrid {
    make_grid("bool_grid", false, |i| i % 3 != 1)
}

fn band_grid() -> FloatGrid {
    let mut g = FloatGrid::new(0.3);
    g.set_name("band");
    g.set_grid_class(VdbGridClass::LevelSet);
    g.set_transform(Transform::linear(0.1));
    let tree = g.tree_mut();
    for i in -20..=20i32 {
        for j in -20..=20i32 {
            for k in -20..=20i32 {
                let d = ((i * i + j * j + k * k) as f32).sqrt() - 15.0;
                if d.abs() < 3.0 {
                    tree.set_value(&Coord::new(i, j, k), 0.1 * d);
                }
            }
        }
    }
    g
}

fn single_voxel_grid() -> FloatGrid {
    let mut g = FloatGrid::new(0.0);
    g.set_name("single");
    g.tree_mut().set_value(&Coord::new(3, 4, 5), 42.0);
    g
}

fn root_tiles_grid(name: &str) -> FloatGrid {
    let mut g = FloatGrid::new(1.0);
    g.set_name(name);
    g.tree_mut().add_tile(3, &Coord::new(0, 0, 0), 2.0, true);
    g.tree_mut().add_tile(3, &Coord::new(-4096, 0, 4096), 3.0, false);
    g
}

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

/// Assert that two grid buffers are identical, reporting the first difference.
fn assert_same_bytes(rust: &[u8], cpp: &[u8], what: &str) {
    if rust == cpp {
        return;
    }
    let n = rust.len().min(cpp.len());
    let first = (0..n).find(|&i| rust[i] != cpp[i]).unwrap_or(n);
    let count = (0..n).filter(|&i| rust[i] != cpp[i]).count();
    panic!(
        "{}: buffers differ (sizes {} vs {}), {} differing bytes, first at {}: rust {:02x?} cpp {:02x?}",
        what,
        rust.len(),
        cpp.len(),
        count,
        first,
        &rust[first..(first + 16).min(rust.len())],
        &cpp[first..(first + 16).min(cpp.len())]
    );
}

fn single_grid(h: &GridHandle) -> &[u8] {
    assert_eq!(h.grid_count(), 1);
    h.grid_bytes(0)
}

/// Grids in `types.nvdb`, in file order, converted by the Rust port.
fn rust_types() -> Vec<(&'static str, GridHandle)> {
    let f = float_grid();
    let b = bool_grid();
    let c = CreateNanoGrid::new(&f);
    vec![
        ("float", create_nano_grid::<f32, f32>(&f)),
        ("double", create_nano_grid::<f64, f64>(&double_grid())),
        ("int32", create_nano_grid::<i32, i32>(&int32_grid())),
        ("int64", create_nano_grid::<i64, i64>(&int64_grid())),
        ("vec3f", create_nano_grid::<Vec3f, Vec3f>(&vec3f_grid())),
        ("vec3d", create_nano_grid::<Vec3d, Vec3d>(&vec3d_grid())),
        ("bool", create_nano_grid::<bool, bool>(&b)),
        ("mask", create_nano_grid::<bool, ValueMask>(&b)),
        ("fp4", create_nano_grid::<f32, Fp4>(&f)),
        ("fp8", create_nano_grid::<f32, Fp8>(&f)),
        ("fp16", create_nano_grid::<f32, Fp16>(&f)),
        ("fpn", c.get_handle_fpn(AbsDiff::default())),
        ("index", c.get_index_handle::<ValueIndex>(1, true, true)),
        ("onindex", c.get_index_handle::<ValueOnIndex>(1, true, true)),
        ("onindex_bare", c.get_index_handle::<ValueOnIndex>(0, false, false)),
    ]
}

#[test]
fn types_match_cpp_byte_for_byte() {
    let cpp = io::read_grids(data("types.nvdb")).unwrap();
    let rust = rust_types();
    assert_eq!(cpp.len(), rust.len());
    for ((name, r), c) in rust.iter().zip(cpp.iter()) {
        assert_same_bytes(single_grid(r), single_grid(c), name);
    }
}

#[test]
fn modes_match_cpp_byte_for_byte() {
    let cpp = io::read_grids(data("modes.nvdb")).unwrap();
    let f = float_grid();
    let mut long = single_voxel_grid();
    long.set_name(&format!("{}_long_name", "x".repeat(300)));
    let mut empty = FloatGrid::new(5.0);
    empty.set_name("empty");
    let conv =
        |g: &FloatGrid, s: StatsMode, c: CheckMode| CreateNanoGrid::new(g).stats(s).checksum(c).get_handle::<f32>();
    let rust = [
        ("disable/partial", conv(&f, StatsMode::Disable, CheckMode::Partial)),
        ("bbox/full", conv(&f, StatsMode::BBox, CheckMode::Full)),
        ("minmax/disable", conv(&f, StatsMode::MinMax, CheckMode::Disable)),
        ("all/full", conv(&f, StatsMode::All, CheckMode::Full)),
        ("long name", create_nano_grid::<f32, f32>(&long)),
        ("empty", conv(&empty, StatsMode::All, CheckMode::Full)),
    ];
    assert_eq!(cpp.len(), rust.len());
    for ((name, r), c) in rust.iter().zip(cpp.iter()) {
        assert_same_bytes(single_grid(r), single_grid(c), name);
    }
    let h = cpp[4].header(0).unwrap();
    assert!(h.has_long_grid_name());
    assert_eq!(h.grid_name(), long.name());
    assert_eq!(h.blind_data_count(), 1);
}

#[test]
fn band_quantization_matches_cpp_byte_for_byte() {
    let cpp = io::read_grids(data("band.nvdb")).unwrap();
    let g = band_grid();
    let c = || CreateNanoGrid::new(&g).checksum(CheckMode::Full);
    let rust = [
        ("float", create_nano_grid::<f32, f32>(&g)),
        ("fp4 dithered", c().dither(true).get_handle::<Fp4>()),
        ("fp16", c().get_handle::<Fp16>()),
        ("fpn dithered", c().dither(true).get_handle_fpn(AbsDiff::default())),
        ("fpn abs 0.001", c().stats(StatsMode::MinMax).get_handle_fpn(AbsDiff::new(0.001))),
    ];
    assert_eq!(cpp.len(), rust.len());
    for ((name, r), c) in rust.iter().zip(cpp.iter()) {
        assert_same_bytes(single_grid(r), single_grid(c), name);
    }
}

#[test]
fn multi_grid_segment_and_raw_buffers() {
    // One segment with three grids (merged handle).
    let cpp = io::read_grids(data("multi.nvdb")).unwrap();
    assert_eq!(cpp.len(), 1);
    assert_eq!(cpp[0].grid_count(), 3);
    let merged = GridHandle::merge(&[
        create_nano_grid::<f32, f32>(&float_grid()),
        create_nano_grid::<i32, i32>(&int32_grid()),
        create_nano_grid::<Vec3f, Vec3f>(&vec3f_grid()),
    ])
    .unwrap();
    assert_same_bytes(merged.data(), cpp[0].data(), "merged");
    let names: Vec<String> =
        io::read_grid_meta_data(data("multi.nvdb")).unwrap().into_iter().map(|m| m.grid_name).collect();
    assert_eq!(names, ["float_grid", "int32_grid", "vec3f_grid"]);
    // Single grids read by index and by name get index 0 / count 1.
    let one = io::read_grid(data("multi.nvdb"), 1).unwrap();
    assert_eq!(one.grid_type(0), Some(GridType::Int32));
    assert_eq!(one.header(0).unwrap().grid_count(), 1);
    assert!(nanovdb::validate_checksum(one.grid_bytes(0), CheckMode::Full));
    assert_same_bytes(one.grid_bytes(0), single_grid(&create_nano_grid::<i32, i32>(&int32_grid())), "by index");
    let by_name = io::read_grid_by_name(data("multi.nvdb"), "vec3f_grid").unwrap();
    assert_eq!(by_name.grid_type(0), Some(GridType::Vec3f));
    assert!(io::has_grid(data("multi.nvdb"), "int32_grid").unwrap());
    assert!(!io::has_grid(data("multi.nvdb"), "nope").unwrap());

    // A raw buffer holding two grids (no file header).
    let raw = io::read_grids(data("raw.nvdb")).unwrap();
    assert_eq!(raw.len(), 1);
    let rust = GridHandle::merge(&[
        create_nano_grid::<f32, f32>(&root_tiles_grid("tiles_a")),
        CreateNanoGrid::new(&root_tiles_grid("tiles_b")).checksum(CheckMode::Full).get_handle::<f32>(),
    ])
    .unwrap();
    assert_same_bytes(rust.data(), raw[0].data(), "raw");
    let b = io::read_grid(data("raw.nvdb"), 1).unwrap();
    assert_eq!(b.header(0).unwrap().grid_name(), "tiles_b");

    // An uncompressed single grid.
    let single = io::read_grid(data("single_none.nvdb"), 0).unwrap();
    assert_same_bytes(single.data(), create_nano_grid::<f32, f32>(&single_voxel_grid()).data(), "single");
}

#[test]
fn primitives_match_cpp_byte_for_byte() {
    use vdb::nanovdb::primitives::*;
    let cpp = io::read_grids(data("primitives.nvdb")).unwrap();
    let c = Vec3d::new(1.5, 2.0, -3.0);
    let o = Vec3d::new(0.0, 0.0, 0.0);
    let rust = [
        ("sphere", create_level_set_sphere::<f32>(5.0, c, 0.5, 3.0, o, "sphere")),
        ("sphere_fp8", create_level_set_sphere::<Fp8>(5.0, c, 0.5, 3.0, o, "sphere_fp8")),
        ("torus", create_level_set_torus::<f32>(6.0, 2.0, o, 0.5, 3.0, Vec3d::new(0.25, 0.0, 0.0), "torus")),
        ("box", create_level_set_box::<f32>(10.0, 6.0, 4.0, Vec3d::new(1.0, 0.0, 0.0), 0.5, 3.0, o, "box")),
        ("fog", create_fog_volume_sphere::<f32>(5.0, Vec3d::new(-2.0, 0.0, 1.0), 0.25, 3.0, o, "fog_sphere")),
    ];
    assert_eq!(cpp.len(), rust.len());
    for ((name, r), c) in rust.iter().zip(cpp.iter()) {
        assert_same_bytes(single_grid(r), single_grid(c), name);
    }
}

// ---------------------------------------------------------------------------------------
// Values, statistics and conversions
// ---------------------------------------------------------------------------------------

/// Coordinates probing voxels, tiles at every level, inactive values and the background.
fn probe_coords<T: Value>(src: &Grid<T>) -> Vec<Coord> {
    let mut v: Vec<Coord> = src.tree().active_voxels().map(|(c, _)| c).collect();
    v.extend([
        Coord::new(10, 10, 10),
        Coord::new(-200, -150, -140),
        Coord::new(66, 67, 68),
        Coord::new(8192 + 5, 17, 4000),
        Coord::new(-8192, 4100, 3),
        Coord::new(-1, -1, -1),
        Coord::new(5, 5, 5),
        Coord::new(1 << 20, -(1 << 20), 77),
        Coord::new(i32::MIN + 5, i32::MAX - 5, 0),
    ]);
    v
}

fn check_values<T: Value, B: BuildType>(src: &Grid<T>, h: &GridHandle, conv: impl Fn(B::Value) -> T, what: &str) {
    let g = h.grid::<B>(0).unwrap_or_else(|| panic!("{}: wrong grid type", what));
    let (tree, acc) = (g.tree(), g.accessor());
    assert_eq!(g.active_voxel_count(), src.active_voxel_count(), "{}", what);
    assert_eq!(g.tree().background(), g.tree().root().background());
    for c in probe_coords(src) {
        let (sv, sa) = src.tree().probe_value(&c);
        let (nv, na) = tree.probe_value(c);
        assert_eq!(conv(nv), sv, "{} value at {:?}", what, c);
        assert_eq!(na, sa, "{} state at {:?}", what, c);
        assert_eq!(conv(acc.get_value(c)), sv, "{} accessor value at {:?}", what, c);
        assert_eq!(acc.is_active(c), sa, "{} accessor state at {:?}", what, c);
    }
}

#[test]
fn values_match_source_grids() {
    let rust = rust_types();
    check_values::<f32, f32>(&float_grid(), &rust[0].1, |v| v, "float");
    check_values::<f64, f64>(&double_grid(), &rust[1].1, |v| v, "double");
    check_values::<i32, i32>(&int32_grid(), &rust[2].1, |v| v, "int32");
    check_values::<i64, i64>(&int64_grid(), &rust[3].1, |v| v, "int64");
    check_values::<Vec3f, Vec3f>(&vec3f_grid(), &rust[4].1, |v| v, "vec3f");
    check_values::<Vec3d, Vec3d>(&vec3d_grid(), &rust[5].1, |v| v, "vec3d");
    check_values::<bool, bool>(&bool_grid(), &rust[6].1, |v| v, "bool");

    // Masks: the value is the active state.
    let src = bool_grid();
    let mask = rust[7].1.grid::<ValueMask>(0).unwrap();
    assert_eq!(mask.grid_class(), nanovdb::GridClass::Topology);
    for c in probe_coords(&src) {
        assert_eq!(mask.tree().is_active(c), src.tree().is_value_on(&c), "{:?}", c);
    }

    // Quantized grids: tiles are exact, voxels within one quantization step of the leaf range.
    let f = float_grid();
    // FpN uses the default fog-volume tolerance of 0.01.
    for (k, tol) in [(8usize, 8.0 / 15.0f32), (9, 8.0 / 255.0), (10, 8.0 / 65535.0), (11, 0.01)] {
        let h = &rust[k].1;
        let ty = h.grid_type(0).unwrap();
        let check = |get: &dyn Fn(Coord) -> (f32, bool)| {
            for c in probe_coords(&f) {
                let (sv, sa) = f.tree().probe_value(&c);
                let (nv, na) = get(c);
                assert_eq!(na, sa);
                assert!((nv - sv).abs() <= tol, "{:?} at {:?}: {} vs {}", ty, c, nv, sv);
            }
        };
        match ty {
            GridType::Fp4 => check(&|c| h.grid::<Fp4>(0).unwrap().tree().probe_value(c)),
            GridType::Fp8 => check(&|c| h.grid::<Fp8>(0).unwrap().tree().probe_value(c)),
            GridType::Fp16 => check(&|c| h.grid::<Fp16>(0).unwrap().tree().probe_value(c)),
            GridType::FpN => check(&|c| h.grid::<FpN>(0).unwrap().tree().probe_value(c)),
            t => panic!("unexpected type {:?}", t),
        }
    }

    // Index grids: values are offsets into the channel stored as blind data.
    let idx = rust[12].1.grid::<ValueIndex>(0).unwrap();
    let chan: Vec<f32> = idx.blind_values(0).unwrap();
    assert_eq!(chan.len() as u64, idx.value_count());
    let on = rust[13].1.grid::<ValueOnIndex>(0).unwrap();
    let on_chan: Vec<f32> = on.blind_values(0).unwrap();
    assert_eq!(on.blind_meta_data(0).name, "channel_0");
    for c in probe_coords(&f) {
        let (sv, sa) = f.tree().probe_value(&c);
        let (i, a) = idx.tree().probe_value(c);
        assert_eq!(a, sa);
        assert_eq!(chan[i as usize], sv, "index value at {:?}", c);
        let (j, a) = on.tree().probe_value(c);
        assert_eq!(a, sa);
        assert_eq!(on_chan[j as usize], if sa { sv } else { *f.background() }, "on-index value at {:?}", c);
    }
    let back = vdb::nanovdb::index_to_openvdb(&on, &on_chan, nanovdb::GridClass::FogVolume, Some("back"));
    for (c, v) in f.tree().active_voxels() {
        assert_eq!(back.tree().get_value(&c), v);
    }
}

#[test]
fn stats_bboxes_and_metadata() {
    let f = float_grid();
    let h = create_nano_grid::<f32, f32>(&f);
    let g = h.grid::<f32>(0).unwrap();
    assert_eq!(g.grid_name(), "float_grid");
    assert_eq!(g.grid_class(), nanovdb::GridClass::FogVolume);
    assert_eq!(g.index_bbox(), f.eval_active_voxel_bbox());
    assert_eq!(g.voxel_size(), Vec3d::splat(0.5));
    assert!(g.has_bbox() && g.has_min_max() && g.has_average() && g.has_std_deviation());
    assert_eq!(g.version().to_string(), "32.9.2");
    assert!(nanovdb::validate_checksum(g.bytes(), CheckMode::Full));
    // World bounds of the index bbox (max exclusive).
    let (lo, hi) = g.world_bbox();
    let b = g.index_bbox();
    assert_eq!(lo, f.transform().index_to_world(b.min.as_vec3d()));
    assert_eq!(hi, f.transform().index_to_world(b.max.offset(1).as_vec3d()));
    assert_eq!(g.world_to_index(g.index_to_world(Vec3d::new(1.0, 2.0, 3.0))), Vec3d::new(1.0, 2.0, 3.0));
    // Leaf statistics are those of the active voxels.
    for leaf in g.tree().leaves() {
        let vals: Vec<f32> = leaf.iter_on().map(|(_, v)| v).collect();
        if vals.is_empty() {
            assert!(!leaf.has_bbox());
            continue;
        }
        let mn = vals.iter().cloned().fold(f32::MAX, f32::min);
        let mx = vals.iter().cloned().fold(f32::MIN, f32::max);
        assert_eq!((leaf.minimum(), leaf.maximum()), (mn, mx));
        let avg = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        assert!((leaf.average() as f64 - avg).abs() < 1e-5);
        assert!(leaf.bbox().is_inside(&leaf.iter_on().next().unwrap().0));
    }
    let root = g.tree().root();
    assert_eq!(root.tile_count() as usize, f.tree().root().table().len());
    let (mn, mx) = g.tree().extrema();
    assert!(mn <= mx);
    assert_eq!(g.tree().node_count(0) as usize, f.tree().leaf_count());
    assert_eq!(g.tree().upper_nodes().count(), 4);
    assert_eq!(g.tree().lower_nodes().count(), g.tree().node_count(1) as usize);
    // Tampering is detected by the checksums.
    let mut full = CreateNanoGrid::new(&f).checksum(CheckMode::Full).get_handle::<f32>().data().to_vec();
    assert!(nanovdb::validate_checksum(&full, CheckMode::Full));
    let n = full.len() - 1000;
    full[n] ^= 1;
    assert!(!nanovdb::validate_checksum(&full, CheckMode::Full));
    let mut bytes = h.data().to_vec();
    bytes[100] ^= 1; // grid name: covered by the head checksum
    assert!(!nanovdb::validate_checksum(&bytes, CheckMode::Partial));
}

#[test]
fn round_trip_to_openvdb() {
    let rust = rust_types();
    let f = float_grid();
    let back = nano_to_openvdb(&rust[0].1.grid::<f32>(0).unwrap());
    assert!(*back.tree() == *f.tree());
    assert_eq!(back.name(), f.name());
    assert_eq!(back.grid_class(), VdbGridClass::FogVolume);
    assert!(back.transform().is_approx_equal(f.transform(), 0.0));
    assert!(*nano_to_openvdb(&rust[1].1.grid::<f64>(0).unwrap()).tree() == *double_grid().tree());
    assert!(*nano_to_openvdb(&rust[2].1.grid::<i32>(0).unwrap()).tree() == *int32_grid().tree());
    assert!(*nano_to_openvdb(&rust[3].1.grid::<i64>(0).unwrap()).tree() == *int64_grid().tree());
    assert!(*nano_to_openvdb(&rust[4].1.grid::<Vec3f>(0).unwrap()).tree() == *vec3f_grid().tree());
    assert!(*nano_to_openvdb(&rust[5].1.grid::<Vec3d>(0).unwrap()).tree() == *vec3d_grid().tree());
    assert!(*nano_to_openvdb(&rust[6].1.grid::<bool>(0).unwrap()).tree() == *bool_grid().tree());
    // Converting back and forth again gives the same buffer.
    let again = create_nano_grid::<f32, f32>(&back);
    assert_same_bytes(again.data(), rust[0].1.data(), "float round trip");
    let fp = nano_to_openvdb(&rust[10].1.grid::<Fp16>(0).unwrap());
    assert_eq!(fp.active_voxel_count(), f.active_voxel_count());
    let mask = nano_to_openvdb(&rust[7].1.grid::<ValueMask>(0).unwrap());
    assert_eq!(mask.active_voxel_count(), bool_grid().active_voxel_count());
    // The quantized C++ sphere decodes to the float sphere within one 8-bit step.
    let cpp = io::read_grids(data("primitives.nvdb")).unwrap();
    let ls = cpp[0].grid::<f32>(0).unwrap();
    let q = cpp[1].grid::<Fp8>(0).unwrap();
    for leaf in ls.tree().leaves() {
        for (c, v) in leaf.iter_on() {
            assert!((q.tree().get_value(c) - v).abs() <= 3.0 / 255.0 + 1e-6);
        }
    }
}

fn scratch_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("vdb_nanovdb_{}_{}", name, std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn io_round_trip_all_codecs() {
    let dir = scratch_dir("io");
    let handles: Vec<GridHandle> = rust_types().into_iter().map(|(_, h)| h).collect();
    for codec in [Codec::None, Codec::Zip, Codec::Blosc] {
        let path = dir.join(format!("types_{}.nvdb", codec.as_str()));
        io::write_grids(&path, &handles, codec).unwrap();
        let back = io::read_grids(&path).unwrap();
        assert_eq!(back.len(), handles.len());
        for (a, b) in handles.iter().zip(back.iter()) {
            assert_same_bytes(b.data(), a.data(), codec.as_str());
        }
        let meta = io::read_grid_meta_data(&path).unwrap();
        assert_eq!(meta.len(), handles.len());
        for (m, h) in meta.iter().zip(handles.iter()) {
            let g = h.header(0).unwrap();
            assert_eq!(m.grid_name, g.grid_name());
            assert_eq!(m.grid_type, g.grid_type());
            assert_eq!(m.voxel_count, g.active_voxel_count());
            assert_eq!(m.grid_size, g.grid_size());
            assert_eq!(m.codec, codec);
            assert_eq!(m.index_bbox, g.index_bbox());
        }
        let g = io::read_grid(&path, 4).unwrap();
        assert_eq!(g.grid_type(0), Some(GridType::Vec3f));
        let all = io::read_all_grids(&path).unwrap();
        assert_eq!(all.grid_count(), handles.len());
        assert_eq!(all.header(3).unwrap().grid_index(), 3);
    }
    // A merged handle written as one segment, and as a raw buffer.
    let merged = GridHandle::merge(&handles[..3]).unwrap();
    let seg = dir.join("merged.nvdb");
    io::write_grid(&seg, &merged, Codec::Zip).unwrap();
    let back = io::read_grids(&seg).unwrap();
    assert_eq!(back.len(), 1);
    assert_same_bytes(back[0].data(), merged.data(), "merged segment");
    let raw = dir.join("merged_raw.nvdb");
    merged.write_raw(&raw).unwrap();
    assert_same_bytes(io::read_grids(&raw).unwrap()[0].data(), merged.data(), "raw");
    assert_eq!(io::read_grid_by_name(&raw, "int32_grid").unwrap().grid_type(0), Some(GridType::Int32));
    let parts = merged.split();
    assert_eq!(parts.len(), 3);
    assert_same_bytes(parts[1].data(), handles[1].data(), "split");
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------------------
// Sampling and ray marching
// ---------------------------------------------------------------------------------------

#[test]
fn samplers_and_hdda_match_cpp() {
    use vdb::nanovdb::hdda::{first_active, zero_crossing, Ray};
    use vdb::nanovdb::sampler::*;
    let handle = vdb::nanovdb::primitives::create_level_set_sphere::<f32>(
        5.0,
        Vec3d::new(1.5, 2.0, -3.0),
        0.5,
        3.0,
        Vec3d::splat(0.0),
        "sphere",
    );
    let grid = handle.grid::<f32>(0).unwrap();
    let text = std::fs::read_to_string(data("samples.txt")).unwrap();
    let acc = grid.accessor();
    let cached0 = NearestSampler::new(&acc, true);
    let cached1 = TrilinearSampler::new(&acc, true);
    let cached2 = TriquadraticSampler::new(&acc, true);
    let cached3 = TricubicSampler::new(&acc, true);
    let tri = TrilinearSampler::new(&acc, false);
    let uncached = |order: usize, p: Vec3d| match order {
        0 => NearestSampler::new(&acc, false).sample(p),
        1 => TrilinearSampler::new(&acc, false).sample(p),
        2 => TriquadraticSampler::new(&acc, false).sample(p),
        _ => TricubicSampler::new(&acc, false).sample(p),
    };
    let (mut ns, mut nz) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let hex = |s: &str| f32::from_bits(u32::from_str_radix(s, 16).unwrap());
        let num = |s: &str| s.parse::<f64>().unwrap();
        let int = |s: &str| s.parse::<i32>().unwrap();
        match f[0] {
            "s" => {
                let (order, cache) = (int(f[1]) as usize, f[2] == "1");
                let p = Vec3d::new(num(f[3]), num(f[4]), num(f[5]));
                let v = if cache {
                    match order {
                        0 => cached0.sample(p),
                        1 => cached1.sample(p),
                        2 => cached2.sample(p),
                        _ => cached3.sample(p),
                    }
                } else {
                    uncached(order, p)
                };
                let want = hex(f[6]);
                assert_eq!(
                    v.to_bits(),
                    want.to_bits(),
                    "order {} cache {} at {:?}: {} vs {}",
                    order,
                    cache,
                    p,
                    v,
                    want
                );
                ns += 1;
            }
            "g" => {
                let p = Vec3d::new(num(f[1]), num(f[2]), num(f[3]));
                let g = tri.gradient(p);
                for k in 0..3 {
                    assert_eq!(g[k].to_bits(), hex(f[4 + k]).to_bits(), "gradient at {:?}", p);
                }
            }
            "z" | "a" => {
                let k = int(f[1]) as f32;
                let eye = [-20.0f32, -2.0 + 0.5 * k, -10.0 + 0.25 * k];
                let dir = [1.0f32, 0.05 * k - 0.3, 0.02 * k];
                let mut ray = Ray::<f32>::with_dir(eye, dir);
                let acc = grid.accessor();
                let hit = f[2] == "1";
                let ijk = Coord::new(int(f[3]), int(f[4]), int(f[5]));
                if f[0] == "z" {
                    let r = zero_crossing(&mut ray, &acc);
                    assert_eq!(r.is_some(), hit, "ray {}", k);
                    if let Some((c, v, t)) = r {
                        assert_eq!(c, ijk, "ray {}", k);
                        assert_eq!(v.to_bits(), hex(f[6]).to_bits(), "ray {} value", k);
                        assert_eq!(t.to_bits(), hex(f[7]).to_bits(), "ray {} time", k);
                    }
                    nz += 1;
                } else {
                    let r = first_active(&mut ray, &acc);
                    assert_eq!(r.is_some(), hit, "ray {}", k);
                    if let Some((c, t)) = r {
                        assert_eq!(c, ijk, "ray {}", k);
                        assert_eq!(t.to_bits(), hex(f[6]).to_bits(), "ray {} time", k);
                    }
                }
            }
            _ => panic!("bad line {}", line),
        }
    }
    assert_eq!(ns, 8 * 64);
    assert_eq!(nz, 16);
}

// ---------------------------------------------------------------------------------------
// C++ reads Rust files
// ---------------------------------------------------------------------------------------

fn nanovdb_ref_binary() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NANOVDB_REF") {
        return Some(p.into());
    }
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/cpp/nanovdb_ref");
    p.exists().then_some(p)
}

/// The line `nanovdb_ref check` prints for grid `n` of `h`.
fn expected_check_line(h: &GridHandle, n: usize) -> String {
    let g = h.header(n).unwrap();
    let full = nanovdb::eval_checksum(g.bytes(), CheckMode::Full);
    let valid = if g.node_count(0) == 0 {
        2
    } else if g.grid_type() == GridType::FpN {
        0 // the C++ validator rejects the over-allocated FpN buffers that C++ itself writes
    } else {
        1
    };
    format!(
        "{}|{}|{}|{}|{}|{},{},{}|{},{},{}|{:016x}|1|{}|{:016x}",
        g.grid_name(),
        g.grid_type().as_str(),
        g.grid_class().as_str(),
        g.grid_size(),
        g.active_voxel_count(),
        g.node_count(0),
        g.node_count(1),
        g.node_count(2),
        g.tile_count(1),
        g.tile_count(2),
        g.tile_count(3),
        g.checksum().0,
        valid,
        full.0
    )
}

#[test]
#[ignore = "needs the C++ checker: build tests/data/cpp/nanovdb_ref or set NANOVDB_REF"]
fn cpp_reads_rust_nvdb_files() {
    let bin = nanovdb_ref_binary().expect("nanovdb_ref not found: set NANOVDB_REF");
    let dir = scratch_dir("cpp");
    let mut handles: Vec<GridHandle> = rust_types().into_iter().map(|(_, h)| h).collect();
    let g = band_grid();
    handles.push(CreateNanoGrid::new(&g).checksum(CheckMode::Full).dither(true).get_handle_fpn(AbsDiff::default()));
    handles.push(vdb::nanovdb::primitives::create_level_set_torus::<f32>(
        6.0,
        2.0,
        Vec3d::splat(0.0),
        0.5,
        3.0,
        Vec3d::new(0.25, 0.0, 0.0),
        "torus",
    ));
    let mut long = single_voxel_grid();
    long.set_name(&"y".repeat(400));
    handles.push(CreateNanoGrid::new(&long).checksum(CheckMode::Full).get_handle::<f32>());
    let mut cases = Vec::new();
    for codec in [Codec::None, Codec::Zip, Codec::Blosc] {
        let path = dir.join(format!("rust_{}.nvdb", codec.as_str()));
        io::write_grids(&path, &handles, codec).unwrap();
        cases.push((path, handles.clone()));
    }
    let merged = GridHandle::merge(&handles[..4]).unwrap();
    let path = dir.join("rust_merged.nvdb");
    io::write_grid(&path, &merged, Codec::Zip).unwrap();
    cases.push((path, vec![merged.clone()]));
    let path = dir.join("rust_raw.nvdb");
    merged.write_raw(&path).unwrap();
    cases.push((path, vec![merged]));
    for (path, hs) in cases {
        let out = std::process::Command::new(&bin).arg("check").arg(&path).output().unwrap();
        assert!(out.status.success(), "{:?}: {}", path, String::from_utf8_lossy(&out.stderr));
        let cpp = String::from_utf8(out.stdout).unwrap();
        let expected: Vec<String> =
            hs.iter().flat_map(|h| (0..h.grid_count()).map(move |n| expected_check_line(h, n))).collect();
        let got: Vec<&str> = cpp.lines().collect();
        assert_eq!(got.len(), expected.len(), "{:?}", path);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_eq!(g, e, "{:?}", path);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
