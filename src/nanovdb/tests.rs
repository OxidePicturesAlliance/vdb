//! Unit tests of the NanoVDB module (the C++ interoperability tests live in
//! `tests/nanovdb.rs`).

use super::hdda::{first_active, zero_crossing, Dda, Hdda, Ray};
use super::sampler::{NearestSampler, Sampler, TricubicSampler, TrilinearSampler, TriquadraticSampler};
use super::tree::{coord_to_key, key_to_coord};
use super::*;
use crate::grid::{FloatGrid, Grid};
use crate::math::{Coord, CoordBBox, Vec3d};

fn small_grid() -> FloatGrid {
    let mut g = FloatGrid::new(-1.0);
    g.set_name("small");
    for (i, c) in
        [Coord::new(0, 0, 0), Coord::new(7, 7, 7), Coord::new(8, 0, 0), Coord::new(-1, -200, 5000)].iter().enumerate()
    {
        g.tree_mut().set_value(c, i as f32 + 0.5);
    }
    g
}

#[test]
fn root_keys_round_trip() {
    for c in [Coord::new(0, 0, 0), Coord::new(4096, -4096, 8192), Coord::new(-(1 << 30), 1 << 30, -4096)] {
        assert_eq!(key_to_coord(coord_to_key(c)), c);
    }
    assert_eq!(coord_to_key(Coord::new(4095, 4095, 4095)), 0);
}

#[test]
fn small_grid_layout_and_values() {
    let g = small_grid();
    let h = create_nano_grid::<f32, f32>(&g);
    assert_eq!(h.grid_count(), 1);
    let n = h.grid::<f32>(0).unwrap();
    assert!(h.grid::<f64>(0).is_none(), "type mismatch must be rejected");
    assert_eq!(n.grid_type(), GridType::Float);
    assert_eq!(n.grid_name(), "small");
    assert_eq!(n.magic(), MAGIC_GRID);
    assert_eq!(n.node_count(0), 3);
    assert_eq!(n.node_count(1), 2);
    assert_eq!(n.node_count(2), 2);
    assert_eq!(n.tile_count(3), 0);
    assert_eq!(n.active_voxel_count(), 4);
    assert_eq!(n.index_bbox(), CoordBBox::new(Coord::new(-1, -200, 0), Coord::new(8, 7, 5000)));
    let tree = n.tree();
    assert_eq!(tree.background(), -1.0);
    assert_eq!(tree.get_value(Coord::new(7, 7, 7)), 1.5);
    assert_eq!(tree.get_value(Coord::new(6, 7, 7)), -1.0);
    assert!(!tree.is_active(Coord::new(6, 7, 7)));
    assert_eq!(tree.probe_value(Coord::new(-1, -200, 5000)), (3.5, true));
    let root = tree.root();
    assert_eq!(root.minimum(), 0.5);
    assert_eq!(root.maximum(), 3.5);
    assert!((root.average() - 2.0).abs() < 1e-6);
    let leaf = root.probe_leaf(Coord::new(1, 1, 1)).unwrap();
    assert_eq!(leaf.origin(), Coord::new(0, 0, 0));
    assert_eq!(leaf.bbox(), CoordBBox::new(Coord::new(0, 0, 0), Coord::new(7, 7, 7)));
    assert_eq!(leaf.value_mask().count_on(), 2);
    assert_eq!(leaf.value_mask().iter_on().collect::<Vec<_>>(), vec![0, 511]);
    assert_eq!(leaf.mem_usage(), 2144);
    let upper = root.probe_child(Coord::new(0, 0, 0)).unwrap();
    assert_eq!(upper.origin(), Coord::new(0, 0, 0));
    assert!(upper.flags() & 2 != 0);
}

#[test]
fn accessor_caches_nodes() {
    let g = small_grid();
    let h = create_nano_grid::<f32, f32>(&g);
    let n = h.grid::<f32>(0).unwrap();
    let acc = n.accessor();
    assert_eq!(acc.cached_node(0), None);
    assert_eq!(acc.get_value(Coord::new(0, 0, 0)), 0.5);
    let leaf = acc.cached_node(0).unwrap();
    // Same leaf: served from the cache.
    assert_eq!(acc.get_value(Coord::new(7, 7, 7)), 1.5);
    assert_eq!(acc.cached_node(0), Some(leaf));
    assert!(matches!(acc.locate(Coord::new(3, 3, 3)), ValueLocation::Leaf(o, _) if o == leaf));
    // Neighbouring leaf in the same lower node.
    assert_eq!(acc.get_value(Coord::new(8, 0, 0)), 2.5);
    assert_ne!(acc.cached_node(0), Some(leaf));
    assert_eq!(acc.get_value(Coord::new(-1, -200, 5000)), 3.5);
    assert_eq!(acc.get_value(Coord::new(100000, 0, 0)), -1.0);
    assert!(matches!(acc.locate(Coord::new(100000, 0, 0)), ValueLocation::Background));
    assert!(matches!(acc.locate(Coord::new(200, 0, 0)), ValueLocation::Upper(..)));
    assert!(matches!(acc.locate(Coord::new(20, 0, 0)), ValueLocation::Lower(..)));
    acc.clear();
    assert_eq!(acc.cached_node(0), None);
    assert_eq!(acc.get_dim(Coord::new(1, 1, 1)), 1);
    assert_eq!(acc.get_dim(Coord::new(20, 0, 0)), 8);
    assert_eq!(acc.get_dim(Coord::new(200, 0, 0)), 128);
    assert_eq!(acc.get_dim(Coord::new(100000, 0, 0)), 4096);
}

#[test]
fn invalid_buffers_are_rejected() {
    assert!(GridHandle::from_bytes(&[0u8; 100]).is_err());
    let mut bytes = create_nano_grid::<f32, f32>(&small_grid()).data().to_vec();
    bytes[0] ^= 0xff;
    assert!(GridHandle::from_bytes(&bytes).is_err());
    let h = create_nano_grid::<f32, f32>(&small_grid());
    assert!(GridHandle::from_bytes(&h.data()[..h.size() - 1]).is_err());
    assert!(matches!(NanoGrid::<i32>::new(h.data()), Err(NanoError::TypeMismatch { .. })));
}

#[test]
fn checksum_modes() {
    let g = small_grid();
    let none = CreateNanoGrid::new(&g).checksum(CheckMode::Disable).get_handle::<f32>();
    assert!(none.header(0).unwrap().checksum().is_empty());
    let part = CreateNanoGrid::new(&g).get_handle::<f32>();
    let full = CreateNanoGrid::new(&g).checksum(CheckMode::Full).get_handle::<f32>();
    let (cp, cf) = (part.header(0).unwrap().checksum(), full.header(0).unwrap().checksum());
    assert!(cp.is_half() && cf.is_full());
    assert_eq!(cp.head(), cf.head());
    assert_eq!(cf.mode(), CheckMode::Full);
    assert_eq!(eval_checksum(full.data(), CheckMode::Full), cf);
    let mut b = full.data().to_vec();
    update_checksum(&mut b, CheckMode::Partial);
    assert_eq!(&b[..], part.data());
}

#[test]
fn stats_modes() {
    let g = small_grid();
    let h = CreateNanoGrid::new(&g).stats(StatsMode::Disable).get_handle::<f32>();
    let n = h.grid::<f32>(0).unwrap();
    assert!(!n.has_bbox() && !n.has_min_max());
    assert!(data::bbox_is_empty(&n.index_bbox()));
    let mut b = h.data().to_vec();
    update_grid_stats::<f32>(&mut b, StatsMode::All);
    update_checksum(&mut b, CheckMode::Partial);
    assert_eq!(&b[..], create_nano_grid::<f32, f32>(&g).data());
    let mm = CreateNanoGrid::new(&g).stats(StatsMode::MinMax).get_handle::<f32>();
    let n = mm.grid::<f32>(0).unwrap();
    assert!(n.has_min_max() && !n.has_average());
    assert_eq!(n.tree().root().maximum(), 3.5);
}

#[test]
fn quantized_and_index_types() {
    let mut g = Grid::<f32>::new(0.0);
    for i in 0..8 {
        for j in 0..8 {
            g.tree_mut().set_value(&Coord::new(i, j, 3), (i * 8 + j) as f32 / 63.0);
        }
    }
    let c = CreateNanoGrid::new(&g);
    for (bits, h) in [(4, c.get_handle::<Fp4>()), (8, c.get_handle::<Fp8>()), (16, c.get_handle::<Fp16>())] {
        let tol = 1.0 / ((1u32 << bits) - 1) as f32;
        let check = |get: &dyn Fn(Coord) -> f32| {
            for (ijk, v) in g.tree().active_voxels() {
                assert!((get(ijk) - v).abs() <= tol * 0.5 + 1e-6, "{} bits at {:?}", bits, ijk);
            }
        };
        match bits {
            4 => check(&|ijk| h.grid::<Fp4>(0).unwrap().tree().get_value(ijk)),
            8 => check(&|ijk| h.grid::<Fp8>(0).unwrap().tree().get_value(ijk)),
            _ => check(&|ijk| h.grid::<Fp16>(0).unwrap().tree().get_value(ijk)),
        }
    }
    // A tight tolerance forces 16 bits, a loose one fewer.
    let tight = c.get_handle_fpn(AbsDiff::new(1e-6));
    let leaf = tight.grid::<FpN>(0).unwrap().tree().leaves().next().unwrap();
    assert_eq!(FpN::log2_bit_width(leaf.bytes(), 0), 4);
    let loose = c.get_handle_fpn(AbsDiff::new(0.05));
    let leaf = loose.grid::<FpN>(0).unwrap().tree().leaves().next().unwrap();
    assert!(FpN::log2_bit_width(leaf.bytes(), 0) < 4);
    for (ijk, v) in g.tree().active_voxels() {
        assert!((loose.grid::<FpN>(0).unwrap().tree().get_value(ijk) - v).abs() <= 0.05);
    }
    assert!(RelDiff::new(0.1).accept(1.0, 1.05));
    // Index grids.
    let on = c.get_index_handle::<ValueOnIndex>(1, false, true);
    let n = on.grid::<ValueOnIndex>(0).unwrap();
    assert_eq!(n.value_count(), 65);
    let leaf = n.tree().leaves().next().unwrap();
    assert_eq!(ValueOnIndex::leaf_value_count(leaf.bytes(), 0), 64);
    let vals: Vec<f32> = n.blind_values(0).unwrap();
    for (ijk, v) in g.tree().active_voxels() {
        assert_eq!(vals[n.tree().get_value(ijk) as usize], v);
    }
    assert_eq!(n.tree().get_value(Coord::new(0, 0, 0)), 0);
}

#[test]
fn samplers_reproduce_linear_fields() {
    let mut g = Grid::<f64>::new(0.0);
    let f = |c: Coord| 1.0 + 2.0 * c[0] as f64 - 0.5 * c[1] as f64 + 0.25 * c[2] as f64;
    for i in -4..12 {
        for j in -4..12 {
            for k in -4..12 {
                let c = Coord::new(i, j, k);
                g.tree_mut().set_value(&c, f(c));
            }
        }
    }
    let h = create_nano_grid::<f64, f64>(&g);
    let n = h.grid::<f64>(0).unwrap();
    let acc = n.accessor();
    let p = Vec3d::new(3.25, 4.5, 5.75);
    let exact = 1.0 + 2.0 * p[0] - 0.5 * p[1] + 0.25 * p[2];
    assert_eq!(NearestSampler::new(&acc, true).sample(p), f(Coord::new(3, 5, 6)));
    for cache in [false, true] {
        assert!((TrilinearSampler::new(&acc, cache).sample(p) - exact).abs() < 1e-12);
        assert!((TriquadraticSampler::new(&acc, cache).sample(p) - exact).abs() < 1e-12);
        assert!((TricubicSampler::new(&acc, cache).sample(p) - exact).abs() < 1e-12);
    }
    let grad = TrilinearSampler::new(&acc, false).gradient(p);
    assert!((grad[0] - 2.0).abs() < 1e-12 && (grad[1] + 0.5).abs() < 1e-12 && (grad[2] - 0.25).abs() < 1e-12);
    assert!(!TrilinearSampler::new(&acc, false).zero_crossing(p));
}

#[test]
fn hdda_and_ray_marching() {
    // Index-space level set of a sphere of radius 10 around the origin.
    let mut g = FloatGrid::new(3.0);
    for i in -16..=16i32 {
        for j in -16..=16i32 {
            for k in -16..=16i32 {
                let d = ((i * i + j * j + k * k) as f32).sqrt() - 10.0;
                if d.abs() < 3.0 {
                    g.tree_mut().set_value(&Coord::new(i, j, k), d);
                }
            }
        }
    }
    let h = create_nano_grid::<f32, f32>(&g);
    let n = h.grid::<f32>(0).unwrap();
    let acc = n.accessor();
    let mut ray = Ray::<f32>::with_dir([-50.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let (ijk, v, t) = zero_crossing(&mut ray, &acc).expect("hit");
    assert_eq!(ijk, Coord::new(-9, 0, 0));
    assert!(v < 0.0 && (t - 41.0).abs() < 1e-4);
    let mut ray = Ray::<f32>::with_dir([-50.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let (c, _) = first_active(&mut ray, &acc).expect("active");
    assert_eq!(c, Coord::new(-12, 0, 0));
    let mut miss = Ray::<f32>::with_dir([-50.0, 40.0, 0.5], [1.0, 0.0, 0.0]);
    assert!(zero_crossing(&mut miss, &acc).is_none());
    // Plain DDA visits every voxel along an axis-aligned ray.
    let ray = Ray::<f64>::new([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 0.0, 5.0);
    let mut dda = Dda::new(&ray, 1);
    let mut visited = vec![dda.voxel()];
    while dda.step() {
        visited.push(dda.voxel());
    }
    assert_eq!(visited, (0..6).map(|i| Coord::new(i, 0, 0)).collect::<Vec<_>>());
    let mut hdda = Hdda::new(&ray, 8);
    assert_eq!(hdda.voxel(), Coord::new(0, 0, 0));
    assert!(hdda.update(&ray, 1));
    assert!(!hdda.update(&ray, 1));
    assert!(hdda.step());
    assert_eq!(hdda.voxel(), Coord::new(1, 0, 0));
    // World to index ray transformation.
    let map = Map::from_scale_translation(0.5, Vec3d::new(1.0, 0.0, 0.0));
    let w = Ray::<f64>::new([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.0, 1.0);
    let i = w.world_to_index(&map);
    assert_eq!(i.eye(), [0.0, 0.0, 0.0]);
    assert_eq!(i.dir(), [1.0, 0.0, 0.0]);
    assert_eq!(i.t1(), 4.0);
}

#[test]
fn map_and_blind_data() {
    let m = Map::from_scale_translation(0.25, Vec3d::new(1.0, 2.0, 3.0));
    assert_eq!(m.voxel_size(), Vec3d::splat(0.25));
    let p = Vec3d::new(4.0, -8.0, 12.0);
    assert_eq!(m.apply_inverse_map(m.apply_map(p)), p);
    let mut b = vec![0u8; Map::SIZE];
    m.write(&mut b, 0);
    assert_eq!(Map::read(&b, 0), m);
    let mut long = small_grid();
    long.set_name(&"n".repeat(300));
    let h = create_nano_grid::<f32, f32>(&long);
    let hd = h.header(0).unwrap();
    assert!(hd.has_long_grid_name());
    assert_eq!(hd.grid_name(), long.name());
    assert_eq!(hd.short_grid_name().len(), 255);
    assert_eq!(hd.blind_meta_data(0).data_class, GridBlindDataClass::GridName);
    assert_eq!(hd.find_blind_data("grid name"), Some(0));
}
