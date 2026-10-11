//! More comparisons against reference results produced with the C++ OpenVDB 10 library:
//! enclosed regions with fill masks, activation, dense conversion, counting, CSG with
//! cancelled-tile pruning, tree merge policies, flood filling, pruning and active-tile
//! queries.

use vdb::grid::Grid;
use vdb::math::{Coord, CoordBBox, Vec3f};
use vdb::tools::activate::*;
use vdb::tools::composite::*;
use vdb::tools::count::*;
use vdb::tools::dense::*;
use vdb::tools::diagnostics::*;
use vdb::tools::find_active_values::*;
use vdb::tools::level_set_platonic::*;
use vdb::tools::level_set_sphere::*;
use vdb::tools::level_set_util::*;
use vdb::tools::prune::*;
use vdb::tools::signed_flood_fill::*;
use vdb::tools::tree_ops::{topology_union, MergePolicy};
use vdb::tree::{Node, Tree};

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-12)
}

fn sphere(r: f32, c: Vec3f, vs: f32) -> Grid<f32> {
    create_level_set_sphere::<f32>(r, c, vs, 3.0, true).unwrap()
}

/// `(active voxels, leaves, sum of active values, active tiles, sum of all leaf values)`.
fn stats(t: &Tree<f32>) -> (u64, usize, f64, u64, f64) {
    let mut s = 0.0;
    t.visit_tiles(&mut |b, v, on, _| {
        if on {
            s += *v as f64 * b.volume() as f64;
        }
    });
    s += t.active_voxels().map(|(_, v)| v as f64).sum::<f64>();
    let all: f64 = t.leaves().flat_map(|l| l.buffer().iter().map(|v| *v as f64)).sum();
    (t.active_voxel_count(), t.leaf_count(), s, count_active_tiles(t), all)
}

fn check_tree(name: &str, t: &Tree<f32>, e: (u64, usize, f64, u64, f64)) {
    let r = stats(t);
    assert_eq!((r.0, r.1, r.3), (e.0, e.1, e.3), "{name}");
    assert!(close(r.2, e.2, 1e-8) || (r.2 == 0.0 && e.2 == 0.0), "{name}: sum {} vs {}", r.2, e.2);
    assert!(close(r.4, e.4, 1e-8), "{name}: all {} vs {}", r.4, e.4);
}

#[test]
fn enclosed_regions_with_fill_mask() {
    let mut big = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let small = sphere(0.5, Vec3f::splat(0.0), 0.05);
    csg_difference(&mut big, &mut small.clone(), true, false).unwrap();
    let mut full = Tree::<bool>::new(false);
    full.fill(&CoordBBox::new(Coord::splat(-30), Coord::splat(30)), true, true);
    let e1 = extract_enclosed_region(&big, 0.0, Some(&full));
    assert_eq!((e1.active_voxel_count(), e1.tree().leaf_count()), (33371, 139));
    let mut part = Tree::<bool>::new(false);
    part.fill(&CoordBBox::new(Coord::new(0, -30, -30), Coord::splat(30)), true, true);
    let e2 = extract_enclosed_region(&big, 0.0, Some(&part));
    assert_eq!((e2.active_voxel_count(), e2.tree().leaf_count()), (33371, 139));
    let mut sp = Tree::<bool>::new(false);
    topology_union(&mut sp, small.tree(), false);
    let e3 = extract_enclosed_region(&big, 0.0, Some(&sp));
    assert_eq!((e3.active_voxel_count(), e3.tree().leaf_count()), (29202, 139));
}

#[test]
fn activation_and_counting() {
    let s = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let mut a = s.clone();
    activate(&mut a, -0.15, 0.0, true);
    assert_eq!((a.active_voxel_count(), count_active_tiles(a.tree())), (50733, 8));
    let mut b = s.clone();
    deactivate(&mut b, 0.0, 0.05, true);
    assert_eq!(b.active_voxel_count(), 19888);

    let q = CoordBBox::new(Coord::new(-3, -5, 0), Coord::new(30, 7, 9));
    assert_eq!(count_active_voxels_in_bbox(s.tree(), &q), 826);
    assert_eq!(count_active_leaf_voxels_in_bbox(s.tree(), &q), 826);
    assert_eq!(count_inactive_voxels(s.tree()), 549755783634);
    assert_eq!(count_inactive_leaf_voxels(s.tree()), 50642);
    let mm = min_max(s.tree());
    assert!(close(mm.min as f64, -0.148530677, 1e-8) && close(mm.max as f64, 0.1489125341, 1e-8));
}

#[test]
fn dense_conversion() {
    let bbox = CoordBBox::new(Coord::new(-5, -3, -7), Coord::new(27, 20, 9));
    let mut dense = Dense::<f32>::new(bbox, MemoryLayout::LayoutZYX).unwrap();
    for n in 0..dense.value_count() as usize {
        let c = dense.offset_to_coord(n);
        let inside = c[0] > 10 && c[0] < 25 && c[1] >= 0 && c[1] < 16 && c[2] >= -7 && c[2] < 1;
        // Rust's `%` truncates toward zero like C++'s, so negative remainders are kept.
        let v = if inside { 2.0 } else { ((c[0] * 13 + c[1] * 7 + c[2] * 3) % 5) as f32 * 0.25 };
        dense.set_value_at(n, v);
    }
    let mut t = Tree::<f32>::new(0.0);
    copy_from_dense(&dense, &mut t, 0.3, false);
    check_tree("fromDense", &t, (8785, 60, 8092.5, 0, 8092.5));
    let mut t2 = Tree::<f32>::new(0.0);
    t2.set_value(&Coord::new(100, 100, 100), 1.0);
    t2.fill(&CoordBBox::new(Coord::splat(-8), Coord::splat(-1)), 5.0, true);
    copy_from_dense(&dense, &mut t2, 0.0, false);
    check_tree("fromDense2", &t2, (11537, 61, 10629.5, 0, 10629.5));
    let mut d2 =
        Dense::<f32>::new(CoordBBox::new(Coord::splat(-10), Coord::splat(10)), MemoryLayout::LayoutXYZ).unwrap();
    copy_to_dense(&t2, &mut d2, false);
    let sd: f64 = (0..d2.value_count() as usize).map(|n| d2.get_value_at(n) as f64 * (n % 7) as f64).sum();
    assert!(close(sd, 9485.25, 1e-12));
}

#[test]
fn csg_cancelled_tiles() {
    let s = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let (mut x, mut y) = (s.clone(), s.clone());
    csg_difference(&mut x, &mut y, true, true).unwrap();
    assert_eq!((x.active_voxel_count(), x.tree().leaf_count()), (0, 0));
    let (mut x, mut y) = (s.clone(), s.clone());
    csg_difference(&mut x, &mut y, true, false).unwrap();
    assert_eq!((x.active_voxel_count(), x.tree().leaf_count()), (30254, 158));
}

#[test]
fn merge_policies() {
    let expected = [
        (68719493002, 21, 5.497558834e+11, 17, 42033.0),
        (12681, 21, 51632.0, 9, 42026.0),
        (68719493002, 21, 5.497558834e+11, 17, 42033.0),
    ];
    let policies = [MergePolicy::ActiveStates, MergePolicy::Nodes, MergePolicy::ActiveStatesAndNodes];
    for (policy, e) in policies.into_iter().zip(expected) {
        let mut a = Tree::<f32>::new(0.0);
        let mut b = Tree::<f32>::new(1.0);
        a.fill(&CoordBBox::new(Coord::splat(0), Coord::splat(15)), 2.0, true);
        a.fill(&CoordBBox::new(Coord::new(128, 0, 0), Coord::new(255, 127, 127)), 3.0, false);
        a.set_value(&Coord::new(300, 0, 0), 4.0);
        a.set_value_off(&Coord::new(301, 0, 0), 1.0);
        b.fill(&CoordBBox::new(Coord::splat(8), Coord::splat(30)), 5.0, true);
        b.fill(&CoordBBox::new(Coord::new(128, 0, 0), Coord::new(135, 7, 7)), 6.0, true);
        b.set_value(&Coord::new(300, 0, 1), 7.0);
        b.set_value_off(&Coord::new(302, 0, 0), -1.0);
        b.fill(&CoordBBox::new(Coord::new(4096, 0, 0), Coord::new(8191, 4095, 4095)), 8.0, true);
        b.set_value(&Coord::new(-4000, 0, 0), 9.0);
        a.merge(&mut b, policy);
        assert!(b.is_empty());
        let r = stats(&a);
        assert_eq!((r.0, r.1, r.3), (e.0, e.1, e.3), "merge {policy:?}");
        assert!(close(r.2, e.2, 1e-9) && close(r.4, e.4, 1e-12), "merge {policy:?}: {r:?}");
    }
}

#[test]
fn flood_fill_and_prune() {
    let s = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let mut f = s.clone();
    signed_flood_fill_with_values(f.tree_mut(), 2.0, -7.0, true).unwrap();
    check_tree("sffValues", f.tree(), (30254, 158, 402.4881175, 0, -45760.51188));
    let mut t = s.clone();
    for l in t.tree_mut().leaves_mut() {
        if l.origin()[0] > 0 {
            l.value_mask_mut().set_all_off();
        }
    }
    let mut u = t.clone();
    prune_inactive(t.tree_mut(), true);
    check_tree("pruneInactive", t.tree(), (20761, 106, 229.7602427, 0, 1513.310294));
    prune_level_set(u.tree_mut(), true).unwrap();
    check_tree("pruneLevelSet", u.tree(), (20761, 106, 229.7602427, 0, 1513.310294));
    let mut w = s.clone();
    prune_level_set_with_values(w.tree_mut(), 1.0, -2.0, true).unwrap();
    check_tree("pruneLevelSet2", w.tree(), (30254, 158, 402.4881175, 0, 3083.888224));
}

#[test]
fn active_tile_queries() {
    let mut t = Tree::<f32>::new(0.0);
    t.fill(&CoordBBox::new(Coord::splat(0), Coord::splat(255)), 1.0, true);
    t.fill(&CoordBBox::new(Coord::new(256, 0, 0), Coord::new(263, 7, 7)), 2.0, true);
    t.set_value(&Coord::new(300, 5, 5), 3.0);
    t.add_tile(3, &Coord::new(-4096, 0, 0), 4.0, true);
    let q = CoordBBox::new(Coord::new(-10, 1, 2), Coord::new(310, 6, 6));
    let tiles: Vec<_> = active_tiles(&t, &q).iter().map(|d| (d.bbox, d.value, d.level)).collect();
    assert_eq!(
        tiles,
        vec![
            (CoordBBox::new(Coord::new(-10, 1, 2), Coord::new(-1, 6, 6)), 4.0, 3),
            (CoordBBox::new(Coord::new(256, 1, 2), Coord::new(263, 6, 6)), 2.0, 1),
            (CoordBBox::new(Coord::new(0, 1, 2), Coord::new(127, 6, 6)), 1.0, 2),
            (CoordBBox::new(Coord::new(128, 1, 2), Coord::new(255, 6, 6)), 1.0, 2),
        ]
    );
    assert_eq!(count_active_values(&t, &q), 8221);
    assert!(any_active_voxels(&t, &q));
    assert!(!any_active_tiles(&t, &CoordBBox::new(Coord::new(300, 5, 5), Coord::new(300, 5, 5))));
}

#[test]
fn diagnostics_reference() {
    let mut a = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let mut b = sphere(0.8, Vec3f::new(0.7, 0.1, 0.0), 0.05);
    csg_union(&mut a, &mut b, true, false).unwrap();
    assert_eq!(check_level_set(&a, 9), "2 voxels are outside the range of NormGrad [0.5,1.5]\n");
    let mut c = CheckLevelSet::new(&a);
    assert_eq!(c.check_eikonal(true, 0.9, 1.1), "77 voxels are outside the range of NormGrad [0.9,1.1]\n");
    assert_eq!((c.diagnose().value_count(), c.diagnose().failure_count()), (77, 77));
    let mut d = Diagnose::new(&a);
    let ng = CheckNormGrad::new(&a, 0.95f32 as f64, 1.05f32 as f64).unwrap();
    assert_eq!(d.check(&ng, true, true, false, false), "411 voxels are outside the range of NormGrad [0.95,1.05]\n");
    assert_eq!(d.value_count(), 411);
    let cr = CheckRange::new(-0.1f32 as f64, 0.1f32 as f64).unwrap();
    assert_eq!(
        d.check(&cr, false, true, true, true),
        "Background is outside the value range [-0.1,0.1]\n11664 voxels are outside the value range [-0.1,0.1]\n"
    );
    let mut f = a.clone();
    sdf_to_fog_volume(&mut f, f32::MAX);
    assert_eq!(check_fog_volume(&f, 6), "");
    f.set_value(&Coord::new(0, 0, 0), 2.0);
    assert_eq!(check_fog_volume(&f, 6), "1 voxel is outside the value range [0,1]\n");
}

/// OpenVDB builds Platonic solids with mesh scan conversion, which uses a slightly
/// narrower band near edges; this port uses exact convex-polytope distances, so its active
/// set is a slight superset of OpenVDB's (counts below are OpenVDB 10's).
#[test]
fn platonic_vs_mesh_to_level_set() {
    let reference = [(4u32, 10724u64), (6, 11966), (8, 17204), (12, 24983), (20, 23136)];
    for (faces, cpp_count) in reference {
        let g = create_level_set_platonic::<f32>(faces, 1.0, Vec3f::new(0.1, 0.2, -0.05), 0.05, 3.0).unwrap();
        let n = g.active_voxel_count();
        println!("faces {faces}: {n} active voxels (OpenVDB: {cpp_count})");
        assert!(n >= cpp_count && (n - cpp_count) as f64 <= 0.025 * cpp_count as f64, "faces {faces}: {n}");
        assert_eq!(g.get_value(&Coord::new(5, 3, 1)), -0.15);
    }
    let tet = create_level_set_tetrahedron::<f32>(1.0, Vec3f::new(0.1, 0.2, -0.05), 0.05, 3.0).unwrap();
    assert!((tet.get_value(&Coord::new(0, 0, 0)) - -0.13333).abs() < 1e-5);
}
