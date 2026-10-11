//! Comparison of the tools against reference results produced with the C++ OpenVDB 10
//! library (the generating program performs the same operations with the same inputs).
//!
//! Each check compares active voxel counts, leaf and tile counts, and the sum of active
//! values (tiles weighted by their voxel count).

use vdb::grid::Grid;
use vdb::math::{Coord, CoordBBox, Transform, Vec3f};
use vdb::tools::change_background::*;
use vdb::tools::composite::*;
use vdb::tools::count::*;
use vdb::tools::diagnostics::*;
use vdb::tools::level_set_measure::*;
use vdb::tools::level_set_sphere::*;
use vdb::tools::level_set_util::*;
use vdb::tools::morphology::*;
use vdb::tools::prune::*;
use vdb::tools::statistics::*;
use vdb::tools::topology_to_level_set::*;
use vdb::types::Value;

fn sum_on<T: Value + Into<f64>>(g: &Grid<T>) -> f64 {
    let mut s = 0.0;
    g.tree().visit_tiles(&mut |b, v, on, _| {
        if on {
            s += (*v).into() * b.volume() as f64;
        }
    });
    s + g.tree().active_voxels().map(|(_, v)| v.into()).sum::<f64>()
}

fn sum_all<T: Value + Into<f64>>(g: &Grid<T>) -> f64 {
    g.tree().leaves().flat_map(|l| l.buffer().iter().map(|v| (*v).into())).sum()
}

/// Assert `(active voxels, leaves, sum of active values, active tiles)`.
fn check<T: Value + Into<f64>>(name: &str, g: &Grid<T>, active: u64, leaves: usize, sum: f64, tiles: u64) {
    assert_eq!(g.active_voxel_count(), active, "{name}: active voxel count");
    assert_eq!(g.tree().leaf_count(), leaves, "{name}: leaf count");
    let s = sum_on(g);
    assert!((s - sum).abs() <= 1e-6 * sum.abs().max(1.0), "{name}: sum {s} != {sum}");
    assert_eq!(count_active_tiles(g.tree()), tiles, "{name}: active tile count");
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-12)
}

fn sphere(r: f32, c: Vec3f, vs: f32) -> Grid<f32> {
    create_level_set_sphere::<f32>(r, c, vs, 3.0, true).unwrap()
}

#[test]
fn level_set_spheres() {
    check("sphereA", &sphere(5.0, Vec3f::new(0.3, -0.2, 0.1), 0.5), 7763, 54, 2285.320629, 0);
    check("sphereB", &sphere(1.0, Vec3f::splat(0.0), 0.05), 30254, 158, 402.4881175, 0);
    let d = create_level_set_sphere::<f64>(2.0, Vec3f::new(0.11, 0.2, 0.3), 0.1, 4.0, true).unwrap();
    check("sphereD", &d, 40688, 195, 2099.969383, 0);
}

#[test]
fn csg() {
    let a = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let b = sphere(0.8, Vec3f::new(0.7, 0.1, 0.0), 0.05);
    check("unionCopy", &csg_union_copy(&a, &b), 35143, 186, 434.7259619, 0);
    check("interCopy", &csg_intersection_copy(&a, &b), 14485, 85, 296.8179278, 0);
    check("diffCopy", &csg_difference_copy(&a, &b), 32784, 163, 609.4984707, 0);
    let (mut x, mut y) = (a.clone(), b.clone());
    csg_union(&mut x, &mut y, true, false).unwrap();
    check("union", &x, 35143, 184, 434.7259619, 0);
    let (mut x, mut y) = (a.clone(), b.clone());
    csg_intersection(&mut x, &mut y, true, false).unwrap();
    check("inter", &x, 14485, 85, 296.8179278, 0);
    let (mut x, mut y) = (a.clone(), b.clone());
    csg_difference(&mut x, &mut y, true, false).unwrap();
    check("diff", &x, 32784, 162, 609.4984707, 0);
    let (mut x, mut y) = (a.clone(), b.clone());
    csg_union(&mut x, &mut y, false, false).unwrap();
    check("unionNoPrune", &x, 35143, 186, 434.7259619, 0);
    let c = sphere(1.0, Vec3f::new(500.0, 0.0, 0.0), 0.05);
    let (mut x, mut y) = (a.clone(), c.clone());
    csg_union(&mut x, &mut y, true, false).unwrap();
    check("unionFar", &x, 60508, 316, 804.9762351, 0);
    let (mut x, mut y) = (a.clone(), c.clone());
    csg_intersection(&mut x, &mut y, true, false).unwrap();
    check("interFar", &x, 0, 0, 0.0, 0);
    check("unionFarCopy", &csg_union_copy(&a, &c), 60508, 316, 804.9762351, 0);
}

#[test]
fn compositing() {
    let mut a = Grid::<f32>::new(1.0);
    let mut b = Grid::<f32>::new(2.0);
    a.tree_mut().fill(&CoordBBox::new(Coord::splat(0), Coord::splat(20)), 3.0, true);
    b.tree_mut().fill(&CoordBBox::new(Coord::splat(10), Coord::splat(40)), -1.0, true);
    a.tree_mut().set_value(&Coord::new(-5, 0, 0), 7.0);
    b.tree_mut().set_value_off(&Coord::new(-5, 0, 1), 9.0);
    let cases: [(&str, fn(&mut Grid<f32>, &mut Grid<f32>), f64); 5] = [
        ("compMax", comp_max, 56250.0),
        ("compMin", comp_min, -13929.0),
        ("compSum", comp_sum, 42321.0),
        ("compMul", comp_mul, 15141.0),
        ("compDiv", comp_div, -20554.5),
    ];
    for (name, op, sum) in cases {
        let (mut x, mut y) = (a.clone(), b.clone());
        op(&mut x, &mut y);
        check(name, &x, 37722, 112, sum, 33);
        assert!(y.empty());
        if name == "compMax" {
            assert_eq!(*x.background(), 2.0);
        }
    }
    let mut x = a.clone();
    comp_replace(&mut x, &b);
    check("compReplace", &x, 37722, 111, -5994.0, 34);
}

#[test]
fn morphology() {
    let s = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let nns = [NearestNeighbors::Face, NearestNeighbors::FaceEdge, NearestNeighbors::FaceEdgeVertex];
    let dilated = [(47450, 191), (57204, 221), (61432, 225)];
    let eroded = [(13366, 158), (4134, 158), (1278, 158)];
    for i in 0..3 {
        let mut x = s.clone();
        dilate_active_values(x.tree_mut(), 2, nns[i], TilePolicy::IgnoreTiles, true);
        assert_eq!((x.active_voxel_count(), x.tree().leaf_count()), dilated[i], "dilate {:?}", nns[i]);
        let mut y = s.clone();
        erode_active_values(y.tree_mut(), 2, nns[i], TilePolicy::IgnoreTiles, true);
        assert_eq!((y.active_voxel_count(), y.tree().leaf_count()), eroded[i], "erode {:?}", nns[i]);
    }

    let mut t = Grid::<f32>::new(0.0);
    t.tree_mut().fill(&CoordBBox::new(Coord::splat(0), Coord::splat(127)), 1.0, true);
    t.tree_mut().fill(&CoordBBox::new(Coord::new(130, 0, 0), Coord::new(140, 10, 10)), 2.0, true);
    t.tree_mut().set_value(&Coord::new(-3, -3, -3), 5.0);
    let policies = [TilePolicy::IgnoreTiles, TilePolicy::ExpandTiles, TilePolicy::PreserveTiles];
    // (active voxels, leaves, active tiles) for [nn][policy].
    let dil: [[(u64, usize, u64); 3]; 3] = [
        [(2101007, 22, 1), (2399990, 5841, 0), (2399990, 1745, 1)],
        [(2102058, 30, 1), (2409968, 5841, 0), (2409968, 1745, 1)],
        [(2102211, 30, 1), (2410140, 5841, 0), (2410140, 1745, 1)],
    ];
    let ero: [(u64, usize, u64); 3] = [(2097277, 9, 1), (1815973, 4105, 0), (1815973, 1354, 2744)];
    for i in 0..3 {
        for p in 0..3 {
            let mut x = t.clone();
            dilate_active_values(x.tree_mut(), 3, nns[i], policies[p], true);
            let r = (x.active_voxel_count(), x.tree().leaf_count(), count_active_tiles(x.tree()));
            assert_eq!(r, dil[i][p], "dilate {:?} {:?}", nns[i], policies[p]);
            let mut y = t.clone();
            erode_active_values(y.tree_mut(), 3, nns[i], policies[p], true);
            let r = (y.active_voxel_count(), y.tree().leaf_count(), count_active_tiles(y.tree()));
            assert_eq!(r, ero[p], "erode {:?} {:?}", nns[i], policies[p]);
        }
    }
}

#[test]
fn prune_median() {
    let mut g = Grid::<f32>::new(0.0);
    for x in 0..32 {
        for y in 0..16 {
            for z in 0..16 {
                g.tree_mut().set_value(&Coord::new(x, y, z), 1.0 + 0.001 * ((x * 7 + y * 3 + z * 11) % 13) as f32);
            }
        }
    }
    prune(g.tree_mut(), 0.02, true);
    assert_eq!(g.tree().leaf_count(), 0);
    assert_eq!(g.get_value(&Coord::new(3, 3, 3)), 1.006);
    assert_eq!(g.get_value(&Coord::new(20, 3, 3)), 1.006);
    assert_eq!(g.active_voxel_count(), 8192);
}

#[test]
fn level_set_util() {
    let s = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let m = sdf_interior_mask(&s, 0.0);
    assert_eq!((m.active_voxel_count(), m.tree().leaf_count()), (33371, 131));
    assert_eq!(extract_isosurface_mask(&s, 0.0).active_voxel_count(), 8328);
    let mut f = s.clone();
    sdf_to_fog_volume(&mut f, f32::MAX);
    check("fog", &f, 33371, 131, 26696.5297, 8);
    let mut f2 = s.clone();
    sdf_to_fog_volume(&mut f2, 0.1);
    check("fog2", &f2, 33371, 131, 37996.79466, 8);

    let mut big = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let mut small = sphere(0.5, Vec3f::splat(0.0), 0.05);
    csg_difference(&mut big, &mut small, true, false).unwrap();
    let enc = extract_enclosed_region(&big, 0.0, None);
    assert_eq!((enc.active_voxel_count(), enc.tree().leaf_count()), (33371, 139));
    assert_eq!(sdf_interior_mask(&big, 0.0).active_voxel_count(), 29202);

    let mut a = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let mut b = sphere(0.7, Vec3f::new(3.0, 0.0, 0.0), 0.05);
    csg_union(&mut a, &mut b, true, false).unwrap();
    let segs = segment_active_voxels(&a);
    let r: Vec<_> = segs.iter().map(|g| (g.active_voxel_count(), g.tree().leaf_count())).collect();
    assert_eq!(r, vec![(30254, 158), (14802, 84)]);
    let sdfs = segment_sdf(&a);
    let r: Vec<_> = sdfs.iter().map(|g| (g.active_voxel_count(), g.tree().leaf_count())).collect();
    assert_eq!(r, vec![(30254, 158), (14802, 84)]);
    assert!(close(sum_all(&sdfs[0]), 3083.888224, 1e-8));
    assert!(close(sum_all(&sdfs[1]), 2998.325917, 1e-8));
    let masks: Vec<_> = extract_active_voxel_segment_masks(&a).iter().map(|m| m.active_voxel_count()).collect();
    assert_eq!(masks, vec![30254, 14802]);
}

#[test]
fn measures() {
    let s = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let mut m = LevelSetMeasure::new(&s).unwrap();
    assert!(close(m.area(true), 12.566670284, 1e-9));
    assert!(close(m.volume(true), 4.19482098767, 1e-9));
    assert!(close(m.total_mean_curvature(true), 12.5571902482, 1e-9));
    assert!(close(m.total_gaussian_curvature(true), 12.5569212852, 1e-9));
    assert_eq!((m.euler_characteristic(), m.genus()), (2, 0));
}

#[test]
fn topology_to_level_set_reference() {
    let mut g = Grid::<bool>::new(false);
    g.tree_mut().fill(&CoordBBox::new(Coord::splat(0), Coord::splat(19)), true, true);
    check("ttls", &topology_to_level_set(&g, 3, 1, 0, 0).unwrap(), 9008, 63, 2321.307892, 0);

    let mut r = Grid::<bool>::new(false).with_transform(Transform::linear(0.1));
    for x in -20..=20 {
        for y in -20..=20 {
            for z in -4..=4 {
                let d = ((x * x + y * y) as f64).sqrt();
                if d > 8.0 && d < 18.0 {
                    r.tree_mut().set_value(&Coord::new(x, y, z), true);
                }
            }
        }
    }
    let lr = topology_to_level_set(&r, 3, 1, 0, 0).unwrap();
    check("ttlsRing", &lr, 12124, 64, 453.8882909, 0);
    let mut mr = LevelSetMeasure::new(&lr).unwrap();
    assert!(close(mr.area(true), 31.2506287251, 1e-8));
    assert!(close(mr.volume(true), 7.76222496033, 1e-8));
    assert_eq!((mr.euler_characteristic(), mr.genus()), (4, -1));
    check("ttlsRingSmooth", &topology_to_level_set(&r, 3, 1, 1, 2).unwrap(), 14132, 64, -143.3916733, 0);
}

#[test]
fn statistics_reference() {
    let s = sphere(1.0, Vec3f::splat(0.0), 0.05);
    let st = statistics(s.tree(), ValueIterKind::On, true);
    assert_eq!(st.size(), 30254);
    assert!(close(st.min(), -0.148530676961, 1e-9) && close(st.max(), 0.148912534118, 1e-9));
    assert!(close(st.mean(), 0.0133036331569, 1e-8) && close(st.std_dev(), 0.0854603456209, 1e-8));
    let sa = statistics(s.tree(), ValueIterKind::All, true);
    assert_eq!(sa.size(), 549755813888);
    assert!(close(sa.mean(), 0.149999987263, 1e-9) && close(sa.std_dev(), 6.91580564775e-05, 1e-6));
    let h = histogram(s.tree(), ValueIterKind::On, -0.15, 0.15, 6, true).unwrap();
    let counts: Vec<u64> = (0..6).map(|i| h.count(i)).collect();
    assert_eq!(counts, vec![3926, 4266, 4700, 5270, 5754, 6338]);
    let e = extrema(s.tree(), ValueIterKind::Off, true);
    assert_eq!(e.size(), 549755783634);
    assert_eq!((e.min(), e.max()), (-0.15000000596046448, 0.15000000596046448));
}

#[test]
fn backgrounds_and_diagnostics() {
    let mut s = sphere(1.0, Vec3f::splat(0.0), 0.05);
    change_level_set_background(s.tree_mut(), 0.5, true).unwrap();
    assert!(close(sum_all(&s), 9340.488118, 1e-9));
    assert_eq!(*s.background(), 0.5);
    let mut t = sphere(1.0, Vec3f::splat(0.0), 0.05);
    change_background(t.tree_mut(), 0.7, true);
    assert!(close(sum_all(&t), 12915.6879, 1e-9));
    assert_eq!(check_level_set(&t, 9), "");
}
