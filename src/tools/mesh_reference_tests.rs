//! Regression tests against reference numbers produced with the C++ OpenVDB library
//! (OpenVDB 10.0.1 as installed, plus the OpenVDB 13 headers of `MeshToVolume.h`,
//! `LevelSetTubes.h` and `LevelSetDilatedMesh.h` compiled against it). The reference
//! programs build exactly the same inputs; the full voxel/point dumps were compared during
//! development and matched (see the crate report). Here the summary statistics are checked.

use super::level_set_dilated_mesh::*;
use super::level_set_tubes::*;
use super::mesh_support::signed_flood_fill_with_values;
use super::mesh_to_volume::*;
use super::particles_to_level_set::*;
use super::point_index_grid::*;
use super::point_scatter::*;
use super::points_to_mask::create_point_mask;
use super::volume_to_mesh::*;
use super::volume_to_spheres::*;
use crate::grid::{Grid, GridClass};
use crate::math::{Coord, CoordBBox, Transform, Vec3d, Vec3f};

/// Replica of `tools::createLevelSetSphere<FloatGrid>` (OpenVDB's `LevelSetSphere`).
pub(crate) fn level_set_sphere(radius: f32, center: Vec3f, dx: f32, w: f32) -> Grid<f32> {
    let mut g = Grid::<f32>::new(dx * w).with_transform(Transform::linear(dx as f64));
    let r0 = radius / dx;
    let rmax = r0 + w;
    let c = Vec3f::new(center[0] / dx, center[1] / dx, center[2] / dx);
    let (imin, imax) = ((c[0] - rmax).floor() as i32, (c[0] + rmax).ceil() as i32);
    let (jmin, jmax) = ((c[1] - rmax).floor() as i32, (c[1] + rmax).ceil() as i32);
    let (kmin, kmax) = ((c[2] - rmax).floor() as i32, (c[2] + rmax).ceil() as i32);
    {
        let tree = g.tree_mut();
        let mut acc = tree.accessor_mut();
        for i in imin..imax {
            let x2 = (i as f32 - c[0]) * (i as f32 - c[0]);
            for j in jmin..=jmax {
                let x2y2 = (j as f32 - c[1]) * (j as f32 - c[1]) + x2;
                let mut k = kmin;
                while k <= kmax {
                    let mut m = 1;
                    let v = (x2y2 + (k as f32 - c[2]) * (k as f32 - c[2])).sqrt() - r0;
                    let d = v.abs();
                    if d < w {
                        acc.set_value(&Coord::new(i, j, k), dx * v);
                    } else {
                        m += (d - w).floor() as i32;
                    }
                    k += m;
                }
            }
        }
    }
    let bg = dx * w;
    signed_flood_fill_with_values(g.tree_mut(), bg, -bg);
    g.set_grid_class(GridClass::LevelSet);
    g
}

fn sphere_mesh(rings: i32, segs: i32, r: f64, c: Vec3d) -> (Vec<Vec3f>, Vec<[u32; 3]>, Vec<[u32; 4]>) {
    use std::f64::consts::PI;
    let mut pts = vec![Vec3f::new(c[0] as f32, (c[1] + r) as f32, c[2] as f32)];
    for i in 1..rings {
        let th = PI * i as f64 / rings as f64;
        for j in 0..segs {
            let ph = 2.0 * PI * j as f64 / segs as f64;
            pts.push(Vec3f::new(
                (c[0] + r * th.sin() * ph.cos()) as f32,
                (c[1] + r * th.cos()) as f32,
                (c[2] + r * th.sin() * ph.sin()) as f32,
            ));
        }
    }
    pts.push(Vec3f::new(c[0] as f32, (c[1] - r) as f32, c[2] as f32));
    let south = (pts.len() - 1) as u32;
    let segs = segs as u32;
    let mut tris = Vec::new();
    let mut quads = Vec::new();
    for j in 0..segs {
        tris.push([0, 1 + (j + 1) % segs, 1 + j]);
    }
    for i in 0..(rings as u32 - 2) {
        for j in 0..segs {
            let a = 1 + i * segs + j;
            let b = 1 + i * segs + (j + 1) % segs;
            quads.push([a, b, b + segs, a + segs]);
        }
    }
    let base = 1 + (rings as u32 - 2) * segs;
    for j in 0..segs {
        tris.push([south, base + j, base + (j + 1) % segs]);
    }
    (pts, tris, quads)
}

fn stats(g: &Grid<f32>) -> (u64, usize, f64) {
    let sum: f64 = g.tree().active_voxels().map(|(_, v)| v as f64).sum();
    (g.active_voxel_count(), g.tree().leaf_count(), sum)
}

fn assert_stats(g: &Grid<f32>, active: u64, leaves: usize, sum: f64) {
    let (a, l, s) = stats(g);
    assert_eq!(a, active);
    assert_eq!(l, leaves);
    assert!((s - sum).abs() < 1e-5 * sum.abs().max(1.0), "sum {s} != {sum}");
}

#[test]
fn mesh_to_volume_matches_cpp() {
    let cube: Vec<Vec3f> = [
        [-5., -5., -5.],
        [-5., -5., 5.],
        [5., -5., 5.],
        [5., -5., -5.],
        [-5., 5., -5.],
        [-5., 5., 5.],
        [5., 5., 5.],
        [5., 5., -5.],
    ]
    .iter()
    .map(|p| Vec3f::new(p[0], p[1], p[2]))
    .collect();
    let q = [[0, 1, 2, 3], [7, 6, 5, 4], [4, 5, 1, 0], [6, 7, 3, 2], [0, 3, 7, 4], [1, 5, 6, 2]];
    let xf = Transform::linear(1.0);
    let g: Grid<f32> = mesh_to_level_set(&xf, &cube, &[], &q, 3.0);
    assert_stats(&g, 3086, 8, 2205.64213);
    let g: Grid<f32> = mesh_to_signed_distance_field(&xf, &cube, &[], &q, 3.0, f32::MAX);
    assert_stats(&g, 3211, 8, 1802.64212);
    // With an interior test instead of the exterior boundary tracing.
    let mesh = QuadAndTriangleDataAdapter::new(&cube, &q);
    let inside = |c: &Coord| c[0].abs() < 5 && c[1].abs() < 5 && c[2].abs() < 5;
    for strategy in [InteriorTestStrategy::EvalEveryVoxel, InteriorTestStrategy::EvalEveryTile] {
        let g: Grid<f32> = mesh_to_volume_with_interior_test(&mesh, &xf, 3.0, 3.0, 0, None, inside, strategy);
        assert_stats(&g, 3086, 8, 2205.64213);
    }

    let tet = vec![
        Vec3f::new(0.3, 0.1, 0.2),
        Vec3f::new(7.7, 0.4, -0.3),
        Vec3f::new(1.1, 6.9, 0.5),
        Vec3f::new(0.6, 1.2, 8.1),
    ];
    let t = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let xf = Transform::linear(0.5);
    assert_stats(&mesh_to_level_set(&xf, &tet, &t, &[], 3.0), 3018, 32, 1924.69634);
    assert_stats(&mesh_to_signed_distance_field(&xf, &tet, &t, &[], 2.0, 5.0), 1945, 22, 574.311314);
    assert_stats(&mesh_to_unsigned_distance_field(&xf, &tet, &t, &[], 1.5), 1420, 15, 526.413823);
    assert_stats(&mesh_to_level_set(&xf, &tet, &t, &[], 1.0), 965, 12, 79.6384786);

    let (pts, t, q) = sphere_mesh(12, 16, 6.3, Vec3d::new(0.3, -0.2, 0.15));
    let xf = Transform::linear(0.25);
    assert_stats(&mesh_to_level_set(&xf, &pts, &t, &q, 3.0), 47084, 264, 2851.79639);
    // Open mesh: the sign heuristics classify everything as outside, as in OpenVDB.
    assert_stats(&mesh_to_level_set(&xf, &pts, &t, &q[4..], 3.0), 46954, 264, 17655.0894);

    let xf = Transform::linear(0.5);
    let g: Grid<f32> = create_level_set_box(Vec3d::new(-1.3, -2.1, 0.4), Vec3d::new(3.2, 1.7, 5.9), &xf, 3.0);
    assert_stats(&g, 3173, 18, 1740.5999);
}

fn mesh_counts(g: &Grid<f32>, iso: f64, adapt: f64, relax: bool) -> (usize, usize, usize) {
    let (p, t, q) = volume_to_mesh_adaptive(g, iso, adapt, relax);
    (p.len(), t.len(), q.len())
}

#[test]
fn volume_to_mesh_matches_cpp() {
    let s1 = level_set_sphere(5.3, Vec3f::new(0.31, -0.27, 0.13), 0.25, 3.0);
    assert_eq!(s1.active_voxel_count(), 34102);
    assert_eq!(mesh_counts(&s1, 0.0, 0.0, true), (8484, 0, 8482));
    assert_eq!(mesh_counts(&s1, 0.1, 0.0, true), (8796, 0, 8794));
    assert_eq!(mesh_counts(&s1, 0.0, 0.3, true), (163, 28, 147));
    assert_eq!(mesh_counts(&s1, 0.0, 0.7, false), (163, 28, 147));
    let s2 = level_set_sphere(2.0, Vec3f::new(0.0, 0.0, 0.0), 0.1, 3.0);
    assert_eq!(s2.active_voxel_count(), 30254);
    assert_eq!(mesh_counts(&s2, 0.0, 0.0, true), (7472, 0, 7470));
    assert_eq!(mesh_counts(&s2, 0.0, 0.5, true), (128, 0, 126));

    // A fog-like grid (grid class unknown) with active tiles.
    let mut fog = Grid::<f32>::new(0.0);
    fog.tree_mut().fill(&CoordBBox::new(Coord::new(-20, -10, -5), Coord::new(27, 40, 33)), 1.0, true);
    fog.tree_mut().set_value(&Coord::new(3, 4, 5), 0.0);
    assert_eq!(fog.tree().active_tile_count(), 119);
    assert_eq!(mesh_counts(&fog, 0.5, 0.0, true), (12628, 0, 12624));
    assert_eq!(mesh_counts(&fog, 0.5, 0.5, true), (867, 434, 646));

    // The first points and quad of the uniform mesh.
    let (p, _, q) = volume_to_mesh_adaptive(&s1, 0.0, 0.0, true);
    assert!((p[0] - Vec3f::new(-4.00015736, -2.50029349, -2.00030637)).length() < 1e-6, "{:?}", p[0]);
    assert!((p[1] - Vec3f::new(-4.03020573, -2.375, -2.05883694)).length() < 1e-6, "{:?}", p[1]);
    assert_eq!(q[0], [11, 5, 4, 10]);

    // Reference (fracture seam) meshing.
    let mut frag = s1.clone();
    frag.tree_mut().leaves_mut().for_each(|l| {
        for n in 0..512 {
            let c = l.offset_to_global_coord(n);
            let v = l.buffer()[n];
            l.buffer_mut()[n] = v.max(0.3f32 - c[0] as f32 * 0.25f32);
        }
    });
    for (adapt, expect) in [(0.0, (4816, 660, 4412, 4203, 670, 660, 170)), (0.3, (1107, 840, 623, 618, 662, 640, 170))]
    {
        let mut mesher = VolumeToMesh::new(0.0, adapt, true);
        mesher.set_ref_grid(&s1, 0.5);
        mesher.run(&frag);
        let (mut nt, mut nq, mut fe, mut fs, mut fsub) = (0, 0, 0, 0, 0);
        for pool in mesher.polygon_pool_list() {
            nq += pool.quads.len();
            nt += pool.triangles.len();
            for &f in pool.quad_flags.iter().chain(pool.triangle_flags.iter()) {
                fe += (f & POLYFLAG_EXTERIOR != 0) as usize;
                fs += (f & POLYFLAG_FRACTURE_SEAM != 0) as usize;
                fsub += (f & POLYFLAG_SUBDIVIDED != 0) as usize;
            }
        }
        let pf: usize = mesher.point_flags().iter().map(|&f| f as usize).sum();
        assert_eq!((mesher.point_list_size(), nt, nq, fe, fs, fsub, pf), expect);
    }

    // Surface mask.
    let mut mask = Grid::<bool>::new(false).with_transform(s1.transform().clone());
    mask.tree_mut().fill(&CoordBBox::new(Coord::new(-40, -40, -40), Coord::new(3, 40, 40)), true, true);
    for (inv, expect) in [(false, (521, 162, 355)), (true, (263, 146, 105))] {
        let mut mesher = VolumeToMesh::new(0.0, 0.4, true);
        mesher.set_surface_mask(&mask, inv);
        mesher.run(&s1);
        let nq: usize = mesher.polygon_pool_list().iter().map(|p| p.quads.len()).sum();
        let nt: usize = mesher.polygon_pool_list().iter().map(|p| p.triangles.len()).sum();
        assert_eq!((mesher.point_list_size(), nt, nq), expect);
    }
}

fn scatter_grid() -> Grid<f32> {
    let mut grid = Grid::<f32>::new(0.0).with_transform(Transform::linear(0.5));
    grid.tree_mut().fill(&CoordBBox::new(Coord::new(-3, -20, 0), Coord::new(40, 17, 21)), 1.0, true);
    for i in 0..300 {
        let c = Coord::new(60 + (i * 7) % 23, (i * 13) % 31 - 15, (i * 5) % 17);
        grid.tree_mut().set_value(&c, 0.25 + (i % 11) as f32 * 0.3);
    }
    grid
}

#[test]
fn point_scatter_and_index_grid_match_cpp() {
    let grid = scatter_grid();
    assert_eq!(grid.active_voxel_count(), 37084);
    let close = |p: Vec3d, e: [f64; 3]| (p - Vec3d::new(e[0], e[1], e[2])).length() < 1e-12;

    let mut pl: Vec<Vec3d> = Vec::new();
    UniformPointScatter::with_count(&mut pl, 7777, &Mt19937::default(), 0.8).run(&grid);
    assert_eq!(pl.len(), 7777);
    assert!(close(pl[0], [-1.3124528915503073, -9.8659965640021685, -0.1458091982812878]));
    assert!(close(pl[1], [-1.4811117614528593, -9.5767331797971984, 1.8884136171930819]));

    let mut d: Vec<Vec3d> = Vec::new();
    DenseUniformPointScatter::new(&mut d, 0.3, &Mt19937::new(42), 1.0).run(&grid);
    assert_eq!(d.len(), 10984);
    assert!(close(d[0], [-1.5270836212257832, -9.9515749192099712, 0.63984549880633068]));

    let mut n: Vec<Vec3d> = Vec::new();
    NonUniformPointScatter::new(&mut n, 2.5, &Mt19937::new(7), 0.5).run(&grid);
    assert_eq!(n.len(), 11694);
    assert!(close(n[1], [-1.5201569473096139, -10.103314141189847, 0.44096771019618586]));

    let mut u: Vec<Vec3d> = Vec::new();
    UniformPointScatter::with_density(&mut u, 0.37, &Mt19937::new(12345), 1.0).run(&grid);
    assert_eq!(u.len(), 1715);

    let xf = Transform::linear(0.7);
    let pig = create_point_index_grid(&pl, &xf);
    assert_eq!((pig.tree().leaf_count(), pig.tree().active_voxel_count()), (62, 5551));
    assert!(is_valid_partition(&pl, &pig));
    let it = PointIndexIterator::from_bbox(&CoordBBox::new(Coord::new(-2, -5, 0), Coord::new(9, 3, 11)), pig.tree());
    assert_eq!(it.size(), 761);
    assert_eq!(&it.to_vec()[..8], &[162, 164, 163, 198, 199, 201, 211, 212]);
    let mut it = PointIndexIterator::new();
    it.world_space_search_and_update_radial(Vec3d::new(3.1, -1.2, 4.4), 4.3, pig.tree(), &pl, &xf, true);
    assert_eq!(it.size(), 566);
    assert_eq!(&it.to_vec()[..8], &[959, 1083, 1154, 1157, 985, 987, 1009, 1055]);
    it.world_space_search_and_update_radial(Vec3d::new(3.1, -1.2, 4.4), 4.3, pig.tree(), &pl, &xf, false);
    assert_eq!(it.size(), 836);
    it.world_space_search_and_update_bbox(Vec3d::new(-1.3, -4.0, 0.2), Vec3d::new(8.8, 5.5, 9.1), pig.tree(), &pl, &xf);
    assert_eq!(it.size(), 1469);
    assert_eq!(&it.to_vec()[..4], &[182, 186, 183, 184]);
    it.search_and_update_coord(&Coord::new(3, 2, 5), pig.tree());
    assert_eq!(it.to_vec(), vec![4304]);

    let mask = create_point_mask(&pl, &xf);
    assert_eq!((mask.active_voxel_count(), mask.tree().leaf_count()), (5551, 62));
}

fn particles() -> SimpleParticleList {
    let mut pl = SimpleParticleList::default();
    let mut s: u32 = 12345;
    let mut next = || {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 8) & 0xFFFF) as f64 / 65535.0
    };
    for _ in 0..300 {
        let (x, y, z) = (next() * 6.0 - 3.0, next() * 6.0 - 3.0, next() * 6.0 - 3.0);
        let r = 0.05 + next() * 0.45;
        let (vx, vy, vz) = (next() - 0.5, next() - 0.5, next() - 0.5);
        pl.positions.push(Vec3d::new(x, y, z));
        pl.radii.push(r);
        pl.velocities.push(Vec3d::new(vx, vy, vz));
    }
    pl
}

#[test]
fn particles_to_level_set_matches_cpp() {
    let pl = particles();
    let ls = |hw: f64| {
        let mut g = Grid::<f32>::new((0.05 * hw) as f32).with_transform(Transform::linear(0.05));
        g.set_grid_class(GridClass::LevelSet);
        g
    };
    let mut g = ls(2.0);
    let mut p = ParticlesToLevelSet::new(&mut g);
    p.rasterize_spheres_with_radius(&pl, 0.13);
    p.finalize(false);
    assert_stats(&g, 117900, 1570, 4940.0552);

    let mut g = ls(3.0);
    let mut p = ParticlesToLevelSet::<f32, i32>::with_attributes(&mut g);
    p.rasterize_spheres_att(&pl);
    assert_eq!(p.min_count(), 10);
    p.finalize(false);
    let att = p.take_attribute_grid().unwrap();
    assert_stats(&g, 709075, 3419, 22939.16);
    let s: f64 = att.tree().active_voxels().map(|(_, v)| v as f64).sum();
    assert_eq!((att.active_voxel_count(), s), (709075, 100740472.0));

    let mut m = Grid::<bool>::new(false).with_transform(Transform::linear(0.05));
    let mut p = ParticlesToLevelSet::new(&mut m);
    p.rasterize_spheres(&pl);
    p.finalize(true);
    assert_eq!((m.active_voxel_count(), m.tree().leaf_count(), m.tree().active_tile_count()), (317581, 2238, 31));
}

#[test]
fn volume_to_spheres_matches_cpp() {
    let s1 = level_set_sphere(3.0, Vec3f::new(0.2, -0.1, 0.3), 0.1, 3.0);
    let spheres = fill_with_spheres(&s1, (1, 8), false, 1.0, f32::MAX, 0.0, 1000);
    assert_eq!(spheres.len(), 8);
    let expect =
        [[-0.0378021598, -0.212479979, 0.308800459, 2.73589301], [2.37726116, 0.244719058, 2.01328802, 0.209501952]];
    for (s, e) in [spheres[0], spheres[7]].iter().zip(expect.iter()) {
        for k in 0..4 {
            assert!((s[k] as f64 - e[k]).abs() < 1e-6, "{:?} vs {:?}", s, e);
        }
    }
    // Small grids are resampled with levelSetRebuild first.
    let s2 = level_set_sphere(1.0, Vec3f::new(0.0, 0.0, 0.0), 0.1, 3.0);
    let spheres = fill_with_spheres(&s2, (1, 6), false, 1.0, f32::MAX, 0.0, 500);
    assert_eq!(spheres.len(), 6);
    assert!((spheres[0][3] - 0.842390895).abs() < 1e-6);
    assert!((spheres[5][0] - -0.220319211).abs() < 1e-6);

    let csp = ClosestSurfacePoint::create(&s1, 0.0).unwrap();
    let mut pts = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(5.0, 1.0, -2.0)];
    let d = csp.search_and_replace(&mut pts);
    assert!((d[0] - 2.62485695).abs() < 1e-6 && (d[1] - 2.43691325).abs() < 1e-6);
    assert!((pts[0] - Vec3d::new(-1.44244361, 0.744722784, -2.06267262)).length() < 1e-6);
}

#[test]
fn tubes_and_dilated_mesh_match_cpp() {
    let g: Grid<f32> =
        create_level_set_capsule(&Vec3f::new(-1.2, 0.3, 0.1), &Vec3f::new(1.7, -0.4, 0.9), 0.8, 0.1, 3.0);
    assert_stats(&g, 14290, 82, 687.138738);
    let v = vec![
        Vec3f::new(0.0, 0.0, 0.0),
        Vec3f::new(2.0, 0.5, 0.0),
        Vec3f::new(2.5, 2.0, 0.5),
        Vec3f::new(0.5, 2.5, -0.5),
        Vec3f::new(-1.5, 1.0, 1.0),
    ];
    let s = vec![[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 2]];
    let g: Grid<f32> = create_level_set_tube_complex(&v, &s, 0.3, 0.05, 3.0);
    assert_stats(&g, 59828, 300, 1348.40449);
    let g: Grid<f32> = create_level_set_tube_complex_with_radii(
        &v,
        &s,
        &[0.2f32, 0.3, 0.25, 0.35, 0.15, 0.4],
        0.05,
        3.0,
        TubeRadiiPolicy::SegmentRadii,
    );
    assert_stats(&g, 63274, 314, 1411.99967);
    // Tapered capsules: OpenVDB's scan conversion misses one band voxel here.
    let g: Grid<f32> =
        create_level_set_tapered_capsule(&Vec3f::new(0.5, 0.3, 0.1), &Vec3f::new(-0.7, 1.4, -0.9), 0.2, 1.1, 0.1, 3.0);
    assert_stats(&g, 10655, 63, 544.532913);

    let tv = vec![
        Vec3f::new(0.3, 0.1, 0.2),
        Vec3f::new(2.7, 0.4, -0.3),
        Vec3f::new(1.1, 2.9, 0.5),
        Vec3f::new(0.6, 1.2, 3.1),
    ];
    let tt = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let g: Grid<f32> = create_level_set_dilated_mesh(&tv, &tt, 0.25, 0.05, 3.0);
    let (a, l, sum) = stats(&g);
    assert_eq!((a, l), (74825, 352));
    assert!((sum - 177.911324).abs() < 2e-3);
}
