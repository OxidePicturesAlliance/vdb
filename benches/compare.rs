//! Rust side of the Rust ↔ C++ OpenVDB comparison benchmark.
//!
//! Every benchmark here has a twin with the same name, inputs and parameters in
//! `benches/cpp/compare.cpp`. Each prints one tab-separated line:
//!
//! ```text
//! name  median_ms  min_ms  check
//! ```
//!
//! where `check` is a result summary (voxel counts, sums...) that must agree with the C++
//! run, so that both sides are known to do the same work. Run both with
//! `benches/compare.sh`, or this side alone with:
//!
//! ```text
//! cargo bench --bench compare -- [filter] [--reps N] [--threads N]
//! ```

use std::hint::black_box;
use std::time::Instant;

use vdb::io::{Compression, WriteOptions};
use vdb::math::{Ray, Transform, Vec3d, Vec3f};
use vdb::points::codec::Codec;
use vdb::points::point_conversion::create_point_data_grid;
use vdb::prelude::*;
use vdb::tools::composite::{csg_difference, csg_union};
use vdb::tools::filter::Filter;
use vdb::tools::grid_operators::{gradient, laplacian, mean_curvature};
use vdb::tools::interpolation::{BoxSampler, GridSampler};
use vdb::tools::level_set_filter::LevelSetFilter;
use vdb::tools::level_set_measure::LevelSetMeasure;
use vdb::tools::level_set_sphere::{create_level_set, create_level_set_sphere};
use vdb::tools::mesh_to_volume::mesh_to_level_set;
use vdb::tools::morphology::{dilate_active_values, erode_active_values, NearestNeighbors, TilePolicy};
use vdb::tools::particles_to_level_set::{particles_to_sdf, SimpleParticleList};
use vdb::tools::point_index_grid::create_point_index_grid;
use vdb::tools::ray_intersector::LevelSetRayIntersector;
use vdb::tools::signed_flood_fill::signed_flood_fill;
use vdb::tools::volume_to_mesh::volume_to_mesh_adaptive;
use vdb::AnyGrid;

// Shared parameters (keep in sync with compare.cpp).
const VOXEL: f32 = 0.004; // sphere of radius 1 → 250 voxels
const HALF_WIDTH: f32 = 3.0;

/// SplitMix64, identical to the C++ side, so both use the same random inputs.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn range_f(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn range_i(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u64() % (hi - lo) as u64) as i32
    }
}

struct Ctx {
    filter: Option<String>,
    reps: usize,
}

impl Ctx {
    /// Time `run` over `reps` repetitions, each on a fresh state from `setup` (untimed);
    /// `check` summarizes the last state (untimed).
    fn bench<S, R>(
        &self,
        name: &str,
        setup: impl Fn() -> S,
        run: impl Fn(&mut S) -> R,
        check: impl Fn(&S, R) -> String,
    ) {
        if let Some(f) = &self.filter {
            if !name.contains(f.as_str()) {
                return;
            }
        }
        let mut times = Vec::with_capacity(self.reps);
        let mut last = None;
        for _ in 0..self.reps {
            let mut state = setup();
            let t = Instant::now();
            let r = black_box(run(&mut state));
            times.push(t.elapsed().as_secs_f64() * 1e3);
            last = Some((state, r));
        }
        let (state, r) = last.unwrap();
        let c = check(&state, r);
        times.sort_by(|a, b| a.total_cmp(b));
        println!("{name}\t{:.3}\t{:.3}\t{c}", times[times.len() / 2], times[0]);
    }
}

fn sphere(center: Vec3f) -> FloatGrid {
    create_level_set_sphere::<f32>(1.0, center, VOXEL, HALF_WIDTH, true).unwrap()
}

/// Torus quad mesh (same construction as compare.cpp).
fn torus(big_r: f32, r: f32, n_major: u32, n_minor: u32) -> (Vec<Vec3f>, Vec<[u32; 4]>) {
    let mut points = Vec::new();
    for i in 0..n_major {
        let u = i as f32 / n_major as f32 * std::f32::consts::TAU;
        for j in 0..n_minor {
            let v = j as f32 / n_minor as f32 * std::f32::consts::TAU;
            let w = big_r + r * v.cos();
            points.push(Vec3f::new(w * u.cos(), r * v.sin(), w * u.sin()));
        }
    }
    let mut quads = Vec::new();
    for i in 0..n_major {
        for j in 0..n_minor {
            let a = i * n_minor + j;
            let b = ((i + 1) % n_major) * n_minor + j;
            let c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor;
            let d = i * n_minor + (j + 1) % n_minor;
            quads.push([a, d, c, b]);
        }
    }
    (points, quads)
}

fn sum_active(grid: &FloatGrid) -> f64 {
    grid.tree().active_voxels().map(|(_, v)| v as f64).sum()
}

fn main() {
    let mut ctx = Ctx { filter: None, reps: 5 };
    let mut threads = 0usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reps" => ctx.reps = args.next().unwrap().parse().unwrap(),
            "--threads" => threads = args.next().unwrap().parse().unwrap(),
            "--bench" => {} // passed by `cargo bench`
            _ => ctx.filter = Some(a),
        }
    }
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }
    eprintln!("rust: {} threads, {} reps", rayon::current_num_threads(), ctx.reps);

    let base = sphere(Vec3f::zero());
    let n = |g: &FloatGrid| g.active_voxel_count().to_string();

    // ---- Tree construction and access ----
    ctx.bench("sphere_create", || (), |_| sphere(Vec3f::zero()), |_, g| n(&g));
    ctx.bench("grid_copy", || (), |_| base.clone(), |_, g| n(&g));
    ctx.bench("active_voxel_count", || (), |_| base.active_voxel_count(), |_, c| c.to_string());
    ctx.bench("iterate_active", || (), |_| sum_active(&base), |_, s| format!("{s:.6e}"));

    let mut rng = Rng(1);
    let insert_coords: Vec<Coord> = (0..2_000_000)
        .map(|_| Coord::new(rng.range_i(-256, 256), rng.range_i(-256, 256), rng.range_i(-256, 256)))
        .collect();
    ctx.bench(
        "random_insert",
        || FloatGrid::new(0.0),
        |g| {
            let mut acc = g.accessor_mut();
            for (i, c) in insert_coords.iter().enumerate() {
                acc.set_value(c, i as f32);
            }
        },
        |g, _| n(g),
    );
    let read_coords: Vec<Coord> = (0..4_000_000)
        .map(|_| Coord::new(rng.range_i(-260, 260), rng.range_i(-260, 260), rng.range_i(-260, 260)))
        .collect();
    ctx.bench(
        "random_read",
        || (),
        |_| {
            let acc = base.accessor();
            read_coords.iter().map(|c| acc.get_value(c) as f64).sum::<f64>()
        },
        |_, s| format!("{s:.6e}"),
    );
    ctx.bench(
        "coherent_read",
        || (),
        |_| {
            let acc = base.accessor();
            let mut s = 0.0f64;
            for i in -128..128 {
                for j in -128..128 {
                    for k in 0..256 {
                        s += acc.get_value(&Coord::new(i, j, k)) as f64;
                    }
                }
            }
            s
        },
        |_, s| format!("{s:.6e}"),
    );
    let sample_points: Vec<Vec3d> = (0..2_000_000)
        .map(|_| Vec3d::new(rng.range_f(-1.05, 1.05), rng.range_f(-1.05, 1.05), rng.range_f(-1.05, 1.05)))
        .collect();
    ctx.bench(
        "box_sample",
        || (),
        |_| {
            let s = GridSampler::<_, BoxSampler>::from_grid(&base);
            sample_points.iter().map(|p| s.ws_sample(*p) as f64).sum::<f64>()
        },
        |_, s| format!("{s:.6e}"),
    );

    // ---- I/O (in memory) ----
    let mut fog = base.clone();
    fog.set_grid_class(GridClass::Unknown); // so that zip is applied
    let fog_any: Vec<AnyGrid> = vec![fog.into()];
    for (label, comp) in [
        ("none", Compression::ACTIVE_MASK),
        ("zip", Compression::ZIP | Compression::ACTIVE_MASK),
        ("blosc", Compression::BLOSC | Compression::ACTIVE_MASK),
    ] {
        let opts = WriteOptions::with_compression(comp);
        let write = || {
            let mut buf = Vec::new();
            vdb::io::write_stream(&mut buf, &fog_any, &MetaMap::new(), &opts).unwrap();
            buf
        };
        ctx.bench(&format!("write_{label}"), || (), |_| write(), |_, b| format!("{} bytes", b.len()));
        let bytes = write();
        ctx.bench(
            &format!("read_{label}"),
            || (),
            |_| vdb::io::read_stream(&bytes[..]).unwrap().1,
            |_, g| {
                let g = g[0].as_float().unwrap();
                format!("{} {:.6e}", g.active_voxel_count(), sum_active(g))
            },
        );
    }

    // ---- Level set tools ----
    let other = sphere(Vec3f::new(0.5, 0.0, 0.0));
    ctx.bench(
        "csg_union",
        || (base.clone(), other.clone()),
        |(a, b)| csg_union(a, b, true, true).unwrap(),
        |(a, _), _| n(a),
    );
    ctx.bench(
        "csg_difference",
        || (base.clone(), other.clone()),
        |(a, b)| csg_difference(a, b, true, true).unwrap(),
        |(a, _), _| n(a),
    );
    ctx.bench(
        "dilate_3",
        || base.clone(),
        |g| dilate_active_values(g.tree_mut(), 3, NearestNeighbors::Face, TilePolicy::IgnoreTiles, true),
        |g, _| n(g),
    );
    ctx.bench(
        "erode_3",
        || base.clone(),
        |g| erode_active_values(g.tree_mut(), 3, NearestNeighbors::Face, TilePolicy::IgnoreTiles, true),
        |g, _| n(g),
    );
    ctx.bench("signed_flood_fill", || base.clone(), |g| signed_flood_fill(g.tree_mut(), true).unwrap(), |g, _| n(g));
    ctx.bench(
        "filter_gaussian",
        || base.clone(),
        |g| Filter::new(g).gaussian(1, 1),
        |g, _| format!("{:.6e}", sum_active(g)),
    );
    ctx.bench(
        "filter_median",
        || base.clone(),
        |g| Filter::new(g).median(1, 1),
        |g, _| format!("{:.6e}", sum_active(g)),
    );
    ctx.bench(
        "ls_filter_mean_curvature",
        || base.clone(),
        |g| LevelSetFilter::new(g).unwrap().mean_curvature(),
        |g, _| n(g),
    );
    ctx.bench("gradient", || (), |_| gradient(&base), |_, g| g.active_voxel_count().to_string());
    ctx.bench("laplacian", || (), |_| laplacian(&base), |_, g| g.active_voxel_count().to_string());
    ctx.bench("mean_curvature", || (), |_| mean_curvature(&base), |_, g| g.active_voxel_count().to_string());
    ctx.bench(
        "ls_measure",
        || (),
        |_| {
            let mut m = LevelSetMeasure::new(&base).unwrap();
            (m.area(true), m.volume(true))
        },
        |_, (a, v)| format!("{a:.5} {v:.5}"),
    );
    for (label, adaptivity) in [("volume_to_mesh", 0.0), ("volume_to_mesh_adaptive", 0.5)] {
        ctx.bench(
            label,
            || (),
            |_| volume_to_mesh_adaptive(&base, 0.0, adaptivity, true),
            |_, (p, t, q)| format!("{} pts {} tris {} quads", p.len(), t.len(), q.len()),
        );
    }
    let (tpoints, tquads) = torus(1.0, 0.35, 512, 256);
    ctx.bench(
        "mesh_to_level_set",
        || (),
        |_| mesh_to_level_set::<f32>(&Transform::linear(VOXEL as f64), &tpoints, &[], &tquads, HALF_WIDTH),
        |_, g| n(&g),
    );
    let rays: Vec<Ray> = (0..500_000)
        .map(|_| {
            let eye = Vec3d::new(rng.range_f(-1.0, 1.0), rng.range_f(-1.0, 1.0), 3.0);
            let target = Vec3d::new(rng.range_f(-0.5, 0.5), rng.range_f(-0.5, 0.5), 0.0);
            Ray::new(eye, (target - eye).unit())
        })
        .collect();
    ctx.bench(
        "ray_intersect",
        || (),
        |_| {
            let mut lsri = LevelSetRayIntersector::new(&base).unwrap();
            let (mut hits, mut t) = (0usize, 0.0f64);
            for r in &rays {
                if let Some(h) = lsri.intersects_ws(r) {
                    hits += 1;
                    t += h.time;
                }
            }
            (hits, t)
        },
        |_, (h, t)| format!("{h} hits {t:.4e}"),
    );

    // ---- Particles and points ----
    let mut particles = SimpleParticleList::default();
    for _ in 0..50_000 {
        particles.positions.push(Vec3d::new(rng.range_f(-1.0, 1.0), rng.range_f(-1.0, 1.0), rng.range_f(-1.0, 1.0)));
        particles.radii.push(0.01);
    }
    ctx.bench(
        "particles_to_sdf",
        || create_level_set::<f32>(VOXEL as f64, HALF_WIDTH as f64),
        |g| particles_to_sdf(&particles, g),
        |g, _| n(g),
    );
    let positions: Vec<Vec3f> = (0..2_000_000)
        .map(|_| {
            Vec3f::new(rng.range_f(-1.0, 1.0) as f32, rng.range_f(-1.0, 1.0) as f32, rng.range_f(-1.0, 1.0) as f32)
        })
        .collect();
    let pxform = Transform::linear(0.02);
    ctx.bench(
        "point_index_grid",
        || (),
        |_| create_point_index_grid(&positions[..], &pxform),
        |_, g| format!("{} leaves", g.tree().leaf_count()),
    );
    ctx.bench(
        "point_data_grid",
        || (),
        |_| create_point_data_grid(&positions[..], Codec::Null, &pxform).unwrap(),
        |_, g| format!("{} leaves", g.tree().leaf_count()),
    );
}
