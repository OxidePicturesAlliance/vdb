//! Mesh → level set → mesh round trip.
//!
//! Builds a torus as a quad mesh, converts it to a narrow-band level set with
//! `mesh_to_level_set`, extracts the zero isosurface again with `volume_to_mesh_adaptive`
//! and writes the result as a Wavefront OBJ file.
//!
//! ```text
//! cargo run --release --example mesh_roundtrip -- [output.obj] [voxel_size] [adaptivity]
//! ```

use std::io::Write;
use vdb::math::{Transform, Vec3f};
use vdb::tools::mesh_to_volume::mesh_to_level_set;
use vdb::tools::volume_to_mesh::volume_to_mesh_adaptive;
use vdb::Grid;

/// A torus with major radius `big_r` and minor radius `r` as a quad mesh.
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

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).cloned().unwrap_or_else(|| "mesh_roundtrip.obj".to_string());
    let voxel_size: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.05);
    let adaptivity: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let (points, quads) = torus(1.0, 0.35, 48, 24);
    println!("input mesh: {} points, {} quads", points.len(), quads.len());

    let xform = Transform::linear(voxel_size);
    let grid: Grid<f32> = mesh_to_level_set(&xform, &points, &[], &quads, 3.0);
    println!(
        "level set: {} active voxels in {} leaf nodes (voxel size {voxel_size})",
        grid.active_voxel_count(),
        grid.tree().leaf_count()
    );

    let (out_points, triangles, out_quads) = volume_to_mesh_adaptive(&grid, 0.0, adaptivity, true);
    println!(
        "output mesh: {} points, {} triangles, {} quads (adaptivity {adaptivity})",
        out_points.len(),
        triangles.len(),
        out_quads.len()
    );

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
    writeln!(f, "# vdb mesh_roundtrip example")?;
    for p in &out_points {
        writeln!(f, "v {} {} {}", p[0], p[1], p[2])?;
    }
    for t in &triangles {
        writeln!(f, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1)?;
    }
    for q in &out_quads {
        writeln!(f, "f {} {} {} {}", q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1)?;
    }
    println!("wrote {out}");
    Ok(())
}
