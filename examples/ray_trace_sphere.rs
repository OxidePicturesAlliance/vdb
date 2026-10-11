//! Ray trace a narrow-band level set sphere and save the image as `sphere.ppm`.
//!
//! ```text
//! cargo run --release --example ray_trace_sphere [output.ppm] [width] [height]
//! ```

use vdb::prelude::*;
use vdb::tools::ray_tracer::{ray_trace, DiffuseShader, Film, NormalShader, PerspectiveCamera, Rgba, ShaderColor};

/// Build a narrow-band signed distance field of a sphere voxel by voxel.
fn sphere(radius: f64, center: Vec3d, voxel_size: f64, half_width: f64) -> FloatGrid {
    let bg = half_width * voxel_size;
    let mut grid = FloatGrid::new(bg as f32).with_transform(Transform::linear(voxel_size));
    grid.set_grid_class(GridClass::LevelSet);
    let ext = radius + bg;
    let lo = Coord::floor((center - Vec3d::splat(ext)) / voxel_size).offset(-1);
    let hi = Coord::ceil((center + Vec3d::splat(ext)) / voxel_size).offset(1);
    let xform = grid.transform().clone();
    {
        let mut acc = grid.accessor_mut();
        for ijk in CoordBBox::new(lo, hi) {
            let d = (xform.index_to_world(ijk.as_vec3d()) - center).length() - radius;
            if d.abs() < bg {
                acc.set_value_on(&ijk, d as f32);
            } else {
                acc.set_value_off(&ijk, if d < 0.0 { -bg as f32 } else { bg as f32 });
            }
        }
    }
    grid
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).cloned().unwrap_or_else(|| "sphere.ppm".to_string());
    let width: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(640);
    let height: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(480);

    let grid = sphere(1.0, Vec3d::zero(), 0.01, 3.0);
    println!("sphere: {} active voxels", grid.active_voxel_count());

    let mut film = Film::new(width, height);
    film.checkerboard(Rgba::gray(0.3), Rgba::gray(0.6), 32);
    let mut camera = PerspectiveCamera::new(&film, Vec3d::zero(), Vec3d::new(2.5, 1.5, 4.0), 50.0, 41.2136);
    camera.look_at(Vec3d::zero(), Vec3d::new(0.0, 1.0, 0.0));

    let start = std::time::Instant::now();
    if out.contains("normal") {
        ray_trace(&grid, &NormalShader::new(ShaderColor::default()), &camera, &mut film, 1, 0)?;
    } else {
        let shader = DiffuseShader::new(ShaderColor::Constant(Rgba::new(1.0, 0.8, 0.5)));
        ray_trace(&grid, &shader, &camera, &mut film, 4, 0)?;
    }
    println!("rendered {width}x{height} in {:.2?}", start.elapsed());
    film.save_ppm(&out)?;
    println!("wrote {out}");
    Ok(())
}
