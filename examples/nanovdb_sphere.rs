//! Build a NanoVDB level-set sphere, save it as a `.nvdb` file, read it back and ray trace
//! it into an ASCII image with hierarchical DDA and trilinear normals.
//!
//! Run with `cargo run --release --example nanovdb_sphere [out.nvdb]`.

use vdb::math::Vec3d;
use vdb::nanovdb::hdda::{zero_crossing, Ray};
use vdb::nanovdb::io::{self, Codec};
use vdb::nanovdb::primitives::create_level_set_sphere;
use vdb::nanovdb::sampler::TrilinearSampler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| std::env::temp_dir().join("nanovdb_sphere.nvdb").to_string_lossy().into_owned());
    let handle = create_level_set_sphere::<f32>(1.0, Vec3d::new(0.0, 0.0, 0.0), 0.02, 3.0, Vec3d::splat(0.0), "sphere");
    io::write_grid(&path, &handle, Codec::Blosc)?;
    for m in io::read_grid_meta_data(&path)? {
        println!(
            "{}: {} {}, {} voxels, {} bytes in memory, {} on disk",
            m.grid_name,
            m.grid_type.as_str(),
            m.grid_class.as_str(),
            m.voxel_count,
            m.grid_size,
            m.file_size
        );
    }
    let handle = io::read_grid(&path, 0)?;
    let grid = handle.grid::<f32>(0).ok_or("not a float grid")?;
    let acc = grid.accessor();
    let sampler = TrilinearSampler::new(&acc, true);
    let map = grid.map();
    let light = Vec3d::new(-0.6, 0.6, 0.5).unit();
    let shades = b" .:-=+*#%@";
    let (w, h) = (64, 32);
    for j in 0..h {
        let mut line = String::new();
        for i in 0..w {
            // Orthographic camera looking down -z, in world space.
            let x = (i as f64 + 0.5) / w as f64 * 2.4 - 1.2;
            let y = 1.2 - (j as f64 + 0.5) / h as f64 * 2.4;
            let world = Ray::<f64>::new([x, y, 5.0], [0.0, 0.0, -1.0], 0.0, 10.0);
            let mut ray = world.world_to_index(&map);
            let c = match zero_crossing(&mut ray, &acc) {
                Some((_, _, t)) => {
                    let p = ray.at(t);
                    let g = sampler.gradient(Vec3d::new(p[0], p[1], p[2]));
                    let n = grid.index_to_world_grad(Vec3d::new(g[0] as f64, g[1] as f64, g[2] as f64)).unit();
                    let d = n.dot(&light).max(0.0);
                    shades[1 + (d * (shades.len() - 2) as f64).round() as usize] as char
                }
                None => ' ',
            };
            line.push(c);
        }
        println!("{}", line);
    }
    Ok(())
}
