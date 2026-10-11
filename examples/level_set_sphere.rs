//! Build two level-set spheres, combine them with a CSG union, and print statistics.
//!
//! Run with `cargo run --release --example level_set_sphere`.

use vdb::math::Vec3f;
use vdb::tools::composite::csg_union;
use vdb::tools::count::{count_active_tiles, min_max};
use vdb::tools::diagnostics::check_level_set;
use vdb::tools::level_set_measure::LevelSetMeasure;
use vdb::tools::level_set_sphere::create_level_set_sphere;
use vdb::tools::statistics::{statistics, ValueIterKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let voxel_size = 0.05;
    let half_width = 3.0;
    let mut a = create_level_set_sphere::<f32>(1.0, Vec3f::new(0.0, 0.0, 0.0), voxel_size, half_width, true)?;
    let mut b = create_level_set_sphere::<f32>(0.8, Vec3f::new(0.7, 0.1, 0.0), voxel_size, half_width, true)?;
    println!("sphere A: {} active voxels, {} leaves", a.active_voxel_count(), a.tree().leaf_count());
    println!("sphere B: {} active voxels, {} leaves", b.active_voxel_count(), b.tree().leaf_count());

    csg_union(&mut a, &mut b, true, false)?;
    a.set_name("union");

    let tree = a.tree();
    println!("\nunion of A and B:");
    println!("  active voxels: {}", a.active_voxel_count());
    println!("  leaf nodes:    {}", tree.leaf_count());
    println!("  active tiles:  {}", count_active_tiles(tree));
    let mm = min_max(tree);
    println!("  value range:   [{}, {}]", mm.min, mm.max);
    let st = statistics(tree, ValueIterKind::On, true);
    println!("  mean / std:    {:.6} / {:.6}", st.mean(), st.std_dev());

    let mut measure = LevelSetMeasure::new(&a)?;
    println!("  surface area:  {:.4}", measure.area(true));
    println!("  volume:        {:.4}", measure.volume(true));
    println!("  genus:         {}", measure.genus());

    let report = check_level_set(&a, 9);
    println!("  level set check: {}", if report.is_empty() { "ok" } else { report.trim_end() });
    Ok(())
}
