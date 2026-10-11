//! Print information about the grids in `.vdb` files (port of OpenVDB's `vdb_print`).
//!
//! ```text
//! cargo run --example vdb_print -- in.vdb [in.vdb ...] [-l | -stats] [-m | -metadata] [-version]
//! ```

use std::process::ExitCode;
use vdb::io::{File, GridMetadata};
use vdb::metadata::MetaMap;
use vdb::{with_any_grid, AnyGrid, Coord};

const INDENT: &str = "   ";

fn usage(prog: &str) -> String {
    format!(
        "Usage: {} in.vdb [in.vdb ...] [options]\n\
         Which: prints information about OpenVDB grids\n\
         Options:\n    \
         -l, -stats     long printout, including grid statistics\n    \
         -m, -metadata  print per-file and per-grid metadata\n    \
         -version       print version information\n",
        prog
    )
}

/// Format like C++ `ostream << std::setprecision(3) << x` (i.e. `%.3g`).
fn g3(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let digits = x.abs().log10().floor() as i32 + 1;
    let decimals = (3 - digits).max(0) as usize;
    let s = format!("{:.*}", decimals, x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

fn size_as_string(n: u64, units: &str) -> String {
    let s = if n < 1000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{}K", g3(n as f64 / 1e3))
    } else if n < 1_000_000_000 {
        format!("{}M", g3(n as f64 / 1e6))
    } else {
        format!("{}G", g3(n as f64 / 1e9))
    };
    s + units
}

fn bytes_as_string(n: u64) -> String {
    if n >> 30 != 0 {
        format!("{}GB", g3(n as f64 / (1u64 << 30) as f64))
    } else if n >> 20 != 0 {
        format!("{}MB", g3(n as f64 / (1u64 << 20) as f64))
    } else if n >> 10 != 0 {
        format!("{}KB", g3(n as f64 / (1u64 << 10) as f64))
    } else {
        format!("{}B", n)
    }
}

fn coord_as_string(c: Coord, sep: &str, start: &str, stop: &str) -> String {
    format!("{}{}{}{}{}{}{}", start, c[0], sep, c[1], sep, c[2], stop)
}

fn print_meta(meta: &MetaMap, indent: &str) {
    for (k, v) in meta {
        println!("{}{}: {}", indent, k, v);
    }
}

fn background_string(grid: &AnyGrid) -> String {
    with_any_grid!(grid, g => format!("{:?}", g.background()))
}

fn print_transform(grid: &AnyGrid, map_type: &str, indent: &str) {
    let t = grid.transform();
    let m = t.map().affine_matrix();
    println!("{}Transform Type: {}", indent, map_type);
    println!("{}Voxel size: {}", indent, t.voxel_size());
    for (i, row) in m.0.iter().enumerate() {
        let label = if i == 0 { "Matrix: " } else { "        " };
        println!("{}{}[{}, {}, {}, {}]", indent, label, row[0], row[1], row[2], row[3]);
    }
}

fn value_type(grid: &AnyGrid) -> String {
    let t = grid.grid_type();
    t.strip_prefix("Tree_").and_then(|s| s.strip_suffix("_5_4_3")).unwrap_or(&t).to_string()
}

fn print_short(file: &File, grids: &[(GridMetadata, Option<AnyGrid>)], one_file: bool, metadata: bool) {
    let indent = if one_file { "" } else { INDENT };
    let indent2 = format!("{}{}", indent, INDENT);
    if metadata {
        print_meta(file.metadata(), indent);
        if !file.metadata().is_empty() {
            println!();
        }
    }
    for (gm, grid) in grids {
        let name = gm.metadata.get_str("name").unwrap_or(&gm.descriptor.grid_name).to_string();
        let Some(grid) = grid else {
            println!(
                "{}{:<11} {:>6} (not loaded: unsupported grid type {})",
                indent,
                name,
                gm.descriptor.value_type().unwrap_or("?"),
                gm.descriptor.grid_type
            );
            continue;
        };
        let bbox = grid.eval_active_voxel_bbox();
        let (box_str, dim_str) = if bbox.is_empty() {
            ("(empty)".to_string(), "0x0x0".to_string())
        } else {
            (
                format!("{}->{}", coord_as_string(bbox.min, ",", "(", ")"), coord_as_string(bbox.max, ",", "(", ")")),
                coord_as_string(bbox.dim(), "x", "", ""),
            )
        };
        let pad = 40usize.saturating_sub(box_str.len() + dim_str.len()).max(1);
        let box_str = format!("{}{}{}", box_str, " ".repeat(pad), dim_str);
        let mem = with_any_grid!(grid, g => g.mem_usage()) as u64;
        println!(
            "{}{:<11} {:>6} {:<40}  {:>8} {:>6} (Total)",
            indent,
            name,
            value_type(grid),
            box_str,
            size_as_string(grid.active_voxel_count(), "Vox"),
            bytes_as_string(mem)
        );
        if metadata {
            println!("{}background: {}", indent2, background_string(grid));
            print_transform(grid, &gm.map_type, &indent2);
            print_meta(grid.metadata(), &indent2);
            println!();
        }
    }
}

fn print_long(file: &File, grids: &[(GridMetadata, Option<AnyGrid>)]) {
    println!("VDB version: {}", file.version());
    print_meta(file.metadata(), "");
    println!();
    for (i, (gm, grid)) in grids.iter().enumerate() {
        if i > 0 {
            println!("\n");
        }
        let name = gm.metadata.get_str("name").unwrap_or(&gm.descriptor.grid_name);
        println!("Name: {}", name);
        let Some(grid) = grid else {
            println!("    Grid Type: {} (not loaded: unsupported)", gm.descriptor.grid_type);
            continue;
        };
        println!("    Grid Type: {}", grid.grid_type());
        println!("    Grid Class: {}", grid.grid_class().as_str());
        println!("    Compression: {}", gm.compression);
        println!("    Saved as half float: {}", gm.descriptor.save_float_as_half);
        with_any_grid!(grid, g => {
            let tree = g.tree();
            let counts = tree.node_count();
            println!("    Background: {:?}", tree.background());
            println!("    Active Voxel Count: {}", tree.active_voxel_count());
            println!("    Active tile Count: {}", tree.active_tile_count());
            println!("    Inactive Voxel Count: {}", tree.inactive_voxel_count());
            println!("    Leaf Node Count: {}", tree.leaf_count());
            println!("    Non-leaf Node Count: {}", counts.iter().skip(1).sum::<usize>());
            println!("    Nodes per level (leaf to root): {:?}", counts);
            println!("    Memory Usage: {}", bytes_as_string(tree.mem_usage() as u64));
        });
        let bbox = grid.eval_active_voxel_bbox();
        if !bbox.is_empty() {
            println!("    Active voxel bounding box: {} -> {}", bbox.min, bbox.max);
            println!("    Active voxel dimensions: {}", coord_as_string(bbox.dim(), " x ", "", ""));
        }
        print_transform(grid, &gm.map_type, "    ");
        println!("    Metadata:");
        print_meta(grid.metadata(), "      ");
    }
}

/// A file and its grids (`None` for unsupported grid types).
type Loaded = (File, Vec<(GridMetadata, Option<AnyGrid>)>);

fn load(path: &str) -> vdb::io::Result<Loaded> {
    let file = File::open(path)?;
    let metas = file.read_all_grid_metadata()?;
    let mut grids = Vec::with_capacity(metas.len());
    for gm in metas {
        let grid = if gm.descriptor.is_supported() { Some(file.read_grid(&gm.descriptor.unique_name)?) } else { None };
        grids.push((gm, grid));
    }
    Ok((file, grids))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let prog = args.first().and_then(|p| p.rsplit('/').next()).unwrap_or("vdb_print").to_string();
    let (mut stats, mut metadata, mut version) = (false, false, false);
    let mut files = Vec::new();
    for arg in &args[1..] {
        match arg.as_str() {
            "-m" | "-metadata" => metadata = true,
            "-l" | "-stats" => stats = true,
            "-version" | "--version" => version = true,
            "-h" | "-help" | "--help" => {
                print!("{}", usage(&prog));
                return ExitCode::SUCCESS;
            }
            a if a.starts_with('-') => {
                eprintln!("{}: \"{}\" is not a valid option\n{}", prog, a, usage(&prog));
                return ExitCode::FAILURE;
            }
            a if !a.is_empty() => files.push(a.to_string()),
            _ => {}
        }
    }
    if version {
        println!("OpenVDB library version: {}.{}", vdb::io::LIBRARY_MAJOR_VERSION, vdb::io::LIBRARY_MINOR_VERSION);
        println!("OpenVDB file format version: {}", vdb::io::FILE_VERSION);
        if files.is_empty() {
            return ExitCode::SUCCESS;
        }
    }
    if files.is_empty() {
        eprintln!("{}: expected one or more OpenVDB files\n{}", prog, usage(&prog));
        return ExitCode::FAILURE;
    }
    let one_file = files.len() == 1;
    let mut status = ExitCode::SUCCESS;
    for (i, path) in files.iter().enumerate() {
        let (file, grids) = match load(path) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("{}: {} ({})", prog, e, path);
                status = ExitCode::FAILURE;
                continue;
            }
        };
        if stats {
            if !one_file {
                if i > 0 {
                    println!("\n{}\n", "-".repeat(40));
                }
                println!("{}\n", path);
            }
            print_long(&file, &grids);
        } else {
            if !one_file {
                if metadata && i > 0 {
                    println!();
                }
                println!("{}:", path);
            }
            print_short(&file, &grids, one_file, metadata);
        }
    }
    status
}
