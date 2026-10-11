//! Shared helpers for the I/O integration tests.
//!
//! [`dump_file`] produces the same canonical text as `vdb_ref dump` (tests/data/cpp/vdb_ref.cpp),
//! so a file's content as seen by this crate can be compared with what the C++ library sees.

#![allow(dead_code)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use vdb::io::File;
use vdb::math::{Vec3, Vec3d, Vec3f, Vec3i};
use vdb::metadata::{MetaMap, MetaValue};
use vdb::tree::Tree;
use vdb::types::Value;
use vdb::AnyGrid;

/// Path of a file in tests/data.
pub fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(name)
}

/// A fresh scratch directory for test output.
pub fn scratch_dir(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("io_tests").join(test);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn mix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

fn coord_hash(c: vdb::Coord) -> u64 {
    let mut h = mix(c[0] as u32 as u64);
    h = mix(h ^ c[1] as u32 as u64);
    mix(h ^ c[2] as u32 as u64)
}

/// The bit pattern of a value, as hashed by `vdb_ref`.
pub trait ValueBits {
    fn bits(&self) -> u64;
}
impl ValueBits for f32 {
    fn bits(&self) -> u64 {
        self.to_bits() as u64
    }
}
impl ValueBits for f64 {
    fn bits(&self) -> u64 {
        self.to_bits()
    }
}
impl ValueBits for i32 {
    fn bits(&self) -> u64 {
        *self as u32 as u64
    }
}
impl ValueBits for i64 {
    fn bits(&self) -> u64 {
        *self as u64
    }
}
impl ValueBits for bool {
    fn bits(&self) -> u64 {
        *self as u64
    }
}
impl<T: ValueBits> ValueBits for Vec3<T> {
    fn bits(&self) -> u64 {
        mix(mix(mix(self.0[0].bits()) ^ self.0[1].bits()) ^ self.0[2].bits())
    }
}

fn hex(v: u64) -> String {
    format!("{:016x}", v)
}

fn meta_string(v: &MetaValue) -> String {
    let f = |x: f32| hex(x.to_bits() as u64);
    let d = |x: f64| hex(x.to_bits());
    let join = |v: Vec<String>| v.join(" ");
    match v {
        MetaValue::Bool(b) => (*b as i32).to_string(),
        MetaValue::Int32(x) => x.to_string(),
        MetaValue::Int64(x) => x.to_string(),
        MetaValue::Float(x) => f(*x),
        MetaValue::Double(x) => d(*x),
        MetaValue::String(s) => format!("\"{}\"", s),
        MetaValue::Vec2i(a) => join(a.iter().map(|x| x.to_string()).collect()),
        MetaValue::Vec2s(a) => join(a.iter().map(|x| f(*x)).collect()),
        MetaValue::Vec2d(a) => join(a.iter().map(|x| d(*x)).collect()),
        MetaValue::Vec3i(a) => join(a.0.iter().map(|x| x.to_string()).collect()),
        MetaValue::Vec3s(a) => join(a.0.iter().map(|x| f(*x)).collect()),
        MetaValue::Vec3d(a) => join(a.0.iter().map(|x| d(*x)).collect()),
        MetaValue::Vec4i(a) => join(a.iter().map(|x| x.to_string()).collect()),
        MetaValue::Vec4s(a) => join(a.iter().map(|x| f(*x)).collect()),
        MetaValue::Vec4d(a) => join(a.iter().map(|x| d(*x)).collect()),
        MetaValue::Mat4s(m) => join(m.0.iter().flatten().map(|x| f(*x)).collect()),
        MetaValue::Mat4d(m) => join(m.0.iter().flatten().map(|x| d(*x)).collect()),
        MetaValue::Unknown { data, .. } => format!("<{} bytes>", data.len()),
    }
}

pub fn dump_meta(out: &mut String, meta: &MetaMap, indent: &str) {
    for (k, v) in meta {
        writeln!(out, "{}meta {} {} {}", indent, k, v.type_name(), meta_string(v)).unwrap();
    }
}

fn dump_tree<T: Value + ValueBits>(out: &mut String, tree: &Tree<T>, is_mask: bool) {
    writeln!(out, "  background {}", hex(tree.background().bits())).unwrap();
    writeln!(out, "  active_voxels {}", tree.active_voxel_count()).unwrap();
    writeln!(out, "  active_tiles {}", tree.active_tile_count()).unwrap();
    writeln!(out, "  leaf_count {}", tree.leaf_count()).unwrap();
    let bbox = tree.eval_active_voxel_bbox();
    if bbox.is_empty() {
        writeln!(out, "  bbox empty").unwrap();
    } else {
        let (a, b) = (bbox.min, bbox.max);
        writeln!(out, "  bbox {} {} {} {} {} {}", a[0], a[1], a[2], b[0], b[1], b[2]).unwrap();
    }
    let (mut active_hash, mut leaf_hash, mut tile_hash) = (0u64, 0u64, 0u64);
    for leaf in tree.leaves() {
        for i in 0..512 {
            let c = leaf.offset_to_global_coord(i);
            let on = leaf.is_value_on_at(i);
            let v = if is_mask { on as u64 } else { leaf.get_value_at(i).bits() };
            let h = mix(coord_hash(c) ^ v);
            if on {
                active_hash = active_hash.wrapping_add(h);
            }
            leaf_hash = leaf_hash.wrapping_add(mix(h ^ if on { 0x5555 } else { 0xaaaa }));
        }
    }
    tree.visit_tiles(&mut |bbox, v, on, _level| {
        let dim = bbox.dim()[0] as u64;
        let h = mix(coord_hash(bbox.min) ^ v.bits());
        tile_hash = tile_hash.wrapping_add(mix(h ^ mix(dim * 2 + on as u64)));
    });
    writeln!(out, "  active_hash {}", hex(active_hash)).unwrap();
    writeln!(out, "  leaf_hash {}", hex(leaf_hash)).unwrap();
    writeln!(out, "  tile_hash {}", hex(tile_hash)).unwrap();
}

/// Canonical description of a loaded grid's tree and transform.
pub fn dump_grid(out: &mut String, grid: &AnyGrid) {
    match grid {
        AnyGrid::Float(g) => dump_tree(out, g.tree(), false),
        AnyGrid::Double(g) => dump_tree(out, g.tree(), false),
        AnyGrid::Int32(g) => dump_tree(out, g.tree(), false),
        AnyGrid::Int64(g) => dump_tree(out, g.tree(), false),
        AnyGrid::Bool(g) => dump_tree(out, g.tree(), false),
        AnyGrid::Mask(g) => dump_tree(out, g.tree(), true),
        AnyGrid::Vec3S(g) => dump_tree::<Vec3f>(out, g.tree(), false),
        AnyGrid::Vec3D(g) => dump_tree::<Vec3d>(out, g.tree(), false),
        AnyGrid::Vec3I(g) => dump_tree::<Vec3i>(out, g.tree(), false),
        // Like `vdb_ref dump`, point data grids are described by tests/points.rs instead.
        AnyGrid::PointData(_) | AnyGrid::PointIndex(_) => {
            out.push_str("  unsupported\n");
            return;
        }
    }
    let m = grid.transform().map().affine_matrix();
    out.push_str("  transform");
    for v in m.0.iter().flatten() {
        write!(out, " {}", hex(v.to_bits())).unwrap();
    }
    out.push('\n');
}

/// Canonical description of a file, identical in format and grid order to `vdb_ref dump`.
pub fn dump_file(path: &Path) -> String {
    let file = File::open(path).unwrap();
    let mut out = String::new();
    writeln!(out, "file_version {}", file.header().file_version).unwrap();
    dump_meta(&mut out, file.metadata(), "");
    let metas = file.read_all_grid_metadata().unwrap();
    let mut order: Vec<usize> = (0..metas.len()).collect();
    if file.header().has_grid_offsets {
        // OpenVDB's File::getGrids() returns grids ordered by name.
        order.sort_by(|&a, &b| metas[a].descriptor.grid_name.cmp(&metas[b].descriptor.grid_name));
    }
    for i in order {
        let m = &metas[i];
        writeln!(out, "grid {}", m.metadata.get_str("name").unwrap_or("")).unwrap();
        writeln!(out, "  type {}", m.descriptor.grid_type).unwrap();
        dump_meta(&mut out, &m.metadata, "  ");
        if m.descriptor.is_supported() {
            let grid = file.read_grid(&m.descriptor.unique_name).unwrap();
            dump_grid(&mut out, &grid);
        } else {
            out.push_str("  unsupported\n");
        }
    }
    out
}

/// Location of the C++ `vdb_ref` tool: `$VDB_REF`, or tests/data/cpp/vdb_ref if built there.
pub fn vdb_ref_binary() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VDB_REF") {
        return Some(PathBuf::from(p));
    }
    let p = data("cpp").join("vdb_ref");
    p.exists().then_some(p)
}
