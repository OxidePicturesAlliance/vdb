//! Recomputation of bounding boxes and statistics of every node of a NanoVDB grid
//! (port of `nanovdb/tools/GridStats.h`, serial reduction order).

use super::build_types::{BuildType, NanoPod, StatsValue};
use super::data::{
    bbox_is_empty, empty_bbox, grid_off, internal_off, leaf_off, rd_coord, rd_i64, rd_u32, rd_u64, root_off, tile_off,
    tree_off, wr_bbox, wr_u32, wr_u64, wr_vec3d, Map,
};
use super::tree::{key_to_coord, NanoInternal};
use super::{grid_flags, GRID_DATA_SIZE};
use crate::math::{Coord, CoordBBox, Vec3d};

/// What statistics to compute (`nanovdb::tools::StatsMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum StatsMode {
    /// No statistics at all (fastest).
    Disable = 0,
    /// Only bounding boxes of active values.
    BBox = 1,
    /// Bounding boxes and min/max values.
    MinMax = 2,
    /// Bounding boxes, min/max, average and standard deviation (the default).
    #[default]
    All = 3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Noop,
    Extrema,
    Stats,
}

/// Running statistics (`Extrema`, `Stats` or `NoopStats` depending on `kind`).
#[derive(Clone, Copy)]
struct Acc<V: StatsValue> {
    kind: Kind,
    min: (V::Key, V),
    max: (V::Key, V),
    size: u64,
    avg: f64,
    aux: f64,
}

impl<V: StatsValue> Acc<V> {
    fn new(kind: Kind) -> Self {
        Acc { kind, min: V::init_min(), max: V::init_max(), size: 0, avg: 0.0, aux: 0.0 }
    }
    #[inline]
    fn extrema(&mut self, k: V::Key, v: V) {
        if k < self.min.0 {
            self.min = (k, v);
        }
        if self.max.0 < k {
            self.max = (k, v);
        }
    }
    fn add(&mut self, v: V) {
        match self.kind {
            Kind::Noop => {}
            Kind::Extrema => self.extrema(v.key(), v),
            Kind::Stats => {
                let k = v.key();
                self.extrema(k, v);
                self.size += 1;
                let x = V::key_f64(k);
                let delta = x - self.avg;
                self.avg += delta / self.size as f64;
                self.aux += delta * (x - self.avg);
            }
        }
    }
    fn add_n(&mut self, v: V, n: u64) {
        match self.kind {
            Kind::Noop => {}
            Kind::Extrema => self.extrema(v.key(), v),
            Kind::Stats => {
                let k = v.key();
                let denom = 1.0 / (self.size + n) as f64;
                let delta = V::key_f64(k) - self.avg;
                self.avg += denom * delta * n as f64;
                self.aux += denom * delta * delta * self.size as f64 * n as f64;
                self.extrema(k, v);
                self.size += n;
            }
        }
    }
    fn merge(&mut self, o: &Self) {
        match self.kind {
            Kind::Noop => {}
            Kind::Extrema => {
                if o.min.0 < self.min.0 {
                    self.min = o.min;
                }
                if self.max.0 < o.max.0 {
                    self.max = o.max;
                }
            }
            Kind::Stats => {
                if o.size > 0 {
                    let denom = 1.0 / (self.size + o.size) as f64;
                    let delta = o.avg - self.avg;
                    self.avg += denom * delta * o.size as f64;
                    self.aux += o.aux + denom * delta * delta * self.size as f64 * o.size as f64;
                    if o.min.0 < self.min.0 {
                        self.min = o.min;
                    }
                    if self.max.0 < o.max.0 {
                        self.max = o.max;
                    }
                    self.size += o.size;
                }
            }
        }
    }
    fn std(&self) -> f64 {
        let var = if self.size < 2 { 0.0 } else { self.aux / self.size as f64 };
        var.sqrt()
    }
}

struct NodeStats<V: StatsValue> {
    acc: Acc<V>,
    bbox: CoordBBox,
}

impl<V: StatsValue> NodeStats<V> {
    fn new(kind: Kind) -> Self {
        NodeStats { acc: Acc::new(kind), bbox: empty_bbox() }
    }
    fn add(&mut self, o: &Self) {
        self.acc.merge(&o.acc);
        self.bbox.min = self.bbox.min.min_elem(&o.bbox.min);
        self.bbox.max = self.bbox.max.max_elem(&o.bbox.max);
    }
    fn expand(&mut self, lo: Coord, hi: Coord) {
        self.bbox.min = self.bbox.min.min_elem(&lo);
        self.bbox.max = self.bbox.max.max_elem(&hi);
    }
}

/// Write min/max (and avg/stddev for [`Kind::Stats`]) at the given offsets.
fn write_stats<B: BuildType>(buf: &mut [u8], acc: &Acc<B::Value>, offs: [usize; 4]) {
    if acc.kind == Kind::Noop {
        return;
    }
    acc.min.1.write(buf, offs[0]);
    acc.max.1.write(buf, offs[1]);
    if acc.kind == Kind::Stats {
        B::Stats::from_f64(acc.avg).write(buf, offs[2]);
        B::Stats::from_f64(acc.std()).write(buf, offs[3]);
    }
}

/// Update the bounding box of a leaf from its value mask (`LeafNode::updateBBox`).
fn update_leaf_bbox(buf: &mut [u8], off: usize) -> bool {
    let w: Vec<u64> = (0..8).map(|i| rd_u64(buf, off + leaf_off::VALUE_MASK + 8 * i)).collect();
    if w.iter().all(|&x| x == 0) {
        buf[off + leaf_off::FLAGS] &= !2;
        return false;
    }
    let mut word64 = w[0];
    let mut xmin = if word64 != 0 { 0 } else { 8 };
    let mut xmax = xmin;
    for (i, &wi) in w.iter().enumerate().skip(1) {
        if wi != 0 {
            word64 |= wi;
            if xmin == 8 {
                xmin = i;
            }
            xmax = i;
        }
    }
    let mut update = |min: u32, max: u32, axis: usize| {
        let o = off + leaf_off::BBOX_MIN + 4 * axis;
        let v = super::data::rd_i32(buf, o);
        super::data::wr_i32(buf, o, (v & !7) + min as i32);
        buf[off + leaf_off::BBOX_DIF + axis] = (max - min) as u8;
    };
    update(xmin as u32, xmax as u32, 0);
    update(word64.trailing_zeros() >> 3, (63 - word64.leading_zeros()) >> 3, 1);
    let word32 = (word64 as u32) | ((word64 >> 32) as u32);
    let word16 = (word32 as u16) | ((word32 >> 16) as u16);
    let byte = ((word16 as u8) | ((word16 >> 8) as u8)) as u32;
    update(byte.trailing_zeros(), 31 - byte.leading_zeros(), 2);
    buf[off + leaf_off::FLAGS] |= 2;
    true
}

fn process_leaf<B: BuildType>(buf: &mut [u8], off: usize, kind: Kind) -> NodeStats<B::Value> {
    let mut local = NodeStats::new(kind);
    if update_leaf_bbox(buf, off) {
        let lo = rd_coord(buf, off + leaf_off::BBOX_MIN);
        let d = |i: usize| buf[off + leaf_off::BBOX_DIF + i] as i32;
        local.bbox = CoordBBox::new(lo, Coord::new(lo[0] + d(0), lo[1] + d(1), lo[2] + d(2)));
        if kind != Kind::Noop {
            for w in 0..8 {
                let mut word = rd_u64(buf, off + leaf_off::VALUE_MASK + 8 * w);
                while word != 0 {
                    let n = (w << 6) + word.trailing_zeros() as usize;
                    word &= word - 1;
                    local.acc.add(B::leaf_value(buf, off, n));
                }
            }
            B::leaf_set_min_max(buf, off, local.acc.min.1, local.acc.max.1);
            if kind == Kind::Stats {
                B::leaf_set_avg_dev(buf, off, local.acc.avg, local.acc.std());
            }
            buf[off + leaf_off::FLAGS] &= !1;
        }
    }
    local
}

fn process_internal<B: BuildType, const L: u32>(buf: &mut [u8], off: usize, kind: Kind) -> NodeStats<B::Value> {
    let lay = *NanoInternal::<B, L>::layout();
    let child_total = NanoInternal::<B, L>::CHILD_TOTAL;
    let mut total = NodeStats::new(kind);
    let m = !((1i32 << (L + child_total)) - 1);
    let raw = rd_coord(buf, off + internal_off::BBOX);
    let origin = Coord::new(raw[0] & m, raw[1] & m, raw[2] & m);
    let words = 1usize << (3 * L - 6);
    // Active tiles.
    for w in 0..words {
        let mut word = rd_u64(buf, off + lay.value_mask + 8 * w);
        while word != 0 {
            let n = (w << 6) + word.trailing_zeros() as usize;
            word &= word - 1;
            if kind != Kind::Noop {
                let v = B::Value::read(buf, off + lay.table + n * lay.stride);
                total.acc.add_n(v, 1u64 << (3 * child_total));
            }
            let ijk = NanoInternal::<B, L>::child_origin(origin, n);
            total.expand(ijk, ijk.offset((1 << child_total) - 1));
        }
    }
    // Children.
    for w in 0..words {
        let mut word = rd_u64(buf, off + lay.child_mask + 8 * w);
        while word != 0 {
            let n = (w << 6) + word.trailing_zeros() as usize;
            word &= word - 1;
            let child = (off as i64 + rd_i64(buf, off + lay.table + n * lay.stride)) as usize;
            let s =
                if L == 5 { process_internal::<B, 4>(buf, child, kind) } else { process_leaf::<B>(buf, child, kind) };
            total.add(&s);
        }
    }
    wr_bbox(buf, off + internal_off::BBOX, &total.bbox);
    let mut flags = rd_u64(buf, off + internal_off::FLAGS);
    if bbox_is_empty(&total.bbox) {
        flags |= 1;
        flags &= !2;
    } else {
        flags |= 2;
        if kind != Kind::Noop {
            write_stats::<B>(buf, &total.acc, [off + lay.min, off + lay.max, off + lay.avg, off + lay.dev]);
            flags &= !1;
        }
    }
    wr_u64(buf, off + internal_off::FLAGS, flags);
    total
}

fn process_root<B: BuildType>(buf: &mut [u8], off: usize, kind: Kind) -> CoordBBox {
    let n = B::NODE;
    let table_size = rd_u32(buf, off + root_off::TABLE_SIZE) as usize;
    if table_size == 0 {
        let bg = B::Value::read(buf, off + n.root_background);
        bg.write(buf, off + n.root_min);
        bg.write(buf, off + n.root_max);
        B::Stats::default().write(buf, off + n.root_avg);
        B::Stats::default().write(buf, off + n.root_dev);
        let e = empty_bbox();
        wr_bbox(buf, off + root_off::BBOX, &e);
        return e;
    }
    let mut total = NodeStats::<B::Value>::new(kind);
    for i in 0..table_size {
        let t = off + n.root_size + i * n.tile_size;
        let child = rd_i64(buf, t + tile_off::CHILD);
        if child != 0 {
            let s = process_internal::<B, 5>(buf, (off as i64 + child) as usize, kind);
            total.add(&s);
        } else if rd_u32(buf, t + tile_off::STATE) != 0 {
            let ijk = key_to_coord(rd_u64(buf, t + tile_off::KEY));
            total.expand(ijk, ijk.offset(4095));
            if kind != Kind::Noop {
                total.acc.add_n(B::Value::read(buf, t + n.tile_value), 1u64 << 36);
            }
        }
    }
    write_stats::<B>(buf, &total.acc, [off + n.root_min, off + n.root_max, off + n.root_avg, off + n.root_dev]);
    wr_bbox(buf, off + root_off::BBOX, &total.bbox);
    total.bbox
}

/// Recompute the bounding boxes and statistics of all nodes of the grid in `grid` (the
/// bytes of one grid with build type `B`), and the grid's world bounding box and flags
/// (`nanovdb::tools::updateGridStats`).
///
/// Types without statistics (`bool`, [`ValueMask`](super::ValueMask) and the index types)
/// only get bounding boxes, whatever the mode.
pub fn update_grid_stats<B: BuildType>(grid: &mut [u8], mode: StatsMode) {
    let kind = match mode {
        StatsMode::Disable => return,
        StatsMode::BBox => Kind::Noop,
        _ if !B::HAS_STATS => Kind::Noop,
        StatsMode::MinMax => Kind::Extrema,
        StatsMode::All => Kind::Stats,
    };
    let root = GRID_DATA_SIZE + rd_i64(grid, GRID_DATA_SIZE + tree_off::NODE_OFFSET + 24) as usize;
    let bbox = process_root::<B>(grid, root, kind);
    let mut flags = rd_u32(grid, grid_off::FLAGS);
    if bbox_is_empty(&bbox) {
        wr_vec3d(grid, grid_off::WORLD_BBOX, Vec3d::splat(f64::MAX));
        wr_vec3d(grid, grid_off::WORLD_BBOX + 24, Vec3d::splat(-f64::MAX));
        flags &= !grid_flags::HAS_BBOX;
    } else {
        let map = Map::read(grid, grid_off::MAP);
        let (lo, hi) = map.transform_bbox(bbox.min, bbox.max.offset(1));
        wr_vec3d(grid, grid_off::WORLD_BBOX, lo);
        wr_vec3d(grid, grid_off::WORLD_BBOX + 24, hi);
        flags |= grid_flags::HAS_BBOX;
    }
    let set = |flags: &mut u32, bit: u32, on: bool| {
        if on {
            *flags |= bit
        } else {
            *flags &= !bit
        }
    };
    set(&mut flags, grid_flags::HAS_MIN_MAX, kind != Kind::Noop);
    set(&mut flags, grid_flags::HAS_AVERAGE, kind == Kind::Stats);
    set(&mut flags, grid_flags::HAS_STD_DEVIATION, kind == Kind::Stats);
    wr_u32(grid, grid_off::FLAGS, flags);
}
