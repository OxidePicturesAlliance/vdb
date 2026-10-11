//! Randomized model-based tests: the VDB tree must behave like a dense map of voxels.
// `is_multiple_of` needs Rust 1.90, newer than the MSRV (1.85).
#![allow(clippy::manual_is_multiple_of, unknown_lints)]

use std::collections::HashMap;
use vdb::math::{Coord, CoordBBox};
use vdb::tree::{FloatTree, Node};

/// Small deterministic PRNG (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u64) as i32
    }
}

/// Reference model: explicit (value, active) per voxel over a bounded domain.
struct Model {
    bg: f32,
    vox: HashMap<Coord, (f32, bool)>,
}
impl Model {
    fn get(&self, c: &Coord) -> (f32, bool) {
        self.vox.get(c).copied().unwrap_or((self.bg, false))
    }
}

const LO: i32 = -40;
const HI: i32 = 40;

fn check(tree: &FloatTree, model: &Model, rng: &mut Rng) {
    let acc = tree.accessor();
    for _ in 0..3000 {
        let c = Coord::new(rng.range(LO, HI), rng.range(LO, HI), rng.range(LO, HI));
        let (v, on) = model.get(&c);
        assert_eq!(tree.get_value(&c), v, "value at {:?}", c);
        assert_eq!(tree.is_value_on(&c), on, "state at {:?}", c);
        assert_eq!(acc.probe_value(&c), (v, on), "accessor at {:?}", c);
    }
    let count = model.vox.values().filter(|(_, on)| *on).count() as u64;
    assert_eq!(tree.active_voxel_count(), count);
    let mut bbox = CoordBBox::empty();
    for (c, (_, on)) in &model.vox {
        if *on {
            bbox.expand_coord(*c);
        }
    }
    assert_eq!(tree.eval_active_voxel_bbox(), bbox);
}

#[test]
fn random_operations_match_model() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for round in 0..4 {
        let mut tree = FloatTree::new(-2.0);
        let mut model = Model { bg: -2.0, vox: HashMap::new() };
        for step in 0..400 {
            let op = rng.next() % 10;
            let c = Coord::new(rng.range(LO, HI), rng.range(LO, HI), rng.range(LO, HI));
            let val = (rng.next() % 5) as f32;
            match op {
                0..=2 => {
                    tree.set_value_on(&c, val);
                    model.vox.insert(c, (val, true));
                }
                3 => {
                    tree.set_value_off(&c, val);
                    model.vox.insert(c, (val, false));
                }
                4 => {
                    let on = rng.next() % 2 == 0;
                    tree.set_active_state(&c, on);
                    let (v, _) = model.get(&c);
                    model.vox.insert(c, (v, on));
                }
                5 => {
                    let d = Coord::new(rng.range(0, 20), rng.range(0, 20), rng.range(0, 20));
                    let b = CoordBBox::new(c, (c + d).min_elem(&Coord::splat(HI)));
                    let on = rng.next() % 3 != 0;
                    tree.fill(&b, val, on);
                    for x in b {
                        model.vox.insert(x, (val, on));
                    }
                }
                6 => {
                    tree.modify_value(&c, |v| *v += 1.0);
                    let (v, _) = model.get(&c);
                    model.vox.insert(c, (v + 1.0, true));
                }
                7 => {
                    let mut acc = tree.accessor_mut();
                    for i in 0..10 {
                        let p = Coord::new(c[0], c[1], (c[2] + i).min(HI));
                        acc.set_value(&p, val + i as f32);
                        model.vox.insert(p, (val + i as f32, true));
                    }
                }
                8 => {
                    tree.prune(0.0);
                }
                _ => {
                    let d = Coord::splat(rng.range(5, 30));
                    let b = CoordBBox::new(c, c + d);
                    tree.clip(&b);
                    let bg = model.bg;
                    for (k, e) in model.vox.iter_mut() {
                        if !b.is_inside(k) {
                            *e = (bg, false);
                        }
                    }
                }
            }
            if step % 50 == 49 {
                check(&tree, &model, &mut rng);
            }
        }
        check(&tree, &model, &mut rng);

        // Voxelizing tiles must not change any value.
        let mut dense = tree.clone();
        dense.voxelize_active_tiles();
        assert_eq!(dense.active_tile_count(), 0);
        check(&dense, &model, &mut rng);

        // Changing the background updates inactive background values only.
        tree.set_background(7.0);
        for e in model.vox.values_mut() {
            if !e.1 && e.0 == -2.0 {
                e.0 = 7.0;
            }
        }
        model.bg = 7.0;
        check(&tree, &model, &mut rng);
        let _ = round;
    }
}

#[test]
fn leaf_iteration_is_complete() {
    let mut tree = FloatTree::new(0.0);
    let mut rng = Rng(42);
    for _ in 0..2000 {
        let c = Coord::new(rng.range(-500, 500), rng.range(-500, 500), rng.range(-500, 500));
        tree.set_value(&c, 1.0);
    }
    let n: u64 = tree.leaves().map(|l| l.on_voxel_count()).sum();
    assert_eq!(n, tree.active_voxel_count());
    assert_eq!(tree.active_voxels().count() as u64, n);
    tree.par_for_each_leaf_mut(|l| l.for_each_on_mut(|_, v| *v *= 3.0));
    assert!(tree.active_voxels().all(|(_, v)| v == 3.0));
    // Leaves come out in a deterministic, sorted-by-root-key order.
    let origins: Vec<Coord> = tree.leaves().map(|l| l.origin()).collect();
    let mut uniq = origins.clone();
    uniq.dedup();
    assert_eq!(uniq.len(), origins.len());
}
