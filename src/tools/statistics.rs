//! Histograms, extrema and statistics of grid values (port of `openvdb/tools/Statistics.h`
//! and of the `Extrema`, `Stats` and `Histogram` accumulators of `openvdb/math/Stats.h`).
//!
//! Values are visited with a [`ValueIterKind`] (active, inactive or all values). Tiles are
//! weighted by the number of voxels they cover; vector values contribute their length.

use crate::math::Coord;
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue};
use crate::tree::{Internal2, LeafNode, Tree, ValueAccessor};
use rayon::prelude::*;

/// Which values of a tree to visit (`ValueOnCIter`, `ValueOffCIter` or `ValueAllCIter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ValueIterKind {
    /// Active voxels and tiles.
    #[default]
    On,
    /// Inactive voxels and tiles (including inactive root tiles).
    Off,
    /// All voxels and tiles.
    All,
}

impl ValueIterKind {
    #[inline]
    pub fn accepts(&self, active: bool) -> bool {
        match self {
            ValueIterKind::On => active,
            ValueIterKind::Off => !active,
            ValueIterKind::All => true,
        }
    }
}

/// Minimum, maximum and sample count (`math::Extrema`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extrema {
    size: u64,
    min: f64,
    max: f64,
}

impl Default for Extrema {
    fn default() -> Self {
        Extrema { size: 0, min: f64::MAX, max: -f64::MAX }
    }
}

impl Extrema {
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a single sample.
    pub fn add(&mut self, val: f64) {
        self.size += 1;
        self.min = val.min(self.min);
        self.max = val.max(self.max);
    }
    /// Add `n` samples of the same value.
    pub fn add_n(&mut self, val: f64, n: u64) {
        self.size += n;
        self.min = val.min(self.min);
        self.max = val.max(self.max);
    }
    /// Merge another accumulator.
    pub fn merge(&mut self, other: &Extrema) {
        if other.size > 0 {
            self.size += other.size;
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn min(&self) -> f64 {
        self.min
    }
    pub fn max(&self) -> f64 {
        self.max
    }
    pub fn range(&self) -> f64 {
        self.max - self.min
    }
}

/// Running mean and variance in addition to the extrema (`math::Stats`, using Welford's
/// algorithm; the variance is the population variance).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Stats {
    extrema: Extrema,
    avg: f64,
    aux: f64,
}

impl Stats {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, val: f64) {
        self.extrema.add(val);
        let delta = val - self.avg;
        self.avg += delta / self.extrema.size as f64;
        self.aux += delta * (val - self.avg);
    }
    pub fn add_n(&mut self, val: f64, n: u64) {
        let denom = 1.0 / (self.extrema.size + n) as f64;
        let delta = val - self.avg;
        self.avg += denom * delta * n as f64;
        self.aux += denom * delta * delta * self.extrema.size as f64 * n as f64;
        self.extrema.add_n(val, n);
    }
    pub fn merge(&mut self, other: &Stats) {
        if other.extrema.size > 0 {
            let denom = 1.0 / (self.extrema.size + other.extrema.size) as f64;
            let delta = other.avg - self.avg;
            self.avg += denom * delta * other.extrema.size as f64;
            self.aux += other.aux + denom * delta * delta * self.extrema.size as f64 * other.extrema.size as f64;
            self.extrema.merge(&other.extrema);
        }
    }
    pub fn size(&self) -> u64 {
        self.extrema.size
    }
    pub fn min(&self) -> f64 {
        self.extrema.min
    }
    pub fn max(&self) -> f64 {
        self.extrema.max
    }
    pub fn range(&self) -> f64 {
        self.extrema.range()
    }
    pub fn mean(&self) -> f64 {
        self.avg
    }
    /// Population variance (zero for fewer than two samples).
    pub fn variance(&self) -> f64 {
        if self.extrema.size < 2 {
            0.0
        } else {
            self.aux / self.extrema.size as f64
        }
    }
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// The extrema part of the statistics.
    pub fn extrema(&self) -> Extrema {
        self.extrema
    }
}

/// A histogram with equal-width bins over `[min, max]` (`math::Histogram`).
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    size: u64,
    min: f64,
    max: f64,
    delta: f64,
    bins: Vec<u64>,
}

impl Histogram {
    /// Return an error unless `min < max` and `num_bins > 0`.
    pub fn new(min: f64, max: f64, num_bins: usize) -> ToolResult<Self> {
        let hmax = max + 1e-10;
        if hmax <= min {
            return Err(ToolError::Value("Histogram: expected min < max".into()));
        }
        if num_bins == 0 {
            return Err(ToolError::Value("Histogram: expected at least one bin".into()));
        }
        Ok(Histogram { size: 0, min, max: hmax, delta: num_bins as f64 / (max - min), bins: vec![0; num_bins] })
    }
    /// A histogram spanning the range of `stats`.
    pub fn from_stats(stats: &Stats, num_bins: usize) -> ToolResult<Self> {
        let (min, max) = (stats.min(), stats.max() + 1e-10);
        if max <= min {
            return Err(ToolError::Value("Histogram: expected min < max".into()));
        }
        if num_bins == 0 {
            return Err(ToolError::Value("Histogram: expected at least one bin".into()));
        }
        Ok(Histogram { size: 0, min, max, delta: num_bins as f64 / (max - min), bins: vec![0; num_bins] })
    }
    /// Add `n` samples of `val`; returns false if the value is outside the range or NaN.
    pub fn add_n(&mut self, val: f64, n: u64) -> bool {
        if val.is_nan() || val < self.min || val > self.max {
            return false;
        }
        let i = (self.delta * (val - self.min)) as usize;
        let last = self.bins.len() - 1;
        self.bins[i.min(last)] += n;
        self.size += n;
        true
    }
    pub fn add(&mut self, val: f64) -> bool {
        self.add_n(val, 1)
    }
    /// Merge a histogram with the same range and bin count; returns false otherwise.
    pub fn merge(&mut self, other: &Histogram) -> bool {
        if !((self.min - other.min).abs() <= 1e-15)
            || !((self.max - other.max).abs() <= 1e-15)
            || self.bins.len() != other.bins.len()
        {
            return false;
        }
        for (a, b) in self.bins.iter_mut().zip(&other.bins) {
            *a += *b;
        }
        self.size += other.size;
        true
    }
    pub fn num_bins(&self) -> usize {
        self.bins.len()
    }
    pub fn min(&self) -> f64 {
        self.min
    }
    pub fn max(&self) -> f64 {
        self.max
    }
    /// Lower bound of bin `n`.
    pub fn bin_min(&self, n: usize) -> f64 {
        self.min + n as f64 / self.delta
    }
    /// Upper bound of bin `n`.
    pub fn bin_max(&self, n: usize) -> f64 {
        self.min + (n + 1) as f64 / self.delta
    }
    /// Number of samples in bin `n`.
    pub fn count(&self, n: usize) -> u64 {
        self.bins[n]
    }
    /// Total number of samples.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Accumulate per-value contributions over the selected values: `add(acc, value, count)`
/// is called once per voxel value and once per tile with its voxel count.
fn accumulate_values<T: ToolValue, A: Send + Sync + Clone>(
    tree: &Tree<T>,
    kind: ValueIterKind,
    threaded: bool,
    init: A,
    add: impl Fn(&mut A, f64, u64) + Sync,
    join: impl Fn(&mut A, &A) + Sync,
) -> A {
    let mut acc = init.clone();
    tree.visit_tiles(&mut |bbox, v, on, _| {
        if kind.accepts(on) {
            add(&mut acc, v.stat_value(), bbox.volume());
        }
    });
    let leaf_op = |leaf: &&LeafNode<T>| {
        let mut a = init.clone();
        for (_, v, on) in leaf.iter_all() {
            if kind.accepts(on) {
                add(&mut a, v.stat_value(), 1);
            }
        }
        a
    };
    let leaves = tree.leaf_refs();
    let parts: Vec<A> =
        if threaded { leaves.par_iter().map(leaf_op).collect() } else { leaves.iter().map(leaf_op).collect() };
    for p in &parts {
        join(&mut acc, p);
    }
    acc
}

/// Extrema of the selected values (`tools::extrema`).
pub fn extrema<T: ToolValue>(tree: &Tree<T>, kind: ValueIterKind, threaded: bool) -> Extrema {
    accumulate_values(
        tree,
        kind,
        threaded,
        Extrema::new(),
        |e, v, n| if n == 1 { e.add(v) } else { e.add_n(v, n) },
        |a, b| a.merge(b),
    )
}

/// Statistics (extrema, mean, variance) of the selected values (`tools::statistics`).
pub fn statistics<T: ToolValue>(tree: &Tree<T>, kind: ValueIterKind, threaded: bool) -> Stats {
    accumulate_values(
        tree,
        kind,
        threaded,
        Stats::new(),
        |s, v, n| if n == 1 { s.add(v) } else { s.add_n(v, n) },
        |a, b| a.merge(b),
    )
}

/// Histogram of the selected values with `num_bins` bins over `[min, max]`
/// (`tools::histogram`).
pub fn histogram<T: ToolValue>(
    tree: &Tree<T>,
    kind: ValueIterKind,
    min: f64,
    max: f64,
    num_bins: usize,
    threaded: bool,
) -> ToolResult<Histogram> {
    let h = Histogram::new(min, max, num_bins)?;
    Ok(accumulate_values(
        tree,
        kind,
        threaded,
        h,
        |h, v, n| {
            h.add_n(v, n);
        },
        |a, b| {
            a.merge(b);
        },
    ))
}

/// Statistics of an operator evaluated at the selected values (`tools::opStatistics`):
/// `op(accessor, ijk)` is evaluated at every voxel; for a tile, OpenVDB evaluates the
/// operator at the tile's origin and adds the result once per covered voxel.
pub fn op_statistics<T: ToolValue>(
    tree: &Tree<T>,
    kind: ValueIterKind,
    op: impl Fn(&ValueAccessor<'_, Internal2<T>>, &Coord) -> f64 + Sync,
    threaded: bool,
) -> Stats {
    op_accumulate(tree, kind, threaded, Stats::new(), &op, |s, v, n| s.add_n(v, n), |a, b| a.merge(b))
}

/// Extrema of an operator evaluated at the selected values (`tools::opExtrema`).
pub fn op_extrema<T: ToolValue>(
    tree: &Tree<T>,
    kind: ValueIterKind,
    op: impl Fn(&ValueAccessor<'_, Internal2<T>>, &Coord) -> f64 + Sync,
    threaded: bool,
) -> Extrema {
    op_accumulate(tree, kind, threaded, Extrema::new(), &op, |s, v, n| s.add_n(v, n), |a, b| a.merge(b))
}

fn op_accumulate<T: ToolValue, A: Send + Sync + Clone>(
    tree: &Tree<T>,
    kind: ValueIterKind,
    threaded: bool,
    init: A,
    op: &(impl Fn(&ValueAccessor<'_, Internal2<T>>, &Coord) -> f64 + Sync),
    add: impl Fn(&mut A, f64, u64) + Sync,
    join: impl Fn(&mut A, &A) + Sync,
) -> A {
    let mut acc = init.clone();
    {
        let a = tree.accessor();
        tree.visit_tiles(&mut |bbox, _, on, _| {
            if kind.accepts(on) {
                add(&mut acc, op(&a, &bbox.min), bbox.volume());
            }
        });
    }
    let leaf_op = |leaf: &&LeafNode<T>| {
        let a = tree.accessor();
        let mut s = init.clone();
        for n in 0..LeafNode::<T>::SIZE {
            if kind.accepts(leaf.is_value_on_at(n)) {
                add(&mut s, op(&a, &leaf.offset_to_global_coord(n)), 1);
            }
        }
        s
    };
    let leaves = tree.leaf_refs();
    let parts: Vec<A> =
        if threaded { leaves.par_iter().map(leaf_op).collect() } else { leaves.iter().map(leaf_op).collect() };
    for p in &parts {
        join(&mut acc, p);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::CoordBBox;
    use crate::tree::FloatTree;

    #[test]
    fn stats_and_histogram() {
        let mut s = Stats::new();
        for v in [1.0, 2.0, 3.0, 4.0] {
            s.add(v);
        }
        assert_eq!(s.mean(), 2.5);
        assert!((s.variance() - 1.25).abs() < 1e-12);
        let mut t = Stats::new();
        t.add_n(2.0, 3);
        t.add(6.0);
        assert_eq!(t.mean(), 3.0);
        assert!((t.variance() - 3.0).abs() < 1e-12);

        let mut tree = FloatTree::new(0.0);
        tree.fill(&CoordBBox::from_coords(0, 0, 0, 7, 7, 7), 1.0, true);
        tree.set_value(&Coord::new(100, 0, 0), 3.0);
        let st = statistics(&tree, ValueIterKind::On, true);
        assert_eq!(st.size(), 513);
        assert_eq!(st.max(), 3.0);
        assert!((st.mean() - (512.0 + 3.0) / 513.0).abs() < 1e-12);
        let ex = extrema(&tree, ValueIterKind::On, false);
        assert_eq!((ex.min(), ex.max()), (1.0, 3.0));
        let h = histogram(&tree, ValueIterKind::On, 0.0, 4.0, 4, true).unwrap();
        assert_eq!((h.count(1), h.count(3)), (512, 1));
        assert!(Histogram::new(1.0, 0.0, 3).is_err());
        let os = op_statistics(&tree, ValueIterKind::On, |acc, c| acc.get_value(c) as f64 * 2.0, true);
        assert_eq!(os.max(), 6.0);
    }
}
