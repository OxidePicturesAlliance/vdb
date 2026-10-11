//! Histograms, extrema and statistics of grid values (port of `openvdb/tools/Statistics.h`),
//! accumulated with the [`Extrema`], [`Stats`] and [`Histogram`] of [`crate::math::stats`].
//!
//! Values are visited with a [`ValueIterKind`] (active, inactive or all values). Tiles are
//! weighted by the number of voxels they cover; vector values contribute their length.

pub use crate::math::stats::{Extrema, Histogram, Stats};
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
        |a, b| a.add_extrema(b),
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
        |a, b| a.add_stats(b),
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
    let h = Histogram::new(min, max, num_bins).map_err(|e| ToolError::Value(e.to_string()))?;
    Ok(accumulate_values(
        tree,
        kind,
        threaded,
        h,
        |h, v, n| {
            h.add(v, n);
        },
        |a, b| {
            a.add_histogram(b);
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
    op_accumulate(tree, kind, threaded, Stats::new(), &op, |s, v, n| s.add_n(v, n), |a, b| a.add_stats(b))
}

/// Extrema of an operator evaluated at the selected values (`tools::opExtrema`).
pub fn op_extrema<T: ToolValue>(
    tree: &Tree<T>,
    kind: ValueIterKind,
    op: impl Fn(&ValueAccessor<'_, Internal2<T>>, &Coord) -> f64 + Sync,
    threaded: bool,
) -> Extrema {
    op_accumulate(tree, kind, threaded, Extrema::new(), &op, |s, v, n| s.add_n(v, n), |a, b| a.add_extrema(b))
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
