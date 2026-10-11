//! Running statistics (port of `openvdb/math/Stats.h`): [`MinMax`], [`Extrema`], [`Stats`] and
//! [`Histogram`].

use super::math_util::is_approx_equal;
use num_traits::Bounded;
use std::fmt;

/// Tracks the minimum and maximum of a sequence of values of any ordered type (`math::MinMax`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMax<T> {
    /// The minimum so far.
    pub min: T,
    /// The maximum so far.
    pub max: T,
}

impl<T: Copy + PartialOrd + Bounded> Default for MinMax<T> {
    /// `min = T::MAX`, `max = T::MIN` (`std::numeric_limits<T>::lowest()`).
    fn default() -> Self {
        MinMax { min: T::max_value(), max: T::min_value() }
    }
}

impl<T: Copy> MinMax<T> {
    /// Start from explicit extrema.
    pub fn with_bounds(min: T, max: T) -> Self {
        MinMax { min, max }
    }
    /// Add a value using a custom strict-weak "less than" ordering.
    pub fn add_by(&mut self, val: T, less: impl Fn(&T, &T) -> bool) {
        if less(&val, &self.min) {
            self.min = val;
        }
        if less(&self.max, &val) {
            self.max = val;
        }
    }
    /// The minimum.
    pub fn min(&self) -> T {
        self.min
    }
    /// The maximum.
    pub fn max(&self) -> T {
        self.max
    }
}

impl<T: Copy + PartialOrd> MinMax<T> {
    /// Add a value.
    pub fn add(&mut self, val: T) {
        if val < self.min {
            self.min = val;
        }
        if self.max < val {
            self.max = val;
        }
    }
    /// Merge with another instance.
    pub fn add_min_max(&mut self, other: &MinMax<T>) {
        if other.min < self.min {
            self.min = other.min;
        }
        if self.max < other.max {
            self.max = other.max;
        }
    }
}

impl<T: fmt::Display> fmt::Display for MinMax<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MinMax   Min={}, Max={}", self.min, self.max)
    }
}

/// Tracks the number of samples and their minimum and maximum (`math::Extrema`).
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
    /// An empty instance.
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a sample.
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
    /// Merge with another instance.
    pub fn add_extrema(&mut self, other: &Extrema) {
        if other.size > 0 {
            self.join(other);
        }
    }
    fn join(&mut self, other: &Extrema) {
        self.size += other.size;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
    /// The number of samples.
    pub fn size(&self) -> u64 {
        self.size
    }
    /// The minimum.
    pub fn min(&self) -> f64 {
        self.min
    }
    /// The maximum.
    pub fn max(&self) -> f64 {
        self.max
    }
    /// `max - min`.
    pub fn range(&self) -> f64 {
        self.max - self.min
    }
    /// A report in the format of `Extrema::print` (precision 3).
    pub fn report(&self, name: &str) -> String {
        let mut s = String::from("Extrema ");
        if !name.is_empty() {
            s += &format!("for \"{name}\" ");
        }
        if self.size > 0 {
            s += &format!(
                "with {} samples:\n  Min={:.3}, Max={:.3}, Range={:.3}\n",
                self.size,
                self.min,
                self.max,
                self.range()
            );
        } else {
            s += ": no samples were added.\n";
        }
        s
    }
}

impl fmt::Display for Extrema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report(""))
    }
}

/// Tracks count, extrema, mean and variance of a sequence of samples using Welford's algorithm
/// (`math::Stats`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Stats {
    ext: Extrema,
    avg: f64,
    aux: f64,
}

impl Stats {
    /// An empty instance.
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a sample.
    pub fn add(&mut self, val: f64) {
        self.ext.add(val);
        let delta = val - self.avg;
        self.avg += delta / self.ext.size as f64;
        self.aux += delta * (val - self.avg);
    }
    /// Add `n` samples of the same value.
    pub fn add_n(&mut self, val: f64, n: u64) {
        let denom = 1.0 / (self.ext.size + n) as f64;
        let delta = val - self.avg;
        self.avg += denom * delta * n as f64;
        self.aux += denom * delta * delta * self.ext.size as f64 * n as f64;
        self.ext.add_n(val, n);
    }
    /// Merge with another instance.
    pub fn add_stats(&mut self, other: &Stats) {
        if other.ext.size > 0 {
            let denom = 1.0 / (self.ext.size + other.ext.size) as f64;
            let delta = other.avg - self.avg;
            self.avg += denom * delta * other.ext.size as f64;
            self.aux += other.aux + denom * delta * delta * self.ext.size as f64 * other.ext.size as f64;
            self.ext.join(&other.ext);
        }
    }
    /// The extrema.
    pub fn extrema(&self) -> &Extrema {
        &self.ext
    }
    /// The number of samples.
    pub fn size(&self) -> u64 {
        self.ext.size
    }
    /// The minimum.
    pub fn min(&self) -> f64 {
        self.ext.min
    }
    /// The maximum.
    pub fn max(&self) -> f64 {
        self.ext.max
    }
    /// `max - min`.
    pub fn range(&self) -> f64 {
        self.ext.range()
    }
    /// The arithmetic mean.
    pub fn avg(&self) -> f64 {
        self.avg
    }
    /// The arithmetic mean.
    pub fn mean(&self) -> f64 {
        self.avg
    }
    /// The population variance (0 for fewer than two samples).
    pub fn var(&self) -> f64 {
        if self.ext.size < 2 {
            0.0
        } else {
            self.aux / self.ext.size as f64
        }
    }
    /// Same as [`var`](Self::var).
    pub fn variance(&self) -> f64 {
        self.var()
    }
    /// The population standard deviation.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }
    /// Same as [`std`](Self::std).
    pub fn std_dev(&self) -> f64 {
        self.std()
    }
    /// A report in the format of `Stats::print` (precision 3).
    pub fn report(&self, name: &str) -> String {
        let mut s = String::from("Statistics ");
        if !name.is_empty() {
            s += &format!("for \"{name}\" ");
        }
        if self.ext.size > 0 {
            s += &format!(
                "with {} samples:\n  Min={:.3}, Max={:.3}, Ave={:.3}, Std={:.3}, Var={:.3}\n",
                self.ext.size,
                self.ext.min,
                self.ext.max,
                self.avg,
                self.std_dev(),
                self.variance()
            );
        } else {
            s += ": no samples were added.\n";
        }
        s
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report(""))
    }
}

/// Error constructing a [`Histogram`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HistogramError {
    /// The range is empty.
    #[error("Histogram: expected min < max")]
    EmptyRange,
    /// Zero bins were requested.
    #[error("Histogram: expected at least one bin")]
    NoBins,
}

/// A histogram of samples over a fixed range with uniform bins (`math::Histogram`).
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    size: u64,
    min: f64,
    max: f64,
    delta: f64,
    bins: Vec<u64>,
}

impl Histogram {
    /// A histogram over `[min, max]` with `num_bins` bins (OpenVDB's default is 10).
    pub fn new(min: f64, max: f64, num_bins: usize) -> Result<Self, HistogramError> {
        let hmax = max + 1e-10;
        if hmax <= min {
            return Err(HistogramError::EmptyRange);
        }
        if num_bins == 0 {
            return Err(HistogramError::NoBins);
        }
        Ok(Histogram { size: 0, min, max: hmax, delta: num_bins as f64 / (max - min), bins: vec![0; num_bins] })
    }
    /// A histogram over the range of `s`.
    pub fn from_stats(s: &Stats, num_bins: usize) -> Result<Self, HistogramError> {
        let (min, max) = (s.min(), s.max() + 1e-10);
        if max <= min {
            return Err(HistogramError::EmptyRange);
        }
        if num_bins == 0 {
            return Err(HistogramError::NoBins);
        }
        Ok(Histogram { size: 0, min, max, delta: num_bins as f64 / (max - min), bins: vec![0; num_bins] })
    }
    /// Add `n` samples of `val`; returns `false` (and adds nothing) if `val` is NaN or out of
    /// range.
    pub fn add(&mut self, val: f64, n: u64) -> bool {
        if val.is_nan() || val < self.min || val > self.max {
            return false;
        }
        let i = (self.delta * (val - self.min)) as usize;
        let last = self.bins.len() - 1;
        self.bins[i.min(last)] += n;
        self.size += n;
        true
    }
    /// Merge with a histogram with the same range and bin count; returns `false` otherwise.
    pub fn add_histogram(&mut self, other: &Histogram) -> bool {
        if !is_approx_equal(self.min, other.min)
            || !is_approx_equal(self.max, other.max)
            || self.bins.len() != other.bins.len()
        {
            return false;
        }
        for (a, b) in self.bins.iter_mut().zip(&other.bins) {
            *a += b;
        }
        self.size += other.size;
        true
    }
    /// The number of bins.
    pub fn num_bins(&self) -> usize {
        self.bins.len()
    }
    /// The lower bound of the range.
    pub fn min(&self) -> f64 {
        self.min
    }
    /// The upper bound of the range (plus 1e-10).
    pub fn max(&self) -> f64 {
        self.max
    }
    /// The lower bound of bin `n`.
    pub fn bin_min(&self, n: usize) -> f64 {
        self.min + n as f64 / self.delta
    }
    /// The upper bound of bin `n`.
    pub fn bin_max(&self, n: usize) -> f64 {
        self.min + (n + 1) as f64 / self.delta
    }
    /// The number of samples in bin `n`.
    pub fn count(&self, n: usize) -> u64 {
        self.bins[n]
    }
    /// The total number of samples.
    pub fn size(&self) -> u64 {
        self.size
    }
    /// A table in the format of `Histogram::print`.
    pub fn report(&self, name: &str) -> String {
        let mut s = String::from("\nHistogram ");
        if !name.is_empty() {
            s += &format!("for \"{name}\" ");
        }
        if self.size > 0 {
            let line = "==============================================================\n";
            s += &format!("with {} samples:\n", self.size);
            s += line;
            s += "||  #   |       Min      |       Max      | Frequency |  %  ||\n";
            s += line;
            for (i, &c) in self.bins.iter().enumerate() {
                s += &format!(
                    "|| {:4} | {:14.6} | {:14.6} | {:9} | {:3} ||\n",
                    i,
                    self.bin_min(i),
                    self.bin_max(i),
                    c,
                    100 * c / self.size
                );
            }
            s += line;
        } else {
            s += ": no samples were added.\n";
        }
        s
    }
}

impl fmt::Display for Histogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut s = Stats::new();
        for &v in &data {
            s.add(v);
        }
        assert_eq!(s.size(), 6);
        assert_eq!(s.min(), 1.0);
        assert_eq!(s.max(), 6.0);
        assert!((s.mean() - 3.5).abs() < 1e-15);
        assert!((s.var() - 35.0 / 12.0).abs() < 1e-14);
        // Merging partial statistics gives the same result.
        let (mut a, mut b) = (Stats::new(), Stats::new());
        data[..2].iter().for_each(|&v| a.add(v));
        data[2..].iter().for_each(|&v| b.add(v));
        a.add_stats(&b);
        assert!((a.var() - s.var()).abs() < 1e-14);
        assert!((a.mean() - s.mean()).abs() < 1e-14);
        let mut c = Stats::new();
        c.add_n(2.0, 3);
        c.add_n(4.0, 1);
        assert!((c.mean() - 2.5).abs() < 1e-15);
        assert!((c.var() - 0.75).abs() < 1e-15);
        assert!(s.report("x").starts_with("Statistics for \"x\" with 6 samples"));
    }

    #[test]
    fn extrema_and_minmax() {
        let mut e = Extrema::new();
        e.add(3.0);
        e.add_n(-1.0, 4);
        assert_eq!((e.size(), e.min(), e.max(), e.range()), (5, -1.0, 3.0, 4.0));
        let mut m: MinMax<i32> = MinMax::default();
        for v in [3, -2, 7] {
            m.add(v);
        }
        assert_eq!((m.min(), m.max()), (-2, 7));
        let mut m2 = MinMax::with_bounds(0, 10);
        m2.add_min_max(&m);
        assert_eq!((m2.min(), m2.max()), (-2, 10));
    }

    #[test]
    fn histogram() {
        let mut h = Histogram::new(0.0, 10.0, 5).unwrap();
        for v in [0.0, 1.0, 2.5, 9.99, 10.0] {
            assert!(h.add(v, 1));
        }
        assert!(!h.add(-1.0, 1));
        assert!(!h.add(f64::NAN, 1));
        assert_eq!(h.count(0), 2);
        assert_eq!(h.count(1), 1);
        assert_eq!(h.count(4), 2);
        assert_eq!(h.size(), 5);
        assert_eq!(h.bin_min(1), 2.0);
        let h2 = h.clone();
        assert!(h.add_histogram(&h2));
        assert_eq!(h.size(), 10);
        assert_eq!(Histogram::new(1.0, 0.0, 3), Err(HistogramError::EmptyRange));
        assert_eq!(Histogram::new(0.0, 1.0, 0), Err(HistogramError::NoBins));
        assert!(h.report("").contains("Frequency"));
    }
}
