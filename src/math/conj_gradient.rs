//! Preconditioned conjugate gradient solver for sparse symmetric positive-definite systems
//! (port of `openvdb/math/ConjGradient.h`, namespace `math::pcg`).
//!
//! * [`SparseStencilMatrix`] — a sparse matrix with at most `N` non-zeros per row, stored row
//!   by row with sorted column indices.
//! * [`JacobiPreconditioner`] and [`IncompleteCholeskyPreconditioner`] implement
//!   [`Preconditioner`].
//! * [`solve`] runs the preconditioned conjugate gradient iteration and reports a [`State`].
//!
//! Vectors are plain slices/`Vec`s; dot products use OpenVDB's deterministic binned reduction,
//! so results are reproducible and match the C++ library.

use crate::types::{RealValue, Value};
use rayon::prelude::*;

/// Termination criteria and solver results (`pcg::State`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    /// True if a termination criterion was met.
    pub success: bool,
    /// Maximum (input) or performed (output) number of iterations.
    pub iterations: usize,
    /// Relative error (infinity norm of the residual over that of `b`).
    pub relative_error: f64,
    /// Absolute error (infinity norm of the residual).
    pub absolute_error: f64,
}

/// OpenVDB's default termination criteria: 50 iterations, relative error 1e-6, absolute error
/// 100 epsilon (`pcg::terminationDefaults`).
pub fn termination_defaults<T: RealValue>() -> State {
    State { success: false, iterations: 50, relative_error: 1.0e-6, absolute_error: T::epsilon().to_f64() * 100.0 }
}

/// Dot product with OpenVDB's deterministic binned summation (`Vector::dot`).
pub fn dot<T: RealValue>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let zero = <T as Value>::zero();
    if n < 1024 {
        return a.iter().zip(b).fold(zero, |s, (x, y)| s + *x * *y);
    }
    const BINS: usize = 100;
    let bin = n / BINS;
    let partial: Vec<T> = (0..BINS)
        .into_par_iter()
        .map(|k| {
            let begin = k * bin;
            let end = if k == BINS - 1 { n } else { begin + bin };
            a[begin..end].iter().zip(&b[begin..end]).fold(zero, |s, (x, y)| s + *x * *y)
        })
        .collect();
    partial.into_iter().fold(zero, |s, x| s + x)
}

/// Infinity norm (`Vector::infNorm`).
pub fn inf_norm<T: RealValue>(a: &[T]) -> T {
    a.iter().fold(<T as Value>::zero(), |m, x| m.max(x.abs()))
}

/// Euclidean norm (`Vector::l2Norm`).
pub fn l2_norm<T: RealValue>(a: &[T]) -> T {
    dot(a, a).sqrt()
}

/// True if every element is finite.
pub fn is_finite<T: RealValue>(a: &[T]) -> bool {
    a.iter().all(|x| x.is_finite())
}

/// `out = a * x + y` (`internal::axpy`).
pub fn axpy<T: RealValue>(a: T, x: &[T], y: &[T], out: &mut [T]) {
    let one = T::from_f64(1.0);
    out.par_iter_mut().enumerate().for_each(|(n, o)| {
        *o = if a == one {
            x[n] + y[n]
        } else if a == -one {
            -x[n] + y[n]
        } else {
            a * x[n] + y[n]
        }
    });
}

fn axpy_in_place<T: RealValue>(a: T, x: &[T], y_out: &mut [T]) {
    let one = T::from_f64(1.0);
    y_out.par_iter_mut().enumerate().for_each(|(n, o)| {
        *o = if a == one {
            x[n] + *o
        } else if a == -one {
            -x[n] + *o
        } else {
            a * x[n] + *o
        }
    });
}

/// A matrix that can be applied to vectors.
pub trait LinearOperator<T>: Sync {
    /// Number of rows (and columns).
    fn num_rows(&self) -> usize;
    /// `out = A * x`.
    fn vector_multiply(&self, x: &[T], out: &mut [T]);
}

/// A sparse square matrix with at most `N` non-zeros per row (`pcg::SparseStencilMatrix`).
#[derive(Clone, Debug)]
pub struct SparseStencilMatrix<T, const N: usize> {
    rows: usize,
    vals: Vec<T>,
    cols: Vec<u32>,
    sizes: Vec<u32>,
}

/// A read-only view of a matrix row.
#[derive(Clone, Copy, Debug)]
pub struct Row<'a, T> {
    vals: &'a [T],
    cols: &'a [u32],
}

impl<'a, T: RealValue> Row<'a, T> {
    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.vals.len()
    }
    /// True if the row has no entries.
    pub fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }
    /// The value in column `col` (zero if not stored).
    pub fn get(&self, col: usize) -> T {
        match self.cols.binary_search(&(col as u32)) {
            Ok(i) => self.vals[i],
            Err(_) => <T as Value>::zero(),
        }
    }
    /// Iterate over `(column, value)` in column order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, T)> + 'a {
        self.cols.iter().zip(self.vals.iter()).map(|(c, v)| (*c as usize, *v))
    }
    /// Dot product with a dense vector.
    pub fn dot(&self, x: &[T]) -> T {
        self.iter().fold(<T as Value>::zero(), |s, (c, v)| s + v * x[c])
    }
}

impl<T: RealValue, const N: usize> SparseStencilMatrix<T, N> {
    /// An `n × n` matrix with no entries.
    pub fn new(n: usize) -> Self {
        SparseStencilMatrix {
            rows: n,
            vals: vec![<T as Value>::zero(); n * N],
            cols: vec![0; n * N],
            sizes: vec![0; n],
        }
    }
    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.rows
    }
    /// Row `r`.
    pub fn row(&self, r: usize) -> Row<'_, T> {
        let s = self.sizes[r] as usize;
        Row { vals: &self.vals[r * N..r * N + s], cols: &self.cols[r * N..r * N + s] }
    }
    /// The value at `(row, col)`.
    pub fn get_value(&self, row: usize, col: usize) -> T {
        self.row(row).get(col)
    }
    /// Set the value at `(row, col)`, inserting a new entry if needed.
    ///
    /// # Panics
    /// If the row already holds `N` other entries.
    pub fn set_value(&mut self, row: usize, col: usize, v: T) {
        let base = row * N;
        let size = self.sizes[row] as usize;
        let cols = &mut self.cols[base..base + N];
        match cols[..size].binary_search(&(col as u32)) {
            Ok(i) => self.vals[base + i] = v,
            Err(i) => {
                assert!(size < N, "SparseStencilMatrix: row {row} is full");
                for k in (i..size).rev() {
                    cols[k + 1] = cols[k];
                    self.vals[base + k + 1] = self.vals[base + k];
                }
                cols[i] = col as u32;
                self.vals[base + i] = v;
                self.sizes[row] += 1;
            }
        }
    }
    /// Remove all entries of a row.
    pub fn clear_row(&mut self, row: usize) {
        self.sizes[row] = 0;
    }
    /// Multiply every entry by `s`.
    pub fn scale(&mut self, s: T) {
        self.vals.par_iter_mut().for_each(|v| *v *= s);
    }
    /// True if every stored value is finite.
    pub fn is_finite(&self) -> bool {
        (0..self.rows).all(|r| self.row(r).iter().all(|(_, v)| v.is_finite()))
    }
    /// True if both matrices have the same entries up to `eps`.
    pub fn eq(&self, other: &Self, eps: T) -> bool {
        self.rows == other.rows
            && (0..self.rows).all(|r| {
                let (a, b) = (self.row(r), other.row(r));
                a.len() == b.len()
                    && a.iter().zip(b.iter()).all(|((ca, va), (cb, vb))| ca == cb && (va - vb).abs() <= eps)
            })
    }
}

impl<T: RealValue, const N: usize> LinearOperator<T> for SparseStencilMatrix<T, N> {
    fn num_rows(&self) -> usize {
        self.rows
    }
    fn vector_multiply(&self, x: &[T], out: &mut [T]) {
        out.par_iter_mut().enumerate().for_each(|(r, o)| *o = self.row(r).dot(x));
    }
}

/// A preconditioner `z = M⁻¹ r` (`pcg::Preconditioner`).
pub trait Preconditioner<T> {
    /// False if the preconditioner could not be built.
    fn is_valid(&self) -> bool {
        true
    }
    /// Apply the preconditioner: `z = M⁻¹ r`.
    fn apply(&mut self, r: &[T], z: &mut [T]);
}

/// Diagonal (Jacobi) preconditioner (`pcg::JacobiPreconditioner`).
#[derive(Clone, Debug)]
pub struct JacobiPreconditioner<T> {
    inv_diag: Vec<T>,
}

impl<T: RealValue> JacobiPreconditioner<T> {
    /// Build from the diagonal of `a`.
    pub fn new<const N: usize>(a: &SparseStencilMatrix<T, N>) -> Self {
        JacobiPreconditioner {
            inv_diag: (0..a.num_rows()).map(|n| T::from_f64(1.0 / a.get_value(n, n).to_f64())).collect(),
        }
    }
}

impl<T: RealValue> Preconditioner<T> for JacobiPreconditioner<T> {
    fn apply(&mut self, r: &[T], z: &mut [T]) {
        z.par_iter_mut().enumerate().for_each(|(n, o)| *o = self.inv_diag[n] * r[n]);
    }
}

/// Incomplete Cholesky factorization preconditioner with the sparsity of the lower triangle
/// of the matrix (`pcg::IncompleteCholeskyPreconditioner`, at most 4 entries per triangular
/// row as in OpenVDB).
#[derive(Clone, Debug)]
pub struct IncompleteCholeskyPreconditioner<T> {
    lower: SparseStencilMatrix<T, 4>,
    upper: SparseStencilMatrix<T, 4>,
    tmp: Vec<T>,
    valid: bool,
}

impl<T: RealValue> IncompleteCholeskyPreconditioner<T> {
    /// Factorize `matrix`; check [`Preconditioner::is_valid`] afterwards.
    pub fn new<const N: usize>(matrix: &SparseStencilMatrix<T, N>) -> Self {
        let n = matrix.num_rows();
        let mut lower = SparseStencilMatrix::<T, 4>::new(n);
        let mut upper = SparseStencilMatrix::<T, 4>::new(n);
        for r in 0..n {
            for (c, v) in matrix.row(r).iter() {
                if c <= r {
                    lower.set_value(r, c, v);
                }
            }
        }
        let mut valid = true;
        for k in 0..n {
            let mut diag = lower.get_value(k, k);
            if diag < T::from_f64(1.0e-5) {
                valid = false;
                break;
            }
            diag = diag.sqrt();
            lower.set_value(k, k, diag);
            let src = matrix.row(k);
            for (ii, v) in src.iter() {
                if ii < k + 1 {
                    continue;
                }
                lower.set_value(ii, k, v / diag);
            }
            for (j, _) in src.iter() {
                if j < k + 1 {
                    continue;
                }
                let a_jk = lower.get_value(j, k);
                for (i, _) in matrix.row(j).iter() {
                    if i < j {
                        continue;
                    }
                    let mut a_ij = lower.get_value(i, j);
                    let a_ik = lower.get_value(i, k);
                    a_ij -= a_ik * a_jk;
                    lower.set_value(i, j, a_ij);
                }
            }
        }
        for r in 0..n {
            for (c, _) in matrix.row(r).iter() {
                if c < r {
                    continue;
                }
                upper.set_value(r, c, lower.get_value(c, r));
            }
        }
        IncompleteCholeskyPreconditioner { lower, upper, tmp: vec![<T as Value>::zero(); n], valid }
    }
    /// The lower-triangular factor.
    pub fn lower_matrix(&self) -> &SparseStencilMatrix<T, 4> {
        &self.lower
    }
    /// The upper-triangular factor (transpose of the lower one).
    pub fn upper_matrix(&self) -> &SparseStencilMatrix<T, 4> {
        &self.upper
    }
}

impl<T: RealValue> Preconditioner<T> for IncompleteCholeskyPreconditioner<T> {
    fn is_valid(&self) -> bool {
        self.valid
    }
    #[allow(clippy::needless_range_loop)]
    fn apply(&mut self, r: &[T], z: &mut [T]) {
        assert!(self.valid, "invalid Cholesky decomposition");
        let size = self.lower.num_rows();
        z.iter_mut().for_each(|v| *v = <T as Value>::zero());
        if size == 0 {
            return;
        }
        self.tmp.iter_mut().for_each(|v| *v = <T as Value>::zero());
        for i in 0..size {
            let row = self.lower.row(i);
            let d = row.get(i);
            let dot = row.dot(&self.tmp);
            self.tmp[i] = (r[i] - dot) / d;
        }
        for ii in 0..size {
            let i = size - 1 - ii;
            let row = self.upper.row(i);
            let d = row.get(i);
            let dot = row.dot(z);
            z[i] = (self.tmp[i] - dot) / d;
        }
    }
}

/// `r = b - A x` (`internal::computeResidual`).
fn compute_residual<T: RealValue, A: LinearOperator<T>>(a: &A, x: &[T], b: &[T], r: &mut [T]) {
    a.vector_multiply(x, r);
    let tmp = r.to_vec();
    axpy(-T::from_f64(1.0), &tmp, b, r);
}

/// Solve `A x = b` for symmetric positive-definite `A` by preconditioned conjugate gradients,
/// starting from the initial guess in `x` (`pcg::solve`).
///
/// # Panics
/// If the sizes of `A`, `b` and `x` disagree.
pub fn solve<T, A>(a: &A, b: &[T], x: &mut [T], precond: &mut dyn Preconditioner<T>, termination: &State) -> State
where
    T: RealValue,
    A: LinearOperator<T>,
{
    let mut result = State { success: false, iterations: 0, relative_error: 0.0, absolute_error: 0.0 };
    let size = a.num_rows();
    if size == 0 {
        return result;
    }
    assert_eq!(size, b.len(), "A and b have incompatible sizes");
    assert_eq!(size, x.len(), "A and x have incompatible sizes");
    let zero = <T as Value>::zero();
    let mut z = vec![zero; size];
    let mut p = vec![zero; size];
    let mut q = vec![zero; size];
    let tmp = inf_norm(b);
    let inf_norm_b = if tmp == zero { T::from_f64(1.0) } else { tmp };
    let mut r = vec![zero; size];
    compute_residual(a, x, b, &mut r);
    result.absolute_error = inf_norm(&r).to_f64();
    result.relative_error = (T::from_f64(result.absolute_error) / inf_norm_b).to_f64();
    if result.relative_error <= termination.relative_error {
        result.success = true;
        return result;
    }
    let mut r_dot_z_prev = T::from_f64(1.0);
    let mut min_l2 = T::max_value();
    for iteration in 0..termination.iterations {
        result.iterations = iteration + 1;
        precond.apply(&r, &mut z);
        let r_dot_z = dot(&r, &z);
        if iteration == 0 {
            p.copy_from_slice(&z);
        } else {
            let beta = r_dot_z / r_dot_z_prev;
            let pc = p.clone();
            axpy(beta, &pc, &z, &mut p);
        }
        a.vector_multiply(&p, &mut q);
        let p_ap = dot(&p, &q);
        let alpha = r_dot_z / p_ap;
        r_dot_z_prev = r_dot_z;
        axpy_in_place(alpha, &p, x);
        axpy_in_place(-alpha, &q, &mut r);
        let l2 = l2_norm(&r);
        min_l2 = min_l2.min(l2);
        let abs_err = inf_norm(&r);
        result.absolute_error = abs_err.to_f64();
        result.relative_error = (abs_err / inf_norm_b).to_f64();
        if l2 > T::from_f64(2.0) * min_l2 {
            result.success = false;
            break;
        }
        if !result.absolute_error.is_finite() {
            result.success = false;
            break;
        }
        if result.absolute_error <= termination.absolute_error || result.relative_error <= termination.relative_error {
            result.success = true;
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laplacian_1d(n: usize) -> SparseStencilMatrix<f64, 7> {
        let mut m = SparseStencilMatrix::new(n);
        for i in 0..n {
            m.set_value(i, i, 2.0);
            if i > 0 {
                m.set_value(i, i - 1, -1.0);
            }
            if i + 1 < n {
                m.set_value(i, i + 1, -1.0);
            }
        }
        m
    }

    #[test]
    fn matrix_basics() {
        let mut m = SparseStencilMatrix::<f32, 3>::new(3);
        m.set_value(0, 2, 3.0);
        m.set_value(0, 0, 1.0);
        m.set_value(0, 1, 2.0);
        assert_eq!(m.row(0).iter().collect::<Vec<_>>(), vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        assert_eq!(m.get_value(0, 1), 2.0);
        assert_eq!(m.get_value(1, 1), 0.0);
        let mut out = vec![0.0; 3];
        m.vector_multiply(&[1.0, 1.0, 1.0], &mut out);
        assert_eq!(out, vec![6.0, 0.0, 0.0]);
    }

    #[test]
    fn solves_tridiagonal_system() {
        let n = 2000;
        let a = laplacian_1d(n);
        let exact: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let mut b = vec![0.0; n];
        a.vector_multiply(&exact, &mut b);
        let mut term = termination_defaults::<f64>();
        term.iterations = 5000;
        term.relative_error = 1e-12;
        let mut x = vec![0.0; n];
        let mut ic = IncompleteCholeskyPreconditioner::new(&a);
        assert!(ic.is_valid());
        let st = solve(&a, &b, &mut x, &mut ic, &term);
        assert!(st.success, "{st:?}");
        // IC(0) of a tridiagonal matrix is exact: one iteration suffices.
        assert_eq!(st.iterations, 1);
        assert!(x.iter().zip(&exact).all(|(a, b)| (a - b).abs() < 1e-8));
        // Jacobi-preconditioned CG on a smaller system. OpenVDB's solver aborts when the residual
        // grows past twice its minimum; the C++ library stops at the same iteration with the same
        // error (reference values from `scratchpad/cppref/cg.cpp`).
        let n = 100;
        let a = laplacian_1d(n);
        let exact: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).cos()).collect();
        let mut b = vec![0.0; n];
        a.vector_multiply(&exact, &mut b);
        let mut x = vec![0.0; n];
        let mut jac = JacobiPreconditioner::new(&a);
        let st = solve(&a, &b, &mut x, &mut jac, &term);
        assert!(!st.success);
        assert_eq!(st.iterations, 52);
        assert!((st.relative_error - 0.00481911489036).abs() < 1e-12, "{st:?}");
        assert!((x[5] - 0.871964773583).abs() < 1e-11);
        let mut x = vec![0.0; n];
        let mut ic = IncompleteCholeskyPreconditioner::new(&a);
        let st = solve(&a, &b, &mut x, &mut ic, &term);
        assert!(st.success && st.iterations == 1);
        assert!((x[5] - 0.87758256189).abs() < 1e-10);
    }
}
