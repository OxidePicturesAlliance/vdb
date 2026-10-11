//! Solve the Poisson equation `∇²x = b` on the active voxels of a tree (port of
//! `openvdb/tools/PoissonSolver.h`, namespace `tools::poisson`).
//!
//! The active voxels are numbered by an index tree, the index-space Laplacian is assembled
//! into a [`LaplacianMatrix`] (with user-defined boundary conditions where neighbours fall
//! outside the domain), and the system is solved with the preconditioned conjugate gradient
//! solver of [`crate::math::conj_gradient`].

use super::internal::{erode_face, topology_copy};
use crate::math::conj_gradient::{
    solve as pcg_solve, IncompleteCholeskyPreconditioner, JacobiPreconditioner, Preconditioner, SparseStencilMatrix,
    State,
};
use crate::math::Coord;
use crate::tree::{LeafNode, Node, Tree};
use crate::types::{RealValue, Value};
use rayon::prelude::*;

/// Index type of the degrees of freedom (`VIndex`).
pub type VIndex = i32;
/// The Laplacian matrix type: at most seven non-zeros per row (`LaplacianMatrix`).
pub type LaplacianMatrix = SparseStencilMatrix<f64, 7>;

/// Boundary conditions applied where a voxel's neighbour lies outside the solution domain: the
/// operator may modify the voxel's source value and the diagonal of its matrix row.
pub trait BoundaryOp: Sync {
    /// Called for voxel `ijk` whose neighbour `neighbor` is outside the domain.
    fn apply(&self, ijk: &Coord, neighbor: &Coord, source: &mut f64, diagonal: &mut f64);
}

/// Homogeneous Dirichlet boundary conditions (`x = 0` outside the domain;
/// `DirichletBoundaryOp`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DirichletBoundaryOp;

impl BoundaryOp for DirichletBoundaryOp {
    fn apply(&self, _: &Coord, _: &Coord, _: &mut f64, diagonal: &mut f64) {
        *diagonal -= 1.0;
    }
}

impl<F: Fn(&Coord, &Coord, &mut f64, &mut f64) + Sync> BoundaryOp for F {
    fn apply(&self, ijk: &Coord, neighbor: &Coord, source: &mut f64, diagonal: &mut f64) {
        self(ijk, neighbor, source, diagonal)
    }
}

/// The preconditioner used by [`solve_with_boundary_conditions_and_preconditioner`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PreconditionerKind {
    /// Incomplete Cholesky (falls back to Jacobi if the factorization fails).
    #[default]
    IncompleteCholesky,
    /// Jacobi (diagonal).
    Jacobi,
}

/// Number the active voxels of `tree` in leaf order (`populateIndexTree`).
pub fn populate_index_tree(tree: &mut Tree<VIndex>) {
    let mut idx: VIndex = 0;
    for leaf in tree.leaves_mut() {
        leaf.for_each_on_mut(|_, v| {
            *v = idx;
            idx += 1;
        });
    }
}

/// A tree with the active topology of `tree` (tiles voxelized) whose active voxels hold their
/// degree-of-freedom index and whose background is -1 (`createIndexTree`).
pub fn create_index_tree<T: Value>(tree: &Tree<T>) -> Tree<VIndex> {
    let mut out = topology_copy(tree, -1);
    out.voxelize_active_tiles();
    populate_index_tree(&mut out);
    out
}

/// The values of `tree` at the degrees of freedom of `index` (`createVectorFromTree`).
pub fn create_vector_from_tree<T: RealValue>(tree: &Tree<T>, index: &Tree<VIndex>) -> Vec<f64> {
    let mut out = vec![0.0; index.active_voxel_count() as usize];
    for il in index.leaves() {
        match tree.probe_leaf(&il.origin()) {
            Some(l) => {
                for (n, &i) in il.iter_on() {
                    out[i as usize] = RealValue::to_f64(*l.get_value_at(n));
                }
            }
            None => {
                let v = tree.get_value(&il.origin()).to_f64();
                for (_, &i) in il.iter_on() {
                    out[i as usize] = v;
                }
            }
        }
    }
    out
}

/// A tree with the topology of `index` holding the entries of `values` (`createTreeFromVector`).
pub fn create_tree_from_vector<T: RealValue>(values: &[f64], index: &Tree<VIndex>, background: T) -> Tree<T> {
    let mut out = topology_copy(index, background);
    let origins: Vec<Coord> = index.leaves().map(|l| l.origin()).collect();
    let index_leaves = index.leaf_refs();
    let leaves = out.leaf_refs_mut();
    debug_assert_eq!(leaves.len(), origins.len());
    leaves.into_par_iter().zip(index_leaves.into_par_iter()).for_each(|(l, il)| {
        for (n, &i) in il.iter_on() {
            l.set_value_only_at(n, T::from_f64(values[i as usize]));
        }
    });
    out
}

/// The index-space Laplacian with homogeneous Dirichlet boundary conditions
/// (`createISLaplacian`).
pub fn create_is_laplacian(index: &Tree<VIndex>, interior: &Tree<bool>, staggered: bool) -> LaplacianMatrix {
    let mut unused = vec![0.0; index.active_voxel_count() as usize];
    create_is_laplacian_with_boundary_conditions(index, interior, &DirichletBoundaryOp, &mut unused, staggered)
}

/// The index-space Laplacian with boundary conditions given by `op`, which may also modify
/// the `source` vector (`createISLaplacianWithBoundaryConditions`). With `staggered`, the
/// compact seven-point stencil is used; otherwise OpenVDB's default wide stencil (neighbours
/// two voxels away) is used.
pub fn create_is_laplacian_with_boundary_conditions<B: BoundaryOp>(
    index: &Tree<VIndex>,
    interior: &Tree<bool>,
    op: &B,
    source: &mut [f64],
    staggered: bool,
) -> LaplacianMatrix {
    let n = index.active_voxel_count() as usize;
    let mut m = LaplacianMatrix::new(n);
    let src: &[f64] = source;
    type RowOut = (usize, Vec<(usize, f64)>, f64);
    let rows: Vec<Vec<RowOut>> = index
        .leaf_refs()
        .par_iter()
        .map_init(
            || (index.accessor(), interior.accessor()),
            |(vidx, inner), leaf: &&LeafNode<VIndex>| {
                let mut out = Vec::new();
                let origin = leaf.origin();
                for (nn, &row_num) in leaf.iter_on() {
                    let row = row_num as usize;
                    let ijk = origin + LeafNode::<VIndex>::offset_to_local_coord(nn);
                    let mut entries = Vec::with_capacity(7);
                    let mut s = src[row];
                    if staggered {
                        if inner.is_value_on(&ijk) {
                            for o in [[-1, 0, 0], [0, -1, 0], [0, 0, -1]] {
                                entries.push((vidx.get_value(&ijk.offset_by(o[0], o[1], o[2])) as usize, 1.0));
                            }
                            entries.push((row, -6.0));
                            for o in [[0, 0, 1], [0, 1, 0], [1, 0, 0]] {
                                entries.push((vidx.get_value(&ijk.offset_by(o[0], o[1], o[2])) as usize, 1.0));
                            }
                        } else {
                            let mut diag = 0.0;
                            for o in [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0]] {
                                let nb = ijk.offset_by(o[0], o[1], o[2]);
                                let (col, on) = vidx.probe_value(&nb);
                                if on {
                                    entries.push((col as usize, 1.0));
                                    diag -= 1.0;
                                } else {
                                    op.apply(&ijk, &nb, &mut s, &mut diag);
                                }
                            }
                            entries.push((row, diag));
                        }
                    } else {
                        let mut diag = 0.0;
                        for o in [[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0], [0, 0, -2], [0, 0, 2]] {
                            let nb = ijk.offset_by(o[0], o[1], o[2]);
                            let (col, on) = vidx.probe_value(&nb);
                            if on {
                                entries.push((col as usize, 1.0));
                                diag -= 1.0;
                            } else {
                                op.apply(&ijk, &nb, &mut s, &mut diag);
                            }
                        }
                        entries.push((row, diag));
                    }
                    out.push((row, entries, s));
                }
                out
            },
        )
        .collect();
    for leaf_rows in rows {
        for (row, entries, s) in leaf_rows {
            for (c, v) in entries {
                m.set_value(row, c, v);
            }
            source[row] = s;
        }
    }
    m
}

/// Solve `∇²x = b` with `b` given by the active values of `tree` and homogeneous Dirichlet
/// boundary conditions (`poisson::solve`). `state` holds the termination criteria on input
/// and the solver state on output.
pub fn solve<T: RealValue>(tree: &Tree<T>, state: &mut State, staggered: bool) -> Tree<T> {
    solve_with_boundary_conditions(tree, &DirichletBoundaryOp, state, staggered)
}

/// [`solve`] with custom boundary conditions (`solveWithBoundaryConditions`).
pub fn solve_with_boundary_conditions<T: RealValue, B: BoundaryOp>(
    tree: &Tree<T>,
    op: &B,
    state: &mut State,
    staggered: bool,
) -> Tree<T> {
    solve_with_boundary_conditions_and_preconditioner(
        tree,
        tree,
        op,
        state,
        staggered,
        PreconditionerKind::IncompleteCholesky,
    )
}

/// Solve on the active voxels of `domain` with source values from `tree`
/// (`solveWithBoundaryConditionsAndPreconditioner`).
pub fn solve_with_boundary_conditions_and_preconditioner<T: RealValue, D: Value, B: BoundaryOp>(
    tree: &Tree<T>,
    domain: &Tree<D>,
    op: &B,
    state: &mut State,
    staggered: bool,
    preconditioner: PreconditionerKind,
) -> Tree<T> {
    let index = create_index_tree(domain);
    let mut b = create_vector_from_tree(tree, &index);
    let mut interior = topology_copy(&index, false);
    erode_face(&mut interior, 1);
    let mut lap = create_is_laplacian_with_boundary_conditions(&index, &interior, op, &mut b, staggered);
    lap.scale(-1.0);
    b.iter_mut().for_each(|v| *v = -*v);
    let mut x = vec![0.0; b.len()];
    let mut precond: Box<dyn Preconditioner<f64>> = match preconditioner {
        PreconditionerKind::IncompleteCholesky => {
            let ic = IncompleteCholeskyPreconditioner::new(&lap);
            if ic.is_valid() {
                Box::new(ic)
            } else {
                Box::new(JacobiPreconditioner::new(&lap))
            }
        }
        PreconditionerKind::Jacobi => Box::new(JacobiPreconditioner::new(&lap)),
    };
    *state = pcg_solve(&lap, &b, &mut x, precond.as_mut(), state);
    create_tree_from_vector(&x, &index, <T as Value>::zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::conj_gradient::termination_defaults;
    use crate::tools::test_util::assert_close;

    fn source() -> Tree<f32> {
        let mut src = Tree::<f32>::new(0.0);
        for i in 0..12 {
            for j in 0..10 {
                for k in 0..9 {
                    let v = if i == 5 && j == 5 && k == 4 {
                        1.0
                    } else if (i + j + k) % 5 == 0 {
                        0.25
                    } else {
                        0.0
                    };
                    src.set_value(&Coord::new(i, j, k), v);
                }
            }
        }
        src
    }

    /// Reference values from `scratchpad/cppref/cg.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let src = source();
        let pts = [Coord::new(5, 5, 4), Coord::new(0, 0, 0), Coord::new(3, 7, 2), Coord::new(11, 9, 8)];
        let mut st = termination_defaults::<f32>();
        st.iterations = 100;
        st.relative_error = 1e-6;
        st.absolute_error = 1e-6;
        let sol = solve(&src, &mut st, false);
        assert!(st.success);
        assert_eq!(st.iterations, 8);
        assert_close(st.relative_error, 6.77961035e-07, 1e-12);
        for (p, e) in pts.iter().zip([-0.317259848, -0.0537585914, -0.0982903987, -0.0117709916]) {
            assert_close(sol.get_value(p) as f64, e, 1e-7);
        }
        let mut st = termination_defaults::<f32>();
        st.iterations = 100;
        st.relative_error = 1e-6;
        st.absolute_error = 1e-6;
        let sol = solve(&src, &mut st, true);
        assert!(st.success);
        assert_eq!(st.iterations, 12);
        for (p, e) in pts.iter().zip([-0.579108834, -0.0578700155, -0.219597235, -0.0191866383]) {
            assert_close(sol.get_value(p) as f64, e, 1e-7);
        }
    }

    #[test]
    fn laplacian_of_solution_matches_source() {
        let src = source();
        let mut st = termination_defaults::<f64>();
        st.iterations = 500;
        st.relative_error = 1e-12;
        st.absolute_error = 1e-12;
        let sol = solve(&src, &mut st, true);
        assert!(st.success);
        // Discrete Laplacian (with zero outside) reproduces the source.
        let c = Coord::new(5, 5, 4);
        let lap = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
            .iter()
            .map(|o| sol.get_value(&c.offset_by(o[0], o[1], o[2])) as f64)
            .sum::<f64>()
            - 6.0 * sol.get_value(&c) as f64;
        assert_close(lap, 1.0, 1e-5);
        let index = create_index_tree(&src);
        assert_eq!(index.active_voxel_count(), 12 * 10 * 9);
        assert_eq!(index.get_value(&Coord::new(100, 0, 0)), -1);
    }
}
