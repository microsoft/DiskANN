/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Restarted block Krylov-Schur iteration for real symmetric operators.
//!
//! Operators use flat row-major blocks with shape `dimension x columns`.
//! Results contain eigenvectors in row-major `wanted x dimension` order.
//! Eigenvector `j` starts at `eigenvectors[j * dimension]`. The first row
//! corresponds to the largest returned eigenvalue. Eigenvector signs are
//! arbitrary.

use faer::linalg::solvers::{Qr, SelfAdjointEigen};
use faer::{Mat, MatRef, Side};
use std::fmt;

/// Fallible symmetric operator with a flat row-major block interface.
pub trait SymmetricOperator {
    /// Error returned by [`Self::apply`].
    type Error;
    /// Returns the square operator dimension.
    fn dim(&self) -> usize;
    /// Computes `output = A * rhs` for row-major `dim x columns` blocks.
    ///
    /// Both buffers must have length `dim() * columns`. Element `(i, j)` is at
    /// index `i * columns + j`. Implementations must represent a finite real
    /// symmetric operator.
    fn apply(&self, rhs: &[f32], columns: usize, output: &mut [f32]) -> Result<(), Self::Error>;
}

/// Parameters for one block Krylov-Schur solve.
#[derive(Clone, Copy, Debug)]
pub struct BlockKrylovSchurParams {
    /// Number of largest eigenpairs requested.
    pub wanted: usize,
    /// Krylov basis size. It must be block aligned and at least `wanted + block_size`.
    pub ncv: usize,
    /// Lanczos block size. `wanted` and `ncv` must be block aligned.
    pub block_size: usize,
    /// Maximum number of projection/restart iterations.
    pub max_iterations: usize,
    /// Absolute residual tolerance.
    pub absolute_tolerance: f32,
    /// Relative residual tolerance.
    pub relative_tolerance: f32,
    /// Deterministic initial-block seed.
    pub seed: u64,
}

/// Progress after a block operator application or restart.
#[derive(Clone, Copy, Debug)]
pub struct BlockKrylovSchurProgress {
    /// Completed block operator applications.
    pub block_operator_applications: usize,
    /// Maximum block applications for this parameter set.
    pub max_block_operator_applications: usize,
    /// One-based restart number; zero is used only during initial expansion.
    pub iteration: usize,
    /// Iteration limit.
    pub max_iterations: usize,
    /// Leading Ritz pairs passing the residual test.
    pub converged: usize,
    /// Requested pair count.
    pub wanted: usize,
    /// Largest absolute residual.
    pub max_absolute_residual: f32,
    /// Largest relative residual.
    pub max_relative_residual: f32,
}

/// Terminal solve status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKrylovSchurStatus {
    /// All requested pairs converged under the absolute-plus-relative residual test.
    Converged,
    /// The iteration limit was reached before all requested pairs converged.
    IterationLimit,
}

/// Solve result with row-major `wanted x dimension` eigenvectors.
#[derive(Debug)]
pub struct BlockKrylovSchurResult {
    /// Ritz values in descending order. Values are finite when the solve succeeds.
    pub eigenvalues: Vec<f32>,
    /// Ritz vectors in row-major `wanted x dimension` order.
    ///
    /// Row `j` is the eigenvector for `eigenvalues[j]`. Its element at input
    /// dimension `i` is `eigenvectors[j * dimension + i]`. Thus, the first
    /// `dimension` values form the eigenvector for the largest Ritz value.
    pub eigenvectors: Vec<f32>,
    /// Terminal status.
    pub status: BlockKrylovSchurStatus,
    /// Number of converged leading pairs.
    pub converged: usize,
    /// Projection iterations performed.
    pub iterations: usize,
    /// Block operator applications performed.
    pub block_operator_applications: usize,
}

/// Precise construction, operator, and numerical failure.
#[derive(Debug)]
pub enum BlockKrylovSchurError<E> {
    /// Invalid parameter or operator dimension.
    InvalidParameter(&'static str),
    /// Checked dimension arithmetic overflowed.
    DimensionOverflow {
        rows: usize,
        cols: usize,
        name: &'static str,
    },
    /// A flat buffer has an unexpected length.
    InvalidLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The user operator failed.
    Operator(E),
    /// A non-finite value was observed.
    NonFinite(&'static str),
    /// The projected self-adjoint EVD failed or produced invalid data.
    ProjectedEvd,
    /// A required full block lost numerical rank.
    NumericalBreakdown(&'static str),
}
impl<E: fmt::Display> fmt::Display for BlockKrylovSchurError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(x) => write!(f, "invalid parameter: {x}"),
            Self::DimensionOverflow { rows, cols, name } => {
                write!(f, "dimension overflow for {name}: {rows} * {cols}")
            }
            Self::InvalidLength {
                name,
                expected,
                actual,
            } => write!(
                f,
                "invalid {name} length: expected {expected}, got {actual}"
            ),
            Self::Operator(x) => write!(f, "operator error: {x}"),
            Self::NonFinite(x) => write!(f, "non-finite value in {x}"),
            Self::ProjectedEvd => write!(f, "projected EVD failed"),
            Self::NumericalBreakdown(x) => write!(f, "numerical breakdown: {x}"),
        }
    }
}
impl<E: fmt::Debug + fmt::Display> std::error::Error for BlockKrylovSchurError<E> {}

/// Errors from the built-in flat row-major operators.
#[derive(Debug)]
pub enum BuiltinOperatorError {
    /// An input or output length is invalid.
    InvalidLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A checked size computation overflowed.
    DimensionOverflow {
        rows: usize,
        cols: usize,
        name: &'static str,
    },
    /// The internal row-major SGEMM rejected a dimension.
    Sgemm(crate::SgemmError),
}
impl fmt::Display for BuiltinOperatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength {
                name,
                expected,
                actual,
            } => write!(
                f,
                "invalid {name} length: expected {expected}, got {actual}"
            ),
            Self::DimensionOverflow { rows, cols, name } => {
                write!(f, "dimension overflow for {name}: {rows} * {cols}")
            }
            Self::Sgemm(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for BuiltinOperatorError {}
fn len(rows: usize, cols: usize, name: &'static str) -> Result<usize, BuiltinOperatorError> {
    rows.checked_mul(cols)
        .ok_or(BuiltinOperatorError::DimensionOverflow { rows, cols, name })
}
fn check(rhs: &[f32], out: &[f32], dim: usize, k: usize) -> Result<(), BuiltinOperatorError> {
    let n = len(dim, k, "block")?;
    if rhs.len() != n {
        return Err(BuiltinOperatorError::InvalidLength {
            name: "rhs",
            expected: n,
            actual: rhs.len(),
        });
    }
    if out.len() != n {
        return Err(BuiltinOperatorError::InvalidLength {
            name: "output",
            expected: n,
            actual: out.len(),
        });
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn mul(
    a_t: crate::Transpose,
    b_t: crate::Transpose,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), BuiltinOperatorError> {
    crate::sgemm(a_t, b_t, m, n, k, 1.0, a, b, None, c).map_err(BuiltinOperatorError::Sgemm)
}

/// Explicit symmetric row-major matrix operator.
pub struct DenseSymmetricOperator<'a> {
    dim: usize,
    data: &'a [f32],
}
impl<'a> DenseSymmetricOperator<'a> {
    /// Creates a `dim x dim` row-major symmetric operator.
    ///
    /// The caller is responsible for ensuring that `data` is symmetric and
    /// finite. The solver checks values produced by the operator.
    pub fn new(dim: usize, data: &'a [f32]) -> Result<Self, BuiltinOperatorError> {
        let n = len(dim, dim, "matrix")?;
        if data.len() != n {
            return Err(BuiltinOperatorError::InvalidLength {
                name: "matrix",
                expected: n,
                actual: data.len(),
            });
        }
        Ok(Self { dim, data })
    }
}
impl SymmetricOperator for DenseSymmetricOperator<'_> {
    type Error = BuiltinOperatorError;
    fn dim(&self) -> usize {
        self.dim
    }
    fn apply(&self, rhs: &[f32], k: usize, out: &mut [f32]) -> Result<(), Self::Error> {
        check(rhs, out, self.dim, k)?;
        mul(
            crate::Transpose::None,
            crate::Transpose::None,
            self.dim,
            k,
            self.dim,
            self.data,
            rhs,
            out,
        )
    }
}

/// Right Gram operator for row-major `D`, computing `D^T D`.
pub struct RightGramOperator<'a> {
    rows: usize,
    cols: usize,
    data: &'a [f32],
}
impl<'a> RightGramOperator<'a> {
    /// Creates an operator for row-major `D` without materializing `D^T D`.
    ///
    /// `data` has length `rows * cols`, and element `(i, j)` is at
    /// `data[i * cols + j]`.
    pub fn new(rows: usize, cols: usize, data: &'a [f32]) -> Result<Self, BuiltinOperatorError> {
        let n = len(rows, cols, "matrix")?;
        if data.len() != n {
            return Err(BuiltinOperatorError::InvalidLength {
                name: "matrix",
                expected: n,
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }
}
impl SymmetricOperator for RightGramOperator<'_> {
    type Error = BuiltinOperatorError;
    fn dim(&self) -> usize {
        self.cols
    }
    fn apply(&self, rhs: &[f32], k: usize, out: &mut [f32]) -> Result<(), Self::Error> {
        check(rhs, out, self.cols, k)?;
        let mut temporary = vec![0.0; len(self.rows, k, "temporary")?];
        mul(
            crate::Transpose::None,
            crate::Transpose::None,
            self.rows,
            k,
            self.cols,
            self.data,
            rhs,
            &mut temporary,
        )?;
        mul(
            crate::Transpose::Ordinary,
            crate::Transpose::None,
            self.cols,
            k,
            self.rows,
            self.data,
            &temporary,
            out,
        )
    }
}

/// Left Gram operator for row-major `D`, computing `D D^T`.
pub struct LeftGramOperator<'a> {
    rows: usize,
    cols: usize,
    data: &'a [f32],
}
impl<'a> LeftGramOperator<'a> {
    /// Creates an operator for row-major `D` without materializing `D D^T`.
    ///
    /// `data` has length `rows * cols`, and element `(i, j)` is at
    /// `data[i * cols + j]`.
    pub fn new(rows: usize, cols: usize, data: &'a [f32]) -> Result<Self, BuiltinOperatorError> {
        let n = len(rows, cols, "matrix")?;
        if data.len() != n {
            return Err(BuiltinOperatorError::InvalidLength {
                name: "matrix",
                expected: n,
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }
}
impl SymmetricOperator for LeftGramOperator<'_> {
    type Error = BuiltinOperatorError;
    fn dim(&self) -> usize {
        self.rows
    }
    fn apply(&self, rhs: &[f32], k: usize, out: &mut [f32]) -> Result<(), Self::Error> {
        check(rhs, out, self.rows, k)?;
        let mut temporary = vec![0.0; len(self.cols, k, "temporary")?];
        mul(
            crate::Transpose::Ordinary,
            crate::Transpose::None,
            self.cols,
            k,
            self.rows,
            self.data,
            rhs,
            &mut temporary,
        )?;
        mul(
            crate::Transpose::None,
            crate::Transpose::None,
            self.rows,
            k,
            self.cols,
            self.data,
            &temporary,
            out,
        )
    }
}

struct Rng(u64);
impl Rng {
    fn new(x: u64) -> Self {
        Self(x | 1)
    }
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}
fn mm(a: MatRef<'_, f32>, b: MatRef<'_, f32>) -> Mat<f32> {
    a * b
}
fn diff(a: MatRef<'_, f32>, b: MatRef<'_, f32>) -> Mat<f32> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] - b[(i, j)])
}
fn finite(a: MatRef<'_, f32>, name: &'static str) -> Result<(), BlockKrylovSchurError<()>> {
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if !a[(i, j)].is_finite() {
                return Err(BlockKrylovSchurError::NonFinite(name));
            }
        }
    }
    Ok(())
}
fn sum(a: MatRef<'_, f32>, b: MatRef<'_, f32>) -> Mat<f32> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] + b[(i, j)])
}
fn hcat(a: MatRef<'_, f32>, b: MatRef<'_, f32>) -> Mat<f32> {
    Mat::from_fn(a.nrows(), a.ncols() + b.ncols(), |i, j| {
        if j < a.ncols() {
            a[(i, j)]
        } else {
            b[(i, j - a.ncols())]
        }
    })
}
fn vcat(a: MatRef<'_, f32>, b: MatRef<'_, f32>) -> Mat<f32> {
    Mat::from_fn(a.nrows() + b.nrows(), a.ncols(), |i, j| {
        if i < a.nrows() {
            a[(i, j)]
        } else {
            b[(i - a.nrows(), j)]
        }
    })
}
fn put(a: &mut Mat<f32>, r: usize, c: usize, b: MatRef<'_, f32>) {
    for i in 0..b.nrows() {
        for j in 0..b.ncols() {
            a[(r + i, c + j)] = b[(i, j)];
        }
    }
}
fn diag(x: &[f32]) -> Mat<f32> {
    Mat::from_fn(x.len(), x.len(), |i, j| if i == j { x[i] } else { 0. })
}
fn qr(a: MatRef<'_, f32>, b: usize) -> Result<(Mat<f32>, Mat<f32>), &'static str> {
    let q = Qr::new(a);
    let qq = q.compute_thin_Q();
    let r = q.thin_R().to_owned();
    let mut scale: f32 = 0.;
    for i in 0..r.nrows() {
        for j in 0..r.ncols() {
            scale = scale.max(r[(i, j)].abs());
        }
    }
    let t = 32. * f32::EPSILON * (a.nrows().max(a.ncols()) as f32) * scale;
    let mut rank = 0;
    for i in 0..a.ncols() {
        let mut norm: f32 = 0.;
        for j in 0..a.ncols() {
            norm += r[(i, j)] * r[(i, j)];
        }
        if norm.sqrt() > t {
            rank += 1
        } else {
            break;
        }
    }
    if rank != b {
        Err("full Lanczos block lost rank")
    } else {
        Ok((
            qq.as_ref().subcols(0, b).to_owned(),
            r.as_ref().subrows(0, b).to_owned(),
        ))
    }
}
fn eig(a: MatRef<'_, f32>) -> Result<(Vec<f32>, Mat<f32>), ()> {
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if !a[(i, j)].is_finite() {
                return Err(());
            }
        }
    }
    let e = SelfAdjointEigen::new(a, Side::Lower).map_err(|_| ())?;
    let s = e.S().column_vector();
    let u = e.U();
    let mut ix: Vec<_> = (0..s.nrows()).collect();
    ix.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap_or(std::cmp::Ordering::Equal));
    let values: Vec<_> = ix.iter().map(|&i| s[i]).collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err(());
    }
    Ok((
        values,
        Mat::from_fn(u.nrows(), u.ncols(), |i, j| u[(i, ix[j])]),
    ))
}
fn validate<E>(p: &BlockKrylovSchurParams, n: usize) -> Result<(), BlockKrylovSchurError<E>> {
    if n == 0 {
        return Err(BlockKrylovSchurError::InvalidParameter("dimension"));
    }
    if p.wanted == 0 || p.wanted > n {
        return Err(BlockKrylovSchurError::InvalidParameter("wanted"));
    }
    if p.block_size == 0 || p.block_size > p.wanted || !p.wanted.is_multiple_of(p.block_size) {
        return Err(BlockKrylovSchurError::InvalidParameter("block_size"));
    }
    let wanted_plus_block =
        p.wanted
            .checked_add(p.block_size)
            .ok_or(BlockKrylovSchurError::DimensionOverflow {
                rows: p.wanted,
                cols: p.block_size,
                name: "wanted + block_size",
            })?;
    if p.ncv <= p.wanted
        || p.ncv > n
        || !p.ncv.is_multiple_of(p.block_size)
        || p.ncv < wanted_plus_block
    {
        return Err(BlockKrylovSchurError::InvalidParameter("ncv"));
    }
    if p.max_iterations == 0 {
        return Err(BlockKrylovSchurError::InvalidParameter("max_iterations"));
    }
    if !p.absolute_tolerance.is_finite() || p.absolute_tolerance < 0. {
        return Err(BlockKrylovSchurError::InvalidParameter(
            "absolute_tolerance",
        ));
    }
    if !p.relative_tolerance.is_finite() || p.relative_tolerance < 0. {
        return Err(BlockKrylovSchurError::InvalidParameter(
            "relative_tolerance",
        ));
    }
    p.ncv
        .checked_mul(n)
        .ok_or(BlockKrylovSchurError::DimensionOverflow {
            rows: p.ncv,
            cols: n,
            name: "basis",
        })?;
    Ok(())
}

/// Checked upper bound on block operator applications.
pub fn block_krylov_schur_max_block_operator_applications(
    p: &BlockKrylovSchurParams,
) -> Result<usize, BlockKrylovSchurError<()>> {
    if p.block_size == 0 {
        return Err(BlockKrylovSchurError::InvalidParameter("block_size"));
    }
    if p.wanted == 0 || p.wanted > p.ncv || !p.wanted.is_multiple_of(p.block_size) {
        return Err(BlockKrylovSchurError::InvalidParameter("wanted"));
    }
    if p.block_size > p.wanted {
        return Err(BlockKrylovSchurError::InvalidParameter("block_size"));
    }
    let wanted_plus_block =
        p.wanted
            .checked_add(p.block_size)
            .ok_or(BlockKrylovSchurError::DimensionOverflow {
                rows: p.wanted,
                cols: p.block_size,
                name: "wanted + block_size",
            })?;
    if p.ncv < wanted_plus_block || !p.ncv.is_multiple_of(p.block_size) {
        return Err(BlockKrylovSchurError::InvalidParameter("ncv"));
    }
    if p.max_iterations == 0 {
        return Err(BlockKrylovSchurError::InvalidParameter("max_iterations"));
    }
    let first = 1usize.checked_add(p.ncv / p.block_size - 2).ok_or(
        BlockKrylovSchurError::DimensionOverflow {
            rows: 1,
            cols: p.ncv / p.block_size - 2,
            name: "applications",
        },
    )?;
    let per = p
        .ncv
        .checked_div(p.block_size)
        .and_then(|x| x.checked_sub(p.wanted / p.block_size))
        .and_then(|x| x.checked_sub(1))
        .ok_or(BlockKrylovSchurError::InvalidParameter("ncv"))?;
    first
        .checked_add(p.max_iterations.saturating_sub(1).checked_mul(per).ok_or(
            BlockKrylovSchurError::DimensionOverflow {
                rows: p.max_iterations,
                cols: per,
                name: "applications",
            },
        )?)
        .ok_or(BlockKrylovSchurError::DimensionOverflow {
            rows: first,
            cols: per,
            name: "applications",
        })
}

struct Solver<'a, O: SymmetricOperator + ?Sized> {
    op: &'a O,
    n: usize,
    p: BlockKrylovSchurParams,
    v: Mat<f32>,
    h: Mat<f32>,
    rng: Rng,
    apps: usize,
    iters: usize,
    converged: usize,
    locked: usize,
    abs: f32,
    rel: f32,
}

fn block_aligned_locked_prefix(converged: usize, wanted: usize, block_size: usize) -> usize {
    if converged == wanted {
        wanted
    } else {
        converged / block_size * block_size
    }
}

impl<'a, O: SymmetricOperator + ?Sized> Solver<'a, O> {
    fn apply<F>(
        &mut self,
        x: MatRef<'_, f32>,
        cb: &mut F,
    ) -> Result<Mat<f32>, BlockKrylovSchurError<O::Error>>
    where
        F: FnMut(BlockKrylovSchurProgress),
    {
        let len =
            x.nrows()
                .checked_mul(x.ncols())
                .ok_or(BlockKrylovSchurError::DimensionOverflow {
                    rows: x.nrows(),
                    cols: x.ncols(),
                    name: "operator block",
                })?;
        let mut a = vec![0.; len];
        let mut y = vec![0.; len];
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                a[i * x.ncols() + j] = x[(i, j)]
            }
        }
        self.op
            .apply(&a, x.ncols(), &mut y)
            .map_err(BlockKrylovSchurError::Operator)?;
        if y.iter().any(|z| !z.is_finite()) {
            return Err(BlockKrylovSchurError::NonFinite("operator output"));
        }
        self.apps = self
            .apps
            .checked_add(1)
            .ok_or(BlockKrylovSchurError::DimensionOverflow {
                rows: self.apps,
                cols: 1,
                name: "applications",
            })?;
        cb(BlockKrylovSchurProgress {
            block_operator_applications: self.apps,
            max_block_operator_applications: block_krylov_schur_max_block_operator_applications(
                &self.p,
            )
            .unwrap_or(usize::MAX),
            iteration: self.iters,
            max_iterations: self.p.max_iterations,
            converged: self.converged,
            wanted: self.p.wanted,
            max_absolute_residual: self.abs,
            max_relative_residual: self.rel,
        });
        Ok(Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
            y[i * x.ncols() + j]
        }))
    }
    fn init<F>(&mut self, cb: &mut F) -> Result<(), BlockKrylovSchurError<O::Error>>
    where
        F: FnMut(BlockKrylovSchurProgress),
    {
        let b = self.p.block_size;
        let x = Mat::from_fn(self.n, b, |_, _| self.rng.next());
        let (v0, _) = qr(x.as_ref(), b).map_err(BlockKrylovSchurError::NumericalBreakdown)?;
        let y = self.apply(v0.as_ref(), cb)?;
        let mut h = mm(v0.as_ref().transpose(), y.as_ref());
        let mut f = diff(y.as_ref(), mm(v0.as_ref(), h.as_ref()).as_ref());
        let c = mm(v0.as_ref().transpose(), f.as_ref());
        h = sum(h.as_ref(), c.as_ref());
        finite(h.as_ref(), "projected matrix")
            .map_err(|_| BlockKrylovSchurError::NonFinite("projected matrix"))?;
        f = diff(f.as_ref(), mm(v0.as_ref(), c.as_ref()).as_ref());
        let (q, r) = qr(f.as_ref(), b).map_err(BlockKrylovSchurError::NumericalBreakdown)?;
        self.h = vcat(h.as_ref(), r.as_ref());
        self.v = hcat(v0.as_ref(), q.as_ref());
        Ok(())
    }
    fn expand<F>(&mut self, cb: &mut F) -> Result<(), BlockKrylovSchurError<O::Error>>
    where
        F: FnMut(BlockKrylovSchurProgress),
    {
        let b = self.p.block_size;
        while self.h.nrows() < self.p.ncv {
            let r = self.h.nrows();
            let c = self.h.ncols();
            let x = self.v.as_ref().subcols(c, r - c).to_owned();
            let y = self.apply(x.as_ref(), cb)?;
            let old = self.v.as_ref().subcols(0, r);
            let mut hk = mm(old.transpose(), y.as_ref());
            let mut f = diff(y.as_ref(), mm(old.as_ref(), hk.as_ref()).as_ref());
            for _ in 0..2 {
                let z = mm(old.transpose(), f.as_ref());
                f = diff(f.as_ref(), mm(old.as_ref(), z.as_ref()).as_ref());
                hk = sum(hk.as_ref(), z.as_ref());
            }
            let z = hcat(self.h.as_ref(), hk.as_ref());
            let next_cols = c
                .checked_add(b)
                .ok_or(BlockKrylovSchurError::DimensionOverflow {
                    rows: c,
                    cols: b,
                    name: "projected matrix columns",
                })?;
            self.h = vcat(z.as_ref(), Mat::zeros(b, next_cols).as_ref());
            finite(self.h.as_ref(), "projected matrix")
                .map_err(|_| BlockKrylovSchurError::NonFinite("projected matrix"))?;
            let (q, rr) = qr(f.as_ref(), b).map_err(BlockKrylovSchurError::NumericalBreakdown)?;
            let oldc = self.v.ncols();
            let newc = oldc
                .checked_add(b)
                .ok_or(BlockKrylovSchurError::DimensionOverflow {
                    rows: oldc,
                    cols: b,
                    name: "basis columns",
                })?;
            self.v.resize_with(self.n, newc, |_, _| 0.);
            self.v.as_mut().subcols_mut(oldc, b).copy_from(q.as_ref());
            let hr = self.h.nrows();
            let hc = self.h.ncols();
            put(&mut self.h, hr - b, hc - b, rr.as_ref());
        }
        Ok(())
    }
    fn truncate(&mut self) -> Result<(), BlockKrylovSchurError<O::Error>> {
        let b = self.p.block_size;
        let w = self.p.wanted;
        let c = self.h.ncols();
        debug_assert!(self.locked.is_multiple_of(b));
        debug_assert!(self.locked <= w);
        let subm = self
            .h
            .as_ref()
            .submatrix(self.locked, self.locked, c - self.locked, c - self.locked)
            .to_owned();
        finite(subm.as_ref(), "projected matrix")
            .map_err(|_| BlockKrylovSchurError::NonFinite("projected matrix"))?;
        let (e, u) = eig(subm.as_ref()).map_err(|_| BlockKrylovSchurError::ProjectedEvd)?;
        let start = self.v.as_ref().subcols(self.v.ncols() - b, b).to_owned();
        let keep = self.v.as_ref().subcols(0, self.locked).to_owned();
        let mid = self
            .v
            .as_ref()
            .subcols(self.locked, self.v.ncols() - b - self.locked)
            .to_owned();
        let proj = mm(mid.as_ref(), u.as_ref().subcols(0, w - self.locked));
        self.v = hcat(hcat(keep.as_ref(), proj.as_ref()).as_ref(), start.as_ref());
        let bottom = self
            .h
            .as_ref()
            .submatrix(self.h.nrows() - b, c - b, b, b)
            .to_owned();
        let coup = mm(
            bottom.as_ref(),
            u.as_ref()
                .submatrix(u.nrows() - b, 0, b, w - self.locked)
                .to_owned()
                .as_ref(),
        );
        let mut next = self.h.clone();
        if self.locked > 0 {
            let top = self
                .h
                .as_ref()
                .submatrix(0, self.locked, self.locked, c - self.locked)
                .to_owned();
            let rotated = mm(top.as_ref(), u.as_ref());
            put(&mut next, 0, self.locked, rotated.as_ref());
        }
        put(
            &mut next,
            self.locked,
            self.locked,
            diag(&e[..w - self.locked]).as_ref(),
        );
        put(&mut next, w, self.locked, coup.as_ref());
        self.h = next.submatrix(0, 0, w + b, w).to_owned();
        Ok(())
    }
}

/// Computes the largest eigenpairs with restarted block Krylov-Schur.
///
/// The operator is assumed to be real and symmetric. Operator inputs and
/// outputs are row-major `dimension x block_columns` matrices. The returned
/// eigenvectors are a row-major `wanted x dimension` matrix. Result row `j`
/// corresponds to `eigenvalues[j]`, which are in descending order.
///
/// The solver uses the residual criterion
/// `absolute_tolerance + relative_tolerance * |Ritz value|`, and returns
/// [`BlockKrylovSchurStatus::IterationLimit`] when that criterion is not met
/// for all requested pairs before `max_iterations` restarts. In that case the
/// returned leading Ritz pairs are still finite and orthonormal to solver
/// precision.
pub fn block_krylov_schur_eigenpairs_with_progress<O, F>(
    op: &O,
    p: &BlockKrylovSchurParams,
    mut cb: F,
) -> Result<BlockKrylovSchurResult, BlockKrylovSchurError<O::Error>>
where
    O: SymmetricOperator + ?Sized,
    F: FnMut(BlockKrylovSchurProgress),
{
    let n = op.dim();
    validate(p, n)?;
    let max = block_krylov_schur_max_block_operator_applications(p)
        .map_err(|_| BlockKrylovSchurError::InvalidParameter("applications"))?;
    let mut s = Solver {
        op,
        n,
        p: *p,
        v: Mat::zeros(0, 0),
        h: Mat::zeros(0, 0),
        rng: Rng::new(p.seed),
        apps: 0,
        iters: 0,
        converged: 0,
        locked: 0,
        abs: f32::INFINITY,
        rel: f32::INFINITY,
    };
    s.init(&mut cb)?;
    s.expand(&mut cb)?;
    let mut status = BlockKrylovSchurStatus::IterationLimit;
    for it in 1..=p.max_iterations {
        s.truncate()?;
        s.iters = it;
        let hr = s.h.nrows();
        let mut first = p.wanted;
        s.abs = 0.;
        s.rel = 0.;
        for j in 0..p.wanted {
            let r = (0..p.block_size)
                .map(|i| s.h[(hr - p.block_size + i, j)].powi(2))
                .sum::<f32>()
                .sqrt();
            let ev = s.h[(j, j)].abs();
            if !ev.is_finite() || !r.is_finite() {
                return Err(BlockKrylovSchurError::NonFinite("Ritz values or residuals"));
            }
            s.abs = s.abs.max(r);
            s.rel = s.rel.max(r / ev.max(f32::MIN_POSITIVE));
            if r > p.absolute_tolerance + p.relative_tolerance * ev && first == p.wanted {
                first = j
            }
        }
        s.converged = first;
        s.locked = block_aligned_locked_prefix(first, p.wanted, p.block_size);
        cb(BlockKrylovSchurProgress {
            block_operator_applications: s.apps,
            max_block_operator_applications: max,
            iteration: it,
            max_iterations: p.max_iterations,
            converged: first,
            wanted: p.wanted,
            max_absolute_residual: s.abs,
            max_relative_residual: s.rel,
        });
        if first == p.wanted {
            status = BlockKrylovSchurStatus::Converged;
            break;
        }
        if it < p.max_iterations {
            s.expand(&mut cb)?
        }
    }
    let output_len = n
        .checked_mul(p.wanted)
        .ok_or(BlockKrylovSchurError::DimensionOverflow {
            rows: n,
            cols: p.wanted,
            name: "eigenvectors",
        })?;
    let mut out = Vec::with_capacity(output_len);
    for j in 0..p.wanted {
        for i in 0..n {
            if !s.v[(i, j)].is_finite() {
                return Err(BlockKrylovSchurError::NonFinite("eigenvectors"));
            }
            out.push(s.v[(i, j)])
        }
    }
    let eigenvalues: Vec<_> = (0..p.wanted).map(|i| s.h[(i, i)]).collect();
    if eigenvalues.iter().any(|x| !x.is_finite()) {
        return Err(BlockKrylovSchurError::NonFinite("final eigenvalues"));
    }
    Ok(BlockKrylovSchurResult {
        eigenvalues,
        eigenvectors: out,
        status,
        converged: s.converged,
        iterations: s.iters,
        block_operator_applications: s.apps,
    })
}

/// Computes the largest eigenpairs without progress notifications.
///
/// This is equivalent to [`block_krylov_schur_eigenpairs_with_progress`] with
/// an empty callback. See that function for layout, symmetry, convergence, and
/// iteration-limit semantics.
pub fn block_krylov_schur_eigenpairs<O: SymmetricOperator + ?Sized>(
    op: &O,
    p: &BlockKrylovSchurParams,
) -> Result<BlockKrylovSchurResult, BlockKrylovSchurError<O::Error>> {
    block_krylov_schur_eigenpairs_with_progress(op, p, |_| {})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p() -> BlockKrylovSchurParams {
        BlockKrylovSchurParams {
            wanted: 2,
            ncv: 6,
            block_size: 2,
            max_iterations: 20,
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-5,
            seed: 17,
        }
    }
    fn diag(v: &[f32]) -> Vec<f32> {
        let mut a = vec![0.; v.len() * v.len()];
        for (i, x) in v.iter().enumerate() {
            a[i * v.len() + i] = *x;
        }
        a
    }

    fn gram(data: &[f32], rows: usize, cols: usize, left: bool) -> Vec<f32> {
        let dim = if left { rows } else { cols };
        let mut out = vec![0.; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                out[i * dim + j] = if left {
                    (0..cols)
                        .map(|k| data[i * cols + k] * data[j * cols + k])
                        .sum()
                } else {
                    (0..rows)
                        .map(|k| data[k * cols + i] * data[k * cols + j])
                        .sum()
                };
            }
        }
        out
    }

    fn rectangular_diagonal(rows: usize, cols: usize) -> Vec<f32> {
        let rank = rows.min(cols);
        let mut matrix = vec![0.0; rows * cols];
        for i in 0..rank {
            matrix[i * cols + i] = (rank - i) as f32;
        }
        matrix
    }

    fn assert_rectangular_gram_solution<O>(
        operator: &O,
        params: &BlockKrylovSchurParams,
        dimension: usize,
        rank: usize,
    ) where
        O: SymmetricOperator,
        O::Error: fmt::Debug,
    {
        let result = block_krylov_schur_eigenpairs(operator, params).unwrap();
        assert_eq!(result.status, BlockKrylovSchurStatus::Converged);
        assert_eq!(result.eigenvectors.len(), params.wanted * dimension);

        let mut vectors_by_column = vec![0.0; dimension * params.wanted];
        for j in 0..params.wanted {
            let expected = ((rank - j) * (rank - j)) as f32;
            let eigenvalue_relative_error = (result.eigenvalues[j] - expected).abs() / expected;
            assert!(eigenvalue_relative_error < params.relative_tolerance);
            for i in 0..dimension {
                vectors_by_column[i * params.wanted + j] = result.eigenvectors[j * dimension + i];
            }
        }

        let mut applied = vec![0.0; dimension * params.wanted];
        operator
            .apply(&vectors_by_column, params.wanted, &mut applied)
            .unwrap();
        for j in 0..params.wanted {
            let residual = (0..dimension)
                .map(|i| {
                    let value = applied[i * params.wanted + j]
                        - result.eigenvalues[j] * vectors_by_column[i * params.wanted + j];
                    value * value
                })
                .sum::<f32>()
                .sqrt();
            assert!(residual / result.eigenvalues[j].abs() < params.relative_tolerance);
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(b).all(|(x, y)| (*x - *y).abs() <= tolerance),
            "left={a:?}, right={b:?}"
        );
    }

    struct FixedDim(usize);
    impl SymmetricOperator for FixedDim {
        type Error = ();
        fn dim(&self) -> usize {
            self.0
        }
        fn apply(&self, _: &[f32], _: usize, _: &mut [f32]) -> Result<(), ()> {
            Ok(())
        }
    }

    fn invalid(p: BlockKrylovSchurParams, expected: &'static str) {
        let op = FixedDim(8);
        assert!(matches!(
            block_krylov_schur_eigenpairs(&op, &p),
            Err(BlockKrylovSchurError::InvalidParameter(name)) if name == expected
        ));
    }

    #[test]
    fn block_solver_has_row_major_orthonormal_result() {
        let matrix = diag(&[9., 8., 7., 6., 5., 4.]);
        let op = DenseSymmetricOperator::new(6, &matrix).unwrap();
        let r = block_krylov_schur_eigenpairs(&op, &p()).unwrap();
        assert_eq!(r.eigenvectors.len(), 12);
        assert_eq!(r.status, BlockKrylovSchurStatus::Converged);
        assert!((r.eigenvalues[0] - 9.).abs() < 1e-3);
        assert!((r.eigenvalues[1] - 8.).abs() < 1e-3);
        for j in 0..2 {
            let n = (0..6)
                .map(|i| r.eigenvectors[j * 6 + i].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!((n - 1.).abs() < 1e-3);
        }
        assert!(
            (0..6)
                .map(|i| r.eigenvectors[i] * r.eigenvectors[6 + i])
                .sum::<f32>()
                .abs()
                < 1e-3
        );
    }

    #[test]
    fn solver_matches_faer_full_evd() {
        let dimension = 8;
        let wanted = 2;
        let matrix = Mat::from_fn(dimension, dimension, |i, j| {
            if i == j {
                12.0 - i as f32
            } else if i.abs_diff(j) == 1 {
                0.35
            } else {
                0.0
            }
        });
        let mut row_major = Vec::with_capacity(dimension * dimension);
        for i in 0..dimension {
            for j in 0..dimension {
                row_major.push(matrix[(i, j)]);
            }
        }
        let operator = DenseSymmetricOperator::new(dimension, &row_major).unwrap();
        let result = block_krylov_schur_eigenpairs(
            &operator,
            &BlockKrylovSchurParams {
                wanted,
                ncv: 8,
                block_size: 2,
                max_iterations: 20,
                absolute_tolerance: 1e-5,
                relative_tolerance: 1e-5,
                seed: 17,
            },
        )
        .unwrap();

        let reference = SelfAdjointEigen::new(matrix.as_ref(), Side::Lower).unwrap();
        let reference_values = reference.S().column_vector();
        let reference_vectors = reference.U();

        assert_eq!(result.status, BlockKrylovSchurStatus::Converged);
        for j in 0..wanted {
            let reference_index = dimension - 1 - j;
            assert!((result.eigenvalues[j] - reference_values[reference_index]).abs() < 2e-4);

            // Eigenvector signs are arbitrary. Compare the absolute inner product.
            let alignment = (0..dimension)
                .map(|i| {
                    result.eigenvectors[j * dimension + i] * reference_vectors[(i, reference_index)]
                })
                .sum::<f32>()
                .abs();
            assert!(alignment > 0.999);
        }
    }

    #[test]
    fn materialized_and_right_gram_solutions_agree() {
        let data = [
            1., 2., -1., 0., 3., 1., 2., -2., 1., 1., 0., 4., -1., 2., 3., 0.,
        ];
        let materialized = gram(&data, 4, 4, false);
        let dense = DenseSymmetricOperator::new(4, &materialized).unwrap();
        let right = RightGramOperator::new(4, 4, &data).unwrap();
        let mut params = p();
        params.wanted = 2;
        params.ncv = 4;
        let a = block_krylov_schur_eigenpairs(&dense, &params).unwrap();
        let b = block_krylov_schur_eigenpairs(&right, &params).unwrap();
        assert_close(&a.eigenvalues, &b.eigenvalues, 2e-3);
    }

    #[test]
    fn left_and_right_gram_operators_match_nonzero_spectrum() {
        let data = [
            1., 2., 0., 3., -1., 1., 2., 0., 2., -2., 1., 1., 0., 4., -1., 0.,
        ];
        let right = RightGramOperator::new(4, 4, &data).unwrap();
        let left = LeftGramOperator::new(4, 4, &data).unwrap();
        let right_matrix = gram(&data, 4, 4, false);
        let left_matrix = gram(&data, 4, 4, true);
        let mut right_out = vec![0.; 16];
        let mut left_out = vec![0.; 16];
        right
            .apply(&diag(&[1., 1., 1., 1.]), 4, &mut right_out)
            .unwrap();
        left.apply(&diag(&[1., 1., 1., 1.]), 4, &mut left_out)
            .unwrap();
        assert_close(&right_out, &right_matrix, 1e-5);
        assert_close(&left_out, &left_matrix, 1e-5);
        let mut params = p();
        params.wanted = 1;
        params.block_size = 1;
        params.ncv = 4;
        let right_result = block_krylov_schur_eigenpairs(&right, &params).unwrap();
        let left_result = block_krylov_schur_eigenpairs(&left, &params).unwrap();
        assert_close(&right_result.eigenvalues, &left_result.eigenvalues, 2e-3);
    }

    #[test]
    fn tall_skinny_and_short_wide_matrices_support_dimension_1024() {
        const LONG_DIMENSION: usize = 1024;
        const SHORT_DIMENSION: usize = 32;
        let tall_skinny = rectangular_diagonal(LONG_DIMENSION, SHORT_DIMENSION);
        let short_wide = rectangular_diagonal(SHORT_DIMENSION, LONG_DIMENSION);
        let params = BlockKrylovSchurParams {
            wanted: 4,
            ncv: 24,
            block_size: 4,
            max_iterations: 20,
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-3,
            seed: 17,
        };

        let tall_right =
            RightGramOperator::new(LONG_DIMENSION, SHORT_DIMENSION, &tall_skinny).unwrap();
        let tall_left =
            LeftGramOperator::new(LONG_DIMENSION, SHORT_DIMENSION, &tall_skinny).unwrap();
        let wide_right =
            RightGramOperator::new(SHORT_DIMENSION, LONG_DIMENSION, &short_wide).unwrap();
        let wide_left =
            LeftGramOperator::new(SHORT_DIMENSION, LONG_DIMENSION, &short_wide).unwrap();

        assert_rectangular_gram_solution(&tall_right, &params, SHORT_DIMENSION, SHORT_DIMENSION);
        assert_rectangular_gram_solution(&tall_left, &params, LONG_DIMENSION, SHORT_DIMENSION);
        assert_rectangular_gram_solution(&wide_right, &params, LONG_DIMENSION, SHORT_DIMENSION);
        assert_rectangular_gram_solution(&wide_left, &params, SHORT_DIMENSION, SHORT_DIMENSION);
    }

    #[test]
    fn residuals_and_row_major_indexing_are_explicitly_validated() {
        let values = [9., 8., 7., 6., 5., 4.];
        let matrix = diag(&values);
        let op = DenseSymmetricOperator::new(6, &matrix).unwrap();
        let result = block_krylov_schur_eigenpairs(&op, &p()).unwrap();
        assert_eq!(result.eigenvectors.len(), p().wanted * values.len());
        for (j, eigenvector) in result.eigenvectors.chunks_exact(values.len()).enumerate() {
            let norm = eigenvector.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
            assert!((norm - 1.).abs() < 2e-3);
            for (i, value) in values.iter().enumerate() {
                let mut residual = *value * eigenvector[i];
                residual -= result.eigenvalues[j] * eigenvector[i];
                assert!(residual.abs() < 2e-3);
            }
        }
    }

    #[test]
    fn block_size_and_clustered_spectrum_are_supported() {
        let matrix = diag(&[5., 5., 4.9999, 4.9998, 3., 2., 1., 0.]);
        let op = DenseSymmetricOperator::new(8, &matrix).unwrap();
        let mut params = p();
        params.wanted = 4;
        params.ncv = 8;
        params.block_size = 2;
        let result = block_krylov_schur_eigenpairs(&op, &params).unwrap();
        assert_eq!(result.eigenvalues.len(), 4);
        assert!(result.eigenvalues.iter().all(|x| x.is_finite()));
        assert!(result.eigenvalues.iter().filter(|x| **x > 4.99).count() >= 2);
    }

    #[test]
    fn rank_deficiency_is_a_numerical_breakdown() {
        let matrix = vec![0.; 36];
        let op = DenseSymmetricOperator::new(6, &matrix).unwrap();
        assert!(matches!(
            block_krylov_schur_eigenpairs(&op, &p()),
            Err(BlockKrylovSchurError::NumericalBreakdown(_))
        ));
    }

    #[test]
    fn partial_convergence_locks_only_complete_blocks() {
        let params = BlockKrylovSchurParams {
            wanted: 4,
            ncv: 6,
            block_size: 2,
            max_iterations: 2,
            absolute_tolerance: 0.0,
            relative_tolerance: 1e-3,
            seed: 17,
        };
        let mut solver = Solver {
            op: &FixedDim(8),
            n: 8,
            p: params,
            v: Mat::from_fn(8, 6, |i, j| if i == j { 1.0 } else { 0.0 }),
            h: Mat::from_fn(6, 4, |i, j| if i == j { (4 - i) as f32 } else { 0.0 }),
            rng: Rng::new(params.seed),
            apps: 0,
            iters: 1,
            converged: 3,
            locked: block_aligned_locked_prefix(3, params.wanted, params.block_size),
            abs: 0.0,
            rel: 0.0,
        };

        assert_eq!(solver.converged, 3);
        assert_eq!(solver.locked, 2);
        solver.truncate().unwrap();
        assert_eq!(solver.h.nrows(), params.wanted + params.block_size);
        assert_eq!(solver.h.ncols(), params.wanted);
    }

    #[test]
    fn every_invalid_parameter_is_rejected_individually() {
        let mut q = p();
        assert!(matches!(
            block_krylov_schur_eigenpairs(&FixedDim(0), &q),
            Err(BlockKrylovSchurError::InvalidParameter("dimension"))
        ));
        q.wanted = 0;
        invalid(q, "wanted");
        q = p();
        q.block_size = 0;
        invalid(q, "block_size");
        q = p();
        q.wanted = 3;
        invalid(q, "block_size");
        q = p();
        q.ncv = 5;
        invalid(q, "ncv");
        q = p();
        q.ncv = 10;
        invalid(q, "ncv");
        q = p();
        q.max_iterations = 0;
        invalid(q, "max_iterations");
        q = p();
        q.absolute_tolerance = -1.;
        invalid(q, "absolute_tolerance");
        q = p();
        q.absolute_tolerance = f32::NAN;
        invalid(q, "absolute_tolerance");
        q = p();
        q.relative_tolerance = -1.;
        invalid(q, "relative_tolerance");
        q.relative_tolerance = f32::INFINITY;
        invalid(q, "relative_tolerance");
    }

    #[test]
    fn overflow_and_builtin_length_errors_are_reported() {
        let mut q = p();
        q.wanted = usize::MAX - 1;
        q.block_size = 2;
        q.ncv = 4;
        assert!(matches!(
            block_krylov_schur_eigenpairs(&FixedDim(usize::MAX), &q),
            Err(BlockKrylovSchurError::DimensionOverflow { .. })
        ));
        assert!(matches!(
            DenseSymmetricOperator::new(usize::MAX, &[]),
            Err(BuiltinOperatorError::DimensionOverflow { .. })
        ));
        let op = DenseSymmetricOperator::new(2, &[1., 0., 0., 1.]).unwrap();
        let mut output = [0.; 1];
        assert!(matches!(
            op.apply(&[1., 0.], 1, &mut output),
            Err(BuiltinOperatorError::InvalidLength { name: "output", .. })
        ));
    }

    #[test]
    fn progress_counts_are_monotonic_and_match_the_limit() {
        let matrix = diag(&[6., 5., 4., 3., 2., 1.]);
        let op = DenseSymmetricOperator::new(6, &matrix).unwrap();
        let mut params = p();
        params.absolute_tolerance = 0.;
        params.relative_tolerance = 0.;
        params.max_iterations = 2;
        let expected_max = block_krylov_schur_max_block_operator_applications(&params).unwrap();
        let mut events = Vec::new();
        let result =
            block_krylov_schur_eigenpairs_with_progress(&op, &params, |x| events.push(x)).unwrap();
        assert!(events.windows(2).all(|x| {
            x[0].block_operator_applications <= x[1].block_operator_applications
                && x[0].iteration <= x[1].iteration
        }));
        assert!(events.iter().all(|x| {
            x.max_block_operator_applications == expected_max
                && x.block_operator_applications <= expected_max
        }));
        assert!(events.iter().any(|x| x.iteration > 0));
        assert_eq!(events.last().unwrap().iteration, result.iterations);
        assert_eq!(result.status, BlockKrylovSchurStatus::IterationLimit);
        assert_eq!(result.block_operator_applications, expected_max);
    }

    #[test]
    fn gram_operators_agree_on_nonzero_values() {
        let d = [2., 0., 0., 0., 3., 0., 0., 0., 1., 0., 0., 0.];
        let right = RightGramOperator::new(4, 3, &d).unwrap();
        let left = LeftGramOperator::new(4, 3, &d).unwrap();
        let mut right_out = [0.; 3];
        let mut left_out = [0.; 4];
        right.apply(&[0., 1., 0.], 1, &mut right_out).unwrap();
        left.apply(&[0., 1., 0., 0.], 1, &mut left_out).unwrap();
        assert_eq!(right_out, [0., 9., 0.]);
        assert_eq!(left_out, [0., 9., 0., 0.]);
    }

    #[test]
    fn validation_progress_and_iteration_status() {
        let matrix = diag(&[6., 5., 4., 3., 2., 1.]);
        let op = DenseSymmetricOperator::new(6, &matrix).unwrap();
        let mut q = p();
        q.ncv = 5;
        assert!(matches!(
            block_krylov_schur_eigenpairs(&op, &q),
            Err(BlockKrylovSchurError::InvalidParameter("ncv"))
        ));
        q = p();
        q.max_iterations = 1;
        let mut progress = Vec::new();
        let r = block_krylov_schur_eigenpairs_with_progress(&op, &q, |x| {
            progress.push(x.block_operator_applications)
        })
        .unwrap();
        assert_eq!(r.status, BlockKrylovSchurStatus::IterationLimit);
        assert!(progress.windows(2).all(|x| x[0] <= x[1]));
    }

    #[derive(Debug)]
    struct Failure;
    impl fmt::Display for Failure {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "failure")
        }
    }
    impl std::error::Error for Failure {}
    struct Failing;
    impl SymmetricOperator for Failing {
        type Error = Failure;
        fn dim(&self) -> usize {
            6
        }
        fn apply(&self, _: &[f32], _: usize, _: &mut [f32]) -> Result<(), Failure> {
            Err(Failure)
        }
    }
    #[test]
    fn operator_error_is_preserved() {
        let e = block_krylov_schur_eigenpairs(&Failing, &p()).unwrap_err();
        assert!(matches!(e, BlockKrylovSchurError::Operator(Failure)));
    }
}
