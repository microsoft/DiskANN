/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Exhaustive f32 TopK without materializing the query-by-document score matrix.
//!
//! Each query row is independent. Queries are copied and packed once; documents
//! remain borrowed. Each compute replaces the previous result, including when
//! the document set changes. Rectangular self-search includes the diagonal.

use std::num::NonZeroUsize;

use diskann_vector::{
    Norm, PureDistanceFunction,
    distance::{InnerProduct, Metric},
    norm::FastL2NormSquared,
    topk::{Candidate, insert_sorted},
};
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};
use diskann_wide::arch::{Scalar, Target1};
use thiserror::Error;

use super::isa::{MaxSimIsa, NotSupported, dispatch_isa};
use crate::{
    matrix_kernels::{self as mk, topk::packed_f32_x_unpacked_f32 as f32_kernel},
    multi_vector::{BlockTransposed, Mat, MatMut, MatRef, Overflow, Standard},
};

/// Failure to prepare an f32 TopK query.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BuildTopKError {
    #[error("TopK does not support metric {0:?}")]
    UnsupportedMetric(Metric),
    #[error(transparent)]
    NotSupported(#[from] NotSupported),
    #[error(transparent)]
    Overflow(#[from] Overflow),
}

/// Failure to bind documents and their metric-specific metadata.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PrepareTopKDocumentsError {
    #[error("TopK does not support metric {0:?}")]
    UnsupportedMetric(Metric),
    #[error("document count {0} exceeds the u32 candidate ID range")]
    CandidateCount(usize),
}

/// Failure to compute TopK. Detectable input and extent errors leave output unchanged.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TopKError {
    #[error("document dimension {0} differs from query dimension {1}")]
    UnequalDim(usize, usize),
    #[error("output has {0} rows, expected {1}")]
    OutputRows(usize, usize),
    #[error("document metric {0:?} differs from query metric {1:?}")]
    MetricMismatch(Metric, Metric),
    #[error(transparent)]
    Overflow(#[from] Overflow),
}

/// Borrowed documents and norms computed from those same, immutable rows.
#[derive(Debug)]
pub struct TopKDocuments<'a> {
    matrix: MatRef<'a, Standard<f32>>,
    metric: Metric,
    norms: Vec<f32>,
}

/// Reusable compute storage. Concurrent calls require separate scratch and output.
#[derive(Debug, Default)]
pub struct TopKScratch {
    kth: Vec<f32>,
}

/// A prepared query snapshot with a coarse-grained ISA-dispatched compute boundary.
pub trait TopKKernel: Send + Sync + std::fmt::Debug {
    fn nrows(&self) -> usize;

    /// Return the nearest document rows, ordered by nondecreasing distance.
    ///
    /// k is the output column count. NaN and positive infinity are ignored;
    /// unfilled slots hold `Candidate::EMPTY`. Cutoff ties may select any equal
    /// candidate, without a cross-ISA ID guarantee.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched dimensions, output rows, metric, or
    /// overflowing working extents. Output remains unchanged on these errors.
    fn compute_top_k(
        &self,
        documents: &TopKDocuments<'_>,
        output: MatMut<'_, Standard<Candidate>>,
        scratch: &mut TopKScratch,
    ) -> Result<(), TopKError>;
}

fn supported(metric: Metric) -> bool {
    matches!(metric, Metric::InnerProduct | Metric::L2)
}

fn norms(matrix: MatRef<'_, Standard<f32>>, metric: Metric) -> Vec<f32> {
    if metric != Metric::L2 || matrix.vector_dim() == 0 {
        return Vec::new();
    }
    matrix
        .rows()
        .map(|row| FastL2NormSquared.evaluate(row))
        .collect()
}

impl<'a> TopKDocuments<'a> {
    /// Bind documents and prepare L2 norms once, without copying the document matrix.
    ///
    /// # Errors
    ///
    /// Rejects unsupported metrics or more than `u32::MAX` rows.
    pub fn new(
        documents: MatRef<'a, Standard<f32>>,
        metric: Metric,
    ) -> Result<Self, PrepareTopKDocumentsError> {
        if !supported(metric) {
            return Err(PrepareTopKDocumentsError::UnsupportedMetric(metric));
        }
        if documents.num_vectors() > u32::MAX as usize {
            return Err(PrepareTopKDocumentsError::CandidateCount(
                documents.num_vectors(),
            ));
        }
        Ok(Self {
            matrix: documents,
            metric,
            norms: norms(documents, metric),
        })
    }
}

fn validate(
    rows: usize,
    dimensions: usize,
    metric: Metric,
    documents: &TopKDocuments<'_>,
    output: &MatMut<'_, Standard<Candidate>>,
) -> Result<(), TopKError> {
    if output.num_vectors() != rows {
        return Err(TopKError::OutputRows(output.num_vectors(), rows));
    }
    if documents.matrix.vector_dim() != dimensions {
        return Err(TopKError::UnequalDim(
            documents.matrix.vector_dim(),
            dimensions,
        ));
    }
    if documents.metric != metric {
        return Err(TopKError::MetricMismatch(documents.metric, metric));
    }
    Ok(())
}

/// Handle empty shapes before deriving any unused padded extents.
fn trivial(documents: &TopKDocuments<'_>, output: &mut MatMut<'_, Standard<Candidate>>) -> bool {
    if output.num_vectors() == 0 || output.vector_dim() == 0 {
        return true;
    }
    let count = documents.matrix.num_vectors();
    if count == 0 || documents.matrix.vector_dim() == 0 {
        let distance = if documents.metric == Metric::InnerProduct {
            -0.0
        } else {
            0.0
        };
        for row in output.rows_mut() {
            row.fill(Candidate::EMPTY);
            for (id, slot) in row.iter_mut().take(count).enumerate() {
                *slot = Candidate::new(id as u32, distance);
            }
        }
        return true;
    }
    false
}

#[derive(Debug)]
struct Prepared<A, const MR: usize, const NR: usize> {
    arch: A,
    query: BlockTransposed<f32, MR>,
    norms: Vec<f32>,
    metric: Metric,
}

impl<A: f32_kernel::Kernel<MR>, const MR: usize, const NR: usize> TopKKernel
    for Prepared<A, MR, NR>
{
    fn nrows(&self) -> usize {
        self.query.nrows()
    }

    #[expect(
        clippy::expect_used,
        reason = "empty shapes were handled before building panel views"
    )]
    fn compute_top_k(
        &self,
        documents: &TopKDocuments<'_>,
        mut output: MatMut<'_, Standard<Candidate>>,
        scratch: &mut TopKScratch,
    ) -> Result<(), TopKError> {
        validate(
            self.nrows(),
            self.query.ncols(),
            self.metric,
            documents,
            &output,
        )?;
        if trivial(documents, &mut output) {
            return Ok(());
        }
        let width = output.vector_dim();
        let padded = self.query.padded_nrows();
        let params = f32_kernel::params(
            mk::Cache::detect(),
            self.query.ncols(),
            width,
            MR,
            NR,
            self.metric,
        )
        .ok_or_else(|| Overflow::for_type::<Candidate>(MR, width))?;
        let k = mk::DimK::new(NonZeroUsize::new(self.query.ncols()).expect("nonzero dimension"));
        let a = mk::blocks::packed::View::from_block_transposed(self.query.as_view())
            .expect("nonempty packed query");
        let b = mk::blocks::unpacked::View::from_matrix_view(documents.matrix.as_matrix_view())
            .expect("nonempty documents");
        scratch.kth.resize(padded, f32::INFINITY);
        scratch.kth.fill(f32::INFINITY);
        let slots = output.as_mut_slice();
        slots.fill(Candidate::EMPTY);
        // SAFETY: Validation and checked prepared storage establish dimensions and
        // extents. Thresholds/norms are padded; document IDs were checked at preparation.
        let mut driver = unsafe {
            f32_kernel::Driver::<A, MR, NR>::new(
                self.arch,
                a,
                b,
                f32_kernel::Rows::new(slots, &mut scratch.kth, &self.norms, self.metric, width),
                &documents.norms,
                k,
                params,
            )
        };
        mk::Drive::drive(&mut driver);
        Ok(())
    }
}

#[derive(Debug)]
struct Reference {
    query: Mat<Standard<f32>>,
    norms: Vec<f32>,
    metric: Metric,
}

impl TopKKernel for Reference {
    fn nrows(&self) -> usize {
        self.query.num_vectors()
    }

    fn compute_top_k(
        &self,
        documents: &TopKDocuments<'_>,
        mut output: MatMut<'_, Standard<Candidate>>,
        _: &mut TopKScratch,
    ) -> Result<(), TopKError> {
        validate(
            self.nrows(),
            self.query.vector_dim(),
            self.metric,
            documents,
            &output,
        )?;
        if trivial(documents, &mut output) {
            return Ok(());
        }
        for (i, (query, nearest)) in self.query.rows().zip(output.rows_mut()).enumerate() {
            nearest.fill(Candidate::EMPTY);
            let mut kth = f32::INFINITY;
            for (id, doc) in documents.matrix.rows().enumerate() {
                let distance = InnerProduct::evaluate(query, doc);
                let distance = if self.metric == Metric::L2 {
                    (self.norms[i] + documents.norms[id]) + 2.0 * distance
                } else {
                    distance
                };
                if distance < kth {
                    kth = insert_sorted(nearest, Candidate::new(id as u32, distance));
                }
            }
        }
        Ok(())
    }
}

struct Build(Metric);

fn prepare<A, const MR: usize, const NR: usize>(
    arch: A,
    query: MatRef<'_, Standard<f32>>,
    metric: Metric,
) -> Result<Prepared<A, MR, NR>, BuildTopKError> {
    let packed = BlockTransposed::try_from_matrix_view(query.as_matrix_view())?;
    let mut norms = norms(query, metric);
    if metric == Metric::L2 && query.vector_dim() != 0 {
        norms.resize(packed.padded_nrows(), 0.0);
    }
    Ok(Prepared {
        arch,
        query: packed,
        norms,
        metric,
    })
}

macro_rules! build_arch {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl Target1<$arch, Result<Box<dyn TopKKernel>, BuildTopKError>, MatRef<'_, Standard<f32>>>
            for Build
        {
            fn run(
                self,
                arch: $arch,
                query: MatRef<'_, Standard<f32>>,
            ) -> Result<Box<dyn TopKKernel>, BuildTopKError> {
                if query.num_vectors() == 0 || query.vector_dim() == 0 {
                    return Ok(Box::new(Reference {
                        query: query.to_owned(),
                        norms: Vec::new(),
                        metric: self.0,
                    }));
                }
                Ok(Box::new(prepare::<$arch, $mr, $nr>(arch, query, self.0)?))
            }
        }
    };
}

build_arch!(Scalar, 8, 2);
#[cfg(target_arch = "x86_64")]
build_arch!(V3, 16, 6);
#[cfg(target_arch = "x86_64")]
build_arch!(V4, 32, 6);
#[cfg(target_arch = "aarch64")]
build_arch!(Neon, 8, 6);

/// Prepare an owned query snapshot for f32 IP or squared L2 TopK.
///
/// # Errors
///
/// Rejects unsupported metrics, unavailable explicit ISAs, or packed extent overflow.
///
/// # Example
///
/// ```
/// use diskann_quantization::multi_vector::{
///     Mat, MatRef, Standard, MaxSimIsa, TopKScratch,
///     build_top_k, TopKDocuments,
/// };
/// use diskann_vector::{distance::Metric, topk::Candidate};
/// use diskann_utils::ReborrowMut;
///
/// let query = MatRef::new(Standard::new(1, 1)?, &[2.0f32])?;
/// let data = [10.0f32, 2.0, 3.0];
/// let documents = MatRef::new(Standard::new(3, 1)?, &data)?;
/// let kernel = build_top_k(MaxSimIsa::Auto, query, Metric::L2)?;
/// let documents = TopKDocuments::new(documents, Metric::L2)?;
/// let mut output = Mat::new(Standard::new(1, 2)?, Candidate::EMPTY)?;
/// kernel.compute_top_k(
///     &documents, output.reborrow_mut(), &mut TopKScratch::default(),
/// )?;
/// assert_eq!(output.get_row(0).unwrap(), &[
///     Candidate::new(1, 0.0), Candidate::new(2, 1.0),
/// ]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn build_top_k(
    isa: MaxSimIsa,
    query: MatRef<'_, Standard<f32>>,
    metric: Metric,
) -> Result<Box<dyn TopKKernel>, BuildTopKError> {
    if !supported(metric) {
        return Err(BuildTopKError::UnsupportedMetric(metric));
    }
    dispatch_isa!(isa, Build(metric), query, {
        let kernel: Box<dyn TopKKernel> = Box::new(Reference {
            query: query.to_owned(),
            norms: norms(query, metric),
            metric,
        });
        Ok::<_, BuildTopKError>(kernel)
    })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_utils::ReborrowMut;

    fn isas() -> impl Iterator<Item = MaxSimIsa> {
        MaxSimIsa::all()
            .into_iter()
            .filter(|isa| isa.is_available())
    }

    fn view(data: &[f32], rows: usize, dim: usize) -> MatRef<'_, Standard<f32>> {
        MatRef::new(Standard::new(rows, dim).unwrap(), data).unwrap()
    }

    fn oracle(q: &[f32], d: &[f32], metric: Metric) -> f32 {
        let score = if metric == Metric::InnerProduct {
            -q.iter()
                .zip(d)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>()
        } else {
            q.iter()
                .zip(d)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum()
        };
        score as f32
    }

    fn check_grid(m: usize, n: usize, dim: usize, widths: &[usize]) {
        let q: Vec<_> = (0..m * dim)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 8.0)
            .collect();
        let d: Vec<_> = (0..n * dim)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) / 8.0)
            .collect();
        for metric in [Metric::InnerProduct, Metric::L2] {
            let documents = TopKDocuments::new(view(&d, n, dim), metric).unwrap();
            for isa in isas() {
                let kernel = build_top_k(isa, view(&q, m, dim), metric).unwrap();
                let mut scratch = TopKScratch::default();
                for &width in widths {
                    let mut output = Mat::new(
                        Standard::new(m, width).unwrap(),
                        Candidate::new(42, -1000.0),
                    )
                    .unwrap();
                    kernel
                        .compute_top_k(&documents, output.reborrow_mut(), &mut scratch)
                        .unwrap();
                    for (row_id, row) in output.rows().enumerate() {
                        let query = &q[row_id * dim..(row_id + 1) * dim];
                        let mut scores: Vec<_> = (0..n)
                            .map(|id| oracle(query, &d[id * dim..(id + 1) * dim], metric))
                            .collect();
                        scores.sort_by(f32::total_cmp);
                        let filled = width.min(n);
                        for (rank, candidate) in row[..filled].iter().enumerate() {
                            let id = candidate.local_idx as usize;
                            assert!(id < n, "{isa:?}, m={m}, n={n}, d={dim}, k={width}");
                            assert!(
                                !row[..rank]
                                    .iter()
                                    .any(|c| c.local_idx == candidate.local_idx)
                            );
                            let expected = oracle(query, &d[id * dim..(id + 1) * dim], metric);
                            assert_eq!(candidate.distance, expected);
                            assert_eq!(
                                candidate.distance, scores[rank],
                                "{isa:?}, m={m}, n={n}, d={dim}, k={width}"
                            );
                        }
                        assert!(row[filled..].iter().all(|c| *c == Candidate::EMPTY));
                        if dim == 0 && metric == Metric::InnerProduct {
                            assert!(
                                row[..filled]
                                    .iter()
                                    .all(|c| c.distance.to_bits() == (-0.0f32).to_bits())
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn top_k_matches_independent_sort_across_widths_and_panel_tails() {
        for m in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            for n in [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13] {
                check_grid(m, n, 1, &[0, 1, 2, 3, 4, 7, 8, 10, 16]);
            }
        }
    }

    #[test]
    fn top_k_matches_independent_sort_across_dimensions() {
        for dim in [0, 1, 7, 8, 15, 16, 17, 65, 128, 129, 384, 768] {
            check_grid(33, 17, dim, &[3, 10]);
        }
    }

    #[test]
    fn top_k_workspace_reuse_and_query_snapshot() {
        for isa in isas() {
            let mut q = vec![2.0];
            let kernel = build_top_k(isa, view(&q, 1, 1), Metric::L2).unwrap();
            q[0] = 100.0;
            let mut scratch = TopKScratch::default();
            for data in [
                &[2.0, 10.0, 3.0][..],
                &[20.0][..],
                &[][..],
                &[2.0, 10.0, 3.0][..],
            ] {
                let documents = TopKDocuments::new(view(data, data.len(), 1), Metric::L2).unwrap();
                for width in [1, 4, 2] {
                    let mut output =
                        Mat::new(Standard::new(1, width).unwrap(), Candidate::EMPTY).unwrap();
                    kernel
                        .compute_top_k(&documents, output.reborrow_mut(), &mut scratch)
                        .unwrap();
                    let row = output.get_row(0).unwrap();
                    if data.is_empty() {
                        assert!(row.iter().all(|c| *c == Candidate::EMPTY));
                    } else {
                        assert_eq!(row[0], Candidate::new(0, (data[0] - 2.0).powi(2)));
                        assert!(
                            row[data.len().min(width)..]
                                .iter()
                                .all(|c| *c == Candidate::EMPTY)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn top_k_filters_computed_nonfinite_distances() {
        for isa in isas() {
            let data = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 4.0, 2.0];
            let documents = TopKDocuments::new(view(&data, 5, 1), Metric::InnerProduct).unwrap();
            let kernel = build_top_k(isa, view(&[1.0], 1, 1), Metric::InnerProduct).unwrap();
            let mut output = Mat::new(Standard::new(1, 5).unwrap(), Candidate::EMPTY).unwrap();
            kernel
                .compute_top_k(
                    &documents,
                    output.reborrow_mut(),
                    &mut TopKScratch::default(),
                )
                .unwrap();
            assert_eq!(
                output.get_row(0).unwrap(),
                &[
                    Candidate::new(1, f32::NEG_INFINITY),
                    Candidate::new(3, -4.0),
                    Candidate::new(4, -2.0),
                    Candidate::EMPTY,
                    Candidate::EMPTY,
                ]
            );
            let documents = TopKDocuments::new(view(&data, 5, 1), Metric::L2).unwrap();
            let kernel = build_top_k(isa, view(&[1.0], 1, 1), Metric::L2).unwrap();
            kernel
                .compute_top_k(
                    &documents,
                    output.reborrow_mut(),
                    &mut TopKScratch::default(),
                )
                .unwrap();
            assert_eq!(
                output.get_row(0).unwrap(),
                &[
                    Candidate::new(4, 1.0),
                    Candidate::new(3, 9.0),
                    Candidate::EMPTY,
                    Candidate::EMPTY,
                    Candidate::EMPTY,
                ]
            );
        }
    }

    #[test]
    fn top_k_rectangular_self_search_keeps_self() {
        let data = [1.0, 2.0, 5.0];
        for isa in isas() {
            let kernel = build_top_k(isa, view(&data, 3, 1), Metric::L2).unwrap();
            let documents = TopKDocuments::new(view(&data, 3, 1), Metric::L2).unwrap();
            let mut output = Mat::new(Standard::new(3, 1).unwrap(), Candidate::EMPTY).unwrap();
            kernel
                .compute_top_k(
                    &documents,
                    output.reborrow_mut(),
                    &mut TopKScratch::default(),
                )
                .unwrap();
            for (id, row) in output.rows().enumerate() {
                assert_eq!(row, &[Candidate::new(id as u32, 0.0)]);
            }
        }
    }

    #[test]
    fn top_k_errors_leave_output_unchanged_even_on_fast_paths() {
        for isa in isas() {
            let kernel = build_top_k(isa, view(&[2.0], 1, 1), Metric::L2).unwrap();
            let wrong_dim = TopKDocuments::new(view(&[], 0, 2), Metric::L2).unwrap();
            let wrong_metric = TopKDocuments::new(view(&[], 0, 1), Metric::InnerProduct).unwrap();
            let good = TopKDocuments::new(view(&[], 0, 1), Metric::L2).unwrap();
            let sentinel = Candidate::new(7, 123.0);
            let mut output = Mat::new(Standard::new(1, 2).unwrap(), sentinel).unwrap();
            let mut scratch = TopKScratch::default();
            assert!(matches!(
                kernel.compute_top_k(&wrong_dim, output.reborrow_mut(), &mut scratch),
                Err(TopKError::UnequalDim(..))
            ));
            assert!(matches!(
                kernel.compute_top_k(&wrong_metric, output.reborrow_mut(), &mut scratch),
                Err(TopKError::MetricMismatch(..))
            ));
            assert_eq!(output.get_row(0).unwrap(), &[sentinel; 2]);
            let mut wrong_rows = Mat::new(Standard::new(2, 2).unwrap(), sentinel).unwrap();
            assert!(matches!(
                kernel.compute_top_k(&good, wrong_rows.reborrow_mut(), &mut scratch),
                Err(TopKError::OutputRows(..))
            ));
            assert!(wrong_rows.rows().all(|row| row == [sentinel; 2]));
            let mut zero_width = Mat::new(Standard::new(1, 0).unwrap(), sentinel).unwrap();
            assert!(
                kernel
                    .compute_top_k(&wrong_dim, zero_width.reborrow_mut(), &mut scratch)
                    .is_err()
            );
        }
    }

    #[test]
    fn top_k_rejects_unsupported_metrics_isas_and_candidate_extents() {
        for metric in [Metric::Cosine, Metric::CosineNormalized] {
            assert!(matches!(
                build_top_k(MaxSimIsa::Scalar, view(&[], 0, 0), metric),
                Err(BuildTopKError::UnsupportedMetric(..))
            ));
            assert!(matches!(
                TopKDocuments::new(view(&[], 0, 0), metric),
                Err(PrepareTopKDocumentsError::UnsupportedMetric(..))
            ));
        }
        for isa in MaxSimIsa::all()
            .into_iter()
            .filter(|isa| !isa.is_available())
        {
            assert!(matches!(
                build_top_k(isa, view(&[], 0, 0), Metric::L2),
                Err(BuildTopKError::NotSupported(..))
            ));
        }
        if usize::BITS > 32 {
            let count = (u32::MAX as usize).checked_add(1).unwrap();
            assert!(matches!(
                TopKDocuments::new(view(&[], count, 0), Metric::L2),
                Err(PrepareTopKDocumentsError::CandidateCount(..))
            ));
        }
    }

    #[test]
    fn top_k_empty_paths_do_not_require_padded_storage() {
        for isa in isas() {
            let kernel = build_top_k(isa, view(&[], usize::MAX, 0), Metric::L2).unwrap();
            let documents = TopKDocuments::new(view(&[], 1, 0), Metric::L2).unwrap();
            let mut output =
                Mat::new(Standard::new(usize::MAX, 0).unwrap(), Candidate::EMPTY).unwrap();
            let mut scratch = TopKScratch { kth: vec![123.0] };
            kernel
                .compute_top_k(&documents, output.reborrow_mut(), &mut scratch)
                .unwrap();
            assert_eq!(scratch.kth, [123.0]);
        }
    }

    #[test]
    fn top_k_ordinary_floats_match_a_margin_separated_oracle() {
        let dim = 65;
        let query: Vec<_> = (0..dim).map(|i| (i as f32 * 0.173).sin()).collect();
        let data: Vec<_> = (0..9)
            .flat_map(|id| query.iter().map(move |&x| x + id as f32 * 0.31))
            .collect();
        for isa in isas() {
            let kernel = build_top_k(isa, view(&query, 1, dim), Metric::L2).unwrap();
            let documents = TopKDocuments::new(view(&data, 9, dim), Metric::L2).unwrap();
            let mut output = Mat::new(Standard::new(1, 3).unwrap(), Candidate::EMPTY).unwrap();
            kernel
                .compute_top_k(
                    &documents,
                    output.reborrow_mut(),
                    &mut TopKScratch::default(),
                )
                .unwrap();
            for (rank, candidate) in output.get_row(0).unwrap().iter().enumerate() {
                assert_eq!(candidate.local_idx, rank as u32);
                let expected = oracle(&query, &data[rank * dim..(rank + 1) * dim], Metric::L2);
                let tolerance = 64.0 * f32::EPSILON * dim as f32 * expected.abs().max(1.0);
                assert!((candidate.distance - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn top_k_writes_only_the_logical_output_view() {
        let sentinel = Candidate::new(99, 123.0);
        for isa in isas() {
            for dim in [0, 1] {
                let query = [2.0];
                let data = [10.0, 2.0, 3.0];
                let kernel = build_top_k(isa, view(&query[..dim], 1, dim), Metric::L2).unwrap();
                let documents =
                    TopKDocuments::new(view(&data[..3 * dim], 3, dim), Metric::L2).unwrap();
                let mut backing = [sentinel; 4];
                let output = MatMut::new(Standard::new(1, 2).unwrap(), &mut backing[1..3]).unwrap();
                kernel
                    .compute_top_k(&documents, output, &mut TopKScratch::default())
                    .unwrap();
                assert_eq!(backing[0], sentinel);
                assert_eq!(backing[3], sentinel);
            }
        }
    }

    #[test]
    fn top_k_acceptance_uses_distance_after_finite_overflow() {
        let data = [f32::MAX, f32::MAX, 1.0, 1.0];
        for isa in isas() {
            for metric in [Metric::InnerProduct, Metric::L2] {
                let kernel = build_top_k(isa, view(&[1.0, 1.0], 1, 2), metric).unwrap();
                let documents = TopKDocuments::new(view(&data, 2, 2), metric).unwrap();
                let mut output = Mat::new(Standard::new(1, 2).unwrap(), Candidate::EMPTY).unwrap();
                kernel
                    .compute_top_k(
                        &documents,
                        output.reborrow_mut(),
                        &mut TopKScratch::default(),
                    )
                    .unwrap();
                let expected = if metric == Metric::L2 {
                    [Candidate::new(1, 0.0), Candidate::EMPTY]
                } else {
                    [
                        Candidate::new(0, f32::NEG_INFINITY),
                        Candidate::new(1, -2.0),
                    ]
                };
                assert_eq!(output.get_row(0).unwrap(), &expected);
            }
        }
    }
}
