/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Exhaustive f32 TopK without materializing the query-by-document score matrix.
//!
//! Bind candidate rows once and search them with successive query batches.
//! Each compute replaces the output. Rectangular self-search includes the diagonal.

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{
    Norm, PureDistanceFunction,
    distance::{InnerProduct, Metric},
    norm::FastL2NormSquared,
    topk::{Candidate, insert_sorted},
};
use thiserror::Error;

use super::isa::{MaxSimIsa, NotSupported, RuntimeArch};
use crate::{
    matrix_kernels::{self as mk, topk::packed_f32_x_unpacked_f32 as f32_kernel},
    multi_vector::{BlockTransposed, Overflow},
};
use f32_kernel::{Norms, Scoring};

/// Failure to bind candidates or select a TopK backend.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BuildTopKError {
    #[error("TopK does not support metric {0:?}")]
    UnsupportedMetric(Metric),
    #[error(transparent)]
    NotSupported(#[from] NotSupported),
    #[error("candidate count {0} exceeds the u32 candidate ID range")]
    CandidateCount(usize),
}

/// Failure to compute TopK. Detectable input and extent errors leave output unchanged.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TopKError {
    #[error("query dimension {0} differs from candidate dimension {1}")]
    UnequalDim(usize, usize),
    #[error("output has {0} rows, expected {1}")]
    OutputRows(usize, usize),
    #[error(transparent)]
    Overflow(#[from] Overflow),
}

/// Exhaustive f32 TopK over a fixed, borrowed candidate matrix.
#[derive(Debug)]
pub struct TopKKernel<'a> {
    arch: RuntimeArch,
    candidates: rowmajor::Ref<'a, f32>,
    score: Scoring<Vec<f32>>,
}

impl<'a> TopKKernel<'a> {
    /// Select the backend and bind the candidates without copying the matrix.
    /// Squared L2 prepares candidate norms once; inner product needs none.
    ///
    /// # Errors
    ///
    /// Rejects unsupported metrics, unavailable explicit ISAs, or more than
    /// `u32::MAX` candidate rows.
    ///
    /// # Example
    ///
    /// ```
    /// use diskann_quantization::multi_vector::{MaxSimIsa, TopKKernel};
    /// use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
    /// use diskann_vector::{distance::Metric, topk::Candidate};
    ///
    /// let centroids = rowmajor::Ref::column_vector(&[10.0f32, 2.0, 3.0]);
    /// let query_batches = [
    ///     rowmajor::Ref::row_vector(&[2.0f32]),
    ///     rowmajor::Ref::row_vector(&[10.0f32]),
    /// ];
    ///
    /// // Prepare the centroids and their L2 norms once for all batches.
    /// let kernel = TopKKernel::new(MaxSimIsa::Auto, Metric::L2, centroids)?;
    /// let mut output = rowmajor::Owned::from_element(1, 2, Candidate::EMPTY);
    /// for queries in query_batches {
    ///     kernel.compute_top_k(queries, output.as_view_mut())?;
    ///     // Use this batch's results before reusing the output buffer.
    ///     # let nearest = output.row(0)[0];
    ///     # assert_eq!(nearest.distance, 0.0);
    ///     # assert_eq!(centroids.row(nearest.local_idx as usize), queries.row(0));
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        isa: MaxSimIsa,
        metric: Metric,
        candidates: rowmajor::Ref<'a, f32>,
    ) -> Result<Self, BuildTopKError> {
        if candidates.nrows() > u32::MAX as usize {
            return Err(BuildTopKError::CandidateCount(candidates.nrows()));
        }
        let arch = RuntimeArch::resolve(isa)?;
        let score = match metric {
            Metric::InnerProduct => Scoring::InnerProduct,
            Metric::L2 => Scoring::L2(squared_norms(candidates)),
            _ => return Err(BuildTopKError::UnsupportedMetric(metric)),
        };
        Ok(Self {
            arch,
            candidates,
            score,
        })
    }

    /// Write the nearest candidate row IDs and distances for each query, nearest first.
    ///
    /// k is the output column count. NaN and positive infinity are ignored;
    /// unfilled slots hold `Candidate::EMPTY`. Cutoff ties may select any equal
    /// candidate, without a cross-ISA ID guarantee. Native backends pack this
    /// query batch for the duration of the call. Concurrent calls need distinct outputs.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched dimensions or output rows, or overflowing
    /// packed query or working extents. Output remains unchanged on these errors.
    pub fn compute_top_k(
        &self,
        queries: rowmajor::Ref<'_, f32>,
        mut output: rowmajor::Mut<'_, Candidate>,
    ) -> Result<(), TopKError> {
        if output.nrows() != queries.nrows() {
            return Err(TopKError::OutputRows(output.nrows(), queries.nrows()));
        }
        if queries.ncols() != self.candidates.ncols() {
            return Err(TopKError::UnequalDim(
                queries.ncols(),
                self.candidates.ncols(),
            ));
        }
        if output.nrows() == 0 || output.ncols() == 0 {
            return Ok(());
        }
        let count = self.candidates.nrows();
        if count == 0 || queries.ncols() == 0 {
            let distance = match self.score {
                Scoring::InnerProduct => -0.0,
                Scoring::L2(_) => 0.0,
            };
            for row in output.rows_mut() {
                row.fill(Candidate::EMPTY);
                for (id, slot) in row.iter_mut().take(count).enumerate() {
                    *slot = Candidate::new(id as u32, distance);
                }
            }
            return Ok(());
        }
        let query_norms;
        let score = match &self.score {
            Scoring::InnerProduct => Scoring::InnerProduct,
            Scoring::L2(b) => {
                query_norms = squared_norms(queries);
                Scoring::L2(Norms { a: &query_norms, b })
            }
        };
        match self.arch {
            RuntimeArch::Scalar(a) => native::<_, 8, 2>(a, queries, self.candidates, score, output),
            #[cfg(target_arch = "x86_64")]
            RuntimeArch::V3(a) => native::<_, 16, 6>(a, queries, self.candidates, score, output),
            #[cfg(target_arch = "x86_64")]
            RuntimeArch::V4(a) => native::<_, 32, 6>(a, queries, self.candidates, score, output),
            #[cfg(target_arch = "aarch64")]
            RuntimeArch::Neon(a) => native::<_, 8, 6>(a, queries, self.candidates, score, output),
            RuntimeArch::Reference => {
                reference(queries, self.candidates, score, output);
                Ok(())
            }
        }
    }
}

fn squared_norms(matrix: rowmajor::Ref<'_, f32>) -> Vec<f32> {
    if matrix.ncols() == 0 {
        return Vec::new();
    }
    matrix
        .rows()
        .map(|r| FastL2NormSquared.evaluate(r))
        .collect()
}

fn native<A: f32_kernel::Kernel<MR>, const MR: usize, const NR: usize>(
    arch: A,
    queries: rowmajor::Ref<'_, f32>,
    candidates: rowmajor::Ref<'_, f32>,
    score: Scoring<Norms<'_>>,
    output: rowmajor::Mut<'_, Candidate>,
) -> Result<(), TopKError> {
    let width = output.ncols();
    let params = score
        .params::<MR, NR>(mk::Cache::detect(), queries.ncols(), width)
        .ok_or_else(|| Overflow::for_type::<Candidate>(MR, width))?;
    let packed = BlockTransposed::<f32, MR>::try_from_matrix_view(queries)?;
    f32_kernel::compute::<_, MR, NR>(arch, packed.as_view(), candidates, output, score, params);
    Ok(())
}

fn reference(
    queries: rowmajor::Ref<'_, f32>,
    candidates: rowmajor::Ref<'_, f32>,
    score: Scoring<Norms<'_>>,
    mut output: rowmajor::Mut<'_, Candidate>,
) {
    for (i, (query, nearest)) in queries.rows().zip(output.rows_mut()).enumerate() {
        nearest.fill(Candidate::EMPTY);
        let mut kth = f32::INFINITY;
        for (id, doc) in candidates.rows().enumerate() {
            let neg_dot: f32 = InnerProduct::evaluate(query, doc);
            let distance = score.distance(i, id, -neg_dot);
            if distance < kth {
                kth = insert_sorted(nearest, Candidate::new(id as u32, distance));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isas() -> impl Iterator<Item = MaxSimIsa> {
        MaxSimIsa::all()
            .into_iter()
            .filter(|isa| isa.is_available())
    }

    fn view(data: &[f32], rows: usize, dim: usize) -> rowmajor::Ref<'_, f32> {
        rowmajor::Ref::try_from_data(data, rows, dim).unwrap()
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
            for isa in isas() {
                let kernel = TopKKernel::new(isa, metric, view(&d, n, dim)).unwrap();
                for &width in widths {
                    let mut output =
                        rowmajor::Owned::from_element(m, width, Candidate::new(42, -1000.0));
                    kernel
                        .compute_top_k(view(&q, m, dim), output.as_view_mut())
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
    fn top_k_reuses_candidates_across_query_batches() {
        let centroids = [10.0, 2.0, 3.0];
        for isa in isas() {
            for (metric, expected) in [
                (
                    Metric::L2,
                    [
                        Candidate::new(1, 0.0),
                        Candidate::new(0, 0.0),
                        Candidate::new(1, 9.0),
                    ],
                ),
                (
                    Metric::InnerProduct,
                    [
                        Candidate::new(0, -20.0),
                        Candidate::new(0, -100.0),
                        Candidate::new(1, 2.0),
                    ],
                ),
            ] {
                let kernel =
                    TopKKernel::new(isa, metric, rowmajor::Ref::column_vector(&centroids)).unwrap();
                let mut output = [Candidate::EMPTY; 2];
                for (batch, nearest) in [
                    (&[2.0][..], &expected[..1]),
                    (&[10.0, -1.0][..], &expected[1..]),
                    (&[][..], &expected[..0]),
                    (&[2.0][..], &expected[..1]),
                ] {
                    kernel
                        .compute_top_k(
                            rowmajor::Ref::column_vector(batch),
                            rowmajor::Mut::column_vector(&mut output[..batch.len()]),
                        )
                        .unwrap();
                    assert_eq!(&output[..batch.len()], nearest);
                }
            }
        }
    }

    #[test]
    fn top_k_filters_computed_nonfinite_distances() {
        for isa in isas() {
            let data = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 4.0, 2.0];
            let queries = view(&[1.0], 1, 1);
            let kernel = TopKKernel::new(isa, Metric::InnerProduct, view(&data, 5, 1)).unwrap();
            let mut output = rowmajor::Owned::from_element(1, 5, Candidate::EMPTY);
            kernel.compute_top_k(queries, output.as_view_mut()).unwrap();
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
            let kernel = TopKKernel::new(isa, Metric::L2, view(&data, 5, 1)).unwrap();
            kernel.compute_top_k(queries, output.as_view_mut()).unwrap();
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
            let points = view(&data, 3, 1);
            let kernel = TopKKernel::new(isa, Metric::L2, points).unwrap();
            let mut output = rowmajor::Owned::from_element(3, 1, Candidate::EMPTY);
            kernel.compute_top_k(points, output.as_view_mut()).unwrap();
            for (id, row) in output.rows().enumerate() {
                assert_eq!(row, &[Candidate::new(id as u32, 0.0)]);
            }
        }
    }

    #[test]
    fn top_k_errors_leave_output_unchanged_even_on_fast_paths() {
        for isa in isas() {
            let kernel = TopKKernel::new(isa, Metric::L2, view(&[], 0, 1)).unwrap();
            let wrong_dim = view(&[2.0, 3.0], 1, 2);
            let good = view(&[2.0], 1, 1);
            let sentinel = Candidate::new(7, 123.0);
            let mut output = rowmajor::Owned::from_element(1, 2, sentinel);
            assert!(matches!(
                kernel.compute_top_k(wrong_dim, output.as_view_mut()),
                Err(TopKError::UnequalDim(2, 1))
            ));
            assert_eq!(output.get_row(0).unwrap(), &[sentinel; 2]);
            let mut wrong_rows = rowmajor::Owned::from_element(2, 2, sentinel);
            assert!(matches!(
                kernel.compute_top_k(good, wrong_rows.as_view_mut()),
                Err(TopKError::OutputRows(2, 1))
            ));
            assert!(wrong_rows.rows().all(|row| row == [sentinel; 2]));
            let mut zero_width = rowmajor::Owned::from_element(1, 0, sentinel);
            assert!(
                kernel
                    .compute_top_k(wrong_dim, zero_width.as_view_mut())
                    .is_err()
            );
        }
    }

    #[test]
    fn top_k_rejects_unsupported_metrics_isas_and_candidate_extents() {
        for metric in [Metric::Cosine, Metric::CosineNormalized] {
            assert!(matches!(
                TopKKernel::new(MaxSimIsa::Scalar, metric, view(&[], 0, 0)),
                Err(BuildTopKError::UnsupportedMetric(..))
            ));
        }
        for isa in MaxSimIsa::all()
            .into_iter()
            .filter(|isa| !isa.is_available())
        {
            assert!(matches!(
                TopKKernel::new(isa, Metric::L2, view(&[], 0, 0)),
                Err(BuildTopKError::NotSupported(..))
            ));
        }
        if usize::BITS > 32 {
            let count = (u32::MAX as usize).checked_add(1).unwrap();
            assert!(matches!(
                TopKKernel::new(MaxSimIsa::Scalar, Metric::L2, view(&[], count, 0)),
                Err(BuildTopKError::CandidateCount(n)) if n == count
            ));
        }
    }

    #[test]
    fn top_k_empty_paths_do_not_require_padded_storage() {
        for isa in isas() {
            let kernel = TopKKernel::new(isa, Metric::L2, view(&[], 1, 0)).unwrap();
            let mut output = rowmajor::Owned::from_element(usize::MAX, 0, Candidate::EMPTY);
            kernel
                .compute_top_k(view(&[], usize::MAX, 0), output.as_view_mut())
                .unwrap();
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
            let kernel = TopKKernel::new(isa, Metric::L2, view(&data, 9, dim)).unwrap();
            let mut output = rowmajor::Owned::from_element(1, 3, Candidate::EMPTY);
            kernel
                .compute_top_k(view(&query, 1, dim), output.as_view_mut())
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
                let kernel =
                    TopKKernel::new(isa, Metric::L2, view(&data[..3 * dim], 3, dim)).unwrap();
                let mut backing = [sentinel; 4];
                let output = rowmajor::Mut::try_from_data(&mut backing[1..3], 1, 2).unwrap();
                kernel
                    .compute_top_k(view(&query[..dim], 1, dim), output)
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
                let kernel = TopKKernel::new(isa, metric, view(&data, 2, 2)).unwrap();
                let mut output = rowmajor::Owned::from_element(1, 2, Candidate::EMPTY);
                kernel
                    .compute_top_k(view(&[1.0, 1.0], 1, 2), output.as_view_mut())
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
