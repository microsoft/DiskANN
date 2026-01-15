/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{
    strided::StridedView,
    views::{self, Matrix},
};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;

use crate::{
    algorithms::kmeans::{
        self,
        common::{square_norm, BlockTranspose},
    },
    cancel::Cancelation,
    random::{BoxedRngBuilder, RngBuilder},
    Parallelism,
};

pub struct LightPQTrainingParameters {
    /// The number of centers for each partition.
    ncenters: usize,
    /// The maximum number of iterations for Lloyd's algorithm.
    lloyds_reps: usize,
}

impl LightPQTrainingParameters {
    /// Construct a new light-weight PQ trainer.
    pub fn new(ncenters: usize, lloyds_reps: usize) -> Self {
        Self {
            ncenters,
            lloyds_reps,
        }
    }
}

#[derive(Debug)]
pub struct SimplePivots {
    dim: usize,
    ncenters: usize,
    pivots: Vec<Matrix<f32>>,
}

fn flatten<T: Copy + Default>(pivots: &[Matrix<T>], ncenters: usize, dim: usize) -> Matrix<T> {
    let mut flattened = Matrix::new(T::default(), ncenters, dim);
    let mut col_start = 0;
    for matrix in pivots {
        assert_eq!(matrix.nrows(), flattened.nrows());
        for (row_index, row) in matrix.row_iter().enumerate() {
            let dst = &mut flattened.row_mut(row_index)[col_start..col_start + row.len()];
            dst.copy_from_slice(row);
        }
        col_start += matrix.ncols();
    }
    flattened
}

impl SimplePivots {
    /// Return the selected pivots for each chunk.
    pub fn pivots(&self) -> &[Matrix<f32>] {
        &self.pivots
    }

    /// Concatenate the individual pivots into a dense representation.
    pub fn flatten(&self) -> Vec<f32> {
        flatten(self.pivots(), self.ncenters, self.dim)
            .into_inner()
            .into()
    }
}

pub trait TrainQuantizer {
    type Quantizer;
    type Error: std::error::Error;

    fn train<R, C>(
        &self,
        data: views::MatrixView<f32>,
        schema: crate::views::ChunkOffsetsView<'_>,
        parallelism: Parallelism,
        rng_builder: &R,
        cancelation: &C,
    ) -> Result<Self::Quantizer, Self::Error>
    where
        R: RngBuilder<usize> + Sync,
        C: Cancelation + Sync;
}

impl TrainQuantizer for LightPQTrainingParameters {
    type Quantizer = SimplePivots;
    type Error = PQTrainingError;

    /// Perform product quantization training on the provided training set and return a
    /// `SimplePivots` containing the result of kmeans clustering on each partition.
    ///
    /// # Panics
    ///
    /// Panics if `data.nrows() != schema.dim()`.
    ///
    /// # Errors
    ///
    /// An error type is returned under the following circumstances:
    ///
    /// * A cancellation request is received. This case can be queried by calling
    ///   `was_canceled` on the returned `PQTrainingError`.
    /// * `NaN` or infinities are observed during the training process.
    fn train<R, C>(
        &self,
        data: views::MatrixView<f32>,
        schema: crate::views::ChunkOffsetsView<'_>,
        parallelism: Parallelism,
        rng_builder: &R,
        cancelation: &C,
    ) -> Result<Self::Quantizer, Self::Error>
    where
        R: RngBuilder<usize> + Sync,
        C: Cancelation + Sync,
    {
        // Inner method where we `dyn` away the cancellation token to reduce compile-times.
        // Unfortunately, we can't quite do the same with the RngBuilder.
        #[inline(never)]
        fn train(
            trainer: &LightPQTrainingParameters,
            data: views::MatrixView<f32>,
            schema: crate::views::ChunkOffsetsView<'_>,
            parallelism: Parallelism,
            rng_builder: &(dyn BoxedRngBuilder<usize> + Sync),
            cancelation: &(dyn Cancelation + Sync),
        ) -> Result<SimplePivots, PQTrainingError> {
            // Make sure we're provided sane values for our schema.
            assert_eq!(data.ncols(), schema.dim());

            let thunk = |i| -> Result<Matrix<f32>, PQTrainingError> {
                let range = schema.at(i);

                // Check for cancelation.
                let exit_if_canceled = || -> Result<(), PQTrainingError> {
                    if cancelation.should_cancel() {
                        Err(PQTrainingError {
                            chunk: i,
                            of: schema.len(),
                            dim: range.len(),
                            kind: PQTrainingErrorKind::Canceled,
                        })
                    } else {
                        Ok(())
                    }
                };

                // This is an early check - if another task hit cancelation, this allows
                // the remaining tasks to exit early.
                exit_if_canceled()?;

                let view = StridedView::try_shrink_from(
                    &(data.as_slice()[range.start..]),
                    data.nrows(),
                    range.len(),
                    schema.dim(),
                )
                .map_err(|err| PQTrainingError {
                    chunk: i,
                    of: schema.len(),
                    dim: range.len(),
                    kind: PQTrainingErrorKind::InternalError(Box::new(err.as_static())),
                })?;

                // Allocate scratch data structures.
                let norms: Vec<f32> = view.row_iter().map(square_norm).collect();
                let transpose = BlockTranspose::<16>::from_strided(view);
                let mut centers = Matrix::new(0.0, trainer.ncenters, range.len());

                // Construct the random number generator seeded by the PQ chunk.
                let mut rng = rng_builder.build_boxed_rng(i);

                // Initialization
                kmeans::plusplus::kmeans_plusplus_into_inner(
                    centers.as_mut_view(),
                    view,
                    &transpose,
                    &norms,
                    &mut rng,
                )
                .or_else(|err| {
                    // Suppress recoverable errors.
                    if !err.is_numerically_recoverable() {
                        Err(PQTrainingError {
                            chunk: i,
                            of: schema.len(),
                            dim: range.len(),
                            kind: PQTrainingErrorKind::Initialization(Box::new(err)),
                        })
                    } else {
                        Ok(())
                    }
                })?;

                // Did a cancelation request come while runing `kmeans++`?
                exit_if_canceled()?;

                // Kmeans
                kmeans::lloyds::lloyds_inner(
                    view,
                    &norms,
                    &transpose,
                    centers.as_mut_view(),
                    trainer.lloyds_reps,
                );
                Ok(centers)
            };

            let pivots: Result<Vec<_>, _> = match parallelism {
                Parallelism::Sequential => (0..schema.len()).map(thunk).collect(),

                #[cfg(feature = "rayon")]
                Parallelism::Rayon => (0..schema.len()).into_par_iter().map(thunk).collect(),
            };

            let dim = data.ncols();
            let ncenters = trainer.ncenters;
            Ok(SimplePivots {
                dim,
                ncenters,
                pivots: pivots?,
            })
        }

        train(self, data, schema, parallelism, rng_builder, cancelation)
    }
}

#[derive(Debug, Error)]
#[error("pq training failed on chunk {chunk} of {of} (dim {dim})")]
pub struct PQTrainingError {
    chunk: usize,
    of: usize,
    dim: usize,
    #[source]
    kind: PQTrainingErrorKind,
}

impl PQTrainingError {
    /// Return whether or not this error originated as a cancelation request.
    pub fn was_canceled(&self) -> bool {
        matches!(self.kind, PQTrainingErrorKind::Canceled)
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
enum PQTrainingErrorKind {
    #[error("canceled by request")]
    Canceled,
    #[error("initial pivot selection error")]
    Initialization(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("internal logic error")]
    InternalError(#[source] Box<dyn std::error::Error + Send + Sync>),
}

///////////
// Tests //
///////////

#[cfg(not(miri))]
#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rand::{
        distr::{Distribution, StandardUniform, Uniform},
        rngs::StdRng,
        seq::SliceRandom,
        Rng, SeedableRng,
    };

    use diskann_utils::lazy_format;

    use super::*;
    use crate::{cancel::DontCancel, error::format, random::StdRngBuilder};

    // With this test - we create sub-matrices that when flattened, will yield the output
    // sequence `0, 1, 2, 3, 4, ...`.
    #[test]
    fn test_flatten() {
        // The number of rows in the final matrix.
        let nrows = 5;
        // The dimensions in each sub-matrix.
        let sub_dims = [1, 2, 3, 4, 5];
        // The prefix sum of the sub dimensions.
        let prefix_sum: Vec<usize> = sub_dims
            .iter()
            .scan(0, |state, i| {
                let this = *state;
                *state += *i;
                Some(this)
            })
            .collect();

        let dim: usize = sub_dims.iter().sum();

        // Create the sub matrices.
        let matrices: Vec<Matrix<usize>> = std::iter::zip(sub_dims.iter(), prefix_sum.iter())
            .map(|(&this_dim, &offset)| {
                let mut m = Matrix::new(0, nrows, this_dim);
                for r in 0..nrows {
                    for c in 0..this_dim {
                        m[(r, c)] = dim * r + offset + c;
                    }
                }
                m
            })
            .collect();

        let flattened = flatten(&matrices, nrows, dim);
        // Check that the output is correct.
        for (i, v) in flattened.as_slice().iter().enumerate() {
            assert_eq!(*v, i, "failed at index {i}");
        }
    }

    struct DatasetBuilder {
        nclusters: usize,
        cluster_size: usize,
        step_between_clusters: f32,
    }

    struct ClusteredDataset {
        data: Matrix<f32>,
        // The pre-configured center point for the manufactured clusters.
        centers: Matrix<f32>,
    }

    impl DatasetBuilder {
        fn build<R>(
            &self,
            schema: crate::views::ChunkOffsetsView<'_>,
            rng: &mut R,
        ) -> ClusteredDataset
        where
            R: Rng,
        {
            let ndata = self.nclusters * self.cluster_size;

            // Start the clustering points at a different location for each chunk.
            // The starting offset is chosen from this distribution.
            let offsets_distribution = Uniform::<f32>::new(-100.0, 100.0).unwrap();

            // The perturbation for vectors within a cluster - all centered around some
            // mean.
            let perturbation_distribution = rand_distr::StandardNormal;

            // Indices that we use to shuffle the order of elements in the dataset.
            let mut indices: Vec<usize> = (0..ndata).collect();

            // Construct the dataset in pieces.
            let (pieces, centers): (Vec<_>, Vec<_>) = (0..schema.len())
                .map(|chunk| {
                    let dim = schema.at(chunk).len();

                    let mut initial = Matrix::new(0.0, ndata, dim);
                    let mut centers = Matrix::new(0.0, self.nclusters, 1);

                    // The starting offset for clusters.
                    let offset = offsets_distribution.sample(rng);

                    // Create a dataset with `nclusters`, each cluster
                    for cluster in 0..self.nclusters {
                        let this_offset = offset + (cluster as f32 * self.step_between_clusters);
                        centers[(cluster, 0)] = this_offset;

                        for element in 0..self.cluster_size {
                            let row = initial.row_mut(cluster * self.cluster_size + element);
                            for r in row.iter_mut() {
                                let perturbation: f32 = perturbation_distribution.sample(rng);
                                *r = this_offset + perturbation;
                            }
                        }
                    }

                    // Shuffle the dataset.
                    indices.shuffle(rng);
                    let mut piece = Matrix::new(0.0, ndata, dim);
                    for (dst, src) in indices.iter().enumerate() {
                        piece.row_mut(dst).copy_from_slice(initial.row(*src));
                    }
                    (piece, centers)
                })
                .unzip();

            ClusteredDataset {
                data: flatten(&pieces, ndata, schema.dim()),
                centers: flatten(&centers, self.nclusters, schema.len()),
            }
        }
    }

    fn broadcast_distance(x: &[f32], y: f32) -> f32 {
        x.iter()
            .map(|i| {
                let d = *i - y;
                d * d
            })
            .sum()
    }

    // Happy Path check - varying over parallelism.
    fn test_pq_training_happy_path(parallelism: Parallelism) {
        let mut rng = StdRng::seed_from_u64(0x749cb951cf960384);
        let builder = DatasetBuilder {
            nclusters: 16,
            cluster_size: 20,
            // NOTE: We need to keep the step between clusters fairly large to ensure that
            // kmeans++ adequately initializes.
            step_between_clusters: 20.0,
        };

        let ncenters = builder.nclusters;

        let offsets = [0, 2, 3, 8, 12, 16];
        let schema = crate::views::ChunkOffsetsView::new(&offsets).unwrap();
        let dataset = builder.build(schema, &mut rng);

        let trainer = LightPQTrainingParameters::new(ncenters, 6);

        let quantizer = trainer
            .train(
                dataset.data.as_view(),
                schema,
                parallelism,
                &StdRngBuilder::new(StandardUniform {}.sample(&mut rng)),
                &DontCancel,
            )
            .unwrap();

        // Now that we have trained the quantizer - we need to double check that the chosen
        // centroids match what we expect.
        //
        // To do this - we loop through the centroids that training picked. We match the
        // centroids with one of the known centers in our clustering.
        //
        // We perform two main checks:
        //
        // 1. We ensure that the quantizer's center actually aligns with a cluster (i.e.,
        //    training did not invent values out of thin air).
        // 2. Every clustering in the original dataset has a representative in the quantizer.
        assert_eq!(quantizer.dim, schema.dim());
        assert_eq!(quantizer.ncenters, ncenters);
        assert_eq!(quantizer.pivots.len(), schema.len());
        for (i, pivot) in quantizer.pivots.iter().enumerate() {
            // Make sure the pivot has the correct dimension.
            assert_eq!(
                pivot.ncols(),
                schema.at(i).len(),
                "center {i} has the incorrect number of columns"
            );
            assert_eq!(pivot.nrows(), ncenters);

            // Start matching pivots to expected centers.
            let mut seen: Vec<bool> = (0..dataset.centers.nrows()).map(|_| false).collect();
            for row in pivot.row_iter() {
                let mut min_distance = f32::MAX;
                let mut min_index = 0;
                for c in 0..dataset.centers.nrows() {
                    let distance = broadcast_distance(row, dataset.centers[(c, i)]);
                    if distance < min_distance {
                        min_distance = distance;
                        min_index = c;
                    }
                }

                // Does the minimum distance suggest that we are inside a cluster.
                assert!(
                    min_distance < 1.0,
                    "got a minimum distance of {}, pivot = {}. Row = {:?}",
                    min_distance,
                    i,
                    row,
                );

                // Mark this index as seen.
                let seen_before = &mut seen[min_index];
                assert!(
                    !*seen_before,
                    "cluster {} has more than one assignment",
                    min_index
                );
                *seen_before = true;
            }

            // Make sure that all clusters were seen.
            assert!(seen.iter().all(|i| *i), "not all clusters were seen");
        }

        // Check `flatten`.
        let flattened = quantizer.flatten();
        assert_eq!(
            &flattened,
            flatten(&quantizer.pivots, quantizer.ncenters, quantizer.dim).as_slice()
        );
    }

    #[test]
    fn test_pq_training_happy_path_sequential() {
        test_pq_training_happy_path(Parallelism::Sequential);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_pq_training_happy_path_parallel() {
        test_pq_training_happy_path(Parallelism::Rayon);
    }

    // A canceler that cancels after a set number of invocations.
    struct CancelAfter {
        counter: AtomicUsize,
        after: usize,
    }

    impl CancelAfter {
        fn new(after: usize) -> Self {
            Self {
                counter: AtomicUsize::new(0),
                after,
            }
        }
    }

    impl Cancelation for CancelAfter {
        fn should_cancel(&self) -> bool {
            let v = self.counter.fetch_add(1, Ordering::Relaxed);
            v >= self.after
        }
    }

    #[test]
    fn test_cancel() {
        let mut rng = StdRng::seed_from_u64(0xb85352d38cc5353b);
        let builder = DatasetBuilder {
            nclusters: 16,
            cluster_size: 20,
            // NOTE: We need to keep the step between clusters fairly large to ensure that
            // kmeans++ adequately initializes.
            step_between_clusters: 20.0,
        };

        let offsets = [0, 2, 3, 8, 12, 16];
        let schema = crate::views::ChunkOffsetsView::new(&offsets).unwrap();
        let dataset = builder.build(schema, &mut rng);

        let trainer = LightPQTrainingParameters::new(builder.nclusters, 6);

        for after in 0..10 {
            let parallelism = [
                Parallelism::Sequential,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon,
            ];

            for par in parallelism {
                let result = trainer.train(
                    dataset.data.as_view(),
                    schema,
                    par,
                    &StdRngBuilder::new(StandardUniform {}.sample(&mut rng)),
                    &CancelAfter::new(after),
                );
                assert!(result.is_err(), "expected the operation to be canceled");
                let err = result.unwrap_err();
                assert!(
                    err.was_canceled(),
                    "expected the failure reason to be cancellation"
                );
            }
        }
    }

    // In this test - we ensure that clustering succeeds even if the number of requested
    // pivots exceeds the number of dataset items.
    #[test]
    fn tests_succeeded_with_too_many_pivots() {
        let data = Matrix::<f32>::new(1.0, 10, 5);
        let offsets: Vec<usize> = vec![0, 1, 4, 5];

        let trainer = LightPQTrainingParameters::new(2 * data.nrows(), 6);

        let quantizer = trainer
            .train(
                data.as_view(),
                crate::views::ChunkOffsetsView::new(&offsets).unwrap(),
                Parallelism::Sequential,
                &StdRngBuilder::new(0),
                &DontCancel,
            )
            .unwrap();

        // We are in the special position to actually know how this will behave.
        // Since the input dataset lacks diversity, there should only have been a single
        // pivot actually selected.
        //
        // All the rest should be zero.
        let flat = flatten(&quantizer.pivots, quantizer.ncenters, quantizer.dim);

        assert!(
            flat.row(0).iter().all(|i| *i == 1.0),
            "expected pivot 0 to be the non-zero pivot"
        );

        for (i, row) in flat.row_iter().enumerate() {
            // skip the first row.
            if i == 0 {
                continue;
            }

            assert!(
                row.iter().all(|j| *j == 0.0),
                "expected pivot {i} to be all zeros"
            );
        }
    }

    #[test]
    fn test_infinity_and_nan_is_not_recoverable() {
        let num_trials = 10;
        let nrows = 10;
        let ncols = 5;

        let offsets: Vec<usize> = vec![0, 1, 4, 5];

        let trainer = LightPQTrainingParameters::new(nrows, 6);

        let row_distribution = Uniform::new(0, nrows).unwrap();
        let col_distribution = Uniform::new(0, ncols).unwrap();
        let mut rng = StdRng::seed_from_u64(0xe746cfebba2d7e35);

        for trial in 0..num_trials {
            let context = lazy_format!("trial {} of {}", trial + 1, num_trials);

            let r = row_distribution.sample(&mut rng);
            let c = col_distribution.sample(&mut rng);

            let check_result = |r: Result<_, PQTrainingError>| {
                assert!(
                    r.is_err(),
                    "expected error due to infinities/NaN -- {}",
                    context
                );
                let err = r.unwrap_err();
                assert!(!err.was_canceled());
                assert!(format(&err).contains("infinity"));
            };

            let mut data = Matrix::<f32>::new(1.0, nrows, ncols);

            // Positive Infinity
            data[(r, c)] = f32::INFINITY;
            let result = trainer.train(
                data.as_view(),
                crate::views::ChunkOffsetsView::new(&offsets).unwrap(),
                Parallelism::Sequential,
                &StdRngBuilder::new(0),
                &DontCancel,
            );
            check_result(result);

            // Positive Infinity
            data[(r, c)] = f32::NEG_INFINITY;
            let result = trainer.train(
                data.as_view(),
                crate::views::ChunkOffsetsView::new(&offsets).unwrap(),
                Parallelism::Sequential,
                &StdRngBuilder::new(0),
                &DontCancel,
            );
            check_result(result);

            // NaN
            data[(r, c)] = f32::NAN;
            let result = trainer.train(
                data.as_view(),
                crate::views::ChunkOffsetsView::new(&offsets).unwrap(),
                Parallelism::Sequential,
                &StdRngBuilder::new(0),
                &DontCancel,
            );
            check_result(result);
        }
    }
}
