/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::{SeedableRng, rngs::StdRng};
use std::{fmt::Display, num::NonZeroUsize};
use thiserror::Error;

use diskann_utils::{
    sampling::WithApproximateNorm,
    views::{Matrix, MatrixView},
};

/// 'StartPointStrategy' is an enum that represents the different strategies to select
///  the starting points for the clustering algorithm.
///
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StartPointStrategy {
    /// Randomly select vector(s) with given norm as starting points with seed provided.
    /// Requires the norm (f32), number of samples (usize), and random seed (u64) to be provided.
    RandomVectors {
        norm: f32,
        nsamples: NonZeroUsize,
        seed: u64,
    },

    /// Sample data from the dataset with seed provided.
    /// Requires number of samples (usize) and random seed (u64) to be provided.
    RandomSamples { nsamples: NonZeroUsize, seed: u64 },

    /// Use the medoid as the starting point. Can select only one starting point.
    Medoid,

    /// Use the Latin Hypercube sampling method to select the starting points.
    /// Requires number of samples (usize) and random seed (u64) to be provided.
    LatinHyperCube { nsamples: NonZeroUsize, seed: u64 },

    /// Use the first vector in the dataset as the starting point. Can select only one starting point.
    FirstVector,
}

pub trait SampleableForStart:
    diskann_utils::sampling::medoid::ComputeMedoid
    + diskann_utils::sampling::latin_hypercube::SampleLatinHyperCube
    + diskann_utils::sampling::random::RoundFromf32
{
}

impl<T> SampleableForStart for T where
    T: diskann_utils::sampling::medoid::ComputeMedoid
        + diskann_utils::sampling::latin_hypercube::SampleLatinHyperCube
        + diskann_utils::sampling::random::RoundFromf32
{
}

#[derive(Debug, Clone, Copy, Error)]
pub enum StartPointError {
    #[error(
        "Not enough input data was supplied, {} samples were requested by {} were supplied",
        requested,
        found
    )]
    NotEnoughTrainingData { requested: usize, found: usize },

    #[error("Error getting row_id {} from training data matrix view", row_id)]
    MatrixRowError { row_id: usize },
}

impl StartPointStrategy {
    pub fn count(&self) -> usize {
        match self {
            StartPointStrategy::RandomVectors { nsamples, .. } => nsamples.get(),
            StartPointStrategy::RandomSamples { nsamples, .. } => nsamples.get(),
            StartPointStrategy::Medoid => 1,
            StartPointStrategy::LatinHyperCube { nsamples, .. } => nsamples.get(),
            StartPointStrategy::FirstVector => 1,
        }
    }

    pub fn compute<T>(&self, train_data: MatrixView<'_, T>) -> Result<Matrix<T>, StartPointError>
    where
        T: Copy + SampleableForStart + diskann_utils::sampling::WithApproximateNorm,
    {
        match self {
            StartPointStrategy::RandomSamples { nsamples, seed } => {
                if nsamples.get() > train_data.nrows() {
                    return Err(StartPointError::NotEnoughTrainingData {
                        requested: nsamples.get(),
                        found: train_data.nrows(),
                    });
                }

                let mut rng = StdRng::seed_from_u64(*seed);

                let indices =
                    rand::seq::index::sample(&mut rng, train_data.nrows(), nsamples.get());

                let mut points = Matrix::new(T::default(), nsamples.get(), train_data.ncols());
                std::iter::zip(points.row_iter_mut(), indices).for_each(|(dst, src)| {
                    dst.copy_from_slice(train_data.row(src));
                });

                Ok(points)
            }
            StartPointStrategy::Medoid => Ok(Matrix::row_vector(
                T::compute_medoid(train_data.as_view()).into(),
            )),
            StartPointStrategy::RandomVectors {
                norm,
                nsamples,
                seed,
            } => {
                let mut rng = StdRng::seed_from_u64(*seed);
                let dim = train_data.ncols();
                let mut points = Matrix::new(T::default(), nsamples.get(), dim);
                points.row_iter_mut().for_each(|row| {
                    row.copy_from_slice(&WithApproximateNorm::with_approximate_norm(
                        dim, *norm, &mut rng,
                    ))
                });

                Ok(points)
            }
            StartPointStrategy::LatinHyperCube { nsamples, seed } => Ok(T::sample_latin_hypercube(
                train_data,
                nsamples.get(),
                Some(*seed),
            )),
            StartPointStrategy::FirstVector => match train_data.get_row(0) {
                Some(row) => Ok(Matrix::row_vector(row.into())),
                None => Err(StartPointError::NotEnoughTrainingData {
                    requested: 1,
                    found: 0,
                }),
            },
        }
    }
}

impl Display for StartPointStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartPointStrategy::RandomVectors {
                norm,
                nsamples,
                seed,
            } => {
                write!(
                    f,
                    "RandomVectors(norm: {}, nsamples: {}, seed: {})",
                    *norm, nsamples, seed
                )
            }
            StartPointStrategy::RandomSamples { nsamples, seed } => {
                write!(f, "RandomSamples(nsamples: {}, seed: {})", nsamples, seed)
            }
            StartPointStrategy::Medoid => write!(f, "Medoid"),
            StartPointStrategy::LatinHyperCube { nsamples, seed } => {
                write!(f, "LatinHyperCube(nsamples: {}, seed: {})", nsamples, seed)
            }
            StartPointStrategy::FirstVector => write!(f, "FirstVector"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn test_num_start_points() {
        let strategy = StartPointStrategy::RandomVectors {
            norm: 1.0,
            nsamples: NonZeroUsize::new(5).unwrap(),
            seed: 42,
        };
        assert_eq!(strategy.count(), 5);
    }
    #[test]
    fn test_num_start_points_random_samples() {
        let strategy = StartPointStrategy::RandomSamples {
            nsamples: NonZeroUsize::new(10).unwrap(),
            seed: 42,
        };
        assert_eq!(strategy.count(), 10);
    }
    #[test]
    fn test_num_start_points_medoid() {
        let strategy = StartPointStrategy::Medoid;
        assert_eq!(strategy.count(), 1);
    }
    #[test]
    fn test_num_start_points_latin_hypercube() {
        let strategy = StartPointStrategy::LatinHyperCube {
            nsamples: NonZeroUsize::new(6).unwrap(),
            seed: 42,
        };
        assert_eq!(strategy.count(), 6);
    }
    #[test]
    fn test_num_start_points_first_vector() {
        let strategy = StartPointStrategy::FirstVector;
        assert_eq!(strategy.count(), 1);
    }

    #[test]
    fn test_display_medoid() {
        let strategy = StartPointStrategy::Medoid;
        let mut buffer = String::new();
        write!(&mut buffer, "{}", strategy).unwrap();
        assert_eq!(buffer, "Medoid");
    }

    #[test]
    fn test_display_first_vector() {
        let strategy = StartPointStrategy::FirstVector;
        let mut buffer = String::new();
        write!(&mut buffer, "{}", strategy).unwrap();
        assert_eq!(buffer, "FirstVector");
    }

    #[test]
    fn test_display_random_vectors() {
        let strategy = StartPointStrategy::RandomVectors {
            norm: 1.2,
            nsamples: NonZeroUsize::new(10).unwrap(),
            seed: 42,
        };
        let mut buffer = String::new();
        write!(&mut buffer, "{}", strategy).unwrap();
        assert_eq!(buffer, "RandomVectors(norm: 1.2, nsamples: 10, seed: 42)");
    }

    #[test]
    fn test_display_random_samples() {
        let strategy = StartPointStrategy::RandomSamples {
            nsamples: NonZeroUsize::new(15).unwrap(),
            seed: 99,
        };
        let mut buffer = String::new();
        write!(&mut buffer, "{}", strategy).unwrap();
        assert_eq!(buffer, "RandomSamples(nsamples: 15, seed: 99)");
    }

    #[test]
    fn test_display_random_hypercube() {
        let strategy = StartPointStrategy::LatinHyperCube {
            nsamples: NonZeroUsize::new(15).unwrap(),
            seed: 42,
        };
        let mut buffer = String::new();
        write!(&mut buffer, "{}", strategy).unwrap();
        assert_eq!(buffer, "LatinHyperCube(nsamples: 15, seed: 42)");
    }

    #[test]
    fn test_start_get_first() {
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let matrix = Matrix::try_from(data.into(), 4, 3).unwrap();
        let strategy = StartPointStrategy::FirstVector;
        let start_points = strategy.compute(matrix.as_view()).unwrap();
        assert_eq!(start_points.nrows(), 1);
        assert_eq!(start_points.ncols(), 3);
        assert_eq!(start_points.get_row(0), matrix.get_row(0));
    }

    #[test]
    fn test_start_get_medoid() {
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let matrix = Matrix::try_from(data.into(), 4, 3).unwrap();
        let strategy = StartPointStrategy::Medoid;
        let start_points = strategy.compute(matrix.as_view()).unwrap();
        assert_eq!(start_points.nrows(), 1);
        assert_eq!(start_points.ncols(), 3);
        // The medoid in this simple case should be the second vector
        assert_eq!(start_points.get_row(0).unwrap(), &[4.0f32, 5.0, 6.0]);
    }

    #[test]
    fn test_start_get_random_vectors() {
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let matrix = Matrix::try_from(data.into(), 4, 3).unwrap();
        let strategy = StartPointStrategy::RandomVectors {
            norm: 5.0,
            nsamples: NonZeroUsize::new(2).unwrap(),
            seed: 42,
        };
        let start_points = strategy.compute(matrix.as_view()).unwrap();
        assert_eq!(start_points.nrows(), 2);
        assert_eq!(start_points.ncols(), 3);
        for i in 0..2 {
            let row = start_points.get_row(i).unwrap();
            let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm - 5.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_start_get_random_samples() {
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let matrix = Matrix::try_from(data.into(), 4, 3).unwrap();
        let strategy = StartPointStrategy::RandomSamples {
            nsamples: NonZeroUsize::new(2).unwrap(),
            seed: 42,
        };
        let start_points = strategy.compute(matrix.as_view()).unwrap();
        assert_eq!(start_points.nrows(), 2);
        assert_eq!(start_points.ncols(), 3);
        for i in 0..2 {
            let row = start_points.get_row(i).unwrap();
            assert!(matrix.row_iter().any(|r| r == row));
        }
    }

    #[test]
    fn test_start_get_latin_hypercube() {
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let matrix = Matrix::try_from(data.into(), 4, 3).unwrap();
        let strategy = StartPointStrategy::LatinHyperCube {
            nsamples: NonZeroUsize::new(2).unwrap(),
            seed: 42,
        };
        let start_points = strategy.compute(matrix.as_view()).unwrap();
        assert_eq!(start_points.nrows(), 2);
        assert_eq!(start_points.ncols(), 3);
    }
}
