/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Common utilities for testing PQ-based distance computations.
use approx::assert_relative_eq;
use diskann::utils::IntoUsize;
use diskann_utils::views;
use diskann_vector::{
    Half, PreprocessedDistanceFunction, PureDistanceFunction,
    distance::{Cosine, InnerProduct, SquaredL2},
};
use rand::{Rng, SeedableRng, distr::Distribution};
use rand_distr::{Normal, Uniform};

use crate::model::{FixedChunkPQTable, pq::calculate_chunk_offsets_auto};

/// We need a way to generate random queries.
///
/// The `TestDistribution` is the mechanism by which we can define distributions for
/// the various supported data types.
///
/// Eventually, there should be more rigorous support for this ...
pub(crate) trait TestDistribution: Sized + Copy {
    fn generate<R>(dim: usize, rng: &mut R) -> Vec<Self>
    where
        R: Rng;
}

impl TestDistribution for f32 {
    fn generate<R>(dim: usize, rng: &mut R) -> Vec<Self>
    where
        R: Rng,
    {
        let distribution = Normal::<f32>::new(0.0, 10.0).unwrap();
        (0..dim).map(|_| distribution.sample(rng)).collect()
    }
}

impl TestDistribution for Half {
    fn generate<R>(dim: usize, rng: &mut R) -> Vec<Self>
    where
        R: Rng,
    {
        let distribution = Normal::<f32>::new(0.0, 10.0).unwrap();
        (0..dim)
            .map(|_| Half::from_f32(distribution.sample(rng)))
            .collect()
    }
}

impl TestDistribution for i8 {
    fn generate<R>(dim: usize, rng: &mut R) -> Vec<Self>
    where
        R: Rng,
    {
        let distribution = rand::distr::StandardUniform {};
        (0..dim).map(|_| distribution.sample(rng)).collect()
    }
}

impl TestDistribution for u8 {
    fn generate<R>(dim: usize, rng: &mut R) -> Vec<Self>
    where
        R: Rng,
    {
        let distribution = rand::distr::StandardUniform {};
        (0..dim).map(|_| distribution.sample(rng)).collect()
    }
}

/// Relative and absolute error.
#[derive(Clone, Copy)]
pub(crate) struct RelativeAndAbsolute {
    pub(crate) relative: f32,
    pub(crate) absolute: f32,
}

/// Configuration for the PQ table.
#[derive(Clone, Copy)]
pub(crate) struct TableConfig {
    pub(crate) dim: usize,
    pub(crate) pq_chunks: usize,
    pub(crate) num_pivots: usize,
    // The starting value for chunk 0, pivot 0.
    pub(crate) start_value: f32,
    // Flag to initialize both the transformation matrix and the centroid.
    pub(crate) use_opq: bool,
}

/// With reference to the docstring for `seed_pivot_table`, this function generates
/// the expected reconstructed vector.
///
/// Useful for obtaining values for expected results.
pub(crate) fn generate_expected_vector(
    code: &[u8],
    offsets: &[usize],
    start_value: f32,
) -> Vec<f32> {
    // Since we're in testing mode, ensure that the expected relationship between the
    // PQ code and the offset mapping holds.
    assert_eq!(code.len() + 1, offsets.len());

    let mut v = Vec::new();
    for (i, c) in code.iter().enumerate() {
        let len = offsets[i + 1] - offsets[i];
        for _ in 0..len {
            // We expect the conversion to `f32` to be lossless.
            v.push(start_value + ((i + c.into_usize()) as f32))
        }
    }
    v
}

/// To test the implementation of these distances, we need a way to seed the source
/// pivot table with known contents.
///
/// The layout of the pivot table will look like this:
///
///      chunk 0          chunk 1       ...        chunk K
///
/// | S    S    ... | S+1   S+1   ... | ... | S+K    S+K    ... |   pivot 0
/// | S+1  S+1  ... | S+2   S+2   ... | ... | S+K+1  S+K+1  ... |   pivot 1
/// | S+2  S+2  ... | S+3   S+3   ... | ... | S+K+2  S+K+2  ... |   pivot 2
/// |     ...       |       ...       | ... |        ...        |     ...
/// | S+N  S+N  ... | S+N+1 S+N+1 ... | ... | S+K+N  S+K+N  ... |   pivot N
///
/// where
///
/// * S: The configured start value for chunk 0, pivot 0 (i.e., `config.start_value`)
/// * K + 1: The number of PQ chunks
/// * N + 1: The number of PQ Pivots
pub(crate) fn seed_pivot_table(config: TableConfig) -> FixedChunkPQTable {
    // Get the chunk offsets for the selected dimension and bytes.
    let offsets = calculate_chunk_offsets_auto(config.dim, config.pq_chunks);

    // Create the pivot table following the schema described in the docstring.
    let mut pivots = Vec::<f32>::new();
    for i in 0..config.num_pivots {
        for j in 0..config.pq_chunks {
            let start = offsets[j];
            let stop = offsets[j + 1];

            // Primitive Conversion: We expect the integer values seen here to be small
            // enought to be losslessly convertiblt to `f32`.
            let val = config.start_value + ((i + j) as f32);
            for _ in 0..(stop - start) {
                pivots.push(val);
            }
        }
    }

    assert_eq!(pivots.len(), config.dim * config.num_pivots);

    let (centroid, matrix) = if config.use_opq {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1c3e6b3951ac5b73);
        let dist = Normal::<f32>::new(0.0, 1.0).unwrap();

        let centroid = (0..config.dim).map(|_| dist.sample(&mut rng)).collect();
        let matrix = views::Matrix::new(
            views::Init(|| dist.sample(&mut rng)),
            config.dim,
            config.dim,
        );
        (centroid, Some(matrix))
    } else {
        (vec![0.0f32; config.dim], None)
    };

    FixedChunkPQTable::new(
        config.dim,
        pivots.into(),
        centroid.into(),
        offsets.into(),
        matrix.map(|x| x.into_inner()),
    )
    .unwrap()
}

/// Generate a random PQ code spanning the requested number of pivots and chunks.
///
/// # Panics
///
/// Panics if the range `[0..num_pivots)` is either empty or contains elements that
/// are not representable as 8-bit unsigned integers.
pub(crate) fn generate_random_code<R>(
    num_pivots: usize,
    num_pq_chunks: usize,
    rng: &mut R,
) -> Vec<u8>
where
    R: Rng,
{
    // Ensure that the number of pivots *actually* is encodable as an 8-bit unsigned
    // integer.
    assert!(num_pivots != 0);
    let num_pivots: u8 = (num_pivots - 1).try_into().unwrap();
    let dist = Uniform::try_from(0..=num_pivots).unwrap();
    (0..num_pq_chunks).map(|_| dist.sample(rng)).collect()
}

/// Testing L2 is tricky for the following reasons.
///
/// First, we need to ensure that the centroid is removed from the properly from the
/// query vector.
///
/// Next, if OPQ is used, we need to ensure that the matrix multiplication is applied
/// to the query vector before we can obtain expected results.
pub(super) fn test_l2_inner<'a, T, F, R>(
    create: impl Fn(&'a FixedChunkPQTable, &[T]) -> F,
    table: &'a FixedChunkPQTable,
    num_trials: usize,
    config: TableConfig,
    rng: &mut R,
    errors: RelativeAndAbsolute,
) where
    T: Into<f32> + TestDistribution,
    F: for<'any> PreprocessedDistanceFunction<&'any [u8], f32>,
    R: Rng,
{
    for _ in 0..num_trials {
        let input: Vec<T> = T::generate(config.dim, rng);
        let mut input_f32: Vec<f32> = input.iter().map(|x| (*x).into()).collect();

        table.preprocess_query(&mut input_f32);

        let computer = create(table, &input);
        for _ in 0..num_trials {
            let code = generate_random_code(config.num_pivots, config.pq_chunks, rng);
            let expected_vector =
                generate_expected_vector(&code, table.get_chunk_offsets(), config.start_value);

            let got = computer.evaluate_similarity(&code);
            let expected = SquaredL2::evaluate(input_f32.as_slice(), expected_vector.as_slice());

            // This doesn't need to be exact due to rounding differences.
            assert_relative_eq!(
                got,
                expected,
                epsilon = errors.absolute,
                max_relative = errors.relative
            );
        }
    }
}

pub(super) fn test_ip_inner<'a, T, F, R>(
    create: impl Fn(&'a FixedChunkPQTable, &[T]) -> F,
    table: &'a FixedChunkPQTable,
    num_trials: usize,
    config: TableConfig,
    rng: &mut R,
    errors: RelativeAndAbsolute,
) where
    T: Into<f32> + TestDistribution,
    F: for<'any> PreprocessedDistanceFunction<&'any [u8], f32>,
    R: Rng,
{
    for _ in 0..num_trials {
        let input: Vec<T> = T::generate(config.dim, rng);
        let input_f32: Vec<f32> = input.iter().map(|x| (*x).into()).collect();

        let computer = create(table, &input);
        for _ in 0..num_trials {
            let code = generate_random_code(config.num_pivots, config.pq_chunks, rng);
            let expected_vector =
                generate_expected_vector(&code, table.get_chunk_offsets(), config.start_value);

            let got = computer.evaluate_similarity(&code);
            let expected = InnerProduct::evaluate(input_f32.as_slice(), expected_vector.as_slice());

            // Allow for some variation due to different rounding orders.
            assert_relative_eq!(
                got,
                expected,
                epsilon = errors.absolute,
                max_relative = errors.relative
            );
        }
    }
}

pub(super) fn test_cosine_inner<'a, T, F, R>(
    create: impl Fn(&'a FixedChunkPQTable, &[T]) -> F,
    table: &'a FixedChunkPQTable,
    num_trials: usize,
    config: TableConfig,
    rng: &mut R,
    errors: RelativeAndAbsolute,
) where
    T: Into<f32> + TestDistribution,
    F: for<'any> PreprocessedDistanceFunction<&'any [u8], f32>,
    R: Rng,
{
    for _ in 0..num_trials {
        let input: Vec<T> = T::generate(config.dim, rng);
        let input_f32: Vec<f32> = input.iter().map(|x| (*x).into()).collect();

        let computer = create(table, &input);
        for _ in 0..num_trials {
            let code = generate_random_code(config.num_pivots, config.pq_chunks, rng);
            let expected_vector =
                generate_expected_vector(&code, table.get_chunk_offsets(), config.start_value);

            let got = computer.evaluate_similarity(&code);
            let expected = Cosine::evaluate(input_f32.as_slice(), expected_vector.as_slice());

            // Allow for some variation due to different rounding orders.
            assert_relative_eq!(
                got,
                expected,
                epsilon = errors.absolute,
                max_relative = errors.relative
            );
        }
    }
}
