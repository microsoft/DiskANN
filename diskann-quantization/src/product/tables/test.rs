/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// A collection of test helpers to ensure uniformity across tables.

use std::num::NonZeroUsize;

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{PureDistanceFunction, distance};
#[cfg(not(miri))]
use rand::seq::IndexedRandom;
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

use crate::{
    product::tables::BasicTable,
    test_util::Check,
    traits::CompressInto,
    views::{self, ChunkOffsets, ChunkOffsetsView},
};

//////////////////////
// Distance Helpers //
//////////////////////

/// To test the implementation of distances, we need a way to seed the source pivot table
/// with known contents.
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
/// * S: The configured start value for chunk 0, pivot 0 (i.e., [`Self::start`])
/// * K + 1: The number of PQ chunks ([`Self::chunks`]).
/// * N + 1: The number of PQ Pivots ([`Self::pivots`]).
#[derive(Debug, Clone)]
pub(super) struct DistanceTestTable {
    /// The chunking schemal
    pub(super) offsets: ChunkOffsets,
    /// The number of pivots per chunk.
    pub(super) pivots: usize,
    /// The starting value for chunk 0, pivot 0.
    pub(super) start: f32,
}

/// The position within the chunking scheme.
#[derive(Debug, Clone, Copy)]
struct Location {
    /// The chunk number.
    chunk: usize,
    /// The pivot.
    pivot: usize,
}

#[derive(Debug, Clone)]
pub(super) struct UniformFloat(Uniform<usize>);

impl UniformFloat {
    pub(super) fn new(low: usize, high: usize) -> Result<Self, rand::distr::uniform::Error> {
        Uniform::new(low, high).map(Self)
    }

    pub(super) fn new_inclusive(
        low: usize,
        high: usize,
    ) -> Result<Self, rand::distr::uniform::Error> {
        Uniform::new_inclusive(low, high).map(Self)
    }
}

impl Distribution<f32> for UniformFloat {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        self.0.sample(rng) as f32
    }
}

impl DistanceTestTable {
    pub(super) fn new(dim: usize, chunks: usize, pivots: usize, start: f32) -> Self {
        Self {
            offsets: ChunkOffsets::partition(
                NonZeroUsize::new(dim).unwrap(),
                NonZeroUsize::new(chunks).unwrap(),
            )
            .unwrap(),
            pivots,
            start,
        }
    }

    /// This is mainly a convenience so we don't always have to import `StdRng` and
    /// `SeedableRng` and all that jazz.
    pub(super) fn rng(&self, seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    pub(super) fn offsets(&self) -> ChunkOffsetsView<'_> {
        self.offsets.as_view()
    }

    pub(super) fn chunks(&self) -> usize {
        self.offsets.len()
    }

    pub(super) fn dim(&self) -> usize {
        self.offsets.dim()
    }

    pub(super) fn pivots(&self) -> usize {
        self.pivots
    }

    fn value(&self, loc: Location) -> f32 {
        (loc.chunk + loc.pivot) as f32 + self.start
    }

    pub(super) fn basic_table(&self) -> BasicTable {
        // This creates a base vector like
        // |  chunk 0  |  chunk 1  |  ... |  chunk K  |
        // | 0 0 ... 0 | 1 1 ... 1 |  ... | K K ... K |
        let mut base = Vec::<f32>::new();
        for i in 0..self.offsets.len() {
            let v = (i as f32) + self.start;
            for _ in self.offsets.at(i) {
                base.push(v);
            }
        }

        // Use our base vector to build the rest of the pivot matrix.
        let pivots = rowmajor::Owned::from_fn(self.pivots(), self.dim(), |rc| {
            (rc.row as f32) + base[rc.col]
        });

        BasicTable::new(pivots, self.offsets().to_owned()).unwrap()
    }

    pub(super) fn expected_vector_into(&self, v: &mut [f32], codes: &[u8]) {
        assert_eq!(v.len(), self.dim());
        assert_eq!(codes.len(), self.chunks());

        let mut i = 0;
        for (chunk, pivot) in codes.iter().copied().enumerate() {
            let pivot = usize::from(pivot);
            assert!(pivot < self.pivots());
            let loc = Location { chunk, pivot };

            for _ in self.offsets.at(chunk) {
                v[i] = self.value(loc);
                i += 1;
            }
        }
    }

    pub(super) fn expected_vector(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.chunks());
        let mut v = vec![0.0; self.dim()];
        self.expected_vector_into(&mut v, codes);
        v
    }

    pub(super) fn drive(
        &self,
        num_trials: usize,
        rng: &mut StdRng,
        f: &mut dyn FnMut(&[u8], &[f32], std::fmt::Arguments<'_>),
        ctx: std::fmt::Arguments<'_>,
    ) {
        // Run two fixed trials - one with all zeros and one with the max setting.
        //
        // Then we perform random trials.
        let mut codes = vec![0u8; self.chunks()];
        let mut vector = vec![0.0; self.dim()];
        self.expected_vector_into(&mut vector, &codes);
        f(&codes, &mut vector, format_args!("{ctx}, all zeros"));

        let max = u8::try_from(self.pivots() - 1).unwrap();
        codes.iter_mut().for_each(|c| *c = max);
        self.expected_vector_into(&mut vector, &codes);
        f(&codes, &mut vector, format_args!("{ctx}, all {max}"));

        // Begin random trials.
        let dist = Uniform::new(0, self.pivots()).unwrap();
        for trial in 0..num_trials {
            codes
                .iter_mut()
                .for_each(|c| *c = u8::try_from(dist.sample(rng)).unwrap());
            self.expected_vector_into(&mut vector, &codes);
            f(
                &codes,
                &mut vector,
                format_args!("{ctx}, trial {} of {}", trial + 1, num_trials),
            );
        }
    }

    pub(super) fn drive_unary(
        &self,
        num_trials: usize,
        rng: &mut StdRng,
        check: Check,
        reference: &dyn Fn(&[f32]) -> f32,
        dut: &mut dyn FnMut(&[u8]) -> f32,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let mut f = |codes: &[u8], vector: &[f32], ctx: std::fmt::Arguments<'_>| {
            let expected = reference(vector);
            let got = dut(codes);

            if let Err(reason) = check.check(got, expected) {
                panic!("Check failed: {} -- {}", reason, ctx);
            }
        };

        self.drive(num_trials, rng, &mut f, ctx)
    }

    pub(super) fn drive_query_like(
        &self,
        num_queries: usize,
        num_trials: usize,
        rng: &mut StdRng,
        check: Check,
        f: &dyn Fn(&[f32], &[f32]) -> f32,
        dut: &mut dyn QueryLike,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let dist = UniformFloat::new(0, self.chunks() + self.pivots()).unwrap();
        let mut query = vec![0.0f32; self.dim()];
        for trial in 0..num_queries {
            query.iter_mut().for_each(|q| *q = dist.sample(rng));

            dut.preprocess(&query);
            self.drive_unary(
                num_trials,
                rng,
                check,
                &mut |vector: &[f32]| f(&query, vector),
                &mut |code| dut.evaluate(code),
                format_args!("{ctx}, query {} of {}", trial + 1, num_queries),
            )
        }
    }
}

pub(super) fn squared_l2(x: &[f32], y: &[f32]) -> f32 {
    distance::SquaredL2::evaluate(x, y)
}

pub(super) fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    distance::InnerProduct::evaluate(x, y)
}

pub(super) fn cosine(x: &[f32], y: &[f32]) -> f32 {
    distance::Cosine::evaluate(x, y)
}

/// A trait modeling query-like style distances with split pre-processing and evaluation.
pub(super) trait QueryLike {
    fn preprocess(&mut self, query: &[f32]);
    fn evaluate(&mut self, code: &[u8]) -> f32;
}

/////////////////////////
// Compression Helpers //
/////////////////////////

// TESTING STRATEGY:
//
// We need the following to test the block compression primitive:
//
// 1. A collection of pivots with known entries.
// 2. A data corpus where we know the mapping of chunks and rows to a center in the
//    previously mentioned known collection of pivots.
//
// This test fulfills these goals in the following way:
//
// ## Pivot Seeding
//
// Use a barrel-shifting approach to seeding the contents of each chunk pivots.
//
// Chunk 0 will contain the following values:
// ```
// 0.25     -0.25      0.25     -0.25     ...  +/- 0.25
// 1.25      0.25      1.25      0.25     ...  1.0 +/- 0.25
// ...
// L + 0.25  L - 0.25  L + 0.25  L - 0.25 ...  L +/- 0.25
// ```
// where Chunk 0 has (L+1) centers (of any dimension).
//
// The integer values are offset by `0.25` to yield a non-zero distance and (in this
// example) are configured so a query with all entries equal to `I <= L` will be mapped to
// row `I`.
//
// Chunk 1 will contain the following:
// ```
// 1.25      0.25      1.25      0.25     ...  1.0 +/- 0.25
// 2.25      1.25      2.25      1.25     ...  2.0 +/- 0.25
// ...
// L + 0.25  L - 0.25  L + 0.25  L - 0.25 ...  L +/- 0.25
// 0.25     -0.25      0.25     -0.25     ...    +/- 0.25
// ```
// That is, Chunk 1 will have the same contents Chunk 0, but the first row of Chunk 0 will
// be moved to the end.
//
// Chunk 2 will continue the pattern by moving the first row of Chunk 1 to the end.
//
// This pattern allows us to compute which center a properly seeded dataset chunk should
// match while providing enough entropy to make sure intermediate values are being
// computed properly.
//
// ## Data Seeding
//
// We will keep data seeding simple, using a seeded random number generate to pick a
// value between O and L and initialize all dimensions with that value.
//
// During seeding, we will record this value so that results yielded by the compression
// algorithm can be checked.
//
// The formula for going from a chunk with index "C" with the assigned value "K" is:
// ```math
// K - C mod (L + 1)
// ```

/// Seed pivot tables for the provided schema using the strategy outlined in the
/// introduction documentation to the test module.
pub(super) fn create_pivot_tables(
    schema: ChunkOffsets,
    num_centers: usize,
) -> (rowmajor::Owned<f32>, ChunkOffsets) {
    let mut pivots = rowmajor::Owned::<f32>::from_element(num_centers, schema.dim(), 0.0);

    (0..schema.len()).for_each(|chunk| {
        let range = schema.at(chunk);

        (0..num_centers).for_each(|center| {
            let buffer = &mut pivots.row_mut(center)[range.clone()];

            // It's okay if this conversion is lossy (though the magnitude of the
            // numbers involved means that this is almost certainly a lossless
            // conversion).
            //
            // The "remainder" operation is what performs the "barrel shifting"
            // for the centers.
            let base = ((center + chunk) % num_centers) as f32;
            buffer.iter_mut().enumerate().for_each(|(dim, b)| {
                // Flip-flop adding and subtracting 0.25.
                *b = if dim % 2 == 0 {
                    base + 0.25
                } else {
                    base - 0.25
                };
            });
        });
    });

    (pivots, schema)
}

/// Initialize a dataset for the provided schema using the strategy outlined in
/// the test module introduction documentation.
///
/// Returns:
///
/// * The initialized dataset as a rowmajor::Owned.
/// * The expected center as a rowmajor::Owned.
pub(super) fn create_dataset<R: Rng>(
    schema: ChunkOffsetsView<'_>,
    num_centers: usize,
    num_data: usize,
    rng: &mut R,
) -> (rowmajor::Owned<f32>, rowmajor::Owned<usize>) {
    let mut data = rowmajor::Owned::<f32>::from_element(num_data, schema.dim(), 0.0);
    let mut expected = rowmajor::Owned::<usize>::from_element(num_data, schema.len(), 0);

    let dist = Uniform::new(0, num_centers).unwrap();
    for row_index in 0..data.nrows() {
        let mut row_view = views::MutChunkView::new(data.row_mut(row_index), schema).unwrap();
        for chunk in 0..schema.len() {
            let value = rng.sample(dist);
            row_view[chunk].fill(value as f32);

            // Compute the expected value based on the rotation scheme used in pivot
            // seeding.
            let value: i64 = value.try_into().unwrap();
            let num_centers: i64 = num_centers.try_into().unwrap();
            let chunk_i64: i64 = chunk.try_into().unwrap();

            let expected_index: u64 = (value - chunk_i64)
                .rem_euclid(num_centers)
                .try_into()
                .unwrap();

            *expected.element_mut(row_index, chunk) = expected_index as usize;
        }
    }

    (data, expected)
}

/////////////////////////////////////////
// Testing `CompressInto<[f32], [u8]>` //
/////////////////////////////////////////

// A cantralized test for error handling in `CompressInto<[f32], [u8]>`
pub(super) fn check_pqtable_single_compression_errors<T>(
    build: &dyn Fn(rowmajor::Owned<f32>, ChunkOffsets) -> T,
    context: &dyn std::fmt::Display,
) where
    T: for<'a, 'b> CompressInto<&'a [f32], &'b mut [u8]>,
{
    let dim = 10;
    let num_chunks = 3;
    let offsets = ChunkOffsets::new(Box::new([0, 4, 9, 10])).unwrap();

    // Set up `ncenters > 256`.
    {
        let pivots = rowmajor::Owned::from_element(257, dim, 0.0);
        let table = build(pivots, offsets.clone());

        let input = vec![f32::default(); dim];
        let mut output = vec![u8::MAX; num_chunks];
        let result = table.compress_into(input.as_slice(), output.as_mut_slice());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "num centers (257) must be at most 256 to compress into a byte vector",
            "{}",
            context
        );
        assert!(
            output.iter().all(|i| *i == u8::MAX),
            "output vector should be unmodified -- {}",
            context
        );
    }

    // Setup input dim not equal to expected.
    {
        let pivots = rowmajor::Owned::from_element(10, dim, 0.0);
        let table = build(pivots, offsets.clone());

        let input = vec![f32::default(); dim - 1];
        let mut output = vec![u8::MAX; num_chunks];
        let result = table.compress_into(input.as_slice(), output.as_mut_slice());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            format!("invalid input len - expected {}, got {}", dim, dim - 1),
            "{}",
            context,
        );
        assert!(
            output.iter().all(|i| *i == u8::MAX),
            "output vector should be unmodified -- {}",
            context
        );
    }

    // Setup output dim not equal to expected.
    {
        let pivots = rowmajor::Owned::from_element(10, dim, 0.0);
        let table = build(pivots, offsets.clone());

        let input = vec![f32::default(); dim];
        let mut output = vec![u8::MAX; num_chunks - 1];
        let result = table.compress_into(input.as_slice(), output.as_mut_slice());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "invalid PQ buffer len - expected {}, got {}",
                num_chunks,
                num_chunks - 1,
            ),
            "{}",
            context,
        );
        assert!(
            output.iter().all(|i| *i == u8::MAX),
            "output vector should be unmodified -- {}",
            context,
        );
    }

    // Infinity or NaN detection.
    {
        let offsets = ChunkOffsets::new(Box::new([
            0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136,
        ]))
        .unwrap();

        let (pivots, o) = create_pivot_tables(offsets.clone(), 7);
        let table = build(pivots, o);

        let mut buf: Box<[f32]> = (0..offsets.dim()).map(|_| 0.0).collect();
        let mut output: Box<[u8]> = (0..offsets.len()).map(|_| 0).collect();

        fn clear(x: &mut [f32]) {
            x.iter_mut().for_each(|i| *i = 0.0);
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x90a10423fdd8f1cf);
        let values = [f32::NEG_INFINITY, f32::INFINITY, f32::NAN];

        // Feed in positive infinity, negative infinity, and NaN into each chunk.
        for chunk in 0..offsets.len() {
            let range = offsets.at(chunk);
            let distribution = Uniform::new(range.start, range.end).unwrap();
            let expected = format!(
                "a value of infinity or NaN was observed while compressing chunk {}",
                chunk
            );

            for &value in values.iter() {
                clear(&mut buf);
                buf[distribution.sample(&mut rng)] = value;
                let err = table
                    .compress_into(&buf, &mut output)
                    .unwrap_err()
                    .to_string();

                assert!(
                    err.contains(&expected),
                    "wrong error message for {} - expected \"{}\", got \"{}\"",
                    value,
                    expected,
                    err
                );
            }
        }
    }
}

////////////////////////////////////////////////////////////////////
// Testing `CompressInto<rowmajor::Ref<'_, f32>, MarixView<'_, u8>>` //
////////////////////////////////////////////////////////////////////

// A cantralized test for error handling in `CompressInto<[f32], [u8]>`
#[cfg(not(miri))]
pub(super) fn check_pqtable_batch_compression_errors<T>(
    build: &dyn Fn(rowmajor::Owned<f32>, ChunkOffsets) -> T,
    context: &dyn std::fmt::Display,
) where
    T: for<'a> CompressInto<rowmajor::Ref<'a, f32>, rowmajor::Mut<'a, u8>>,
{
    let dim = 10;
    let num_chunks = 3;
    let offsets = ChunkOffsets::new(Box::new([0, 4, 9, 10])).unwrap();

    let batchsize = 10;

    // Set up `ncenters > 256`.
    {
        let pivots = rowmajor::Owned::from_element(257, dim, 0.0);
        let table = build(pivots, offsets.clone());

        let input = rowmajor::Owned::from_element(batchsize, dim, f32::default());
        let mut output = rowmajor::Owned::from_element(batchsize, num_chunks, u8::MAX);
        let result = table.compress_into(input.as_view(), output.as_view_mut());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "num centers (257) must be at most 256 to compress into a byte vector",
            "{}",
            context
        );
        assert!(
            output.as_slice().iter().all(|i| *i == u8::MAX),
            "output vector should be unmodified -- {}",
            context
        );
    }

    // Setup input dim not equal to expected.
    {
        let pivots = rowmajor::Owned::from_element(10, dim, 0.0);
        let table = build(pivots, offsets.clone());

        let input = rowmajor::Owned::from_element(batchsize, dim - 1, f32::default());
        let mut output = rowmajor::Owned::from_element(batchsize, num_chunks, u8::MAX);
        let result = table.compress_into(input.as_view(), output.as_view_mut());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            format!("invalid input len - expected {}, got {}", dim, dim - 1),
            "{}",
            context,
        );
        assert!(
            output.as_slice().iter().all(|i| *i == u8::MAX),
            "output vector should be unmodified -- {}",
            context
        );
    }

    // Setup output dim not equal to expected.
    {
        let pivots = rowmajor::Owned::from_element(10, dim, 0.0);
        let table = build(pivots, offsets.clone());

        let input = rowmajor::Owned::from_element(batchsize, dim, f32::default());
        let mut output = rowmajor::Owned::from_element(batchsize, num_chunks - 1, u8::MAX);
        let result = table.compress_into(input.as_view(), output.as_view_mut());

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "invalid PQ buffer len - expected {}, got {}",
                num_chunks,
                num_chunks - 1,
            ),
            "{}",
            context,
        );
        assert!(
            output.as_slice().iter().all(|i| *i == u8::MAX),
            "output vector should be unmodified -- {}",
            context,
        );
    }

    // Num rows are different.
    {
        let pivots = rowmajor::Owned::from_element(10, dim, 0.0);
        let table = build(pivots, offsets.clone());

        let input = rowmajor::Owned::from_element(batchsize, dim, f32::default());
        let mut output = rowmajor::Owned::from_element(batchsize - 1, num_chunks, u8::MAX);
        let result = table.compress_into(input.as_view(), output.as_view_mut());

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "input and output must have the same number of rows - instead, got {0} and {1} \
                 (respectively)",
                batchsize,
                batchsize - 1,
            ),
            "{}",
            context,
        );
        assert!(
            output.as_slice().iter().all(|i| *i == u8::MAX),
            "output vector should be unmodified -- {}",
            context,
        );
    }

    // Infinity and NaN detection.
    {
        let offsets = ChunkOffsets::new(Box::new([
            0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136,
        ]))
        .unwrap();

        let (pivots, o) = create_pivot_tables(offsets.clone(), 7);
        let table = build(pivots, o);

        let num_points = 15;
        let mut buf = rowmajor::Owned::<f32>::from_element(num_points, offsets.dim(), 0.0);
        let mut output = rowmajor::Owned::<u8>::from_element(num_points, offsets.len(), 0);

        fn clear<T: Default>(mut x: rowmajor::Mut<T>) {
            x.as_mut_slice().iter_mut().for_each(|i| *i = T::default());
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x8aa9f8cc50260d5c);

        let sample = [f32::NEG_INFINITY, f32::INFINITY, f32::NAN];

        // Feed in positive infinity, negative infinity, and NaN into each chunk.
        for chunk in 0..offsets.len() {
            let range = offsets.at(chunk);
            let distribution = Uniform::new(range.start, range.end).unwrap();

            for row in 0..num_points {
                clear(buf.as_view_mut());
                let value = *sample.choose(&mut rng).unwrap();
                *buf.element_mut(row, distribution.sample(&mut rng)) = value;
                let err = table
                    .compress_into(buf.as_view(), output.as_view_mut())
                    .expect_err(&format!("expected a value of {}", value));

                let message = err.to_string();
                let expected = format!(
                    "a value of infinity or NaN was observed while compressing chunk {} \
                     of batch input {}",
                    chunk, row
                );

                assert!(
                    message.contains(&expected),
                    "wrong error message - expected \"{}\", got \"{}\"",
                    expected,
                    err
                );
            }
        }
    }
}
