/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// A collection of test helpers to ensure uniformity across tables.
use diskann_utils::views::Matrix;
#[cfg(not(miri))]
use diskann_utils::views::{MatrixView, MutMatrixView};
#[cfg(not(miri))]
use rand::seq::IndexedRandom;
use rand::{
    distr::{Distribution, Uniform},
    Rng, SeedableRng,
};

use crate::traits::CompressInto;
use crate::views::{self, ChunkOffsets, ChunkOffsetsView};

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
) -> (Matrix<f32>, ChunkOffsets) {
    let mut pivots = Matrix::<f32>::new(0.0, num_centers, schema.dim());

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
/// * The initialized dataset as a Matrix.
/// * The expected center as a Matrix.
pub(super) fn create_dataset<R: Rng>(
    schema: ChunkOffsetsView<'_>,
    num_centers: usize,
    num_data: usize,
    rng: &mut R,
) -> (Matrix<f32>, Matrix<usize>) {
    let mut data = Matrix::<f32>::new(0.0, num_data, schema.dim());
    let mut expected = Matrix::<usize>::new(0, num_data, schema.len());

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

            expected[(row_index, chunk)] = expected_index as usize;
        }
    }

    (data, expected)
}

/////////////////////////////////////////
// Testing `CompressInto<[f32], [u8]>` //
/////////////////////////////////////////

// A cantralized test for error handling in `CompressInto<[f32], [u8]>`
pub(super) fn check_pqtable_single_compression_errors<T>(
    build: &dyn Fn(Matrix<f32>, ChunkOffsets) -> T,
    context: &dyn std::fmt::Display,
) where
    T: for<'a, 'b> CompressInto<&'a [f32], &'b mut [u8]>,
{
    let dim = 10;
    let num_chunks = 3;
    let offsets = ChunkOffsets::new(Box::new([0, 4, 9, 10])).unwrap();

    // Set up `ncenters > 256`.
    {
        let pivots = Matrix::new(0.0, 257, dim);
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
        let pivots = Matrix::new(0.0, 10, dim);
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
        let pivots = Matrix::new(0.0, 10, dim);
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
// Testing `CompressInto<MatrixView<'_, f32>, MarixView<'_, u8>>` //
////////////////////////////////////////////////////////////////////

// A cantralized test for error handling in `CompressInto<[f32], [u8]>`
#[cfg(not(miri))]
pub(super) fn check_pqtable_batch_compression_errors<T>(
    build: &dyn Fn(Matrix<f32>, ChunkOffsets) -> T,
    context: &dyn std::fmt::Display,
) where
    T: for<'a> CompressInto<MatrixView<'a, f32>, MutMatrixView<'a, u8>>,
{
    let dim = 10;
    let num_chunks = 3;
    let offsets = ChunkOffsets::new(Box::new([0, 4, 9, 10])).unwrap();

    let batchsize = 10;

    // Set up `ncenters > 256`.
    {
        let pivots = Matrix::new(0.0, 257, dim);
        let table = build(pivots, offsets.clone());

        let input = Matrix::new(f32::default(), batchsize, dim);
        let mut output = Matrix::new(u8::MAX, batchsize, num_chunks);
        let result = table.compress_into(input.as_view(), output.as_mut_view());
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
        let pivots = Matrix::new(0.0, 10, dim);
        let table = build(pivots, offsets.clone());

        let input = Matrix::new(f32::default(), batchsize, dim - 1);
        let mut output = Matrix::new(u8::MAX, batchsize, num_chunks);
        let result = table.compress_into(input.as_view(), output.as_mut_view());
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
        let pivots = Matrix::new(0.0, 10, dim);
        let table = build(pivots, offsets.clone());

        let input = Matrix::new(f32::default(), batchsize, dim);
        let mut output = Matrix::new(u8::MAX, batchsize, num_chunks - 1);
        let result = table.compress_into(input.as_view(), output.as_mut_view());

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
        let pivots = Matrix::new(0.0, 10, dim);
        let table = build(pivots, offsets.clone());

        let input = Matrix::new(f32::default(), batchsize, dim);
        let mut output = Matrix::new(u8::MAX, batchsize - 1, num_chunks);
        let result = table.compress_into(input.as_view(), output.as_mut_view());

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
        let mut buf = Matrix::<f32>::new(0.0, num_points, offsets.dim());
        let mut output = Matrix::<u8>::new(0, num_points, offsets.len());

        fn clear<T: Default>(mut x: MutMatrixView<T>) {
            x.as_mut_slice().iter_mut().for_each(|i| *i = T::default());
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x8aa9f8cc50260d5c);

        let sample = [f32::NEG_INFINITY, f32::INFINITY, f32::NAN];

        // Feed in positive infinity, negative infinity, and NaN into each chunk.
        for chunk in 0..offsets.len() {
            let range = offsets.at(chunk);
            let distribution = Uniform::new(range.start, range.end).unwrap();

            for row in 0..num_points {
                clear(buf.as_mut_view());
                let value = *sample.choose(&mut rng).unwrap();
                buf[(row, distribution.sample(&mut rng))] = value;
                let err = table
                    .compress_into(buf.as_view(), output.as_mut_view())
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
