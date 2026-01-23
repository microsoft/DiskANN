/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use super::pivots;
use crate::{
    product::tables::basic::TableCompressionError,
    traits::CompressInto,
    views::{ChunkOffsets, ChunkOffsetsView},
};
use diskann_utils::{
    strided,
    views::{self, MatrixView, MutMatrixView},
};
use thiserror::Error;

/// A PQ table that stores the pivots for each chunk in a miniture block-transpose to
/// facilitate faster compression. The exact layout is not documented (as it is for the
/// `BasicTable`) because it may be subject to change.
///
/// The advantage of this table over the `BasicTable` is that this table uses a more
/// hardware friendly layout for the pivots, meaning that compression (particularly batch
/// compression) is much faster than the basic table, at the cost of a slightly higher
/// memory footprint.
///
/// # Invariants (Dev Docs)
///
/// * `pivots.len() == schema.len()`: The number of PQ chunks must agree.
/// * `pivots[i].dimension() == schema.at(i).len()` for all inbounds `i`.
/// * `pivots[i].total() == ncenters` for all inbounds `i`.
/// * `largest = max(schema.at(i).len() for i)`.
#[derive(Debug)]
pub struct TransposedTable {
    pivots: Box<[pivots::Chunk]>,
    offsets: crate::views::ChunkOffsets,
    /// The largest dimension in offsets.
    largest: usize,
    /// The number of centers in each block.
    ncenters: usize,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TransposedTableError {
    #[error("pivots have {pivot_dim} dimensions while the offsets expect {offsets_dim}")]
    DimMismatch {
        pivot_dim: usize,
        offsets_dim: usize,
    },
    #[error("error constructing pivot {problem} of {total}")]
    PivotError {
        problem: usize,
        total: usize,
        source: pivots::ChunkConstructionError,
    },
}

impl TransposedTable {
    /// Construct a new `TransposedTable` from raw parts.
    ///
    /// # Error
    ///
    /// Returns an error if
    ///
    /// * `pivots.ncols() != offsets.dim()`: Pivots must have the dimensionality expected
    ///   by the offsets.
    ///
    /// * `pivots.nrows() == 0`: The pivot table cannot be empty.
    #[allow(clippy::expect_used)]
    pub fn from_parts(
        pivots: views::MatrixView<f32>,
        offsets: ChunkOffsets,
    ) -> Result<Self, TransposedTableError> {
        let pivot_dim = pivots.ncols();
        let offsets_dim = offsets.dim();
        if pivot_dim != offsets_dim {
            return Err(TransposedTableError::DimMismatch {
                pivot_dim,
                offsets_dim,
            });
        }

        let ncenters = pivots.nrows();
        let mut largest = 0;
        let pivots: Box<[_]> = (0..offsets.len())
            .map(|i| {
                let range = offsets.at(i);
                largest = largest.max(range.len());
                let view = strided::StridedView::try_shrink_from(
                    &(pivots.as_slice()[range.start..]),
                    pivots.nrows(),
                    range.len(),
                    offsets.dim(),
                )
                .expect(
                    "the check on `pivot_dim` and `offsets_dim` should cause this to never error",
                );
                pivots::Chunk::new(view).map_err(|source| TransposedTableError::PivotError {
                    problem: i,
                    total: offsets.len(),
                    source,
                })
            })
            .collect::<Result<Box<[_]>, TransposedTableError>>()?;

        debug_assert_eq!(pivots.len(), offsets.len());
        Ok(Self {
            pivots,
            offsets,
            largest,
            ncenters,
        })
    }

    /// Return the number of pivots in each PQ chunk.
    pub fn ncenters(&self) -> usize {
        self.ncenters
    }

    /// Return the number of PQ chunks.
    pub fn nchunks(&self) -> usize {
        self.offsets.len()
    }

    /// Return the dimensionality of the full-precision vectors associated with this table.
    pub fn dim(&self) -> usize {
        self.offsets.dim()
    }

    /// Return a view over the schema offsets.
    pub fn view_offsets(&self) -> ChunkOffsetsView<'_> {
        self.offsets.as_view()
    }

    /// Perform PQ compression on the dataset by mapping each chunk in `data` to its nearest
    /// neighbor in the corresponding entry in `chunks`.
    ///
    /// The index of the nearest neighbor is provided to `compression_delegate` along with its
    /// corresponding row in data and chunk index.
    ///
    /// Calls to `compression_delegate` may occur in any order.
    ///
    /// Visitor will be invoked for all rows in `0..data.nrows()` and all chunks in
    /// `0..schema.len()`.
    ///
    /// # Panics
    ///
    /// Panics under the following conditions:
    /// * `data.cols() != self.dim()`: The number of columns in the source dataset must match
    ///   the number of dimensions expected by the schema.
    #[allow(clippy::expect_used)]
    pub fn compress_batch<T, F, DelegateError>(
        &self,
        data: views::MatrixView<'_, T>,
        mut compression_delegate: F,
    ) -> Result<(), CompressError<DelegateError>>
    where
        T: Copy + Into<f32>,
        F: FnMut(RowChunk, pivots::CompressionResult) -> Result<(), DelegateError>,
        DelegateError: std::error::Error,
    {
        assert_eq!(
            data.ncols(),
            self.dim(),
            "schema expects {} dimensions but data has {}",
            self.dim(),
            data.ncols()
        );

        // The batch size expected by the compression micro-kernel.
        const SUB_BATCH_SIZE: usize = pivots::Chunk::batchsize();

        let dim_nonzero = self.offsets.dim_nonzero();
        let mut packing_buffer: Box<[f32]> =
            (0..self.largest * SUB_BATCH_SIZE).map(|_| 0.0).collect();

        // Stride along the chunks in the source matrix to keep the associated `Chunk` in the
        // cache.
        let nrows = data.nrows();
        let ncols = data.ncols();
        let slice = data.as_slice();

        for (i, chunk) in self.pivots.iter().enumerate() {
            let range = self.offsets.at(i);
            if let Some(chunk_dim) = NonZeroUsize::new(range.len()) {
                // Construct a view for the packing buffer for this chunk.
                let mut packing_view = views::MutMatrixView::try_from(
                    &mut packing_buffer[..SUB_BATCH_SIZE * chunk_dim.get()],
                    SUB_BATCH_SIZE,
                    chunk_dim.get(),
                )
                .expect("the packing buffer should have been sized correctly");

                for row_start in (0..nrows).step_by(SUB_BATCH_SIZE) {
                    let row_end = nrows.min(row_start + SUB_BATCH_SIZE);

                    // If this is a full batch, the use the batched micro-kernel.
                    if row_end - row_start == SUB_BATCH_SIZE {
                        // When computing the stop offset, don't try to adjust for the length
                        // of the underlying span.
                        //
                        // The control on our loop bounds mean we should never be in a situation
                        // where this would occur, so we'd rather hit the indexing panic early.
                        let mut linear_start = row_start * ncols + range.start;
                        packing_view.row_iter_mut().for_each(|row| {
                            pack(row, &slice[linear_start..linear_start + chunk_dim.get()]);
                            linear_start += dim_nonzero.get();
                        });

                        let result = chunk.find_closest_batch(packing_view.as_view().into());

                        // Invoke the delegate with the results.
                        // If the delegate returns an error, wrap that error inside a
                        // `CompressError` to add context and forward the error upward.
                        for (j, &r) in result.iter().enumerate() {
                            compression_delegate(
                                RowChunk {
                                    row: row_start + j,
                                    chunk: i,
                                },
                                r,
                            )
                            .map_err(|inner| CompressError {
                                inner,
                                row: row_start + j,
                                chunk: i,
                                nearest: r.into_inner(),
                            })?;
                        }
                    } else {
                        // Handle remainders one at a time.
                        for row in row_start..row_end {
                            let linear_start = row * ncols + range.start;
                            let linear_stop = linear_start + range.len();

                            // Pre-convert to f32.
                            let packed = &mut packing_view.row_mut(0);
                            pack(packed, &slice[linear_start..linear_stop]);
                            let result = chunk.find_closest(packed);

                            compression_delegate(RowChunk { row, chunk: i }, result).map_err(
                                |inner| CompressError {
                                    inner,
                                    row,
                                    chunk: i,
                                    nearest: result.into_inner(),
                                },
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute the operation defined by `T` for each chunk in the query on all corresponding
    /// pivots for that chunk, storing the result in the output matrix.
    ///
    /// For example, this can be used to compute squared L2 distances between chunks of the
    /// query and the pivot table to create a fast run-time lookup table for these distances.
    ///
    /// This is currently implemented for the following operation types `T`:
    ///
    /// * `quantization::distances::SquaredL2`
    /// * `quantization::distances::InnerProduct`
    ///
    /// # Arguments
    ///
    /// * `query`: The query slice to process. Must have length `self.dim()`.
    /// * `partials`: Output matrix for the partial results. The result of the computation
    ///   of chunk `i` against pivot `j` will be stored into `pivots[(i, j)]`.
    ///
    ///   Must have `nrows = self.nchunks()` and `ncols = self.ncenters()`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `query.len() != self.dim()`.
    /// * `partisl.nrows() != self.nchunks()`.
    /// * `partisl.ncols() != self.ncenters()`.
    pub fn process_into<T>(&self, query: &[f32], mut partials: MutMatrixView<'_, f32>)
    where
        T: pivots::ProcessInto,
    {
        // Check Requirements
        assert_eq!(
            query.len(),
            self.dim(),
            "query has the wrong number of dimensions"
        );
        assert_eq!(
            partials.ncols(),
            self.ncenters(),
            "output has the wrong number of columns"
        );
        assert_eq!(
            partials.nrows(),
            self.nchunks(),
            "output has the wrong number of rows"
        );

        // Loop over each chunk.
        std::iter::zip(self.pivots.iter(), partials.row_iter_mut())
            .enumerate()
            .for_each(|(i, (pivot, out))| {
                let range = self.offsets.at(i);
                T::process_into(pivot, &query[range], out);
            });
    }
}

/// Row and chunk indexes provided to the `compression_delegate` argument of `compress` to
/// describe the position of the nearest neighbor being provided.
pub struct RowChunk {
    row: usize,
    chunk: usize,
}

#[derive(Error, Debug)]
#[error(
    "compression delegate returned \"{inner}\" when processing row {row} and chunk \
    {chunk} with nearest center {nearest}"
)]
pub struct CompressError<DelegateError: std::error::Error> {
    inner: DelegateError,
    row: usize,
    chunk: usize,
    nearest: u32,
}

#[inline(always)]
fn pack<T>(dst: &mut [f32], src: &[T])
where
    T: Copy + Into<f32>,
{
    debug_assert_eq!(dst.len(), src.len());
    std::iter::zip(dst.iter_mut(), src.iter()).for_each(|(d, &s)| *d = s.into());
}

///////////////
// Coompress //
///////////////

impl CompressInto<&[f32], &mut [u8]> for TransposedTable {
    type Error = TableCompressionError;
    type Output = ();

    /// Compress the full-precision vector `from` into the PQ byte buffer `to`.
    ///
    /// Compression is performed by partitioning `from` into chunks according to the offsets
    /// schema in the table and then finding the closest pivot according to the L2 distance.
    ///
    /// The final compressed value is the index of the closest pivot.
    ///
    /// # Errors
    ///
    /// Returns errors under the following conditions:
    ///
    /// * `self.ncenters() > 256`: If the number of centers exceeds 256, then it cannot be
    ///   guaranteed that the index of the closest pivot for a chunk will fit losslessly in
    ///   an 8-bit integer.
    ///
    /// * `from.len() != self.dim()`: The full precision vector must have the dimensionality
    ///   expected by the compression.
    ///
    /// * `to.len() != self.nchunks()`: The PQ buffer must be sized appropriately.
    ///
    /// * If any chunk is sufficiently far from all centers that its distance becomes
    ///   infinity to all centers.
    ///
    /// # Allocates
    ///
    /// This function should not allocate when successful.
    ///
    /// # Parallelism
    ///
    /// This function is single-threaded.
    fn compress_into(&self, from: &[f32], to: &mut [u8]) -> Result<(), Self::Error> {
        if self.ncenters() > 256 {
            return Err(Self::Error::CannotCompressToByte(self.ncenters()));
        }
        if from.len() != self.dim() {
            return Err(Self::Error::InvalidInputDim(self.dim(), from.len()));
        }
        if to.len() != self.nchunks() {
            return Err(Self::Error::InvalidOutputDim(self.nchunks(), to.len()));
        }

        std::iter::zip(self.pivots.iter(), to.iter_mut())
            .enumerate()
            .try_for_each(|(i, (pivot, to))| {
                let range = self.offsets.at(i);
                let result = pivot.find_closest(&from[range]);
                result.map(
                    |v| *to = v as u8, // conversion guaranteed to be lossless
                    || Self::Error::InfinityOrNaN(i),
                )
            })
    }
}

#[derive(Error, Debug)]
pub enum TableBatchCompressionError {
    #[error("num centers ({0}) must be at most 256 to compress into a byte vector")]
    CannotCompressToByte(usize),
    #[error("invalid input len - expected {0}, got {1}")]
    InvalidInputDim(usize, usize),
    #[error("invalid PQ buffer len - expected {0}, got {1}")]
    InvalidOutputDim(usize, usize),
    #[error(
        "input and output must have the same number of rows - instead, got {0} and {1} \
         (respectively)"
    )]
    UnequalRows(usize, usize),
    #[error(
        "a value of infinity or NaN was observed while compressing chunk {0} of batch input {1}"
    )]
    InfinityOrNaN(usize, usize),
}

impl<T> CompressInto<MatrixView<'_, T>, MutMatrixView<'_, u8>> for TransposedTable
where
    T: Copy + Into<f32>,
{
    type Error = TableBatchCompressionError;
    type Output = ();

    /// Compress each full-precision row in `from` into the corresponding row in `to`.
    ///
    /// Compression is performed by partitioning `from` into chunks according to the offsets
    /// schema in the table and then finding the closest pivot according to the L2 distance.
    ///
    /// The final compressed value is the index of the closest pivot.
    ///
    /// # Errors
    ///
    /// Returns errors under the following conditions:
    ///
    /// * `self.ncenters() > 256`: If the number of centers exceeds 256, then it cannot be
    ///   guaranteed that the index of the closest pivot for a chunk will fit losslessly in
    ///   an 8-bit integer.
    ///
    /// * `from.ncols() != self.dim()`: The full precision data must have the dimensionality
    ///   expected by the compression.
    ///
    /// * `to.ncols() != self.nchunks()`: The PQ buffer must be sized appropriately.
    ///
    /// * `from.nrows() == to.nrows()`: The input and output buffers must have the same
    ///   number of elements.
    ///
    /// * If any chunk is sufficiently far from all centers that its distance becomes
    ///   infinity to all centers.
    ///
    /// # Allocates
    ///
    /// Allocates scratch memory proportional to the length of the largest chunk.
    ///
    /// # Parallelism
    ///
    /// This function is single-threaded.
    fn compress_into(
        &self,
        from: MatrixView<'_, T>,
        mut to: MutMatrixView<'_, u8>,
    ) -> Result<(), Self::Error> {
        if self.ncenters() > 256 {
            return Err(Self::Error::CannotCompressToByte(self.ncenters()));
        }
        if from.ncols() != self.dim() {
            return Err(Self::Error::InvalidInputDim(self.dim(), from.ncols()));
        }
        if to.ncols() != self.nchunks() {
            return Err(Self::Error::InvalidOutputDim(self.nchunks(), to.ncols()));
        }
        if from.nrows() != to.nrows() {
            return Err(Self::Error::UnequalRows(from.nrows(), to.nrows()));
        }

        // The `CompressionError` already has all the information we need, so we make the
        // `Delegate` error light-weight.
        #[derive(Debug, Error)]
        #[error("unreachable")]
        struct PassThrough;

        // Do the compression. We will reformat the NaN warning from `CompressionError`
        // if needed.
        let result = self.compress_batch(
            from,
            |RowChunk { row, chunk }, result| -> Result<(), PassThrough> {
                result.map(|v| to[(row, chunk)] = v as u8, || PassThrough)
            },
        );

        result.map_err(|err| Self::Error::InfinityOrNaN(err.chunk, err.row))
    }
}

#[cfg(test)]
mod test_compression {
    use std::collections::HashSet;

    use diskann_vector::{distance, PureDistanceFunction};
    use rand::{
        distr::{Distribution, StandardUniform, Uniform},
        rngs::StdRng,
        Rng, SeedableRng,
    };

    use super::*;
    #[cfg(not(miri))]
    use crate::product::tables::test::{
        check_pqtable_batch_compression_errors, check_pqtable_single_compression_errors,
    };
    use crate::{
        distances::{InnerProduct, SquaredL2},
        error::format,
        product::tables::test::{create_dataset, create_pivot_tables},
    };
    use diskann_utils::lazy_format;

    //////////////////////////////
    // Transposed Table Methods //
    //////////////////////////////

    // Test that an error is returned when the dimension between the pivots and offsets
    // disagree.
    #[test]
    fn error_on_mismatch_dim() {
        let pivots = views::Matrix::new(0.0, 3, 5);
        let offsets = ChunkOffsets::new(Box::new([0, 1, 6])).unwrap();
        let result = TransposedTable::from_parts(pivots.as_view(), offsets);
        assert!(result.is_err(), "dimensions are not equal");
        assert_eq!(
            result.unwrap_err().to_string(),
            "pivots have 5 dimensions while the offsets expect 6"
        );
    }

    // Test that an error is returned when the dimension between the pivots and offsets
    // disagree.
    #[test]
    fn error_on_empty() {
        let pivots = views::Matrix::new(0.0, 0, 5);
        let offsets = ChunkOffsets::new(Box::new([0, 1, 5])).unwrap();
        let result = TransposedTable::from_parts(pivots.as_view(), offsets);
        assert!(result.is_err(), "dimensions are not equal");

        let expected = [
            "error constructing pivot 0 of 2",
            "    caused by: cannot construct a Chunk from a source with zero length",
        ]
        .join("\n");

        assert_eq!(format(&result.unwrap_err()), expected,);
    }

    #[test]
    fn basic_table() {
        let mut rng = StdRng::seed_from_u64(0xd96bac968083ec29);
        for dim in [5, 10, 12] {
            // Sweep over enough totals to ensure the inner chunks have a non-trivial layout.
            for total in [1, 2, 3, 7, 8, 9, 10] {
                let pivots = views::Matrix::new(
                    views::Init(|| -> f32 { StandardUniform {}.sample(&mut rng) }),
                    total,
                    dim,
                );
                let offsets = ChunkOffsets::new(Box::new([0, 1, 3, dim])).unwrap();
                let table = TransposedTable::from_parts(pivots.as_view(), offsets.clone()).unwrap();

                assert_eq!(table.ncenters(), total);
                assert_eq!(table.nchunks(), offsets.len());
                assert_eq!(table.dim(), offsets.dim());

                // This kind of looks into the guts of this data structure, but is an extra
                // check that the plumbing was performed properly.
                for chunk in 0..offsets.len() {
                    let range = offsets.at(chunk);
                    let pivot = &table.pivots[chunk];
                    for row in 0..total {
                        let r = &pivots.row(row)[range.clone()];
                        for (col, expected) in r.iter().enumerate() {
                            assert_eq!(pivot.get(row, col), *expected);
                        }
                    }
                }

                assert_eq!(table.view_offsets(), offsets.as_view());
            }
        }
    }

    /////////////////
    // Compression //
    /////////////////

    #[derive(Error, Debug)]
    #[error("unreachable reached")]
    struct Infallible;

    #[test]
    fn test_happy_path() {
        // Feed in chunks of dimension 1, 2, 3, ... 16.
        //
        // If we're using MIRI, max out at 7 dimensions.
        let offsets: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 3, 6, 10, 15, 21, 28, 36]
        } else {
            vec![
                0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136,
            ]
        };

        let schema = ChunkOffsetsView::new(&offsets).unwrap();
        let mut rng = StdRng::seed_from_u64(0x88e3d3366501ad6c);

        let num_data = if cfg!(miri) {
            vec![0, 8, 9, 10, 11]
        } else {
            vec![0, 1, 2, 3, 4, 16, 17, 18, 19]
        };

        let num_trials = if cfg!(miri) { 1 } else { 10 };

        // Strategically pick `num_centers` so we cover the corner cases in lower-level
        // handling.
        //
        // This includes:
        // * Full blocks with an even number of total blocks (num centers = 16);
        // * Full blocks with an odd number of total blocks (num centers = 24);
        // * Partially full blocks with the last block even (num centers = 13);
        // * Partially full blocks with the last block odd (num centers = 17);
        for &num_centers in [16, 24, 13, 17].iter() {
            for &num_data in num_data.iter() {
                for trial in 0..num_trials {
                    let context = lazy_format!(
                        "happy path, num centers = {}, num data = {}, trial = {}",
                        num_centers,
                        num_data,
                        trial,
                    );

                    println!("Currently = {}", context);

                    let (pivots, offsets) = create_pivot_tables(schema.to_owned(), num_centers);
                    let table = TransposedTable::from_parts(pivots.as_view(), offsets).unwrap();
                    let (data, expected) = create_dataset(schema, num_centers, num_data, &mut rng);

                    let mut called = HashSet::<(usize, usize)>::new();

                    // Direct method call.
                    table
                        .compress_batch(
                            data.as_view(),
                            |RowChunk { row, chunk }, value| -> Result<(), Infallible> {
                                assert!(value.is_okay());
                                // Ensure that this is the expected value.
                                assert_eq!(
                                value.unwrap() as usize,
                                expected[(row, chunk)],
                                "failed at (row = {row}, chunk = {chunk}). data = {:?}, context: {}",
                                &(data.row(row)[schema.at(chunk)]),
                                context,
                            );

                                // (A) Ensure that this combination of row and chunk hasn't
                                //     been called before.
                                // (B) Record that this combination has been called.
                                assert!(
                                    called.insert((row, chunk)),
                                    "row {row} and chunk {chunk}, called multiple times. Context = {}",
                                    context,
                                );

                                Ok(())
                            },
                        )
                        .unwrap();

                    assert_eq!(called.len(), num_data * schema.len());

                    // Trait Interface.
                    let mut output = views::Matrix::new(0, num_data, schema.len());
                    table
                        .compress_into(data.as_view(), output.as_mut_view())
                        .unwrap();

                    assert_eq!(output.nrows(), expected.nrows());
                    assert_eq!(output.ncols(), expected.ncols());
                    for row in 0..output.nrows() {
                        for col in 0..output.ncols() {
                            assert_eq!(
                                output[(row, col)] as usize,
                                expected[(row, col)],
                                "failed on row {}, col {}. Context = {}",
                                row,
                                col,
                                context,
                            );
                        }
                    }

                    // Trait inteface - single step.
                    let mut output = vec![0; schema.len()];
                    for (i, (row, expected)) in
                        std::iter::zip(data.row_iter(), expected.row_iter()).enumerate()
                    {
                        table.compress_into(row, output.as_mut_slice()).unwrap();
                        for (d, (o, e)) in
                            std::iter::zip(output.iter(), expected.iter()).enumerate()
                        {
                            assert_eq!(
                                *o as usize, *e,
                                "failed on row {}, col {}. Context = {}",
                                i, d, context
                            );
                        }
                    }
                }
            }
        }
    }

    ////////////////////
    // Error Handling //
    ////////////////////

    #[test]
    #[should_panic(expected = "schema expects 4 dimensions but data has 5")]
    fn panic_on_dim_mismatch() {
        let offsets = [0, 4];
        let data: Vec<f32> = vec![0.0; 5];

        let schema = ChunkOffsetsView::new(&offsets).unwrap();
        let (pivots, offsets) = create_pivot_tables(schema.to_owned(), 3);
        let table = TransposedTable::from_parts(pivots.as_view(), offsets).unwrap();

        // should panic
        let _ = table.compress_batch(
            views::MatrixView::try_from(data.as_slice(), 1, 5).unwrap(),
            |_, _| -> Result<(), Infallible> { panic!("this shouldn't be called") },
        );
    }

    #[derive(Error, Debug)]
    #[error("compression delegate error with {0}")]
    struct DelegateError(u64);

    // The strategy here is to construct a delegate that returns an error on a specific
    // row and chunk. We then make sure that the error is propagated successfully along
    // with the recorded row and chunk.
    #[test]
    fn test_delegate_error_propagation() {
        let offsets: Vec<usize> = vec![0, 1, 7];
        let schema = ChunkOffsetsView::new(&offsets).unwrap();
        let mut rng = StdRng::seed_from_u64(0xc35a90da17fafa2a);

        let num_centers = 3;
        let num_data = 7;

        let (pivots, offsets) = create_pivot_tables(schema.to_owned(), num_centers);
        let table = TransposedTable::from_parts(pivots.as_view(), offsets).unwrap();
        let (data, _) = create_dataset(schema, num_centers, num_data, &mut rng);

        let data_view =
            views::MatrixView::try_from(data.as_slice(), num_data, schema.dim()).unwrap();
        let distribution = rand_distr::StandardUniform {};

        for row in 0..data_view.nrows() {
            for chunk in 0..schema.len() {
                let context = lazy_format!("row = {row}, chunk = {chunk}");

                // Generate a random number for the delegate error.
                let value: u64 = rng.sample(distribution);

                let result = table.compress_batch(
                    data_view,
                    |RowChunk {
                         row: this_row,
                         chunk: this_chunk,
                     },
                     _| {
                        if this_row == row && this_chunk == chunk {
                            Err(DelegateError(value))
                        } else {
                            Ok(())
                        }
                    },
                );
                assert!(result.is_err(), "{}", context);

                let message = result.unwrap_err().to_string();
                assert!(
                    message.contains(&format!("{}", DelegateError(value))),
                    "{}",
                    context
                );
                assert!(message.contains("delegate returned"));
                assert!(message.contains(&format!("when processing row {row} and chunk {chunk}")));
            }
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn test_table_single_compression_errors() {
        check_pqtable_single_compression_errors(
            &|pivots: views::Matrix<f32>, offsets| {
                TransposedTable::from_parts(pivots.as_view(), offsets).unwrap()
            },
            &"TranposedTable",
        )
    }

    #[test]
    #[cfg(not(miri))]
    fn test_table_batch_compression_errors() {
        check_pqtable_batch_compression_errors(
            &|pivots: views::Matrix<f32>, offsets| {
                TransposedTable::from_parts(pivots.as_view(), offsets).unwrap()
            },
            &"TranposedTable",
        )
    }

    /////////////////////////
    // Test `process_into` //
    /////////////////////////

    fn test_process_into_impl(
        num_chunks: usize,
        num_centers: usize,
        num_trials: usize,
        rng: &mut StdRng,
    ) {
        // Choose the chunk size randomly from this distribution. Keep choosing chunks
        // sizes until the desired `num_chunks` is reached.
        //
        // The sum of chunk sizes gives the dimensionality.
        let chunk_size_distribution = Uniform::<usize>::new(1, 6).unwrap();

        // Use integer values for the value distribution to avoid dealing with floating
        // point rounding.
        let value_distribution = Uniform::<i32>::new(-10, 10).unwrap();

        for trial in 0..num_trials {
            let mut offsets: Vec<usize> = vec![0];
            for _ in 0..num_chunks {
                let chunk_size = chunk_size_distribution.sample(rng);
                offsets.push(offsets.last().unwrap() + chunk_size);
            }

            let offsets = ChunkOffsets::new(offsets.into()).unwrap();
            let dim = offsets.dim();
            let pivots = views::Matrix::<f32>::new(
                views::Init(|| value_distribution.sample(rng) as f32),
                num_centers,
                dim,
            );

            let table = TransposedTable::from_parts(pivots.as_view(), offsets.clone()).unwrap();

            let mut output = views::Matrix::<f32>::new(0.0, num_chunks, num_centers);
            let query: Vec<_> = (0..dim)
                .map(|_| value_distribution.sample(rng) as f32)
                .collect();

            // Inner Product
            table.process_into::<InnerProduct>(&query, output.as_mut_view());

            for chunk in 0..num_chunks {
                let range = offsets.at(chunk);
                let query_chunk = &query[range.clone()];
                for center in 0..num_centers {
                    let data_chunk = &pivots.row(center)[range.clone()];
                    let expected: f32 = distance::InnerProduct::evaluate(query_chunk, data_chunk);
                    assert_eq!(
                        output[(chunk, center)],
                        expected,
                        "failed on (chunk, center) = ({}, {}) - offsets = {:?} - trial = {}",
                        chunk,
                        center,
                        offsets,
                        trial,
                    );
                }
            }

            // Squared L2
            table.process_into::<SquaredL2>(&query, output.as_mut_view());

            for chunk in 0..num_chunks {
                let range = offsets.at(chunk);
                let query_chunk = &query[range.clone()];
                for center in 0..num_centers {
                    let data_chunk = &pivots.row(center)[range.clone()];
                    let expected: f32 = distance::SquaredL2::evaluate(query_chunk, data_chunk);
                    assert_eq!(
                        output[(chunk, center)],
                        expected,
                        "failed on (chunk, center) = ({}, {}) - offsets = {:?} - trial = {}",
                        chunk,
                        center,
                        offsets,
                        trial,
                    );
                }
            }
        }
    }

    #[test]
    fn test_process_into() {
        let mut rng = StdRng::seed_from_u64(0x0e3cf3ba4b27e7f8);
        for num_chunks in 1..5 {
            for num_centers in 1..48 {
                test_process_into_impl(num_chunks, num_centers, 2, &mut rng);
            }
        }
    }

    #[test]
    #[should_panic(expected = "query has the wrong number of dimensions")]
    fn test_process_into_panics_query() {
        let offsets = ChunkOffsets::new(Box::new([0, 1, 5])).unwrap();
        let data = views::Matrix::<f32>::new(0.0, 3, 5);
        let table = TransposedTable::from_parts(data.as_view(), offsets).unwrap();
        assert_eq!(table.dim(), 5);

        // query has the wrong length.
        let query = vec![0.0; table.dim() - 1];
        let mut partials = views::Matrix::new(0.0, table.nchunks(), table.ncenters());
        table.process_into::<InnerProduct>(&query, partials.as_mut_view());
    }

    #[test]
    #[should_panic(expected = "output has the wrong number of rows")]
    fn test_process_into_panics_partials_rows() {
        let offsets = ChunkOffsets::new(Box::new([0, 1, 5])).unwrap();
        let data = views::Matrix::<f32>::new(0.0, 3, 5);
        let table = TransposedTable::from_parts(data.as_view(), offsets).unwrap();
        assert_eq!(table.dim(), 5);

        let query = vec![0.0; table.dim()];
        // partials has the wrong numbers of rows.
        let mut partials = views::Matrix::new(0.0, table.nchunks() - 1, table.ncenters());
        table.process_into::<InnerProduct>(&query, partials.as_mut_view());
    }

    #[test]
    #[should_panic(expected = "output has the wrong number of columns")]
    fn test_process_into_panics_partials_cols() {
        let offsets = ChunkOffsets::new(Box::new([0, 1, 5])).unwrap();
        let data = views::Matrix::<f32>::new(0.0, 3, 5);
        let table = TransposedTable::from_parts(data.as_view(), offsets).unwrap();
        assert_eq!(table.dim(), 5);

        let query = vec![0.0; table.dim()];
        // partials has the wrong numbers of rows.
        let mut partials = views::Matrix::new(0.0, table.nchunks(), table.ncenters() - 1);
        table.process_into::<InnerProduct>(&query, partials.as_mut_view());
    }
}
