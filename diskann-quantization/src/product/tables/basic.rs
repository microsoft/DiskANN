/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::traits::CompressInto;
use crate::views::{ChunkOffsetsBase, ChunkOffsetsView};
use diskann_utils::{DenseData, Matrix, MatrixView};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use thiserror::Error;

/// A basic PQ table that stores the pivot table in the following dense, row-major form:
/// ```text
///           | -- chunk 0 -- | -- chunk 1 -- | -- chunk 2 -- | .... | -- chunk N-1 -- |
///           +------------------------------------------------------------------------+
///  pivot 0  | c000 c001 ... | c010 c011 ... | c020 c021 ... | .... |       ...       |
///  pivot 1  | c100 c101 ... | c110 c111 ... | c120 c121 ... | .... |       ...       |
///    ...    |      ...      |      ...      |      ...      | .... |       ...       |
///  pivot K  | cK00 cK01 ... | cK10 cK11 ... | cK20 cK21 ... | .... |       ...       |
/// ```
/// The member `offsets` describes the number of dimensions of each chunk.
///
/// # Invariants
///
/// * `offsets.dim() == pivots.ncols()`: The dimensionality of the two must agree.
#[derive(Debug, Clone)]
pub struct BasicTableBase<M, U>
where
    U: DenseData<Elem = usize>,
{
    pivots: M,
    offsets: ChunkOffsetsBase<U>,
}

/// A `BasicTableBase` that owns its contents.
pub type BasicTable = BasicTableBase<Matrix<f32>, Box<[usize]>>;

/// A `BasicTableBase` that references its contents. Construction of such a table will
/// not result in a memory allocation.
pub type BasicTableView<'a> = BasicTableBase<MatrixView<'a, f32>, &'a [usize]>;

/// Bridges owned ([`Matrix`]) and borrowed ([`MatrixView`]) pivot storage, exposing a
/// [`MatrixView`] so a [`BasicTableBase`] reads its dimensions and rows from one place.
pub trait PivotTable {
    fn view(&self) -> MatrixView<'_, f32>;
}

impl PivotTable for Matrix<f32> {
    fn view(&self) -> MatrixView<'_, f32> {
        self.as_matrix_view()
    }
}

impl PivotTable for MatrixView<'_, f32> {
    fn view(&self) -> MatrixView<'_, f32> {
        self.as_matrix_view()
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum BasicTableError {
    #[error("pivots have {pivot_dim} dimensions while the offsets expect {offsets_dim}")]
    DimMismatch {
        pivot_dim: usize,
        offsets_dim: usize,
    },
    #[error("pivots cannot be empty")]
    PivotsEmpty,
}

impl<M, U> BasicTableBase<M, U>
where
    M: PivotTable,
    U: DenseData<Elem = usize>,
{
    /// Construct a new `BasicTableBase` over the pivot table and offsets.
    ///
    /// # Error
    ///
    /// Returns an error if `pivots.ncols() != offsets.dim()` or if `pivots.nrows() == 0`.
    pub fn new(pivots: M, offsets: ChunkOffsetsBase<U>) -> Result<Self, BasicTableError> {
        let (num_pivots, pivot_dim) = {
            let view = pivots.view();
            (view.nrows(), view.ncols())
        };
        let offsets_dim = offsets.dim();

        if pivot_dim != offsets_dim {
            Err(BasicTableError::DimMismatch {
                pivot_dim,
                offsets_dim,
            })
        } else if num_pivots == 0 {
            Err(BasicTableError::PivotsEmpty)
        } else {
            Ok(Self { pivots, offsets })
        }
    }

    /// Return a view over the pivot table.
    pub fn view_pivots(&self) -> MatrixView<'_, f32> {
        self.pivots.view()
    }

    /// Return a view over the schema offsets.
    pub fn view_offsets(&self) -> ChunkOffsetsView<'_> {
        self.offsets.as_view()
    }

    /// Return the number of pivots in each PQ chunk.
    pub fn ncenters(&self) -> usize {
        self.view_pivots().nrows()
    }

    /// Return the number of PQ chunks.
    pub fn nchunks(&self) -> usize {
        self.offsets.len()
    }

    /// Return the dimensionality of the full-precision vectors associated with this table.
    pub fn dim(&self) -> usize {
        self.view_pivots().ncols()
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TableCompressionError {
    #[error("num centers ({0}) must be at most 256 to compress into a byte vector")]
    CannotCompressToByte(usize),
    #[error("invalid input len - expected {0}, got {1}")]
    InvalidInputDim(usize, usize),
    #[error("invalid PQ buffer len - expected {0}, got {1}")]
    InvalidOutputDim(usize, usize),
    #[error("a value of infinity or NaN was observed while compressing chunk {0}")]
    InfinityOrNaN(usize),
}

impl<M, U> CompressInto<&[f32], &mut [u8]> for BasicTableBase<M, U>
where
    M: PivotTable,
    U: DenseData<Elem = usize>,
{
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

        let pivots = self.view_pivots();
        to.iter_mut().enumerate().try_for_each(|(chunk, to)| {
            let mut min_distance = f32::INFINITY;
            let mut min_index = usize::MAX;
            let range = self.offsets.at(chunk);
            let slice = &from[range.clone()];

            pivots.row_iter().enumerate().for_each(|(index, row)| {
                let distance: f32 = SquaredL2::evaluate(slice, &row[range.clone()]);
                if distance < min_distance {
                    min_distance = distance;
                    min_index = index;
                }
            });

            if min_distance.is_infinite() {
                Err(Self::Error::InfinityOrNaN(chunk))
            } else {
                // This is guaranteed to be lossless because we have at most 256 centers.
                *to = min_index as u8;
                Ok(())
            }
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::{lazy_format, matrix};
    use rand::{
        SeedableRng,
        distr::{Distribution, StandardUniform},
    };

    use super::*;
    use crate::product::tables::test::{
        check_pqtable_single_compression_errors, create_dataset, create_pivot_tables,
    };

    /////////////////////////
    // Basic Table Methods //
    /////////////////////////

    // Test that an error is returned when the dimension between the pivots and offsets
    // disagree.
    #[test]
    fn error_on_mismatch_dim() {
        let pivots = matrix::Matrix::new(0.0, 3, 5);
        let offsets = crate::views::ChunkOffsets::new(Box::new([0, 1, 6])).unwrap();
        let result = BasicTable::new(pivots, offsets);
        assert!(result.is_err(), "dimensions are not equal");
        assert_eq!(
            result.unwrap_err().to_string(),
            "pivots have 5 dimensions while the offsets expect 6"
        );
    }

    // Test that the table constructor errors when there are no pivots.
    #[test]
    fn error_on_no_pivots() {
        let pivots = matrix::Matrix::new(0.0, 0, 5);
        let offsets = crate::views::ChunkOffsets::new(Box::new([0, 1, 2, 5])).unwrap();
        let result = BasicTable::new(pivots, offsets);
        assert!(result.is_err(), "pivots is empty");
        assert_eq!(result.unwrap_err().to_string(), "pivots cannot be empty",);
    }

    #[test]
    fn basic_table() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xd96bac968083ec29);
        for dim in [5, 10, 12] {
            for total in [1, 2, 3] {
                let pivots = matrix::Matrix::new(
                    matrix::Init(|| -> f32 { StandardUniform {}.sample(&mut rng) }),
                    total,
                    dim,
                );
                let offsets = crate::views::ChunkOffsets::new(Box::new([0, 1, 3, dim])).unwrap();

                let table = BasicTable::new(pivots.clone(), offsets.clone()).unwrap();

                assert_eq!(table.ncenters(), total);
                assert_eq!(table.nchunks(), offsets.len());
                assert_eq!(table.dim(), offsets.dim());
                assert_eq!(table.view_pivots().as_view(), pivots.as_view());
                assert_eq!(table.view_offsets().as_view(), offsets.as_view());
            }
        }
    }

    /////////////////
    // Compression //
    /////////////////

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

        let schema = crate::views::ChunkOffsetsView::new(&offsets).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xda5b2e661eabacea);

        let num_data = 20;
        let num_trials = if cfg!(miri) { 1 } else { 10 };

        for &num_centers in [16, 24, 13, 17].iter() {
            for trial in 0..num_trials {
                let context = lazy_format!(
                    "happy path, num centers = {}, num data = {}, trial = {}",
                    num_centers,
                    num_data,
                    trial,
                );

                println!("Currently = {}", context);

                let (pivots, offsets) = create_pivot_tables(schema.to_owned(), num_centers);
                let table = BasicTable::new(pivots, offsets).unwrap();
                let (data, expected) = create_dataset(schema, num_centers, num_data, &mut rng);

                let mut output = vec![0; schema.len()];
                for (input, expected) in std::iter::zip(data.row_iter(), expected.row_iter()) {
                    table.compress_into(input, &mut output).unwrap();
                    for (entry, (e, o)) in
                        std::iter::zip(expected.iter(), output.iter()).enumerate()
                    {
                        let o: usize = (*o).into();
                        assert_eq!(*e, o, "unexpected assignment at dim {}", entry);
                    }
                }
            }
        }
    }

    #[test]
    fn test_compression_error() {
        let dim = 10;
        let num_chunks = 3;
        let offsets = crate::views::ChunkOffsets::new(Box::new([0, 4, 9, 10])).unwrap();

        // Set up `ncenters > 256`.
        {
            let pivots = matrix::Matrix::new(0.0, 257, dim);
            let table = BasicTable::new(pivots, offsets.clone()).unwrap();

            let input = vec![f32::default(); dim];
            let mut output = vec![u8::MAX; num_chunks];
            let result = table.compress_into(&input, &mut output);
            assert!(result.is_err());
            assert_eq!(
                result.unwrap_err().to_string(),
                "num centers (257) must be at most 256 to compress into a byte vector"
            );
            assert!(
                output.iter().all(|i| *i == u8::MAX),
                "output vector should be unmodified"
            );
        }

        // Setup input dim not equal to expected.
        {
            let pivots = matrix::Matrix::new(0.0, 10, dim);
            let table = BasicTable::new(pivots, offsets.clone()).unwrap();

            let input = vec![f32::default(); dim - 1];
            let mut output = vec![u8::MAX; num_chunks];
            let result = table.compress_into(&input, &mut output);
            assert!(result.is_err());
            assert_eq!(
                result.unwrap_err().to_string(),
                format!("invalid input len - expected {}, got {}", dim, dim - 1),
            );
            assert!(
                output.iter().all(|i| *i == u8::MAX),
                "output vector should be unmodified"
            );
        }

        // Setup output dim not equal to expected.
        {
            let pivots = matrix::Matrix::new(0.0, 10, dim);
            let table = BasicTable::new(pivots, offsets.clone()).unwrap();

            let input = vec![f32::default(); dim];
            let mut output = vec![u8::MAX; num_chunks - 1];
            let result = table.compress_into(&input, &mut output);
            assert!(result.is_err());
            assert_eq!(
                result.unwrap_err().to_string(),
                format!(
                    "invalid PQ buffer len - expected {}, got {}",
                    num_chunks,
                    num_chunks - 1
                ),
            );
            assert!(
                output.iter().all(|i| *i == u8::MAX),
                "output vector should be unmodified"
            );
        }
    }

    #[test]
    fn test_table_single_compression_errors() {
        check_pqtable_single_compression_errors(
            &|pivots: matrix::Matrix<f32>, offsets| BasicTable::new(pivots, offsets).unwrap(),
            &"BasicTable",
        )
    }
}
