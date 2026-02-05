/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{self, Matrix};

use crate::recall;

/// A generic wrapper for containing the results IDs for multiple query searches.
///
/// Users should interact with this type via the [`as_rows`](Self::as_rows) methods.
///
/// Note that the slices returned by [`as_rows`](Self::as_rows) may have different lengths
/// depending on how many IDs were actually returned for each query.
#[derive(Debug)]
pub struct ResultIds<I> {
    inner: ResultIdsInner<I>,
}

impl<I> ResultIds<I> {
    /// Return the contained IDs as a [`recall::Rows<I>`].
    pub fn as_rows(&self) -> &dyn recall::Rows<I> {
        self.inner.as_rows()
    }

    pub(crate) fn new(inner: ResultIdsInner<I>) -> Self {
        Self { inner }
    }
}

/// A [`recall::Rows<I>`] implementation that is more efficient when the number of returned
/// IDs is known to be bounded by a fixed size and thus can be stored in a matrix.
///
/// The number of valid IDs per row is allowed to be less than this upper bound and is tracked
/// separately.
#[derive(Debug)]
pub(crate) struct Bounded<I> {
    ids: Matrix<I>,
    // Must have the same length as `matrix.nrows()`.
    lengths: Vec<usize>,
}

impl<I> Bounded<I> {
    /// Create a new `Bounded` instance with the given `ids` matrix and `lengths` vector.
    ///
    /// Argument `lengths` must have the same length as the number of rows in `ids` and the
    /// value of each entry must be less than or equal to the number of columns in `ids`.
    ///
    /// Length values that exceed the number of columns will silently be clamped when accessing rows.
    ///
    /// # Panics
    ///
    /// Panics if the number of rows in `ids` does not match the length of `lengths`.
    pub(crate) fn new(ids: Matrix<I>, lengths: Vec<usize>) -> Self {
        assert_eq!(
            ids.nrows(),
            lengths.len(),
            "an internal invariant was not upheld",
        );
        Self { ids, lengths }
    }

    /// Return the number of rows stored in `self`.
    pub(crate) fn len(&self) -> usize {
        self.lengths.len()
    }

    /// Return an iterator over the valid ID slices. The length of the iterator will be equal to
    /// [`Bounded::len`].
    ///
    /// Note that the yielded slices are not guaranteed to have the same length.
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &[I]> {
        std::iter::zip(self.ids.row_iter(), self.lengths.iter()).map(|(row, len)| {
            match row.get(..*len) {
                Some(v) => v,
                None => row,
            }
        })
    }
}

impl<I> recall::Rows<I> for Bounded<I> {
    fn nrows(&self) -> usize {
        self.len()
    }
    fn row(&self, index: usize) -> &[I] {
        let length = self.lengths[index];
        let row = self.ids.row(index);
        match row.get(..length) {
            Some(v) => v,
            None => row,
        }
    }
    fn ncols(&self) -> Option<usize> {
        None
    }
}

///////////
// Inner //
///////////

/// We internally have two representations for result IDs: either a bounded size
/// container (to reduce the number of heap allocations) or a dynamic vector of vectors.
///
/// The former is used when the number of IDs is known to be bounded.
#[derive(Debug)]
pub(crate) enum ResultIdsInner<I> {
    Fixed(Bounded<I>),
    Dynamic(Vec<Vec<I>>),
}

impl<I> ResultIdsInner<I> {
    pub(crate) fn as_rows(&self) -> &dyn recall::Rows<I> {
        match self {
            Self::Fixed(bounded) => bounded,
            Self::Dynamic(ids) => ids,
        }
    }
}

/// A utility tool for aggregating multiple [`ResultIdsInner<I>`] instances into a single
/// [`ResultIds<I>`]. When aggregating, if all instances are [`ResultIdsInner::Fixed`]
/// with the same upper bound, then the aggregation will also be [`ResultIdsInner::Fixed`].
///
/// Otherwise, the aggregation will be [`ResultIdsInner::Dynamic`].
#[derive(Debug, Default)]
pub(crate) enum IdAggregator<I> {
    /// No ids have been aggregated.
    #[default]
    Empty,
    /// IDs have been aggregated and all of them are bounded with the same size
    /// stored in `num_ids`. The field `len` stores the total number of rows aggregated
    /// to help with the final allocation in [`IdAggergator::finish`].
    Fixed {
        matrices: Vec<Bounded<I>>,
        len: usize,
        num_ids: usize,
    },
    /// At least one aggregated IDs instance was dynamic.
    Dynamic(Vec<ResultIdsInner<I>>),
}

impl<I> IdAggregator<I>
where
    I: Clone + Default,
{
    /// Construct a new empty [`IdAggregator`].
    pub(crate) fn new() -> Self {
        Self::Empty
    }

    /// Push `ids` into the aggregator.
    pub(crate) fn push(&mut self, ids: ResultIdsInner<I>) {
        // The general logic is as follows:
        // - If we are empty, we just take the incoming IDs. If they are bounded, we optimistically assume
        //   that future pushes will also be bounded with the same size. Otherwise, we'll always be `Dynamic`.
        //
        // - If we are `Fixed`, we check if the incoming IDs are also bounded with the same size and if so,
        //   simply append them to the internal list. If not, we convert all previously stored bounded IDs
        //   into dynamic IDs and switch to `Dynamic` mode.
        //
        // - If we are `Dynamic`, we simply append the incoming IDs.
        //
        // Possible transitions:
        // * Empty -> Fixed
        // * Empty -> Dynamic
        // * Fixed -> Dynamic

        *self = match std::mem::take(self) {
            Self::Empty => match ids {
                ResultIdsInner::Fixed(bounded) => {
                    let len = bounded.ids.nrows();
                    let num_ids = bounded.ids.ncols();
                    Self::Fixed {
                        matrices: vec![bounded],
                        len,
                        num_ids,
                    }
                }
                ResultIdsInner::Dynamic(ids) => Self::Dynamic(vec![ResultIdsInner::Dynamic(ids)]),
            },
            Self::Fixed {
                mut matrices,
                len,
                num_ids,
            } => match ids {
                ResultIdsInner::Fixed(bounded) => {
                    if bounded.ids.ncols() == num_ids {
                        let len = len + bounded.len();
                        matrices.push(bounded);
                        Self::Fixed {
                            matrices,
                            len,
                            num_ids,
                        }
                    } else {
                        let mut dynamic: Vec<_> =
                            matrices.into_iter().map(ResultIdsInner::Fixed).collect();
                        dynamic.push(ResultIdsInner::Fixed(bounded));
                        Self::Dynamic(dynamic)
                    }
                }
                ResultIdsInner::Dynamic(ids) => {
                    let mut dynamic: Vec<_> =
                        matrices.into_iter().map(ResultIdsInner::Fixed).collect();
                    dynamic.push(ResultIdsInner::Dynamic(ids));
                    Self::Dynamic(dynamic)
                }
            },
            Self::Dynamic(mut dynamic) => {
                dynamic.push(ids);
                Self::Dynamic(dynamic)
            }
        };
    }

    /// Consume `self`, producing a single [`ResultIds<I>`] containing the concatenation
    /// of all pushed IDs.
    pub(crate) fn finish(self) -> ResultIds<I> {
        // The internal logic is as follows:
        // * If we are empty, we return an empty dynamic IDs container.
        // * If we are fixed, we allocate a new **single** matrix and copy all IDs into it.
        // * If we are dynamic, we concatenate all dynamic ID vectors into a single dynamic container.

        match self {
            Self::Empty => ResultIds::new(ResultIdsInner::Dynamic(Vec::new())),
            Self::Fixed {
                matrices,
                len,
                num_ids,
            } => {
                let mut dst = Matrix::new(views::Init(|| I::default()), len, num_ids);
                let mut lengths = Vec::with_capacity(len);

                let mut output_row = 0;
                for bounded in matrices {
                    for row in bounded.ids.row_iter() {
                        dst.row_mut(output_row).clone_from_slice(row);
                        output_row += 1;
                    }
                    lengths.extend_from_slice(&bounded.lengths);
                }

                ResultIds::new(ResultIdsInner::Fixed(Bounded::new(dst, lengths)))
            }
            Self::Dynamic(all) => {
                let mut dst = Vec::<Vec<I>>::new();
                for ids in all {
                    match ids {
                        ResultIdsInner::Fixed(bounded) => {
                            bounded.iter().for_each(|row| dst.push(row.into()));
                        }
                        ResultIdsInner::Dynamic(dynamic) => {
                            dynamic.into_iter().for_each(|i| dst.push(i));
                        }
                    }
                }

                ResultIds::new(ResultIdsInner::Dynamic(dst))
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::recall::Rows;

    // Helper to create a Bounded instance for testing
    fn make_bounded(data: Vec<Vec<u32>>) -> Bounded<u32> {
        let nrows = data.len();
        let ncols = data.iter().map(|v| v.len()).max().unwrap_or(0);

        let mut matrix = Matrix::new(0u32, nrows, ncols);
        let mut lengths = Vec::with_capacity(nrows);

        for (row, row_data) in std::iter::zip(matrix.row_iter_mut(), data.iter()) {
            let len = std::iter::zip(row.iter_mut(), row_data.iter())
                .map(|(dst, src)| {
                    *dst = *src;
                })
                .count();
            lengths.push(len);
        }

        Bounded::new(matrix, lengths)
    }

    #[test]
    fn test_bounded_new_valid() {
        let matrix = Matrix::new(0u32, 3, 5);
        let lengths = vec![2, 3, 1];
        let bounded = Bounded::new(matrix, lengths);

        assert_eq!(bounded.len(), 3);
    }

    #[test]
    fn test_bounded_length_clamping() {
        let matrix = Matrix::new(0u32, 3, 3);
        let lengths = vec![2, 3, 5]; // Last length exceeds number of columns
        let bounded = Bounded::new(matrix, lengths);

        assert_eq!(bounded.len(), 3);
        assert_eq!(bounded.row(0), &[0, 0]);
        assert_eq!(bounded.row(1), &[0, 0, 0]);
        assert_eq!(bounded.row(2), &[0, 0, 0]); // Clamped to 3 columns

        let rows: Vec<&[u32]> = bounded.iter().collect();
        assert_eq!(rows[0], &[0, 0]);
        assert_eq!(rows[1], &[0, 0, 0]);
        assert_eq!(rows[2], &[0, 0, 0]); // Clamped to 3 columns
    }

    #[test]
    #[should_panic(expected = "an internal invariant was not upheld")]
    fn test_bounded_new_mismatched_lengths() {
        let matrix = Matrix::new(0u32, 3, 5);
        let lengths = vec![2, 3]; // Only 2 lengths for 3 rows
        Bounded::new(matrix, lengths);
    }

    #[test]
    fn test_bounded() {
        let bounded = make_bounded(vec![vec![1, 2], vec![3, 4, 5], vec![6]]);
        assert_eq!(bounded.len(), 3);

        // `Rows`
        assert_eq!(bounded.nrows(), 3);
        assert_eq!(bounded.row(0), &[1, 2]);
        assert_eq!(bounded.row(1), &[3, 4, 5]);
        assert_eq!(bounded.row(2), &[6]);
        assert_eq!(bounded.ncols(), None);

        // Iterator
        let rows: Vec<&[u32]> = bounded.iter().collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], &[1, 2]);
        assert_eq!(rows[1], &[3, 4, 5]);
        assert_eq!(rows[2], &[6]);
    }

    #[test]
    fn test_result_ids_inner_fixed() {
        let bounded = make_bounded(vec![vec![1, 2], vec![3, 4, 5]]);
        let inner = ResultIdsInner::Fixed(bounded);

        let rows = inner.as_rows();
        assert_eq!(rows.nrows(), 2);
        assert_eq!(rows.row(0), &[1, 2]);
        assert_eq!(rows.row(1), &[3, 4, 5]);
    }

    #[test]
    fn test_result_ids_inner_dynamic() {
        let vecs = vec![vec![1, 2, 3], vec![4], vec![5, 6]];
        let inner = ResultIdsInner::Dynamic(vecs);

        let rows = inner.as_rows();
        assert_eq!(rows.nrows(), 3);
        assert_eq!(rows.row(0), &[1, 2, 3]);
        assert_eq!(rows.row(1), &[4]);
        assert_eq!(rows.row(2), &[5, 6]);
    }

    #[test]
    fn test_result_ids_wrapper() {
        let bounded = make_bounded(vec![vec![10], vec![20, 30]]);
        let result = ResultIds::new(ResultIdsInner::Fixed(bounded));

        let rows = result.as_rows();
        assert_eq!(rows.nrows(), 2);
        assert_eq!(rows.row(0), &[10]);
        assert_eq!(rows.row(1), &[20, 30]);
    }

    // IdAggregator Tests

    #[test]
    fn test_aggregator_empty_finish() {
        let aggregator = IdAggregator::<u32>::new();
        let result = aggregator.finish();

        let rows = result.as_rows();
        assert_eq!(rows.nrows(), 0);
        assert_eq!(rows.ncols(), None);
    }

    #[test]
    fn test_aggregator_empty_to_fixed() {
        let mut aggregator = IdAggregator::new();

        let bounded = make_bounded(vec![vec![1, 2], vec![3], vec![4, 5]]);
        aggregator.push(ResultIdsInner::Fixed(bounded));

        // Should be in Fixed state
        match aggregator {
            IdAggregator::Fixed { len, num_ids, .. } => {
                assert_eq!(len, 3);
                assert_eq!(num_ids, 2);
            }
            _ => panic!("Expected Fixed state"),
        }

        let finished = aggregator.finish();
        let rows = finished.as_rows();
        assert_eq!(rows.nrows(), 3);
        assert_eq!(rows.row(0), &[1, 2]);
        assert_eq!(rows.row(1), &[3]);
        assert_eq!(rows.row(2), &[4, 5]);
    }

    #[test]
    fn test_aggregator_empty_to_dynamic() {
        let mut aggregator = IdAggregator::new();

        let vecs = vec![vec![1, 2, 3], vec![4]];
        aggregator.push(ResultIdsInner::Dynamic(vecs));

        // Should be in Dynamic state
        match aggregator {
            IdAggregator::Dynamic(ref inner) => {
                assert_eq!(inner.len(), 1);
            }
            _ => panic!("Expected Dynamic state"),
        }

        let finished = aggregator.finish();
        let rows = finished.as_rows();
        assert_eq!(rows.nrows(), 2);
        assert_eq!(rows.row(0), &[1, 2, 3]);
        assert_eq!(rows.row(1), &[4]);
    }

    #[test]
    fn test_aggregator_fixed_stays_fixed_same_size() {
        let mut aggregator = IdAggregator::new();

        // Push first bounded with 3 columns
        let bounded1 = make_bounded(vec![vec![1, 2, 3], vec![4, 5]]);
        aggregator.push(ResultIdsInner::Fixed(bounded1));

        // Push second bounded with 3 columns
        let bounded2 = make_bounded(vec![vec![6, 7, 8]]);
        aggregator.push(ResultIdsInner::Fixed(bounded2));

        // Should still be in Fixed state with accumulated length
        match &aggregator {
            IdAggregator::Fixed {
                len,
                num_ids,
                matrices,
            } => {
                assert_eq!(*len, 3); // 2 + 1 rows
                assert_eq!(*num_ids, 3);
                assert_eq!(matrices.len(), 2);
            }
            _ => panic!("Expected Fixed state"),
        }

        let finished = aggregator.finish();
        let rows = finished.as_rows();
        assert_eq!(rows.nrows(), 3);
        assert_eq!(rows.row(0), &[1, 2, 3]);
        assert_eq!(rows.row(1), &[4, 5]);
        assert_eq!(rows.row(2), &[6, 7, 8]);
    }

    #[test]
    fn test_aggregator_fixed_to_dynamic_different_sizes() {
        let mut aggregator = IdAggregator::new();

        // Push first bounded with 2 columns
        let bounded1 = make_bounded(vec![vec![1, 2], vec![3, 4]]);
        aggregator.push(ResultIdsInner::Fixed(bounded1));

        // Push second bounded with 3 columns (different size)
        let bounded2 = make_bounded(vec![vec![5, 6, 7]]);
        aggregator.push(ResultIdsInner::Fixed(bounded2));

        // Should transition to Dynamic
        match aggregator {
            IdAggregator::Dynamic(ref inner) => {
                assert_eq!(inner.len(), 2);
            }
            _ => panic!("Expected Dynamic state after size mismatch"),
        }

        let finished = aggregator.finish();
        let rows = finished.as_rows();
        assert_eq!(rows.nrows(), 3);
        assert_eq!(rows.row(0), &[1, 2]);
        assert_eq!(rows.row(1), &[3, 4]);
        assert_eq!(rows.row(2), &[5, 6, 7]);
    }

    #[test]
    fn test_aggregator_fixed_to_dynamic_incoming_dynamic() {
        let mut aggregator = IdAggregator::new();

        // Start with Fixed
        let bounded = make_bounded(vec![vec![1, 2], vec![3, 4]]);
        aggregator.push(ResultIdsInner::Fixed(bounded));

        // Push dynamic
        let vecs = vec![vec![5, 6, 7]];
        aggregator.push(ResultIdsInner::Dynamic(vecs));

        // Should transition to Dynamic
        match aggregator {
            IdAggregator::Dynamic(ref inner) => {
                assert_eq!(inner.len(), 2);
            }
            _ => panic!("Expected Dynamic state"),
        }

        let finished = aggregator.finish();
        let rows = finished.as_rows();
        assert_eq!(rows.nrows(), 3);
        assert_eq!(rows.row(0), &[1, 2]);
        assert_eq!(rows.row(1), &[3, 4]);
        assert_eq!(rows.row(2), &[5, 6, 7]);
    }

    #[test]
    fn test_aggregator_dynamic_stays_dynamic() {
        let mut aggregator = IdAggregator::new();

        // Start with Dynamic
        let vecs1 = vec![vec![1, 2]];
        aggregator.push(ResultIdsInner::Dynamic(vecs1));

        // Push more dynamic
        let vecs2 = vec![vec![3, 4, 5]];
        aggregator.push(ResultIdsInner::Dynamic(vecs2));

        // Push bounded
        let bounded = make_bounded(vec![vec![6, 7]]);
        aggregator.push(ResultIdsInner::Fixed(bounded));

        // Should remain Dynamic
        match aggregator {
            IdAggregator::Dynamic(ref inner) => {
                assert_eq!(inner.len(), 3);
            }
            _ => panic!("Expected Dynamic state"),
        }

        let finished = aggregator.finish();
        let rows = finished.as_rows();
        assert_eq!(rows.nrows(), 3);
        assert_eq!(rows.row(0), &[1, 2]);
        assert_eq!(rows.row(1), &[3, 4, 5]);
        assert_eq!(rows.row(2), &[6, 7]);
    }

    // #[test]
    // fn test_aggregator_finish_fixed_single_matrix() {
    //     let mut aggregator = IdAggregator::new();

    //     let bounded1 = make_bounded(vec![vec![1, 2], vec![3, 4]]);
    //     aggregator.push(ResultIdsInner::Fixed(bounded1));

    //     let bounded2 = make_bounded(vec![vec![5, 6], vec![7, 8]]);
    //     aggregator.push(ResultIdsInner::Fixed(bounded2));

    //     let result = aggregator.finish();
    //     let rows = result.as_rows();

    //     assert_eq!(rows.nrows(), 4);
    //     assert_eq!(rows.row(0), &[1, 2]);
    //     assert_eq!(rows.row(1), &[3, 4]);
    //     assert_eq!(rows.row(2), &[5, 6]);
    //     assert_eq!(rows.row(3), &[7, 8]);
    // }

    // #[test]
    // fn test_aggregator_finish_dynamic() {
    //     let mut aggregator = IdAggregator::new();

    //     let vecs1 = vec![vec![1, 2, 3], vec![4]];
    //     aggregator.push(ResultIdsInner::Dynamic(vecs1));

    //     let bounded = make_bounded(vec![vec![5, 6]]);
    //     aggregator.push(ResultIdsInner::Fixed(bounded));

    //     let vecs2 = vec![vec![7, 8, 9, 10]];
    //     aggregator.push(ResultIdsInner::Dynamic(vecs2));

    //     let result = aggregator.finish();
    //     let rows = result.as_rows();

    //     assert_eq!(rows.nrows(), 4);
    //     assert_eq!(rows.row(0), &[1, 2, 3]);
    //     assert_eq!(rows.row(1), &[4]);
    //     assert_eq!(rows.row(2), &[5, 6]);
    //     assert_eq!(rows.row(3), &[7, 8, 9, 10]);
    // }

    // #[test]
    // fn test_aggregator_finish_preserves_variable_lengths() {
    //     let mut aggregator = IdAggregator::new();

    //     // Different row lengths with same max columns
    //     let bounded = make_bounded(vec![vec![1, 2, 3], vec![4], vec![5, 6]]);
    //     aggregator.push(ResultIdsInner::Fixed(bounded));

    //     let result = aggregator.finish();
    //     let rows = result.as_rows();

    //     assert_eq!(rows.nrows(), 3);
    //     assert_eq!(rows.row(0), &[1, 2, 3]);
    //     assert_eq!(rows.row(1), &[4]);
    //     assert_eq!(rows.row(2), &[5, 6]);
    // }
}
