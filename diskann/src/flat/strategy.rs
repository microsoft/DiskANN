/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Core flat-search trait: [`DistancesUnordered`].

use std::fmt::Debug;

use diskann_utils::future::SendFuture;

use crate::{error::ToRanked, provider::HasId};

/// Receives distances in scan order. Batch submission preserves the order of
/// repeated [`Self::add`] calls, including candidates tied at the top-k boundary.
pub trait DistanceHandler<I: Copy>: Send {
    fn add(&mut self, id: I, distance: f32);

    /// Whether staging individually computed distances into temporary batches
    /// is worthwhile for this handler. Consult once before allocating or filling
    /// such buffers. The default preserves batching for existing handlers.
    ///
    /// This is only a staging hint: already computed blocks (for example PQ or
    /// GEMM output) should still use [`Self::add_batch`], even when this is false.
    #[inline]
    fn prefers_buffered_batches(&self) -> bool {
        true
    }

    /// Submit a batch in slice order. The default implementation forwards each
    /// candidate to [`Self::add`]; optimized handlers can screen a whole batch.
    ///
    /// # Panics
    ///
    /// Panics if the ID and distance slices have different lengths.
    #[track_caller]
    fn add_batch(&mut self, ids: &[I], distances: &[f32]) {
        assert_eq!(ids.len(), distances.len(), "ID and distance lengths differ");
        for (&id, &distance) in ids.iter().zip(distances) {
            self.add(id, distance);
        }
    }
}

impl<I: Copy, F: FnMut(I, f32) + Send> DistanceHandler<I> for F {
    #[inline]
    fn add(&mut self, id: I, distance: f32) {
        self(id, distance);
    }
}

/// A complete, unordered scan of a candidate set.
///
/// Implementations drive the scan rather than exposing elements for the caller to fetch.
/// For each candidate in the scan, the implementation computes its distance and invokes
/// the supplied callback with the candidate's `(id, distance)` pair. No ordering of those
/// callbacks is guaranteed.
///
/// A visitor represents one initialized scan. It may retain any state needed while
/// scanning, and remains available to the search post-processor after the scan completes.
pub trait DistancesUnordered: HasId + Send + Sync {
    /// The error type returned when the visitor cannot complete its scan.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Scan all candidates represented by this visitor and invoke `f` for each result.
    ///
    /// The callback may have been invoked before an error is returned. In that case,
    /// those results form an incomplete scan and must not be treated as exhaustive.
    fn distances_unordered<F>(&mut self, f: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(Self::Id, f32);

    /// Scan into a handler that can accept batches. Existing visitors use the
    /// scalar callback by default; visitors with contiguous distance buffers can
    /// override this to submit them without allocating per-candidate results.
    /// Visitors that would create temporary distance batches solely for submission
    /// should consult [`DistanceHandler::prefers_buffered_batches`] first.
    ///
    /// Emit the same candidates in the same order as [`Self::distances_unordered`].
    /// Submit every candidate the scalar method would emit, including NaNs and
    /// distances rejected by top-k, so counts remain independent of batch size. Errors have
    /// the same incomplete-scan semantics as the scalar method.
    fn distances_unordered_with<H>(
        &mut self,
        handler: &mut H,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        H: DistanceHandler<Self::Id>,
    {
        self.distances_unordered(|id, distance| handler.add(id, distance))
    }
}

#[cfg(test)]
mod tests {
    //! Direct [`DistancesUnordered`] impls over in-memory fixtures, including a
    //! happy-path scanner and one that fails mid-stream.

    use diskann_utils::future::SendFuture;
    use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};

    use super::*;
    use crate::{always_escalate, convert_error, error::Infallible, utils::VectorRepr};

    #[test]
    fn scalar_handler_accepts_empty_batches_and_preserves_order() {
        let mut results = Vec::new();
        let mut handler = |id, distance: f32| results.push((id, distance.to_bits()));
        handler.add_batch(&[], &[]);
        handler.add(7, 0.0);
        handler.add_batch(&[2, 9, 1], &[-0.0, f32::NAN, f32::INFINITY]);
        assert_eq!(
            results,
            [
                (7, 0.0f32.to_bits()),
                (2, (-0.0f32).to_bits()),
                (9, f32::NAN.to_bits()),
                (1, f32::INFINITY.to_bits())
            ]
        );
    }

    #[test]
    #[should_panic(expected = "ID and distance lengths differ")]
    fn mismatched_batch_lengths_are_rejected() {
        let mut handler = |_: u32, _: f32| {};
        handler.add_batch(&[1], &[]);
    }

    /// Sample dataset shared by every test below.
    fn sample_items() -> Vec<(u32, Vec<f32>)> {
        vec![
            (10, vec![0.0, 0.0]),
            (11, vec![1.0, 0.0]),
            (12, vec![0.0, 2.0]),
        ]
    }

    /////////////////////////////
    // Scanner yielding slices //
    /////////////////////////////

    /// Scans `items` in order, scoring each with the supplied computer.
    struct Scanner {
        items: Vec<(u32, Vec<f32>)>,
        computer: <f32 as VectorRepr>::QueryDistance,
    }

    impl HasId for Scanner {
        type Id = u32;
    }

    impl DistancesUnordered for Scanner {
        type Error = Infallible;

        fn distances_unordered<F>(&mut self, mut f: F) -> impl SendFuture<Result<(), Self::Error>>
        where
            F: Send + FnMut(Self::Id, f32),
        {
            async move {
                for (id, v) in &self.items {
                    let dist = self.computer.evaluate_similarity(v.as_slice());
                    f(*id, dist);
                }
                Ok(())
            }
        }
    }

    /// Direct [`DistancesUnordered`] impl yields the expected `(id, distance)` pairs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distances_unordered_scanner() {
        let query = vec![0.5_f32, 0.9];
        let expected_computer = f32::query_distance(&query, Metric::L2);

        let expected: Vec<(u32, f32)> = sample_items()
            .into_iter()
            .map(|(id, v)| (id, expected_computer.evaluate_similarity(v.as_slice())))
            .collect();

        let mut scanner = Scanner {
            items: sample_items(),
            computer: f32::query_distance(&query, Metric::L2),
        };

        let mut seen: Vec<(u32, f32)> = Vec::new();
        scanner
            .distances_unordered(|id, d| seen.push((id, d)))
            .await
            .unwrap();
        assert_eq!(seen, expected);
    }

    struct BorrowingScanner<'a> {
        items: &'a [(u32, f32)],
        query: &'a f32,
    }

    impl HasId for BorrowingScanner<'_> {
        type Id = u32;
    }

    impl DistancesUnordered for BorrowingScanner<'_> {
        type Error = Infallible;

        fn distances_unordered<F>(&mut self, mut f: F) -> impl SendFuture<Result<(), Self::Error>>
        where
            F: Send + FnMut(Self::Id, f32),
        {
            async move {
                for (id, value) in self.items {
                    f(*id, (*value - *self.query).abs());
                }
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn accessor_can_borrow_query_state() {
        let items = [(10, 1.0), (11, 4.0)];
        let query = 2.0;
        let mut scanner = BorrowingScanner {
            items: &items,
            query: &query,
        };
        let mut seen = Vec::new();

        scanner
            .distances_unordered(|id, distance| seen.push((id, distance)))
            .await
            .unwrap();

        assert_eq!(seen, [(10, 1.0), (11, 2.0)]);
    }

    ///////////////////////////
    // Failing scanner       //
    ///////////////////////////

    /// Non-recoverable error type returned by [`Failing`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
    #[error("synthetic scan failure at id {0}")]
    struct Boom(u32);

    always_escalate!(Boom);
    convert_error!(Boom);

    /// Scans `items`, but returns `Err(Boom(id))` exactly once after `fail_after`
    /// successful yields.
    struct Failing {
        items: Vec<(u32, Vec<f32>)>,
        fail_after: usize,
        computer: <f32 as VectorRepr>::QueryDistance,
    }

    impl HasId for Failing {
        type Id = u32;
    }

    impl DistancesUnordered for Failing {
        type Error = Boom;

        fn distances_unordered<F>(&mut self, mut f: F) -> impl SendFuture<Result<(), Self::Error>>
        where
            F: Send + FnMut(Self::Id, f32),
        {
            async move {
                for (i, (id, v)) in self.items.iter().enumerate() {
                    if i == self.fail_after {
                        return Err(Boom(*id));
                    }
                    let dist = self.computer.evaluate_similarity(v.as_slice());
                    f(*id, dist);
                }
                Ok(())
            }
        }
    }

    /// An error returned mid-scan propagates up, and the closure stops being invoked
    /// at the failure point.
    #[tokio::test]
    async fn failures_midstream() {
        let mut scanner = Failing {
            items: sample_items(),
            fail_after: 1, // Yield item 0 successfully, fail on item 1.
            computer: f32::query_distance(&[0.0, 0.0], Metric::L2),
        };

        let mut seen: Vec<u32> = Vec::new();
        let err = scanner
            .distances_unordered(|id, _d| seen.push(id))
            .await
            .expect_err("Failing scanner must surface its error");

        assert_eq!(err, Boom(11));
        assert_eq!(
            seen,
            vec![10],
            "the closure must only see items yielded before the failure",
        );
    }
}
