/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Core flat-search traits: [`DistancesUnordered`] and [`SearchStrategy`].

use std::fmt::Debug;

use diskann_utils::future::SendFuture;

use crate::{
    error::{StandardError, ToRanked},
    provider::{DataProvider, HasId},
};

/// Per-query accessor that drives a complete flat scan.
///
/// The accessor owns the query-specific computation state so implementations can fuse
/// query preprocessing, data access, batching, filtering, and distance computation.
pub trait DistancesUnordered: HasId + Send + Sync {
    /// The error type for [`Self::distances_unordered`].
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Drive the entire scan, invoking `f` with each `(id, distance)` pair.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot complete the scan.
    fn distances_unordered<F>(&mut self, f: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(Self::Id, f32);
}

/// Per-call configuration that constructs a query-aware [`DistancesUnordered`] visitor.
pub trait SearchStrategy<'a, P, T>: Send + Sync
where
    P: DataProvider,
{
    /// The query-aware visitor used to execute the scan.
    type Visitor: DistancesUnordered<Id = P::InternalId>;

    /// An error that can occur while constructing [`Self::Visitor`].
    type Error: StandardError;

    /// Construct a fresh visitor for `query`.
    ///
    /// # Errors
    ///
    /// Returns an error when query preprocessing or visitor initialization fails.
    fn create_visitor(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
        query: T,
    ) -> Result<Self::Visitor, Self::Error>;
}

#[cfg(test)]
mod tests {
    //! Direct [`DistancesUnordered`] impls over in-memory fixtures, including a
    //! happy-path scanner and one that fails mid-stream.

    use diskann_utils::future::SendFuture;
    use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};

    use super::*;
    use crate::{always_escalate, convert_error, error::Infallible, utils::VectorRepr};

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
