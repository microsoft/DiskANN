/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Core flat-search traits: [`DistancesUnordered`] and [`SearchStrategy`].

use std::fmt::Debug;

use diskann_utils::future::SendFuture;
use diskann_vector::PreprocessedDistanceFunction;

use crate::{
    error::{StandardError, ToRanked},
    provider::DataProvider,
};

/// Fused iterate-and-score primitive over the elements of a flat index.
///
/// Implementations drive an entire scan over the underlying data, scoring each element
/// with the supplied computer `C` and invoking `f` with the resulting `(id, distance)`
/// pair. The associated [`Self::ElementRef`] is the reference shape on which `C` must
/// be able to compute distances.
pub trait DistancesUnordered<C>: Send + Sync
where
    C: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32>,
{
    /// Lifetime is intentionally unconstrained so it can appear under HRTB without
    /// inducing a `'static` bound on `Self`.
    type ElementRef<'a>;

    /// Id type yielded by the underlying data backend, used to uniquely identify
    /// each element passed to the closure of [`Self::distances_unordered`].
    type Id;

    /// The error type for [`Self::distances_unordered`].
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Drive the entire scan, scoring each element with `computer` and invoking `f`
    /// with the resulting `(id, distance)` pair.
    fn distances_unordered<F>(
        &mut self,
        computer: &C,
        f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(Self::Id, f32);
}

/// Per-call configuration that knows how to construct a per-query
/// [`DistancesUnordered`] visitor for a provider, and the [`Self::QueryComputer`] used
/// to score each element during the scan.
pub trait SearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
{
    /// The reference element shape on which [`Self::QueryComputer`] computes
    /// distances.
    type ElementRef<'a>;

    /// Id type yielded by the `Self::Visitor`.
    type Id;

    /// The concrete query-computer type.
    type QueryComputer: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32>
        + Send
        + Sync
        + 'static;

    /// The error type for [`Self::build_query_computer`].
    type QueryComputerError: StandardError;

    /// The visitor type produced by [`Self::create_visitor`].
    type Visitor<'a>: for<'b> DistancesUnordered<
            Self::QueryComputer,
            ElementRef<'b> = Self::ElementRef<'b>,
            Id = Self::Id,
        >
    where
        Self: 'a,
        P: 'a;

    /// The error type for [`Self::create_visitor`].
    type Error: StandardError;

    /// Construct a fresh visitor over `provider` for the given request `context`.
    fn create_visitor<'a>(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
    ) -> Result<Self::Visitor<'a>, Self::Error>;

    /// Construct the per-query computer.
    fn build_query_computer(
        &self,
        query: T,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError>;
}

#[cfg(test)]
mod tests {
    //! Direct [`DistancesUnordered`] impls over a few in-memory fixtures: a
    //! happy-path scanner over `&[f32]` elements, a scanner whose `ElementRef<'a>`
    //! is a lifetime-carrying non-reference type, and a scanner that fails
    //! mid-stream.

    use std::marker::PhantomData;

    use diskann_utils::future::SendFuture;
    use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};

    use super::*;
    use crate::{ANNError, always_escalate, error::Infallible, utils::VectorRepr};

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
    }

    impl DistancesUnordered<<f32 as VectorRepr>::QueryDistance> for Scanner {
        type ElementRef<'a> = &'a [f32];
        type Id = u32;
        type Error = Infallible;

        fn distances_unordered<F>(
            &mut self,
            computer: &<f32 as VectorRepr>::QueryDistance,
            mut f: F,
        ) -> impl SendFuture<Result<(), Self::Error>>
        where
            F: Send + FnMut(Self::Id, f32),
        {
            async move {
                for (id, v) in &self.items {
                    let dist = computer.evaluate_similarity(v.as_slice());
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
        let computer = f32::query_distance(&query, Metric::L2);

        let expected: Vec<(u32, f32)> = sample_items()
            .into_iter()
            .map(|(id, v)| (id, computer.evaluate_similarity(v.as_slice())))
            .collect();

        let mut scanner = Scanner {
            items: sample_items(),
        };

        let mut seen: Vec<(u32, f32)> = Vec::new();
        scanner
            .distances_unordered(&computer, |id, d| seen.push((id, d)))
            .await
            .unwrap();
        assert_eq!(seen, expected);
    }

    ///////////////////////////
    // Failing scanner       //
    ///////////////////////////

    /// Non-recoverable error type returned by [`Failing`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
    #[error("synthetic scan failure at id {0}")]
    struct Boom(u32);

    always_escalate!(Boom);

    impl From<Boom> for ANNError {
        #[track_caller]
        fn from(boom: Boom) -> ANNError {
            ANNError::opaque(boom)
        }
    }

    /// Scans `items`, but returns `Err(Boom(id))` exactly once after `fail_after`
    /// successful yields.
    struct Failing {
        items: Vec<(u32, Vec<f32>)>,
        fail_after: usize,
    }

    impl DistancesUnordered<<f32 as VectorRepr>::QueryDistance> for Failing {
        type ElementRef<'a> = &'a [f32];
        type Id = u32;
        type Error = Boom;

        fn distances_unordered<F>(
            &mut self,
            computer: &<f32 as VectorRepr>::QueryDistance,
            mut f: F,
        ) -> impl SendFuture<Result<(), Self::Error>>
        where
            F: Send + FnMut(Self::Id, f32),
        {
            async move {
                for (i, (id, v)) in self.items.iter().enumerate() {
                    if i == self.fail_after {
                        return Err(Boom(*id));
                    }
                    let dist = computer.evaluate_similarity(v.as_slice());
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
        };

        let query = vec![0.0_f32, 0.0];
        let computer = f32::query_distance(&query, Metric::L2);

        let mut seen: Vec<u32> = Vec::new();
        let err = scanner
            .distances_unordered(&computer, |id, _d| seen.push(id))
            .await
            .expect_err("Failing scanner must surface its error");

        assert_eq!(err, Boom(11));
        assert_eq!(
            seen,
            vec![10],
            "the closure must only see items yielded before the failure",
        );
    }

    /////////////////////////////////////////////
    // Lifetime-carrying concrete `ElementRef` //
    /////////////////////////////////////////////

    struct View<'a> {
        ptr: *const f32,
        len: usize,
        _phantom: PhantomData<&'a [f32]>,
    }

    // SAFETY: `View<'a>` semantically carries a `&'a [f32]`, which is `Send + Sync`.
    unsafe impl Send for View<'_> {}
    unsafe impl Sync for View<'_> {}

    /// Computer that reconstructs a `&[f32]` from a [`View`]'s ptr+len and
    /// computes inner product against a stored query.
    struct ViewComputer {
        query: Vec<f32>,
    }

    impl<'a> PreprocessedDistanceFunction<View<'a>, f32> for ViewComputer {
        fn evaluate_similarity(&self, v: View<'a>) -> f32 {
            // SAFETY: `v.ptr` / `v.len` were produced from a `&'a [f32]` held by the
            // scanner that owns the backing `Vec`; the phantom lifetime ties this view
            // to that borrow, so the slice is valid for the duration of this call.
            let s = unsafe { std::slice::from_raw_parts(v.ptr, v.len) };
            s.iter().zip(&self.query).map(|(a, b)| a * b).sum()
        }
    }

    /// Scans `rows`, yielding a [`View`] tied (via its phantom lifetime) to the
    /// borrow of the underlying `Vec<f32>`.
    struct ViewScanner {
        rows: Vec<(u32, Vec<f32>)>,
    }

    impl ViewScanner {
        fn iter<'a>(&self) -> impl Iterator<Item = (u32, View<'a>)> {
            self.rows.iter().map(|(x, y)| {
                (
                    *x,
                    View {
                        ptr: y.as_ptr(),
                        len: y.len(),
                        _phantom: PhantomData,
                    },
                )
            })
        }
    }

    impl DistancesUnordered<ViewComputer> for ViewScanner {
        type ElementRef<'a> = View<'a>;
        type Id = u32;
        type Error = Infallible;

        fn distances_unordered<F>(
            &mut self,
            computer: &ViewComputer,
            mut f: F,
        ) -> impl SendFuture<Result<(), Self::Error>>
        where
            F: Send + FnMut(Self::Id, f32),
        {
            async move {
                for (id, v) in self.iter() {
                    f(id, computer.evaluate_similarity(v));
                }
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn distances_unordered_lifetime_carrying_element_ref() {
        let mut scanner = ViewScanner {
            rows: vec![
                (10, vec![1.0, 0.0]),
                (11, vec![0.5, 0.5]),
                (12, vec![0.0, 2.0]),
            ],
        };
        let computer = ViewComputer {
            query: vec![1.0, 3.0],
        };
        let expected: Vec<(u32, f32)> = vec![
            (10, 1.0 * 1.0 + 0.0 * 3.0),
            (11, 0.5 * 1.0 + 0.5 * 3.0),
            (12, 0.0 * 1.0 + 2.0 * 3.0),
        ];

        let mut seen: Vec<(u32, f32)> = Vec::new();
        scanner
            .distances_unordered(&computer, |id, d| seen.push((id, d)))
            .await
            .unwrap();
        assert_eq!(seen, expected);
    }
}
