/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Trivial in-memory provider, iterator, and strategy used for unit-testing the flat
//! search infrastructure.
//!
//! This is intentionally simple: vectors live in a `Vec<Vec<f32>>`, ids are `u32`, and
//! distance is squared Euclidean. It exists so the trait shapes in [`crate::flat`] can be
//! exercised end-to-end without dragging in any provider-side machinery.

use diskann_utils::future::SendFuture;
use diskann_vector::PreprocessedDistanceFunction;
use thiserror::Error;

use crate::{
    always_escalate,
    ANNError, ANNErrorKind,
    flat::{FlatIterator, FlatPostProcess, FlatSearchStrategy},
    graph::SearchOutputBuffer,
    neighbor::Neighbor,
    provider::{DataProvider, DefaultContext, HasId, NoopGuard},
};

/// Trivial flat provider holding a list of fixed-dimension `f32` vectors.
#[derive(Debug)]
pub struct InMemoryFlatProvider {
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
}

impl InMemoryFlatProvider {
    pub fn new(dim: usize, vectors: Vec<Vec<f32>>) -> Self {
        Self { dim, vectors }
    }
}

#[derive(Debug, Error)]
#[error("invalid vector id {0}")]
pub struct InMemoryProviderError(u32);

impl From<InMemoryProviderError> for ANNError {
    #[track_caller]
    fn from(err: InMemoryProviderError) -> Self {
        ANNError::new(ANNErrorKind::IndexError, err)
    }
}

always_escalate!(InMemoryProviderError);

impl DataProvider for InMemoryFlatProvider {
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = InMemoryProviderError;
    type Guard = NoopGuard<u32>;

    fn to_internal_id(
        &self,
        _context: &Self::Context,
        gid: &u32,
    ) -> Result<u32, Self::Error> {
        if (*gid as usize) < self.vectors.len() {
            Ok(*gid)
        } else {
            Err(InMemoryProviderError(*gid))
        }
    }

    fn to_external_id(
        &self,
        _context: &Self::Context,
        id: u32,
    ) -> Result<u32, Self::Error> {
        if (id as usize) < self.vectors.len() {
            Ok(id)
        } else {
            Err(InMemoryProviderError(id))
        }
    }
}

/// Sequential iterator over [`InMemoryFlatProvider`].
pub struct InMemoryIterator<'a> {
    vectors: &'a [Vec<f32>],
    cursor: u32,
}

#[derive(Debug, Error)]
#[error("in-memory iterator does not error")]
pub struct InMemoryIteratorError;

impl From<InMemoryIteratorError> for ANNError {
    #[track_caller]
    fn from(err: InMemoryIteratorError) -> Self {
        ANNError::new(ANNErrorKind::IndexError, err)
    }
}

always_escalate!(InMemoryIteratorError);

impl<'a> HasId for InMemoryIterator<'a> {
    type Id = u32;
}

impl<'a> FlatIterator for InMemoryIterator<'a> {
    type ElementRef<'b> = &'b [f32];
    type Element<'b>
        = &'b [f32]
    where
        Self: 'b;
    type Error = InMemoryIteratorError;

    fn next(
        &mut self,
    ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>> {
        let idx = self.cursor as usize;
        let result = self.vectors.get(idx).map(|v| {
            self.cursor += 1;
            (idx as u32, v.as_slice())
        });
        std::future::ready(Ok(result))
    }
}

/// Squared Euclidean computer: holds a copy of the query and scores against `&[f32]`.
#[derive(Debug, Clone)]
pub struct L2QueryComputer {
    query: Vec<f32>,
}

impl<'a> PreprocessedDistanceFunction<&'a [f32], f32> for L2QueryComputer {
    fn evaluate_similarity(&self, changing: &'a [f32]) -> f32 {
        debug_assert_eq!(self.query.len(), changing.len());
        self.query
            .iter()
            .zip(changing.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }
}

/// Strategy: produces an [`InMemoryIterator`] and an [`L2QueryComputer`].
#[derive(Debug, Default, Clone, Copy)]
pub struct InMemoryStrategy;

#[derive(Debug, Error)]
pub enum InMemoryStrategyError {
    #[error("query length {query} does not match provider dimension {dim}")]
    DimMismatch { query: usize, dim: usize },
}

impl From<InMemoryStrategyError> for ANNError {
    #[track_caller]
    fn from(err: InMemoryStrategyError) -> Self {
        ANNError::new(ANNErrorKind::IndexError, err)
    }
}

impl FlatSearchStrategy<InMemoryFlatProvider, [f32]> for InMemoryStrategy {
    type Iter<'a> = InMemoryIterator<'a>;
    type QueryComputer = L2QueryComputer;
    type Error = InMemoryStrategyError;

    fn create_iter<'a>(
        &'a self,
        provider: &'a InMemoryFlatProvider,
        _context: &'a DefaultContext,
    ) -> Result<Self::Iter<'a>, Self::Error> {
        Ok(InMemoryIterator {
            vectors: &provider.vectors,
            cursor: 0,
        })
    }

    fn build_query_computer(&self, query: &[f32]) -> Result<Self::QueryComputer, Self::Error> {
        Ok(L2QueryComputer {
            query: query.to_vec(),
        })
    }
}

/// Post-processor that copies the surviving `(id, distance)` pairs straight to output.
///
/// Identical in behavior to [`crate::flat::CopyFlatIds`] but typed concretely against the
/// in-memory iterator, useful in tests where we want to assert against the exact output
/// shape.
#[derive(Debug, Default, Clone, Copy)]
pub struct CopyInMemoryHits;

impl<'a> FlatPostProcess<InMemoryIterator<'a>, [f32]> for CopyInMemoryHits {
    type Error = crate::error::Infallible;

    fn post_process<I, B>(
        &self,
        _iter: &mut InMemoryIterator<'a>,
        _query: &[f32],
        candidates: I,
        output: &mut B,
    ) -> impl SendFuture<Result<usize, Self::Error>>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        let count = output.extend(candidates.map(|n| (n.id, n.distance)));
        std::future::ready(Ok(count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        flat::{CopyFlatIds, FlatIndex, validate_k},
        neighbor::Neighbor,
    };

    fn build_provider() -> InMemoryFlatProvider {
        // 5 two-dimensional points; the closest to (0.0, 0.0) is index 0.
        InMemoryFlatProvider::new(
            2,
            vec![
                vec![0.1, 0.0],  // d^2 = 0.01
                vec![1.0, 0.0],  // d^2 = 1.00
                vec![0.0, 0.5],  // d^2 = 0.25
                vec![5.0, 5.0],  // d^2 = 50.0
                vec![-0.2, 0.1], // d^2 = 0.05
            ],
        )
    }

    #[tokio::test]
    async fn knn_flat_returns_top_k_in_distance_order() {
        let provider = build_provider();
        let index = FlatIndex::new(provider);
        let strategy = InMemoryStrategy;
        let processor = CopyInMemoryHits;
        let query = vec![0.0_f32, 0.0];

        let mut output: Vec<Neighbor<u32>> = Vec::new();
        let stats = index
            .knn_search(
                validate_k(3).unwrap(),
                &strategy,
                &processor,
                &DefaultContext,
                query.as_slice(),
                &mut output,
            )
            .await
            .expect("search succeeds");

        assert_eq!(stats.cmps, 5);
        assert_eq!(stats.result_count, 3);

        let ids: Vec<u32> = output.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![0, 4, 2]);
    }

    #[tokio::test]
    async fn knn_flat_with_k_larger_than_n_returns_all() {
        let provider = build_provider();
        let index = FlatIndex::new(provider);
        let strategy = InMemoryStrategy;
        let processor = CopyFlatIds;
        let query = vec![0.0_f32, 0.0];

        let mut output: Vec<Neighbor<u32>> = Vec::new();
        let stats = index
            .knn_search(
                validate_k(100).unwrap(),
                &strategy,
                &processor,
                &DefaultContext,
                query.as_slice(),
                &mut output,
            )
            .await
            .expect("search succeeds");

        assert_eq!(stats.cmps, 5);
        assert_eq!(stats.result_count, 5);
    }

    #[test]
    fn knn_flat_rejects_zero_k() {
        assert!(validate_k(0).is_err());
    }
}
