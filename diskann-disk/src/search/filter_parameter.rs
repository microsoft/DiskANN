/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Filter types for disk search.
//!
//! See `rfcs/01101-disk-beta-filter.md` for the design rationale.

use thiserror::Error;

/// Closure used to filter vector IDs during disk search.
///
/// The `u32` argument is `DiskProvider::InternalId`. All three invocation
/// sites (`flat_search`, `pq_distances`, `RerankAndFilter::post_process`)
/// pass internal IDs uniformly. On the disk path
/// `InternalId == ExternalId == u32` by construction, so callers keyed on
/// either ID space see the same value.
pub type Predicate<'a> = Box<dyn Fn(u32) -> bool + Send + Sync + 'a>;

/// Top-level search plan: graph traversal vs. linear scan.
///
/// Replaces the previous `(vector_filter, is_flat_search)` parameter pair on
/// `DiskIndexSearcher::search`.
pub enum SearchPlan<'a> {
    /// Brute-force linear scan. `Some(p)` applies `p` inline; `None` scans
    /// every vector (recall baseline).
    FlatScan { filter: Option<Predicate<'a>> },

    /// Graph traversal; `GraphMode` picks the algorithm and any modifier.
    Graph(GraphMode<'a>),
}

/// Graph-search variant. Invalid combinations (e.g. beta without a predicate)
/// are unrepresentable.
pub enum GraphMode<'a> {
    /// Plain greedy beam.
    Unfiltered,

    /// Greedy beam + hard post-filter (applied in `RerankAndFilter`).
    /// Traversal identical to `Unfiltered`.
    PostFilter(Predicate<'a>),

    /// Beta-biased beam: matching vectors' PQ distances multiplied by
    /// `beta` ∈ (0, 1] in `pq_distances`. Predicate also post-filters.
    BetaFilter { predicate: Predicate<'a>, beta: f32 },
    // Future graph algorithms slot in here as new variants. e.g.:
    //   Multihop { predicate: Predicate<'a> },
}

/// Validation error for `GraphMode::beta_filter`.
#[derive(Debug, Error)]
pub enum BetaError {
    /// `beta` was outside the valid `(0, 1]` range.
    #[error("beta must be in (0, 1], got {0}")]
    OutOfRange(f32),
}

impl<'a> GraphMode<'a> {
    /// Graph search + hard post-filter. No traversal-time modifier.
    pub fn post_filter<F>(predicate: F) -> Self
    where
        F: Fn(u32) -> bool + Send + Sync + 'a,
    {
        Self::PostFilter(Box::new(predicate))
    }

    /// Graph search with beta-biased traversal + hard post-filter.
    ///
    /// Fallible — returns `BetaError::OutOfRange` if `beta` is outside
    /// `(0, 1]`. Designed for callers that read `beta` from external input
    /// (JSON config, CLI args). Programmer-supplied literals can `.unwrap()`
    /// or `?`.
    pub fn beta_filter<F>(predicate: F, beta: f32) -> Result<Self, BetaError>
    where
        F: Fn(u32) -> bool + Send + Sync + 'a,
    {
        if !(beta > 0.0 && beta <= 1.0) {
            return Err(BetaError::OutOfRange(beta));
        }
        Ok(Self::BetaFilter {
            predicate: Box::new(predicate),
            beta,
        })
    }
}

impl<'a> SearchPlan<'a> {
    /// Flat scan over all vectors. Recall baseline.
    pub fn flat() -> Self {
        Self::FlatScan { filter: None }
    }

    /// Flat scan restricted to vectors that satisfy `predicate`.
    pub fn flat_filtered<F>(predicate: F) -> Self
    where
        F: Fn(u32) -> bool + Send + Sync + 'a,
    {
        Self::FlatScan {
            filter: Some(Box::new(predicate)),
        }
    }

    /// Graph search over all vectors. The default; no filter.
    pub fn graph() -> Self {
        Self::Graph(GraphMode::Unfiltered)
    }

    /// Graph search using the supplied `GraphMode`.
    pub fn graph_with(mode: GraphMode<'a>) -> Self {
        Self::Graph(mode)
    }
}

/// Disk-local projection of `SearchPlan` used inside the strategy.
///
/// `search_strategy()` is the only site that introspects `GraphMode`'s
/// variants; the resulting `FilterMode` is what `DiskAccessor` and
/// `RerankAndFilter` dispatch on. The sum type makes
/// `(no predicate, Some(beta))` unrepresentable at the strategy layer.
///
/// Variants hold `&dyn Fn` directly (not `&Box<dyn Fn>`), so calling the
/// predicate is one indirection instead of two. The owning `Box` still lives
/// in `SearchPlan`; the strategy projects via `predicate.as_ref()`.
#[derive(Clone, Copy)]
pub(crate) enum FilterMode<'a> {
    None,
    Filter(&'a (dyn Fn(u32) -> bool + Send + Sync + 'a)),
    BetaFilter {
        predicate: &'a (dyn Fn(u32) -> bool + Send + Sync + 'a),
        beta: f32,
    },
    // Future graph algorithms with their own supplementary data slot in here:
    //   Multihop { predicate, depth: u32 },
}

impl<'a> FilterMode<'a> {
    /// Returns the post-filter predicate, if any.
    ///
    /// Used by `RerankAndFilter` to apply a hard post-filter uniformly on
    /// both `Filter` and `BetaFilter` variants. Returns `None` for
    /// `FilterMode::None`.
    pub(crate) fn post_filter(
        &self,
    ) -> Option<&'a (dyn Fn(u32) -> bool + Send + Sync + 'a)> {
        match self {
            FilterMode::None => None,
            FilterMode::Filter(p) | FilterMode::BetaFilter { predicate: p, .. } => Some(*p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `GraphMode` doesn't derive `Debug` (its variants hold trait objects
    /// without `Debug` bounds), so we can't `.unwrap_err()` on the `Result`
    /// — match the err arm explicitly instead.
    fn expect_out_of_range(result: Result<GraphMode<'static>, BetaError>) {
        match result {
            Err(BetaError::OutOfRange(_)) => {}
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn beta_filter_rejects_zero() {
        expect_out_of_range(GraphMode::beta_filter(|_| true, 0.0));
    }

    #[test]
    fn beta_filter_rejects_greater_than_one() {
        expect_out_of_range(GraphMode::beta_filter(|_| true, 1.5));
    }

    #[test]
    fn beta_filter_rejects_negative() {
        expect_out_of_range(GraphMode::beta_filter(|_| true, -0.5));
    }

    #[test]
    fn beta_filter_accepts_one() {
        assert!(GraphMode::beta_filter(|_| true, 1.0).is_ok());
    }

    #[test]
    fn beta_filter_accepts_typical_value() {
        assert!(GraphMode::beta_filter(|_| true, 0.5).is_ok());
    }

    #[test]
    fn flat_no_filter_constructor() {
        let plan = SearchPlan::flat();
        assert!(matches!(plan, SearchPlan::FlatScan { filter: None }));
    }

    #[test]
    fn flat_filtered_constructor() {
        let plan = SearchPlan::flat_filtered(|id| id == 5);
        match plan {
            SearchPlan::FlatScan { filter: Some(p) } => assert!(p(5) && !p(4)),
            _ => panic!("expected FlatScan with filter"),
        }
    }

    #[test]
    fn graph_unfiltered_constructor() {
        let plan = SearchPlan::graph();
        assert!(matches!(plan, SearchPlan::Graph(GraphMode::Unfiltered)));
    }

    #[test]
    fn graph_post_filter_constructor() {
        let plan = SearchPlan::graph_with(GraphMode::post_filter(|id| id == 7));
        match plan {
            SearchPlan::Graph(GraphMode::PostFilter(p)) => assert!(p(7) && !p(8)),
            _ => panic!("expected Graph(PostFilter)"),
        }
    }

    // === FilterMode projection tests ===
    //
    // These verify that the post-filter predicate extraction matches what the
    // strategy/RerankAndFilter consumers expect: `None` projects to "accept all"
    // (no closure invocation); `Filter` and `BetaFilter` both expose the same
    // predicate uniformly (so post-filter behavior is identical between them).

    #[test]
    fn filter_mode_none_post_filter_is_none() {
        // No-filter zero-cost guard: the projection of `SearchPlan::graph()` /
        // `SearchPlan::flat()` is `FilterMode::None`. Consumers short-circuit
        // via `.map_or(true, ...)` and invoke no closure.
        let m: FilterMode<'_> = FilterMode::None;
        assert!(m.post_filter().is_none());
    }

    #[test]
    fn filter_mode_filter_post_filter_returns_predicate() {
        // Projection of `SearchPlan::graph_with(GraphMode::PostFilter(_))` and
        // `SearchPlan::FlatScan { filter: Some(_) }` both map to this variant.
        let pred: &(dyn Fn(u32) -> bool + Send + Sync) = &|id| id == 5;
        let m = FilterMode::Filter(pred);
        let p = m.post_filter().expect("Filter variant must expose a predicate");
        assert!(p(5));
        assert!(!p(4));
    }

    #[test]
    fn filter_mode_beta_filter_post_filter_returns_predicate() {
        // BetaFilter must expose the same post-filter predicate as Filter —
        // so `RerankAndFilter` applies the hard filter identically regardless of
        // which variant the strategy projected from.
        let pred: &(dyn Fn(u32) -> bool + Send + Sync) = &|id| id == 7;
        let m = FilterMode::BetaFilter {
            predicate: pred,
            beta: 0.5,
        };
        let p = m
            .post_filter()
            .expect("BetaFilter variant must expose a predicate");
        assert!(p(7));
        assert!(!p(8));
    }
}
