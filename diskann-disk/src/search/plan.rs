/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Top-level disk search plan.
//!
//! Replaces the previous `(vector_filter, is_flat_search, adaptive_l)`
//! parameter triple on `DiskIndexSearcher::search` with a sum type, making
//! invalid combinations (flat scan with adaptive L, inline filter without a
//! predicate) unrepresentable.

use diskann::graph::search::AdaptiveL;

/// Owned closure used to filter vector IDs during disk search.
///
/// The `&u32` argument is the disk-side internal/external ID (they coincide
/// on the disk path by construction).
pub type SearchPredicate<'a> = Box<dyn Fn(&u32) -> bool + Send + Sync + 'a>;

/// Top-level disk search plan.
///
/// Three variants encode the algorithm + filter combination:
///
/// * `FlatScan` — brute-force linear scan, with or without an inline filter.
/// * `Graph` — plain greedy beam search; the optional filter is applied as a
///   hard post-filter during reranking (no traversal-time effect).
/// * `InlineFilter` — label-filtered graph search; the predicate is consulted
///   at visit time (not just during rerank). `adaptive_l = Some(_)` grows the
///   beam mid-search if the observed match specificity is low.
///
/// `AdaptiveL` is unreachable except via `InlineFilter` — flat scan doesn't
/// benefit from beam widening, and plain graph search without inline tracking
/// can't compute specificity. `InlineFilter` always carries a predicate; an
/// inline filter with no filter degrades to plain Knn with extra bookkeeping
/// (verified — same top-k, slower), so the variant requires one explicitly.
pub enum SearchPlan<'a> {
    /// Brute-force linear scan. `Some(p)` applies `p` inline; `None` scans
    /// every vector (recall baseline).
    FlatScan { filter: Option<SearchPredicate<'a>> },

    /// Plain greedy beam search. The optional post-filter is applied during
    /// reranking via `RerankAndFilter`; traversal is identical to the
    /// unfiltered case.
    Graph { filter: Option<SearchPredicate<'a>> },

    /// Inline label-filtered graph search. The predicate is consulted at
    /// visit time (`QueryLabelProvider::on_visit`) and again at rerank time.
    /// `adaptive_l = Some(_)` enables mid-search beam widening based on
    /// observed match specificity.
    InlineFilter {
        predicate: SearchPredicate<'a>,
        adaptive_l: Option<AdaptiveL>,
    },
}

impl<'a> SearchPlan<'a> {
    /// Flat scan over all vectors. Recall baseline.
    pub fn flat() -> Self {
        Self::FlatScan { filter: None }
    }

    /// Flat scan restricted to vectors that satisfy `predicate`.
    pub fn flat_filtered<F>(predicate: F) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::FlatScan {
            filter: Some(Box::new(predicate)),
        }
    }

    /// Plain greedy graph search; no filter.
    pub fn graph() -> Self {
        Self::Graph { filter: None }
    }

    /// Plain greedy graph search with a hard post-filter applied during
    /// reranking. Traversal is unaffected.
    pub fn graph_filtered<F>(predicate: F) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::Graph {
            filter: Some(Box::new(predicate)),
        }
    }

    /// Inline label-filtered graph search. `adaptive_l = Some(_)` enables
    /// mid-search beam widening; `None` runs inline tracking only (no
    /// resizing).
    pub fn inline_filter<F>(predicate: F, adaptive_l: Option<AdaptiveL>) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::InlineFilter {
            predicate: Box::new(predicate),
            adaptive_l,
        }
    }

    /// Borrow the predicate carried by the plan, if any.
    ///
    /// `FlatScan { None }` and `Graph { None }` return `None` (accept-all,
    /// consumers short-circuit). `FlatScan { Some(p) }`, `Graph { Some(p) }`,
    /// and `InlineFilter { predicate: p, .. }` all return `Some(p)`.
    pub fn predicate(&self) -> Option<&(dyn Fn(&u32) -> bool + Send + Sync)> {
        match self {
            SearchPlan::FlatScan { filter: None } | SearchPlan::Graph { filter: None } => None,
            SearchPlan::FlatScan { filter: Some(p) }
            | SearchPlan::Graph { filter: Some(p) }
            | SearchPlan::InlineFilter { predicate: p, .. } => Some(p.as_ref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_no_filter_constructor() {
        let plan = SearchPlan::flat();
        assert!(matches!(plan, SearchPlan::FlatScan { filter: None }));
        assert!(plan.predicate().is_none());
    }

    #[test]
    fn flat_filtered_constructor() {
        let plan = SearchPlan::flat_filtered(|id| *id == 5);
        let p = plan.predicate().expect("FlatScan { Some } exposes predicate");
        assert!(p(&5));
        assert!(!p(&4));
    }

    #[test]
    fn graph_no_filter_constructor() {
        let plan = SearchPlan::graph();
        assert!(matches!(plan, SearchPlan::Graph { filter: None }));
        assert!(plan.predicate().is_none());
    }

    #[test]
    fn graph_filtered_constructor() {
        let plan = SearchPlan::graph_filtered(|id| *id == 7);
        let p = plan.predicate().expect("Graph { Some } exposes predicate");
        assert!(p(&7));
        assert!(!p(&6));
    }

    #[test]
    fn inline_filter_constructor_without_adaptive_l() {
        let plan = SearchPlan::inline_filter(|id| *id == 3, None);
        match &plan {
            SearchPlan::InlineFilter {
                adaptive_l: None, ..
            } => {}
            _ => panic!("expected InlineFilter with adaptive_l = None"),
        }
        let p = plan
            .predicate()
            .expect("InlineFilter always exposes a predicate");
        assert!(p(&3));
        assert!(!p(&2));
    }

    #[test]
    fn inline_filter_constructor_with_adaptive_l() {
        let adaptive = AdaptiveL::new(5, 16.0).expect("valid AdaptiveL");
        let plan = SearchPlan::inline_filter(|id| *id == 11, Some(adaptive));
        match &plan {
            SearchPlan::InlineFilter {
                adaptive_l: Some(_),
                ..
            } => {}
            _ => panic!("expected InlineFilter with adaptive_l = Some"),
        }
    }
}
