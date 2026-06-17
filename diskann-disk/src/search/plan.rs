/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Top-level disk search plan.
//!
//! Encodes the algorithm + filter combination for `DiskIndexSearcher::search`
//! as a sum type so that invalid combinations (flat scan with adaptive L,
//! inline filter without a predicate) are unrepresentable at the API boundary.

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

pub enum SearchPlan<'a> {
    FlatScan { filter: Option<SearchPredicate<'a>> },

    Graph { filter: Option<SearchPredicate<'a>> },

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_no_filter_constructor() {
        let plan = SearchPlan::flat();
        assert!(matches!(plan, SearchPlan::FlatScan { filter: None }));
    }

    #[test]
    fn flat_filtered_constructor() {
        let plan = SearchPlan::flat_filtered(|id| *id == 5);
        match &plan {
            SearchPlan::FlatScan { filter: Some(p) } => {
                assert!(p(&5));
                assert!(!p(&4));
            }
            _ => panic!("expected FlatScan with filter"),
        }
    }

    #[test]
    fn graph_no_filter_constructor() {
        let plan = SearchPlan::graph();
        assert!(matches!(plan, SearchPlan::Graph { filter: None }));
    }

    #[test]
    fn graph_filtered_constructor() {
        let plan = SearchPlan::graph_filtered(|id| *id == 7);
        match &plan {
            SearchPlan::Graph { filter: Some(p) } => {
                assert!(p(&7));
                assert!(!p(&6));
            }
            _ => panic!("expected Graph with filter"),
        }
    }

    #[test]
    fn inline_filter_constructor_without_adaptive_l() {
        let plan = SearchPlan::inline_filter(|id| *id == 3, None);
        match &plan {
            SearchPlan::InlineFilter {
                predicate,
                adaptive_l: None,
            } => {
                assert!(predicate(&3));
                assert!(!predicate(&2));
            }
            _ => panic!("expected InlineFilter with adaptive_l = None"),
        }
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
