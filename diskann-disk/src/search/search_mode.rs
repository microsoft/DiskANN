/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Top-level disk search mode.
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

/// Top-level disk search mode.
///
/// Three variants encode the algorithm + filter combination:
///
/// * `FlatScan` — brute-force linear scan, with or without an inline filter.
/// * `Graph` — plain greedy beam search; the optional filter is applied as a
///   hard post-filter during reranking (no traversal-time effect).
/// * `InlineFilter` — label-filtered graph search; the predicate is consulted
///   at visit time (not just during rerank). `adaptive_l = Some(_)` grows the
///   beam mid-search if the observed match specificity is low.

pub enum SearchMode<'a> {
    FlatScan { filter: Option<SearchPredicate<'a>> },

    Graph { filter: Option<SearchPredicate<'a>> },

    InlineFilter {
        predicate: SearchPredicate<'a>,
        adaptive_l: Option<AdaptiveL>,
    },
}

impl<'a> SearchMode<'a> {
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
        let mode = SearchMode::flat();
        assert!(matches!(mode, SearchMode::FlatScan { filter: None }));
    }

    #[test]
    fn flat_filtered_constructor() {
        let mode = SearchMode::flat_filtered(|id| *id == 5);
        match &mode {
            SearchMode::FlatScan { filter: Some(p) } => {
                assert!(p(&5));
                assert!(!p(&4));
            }
            _ => panic!("expected FlatScan with filter"),
        }
    }

    #[test]
    fn graph_no_filter_constructor() {
        let mode = SearchMode::graph();
        assert!(matches!(mode, SearchMode::Graph { filter: None }));
    }

    #[test]
    fn graph_filtered_constructor() {
        let mode = SearchMode::graph_filtered(|id| *id == 7);
        match &mode {
            SearchMode::Graph { filter: Some(p) } => {
                assert!(p(&7));
                assert!(!p(&6));
            }
            _ => panic!("expected Graph with filter"),
        }
    }

    #[test]
    fn inline_filter_constructor_without_adaptive_l() {
        let mode = SearchMode::inline_filter(|id| *id == 3, None);
        match &mode {
            SearchMode::InlineFilter {
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
        let mode = SearchMode::inline_filter(|id| *id == 11, Some(adaptive));
        match &mode {
            SearchMode::InlineFilter {
                adaptive_l: Some(_),
                ..
            } => {}
            _ => panic!("expected InlineFilter with adaptive_l = Some"),
        }
    }
}
