/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Baseline regression tests for single and batch insert followed by search.
//!
//! This module consolidates both single-insert and multi-insert tests into a single
//! parameterized framework. The test matrix covers:
//!
//! - **Grid configurations**: 1D/100, 3D/5, 4D/4
//! - **Insert modes**: single (one-by-one) and batch (via `multi_insert`)
//! - **Batch sizes**: all-at-once and chunked
//! - **Intra-batch candidates**: None, Some(4), All
//!
//! For multi-insert tests, thread-count invariance is verified: the same baseline must
//! hold regardless of whether the tokio runtime is single-threaded or multi-threaded.

use std::{num::NonZeroUsize, sync::Arc};

use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, DiskANNIndex,
        config::IntraBatchCandidates,
        search::Knn,
        test::{provider as test_provider, synthetic::Grid},
    },
    neighbor::Neighbor,
    test::{
        TestPath, TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
        tokio::current_thread_runtime,
    },
    utils::{IntoUsize, async_tools::VectorIdBoxSlice},
};

use super::DUMP_GRAPH_STATE;

fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/grid_insert")
}

/////////////
// Helpers //
/////////////

/// Create a provider with only a start point — no pre-existing graph data.
///
/// The start point is placed at `grid.start_point(size)` (all coordinates = `size`),
/// matching the convention used by [`Grid::setup`]. The start point ID is `u32::MAX`.
fn empty_provider(grid: Grid, size: usize) -> test_provider::Provider {
    let max_degree: usize = (grid.dim() as usize) * 2;
    let start_vector = grid.start_point(size);

    let config = test_provider::Config::new(
        Metric::L2,
        max_degree,
        test_provider::StartPoint::new(u32::MAX, start_vector),
    )
    .unwrap();

    test_provider::Provider::new(config)
}

/// Build a [`DiskANNIndex`] around the given provider.
fn build_index(
    provider: test_provider::Provider,
    intra_batch_candidates: IntraBatchCandidates,
    max_minibatch_par: usize,
) -> Arc<DiskANNIndex<test_provider::Provider>> {
    let provider_degree = provider.max_degree();

    let target_degree = match provider_degree.checked_sub(2) {
        Some(degree) => degree.max(2).min(provider_degree),
        None => provider_degree,
    };

    let index_config = graph::config::Builder::new_with(
        target_degree,
        graph::config::MaxDegree::new(provider_degree),
        100,
        (Metric::L2).into(),
        |b| {
            b.intra_batch_candidates(intra_batch_candidates);
            b.max_minibatch_par(max_minibatch_par);
        },
    )
    .build()
    .unwrap();

    Arc::new(DiskANNIndex::new(index_config, provider, None))
}

/// Run a round of index building on `data`.
///
/// If `batchsize` is `None`, points are inserted one-by-one via [`DiskANNIndex::insert`].
/// Otherwise, data is partitioned into chunks of `batchsize` and fed to
/// [`DiskANNIndex::multi_insert`].
fn run_build(
    index: &Arc<DiskANNIndex<test_provider::Provider>>,
    data: MatrixView<'_, f32>,
    batchsize: Option<NonZeroUsize>,
    runtime: &tokio::runtime::Runtime,
) -> test_provider::Context {
    let strategy = test_provider::Strategy::new();
    let context = test_provider::Context::new();

    match batchsize {
        None => {
            for (id, vector) in data.row_iter().enumerate() {
                runtime
                    .block_on(index.insert(strategy, &context, &(id as u32), vector))
                    .unwrap();
            }
        }
        Some(batchsize) => {
            let mut start = 0;
            while start < data.nrows() {
                let stop = (start + batchsize.get()).min(data.nrows());
                let vectors: Box<[VectorIdBoxSlice<u32, f32>]> = (start..stop)
                    .map(|i| VectorIdBoxSlice::new(i as u32, data.row(i).into()))
                    .collect();
                runtime
                    .block_on(
                        index.multi_insert::<test_provider::Strategy, _>(
                            strategy, &context, vectors,
                        ),
                    )
                    .unwrap();

                start = stop;
            }
        }
    }

    context
}

/// Capture the graph state as sorted adjacency lists, if [`DUMP_GRAPH_STATE`] is enabled.
fn maybe_dump_graph(index: &DiskANNIndex<test_provider::Provider>) -> Option<Vec<(u32, Vec<u32>)>> {
    if !DUMP_GRAPH_STATE {
        return None;
    }

    let mut neighbors: Vec<(u32, Vec<u32>)> = index
        .provider()
        .dump_neighbors()
        .into_iter()
        .map(|(id, adj)| {
            let mut n: Vec<u32> = adj.into();
            n.sort();
            (id, n)
        })
        .collect();

    neighbors.sort_by_key(|(id, _)| *id);
    Some(neighbors)
}

/// Search the index with standard query vectors and return results.
fn run_searches(
    index: &DiskANNIndex<test_provider::Provider>,
    grid: Grid,
    size: usize,
    description_prefix: &str,
    runtime: &tokio::runtime::Runtime,
) -> Vec<InsertSearchResult> {
    let desc_0 = format!(
        "{} Search with query of all -1s. \
         The nearest neighbor should be coordinate 0 (all zeros).",
        description_prefix,
    );

    let desc_1 = format!(
        "{} Search with query of all `size`. \
         The start point should appear as it is not filtered by default.",
        description_prefix,
    );

    let queries = [
        (vec![-1.0f32; grid.dim().into()], desc_0),
        (vec![size as f32; grid.dim().into()], desc_1),
    ];

    let mut results = Vec::new();
    for (query, desc) in queries {
        let params = Knn::new(10, 10, None).unwrap();
        let beam_width = params.beam_width().get();
        let search_ctx = test_provider::Context::new();

        let mut neighbors = vec![Neighbor::<u32>::default(); params.k_value().get()];
        let graph::index::SearchStats {
            cmps,
            hops,
            result_count,
            range_search_second_round,
        } = runtime
            .block_on(index.search(
                params,
                &test_provider::Strategy::new(),
                &search_ctx,
                query.as_slice(),
                &mut crate::neighbor::BackInserter::new(neighbors.as_mut_slice()),
            ))
            .unwrap();

        assert!(
            !range_search_second_round,
            "range search should not activate for k-nearest-neighbors",
        );

        let metrics = index.provider().metrics();

        results.push(InsertSearchResult {
            query: query.clone(),
            description: desc,
            results: neighbors.into_iter().map(|i| i.as_tuple()).collect(),
            comparisons: cmps.into_usize(),
            hops: hops.into_usize(),
            num_results: result_count.into_usize(),
            grid_dims: grid.dim().into(),
            grid_size: size,
            beam_width,
            metrics,
        });
    }

    results
}

/////////////
// Results //
/////////////

/// A single post-insert search result for baseline comparison.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct InsertSearchResult {
    description: String,

    /// The query vector given to search.
    query: Vec<f32>,

    /// The k-NN results from search (id, distance).
    results: Vec<(u32, f32)>,

    /// The number of comparisons recorded during search.
    comparisons: usize,

    /// The number of hops recorded during search.
    hops: usize,

    /// The number of results returned from search.
    num_results: usize,

    /// Grid dimensionality.
    grid_dims: usize,

    /// Grid edge size.
    grid_size: usize,

    /// Beam width used for search.
    beam_width: usize,

    /// Provider-level metrics at the time of the search (cumulative from insert + search).
    metrics: test_provider::Metrics,
}

verbose_eq!(InsertSearchResult {
    description,
    query,
    results,
    comparisons,
    hops,
    num_results,
    grid_dims,
    grid_size,
    beam_width,
    metrics,
});

/// Full baseline for a grid insert test: insert-phase metrics + search results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GridInsertBaseline {
    /// Description of the test scenario.
    description: String,

    /// Grid dimensionality.
    grid_dims: usize,

    /// Grid edge size.
    grid_size: usize,

    /// Total number of points inserted (excludes start point).
    num_inserted: usize,

    /// Provider-level metrics captured immediately after all inserts complete.
    insert_metrics: test_provider::Metrics,

    /// Context-level metrics across all searches.
    context_metrics: test_provider::ContextMetrics,

    /// Search results collected after all inserts.
    searches: Vec<InsertSearchResult>,

    /// Optional: sorted adjacency lists for every point. Only populated when
    /// `DUMP_GRAPH_STATE` is `true`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    graph_state: Option<Vec<(u32, Vec<u32>)>>,
}

verbose_eq!(GridInsertBaseline {
    description,
    grid_dims,
    grid_size,
    num_inserted,
    insert_metrics,
    context_metrics,
    searches,
    graph_state,
});

////////////////
// Test logic //
////////////////

/// Produce a human-readable description of the [`IntraBatchCandidates`] setting.
fn ibc_label(ibc: IntraBatchCandidates) -> String {
    match ibc {
        IntraBatchCandidates::None => "ibc=none".to_string(),
        IntraBatchCandidates::Max(n) => format!("ibc=max({})", n),
        IntraBatchCandidates::All => "ibc=all".to_string(),
    }
}

/// Produce a deterministic baseline file name encoding all test parameters.
fn baseline_name(
    grid: Grid,
    size: usize,
    batchsize: Option<NonZeroUsize>,
    ibc: IntraBatchCandidates,
) -> String {
    let batch_tag = match batchsize {
        None => "single".to_string(),
        Some(bs) => format!("batch_{}", bs),
    };
    let ibc_tag = match ibc {
        IntraBatchCandidates::None => "ibc_none".to_string(),
        IntraBatchCandidates::Max(n) => format!("ibc_max_{}", n),
        IntraBatchCandidates::All => "ibc_all".to_string(),
    };
    format!("insert_{}_{}_{}/{}", grid.dim(), size, batch_tag, ibc_tag)
}

/// Parameters for a single test run.
struct TestParams {
    grid: Grid,
    size: usize,
    /// `None` = single insert, `Some(n)` = multi_insert with batch size `n`.
    batchsize: Option<NonZeroUsize>,
    intra_batch_candidates: IntraBatchCandidates,
    /// Maximum parallelism for multi_insert. Use 1 for single insert, > 1 for batch.
    max_minibatch_par: usize,
}

/// Core test function: build an index, insert data, search, and compare against baseline.
fn _grid_build_and_search(params: TestParams, mut parent: TestPath<'_>) {
    let rt = current_thread_runtime();

    let TestParams {
        grid,
        size,
        batchsize,
        intra_batch_candidates,
        max_minibatch_par,
    } = params;

    let num_points = grid.num_points(size);
    let grid_data = grid.data(size);
    let index = build_index(
        empty_provider(grid, size),
        intra_batch_candidates,
        max_minibatch_par,
    );

    // Build the index.
    let insert_context = run_build(&index, grid_data.as_view(), batchsize, &rt);

    let insert_metrics = index.provider().metrics();
    index.provider().is_consistent().unwrap();

    let graph_state = maybe_dump_graph(&index);

    // Describe the insert mode for the baseline.
    let mode_desc = match batchsize {
        None => "one-by-one".to_string(),
        Some(bs) if bs.get() >= num_points => "batch (all-at-once)".to_string(),
        Some(bs) => format!("batch (chunks of {})", bs),
    };

    let description_prefix = format!(
        "After inserting {} points ({}, {}) into a {}D grid of size {}.",
        num_points,
        mode_desc,
        ibc_label(intra_batch_candidates),
        grid.dim(),
        size,
    );

    let searches = run_searches(&index, grid, size, &description_prefix, &rt);

    let baseline = GridInsertBaseline {
        description: description_prefix,
        grid_dims: grid.dim().into(),
        grid_size: size,
        num_inserted: num_points,
        insert_metrics,
        context_metrics: insert_context.metrics(),
        searches,
        graph_state,
    };

    let name = parent.push(baseline_name(grid, size, batchsize, intra_batch_candidates));
    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);
}

/// Verify that multi-insert produces the same results regardless of runtime thread count.
///
/// Runs the build on both a single-threaded and a multi-threaded tokio runtime and asserts
/// that the resulting baselines are identical.
fn _assert_thread_invariant(
    grid: Grid,
    size: usize,
    batchsize: NonZeroUsize,
    intra_batch_candidates: IntraBatchCandidates,
    max_minibatch_par: usize,
) {
    // Build with single-threaded runtime.
    let rt_st = current_thread_runtime();
    let grid_data = grid.data(size);

    let index_st = build_index(
        empty_provider(grid, size),
        intra_batch_candidates,
        max_minibatch_par,
    );
    run_build(&index_st, grid_data.as_view(), Some(batchsize), &rt_st);
    let metrics_st = index_st.provider().metrics();

    // Build with multi-threaded runtime (2 worker threads).
    let rt_mt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("multi-thread runtime should build");

    let index_mt = build_index(
        empty_provider(grid, size),
        intra_batch_candidates,
        max_minibatch_par,
    );
    run_build(&index_mt, grid_data.as_view(), Some(batchsize), &rt_mt);
    let metrics_mt = index_mt.provider().metrics();

    // Metrics must match exactly.
    assert_eq_verbose!(metrics_st, metrics_mt);

    // Search results must match exactly.
    let prefix = "Thread invariance check.";
    let searches_st = run_searches(&index_st, grid, size, prefix, &rt_st);
    let searches_mt = run_searches(&index_mt, grid, size, prefix, &rt_mt);
    assert_eq_verbose!(searches_st, searches_mt);
}

///////////////////
// Single-insert //
///////////////////

#[test]
fn single_1d_100() {
    _grid_build_and_search(
        TestParams {
            grid: Grid::One,
            size: 100,
            batchsize: None,
            intra_batch_candidates: IntraBatchCandidates::None,
            max_minibatch_par: 1,
        },
        root().path(),
    );
}

#[test]
fn single_3d_5() {
    _grid_build_and_search(
        TestParams {
            grid: Grid::Three,
            size: 5,
            batchsize: None,
            intra_batch_candidates: IntraBatchCandidates::None,
            max_minibatch_par: 1,
        },
        root().path(),
    );
}

#[test]
fn single_4d_4() {
    _grid_build_and_search(
        TestParams {
            grid: Grid::Four,
            size: 4,
            batchsize: None,
            intra_batch_candidates: IntraBatchCandidates::None,
            max_minibatch_par: 1,
        },
        root().path(),
    );
}

/////////////////////////////////////////
// Multi-insert: all-at-once, ibc=none //
/////////////////////////////////////////

fn all_at_once(grid: Grid, size: usize) -> NonZeroUsize {
    NonZeroUsize::new(grid.num_points(size)).unwrap()
}

#[test]
fn batch_all_ibc_none_1d_100() {
    let (grid, size) = (Grid::One, 100);
    _grid_build_and_search(
        TestParams {
            grid,
            size,
            batchsize: Some(all_at_once(grid, size)),
            intra_batch_candidates: IntraBatchCandidates::None,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

#[test]
fn batch_all_ibc_none_3d_5() {
    let (grid, size) = (Grid::Three, 5);
    _grid_build_and_search(
        TestParams {
            grid,
            size,
            batchsize: Some(all_at_once(grid, size)),
            intra_batch_candidates: IntraBatchCandidates::None,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

#[test]
fn batch_all_ibc_none_4d_4() {
    let (grid, size) = (Grid::Four, 4);
    _grid_build_and_search(
        TestParams {
            grid,
            size,
            batchsize: Some(all_at_once(grid, size)),
            intra_batch_candidates: IntraBatchCandidates::None,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

///////////////////////////////////////////
// Multi-insert: all-at-once, ibc=max(4) //
///////////////////////////////////////////

#[test]
fn batch_all_ibc_4_3d_5() {
    let (grid, size) = (Grid::Three, 5);
    _grid_build_and_search(
        TestParams {
            grid,
            size,
            batchsize: Some(all_at_once(grid, size)),
            intra_batch_candidates: IntraBatchCandidates::new(4),
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

#[test]
fn batch_all_ibc_4_4d_4() {
    let (grid, size) = (Grid::Four, 4);
    _grid_build_and_search(
        TestParams {
            grid,
            size,
            batchsize: Some(all_at_once(grid, size)),
            intra_batch_candidates: IntraBatchCandidates::new(4),
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

////////////////////////////////////////
// Multi-insert: all-at-once, ibc=all //
////////////////////////////////////////

#[test]
fn batch_all_ibc_all_1d_100() {
    let (grid, size) = (Grid::One, 100);
    _grid_build_and_search(
        TestParams {
            grid,
            size,
            batchsize: Some(all_at_once(grid, size)),
            intra_batch_candidates: IntraBatchCandidates::All,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

#[test]
fn batch_all_ibc_all_3d_5() {
    let (grid, size) = (Grid::Three, 5);
    _grid_build_and_search(
        TestParams {
            grid,
            size,
            batchsize: Some(all_at_once(grid, size)),
            intra_batch_candidates: IntraBatchCandidates::All,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

#[test]
fn batch_all_ibc_all_4d_4() {
    let (grid, size) = (Grid::Four, 4);
    _grid_build_and_search(
        TestParams {
            grid,
            size,
            batchsize: Some(all_at_once(grid, size)),
            intra_batch_candidates: IntraBatchCandidates::All,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

/////////////////////////////////////////
// Multi-insert: chunked(25), ibc=none //
/////////////////////////////////////////

#[test]
fn batch_25_ibc_none_3d_5() {
    _grid_build_and_search(
        TestParams {
            grid: Grid::Three,
            size: 5,
            batchsize: NonZeroUsize::new(25),
            intra_batch_candidates: IntraBatchCandidates::None,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

#[test]
fn batch_25_ibc_none_4d_4() {
    _grid_build_and_search(
        TestParams {
            grid: Grid::Four,
            size: 4,
            batchsize: NonZeroUsize::new(25),
            intra_batch_candidates: IntraBatchCandidates::None,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

////////////////////////////////////////
// Multi-insert: chunked(25), ibc=all //
////////////////////////////////////////

#[test]
fn batch_25_ibc_all_3d_5() {
    _grid_build_and_search(
        TestParams {
            grid: Grid::Three,
            size: 5,
            batchsize: NonZeroUsize::new(25),
            intra_batch_candidates: IntraBatchCandidates::All,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

#[test]
fn batch_25_ibc_all_4d_4() {
    _grid_build_and_search(
        TestParams {
            grid: Grid::Four,
            size: 4,
            batchsize: NonZeroUsize::new(25),
            intra_batch_candidates: IntraBatchCandidates::All,
            max_minibatch_par: 2,
        },
        root().path(),
    );
}

/////////////////////////////
// Thread-count invariance //
/////////////////////////////

#[test]
fn thread_invariant_batch_all_ibc_all_3d_5() {
    _assert_thread_invariant(
        Grid::Three,
        5,
        all_at_once(Grid::Three, 5),
        IntraBatchCandidates::All,
        2,
    );
}

#[test]
fn thread_invariant_batch_25_ibc_none_4d_4() {
    _assert_thread_invariant(
        Grid::Four,
        4,
        NonZeroUsize::new(25).unwrap(),
        IntraBatchCandidates::None,
        2,
    );
}
