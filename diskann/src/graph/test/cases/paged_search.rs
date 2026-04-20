/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for paged (iterative) search.
//!
//! Paged search returns results in pages of k neighbors via a stateful
//! `SearchState`. Tests cover basic pagination, single-page retrieval,
//! and small page sizes that stress the iteration machinery.

use std::sync::Arc;

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, DiskANNIndex,
        test::{provider as test_provider, synthetic::Grid},
    },
    neighbor::Neighbor,
    test::{
        TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
        tokio::current_thread_runtime,
    },
};

fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/pagedSearch")
}

fn setup_grid_index(grid_size: usize, dims: Grid) -> Arc<DiskANNIndex<test_provider::Provider>> {
    let provider = test_provider::Provider::grid(dims, grid_size).unwrap();
    let index_config = graph::config::Builder::new(
        provider.max_degree(),
        graph::config::MaxDegree::same(),
        100,
        Metric::L2.into(),
    )
    .build()
    .unwrap();
    Arc::new(DiskANNIndex::new(index_config, provider, None))
}

fn setup_grid_index_and_basic_query(
    grid_size: usize,
    dims: Grid,
) -> (Arc<DiskANNIndex<test_provider::Provider>>, Vec<f32>) {
    let index = setup_grid_index(grid_size, dims);
    let query = vec![grid_size as f32; dims.dim().into()];
    (index, query)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PagedSearchBaseline {
    grid_size: usize,
    dims: usize,
    query: Vec<f32>,
    search_l: usize,
    page_size: usize,
    pages: Vec<Vec<(u32, f32)>>,
    total_results: usize,
}

verbose_eq!(PagedSearchBaseline {
    grid_size,
    dims,
    query,
    search_l,
    page_size,
    pages,
    total_results
});

/// assert no duplicate IDs across pages
fn assert_no_duplicates_across_pages(pages: &[Vec<Neighbor<u32>>]) {
    let mut seen = std::collections::HashSet::new();
    for (page_idx, page) in pages.iter().enumerate() {
        for n in page {
            assert!(
                seen.insert(n.id),
                "duplicate id {} found on page {}",
                n.id,
                page_idx
            );
        }
    }
}

/// assert distances are non-decreasing across full result sequence
fn assert_non_decreasing_distances(pages: &[Vec<Neighbor<u32>>]) {
    let all_results: Vec<&Neighbor<u32>> = pages.iter().flat_map(|p| p.iter()).collect();
    for window in all_results.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "distances not non-decreasing: id {} dist {} followed by id {} dist {}",
            window[0].id,
            window[0].distance,
            window[1].id,
            window[1].distance,
        );
    }
}

/// assert each page respects the max page size
fn assert_page_sizes(pages: &[Vec<Neighbor<u32>>], max_page_size: usize) {
    for (i, page) in pages.iter().enumerate() {
        assert!(
            page.len() <= max_page_size,
            "page {} has {} results, exceeding max {}",
            i,
            page.len(),
            max_page_size
        );
    }
}

fn build_baseline(
    grid_size: usize,
    dims: &Grid,
    query: &[f32],
    search_l: usize,
    page_size: usize,
    pages: &[Vec<Neighbor<u32>>],
) -> PagedSearchBaseline {
    PagedSearchBaseline {
        grid_size,
        dims: dims.dim() as usize,
        query: query.to_vec(),
        search_l,
        page_size,
        pages: pages
            .iter()
            .map(|p| p.iter().map(|n| (n.id, n.distance)).collect())
            .collect(),
        total_results: pages.iter().map(|p| p.len()).sum(),
    }
}

#[test]
fn basic_paged_search() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let test_name = path.push("basic_paged_search");

    let grid_size = 5;
    let dims = Grid::Three;
    let (index, query) = setup_grid_index_and_basic_query(grid_size, dims);
    let search_l = 32;
    let page_size = 4;

    let mut state = rt
        .block_on(index.start_paged_search(
            test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            search_l,
        ))
        .unwrap();

    let mut pages: Vec<Vec<Neighbor<u32>>> = Vec::new();
    let mut buffer = vec![Neighbor::<u32>::default(); page_size];

    loop {
        let count = rt
            .block_on(
                index.next_search_results::<test_provider::Strategy, &[f32]>(
                    &test_provider::Context::new(),
                    &mut state,
                    page_size,
                    &mut buffer,
                ),
            )
            .unwrap();

        if count == 0 {
            break;
        }
        pages.push(buffer[..count].to_vec());
    }

    let baseline = build_baseline(grid_size, &dims, &query, search_l, page_size, &pages);

    let expected = get_or_save_test_results(&test_name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert_no_duplicates_across_pages(&pages);
    assert_non_decreasing_distances(&pages);
    assert_page_sizes(&pages, page_size);
}

#[test]
fn single_page() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let test_name = path.push("single_page");

    let grid_size = 5;
    let dims = Grid::Three;
    let (index, query) = setup_grid_index_and_basic_query(grid_size, dims);
    let search_l = 200;
    let page_size = 200; // larger than total points (125)

    let mut state = rt
        .block_on(index.start_paged_search(
            test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            search_l,
        ))
        .unwrap();

    let mut buffer = vec![Neighbor::<u32>::default(); page_size];

    let count = rt
        .block_on(
            index.next_search_results::<test_provider::Strategy, &[f32]>(
                &test_provider::Context::new(),
                &mut state,
                page_size,
                &mut buffer,
            ),
        )
        .unwrap();

    let results: Vec<Neighbor<u32>> = buffer[..count].to_vec();
    let pages = vec![results.clone()];

    let baseline = build_baseline(grid_size, &dims, &query, search_l, page_size, &pages);

    let expected = get_or_save_test_results(&test_name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert_no_duplicates_across_pages(&pages);
    assert_non_decreasing_distances(&pages);

    // Verify second call returns 0 (nothing left)
    let count2 = rt
        .block_on(
            index.next_search_results::<test_provider::Strategy, &[f32]>(
                &test_provider::Context::new(),
                &mut state,
                page_size,
                &mut buffer,
            ),
        )
        .unwrap();
    assert_eq!(count2, 0, "second page should be empty");
}

#[test]
fn small_page_size() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let test_name = path.push("small_page_size");

    let grid_size = 5;
    let dims = Grid::Three;
    let (index, query) = setup_grid_index_and_basic_query(grid_size, dims);
    let search_l = 32;
    let page_size = 1; // one result per page, maximum iterations

    let mut state = rt
        .block_on(index.start_paged_search(
            test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            search_l,
        ))
        .unwrap();

    let mut pages: Vec<Vec<Neighbor<u32>>> = Vec::new();
    let mut buffer = vec![Neighbor::<u32>::default(); page_size];

    loop {
        let count = rt
            .block_on(
                index.next_search_results::<test_provider::Strategy, &[f32]>(
                    &test_provider::Context::new(),
                    &mut state,
                    page_size,
                    &mut buffer,
                ),
            )
            .unwrap();

        if count == 0 {
            break;
        }
        pages.push(buffer[..count].to_vec());
    }

    let baseline = build_baseline(grid_size, &dims, &query, search_l, page_size, &pages);

    let expected = get_or_save_test_results(&test_name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert_no_duplicates_across_pages(&pages);
    assert_non_decreasing_distances(&pages);
    assert_page_sizes(&pages, page_size);

    // Every page should have exactly 1 result
    for (i, page) in pages.iter().enumerate() {
        assert_eq!(page.len(), 1, "page {} should have exactly 1 result", i);
    }
}
