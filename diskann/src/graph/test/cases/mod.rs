/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod consolidate;
mod grid_insert;
mod grid_search;
mod helpers;
mod index;
mod inplace_delete;
mod paged_search;
mod range_search;

/// Set to `true` and recompile to include full adjacency list state in participating
/// baselines.
///
/// Useful for debugging regressions, but produces large baseline files.
const DUMP_GRAPH_STATE: bool = false;
