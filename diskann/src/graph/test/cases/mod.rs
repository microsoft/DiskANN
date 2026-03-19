/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod grid_insert;
mod grid_multihop_search;
mod grid_range_search;
mod grid_search;
mod inplace_delete;

/// Set to `true` and recompile to include full adjacency list state in participating
/// baselines.
///
/// Useful for debugging regressions, but produces large baseline files.
const DUMP_GRAPH_STATE: bool = false;
