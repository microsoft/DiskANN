/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod grid;
mod grid_insert;
mod inplace_delete;

/// Set to `true` and recompile to include full adjacency list state in participating
/// baselines.
///
/// Useful for debugging regressions, but produces large baseline files.
const DUMP_GRAPH_STATE: bool = false;
