/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod grid_insert;
mod insert;
mod reassign;
mod search;
mod split;

/// Set to `true` and recompile to record and compare centroids and ordered members.
/// Regenerate grid baselines after changing this flag; list sizes are always checked.
const DUMP_IVF_STATE: bool = false;
