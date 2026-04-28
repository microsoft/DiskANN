/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Statistics returned by a flat search.

/// Statistics collected during a single flat search invocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatSearchStats {
    /// Number of distance computations performed (i.e., elements visited by the scanner).
    pub cmps: u32,

    /// Number of results written into the caller-provided output buffer.
    pub result_count: u32,
}
