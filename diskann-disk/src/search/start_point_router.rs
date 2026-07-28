/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query-time start-point routers for disk graph search.

use crate::search::ivf_pq_router::IvfPqStartPointRouter;

/// Query-time router used to seed disk Vamana traversal.
#[derive(Debug, Clone)]
pub enum StartPointRouter {
    /// IVF router that samples probed posting lists and scores IDs with global PQ.
    IvfPq(Box<IvfPqStartPointRouter>),
}

impl StartPointRouter {
    /// Return resident router bytes, excluding allocator overhead.
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::IvfPq(router) => router.memory_bytes(),
        }
    }
}
