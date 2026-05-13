/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub mod async_;

// Layers for the async index.
pub mod layers;

/// Determinant-diversity post-processing algorithm.
///
/// This module is not async-specific and is re-exported here for clarity.
/// It provides diversity-promoting reranking for nearest neighbor search results.
pub mod determinant_diversity;
pub use determinant_diversity::determinant_diversity_post_process;
