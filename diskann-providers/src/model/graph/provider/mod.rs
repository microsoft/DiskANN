/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub mod async_;

// Layers for the async index.
pub mod layers;

/// Determinant-diversity post-processing algorithm.
///
/// This module is not async-specific.
/// It provides diversity-promoting reranking for nearest neighbor search results.
mod determinant_diversity;
pub use determinant_diversity::{
    DeterminantDiversityError, DeterminantDiversityParams, determinant_diversity,
};
