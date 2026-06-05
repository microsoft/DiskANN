/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Operates entirely with full precision
///
/// All indexing and search operations use the uncompressed full-precision vectors.
#[derive(Debug, Clone, Copy)]
pub struct FullPrecision;

/// Operates entirely in the quantized space
///
/// All indexing and search operations use quantized vectors.
/// If full-precision vectors are available, they are only used for the final reranking step.
#[derive(Debug, Clone, Copy)]
pub struct Quantized;
