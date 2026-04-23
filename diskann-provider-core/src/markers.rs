/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{CreateDeleteProvider, postprocess};

/// Operates entirely in full precision.
///
/// All indexing and search operations use the uncompressed full-precision vectors.
#[derive(Debug, Clone, Copy)]
pub struct FullPrecision;

/// Operates entirely in the quantized space.
///
/// All indexing and search operations use quantized vectors.
/// If full-precision vectors are available, they are only used for the final reranking step.
#[derive(Debug, Clone, Copy)]
pub struct Quantized;

/// Operates primarily in the quantized space with selective use of full precision.
///
/// # Search
/// Search is performed in the quantized space. Full-precision vectors are used only
/// to rerank the final candidate set.
///
/// # Insert and Prune
/// During insert operations, the search step uses quantized vectors.
/// Pruning then combines quantized vectors with a limited number of full-precision vectors.
///
/// The number of full-precision vectors used in pruning can be configured with
/// the `max_fp_vecs_per_prune` option when constructing a `BfTreeProvider`.
#[derive(Debug, Clone, Copy)]
pub struct Hybrid {
    /// Maximum number of full-precision vectors to use during pruning.
    /// This field is ignored during search, where full-precision vectors are never used.
    /// `None` defaults to use all full-precision vectors.
    pub max_fp_vecs_per_prune: Option<usize>,
}

impl Hybrid {
    /// Create a new `Hybrid` strategy with the specified maximum number of full-precision vectors
    /// to use during pruning.
    ///
    /// If `max_fp_vecs_per_prune` is `None`, use all full-precision vectors.
    pub fn new(max_fp_vecs_per_prune: Option<usize>) -> Self {
        Self {
            max_fp_vecs_per_prune,
        }
    }
}

/// A tag type to indicate that no deletes are allowed for this provider.
///
/// This effectively disables deletion support at compile-time.
#[derive(Debug, Clone, Copy)]
pub struct NoDeletes;

impl postprocess::DeletionCheck for NoDeletes {
    /// Always mark IDs as not deleted.
    ///
    /// We rely on constant propagation and dead-code elimination to optimize call-sites
    /// accordingly.
    #[inline(always)]
    fn deletion_check(&self, _: u32) -> bool {
        false
    }
}

impl CreateDeleteProvider for NoDeletes {
    type Target = Self;
    fn create(self, _: usize) -> Self {
        Self
    }
}

/// A tag type used to indicate that the `TableDeleteProviderAsync` should be used.
#[derive(Debug, Clone, Copy)]
pub struct TableBasedDeletes;

impl CreateDeleteProvider for TableBasedDeletes {
    type Target = crate::TableDeleteProviderAsync;
    fn create(self, total_points: usize) -> Self::Target {
        crate::TableDeleteProviderAsync::new(total_points)
    }
}

/// A tag type used to indicate that no store should be used.
///
/// Typically this would be for full precision only or quant only setups.
#[derive(Debug, Clone, Copy)]
pub struct NoStore;

/// A ZST for [`MultiInsertStrategy::seed`](diskann::graph::glue::MultiInsertStrategy::Seed)
/// indicating no use of the input batch.
///
/// Inmem providers typically don't use a working set at all, instead passing through accesses
/// directly to the underlying provider. As such, no seeding is needed.
#[derive(Debug, Clone, Copy)]
pub struct Unseeded;

/// Prefetch cache line level.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub enum PrefetchCacheLineLevel {
    /// 4 cache lines
    CacheLine4,
    /// 8 cache lines
    CacheLine8,
    /// 16 cache lines
    #[default]
    CacheLine16,
    /// prefetch all cache lines
    All,
}
