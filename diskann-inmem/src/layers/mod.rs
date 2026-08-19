/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Distance layers indexing.
//!
//! An important assumption made by this module is that the data within each layer is
//! uniformly sized: each entry occupies the same number of bytes. Furthermore, the data
//! to be stored may not assume any particular alignment. Implementations will strive to
//! achieve a reasonable alignment, but this may not be relied on.
//!
//! # Query Distance Specialization
//!
//! The design of this module allows aggressive optimization of graph search kernels via
//! the [`Search`] and [`QueryVisitor`] pairs of traits.
//!
//! Implementations of [`Search`] can pass a [`QueryDistance`] kernel specialized to
//! a specific geometry (dimensionality or metric type) which upstream [`QueryVisitor`]
//! will fuse into larger kernels. While this allows for high performance graph kernels,
//! some considerations should be taken into account:
//!
//! 1. For correctness purposes, upstream callers cannot do any kind of caching. As such,
//!    the dispatch layer used to select the kernel passed to the [`QueryVisitor`] should
//!    be relatively efficient.
//!
//! 2. Keep the number of specializations bounded for compile time reasons.

use std::num::NonZeroU16;

use diskann::ANNResult;
use thiserror::Error;

use crate::{
    counters::LocalCounters,
    num::{Capacity, IdLimit, MaxDegree},
};

pub mod full;
pub use full::{Full, FullPrecision};

pub trait LayerConfig {
    type Layer: Layer;

    fn build(self) -> ANNResult<Self::Layer>;
}

/// Base layer for data representations.
pub trait Layer: Send + Sync + 'static {
    fn max_degree(&self) -> MaxDegree;

    fn id_limit(&self) -> IdLimit;

    fn capacity(&self) -> Capacity;

    fn retire(&self, i: u32) -> ANNResult<()>;

    fn is_readable(&self, i: u32) -> Option<bool>;
}

pub trait Set<T>: Layer {
    type Guard<'a>: Guard;
    fn set(&self, element: T) -> ANNResult<Self::Guard<'_>>;
}

pub trait Guard {
    fn id(&self) -> u32;
    fn publish(self);
}

/// Trait object based implementation of [`diskann::graph::glue::SearchAccessor::expand_beam`].
///
/// Dynamic dispatch is used to enable aggressive specialization of this primitive without
/// monomorphizing the entire search algorithm. Examples specializations include:
///
/// * Optimizing for certain fixed dimensions.
/// * Inlining metric specific distance functions.
/// * Tailoring prefetching to the dimension.
///
/// # Safety
///
/// This trait is `unsafe` because [`Self::id_limit`] **must** work for [`Self::expand_beam`]'s
/// safety pre-conditions.
pub(crate) unsafe trait ExpandBeam: Send + Sync + std::fmt::Debug {
    /// Evaluate a raw distance against index `i`.
    fn evaluate(&self, i: u32) -> ANNResult<Option<f32>>;

    /// Return an [`IdLimit`] for this primitive.
    ///
    /// Callers must be able to use this limit to satisfy the safety pre-conditions for
    /// [`Self::expand_beam`].
    ///
    /// See also: [`IdLimit::is_in_bound`].
    fn id_limit(&self) -> IdLimit;

    /// Compute the distance between the query and each neighbor in `list`.
    ///
    /// # Safety
    ///
    /// * All items in `list` must in-bounds with respect to [`Self::id_limit`].
    /// * `buffer.len() >= list.len()`.
    unsafe fn expand_beam(&self, list: &[u32], buffer: &mut [(u32, f32)]) -> ANNResult<usize>;
}

#[derive(Debug, Clone, Copy)]
pub struct PruneKey(NonZeroU16);

impl PruneKey {
    const ONE: Self = Self(NonZeroU16::new(1).unwrap());

    pub(crate) fn counter() -> Self {
        Self::ONE
    }

    pub(crate) fn inc(self) -> Result<Self, Overflow> {
        match self.0.checked_add(1) {
            Some(v) => Ok(Self(v)),
            None => Err(Overflow),
        }
    }

    pub(crate) fn as_u64(self) -> u64 {
        u64::from(self.0.get()) - 1
    }

    pub(crate) fn index(self) -> usize {
        usize::from(self.0.get()) - 1
    }
}

impl<'a> diskann_utils::Reborrow<'a> for PruneKey {
    type Target = PruneKey;
    fn reborrow(&'a self) -> Self::Target {
        *self
    }
}

#[derive(Debug, Error)]
#[error("prune list exceeded u16::MAX")]
pub(crate) struct Overflow;

diskann::convert_error!(Overflow);

/// Enable search over vectors defined by a [`Layer`].
pub trait Search: Send + Sync + 'static {
    /// The type of the query. This should be equivalent to the generic parameter in
    /// [`Set`], but needs to be replicated here due to limitations in the current trait
    /// design.
    type Query<'a>;

    #[doc(hidden)]
    fn search_accessor<'a>(
        &'a self,
        query: Self::Query<'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>>;
}

// TODO: Try to hide?
#[doc(hidden)]
pub(crate) trait Prune: Send + Sync + std::fmt::Debug {
    fn prepare(
        &mut self,
        items: hashbrown::hash_map::IterMut<'_, u32, Option<PruneKey>>,
    ) -> ANNResult<PruneKey>;

    fn evaluate(&self, a: PruneKey, b: PruneKey) -> f32;
}

/// A insert-specific specialization of [`Search`].
///
/// Note that the bounds for this trait are unnecessarily complicated, but rely on changes
/// to `diskann` to full resolve.
pub trait Insert: Search + for<'a> Set<Self::Query<'a>> {
    #[doc(hidden)]
    fn insert_search_accessor<'a>(
        &'a self,
        query: Self::Query<'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        self.search_accessor(query, provider, counters)
    }

    #[doc(hidden)]
    fn prune_accessor<'a>(
        &'a self,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>>;
}
