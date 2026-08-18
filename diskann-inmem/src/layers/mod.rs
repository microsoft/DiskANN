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

use crate::{Hidden, counters::LocalCounters, num::Bytes, store::Store};

mod full;
pub use full::{Full, FullPrecision};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Available,
    Published,
    Retiring,
    Frozen,
}

/// Base layer for data representations.
pub trait Layer: Send + Sync + 'static {
    fn bytes(&self) -> Bytes;
}

pub trait Set<T>: Layer {
    fn set(&self, element: T, bytes: &mut [u8]) -> ANNResult<()>;
}

// TODO: Try to hide?
pub(crate) trait __ExpandBeam: Send + Sync + std::fmt::Debug {
    /// Evaluate a raw distance against index `i`.
    fn __evaluate(&self, i: u32, _: Hidden) -> ANNResult<Option<f32>>;

    /// Compute the distance between the query and each neighbor in `list`.
    ///
    /// # Safety
    ///
    /// * All items in `list` must in-bounds with respect to `reader`.
    /// * `buffer.len() >= list.len()`.
    unsafe fn __expand_beam(
        &self,
        list: &[u32],
        buffer: &mut [(u32, f32)],
        _: Hidden,
    ) -> ANNResult<usize>;
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
struct Overflow;

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
        store: &'a Store,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>>;
}

// TODO: Try to hide?
#[doc(hidden)]
pub(crate) trait __Prune: Send + Sync + std::fmt::Debug {
    fn __prepare(
        &mut self,
        items: hashbrown::hash_map::IterMut<'_, u32, Option<PruneKey>>,
    ) -> ANNResult<PruneKey>;

    fn __evaluate(&self, a: PruneKey, b: PruneKey) -> f32;
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
        store: &'a Store,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        self.search_accessor(query, store, provider, counters)
    }

    #[doc(hidden)]
    fn prune_accessor<'a>(
        &'a self,
        store: &'a Store,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>>;
}
