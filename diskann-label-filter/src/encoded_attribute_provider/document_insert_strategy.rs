/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

//! A strategy wrapper that enables insertion of [Document] objects into a
//! [DiskANNIndex] using a [DocumentProvider].

use diskann::{
    graph::glue::{InsertStrategy, PruneStrategy, SearchAccessor, SearchStrategy},
    provider::{DataProvider, HasId},
    ANNResult,
};

use super::document_provider::DocumentProvider;
use crate::document::Document;
use crate::encoded_attribute_provider::roaring_attribute_store::RoaringAttributeStore;

/// A strategy wrapper that enables insertion of [Document] objects.
pub struct DocumentInsertStrategy<Inner> {
    inner: Inner,
}

impl<Inner: Clone> Clone for DocumentInsertStrategy<Inner> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<Inner: Copy> Copy for DocumentInsertStrategy<Inner> {}

impl<Inner> DocumentInsertStrategy<Inner> {
    pub fn new(inner: Inner) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &Inner {
        &self.inner
    }
}

/// Wrapper accessor for Document queries
pub struct DocumentSearchAccessor<Inner> {
    inner: Inner,
}

impl<Inner> DocumentSearchAccessor<Inner> {
    pub fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

impl<Inner> HasId for DocumentSearchAccessor<Inner>
where
    Inner: HasId,
{
    type Id = Inner::Id;
}

impl<Inner> SearchAccessor for DocumentSearchAccessor<Inner>
where
    Inner: SearchAccessor,
{
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        self.inner.starting_points()
    }

    fn start_point_distances<F>(
        &mut self,
        f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        self.inner.start_point_distances(f)
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: diskann::graph::glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        self.inner.expand_beam(ids, pred, on_neighbors)
    }

    fn terminate_early(&mut self) -> bool {
        self.inner.terminate_early()
    }

    fn is_not_start_point(
        &self,
    ) -> impl std::future::Future<
        Output = ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>,
    > + Send {
        self.inner.is_not_start_point()
    }
}

impl<'a, 'doc, Inner, DP, VT>
    SearchStrategy<
        'a,
        DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        &'doc Document<'doc, VT>,
    > for DocumentInsertStrategy<Inner>
where
    Inner: SearchStrategy<'a, DP, &'doc VT>,
    DP: DataProvider,
    VT: Sync + Send + ?Sized + 'static,
{
    type SearchAccessor = DocumentSearchAccessor<Inner::SearchAccessor>;
    type SearchAccessorError = Inner::SearchAccessorError;

    fn search_accessor(
        &'a self,
        provider: &'a DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'a <DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>> as DataProvider>::Context,
        query: &'doc Document<'doc, VT>,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let inner_accessor = self
            .inner
            .search_accessor(provider.inner_provider(), context, query.vector())?;
        Ok(DocumentSearchAccessor::new(inner_accessor))
    }
}

impl<'a, 'doc, Inner, DP, VT>
    InsertStrategy<
        'a,
        DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        &'doc Document<'doc, VT>,
    > for DocumentInsertStrategy<Inner>
where
    Inner: InsertStrategy<'a, DP, &'doc VT>,
    DP: DataProvider,
    VT: Sync + Send + ?Sized + 'static,
{
    type PruneStrategy = DocumentPruneStrategy<Inner::PruneStrategy>;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        DocumentPruneStrategy::new(self.inner.prune_strategy())
    }

    fn insert_search_accessor(
        &'a self,
        provider: &'a DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'a <DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>> as DataProvider>::Context,
        vector: &'doc Document<'doc, VT>,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let inner_accessor = self
            .inner
            .insert_search_accessor(provider.inner_provider(), context, vector.vector())?;
        Ok(DocumentSearchAccessor::new(inner_accessor))
    }
}

#[derive(Clone, Copy)]
pub struct DocumentPruneStrategy<Inner> {
    inner: Inner,
}

impl<Inner> DocumentPruneStrategy<Inner> {
    pub fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

impl<Inner, DP> PruneStrategy<DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>>
    for DocumentPruneStrategy<Inner>
where
    DP: DataProvider,
    Inner: PruneStrategy<DP>,
{
    type PruneAccessor<'a> = Inner::PruneAccessor<'a>;
    type PruneAccessorError = Inner::PruneAccessorError;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'a DP::Context,
        capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        self.inner
            .prune_accessor(provider.inner_provider(), context, capacity)
    }
}
