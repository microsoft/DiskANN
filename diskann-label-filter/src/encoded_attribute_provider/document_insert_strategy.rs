/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

//! A strategy wrapper that enables insertion of [Document] objects into a
//! [DiskANNIndex] using a [DocumentProvider].

use std::marker::PhantomData;

use diskann::{
    graph::glue::{self, ExpandBeam, InsertStrategy, PruneStrategy, SearchExt, SearchStrategy},
    provider::{Accessor, BuildQueryComputer, DataProvider, DelegateNeighbor, HasId},
    ANNResult,
};

use super::document_provider::DocumentProvider;
use crate::document::Document;
use crate::encoded_attribute_provider::roaring_attribute_store::RoaringAttributeStore;

/// A strategy wrapper that enables insertion of [Document] objects.
pub struct DocumentInsertStrategy<Inner, VT: ?Sized> {
    inner: Inner,
    _phantom: PhantomData<fn() -> VT>,
}

impl<Inner: Clone, VT: ?Sized> Clone for DocumentInsertStrategy<Inner, VT> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<Inner: Copy, VT: ?Sized> Copy for DocumentInsertStrategy<Inner, VT> {}

impl<Inner, VT: ?Sized> DocumentInsertStrategy<Inner, VT> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    pub fn inner(&self) -> &Inner {
        &self.inner
    }
}

/// Wrapper accessor for Document queries
pub struct DocumentSearchAccessor<Inner, VT: ?Sized> {
    inner: Inner,
    _phantom: PhantomData<fn() -> VT>,
}

impl<Inner, VT: ?Sized> DocumentSearchAccessor<Inner, VT> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<Inner, VT> HasId for DocumentSearchAccessor<Inner, VT>
where
    Inner: HasId,
    VT: ?Sized,
{
    type Id = Inner::Id;
}

impl<Inner, VT> Accessor for DocumentSearchAccessor<Inner, VT>
where
    Inner: Accessor,
    VT: ?Sized,
{
    type ElementRef<'a> = Inner::ElementRef<'a>;
    type Element<'a>
        = Inner::Element<'a>
    where
        Self: 'a;
    type Extended = Inner::Extended;
    type GetError = Inner::GetError;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl std::future::Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        self.inner.get_element(id)
    }

    fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        f: F,
    ) -> impl std::future::Future<Output = Result<(), Self::GetError>> + Send
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + for<'b> FnMut(Self::ElementRef<'b>, Self::Id),
    {
        self.inner.on_elements_unordered(itr, f)
    }
}

impl<'doc, Inner, VT> BuildQueryComputer<Document<'doc, VT>> for DocumentSearchAccessor<Inner, VT>
where
    Inner: BuildQueryComputer<VT>,
    VT: ?Sized,
{
    type QueryComputerError = Inner::QueryComputerError;
    type QueryComputer = Inner::QueryComputer;

    fn build_query_computer(
        &self,
        from: &Document<'doc, VT>,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.inner.build_query_computer(from.vector())
    }
}

impl<'this, Inner, VT> DelegateNeighbor<'this> for DocumentSearchAccessor<Inner, VT>
where
    Inner: DelegateNeighbor<'this>,
    VT: ?Sized,
{
    type Delegate = Inner::Delegate;
    fn delegate_neighbor(&'this mut self) -> Self::Delegate {
        self.inner.delegate_neighbor()
    }
}

impl<'doc, Inner, VT> ExpandBeam<Document<'doc, VT>> for DocumentSearchAccessor<Inner, VT>
where
    Inner: ExpandBeam<VT>,
    VT: ?Sized,
{
}

impl<Inner, VT> SearchExt for DocumentSearchAccessor<Inner, VT>
where
    Inner: SearchExt,
    VT: ?Sized,
{
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        self.inner.starting_points()
    }
    fn terminate_early(&mut self) -> bool {
        self.inner.terminate_early()
    }
}

impl<'doc, Inner, DP, VT>
    SearchStrategy<DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>, Document<'doc, VT>>
    for DocumentInsertStrategy<Inner, VT>
where
    Inner: InsertStrategy<DP, VT>,
    DP: DataProvider,
    VT: Sync + Send + ?Sized + 'static,
{
    type QueryComputer = Inner::QueryComputer;
    type PostProcessor = glue::CopyIds;
    type SearchAccessorError = Inner::SearchAccessorError;
    type SearchAccessor<'a> = DocumentSearchAccessor<Inner::SearchAccessor<'a>, VT>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'a <DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        let inner_accessor = self
            .inner
            .search_accessor(provider.inner_provider(), context)?;
        Ok(DocumentSearchAccessor::new(inner_accessor))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        glue::CopyIds
    }
}

impl<'doc, Inner, DP, VT>
    InsertStrategy<DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>, Document<'doc, VT>>
    for DocumentInsertStrategy<Inner, VT>
where
    Inner: InsertStrategy<DP, VT>,
    DP: DataProvider,
    VT: Sync + Send + ?Sized + 'static,
{
    type PruneStrategy = DocumentPruneStrategy<Inner::PruneStrategy>;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        DocumentPruneStrategy::new(self.inner.prune_strategy())
    }

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'a <DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        let inner_accessor = self
            .inner
            .insert_search_accessor(provider.inner_provider(), context)?;
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
    type DistanceComputer = Inner::DistanceComputer;
    type PruneAccessor<'a> = Inner::PruneAccessor<'a>;
    type PruneAccessorError = Inner::PruneAccessorError;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'a DP::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        self.inner
            .prune_accessor(provider.inner_provider(), context)
    }
}
