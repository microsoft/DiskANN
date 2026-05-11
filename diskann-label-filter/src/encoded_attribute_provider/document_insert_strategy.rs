/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

//! A strategy wrapper that enables insertion of [Document] objects into a
//! [DiskANNIndex] using a [DocumentProvider].

use diskann::{
    graph::glue::{ExpandBeam, InsertStrategy, PruneStrategy, SearchExt, SearchStrategy},
    provider::{Accessor, BuildQueryComputer, DataProvider, DelegateNeighbor, HasId},
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

impl<Inner> Accessor for DocumentSearchAccessor<Inner>
where
    Inner: Accessor,
{
    type ElementRef<'a> = Inner::ElementRef<'a>;
    type Element<'a>
        = Inner::Element<'a>
    where
        Self: 'a;
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

impl<'doc, Inner, VT> BuildQueryComputer<&'doc Document<'doc, VT>> for DocumentSearchAccessor<Inner>
where
    Inner: BuildQueryComputer<&'doc VT>,
    VT: ?Sized,
{
    type QueryComputerError = Inner::QueryComputerError;
    type QueryComputer = Inner::QueryComputer;

    fn build_query_computer(
        &self,
        from: &'doc Document<'doc, VT>,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.inner.build_query_computer(from.vector())
    }
}

impl<'this, Inner> DelegateNeighbor<'this> for DocumentSearchAccessor<Inner>
where
    Inner: DelegateNeighbor<'this>,
{
    type Delegate = Inner::Delegate;
    fn delegate_neighbor(&'this mut self) -> Self::Delegate {
        self.inner.delegate_neighbor()
    }
}

impl<'doc, Inner, VT> ExpandBeam<&'doc Document<'doc, VT>> for DocumentSearchAccessor<Inner>
where
    Inner: ExpandBeam<&'doc VT>,
    VT: ?Sized,
{
}

impl<Inner> SearchExt for DocumentSearchAccessor<Inner>
where
    Inner: SearchExt,
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
    SearchStrategy<
        DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        &'doc Document<'doc, VT>,
    > for DocumentInsertStrategy<Inner>
where
    Inner: InsertStrategy<DP, &'doc VT>,
    DP: DataProvider,
    VT: Sync + Send + ?Sized + 'static,
{
    type QueryComputer = Inner::QueryComputer;
    type SearchAccessorError = Inner::SearchAccessorError;
    type SearchAccessor<'a> = DocumentSearchAccessor<Inner::SearchAccessor<'a>>;

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
}

impl<'doc, Inner, DP, VT>
    InsertStrategy<
        DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        &'doc Document<'doc, VT>,
    > for DocumentInsertStrategy<Inner>
where
    Inner: InsertStrategy<DP, &'doc VT>,
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
    type WorkingSet = Inner::WorkingSet;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'a DP::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        self.inner
            .prune_accessor(provider.inner_provider(), context)
    }

    fn create_working_set(&self, _capacity: usize) -> Self::WorkingSet {
        self.inner.create_working_set(_capacity)
    }
}
