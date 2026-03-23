/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

//! A strategy wrapper that enables insertion of [Document] objects into a
//! [DiskANNIndex] using a [DocumentProvider].

use diskann::{
    graph::glue::{self, ExpandBeam, InsertStrategy, PruneStrategy, SearchExt, SearchStrategy},
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
    // _phantom: PhantomData<fn() -> VT>,
}

impl<Inner> DocumentSearchAccessor<Inner> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            // _phantom: PhantomData,
        }
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

impl<'doc, Inner, VT> BuildQueryComputer<Document<'doc, VT>> for DocumentSearchAccessor<Inner>
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

impl<'this, Inner> DelegateNeighbor<'this> for DocumentSearchAccessor<Inner>
where
    Inner: DelegateNeighbor<'this>,
{
    type Delegate = Inner::Delegate;
    fn delegate_neighbor(&'this mut self) -> Self::Delegate {
        self.inner.delegate_neighbor()
    }
}

impl<'doc, Inner, VT> ExpandBeam<Document<'doc, VT>> for DocumentSearchAccessor<Inner>
where
    Inner: ExpandBeam<VT>,
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
    SearchStrategy<DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>, Document<'doc, VT>>
    for DocumentInsertStrategy<Inner>
where
    Inner: InsertStrategy<DP, VT>,
    DP: DataProvider,
    VT: Sync + Send + ?Sized + 'static,
{
    type QueryComputer = Inner::QueryComputer;
    type PostProcessor = glue::CopyIds;
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

    fn post_processor(&self) -> Self::PostProcessor {
        glue::CopyIds
    }
}

impl<'doc, Inner, DP, VT>
    InsertStrategy<DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>, Document<'doc, VT>>
    for DocumentInsertStrategy<Inner>
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

#[cfg(test)]
mod tests {
    use super::{DocumentInsertStrategy, DocumentPruneStrategy, DocumentSearchAccessor};
    use diskann::{
        graph::{
            glue::{InsertStrategy, PruneStrategy, SearchExt, SearchStrategy},
            test::provider::{Config, Context, Provider, StartPoint, Strategy},
        },
        provider::BuildQueryComputer,
    };
    use diskann_vector::distance::Metric;

    use crate::{
        document::Document,
        encoded_attribute_provider::{
            document_provider::DocumentProvider, roaring_attribute_store::RoaringAttributeStore,
        },
    };

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Build a minimal test provider with a single start point and three dimensions.
    fn make_test_provider() -> Provider {
        let config = Config::new(
            Metric::L2,
            10,
            StartPoint::new(u32::MAX, vec![1.0f32, 2.0, 0.0]),
        )
        .expect("test provider config should be valid");
        Provider::new(config)
    }

    fn make_doc_provider(
        provider: Provider,
    ) -> DocumentProvider<Provider, RoaringAttributeStore<u32>> {
        DocumentProvider::new(provider, RoaringAttributeStore::new())
    }

    /// `search_accessor` successfully creates a `DocumentSearchAccessor` wrapping the
    /// inner accessor.
    #[test]
    fn test_search_accessor_creates_wrapped_accessor() {
        let strategy = DocumentInsertStrategy::new(Strategy::new());
        let provider = make_doc_provider(make_test_provider());
        let context = Context::new();

        let result = <DocumentInsertStrategy<Strategy> as SearchStrategy<
            DocumentProvider<Provider, RoaringAttributeStore<u32>>,
            Document<'_, [f32]>,
        >>::search_accessor(&strategy, &provider, &context);

        assert!(result.is_ok());
    }

    #[test]
    fn test_insert_search_accessor_creates_wrapped_accessor() {
        let strategy = DocumentInsertStrategy::new(Strategy::new());
        let provider = make_doc_provider(make_test_provider());
        let context = Context::new();

        let result = <DocumentInsertStrategy<Strategy> as InsertStrategy<
            DocumentProvider<Provider, RoaringAttributeStore<u32>>,
            Document<'_, [f32]>,
        >>::insert_search_accessor(&strategy, &provider, &context);

        assert!(result.is_ok());
    }

    #[test]
    fn test_prune_accessor_delegates_to_inner_provider() {
        let doc_prune_strategy = DocumentPruneStrategy::new(Strategy::new());
        let provider = make_doc_provider(make_test_provider());
        let context = Context::new();

        let result = <DocumentPruneStrategy<Strategy> as PruneStrategy<
            DocumentProvider<Provider, RoaringAttributeStore<u32>>,
        >>::prune_accessor(&doc_prune_strategy, &provider, &context);

        assert!(result.is_ok());
    }

    #[test]
    fn test_build_query_computer_extracts_vector_from_document() {
        let provider = make_test_provider();
        let context = Context::new();
        let strategy_inner = Strategy::new();
        let inner_accessor = strategy_inner
            .search_accessor(&provider, &context)
            .expect("creating search accessor should succeed");
        let doc_accessor = DocumentSearchAccessor::new(inner_accessor);

        let vector = vec![1.0f32, 2.0, 0.0];
        let doc = Document::new(vector.as_slice(), vec![]);

        let result = <DocumentSearchAccessor<_> as BuildQueryComputer<
            Document<'_, [f32]>,
        >>::build_query_computer(&doc_accessor, &doc);

        assert!(
            result.is_ok(),
            "build_query_computer should succeed for a valid vector"
        );
    }

    #[test]
    fn test_terminate_early_delegates_to_inner() {
        let provider = make_test_provider();
        let context = Context::new();
        let strategy_inner = Strategy::new();
        let mut inner_accessor = strategy_inner
            .search_accessor(&provider, &context)
            .expect("creating search accessor should succeed");
        let inner_terminate_early = inner_accessor.terminate_early();
        let mut doc_accessor = DocumentSearchAccessor::new(inner_accessor);
        assert_eq!(
            inner_terminate_early,
            doc_accessor.terminate_early(),
            "terminate_early should have same value as inner accessor"
        );
    }
}
