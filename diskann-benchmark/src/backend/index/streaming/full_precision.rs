/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::{DiskANNIndex, InplaceDeleteMethod},
    provider::{self, Delete},
    utils::{VectorRepr, ONE},
    ANNError, ANNErrorKind, ANNResult,
};
use diskann_benchmark_core::recall::Rows;
use diskann_providers::model::graph::provider::async_::{
    common,
    inmem::{self, DefaultProvider},
    TableDeleteProviderAsync,
};
use diskann_utils::{
    future::AsyncFriendly,
    views::{Matrix, MatrixView},
};

use super::{
    stats::{GenericStats, StreamStats},
    ManagedStream,
};
use crate::{
    backend::index::{
        build::{BuildKind, BuildStats},
        search::knn,
    },
    inputs::async_::TopkSearchPhase,
};

type FullPrecisionIndex<T> = Arc<
    DiskANNIndex<
        DefaultProvider<inmem::FullPrecisionStore<T>, common::NoStore, TableDeleteProviderAsync>,
    >,
>;

/// Full-Precision Streaming Index Implementation.
///
/// ## Behavior with Deletes
///
/// Deletes are processed by using `inplace_delete` to soft-delete data. Slots deleted this way
/// are not reused until maintenance is run, which drops deleted neighbors and releases the
/// deleted data points.
pub(crate) struct FullPrecisionStream<T>
where
    T: VectorRepr,
{
    pub(crate) index: FullPrecisionIndex<T>,
    pub(crate) search: TopkSearchPhase,
    pub(crate) runtime: tokio::runtime::Runtime,
    pub(crate) ntasks: NonZeroUsize,
    pub(crate) inplace_delete_num_to_replace: usize,
    pub(crate) inplace_delete_method: InplaceDeleteMethod,
}

impl<T> FullPrecisionStream<T>
where
    T: VectorRepr,
{
    // Common code-path for both inserts and replace.
    fn insert_(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<BuildStats> {
        let runner = diskann_benchmark_core::build::graph::SingleInsert::new(
            self.index.clone(),
            Arc::new(data.to_owned()),
            common::FullPrecision,
            diskann_benchmark_core::build::ids::Slice::new(slots.into()),
        );

        let results = diskann_benchmark_core::build::build(
            runner,
            diskann_benchmark_core::build::Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        BuildStats::new(BuildKind::SingleInsert, results)
    }
}

impl<T> ManagedStream<T> for FullPrecisionStream<T>
where
    T: VectorRepr,
{
    type Output = StreamStats;

    fn search(
        &self,
        queries: Arc<Matrix<T>>,
        groundtruth: &dyn Rows<u32>,
    ) -> anyhow::Result<Self::Output> {
        let knn = diskann_benchmark_core::search::graph::KNN::new(
            self.index.clone(),
            queries,
            diskann_benchmark_core::search::graph::Strategy::broadcast(common::FullPrecision),
        )?;

        let steps = knn::SearchSteps::new(
            self.search.reps,
            &self.search.num_threads,
            &self.search.runs,
        );
        let results = knn::run(&knn, groundtruth, steps)?;
        Ok(StreamStats::Search(results))
    }

    fn insert(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Insert(self.insert_(data, slots)?))
    }

    fn replace(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Replace(self.insert_(data, slots)?))
    }

    fn delete(&self, slots: &[u32]) -> anyhow::Result<Self::Output> {
        let runner = diskann_benchmark_core::streaming::graph::InplaceDelete::new(
            self.index.clone(),
            common::FullPrecision,
            self.inplace_delete_num_to_replace,
            self.inplace_delete_method,
            diskann_benchmark_core::build::ids::Slice::new(slots.into()),
        );

        let results = diskann_benchmark_core::build::build(
            runner,
            diskann_benchmark_core::build::Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        Ok(StreamStats::Delete(GenericStats::new(
            Cow::Borrowed("Delete"),
            results,
        )?))
    }

    fn maintain(&self) -> anyhow::Result<Self::Output> {
        let range = self.index.provider().iter();

        let runner = diskann_benchmark_core::streaming::graph::DropDeleted::new(
            self.index.clone(),
            false,
            diskann_benchmark_core::build::ids::Range::new(range),
        );

        let drop_deleted = diskann_benchmark_core::build::build(
            runner,
            diskann_benchmark_core::build::Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        let release = diskann_benchmark_core::build::build(
            Release::new(self.index.clone()),
            diskann_benchmark_core::build::Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        Ok(StreamStats::Maintain {
            drop_deleted: GenericStats::new(Cow::Borrowed("Drop Deleted"), drop_deleted)?,
            release: GenericStats::new(Cow::Borrowed("Release"), release)?,
        })
    }
}

/////////////
// Helpers //
/////////////

struct Release<U, V>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
{
    index: Arc<DiskANNIndex<DefaultProvider<U, V, TableDeleteProviderAsync>>>,
}

impl<U, V> Release<U, V>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
{
    fn new(index: Arc<DiskANNIndex<DefaultProvider<U, V, TableDeleteProviderAsync>>>) -> Arc<Self> {
        Arc::new(Self { index })
    }
}

/// NOTE: The implementation here strictly targets the implementation of [`provider::Delete`]
/// for [`DefaultProvider`] with the [`TableDeleteProviderAsync`] delete provider.
///
/// Trying to make this generic over the delete provider is not a recipe for a good time.
impl<U, V> diskann_benchmark_core::build::Build for Release<U, V>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.index.provider().total_points()
    }

    async fn build(&self, range: std::ops::Range<usize>) -> ANNResult<()> {
        let provider = self.index.provider();
        let ctx = &provider::DefaultContext;

        for internal_id in range {
            let internal_id: u32 = internal_id
                .try_into()
                .map_err(|_| ANNError::message(ANNErrorKind::Opaque, "invalid id provided"))?;
            if provider
                .status_by_external_id(ctx, &internal_id)
                .await?
                .is_deleted()
            {
                provider.release(ctx, internal_id).await?;
            }
        }
        Ok(())
    }
}
