/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::DiskANNIndex,
    provider::{self, Delete},
    utils::ONE,
    ANNError, ANNErrorKind, ANNResult,
};
use diskann_benchmark_core::{
    build::{self, ids::Range, Parallelism},
    streaming::graph::DropDeleted,
};
use diskann_providers::model::graph::provider::async_::{
    inmem::DefaultProvider, TableDeleteProviderAsync,
};
use diskann_utils::future::AsyncFriendly;
use tokio::runtime::Runtime;

use crate::index::streaming::{
    runner::Maintainer,
    stats::{GenericStats, StreamStats},
};

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

//////////////////////
// Inmem Maintainer //
//////////////////////

/// Inmem maintenance: runs `DropDeleted` to unlink deleted neighbors, then `Release`
/// to free deleted slots for reuse.
pub(crate) struct InmemMaintainer;

impl<U, V> Maintainer<DefaultProvider<U, V, TableDeleteProviderAsync>> for InmemMaintainer
where
    U: AsyncFriendly,
    V: AsyncFriendly,
{
    fn maintain(
        &self,
        index: &Arc<DiskANNIndex<DefaultProvider<U, V, TableDeleteProviderAsync>>>,
        runtime: &Runtime,
        ntasks: NonZeroUsize,
    ) -> anyhow::Result<StreamStats> {
        let range = index.provider().iter();

        let runner = DropDeleted::new(index.clone(), false, Range::new(range));

        let drop_deleted = build::build(runner, Parallelism::fixed(Some(ONE), ntasks), runtime)?;

        let release = build::build(
            Release::new(index.clone()),
            Parallelism::fixed(Some(ONE), ntasks),
            runtime,
        )?;

        Ok(StreamStats::Maintain(vec![
            GenericStats::new(Cow::Borrowed("Drop Deleted"), drop_deleted)?,
            GenericStats::new(Cow::Borrowed("Release"), release)?,
        ]))
    }
}
