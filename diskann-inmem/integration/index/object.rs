/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, pin::Pin, sync::Arc};

use diskann::{
    graph::{
        DiskANNIndex, glue,
        search::{InlineFilterSearch, Knn, MultihopFilterSearch},
    },
    neighbor::Neighbor,
};
use diskann_utils::views::rowmajor::Matrix;

use diskann_inmem::{self as di, Context, Provider, Strategy, repr};

use super::stats::{Counters, KnnSearch};
use crate::support::datatype::{AsDataType, DatasetView, Slice, TryCast};

pub(crate) trait PartialIndex {
    //--------//
    // Search //
    //--------//

    fn search<'a>(
        &'a self,
        query: Slice<'a>,
        knn: Knn,
        neighbors: &'a mut Vec<Neighbor<u64>>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<KnnSearch>> + 'a>>;

    //--------//
    // Insert //
    //--------//

    fn insert<'a>(
        &'a self,
        vector: Slice<'a>,
        id: u64,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;

    //----------//
    // Counters //
    //----------//

    fn counters(&self) -> Counters;
}

#[derive(Debug, Clone)]
pub(crate) enum Filter {
    Inline(InlineFilterSearch),
    Multihop(MultihopFilterSearch),
}

pub(crate) trait Index: PartialIndex {
    //--------//
    // Search //
    //--------//

    // NOTE: This is a synchronous method.
    fn filter_search<'a>(
        &'a self,
        query: Slice<'a>,
        bitmap: &'a [bool],
        filter: Filter,
        neighbors: &'a mut Vec<Neighbor<u64>>,
    ) -> anyhow::Result<KnnSearch>;

    //--------//
    // Insert //
    //--------//

    fn multi_insert(
        self: Arc<Self>,
        batch: DatasetView<'_>,
        ids: std::ops::Range<u64>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + '_>>;
}

///////////
// Impls //
///////////

macro_rules! index {
    ($T:ty) => {
        index!({} $T where);
    };
    ({ $($generics:ident),* $(,)? } $T:ty where $($where:tt)*) => {
        impl<$($generics),*> PartialIndex for DiskANNIndex<Provider<$T, u64>>
        where
            $($where)*
        {
            //--------//
            // Search //
            //--------//

            fn search<'a>(
                &'a self,
                query: Slice<'a>,
                knn: Knn,
                neighbors: &'a mut Vec<Neighbor<u64>>,
            ) -> Pin<Box<dyn Future<Output = anyhow::Result<KnnSearch>> + 'a>> {
                let fut = async move {
                    let query = query.try_cast()?;
                    let stats = self
                        .search(knn, &Strategy, &Context, query, neighbors)
                        .await?;

                    Ok(stats.into())
                };

                Box::pin(fut)
            }

            //--------//
            // Insert //
            //--------//

            fn insert<'a>(
                &'a self,
                vector: Slice<'a>,
                id: u64,
            ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>> {
                let fut = async move {
                    let vector = vector.try_cast()?;
                    self.insert(&Strategy, &Context, &id, vector).await?;

                    Ok(())
                };

                Box::pin(fut)
            }

            //----------//
            // Counters //
            //----------//

            fn counters(&self) -> Counters {
                self.provider().counters().into()
            }
        }

        impl<$($generics),*> Index for DiskANNIndex<Provider<$T, u64>>
        where
            $($where)*
        {
            //--------//
            // Search //
            //--------//

            fn filter_search(
                &self,
                query: Slice<'_>,
                predicate: &[bool],
                filter: Filter,
                neighbors: &mut Vec<Neighbor<u64>>,
            ) -> anyhow::Result<KnnSearch> {
                anyhow::ensure!(
                    predicate.len() == self.provider().capacity().value(),
                    "expected predicate to densly represent external IDs",
                );

                let query = query.try_cast()?;
                let scoped = di::provider::scoped(&self)?;
                let mut filtered = scoped.filtered(
                    query,
                    |i| predicate[*i as usize],
                )?;

                let stats = match filter {
                    Filter::Inline(inline) => filtered.inline_filtered(inline, neighbors)?,
                    Filter::Multihop(multihop) => filtered.multihop_filtered(multihop, neighbors)?,
                };

                Ok(stats.into())
            }

            //--------//
            // Insert //
            //--------//

            fn multi_insert(
                self: Arc<Self>,
                batch: DatasetView<'_>,
                ids: std::ops::Range<u64>,
            ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + '_>> {
                let fut = async move {
                    let vectors = Arc::new(batch.try_cast()?.to_rowmajor_owned());
                    let ids: Arc<[u64]> = ids.collect();
                    (&self).multi_insert(Strategy, &Context, vectors, ids).await?;
                    Ok(())
                };

                Box::pin(fut)
            }
        }
    }
}

index!({ T } repr::Full<T> where T: repr::FullPrecision + TryCast + AsDataType);
index!(repr::Quantized);

impl PartialIndex for DiskANNIndex<di::provider::buildonly::Provider<repr::Full<f32>>> {
    fn search<'a>(
        &'a self,
        query: Slice<'a>,
        knn: Knn,
        neighbors: &'a mut Vec<Neighbor<u64>>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<KnnSearch>> + 'a>> {
        let fut = async move {
            let query = query.try_cast()?;
            let mut neighbors_u32 = Vec::<Neighbor<u32>>::new();
            let post_process = glue::Pipeline::new(glue::FilterStartPoints, glue::CopyIds);

            let stats = self
                .search_with(
                    knn,
                    &Strategy,
                    post_process,
                    &Context,
                    query,
                    &mut neighbors_u32,
                )
                .await?;

            neighbors.extend(
                neighbors_u32
                    .into_iter()
                    .map(|n| Neighbor::new(u64::from(*n.id()), *n.distance())),
            );

            Ok(stats.into())
        };

        Box::pin(fut)
    }

    fn insert<'a>(
        &'a self,
        vector: Slice<'a>,
        id: u64,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>> {
        let fut = async move {
            let id: u32 =
                u32::try_from(id).map_err(|_| anyhow::anyhow!("id {id} exceeds u32::MAX"))?;
            let vector = vector.try_cast()?;
            self.insert(&Strategy, &Context, &id, vector).await?;

            Ok(())
        };

        Box::pin(fut)
    }

    fn counters(&self) -> Counters {
        self.provider().counters().into()
    }
}
