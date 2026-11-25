/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, pin::Pin, sync::Arc};

use diskann::{
    graph::search::{InlineFilterSearch, Knn, MultihopFilterSearch},
    neighbor::Neighbor,
};
use diskann_utils::views::rowmajor::Matrix;

use diskann_inmem::{self as di, repr};

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
    ) -> anyhow::Result<KnnSearch>;

    //--------//
    // Insert //
    //--------//

    fn insert<'a>(&'a self, vector: Slice<'a>, id: u64) -> anyhow::Result<()>;

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

    fn multi_insert<'a>(
        self: Arc<Self>,
        batch: DatasetView<'a>,
        ids: std::ops::Range<u64>,
        handl: &'a tokio::runtime::Handle,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;
}

///////////
// Impls //
///////////

macro_rules! index {
    ($T:ty) => {
        index!({} $T where);
    };
    ({ $($generics:ident),* $(,)? } $T:ty where $($where:tt)*) => {
        impl<$($generics),*> PartialIndex for di::index::Index<$T, u64>
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
            ) -> anyhow::Result<KnnSearch> {
                let query = query.try_cast()?;
                let stats = self.read()?.search(query, knn, neighbors)?;
                Ok(stats.into())
            }

            //--------//
            // Insert //
            //--------//

            fn insert<'a>(
                &'a self,
                vector: Slice<'a>,
                id: u64,
            ) -> anyhow::Result<()> {
                let vector = vector.try_cast()?;
                self.write()?.insert(vector, id)?;
                Ok(())
            }

            //----------//
            // Counters //
            //----------//

            fn counters(&self) -> Counters {
                self.counters().into()
            }
        }

        impl<$($generics),*> Index for di::index::Index<$T, u64>
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
                    predicate.len() == self.capacity().value(),
                    "expected predicate to densly represent external IDs",
                );

                let query = query.try_cast()?;
                let read = self.read()?;
                let mut filtered = read.filtered(
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

            fn multi_insert<'a>(
                self: Arc<Self>,
                batch: DatasetView<'a>,
                ids: std::ops::Range<u64>,
                handle: &'a tokio::runtime::Handle,
            ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>> {
                let fut = async move {
                    let vectors = batch.try_cast()?.to_rowmajor_owned();
                    let ids: Vec<u64> = ids.collect();
                    self.batch_insert_builtin(vectors, &ids, handle).await?;
                    Ok(())
                };

                Box::pin(fut)
            }
        }
    }
}

index!({ T } repr::Full<T> where T: repr::FullPrecision + TryCast + AsDataType);
index!(repr::Quantized);

impl PartialIndex for di::index::buildonly::Index<repr::Full<f32>> {
    fn search<'a>(
        &'a self,
        query: Slice<'a>,
        knn: Knn,
        neighbors: &'a mut Vec<Neighbor<u64>>,
    ) -> anyhow::Result<KnnSearch> {
        let query = query.try_cast()?;
        let mut neighbors_u32 = Vec::<Neighbor<u32>>::new();
        let stats = self.read()?.search(query, knn, &mut neighbors_u32)?;

        neighbors.extend(
            neighbors_u32
                .into_iter()
                .map(|n| Neighbor::new(u64::from(*n.id()), *n.distance())),
        );

        Ok(stats.into())
    }

    fn insert<'a>(&'a self, vector: Slice<'a>, id: u64) -> anyhow::Result<()> {
        let id: u32 = u32::try_from(id).map_err(|_| anyhow::anyhow!("id {id} exceeds u32::MAX"))?;
        let vector = vector.try_cast()?;
        self.write()?.insert(vector, id)?;
        Ok(())
    }

    fn counters(&self) -> Counters {
        self.counters().into()
    }
}
