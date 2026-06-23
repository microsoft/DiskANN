/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, pin::Pin};

use diskann::{
    graph::{DiskANNIndex, search::Knn},
    neighbor::Neighbor,
};
use half::f16;

use diskann_inmem::{Context, Provider, Strategy, layers};

use super::datatype::{AsDataType, DataType, FromSlice, Slice};

pub(crate) trait Index {
    fn data_type(&self) -> DataType;

    fn search<'a>(
        &'a self,
        query: Slice<'a>,
        knn: Knn,
        neighbors: &'a mut Vec<Neighbor<u64>>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;

    fn insert<'a>(
        &'a self,
        vector: Slice<'a>,
        id: u64,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;

    // fn retire(&self, id: u64) -> anyhow::Result<()>;
}

///////////
// Impls //
///////////

impl<T> Index for DiskANNIndex<Provider<layers::Full<T>, u64>>
where
    layers::Full<T>: for<'a> layers::Insert<Query<'a> = &'a [T]>,
    T: FromSlice + AsDataType + Send + Sync + 'static,
{
    fn data_type(&self) -> DataType {
        T::DATA_TYPE
    }

    fn search<'a>(
        &'a self,
        query: Slice<'a>,
        knn: Knn,
        neighbors: &'a mut Vec<Neighbor<u64>>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>> {
        let fut = async move {
            let query = query.try_cast()?;
            let _ = self
                .search(knn, &Strategy, &Context, query, neighbors)
                .await?;

            Ok(())
        };

        Box::pin(fut)
    }

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

    // fn retire(&self, id: u64) -> anyhow::Result<()> {
    // }
}
