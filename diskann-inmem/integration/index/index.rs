/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, pin::Pin};

use diskann::{
    graph::{DiskANNIndex, search::Knn},
    neighbor::Neighbor,
    utils::IntoUsize,
};
use diskann_benchmark_runner::utils::fmt::KeyValue;
use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use diskann_inmem::{Context, Provider, Strategy, integration, layers};

use crate::support::datatype::{AsDataType, DataType, FromSlice, Slice};

pub(crate) trait Index {
    fn data_type(&self) -> DataType;

    fn search<'a>(
        &'a self,
        query: Slice<'a>,
        knn: Knn,
        neighbors: &'a mut Vec<Neighbor<u64>>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<KnnSearch>> + 'a>>;

    fn insert<'a>(
        &'a self,
        vector: Slice<'a>,
        id: u64,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;

    fn counters(&self) -> Counters;
    // fn retire(&self, id: u64) -> anyhow::Result<()>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KnnSearch {
    hops: usize,
    cmps: usize,
}

impl KnnSearch {
    pub(crate) fn new() -> Self {
        Self { hops: 0, cmps: 0 }
    }
}

impl From<diskann::graph::index::SearchStats> for KnnSearch {
    fn from(stats: diskann::graph::index::SearchStats) -> Self {
        Self {
            hops: stats.hops.into_usize(),
            cmps: stats.cmps.into_usize(),
        }
    }
}

impl std::ops::AddAssign for KnnSearch {
    fn add_assign(&mut self, rhs: Self) {
        self.hops = self.hops.wrapping_add(rhs.hops);
        self.cmps = self.cmps.wrapping_add(rhs.cmps);
    }
}

impl std::fmt::Display for KnnSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hops = {}, cmps = {}", self.hops, self.cmps)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Counters {
    query_distance: u64,
    distance: u64,
    get_vector: u64,
    set_vector: u64,
    get_neighbors: u64,
    set_neighbors: u64,
    append_neighbors: u64,
}

impl Counters {
    pub(crate) fn delta(&self, after: &Counters) -> anyhow::Result<Self> {
        #[derive(Debug, Error)]
        #[error(
            "counter \"{}\" non-monotonically increasing from {} to {}",
            self.0,
            self.1,
            self.2
        )]
        struct NonMonotonic(&'static str, u64, u64);

        fn check(before: u64, after: u64, field: &'static str) -> Result<u64, NonMonotonic> {
            after
                .checked_sub(before)
                .ok_or(NonMonotonic(field, before, after))
        }

        let delta = Self {
            query_distance: check(self.query_distance, after.query_distance, "query_distance")?,
            distance: check(self.distance, after.distance, "distance")?,
            get_vector: check(self.get_vector, after.get_vector, "get_vector")?,
            set_vector: check(self.set_vector, after.set_vector, "set_vector")?,
            get_neighbors: check(self.get_neighbors, after.get_neighbors, "get_neighbors")?,
            set_neighbors: check(self.set_neighbors, after.set_neighbors, "set_neighbors")?,
            append_neighbors: check(
                self.append_neighbors,
                after.append_neighbors,
                "append_neighbors",
            )?,
        };

        Ok(delta)
    }
}

impl std::fmt::Display for Counters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("query_distance", &self.query_distance);
        kv.push("distance", &self.distance);
        kv.push("get_vector", &self.get_vector);
        kv.push("set_vector", &self.set_vector);
        kv.push("get_neighbors", &self.get_neighbors);
        kv.push("set_neighbors", &self.set_neighbors);
        kv.push("append_neighbors", &self.append_neighbors);
        kv.render(f)
    }
}

impl From<integration::counters::CounterSnapshot> for Counters {
    fn from(snapshot: integration::counters::CounterSnapshot) -> Self {
        Self {
            query_distance: snapshot.query_distance,
            distance: snapshot.distance,
            get_vector: snapshot.get_vector,
            set_vector: snapshot.set_vector,
            get_neighbors: snapshot.get_neighbors,
            set_neighbors: snapshot.set_neighbors,
            append_neighbors: snapshot.append_neighbors,
        }
    }
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

    fn counters(&self) -> Counters {
        self.provider().counters().into()
    }

    // fn retire(&self, id: u64) -> anyhow::Result<()> {
    // }
}
