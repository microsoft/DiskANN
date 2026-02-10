/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Range, sync::Arc};

use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    graph::{self, glue},
    provider,
    utils::async_tools::VectorIdBoxSlice,
};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::build::{Build, ids::ToId};

/// A built-in helper for benchmarking [multi-insert](graph::DiskANNIndex::multi_insert).
///
/// This is intended to be used in conjunction with [`crate::build::build`] and [`crate::build::build_tracked`].
///
/// # Notes
///
/// The multi-insert API for [`diskann::graph::DiskANNIndex`] parallelizes insertion internally. When using
/// [`crate::build::build`], users should use [`crate::build::Parallelism::sequential`].
pub struct MultiInsert<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    data: Arc<Matrix<T>>,
    strategy: S,
    to_id: Box<dyn ToId<DP::ExternalId>>,
}

impl<DP, T, S> MultiInsert<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`MultiInsert`] builder for the given `index`.
    ///
    /// Vectors will be inserted using all rows of `data` with `strategy` used
    /// for the [`diskann::graph::glue::InsertStrategy`].
    ///
    /// Parameter `to_id` will be used to convert row indices of `data` (`0..data.nrows()`)
    /// to external IDs.
    pub fn new<I>(
        index: Arc<graph::DiskANNIndex<DP>>,
        data: Arc<Matrix<T>>,
        strategy: S,
        to_id: I,
    ) -> Arc<Self>
    where
        I: ToId<DP::ExternalId>,
    {
        Arc::new(Self {
            index,
            data,
            strategy,
            to_id: Box::new(to_id),
        })
    }
}

impl<DP, T, S> Build for MultiInsert<DP, T, S>
where
    DP: provider::DataProvider<Context: Default> + provider::SetElement<[T]>,
    S: glue::InsertStrategy<DP, [T], PruneStrategy: Clone> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
    // TODO (Mark): This is a very very unfortunate bound and should be cleaned up with
    // an overhaul to the working set.
    for<'a> glue::aliases::InsertPruneAccessor<'a, S, DP, [T]>: glue::AsElement<&'a [T]>,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.data.nrows()
    }

    async fn build(&self, range: Range<usize>) -> ANNResult<Self::Output> {
        let vectors: ANNResult<Box<[_]>> = range
            .into_iter()
            .map(|i| {
                let id = self.to_id.to_id(i)?;
                let vector = self.data.get_row(i).ok_or_else(|| {
                    #[derive(Debug)]
                    struct OutOfBounds {
                        max: usize,
                        accessed: usize,
                    }

                    impl std::fmt::Display for OutOfBounds {
                        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                            write!(
                                f,
                                "tried to access data with {} rows at index {}",
                                self.max, self.accessed
                            )
                        }
                    }

                    ANNError::message(
                        ANNErrorKind::Opaque,
                        OutOfBounds {
                            max: self.data.nrows(),
                            accessed: i,
                        },
                    )
                })?;

                Ok(VectorIdBoxSlice::new(id, vector.into()))
            })
            .collect();

        let context = DP::Context::default();
        self.index
            .multi_insert(self.strategy.clone(), &context, vectors?)
            .await?;

        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroUsize;

    use diskann::{
        graph::{
            DiskANNIndex,
            test::{provider, synthetic},
        },
        provider::NeighborAccessor,
        utils::IntoUsize,
    };

    use crate::build;

    #[test]
    fn test_multi_insert() {
        let grid = synthetic::Grid::Four;
        let size = 4;
        let start_id = u32::MAX;
        let distance = diskann_vector::distance::Metric::L2;

        let start_point = grid.start_point(size);
        let data = Arc::new(grid.data(size));

        let provider_config = provider::Config::new(
            distance,
            2 * grid.dim().into_usize(),
            std::iter::once(provider::StartPoint::new(start_id, start_point)),
        )
        .unwrap();

        let provider = provider::Provider::new(provider_config);

        let index_config = diskann::graph::config::Builder::new(
            provider.max_degree().checked_sub(3).unwrap(),
            diskann::graph::config::MaxDegree::new(provider.max_degree()),
            20,
            distance.into(),
        )
        .build()
        .unwrap();

        let index = Arc::new(DiskANNIndex::new(index_config, provider, None));

        let rt = crate::tokio::runtime(1).unwrap();
        let builder = MultiInsert::new(
            index.clone(),
            data.clone(),
            provider::Strategy::new(),
            build::ids::Identity::<u32>::new(),
        );
        let _ = build::build(
            builder.clone(),
            build::Parallelism::sequential(NonZeroUsize::new(10).unwrap()),
            &rt,
        )
        .unwrap();

        // Ensure that the index is correctly populated.
        let accessor = index.provider().neighbors();
        let mut v = diskann::graph::AdjacencyList::new();

        for i in 0..data.nrows() {
            rt.block_on(accessor.get_neighbors(i.try_into().unwrap(), &mut v))
                .unwrap();
            assert!(!v.is_empty());
        }

        // Check the start point.
        rt.block_on(accessor.get_neighbors(start_id, &mut v))
            .unwrap();
        assert!(!v.is_empty());

        // Test that we correctly get an indexing error for out-of-bounds accesses.
        let err = rt
            .block_on(builder.build(data.nrows()..data.nrows() + 1))
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("tried to access data"),
            "actual message: {msg}"
        );
    }
}
