/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Range, sync::Arc};

use diskann::{
    ANNResult,
    graph::{self, glue},
    provider,
};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::build::{Build, ids::ToId};

/// A built-in helper for benchmarking [insert](graph::DiskANNIndex::insert).
///
/// This is intended to be used in conjunction with [`crate::build::build`] and [`crate::build::build_tracked`].
#[derive(Debug)]
pub struct SingleInsert<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    data: Arc<Matrix<T>>,
    strategy: S,
    to_id: Box<dyn ToId<DP::ExternalId>>,
}

impl<DP, T, S> SingleInsert<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`SingleInsert`] builder for the given `index`.
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

impl<DP, T, S> Build for SingleInsert<DP, T, S>
where
    DP: provider::DataProvider<Context: Default> + provider::SetElement<[T]>,
    S: glue::InsertStrategy<DP, [T]> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.data.nrows()
    }

    async fn build(&self, range: Range<usize>) -> ANNResult<Self::Output> {
        for i in range {
            let context = DP::Context::default();
            self.index
                .insert(
                    self.strategy.clone(),
                    &context,
                    &self.to_id.to_id(i)?,
                    self.data.row(i),
                )
                .await?;
        }
        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::{
        graph::test::{provider, synthetic},
        provider::NeighborAccessor,
        utils::{IntoUsize, ONE},
    };

    use crate::build;

    #[test]
    fn test_single_insert() {
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

        let index = Arc::new(diskann::graph::DiskANNIndex::new(
            index_config,
            provider,
            None,
        ));

        let rt = crate::tokio::runtime(1).unwrap();
        let _ = build::build(
            SingleInsert::new(
                index.clone(),
                data.clone(),
                provider::Strategy::new(),
                build::ids::Identity::<u32>::new(),
            ),
            build::Parallelism::dynamic(ONE, ONE),
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
    }
}
