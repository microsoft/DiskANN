/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Range, sync::Arc};

use diskann::{
    ANNResult,
    graph::{self, compat},
};
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, Matrix},
};

use crate::build::{Build, ids::ToId};

/// A built-in helper for benchmarking [insert](graph::compat::DiskANNIndex::insert).
///
/// This is intended to be used in conjunction with [`crate::build::build`] and [`crate::build::build_tracked`].
#[derive(Debug)]
pub struct SingleInsert<DP, T, S>
where
    DP: compat::DataProvider,
{
    index: Arc<graph::compat::DiskANNIndex<DP>>,
    data: Arc<rowmajor::Owned<T>>,
    strategy: S,
    to_id: Box<dyn ToId<DP::ExternalId>>,
}

impl<DP, T, S> SingleInsert<DP, T, S>
where
    DP: compat::DataProvider,
{
    /// Construct a new [`SingleInsert`] builder for the given `index`.
    ///
    /// Vectors will be inserted using all rows of `data` with `strategy` used
    /// for the [`diskann::graph::compat::InsertStrategy`].
    ///
    /// Parameter `to_id` will be used to convert row indices of `data` (`0..data.nrows()`)
    /// to external IDs.
    pub fn new<I>(
        index: Arc<graph::compat::DiskANNIndex<DP>>,
        data: Arc<rowmajor::Owned<T>>,
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
    DP: compat::DataProvider<Context: Default> + for<'a> compat::SetElement<&'a [T]>,
    S: for<'a> compat::InsertStrategy<'a, DP, &'a [T]> + Clone + AsyncFriendly,
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
                    &self.strategy,
                    &context,
                    &self.to_id.to_id(i)?,
                    self.data.row(i),
                )
                .await?;
        }
        Ok(())
    }
}
