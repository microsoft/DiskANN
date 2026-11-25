/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Range, sync::Arc};

use diskann::{
    ANNError, ANNResult,
    graph::{self, compat},
};
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, Matrix},
};

use crate::build::{Build, ids::ToId};

/// A built-in helper for benchmarking [multi-insert](graph::compat::DiskANNIndex::multi_insert).
///
/// This is intended to be used in conjunction with [`crate::build::build`] and [`crate::build::build_tracked`].
///
/// # Notes
///
/// The multi-insert API for [`diskann::graph::compat::DiskANNIndex`] parallelizes insertion internally. When using
/// [`crate::build::build`], users should use [`crate::build::Parallelism::sequential`].
pub struct MultiInsert<DP, T, S>
where
    DP: compat::DataProvider,
{
    index: Arc<graph::compat::DiskANNIndex<DP>>,
    data: Arc<rowmajor::Owned<T>>,
    strategy: S,
    to_id: Box<dyn ToId<DP::ExternalId>>,
}

impl<DP, T, S> MultiInsert<DP, T, S>
where
    DP: compat::DataProvider,
{
    /// Construct a new [`MultiInsert`] builder for the given `index`.
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

impl<DP, T, S> Build for MultiInsert<DP, T, S>
where
    DP: compat::DataProvider<Context: Default> + for<'a> compat::SetElement<&'a [T]>,
    S: compat::MultiInsertStrategy<DP, rowmajor::Owned<T>> + Clone + 'static,
    T: AsyncFriendly + Clone,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.data.nrows()
    }

    async fn build(&self, range: Range<usize>) -> ANNResult<Self::Output> {
        let vectors = self
            .data
            .subview(range.clone())
            .ok_or_else(|| {
                #[derive(Debug)]
                struct OutOfBounds {
                    max: usize,
                    start: usize,
                    end: usize,
                }

                impl std::fmt::Display for OutOfBounds {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(
                            f,
                            "tried to access data with {} rows at range [{}, {})",
                            self.max, self.start, self.end
                        )
                    }
                }

                ANNError::message(OutOfBounds {
                    max: self.data.nrows(),
                    start: range.start,
                    end: range.end,
                })
            })?
            .to_rowmajor_owned();

        let ids: ANNResult<Arc<[_]>> = range.into_iter().map(|i| self.to_id.to_id(i)).collect();
        let context = DP::Context::default();
        self.index
            .multi_insert::<S, _>(self.strategy.clone(), &context, Arc::new(vectors), ids?)
            .await?;

        Ok(())
    }
}
