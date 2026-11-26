/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Range, sync::Arc};

use diskann::{
    ANNResult,
    graph::{self, compat},
};

use crate::build::{Build, ids::ToIdSized};

/// A [`Build`] stage that invokes
/// [`drop_deleted_neighbors`](diskann::graph::compat::DiskANNIndex::drop_deleted_neighbors)
/// on a collection of points.
///
/// The collection of points is determined by an implementation of [`ToIdSized`].
pub struct DropDeleted<DP>
where
    DP: compat::DataProvider,
{
    index: Arc<graph::compat::DiskANNIndex<DP>>,
    only_orphans: bool,
    to_id: Box<dyn ToIdSized<DP::InternalId>>,
}

impl<DP> DropDeleted<DP>
where
    DP: compat::DataProvider,
{
    /// Construct a new [`DropDeleted`] build stage.
    ///
    /// This [`Build`] object will run for all Ids provided by `to_id`, invoking
    /// [`diskann::graph::compat::DiskANNIndex::drop_deleted_neighbors`] on each ID.
    ///
    /// Argument `only_orphans` is passed directly to the method on [`diskann::graph::compat::DiskANNIndex`].
    ///
    /// # Notes
    ///
    /// This method is a little different from other stages since it uses internal IDs
    /// rather than external IDs. As such, users are **not** encouraged to use it.
    pub fn new(
        index: Arc<graph::compat::DiskANNIndex<DP>>,
        only_orphans: bool,
        to_id: impl ToIdSized<DP::InternalId> + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            index,
            only_orphans,
            to_id: Box::new(to_id),
        })
    }
}

impl<DP> Build for DropDeleted<DP>
where
    DP: compat::DataProvider<Context: Default> + compat::Delete + compat::DefaultAccessor,
    for<'a> <DP as compat::DefaultAccessor>::Accessor<'a>:
        graph::glue::NeighborAccessorMut<Id = DP::InternalId>,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.to_id.len()
    }

    async fn build(&self, range: Range<usize>) -> ANNResult<Self::Output> {
        let mut accessor = self.index.provider().default_accessor();
        for i in range {
            let context = DP::Context::default();
            self.index
                .drop_deleted_neighbors(
                    &context,
                    &mut accessor,
                    self.to_id.to_id(i)?,
                    self.only_orphans,
                )
                .await?;
        }
        Ok(())
    }
}
