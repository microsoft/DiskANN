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

/// A built-in helper for running and benchmarking
/// [`inplace_delete`](graph::compat::DiskANNIndex::inplace_delete).
///
/// This is intended to be used in conjunction with [`crate::build::build`] and
/// [`crate::build::build_tracked`].
pub struct InplaceDelete<DP, S>
where
    DP: compat::DataProvider,
{
    index: Arc<graph::compat::DiskANNIndex<DP>>,
    strategy: S,
    num_to_replace: usize,
    inplace_delete_method: compat::InplaceDeleteMethod,
    to_id: Box<dyn ToIdSized<DP::ExternalId>>,
}

impl<DP, S> InplaceDelete<DP, S>
where
    DP: compat::DataProvider,
{
    /// Construct a new [`InplaceDelete`] build stage.
    ///
    /// This [`Build`] object will run for all Ids provided by `to_id`, invoking
    /// [`diskann::graph::compat::DiskANNIndex::inplace_delete`] on each ID.
    ///
    /// Arguments `num_to_replace` and `inplace_delete_method` are passed directly to the method
    /// on [`diskann::graph::compat::DiskANNIndex`].
    pub fn new(
        index: Arc<graph::compat::DiskANNIndex<DP>>,
        strategy: S,
        num_to_replace: usize,
        inplace_delete_method: compat::InplaceDeleteMethod,
        to_id: impl ToIdSized<DP::ExternalId> + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            index,
            strategy,
            num_to_replace,
            inplace_delete_method,
            to_id: Box::new(to_id),
        })
    }
}

impl<DP, S> Build for InplaceDelete<DP, S>
where
    DP: compat::DataProvider<Context: Default> + compat::Delete,
    S: compat::InplaceDeleteStrategy<DP> + Clone,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.to_id.len()
    }

    async fn build(&self, range: Range<usize>) -> ANNResult<Self::Output> {
        for i in range {
            let context = DP::Context::default();
            self.index
                .inplace_delete(
                    self.strategy.clone(),
                    &context,
                    &self.to_id.to_id(i)?,
                    self.num_to_replace,
                    self.inplace_delete_method,
                )
                .await?;
        }
        Ok(())
    }
}
