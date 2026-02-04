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

use crate::build::{Build, ids::ToIdSized};

/// A built-in helper for running and benchmarking
/// [`inplace_delete`](graph::DiskANNIndex::inplace_delete).
///
/// This is intended to be used in conjunction with [`crate::build::build`] and
/// [`crate::build::build_tracked`].
pub struct InplaceDelete<DP, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    strategy: S,
    num_to_replace: usize,
    inplace_delete_method: graph::InplaceDeleteMethod,
    to_id: Box<dyn ToIdSized<DP::ExternalId>>,
}

impl<DP, S> InplaceDelete<DP, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`InplaceDelete`] build stage.
    ///
    /// This [`Build`] object will run for all Ids provided by `to_id`, invoking
    /// [`diskann::graph::DiskANNIndex::inplace_delete`] on each ID.
    ///
    /// Arguments `num_to_replace` and `inplace_delete_method` are passed directly to the method
    /// on [`diskann::graph::DiskANNIndex`].
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        strategy: S,
        num_to_replace: usize,
        inplace_delete_method: graph::InplaceDeleteMethod,
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
    DP: provider::DataProvider<Context: Default> + provider::Delete,
    S: glue::InplaceDeleteStrategy<DP> + Clone,
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroUsize;

    use diskann::{graph::test::provider, provider::Delete, utils::ONE};

    use crate::{build, streaming::graph::test};

    #[test]
    fn test_inplace_delete() {
        let (index, num_points) = test::build_test_index();
        let points_to_delete = [10, 20, 30, 40];

        let rt = crate::tokio::runtime(2).unwrap();
        let _ = build::build(
            InplaceDelete::new(
                index.clone(),
                provider::Strategy::new(),
                4,
                graph::InplaceDeleteMethod::TwoHopAndOneHop,
                build::ids::Slice::new(points_to_delete.into()),
            ),
            build::Parallelism::dynamic(ONE, NonZeroUsize::new(2).unwrap()),
            &rt,
        )
        .unwrap();

        let num_points: u32 = num_points.try_into().unwrap();
        let ctx = provider::Context::new();
        for i in 0..num_points {
            let is_deleted = rt
                .block_on(index.provider().status_by_external_id(&ctx, &i))
                .unwrap()
                .is_deleted();
            if points_to_delete.contains(&i) {
                assert!(is_deleted, "expected {i} to be deleted");
            } else {
                assert!(!is_deleted, "expected {i} to not be deleted");
            }
        }
    }
}
