/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Range, sync::Arc};

use diskann::{ANNResult, graph, provider};

use crate::build::{Build, ids::ToIdSized};

/// A [`Build`] stage that invokes
/// [`drop_deleted_neighbors`](diskann::graph::DiskANNIndex::drop_deleted_neighbors)    
/// on a collection of points.
///
/// The collection of points is determined by an implementation of [`ToIdSized`].
pub struct DropDeleted<DP>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    only_orphans: bool,
    to_id: Box<dyn ToIdSized<DP::InternalId>>,
}

impl<DP> DropDeleted<DP>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`DropDeleted`] build stage.
    ///
    /// This [`Build`] object will run for all Ids provided by `to_id`, invoking
    /// [`diskann::graph::DiskANNIndex::drop_deleted_neighbors`] on each ID.
    ///
    /// Argument `only_orphans` is passed directly to the method on [`diskann::graph::DiskANNIndex`].
    ///
    /// # Notes
    ///
    /// This method is a little different from other stages since it uses internal IDs
    /// rather than external IDs. As such, users are **not** encouraged to use it.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
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
    DP: provider::DataProvider<Context: Default> + provider::Delete + provider::DefaultAccessor,
    for<'a> <DP as provider::DefaultAccessor>::Accessor<'a>: provider::AsNeighborMut,
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroUsize;

    use diskann::{
        graph::test::provider,
        provider::{Delete, NeighborAccessor},
        utils::ONE,
    };

    use crate::{build, streaming::graph::test};

    // In this test - we build an index, delete all the even numbered entries, then run
    // `drop_deleted` on the whole index.
    //
    // This will leave a broken index in the end, but we mainly care that `drop_deleted`
    // runs correctly.
    #[test]
    fn test_drop_deleted() {
        let (index, num_points) = test::build_test_index();
        let rt = crate::tokio::runtime(2).unwrap();

        let ctx = provider::Context::new();
        let num_points: u32 = num_points.try_into().unwrap();

        for i in (0..num_points).filter(|i| i.is_multiple_of(2)) {
            rt.block_on(index.provider().delete(&ctx, &i)).unwrap();
        }

        let _ = build::build(
            DropDeleted::new(index.clone(), false, build::ids::Range::new(0..num_points)),
            build::Parallelism::dynamic(ONE, NonZeroUsize::new(2).unwrap()),
            &rt,
        )
        .unwrap();

        let accessor = index.provider().neighbors();
        let mut v = diskann::graph::AdjacencyList::new();

        // `drop_deleted` short-circuits already deleted entries. So we should only check
        // the odd indices.
        for i in (0..num_points).filter(|i| !i.is_multiple_of(2)) {
            rt.block_on(accessor.get_neighbors(i, &mut v)).unwrap();

            assert!(!v.is_empty());
            for n in v.iter() {
                assert!(
                    !n.is_multiple_of(2),
                    "all multiples of 2 should be removed for entry {}: {:?}",
                    i,
                    v,
                );
            }
        }
    }
}
