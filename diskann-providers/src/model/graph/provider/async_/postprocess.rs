/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared search post-processing.

use diskann::{
    graph::{SearchOutputBuffer, glue},
    neighbor::Neighbor,
    provider::BuildQueryComputer,
};

/// A bridge allowing `Accessors` to opt-in to [`RemoveDeletedIdsAndCopy`] by delegating to
/// an implementation of the [`DeletionCheck`] trait.
///
/// # Note
///
/// This **must not** be used as a general replacement for [`diskann::provider::Delete`].
/// This must only be used as a performance improvement for [`RemoveDeletedIdsAndCopy`].
pub(crate) trait AsDeletionCheck {
    type Checker: DeletionCheck;
    fn as_deletion_check(&self) -> &Self::Checker;
}

/// A light-weight, synchronous alternative to [`Delete`], targeted at quickly filtering out
/// deleted IDs during search post-processing.
///
/// For the [`NoDeletes`] case, we rely on constant-propagation and dead code elimination
/// to optimize away filters.
pub(crate) trait DeletionCheck {
    fn deletion_check(&self, id: u32) -> bool;
}

/// A [`SearchPostProcess`] routine that fuses the removal of deleted elements with the
/// copying of IDs into an output buffer.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoveDeletedIdsAndCopy;

impl<A, T> glue::SearchPostProcess<A, T> for RemoveDeletedIdsAndCopy
where
    A: BuildQueryComputer<T, Id = u32> + AsDeletionCheck,
    T: ?Sized,
{
    type Error = std::convert::Infallible;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        _query: &T,
        _computer: &<A as BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        let checker = accessor.as_deletion_check();
        let count = output.extend(candidates.filter_map(|n| {
            if checker.deletion_check(n.id) {
                None
            } else {
                Some((n.id, n.distance))
            }
        }));
        std::future::ready(Ok(count))
    }
}
