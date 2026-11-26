/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::ANNResult;
use diskann_utils::views::rowmajor::{self, Matrix};
use hashbrown::HashMap;

use crate::{
    index::{Id, Index, accessors::Prune, algorithms::Insert},
    repr,
};

//--------------//
// Batch Insert //
//--------------//

/// This struct exists solely for testing the standard implementation of batch-insert in
/// integration-sized contexts.
///
/// By virtue of being synchronous `diskann-inmem` can implement batch insert entirely
/// synchronously with rayon based parallelism. However, testing the provided methods within
/// `diskann` is still important.
#[derive(Debug)]
#[cfg(feature = "diskann-builtins")]
struct BatchInsertInner<R, M, T>
where
    M: Id,
{
    index: Arc<Index<R, M>>,
    data: rowmajor::Owned<T>,
    ids: HashMap<u32, usize>,
}

#[derive(Debug)]
#[cfg(feature = "diskann-builtins")]
pub(super) struct BatchInsert<R, M: Id, T>(Arc<BatchInsertInner<R, M, T>>);

#[cfg(feature = "diskann-builtins")]
impl<R, M, T> BatchInsert<R, M, T>
where
    M: Id,
{
    pub(super) fn new(
        index: Arc<Index<R, M>>,
        data: rowmajor::Owned<T>,
        ids: &[u32],
    ) -> ANNResult<Self> {
        // Convert IDs to positions in `data`.
        let ids: HashMap<_, _> = ids.iter().enumerate().map(|(pos, i)| (*i, pos)).collect();
        if ids.len() != data.nrows() {
            return Err(diskann::error::ANNError::message(
                "mismatch in number of ids",
            ));
        }

        Ok(Self(Arc::new(BatchInsertInner { index, data, ids })))
    }
}

#[cfg(feature = "diskann-builtins")]
impl<R, M: Id, T> Clone for BatchInsert<R, M, T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(feature = "diskann-builtins")]
impl<R, M, T> diskann::HasId for BatchInsert<R, M, T>
where
    M: Id,
{
    type Id = u32;
}

#[cfg(feature = "diskann-builtins")]
impl<R, M, T> diskann::graph::algorithms::batch::insert::Standard for BatchInsert<R, M, T>
where
    for<'a> R: repr::Insert<Vector<'a> = &'a [T]>,
    M: Id,
    T: std::fmt::Debug + Send + Sync + 'static,
{
    type InsertAccessor<'a> = Insert<'a>;
    type PruneAccessor<'a> = Prune<'a>;

    fn insert_accessor(
        &mut self,
        id: u32,
    ) -> ANNResult<impl std::borrow::BorrowMut<Self::InsertAccessor<'_>> + Send> {
        self.0
            .index
            .write()?
            .insert_access(self.0.data.row(self.0.ids[&id]))
    }

    fn prune_accessor(
        &mut self,
    ) -> ANNResult<impl std::borrow::BorrowMut<Self::PruneAccessor<'_>> + Send> {
        self.0.index.write()?.prune_accessor()
    }
}
