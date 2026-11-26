/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::borrow::BorrowMut;

use diskann::{ANNResult, graph::AdjacencyList};

use crate::index::accessors::{Prune, Search};

//--------//
// Insert //
//--------//

#[derive(Debug)]
pub(super) struct Insert<'a> {
    search: Search<'a>,
    prune: Prune<'a>,
}

impl<'a> Insert<'a> {
    pub(super) fn new(search: Search<'a>, prune: Prune<'a>) -> Self {
        Self { search, prune }
    }
}

impl diskann::HasId for Insert<'_> {
    type Id = u32;
}

impl<'a> diskann::graph::algorithms::insert::Access for Insert<'a> {
    type SearchAccessor<'x>
        = Search<'a>
    where
        Self: 'x;

    type PruneAccessor<'x>
        = Prune<'a>
    where
        Self: 'x;

    fn search_accessor(&mut self) -> impl BorrowMut<Search<'a>> + Send {
        &mut self.search
    }

    fn prune_accessor(&mut self) -> impl BorrowMut<Prune<'a>> + Send {
        &mut self.prune
    }
}

//----------------//
// Inplace Delete //
//----------------//

#[derive(Debug)]
pub(super) struct InplaceDelete<'a> {
    prune: Prune<'a>,
}

impl<'a> InplaceDelete<'a> {
    pub(super) fn new(prune: Prune<'a>) -> Self {
        Self { prune }
    }
}

impl diskann::HasId for InplaceDelete<'_> {
    type Id = u32;
}

impl<'a> diskann::graph::algorithms::inplace_delete::Access for InplaceDelete<'a> {
    type PruneAccessor<'x>
        = Prune<'a>
    where
        Self: 'x;

    fn prune_accessor(&mut self) -> impl std::borrow::BorrowMut<Prune<'a>> + Send {
        &mut self.prune
    }

    async fn keep_live(&mut self, ids: &mut AdjacencyList<u32>) -> ANNResult<()> {
        self.prune.partition_live(ids, None);
        Ok(())
    }
}
