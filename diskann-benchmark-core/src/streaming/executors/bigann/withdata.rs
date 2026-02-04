/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Range, path::Path, sync::Arc};

use diskann_utils::views::{Matrix, MatrixView};

use crate::{recall, streaming};

use super::{Args, Delete, Insert, Replace, Search};

type LoadGroundtruth<I> = dyn FnMut(&Path) -> anyhow::Result<Box<dyn recall::Rows<I>>>;

/// An adaptor to execute a [`super::RunBook`] out of a set data set and queries.
pub struct WithData<T, I, Inner> {
    inner: Inner,
    dataset: Matrix<T>,
    queries: Arc<Matrix<T>>,
    load_groundtruth: Box<LoadGroundtruth<I>>,
}

impl<T, I, Inner> WithData<T, I, Inner> {
    /// Create a new [`WithData`] adaptor over the `dataset` and `queries`.
    ///
    /// Argument `load_groundtruth` is a callback responsible for loading the groundtruth
    /// for a given path.
    pub fn new(
        inner: Inner,
        dataset: Matrix<T>,
        queries: Arc<Matrix<T>>,
        load_groundtruth: impl FnMut(&Path) -> anyhow::Result<Box<dyn recall::Rows<I>>> + 'static,
    ) -> Self {
        Self {
            inner,
            dataset,
            queries,
            load_groundtruth: Box::new(load_groundtruth),
        }
    }
}

/// An adaptor for [`super::Args`] that provides data and groundtruth instead of the
/// raw ranges from [`super::RunBook`].
#[derive(Debug, Clone, Copy)]
pub struct DataArgs<T, I> {
    _marker: std::marker::PhantomData<(T, I)>,
}

impl<T, I> streaming::Arguments for DataArgs<T, I>
where
    T: 'static,
    I: 'static,
{
    /// A tuple consisting of the queries for search as well as the corresponding
    /// groundtruth (as [`recall::Rows`]).
    type Search<'a> = (Arc<Matrix<T>>, &'a dyn recall::Rows<I>);

    /// A tuple consisting of the data to insert as well as the external IDs (stored
    /// as `usize`) for the data. It is assumed that the length of the IDs range matches
    /// the number of rows in the data matrix.
    type Insert<'a> = (MatrixView<'a, T>, Range<usize>);

    /// The external IDs (stored as `usize`) to delete.
    type Delete<'a> = Range<usize>;

    /// A tuple consisting of the data to replace as well as the external IDs (stored
    /// as `usize`) for the data. It is assumed that the length of the IDs range matches
    /// the number of rows in the data matrix.
    type Replace<'a> = (MatrixView<'a, T>, Range<usize>);
    type Maintain<'a> = ();
}

impl<T, I, Inner> streaming::Stream<Args> for WithData<T, I, Inner>
where
    Inner: streaming::Stream<DataArgs<T, I>>,
    T: 'static,
    I: 'static,
{
    type Output = Inner::Output;

    fn search(&mut self, args: Search<'_>) -> anyhow::Result<Self::Output> {
        let groundtruth = (self.load_groundtruth)(args.groundtruth)?;
        self.inner.search((self.queries.clone(), &*groundtruth))
    }

    fn insert(&mut self, args: Insert) -> anyhow::Result<Self::Output> {
        let data = self.dataset.subview(args.offsets).unwrap();
        self.inner.insert((data, args.ids))
    }

    fn replace(&mut self, args: Replace) -> anyhow::Result<Self::Output> {
        let data = self.dataset.subview(args.offsets).unwrap();
        self.inner.replace((data, args.ids))
    }

    fn delete(&mut self, args: Delete) -> anyhow::Result<Self::Output> {
        self.inner.delete(args.ids)
    }

    fn maintain(&mut self, _args: ()) -> anyhow::Result<Self::Output> {
        self.inner.maintain(())
    }

    fn needs_maintenance(&mut self) -> bool {
        self.inner.needs_maintenance()
    }
}
