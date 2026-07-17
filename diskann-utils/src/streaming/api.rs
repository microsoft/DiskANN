/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::any::Any;

/// A streaming interface for performing dynamic (streaming) operations on an index.
pub trait Stream<A>
where
    A: Arguments,
{
    /// Output type for all operations. The `'static` is to allow results to be
    /// aggregated in [`Any`] for type erasure in higher level [`Executor`]s.
    type Output: 'static;

    /// Perform a search operation.
    fn search(&mut self, args: A::Search<'_>) -> anyhow::Result<Self::Output>;

    /// Perform an insert operation.
    fn insert(&mut self, args: A::Insert<'_>) -> anyhow::Result<Self::Output>;

    /// Perform a replace operation.
    fn replace(&mut self, args: A::Replace<'_>) -> anyhow::Result<Self::Output>;

    /// Perform a delete operation.
    fn delete(&mut self, args: A::Delete<'_>) -> anyhow::Result<Self::Output>;

    /// Perform a maintain operation.
    fn maintain(&mut self, args: A::Maintain<'_>) -> anyhow::Result<Self::Output>;

    /// Indicate whether or not maintenance is needed. [`Executor`] implementations
    /// are responsible periodically checking this.
    fn needs_maintenance(&mut self) -> bool;
}

/// Operation arguments to [`Stream`].
pub trait Arguments: 'static {
    /// Argument to [`Stream::search`].
    type Search<'a>;
    /// Argument to [`Stream::insert`].
    type Insert<'a>;
    /// Argument to [`Stream::replace`].
    type Replace<'a>;
    /// Argument to [`Stream::delete`].
    type Delete<'a>;
    /// Argument to [`Stream::maintain`].
    type Maintain<'a>;
}

/// A sequential executor for [`Stream`]s.
pub trait Executor {
    /// The argument collection type for the underlying [`Stream`].
    type Args: Arguments;

    /// Execute a series of operations on `stream`. As outputs are produced, they will be
    /// passed to `collect` for aggregation.
    fn run_with<S, F, O>(&mut self, stream: &mut S, collect: F) -> anyhow::Result<()>
    where
        S: Stream<Self::Args, Output = O>,
        O: 'static,
        F: FnMut(O) -> anyhow::Result<()>;

    /// Execute a series of operations on `stream`. The outputs of each operation will be
    /// collected in the returned `Vec` in-order.
    fn run<S>(&mut self, stream: &mut S) -> anyhow::Result<Vec<S::Output>>
    where
        S: Stream<Self::Args>,
    {
        let mut outputs = Vec::new();
        self.run_with(stream, |output| {
            outputs.push(output);
            Ok(())
        })?;
        Ok(outputs)
    }
}

/// A type-erased [`Stream`] implementation that wraps stream outputs in [`Box<dyn Any>`].
#[derive(Debug)]
pub struct AnyStream<'a, T>(&'a mut T);

impl<'a, T> AnyStream<'a, T> {
    /// Wrap `stream` in an [`AnyStream`].
    pub fn new(stream: &'a mut T) -> Self {
        Self(stream)
    }
}

fn boxed<T>(x: T) -> Box<dyn Any>
where
    T: Any,
{
    Box::new(x)
}

impl<A, T> Stream<A> for AnyStream<'_, T>
where
    A: Arguments,
    T: Stream<A>,
{
    type Output = Box<dyn Any>;

    fn search(&mut self, args: A::Search<'_>) -> anyhow::Result<Self::Output> {
        self.0.search(args).map(boxed)
    }

    fn insert(&mut self, args: A::Insert<'_>) -> anyhow::Result<Self::Output> {
        self.0.insert(args).map(boxed)
    }

    fn replace(&mut self, args: A::Replace<'_>) -> anyhow::Result<Self::Output> {
        self.0.replace(args).map(boxed)
    }

    fn delete(&mut self, args: A::Delete<'_>) -> anyhow::Result<Self::Output> {
        self.0.delete(args).map(boxed)
    }

    fn maintain(&mut self, args: A::Maintain<'_>) -> anyhow::Result<Self::Output> {
        self.0.maintain(args).map(boxed)
    }

    fn needs_maintenance(&mut self) -> bool {
        self.0.needs_maintenance()
    }
}
