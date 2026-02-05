/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::any::Any;

/// A streaming interface for performing dynamic (streaming) operations on an index.
///
/// Streams are characterized by five operations:
///
/// * `search`: This is, after all, the whole reason for building an index.
/// * `insert`: Insert new points into the index that do not already exist.
/// * `replace`: Replace existing points in the index with new data.
/// * `delete`: Remove points from the index.
/// * `maintain`: Perform maintenance operations on the index. Examples may
///   include fully removing deleted points from internal references.
///
/// This trait is parameterized by an [`Arguments`] proxy trait, which defines the
/// argument types for each of the operations. The motivation here is to allow nesting
/// of [`Stream`] implementations that progressively modify or adapt the arguments
/// for better code reuse. An example of this is
/// [`crate::streaming::executors::bigann::WithData`], which is a stream layer adapting
/// the raw ranges used by [`crate::streaming::executors::bigann::RunBook`] into
/// actual data slices.
///
/// Runners for [`Stream`]s use the [`Executor`] trait to invoke the stream operations
/// in a structured way.
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
///
/// Implementations invoke the operations of a [`Stream`] in a structured way (which should
/// be reflected in the associated documentation) and aggregate the results.
pub trait Executor {
    /// The argument collection type for the underlying [`Stream`].
    type Args: Arguments;

    /// Execute a series of operations on `stream`. As outputs are produced, they will be
    /// passed to `collect` for aggregation.
    ///
    /// Since dynamic execution may be long-running, this allows implementations of `collect`
    /// to perform operations like status updates or partial saving of results as they are
    /// generated.
    ///
    /// See also: [`Executor::run`].
    fn run_with<S, F, O>(&mut self, stream: &mut S, collect: F) -> anyhow::Result<()>
    where
        S: Stream<Self::Args, Output = O>,
        O: 'static,
        F: FnMut(O) -> anyhow::Result<()>;

    /// Execute a series of operations on `stream`. The outputs of each operation will be
    /// collected in the retuned `Vec` in-order.
    ///
    /// See also: [`Executor::run_with`].
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

/// A type-erased [`Stream`] implementation that wraps stream (outputs)[`Stream::Output`] in
/// [`Box<dyn Any>`].
///
/// This is useful as the final layer in a stack of nested [`Stream`]s that allows the top
/// level [`Executor`] to operate without knowledge of the concrete output types.
///
/// From a performance perspective, this is usually fine since
///
/// 1. Individual [`Stream`] operations are typically expensive relative to the cost of boxing.
/// 2. Many concrete [`Stream`] implementations will use the same [`Executor`] implementation.
///    Thus, boxing can help reduce code bloat without significant performance impact.
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    // The main thing we're testing here is the implementation of [`Executor::run`] and
    // to a lesser extent `AnyStream`.

    struct Search;
    struct Insert;
    struct Replace;
    struct Delete;
    struct Maintain;

    #[derive(Debug, PartialEq)]
    enum Op {
        Search,
        Insert,
        Replace,
        Delete,
        Maintain,
    }

    struct TestArgs;

    impl Arguments for TestArgs {
        type Search<'a> = Search;
        type Insert<'a> = Insert;
        type Replace<'a> = Replace;
        type Delete<'a> = Delete;
        type Maintain<'a> = Maintain;
    }

    struct TestStream {
        needs_maintenance: bool,
    }

    impl TestStream {
        fn new(needs_maintenance: bool) -> Self {
            Self { needs_maintenance }
        }
    }

    impl Stream<TestArgs> for TestStream {
        type Output = Op;

        fn search(&mut self, _args: Search) -> anyhow::Result<Self::Output> {
            Ok(Op::Search)
        }

        fn insert(&mut self, _args: Insert) -> anyhow::Result<Self::Output> {
            Ok(Op::Insert)
        }

        fn replace(&mut self, _args: Replace) -> anyhow::Result<Self::Output> {
            Ok(Op::Replace)
        }

        fn delete(&mut self, _args: Delete) -> anyhow::Result<Self::Output> {
            Ok(Op::Delete)
        }

        fn maintain(&mut self, _args: Maintain) -> anyhow::Result<Self::Output> {
            Ok(Op::Maintain)
        }

        fn needs_maintenance(&mut self) -> bool {
            self.needs_maintenance
        }
    }

    struct TestExecutor;

    impl Executor for TestExecutor {
        type Args = TestArgs;

        fn run_with<S, F, O>(&mut self, stream: &mut S, mut collect: F) -> anyhow::Result<()>
        where
            S: Stream<Self::Args, Output = O>,
            O: 'static,
            F: FnMut(O) -> anyhow::Result<()>,
        {
            collect(stream.search(Search)?)?;
            collect(stream.insert(Insert)?)?;
            collect(stream.replace(Replace)?)?;
            collect(stream.delete(Delete)?)?;
            collect(stream.maintain(Maintain)?)?;
            Ok(())
        }
    }

    #[test]
    fn test_executor_run() -> anyhow::Result<()> {
        let mut stream = TestStream::new(false);
        let mut executor = TestExecutor;

        let outputs = executor.run(&mut stream)?;

        assert_eq!(outputs.len(), 5);
        assert!(matches!(outputs[0], Op::Search));
        assert!(matches!(outputs[1], Op::Insert));
        assert!(matches!(outputs[2], Op::Replace));
        assert!(matches!(outputs[3], Op::Delete));
        assert!(matches!(outputs[4], Op::Maintain));

        Ok(())
    }

    #[test]
    fn test_any_stream() {
        let mut stream = TestStream::new(false);
        let mut any_stream = AnyStream::new(&mut stream);

        assert!(
            !any_stream.needs_maintenance(),
            "AnyStream should forward `needs_maintenance`"
        );
        assert_eq!(
            any_stream
                .search(Search)
                .unwrap()
                .downcast_ref::<Op>()
                .unwrap(),
            &Op::Search
        );
        assert_eq!(
            any_stream
                .insert(Insert)
                .unwrap()
                .downcast_ref::<Op>()
                .unwrap(),
            &Op::Insert
        );
        assert_eq!(
            any_stream
                .replace(Replace)
                .unwrap()
                .downcast_ref::<Op>()
                .unwrap(),
            &Op::Replace
        );
        assert_eq!(
            any_stream
                .delete(Delete)
                .unwrap()
                .downcast_ref::<Op>()
                .unwrap(),
            &Op::Delete
        );
        assert_eq!(
            any_stream
                .maintain(Maintain)
                .unwrap()
                .downcast_ref::<Op>()
                .unwrap(),
            &Op::Maintain
        );

        let mut stream = TestStream::new(true);
        let mut any_stream = AnyStream::new(&mut stream);
        assert!(
            any_stream.needs_maintenance(),
            "AnyStream should forward `needs_maintenance`"
        );
    }
}
