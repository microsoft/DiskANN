/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, sync::Arc};

use diskann::ANNError;
use thiserror::Error;

/// A dynamic strategy (e.g. `diskann::graph::glue::SearchStrategy`) manager for built-in
/// searcher such as [`super::KNN`], [`super::Range`], and [`super::MultiHop`].
///
/// This provides an efficient means for either broadcasting a single strategy to all
/// search queries, or maintaining a collection of strategies, one for each query.
#[derive(Debug)]
pub enum Strategy<S> {
    /// Use the same strategy for all queries.
    Broadcast(S),
    /// Use a custom strategy for each query.
    Collection(Box<[S]>),
    /// Use a custom strategy for each query via an [`Indexable`] trait object.
    Indexable(Box<dyn Indexable<S> + Send + Sync>),
}

impl<S> Strategy<S> {
    /// Create a strategy that broadcasts `strategy` to all queries.
    pub fn broadcast(strategy: S) -> Self {
        Self::Broadcast(strategy)
    }

    /// Create a strategy that uses the strategies in `itr` for each query.
    pub fn collection<I>(itr: I) -> Self
    where
        I: IntoIterator<Item = S>,
    {
        Self::Collection(itr.into_iter().collect())
    }

    /// Create a strategy that uses `indexable` for each query's strategy.
    ///
    /// This method is most useful when the strategies are stored in a custom data
    /// structure and can avoid the cost of rematerializing a collection.
    pub fn from_indexable<I>(indexable: I) -> Self
    where
        S: std::fmt::Debug,
        I: Indexable<S> + Send + Sync + 'static,
    {
        Self::Indexable(Box::new(indexable))
    }

    /// Get the strategy for the query at `index`.
    pub fn get(&self, index: usize) -> Result<&S, Error> {
        match self {
            Self::Broadcast(s) => Ok(s),
            Self::Collection(strategies) => get_as_slice(strategies, index),
            Self::Indexable(indexable) => indexable.get(index),
        }
    }

    /// Return the number of strategies contained in `self`, or `None` if there are
    /// an unbounded number of strategies.
    ///
    /// ```rust
    /// use diskann_benchmark_core::search::graph::Strategy;
    ///
    /// let strategy = Strategy::broadcast(42usize);
    /// assert_eq!(*strategy.get(0).unwrap(), 42);
    /// assert!(
    ///     strategy.len().is_none(),
    ///     "broadcasted strategies can be retrieved from any index",
    /// );
    ///
    /// let strategy = Strategy::collection([42usize, 128usize]);
    /// assert_eq!(*strategy.get(0).unwrap(), 42);
    /// assert_eq!(*strategy.get(1).unwrap(), 128);
    /// assert_eq!(strategy.len(), Some(2));
    /// ```
    pub fn len(&self) -> Option<usize> {
        match self {
            Self::Broadcast(_) => None,
            Self::Collection(strategies) => Some(strategies.len()),
            Self::Indexable(indexable) => Some(indexable.len()),
        }
    }

    /// Return `true` only if the number of strategies is bounded and equal to zero.
    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Check if the number of strategies in `self` is compatible with `expected`.
    ///
    /// [`Self::Broadcast`] is always compatible. Otherwise, the number of strategies must
    /// exactly match `expected`.
    pub fn length_compatible(&self, expected: usize) -> Result<(), LengthIncompatible> {
        if let Some(len) = self.len()
            && len != expected
        {
            Err(LengthIncompatible {
                strategies: len,
                expected,
            })
        } else {
            Ok(())
        }
    }
}

/// A helper trait for [`Strategy`] that allows custom collections of strategies.
pub trait Indexable<S>: std::fmt::Debug {
    /// Return the number of strategies in the collection.
    ///
    /// Implementations should ensure that `get(i)` returns `Ok(s)` for all `i < Self::len()`.
    fn len(&self) -> usize;

    /// Return the strategy at `index`.
    fn get(&self, index: usize) -> Result<&S, Error>;

    /// Return `true` if the collection is empty. Otherwise, return `false`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn get_as_slice<T>(x: &[T], index: usize) -> Result<&T, Error> {
    x.get(index).ok_or_else(|| Error::new(index, x.len()))
}

impl<S> Indexable<S> for Arc<[S]>
where
    S: std::fmt::Debug,
{
    fn len(&self) -> usize {
        <[S]>::len(self)
    }

    fn get(&self, index: usize) -> Result<&S, Error> {
        get_as_slice(self, index)
    }
}

impl<S> Indexable<S> for Box<[S]>
where
    S: std::fmt::Debug,
{
    fn len(&self) -> usize {
        <[S]>::len(self)
    }

    fn get(&self, index: usize) -> Result<&S, Error> {
        get_as_slice(self, index)
    }
}

/// An error indicating that an attempt was made to index a strategy collection
/// at an out-of-bounds index.
#[derive(Debug, Clone, Copy, Error)]
#[error("Tried to index a strategy collection of length {} at index {}", self.len, self.index)]
pub struct Error {
    index: usize,
    len: usize,
}

impl Error {
    fn new(index: usize, len: usize) -> Self {
        Self { index, len }
    }
}

impl From<Error> for ANNError {
    #[track_caller]
    fn from(error: Error) -> ANNError {
        ANNError::opaque(error)
    }
}

/// Error for an incorrect number of strategies.
///
/// See: [`Strategy::length_compatible`].
#[derive(Debug, Clone)]
pub struct LengthIncompatible {
    strategies: usize,
    expected: usize,
}

impl std::fmt::Display for LengthIncompatible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Plural {
            value: usize,
            singular: &'static str,
            plural: &'static str,
        }

        impl std::fmt::Display for Plural {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if self.value == 1 {
                    write!(f, "{} {}", self.value, self.singular)
                } else {
                    write!(f, "{} {}", self.value, self.plural)
                }
            }
        }

        let strategies = Plural {
            value: self.strategies,
            singular: "strategy was",
            plural: "strategies were",
        };

        let expected = Plural {
            value: self.expected,
            singular: "was expected",
            plural: "were expected",
        };

        write!(f, "{strategies} provided when {expected}")
    }
}

impl std::error::Error for LengthIncompatible {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test strategy type
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestStrategy(u32);

    // Custom indexable implementation for testing
    #[derive(Debug)]
    struct CustomIndexable {
        strategies: Vec<TestStrategy>,
    }

    impl Indexable<TestStrategy> for CustomIndexable {
        fn len(&self) -> usize {
            self.strategies.len()
        }

        fn get(&self, index: usize) -> Result<&TestStrategy, Error> {
            get_as_slice(&self.strategies, index)
        }
    }

    #[test]
    fn test_strategy_broadcast() {
        let strategy = TestStrategy(42);
        let broadcast = Strategy::broadcast(strategy.clone());

        match &broadcast {
            Strategy::Broadcast(s) => assert_eq!(*s, strategy),
            _ => panic!("Expected Broadcast variant"),
        }

        for i in 0..10 {
            assert_eq!(broadcast.get(i).unwrap(), &strategy);
        }
    }

    #[test]
    fn test_strategy_collection() {
        let strategies = [TestStrategy(1), TestStrategy(2), TestStrategy(3)];
        let collection = Strategy::collection(strategies.clone());

        match &collection {
            Strategy::Collection(s) => {
                assert_eq!(s.len(), 3);
                assert_eq!(s[0], strategies[0]);
                assert_eq!(s[1], strategies[1]);
                assert_eq!(s[2], strategies[2]);
            }
            _ => panic!("Expected Collection variant"),
        }

        assert_eq!(collection.get(0).unwrap(), &TestStrategy(1));
        assert_eq!(collection.get(1).unwrap(), &TestStrategy(2));
        assert_eq!(collection.get(2).unwrap(), &TestStrategy(3));

        let err = collection.get(3).unwrap_err();
        assert_eq!(err.index, 3);
        assert_eq!(err.len, 3);
    }

    #[test]
    fn test_strategy_collection_empty() {
        let collection = Strategy::<TestStrategy>::collection(vec![]);

        let result = collection.get(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_indexable() {
        let custom = CustomIndexable {
            strategies: vec![TestStrategy(100), TestStrategy(200)],
        };

        let strategy = Strategy::from_indexable(custom);

        match strategy {
            Strategy::Indexable(_) => {
                assert_eq!(strategy.get(0).unwrap(), &TestStrategy(100));
                assert_eq!(strategy.get(1).unwrap(), &TestStrategy(200));
            }
            _ => panic!("Expected Indexable variant"),
        }

        assert_eq!(strategy.get(0).unwrap(), &TestStrategy(100));
        assert_eq!(strategy.get(1).unwrap(), &TestStrategy(200));
        let err = strategy.get(5).unwrap_err();
        assert_eq!(err.index, 5);
        assert_eq!(err.len, 2);
    }

    #[test]
    fn test_indexable_arc_slice() {
        let strategies: Arc<[TestStrategy]> =
            Arc::from(vec![TestStrategy(1), TestStrategy(2), TestStrategy(3)]);

        assert_eq!(strategies.len(), 3);
        assert!(!strategies.is_empty());

        assert_eq!(strategies.get(0).unwrap(), &TestStrategy(1));
        assert_eq!(strategies.get(1).unwrap(), &TestStrategy(2));
        assert_eq!(strategies.get(2).unwrap(), &TestStrategy(3));

        assert!(strategies.get(10).is_err());
    }

    #[test]
    fn test_indexable_box_slice() {
        let strategies: Box<[TestStrategy]> =
            vec![TestStrategy(5), TestStrategy(10)].into_boxed_slice();

        assert_eq!(strategies.len(), 2);
        assert!(!strategies.is_empty());

        assert_eq!(strategies.get(0).unwrap(), &TestStrategy(5));
        assert_eq!(strategies.get(1).unwrap(), &TestStrategy(10));

        assert!(strategies.get(5).is_err());
    }

    #[test]
    fn test_indexable_is_empty() {
        let empty: Box<[TestStrategy]> = vec![].into_boxed_slice();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let non_empty: Box<[TestStrategy]> = vec![TestStrategy(1)].into_boxed_slice();
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.len(), 1);
    }

    #[test]
    fn test_error_to_ann_error() {
        let error = Error::new(3, 2);
        let ann_error: ANNError = error.into();

        // Verify it converts without panicking
        let message = format!("{:?}", ann_error);
        assert!(!message.is_empty());
    }

    #[test]
    fn test_strategy_len() {
        // Broadcast returns None (unbounded)
        let broadcast = Strategy::broadcast(TestStrategy(1));
        assert_eq!(broadcast.len(), None);
        assert!(!broadcast.is_empty());

        // Collection returns Some(len)
        let collection =
            Strategy::collection(vec![TestStrategy(1), TestStrategy(2), TestStrategy(3)]);
        assert_eq!(collection.len(), Some(3));
        assert!(!collection.is_empty());

        // Empty collection returns Some(0)
        let empty_collection = Strategy::<TestStrategy>::collection(vec![]);
        assert_eq!(empty_collection.len(), Some(0));
        assert!(empty_collection.is_empty());

        // Indexable returns Some(len)
        let custom = CustomIndexable {
            strategies: vec![TestStrategy(1), TestStrategy(2)],
        };
        let indexable = Strategy::from_indexable(custom);
        assert_eq!(indexable.len(), Some(2));
        assert!(!indexable.is_empty());

        // Empty indexable returns Some(0)
        let empty_custom = CustomIndexable { strategies: vec![] };
        let empty_indexable = Strategy::from_indexable(empty_custom);
        assert_eq!(empty_indexable.len(), Some(0));
        assert!(empty_indexable.is_empty());
    }

    #[test]
    fn test_length_compatible_broadcast() {
        // Broadcast is always compatible with any expected length
        let broadcast = Strategy::broadcast(1usize);
        assert!(broadcast.length_compatible(0).is_ok());
        assert!(broadcast.length_compatible(1).is_ok());
        assert!(broadcast.length_compatible(100).is_ok());
        assert!(broadcast.length_compatible(usize::MAX).is_ok());
    }

    #[test]
    fn test_length_compatible_collection() {
        let collection = Strategy::collection([1usize, 2, 3]);
        assert!(collection.length_compatible(3).is_ok());

        // Incompatible when expected doesn't match
        let err = collection.length_compatible(2).unwrap_err();
        assert_eq!(
            err.to_string(),
            "3 strategies were provided when 2 were expected"
        );

        let err = collection.length_compatible(5).unwrap_err();
        assert_eq!(
            err.to_string(),
            "3 strategies were provided when 5 were expected"
        );

        // One Strategy
        let single = Strategy::collection([1usize]);
        assert!(single.length_compatible(1).is_ok());

        let err = single.length_compatible(0).unwrap_err();
        assert_eq!(
            err.to_string(),
            "1 strategy was provided when 0 were expected"
        );

        // Empty collection
        let empty = Strategy::<usize>::collection([]);
        assert!(empty.length_compatible(0).is_ok());

        let err = empty.length_compatible(1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "0 strategies were provided when 1 was expected"
        );
    }

    #[test]
    fn test_length_compatible_indexable() {
        let custom = CustomIndexable {
            strategies: vec![TestStrategy(1), TestStrategy(2)],
        };
        let indexable = Strategy::from_indexable(custom);
        assert!(indexable.length_compatible(2).is_ok());

        // Incompatible when expected doesn't match
        let err = indexable.length_compatible(1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "2 strategies were provided when 1 was expected"
        );

        let err = indexable.length_compatible(10).unwrap_err();
        assert_eq!(
            err.to_string(),
            "2 strategies were provided when 10 were expected"
        );

        // Empty indexable
        let empty_custom = CustomIndexable { strategies: vec![] };
        let empty_indexable = Strategy::from_indexable(empty_custom);
        assert!(empty_indexable.length_compatible(0).is_ok());

        let err = empty_indexable.length_compatible(5).unwrap_err();
        assert_eq!(
            err.to_string(),
            "0 strategies were provided when 5 were expected"
        );
    }
}
