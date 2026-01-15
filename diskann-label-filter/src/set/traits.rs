/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::borrow::Cow;
use std::hash::Hash;

use diskann::ANNResult;
use diskann_utils::future::AsyncFriendly;

/// A simple abstraction over a mathematical set, parameterized by the element type.
///
/// This trait lets you plug in different concrete set backends, for example a roaring bitmap
/// for integer sets, without changing call sites. Each concrete `Set` only accepts its own
/// element type.
///
/// Requirements:
/// - `Clone` and `Default` so callers can duplicate and construct sets easily
/// - `IntoIterator<Item = Element>` so callers can iterate elements directly
pub trait Set<Element>: Clone + Default + IntoIterator<Item = Element> {
    /// Create an empty set, equivalent to `Default::default`.
    fn empty_set() -> Self;

    /// Return the intersection of `self` and `other`.
    fn intersection(&self, other: &Self) -> Self;

    /// Return the union of `self` and `other`.
    fn union(&self, other: &Self) -> Self;

    /// Insert a value, return false if the element already exists
    /// and true otherwise.
    /// Return ANNError if there was an error in the operation.
    fn insert(&mut self, value: &Element) -> ANNResult<bool>;

    /// Remove a value, return true if the value was present,
    /// false otherwise.
    /// Return ANNError if there was an underlying error.
    fn remove(&mut self, value: &Element) -> ANNResult<bool>;

    /// Return true if `value` is a member of the set.
    /// Return ANNError if the operation failed.
    fn contains(&self, value: &Element) -> ANNResult<bool>;

    /// Remove all elements from the set.
    /// Return ANNError if the operation failed.
    fn clear(&mut self) -> ANNResult<()>;

    /// Return the number of elements in the set.
    /// Return ANNError if the operation failed.
    fn len(&self) -> ANNResult<usize>;

    /// Return true if the set is empty.
    /// Return ANNError if the operation failed.
    fn is_empty(&self) -> ANNResult<bool>;
}

/// Provider for sets that may live in memory or in a storage layer.
///
/// A `SetProvider` owns or manages many sets, addressed by a key,
/// for example vector identifiers to label sets. Each implementation
/// chooses a concrete set type that fits its backend and usage.
///
/// Type parameters:
/// - `Key` is the lookup key for an individual set, for example a vector id
/// - `Value` is the element type stored inside each set
pub trait SetProvider<Key, Value>: AsyncFriendly
where
    Key: Clone + Eq + Hash,
    Value: Clone + Eq + Hash,
{
    /// Concrete set type managed by this provider.
    type S: Set<Value> + AsyncFriendly;

    /// Get the set for `key`.
    ///
    /// Implementations may borrow internally and return `Cow::Borrowed`,
    /// or materialize an owned set and return `Cow::Owned`.
    /// Returns:
    ///     Ok(Some(Set)) of the key if the key exists
    ///     Ok(None) if the key doesn't exist
    ///     ANNError if there was an error, say in the underlying store.
    fn get(&'_ self, key: &Key) -> ANNResult<Option<Cow<'_, Self::S>>>;

    /// Number of keys managed by this provider.
    /// Returns ANNError if there was an underlying error while retrieving
    /// the result (for instance, in the backing store)
    fn count(&self) -> ANNResult<usize>;

    /// Check if a key exists.
    /// Returns true if the key exists, false otherwise,
    /// and ANNError if an error was encountered while
    /// processing the operation.
    fn exists(&self, key: &Key) -> ANNResult<bool>;

    /// Insert `value` into the set for `key`.
    ///
    /// If the key is missing, create an empty set first, then insert.
    /// Returns true if the value was inserted, false if the *value*
    /// already exists, has nothing to do with the key. This is because
    /// we always expect multiple insert calls with the same key, so it
    /// is not important whether the key exists before insert or not.
    ///
    /// Returns ANNError if there was an underlying error in the operation.
    fn insert(&mut self, key: &Key, value: &Value) -> ANNResult<bool>;

    /// Insert values for a key
    /// If the key is missing, it will create a new entry.
    /// Returns true if the all the values were inserted, false
    /// if any of them already exist.
    ///
    /// Returns ANNError if there is an underlying failure.
    /// Currently, there is no way to indicate the insertion status
    /// of an individual value.
    fn insert_values(&mut self, key: &Key, values: &[Value]) -> ANNResult<bool>;

    /// Delete the entire set for `key`.
    ///
    /// Returns true if the key is present. false, otherwise, and ANNError
    /// if an error was encountered during the operation.
    fn delete(&mut self, key: &Key) -> ANNResult<bool>;

    /// Remove `value` from the set for `key`.
    ///
    /// Returns true if the value was present and removed. false otherwise.
    /// ANNError is returned if the operation fails.
    fn delete_from_set(&mut self, key: &Key, value: &Value) -> ANNResult<bool>;

    ///Clear all the keys and values in the set.
    ///
    /// Returns error if the operation failedd
    fn clear(&mut self) -> ANNResult<()>;
}
