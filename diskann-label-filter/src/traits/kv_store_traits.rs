/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::result::Result;

/// A trait for key-value storage operations.
///
/// This trait defines the interface for a key-value store that supports
/// basic CRUD operations, range queries, and batch operations. Implementations
/// should be thread-safe (Send + Sync).
pub trait KvStore: Send + Sync {
    /// The error type returned by operations on this store.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Retrieves the value associated with a key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Some(value))` if the key exists, `Ok(None)` if the key doesn't exist,
    /// or an error if the operation fails.
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error>;

    /// Sets the value for a key, creating or overwriting it.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to set.
    /// * `value` - The value to associate with the key.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the operation fails.
    fn set(&self, key: &[u8], value: &[u8]) -> Result<(), Self::Error>;

    /// Deletes a key and its associated value.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success (even if the key didn't exist), or an error if the operation fails.
    fn del(&self, key: &[u8]) -> Result<(), Self::Error>;

    /// Returns an iterator over key-value pairs within a specified range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of keys to iterate over (supports standard Rust range types).
    ///
    /// # Returns
    ///
    /// Returns an iterator over `(key, value)` pairs, or an error if the operation fails.
    fn range<R>(&self, range: R) -> Result<KvIterator<'_, Self::Error>, Self::Error>
    where
        R: Into<KeyRange>;

    /// Retrieves multiple values for a batch of keys.
    ///
    /// # Arguments
    ///
    /// * `keys` - A slice of keys to look up.
    ///
    /// # Returns
    ///
    /// Returns a vector where each element is `Some(value)` if the corresponding key exists,
    /// or `None` if it doesn't, or an error if the operation fails.
    fn batch_get(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>, Self::Error>;

    /// Sets multiple key-value pairs in a batch.
    ///
    /// # Arguments
    ///
    /// * `entries` - A slice of (key, value) pairs to set.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the operation fails.
    fn batch_set(&self, entries: &[(&[u8], &[u8])]) -> Result<(), Self::Error>;

    /// Deletes multiple keys in a batch.
    ///
    /// # Arguments
    ///
    /// * `keys` - A slice of keys to delete.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the operation fails.
    fn batch_del(&self, keys: &[&[u8]]) -> Result<(), Self::Error>;
}

/// Represents a range of keys for iteration.
///
/// This struct defines the start and end bounds for key ranges in queries.
#[derive(Debug, Clone)]
pub struct KeyRange {
    /// The starting bound of the range.
    pub start: RangeBound,
    /// The ending bound of the range.
    pub end: RangeBound,
}

/// Represents the bound of a key range (unbounded, included, or excluded).
#[derive(Debug, Clone)]
pub enum RangeBound {
    /// No bound (extends to infinity in this direction).
    Unbounded,
    /// The bound is included in the range.
    Included(Vec<u8>),
    /// The bound is excluded from the range.
    Excluded(Vec<u8>),
}

impl KeyRange {
    /// Creates a full range with no bounds (all keys).
    ///
    /// # Returns
    ///
    /// A `KeyRange` that includes all possible keys.
    pub fn full() -> Self {
        KeyRange {
            start: RangeBound::Unbounded,
            end: RangeBound::Unbounded,
        }
    }

    /// Checks if a key is contained within this range.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to check.
    ///
    /// # Returns
    ///
    /// `true` if the key is within the range, `false` otherwise.
    pub fn contains(&self, key: &[u8]) -> bool {
        let start_ok = match &self.start {
            RangeBound::Unbounded => true,
            RangeBound::Included(s) => key >= s.as_slice(),
            RangeBound::Excluded(s) => key > s.as_slice(),
        };

        let end_ok = match &self.end {
            RangeBound::Unbounded => true,
            RangeBound::Included(e) => key <= e.as_slice(),
            RangeBound::Excluded(e) => key < e.as_slice(),
        };

        start_ok && end_ok
    }
}

impl<T: AsRef<[u8]> + ?Sized> From<std::ops::Range<&T>> for KeyRange {
    fn from(r: std::ops::Range<&T>) -> Self {
        KeyRange {
            start: RangeBound::Included(r.start.as_ref().to_vec()),
            end: RangeBound::Excluded(r.end.as_ref().to_vec()),
        }
    }
}

impl<T: AsRef<[u8]> + ?Sized> From<std::ops::RangeInclusive<&T>> for KeyRange {
    fn from(r: std::ops::RangeInclusive<&T>) -> Self {
        KeyRange {
            start: RangeBound::Included(r.start().as_ref().to_vec()),
            end: RangeBound::Included(r.end().as_ref().to_vec()),
        }
    }
}

impl<T: AsRef<[u8]> + ?Sized> From<std::ops::RangeFrom<&T>> for KeyRange {
    fn from(r: std::ops::RangeFrom<&T>) -> Self {
        KeyRange {
            start: RangeBound::Included(r.start.as_ref().to_vec()),
            end: RangeBound::Unbounded,
        }
    }
}

impl<T: AsRef<[u8]> + ?Sized> From<std::ops::RangeTo<&T>> for KeyRange {
    fn from(r: std::ops::RangeTo<&T>) -> Self {
        KeyRange {
            start: RangeBound::Unbounded,
            end: RangeBound::Excluded(r.end.as_ref().to_vec()),
        }
    }
}

impl<T: AsRef<[u8]> + ?Sized> From<std::ops::RangeToInclusive<&T>> for KeyRange {
    fn from(r: std::ops::RangeToInclusive<&T>) -> Self {
        KeyRange {
            start: RangeBound::Unbounded,
            end: RangeBound::Included(r.end.as_ref().to_vec()),
        }
    }
}

impl From<std::ops::RangeFull> for KeyRange {
    fn from(_: std::ops::RangeFull) -> Self {
        KeyRange::full()
    }
}

/// Type alias for an iterator over key-value pairs.
///
/// This iterator yields `Result<(key, value), Error>` items and is Send-safe.
pub type KvIterator<'a, E> = Box<dyn Iterator<Item = Result<(Vec<u8>, Vec<u8>), E>> + Send + 'a>;

#[cfg(test)]
mod implementations {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::{Arc, RwLock};

    pub struct MemoryStore {
        data: Arc<RwLock<BTreeMap<Vec<u8>, Vec<u8>>>>,
    }
    #[derive(Debug)]
    pub struct KvStoreError(pub String);

    impl std::fmt::Display for KvStoreError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "KV store error: {}", self.0)
        }
    }

    impl std::error::Error for KvStoreError {}

    impl From<String> for KvStoreError {
        fn from(s: String) -> Self {
            KvStoreError(s)
        }
    }

    impl From<&str> for KvStoreError {
        fn from(s: &str) -> Self {
            KvStoreError(s.to_string())
        }
    }
    impl MemoryStore {
        pub fn new() -> Self {
            Self {
                data: Arc::new(RwLock::new(BTreeMap::new())),
            }
        }
    }

    impl KvStore for MemoryStore {
        type Error = KvStoreError;

        fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
            Ok(self.data.read().unwrap().get(key).cloned())
        }

        fn set(&self, key: &[u8], value: &[u8]) -> Result<(), Self::Error> {
            self.data
                .write()
                .unwrap()
                .insert(key.to_vec(), value.to_vec());
            Ok(())
        }

        fn del(&self, key: &[u8]) -> Result<(), Self::Error> {
            self.data.write().unwrap().remove(key);
            Ok(())
        }

        fn range<R>(&self, range: R) -> Result<KvIterator<'_, Self::Error>, Self::Error>
        where
            R: Into<KeyRange>,
        {
            let key_range = range.into();
            let data = self.data.read().unwrap();

            let items: Vec<_> = data
                .iter()
                .filter(|(k, _)| key_range.contains(k))
                .map(|(k, v)| Ok((k.clone(), v.clone())))
                .collect();

            Ok(Box::new(items.into_iter()))
        }

        fn batch_set(&self, entries: &[(&[u8], &[u8])]) -> Result<(), Self::Error> {
            let mut data = self.data.write().unwrap();
            for (k, v) in entries {
                data.insert(k.to_vec(), v.to_vec());
            }
            Ok(())
        }

        fn batch_get(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>, Self::Error> {
            let data = self.data.read().unwrap();
            let results = keys.iter().map(|k| data.get(*k).cloned()).collect();
            Ok(results)
        }

        fn batch_del(&self, keys: &[&[u8]]) -> Result<(), Self::Error> {
            let mut data = self.data.write().unwrap();
            for k in keys {
                data.remove(*k);
            }
            Ok(())
        }
    }

    #[test]
    fn test_basic_operations() {
        let store = MemoryStore::new();

        store.set(b"key1", b"value1").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"value1".to_vec()));

        store.set(b"key1", b"updated").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"updated".to_vec()));

        store.del(b"key1").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), None);

        assert_eq!(store.get(b"missing").unwrap(), None);
    }

    #[test]
    fn test_range_scan_exclusive() {
        let store = MemoryStore::new();

        store.set(b"a", b"1").unwrap();
        store.set(b"b", b"2").unwrap();
        store.set(b"c", b"3").unwrap();
        store.set(b"d", b"4").unwrap();
        store.set(b"e", b"5").unwrap();

        let results: Vec<_> = store
            .range(b"b"..b"d")
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, b"b");
        assert_eq!(results[0].1, b"2");
        assert_eq!(results[1].0, b"c");
        assert_eq!(results[1].1, b"3");
    }

    #[test]
    fn test_range_scan_inclusive() {
        let store = MemoryStore::new();

        store.set(b"a", b"1").unwrap();
        store.set(b"b", b"2").unwrap();
        store.set(b"c", b"3").unwrap();
        store.set(b"d", b"4").unwrap();

        let results: Vec<_> = store
            .range(b"b"..=b"d")
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, b"b");
        assert_eq!(results[1].0, b"c");
        assert_eq!(results[2].0, b"d");
    }

    #[test]
    fn test_range_from_start() {
        let store = MemoryStore::new();

        store.set(b"a", b"1").unwrap();
        store.set(b"b", b"2").unwrap();
        store.set(b"c", b"3").unwrap();

        let results: Vec<_> = store
            .range(..b"c")
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 2); // a, b only
        assert_eq!(results[0].0, b"a");
        assert_eq!(results[1].0, b"b");
    }

    #[test]
    fn test_range_to_end() {
        let store = MemoryStore::new();

        store.set(b"a", b"1").unwrap();
        store.set(b"b", b"2").unwrap();
        store.set(b"c", b"3").unwrap();
        store.set(b"d", b"4").unwrap();

        let results: Vec<_> = store
            .range(b"c"..)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 2); // c, d
        assert_eq!(results[0].0, b"c");
        assert_eq!(results[1].0, b"d");
    }

    #[test]
    fn test_full_scan() {
        let store = MemoryStore::new();

        store.set(b"a", b"1").unwrap();
        store.set(b"b", b"2").unwrap();
        store.set(b"c", b"3").unwrap();

        let results: Vec<_> = store
            .range(..)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_batch_operations() {
        let store = MemoryStore::new();

        let entries = vec![
            (b"k1".as_slice(), b"v1".as_slice()),
            (b"k2".as_slice(), b"v2".as_slice()),
            (b"k3".as_slice(), b"v3".as_slice()),
        ];
        store.batch_set(&entries).unwrap();

        assert_eq!(store.get(b"k1").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(store.get(b"k2").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(store.get(b"k3").unwrap(), Some(b"v3".to_vec()));

        let results = store.batch_get(&[b"k1", b"k2", b"k3", b"k4"]).unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], Some(b"v1".to_vec()));
        assert_eq!(results[1], Some(b"v2".to_vec()));
        assert_eq!(results[2], Some(b"v3".to_vec()));
        assert_eq!(results[3], None);

        store.batch_del(&[b"k1", b"k2"]).unwrap();
        assert_eq!(store.get(b"k1").unwrap(), None);
        assert_eq!(store.get(b"k2").unwrap(), None);
        assert_eq!(store.get(b"k3").unwrap(), Some(b"v3".to_vec()));
    }

    #[test]
    fn test_empty_range() {
        let store = MemoryStore::new();

        store.set(b"a", b"1").unwrap();
        store.set(b"c", b"3").unwrap();

        let results: Vec<_> = store
            .range(b"x"..b"z")
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_idempotent_delete() {
        let store = MemoryStore::new();

        store.set(b"key", b"value").unwrap();

        store.del(b"key").unwrap();
        assert_eq!(store.get(b"key").unwrap(), None);

        store.del(b"key").unwrap();
        assert_eq!(store.get(b"key").unwrap(), None);
    }
}
