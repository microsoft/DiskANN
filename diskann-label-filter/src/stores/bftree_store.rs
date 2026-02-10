/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! BfTree-backed key-value store implementation with improvements based on code review.

use crate::traits::kv_store_traits::{KeyRange, KvIterator, KvStore};
use bf_tree::{BfTree, Config, LeafInsertResult, LeafReadResult};
use std::path::Path;
use std::sync::Arc;

/// A persistent key-value store backed by BfTree.
///
/// This implementation provides:
/// - Persistent storage on disk
/// - Thread-safe concurrent access (BfTree has internal synchronization)
/// - Efficient point queries (get/set/delete)
/// - Configurable page sizes and cache settings
///
/// # Limitations
///
/// - **Range queries**: Not currently implemented. Calling `range()` will return an error.
/// - **Batch atomicity**: Batch operations are NOT atomic. If a batch operation fails
///   partway through, previous operations in the batch will have succeeded.
/// - **Value size**: Large values may cause performance issues.
///
/// # Performance Characteristics
///
/// - **Reads**: O(log n) with page cache
/// - **Writes**: O(log n) with write-ahead logging
/// - **Scans**: Not yet implemented (range queries return error)
/// - **Batch operations**: Sequential
///
/// # Thread Safety
///
/// BfTreeStore is thread-safe via Arc<BfTree>. Multiple readers and writers can
/// safely access the store concurrently. BfTree internally handles synchronization.
///
/// # Note on Batch Operations
///
/// Batch operations (`batch_set`, `batch_del`) are not atomic. Each operation
/// within a batch is executed sequentially. If you need atomic batch operations,
/// consider using a different storage backend that supports transactions.
#[derive(Clone)]
pub struct BfTreeStore {
    tree: Arc<BfTree>,
}

#[derive(Debug)]
pub struct BfTreeStoreError(pub String);

impl std::fmt::Display for BfTreeStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BfTree store error: {}", self.0)
    }
}

impl std::error::Error for BfTreeStoreError {}

impl From<String> for BfTreeStoreError {
    fn from(s: String) -> Self {
        BfTreeStoreError(s)
    }
}

impl From<&str> for BfTreeStoreError {
    fn from(s: &str) -> Self {
        BfTreeStoreError(s.to_string())
    }
}

impl From<bf_tree::ConfigError> for BfTreeStoreError {
    fn from(e: bf_tree::ConfigError) -> Self {
        BfTreeStoreError(format!("{:?}", e))
    }
}

impl BfTreeStore {
    pub const MAX_RECORD_TOTAL_SIZE: usize = 4 * 1024;

    /// setting leaf page size to 16KB forces the minimum record size to be at least 16 * 1024 / 4096 = 4 bytes
    pub const LEAF_PAGE_SIZE: usize = 16 * 1024;
    /// Default cache size (32MB)
    pub const DEFAULT_CACHE_SIZE: usize = 32 * 1024 * 1024;

    /// Creates a new BfTreeStore with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - BfTree configuration including path, page size, and cache settings
    ///
    /// # Example
    ///
    /// ```ignore
    /// use bf_tree::Config;
    /// use diskann_label_filter::stores::bftree_store::BfTreeStore;
    ///
    /// let config = Config::new("./data/index", 32 * 1024 * 1024); // 32MB cache
    /// let store = BfTreeStore::new(config)?;
    /// ```
    pub fn new(config: Config) -> Result<Self, BfTreeStoreError> {
        let tree = BfTree::with_config(config, None)?;
        Ok(Self {
            tree: Arc::new(tree),
        })
    }

    /// Creates an in-memory BfTreeStore for testing or caching.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use diskann_label_filter::stores::bftree_store::BfTreeStore;
    ///
    /// let store = BfTreeStore::memory()?;
    /// ```
    pub fn memory() -> Result<Self, BfTreeStoreError> {
        Self::memory_with_cache_size(
            Self::DEFAULT_CACHE_SIZE,
            Self::MAX_RECORD_TOTAL_SIZE,
            Self::LEAF_PAGE_SIZE,
        )
    }

    /// Creates an in-memory BfTreeStore with custom cache size.
    ///
    /// # Arguments
    ///
    /// * `cache_size` - Size of the cache in bytes
    /// * `max_record_size` - max size of the record in bftree
    /// * `leaft_page_size` - leaf page size of the bftree
    pub fn memory_with_cache_size(
        cache_size: usize,
        max_record_size: usize,
        leaf_page_size: usize,
    ) -> Result<Self, BfTreeStoreError> {
        let mut config = Config::new(":memory:", cache_size);
        config.cb_max_key_len(1024);
        config.cb_max_record_size(max_record_size);
        config.leaf_page_size(leaf_page_size);
        Self::new(config)
    }

    /// Opens an existing BfTree at the specified path with default configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the BfTree data directory
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, BfTreeStoreError> {
        Self::open_with_cache_size(path, Self::DEFAULT_CACHE_SIZE)
    }

    /// Opens a BfTree with custom cache size.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the BfTree data directory
    /// * `cache_size` - Size of the cache in bytes
    pub fn open_with_cache_size<P: AsRef<Path>>(
        path: P,
        cache_size: usize,
    ) -> Result<Self, BfTreeStoreError> {
        let config = Config::new(path.as_ref(), cache_size);
        Self::new(config)
    }
}

impl KvStore for BfTreeStore {
    type Error = BfTreeStoreError;

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
        // Start with a reasonable size, grow if needed
        let mut buffer_size = 4096; // 4KB initial
        let mut buffer: Vec<u8> = vec![0u8; buffer_size];

        loop {
            match self.tree.read(key, &mut buffer) {
                LeafReadResult::Found(n) => {
                    let size = n as usize;
                    if size <= buffer_size {
                        buffer.truncate(size);
                        return Ok(Some(buffer));
                    }
                    // Buffer too small, grow and retry
                    buffer_size = size.min(Self::MAX_RECORD_TOTAL_SIZE);
                    if buffer_size >= Self::MAX_RECORD_TOTAL_SIZE {
                        return Err(BfTreeStoreError(format!(
                            "Value size {} exceeds maximum {}",
                            size,
                            Self::MAX_RECORD_TOTAL_SIZE
                        )));
                    }
                    buffer.resize(buffer_size, 0u8);
                }
                LeafReadResult::NotFound => return Ok(None),
                LeafReadResult::Deleted => return Ok(None), // Treat deleted keys as not found
                LeafReadResult::InvalidKey => {
                    let error_msg = format!("Invalid key:{:?}", key);
                    return Err(BfTreeStoreError(error_msg));
                }
            }
        }
    }

    fn set(&self, key: &[u8], value: &[u8]) -> Result<(), Self::Error> {
        match self.tree.insert(key, value) {
            LeafInsertResult::Success => Ok(()),
            LeafInsertResult::InvalidKV(msg) => {
                Err(BfTreeStoreError(format!("Invalid key-value pair: {}", msg)))
            }
        }
    }

    fn del(&self, key: &[u8]) -> Result<(), Self::Error> {
        // Note: BfTree::delete() doesn't return a result, it marks as deleted
        self.tree.delete(key);
        Ok(())
    }

    fn range<R>(&self, range: R) -> Result<KvIterator<'_, Self::Error>, Self::Error>
    where
        R: Into<KeyRange>,
    {
        let _key_range = range.into();

        // BfTree's scan API requires understanding its internal serialization format.
        // Rather than silently returning empty results, we return a clear error.
        Err(BfTreeStoreError(
            "Range queries are not yet implemented for BfTreeStore. \
             BfTree's scan API returns serialized key-value pairs that require \
             parsing the internal format. Contributions welcome!"
                .to_string(),
        ))
    }

    fn batch_get(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>, Self::Error> {
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            results.push(self.get(key)?);
        }
        Ok(results)
    }

    fn batch_set(&self, entries: &[(&[u8], &[u8])]) -> Result<(), Self::Error> {
        // Note: Not atomic - each insert is independent
        for (i, (key, value)) in entries.iter().enumerate() {
            match self.tree.insert(key, value) {
                LeafInsertResult::Success => {}
                LeafInsertResult::InvalidKV(msg) => {
                    return Err(BfTreeStoreError(format!(
                        "Batch set failed at index {}/{}: {}. \
                         Note: Previous {} operations succeeded (not atomic).",
                        i,
                        entries.len(),
                        msg,
                        i
                    )));
                }
            }
        }
        Ok(())
    }

    fn batch_del(&self, keys: &[&[u8]]) -> Result<(), Self::Error> {
        // Note: Not atomic - each delete is independent
        for key in keys {
            self.tree.delete(key);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_key_small_value() {
        let store = BfTreeStore::memory().unwrap();
        let key = b"key";
        let value = b"v";

        store.set(key, value).unwrap();
        let retrieved = store.get(key).unwrap();
        assert_eq!(retrieved, Some(value.to_vec()));
    }

    #[test]
    fn larger_key_small_value() {
        let store = BfTreeStore::memory().unwrap();
        let key = &[b'k'; 1000];
        let value = b"v";

        store.set(key, value).unwrap();
        let retrieved = store.get(key).unwrap();
        assert_eq!(retrieved, Some(value.to_vec()));
    }

    #[test]
    fn small_key_larger_value() {
        let store = BfTreeStore::memory().unwrap();
        let key = b"k";
        let value = &[b'v'; 1000];

        store.set(key, value).unwrap();
        let retrieved = store.get(key).unwrap();
        assert_eq!(retrieved, Some(value.to_vec()));
    }

    #[test]
    fn larger_key_larger_value() {
        println!("Test: larger_key_larger_value");
        let store = BfTreeStore::memory().unwrap();
        println!("Created in-memory BfTreeStore");
        let key = &[b'k'; 1000];
        let value = &[b'v'; 1000];

        store.set(key, value).unwrap();
        let retrieved = store.get(key).unwrap();
        assert_eq!(retrieved, Some(value.to_vec()));
    }

    #[test]
    fn test_basic_operations() {
        let store = BfTreeStore::memory().unwrap();

        // Test set and get
        store.set(b"key1", b"value1").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"value1".to_vec()));

        // Test overwrite
        store.set(b"key1", b"value2").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"value2".to_vec()));

        // Test delete
        store.del(b"key1").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), None);
    }

    #[test]
    fn test_batch_operations() {
        let store = BfTreeStore::memory().unwrap();

        // Test batch_set
        store
            .batch_set(&[(b"k1", b"v1"), (b"k2", b"v2"), (b"k3", b"v3")])
            .unwrap();

        // Test batch_get
        let results = store.batch_get(&[b"k1", b"k2", b"k3", b"k4"]).unwrap();
        assert_eq!(results[0], Some(b"v1".to_vec()));
        assert_eq!(results[1], Some(b"v2".to_vec()));
        assert_eq!(results[2], Some(b"v3".to_vec()));
        assert_eq!(results[3], None);

        // Test batch_del
        store.batch_del(&[b"k1", b"k3"]).unwrap();
        assert_eq!(store.get(b"k1").unwrap(), None);
        assert_eq!(store.get(b"k2").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(store.get(b"k3").unwrap(), None);
    }

    #[test]
    fn test_in_memory_isolation() {
        let store1 = BfTreeStore::memory().unwrap();
        let store2 = BfTreeStore::memory().unwrap();

        // Write to store1
        store1.set(b"key", b"value1").unwrap();

        // Store2 should not see store1's data
        assert_eq!(store2.get(b"key").unwrap(), None);

        // Write to store2
        store2.set(b"key", b"value2").unwrap();

        // Store1 should still have its own value
        assert_eq!(store1.get(b"key").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(store2.get(b"key").unwrap(), Some(b"value2".to_vec()));
    }

    #[test]
    fn test_large_values() {
        let store = BfTreeStore::memory().unwrap();

        // Test with a value that fits within BfTree limits
        let large_value = vec![42u8; 1024];
        store.set(b"large_key", &large_value).unwrap();
        assert_eq!(store.get(b"large_key").unwrap(), Some(large_value));
    }

    #[test]
    fn test_oversized_value_error() {
        let store = BfTreeStore::memory().unwrap();

        // BfTree has maximum record size limits
        // Try a very large value that should be rejected
        let huge_value = vec![42u8; 1024 * 1024]; // 1MB
        let result = store.set(b"huge_key", &huge_value);

        // Should return an error about invalid key-value pair
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = e.to_string();
            assert!(msg.contains("Invalid key-value pair") || msg.contains("too large"));
        }
    }

    #[test]
    fn test_range_query_returns_error() {
        let store = BfTreeStore::memory().unwrap();

        // Should return error, not empty results
        let result = store.range(b"a"..b"z");
        assert!(result.is_err());

        // Check the error message
        if let Err(e) = result {
            assert!(e.to_string().contains("not yet implemented"));
        }
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(BfTreeStore::memory().unwrap());
        let mut handles = vec![];

        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let key = format!("key_{}", i);
                let value = format!("value_{}", i);
                store_clone.set(key.as_bytes(), value.as_bytes()).unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all writes succeeded
        for i in 0..10 {
            let key = format!("key_{}", i);
            let expected_value = format!("value_{}", i);
            assert_eq!(
                store.get(key.as_bytes()).unwrap(),
                Some(expected_value.into_bytes())
            );
        }
    }
}
