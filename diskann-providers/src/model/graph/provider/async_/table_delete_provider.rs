/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::atomic::{AtomicU32, Ordering};

use diskann::utils::IntoUsize;

use super::postprocess;

pub struct TableDeleteProviderAsync {
    delete_table: Vec<AtomicU32>,
    pub max_size: usize,
}

impl Default for TableDeleteProviderAsync {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Default data structure for the delete provider
/// Stores an array of (atomic u32-packed) bools which are
/// set true when an id is deleted, and false when not deleted
/// Designed to minimize contention at the expense of slightly
/// higher than optimal space usage in some cases
/// Warning: count(), clear(), and is_empty() are NOT LINEARIZABLE
impl TableDeleteProviderAsync {
    pub fn new(max_size: usize) -> Self {
        let size = max_size.div_ceil(32);
        let delete_table: Vec<_> = (0..size).map(|_| AtomicU32::new(0)).collect();
        TableDeleteProviderAsync {
            delete_table,
            max_size,
        }
    }

    #[inline]
    pub(crate) fn is_deleted(&self, vector_id: usize) -> bool {
        assert!(vector_id < self.max_size);
        let slot = vector_id / 32;
        let bit = vector_id % 32;
        let mask: u32 = 1 << (bit as u32);
        (self.delete_table[slot].load(Ordering::Acquire) & mask) != 0
    }

    pub(crate) fn delete(&self, vector_id: usize) {
        assert!(vector_id < self.max_size);
        let slot = vector_id / 32;
        let bit = vector_id % 32;
        let mask: u32 = 1 << (bit as u32);
        self.delete_table[slot].fetch_or(mask, Ordering::AcqRel);
    }

    // private method for now, but may need to become public
    pub(crate) fn undelete(&self, vector_id: usize) {
        assert!(vector_id < self.max_size);
        let slot = vector_id / 32;
        let bit = vector_id % 32;
        let mask: u32 = 1 << (bit as u32);
        self.delete_table[slot].fetch_and(!mask, Ordering::AcqRel);
    }

    pub(crate) fn clear(&self) {
        for i in 0..self.max_size {
            self.undelete(i);
        }
    }

    /// Serialize the delete bitmap to bytes (little-endian u32 values)
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(self.delete_table.as_slice()));
        for atomic_val in &self.delete_table {
            let val = atomic_val.load(Ordering::Relaxed);
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Create a TableDeleteProviderAsync from serialized bytes
    pub(crate) fn from_bytes(bytes: &[u8], max_size: usize) -> Result<Self, String> {
        let expected_len = max_size.div_ceil(32);

        let (chunks, remainder) = bytes.as_chunks::<{ std::mem::size_of::<u32>() }>();
        if chunks.len() != expected_len {
            return Err(format!(
                "Delete bitmap size mismatch: expected {} u32 values, got {}",
                expected_len,
                chunks.len()
            ));
        }
        if !remainder.is_empty() {
            return Err("Length of bytes is not a multiple of 4".to_string());
        }

        let delete_table = chunks
            .iter()
            .map(|chunk| AtomicU32::new(u32::from_le_bytes(*chunk)))
            .collect();

        Ok(TableDeleteProviderAsync {
            delete_table,
            max_size,
        })
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        let mut count = 0;
        for i in 0..self.max_size {
            if self.is_deleted(i) {
                count += 1;
            }
        }
        count
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        for i in 0..self.max_size {
            if self.is_deleted(i) {
                return false;
            }
        }
        true
    }
}

impl postprocess::DeletionCheck for TableDeleteProviderAsync {
    fn deletion_check(&self, id: u32) -> bool {
        self.is_deleted(id.into_usize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_async_delete_provider() {
        {
            // test max_size not a multiple of 32
            let delete_provider = get_test_delete_table_provider(50, &[0, 20, 34, 48]);

            assert!(!delete_provider.is_empty());
            assert!(delete_provider.count() == 4);
            assert!(delete_provider.is_deleted(0));
            assert!(delete_provider.is_deleted(20));
            assert!(!delete_provider.is_deleted(2));
            assert!(delete_provider.is_deleted(34));
            assert!(!delete_provider.is_deleted(40));

            delete_provider.delete(28);
            assert!(delete_provider.count() == 5);
            delete_provider.delete(37);
            assert!(delete_provider.count() == 6);
            delete_provider.delete(37);
            assert!(delete_provider.count() == 6);

            delete_provider.clear();

            assert!(delete_provider.count() == 0);
            assert!(delete_provider.is_empty());

            assert!(!delete_provider.is_deleted(0));
            assert!(!delete_provider.is_deleted(36));

            delete_provider.delete(0);
            assert!(delete_provider.is_deleted(0));

            assert!(delete_provider.count() == 1);
            assert!(!delete_provider.is_empty());

            delete_provider.undelete(0);
            assert!(delete_provider.is_empty());

            // check for access past max size -- should panic
            {
                let res = std::panic::catch_unwind(|| {
                    delete_provider.is_deleted(50);
                });
                assert!(res.is_err());
            }
            {
                let res = std::panic::catch_unwind(|| {
                    delete_provider.is_deleted(55);
                });
                assert!(res.is_err());
            }
            {
                let res = std::panic::catch_unwind(|| {
                    delete_provider.is_deleted(67);
                });
                assert!(res.is_err());
            }
        }

        {
            // test max_size a multiple of 32
            let delete_provider = get_test_delete_table_provider(64, &[0, 20, 34, 48, 55]);

            // access below max size -- should not panic

            delete_provider.is_deleted(5);
            delete_provider.is_deleted(32);
            delete_provider.is_deleted(63);

            // access past max size -- should panic
            {
                let res = std::panic::catch_unwind(|| {
                    delete_provider.is_deleted(64);
                });
                assert!(res.is_err());
            }
            {
                let res = std::panic::catch_unwind(|| {
                    delete_provider.is_deleted(70);
                });
                assert!(res.is_err());
            }
        }
    }

    fn get_test_delete_table_provider(
        max_points: usize,
        ids: &[usize],
    ) -> TableDeleteProviderAsync {
        let delete_provider = TableDeleteProviderAsync::new(max_points);
        for id in ids {
            delete_provider.delete(*id);
        }
        delete_provider
    }

    #[test]
    fn test_save_load_roundtrip() {
        let original = get_test_delete_table_provider(50, &[0, 5, 20, 34, 48]);
        let bytes = original.to_bytes();
        let loaded = TableDeleteProviderAsync::from_bytes(&bytes, 50).unwrap();

        assert_eq!(original.count(), loaded.count());
        for i in 0..50 {
            assert_eq!(original.is_deleted(i), loaded.is_deleted(i));
        }
    }

    #[test]
    fn test_from_bytes_size_mismatch() {
        // max_size=50 requires ceil(50/32) = 2 u32 values = 8 bytes
        // Provide wrong number of bytes (e.g., 4 bytes = 1 u32)
        let bytes = vec![0u8; 4]; // Only 1 u32 worth of bytes
        let result = TableDeleteProviderAsync::from_bytes(&bytes, 50);
        assert!(result.is_err());
    }
}
