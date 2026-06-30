/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! InvertedIndex trait implementation for GenericIndex.

use crate::traits::inverted_index_trait::InvertedIndexProvider;
use crate::traits::key_codec::KeyCodec;
use crate::traits::kv_store_traits::KvStore;
use crate::traits::posting_list_trait::PostingList;
use crate::utils::flatten_utils::Attributes;

use super::error::{IndexError, Result};
use super::generic_index::{GenericIndex, DATA_TYPE_POSTING_LIST};

impl<S, PL, K> InvertedIndexProvider for GenericIndex<S, PL, K>
where
    S: KvStore,
    PL: PostingList,
    K: KeyCodec + Default,
    S::Error: std::error::Error + Send + Sync + 'static,
    PL::Error: std::error::Error + Send + Sync + 'static,
{
    type Error = IndexError;
    type DocId = usize;

    fn insert(&mut self, doc_id: Self::DocId, attributes: &Attributes) -> Result<()> {
        let codec = K::default();
        // Attributes is already HashMap<String, AttributeValue>, no need to flatten
        let mut keys = Vec::with_capacity(attributes.len());

        for (field, value) in attributes.iter() {
            let key = codec.encode_field_value_key(field, value);
            keys.push(key.clone());

            // Get existing posting list
            let mut posting_list = self.get_or_empty_posting_list(&key)?;

            // Add document ID
            posting_list.insert(doc_id);

            // Serialize and store
            let bytes = posting_list.serialize();
            self.store
                .set(&key, &bytes)
                .map_err(|e| IndexError::kv_store_with_key("set", key.clone(), e))?;
        }

        // Store reverse mapping
        let reverse_key = Self::reverse_key(doc_id);
        let reverse_value = Self::serialize_key_list(&keys)?;
        self.store
            .set(&reverse_key, &reverse_value)
            .map_err(|e| IndexError::kv_store("set", e))?;

        Ok(())
    }

    fn delete(&mut self, doc_id: Self::DocId) -> Result<()> {
        // Load reverse mapping
        let reverse_key = Self::reverse_key(doc_id);
        let reverse_bytes = self
            .store
            .get(&reverse_key)
            .map_err(|e| IndexError::kv_store("get", e))?;

        let keys = match reverse_bytes {
            Some(bytes) => Self::deserialize_key_list(&bytes)?,
            None => return Ok(()), // Document doesn't exist, nothing to delete
        };

        // Remove document from each posting list
        for key in &keys {
            if let Some(bytes) = self
                .store
                .get(key)
                .map_err(|e| IndexError::kv_store_with_key("get", key.clone(), e))?
            {
                let mut posting_list = PL::deserialize(&bytes)
                    .map_err(|e| IndexError::serialization(DATA_TYPE_POSTING_LIST, e))?;

                posting_list.remove(doc_id);

                if posting_list.is_empty() {
                    // Remove empty posting list
                    self.store
                        .del(key)
                        .map_err(|e| IndexError::kv_store_with_key("delete", key.clone(), e))?;
                } else {
                    // Update posting list
                    let bytes = posting_list.serialize();
                    self.store
                        .set(key, &bytes)
                        .map_err(|e| IndexError::kv_store_with_key("set", key.clone(), e))?;
                }
            }
        }

        // Delete reverse mapping
        self.store
            .del(&reverse_key)
            .map_err(|e| IndexError::kv_store("delete", e))?;

        Ok(())
    }

    fn update(&mut self, doc_id: Self::DocId, attributes: &Attributes) -> Result<()> {
        self.delete(doc_id)?;
        self.insert(doc_id, attributes)?;
        Ok(())
    }

    /// Batch insert multiple documents.
    ///
    /// # Atomicity
    ///
    /// This operation is **not atomic**. If an error occurs during processing,
    /// some documents may have been inserted while others were not. The caller
    /// should handle partial failures appropriately.
    ///
    /// # Performance
    ///
    /// Currently implemented as sequential inserts. Future optimizations may
    /// include batched KV store writes for better performance.
    fn batch_insert(&mut self, pairs: &[(Self::DocId, Attributes)]) -> Result<()> {
        for (doc_id, attributes) in pairs {
            self.insert(*doc_id, attributes)?;
        }
        Ok(())
    }

    /// Batch delete multiple documents.
    ///
    /// # Atomicity
    ///
    /// This operation is **not atomic**. If an error occurs during processing,
    /// some documents may have been deleted while others were not. The caller
    /// should handle partial failures appropriately.
    ///
    /// # Performance
    ///
    /// Currently implemented as sequential deletes. Future optimizations may
    /// include batched KV store operations for better performance.
    fn batch_delete(&mut self, doc_ids: &[Self::DocId]) -> Result<()> {
        for doc_id in doc_ids {
            self.delete(*doc_id)?;
        }
        Ok(())
    }

    /// Batch update multiple documents.
    ///
    /// # Atomicity
    ///
    /// This operation is **not atomic**. Each update is performed as a
    /// delete followed by an insert. If an error occurs during processing,
    /// the index may be in an inconsistent state. The caller should handle
    /// partial failures appropriately.
    ///
    /// # Performance
    ///
    /// Currently implemented as sequential delete+insert operations. Future
    /// optimizations may include batched operations for better performance.
    fn batch_update(&mut self, pairs: &[(Self::DocId, Attributes)]) -> Result<()> {
        for (doc_id, attributes) in pairs {
            self.update(*doc_id, attributes)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::AttributeValue;
    use crate::stores::bftree_store::BfTreeStore;
    use crate::traits::key_codec::DefaultKeyCodec;
    use crate::traits::posting_list_trait::RoaringPostingList;
    use std::collections::HashMap;
    use std::sync::Arc;

    type TestIndex = GenericIndex<BfTreeStore, RoaringPostingList, DefaultKeyCodec>;

    fn index() -> TestIndex {
        let store = BfTreeStore::memory().expect("memory store");
        TestIndex::new(Arc::new(store))
    }

    fn attrs(pairs: &[(&str, &str)]) -> Attributes {
        pairs
            .iter()
            .map(|(f, v)| (f.to_string(), AttributeValue::String(v.to_string())))
            .collect::<HashMap<_, _>>()
    }

    /// Read the posting list for a string field/value pair.
    fn posting(index: &TestIndex, field: &str, value: &str) -> RoaringPostingList {
        let codec = DefaultKeyCodec::default();
        let key = codec.encode_field_value_key(field, &AttributeValue::String(value.to_string()));
        index.get_or_empty_posting_list(&key).unwrap()
    }

    #[test]
    fn insert_adds_document_to_posting_list() {
        let mut idx = index();
        idx.insert(1, &attrs(&[("color", "red")])).unwrap();

        let pl = posting(&idx, "color", "red");
        assert!(pl.contains(1));
    }

    #[test]
    fn delete_removes_document_and_empties_posting_list() {
        let mut idx = index();
        idx.insert(1, &attrs(&[("color", "red")])).unwrap();
        idx.delete(1).unwrap();

        // Posting list became empty and was removed from the store.
        assert_eq!(posting(&idx, "color", "red").len(), 0);
    }

    #[test]
    fn delete_missing_document_is_noop() {
        let mut idx = index();
        // No reverse mapping exists -> early return, still Ok.
        idx.delete(99).unwrap();
    }

    #[test]
    fn delete_keeps_posting_list_with_remaining_documents() {
        let mut idx = index();
        idx.insert(1, &attrs(&[("color", "red")])).unwrap();
        idx.insert(2, &attrs(&[("color", "red")])).unwrap();

        idx.delete(1).unwrap();

        let pl = posting(&idx, "color", "red");
        assert!(!pl.contains(1));
        assert!(pl.contains(2));
    }

    #[test]
    fn update_moves_document_between_values() {
        let mut idx = index();
        idx.insert(7, &attrs(&[("color", "red")])).unwrap();
        idx.update(7, &attrs(&[("color", "blue")])).unwrap();

        assert!(!posting(&idx, "color", "red").contains(7));
        assert!(posting(&idx, "color", "blue").contains(7));
    }

    #[test]
    fn batch_insert_delete_update_round_trip() {
        let mut idx = index();

        idx.batch_insert(&[
            (10, attrs(&[("color", "red")])),
            (11, attrs(&[("color", "red")])),
        ])
        .unwrap();
        assert!(posting(&idx, "color", "red").contains(10));
        assert!(posting(&idx, "color", "red").contains(11));

        idx.batch_update(&[(10, attrs(&[("color", "green")]))])
            .unwrap();
        assert!(posting(&idx, "color", "green").contains(10));
        assert!(!posting(&idx, "color", "red").contains(10));

        idx.batch_delete(&[11]).unwrap();
        assert!(!posting(&idx, "color", "red").contains(11));
    }
}
