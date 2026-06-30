/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PostingListProvider trait implementation for GenericIndex.

use crate::attribute::AttributeValue;
use crate::traits::key_codec::KeyCodec;
use crate::traits::kv_store_traits::KvStore;
use crate::traits::posting_list_trait::{PostingList, PostingListAccessor};

use super::error::{IndexError, Result};
use super::generic_index::{GenericIndex, DATA_TYPE_POSTING_LIST};

impl<S, PL, K> PostingListAccessor for GenericIndex<S, PL, K>
where
    S: KvStore,
    PL: PostingList,
    K: KeyCodec + Default,
    S::Error: std::error::Error + Send + Sync + 'static,
    PL::Error: std::error::Error + Send + Sync + 'static,
{
    type Error = IndexError;
    type PostingList = PL;
    type DocId = usize;

    fn get_posting_list(
        &self,
        field: &str,
        value: &AttributeValue,
    ) -> Result<Option<Self::PostingList>> {
        let codec = K::default();
        let key = codec.encode_field_value_key(field, value);

        match self
            .store
            .get(&key)
            .map_err(|e| IndexError::kv_store_with_key("get", key.clone(), e))?
        {
            Some(bytes) => {
                let pl = PL::deserialize(&bytes)
                    .map_err(|e| IndexError::serialization(DATA_TYPE_POSTING_LIST, e))?;
                Ok(Some(pl))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stores::bftree_store::BfTreeStore;
    use crate::traits::key_codec::DefaultKeyCodec;
    use crate::traits::posting_list_trait::RoaringPostingList;
    use std::sync::Arc;

    type TestIndex = GenericIndex<BfTreeStore, RoaringPostingList, DefaultKeyCodec>;

    fn index() -> TestIndex {
        let store = BfTreeStore::memory().expect("memory store");
        TestIndex::new(Arc::new(store))
    }

    fn key(field: &str, value: &AttributeValue) -> Vec<u8> {
        DefaultKeyCodec::default().encode_field_value_key(field, value)
    }

    #[test]
    fn get_posting_list_returns_stored_list() {
        let idx = index();
        let value = AttributeValue::String("red".to_string());

        let mut pl = RoaringPostingList::empty();
        pl.insert(3);
        pl.insert(9);
        idx.store()
            .set(&key("color", &value), &pl.serialize())
            .unwrap();

        let loaded = idx.get_posting_list("color", &value).unwrap().unwrap();
        assert!(loaded.contains(3));
        assert!(loaded.contains(9));
    }

    #[test]
    fn get_posting_list_missing_returns_none() {
        let idx = index();
        let value = AttributeValue::String("absent".to_string());
        assert!(idx.get_posting_list("color", &value).unwrap().is_none());
    }

    #[test]
    fn get_posting_list_corrupt_is_error() {
        let idx = index();
        let value = AttributeValue::String("bad".to_string());
        idx.store()
            .set(&key("color", &value), b"not-valid")
            .unwrap();

        assert!(idx.get_posting_list("color", &value).is_err());
    }
}
