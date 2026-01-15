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
