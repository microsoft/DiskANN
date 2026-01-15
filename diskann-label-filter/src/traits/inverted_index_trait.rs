/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::utils::flatten_utils::Attributes;

/// An inverted index trait for storing and modifying document attributes.
///
/// This trait defines the write operations for an inverted index that maps
/// field-value pairs to document IDs (posting lists). It supports insertion,
/// deletion, and updates.
///
/// For query evaluation capabilities, see the `QueryEvaluator` trait
pub trait InvertedIndexProvider {
    /// The error type returned by operations on this inverted index.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The document ID type used to identify documents in the index.
    type DocId: Copy + Into<usize> + From<usize> + std::fmt::Debug;

    /// Inserts a document with its attributes into the index.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The unique identifier for the document.
    /// * `attributes` - The attributes (field-value pairs) of the document.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the insertion fails.
    fn insert(
        &mut self,
        doc_id: Self::DocId,
        attributes: &Attributes,
    ) -> std::result::Result<(), Self::Error>;

    /// Inserts multiple documents with their attributes into the index.
    ///
    /// This is more efficient than calling `insert` multiple times individually.
    ///
    /// # Arguments
    ///
    /// * `attribute_id_pairs` - A slice of (document ID, attributes) pairs.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the batch insertion fails.
    fn batch_insert(
        &mut self,
        attribute_id_pairs: &[(Self::DocId, Attributes)],
    ) -> std::result::Result<(), Self::Error>;

    /// Deletes a document from the index.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The unique identifier of the document to delete.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the deletion fails.
    fn delete(&mut self, doc_id: Self::DocId) -> std::result::Result<(), Self::Error>;

    /// Deletes multiple documents from the index.
    ///
    /// This is more efficient than calling `delete` multiple times individually.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - A slice of document IDs to delete.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the batch deletion fails.
    fn batch_delete(&mut self, doc_ids: &[Self::DocId]) -> std::result::Result<(), Self::Error>;

    /// Updates a document's attributes in the index.
    ///
    /// This operation typically removes the old attributes and inserts the new ones.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The unique identifier of the document to update.
    /// * `attributes` - The new attributes for the document.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the update fails.
    fn update(
        &mut self,
        doc_id: Self::DocId,
        attributes: &Attributes,
    ) -> std::result::Result<(), Self::Error>;

    /// Updates multiple documents' attributes in the index.
    ///
    /// This is more efficient than calling `update` multiple times individually.
    ///
    /// # Arguments
    ///
    /// * `doc_id_attribute_pairs` - A slice of (document ID, new attributes) pairs.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the batch update fails.
    fn batch_update(
        &mut self,
        doc_id_attribute_pairs: &[(Self::DocId, Attributes)],
    ) -> std::result::Result<(), Self::Error>;
}

#[cfg(test)]
mod reference_impl {

    use super::*;
    use crate::attribute::AttributeValue;
    use crate::traits::key_codec::KeyCodec;
    use crate::traits::kv_store_traits::KvStore;
    use crate::traits::posting_list_trait::{PostingList, PostingListAccessor};
    use crate::traits::query_evaluator::QueryEvaluator;
    use crate::utils::flatten_utils::Attributes;
    use crate::ASTExpr;
    use serde_json::Value;
    use std::marker::PhantomData;
    use std::sync::Arc;

    #[derive(Debug)]
    pub struct IndexError(pub String);

    impl std::fmt::Display for IndexError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for IndexError {}

    impl From<&str> for IndexError {
        fn from(s: &str) -> Self {
            IndexError(s.to_string())
        }
    }

    impl From<String> for IndexError {
        fn from(s: String) -> Self {
            IndexError(s)
        }
    }

    type Result<T> = std::result::Result<T, IndexError>;

    pub struct GenericInvertedIndex<S: KvStore, PL: PostingList, K: KeyCodec + Default> {
        store: Arc<S>,
        _pl: PhantomData<PL>,
        _kc: PhantomData<K>,
    }

    impl<S, PL, K> GenericInvertedIndex<S, PL, K>
    where
        S: KvStore,
        PL: PostingList,
        K: KeyCodec + Default,
    {
        pub fn new(store: Arc<S>) -> Self {
            Self {
                store,
                _pl: PhantomData,
                _kc: PhantomData,
            }
        }

        fn reverse_key(label_id: usize) -> Vec<u8> {
            format!("@R:{}", label_id).into_bytes()
        }

        fn serialize_key_list(keys: &[Vec<u8>]) -> Vec<u8> {
            let mut out = Vec::with_capacity(4 + keys.iter().map(|k| 4 + k.len()).sum::<usize>());
            let count = keys.len() as u32;
            out.extend_from_slice(&count.to_le_bytes());
            for k in keys {
                let len = k.len() as u32;
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(k);
            }
            out
        }

        fn deserialize_key_list(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
            if bytes.len() < 4 {
                return Err("corrupt reverse key list: too short".into());
            }
            let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
            let mut pos = 4;
            let mut keys = Vec::with_capacity(count);
            for _ in 0..count {
                if pos + 4 > bytes.len() {
                    return Err("corrupt reverse key list: length header overflow".into());
                }
                let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                if pos + len > bytes.len() {
                    return Err("corrupt reverse key list: element overflow".into());
                }
                keys.push(bytes[pos..pos + len].to_vec());
                pos += len;
            }
            if pos != bytes.len() {
                return Err("corrupt reverse key list: trailing bytes".into());
            }
            Ok(keys)
        }

        fn load_reverse_keys(&self, label_id: usize) -> Result<Vec<Vec<u8>>> {
            let key = Self::reverse_key(label_id);
            match self
                .store
                .get(&key)
                .map_err(|e| IndexError(e.to_string()))?
            {
                Some(bytes) => Self::deserialize_key_list(&bytes),
                None => Ok(Vec::new()),
            }
        }

        fn save_reverse_keys(&self, label_id: usize, keys: &[Vec<u8>]) -> Result<()> {
            let key = Self::reverse_key(label_id);
            let bytes = Self::serialize_key_list(keys);
            self.store
                .set(&key, &bytes)
                .map_err(|e| IndexError(e.to_string()))
        }

        fn delete_reverse_keys(&self, label_id: usize) -> Result<()> {
            let key = Self::reverse_key(label_id);
            self.store.del(&key).map_err(|e| IndexError(e.to_string()))
        }

        fn get_posting_for(&self, field: &str, value: &Value) -> Result<Option<PL>> {
            let codec = K::default();
            match &AttributeValue::try_from(value) {
                Ok(real_value) => {
                    let key_str = codec.encode_field_value_key(field, real_value);
                    match self
                        .store
                        .get(&key_str)
                        .map_err(|e| IndexError(e.to_string()))?
                    {
                        Some(bytes) => Ok(Some(
                            PL::deserialize(&bytes).map_err(|e| IndexError(e.to_string()))?,
                        )),
                        None => Ok(None),
                    }
                }
                Err(e) => {
                    panic!(
                        "Failed to convert JSON value to AttributeValue because: {}",
                        e
                    );
                }
            }
        }

        fn range_query_gte(&self, field: &str, value: f64) -> Result<PL> {
            let codec = K::default();
            let mut result = PL::empty();

            if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                let int_value = AttributeValue::try_from(&serde_json::json!(value as i64))
                    .map_err(|_| IndexError("Failed to convert to integer".to_string()))?;
                let start_key = codec.encode_field_value_key(field, &int_value);
                let end_key = self.create_field_type_upper_bound(field, &int_value);
                result = result.union(&self.union_posting_lists_in_range(&start_key, &end_key)?);
            }

            let float_value = AttributeValue::try_from(&serde_json::json!(value))
                .map_err(|_| IndexError("Failed to convert to float".to_string()))?;
            let start_key = codec.encode_field_value_key(field, &float_value);
            let end_key = self.create_field_type_upper_bound(field, &float_value);
            result = result.union(&self.union_posting_lists_in_range(&start_key, &end_key)?);

            Ok(result)
        }

        fn range_query_gt(&self, field: &str, value: f64) -> Result<PL> {
            let mut result = self.range_query_gte(field, value)?;

            if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                if let Some(exact_int) =
                    self.get_posting_for(field, &serde_json::json!(value as i64))?
                {
                    result = self.subtract_posting_lists(&result, &exact_int);
                }
            }
            if let Some(exact_float) = self.get_posting_for(field, &serde_json::json!(value))? {
                result = self.subtract_posting_lists(&result, &exact_float);
            }

            Ok(result)
        }

        fn range_query_lte(&self, field: &str, value: f64) -> Result<PL> {
            let codec = K::default();
            let mut result = PL::empty();

            if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                let int_value = AttributeValue::try_from(&serde_json::json!(value as i64))
                    .map_err(|_| IndexError("Failed to convert to integer".to_string()))?;
                let start_key = self.create_field_type_lower_bound(field, &int_value);
                let end_key = codec.encode_field_value_key(field, &int_value);
                let end_key_inclusive = self.increment_key(&end_key);
                result = result
                    .union(&self.union_posting_lists_in_range(&start_key, &end_key_inclusive)?);
            }

            let float_value = AttributeValue::try_from(&serde_json::json!(value))
                .map_err(|_| IndexError("Failed to convert to float".to_string()))?;
            let start_key = self.create_field_type_lower_bound(field, &float_value);
            let end_key = codec.encode_field_value_key(field, &float_value);
            let end_key_inclusive = self.increment_key(&end_key);
            result =
                result.union(&self.union_posting_lists_in_range(&start_key, &end_key_inclusive)?);

            Ok(result)
        }

        fn range_query_lt(&self, field: &str, value: f64) -> Result<PL> {
            let codec = K::default();
            let mut result = PL::empty();

            if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                let int_value = AttributeValue::try_from(&serde_json::json!(value as i64))
                    .map_err(|_| IndexError("Failed to convert to integer".to_string()))?;
                let start_key = self.create_field_type_lower_bound(field, &int_value);
                let end_key = codec.encode_field_value_key(field, &int_value);
                result = result.union(&self.union_posting_lists_in_range(&start_key, &end_key)?);
            }

            let float_value = AttributeValue::try_from(&serde_json::json!(value))
                .map_err(|_| IndexError("Failed to convert to float".to_string()))?;
            let start_key = self.create_field_type_lower_bound(field, &float_value);
            let end_key = codec.encode_field_value_key(field, &float_value);
            result = result.union(&self.union_posting_lists_in_range(&start_key, &end_key)?);

            Ok(result)
        }

        fn union_posting_lists_in_range(&self, start_key: &[u8], end_key: &[u8]) -> Result<PL> {
            let mut result = PL::empty();
            for item in self
                .store
                .range(start_key..end_key)
                .map_err(|e| IndexError(e.to_string()))?
            {
                let (_key, value_bytes) = item.map_err(|e| IndexError(e.to_string()))?;
                let pl = PL::deserialize(&value_bytes).map_err(|e| IndexError(e.to_string()))?;
                result = result.union(&pl);
            }
            Ok(result)
        }

        fn subtract_posting_lists(&self, a: &PL, b: &PL) -> PL {
            a.difference(b)
        }

        fn create_field_type_lower_bound(
            &self,
            field: &str,
            sample_value: &AttributeValue,
        ) -> Vec<u8> {
            match sample_value {
                AttributeValue::Real(_) | AttributeValue::Integer(_) => {
                    let codec = K::default();
                    let sample_key = codec.encode_field_value_key(field, sample_value);
                    if let Some(pos) = sample_key.iter().position(|&b| b == 0) {
                        if pos + 1 < sample_key.len() {
                            let type_prefix = sample_key[pos + 1];
                            return format!("{}\0{}00000000000000000", field, type_prefix as char)
                                .into_bytes();
                        }
                    }
                    format!("{}\0", field).into_bytes()
                }
                _ => format!("{}\0", field).into_bytes(),
            }
        }

        fn create_field_type_upper_bound(
            &self,
            field: &str,
            sample_value: &AttributeValue,
        ) -> Vec<u8> {
            match sample_value {
                AttributeValue::Real(_) | AttributeValue::Integer(_) => {
                    let codec = K::default();
                    let sample_key = codec.encode_field_value_key(field, sample_value);
                    if let Some(pos) = sample_key.iter().position(|&b| b == 0) {
                        if pos + 1 < sample_key.len() {
                            let type_prefix = sample_key[pos + 1];
                            return format!(
                                "{}\0{}ffffffffffffffffffffffff",
                                field, type_prefix as char
                            )
                            .into_bytes();
                        }
                    }
                    format!("{}\0\u{FFFF}", field).into_bytes()
                }
                _ => format!("{}\0\u{FFFF}", field).into_bytes(),
            }
        }

        fn increment_key(&self, key: &[u8]) -> Vec<u8> {
            let mut incremented = key.to_vec();
            incremented.push(0x00);
            incremented
        }
    }

    impl<S, PL, K> QueryEvaluator for GenericInvertedIndex<S, PL, K>
    where
        S: KvStore,
        PL: PostingList,
        K: KeyCodec + Default,
    {
        type PostingList = PL;
        type DocId = usize;
        type Error = IndexError;

        fn evaluate_query(&self, expr: &ASTExpr) -> Result<Self::PostingList> {
            match expr {
                ASTExpr::And(subs) => {
                    if subs.is_empty() {
                        return Ok(PL::empty());
                    }
                    let mut acc = self.evaluate_query(&subs[0])?;
                    for s in subs.iter().skip(1) {
                        let r = self.evaluate_query(s)?;
                        acc = acc.intersect(&r);
                        if acc.len() == 0 {
                            break;
                        }
                    }
                    Ok(acc)
                }
                ASTExpr::Or(subs) => {
                    if subs.is_empty() {
                        return Ok(PL::empty());
                    }
                    let mut acc = PL::empty();
                    for s in subs.iter() {
                        let r = self.evaluate_query(s)?;
                        acc = acc.union(&r);
                    }
                    Ok(acc)
                }
                ASTExpr::Not(sub) => {
                    let r = self.evaluate_query(sub)?;
                    Ok(r)
                }
                ASTExpr::Compare { field, op } => match op {
                    crate::CompareOp::Eq(v) => {
                        if let Some(pl) = self.get_posting_for(field, v)? {
                            return Ok(pl);
                        }
                        Ok(PL::empty())
                    }
                    crate::CompareOp::Ne(_v) => {
                        todo!();
                    }
                    crate::CompareOp::Lt(num) => self.range_query_lt(field, *num),
                    crate::CompareOp::Lte(num) => self.range_query_lte(field, *num),
                    crate::CompareOp::Gt(num) => self.range_query_gt(field, *num),
                    crate::CompareOp::Gte(num) => self.range_query_gte(field, *num),
                },
            }
        }
    }

    impl<S, PL, K> PostingListAccessor for GenericInvertedIndex<S, PL, K>
    where
        S: KvStore,
        PL: PostingList,
        K: KeyCodec + Default,
    {
        type PostingList = PL;
        type DocId = usize;
        type Error = IndexError;

        fn get_posting_list(
            &self,
            field: &str,
            value: &AttributeValue,
        ) -> Result<Option<Self::PostingList>> {
            let codec = K::default();
            let key_str = codec.encode_field_value_key(field, value);
            match self
                .store
                .get(&key_str)
                .map_err(|e| IndexError(e.to_string()))?
            {
                Some(bytes) => Ok(Some(
                    PL::deserialize(&bytes).map_err(|e| IndexError(e.to_string()))?,
                )),
                None => Ok(None),
            }
        }
    }

    impl<S, PL, K> InvertedIndexProvider for GenericInvertedIndex<S, PL, K>
    where
        S: KvStore,
        PL: PostingList,
        K: KeyCodec + Default,
    {
        type DocId = usize;
        type Error = IndexError;

        fn batch_insert(&mut self, attribute_id_pairs: &[(Self::DocId, Attributes)]) -> Result<()> {
            for (label_id, attributes) in attribute_id_pairs {
                self.insert(*label_id, attributes)?;
            }
            Ok(())
        }

        fn insert(&mut self, label_id: Self::DocId, attributes: &Attributes) -> Result<()> {
            if !self.load_reverse_keys(label_id)?.is_empty() {
                self.delete(label_id)?;
            }
            let mut keys_for_label = Vec::new();
            let codec = K::default();
            for (field, value) in attributes {
                let key = codec.encode_field_value_key(field, value);
                match self
                    .store
                    .get(&key)
                    .map_err(|e| IndexError(e.to_string()))?
                {
                    Some(existing) => {
                        let mut pl =
                            PL::deserialize(&existing).map_err(|e| IndexError(e.to_string()))?;
                        if pl.insert(label_id) {
                            let bytes = pl.serialize();
                            self.store
                                .set(&key, &bytes)
                                .map_err(|e| IndexError(e.to_string()))?;
                        }
                    }
                    None => {
                        let mut pl = PL::empty();
                        pl.insert(label_id);
                        let bytes = pl.serialize();
                        self.store
                            .set(&key, &bytes)
                            .map_err(|e| IndexError(e.to_string()))?;
                    }
                }
                keys_for_label.push(key);
            }
            self.save_reverse_keys(label_id, &keys_for_label)?;
            Ok(())
        }

        fn delete(&mut self, label_id: Self::DocId) -> Result<()> {
            let keys = self.load_reverse_keys(label_id)?;
            for key in keys.iter() {
                if let Some(existing) =
                    self.store.get(key).map_err(|e| IndexError(e.to_string()))?
                {
                    let mut pl =
                        PL::deserialize(&existing).map_err(|e| IndexError(e.to_string()))?;
                    if pl.remove(label_id) {
                        if pl.len() == 0 {
                            self.store.del(key).map_err(|e| IndexError(e.to_string()))?;
                        } else {
                            let bytes = pl.serialize();
                            self.store
                                .set(key, &bytes)
                                .map_err(|e| IndexError(e.to_string()))?;
                        }
                    }
                }
            }
            self.delete_reverse_keys(label_id)?;
            Ok(())
        }

        fn update(&mut self, label_id: Self::DocId, attributes: &Attributes) -> Result<()> {
            self.delete(label_id)?;
            self.insert(label_id, attributes)
        }

        fn batch_delete(&mut self, doc_ids: &[Self::DocId]) -> Result<()> {
            for doc_id in doc_ids {
                self.delete(*doc_id)?;
            }
            Ok(())
        }

        fn batch_update(
            &mut self,
            doc_id_attribute_pairs: &[(Self::DocId, Attributes)],
        ) -> Result<()> {
            for (doc_id, attributes) in doc_id_attribute_pairs {
                self.update(*doc_id, attributes)?;
            }
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::traits::{key_codec::DefaultKeyCodec, posting_list_trait::RoaringPostingList};
        use hashbrown::HashMap;
        use serde_json::json;

        #[derive(Default)]
        struct DummyKvStore {
            map: std::sync::Mutex<HashMap<Vec<u8>, Vec<u8>>>,
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
        impl KvStore for DummyKvStore {
            type Error = KvStoreError;

            fn get(&self, key: &[u8]) -> std::result::Result<Option<Vec<u8>>, Self::Error> {
                let map = self.map.lock().unwrap();
                Ok(map.get(key).cloned())
            }

            fn set(&self, key: &[u8], value: &[u8]) -> std::result::Result<(), Self::Error> {
                self.map
                    .lock()
                    .unwrap()
                    .insert(key.to_vec(), value.to_vec());
                Ok(())
            }

            fn del(&self, key: &[u8]) -> std::result::Result<(), Self::Error> {
                self.map.lock().unwrap().remove(key);
                Ok(())
            }

            fn range<R>(
                &self,
                range: R,
            ) -> std::result::Result<
                crate::traits::kv_store_traits::KvIterator<'_, Self::Error>,
                Self::Error,
            >
            where
                R: Into<crate::traits::kv_store_traits::KeyRange>,
            {
                use crate::traits::kv_store_traits::KeyRange;
                let key_range: KeyRange = range.into();
                let map = self.map.lock().unwrap();
                let items: Vec<_> = map
                    .iter()
                    .filter(|(k, _)| key_range.contains(k))
                    .map(|(k, v)| Ok((k.clone(), v.clone())))
                    .collect();
                Ok(Box::new(items.into_iter()))
            }

            fn batch_set(
                &self,
                entries: &[(&[u8], &[u8])],
            ) -> std::result::Result<(), Self::Error> {
                let mut map = self.map.lock().unwrap();
                for (key, value) in entries {
                    map.insert(key.to_vec(), value.to_vec());
                }
                Ok(())
            }

            fn batch_get(
                &self,
                keys: &[&[u8]],
            ) -> std::result::Result<Vec<Option<Vec<u8>>>, Self::Error> {
                let map = self.map.lock().unwrap();
                let results = keys.iter().map(|k| map.get(*k).cloned()).collect();
                Ok(results)
            }

            fn batch_del(&self, keys: &[&[u8]]) -> std::result::Result<(), Self::Error> {
                let mut map = self.map.lock().unwrap();
                for k in keys {
                    map.remove(*k);
                }
                Ok(())
            }
        }

        fn make_index() -> GenericInvertedIndex<DummyKvStore, RoaringPostingList, DefaultKeyCodec> {
            GenericInvertedIndex::new(Arc::new(DummyKvStore::default()))
        }

        #[test]
        fn test_insert_and_get_posting_list() {
            let mut index = make_index();
            let mut attrs = Attributes::new();
            let red_value = AttributeValue::try_from(&json!("red")).unwrap();
            attrs.insert("color".to_string(), red_value.clone());
            index.insert(1, &attrs).unwrap();

            let pl = index
                .get_posting_list("color", &red_value)
                .unwrap()
                .unwrap();
            assert_eq!(pl.len(), 1);
            assert!(pl.contains(1));
        }

        #[test]
        fn test_update_and_delete() {
            let mut index = make_index();
            let mut attrs = Attributes::new();
            attrs.insert(
                "shape".to_string(),
                AttributeValue::try_from(&json!("circle")).unwrap(),
            );
            index.insert(2, &attrs).unwrap();

            let mut new_attrs = Attributes::new();
            new_attrs.insert(
                "shape".to_string(),
                AttributeValue::try_from(&json!("square")).unwrap(),
            );
            index.update(2, &new_attrs).unwrap();

            let circle_value = AttributeValue::try_from(&json!("circle")).unwrap();
            let square_value = AttributeValue::try_from(&json!("square")).unwrap();

            assert!(index
                .get_posting_list("shape", &circle_value)
                .unwrap()
                .is_none());
            let pl = index
                .get_posting_list("shape", &square_value)
                .unwrap()
                .unwrap();
            assert_eq!(pl.len(), 1);
            assert!(pl.contains(2));

            index.delete(2).unwrap();
            assert!(index
                .get_posting_list("shape", &square_value)
                .unwrap()
                .is_none());
        }

        #[test]
        fn test_build_bulk_insert() {
            let mut index = make_index();
            let mut attrs1 = Attributes::new();
            attrs1.insert(
                "type".to_string(),
                AttributeValue::try_from(&json!("A")).unwrap(),
            );
            let mut attrs2 = Attributes::new();
            attrs2.insert(
                "type".to_string(),
                AttributeValue::try_from(&json!("B")).unwrap(),
            );
            let base = vec![(10, attrs1), (20, attrs2)];
            index.batch_insert(&base).unwrap();

            let type_a = AttributeValue::try_from(&json!("A")).unwrap();
            let type_b = AttributeValue::try_from(&json!("B")).unwrap();
            let pl_a = index.get_posting_list("type", &type_a).unwrap().unwrap();
            let pl_b = index.get_posting_list("type", &type_b).unwrap().unwrap();
            assert_eq!(pl_a.len(), 1);
            assert_eq!(pl_b.len(), 1);
        }

        #[test]
        fn test_evaluate_query_and_or() {
            let mut index = make_index();
            let mut attrs1 = Attributes::new();
            attrs1.insert(
                "x".to_string(),
                AttributeValue::try_from(&json!(1)).unwrap(),
            );
            let mut attrs2 = Attributes::new();
            attrs2.insert(
                "x".to_string(),
                AttributeValue::try_from(&json!(2)).unwrap(),
            );
            index.insert(1, &attrs1).unwrap();
            index.insert(2, &attrs2).unwrap();

            let expr = ASTExpr::Or(vec![
                ASTExpr::Compare {
                    field: "x".to_string(),
                    op: crate::CompareOp::Eq(json!(1)),
                },
                ASTExpr::Compare {
                    field: "x".to_string(),
                    op: crate::CompareOp::Eq(json!(2)),
                },
            ]);
            let pl = index.evaluate_query(&expr).unwrap();
            assert!(pl.contains(1));
            assert!(pl.contains(2));

            let expr_and = ASTExpr::And(vec![
                ASTExpr::Compare {
                    field: "x".to_string(),
                    op: crate::CompareOp::Eq(json!(1)),
                },
                ASTExpr::Compare {
                    field: "x".to_string(),
                    op: crate::CompareOp::Eq(json!(2)),
                },
            ]);
            let pl_and = index.evaluate_query(&expr_and).unwrap();
            assert!(pl_and.is_empty());
        }

        #[test]
        fn test_reverse_key_serialization_roundtrip() {
            let keys = vec![b"foo".to_vec(), b"bar".to_vec()];
            let bytes = GenericInvertedIndex::<
                DummyKvStore,
                RoaringPostingList,
                DefaultKeyCodec,
            >::serialize_key_list(&keys);
            let roundtrip = GenericInvertedIndex::<
                DummyKvStore,
                RoaringPostingList,
                DefaultKeyCodec,
            >::deserialize_key_list(&bytes)
            .unwrap();
            assert_eq!(keys, roundtrip);
        }

        #[test]
        fn test_get_posting_for_nonexistent() {
            let index = make_index();
            let test_value = AttributeValue::try_from(&json!("value")).unwrap();
            let pl = index.get_posting_list("nonexistent", &test_value).unwrap();
            assert!(pl.is_none());
        }

        #[test]
        fn test_range_query_graceful_degradation() {
            let mut index = make_index();
            for (id, age) in [(1, 10), (2, 25), (3, 30), (4, 35), (5, 50)] {
                let mut attrs = Attributes::new();
                attrs.insert(
                    "age".to_string(),
                    AttributeValue::try_from(&json!(age)).unwrap(),
                );
                index.insert(id, &attrs).unwrap();
            }

            let expr = ASTExpr::Compare {
                field: "age".to_string(),
                op: crate::CompareOp::Gte(25.0),
            };
            let pl = index.evaluate_query(&expr).unwrap();

            assert!(pl.contains(2));
            assert!(pl.contains(3));
            assert!(pl.contains(4));
            assert!(pl.contains(5));
            assert!(!pl.contains(1));
        }

        #[test]
        fn test_range_query_types() {
            let mut index = make_index();
            for (id, score) in [(1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0), (5, 50.0)] {
                let mut attrs = Attributes::new();
                attrs.insert(
                    "score".to_string(),
                    AttributeValue::try_from(&json!(score)).unwrap(),
                );
                index.insert(id, &attrs).unwrap();
            }

            let pl_gte = index
                .evaluate_query(&ASTExpr::Compare {
                    field: "score".to_string(),
                    op: crate::CompareOp::Gte(30.0),
                })
                .unwrap();
            assert_eq!(pl_gte.len(), 3);

            let pl_gt = index
                .evaluate_query(&ASTExpr::Compare {
                    field: "score".to_string(),
                    op: crate::CompareOp::Gt(30.0),
                })
                .unwrap();
            assert_eq!(pl_gt.len(), 2);

            let pl_lte = index
                .evaluate_query(&ASTExpr::Compare {
                    field: "score".to_string(),
                    op: crate::CompareOp::Lte(30.0),
                })
                .unwrap();
            assert_eq!(pl_lte.len(), 3);

            let pl_lt = index
                .evaluate_query(&ASTExpr::Compare {
                    field: "score".to_string(),
                    op: crate::CompareOp::Lt(30.0),
                })
                .unwrap();
            assert_eq!(pl_lt.len(), 2);
        }

        #[test]
        fn test_range_query_combined_with_and() {
            let mut index = make_index();
            for (id, age) in [(1, 10), (2, 25), (3, 30), (4, 35), (5, 40), (6, 50)] {
                let mut attrs = Attributes::new();
                attrs.insert(
                    "age".to_string(),
                    AttributeValue::try_from(&json!(age)).unwrap(),
                );
                index.insert(id, &attrs).unwrap();
            }

            let expr = ASTExpr::And(vec![
                ASTExpr::Compare {
                    field: "age".to_string(),
                    op: crate::CompareOp::Gte(25.0),
                },
                ASTExpr::Compare {
                    field: "age".to_string(),
                    op: crate::CompareOp::Lt(40.0),
                },
            ]);

            let pl = index.evaluate_query(&expr).unwrap();
            assert_eq!(pl.len(), 3);
            assert!(pl.contains(2));
            assert!(pl.contains(3));
            assert!(pl.contains(4));
            assert!(!pl.contains(1));
            assert!(!pl.contains(5));
            assert!(!pl.contains(6));
        }

        #[test]
        fn test_batch_delete() {
            let mut index = make_index();
            // Insert multiple documents
            for id in 1..=5 {
                let mut attrs = Attributes::new();
                attrs.insert(
                    "tag".to_string(),
                    AttributeValue::try_from(&json!("test")).unwrap(),
                );
                index.insert(id, &attrs).unwrap();
            }

            // Verify all documents are present
            let test_value = AttributeValue::try_from(&json!("test")).unwrap();
            let pl = index.get_posting_list("tag", &test_value).unwrap().unwrap();
            assert_eq!(pl.len(), 5);

            // Batch delete documents 2, 3, and 4
            index.batch_delete(&[2, 3, 4]).unwrap();

            // Verify only documents 1 and 5 remain
            let pl = index.get_posting_list("tag", &test_value).unwrap().unwrap();
            assert_eq!(pl.len(), 2);
            assert!(pl.contains(1));
            assert!(!pl.contains(2));
            assert!(!pl.contains(3));
            assert!(!pl.contains(4));
            assert!(pl.contains(5));
        }

        #[test]
        fn test_batch_update() {
            let mut index = make_index();
            // Insert documents with color "red"
            for id in 1..=3 {
                let mut attrs = Attributes::new();
                attrs.insert(
                    "color".to_string(),
                    AttributeValue::try_from(&json!("red")).unwrap(),
                );
                index.insert(id, &attrs).unwrap();
            }

            // Batch update documents 1 and 3 to color "blue"
            let mut blue_attrs = Attributes::new();
            blue_attrs.insert(
                "color".to_string(),
                AttributeValue::try_from(&json!("blue")).unwrap(),
            );

            let updates = vec![(1, blue_attrs.clone()), (3, blue_attrs.clone())];
            index.batch_update(&updates).unwrap();

            // Verify document 2 is still red
            let red_value = AttributeValue::try_from(&json!("red")).unwrap();
            let blue_value = AttributeValue::try_from(&json!("blue")).unwrap();
            let red_pl = index
                .get_posting_list("color", &red_value)
                .unwrap()
                .unwrap();
            assert_eq!(red_pl.len(), 1);
            assert!(red_pl.contains(2));

            // Verify documents 1 and 3 are now blue
            let blue_pl = index
                .get_posting_list("color", &blue_value)
                .unwrap()
                .unwrap();
            assert_eq!(blue_pl.len(), 2);
            assert!(blue_pl.contains(1));
            assert!(blue_pl.contains(3));
        }
    }
}
