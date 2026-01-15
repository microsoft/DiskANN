/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Generic inverted index implementation.

use crate::attribute::AttributeValue;
use crate::parser::ast::CompareOp;
use crate::traits::key_codec::{KeyCodec, FIELD_SEPARATOR, TYPE_PREFIX_FLOAT, TYPE_PREFIX_INTEGER};
use crate::traits::kv_store_traits::KvStore;
use crate::traits::posting_list_trait::PostingList;
use std::marker::PhantomData;
use std::sync::Arc;

use super::error::{IndexError, Result};

// Constants for error messages and serialization locations
const LOCATION_REVERSE_KEY_LIST: &str = "reverse_key_list";
const LOCATION_SERIALIZE_KEY_LIST: &str = "serialize_key_list";
pub(crate) const DATA_TYPE_POSTING_LIST: &str = "posting_list";

/// A generic inverted index implementation.
///
/// This index can work with any combination of:
/// - KvStore backend (BfTree, Memory, RocksDB, etc.)
/// - PostingList implementation (RoaringBitmap, BitSet, etc.)
/// - KeyCodec strategy (DefaultKeyCodec, custom encodings, etc.)
///
/// # Type Parameters
///
/// * `S` - The KvStore backend implementation
/// * `PL` - The PostingList implementation
/// * `K` - The KeyCodec strategy
///
/// # Example
///
/// ```ignore
/// use diskann_label_filter::index::GenericIndex;
/// use diskann_label_filter::stores::bftree_store::BfTreeStore;
/// use diskann_label_filter::traits::posting_list_provider::RoaringPostingList;
/// use diskann_label_filter::traits::key_codec::DefaultKeyCodec;
///
/// let store = Arc::new(BfTreeStore::open("./data/index")?);
/// let index = GenericIndex::<BfTreeStore, RoaringPostingList, DefaultKeyCodec>::new(store);
/// ```
pub struct GenericIndex<S: KvStore, PL: PostingList, K: KeyCodec + Default> {
    pub(crate) store: Arc<S>,
    pub(crate) _pl: PhantomData<PL>,
    pub(crate) _kc: PhantomData<K>,
    /// Optional field normalizer function for query processing
    #[allow(clippy::type_complexity)]
    pub(crate) field_normalizer: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
}

impl<S, PL, K> GenericIndex<S, PL, K>
where
    S: KvStore,
    PL: PostingList,
    K: KeyCodec + Default,
{
    /// Creates a new GenericIndex with the provided KvStore.
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            _pl: PhantomData,
            _kc: PhantomData,
            field_normalizer: None,
        }
    }

    /// Set a custom field normalizer function for query processing.
    ///
    /// The normalizer function converts field names from query expressions
    /// into the format used for indexing (e.g., "field.nested" -> "/field/nested").
    ///
    /// # Example
    ///
    /// ```ignore
    /// let index = GenericIndex::new(store)
    ///     .with_field_normalizer(|field| format!("/{}", field.replace(".", "/")));
    /// ```
    pub fn with_field_normalizer<F>(mut self, normalizer: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.field_normalizer = Some(Arc::new(normalizer));
        self
    }

    /// Normalize field name using custom function or default logic.
    ///
    /// Default normalization converts dot notation to JSON pointer format:
    /// "field.nested" -> "/field/nested"
    pub(crate) fn normalize_field(&self, field: &str) -> String {
        if let Some(normalizer) = &self.field_normalizer {
            normalizer(field)
        } else {
            field.to_string() // can do Cow optimaztion later, but now we assume we need to do the normalize for query so no need to optimzie
        }
    }

    /// Returns a reference to the underlying KvStore.
    pub fn store(&self) -> &Arc<S> {
        &self.store
    }

    /// Generates the reverse mapping key for a document ID.
    ///
    /// Format: "@R:{doc_id}"
    pub(crate) fn reverse_key(doc_id: usize) -> Vec<u8> {
        format!("@R:{}", doc_id).into_bytes()
    }

    /// Serializes a list of keys for reverse mapping storage.
    pub(crate) fn serialize_key_list(keys: &[Vec<u8>]) -> Result<Vec<u8>> {
        // Calculate total size with overflow protection
        let total_size = keys
            .iter()
            .try_fold(4usize, |acc, k| {
                let with_len = acc.checked_add(4)?;
                with_len.checked_add(k.len())
            })
            .ok_or_else(|| IndexError::overflow(LOCATION_SERIALIZE_KEY_LIST))?;

        let mut out = Vec::with_capacity(total_size);
        let count = keys.len() as u32;
        out.extend_from_slice(&count.to_le_bytes());
        for k in keys {
            let len = k.len() as u32;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(k);
        }
        Ok(out)
    }

    /// Deserializes a list of keys from reverse mapping storage.
    pub(crate) fn deserialize_key_list(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
        if bytes.len() < 4 {
            return Err(IndexError::corrupt_data(
                LOCATION_REVERSE_KEY_LIST,
                "buffer too short for header (need 4 bytes)",
            ));
        }

        let count_bytes = bytes[0..4].try_into().map_err(|_| {
            IndexError::corrupt_data(LOCATION_REVERSE_KEY_LIST, "invalid count bytes")
        })?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut pos = 4;
        let mut keys = Vec::with_capacity(count);

        for i in 0..count {
            if pos + 4 > bytes.len() {
                return Err(IndexError::corrupt_data(
                    LOCATION_REVERSE_KEY_LIST,
                    format!("length header overflow at key {}", i),
                ));
            }

            let len_bytes = bytes[pos..pos + 4].try_into().map_err(|_| {
                IndexError::corrupt_data(
                    LOCATION_REVERSE_KEY_LIST,
                    format!("invalid length bytes at key {}", i),
                )
            })?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            pos += 4;

            if pos + len > bytes.len() {
                return Err(IndexError::corrupt_data(
                    LOCATION_REVERSE_KEY_LIST,
                    format!(
                        "data overflow at key {} (expected {} bytes, have {})",
                        i,
                        len,
                        bytes.len() - pos
                    ),
                ));
            }

            keys.push(bytes[pos..pos + len].to_vec());
            pos += len;
        }
        Ok(keys)
    }

    /// Helper to get a posting list from the store, or create an empty one.
    pub(crate) fn get_or_empty_posting_list(&self, key: &[u8]) -> Result<PL> {
        let kv_result = self
            .store
            .get(key)
            .map_err(|e| IndexError::kv_store("get", e))?;

        match kv_result {
            Some(bytes) => PL::deserialize(&bytes)
                .map_err(|e| IndexError::serialization(DATA_TYPE_POSTING_LIST, e)),
            None => Ok(PL::empty()),
        }
    }

    /// Helper to extract numeric value from CompareOp
    pub(crate) fn get_compare_value(op: &CompareOp) -> Option<f64> {
        match op {
            CompareOp::Eq(v) | CompareOp::Ne(v) => v.as_f64(),
            CompareOp::Lt(v) | CompareOp::Lte(v) | CompareOp::Gt(v) | CompareOp::Gte(v) => Some(*v),
        }
    }

    /// Helper to get start/end keys for integer range queries.
    ///
    /// # Range Semantics
    ///
    /// Rust ranges are exclusive on the end, so:
    /// - `>=` (Gte): range from value to max → [value, ∞)
    /// - `>` (Gt): range from value+1 to max → (value, ∞) = [value+1, ∞)
    /// - `<=` (Lte): range from min to value+1 → (-∞, value] = (-∞, value+1)
    /// - `<` (Lt): range from min to value → (-∞, value)
    ///
    /// # Returns
    ///
    /// Returns (start_key, end_key) tuple. If the operation is not a range op,
    /// returns (empty, empty) to signal no range scan needed.
    pub(crate) fn get_integer_range_bounds(field: &str, op: &CompareOp) -> (Vec<u8>, Vec<u8>) {
        let codec = K::default();
        let value = match Self::get_compare_value(op) {
            Some(v) => v,
            None => return (vec![], vec![]),
        };

        match op {
            CompareOp::Gte(_) | CompareOp::Gt(_) => {
                let val = value as i64;
                // For Gt, start from value+1 (exclusive lower bound)
                // For Gte, start from value (inclusive lower bound)
                let adjusted_val = if matches!(op, CompareOp::Gt(_)) {
                    val + 1
                } else {
                    val
                };
                let start_key =
                    codec.encode_field_value_key(field, &AttributeValue::Integer(adjusted_val));
                // Upper bound: maximum Unicode code point for this field+type prefix
                let end_key = format!(
                    "{}{}{}{}",
                    field, FIELD_SEPARATOR, TYPE_PREFIX_INTEGER, "\u{10FFFF}"
                )
                .into_bytes();
                (start_key, end_key)
            }
            CompareOp::Lte(_) | CompareOp::Lt(_) => {
                let val = value as i64;
                // For Lte, end at value+1 (exclusive upper bound = inclusive value)
                // For Lt, end at value (exclusive upper bound)
                let adjusted_val = if matches!(op, CompareOp::Lte(_)) {
                    val + 1
                } else {
                    val
                };
                let start_key =
                    format!("{}{}{}", field, FIELD_SEPARATOR, TYPE_PREFIX_INTEGER).into_bytes();
                let end_key =
                    codec.encode_field_value_key(field, &AttributeValue::Integer(adjusted_val));
                (start_key, end_key)
            }
            _ => (vec![], vec![]),
        }
    }

    /// Helper to get start/end keys for float range queries.
    ///
    /// # Range Semantics
    ///
    /// Similar to integer ranges, but without the +1 adjustment since
    /// floats don't have a meaningful "next" value. We rely on the
    /// lexicographic ordering of the encoded float representation.
    ///
    /// # Returns
    ///
    /// Returns (start_key, end_key) tuple. If the value is NaN/infinity
    /// or the operation is not a range op, returns (empty, empty).
    pub(crate) fn get_float_range_bounds(field: &str, op: &CompareOp) -> (Vec<u8>, Vec<u8>) {
        let codec = K::default();
        let value = match Self::get_compare_value(op) {
            Some(v) => v,
            None => return (vec![], vec![]),
        };

        match op {
            CompareOp::Gte(_) | CompareOp::Gt(_) => {
                // Note: If from_f64 returns None (for NaN/infinity), we return empty bounds
                // The caller will handle this by returning an empty result set
                if let Some(_num) = serde_json::Number::from_f64(value) {
                    let start_key =
                        codec.encode_field_value_key(field, &AttributeValue::Real(value));
                    // Upper bound: maximum Unicode code point for this field+type prefix
                    let end_key = format!(
                        "{}{}{}{}",
                        field, FIELD_SEPARATOR, TYPE_PREFIX_FLOAT, "\u{10FFFF}"
                    )
                    .into_bytes();
                    (start_key, end_key)
                } else {
                    (vec![], vec![])
                }
            }
            CompareOp::Lte(_) | CompareOp::Lt(_) => {
                if let Some(_num) = serde_json::Number::from_f64(value) {
                    let start_key =
                        format!("{}{}{}", field, FIELD_SEPARATOR, TYPE_PREFIX_FLOAT).into_bytes();
                    let end_key = codec.encode_field_value_key(field, &AttributeValue::Real(value));
                    (start_key, end_key)
                } else {
                    (vec![], vec![])
                }
            }
            _ => (vec![], vec![]),
        }
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::stores::bftree_store::BfTreeStore;
    use crate::traits::key_codec::DefaultKeyCodec;
    use crate::traits::posting_list_trait::RoaringPostingList;
    use serde_json::json;

    // Type alias for tests (BfTreeStore requires a path, so we use methods that don't need an instance)
    type TestIndex = GenericIndex<BfTreeStore, RoaringPostingList, DefaultKeyCodec>;

    #[test]
    fn test_reverse_key_format() {
        let key = TestIndex::reverse_key(42);
        assert_eq!(key, b"@R:42");

        let key = TestIndex::reverse_key(0);
        assert_eq!(key, b"@R:0");

        let key = TestIndex::reverse_key(99999);
        assert_eq!(key, b"@R:99999");
    }

    #[test]
    fn test_serialize_deserialize_key_list_empty() {
        let keys: Vec<Vec<u8>> = vec![];
        let serialized =
            TestIndex::serialize_key_list(&keys).expect("serialization should succeed");

        assert_eq!(serialized.len(), 4); // Just the count header
        assert_eq!(&serialized[0..4], &[0, 0, 0, 0]); // count = 0

        let deserialized =
            TestIndex::deserialize_key_list(&serialized).expect("deserialization should succeed");
        assert_eq!(deserialized.len(), 0);
    }

    #[test]
    fn test_serialize_deserialize_key_list_single() {
        let keys = vec![b"test_key".to_vec()];
        let serialized =
            TestIndex::serialize_key_list(&keys).expect("serialization should succeed");

        let deserialized =
            TestIndex::deserialize_key_list(&serialized).expect("deserialization should succeed");
        assert_eq!(deserialized.len(), 1);
        assert_eq!(deserialized[0], b"test_key");
    }

    #[test]
    fn test_serialize_deserialize_key_list_multiple() {
        let keys = vec![
            b"key1".to_vec(),
            b"key2_longer".to_vec(),
            b"k3".to_vec(),
            vec![0u8, 1, 2, 255], // Binary data
        ];

        let serialized =
            TestIndex::serialize_key_list(&keys).expect("serialization should succeed");

        let deserialized =
            TestIndex::deserialize_key_list(&serialized).expect("deserialization should succeed");

        assert_eq!(deserialized.len(), 4);
        assert_eq!(deserialized[0], b"key1");
        assert_eq!(deserialized[1], b"key2_longer");
        assert_eq!(deserialized[2], b"k3");
        assert_eq!(deserialized[3], vec![0u8, 1, 2, 255]);
    }

    #[test]
    fn test_serialize_deserialize_key_list_empty_keys() {
        let keys = vec![vec![], b"non_empty".to_vec(), vec![]];

        let serialized =
            TestIndex::serialize_key_list(&keys).expect("serialization should succeed");

        let deserialized =
            TestIndex::deserialize_key_list(&serialized).expect("deserialization should succeed");

        assert_eq!(deserialized.len(), 3);
        assert_eq!(deserialized[0], Vec::<u8>::new());
        assert_eq!(deserialized[1], b"non_empty");
        assert_eq!(deserialized[2], Vec::<u8>::new());
    }

    #[test]
    fn test_deserialize_key_list_corrupt_too_short() {
        let bytes = vec![1, 2]; // Less than 4 bytes
        let result = TestIndex::deserialize_key_list(&bytes);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, IndexError::CorruptData { .. }));
        assert!(err.to_string().contains("buffer too short"));
    }

    #[test]
    fn test_deserialize_key_list_corrupt_truncated_length() {
        // Count says 1 key, but no length header
        let bytes = vec![1, 0, 0, 0];
        let result = TestIndex::deserialize_key_list(&bytes);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, IndexError::CorruptData { .. }));
        assert!(err.to_string().contains("length header overflow"));
    }

    #[test]
    fn test_deserialize_key_list_corrupt_truncated_data() {
        // Count says 1 key of length 10, but only 5 bytes provided
        let mut bytes = vec![1, 0, 0, 0]; // count = 1
        bytes.extend_from_slice(&[10, 0, 0, 0]); // length = 10
        bytes.extend_from_slice(b"short"); // only 5 bytes

        let result = TestIndex::deserialize_key_list(&bytes);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, IndexError::CorruptData { .. }));
        assert!(err.to_string().contains("data overflow"));
    }

    #[test]
    fn test_get_compare_value() {
        use serde_json::json;

        // Eq with integer
        let op = CompareOp::Eq(json!(42));
        assert_eq!(TestIndex::get_compare_value(&op), Some(42.0));

        // Eq with float
        let op = CompareOp::Eq(json!(3.14));
        assert_eq!(TestIndex::get_compare_value(&op), Some(3.14));

        // Eq with string (no numeric value)
        let op = CompareOp::Eq(json!("hello"));
        assert_eq!(TestIndex::get_compare_value(&op), None);

        // Lt
        let op = CompareOp::Lt(10.5);
        assert_eq!(TestIndex::get_compare_value(&op), Some(10.5));

        // Gte
        let op = CompareOp::Gte(100.0);
        assert_eq!(TestIndex::get_compare_value(&op), Some(100.0));
    }

    #[test]
    fn test_get_integer_range_bounds_gte() {
        let op = CompareOp::Gte(10.0);
        let (start, end) = TestIndex::get_integer_range_bounds("age", &op);

        assert!(!start.is_empty());
        assert!(!end.is_empty());

        // Start key should encode integer 10
        let start_str = String::from_utf8_lossy(&start);
        assert!(start_str.starts_with("age"));
        assert!(start_str.contains(TYPE_PREFIX_INTEGER));

        // End key should have the max marker
        let end_str = String::from_utf8_lossy(&end);
        assert!(end_str.starts_with("age"));
        assert!(end_str.contains("\u{10FFFF}"));
    }

    #[test]
    fn test_get_integer_range_bounds_gt() {
        let op = CompareOp::Gt(10.0);
        let (start, _) = TestIndex::get_integer_range_bounds("age", &op);

        // GT 10 should start at 11
        let start_str = String::from_utf8_lossy(&start);
        assert!(start_str.starts_with("age"));
        assert!(start_str.contains(TYPE_PREFIX_INTEGER));
    }

    #[test]
    fn test_get_integer_range_bounds_lte() {
        let op = CompareOp::Lte(20.0);
        let (start, end) = TestIndex::get_integer_range_bounds("age", &op);

        assert!(!start.is_empty());
        assert!(!end.is_empty());

        // Start should be at the beginning of integers
        let start_str = String::from_utf8_lossy(&start);
        assert!(start_str.starts_with("age"));
        assert!(start_str.contains(TYPE_PREFIX_INTEGER));

        // End should be at 21 (exclusive)
        let end_str = String::from_utf8_lossy(&end);
        assert!(end_str.starts_with("age"));
    }

    #[test]
    fn test_get_integer_range_bounds_lt() {
        let op = CompareOp::Lte(20.0);
        let (start, end) = TestIndex::get_integer_range_bounds("age", &op);

        assert!(!start.is_empty());
        assert!(!end.is_empty());

        // LT 20 should end at 20 (exclusive)
        let start_str = String::from_utf8_lossy(&start);
        assert!(start_str.starts_with("age"));

        let end_str = String::from_utf8_lossy(&end);
        assert!(end_str.starts_with("age"));
    }

    #[test]
    fn test_get_integer_range_bounds_eq_returns_empty() {
        let op = CompareOp::Eq(json!(42));
        let (start, end) = TestIndex::get_integer_range_bounds("age", &op);

        // Eq is not a range operation
        assert!(start.is_empty());
        assert!(end.is_empty());
    }

    #[test]
    fn test_get_float_range_bounds_gte() {
        let op = CompareOp::Gte(3.14);
        let (start, end) = TestIndex::get_float_range_bounds("price", &op);

        assert!(!start.is_empty());
        assert!(!end.is_empty());

        let start_str = String::from_utf8_lossy(&start);
        assert!(start_str.starts_with("price"));
        assert!(start_str.contains(TYPE_PREFIX_FLOAT));

        let end_str = String::from_utf8_lossy(&end);
        assert!(end_str.contains("\u{10FFFF}"));
    }

    #[test]
    fn test_get_float_range_bounds_lte() {
        let op = CompareOp::Lte(99.9);
        let (start, end) = TestIndex::get_float_range_bounds("price", &op);

        assert!(!start.is_empty());
        assert!(!end.is_empty());

        let start_str = String::from_utf8_lossy(&start);
        assert!(start_str.starts_with("price"));
        assert!(start_str.contains(TYPE_PREFIX_FLOAT));
    }

    #[test]
    fn test_get_float_range_bounds_nan_returns_empty() {
        let op = CompareOp::Gte(f64::NAN);
        let (start, end) = TestIndex::get_float_range_bounds("price", &op);

        // NaN should result in empty bounds
        assert!(start.is_empty());
        assert!(end.is_empty());
    }

    #[test]
    fn test_get_float_range_bounds_infinity_returns_empty() {
        let op = CompareOp::Lte(f64::INFINITY);
        let (start, end) = TestIndex::get_float_range_bounds("price", &op);

        // Infinity should result in empty bounds (from_f64 returns None)
        assert!(start.is_empty());
        assert!(end.is_empty());
    }

    #[test]
    fn test_constants_values() {
        // Verify constants are properly defined
        assert_eq!(LOCATION_REVERSE_KEY_LIST, "reverse_key_list");
        assert_eq!(LOCATION_SERIALIZE_KEY_LIST, "serialize_key_list");
        assert_eq!(DATA_TYPE_POSTING_LIST, "posting_list");
    }
}
