/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! QueryEvaluator trait implementation for GenericIndex.
//!
//! # Range Query Strategy
//!
//! This implementation maintains separate index spaces for integers and floats to:
//! 1. Preserve exact integer semantics (no floating point errors)
//! 2. Enable efficient range scans without type mixing
//! 3. Allow queries to match both numeric types when appropriate
//!
//! For example, a query like `age >= 18` will check both:
//! - Integer keys: "age|I|18" -> "age|I|\u{10FFFF}"
//! - Float keys: "age|F|18.0" -> "age|F|\u{10FFFF}"
//!
//! # NOT/NE Operations
//!
//! The NOT and NE operations are currently unsupported

use crate::attribute::AttributeValue;
use crate::parser::ast::CompareOp;
use crate::traits::key_codec::KeyCodec;
use crate::traits::kv_store_traits::KvStore;
use crate::traits::posting_list_trait::{PostingList, PostingListAccessor};
use crate::traits::query_evaluator::QueryEvaluator;
use crate::ASTExpr;

use super::error::QueryError;
use super::generic_index::{GenericIndex, DATA_TYPE_POSTING_LIST};

impl<S, PL, K> QueryEvaluator for GenericIndex<S, PL, K>
where
    S: KvStore,
    PL: PostingList,
    K: KeyCodec + Default,
    S::Error: std::error::Error + Send + Sync + 'static,
    PL::Error: std::error::Error + Send + Sync + 'static,
{
    type Error = QueryError;
    type PostingList = PL;
    type DocId = usize;

    fn evaluate_query(&self, expr: &ASTExpr) -> Result<Self::PostingList, Self::Error> {
        match expr {
            ASTExpr::And(subs) => {
                if subs.is_empty() {
                    return Ok(PL::empty());
                }
                let mut acc = self.evaluate_query(&subs[0])?;
                for sub in subs.iter().skip(1) {
                    let result = self.evaluate_query(sub)?;
                    acc = acc.intersect(&result);
                    if acc.is_empty() {
                        break; // Early termination
                    }
                }
                Ok(acc)
            }
            ASTExpr::Or(subs) => {
                let mut acc = PL::empty();
                for sub in subs {
                    let result = self.evaluate_query(sub)?;
                    acc = acc.union(&result);
                }
                Ok(acc)
            }
            ASTExpr::Not(_) => {
                // NOT operation requires a document universe to compute the complement
                Err(QueryError::unsupported(
                    "NOT",
                    "NOT operation requires document universe tracking. \
                     Use De Morgan's laws to rewrite as AND/OR, or implement document tracking.",
                ))
            }
            ASTExpr::Compare { field, op } => match op {
                CompareOp::Eq(v) => {
                    // Try to convert the JSON value to AttributeValue
                    let attr_value = match AttributeValue::try_from(v) {
                        Ok(av) => av,
                        Err(e) => {
                            return Err(QueryError::invalid_query(field, e.to_string()));
                        }
                    };

                    // Normalize query field using configurable normalizer
                    let normalized_field = self.normalize_field(field);
                    match self.get_posting_list(&normalized_field, &attr_value)? {
                        Some(pl) => Ok(pl),
                        None => Ok(PL::empty()),
                    }
                }
                CompareOp::Ne(_) => {
                    // NE operation requires a document universe to compute the complement
                    Err(QueryError::unsupported(
                        "NE",
                        "NE operation requires document universe tracking. \
                         Rewrite query using complemented AND/OR logic.",
                    ))
                }
                CompareOp::Gte(_) | CompareOp::Gt(_) | CompareOp::Lte(_) | CompareOp::Lt(_) => {
                    // Range query - scan both integer and float ranges
                    let mut result = PL::empty();

                    // Normalize the field name for range queries
                    let normalized_field = self.normalize_field(field);

                    // Try integer range first
                    let (start_key, end_key) =
                        Self::get_integer_range_bounds(&normalized_field, op);
                    if !start_key.is_empty() {
                        let iter = self
                            .store
                            .range(&start_key[..]..&end_key[..])
                            .map_err(|e| {
                                QueryError::from(crate::kv_index::error::IndexError::kv_store(
                                    "range", e,
                                ))
                            })?;

                        for item in iter {
                            let (_, value) = item.map_err(|e| {
                                QueryError::from(crate::kv_index::error::IndexError::kv_store(
                                    "range_next",
                                    e,
                                ))
                            })?;
                            let pl = PL::deserialize(&value).map_err(|e| {
                                QueryError::from(crate::kv_index::error::IndexError::serialization(
                                    DATA_TYPE_POSTING_LIST,
                                    e,
                                ))
                            })?;
                            result = result.union(&pl);
                        }
                    }

                    // Try float range
                    let (start_key, end_key) = Self::get_float_range_bounds(&normalized_field, op);
                    if !start_key.is_empty() {
                        let iter = self
                            .store
                            .range(&start_key[..]..&end_key[..])
                            .map_err(|e| {
                                QueryError::from(crate::kv_index::error::IndexError::kv_store(
                                    "range", e,
                                ))
                            })?;

                        for item in iter {
                            let (_, value) = item.map_err(|e| {
                                QueryError::from(crate::kv_index::error::IndexError::kv_store(
                                    "range_next",
                                    e,
                                ))
                            })?;
                            let pl = PL::deserialize(&value).map_err(|e| {
                                QueryError::from(crate::kv_index::error::IndexError::serialization(
                                    DATA_TYPE_POSTING_LIST,
                                    e,
                                ))
                            })?;
                            result = result.union(&pl);
                        }
                    }

                    Ok(result)
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_index::GenericIndex;
    use crate::traits::inverted_index_trait::InvertedIndexProvider;
    use crate::traits::key_codec::DefaultKeyCodec;
    use crate::traits::kv_store_traits::KvStore;
    use crate::traits::posting_list_trait::RoaringPostingList;
    use crate::utils::flatten_utils::Attributes;
    use hashbrown::HashMap;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    // DummyKvStore for testing
    #[derive(Default)]
    struct DummyKvStore {
        map: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
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

        fn batch_set(&self, entries: &[(&[u8], &[u8])]) -> std::result::Result<(), Self::Error> {
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

    type TestIndex = GenericIndex<DummyKvStore, RoaringPostingList, DefaultKeyCodec>;

    fn make_index() -> TestIndex {
        let store = Arc::new(DummyKvStore::default());
        GenericIndex::new(store)
    }

    #[test]
    fn test_evaluate_query_eq_string() {
        let mut index = make_index();

        // Insert documents
        let mut attrs1 = Attributes::new();
        attrs1.insert(
            "color".to_string(),
            AttributeValue::try_from(&json!("red")).unwrap(),
        );
        index.insert(1, &attrs1).unwrap();

        let mut attrs2 = Attributes::new();
        attrs2.insert(
            "color".to_string(),
            AttributeValue::try_from(&json!("blue")).unwrap(),
        );
        index.insert(2, &attrs2).unwrap();

        // Query for color = "red"
        let expr = ASTExpr::Compare {
            field: "color".to_string(),
            op: CompareOp::Eq(json!("red")),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(1));
        assert!(!result.contains(2));
    }

    #[test]
    fn test_evaluate_query_eq_integer() {
        let mut index = make_index();

        let mut attrs1 = Attributes::new();
        attrs1.insert(
            "age".to_string(),
            AttributeValue::try_from(&json!(25)).unwrap(),
        );
        index.insert(1, &attrs1).unwrap();

        let mut attrs2 = Attributes::new();
        attrs2.insert(
            "age".to_string(),
            AttributeValue::try_from(&json!(30)).unwrap(),
        );
        index.insert(2, &attrs2).unwrap();

        let expr = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Eq(json!(25)),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(1));
    }

    #[test]
    fn test_evaluate_query_eq_float() {
        let mut index = make_index();

        let mut attrs = Attributes::new();
        attrs.insert(
            "price".to_string(),
            AttributeValue::try_from(&json!(19.99)).unwrap(),
        );
        index.insert(1, &attrs).unwrap();

        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Eq(json!(19.99)),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(1));
    }

    #[test]
    fn test_evaluate_query_eq_bool() {
        let mut index = make_index();

        let mut attrs = Attributes::new();
        attrs.insert(
            "active".to_string(),
            AttributeValue::try_from(&json!(true)).unwrap(),
        );
        index.insert(1, &attrs).unwrap();

        let expr = ASTExpr::Compare {
            field: "active".to_string(),
            op: CompareOp::Eq(json!(true)),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(1));
    }

    #[test]
    fn test_evaluate_query_eq_invalid_nan() {
        let index = make_index();

        let expr = ASTExpr::Compare {
            field: "value".to_string(),
            op: CompareOp::Eq(json!(f64::NAN)),
        };

        let result = index.evaluate_query(&expr);
        assert!(result.is_err());
        // NaN in JSON becomes null, which is unsupported
        match result {
            Err(QueryError::InvalidQuery { .. }) => {
                // Success - we expect an InvalidQuery error
            }
            _ => panic!("Expected InvalidQuery error, got: {:?}", result),
        }
    }

    #[test]
    fn test_evaluate_query_eq_invalid_infinity() {
        let index = make_index();

        let expr = ASTExpr::Compare {
            field: "value".to_string(),
            op: CompareOp::Eq(json!(f64::INFINITY)),
        };

        let result = index.evaluate_query(&expr);
        assert!(result.is_err());
        match result {
            Err(QueryError::InvalidQuery { .. }) => {}
            _ => panic!("Expected InvalidQuery error"),
        }
    }

    #[test]
    fn test_evaluate_query_eq_unsupported_null() {
        let index = make_index();

        let expr = ASTExpr::Compare {
            field: "value".to_string(),
            op: CompareOp::Eq(json!(null)),
        };

        let result = index.evaluate_query(&expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_query_and_basic() {
        let mut index = make_index();

        let mut attrs1 = Attributes::new();
        attrs1.insert(
            "color".to_string(),
            AttributeValue::try_from(&json!("red")).unwrap(),
        );
        attrs1.insert(
            "size".to_string(),
            AttributeValue::try_from(&json!("large")).unwrap(),
        );
        index.insert(1, &attrs1).unwrap();

        let mut attrs2 = Attributes::new();
        attrs2.insert(
            "color".to_string(),
            AttributeValue::try_from(&json!("red")).unwrap(),
        );
        attrs2.insert(
            "size".to_string(),
            AttributeValue::try_from(&json!("small")).unwrap(),
        );
        index.insert(2, &attrs2).unwrap();

        // Query: color = "red" AND size = "large"
        let expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "color".to_string(),
                op: CompareOp::Eq(json!("red")),
            },
            ASTExpr::Compare {
                field: "size".to_string(),
                op: CompareOp::Eq(json!("large")),
            },
        ]);

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(1));
        assert!(!result.contains(2));
    }

    #[test]
    fn test_evaluate_query_and_empty() {
        let index = make_index();

        let expr = ASTExpr::And(vec![]);
        let result = index.evaluate_query(&expr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_query_and_early_termination() {
        let mut index = make_index();

        let mut attrs = Attributes::new();
        attrs.insert(
            "field1".to_string(),
            AttributeValue::try_from(&json!("value1")).unwrap(),
        );
        index.insert(1, &attrs).unwrap();

        // Query: field1 = "value1" AND field2 = "nonexistent"
        // Should short-circuit when field2 returns empty
        let expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "field1".to_string(),
                op: CompareOp::Eq(json!("value1")),
            },
            ASTExpr::Compare {
                field: "field2".to_string(),
                op: CompareOp::Eq(json!("nonexistent")),
            },
        ]);

        let result = index.evaluate_query(&expr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_query_or_basic() {
        let mut index = make_index();

        let mut attrs1 = Attributes::new();
        attrs1.insert(
            "color".to_string(),
            AttributeValue::try_from(&json!("red")).unwrap(),
        );
        index.insert(1, &attrs1).unwrap();

        let mut attrs2 = Attributes::new();
        attrs2.insert(
            "color".to_string(),
            AttributeValue::try_from(&json!("blue")).unwrap(),
        );
        index.insert(2, &attrs2).unwrap();

        // Query: color = "red" OR color = "blue"
        let expr = ASTExpr::Or(vec![
            ASTExpr::Compare {
                field: "color".to_string(),
                op: CompareOp::Eq(json!("red")),
            },
            ASTExpr::Compare {
                field: "color".to_string(),
                op: CompareOp::Eq(json!("blue")),
            },
        ]);

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains(1));
        assert!(result.contains(2));
    }

    #[test]
    fn test_evaluate_query_or_empty() {
        let index = make_index();

        let expr = ASTExpr::Or(vec![]);
        let result = index.evaluate_query(&expr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_query_not_unsupported() {
        let index = make_index();

        let expr = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: "field".to_string(),
            op: CompareOp::Eq(json!("value")),
        }));

        let result = index.evaluate_query(&expr);
        assert!(result.is_err());
        match result {
            Err(QueryError::UnsupportedOperation { operation, .. }) => {
                assert_eq!(operation, "NOT");
            }
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    #[test]
    fn test_evaluate_query_ne_unsupported() {
        let index = make_index();

        let expr = ASTExpr::Compare {
            field: "field".to_string(),
            op: CompareOp::Ne(json!("value")),
        };

        let result = index.evaluate_query(&expr);
        assert!(result.is_err());
        match result {
            Err(QueryError::UnsupportedOperation { operation, .. }) => {
                assert_eq!(operation, "NE");
            }
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    #[test]
    fn test_evaluate_query_gte_integer() {
        let mut index = make_index();

        for age in [10, 20, 30, 40, 50] {
            let mut attrs = Attributes::new();
            attrs.insert(
                "age".to_string(),
                AttributeValue::try_from(&json!(age)).unwrap(),
            );
            index.insert(age, &attrs).unwrap();
        }

        let expr = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Gte(30.0),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 3); // 30, 40, 50
        assert!(result.contains(30));
        assert!(result.contains(40));
        assert!(result.contains(50));
        assert!(!result.contains(10));
        assert!(!result.contains(20));
    }

    #[test]
    fn test_evaluate_query_gt_integer() {
        let mut index = make_index();

        for age in [10, 20, 30, 40] {
            let mut attrs = Attributes::new();
            attrs.insert(
                "age".to_string(),
                AttributeValue::try_from(&json!(age)).unwrap(),
            );
            index.insert(age, &attrs).unwrap();
        }

        let expr = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Gt(20.0),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 2); // 30, 40
        assert!(result.contains(30));
        assert!(result.contains(40));
        assert!(!result.contains(20));
    }

    #[test]
    fn test_evaluate_query_lte_integer() {
        let mut index = make_index();

        for age in [10, 20, 30, 40] {
            let mut attrs = Attributes::new();
            attrs.insert(
                "age".to_string(),
                AttributeValue::try_from(&json!(age)).unwrap(),
            );
            index.insert(age, &attrs).unwrap();
        }

        let expr = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Lte(30.0),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 3); // 10, 20, 30
        assert!(result.contains(10));
        assert!(result.contains(20));
        assert!(result.contains(30));
        assert!(!result.contains(40));
    }

    #[test]
    fn test_evaluate_query_lt_integer() {
        let mut index = make_index();

        for age in [10, 20, 30, 40] {
            let mut attrs = Attributes::new();
            attrs.insert(
                "age".to_string(),
                AttributeValue::try_from(&json!(age)).unwrap(),
            );
            index.insert(age, &attrs).unwrap();
        }

        let expr = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Lt(30.0),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 2); // 10, 20
        assert!(result.contains(10));
        assert!(result.contains(20));
        assert!(!result.contains(30));
    }

    #[test]
    fn test_evaluate_query_range_float() {
        let mut index = make_index();

        for (id, price) in [(1, 9.99), (2, 19.99), (3, 29.99), (4, 39.99)] {
            let mut attrs = Attributes::new();
            attrs.insert(
                "price".to_string(),
                AttributeValue::try_from(&json!(price)).unwrap(),
            );
            index.insert(id, &attrs).unwrap();
        }

        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Gte(20.0),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 2); // 29.99, 39.99
        assert!(result.contains(3));
        assert!(result.contains(4));
    }

    #[test]
    fn test_evaluate_query_range_mixed_int_float() {
        let mut index = make_index();

        // Insert integers
        let mut attrs1 = Attributes::new();
        attrs1.insert(
            "value".to_string(),
            AttributeValue::try_from(&json!(10)).unwrap(),
        );
        index.insert(1, &attrs1).unwrap();

        // Insert floats
        let mut attrs2 = Attributes::new();
        attrs2.insert(
            "value".to_string(),
            AttributeValue::try_from(&json!(15.5)).unwrap(),
        );
        index.insert(2, &attrs2).unwrap();

        let mut attrs3 = Attributes::new();
        attrs3.insert(
            "value".to_string(),
            AttributeValue::try_from(&json!(20)).unwrap(),
        );
        index.insert(3, &attrs3).unwrap();

        // Query >= 15
        let expr = ASTExpr::Compare {
            field: "value".to_string(),
            op: CompareOp::Gte(15.0),
        };

        let result = index.evaluate_query(&expr).unwrap();
        // Should match both 15.5 (float) and 20 (int)
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(!result.contains(1));
    }

    #[test]
    fn test_evaluate_query_complex_nested() {
        let mut index = make_index();

        let mut attrs1 = Attributes::new();
        attrs1.insert(
            "type".to_string(),
            AttributeValue::try_from(&json!("A")).unwrap(),
        );
        attrs1.insert(
            "status".to_string(),
            AttributeValue::try_from(&json!("active")).unwrap(),
        );
        index.insert(1, &attrs1).unwrap();

        let mut attrs2 = Attributes::new();
        attrs2.insert(
            "type".to_string(),
            AttributeValue::try_from(&json!("B")).unwrap(),
        );
        attrs2.insert(
            "status".to_string(),
            AttributeValue::try_from(&json!("active")).unwrap(),
        );
        index.insert(2, &attrs2).unwrap();

        let mut attrs3 = Attributes::new();
        attrs3.insert(
            "type".to_string(),
            AttributeValue::try_from(&json!("A")).unwrap(),
        );
        attrs3.insert(
            "status".to_string(),
            AttributeValue::try_from(&json!("inactive")).unwrap(),
        );
        index.insert(3, &attrs3).unwrap();

        // Query: (type = "A" OR type = "B") AND status = "active"
        let expr = ASTExpr::And(vec![
            ASTExpr::Or(vec![
                ASTExpr::Compare {
                    field: "type".to_string(),
                    op: CompareOp::Eq(json!("A")),
                },
                ASTExpr::Compare {
                    field: "type".to_string(),
                    op: CompareOp::Eq(json!("B")),
                },
            ]),
            ASTExpr::Compare {
                field: "status".to_string(),
                op: CompareOp::Eq(json!("active")),
            },
        ]);

        let result = index.evaluate_query(&expr).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains(1)); // type=A, status=active
        assert!(result.contains(2)); // type=B, status=active
        assert!(!result.contains(3)); // type=A, status=inactive
    }

    #[test]
    fn test_evaluate_query_nonexistent_field() {
        let mut index = make_index();

        let mut attrs = Attributes::new();
        attrs.insert(
            "field1".to_string(),
            AttributeValue::try_from(&json!("value")).unwrap(),
        );
        index.insert(1, &attrs).unwrap();

        // Query for a field that doesn't exist
        let expr = ASTExpr::Compare {
            field: "nonexistent".to_string(),
            op: CompareOp::Eq(json!("value")),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_query_empty_index() {
        let index = make_index();

        let expr = ASTExpr::Compare {
            field: "field".to_string(),
            op: CompareOp::Eq(json!("value")),
        };

        let result = index.evaluate_query(&expr).unwrap();
        assert!(result.is_empty());
    }
}
