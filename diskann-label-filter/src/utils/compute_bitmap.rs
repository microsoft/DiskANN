/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::AttributeValue;
use crate::parser::evaluator::eval_query_expr;
use crate::utils::flatten_utils::{flatten_json_pointers_with_config, FlattenConfig};
use crate::{parser::format::Document, ASTExpr, CompareOp};
use bit_set::BitSet;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::mem::discriminant;
use std::ops::Bound::{Excluded, Included, Unbounded};

struct NotNonNan;

impl std::fmt::Display for NotNonNan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NotNonNan")
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct OrderedFloat(f64);

impl OrderedFloat {
    pub fn new(v: f64) -> Result<Self, NotNonNan> {
        if v.is_nan() {
            Err(NotNonNan)
        } else {
            Ok(Self(v))
        }
    }
}

impl Eq for OrderedFloat {}
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        // By construction, we know the partial comparison will succeed.
        // Return `Eq` if it doesn't for better code-gen.
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

enum QueryAccelerator {
    InvertedIndex(HashMap<AttributeValue, BitSet>),
    BTree(BTreeMap<OrderedFloat, Vec<usize>>),
}

fn check_for_disallowed_operators(query_expr: &ASTExpr) -> bool {
    match query_expr {
        ASTExpr::Not(_) => true,
        ASTExpr::And(subs) => subs.iter().any(check_for_disallowed_operators),
        ASTExpr::Or(subs) => subs.iter().any(check_for_disallowed_operators),
        ASTExpr::Compare { .. } => false,
    }
}

fn insert_into_bitset(ids: Vec<usize>) -> BitSet {
    let mut bitset = BitSet::new();
    bitset.extend(ids);
    bitset
}

fn eval_query_using_accelerators(
    query_expr: &ASTExpr,
    query_accelerators: &HashMap<String, QueryAccelerator>,
) -> Result<BitSet, anyhow::Error> {
    match query_expr {
        ASTExpr::And(subs) => {
            let mut acc: Option<BitSet> = None;
            for e in subs {
                let b = eval_query_using_accelerators(e, query_accelerators)?;
                acc = Some(match acc {
                    None => b,
                    Some(acc_b) => acc_b.intersection(&b).collect(),
                });
            }
            Ok(acc.unwrap_or_else(BitSet::new))
        }
        ASTExpr::Or(subs) => {
            let mut acc: Option<BitSet> = None;
            for e in subs {
                let b = eval_query_using_accelerators(e, query_accelerators)?;
                acc = Some(match acc {
                    None => b,
                    Some(acc_b) => acc_b.union(&b).collect(),
                });
            }
            Ok(acc.unwrap_or_else(BitSet::new))
        }
        ASTExpr::Not(_) => Err(anyhow::anyhow!(
            "NOT operator is not supported when using query accelerators"
        )),
        ASTExpr::Compare { field, op } => {
            let separator = FlattenConfig::dot_notation().separator;
            let field = if !field.starts_with(&separator) {
                format!("{}{}", separator, field)
            } else {
                field.clone()
            };
            if let Some(accelerator) = query_accelerators.get(&field) {
                match accelerator {
                    QueryAccelerator::InvertedIndex(bitmap) => {
                        match op {
                            CompareOp::Eq(value) => {
                                let attr_val = AttributeValue::try_from(value).map_err(|e| anyhow::anyhow!("Failed to convert value for Eq: {e}"))?;
                                Ok(bitmap.get(&attr_val).cloned().unwrap_or_default())
                            }
                            CompareOp::Ne(value) => {
                                let attr_val = AttributeValue::try_from(value).map_err(|e| anyhow::anyhow!("Failed to convert value for Ne: {e}"))?;
                                let mut result = BitSet::new();
                                for (val, bits) in bitmap.iter() {
                                    if val != &attr_val {
                                        result.extend(bits);
                                    }
                                }
                                Ok(result)
                            }
                            _ => {
                                Err(anyhow::anyhow!("Only equality comparisons are supported with the inverted index accelerator"))
                            }
                        }
                    }
                    QueryAccelerator::BTree(btree) => {
                        match op {
                            CompareOp::Eq(value) => {
                                let fval = value.as_f64().ok_or_else(|| anyhow::anyhow!("Failed to convert value to f64 for Eq"))?;
                                let fval = OrderedFloat::new(fval).map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                                if let Some(ids) = btree.get(&fval) {
                                    let mut bitset = BitSet::new();
                                    bitset.extend(ids.iter().cloned());
                                    Ok(bitset)
                                } else {
                                    Ok(BitSet::new())
                                }
                            }
                            CompareOp::Ne(value) => {
                                let fval = value.as_f64().ok_or_else(|| anyhow::anyhow!("Failed to convert value to f64 for Ne"))?;
                                let fval = OrderedFloat::new(fval).map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                                let mut bitset = BitSet::new();
                                for (val, ids) in btree.iter() {
                                    if val != &fval {
                                        bitset.extend(ids.iter().cloned());
                                    }
                                }
                                Ok(bitset)
                            }
                            CompareOp::Lt(num) => {
                                let fval = OrderedFloat::new(*num).map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                                let iter = btree.range((Unbounded, Excluded(fval)));
                                Ok(insert_into_bitset(iter.flat_map(|(_, ids)| ids.iter().cloned()).collect::<Vec<_>>()))
                            }
                            CompareOp::Lte(num) => {
                                let fval = OrderedFloat::new(*num).map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                                let iter = btree.range((Unbounded, Included(fval)));
                                Ok(insert_into_bitset(iter.flat_map(|(_, ids)| ids.iter().cloned()).collect::<Vec<_>>()))
                            }
                            CompareOp::Gt(num) => {
                                let fval = OrderedFloat::new(*num).map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                                let iter = btree.range((Excluded(fval), Unbounded));
                                Ok(insert_into_bitset(iter.flat_map(|(_, ids)| ids.iter().cloned()).collect::<Vec<_>>()))
                            }
                            CompareOp::Gte(num) => {
                                let fval = OrderedFloat::new(*num).map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                                let iter = btree.range((Included(fval), Unbounded));
                                Ok(insert_into_bitset(iter.flat_map(|(_, ids)| ids.iter().cloned()).collect::<Vec<_>>()))
                            }
                        }
                    }
                }
            } else {
                // if field not present, return an empty bitset
                Ok(BitSet::new())
            }
        }
    }
}

fn compute_inverted_index_accelerator(
    key: &str,
    doc_ids: &[usize],
    labels: &[HashMap<String, AttributeValue>],
) -> Result<HashMap<AttributeValue, BitSet>, anyhow::Error> {
    let mut inverted_index: HashMap<AttributeValue, BitSet> = HashMap::new();
    for (doc_id, label) in doc_ids.iter().zip(labels.iter()) {
        if let Some(value) = label.get(key) {
            inverted_index
                .entry(value.clone())
                .or_insert_with(BitSet::new)
                .insert(*doc_id);
        }
    }
    Ok(inverted_index)
}

fn compute_btree_accelerator(
    key: &str,
    labels: &[HashMap<String, AttributeValue>],
    doc_ids: &[usize],
) -> Result<BTreeMap<OrderedFloat, Vec<usize>>, anyhow::Error> {
    // Implementation for computing BTree accelerator
    let mut map: BTreeMap<OrderedFloat, Vec<usize>> = BTreeMap::new();
    for (label, doc_id) in labels.iter().zip(doc_ids.iter().copied()) {
        if let Some(value) = label.get(key) {
            if let Some(f64_value) = value.as_float() {
                let f64_value = OrderedFloat::new(f64_value)
                    .map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                map.entry(f64_value).or_default().push(doc_id);
            } else if let Some(i64_value) = value.as_integer() {
                let i64_value = OrderedFloat::new(i64_value as f64)
                    .map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                map.entry(i64_value).or_default().push(doc_id);
            } else {
                // Error for other attribute values
                return Err(anyhow::anyhow!(
                    "Unsupported attribute value for key: {}",
                    key
                ));
            }
        }
    }
    Ok(map)
}

// Compute a global label set across all documents with a representative element
// Make sure that each global label only maps to the same type of AttributeValue, and throw an error otherwise
fn compute_global_label_set(
    flattened_base_labels: &Vec<HashMap<std::string::String, AttributeValue>>,
) -> Result<HashMap<String, AttributeValue>, anyhow::Error> {
    let mut global_label_set = HashMap::new();
    for labels in flattened_base_labels {
        for (key, value) in labels {
            if let Some(existing_value) = global_label_set.get(key) {
                if discriminant(existing_value) != discriminant(value) {
                    return Err(anyhow::anyhow!("Inconsistent types for key: {}", key));
                }
            }
            global_label_set.insert(key.clone(), value.clone());
        }
    }
    Ok(global_label_set)
}

fn compute_query_accelerator(
    key: String,
    value: AttributeValue,
    doc_ids: &[usize],
    flattened_base_labels: &[HashMap<String, AttributeValue>],
) -> Result<QueryAccelerator, anyhow::Error> {
    match value {
        AttributeValue::String(_) | AttributeValue::Bool(_) => {
            let bitmap = compute_inverted_index_accelerator(&key, doc_ids, flattened_base_labels)?;
            Ok(QueryAccelerator::InvertedIndex(bitmap))
        }
        AttributeValue::Integer(_) | AttributeValue::Real(_) => {
            // For integers and reals, we use an BTree
            let btree = compute_btree_accelerator(&key, flattened_base_labels, doc_ids)?;
            Ok(QueryAccelerator::BTree(btree))
        }
        AttributeValue::Empty => Err(anyhow::anyhow!("Empty attribute value is not allowed")),
    }
}

pub fn compute_query_bitmaps(
    base_labels: Vec<Document>,
    query_labels: Vec<(usize, ASTExpr)>,
) -> Result<Vec<BitSet>, anyhow::Error> {
    // read query labels and differentiate between fast and slow path
    let bitmaps = if query_labels
        .iter()
        .any(|(_, expr)| check_for_disallowed_operators(expr))
    {
        // using the global threadpool is fine here
        #[allow(clippy::disallowed_methods)]
        let query_bitmaps: Vec<BitSet> = query_labels
            .par_iter()
            .map(|(_query_id, query_expr)| {
                let mut bitmap = BitSet::new();
                for base_label in base_labels.iter() {
                    if eval_query_expr(query_expr, &base_label.label) {
                        bitmap.insert(base_label.doc_id);
                    }
                }
                bitmap
            })
            .collect();
        query_bitmaps
    } else {
        // Flatten base labels so that nested structures are converted to a flat list of key-value pairs
        let flattened_base_labels: Vec<Vec<(std::string::String, AttributeValue)>> = base_labels
            .iter()
            .map(|base_label| {
                flatten_json_pointers_with_config(&base_label.label, &FlattenConfig::dot_notation())
            })
            .collect();

        let flattened_base_label_hashmaps: Result<
            Vec<HashMap<String, AttributeValue>>,
            anyhow::Error,
        > = flattened_base_labels
            .iter()
            .map(|labels| {
                let mut map = HashMap::new();
                for (key, value) in labels {
                    // a base label may not have two values for the same key
                    if let Some(_existing_value) = map.get(key) {
                        return Err(anyhow::anyhow!(
                            "Duplicate keys in the same document: {}",
                            key
                        ));
                    }
                    map.insert(key.clone(), value.clone());
                }
                Ok(map)
            })
            .collect();

        let flattened_base_label_hashmaps = flattened_base_label_hashmaps?;
        let base_doc_ids: Vec<usize> = base_labels
            .iter()
            .map(|base_label| base_label.doc_id)
            .collect();

        // compute the global set of labels ahead of time so that we can compute
        // each accelerator in parallel
        let global_label_set = compute_global_label_set(&flattened_base_label_hashmaps)?;

        // Compute the accelerators for each label in the global set
        #[allow(clippy::disallowed_methods)]
        let query_accelerators: Result<Vec<(String, QueryAccelerator)>, anyhow::Error> =
            global_label_set
                .par_iter()
                .map(|(key, value)| {
                    compute_query_accelerator(
                        key.clone(),
                        value.clone(),
                        &base_doc_ids,
                        &flattened_base_label_hashmaps,
                    )
                    .map(|accel| (key.clone(), accel))
                })
                .collect();

        // Convert the query accelerators to a hashmap for faster lookups
        let query_accelerators: HashMap<String, QueryAccelerator> =
            query_accelerators?.into_iter().collect();

        // Evaluate each query using the precomputed accelerators
        #[allow(clippy::disallowed_methods)]
        let query_bitmaps: Result<Vec<BitSet>, anyhow::Error> = query_labels
            .par_iter()
            .map(|(_query_id, query_expr)| {
                eval_query_using_accelerators(query_expr, &query_accelerators)
            })
            .collect();

        query_bitmaps?
    };

    Ok(bitmaps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::AttributeValue;
    use crate::parser::format::Document;
    use crate::{ASTExpr, CompareOp};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn test_compute_query_bitmap_duplicate_key_in_doc() {
        // serde_json does not allow duplicate keys, but we can simulate this by flattening a document with a nested object that, when flattened, produces duplicate keys
        // For this test, we will directly call compute_query_bitmaps with a document that, after flattening, would have duplicate keys
        // This is a synthetic test: we create a document with a nested object and a top-level key that would flatten to the same key
        let base_labels = vec![Document {
            doc_id: 0,
            label: json!({"color": {"color": "red"}, "color.color": "blue"}),
        }];
        // Query: color == "red"
        let query = ASTExpr::Compare {
            field: "color".to_string(),
            op: CompareOp::Eq(json!("red")),
        };
        let result = compute_query_bitmaps(base_labels.clone(), vec![(0, query)]);
        assert!(
            result.is_err(),
            "Should error on duplicate keys in the same document"
        );
    }

    #[test]
    fn test_compute_query_bitmap_inconsistent_types() {
        // Two documents, same key, different value types
        let base_labels = vec![
            Document {
                doc_id: 0,
                label: json!({"foo": "bar"}),
            },
            Document {
                doc_id: 1,
                label: json!({"foo": 123}),
            },
        ];
        // Query: foo == "bar"
        let query = ASTExpr::Compare {
            field: "foo".to_string(),
            op: CompareOp::Eq(json!("bar")),
        };
        let result = compute_query_bitmaps(base_labels.clone(), vec![(0, query)]);
        assert!(result.is_err(), "Should error on inconsistent value types");
    }

    #[test]
    fn test_compute_query_bitmap_missing_field() {
        use crate::parser::format::Document;
        use serde_json::json;
        // Three documents, one missing the 'color' field
        let base_labels = vec![
            Document {
                doc_id: 0,
                label: json!({"weight": 30}), // no color field
            },
            Document {
                doc_id: 1,
                label: json!({"color": "red", "weight": 10}),
            },
            Document {
                doc_id: 2,
                label: json!({"color": "blue", "weight": 20}),
            },
        ];

        // Query: color == "red"
        let query_color = ASTExpr::Compare {
            field: "color".to_string(),
            op: CompareOp::Eq(json!("red")),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_color)])
            .expect("should succeed");
        assert!(!bitmaps[0].contains(0));
        assert!(bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(2));

        // Query: weight >= 20
        let query_weight = ASTExpr::Compare {
            field: "weight".to_string(),
            op: CompareOp::Gte(20.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_weight)])
            .expect("should succeed");
        assert!(!bitmaps[0].contains(1));
        assert!(bitmaps[0].contains(2));
        assert!(bitmaps[0].contains(0));
    }

    #[test]
    fn test_compute_query_bitmap_nested_value() {
        use crate::parser::format::Document;
        use serde_json::json;
        // Two documents with nested car.color
        let base_labels = vec![
            Document {
                doc_id: 0,
                label: json!({"car": {"color": "red"}}),
            },
            Document {
                doc_id: 1,
                label: json!({"car": {"color": "blue"}}),
            },
        ];

        // Query: car.color == "red"
        let query_eq = ASTExpr::Compare {
            field: "car.color".to_string(),
            op: CompareOp::Eq(json!("red")),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_eq)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(1));

        // Query: NOT car.color == "red" (should match blue)
        let query_not = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: "car.color".to_string(),
            op: CompareOp::Eq(json!("red")),
        }));
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_not)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(0));
    }

    #[test]
    fn test_compute_query_bitmap_floats() {
        let base_labels = vec![
            Document {
                doc_id: 0,
                label: json!({"score": 1.5}),
            },
            Document {
                doc_id: 1,
                label: json!({"score": 2.0}),
            },
            Document {
                doc_id: 2,
                label: json!({"score": 3.5}),
            },
        ];

        // score < 2.0
        let query_lt = ASTExpr::Compare {
            field: "score".to_string(),
            op: CompareOp::Lt(2.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_lt)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(2));

        // score > 2.0
        let query_gt = ASTExpr::Compare {
            field: "score".to_string(),
            op: CompareOp::Gt(2.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_gt)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(2));
        assert!(!bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(1));

        // score <= 2.0
        let query_lte = ASTExpr::Compare {
            field: "score".to_string(),
            op: CompareOp::Lte(2.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_lte)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(0));
        assert!(bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(2));

        // score >= 2.0
        let query_gte = ASTExpr::Compare {
            field: "score".to_string(),
            op: CompareOp::Gte(2.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_gte)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(1));
        assert!(bitmaps[0].contains(2));
        assert!(!bitmaps[0].contains(0));

        // score >= 2.0 AND score <= 3.5 (range: [2.0, 3.5])
        let query_range = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "score".to_string(),
                op: CompareOp::Gte(2.0),
            },
            ASTExpr::Compare {
                field: "score".to_string(),
                op: CompareOp::Lte(3.5),
            },
        ]);
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_range)])
            .expect("should succeed");
        // Should match doc 1 (2.0) and doc 2 (3.5)
        assert!(bitmaps[0].contains(1));
        assert!(bitmaps[0].contains(2));
        assert!(!bitmaps[0].contains(0));
    }

    #[test]
    fn test_compute_query_bitmap_ints() {
        let base_labels = vec![
            Document {
                doc_id: 0,
                label: json!({"age": 10}),
            },
            Document {
                doc_id: 1,
                label: json!({"age": 20}),
            },
            Document {
                doc_id: 2,
                label: json!({"age": 30}),
            },
        ];

        // age < 20
        let query_lt = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Lt(20.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_lt)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(2));

        // age > 20
        let query_gt = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Gt(20.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_gt)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(2));
        assert!(!bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(1));

        // age <= 20
        let query_lte = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Lte(20.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_lte)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(0));
        assert!(bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(2));

        // age >= 20
        let query_gte = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Gte(20.0),
        };
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_gte)])
            .expect("should succeed");
        assert!(bitmaps[0].contains(1));
        assert!(bitmaps[0].contains(2));
        assert!(!bitmaps[0].contains(0));

        // age >= 20 AND age <= 30 (range: [20, 30])
        let query_range = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "age".to_string(),
                op: CompareOp::Gte(20.0),
            },
            ASTExpr::Compare {
                field: "age".to_string(),
                op: CompareOp::Lte(30.0),
            },
        ]);
        let bitmaps = compute_query_bitmaps(base_labels.clone(), vec![(0, query_range)])
            .expect("should succeed");
        // Should match doc 1 (20) and doc 2 (30)
        assert!(bitmaps[0].contains(1));
        assert!(bitmaps[0].contains(2));
        assert!(!bitmaps[0].contains(0));
    }

    #[test]
    fn test_compute_query_bitmap_ints_uses_document_ids_in_accelerator() {
        let base_labels = vec![
            Document {
                doc_id: 10,
                label: json!({"age": 10}),
            },
            Document {
                doc_id: 20,
                label: json!({"age": 20}),
            },
            Document {
                doc_id: 30,
                label: json!({"age": 30}),
            },
        ];

        let query_gte = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Gte(20.0),
        };
        let bitmaps =
            compute_query_bitmaps(base_labels, vec![(0, query_gte)]).expect("should succeed");

        assert!(bitmaps[0].contains(20));
        assert!(bitmaps[0].contains(30));
        assert!(!bitmaps[0].contains(10));
        assert!(!bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(2));
    }

    #[test]
    fn test_compute_query_bitmap_bools() {
        use crate::parser::format::Document;
        use serde_json::json;
        // Two documents with a boolean field
        let base_labels = vec![
            Document {
                doc_id: 0,
                label: json!({"flag": true}),
            },
            Document {
                doc_id: 1,
                label: json!({"flag": false}),
            },
        ];

        // Query: flag == true
        let query = ASTExpr::Compare {
            field: "flag".to_string(),
            op: CompareOp::Eq(json!(true)),
        };
        let queries = vec![(0, query)];
        let bitmaps = compute_query_bitmaps(base_labels.clone(), queries).expect("should succeed");
        // Only doc 0 should match
        assert!(bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(1));
    }

    #[test]
    fn test_compute_query_bitmaps_mixed_labels() {
        let base_labels = vec![
            Document {
                doc_id: 0,
                label: json!({"color": "red", "size": 10}),
            },
            Document {
                doc_id: 1,
                label: json!({"color": "blue", "size": 20}),
            },
            Document {
                doc_id: 2,
                label: json!({"color": "red", "size": 20}),
            },
        ];

        // Query: color == "red"
        let query1 = ASTExpr::Compare {
            field: "color".to_string(),
            op: CompareOp::Eq(serde_json::Value::String("red".to_string())),
        };
        // Query: size == 20
        let query2 = ASTExpr::Compare {
            field: "size".to_string(),
            op: CompareOp::Eq(20.into()),
        };
        // Query: color == "red" AND size == 20
        let query3 = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "color".to_string(),
                op: CompareOp::Eq(serde_json::Value::String("red".to_string())),
            },
            ASTExpr::Compare {
                field: "size".to_string(),
                op: CompareOp::Eq(20.into()),
            },
        ]);
        // Query: color == "red" OR size == 10
        let query4 = ASTExpr::Or(vec![
            ASTExpr::Compare {
                field: "color".to_string(),
                op: CompareOp::Eq(serde_json::Value::String("red".to_string())),
            },
            ASTExpr::Compare {
                field: "size".to_string(),
                op: CompareOp::Eq(10.into()),
            },
        ]);

        let queries = vec![(0, query1), (1, query2), (2, query3), (3, query4)];

        let bitmaps = compute_query_bitmaps(base_labels.clone(), queries).expect("should succeed");
        // color == "red" => doc 0, 2
        assert!(bitmaps[0].contains(0));
        assert!(bitmaps[0].contains(2));
        assert!(!bitmaps[0].contains(1));
        // size == 20 => doc 1, 2
        assert!(bitmaps[1].contains(1));
        assert!(bitmaps[1].contains(2));
        assert!(!bitmaps[1].contains(0));
        // color == "red" AND size == 20 => doc 2
        assert!(bitmaps[2].contains(2));
        assert!(!bitmaps[2].contains(0));
        assert!(!bitmaps[2].contains(1));
        // color == "red" OR size == 10 => doc 0, 2
        assert!(bitmaps[3].contains(0));
        assert!(bitmaps[3].contains(2));
        assert!(!bitmaps[3].contains(1));

        // Query: NOT color == "red"
        let not_query = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: "color".to_string(),
            op: CompareOp::Eq(serde_json::json!("red")),
        }));
        let queries_with_not = vec![(0, not_query)];
        let result = compute_query_bitmaps(base_labels.clone(), queries_with_not);
        assert!(
            result.is_ok(),
            "Slow path should not error, but NOT is not accelerated"
        );
        // The result should be a bitmap with doc 1 (not red)
        let bitmaps = result.unwrap();
        assert!(bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(2));
    }

    #[test]
    fn test_compute_query_accelerator() {
        // Prepare base labels
        let mut doc1 = HashMap::new();
        doc1.insert("foo".to_string(), AttributeValue::String("bar".to_string()));
        doc1.insert("num".to_string(), AttributeValue::Integer(42));
        doc1.insert("real".to_string(), AttributeValue::Real(3.13));
        doc1.insert("flag".to_string(), AttributeValue::Bool(true));
        let mut doc2 = HashMap::new();
        doc2.insert("foo".to_string(), AttributeValue::String("baz".to_string()));
        doc2.insert("num".to_string(), AttributeValue::Integer(7));
        doc2.insert("real".to_string(), AttributeValue::Real(2.71));
        doc2.insert("flag".to_string(), AttributeValue::Bool(false));
        let base = vec![doc1, doc2];
        let doc_ids = vec![10, 42];

        // String
        let accel = compute_query_accelerator(
            "foo".to_string(),
            AttributeValue::String("bar".to_string()),
            &doc_ids,
            &base,
        )
        .expect("Should succeed for String");
        match accel {
            QueryAccelerator::InvertedIndex(map) => {
                assert!(map.contains_key(&AttributeValue::String("bar".to_string())));
                assert!(map.contains_key(&AttributeValue::String("baz".to_string())));
                assert_eq!(
                    map.get(&AttributeValue::String("bar".to_string()))
                        .expect("bar key should exist")
                        .iter()
                        .collect::<Vec<_>>(),
                    vec![10]
                );
                assert_eq!(
                    map.get(&AttributeValue::String("baz".to_string()))
                        .expect("baz key should exist")
                        .iter()
                        .collect::<Vec<_>>(),
                    vec![42]
                );
            }
            _ => panic!("Expected InvertedIndex for String"),
        }

        // Bool
        let accel = compute_query_accelerator(
            "flag".to_string(),
            AttributeValue::Bool(true),
            &doc_ids,
            &base,
        )
        .expect("Should succeed for Bool");
        match accel {
            QueryAccelerator::InvertedIndex(map) => {
                assert!(map.contains_key(&AttributeValue::Bool(true)));
                assert!(map.contains_key(&AttributeValue::Bool(false)));
            }
            _ => panic!("Expected InvertedIndex for Bool"),
        }

        // Integer
        let accel = compute_query_accelerator(
            "num".to_string(),
            AttributeValue::Integer(42),
            &doc_ids,
            &base,
        )
        .expect("Should succeed for Integer");
        match accel {
            QueryAccelerator::BTree(map) => {
                assert!(map.contains_key(&super::OrderedFloat(42.0)));
                assert!(map.contains_key(&super::OrderedFloat(7.0)));
            }
            _ => panic!("Expected BTree for Integer"),
        }

        // Real
        let accel = compute_query_accelerator(
            "real".to_string(),
            AttributeValue::Real(3.13),
            &doc_ids,
            &base,
        )
        .expect("Should succeed for Real");
        match accel {
            QueryAccelerator::BTree(map) => {
                assert!(map.contains_key(&super::OrderedFloat(3.13)));
                assert!(map.contains_key(&super::OrderedFloat(2.71)));
            }
            _ => panic!("Expected BTree for Real"),
        }

        // Empty
        let err =
            compute_query_accelerator("none".to_string(), AttributeValue::Empty, &doc_ids, &base);
        assert!(err.is_err());
    }

    #[test]
    fn test_check_for_disallowed_operators() {
        // Compare only (no NOT)
        let expr = ASTExpr::Compare {
            field: "foo".to_string(),
            op: CompareOp::Eq(serde_json::Value::String("bar".to_string())),
        };
        assert!(!check_for_disallowed_operators(&expr));

        // NOT at root
        let expr = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: "foo".to_string(),
            op: CompareOp::Eq(serde_json::Value::String("bar".to_string())),
        }));
        assert!(check_for_disallowed_operators(&expr));

        // AND with NOT inside
        let expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "foo".to_string(),
                op: CompareOp::Eq(serde_json::Value::String("bar".to_string())),
            },
            ASTExpr::Not(Box::new(ASTExpr::Compare {
                field: "baz".to_string(),
                op: CompareOp::Eq(serde_json::Value::String("qux".to_string())),
            })),
        ]);
        assert!(check_for_disallowed_operators(&expr));

        // OR with only Compare
        let expr = ASTExpr::Or(vec![
            ASTExpr::Compare {
                field: "foo".to_string(),
                op: CompareOp::Eq(serde_json::Value::String("bar".to_string())),
            },
            ASTExpr::Compare {
                field: "baz".to_string(),
                op: CompareOp::Eq(serde_json::Value::String("qux".to_string())),
            },
        ]);
        assert!(!check_for_disallowed_operators(&expr));

        // Nested AND/OR with NOT deep inside
        let expr = ASTExpr::And(vec![
            ASTExpr::Or(vec![
                ASTExpr::Compare {
                    field: "a".to_string(),
                    op: CompareOp::Eq(serde_json::Value::String("b".to_string())),
                },
                ASTExpr::Not(Box::new(ASTExpr::Compare {
                    field: "c".to_string(),
                    op: CompareOp::Eq(serde_json::Value::String("d".to_string())),
                })),
            ]),
            ASTExpr::Compare {
                field: "e".to_string(),
                op: CompareOp::Eq(serde_json::Value::String("f".to_string())),
            },
        ]);
        assert!(check_for_disallowed_operators(&expr));
    }
}
