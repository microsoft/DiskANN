/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
use diskann_label_filter::attribute::AttributeValue;
use diskann_label_filter::parser::format::Document;
use diskann_label_filter::utils::flatten_utils::{
    flatten_json_pointers_with_config, FlattenConfig,
};
use diskann_label_filter::{ASTExpr, CompareOp};
use rayon::prelude::*;
use std::any::Any;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::mem::discriminant;
use std::ops::Bound::{Excluded, Included, Unbounded};

// In order to construct a B-Tree over floats, we need to create a total
// ordering on the float values by excluding NaN values. This struct is
// used to throw an error if a NaN value is encountered when constructing
// the OrderedFloat type.
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

trait QueryAccelerator: Send + Sync {
    fn eval(&self, op: &CompareOp) -> Result<BitSet, anyhow::Error>;

    fn universe(&self) -> BitSet;

    // method for testing
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn Any;
}

struct InvertedIndexAccelerator {
    map: HashMap<AttributeValue, BitSet>,
}

impl QueryAccelerator for InvertedIndexAccelerator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn universe(&self) -> BitSet {
        let mut result = BitSet::new();
        for (_, bits) in self.map.iter() {
            result.extend(bits);
        }
        result
    }

    fn eval(&self, op: &CompareOp) -> Result<BitSet, anyhow::Error> {
        match op {
            CompareOp::Eq(v) => {
                let attr_val = AttributeValue::try_from(v)
                    .map_err(|e| anyhow::anyhow!("Failed to convert value for Eq: {e}"))?;
                Ok(self.map.get(&attr_val).cloned().unwrap_or_default())
            }
            CompareOp::Ne(v) => {
                let attr_val = AttributeValue::try_from(v)
                    .map_err(|e| anyhow::anyhow!("Failed to convert value for Ne: {e}"))?;
                let mut result = BitSet::new();
                for (val, bits) in self.map.iter() {
                    if val != &attr_val {
                        result.extend(bits);
                    }
                }
                Ok(result)
            }
            _ => Err(anyhow::anyhow!(
                "Only equality comparisons are supported with the inverted index accelerator"
            )),
        }
    }
}

struct BTreeAccelerator {
    map: BTreeMap<OrderedFloat, Vec<usize>>,
}

impl QueryAccelerator for BTreeAccelerator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn universe(&self) -> BitSet {
        let mut result = BitSet::new();
        for (_, ids) in self.map.iter() {
            result.extend(ids.iter().cloned());
        }
        result
    }

    fn eval(&self, op: &CompareOp) -> Result<BitSet, anyhow::Error> {
        match op {
            CompareOp::Eq(v) => {
                let fval = v
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert value to f64 for Eq"))?;
                let fval = OrderedFloat::new(fval)
                    .map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                if let Some(ids) = self.map.get(&fval) {
                    Ok(insert_into_bitset(ids.to_vec()))
                } else {
                    Ok(BitSet::new())
                }
            }
            CompareOp::Ne(v) => {
                let fval = v
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert value to f64 for Ne"))?;
                let fval = OrderedFloat::new(fval)
                    .map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                let mut bitset = BitSet::new();
                for (val, ids) in self.map.iter() {
                    if val != &fval {
                        bitset.extend(ids.iter().cloned());
                    }
                }
                Ok(bitset)
            }
            CompareOp::Lt(num) => {
                let fval = OrderedFloat::new(*num)
                    .map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                let iter = self.map.range((Unbounded, Excluded(fval)));
                Ok(insert_into_bitset(
                    iter.flat_map(|(_, ids)| ids.iter().cloned()).collect(),
                ))
            }
            CompareOp::Lte(num) => {
                let fval = OrderedFloat::new(*num)
                    .map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                let iter = self.map.range((Unbounded, Included(fval)));
                Ok(insert_into_bitset(
                    iter.flat_map(|(_, ids)| ids.iter().cloned()).collect(),
                ))
            }
            CompareOp::Gt(num) => {
                let fval = OrderedFloat::new(*num)
                    .map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                let iter = self.map.range((Excluded(fval), Unbounded));
                Ok(insert_into_bitset(
                    iter.flat_map(|(_, ids)| ids.iter().cloned()).collect(),
                ))
            }
            CompareOp::Gte(num) => {
                let fval = OrderedFloat::new(*num)
                    .map_err(|e| anyhow::anyhow!("Failed to create OrderedFloat: {e}"))?;
                let iter = self.map.range((Included(fval), Unbounded));
                Ok(insert_into_bitset(
                    iter.flat_map(|(_, ids)| ids.iter().cloned()).collect(),
                ))
            }
            _ => Err(anyhow::anyhow!("Unsupported comparison operation")),
        }
    }
}

// Helper to prepend the separator if not present
fn prepend_separator(field: &str) -> String {
    let separator = FlattenConfig::dot_notation().separator;
    if !field.starts_with(&separator) {
        format!("{}{}", separator, field)
    } else {
        field.to_string()
    }
}

// Takes in an expression and returns a vector of all the labels used in the expression (raw field names, no separator prepending)
fn compute_label_set(expr: &ASTExpr) -> Vec<String> {
    match expr {
        ASTExpr::Not(sub) => compute_label_set(sub),
        ASTExpr::And(subs) => subs.iter().flat_map(compute_label_set).collect(),
        ASTExpr::Or(subs) => subs.iter().flat_map(compute_label_set).collect(),
        ASTExpr::Compare { field, .. } => vec![field.clone()],
    }
}

// Takes in a set of labels and returns the universe of all possible values for those labels
fn compute_universe(
    universe_labels: Vec<String>,
    query_accelerators: &HashMap<String, Box<dyn QueryAccelerator>>,
) -> BitSet {
    let mut universe_iter = universe_labels.iter();
    // Initialize universe to the first accelerator's universe, then intersect with the rest
    let mut universe = if let Some(first_label) = universe_iter.next() {
        if let Some(accelerator) = query_accelerators.get(first_label) {
            accelerator.universe()
        } else {
            BitSet::new()
        }
    } else {
        BitSet::new()
    };
    for label in universe_iter {
        if let Some(accelerator) = query_accelerators.get(label) {
            universe = universe.intersection(&accelerator.universe()).collect();
        }
    }
    universe
}

fn insert_into_bitset(ids: Vec<usize>) -> BitSet {
    let mut bitset = BitSet::new();
    bitset.extend(ids);
    bitset
}

fn eval_query_using_accelerators(
    query_expr: &ASTExpr,
    query_accelerators: &HashMap<String, Box<dyn QueryAccelerator>>,
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
        ASTExpr::Not(sub) => {
            // compute the universe of all possible values
            let universe_labels_raw = compute_label_set(query_expr);
            let universe_labels: Vec<String> = universe_labels_raw
                .iter()
                .map(|f| prepend_separator(f))
                .collect();
            let universe = compute_universe(universe_labels, query_accelerators);

            // Evaluate the sub-expression
            let sub_result = eval_query_using_accelerators(sub, query_accelerators)?;

            // Return the difference between the sub-expression result and the universe
            Ok(universe.difference(&sub_result).collect())
        }
        ASTExpr::Compare { field, op } => {
            let field = prepend_separator(field);
            if let Some(accelerator) = query_accelerators.get(&field) {
                accelerator.eval(op)
            } else {
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
                // convert from i64 to f64
                let f = i64_value as f64;
                if f as i64 != i64_value {
                    return Err(anyhow::anyhow!(
                        "i64 value cannot be exactly represented as f64: {}",
                        i64_value
                    ));
                }
                let i64_value = OrderedFloat::new(f)
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
    key: &str,
    value: &AttributeValue,
    doc_ids: &[usize],
    flattened_base_labels: &[HashMap<String, AttributeValue>],
) -> Result<Box<dyn QueryAccelerator>, anyhow::Error> {
    match value {
        AttributeValue::String(_) | AttributeValue::Bool(_) => {
            let bitmap = compute_inverted_index_accelerator(key, doc_ids, flattened_base_labels)?;
            Ok(Box::new(InvertedIndexAccelerator { map: bitmap }))
        }
        AttributeValue::Integer(_) | AttributeValue::Real(_) => {
            let btree = compute_btree_accelerator(key, flattened_base_labels, doc_ids)?;
            Ok(Box::new(BTreeAccelerator { map: btree }))
        }
        AttributeValue::Empty => Err(anyhow::anyhow!("Empty attribute value is not allowed")),
    }
}

pub fn compute_query_bitmaps(
    base_labels: Vec<Document>,
    query_labels: Vec<(usize, ASTExpr)>,
) -> Result<Vec<BitSet>, anyhow::Error> {
    // Flatten base labels so that nested structures are converted to a flat list of key-value pairs
    let flattened_base_labels: Vec<Vec<(std::string::String, AttributeValue)>> = base_labels
        .iter()
        .map(|base_label| {
            flatten_json_pointers_with_config(&base_label.label, &FlattenConfig::dot_notation())
        })
        .collect();

    let flattened_base_label_hashmaps: Result<Vec<HashMap<String, AttributeValue>>, anyhow::Error> =
        flattened_base_labels
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
    let query_accelerators: HashMap<String, Box<dyn QueryAccelerator>> = global_label_set
        .par_iter()
        .map(|(key, value)| {
            compute_query_accelerator(key, value, &base_doc_ids, &flattened_base_label_hashmaps)
                .map(|accel| (key.clone(), accel))
        })
        .collect::<Result<_, _>>()?;

    // Evaluate each query using the precomputed accelerators
    #[allow(clippy::disallowed_methods)]
    let query_bitmaps: Result<Vec<BitSet>, anyhow::Error> = query_labels
        .par_iter()
        .map(|(_query_id, query_expr)| {
            eval_query_using_accelerators(query_expr, &query_accelerators)
        })
        .collect();

    let query_bitmaps = query_bitmaps?;

    Ok(query_bitmaps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_label_filter::attribute::AttributeValue;
    use diskann_label_filter::parser::format::Document;
    use diskann_label_filter::{ASTExpr, CompareOp};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn test_compute_query_bitmap_not_with_missing_field() {
        // Three documents: two with "color", one without
        let base_labels = vec![
            Document {
                doc_id: 0,
                label: json!({"color": "red"}),
            },
            Document {
                doc_id: 1,
                label: json!({"color": "blue"}),
            },
            Document {
                doc_id: 2,
                label: json!({"shape": "circle"}), // no color field
            },
        ];

        // Query: NOT color == "red"
        let not_query = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: "color".to_string(),
            op: CompareOp::Eq(json!("red")),
        }));
        let queries = vec![(0, not_query)];
        let bitmaps = compute_query_bitmaps(base_labels.clone(), queries).expect("Should succeed");
        // Only doc 1 should match (has color and is not red)
        assert!(bitmaps[0].contains(1));
        assert!(!bitmaps[0].contains(0));
        // Doc 2 does not have color, so should not be included in the NOT universe
        assert!(!bitmaps[0].contains(2));
    }

    #[test]
    fn test_compute_universe_function() {
        // Sub-test 1: universe label not in query_accelerators, should return empty
        let query_accelerators: HashMap<String, Box<dyn QueryAccelerator>> = HashMap::new();
        let universe_labels = vec!["missing_label".to_string()];
        let result = compute_universe(universe_labels, &query_accelerators);
        assert!(
            result.is_empty(),
            "Universe should be empty if label is missing"
        );

        // Sub-test 2: both accelerator types, non-empty intersection
        // InvertedIndexAccelerator for 'foo' with docs 1, 2
        let mut inv_map = HashMap::new();
        inv_map.insert(
            AttributeValue::String("a".to_string()),
            [1, 2].iter().cloned().collect(),
        );
        let inv_accel = Box::new(InvertedIndexAccelerator { map: inv_map });

        // BTreeAccelerator for 'bar' with docs 2, 3
        let mut btree_map = BTreeMap::new();
        btree_map.insert(OrderedFloat(1.0), vec![2, 3]);
        let btree_accel = Box::new(BTreeAccelerator { map: btree_map });

        let mut query_accelerators: HashMap<String, Box<dyn QueryAccelerator>> = HashMap::new();
        query_accelerators.insert("foo".to_string(), inv_accel);
        query_accelerators.insert("bar".to_string(), btree_accel);

        // The intersection of {1,2} and {2,3} is {2}
        let universe_labels = vec!["foo".to_string(), "bar".to_string()];
        let result = compute_universe(universe_labels, &query_accelerators);
        let expected: BitSet = [2].iter().cloned().collect();
        assert_eq!(
            result, expected,
            "Universe should be the intersection of both accelerator universes"
        );
    }

    #[test]
    fn test_compute_label_set() {
        // OR expression: foo == 1 OR bar == 2
        let expr_or = ASTExpr::Or(vec![
            ASTExpr::Compare {
                field: "foo".to_string(),
                op: CompareOp::Eq(json!(1)),
            },
            ASTExpr::Compare {
                field: "bar".to_string(),
                op: CompareOp::Eq(json!(2)),
            },
        ]);
        let mut result_or = compute_label_set(&expr_or);
        result_or.sort();
        assert_eq!(result_or, vec!["bar".to_string(), "foo".to_string()]);

        // NOT expression: NOT (baz == 3)
        let expr_not = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: "baz".to_string(),
            op: CompareOp::Eq(json!(3)),
        }));
        let result_not = compute_label_set(&expr_not);
        assert_eq!(result_not, vec!["baz".to_string()]);
    }

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

        // Query: NOT .car.color == "red" (should match blue)
        let query_not = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: ".car.color".to_string(),
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
        let bitmaps =
            compute_query_bitmaps(base_labels.clone(), queries_with_not).expect("Should succeed");
        // The result should be a bitmap with doc 1 (not red)
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
            "foo",
            &AttributeValue::String("bar".to_string()),
            &doc_ids,
            &base,
        )
        .expect("Should succeed for String");
        let accel = accel
            .as_any()
            .downcast_ref::<InvertedIndexAccelerator>()
            .expect("Expected InvertedIndexAccelerator");
        assert!(accel
            .map
            .contains_key(&AttributeValue::String("bar".to_string())));
        assert!(accel
            .map
            .contains_key(&AttributeValue::String("baz".to_string())));
        assert_eq!(
            accel
                .map
                .get(&AttributeValue::String("bar".to_string()))
                .expect("bar key should exist")
                .iter()
                .collect::<Vec<_>>(),
            vec![10]
        );
        assert_eq!(
            accel
                .map
                .get(&AttributeValue::String("baz".to_string()))
                .expect("baz key should exist")
                .iter()
                .collect::<Vec<_>>(),
            vec![42]
        );

        // Bool
        let accel = compute_query_accelerator("flag", &AttributeValue::Bool(true), &doc_ids, &base)
            .expect("Should succeed for Bool");
        let accel = accel
            .as_any()
            .downcast_ref::<InvertedIndexAccelerator>()
            .expect("Expected InvertedIndexAccelerator");
        assert!(accel.map.contains_key(&AttributeValue::Bool(true)));
        assert!(accel.map.contains_key(&AttributeValue::Bool(false)));

        // Integer
        let accel = compute_query_accelerator("num", &AttributeValue::Integer(42), &doc_ids, &base)
            .expect("Should succeed for Integer");
        let accel = accel
            .as_any()
            .downcast_ref::<BTreeAccelerator>()
            .expect("Expected BTreeAccelerator");
        assert!(accel.map.contains_key(&super::OrderedFloat(42.0)));
        assert!(accel.map.contains_key(&super::OrderedFloat(7.0)));

        // Real
        let accel = compute_query_accelerator("real", &AttributeValue::Real(3.13), &doc_ids, &base)
            .expect("Should succeed for Real");
        let accel = accel
            .as_any()
            .downcast_ref::<BTreeAccelerator>()
            .expect("Expected BTreeAccelerator");
        assert!(accel.map.contains_key(&super::OrderedFloat(3.13)));
        assert!(accel.map.contains_key(&super::OrderedFloat(2.71)));

        // Empty
        let err = compute_query_accelerator("none", &AttributeValue::Empty, &doc_ids, &base);
        assert!(err.is_err());
    }
}
