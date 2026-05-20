/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
use std::collections::HashMap;
use std::mem::discriminant;
// use serde_json::Value;
use crate::{ASTExpr, CompareOp};
use crate::attribute::AttributeValue;
use crate::parser::evaluator::eval_query_expr;
use crate::utils::jsonl_reader::read_and_parse_queries;
use crate::utils::jsonl_reader::read_baselabels; 
use crate::utils::{flatten_utils::flatten_json_pointers};
// use crate::parser::{format::Document};
use rayon::prelude::*;

pub enum QueryAccelerator {
    InvertedIndex(HashMap<AttributeValue, BitSet>),
    RTree, // Placeholder for future implementation
}

pub fn eval_query_using_accelerators(
    query_expr: &ASTExpr,
    query_accelerators: &HashMap<String, QueryAccelerator>,
) -> BitSet {
    let bitset = match query_expr {
        ASTExpr::And(subs) => 
         subs.iter().map(|e| eval_query_using_accelerators(e, query_accelerators)).fold(BitSet::new(), |acc, b| {
            if acc.is_empty() {
                b
            } else {
                acc.intersection(&b).collect()
            }
        }),
        ASTExpr::Or(subs) => subs.iter().map(|e| eval_query_using_accelerators(e, query_accelerators)).fold(BitSet::new(), |acc, b| {
            if acc.is_empty() {
                b
            } else {
                acc.union(&b).collect()
            }
        }),
        ASTExpr::Not(sub) => {
            // Want to flip all the bits in the bitmap, but we need to know
            // the document universe for that. We could potentially disallow
            // empty labels, and then compute the complement
            panic!("NOT operator is not supported yet");
        },
        ASTExpr::Compare { field, op } => {
            if let Some(accelerator) = query_accelerators.get(field) {
                match accelerator {
                    QueryAccelerator::InvertedIndex(bitmap) => {
                        // We can only accelerate equality comparisons 
                        // with the inverted index
                        match op {
                            CompareOp::Eq(value) => bitmap.get(&AttributeValue::try_from(value).unwrap()).cloned().unwrap_or_default(),
                            CompareOp::Ne(value) => {
                                let mut result = BitSet::new();
                                for (val, bits) in bitmap.iter() {
                                    if val != &AttributeValue::try_from(value).unwrap() {
                                        result.extend(bits);
                                    }
                                }
                                result
                            },
                            _ => {
                                // For other comparison operators, we would need a different accelerator (e.g. an RTree)
                                panic!("Only equality comparisons are supported with the inverted index accelerator");
                            }
                        }
                    },
                    QueryAccelerator::RTree => {
                         // RTree acceleration logic would go here
                         panic!("RTree accelerator is not implemented yet");
                //          match op {
                //     CompareOp::Eq(value) => field_val == value,
                //     CompareOp::Ne(value) => field_val != value,
                //     CompareOp::Lt(num) => {
                //         if let Some(f1) = field_val.as_f64() {
                //             f1 < *num
                //         } else {
                //             false
                //         }
                //     }
                //     CompareOp::Lte(num) => {
                //         if let Some(f1) = field_val.as_f64() {
                //             f1 <= *num
                //         } else {
                //             false
                //         }
                //     }
                //     CompareOp::Gt(num) => {
                //         if let Some(f1) = field_val.as_f64() {
                //             f1 > *num
                //         } else {
                //             false
                //         }
                //     }
                //     CompareOp::Gte(num) => {
                //         if let Some(f1) = field_val.as_f64() {
                //             f1 >= *num
                //         } else {
                //             false
                //         }
                //     }
                // }
                    }
                }
            } else{
                // if field not present, return an empty bitset
                return BitSet::new();
            }
        }
    };
    bitset
}

pub fn compute_inverted_index_bitmap(
    key: String, 
    labels: Vec<HashMap<String, AttributeValue>>) 
    -> Result<HashMap<AttributeValue, BitSet>, anyhow::Error>
{
    let mut inverted_index: HashMap<AttributeValue, BitSet> = HashMap::new();
    for (doc_id, label) in labels.iter().enumerate() {
        if let Some(value) = label.get(&key) {
            inverted_index.entry(value.clone()).or_insert_with(BitSet::new).insert(doc_id);
        }
    }
    Ok(inverted_index)
}

// Compute a global label set across all documents with a representative element
// Make sure that each global label only maps to the same type of AttributeValue, and throw an error otherwise
pub fn compute_global_label_set(flattened_base_labels: &Vec<HashMap<std::string::String, AttributeValue>>) -> Result<HashMap<String, AttributeValue>, anyhow::Error> {
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

pub fn compute_query_accelerator(
    key: String,
    value: AttributeValue,
    flattened_base_labels: &Vec<HashMap<String, AttributeValue>>
) -> QueryAccelerator {
    match value {
        AttributeValue::String(_) | AttributeValue::Bool(_) => {
            let bitmap = compute_inverted_index_bitmap(key.clone(), flattened_base_labels.clone()).unwrap_or_default();
            QueryAccelerator::InvertedIndex(bitmap)
        },
        AttributeValue::Integer(_) | AttributeValue::Real(_) => {
            // For integers and reals, we use an RTree
            QueryAccelerator::RTree
        }
        AttributeValue::Empty => {
            // TODO this should be an error or something
            QueryAccelerator::RTree
        }
    }
}

pub fn read_labels_and_compute_bitmap(
    base_label_filename: &str,
    query_label_filename: &str,
) -> Result<Vec<BitSet>, anyhow::Error> {
    // Read base labels
    let base_labels = read_baselabels(base_label_filename)?;

    // Flatten base labels
    let flattened_base_labels: Vec<Vec<(std::string::String, AttributeValue)>> = base_labels
        .iter()
        .map(|base_label| flatten_json_pointers(&base_label.label))
        .collect();

    // print the string in the first document for debugging
    if let Some(first_doc) = flattened_base_labels.first() {
        println!("First document flattened labels:");
        for (key, value) in first_doc {
            println!("{}: {:?}", key, value);
        }
    }

    let flattened_base_label_hashmaps: Result<Vec<HashMap<String, AttributeValue>>, anyhow::Error> =
        flattened_base_labels.iter().map(|labels| {
            let mut map = HashMap::new();
            for (key, value) in labels {
                if let Some(_existing_value) = map.get(key) {
                    return Err(anyhow::anyhow!("Duplicate keys in the same document: {}", key));
                }
                map.insert(key.clone(), value.clone());
            }
            Ok(map)
        }).collect();

    let flattened_base_label_hashmaps = flattened_base_label_hashmaps?;

    // print all the keys in the global label set for debugging
    let global_label_set = compute_global_label_set(&flattened_base_label_hashmaps)?;
    println!("Global label set:");
    for (key, value) in &global_label_set {
        println!("{}: {:?}", key, value);
    }

    // let global_label_set = compute_global_label_set(&flattened_base_label_hashmaps)?;

    #[allow(clippy::disallowed_methods)]
    let query_accelerators: Vec<(String, QueryAccelerator)> = global_label_set.par_iter()
        .map(|(key, value)| {
            (key.clone(), compute_query_accelerator(key.clone(), value.clone(), &flattened_base_label_hashmaps))
        })
        .collect();
    
    // Convert to a HashMap for easier access during query evaluation
    let query_accelerators: HashMap<String, QueryAccelerator> = query_accelerators.into_iter().collect();

    // Parse queries and evaluate against labels
    let parsed_queries = read_and_parse_queries(query_label_filename)?;

    // using the global threadpool is fine here
    #[allow(clippy::disallowed_methods)]
    let query_bitmaps: Vec<BitSet> = parsed_queries
        .par_iter()
        .map(|(_query_id, query_expr)| {
            eval_query_using_accelerators(query_expr, &query_accelerators)
        })
        .collect();

    Ok(query_bitmaps)
}