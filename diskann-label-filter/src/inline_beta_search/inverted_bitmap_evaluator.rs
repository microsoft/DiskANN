/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */


use roaring::RoaringTreemap;

use crate::encoded_attribute_provider::{
        ast_id_expr::ASTIdExpr, encoded_filter_expr::EncodedFilterExpr,
        roaring_attribute_store::RoaringAttributeStore,
    };

/// Evaluates an encoded filter expression using the inverted index and
/// returns the bitmap of matching point IDs.
pub struct InvertedBitmapEvaluator<'a> {
    attribute_store: &'a RoaringAttributeStore<u32>,
    universe: &'a RoaringTreemap,
    inplace_eval: bool,
}

impl<'a> InvertedBitmapEvaluator<'a> {
    pub fn new(
        attribute_store: &'a RoaringAttributeStore<u32>,
        universe: &'a RoaringTreemap,
    ) -> Self {
        Self {
            attribute_store,
            universe,
            inplace_eval: false,
        }
    }

    pub fn evaluate(&self, expr: &EncodedFilterExpr) -> diskann::ANNResult<RoaringTreemap> {
        if self.inplace_eval {
            let mut bitmap = RoaringTreemap::new();
            let mut write_result = |partial_bitmap: &RoaringTreemap| {
                bitmap = partial_bitmap.clone();
            };
            self.evaluate_ast_inplace(expr.encoded_filter_expr(), &mut write_result)?;
            Ok(bitmap)
        } else {
            self.evaluate_ast(expr.encoded_filter_expr())
        }
    }

    fn evaluate_ast(&self, expr: &ASTIdExpr<u64>) -> diskann::ANNResult<RoaringTreemap> {
        match expr {
            ASTIdExpr::Eq(attr_id) => self.attribute_store.posting_bitmap_for_attr_id(*attr_id),
            ASTIdExpr::Lte(field, value) => self.evaluate_numeric_range(field, *value, true),
            ASTIdExpr::Gte(field, value) => self.evaluate_numeric_range(field, *value, false),
            ASTIdExpr::Between(field, min, max) => self.evaluate_numeric_between(field, *min, *max),
            ASTIdExpr::And(exprs) => {
                if exprs.is_empty() {
                    return Ok(self.universe.clone());
                }

                let mut iter = exprs.iter();
                let mut acc = self.evaluate_ast(iter.next().expect("non-empty ensured"))?;
                for expr in iter {
                    let rhs = self.evaluate_ast(expr)?;
                    acc &= &rhs;
                }
                Ok(acc)
            }
            ASTIdExpr::Or(exprs) => {
                if exprs.is_empty() {
                    return Ok(RoaringTreemap::new());
                }

                let mut iter = exprs.iter();
                let mut acc = self.evaluate_ast(iter.next().expect("non-empty ensured"))?;
                for expr in iter {
                    let rhs = self.evaluate_ast(expr)?;
                    acc |= &rhs;
                }
                Ok(acc)
            }
            ASTIdExpr::Not(expr) => {
                let rhs = self.evaluate_ast(expr)?;
                Ok(self.universe - &rhs)
            }
        }
    }

    fn evaluate_numeric_range(
        &self,
        field: &str,
        value: f64,
        is_lte: bool,
    ) -> diskann::ANNResult<RoaringTreemap> {
        let field_attr_ids = self.attribute_store.attribute_ids_for_field(field)?;
        if field_attr_ids.is_empty() {
            return Ok(RoaringTreemap::new());
        }

        let attribute_map = self.attribute_store.attribute_map();
        let encoder = attribute_map.read().map_err(|_| {
            diskann::ANNError::message(
                diskann::ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on attribute map",
            )
        })?;

        let mut bitmap = RoaringTreemap::new();
        for attr_id in field_attr_ids {
            let matches = encoder
                .get_by_id(attr_id)
                .and_then(|attribute| attribute.attr_value().as_numeric_f64())
                .map(|attr_value| {
                    if is_lte {
                        attr_value <= value
                    } else {
                        attr_value >= value
                    }
                })
                .unwrap_or(false);

            if matches {
                let mut merge_postings = |postings: &RoaringTreemap| {
                    bitmap |= postings;
                };
                self.attribute_store
                    .posting_bitmap_read_inplace_for_attr_id(attr_id, &mut merge_postings)?;
            }
        }

        Ok(bitmap)
    }

    fn evaluate_ast_inplace(
        &self,
        expr: &ASTIdExpr<u64>,
        func: &mut dyn FnMut(&RoaringTreemap),
    ) -> diskann::ANNResult<()> {
        match expr {
            ASTIdExpr::Eq(attr_id) => self.attribute_store.posting_bitmap_read_inplace_for_attr_id(*attr_id, func),
            ASTIdExpr::Lte(field, value) => self.evaluate_numeric_range_inplace(field, *value, true, func),
            ASTIdExpr::Gte(field, value) => self.evaluate_numeric_range_inplace(field, *value, false, func),
            ASTIdExpr::Between(field, min, max) => {
                self.evaluate_numeric_between_inplace(field, *min, *max, func)
            }
            ASTIdExpr::And(exprs) => {
                if exprs.is_empty() {
                    func(self.universe);
                    return Ok(());
                }

                let mut iter = exprs.iter();
                let mut acc = RoaringTreemap::new();
                let mut seed_acc = |rhs: &RoaringTreemap| {
                    acc = rhs.clone();
                };
                self.evaluate_ast_inplace(iter.next().expect("non-empty ensured"), &mut seed_acc)?;

                for expr in iter {
                    let mut intersect_rhs = |rhs: &RoaringTreemap| {
                        acc &= rhs;
                    };
                    self.evaluate_ast_inplace(expr, &mut intersect_rhs)?;
                }
                func(&acc);
                Ok(())
            }
            ASTIdExpr::Or(exprs) => {
                if exprs.is_empty() {
                    let empty = RoaringTreemap::new();
                    func(&empty);
                    return Ok(());
                }

                let mut acc = RoaringTreemap::new();

                for expr in exprs {
                    let mut union_rhs = |rhs: &RoaringTreemap| {
                        acc |= rhs;
                    };
                    self.evaluate_ast_inplace(expr, &mut union_rhs)?;
                }
                func(&acc);
                Ok(())
            }
            ASTIdExpr::Not(expr) => {
                let mut negate_rhs = |rhs: &RoaringTreemap| {
                    let negated = self.universe - rhs;
                    func(&negated);
                };
                self.evaluate_ast_inplace(expr, &mut negate_rhs)?;
                Ok(())
            }
        }
    }

    fn evaluate_numeric_range_inplace(
        &self,
        field: &str,
        value: f64,
        is_lte: bool,
        func: &mut dyn FnMut(&RoaringTreemap),
    ) -> diskann::ANNResult<()> {
        let field_attr_ids = self.attribute_store.attribute_ids_for_field(field)?;
        if field_attr_ids.is_empty() {
            return Ok(());
        }

        let attribute_map = self.attribute_store.attribute_map();
        let encoder = attribute_map.read().map_err(|_| {
            diskann::ANNError::message(
                diskann::ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on attribute map",
            )
        })?;

        let mut bitmap = RoaringTreemap::new();
        for attr_id in field_attr_ids {
            let matches = encoder
                .get_by_id(attr_id)
                .and_then(|attribute| attribute.attr_value().as_numeric_f64())
                .map(|attr_value| {
                    if is_lte {
                        attr_value <= value
                    } else {
                        attr_value >= value
                    }
                })
                .unwrap_or(false);

            if matches {
                let mut merge_postings = |postings: &RoaringTreemap| {
                    bitmap |= postings;
                };
                self.attribute_store
                    .posting_bitmap_read_inplace_for_attr_id(attr_id, &mut merge_postings)?;
            }
        }
        func(&bitmap);
        Ok(())
    }

    fn evaluate_numeric_between(
        &self,
        field: &str,
        min: f64,
        max: f64,
    ) -> diskann::ANNResult<RoaringTreemap> {
        let field_attr_ids = self.attribute_store.attribute_ids_for_field(field)?;
        if field_attr_ids.is_empty() {
            return Ok(RoaringTreemap::new());
        }

        let attribute_map = self.attribute_store.attribute_map();
        let encoder = attribute_map.read().map_err(|_| {
            diskann::ANNError::message(
                diskann::ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on attribute map",
            )
        })?;

        let mut bitmap = RoaringTreemap::new();
        for attr_id in field_attr_ids {
            let matches = encoder
                .get_by_id(attr_id)
                .and_then(|attribute| attribute.attr_value().as_numeric_f64())
                .map(|attr_value| { attr_value >= min && attr_value <= max })
                .unwrap_or(false);

            if matches {
                let mut merge_postings = |postings: &RoaringTreemap| {
                    bitmap |= postings;
                };
                self.attribute_store
                    .posting_bitmap_read_inplace_for_attr_id(attr_id, &mut merge_postings)?;
            }
        }

        Ok(bitmap)
    }

    fn evaluate_numeric_between_inplace(
        &self,
        field: &str,
        min: f64,
        max: f64,
        func: &mut dyn FnMut(&RoaringTreemap),
    ) -> diskann::ANNResult<()> {
        let bitmap = self.evaluate_numeric_between(field, min, max)?;
        func(&bitmap);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::InvertedBitmapEvaluator;
    use crate::{
        attribute::{Attribute, AttributeValue},
        encoded_attribute_provider::{
            encoded_filter_expr::EncodedFilterExpr,
            roaring_attribute_store::RoaringAttributeStore,
        },
        parser::ast::{ASTExpr, CompareOp},
        traits::attribute_store::AttributeStore,
    };
    use roaring::RoaringTreemap;

    fn build_store() -> RoaringAttributeStore<u32> {
        let store = RoaringAttributeStore::new();

        store
            .set_element(
                &1,
                &[
                    Attribute::from_value("color", AttributeValue::String("red".to_string())),
                    Attribute::from_value("price", AttributeValue::Integer(100)),
                ],
            )
            .expect("set attrs for doc 1");
        store
            .set_element(
                &2,
                &[
                    Attribute::from_value("color", AttributeValue::String("blue".to_string())),
                    Attribute::from_value("price", AttributeValue::Integer(200)),
                ],
            )
            .expect("set attrs for doc 2");
        store
            .set_element(
                &3,
                &[
                    Attribute::from_value("color", AttributeValue::String("red".to_string())),
                    Attribute::from_value("price", AttributeValue::Integer(300)),
                ],
            )
            .expect("set attrs for doc 3");

        store
    }

    fn build_universe(ids: &[u32]) -> RoaringTreemap {
        let mut universe = RoaringTreemap::new();
        universe.extend(ids.iter().copied().map(u64::from));
        universe
    }

    fn evaluate_with_mode(
        expr: &ASTExpr,
        store: &RoaringAttributeStore<u32>,
        universe: &RoaringTreemap,
        inplace_eval: bool,
    ) -> RoaringTreemap {
        let encoded = EncodedFilterExpr::from_attribute_store(expr, store).expect("encode expr");
        let evaluator = InvertedBitmapEvaluator {
            attribute_store: store,
            universe,
            inplace_eval,
        };
        evaluator.evaluate(&encoded).expect("evaluate expr")
    }

    fn assert_same_modes(
        expr: &ASTExpr,
        store: &RoaringAttributeStore<u32>,
        universe: &RoaringTreemap,
        expected: &[u32],
    ) {
        let non_inplace = evaluate_with_mode(expr, store, universe, false);
        let inplace = evaluate_with_mode(expr, store, universe, true);

        let mut expected_bitmap = RoaringTreemap::new();
        expected_bitmap.extend(expected.iter().copied().map(u64::from));

        assert_eq!(non_inplace, expected_bitmap, "non-inplace result mismatch");
        assert_eq!(inplace, expected_bitmap, "inplace result mismatch");
        assert_eq!(inplace, non_inplace, "modes should produce identical results");
    }

    #[test]
    fn test_evaluate_eq_same_for_inplace_and_non_inplace() {
        let store = build_store();
        let universe = build_universe(&[1, 2, 3]);
        let expr = ASTExpr::Compare {
            field: "color".to_string(),
            op: CompareOp::Eq(json!("red")),
        };

        assert_same_modes(&expr, &store, &universe, &[1, 3]);
    }

    #[test]
    fn test_evaluate_numeric_range_same_for_inplace_and_non_inplace() {
        let store = build_store();
        let universe = build_universe(&[1, 2, 3]);
        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Lte(200.0),
        };

        assert_same_modes(&expr, &store, &universe, &[1, 2]);
    }

    #[test]
    fn test_evaluate_between_same_for_inplace_and_non_inplace() {
        let store = build_store();
        let universe = build_universe(&[1, 2, 3]);
        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Between(150.0, 300.0),
        };

        assert_same_modes(&expr, &store, &universe, &[2, 3]);
    }

    #[test]
    fn test_evaluate_logical_expression_same_for_inplace_and_non_inplace() {
        let store = build_store();
        let universe = build_universe(&[1, 2, 3]);
        let expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "color".to_string(),
                op: CompareOp::Eq(json!("red")),
            },
            ASTExpr::Not(Box::new(ASTExpr::Compare {
                field: "price".to_string(),
                op: CompareOp::Gte(300.0),
            })),
        ]);

        assert_same_modes(&expr, &store, &universe, &[1]);
    }

    #[test]
    fn test_evaluate_empty_logicals_same_for_inplace_and_non_inplace() {
        let store = build_store();
        let universe = build_universe(&[1, 2, 3]);

        assert_same_modes(&ASTExpr::And(vec![]), &store, &universe, &[1, 2, 3]);
        assert_same_modes(&ASTExpr::Or(vec![]), &store, &universe, &[]);
    }
}
