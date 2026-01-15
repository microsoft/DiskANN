/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{parser::ast::ASTVisitor, ASTExpr, CompareOp};
use diskann::{ANNError, ANNErrorKind, ANNResult};
use std::sync::{Arc, RwLock};

use crate::{
    attribute::Attribute,
    encoded_attribute_provider::{ast_id_expr::ASTIdExpr, attribute_encoder::AttributeEncoder},
};

///Struct that maps an [`ASTExpr`] into an [`ASTIdExpr`]. For a
/// subset of filter expressions that do not use relational
/// operators (<, <=, >, >=), we can optimize filter matching
/// by mapping attributes to unique ids, so that at search time
/// we end up comparing numbers instead of "field-name + value"
/// combinations. At insertion time, this is managed by the
/// [`RoaringAttributeStore`] struct which stores a map of the
/// attribute->id pairs. At search time, this struct converts
/// the attributes specified in the filter expression into
/// the same set of u64s.
pub struct ASTLabelIdMapper {
    attribute_map: Arc<RwLock<AttributeEncoder>>,
}

impl ASTLabelIdMapper {
    pub fn new(attribute_map: Arc<RwLock<AttributeEncoder>>) -> Self {
        Self { attribute_map }
    }

    fn _lookup(
        encoder: &AttributeEncoder,
        attribute: &Attribute,
        field: &str,
        op: &CompareOp,
    ) -> ANNResult<ASTIdExpr<u64>> {
        match encoder.get(attribute) {
            Some(attribute_id) => Ok(ASTIdExpr::Terminal(attribute_id)),
            None => Err(ANNError::message(
                ANNErrorKind::Opaque,
                format!(
                    "{}+{} present in the query does not exist in the dataset.",
                    field, op
                ),
            )),
        }
    }
}

impl ASTVisitor for ASTLabelIdMapper {
    type Output = ANNResult<ASTIdExpr<u64>>;

    /// Visit an AST expression
    fn visit(&mut self, expr: &ASTExpr) -> Self::Output {
        match expr {
            ASTExpr::And(exprs) => self.visit_and(exprs),
            ASTExpr::Or(exprs) => self.visit_or(exprs),
            ASTExpr::Not(expr) => self.visit_not(expr),
            ASTExpr::Compare { field, op } => self.visit_compare(field, op),
        }
    }

    /// Visit an AND expression
    fn visit_and(&mut self, exprs: &[ASTExpr]) -> Self::Output {
        let mut id_exprs: Vec<ASTIdExpr<u64>> = Vec::with_capacity(exprs.len());
        for expr in exprs {
            id_exprs.push(self.visit(expr)?);
        }
        Ok(ASTIdExpr::And(id_exprs))
    }

    /// Visit an OR expression
    fn visit_or(&mut self, exprs: &[ASTExpr]) -> Self::Output {
        let mut id_exprs: Vec<ASTIdExpr<u64>> = Vec::with_capacity(exprs.len());
        for expr in exprs {
            id_exprs.push(self.visit(expr)?);
        }
        Ok(ASTIdExpr::Or(id_exprs))
    }

    /// Visit a NOT expression
    fn visit_not(&mut self, expr: &ASTExpr) -> Self::Output {
        let id_expr = Box::new(self.visit(expr)?);
        Ok(ASTIdExpr::Not(id_expr))
    }

    /// Visit a comparison expression
    fn visit_compare(&mut self, field: &str, op: &CompareOp) -> Self::Output {
        let label_or_none = match op {
            CompareOp::Eq(value) => match Attribute::from_json_value(field, value) {
                Ok(v) => Some(v),
                Err(json_e) => {
                    return Err(ANNError::new(ANNErrorKind::Opaque, json_e));
                }
            },
            CompareOp::Ne(_value) => {
                tracing::warn!("<> Not supported ");
                None
            }
            CompareOp::Lt(_num) => {
                tracing::warn!("< Not supported ");
                None
            }
            CompareOp::Lte(_num) => {
                tracing::warn!("<= Not supported ");
                None
            }
            CompareOp::Gt(_num) => {
                tracing::warn!("> Not supported ");
                None
            }
            CompareOp::Gte(_num) => {
                tracing::warn!(">= Not supported ");
                None
            }
        };

        if let Some(attribute) = label_or_none {
            match self.attribute_map.read() {
                Ok(guard) => Self::_lookup(&guard, &attribute, field, op),
                Err(poison_error) => {
                    let attr_map = poison_error.into_inner();
                    Self::_lookup(&attr_map, &attribute, field, op)
                }
            }
        } else {
            Err(ANNError::message(
                ANNErrorKind::Opaque,
                format!(
                    "CompareOp {} is not supported in the mapped filter search scenario.",
                    op
                ),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attribute::{Attribute, AttributeValue},
        encoded_attribute_provider::{ast_id_expr::ASTIdExpr, attribute_encoder::AttributeEncoder},
        parser::ast::ASTVisitor,
        ASTExpr, CompareOp,
    };
    use diskann::ANNErrorKind;
    use serde_json::Value;
    use std::sync::{Arc, RwLock};

    /// Helper function to create a test AttributeMap with some test data
    fn create_test_attribute_map() -> Arc<RwLock<AttributeEncoder>> {
        let mut attribute_map = AttributeEncoder::new();

        // Add some test attributes
        let attr1 = Attribute::from_value(
            "category".to_string(),
            AttributeValue::String("electronics".to_string()),
        );
        let attr2 = Attribute::from_value("price".to_string(), AttributeValue::Real(299.99));
        let attr3 = Attribute::from_value(
            "brand".to_string(),
            AttributeValue::String("apple".to_string()),
        );
        let attr4 = Attribute::from_value("in_stock".to_string(), AttributeValue::Bool(true));

        // Insert attributes and store the returned IDs
        let _id1 = attribute_map.insert(&attr1);
        let _id2 = attribute_map.insert(&attr2);
        let _id3 = attribute_map.insert(&attr3);
        let _id4 = attribute_map.insert(&attr4);

        Arc::new(RwLock::new(attribute_map))
    }

    /// Helper function to create an ASTLabelIdMapper with test data
    fn create_test_mapper() -> ASTLabelIdMapper {
        let attribute_map = create_test_attribute_map();
        ASTLabelIdMapper::new(attribute_map)
    }

    #[test]
    fn test_visit_compare_eq_existing_attribute() {
        let mut mapper = create_test_mapper();

        // Test with existing string attribute
        let expr = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::String("electronics".to_string())),
        };

        let result = mapper.visit(&expr).unwrap();
        assert!(matches!(result, ASTIdExpr::Terminal(_)));

        // Test with existing number attribute
        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Eq(Value::Number(serde_json::Number::from_f64(299.99).unwrap())),
        };

        let result = mapper.visit(&expr).unwrap();
        assert!(matches!(result, ASTIdExpr::Terminal(_)));

        // Test with existing boolean attribute
        let expr = ASTExpr::Compare {
            field: "in_stock".to_string(),
            op: CompareOp::Eq(Value::Bool(true)),
        };

        let result = mapper.visit(&expr).unwrap();
        assert!(matches!(result, ASTIdExpr::Terminal(_)));
    }

    #[test]
    fn test_visit_compare_eq_nonexistent_attribute() {
        let mut mapper = create_test_mapper();

        // Test with non-existent attribute
        let expr = ASTExpr::Compare {
            field: "nonexistent_field".to_string(),
            op: CompareOp::Eq(Value::String("some_value".to_string())),
        };

        let result = mapper.visit(&expr);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert_eq!(error.kind(), ANNErrorKind::Opaque);
        assert!(error.to_string().contains("does not exist in the dataset"));
    }

    #[test]
    fn test_visit_compare_eq_wrong_value_type() {
        let mut mapper = create_test_mapper();

        // Test with existing field but wrong value type
        let expr = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::String("wrong_value".to_string())),
        };

        let result = mapper.visit(&expr);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert_eq!(error.kind(), ANNErrorKind::Opaque);
        assert!(error.to_string().contains("does not exist in the dataset"));
    }

    #[test]
    fn test_visit_compare_unsupported_operations() {
        let mut mapper = create_test_mapper();

        // Test Ne operation (not supported)
        let expr = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Ne(Value::String("electronics".to_string())),
        };

        let result = mapper.visit(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not supported"));

        // Test Lt operation (not supported)
        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Lt(100.0),
        };

        let result = mapper.visit(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not supported"));

        // Test Lte operation (not supported)
        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Lte(300.0),
        };

        let result = mapper.visit(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not supported"));

        // Test Gt operation (not supported)
        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Gt(200.0),
        };

        let result = mapper.visit(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not supported"));

        // Test Gte operation (not supported)
        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Gte(250.0),
        };

        let result = mapper.visit(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not supported"));
    }

    #[test]
    fn test_visit_and_expression() {
        let mut mapper = create_test_mapper();

        // Create an AND expression with two comparison expressions
        let expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "category".to_string(),
                op: CompareOp::Eq(Value::String("electronics".to_string())),
            },
            ASTExpr::Compare {
                field: "brand".to_string(),
                op: CompareOp::Eq(Value::String("apple".to_string())),
            },
        ]);

        let result = mapper.visit(&expr).unwrap();

        match result {
            ASTIdExpr::And(exprs) => {
                assert_eq!(exprs.len(), 2);
                assert!(matches!(exprs[0], ASTIdExpr::Terminal(_)));
                assert!(matches!(exprs[1], ASTIdExpr::Terminal(_)));
            }
            _ => panic!("Expected ASTIdExpr::And"),
        }
    }

    #[test]
    fn test_visit_or_expression() {
        let mut mapper = create_test_mapper();

        // Create an OR expression with two comparison expressions
        let expr = ASTExpr::Or(vec![
            ASTExpr::Compare {
                field: "category".to_string(),
                op: CompareOp::Eq(Value::String("electronics".to_string())),
            },
            ASTExpr::Compare {
                field: "brand".to_string(),
                op: CompareOp::Eq(Value::String("apple".to_string())),
            },
        ]);

        let result = mapper.visit(&expr).unwrap();

        match result {
            ASTIdExpr::Or(exprs) => {
                assert_eq!(exprs.len(), 2);
                assert!(matches!(exprs[0], ASTIdExpr::Terminal(_)));
                assert!(matches!(exprs[1], ASTIdExpr::Terminal(_)));
            }
            _ => panic!("Expected ASTIdExpr::Or"),
        }
    }

    #[test]
    fn test_visit_not_expression() {
        let mut mapper = create_test_mapper();

        // Create a NOT expression - this should now work as NOT is supported
        let expr = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::String("electronics".to_string())),
        }));

        let result = mapper.visit(&expr);
        assert!(result.is_ok());

        let id_expr = result.unwrap();
        // Verify it's a NOT expression wrapping a terminal
        match id_expr {
            ASTIdExpr::Not(inner) => match inner.as_ref() {
                ASTIdExpr::Terminal(_) => {} // Expected: NOT wrapping a terminal (label ID)
                _ => panic!("Expected NOT to wrap a terminal expression"),
            },
            _ => panic!("Expected a NOT expression"),
        }
    }

    #[test]
    fn test_visit_nested_and_or_expressions() {
        let mut mapper = create_test_mapper();

        // Create a complex nested expression: (category=electronics OR brand=apple) AND in_stock=true
        let expr = ASTExpr::And(vec![
            ASTExpr::Or(vec![
                ASTExpr::Compare {
                    field: "category".to_string(),
                    op: CompareOp::Eq(Value::String("electronics".to_string())),
                },
                ASTExpr::Compare {
                    field: "brand".to_string(),
                    op: CompareOp::Eq(Value::String("apple".to_string())),
                },
            ]),
            ASTExpr::Compare {
                field: "in_stock".to_string(),
                op: CompareOp::Eq(Value::Bool(true)),
            },
        ]);

        let result = mapper.visit(&expr).unwrap();

        match result {
            ASTIdExpr::And(and_exprs) => {
                assert_eq!(and_exprs.len(), 2);

                // First element should be an OR expression
                match &and_exprs[0] {
                    ASTIdExpr::Or(or_exprs) => {
                        assert_eq!(or_exprs.len(), 2);
                        assert!(matches!(or_exprs[0], ASTIdExpr::Terminal(_)));
                        assert!(matches!(or_exprs[1], ASTIdExpr::Terminal(_)));
                    }
                    _ => panic!("Expected ASTIdExpr::Or as first element"),
                }

                // Second element should be a terminal
                assert!(matches!(and_exprs[1], ASTIdExpr::Terminal(_)));
            }
            _ => panic!("Expected ASTIdExpr::And"),
        }
    }

    #[test]
    fn test_visit_empty_and_expression() {
        let mut mapper = create_test_mapper();

        // Create an empty AND expression
        let expr = ASTExpr::And(vec![]);

        let result = mapper.visit(&expr).unwrap();

        match result {
            ASTIdExpr::And(exprs) => {
                assert_eq!(exprs.len(), 0);
            }
            _ => panic!("Expected ASTIdExpr::And"),
        }
    }

    #[test]
    fn test_visit_empty_or_expression() {
        let mut mapper = create_test_mapper();

        // Create an empty OR expression
        let expr = ASTExpr::Or(vec![]);

        let result = mapper.visit(&expr).unwrap();

        match result {
            ASTIdExpr::Or(exprs) => {
                assert_eq!(exprs.len(), 0);
            }
            _ => panic!("Expected ASTIdExpr::Or"),
        }
    }

    #[test]
    fn test_visit_and_with_error_propagation() {
        let mut mapper = create_test_mapper();

        // Create an AND expression where one sub-expression will fail
        let expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "category".to_string(),
                op: CompareOp::Eq(Value::String("electronics".to_string())),
            },
            ASTExpr::Compare {
                field: "nonexistent_field".to_string(),
                op: CompareOp::Eq(Value::String("some_value".to_string())),
            },
        ]);

        let result = mapper.visit(&expr);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert_eq!(error.kind(), ANNErrorKind::Opaque);
        assert!(error.to_string().contains("does not exist in the dataset"));
    }

    #[test]
    fn test_visit_or_with_error_propagation() {
        let mut mapper = create_test_mapper();

        // Create an OR expression where one sub-expression will fail
        let expr = ASTExpr::Or(vec![
            ASTExpr::Compare {
                field: "category".to_string(),
                op: CompareOp::Eq(Value::String("electronics".to_string())),
            },
            ASTExpr::Compare {
                field: "nonexistent_field".to_string(),
                op: CompareOp::Eq(Value::String("some_value".to_string())),
            },
        ]);

        let result = mapper.visit(&expr);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert_eq!(error.kind(), ANNErrorKind::Opaque);
        assert!(error.to_string().contains("does not exist in the dataset"));
    }

    #[test]
    fn test_with_poisoned_rwlock() {
        // This test simulates what happens when the RwLock is poisoned
        // We create an AttributeMap and then simulate a panic in another thread
        let attribute_map = create_test_attribute_map();

        // Poison the lock by panicking while holding a write lock in another thread
        let poisoned_map = {
            let map_clone = Arc::clone(&attribute_map);
            let handle = std::thread::spawn(move || {
                let _guard = map_clone.write().unwrap();
                panic!("Intentional panic to poison the lock");
            });

            // Wait for the thread to finish and poison the lock
            let _ = handle.join();
            attribute_map
        };

        let mut mapper = ASTLabelIdMapper::new(poisoned_map);

        // Test that the mapper still works even with a poisoned lock
        let expr = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::String("electronics".to_string())),
        };

        let result = mapper.visit(&expr).unwrap();
        assert!(matches!(result, ASTIdExpr::Terminal(_)));
    }

    #[test]
    fn test_visit_compare_with_null_value() {
        let mut mapper = create_test_mapper();

        // Test with null value
        let expr = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::Null),
        };

        let result = mapper.visit(&expr);
        // This should fail because null values are not supported in our attribute conversion
        assert!(result.is_err());
    }

    #[test]
    fn test_visit_compare_with_array_value() {
        let mut mapper = create_test_mapper();

        // Test with array value
        let expr = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::Array(vec![Value::String("test".to_string())])),
        };

        let result = mapper.visit(&expr);
        // This should fail because array values are not supported in our attribute conversion
        assert!(result.is_err());
    }

    #[test]
    fn test_visit_compare_with_object_value() {
        let mut mapper = create_test_mapper();

        // Test with object value
        let mut obj = serde_json::Map::new();
        obj.insert("test".to_string(), Value::String("value".to_string()));
        let expr = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::Object(obj)),
        };

        let result = mapper.visit(&expr);
        // This should fail because object values are not supported in our attribute conversion
        assert!(result.is_err());
    }

    #[test]
    fn test_visit_compare_with_different_number_formats() {
        let mut mapper = create_test_mapper();

        // Test with integer value that should match a real attribute
        let expr = ASTExpr::Compare {
            field: "price".to_string(),
            op: CompareOp::Eq(Value::Number(serde_json::Number::from(300))),
        };

        let result = mapper.visit(&expr);
        // This should fail because the integer 300 doesn't match the real 299.99
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not exist in the dataset"));
    }

    #[test]
    fn test_with_empty_attribute_map() {
        let empty_map = Arc::new(RwLock::new(AttributeEncoder::new()));
        let mut mapper = ASTLabelIdMapper::new(empty_map);

        // Test with any comparison on empty map
        let expr = ASTExpr::Compare {
            field: "any_field".to_string(),
            op: CompareOp::Eq(Value::String("any_value".to_string())),
        };

        let result = mapper.visit(&expr);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not exist in the dataset"));
    }

    #[test]
    fn test_visit_deeply_nested_expressions() {
        let mut mapper = create_test_mapper();

        // Create a deeply nested expression: ((category=electronics AND brand=apple) OR (price=299.99)) AND in_stock=true
        let expr = ASTExpr::And(vec![
            ASTExpr::Or(vec![
                ASTExpr::And(vec![
                    ASTExpr::Compare {
                        field: "category".to_string(),
                        op: CompareOp::Eq(Value::String("electronics".to_string())),
                    },
                    ASTExpr::Compare {
                        field: "brand".to_string(),
                        op: CompareOp::Eq(Value::String("apple".to_string())),
                    },
                ]),
                ASTExpr::Compare {
                    field: "price".to_string(),
                    op: CompareOp::Eq(Value::Number(serde_json::Number::from_f64(299.99).unwrap())),
                },
            ]),
            ASTExpr::Compare {
                field: "in_stock".to_string(),
                op: CompareOp::Eq(Value::Bool(true)),
            },
        ]);

        let result = mapper.visit(&expr).unwrap();

        // Verify the structure is correct
        match result {
            ASTIdExpr::And(and_exprs) => {
                assert_eq!(and_exprs.len(), 2);

                // First element should be an OR expression
                match &and_exprs[0] {
                    ASTIdExpr::Or(or_exprs) => {
                        assert_eq!(or_exprs.len(), 2);

                        // First OR element should be an AND expression
                        match &or_exprs[0] {
                            ASTIdExpr::And(inner_and_exprs) => {
                                assert_eq!(inner_and_exprs.len(), 2);
                                assert!(matches!(inner_and_exprs[0], ASTIdExpr::Terminal(_)));
                                assert!(matches!(inner_and_exprs[1], ASTIdExpr::Terminal(_)));
                            }
                            _ => panic!("Expected ASTIdExpr::And as first OR element"),
                        }

                        // Second OR element should be a terminal
                        assert!(matches!(or_exprs[1], ASTIdExpr::Terminal(_)));
                    }
                    _ => panic!("Expected ASTIdExpr::Or as first element"),
                }

                // Second element should be a terminal
                assert!(matches!(and_exprs[1], ASTIdExpr::Terminal(_)));
            }
            _ => panic!("Expected ASTIdExpr::And"),
        }
    }
}
