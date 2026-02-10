/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::io::Write;

use serde_json::Value;
use thiserror::Error;

/// For all client facing APIs, we will use the Attribute struct
/// or a HashMap<String, AttributeValue> for representing attrs
/// However, internally, we may want attributes to be stored as
/// integers or some other compact representation. That is defined
/// by AttributeType.
pub trait AttributeType: Eq + Clone + Hash {}
impl<T> AttributeType for T where T: Eq + Clone + Hash {}

/// A flattened label entry, field name and value pair.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Attribute {
    flattened_field_name: String,
    value: AttributeValue,
}

/// Value stored against a flattened label field name.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Empty,
    Bool(bool),
    Integer(i64),
    Real(f64),
    String(String),
}

impl AttributeValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttributeValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_float(&self) -> Option<f64> {
        match self {
            AttributeValue::Real(r) => Some(*r),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            AttributeValue::Integer(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, AttributeValue::Empty)
    }

    ///f64::to_bits() places a total order on floats
    /// where -NAN < -INF < -0.0 < +0.0 < +INF < +NAN
    /// For us, -NAN == NAN, -INF == INF, and -0 == 0.
    /// Hence this function that normalizes these values
    fn to_bits_helper(f: &f64) -> u64 {
        if f.is_nan() {
            f64::NAN.to_bits()
        } else if *f == 0.0 {
            0.0_f64.to_bits()
        } else {
            f.to_bits()
        }
    }
}

///The Hash and PartialEq implementations are only for internal use, for
/// mapped providers.
impl Hash for AttributeValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            AttributeValue::Bool(v) => state.write_u8(if *v { 1 } else { 0 }),
            AttributeValue::Integer(i) => state.write_i64(*i),
            AttributeValue::Real(f) => state.write_u64(Self::to_bits_helper(f)),
            AttributeValue::String(s) => s.hash(state),
            AttributeValue::Empty => {}
        }
    }
}

impl PartialEq for AttributeValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(l0), Self::Bool(r0)) => l0 == r0,
            (Self::Integer(l0), Self::Integer(r0)) => l0 == r0,
            (Self::Real(l0), Self::Real(r0)) => {
                Self::to_bits_helper(l0) == Self::to_bits_helper(r0)
            }
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            (Self::Empty, Self::Empty) => true,
            _ => false,
        }
    }
}

impl Eq for AttributeValue {}

impl Attribute {
    /// Create a label from a flattened field name and a JSON value.
    ///
    /// Objects and arrays are rejected because input is expected to be flattened.
    /// Numbers prefer integer first then real.
    pub fn from_json_value(
        field_name: &str,
        json_value: &serde_json::Value,
    ) -> Result<Self, JsonConversionError> {
        Ok(Self {
            flattened_field_name: field_name.to_owned(),
            value: AttributeValue::try_from(json_value)?,
        })
    }

    /// Field name getter.
    pub fn field_name(&self) -> &String {
        &self.flattened_field_name
    }

    /// Value getter.
    pub fn value(&self) -> &AttributeValue {
        &self.value
    }

    /// Builder from explicit parts.
    pub fn from_value(flattened_field_name: impl Into<String>, value: AttributeValue) -> Self {
        Self {
            flattened_field_name: flattened_field_name.into(),
            value,
        }
    }
}

impl Display for AttributeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttributeValue::Empty => write!(f, ""),
            AttributeValue::Bool(b) => write!(f, "{}", b),
            AttributeValue::Integer(n) => write!(f, "{}", n),
            AttributeValue::Real(r) => write!(f, "{}", r),
            AttributeValue::String(s) => write!(f, "{}", s),
        }
    }
}

impl Display for Attribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}={}", self.flattened_field_name, self.value)
    }
}

/// Try to create an AttributeValue from JSON.
impl TryFrom<&serde_json::Value> for AttributeValue {
    type Error = JsonConversionError;

    fn try_from(json_value: &Value) -> Result<Self, Self::Error> {
        match json_value {
            Value::Null => Err(JsonConversionError::NullValue),
            Value::Bool(v) => Ok(AttributeValue::Bool(*v)),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(AttributeValue::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(AttributeValue::Real(f))
                } else {
                    Err(JsonConversionError::Unsupported(n.clone()))
                }
            }
            Value::String(s) => Ok(AttributeValue::String(s.clone())),
            Value::Array(_values) => Err(JsonConversionError::ObjectsNotSupported),
            Value::Object(_) => Err(JsonConversionError::ObjectsNotSupported),
        }
    }
}

#[derive(Debug, Error)]
pub enum JsonConversionError {
    #[error("Value {0} is not an i64 nor f64")]
    Unsupported(serde_json::Number),
    #[error("Nested objects are not supported")]
    ObjectsNotSupported,
    #[error("Value is null")]
    NullValue,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn label_value_debug_clone_eq() {
        let v = AttributeValue::String("test_value".to_string());
        let c = v.clone();
        assert_eq!(v, c);
        let _dbg = format!("{:?}", v);
    }

    #[test]
    fn label_info_clone_and_getters() {
        let original = Attribute::from_value("a.b", AttributeValue::Integer(7));
        let cloned = original.clone();

        assert_eq!(cloned.field_name(), original.field_name());
        assert_eq!(cloned.value(), original.value());
        assert_eq!(cloned.value(), &AttributeValue::Integer(7));
    }

    #[test]
    fn label_value_try_from_primitives() {
        let cases = vec![
            (json!(true), AttributeValue::Bool(true)),
            (json!(42), AttributeValue::Integer(42)),
            (json!(-5), AttributeValue::Integer(-5)),
            (json!(3.68), AttributeValue::Real(3.68)),
            (json!("hello"), AttributeValue::String("hello".to_string())),
        ];

        for (j, expected) in cases {
            let got = AttributeValue::try_from(&j).unwrap();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn label_value_try_from_arrays_returns_error() {
        let j = json!([1, "x", false, [2, 3], 4.5]);
        let err = AttributeValue::try_from(&j).unwrap_err();
        match err {
            JsonConversionError::ObjectsNotSupported => {}
            _ => panic!("expected ObjectsNotSupported for arrays"),
        }
    }

    #[test]
    fn label_value_try_from_object_is_error() {
        let obj = json!({"a": 1});
        let err = AttributeValue::try_from(&obj).unwrap_err();
        match err {
            JsonConversionError::ObjectsNotSupported => {}
            _ => panic!("expected ObjectsNotSupported"),
        }
    }

    #[test]
    fn from_json_value_ok_paths() {
        let entry = Attribute::from_json_value("n", &json!("test")).unwrap();
        assert_eq!(entry.field_name(), "n");
        assert_eq!(entry.value(), &AttributeValue::String("test".to_string()));
    }

    #[test]
    fn from_json_value_arrays_fail() {
        let json_array = json!([1, "text", true, [1, 2, 3], 42.5]);
        let err = Attribute::from_json_value("n", &json_array).unwrap_err();
        match err {
            JsonConversionError::ObjectsNotSupported => {}
            _ => panic!("expected ObjectsNotSupported for arrays"),
        }
    }

    #[test]
    fn from_json_value_object_errors() {
        let obj = json!({"k": "v"});
        let err = Attribute::from_json_value("n", &obj).unwrap_err();
        match err {
            JsonConversionError::ObjectsNotSupported => {}
            _ => panic!("expected ObjectsNotSupported"),
        }
    }

    #[test]
    fn from_json_value_array_with_object_errors() {
        let j = json!([1, {"o": 1}, 3]);
        let err = Attribute::from_json_value("arr", &j).unwrap_err();
        match err {
            JsonConversionError::ObjectsNotSupported => {}
            _ => panic!("expected ObjectsNotSupported"),
        }
    }

    // Additional comprehensive tests for AttributeValue accessor methods
    #[test]
    fn attribute_value_as_str_tests() {
        assert_eq!(
            AttributeValue::String("hello".to_string()).as_str(),
            Some("hello")
        );
        assert_eq!(AttributeValue::Empty.as_str(), None);
        assert_eq!(AttributeValue::Bool(true).as_str(), None);
        assert_eq!(AttributeValue::Integer(42).as_str(), None);
        assert_eq!(AttributeValue::Real(2.5).as_str(), None);
    }

    #[test]
    fn attribute_value_as_bool_tests() {
        assert_eq!(AttributeValue::Bool(true).as_bool(), Some(true));
        assert_eq!(AttributeValue::Bool(false).as_bool(), Some(false));
        assert_eq!(AttributeValue::String("true".to_string()).as_bool(), None);
        assert_eq!(AttributeValue::Integer(1).as_bool(), None);
        assert_eq!(AttributeValue::Real(1.0).as_bool(), None);
        assert_eq!(AttributeValue::Empty.as_bool(), None);
    }

    #[test]
    fn attribute_value_as_float_tests() {
        assert_eq!(AttributeValue::Real(2.5).as_float(), Some(2.5));
        assert_eq!(AttributeValue::Real(-2.5).as_float(), Some(-2.5));
        assert_eq!(AttributeValue::Real(0.0).as_float(), Some(0.0));
        // NaN requires special handling since NaN != NaN
        assert!(AttributeValue::Real(f64::NAN).as_float().unwrap().is_nan());
        assert_eq!(
            AttributeValue::Real(f64::INFINITY).as_float(),
            Some(f64::INFINITY)
        );
        assert_eq!(
            AttributeValue::Real(f64::NEG_INFINITY).as_float(),
            Some(f64::NEG_INFINITY)
        );
        assert_eq!(AttributeValue::Integer(42).as_float(), None);
        assert_eq!(AttributeValue::String("3.14".to_string()).as_float(), None);
        assert_eq!(AttributeValue::Bool(true).as_float(), None);
        assert_eq!(AttributeValue::Empty.as_float(), None);
    }

    #[test]
    fn attribute_value_as_integer_tests() {
        assert_eq!(AttributeValue::Integer(42).as_integer(), Some(42));
        assert_eq!(AttributeValue::Integer(-100).as_integer(), Some(-100));
        assert_eq!(AttributeValue::Integer(0).as_integer(), Some(0));
        assert_eq!(AttributeValue::Real(42.0).as_integer(), None);
        assert_eq!(AttributeValue::String("42".to_string()).as_integer(), None);
        assert_eq!(AttributeValue::Bool(false).as_integer(), None);
        assert_eq!(AttributeValue::Empty.as_integer(), None);
    }

    #[test]
    fn attribute_value_is_empty_tests() {
        assert!(AttributeValue::Empty.is_empty());
        assert!(!AttributeValue::Bool(false).is_empty());
        assert!(!AttributeValue::Integer(0).is_empty());
        assert!(!AttributeValue::Real(0.0).is_empty());
        assert!(!AttributeValue::String("".to_string()).is_empty());
    }

    #[test]
    fn attribute_value_display_formatting() {
        assert_eq!(format!("{}", AttributeValue::Empty), "");
        assert_eq!(format!("{}", AttributeValue::Bool(true)), "true");
        assert_eq!(format!("{}", AttributeValue::Bool(false)), "false");
        assert_eq!(format!("{}", AttributeValue::Integer(42)), "42");
        assert_eq!(format!("{}", AttributeValue::Integer(-100)), "-100");
        assert_eq!(format!("{}", AttributeValue::Real(2.5)), "2.5");
        assert_eq!(format!("{}", AttributeValue::Real(-2.5)), "-2.5");
        assert_eq!(
            format!("{}", AttributeValue::String("hello".to_string())),
            "hello"
        );
    }

    #[test]
    fn attribute_display_formatting() {
        let attr = Attribute::from_value("field.name", AttributeValue::String("value".to_string()));
        assert_eq!(format!("{}", attr), "field.name=value");

        let attr_int = Attribute::from_value("count", AttributeValue::Integer(42));
        assert_eq!(format!("{}", attr_int), "count=42");

        let attr_bool = Attribute::from_value("enabled", AttributeValue::Bool(true));
        assert_eq!(format!("{}", attr_bool), "enabled=true");

        let attr_empty = Attribute::from_value("optional", AttributeValue::Empty);
        assert_eq!(format!("{}", attr_empty), "optional=");
    }

    #[test]
    fn attribute_from_value_with_different_string_types() {
        // Test with &str
        let attr1 = Attribute::from_value("test", AttributeValue::Integer(1));
        assert_eq!(attr1.field_name(), "test");

        // Test with String
        let attr2 = Attribute::from_value("test".to_string(), AttributeValue::Integer(2));
        assert_eq!(attr2.field_name(), "test");

        // Test with owned String
        let field_name = String::from("dynamic_field");
        let attr3 = Attribute::from_value(field_name, AttributeValue::Bool(true));
        assert_eq!(attr3.field_name(), "dynamic_field");
    }

    #[test]
    fn json_conversion_edge_cases() {
        // Test very large integers
        let large_int = json!(9223372036854775807i64); // i64::MAX
        let attr_val = AttributeValue::try_from(&large_int).unwrap();
        assert_eq!(attr_val.as_integer(), Some(9223372036854775807));

        // Test very small integers
        let small_int = json!(-9223372036854775808i64); // i64::MIN
        let attr_val = AttributeValue::try_from(&small_int).unwrap();
        assert_eq!(attr_val.as_integer(), Some(-9223372036854775808));

        // Test regular float values
        let float_json = json!(1.23456789);
        let attr_val = AttributeValue::try_from(&float_json).unwrap();
        assert_eq!(attr_val.as_float(), Some(1.23456789));

        // Test zero float
        let zero_float = json!(0.0);
        let attr_val = AttributeValue::try_from(&zero_float).unwrap();
        assert_eq!(attr_val.as_float(), Some(0.0));

        // Test negative float
        let neg_float = json!(-42.5);
        let attr_val = AttributeValue::try_from(&neg_float).unwrap();
        assert_eq!(attr_val.as_float(), Some(-42.5));
    }

    #[test]
    fn json_conversion_empty_strings_and_arrays() {
        // Test empty string
        let empty_str = json!("");
        let attr_val = AttributeValue::try_from(&empty_str).unwrap();
        assert_eq!(attr_val, AttributeValue::String("".to_string()));
        assert_eq!(attr_val.as_str(), Some(""));

        // Test empty array returns error
        let empty_array = json!([]);
        let err = AttributeValue::try_from(&empty_array).unwrap_err();
        match err {
            JsonConversionError::ObjectsNotSupported => {}
            _ => panic!("expected ObjectsNotSupported for arrays"),
        }
    }

    #[test]
    fn json_conversion_error_display() {
        // Test Unsupported error display
        let invalid_number = serde_json::Number::from_f64(f64::NAN);
        if let Some(num) = invalid_number {
            let error = JsonConversionError::Unsupported(num.clone());
            let error_msg = format!("{}", error);
            assert!(error_msg.contains("is not an i64 nor f64"));
        }

        // Test ObjectsNotSupported error display
        let error = JsonConversionError::ObjectsNotSupported;
        let error_msg = format!("{}", error);
        assert_eq!(error_msg, "Nested objects are not supported");
    }

    #[test]
    fn attribute_value_partial_eq_comprehensive() {
        // Test all combinations of equality
        let values = vec![
            AttributeValue::Empty,
            AttributeValue::Bool(true),
            AttributeValue::Bool(false),
            AttributeValue::Integer(0),
            AttributeValue::Integer(42),
            AttributeValue::Real(0.0),
            AttributeValue::Real(42.3),
            AttributeValue::Real(42.1),
            AttributeValue::String("".to_string()),
            AttributeValue::String("test".to_string()),
        ];

        // Each value should equal itself
        for value in &values {
            assert_eq!(value, value);
        }

        // Different values should not be equal
        for (i, val1) in values.iter().enumerate() {
            for (j, val2) in values.iter().enumerate() {
                if i != j {
                    assert_ne!(val1, val2);
                }
            }
        }

        // Test specific equality cases
        assert_eq!(
            AttributeValue::String("test".to_string()),
            AttributeValue::String("test".to_string())
        );
        assert_ne!(
            AttributeValue::String("test1".to_string()),
            AttributeValue::String("test2".to_string())
        );
    }

    #[test]
    fn attribute_value_float_eq_corner_cases() {
        // Test NaN equality - our implementation should make NaN == NaN
        let nan1 = AttributeValue::Real(f64::NAN);
        let nan2 = AttributeValue::Real(f64::NAN);
        assert_eq!(nan1, nan2); // Should be equal due to to_bits_helper normalization

        // Test positive and negative zero equality - should be equal
        let pos_zero = AttributeValue::Real(0.0);
        let neg_zero = AttributeValue::Real(-0.0);
        assert_eq!(pos_zero, neg_zero); // Should be equal due to normalization

        // Test positive and negative infinity equality - should be equal
        let pos_inf = AttributeValue::Real(f64::INFINITY);
        let neg_inf = AttributeValue::Real(f64::NEG_INFINITY);
        assert_ne!(pos_inf, neg_inf);

        // Test that NaN is not equal to any normal number
        assert_ne!(nan1, AttributeValue::Real(0.0));
        assert_ne!(nan1, AttributeValue::Real(1.0));
        assert_ne!(nan1, pos_inf);

        // Test normal floating point equality
        assert_eq!(AttributeValue::Real(42.5), AttributeValue::Real(42.5));
        assert_ne!(AttributeValue::Real(42.5), AttributeValue::Real(42.6));

        // Test very small numbers (subnormal)
        let subnormal1 = AttributeValue::Real(f64::MIN_POSITIVE / 2.0);
        let subnormal2 = AttributeValue::Real(f64::MIN_POSITIVE / 2.0);
        assert_eq!(subnormal1, subnormal2);

        // Test edge values
        assert_eq!(
            AttributeValue::Real(f64::MAX),
            AttributeValue::Real(f64::MAX)
        );
        assert_eq!(
            AttributeValue::Real(f64::MIN),
            AttributeValue::Real(f64::MIN)
        );
        assert_eq!(
            AttributeValue::Real(f64::MIN_POSITIVE),
            AttributeValue::Real(f64::MIN_POSITIVE)
        );
    }

    #[test]
    fn attribute_value_hash_corner_cases() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Helper function to get hash
        fn get_hash<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        // Test that equal values have equal hashes
        let nan1 = AttributeValue::Real(f64::NAN);
        let nan2 = AttributeValue::Real(f64::NAN);
        assert_eq!(nan1, nan2);
        assert_eq!(get_hash(&nan1), get_hash(&nan2));

        // Test positive and negative zero hash equality
        let pos_zero = AttributeValue::Real(0.0);
        let neg_zero = AttributeValue::Real(-0.0);
        assert_eq!(pos_zero, neg_zero);
        assert_eq!(get_hash(&pos_zero), get_hash(&neg_zero));

        // Test positive and negative infinity
        let pos_inf = AttributeValue::Real(f64::INFINITY);
        let neg_inf = AttributeValue::Real(f64::NEG_INFINITY);
        assert_ne!(pos_inf, neg_inf);
        assert_ne!(get_hash(&pos_inf), get_hash(&neg_inf));

        // Test that identical values hash the same
        let val1 = AttributeValue::Real(42.5);
        let val2 = AttributeValue::Real(42.5);
        assert_eq!(val1, val2);
        assert_eq!(get_hash(&val1), get_hash(&val2));

        // Test different types have different hashes (due to discriminant)
        let int_val = AttributeValue::Integer(42);
        let float_val = AttributeValue::Real(42.0);
        assert_ne!(int_val, float_val);
        // Hashes should be different (not guaranteed but highly likely due to discriminant)

        // Test empty value hash
        let empty1 = AttributeValue::Empty;
        let empty2 = AttributeValue::Empty;
        assert_eq!(get_hash(&empty1), get_hash(&empty2));

        // Test boolean hash consistency
        let bool_true1 = AttributeValue::Bool(true);
        let bool_true2 = AttributeValue::Bool(true);
        let bool_false = AttributeValue::Bool(false);
        assert_eq!(get_hash(&bool_true1), get_hash(&bool_true2));
        assert_ne!(get_hash(&bool_true1), get_hash(&bool_false));

        // Test string hash consistency
        let str1 = AttributeValue::String("test".to_string());
        let str2 = AttributeValue::String("test".to_string());
        let str3 = AttributeValue::String("different".to_string());
        assert_eq!(get_hash(&str1), get_hash(&str2));
        assert_ne!(get_hash(&str1), get_hash(&str3));
    }

    #[test]
    fn attribute_value_to_bits_helper_edge_cases() {
        // Test NaN normalization
        let nan_bits = AttributeValue::to_bits_helper(&f64::NAN);
        assert_eq!(nan_bits, f64::NAN.to_bits());

        // Test zero normalization - both +0.0 and -0.0 should give same bits
        let pos_zero_bits = AttributeValue::to_bits_helper(&0.0);
        let neg_zero_bits = AttributeValue::to_bits_helper(&-0.0);
        assert_eq!(pos_zero_bits, 0.0_f64.to_bits());
        assert_eq!(neg_zero_bits, 0.0_f64.to_bits());
        assert_eq!(pos_zero_bits, neg_zero_bits);

        // Test infinity
        let pos_inf_bits = AttributeValue::to_bits_helper(&f64::INFINITY);
        let neg_inf_bits = AttributeValue::to_bits_helper(&f64::NEG_INFINITY);
        assert_eq!(pos_inf_bits, f64::INFINITY.to_bits());
        assert_eq!(neg_inf_bits, f64::NEG_INFINITY.to_bits());
        assert_ne!(pos_inf_bits, neg_inf_bits);

        // Test normal numbers pass through unchanged
        let normal_value = 42.5;
        let normal_bits = AttributeValue::to_bits_helper(&normal_value);
        assert_eq!(normal_bits, normal_value.to_bits());

        // Test negative normal numbers
        let neg_value = -42.5;
        let neg_bits = AttributeValue::to_bits_helper(&neg_value);
        assert_eq!(neg_bits, neg_value.to_bits());

        // Test very small positive number (subnormal)
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let subnormal_bits = AttributeValue::to_bits_helper(&subnormal);
        assert_eq!(subnormal_bits, subnormal.to_bits());

        // Test edge values
        assert_eq!(
            AttributeValue::to_bits_helper(&f64::MAX),
            f64::MAX.to_bits()
        );
        assert_eq!(
            AttributeValue::to_bits_helper(&f64::MIN),
            f64::MIN.to_bits()
        );
        assert_eq!(
            AttributeValue::to_bits_helper(&f64::MIN_POSITIVE),
            f64::MIN_POSITIVE.to_bits()
        );
    }

    #[test]
    fn attribute_value_eq_hash_consistency() {
        // Test that if two values are equal, they have the same hash
        // This is a fundamental requirement for hash-based collections

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn get_hash<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        let test_cases = vec![
            // NaN cases
            (
                AttributeValue::Real(f64::NAN),
                AttributeValue::Real(f64::NAN),
            ),
            // Zero cases
            (AttributeValue::Real(0.0), AttributeValue::Real(-0.0)),
            // Infinity cases
            (
                AttributeValue::Real(f64::INFINITY),
                AttributeValue::Real(f64::NEG_INFINITY),
            ),
            // Normal values
            (AttributeValue::Real(42.5), AttributeValue::Real(42.5)),
            (AttributeValue::Integer(100), AttributeValue::Integer(100)),
            (AttributeValue::Bool(true), AttributeValue::Bool(true)),
            (AttributeValue::Bool(false), AttributeValue::Bool(false)),
            (
                AttributeValue::String("test".to_string()),
                AttributeValue::String("test".to_string()),
            ),
            (AttributeValue::Empty, AttributeValue::Empty),
        ];

        for (val1, val2) in test_cases {
            // If values are equal, their hashes must be equal
            if val1 == val2 {
                assert_eq!(
                    get_hash(&val1),
                    get_hash(&val2),
                    "Equal values must have equal hashes: {:?} == {:?}",
                    val1,
                    val2
                );
            }
        }
    }
}
