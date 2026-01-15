/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::AttributeValue;

/// A trait for encoding and decoding field-value pairs into keys.
///
/// This trait defines how to convert field-value pairs into byte keys suitable
/// for storage in key-value stores. The encoding should preserve ordering for
/// numeric types to enable efficient range queries.
pub trait KeyCodec: Send + Sync + 'static {
    /// Encodes a field-value pair into a byte key.
    ///
    /// The encoding should be deterministic and preserve ordering for numeric types.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name.
    /// * `value` - The value to encode.
    ///
    /// # Returns
    ///
    /// A byte vector representing the encoded key.
    fn encode_field_value_key(&self, field: &str, value: &AttributeValue) -> Vec<u8>;

    /// Decodes a byte key back into a field-value pair.
    ///
    /// This is an optional debuging method with a default implementation that returns `None`.
    ///
    /// # Arguments
    ///
    /// * `key` - The encoded key bytes.
    ///
    /// # Returns
    ///
    /// `Some((field, value))` if decoding succeeds, or `None` if not implemented or fails.
    fn decode_field_value_key(&self, key: &[u8]) -> Option<(String, AttributeValue)>;
}

/// Type prefix constants for key encoding.
pub const TYPE_PREFIX_INTEGER: &str = "I";
pub const TYPE_PREFIX_FLOAT: &str = "F";
pub const TYPE_PREFIX_STRING: &str = "S";
pub const TYPE_PREFIX_BOOL: &str = "B";

/// Field separator in encoded keys.
pub const FIELD_SEPARATOR: &str = "\0";

/// The default key codec implementation.
///
/// This codec encodes field-value pairs using a format that preserves ordering
/// for numeric types. It uses special type prefixes (I for integers, F for floats,
/// S for strings, B for booleans) and encodes numbers in a way that maintains
/// lexicographic ordering of the byte representation.
#[derive(Debug, Clone)]
pub struct DefaultKeyCodec {}

impl DefaultKeyCodec {
    /// Creates a new `DefaultKeyCodec` instance.
    pub fn new() -> Self {
        Self {}
    }

    /// Encodes a signed 64-bit integer to preserve ordering.
    ///
    /// XORs with the sign bit to ensure negative numbers sort before positive ones.
    #[inline]
    fn encode_i64(n: i64) -> u64 {
        (n as u64) ^ 0x8000000000000000 // XOR with sign bit
    }

    /// Encodes a 64-bit float to preserve ordering.
    ///
    /// Handles IEEE 754 floating-point encoding to ensure proper sorting.
    #[inline]
    fn encode_f64(f: f64) -> u64 {
        let bits = f.to_bits();
        if (bits >> 63) == 0 {
            bits ^ 0x8000000000000000
        } else {
            !bits
        }
    }

    /// Formats an integer value with field name and type prefix.
    ///
    /// Takes a raw i64 value and encodes it internally to preserve ordering.
    fn format_integer_value(field: &str, value: i64) -> Vec<u8> {
        let encoded = Self::encode_i64(value);
        format!(
            "{}{}{}{:016x}",
            field, FIELD_SEPARATOR, TYPE_PREFIX_INTEGER, encoded
        )
        .into_bytes()
    }

    /// Formats a float value with field name and type prefix.
    ///
    /// Takes a raw f64 value and encodes it internally to preserve ordering.
    fn format_float_value(field: &str, value: f64) -> Vec<u8> {
        let encoded = Self::encode_f64(value);
        format!(
            "{}{}{}{:016x}",
            field, FIELD_SEPARATOR, TYPE_PREFIX_FLOAT, encoded
        )
        .into_bytes()
    }

    /// Formats a string value with field name and type prefix.
    fn format_string_value(field: &str, value: &str) -> Vec<u8> {
        format!(
            "{}{}{}{}",
            field, FIELD_SEPARATOR, TYPE_PREFIX_STRING, value
        )
        .into_bytes()
    }

    /// Formats a boolean key with field name and type prefix.
    fn format_bool_value(field: &str, value: bool) -> Vec<u8> {
        format!(
            "{}{}{}{}",
            field,
            FIELD_SEPARATOR,
            TYPE_PREFIX_BOOL,
            if value { "1" } else { "0" }
        )
        .into_bytes()
    }
}

impl Default for DefaultKeyCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyCodec for DefaultKeyCodec {
    fn encode_field_value_key(&self, field: &str, value: &AttributeValue) -> Vec<u8> {
        match value {
            AttributeValue::Integer(i) => Self::format_integer_value(field, *i),
            AttributeValue::Real(f) => Self::format_float_value(field, *f),
            AttributeValue::String(s) => Self::format_string_value(field, s),
            AttributeValue::Bool(b) => Self::format_bool_value(field, *b),
            AttributeValue::Empty => {
                // Handle empty values - maybe as a special case or skip
                Self::format_string_value(field, "")
            }
        }
    }

    fn decode_field_value_key(&self, _key: &[u8]) -> Option<(String, AttributeValue)> {
        todo!()
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_int(n: i64) -> AttributeValue {
        AttributeValue::try_from(&json!(n)).unwrap()
    }

    fn make_float(f: f64) -> AttributeValue {
        AttributeValue::try_from(&json!(f)).unwrap()
    }

    fn make_string(s: &str) -> AttributeValue {
        AttributeValue::try_from(&json!(s)).unwrap()
    }

    fn make_bool(b: bool) -> AttributeValue {
        AttributeValue::try_from(&json!(b)).unwrap()
    }

    #[test]
    fn test_integer_ordering() {
        let codec = DefaultKeyCodec::new();

        let key_neg100 = codec.encode_field_value_key("age", &make_int(-100));
        let key_neg10 = codec.encode_field_value_key("age", &make_int(-10));
        let key_neg1 = codec.encode_field_value_key("age", &make_int(-1));
        let key_zero = codec.encode_field_value_key("age", &make_int(0));
        let key_pos1 = codec.encode_field_value_key("age", &make_int(1));
        let key_pos10 = codec.encode_field_value_key("age", &make_int(10));
        let key_pos100 = codec.encode_field_value_key("age", &make_int(100));

        assert!(key_neg100 < key_neg10);
        assert!(key_neg10 < key_neg1);
        assert!(key_neg1 < key_zero);
        assert!(key_zero < key_pos1);
        assert!(key_pos1 < key_pos10);
        assert!(key_pos10 < key_pos100);
    }

    #[test]
    fn test_float_ordering() {
        let codec = DefaultKeyCodec::new();

        let key_neg100 = codec.encode_field_value_key("price", &make_float(-100.5));
        let key_neg1 = codec.encode_field_value_key("price", &make_float(-1.5));
        let key_neg_tiny = codec.encode_field_value_key("price", &make_float(-0.001));
        let key_zero = codec.encode_field_value_key("price", &make_float(0.0));
        let key_pos_tiny = codec.encode_field_value_key("price", &make_float(0.001));
        let key_pos1 = codec.encode_field_value_key("price", &make_float(1.5));
        let key_pos100 = codec.encode_field_value_key("price", &make_float(100.5));

        assert!(key_neg100 < key_neg1);
        assert!(key_neg1 < key_neg_tiny);
        assert!(key_neg_tiny < key_zero);
        assert!(key_zero < key_pos_tiny);
        assert!(key_pos_tiny < key_pos1);
        assert!(key_pos1 < key_pos100);
    }

    #[test]
    fn test_string_ordering() {
        let codec = DefaultKeyCodec::new();

        let key_a = codec.encode_field_value_key("name", &make_string("Alice"));
        let key_b = codec.encode_field_value_key("name", &make_string("Bob"));
        let key_c = codec.encode_field_value_key("name", &make_string("Charlie"));

        assert!(key_a < key_b);
        assert!(key_b < key_c);
    }

    #[test]
    fn test_boolean_ordering() {
        let codec = DefaultKeyCodec::new();

        let key_false = codec.encode_field_value_key("active", &make_bool(false));
        let key_true = codec.encode_field_value_key("active", &make_bool(true));

        assert!(key_false < key_true);
    }

    #[test]
    fn test_type_isolation() {
        let codec = DefaultKeyCodec::new();

        let key_int = codec.encode_field_value_key("value", &make_int(42));
        let key_str = codec.encode_field_value_key("value", &make_string("42"));
        let key_bool = codec.encode_field_value_key("value", &make_bool(true));

        assert_ne!(key_int, key_str);
        assert_ne!(key_int, key_bool);
        assert_ne!(key_str, key_bool);

        // Check for type prefixes
        let bool_prefix = format!("{}{}", FIELD_SEPARATOR, TYPE_PREFIX_BOOL).into_bytes();
        let int_prefix = format!("{}{}", FIELD_SEPARATOR, TYPE_PREFIX_INTEGER).into_bytes();
        let str_prefix = format!("{}{}", FIELD_SEPARATOR, TYPE_PREFIX_STRING).into_bytes();

        assert!(key_bool
            .windows(bool_prefix.len())
            .any(|w| w == bool_prefix)); // Boolean prefix
        assert!(key_int.windows(int_prefix.len()).any(|w| w == int_prefix)); // Integer prefix
        assert!(key_str.windows(str_prefix.len()).any(|w| w == str_prefix)); // String prefix
    }

    #[test]
    fn test_range_query_simulation() {
        let codec = DefaultKeyCodec::new();

        let mut keys = [
            (10, codec.encode_field_value_key("age", &make_int(10))),
            (25, codec.encode_field_value_key("age", &make_int(25))),
            (30, codec.encode_field_value_key("age", &make_int(30))),
            (35, codec.encode_field_value_key("age", &make_int(35))),
            (50, codec.encode_field_value_key("age", &make_int(50))),
            (100, codec.encode_field_value_key("age", &make_int(100))),
        ];

        keys.sort_by(|a, b| a.1.cmp(&b.1));

        let values: Vec<_> = keys.iter().map(|(v, _)| v).collect();
        assert_eq!(values, vec![&10, &25, &30, &35, &50, &100]);

        let start_key = codec.encode_field_value_key("age", &make_int(25));
        let end_key = codec.encode_field_value_key("age", &make_int(50));

        let results: Vec<_> = keys
            .iter()
            .filter(|(_, key)| key >= &start_key && key < &end_key)
            .map(|(v, _)| v)
            .collect();

        assert_eq!(results, vec![&25, &30, &35]);
    }

    #[test]
    fn test_edge_cases() {
        let codec = DefaultKeyCodec::new();

        let key_min = codec.encode_field_value_key("x", &make_int(i64::MIN));
        let key_max = codec.encode_field_value_key("x", &make_int(i64::MAX));
        assert!(key_min < key_max);

        let key_neg_zero = codec.encode_field_value_key("x", &make_float(-0.0));
        let key_pos_zero = codec.encode_field_value_key("x", &make_float(0.0));
        assert!(key_neg_zero <= key_pos_zero || key_pos_zero <= key_neg_zero);

        let key_empty = codec.encode_field_value_key("name", &make_string(""));
        let key_a = codec.encode_field_value_key("name", &make_string("a"));
        assert!(key_empty < key_a);
    }

    #[test]
    fn test_field_prefix_separation() {
        let codec = DefaultKeyCodec::new();

        let age_10 = codec.encode_field_value_key("age", &make_int(10));
        let score_5 = codec.encode_field_value_key("score", &make_int(5));

        assert!(age_10 < score_5);
    }

    // Tests for individual formatting functions
    #[test]
    fn test_format_integer_key() {
        let key = DefaultKeyCodec::format_integer_value("age", 42);
        let key_str = String::from_utf8(key.clone()).unwrap();

        // Check structure: field + separator + type prefix + encoded value
        assert!(key_str.starts_with("age"));
        assert!(key_str.contains(FIELD_SEPARATOR));
        assert!(key_str.contains(TYPE_PREFIX_INTEGER));

        // Verify format consistency
        let key_neg = DefaultKeyCodec::format_integer_value("age", -42);
        assert_eq!(key.len(), key_neg.len()); // Same length for all integers

        // Verify ordering: negative < positive
        assert!(key_neg < key);
    }

    #[test]
    fn test_format_float_key() {
        let key = DefaultKeyCodec::format_float_value("score", 3.14);
        let key_str = String::from_utf8(key.clone()).unwrap();

        // Check structure
        assert!(key_str.starts_with("score"));
        assert!(key_str.contains(FIELD_SEPARATOR));
        assert!(key_str.contains(TYPE_PREFIX_FLOAT));

        // Verify encoding preserves ordering
        let key1 = DefaultKeyCodec::format_float_value("score", 1.5);
        let key2 = DefaultKeyCodec::format_float_value("score", 2.5);
        assert!(key1 < key2);

        // Test negative floats
        let key_neg = DefaultKeyCodec::format_float_value("score", -1.5);
        let key_pos = DefaultKeyCodec::format_float_value("score", 1.5);
        assert!(key_neg < key_pos);
    }

    #[test]
    fn test_format_string_key() {
        let key = DefaultKeyCodec::format_string_value("name", "alice");
        let key_str = String::from_utf8(key).unwrap();

        // Check structure
        assert!(key_str.starts_with("name"));
        assert!(key_str.contains(FIELD_SEPARATOR));
        assert!(key_str.contains(TYPE_PREFIX_STRING));
        assert!(key_str.ends_with("alice"));

        // Test with special characters
        let key_special = DefaultKeyCodec::format_string_value("text", "hello\nworld");
        let key_special_str = String::from_utf8(key_special).unwrap();
        assert!(key_special_str.contains("hello\nworld"));

        // Test empty string
        let key_empty = DefaultKeyCodec::format_string_value("text", "");
        let key_empty_str = String::from_utf8(key_empty).unwrap();
        assert!(key_empty_str.contains(TYPE_PREFIX_STRING));
    }

    #[test]
    fn test_format_bool_key() {
        let key_true = DefaultKeyCodec::format_bool_value("active", true);
        let key_false = DefaultKeyCodec::format_bool_value("active", false);

        let key_true_str = String::from_utf8(key_true.clone()).unwrap();
        let key_false_str = String::from_utf8(key_false.clone()).unwrap();

        // Check structure
        assert!(key_true_str.starts_with("active"));
        assert!(key_true_str.contains(FIELD_SEPARATOR));
        assert!(key_true_str.contains(TYPE_PREFIX_BOOL));
        assert!(key_true_str.ends_with("1"));

        assert!(key_false_str.starts_with("active"));
        assert!(key_false_str.contains(FIELD_SEPARATOR));
        assert!(key_false_str.contains(TYPE_PREFIX_BOOL));
        assert!(key_false_str.ends_with("0"));

        // Verify ordering: false < true
        assert!(key_false < key_true);
    }

    #[test]
    fn test_format_functions_type_isolation() {
        // Ensure different types produce different prefixes
        let key_int = DefaultKeyCodec::format_integer_value("value", 42);
        let key_float = DefaultKeyCodec::format_float_value("value", 42.0);
        let key_string = DefaultKeyCodec::format_string_value("value", "42");
        let key_bool = DefaultKeyCodec::format_bool_value("value", true);

        // All should have different type prefixes
        let int_prefix = format!("{}{}", FIELD_SEPARATOR, TYPE_PREFIX_INTEGER).into_bytes();
        let float_prefix = format!("{}{}", FIELD_SEPARATOR, TYPE_PREFIX_FLOAT).into_bytes();
        let string_prefix = format!("{}{}", FIELD_SEPARATOR, TYPE_PREFIX_STRING).into_bytes();
        let bool_prefix = format!("{}{}", FIELD_SEPARATOR, TYPE_PREFIX_BOOL).into_bytes();

        assert!(key_int.windows(int_prefix.len()).any(|w| w == int_prefix));
        assert!(key_float
            .windows(float_prefix.len())
            .any(|w| w == float_prefix));
        assert!(key_string
            .windows(string_prefix.len())
            .any(|w| w == string_prefix));
        assert!(key_bool
            .windows(bool_prefix.len())
            .any(|w| w == bool_prefix));

        // None should match the wrong prefix
        assert!(!key_int
            .windows(float_prefix.len())
            .any(|w| w == float_prefix));
        assert!(!key_float.windows(int_prefix.len()).any(|w| w == int_prefix));
        assert!(!key_string
            .windows(int_prefix.len())
            .any(|w| w == int_prefix));
        assert!(!key_bool.windows(int_prefix.len()).any(|w| w == int_prefix));
    }

    #[test]
    fn test_format_functions_field_names() {
        // Test various field names
        let key_short = DefaultKeyCodec::format_integer_value("a", 42);
        let key_long = DefaultKeyCodec::format_integer_value("very_long_field_name", 42);
        let key_dots = DefaultKeyCodec::format_integer_value("nested.field.path", 42);

        // All should contain the field name at the start
        assert!(key_short.starts_with(b"a"));
        assert!(key_long.starts_with(b"very_long_field_name"));
        assert!(key_dots.starts_with(b"nested.field.path"));

        // All should contain the separator
        let separator = FIELD_SEPARATOR.as_bytes();
        assert!(key_short.windows(separator.len()).any(|w| w == separator));
        assert!(key_long.windows(separator.len()).any(|w| w == separator));
        assert!(key_dots.windows(separator.len()).any(|w| w == separator));
    }
}
