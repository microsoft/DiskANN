/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::AttributeValue;
use crate::parser::format::Document;
use serde_json::Value;
use std::fmt::{self, Write};
use std::{collections::HashMap, fmt::Display};
pub type Attributes = HashMap<String, AttributeValue>;

/// Configuration for JSON flattening behavior
#[derive(Debug, Clone)]
pub struct FlattenConfig {
    /// Separator used between path segments (default: "/")
    pub separator: String,
    /// Whether to include array indices in paths (default: true)
    pub include_array_indices: bool,
    /// Custom root prefix (default: "")
    pub root_prefix: String,
}

impl Default for FlattenConfig {
    fn default() -> Self {
        Self {
            separator: "/".to_string(),
            include_array_indices: true,
            root_prefix: "".to_string(),
        }
    }
}

impl FlattenConfig {
    /// Create config for JSON Pointer format (RFC 6901)
    pub fn json_pointer() -> Self {
        Self::default()
    }

    /// Create config for dot-separated paths
    pub fn dot_notation() -> Self {
        Self {
            separator: ".".to_string(),
            include_array_indices: true,
            root_prefix: "".to_string(),
        }
    }

    /// Create config for underscore-separated paths  
    pub fn underscore_notation() -> Self {
        Self {
            separator: "_".to_string(),
            include_array_indices: true,
            root_prefix: "".to_string(),
        }
    }

    /// Create config with custom separator
    pub fn with_separator(separator: &str) -> Self {
        Self {
            separator: separator.to_string(),
            ..Self::default()
        }
    }
}

impl Document {
    /// Flattens all nested fields in metadata to a simple key-value structure
    /// For example, {"details": {"color": "red"}} becomes {"/details/color": "red"}
    pub fn flatten_metadata(&self) -> Attributes {
        flatten_json_pointers_map(&self.label)
    }

    /// Configurable version that uses FlattenConfig
    /// For example, with config.separator=".": {"details": {"color": "red"}} becomes {".details.color": "red"}
    pub fn flatten_metadata_with_config(&self, config: &FlattenConfig) -> Attributes {
        flatten_json_pointers_map_with_config(&self.label, config)
    }

    /// Convenience wrapper for custom separator
    pub fn flatten_metadata_with_separator(&self, separator: &str) -> Attributes {
        let config = FlattenConfig {
            separator: separator.to_string(),
            ..FlattenConfig::default()
        };
        self.flatten_metadata_with_config(&config)
    }
}

/// Use a string as a pre-formatter buffer. For recursive calls,
/// `StringStack::push` can be used to format a value into the
/// underlying string and create a new `StringStack`. When that
/// stack is destroyed, the string will be truncated back to the
/// length it was before the new value was formatted.
struct StringStack<'a> {
    string: &'a mut String,
    prefix: usize,
}

impl<'a> StringStack<'a> {
    fn new(string: &'a mut String) -> Self {
        let prefix = string.len();
        Self { string, prefix }
    }
}

impl StringStack<'_> {
    fn push<'a, T>(&'a mut self, value: &T, separator: &str) -> StringStack<'a>
    where
        T: Display,
    {
        let prefix = self.string.len();
        write!(self.string, "{}{}", separator, value).unwrap();
        StringStack {
            string: self.string,
            prefix,
        }
    }
}

impl fmt::Display for StringStack<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.string)
    }
}

impl Drop for StringStack<'_> {
    fn drop(&mut self) {
        self.string.truncate(self.prefix)
    }
}

/// Recursively walk a JSON value producing JSON Pointer paths to every leaf (non-container) value.
fn flatten_json_pointer_inner(
    value: &Value,
    current: &str,
    out: &mut Vec<(String, AttributeValue)>,
    separator: &str,
) {
    let mut string = current.to_string();
    let stack = StringStack::new(&mut string);

    fn flatten_recursive(
        value: &Value,
        mut stack: StringStack<'_>,
        out: &mut Vec<(String, AttributeValue)>,
        separator: &str,
    ) {
        match value {
            Value::Object(map) => {
                for (k, v) in map.iter() {
                    flatten_recursive(v, stack.push(&k, separator), out, separator);
                }
            }
            Value::Array(arr) => {
                for (i, item) in arr.iter().enumerate() {
                    flatten_recursive(item, stack.push(&i, separator), out, separator);
                }
            }
            _ => {
                // Primitive leaf
                let rv = match AttributeValue::try_from(value) {
                    Ok(v) => v,
                    Err(_err) => {
                        panic!("Could not convert value");
                    }
                };
                out.push((stack.to_string(), rv));
            }
        }
    }

    flatten_recursive(value, stack, out, separator)
}

type AttributesVec = Vec<(String, AttributeValue)>;

/// Public API: Flatten a serde_json::Value into (JSON Pointer path, value) pairs.
///
/// - Produces one entry per primitive leaf (string/number/bool) and empty containers.
/// - Object keys and array indices are escaped per RFC6901.
/// - The root primitive (if the whole value is a primitive) uses the empty string path "".
///
/// Example:
/// {"a": {"b": [1, 2]}} -> [ ("/a/b/0", 1), ("/a/b/1", 2) ]
pub fn flatten_json_pointers(value: &Value) -> AttributesVec {
    let config = FlattenConfig::default();
    flatten_json_pointers_with_config(value, &config)
}

/// Configurable version that uses FlattenConfig
///
/// Example:
/// With config.separator="/": {"a": {"b": [1, 2]}} -> [ ("/a/b/0", 1), ("/a/b/1", 2) ]
pub fn flatten_json_pointers_with_config(value: &Value, config: &FlattenConfig) -> AttributesVec {
    let mut out = Vec::new();
    flatten_json_pointer_inner(value, &config.root_prefix, &mut out, &config.separator);
    out
}

/// Convenience wrapper for custom separator (uses default config with custom separator)
pub fn flatten_json_pointers_with_separator(value: &Value, separator: &str) -> AttributesVec {
    let config = FlattenConfig {
        separator: separator.to_string(),
        ..FlattenConfig::default()
    };
    flatten_json_pointers_with_config(value, &config)
}

/// Convenience helper returning a HashMap for pointer -> value (last write wins if duplicates)
pub fn flatten_json_pointers_map(value: &Value) -> HashMap<String, AttributeValue> {
    flatten_json_pointers(value)
        .into_iter()
        .collect::<HashMap<String, AttributeValue>>()
}

/// Configurable version that uses FlattenConfig
pub fn flatten_json_pointers_map_with_config(
    value: &Value,
    config: &FlattenConfig,
) -> HashMap<String, AttributeValue> {
    flatten_json_pointers_with_config(value, config)
        .into_iter()
        .collect::<HashMap<String, AttributeValue>>()
}

/// Convenience wrapper for custom separator
pub fn flatten_json_pointers_map_with_separator(
    value: &Value,
    separator: &str,
) -> HashMap<String, AttributeValue> {
    let config = FlattenConfig {
        separator: separator.to_string(),
        ..FlattenConfig::default()
    };
    flatten_json_pointers_map_with_config(value, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::flatten_utils::Document;
    use serde_json::json;

    #[test]
    fn test_flatten_metadata() {
        let label = Document {
            doc_id: 123,
            label: json!({
                "name": "Product X",
                "details": {
                    "color": "blue",
                    "dimensions": {
                        "width": 10,
                        "height": 20
                    }
                },
                "tags": ["electronics", "featured"]
            }),
        };

        let flattened = label.flatten_metadata();

        // Check flattened fields
        assert_eq!(
            flattened.get("/name").unwrap(),
            &AttributeValue::String("Product X".into())
        );
        assert_eq!(
            flattened.get("/details/color").unwrap(),
            &AttributeValue::String("blue".into())
        );
        assert_eq!(
            flattened.get("/details/dimensions/width").unwrap(),
            &AttributeValue::Integer(10)
        );
        assert_eq!(
            flattened.get("/details/dimensions/height").unwrap(),
            &AttributeValue::Integer(20)
        );
        assert_eq!(
            flattened.get("/tags/0").unwrap(),
            &AttributeValue::String("electronics".into())
        );
        assert_eq!(
            flattened.get("/tags/1").unwrap(),
            &AttributeValue::String("featured".into())
        );
    }

    #[test]
    fn test_flatten_json_path() {
        let value = json!({
            "name": "Widget",
            "details": {"size": {"w": 10, "h": 20}},
            "tags": ["a", "b"],
            "empty_obj": {},
            "empty_arr": [],
            "arr": [1,2]
        });

        let flattened = flatten_json_pointers(&value);
        let map: HashMap<_, _> = flattened.iter().cloned().collect();

        assert_eq!(
            map.get("/name").unwrap(),
            &AttributeValue::String("Widget".into())
        );
        assert_eq!(
            map.get("/details/size/w").unwrap(),
            &AttributeValue::Integer(10)
        );
        assert_eq!(
            map.get("/details/size/h").unwrap(),
            &AttributeValue::Integer(20)
        );
        assert_eq!(
            map.get("/tags/0").unwrap(),
            &AttributeValue::String("a".into())
        );
        assert_eq!(
            map.get("/tags/1").unwrap(),
            &AttributeValue::String("b".into())
        );
        // empty containers omitted
        assert!(!map.contains_key("/empty_obj"));
        assert!(!map.contains_key("/empty_arr"));
        assert_eq!(map.get("/arr/0").unwrap(), &AttributeValue::Integer(1));
        assert_eq!(map.get("/arr/1").unwrap(), &AttributeValue::Integer(2));
    }

    #[test]
    fn test_flatten_with_config_dot_notation() {
        let value = json!({
            "name": "Test",
            "details": {"size": 42}
        });

        let config = FlattenConfig::dot_notation();
        let map = flatten_json_pointers_map_with_config(&value, &config);

        assert_eq!(
            map.get(".name").unwrap(),
            &AttributeValue::String("Test".into())
        );
        assert_eq!(
            map.get(".details.size").unwrap(),
            &AttributeValue::Integer(42)
        );
    }

    #[test]
    fn test_flatten_with_config_custom_separator() {
        let value = json!({
            "user": {"name": "John"}
        });

        let config = FlattenConfig::with_separator("_");
        let map = flatten_json_pointers_map_with_config(&value, &config);

        assert_eq!(
            map.get("_user_name").unwrap(),
            &AttributeValue::String("John".into())
        );
    }

    #[test]
    fn test_document_flatten_with_config() {
        let doc = Document {
            doc_id: 1,
            label: json!({
                "category": "test",
                "metadata": {"version": 1}
            }),
        };

        // Test with dot notation
        let config = FlattenConfig::dot_notation();
        let attrs = doc.flatten_metadata_with_config(&config);

        assert_eq!(
            attrs.get(".category").unwrap(),
            &AttributeValue::String("test".into())
        );
        assert_eq!(
            attrs.get(".metadata.version").unwrap(),
            &AttributeValue::Integer(1)
        );

        // Test with separator convenience method
        let attrs_sep = doc.flatten_metadata_with_separator(":");
        assert_eq!(
            attrs_sep.get(":category").unwrap(),
            &AttributeValue::String("test".into())
        );
    }
}
