/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Error types for the index module.

/// Error types for query evaluation operations.
///
/// This enum represents errors that can occur during query evaluation,
/// separate from underlying storage or serialization errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    /// Query operation not supported
    UnsupportedOperation {
        /// The operation that's not supported (e.g., "NOT", "NE")
        operation: String,
        /// Explanation and possible alternatives
        reason: String,
    },

    /// Invalid query value or expression
    InvalidQuery {
        /// The field or expression that's invalid
        context: String,
        /// Why the query is invalid
        reason: String,
    },

    /// Underlying index error (storage, serialization, etc.)
    Index(IndexError),
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::UnsupportedOperation { operation, reason } => {
                write!(
                    f,
                    "Query operation '{}' not supported: {}",
                    operation, reason
                )
            }
            QueryError::InvalidQuery { context, reason } => {
                write!(f, "Invalid query at '{}': {}", context, reason)
            }
            QueryError::Index(err) => {
                write!(f, "Index error during query: {}", err)
            }
        }
    }
}

impl std::error::Error for QueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            QueryError::Index(err) => Some(err),
            _ => None,
        }
    }
}

impl From<IndexError> for QueryError {
    fn from(err: IndexError) -> Self {
        QueryError::Index(err)
    }
}

impl QueryError {
    /// Create an unsupported operation error
    pub fn unsupported(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create an invalid query error
    pub fn invalid_query(context: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidQuery {
            context: context.into(),
            reason: reason.into(),
        }
    }
}

/// Error types for index operations.
///
/// This enum provides structured error information that can be matched on
/// programmatically, enabling proper error recovery and handling logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexError {
    /// Key-value store operation failed
    KvStore {
        /// The operation that failed (e.g., "get", "set", "delete", "range")
        operation: String,
        /// The key involved in the operation, if applicable
        key: Option<Vec<u8>>,
        /// The underlying error message from the KV store
        source: String,
    },

    /// Failed to serialize or deserialize data
    Serialization {
        /// The type of data being serialized (e.g., "posting_list", "key_list")
        data_type: String,
        /// The underlying error message
        source: String,
    },

    /// Data corruption detected during deserialization
    CorruptData {
        /// Where the corruption was detected (e.g., "reverse_key_list", "posting_list")
        location: String,
        /// Description of what's wrong
        reason: String,
    },

    /// Integer overflow in size calculation
    Overflow {
        /// The operation that caused the overflow
        operation: String,
    },

    /// Invalid input value (e.g., NaN, infinity for floats)
    InvalidValue {
        /// The field name
        field: String,
        /// Why the value is invalid
        reason: String,
    },

    /// Operation not supported
    Unsupported {
        /// The operation that's not supported
        operation: String,
        /// Explanation and possible alternatives
        reason: String,
    },

    /// Document not found in the index
    DocumentNotFound {
        /// The document ID that wasn't found
        doc_id: usize,
    },
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::KvStore {
                operation,
                key,
                source,
            } => {
                write!(f, "KV store error during '{}': {}", operation, source)?;
                if let Some(k) = key {
                    write!(f, " (key: {} bytes)", k.len())?;
                }
                Ok(())
            }
            IndexError::Serialization { data_type, source } => {
                write!(f, "Serialization error for '{}': {}", data_type, source)
            }
            IndexError::CorruptData { location, reason } => {
                write!(f, "Corrupt data at '{}': {}", location, reason)
            }
            IndexError::Overflow { operation } => {
                write!(f, "Integer overflow in '{}'", operation)
            }
            IndexError::InvalidValue { field, reason } => {
                write!(f, "Invalid value for field '{}': {}", field, reason)
            }
            IndexError::Unsupported { operation, reason } => {
                write!(f, "Operation '{}' not supported: {}", operation, reason)
            }
            IndexError::DocumentNotFound { doc_id } => {
                write!(f, "Document {} not found in index", doc_id)
            }
        }
    }
}

impl std::error::Error for IndexError {}

// Ergonomic constructor methods
impl IndexError {
    /// Create a KV store error without key information
    pub fn kv_store(operation: impl Into<String>, source: impl std::fmt::Display) -> Self {
        Self::KvStore {
            operation: operation.into(),
            key: None,
            source: source.to_string(),
        }
    }

    /// Create a KV store error with key information
    pub fn kv_store_with_key(
        operation: impl Into<String>,
        key: Vec<u8>,
        source: impl std::fmt::Display,
    ) -> Self {
        Self::KvStore {
            operation: operation.into(),
            key: Some(key),
            source: source.to_string(),
        }
    }

    /// Create a serialization error
    pub fn serialization(data_type: impl Into<String>, source: impl std::fmt::Display) -> Self {
        Self::Serialization {
            data_type: data_type.into(),
            source: source.to_string(),
        }
    }

    /// Create a corrupt data error
    pub fn corrupt_data(location: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::CorruptData {
            location: location.into(),
            reason: reason.into(),
        }
    }

    /// Create an overflow error
    pub fn overflow(operation: impl Into<String>) -> Self {
        Self::Overflow {
            operation: operation.into(),
        }
    }

    /// Create an invalid value error
    pub fn invalid_value(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidValue {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Create an unsupported operation error
    pub fn unsupported(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Unsupported {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create a document not found error
    pub fn document_not_found(doc_id: usize) -> Self {
        Self::DocumentNotFound { doc_id }
    }
}

/// Result type alias for index operations.
pub type Result<T> = std::result::Result<T, IndexError>;

#[cfg(test)]
mod error_handling_tests {
    use crate::kv_index::error::{IndexError, QueryError};
    use std::error::Error;

    #[test]
    fn test_error_kv_store_variant() {
        let err = IndexError::kv_store("get", "test message");
        assert!(matches!(
            &err,
            IndexError::KvStore { operation, key: None, source }
            if operation == "get" && source == "test message"
        ));
        assert!(err.to_string().contains("test message"));
    }

    #[test]
    fn test_error_kv_store_with_key_variant() {
        let test_key = vec![1, 2, 3];
        let err = IndexError::kv_store_with_key("set", test_key.clone(), "write failed");
        assert!(matches!(
            err,
            IndexError::KvStore { operation, key: Some(k), source }
            if operation == "set" && k == test_key && source == "write failed"
        ));
    }

    #[test]
    fn test_error_serialization_variant() {
        let err = IndexError::serialization("posting_list", "invalid format");
        assert!(matches!(
            err,
            IndexError::Serialization { data_type, source }
            if data_type == "posting_list" && source == "invalid format"
        ));
    }

    #[test]
    fn test_error_corrupt_data_variant() {
        let err = IndexError::corrupt_data("reverse_key_list", "buffer too short");
        assert!(matches!(
            err,
            IndexError::CorruptData { location, reason }
            if location == "reverse_key_list" && reason == "buffer too short"
        ));
    }

    #[test]
    fn test_error_overflow_variant() {
        let err = IndexError::overflow("serialize_key_list");
        assert!(matches!(
            &err,
            IndexError::Overflow { operation }
            if operation == "serialize_key_list"
        ));
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn test_error_invalid_value_variant() {
        let err = IndexError::invalid_value("min_bound", "NaN is not valid");
        assert!(matches!(
            err,
            IndexError::InvalidValue { field, reason }
            if field == "min_bound" && reason == "NaN is not valid"
        ));
    }

    #[test]
    fn test_error_unsupported_variant() {
        let err = IndexError::unsupported("NOT", "operation not implemented");
        assert!(matches!(
            &err,
            IndexError::Unsupported { operation, reason }
            if operation == "NOT" && reason == "operation not implemented"
        ));
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn test_error_document_not_found_variant() {
        let err = IndexError::document_not_found(12345);
        assert!(matches!(
            err,
            IndexError::DocumentNotFound { doc_id }
            if doc_id == 12345
        ));
        assert!(err.to_string().contains("12345"));
    }

    #[test]
    fn test_error_display_implementation() {
        let err = IndexError::kv_store("get", "key not found");
        let display_str = format!("{}", err);
        assert!(display_str.contains("get"));
        assert!(display_str.contains("key not found"));
    }

    #[test]
    fn test_error_is_send_sync() {
        // This is a compile-time test to ensure IndexError implements Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<IndexError>();
    }

    #[test]
    fn test_error_is_std_error() {
        // Test that IndexError implements std::error::Error
        let err = IndexError::overflow("test");
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_error_display_messages() {
        let err = IndexError::kv_store("get", "connection timeout");
        assert_eq!(
            err.to_string(),
            "KV store error during 'get': connection timeout"
        );

        let err = IndexError::corrupt_data("posting_list", "invalid format");
        assert_eq!(
            err.to_string(),
            "Corrupt data at 'posting_list': invalid format"
        );

        let err = IndexError::overflow("capacity calculation");
        assert_eq!(
            err.to_string(),
            "Integer overflow in 'capacity calculation'"
        );

        let err = IndexError::invalid_value("price", "NaN or infinity");
        assert_eq!(
            err.to_string(),
            "Invalid value for field 'price': NaN or infinity"
        );

        let err = IndexError::unsupported("NOT", "requires document universe");
        assert_eq!(
            err.to_string(),
            "Operation 'NOT' not supported: requires document universe"
        );

        let err = IndexError::document_not_found(42);
        assert_eq!(err.to_string(), "Document 42 not found in index");
    }

    #[test]
    fn test_error_cloneable() {
        let err = IndexError::kv_store("test", "error");
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_serialization_error() {
        let err = IndexError::serialization("posting_list", "parse error");
        assert!(matches!(err, IndexError::Serialization { .. }));
    }

    #[test]
    fn test_kv_store_error_with_key() {
        let key = vec![1, 2, 3];
        let err = IndexError::kv_store_with_key("set", key.clone(), "disk full");

        if let IndexError::KvStore {
            operation,
            key: Some(k),
            source,
        } = err
        {
            assert_eq!(operation, "set");
            assert_eq!(k, key);
            assert_eq!(source, "disk full");
        } else {
            panic!("Expected KvStore error with key");
        }
    }

    #[test]
    fn test_unsupported_operation_error() {
        let err = IndexError::unsupported("NOT", "requires document universe tracking");

        if let IndexError::Unsupported { operation, reason } = err {
            assert_eq!(operation, "NOT");
            assert!(reason.contains("document universe"));
        } else {
            panic!("Expected Unsupported error");
        }
    }

    // QueryError tests
    #[test]
    fn test_query_error_unsupported() {
        let err = QueryError::unsupported("NOT", "requires universe");
        assert!(matches!(err, QueryError::UnsupportedOperation { .. }));
        assert!(err.to_string().contains("NOT"));
        assert!(err.to_string().contains("requires universe"));
    }

    #[test]
    fn test_query_error_invalid_query() {
        let err = QueryError::invalid_query("field1", "NaN value");
        assert!(matches!(err, QueryError::InvalidQuery { .. }));
        assert!(err.to_string().contains("field1"));
        assert!(err.to_string().contains("NaN value"));
    }

    #[test]
    fn test_query_error_from_index_error() {
        let index_err = IndexError::kv_store("get", "timeout");
        let query_err: QueryError = index_err.into();
        assert!(matches!(query_err, QueryError::Index(_)));
        assert!(query_err.to_string().contains("timeout"));
    }

    #[test]
    fn test_query_error_source() {
        let index_err = IndexError::kv_store("get", "source error");
        let query_err = QueryError::Index(index_err);

        // Test that source() returns the inner error
        let source = query_err.source();
        assert!(source.is_some());
        assert!(source.unwrap().to_string().contains("source error"));
    }

    #[test]
    fn test_query_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QueryError>();
    }

    #[test]
    fn test_query_error_is_std_error() {
        let err = QueryError::unsupported("test", "reason");
        let _: &dyn std::error::Error = &err;
    }
}
