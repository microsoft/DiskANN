/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNError;
use thiserror::Error;

/// A general utility struct for type-erasing errors related to cache reads and writes.
///
/// This error records:
///
/// 1. The actual error message via [`std::error::Error::source()`].
/// 2. The key being accessed.
/// 3. The type of operation (read or write).
/// 4. The file and line where the [`CacheAccessError`] is created.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct CacheAccessError(Box<dyn std::error::Error + Send + Sync>);

impl From<CacheAccessError> for ANNError {
    #[track_caller]
    fn from(err: CacheAccessError) -> Self {
        Self::opaque(err)
    }
}

impl CacheAccessError {
    /// Construct a new [`CacheAccessError`] recording a critical read failure for the key
    /// with error payload `source.
    ///
    /// The error payload is available using the [`std::error::Error::source`] method.
    #[track_caller]
    #[inline(never)]
    pub fn read<K, I>(key: K, source: I) -> Self
    where
        K: Send + Sync + std::fmt::Debug + 'static,
        I: std::error::Error + Send + Sync + 'static,
    {
        Self(Box::new(Inner {
            op: Op::Read,
            key,
            source,
            location: std::panic::Location::caller(),
        }))
    }

    /// Construct a new [`CacheAccessError`] recording a critical write failure for the key
    /// with error payload `source.
    ///
    /// The error payload is available using the [`std::error::Error::source`] method.
    #[track_caller]
    #[inline(never)]
    pub fn write<K, I>(key: K, source: I) -> Self
    where
        K: Send + Sync + std::fmt::Debug + 'static,
        I: std::error::Error + Send + Sync + 'static,
    {
        Self(Box::new(Inner {
            op: Op::Write,
            key,
            source,
            location: std::panic::Location::caller(),
        }))
    }
}

#[derive(Debug)]
enum Op {
    Read,
    Write,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "reading"),
            Self::Write => write!(f, "writing"),
        }
    }
}

#[derive(Debug, Error)]
#[error("cache access error while {op} key {key:?} at {}:{}", self.location.file(), self.location.line())]
struct Inner<K, I> {
    op: Op,
    key: K,
    source: I,
    location: &'static std::panic::Location<'static>,
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Error)]
    #[error("error A")]
    struct ErrorA {}

    #[derive(Debug, Error)]
    #[error("error B, value = {value}")]
    struct ErrorB {
        value: usize,
        source: ErrorA,
    }

    #[test]
    fn test_read_error() {
        use std::error::Error;

        let err = ErrorB {
            value: 10,
            source: ErrorA {},
        };

        let line = line!() + 1;
        let err = CacheAccessError::read(5, err);

        let msg = err.to_string();
        assert!(
            msg.starts_with("cache access error while reading key 5"),
            "{}",
            msg
        );
        assert!(msg.contains(&line.to_string()), "{}", msg);
        assert!(!msg.contains("error A"), "{}", msg);
        assert!(!msg.contains("error B"), "{}", msg);

        let source = err.source().unwrap();
        let source_msg = source.to_string();
        assert_eq!(source_msg, "error B, value = 10");

        let source = source.source().unwrap();
        let source_msg = source.to_string();
        assert_eq!(source_msg, "error A");
    }

    #[test]
    fn test_write_error() {
        use std::error::Error;

        let err = ErrorB {
            value: 10,
            source: ErrorA {},
        };

        let line = line!() + 1;
        let err = CacheAccessError::write(5, err);

        let msg = err.to_string();
        assert!(
            msg.starts_with("cache access error while writing key 5"),
            "{}",
            msg
        );
        assert!(msg.contains(&line.to_string()), "{}", msg);
        assert!(!msg.contains("error A"), "{}", msg);
        assert!(!msg.contains("error B"), "{}", msg);

        let source = err.source().unwrap();
        let source_msg = source.to_string();
        assert_eq!(source_msg, "error B, value = 10");

        let source = source.source().unwrap();
        let source_msg = source.to_string();
        assert_eq!(source_msg, "error A");
    }

    #[test]
    fn test_conversion_to_ann_error() {
        let err = ErrorB {
            value: 10,
            source: ErrorA {},
        };

        let first_line = line!() + 1;
        let err = CacheAccessError::write(5, err);

        let second_line = line!() + 1;
        let err: ANNError = err.into();
        let msg = err.to_string();

        assert!(msg.contains(&first_line.to_string()), "msg = {}", msg);
        assert!(msg.contains(&second_line.to_string()), "msg = {}", msg);

        assert!(
            msg.contains("cache access error while writing key 5"),
            "msg = {}",
            msg
        );
        assert!(msg.contains("error B, value = 10"), "msg = {}", msg);
        assert!(msg.contains("error A"), "msg = {}", msg);
    }
}
