/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, always_escalate};

/// A tag type indicating that a method fails via panic instead of returning an error.
///
/// This is an enum with no alternatives, so is impossible to construct. Therefore, we know
/// that there can never be an actual value with this type.
///
#[derive(Debug)]
pub enum Panics {}

impl std::fmt::Display for Panics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "panics")
    }
}

impl std::error::Error for Panics {}
impl From<Panics> for ANNError {
    #[cold]
    fn from(_: Panics) -> ANNError {
        ANNError::log_async_error("unreachable")
    }
}

always_escalate!(Panics);
