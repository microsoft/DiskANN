/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Dispatch Rules
//!
//! This module provides the [`DispatchRule`] trait and supporting types for value-to-type
//! matching and conversion.
//!
//! [`DispatchRule`] is used by benchmark implementations to match runtime enum values
//! (e.g., `DataType::Float32`) to static Rust types (e.g., `Type<f32>`), enabling
//! type-driven overload resolution.

mod api;

pub use api::{
    Description, DispatchRule, FailureScore, MatchScore, TaggedFailureScore, Why,
    IMPLICIT_MATCH_SCORE,
};

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDescription;

    impl DispatchRule<usize> for TestDescription {
        type Error = std::convert::Infallible;

        fn try_match(_from: &usize) -> Result<MatchScore, FailureScore> {
            panic!("should not be called");
        }

        fn convert(_from: usize) -> Result<Self, Self::Error> {
            panic!("should not be called");
        }
    }

    ///////////////////
    // Test Routines //
    ///////////////////

    #[test]
    fn test_empty_description() {
        assert_eq!(
            Description::<usize, TestDescription>::new().to_string(),
            "<no description>"
        );
        assert_eq!(
            Why::<usize, TestDescription>::new(&0).to_string(),
            "<no description>"
        );
    }
}
