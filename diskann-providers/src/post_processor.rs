/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Unified post-processor parameter types with validation.
//!
//! This module provides centralized definitions and validation for post-processor
//! parameters like Determinant-Diversity, ensuring consistent validation across
//! different search contexts (in-memory, disk, benchmarking).

use std::fmt;

/// Parameters for Determinant-Diversity post-processor with validation.
///
/// Determinant-Diversity is a diversity-promoting reranking algorithm that takes
/// relevance-ranked neighbors and reorders them to maximize geometric diversity
/// while maintaining relevance.
///
/// # Parameters
///
/// - `power`: Relevance weighting exponent. Controls the emphasis on maintaining
///   relevance scores from the original search. Must be > 0.0.
///
/// - `eta`: Numerical stability parameter for ridge-regularization. Controls the
///   trade-off between exact determinant computation (eta=0) and numerical robustness
///   (eta>0). Must be >= 0.0.
///
/// # Errors
///
/// Construction fails if:
/// - `power` is non-finite or `<= 0.0` (invalid power weighting)
/// - `eta` is non-finite or `< 0.0` (negative stability parameter)
#[derive(Debug, Clone, Copy)]
pub struct DeterminantDiversityParams {
    /// Relevance weighting exponent. Must be > 0.0.
    power: f32,
    /// Numerical stability parameter. Must be >= 0.0.
    eta: f32,
}

impl DeterminantDiversityParams {
    /// Create and validate new Determinant-Diversity parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails:
    /// - `power` is non-finite or `<= 0.0`: invalid relevance weighting
    /// - `eta` is non-finite or `< 0.0`: invalid numerical stability parameter
    pub fn new(power: f32, eta: f32) -> Result<Self, DeterminantDiversityError> {
        if !power.is_finite() || power <= 0.0 {
            return Err(DeterminantDiversityError::InvalidPower(power));
        }
        if !eta.is_finite() || eta < 0.0 {
            return Err(DeterminantDiversityError::InvalidEta(eta));
        }
        Ok(Self { power, eta })
    }

    /// Get power parameter.
    #[inline]
    pub fn power(&self) -> f32 {
        self.power
    }

    /// Get eta parameter.
    #[inline]
    pub fn eta(&self) -> f32 {
        self.eta
    }
}

impl fmt::Display for DeterminantDiversityParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeterminantDiversity(power={}, eta={})",
            self.power, self.eta
        )
    }
}

/// Validation error for Determinant-Diversity parameters.
#[derive(Debug, Clone)]
pub enum DeterminantDiversityError {
    /// Power parameter <= 0.0
    InvalidPower(f32),
    /// Eta parameter < 0.0
    InvalidEta(f32),
}

impl fmt::Display for DeterminantDiversityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPower(p) => {
                write!(f, "determinant-diversity power must be > 0.0, got: {}", p)
            }
            Self::InvalidEta(e) => {
                write!(f, "determinant-diversity eta must be >= 0.0, got: {}", e)
            }
        }
    }
}

impl std::error::Error for DeterminantDiversityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_params() {
        assert!(DeterminantDiversityParams::new(1.0, 0.0).is_ok());
        assert!(DeterminantDiversityParams::new(0.5, 1.5).is_ok());
        assert!(DeterminantDiversityParams::new(2.0, 0.1).is_ok());
    }

    #[test]
    fn test_invalid_power() {
        assert!(DeterminantDiversityParams::new(0.0, 1.0).is_err());
        assert!(DeterminantDiversityParams::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_invalid_eta() {
        assert!(DeterminantDiversityParams::new(1.0, -0.1).is_err());
    }

    #[test]
    fn test_invalid_non_finite_values() {
        assert!(DeterminantDiversityParams::new(f32::NAN, 0.1).is_err());
        assert!(DeterminantDiversityParams::new(f32::INFINITY, 0.1).is_err());
        assert!(DeterminantDiversityParams::new(1.0, f32::NAN).is_err());
        assert!(DeterminantDiversityParams::new(1.0, f32::INFINITY).is_err());
    }

    #[test]
    fn test_display() {
        let params = DeterminantDiversityParams::new(1.5, 0.5).unwrap();
        assert_eq!(
            params.to_string(),
            "DeterminantDiversity(power=1.5, eta=0.5)"
        );
    }
}
