/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

/// Failures specific to e-means training.
#[derive(Debug, Error)]
pub enum EmeansError<E = std::convert::Infallible> {
    #[error("invalid e-means configuration: {0}")]
    InvalidConfig(&'static str),
    #[error(
        "e-means requires nonempty matrices with equal, nonzero dimensions and 1 <= k <= min(n, u32::MAX)"
    )]
    InvalidShape,
    #[error("e-means workspace or update count overflows its supported size")]
    SizeOverflow,
    #[error("e-means requires finite coordinates and squared row norms <= f32::MAX / 8")]
    NumericalRange,
    #[error("e-means training was canceled")]
    Canceled,
    #[error("assignment backend failed")]
    Assignment(#[source] E),
    #[error(
        "assignment backend returned cluster {label} for input row {row}, expected an index below {clusters}"
    )]
    InvalidAssignment {
        row: usize,
        label: u32,
        clusters: usize,
    },
}

impl EmeansError {
    pub(super) fn with_assignment_error<E>(self) -> EmeansError<E> {
        match self {
            Self::InvalidConfig(message) => EmeansError::InvalidConfig(message),
            Self::InvalidShape => EmeansError::InvalidShape,
            Self::SizeOverflow => EmeansError::SizeOverflow,
            Self::NumericalRange => EmeansError::NumericalRange,
            Self::Canceled => EmeansError::Canceled,
            Self::InvalidAssignment {
                row,
                label,
                clusters,
            } => EmeansError::InvalidAssignment {
                row,
                label,
                clusters,
            },
            Self::Assignment(never) => match never {},
        }
    }
}
