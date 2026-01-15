/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod ann_error;
pub(crate) use ann_error::ensure_positive;
pub use ann_error::{ANNError, ANNErrorKind, ANNResult, DiskANNError, ErrorContext, IntoANNResult};

pub(crate) mod ranked;
pub use ranked::{ErrorExt, Infallible, NeverTransient, RankedError, ToRanked, TransientError};

pub trait StandardError: std::error::Error + Send + Sync + 'static + Into<ANNError> {}
impl<T> StandardError for T where T: std::error::Error + Send + Sync + 'static + Into<ANNError> {}

#[cfg(any(test, feature = "testing"))]
macro_rules! message {
    ($kind:ident, $($args:tt)*) => {
        $crate::ANNError::message($kind, format!($($args)*))
    };
    ($($args:tt)*) => {
        $crate::ANNError::message($crate::ANNErrorKind::Opaque, format!($($args)*))
    };
}

#[cfg(any(test, feature = "testing"))]
pub(crate) use message;
