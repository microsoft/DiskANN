/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Borrowed-matrix fitting and assignment helpers and squared-Euclidean kernels.

mod clustering;
mod kernels;

use std::fmt::{Debug, Display};

pub(super) use clustering::{assign_nearest, fit_two_means};
pub(super) use kernels::LloydScratch;

use crate::ANNError;

#[track_caller]
pub(super) fn index_error<D>(message: D) -> ANNError
where
    D: Display + Debug + Send + Sync + 'static,
{
    ANNError::message(message)
}
