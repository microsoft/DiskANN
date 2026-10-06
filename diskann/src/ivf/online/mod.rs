/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The online insert algorithm: stage and route a batch, then split every list the
//! batch overflows.
//!
//! Planning performs all of its accessor reads first and then computes the update
//! from owned, densely indexed data. Everything read from the accessor is validated
//! as it arrives, so the updates built from it need no further checks. Nothing here
//! mutates the index; the caller applies the planned update.

mod kernels;
mod split;

use std::fmt::{Debug, Display};

pub(super) use split::{TwoMeans, two_means};

use crate::{ANNError, ANNErrorKind};

#[track_caller]
pub(super) fn index_error<D>(message: D) -> ANNError
where
    D: Display + Debug + Send + Sync + 'static,
{
    ANNError::message(ANNErrorKind::IndexError, message)
}
