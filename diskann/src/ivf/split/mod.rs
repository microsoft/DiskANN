/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete regional split operations, independent of physical list storage.

use std::fmt::Debug;

use crate::{
    ANNError,
    ivf::region::{Update, View},
};

pub(crate) mod plan;
pub(crate) mod policies;

pub(crate) use plan::{Dst, Plan};
pub(crate) use policies::TwoMeansSplit;

/// Replace parents and reassign their members together with selected neighbors.
pub(crate) trait Split<I, L, R>
where
    R: View<I, L>,
{
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Every list in `plan` must be filled. Other filled sources must not be edited.
    /// Implementations choose whether to restrict movement to parent neighborhoods.
    ///
    /// # Errors
    ///
    /// A failure may leave partial private updates. The caller must discard the
    /// operation rather than finalize it.
    fn split<U>(
        &mut self,
        region: &R,
        update: &mut U,
        plan: &Plan<L>,
    ) -> Result<(), Self::Error> where U: Update<I, L>;
}
