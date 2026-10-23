/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete regional split operations, independent of physical list storage.

use std::fmt::Debug;

use crate::{
    ANNError,
    ivf::region::{RegionUpdate, RegionView},
};

pub(crate) mod twomeans;
pub(crate) use twomeans::TwoMeansSplitter;

/// Replace parents and reassign their members together with selected neighbors.
pub(crate) trait Split<I, L, R: RegionView<I, L>> {
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Parents and neighbors must be distinct, disjoint filled sources.
    /// Other filled sources must not be edited.
    ///
    /// # Errors
    ///
    /// A failure may leave partial private updates. The caller must discard the
    /// operation rather than finalize it.
    fn split<U: RegionUpdate<I, L>>(
        &mut self,
        region: &R,
        update: &mut U,
        parents: &[L],
        neighbors: &[L],
    ) -> Result<(), Self::Error>;
}
