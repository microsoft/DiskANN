/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Regional split and reassignment operations, independent of physical list storage.

use std::fmt::Debug;

use crate::{
    ANNError,
    ivf::region::{Update, View},
};

pub(crate) mod links;
pub(crate) mod policies;

pub(crate) use links::Links;
pub(crate) use policies::{KMeansSplit, Linked};

/// Replace parents with new lists that take over their members.
pub(crate) trait Split<I, L, R>
where
    R: View<I, L>,
{
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Retire every parent and create lists holding all of its members, calling
    /// `emit(parent, child)` once per created list.
    ///
    /// # Errors
    ///
    /// A failure may leave partial private updates. The caller must discard the
    /// operation rather than finalize it.
    fn split<U, F>(
        &mut self,
        region: &R,
        update: &mut U,
        parents: &[L],
        emit: F,
    ) -> Result<(), Self::Error>
    where
        U: Update<I, L>,
        F: FnMut(L, L);
}

/// Move points between linked lists without changing any centroid.
pub(crate) trait Reassign<I, L, R>
where
    R: View<I, L>,
{
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Every list in `links` must be filled into `region`
    ///
    /// # Errors
    ///
    /// A failure may leave partial private updates. The caller must discard the
    /// operation rather than finalize it.
    fn reassign<U>(
        &mut self,
        region: &R,
        update: &mut U,
        links: &Links<L>,
    ) -> Result<(), Self::Error>
    where
        U: Update<I, L>;
}
