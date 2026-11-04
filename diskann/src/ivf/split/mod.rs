/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Policies that restructure the lists filled for one stage of a build operation.
//!
//! A split replaces lists with fitted children, and a reassignment moves points between
//! linked lists. Policies read a [`View`] and record changes through an [`Update`]; the
//! index finishes the update.

use std::fmt::Debug;

use diskann_utils::Parallelism;

use crate::{
    ANNError,
    ivf::region::{Update, View},
};

pub(crate) mod links;
pub(crate) mod policies;

pub(crate) use links::Links;
pub(crate) use policies::{KMeansSplit, Linked};

/////////////
/// Split ///
/////////////

/// A policy that replaces parent lists with new lists that take over their members.
pub(crate) trait Split<I, L, R>
where
    R: View<I, L>,
{
    /// The error returned by a failed split.
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Retire every parent and create lists holding all of its members, calling
    /// `emit(parent, child)` once per created list.
    ///
    /// Every parent must be filled into `region`.
    ///
    /// # Errors
    ///
    /// A failure may leave partial changes in `update`, so the caller must abandon the
    /// operation.
    fn split<U, F>(
        &mut self,
        region: &R,
        update: &mut U,
        parents: &[L],
        parallelism: Parallelism<'_>,
        emit: F,
    ) -> Result<(), Self::Error>
    where
        U: Update<I, L>,
        F: FnMut(L, L);
}

////////////////
/// Reassign ///
////////////////

/// A policy that moves points between linked lists without changing any centroid.
pub(crate) trait Reassign<I, L, R>
where
    R: View<I, L>,
{
    /// The error returned by a failed reassignment.
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Potentially re-assign points using links
    ///
    /// Every list in `links` must be filled into `region`.
    ///
    /// # Errors
    ///
    /// A failure may leave partial changes in `update`, so the caller must abandon the
    /// operation.
    fn reassign<U>(
        &mut self,
        region: &R,
        update: &mut U,
        links: &Links<L>,
        parallelism: Parallelism<'_>,
    ) -> Result<(), Self::Error>
    where
        U: Update<I, L>;
}
