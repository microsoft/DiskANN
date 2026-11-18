/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Policies that restructure the lists filled for one stage of a build operation.
//!
//! A split replaces lists with fitted children, and a reassignment moves points between
//! linked lists. Policies read a [`Region`] and record changes through it; the index
//! finishes the region.

use std::fmt::Debug;

use diskann_quantization::Parallelism;

use crate::{ANNError, ivf::region::Region};

pub(crate) mod links;
pub(crate) mod policies;

pub(crate) use links::Links;
pub(crate) use policies::{KMeansSplit, Linked};

///////////
// Split //
///////////

/// A policy that replaces parent lists with new lists that take over their members.
pub(crate) trait Split<I, L, R>
where
    R: Region<I, L>,
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
    /// A failure may leave partial changes in `region`, so the caller must abandon the
    /// operation.
    fn split<F>(
        &mut self,
        region: &mut R,
        parents: &[L],
        parallelism: Parallelism,
        emit: F,
    ) -> Result<(), Self::Error>
    where
        F: FnMut(L, L);
}

//////////////
// Reassign //
//////////////

/// A policy that moves points between linked lists without changing any centroid.
pub(crate) trait Reassign<I, L, R>
where
    R: Region<I, L>,
{
    /// The error returned by a failed reassignment.
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Potentially re-assign points using links
    ///
    /// Every list in `links` must be filled into `region`.
    ///
    /// # Errors
    ///
    /// A failure may leave partial changes in `region`, so the caller must abandon the
    /// operation.
    fn reassign(
        &mut self,
        region: &mut R,
        links: &Links<L>,
        parallelism: Parallelism,
    ) -> Result<(), Self::Error>;
}
