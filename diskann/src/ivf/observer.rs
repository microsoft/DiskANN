/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Optional observation points for IVF insertion.

/// A measured phase within an IVF split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertPhase {
    /// Fit two centroids for every split parent.
    TwoMeans,
    /// Assign the complete split working set across all fitted children.
    Reassignment,
}

/// An event emitted by [`super::IVFIndex::insert_batch_observed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertEvent {
    /// A split phase is about to begin.
    PhaseStarted(InsertPhase),
    /// A split phase completed successfully.
    PhaseFinished(InsertPhase),
}
