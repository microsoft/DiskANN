/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete destination memberships produced by regional clustering.

/// Complete membership of a new or retained destination list.
#[derive(Debug)]
pub struct Partition<I, L> {
    /// `Some` retains a source list's identity and centroid; `None` creates a new list.
    pub id: Option<L>,
    pub members: Vec<I>,
}
