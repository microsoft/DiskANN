/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Semver-style version stamps embedded in every saved object.

use serde::{Deserialize, Serialize};

/// A semver-style schema version attached to every saved record.
///
/// Each [`crate::save::Save`] / [`crate::load::Load`] impl declares its
/// `const VERSION: Version`. On load, the version recorded in the manifest is compared
/// against the declared version to dispatch between
/// [`Load::load`](crate::load::Load::load) and
/// [`Load::load_legacy`](crate::load::Load::load_legacy).
///
/// The framework treats versions as opaque triples and only checks them for equality;
/// ordering / semver semantics are entirely up to the implementing type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    /// Construct a [`Version`] from its three components.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}
