/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Set module for label storage backends.
//!
//! This module defines traits and implementations for working with labels as sets.
//! It provides a common abstraction over different backends, so the same interface can be
//! used regardless of the underlying system.

pub mod roaring_set;
pub mod roaring_set_provider;
pub mod traits;

pub use traits::{Set, SetProvider};
