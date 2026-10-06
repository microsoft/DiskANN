/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Layers that can be added to existing strategies such as
//! [`diskann::graph::glue::SearchStrategy`].
//!
//! Implementations:
//!
//! * [`BetaFilter`]

pub mod betafilter;
pub use betafilter::BetaFilter;
