/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Layers that can be added to existing strategies such as
//! [`diskann::glue::SearchStrategy`].
//!
//! Implementations:
//!
//! * [`BetaFilter`]

mod betafilter;
pub use betafilter::BetaFilter;
