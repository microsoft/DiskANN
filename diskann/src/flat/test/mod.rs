/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Test fixtures and helpers for the flat module.
//!
//! Mirrors the layout of [`crate::graph::test`]: shared visitor / strategy / data helpers
//! live at this level, while end-to-end baseline-cached tests live under [`cases`].

pub(crate) mod harness;
pub(crate) mod provider;

mod cases;
