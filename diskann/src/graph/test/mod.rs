/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod provider;
pub mod synthetic;

// Every case in this module drives the index through the tokio runtime.
#[cfg(all(test, feature = "tokio"))]
mod cases;
