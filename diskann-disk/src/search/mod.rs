/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Model module containing data structures, providers, and traits for disk index operations

pub mod provider;
pub mod traits;

pub(crate) mod sector_math;
pub mod search_trace;

#[cfg(target_os = "linux")]
pub mod pipelined;
