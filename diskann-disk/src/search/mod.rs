/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Model module containing data structures, providers, and traits for disk index operations

pub mod provider;
pub mod traits;

#[cfg(target_os = "linux")]
pub mod pipelined;
