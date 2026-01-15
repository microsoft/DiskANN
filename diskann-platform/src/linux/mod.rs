/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

mod perf;
pub use perf::{
    get_number_of_processors, get_peak_workingset_size, get_process_cycle_time, get_process_time,
    get_system_time,
};

pub mod ssd_io_context;
