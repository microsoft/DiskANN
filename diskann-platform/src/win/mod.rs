/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

pub use file_handle::{AccessMode, FileHandle, ShareMode};
pub use file_io::{get_queued_completion_status, read_file_to_slice};
pub use io_completion_port::IOCompletionPort;
pub use perf::{
    get_number_of_processors, get_peak_workingset_size, get_process_cycle_time, get_process_time,
    get_system_time,
};
pub use thread_safe_handle::ThreadSafeHandle;

mod file_handle;
mod file_io;
mod io_completion_port;
mod perf;
pub mod ssd_io_context;
mod thread_safe_handle;

#[allow(non_camel_case_types)]
pub type ULONG_PTR = usize;
pub type DWORD = u32;
pub type OVERLAPPED = windows_sys::Win32::System::IO::OVERLAPPED;
