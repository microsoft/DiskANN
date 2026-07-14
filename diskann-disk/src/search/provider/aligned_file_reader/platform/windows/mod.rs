/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub use file_handle::FileHandle;
pub use file_io::{get_queued_completion_status, read_file_to_slice};
pub use io_completion_port::IOCompletionPort;
pub use ssd_io_context::IOContext;

mod file_handle;
mod file_io;
mod io_completion_port;
mod ssd_io_context;

#[allow(non_camel_case_types)]
pub type ULONG_PTR = usize;
#[allow(clippy::upper_case_acronyms)]
pub type DWORD = u32;
#[allow(clippy::upper_case_acronyms)]
pub type OVERLAPPED = windows_sys::Win32::System::IO::OVERLAPPED;
