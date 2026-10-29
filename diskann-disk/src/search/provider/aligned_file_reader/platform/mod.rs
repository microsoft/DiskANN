/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Platform-specific I/O primitives backing the native aligned file readers.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::IOContext;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::{
    DWORD, FileHandle, IOCompletionPort, IOContext, OVERLAPPED, ULONG_PTR,
    get_queued_completion_status, read_file_to_slice,
};
