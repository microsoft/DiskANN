/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(target_os = "linux")]
use std::fs::File;

#[cfg(target_os = "linux")]
use io_uring::IoUring;

// The IOContext struct for disk I/O. One for each thread.
#[cfg(target_os = "linux")]
pub struct IOContext {
    pub file_handle: File,
    pub ring: IoUring,
}

#[cfg(target_os = "linux")]
impl IOContext {
    pub fn new(file_handle: File, ring: IoUring) -> Self {
        IOContext { file_handle, ring }
    }
}
