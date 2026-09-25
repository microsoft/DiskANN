/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fs::File;

use io_uring::IoUring;

// The IOContext struct for disk I/O. One for each thread.
pub struct IOContext {
    // Kept alive so the file descriptor registered with the io_uring ring stays valid;
    // the fd is used via `register_files`, so this field is never read directly.
    #[allow(dead_code)]
    pub file_handle: File,
    pub ring: IoUring,
}

impl IOContext {
    pub fn new(file_handle: File, ring: IoUring) -> Self {
        IOContext { file_handle, ring }
    }
}
