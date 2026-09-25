/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{FileHandle, IOCompletionPort};

// The IOContext struct for disk I/O. One for each thread.
#[derive(Default)]
pub struct IOContext {
    pub file_handle: FileHandle,
    pub io_completion_port: IOCompletionPort,
}

impl IOContext {
    pub fn new() -> Self {
        Self::default()
    }
}
