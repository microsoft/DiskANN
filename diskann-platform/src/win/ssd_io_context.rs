/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{FileHandle, IOCompletionPort};

// The IOContext struct for disk I/O. One for each thread.
pub struct IOContext {
    pub status: Status,
    pub file_handle: FileHandle,
    pub io_completion_port: IOCompletionPort,
}

impl Default for IOContext {
    fn default() -> Self {
        IOContext {
            status: Status::ReadWait,
            file_handle: FileHandle::default(),
            io_completion_port: IOCompletionPort::default(),
        }
    }
}

impl IOContext {
    pub fn new() -> Self {
        Self::default()
    }
}

pub enum Status {
    ReadWait,
    ReadSuccess,
    ProcessComplete,
}
