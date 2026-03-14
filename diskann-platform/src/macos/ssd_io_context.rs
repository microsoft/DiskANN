/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fs::File;

/// The IOContext struct for disk I/O on macOS.
///
/// macOS does not have io_uring (Linux-only) or IOCompletionPort (Windows-only).
/// This is a simple implementation that just holds a file handle.
/// For actual async I/O operations on macOS, use the StorageProviderAlignedFileReader
/// which provides a cross-platform fallback implementation.
pub struct IOContext {
    pub file_handle: File,
}

impl IOContext {
    pub fn new(file_handle: File) -> Self {
        IOContext { file_handle }
    }
}
