/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io,
    sync::{Mutex, MutexGuard},
};

use windows_sys::Win32::Foundation::HANDLE;

/// `ThreadSafeHandle` struct that wraps a native Windows `HANDLE` object with a mutex to ensure thread safety.
pub struct ThreadSafeHandle(Mutex<HANDLE>);

/// Implement `Send` and `Sync` for `ThreadSafeHandle` to allow it to be shared across threads.
unsafe impl Send for ThreadSafeHandle {}
unsafe impl Sync for ThreadSafeHandle {}

impl ThreadSafeHandle {
    /// Lock the mutex and return a guard to the handle.
    pub fn lock(&self) -> io::Result<MutexGuard<'_, HANDLE>> {
        self.0.lock().map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                "Unable to acquire lock on ThreadSafeHandle.",
            )
        })
    }

    /// Create a new `ThreadSafeHandle` from a native Windows `HANDLE`.
    pub fn new(handle: HANDLE) -> Self {
        Self(Mutex::new(handle))
    }
}
