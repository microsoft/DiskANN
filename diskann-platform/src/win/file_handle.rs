/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{ffi::CString, io, ptr};

use windows_sys::Win32::{
    Foundation::{
        CloseHandle, GetLastError, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
    },
    Storage::FileSystem::{
        CreateFileA, FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED, FILE_FLAG_RANDOM_ACCESS,
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    },
};

use super::DWORD;

pub const FILE_ATTRIBUTE_READONLY: DWORD = 0x00000001;

/// `AccessMode` determines how a file can be accessed.
/// These modes are used when creating or opening a file to decide what operations are allowed
/// to be performed on the file.
///
/// # Variants
///
/// - `Read`: The file is opened in read-only mode.
///
/// - `Write`: The file is opened in write-only mode.
///
/// - `ReadWrite`: The file is opened for both reading and writing.
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

/// `ShareMode` determines how a file can be shared.
///
/// These modes are used when creating or opening a file to decide what operations other
/// opening instances of the file can perform on it.
/// # Variants
/// - `None`: Prevents other processes from opening a file if they request delete,
///   read, or write access.
///
/// - `Read`: Allows subsequent open operations on the same file to request read access.
///
/// - `Write`: Allows subsequent open operations on the same file file to request write access.
///
/// - `Delete`: Allows subsequent open operations on the same file file to request delete access.
pub enum ShareMode {
    None,
    Read,
    Write,
    Delete,
}

/// # Windows File Handle Wrapper
///
/// Introduces a Rust-friendly wrapper around the native Windows `HANDLE` object, `FileHandle`.
/// `FileHandle` provides safe creation and automatic cleanup of Windows file handles, leveraging Rust's ownership model.
///
/// `FileHandle` struct that wraps a native Windows `HANDLE` object
pub struct FileHandle {
    pub handle: HANDLE,
}
// SAFETY: THIS IS NOT ENTIRELY SAFE! PLEASE READ!
//
// The Windows API functions `ReadFile` and `GetQueuedCompletionStatus` are safe to call
// from multiple threads when using the OVERLAPPED API.
// ReadFile Function - https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile
// Synchronous and Asynchronous I/O - https://learn.microsoft.com/en-us/windows/win32/FileIO/synchronous-and-asynchronous-i-o
//
// However, `GetQueuedCompletionStatus` will not behave as expected as it will return when
// **any** request completes, not just the request associated with a particular thread.
//
// Our code uses `GetQueuedCompletionStatus` and therefore functions on the `FileHandle`
// may only be reliably called when a thread has exclusive access to that handle.
//
// This is done through the `WindowsAlignedFileReader` but is not captured in the type system.
//
// The correct long-term solution is to remove these implementations to make `FileHandle`
// not sharable between threads and embed it in a higher-level wrapper that is acquired early
// in search to guarantee exclusive access.
unsafe impl Send for FileHandle {}
unsafe impl Sync for FileHandle {}

impl FileHandle {
    /// Creates a new `FileHandle` by opening an existing file with the given access and shared mode.
    ///
    /// This function is marked unsafe because it creates a raw pointer to the filename and try to create
    /// a Windows `HANDLE` object without checking if you have sufficient permissions.
    ///
    /// # Safety
    ///
    /// Ensure that the file specified by `file_name` is valid and the calling process has
    /// sufficient permissions to perform the specified `access_mode` and `share_mode` operations.
    ///
    /// # Parameters
    ///
    /// - `file_name`: The name of the file.
    /// - `access_mode`: The access mode to be used for the file.
    /// - `share_mode`: The share mode to be used for the file
    ///
    /// # Errors
    /// This function will return an error if the `file_name` is invalid or if the file cannot
    /// be opened with the specified `access_mode` and `share_mode`.
    pub unsafe fn new(
        file_name: &str,
        access_mode: AccessMode,
        share_mode: ShareMode,
    ) -> io::Result<Self> {
        let file_name_c = CString::new(file_name).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid file name. {}", file_name),
            )
        })?;

        let dw_desired_access = match access_mode {
            AccessMode::Read => GENERIC_READ,
            AccessMode::Write => GENERIC_WRITE,
            AccessMode::ReadWrite => GENERIC_READ | GENERIC_WRITE,
        };

        let dw_share_mode = match share_mode {
            ShareMode::None => 0,
            ShareMode::Read => FILE_SHARE_READ,
            ShareMode::Write => FILE_SHARE_WRITE,
            ShareMode::Delete => FILE_SHARE_DELETE,
        };

        let dw_flags_and_attributes = FILE_ATTRIBUTE_READONLY
            | FILE_FLAG_NO_BUFFERING
            | FILE_FLAG_OVERLAPPED
            | FILE_FLAG_RANDOM_ACCESS;

        let handle = unsafe {
            CreateFileA(
                Self::as_windows_pcstr(&file_name_c),
                dw_desired_access,
                dw_share_mode,
                ptr::null_mut(),
                OPEN_EXISTING,
                dw_flags_and_attributes,
                std::ptr::null_mut(),
            )
        };

        if handle == INVALID_HANDLE_VALUE {
            let error_code = unsafe { GetLastError() };
            Err(io::Error::from_raw_os_error(error_code as i32))
        } else {
            Ok(Self { handle })
        }
    }

    fn as_windows_pcstr(str: &CString) -> ::windows_sys::core::PCSTR {
        str.as_ptr() as ::windows_sys::core::PCSTR
    }
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        let result = unsafe { CloseHandle(self.handle) };
        if result == 0 {
            let error_code = unsafe { GetLastError() };
            let error = io::Error::from_raw_os_error(error_code as i32);
            tracing::warn!("Error when dropping FileHandle: {:?}", error);
        }
    }
}

/// Returns a `FileHandle` with an `INVALID_HANDLE_VALUE`.
impl Default for FileHandle {
    fn default() -> Self {
        FileHandle {
            handle: INVALID_HANDLE_VALUE,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::Path};

    use super::*;

    #[test]
    fn test_create_file() {
        // Create a dummy file
        let dummy_file_path = "dummy_file.txt";
        {
            let _file = File::create(dummy_file_path).expect("Failed to create dummy file.");
        }

        let path = Path::new(dummy_file_path);
        {
            let file_handle = unsafe {
                FileHandle::new(path.to_str().unwrap(), AccessMode::Read, ShareMode::Read)
            };

            // Check that the file handle is valid
            assert!(file_handle.is_ok());
        }

        // Try to delete the file. If the handle was correctly dropped, this should succeed.
        match std::fs::remove_file(dummy_file_path) {
            Ok(()) => (), // File was deleted successfully, which means the handle was closed.
            Err(e) => panic!("Failed to delete file: {}", e), // Failed to delete the file, likely because the handle is still open.
        }
    }

    #[test]
    fn test_file_not_found() {
        let path = Path::new("non_existent_file.txt");
        let file_handle =
            unsafe { FileHandle::new(path.to_str().unwrap(), AccessMode::Read, ShareMode::Read) };

        // Check that opening a non-existent file returns an error
        assert!(file_handle.is_err());
    }
}
