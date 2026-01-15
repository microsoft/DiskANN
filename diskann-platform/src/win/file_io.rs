/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
/// The module provides unsafe wrappers around two Windows API functions: `ReadFile` and `GetQueuedCompletionStatus`.
///
/// These wrappers aim to simplify and abstract the use of these functions, providing easier error handling and a safer interface.
/// They return standard Rust `io::Result` types for convenience and consistency with the rest of the Rust standard library.
use std::io;
use std::ptr;

use windows_sys::Win32::{
    Foundation::{GetLastError, ERROR_IO_PENDING, WAIT_TIMEOUT},
    Storage::FileSystem::ReadFile,
    System::IO::{GetQueuedCompletionStatus, OVERLAPPED},
};

use super::{DWORD, ULONG_PTR};
use crate::{FileHandle, IOCompletionPort};

/// Asynchronously queue a read request from a file into a buffer slice.
///
/// Wraps the unsafe Windows API function `ReadFile`, making it safe to call only when the overlapped buffer
/// remains valid and unchanged anywhere else during the entire async operation.
///
/// Returns a boolean indicating whether the read operation completed synchronously or is pending.
///
/// # Safety
///
/// This function is marked as `unsafe` because it uses raw pointers and requires the caller to ensure
/// that the buffer slice and the overlapped buffer stay valid during the whole async operation.
///
/// SAFETY: THIS IS NOT ENTIRELY SAFE! PLEASE READ!
///
/// This function is thread safe i.e. the same file handle can be used by multiple threads to read from the file
/// as it uses the windows ReadFile API with async mode using OVERLAPPED structure.
/// ReadFile Function - https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile
/// Synchronous and Asynchronous I/O - https://learn.microsoft.com/en-us/windows/win32/FileIO/synchronous-and-asynchronous-i-o
///
/// The only caveat is read operation is followed by polling on the handle using GetQueuedCompletionStatus API.
/// If multiple threads are submitting read requests and polling then polling will return the completion status of any of the read requests.
/// This is because GetQueuedCompletionStatus API returns the completion status of any of the read requests that are completed.
pub unsafe fn read_file_to_slice<T>(
    file_handle: &FileHandle,
    buffer_slice: &mut [T],
    overlapped: *mut OVERLAPPED,
    offset: u64,
) -> io::Result<bool> {
    let num_bytes = std::mem::size_of_val(buffer_slice);
    unsafe {
        ptr::write(overlapped, std::mem::zeroed());
        (*overlapped).Anonymous.Anonymous.Offset = offset as u32;
        (*overlapped).Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
    }

    let win32_result: i32 = unsafe {
        ReadFile(
            file_handle.handle,
            buffer_slice.as_mut_ptr().cast::<u8>(),
            num_bytes as DWORD,
            ptr::null_mut(),
            overlapped,
        )
    };

    // `ReadFile` returns zero on failure.
    if win32_result == 0 {
        let error = unsafe { GetLastError() };
        return if error != ERROR_IO_PENDING {
            Err(io::Error::from_raw_os_error(error as i32))
        } else {
            Ok(false)
        };
    }

    Ok(true)
}

/// Retrieves the results of an asynchronous I/O operation on an I/O completion port.
///
/// Wraps the unsafe Windows API function `GetQueuedCompletionStatus`, making it safe to call only when the overlapped buffer
/// remains valid and unchanged anywhere else during the entire async operation.
///
/// Returns a boolean indicating whether an I/O operation completed synchronously or is still pending.
///
/// # Safety
///
/// This function is marked as `unsafe` because it uses raw pointers and requires the caller to ensure
/// that the overlapped buffer stays valid during the whole async operation.
///
/// SAFETY: THIS IS NOT ENTIRELY SAFE! PLEASE READ!
///
/// This function is thread safe i.e. the same file handle can be used by multiple threads to read from the file
/// as it uses the windows ReadFile API with async mode using OVERLAPPED structure.
/// ReadFile Function - https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile
/// Synchronous and Asynchronous I/O - https://learn.microsoft.com/en-us/windows/win32/FileIO/synchronous-and-asynchronous-i-o
///
/// The only caveat is read operation is followed by polling on the handle using GetQueuedCompletionStatus API.
/// If multiple threads are submitting read requests and polling then polling will return the completion status of any of the read requests.
/// This is because GetQueuedCompletionStatus API returns the completion status of any of the read requests that are completed.
pub unsafe fn get_queued_completion_status(
    completion_port: &IOCompletionPort,
    lp_number_of_bytes: &mut DWORD,
    lp_completion_key: &mut ULONG_PTR,
    lp_overlapped: *mut *mut OVERLAPPED,
    dw_milliseconds: DWORD,
) -> io::Result<bool> {
    let result = unsafe {
        GetQueuedCompletionStatus(
            *completion_port.mutex_guarded_handle()?,
            lp_number_of_bytes,
            lp_completion_key,
            lp_overlapped,
            dw_milliseconds,
        )
    };

    match result {
        0 => {
            let error = unsafe { GetLastError() };
            if error == WAIT_TIMEOUT {
                Ok(false)
            } else {
                Err(io::Error::from_raw_os_error(error as i32))
            }
        }
        _ => Ok(true),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write, path::Path};

    use super::*;
    use crate::win::file_handle::{AccessMode, ShareMode};

    #[test]
    fn test_read_file_to_slice() {
        // Create a temporary file and write some data into it
        let path = Path::new("temp.txt");
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"Hello, world!").unwrap();
        }

        let mut buffer: [u8; 512] = [0; 512];
        let mut overlapped = unsafe { std::mem::zeroed::<OVERLAPPED>() };
        {
            let file_handle = unsafe {
                FileHandle::new(path.to_str().unwrap(), AccessMode::Read, ShareMode::Read)
            }
            .unwrap();

            // Call the function under test
            let result =
                unsafe { read_file_to_slice(&file_handle, &mut buffer, &mut overlapped, 0) };

            assert!(result.is_ok());
            let result_str = std::str::from_utf8(&buffer[.."Hello, world!".len()]).unwrap();
            assert_eq!(result_str, "Hello, world!");
        }

        // Clean up
        std::fs::remove_file("temp.txt").unwrap();
    }
}
