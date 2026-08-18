/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Concrete [`AlignedFileReader`](super::traits::AlignedFileReader) implementations.

cfg_if::cfg_if! {
    if #[cfg(all(not(miri), target_os = "linux"))] {
        mod linux;
        pub use linux::LinuxAlignedFileReader;
    } else if #[cfg(all(not(miri), target_os = "windows"))] {
        mod windows;
        pub use windows::WindowsAlignedFileReader;
    }
}

mod storage_provider;
pub use storage_provider::StorageProviderAlignedFileReader;
