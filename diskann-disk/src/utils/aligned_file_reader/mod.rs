/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod traits;

mod aligned_read;
pub use aligned_read::AlignedRead;

cfg_if::cfg_if! {
    if #[cfg(all(not(miri), target_os = "linux"))] {
        pub mod linux_aligned_file_reader;
        pub use linux_aligned_file_reader::LinuxAlignedFileReader;
    } else if #[cfg(all(not(miri), target_os = "windows"))] {
        #[allow(clippy::module_inception)]
        pub mod windows_aligned_file_reader;
        pub use windows_aligned_file_reader::WindowsAlignedFileReader;
    }

}

cfg_if::cfg_if! {
    if #[cfg(any(feature = "virtual_storage", test))] {
        pub mod virtual_aligned_reader_factory;
        pub use virtual_aligned_reader_factory::VirtualAlignedReaderFactory;
    }
}

pub mod storage_provider_aligned_file_reader;
pub use storage_provider_aligned_file_reader::StorageProviderAlignedFileReader;

pub mod aligned_file_reader_factory;
pub use aligned_file_reader_factory::AlignedFileReaderFactory;
