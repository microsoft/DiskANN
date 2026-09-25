/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod traits;

mod aligned_read;
pub use aligned_read::{AlignedRead, Alignment, A1, A512};

#[cfg(all(not(miri), any(target_os = "linux", target_os = "windows")))]
mod platform;

mod reader;
#[cfg(all(not(miri), target_os = "linux"))]
pub use reader::LinuxAlignedFileReader;
pub use reader::StorageProviderAlignedFileReader;
#[cfg(all(not(miri), target_os = "windows"))]
pub use reader::WindowsAlignedFileReader;

mod factory;
pub use factory::AlignedFileReaderFactory;
#[cfg(test)]
pub(crate) use factory::VirtualAlignedReaderFactory;
