/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

#[cfg(all(not(miri), target_os = "linux"))]
use super::LinuxAlignedFileReader;
#[cfg(miri)]
use super::StorageProviderAlignedFileReader;
#[cfg(all(not(miri), target_os = "windows"))]
use super::WindowsAlignedFileReader;
use crate::utils::aligned_file_reader::traits::AlignedReaderFactory;

pub struct AlignedFileReaderFactory {
    pub file_path: String,
}

impl AlignedReaderFactory for AlignedFileReaderFactory {
    /*
        Fall back to the StorageProviderAlignedFileReader when running in miri. Otherwise, miri fails with this error:
       --> C:\Users\<user>\.cargo\registry\src\msdata.pkgs.visualstudio.com-32ec7033fece98f6\io-uring-0.6.3\src\sys\mod.rs:97:15
        |
    97  |     to_result(syscall(SYSCALL_SETUP, entries as c_long, p as c_long) as _)
        |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ can't execute syscall with ID 425
        |
        = help: this is likely not a bug in the program; it indicates that the program performed an operation that the interpreter does not support
        = note: BACKTRACE on thread `algorithm::sear`:
        = note: inside `io_uring::sys::io_uring_setup` at C:\Users\<user>\.cargo\registry\src\msdata.pkgs.visualstudio.com-32ec7033fece98f6\io-uring-0.6.3\src\sys\mod.rs:97:15: 97:69
        = note: inside `io_uring::IoUring::with_params` at C:\Users\<user>\.cargo\registry\src\msdata.pkgs.visualstudio.com-32ec7033fece98f6\io-uring-0.6.3\src\lib.rs:152:57: 152:93
        = note: inside `io_uring::Builder::build` at C:\Users\<user>\.cargo\registry\src\msdata.pkgs.visualstudio.com-32ec7033fece98f6\io-uring-0.6.3\src\lib.rs:412:20: 412:62
        = note: inside `io_uring::IoUring::new` at C:\Users\<user>\.cargo\registry\src\msdata.pkgs.visualstudio.com-32ec7033fece98f6\io-uring-0.6.3\src\lib.rs:82:9: 82:39
    note: inside `<model::aligned_file_reader::linux_aligned_file_reader::LinuxAlignedFileReader as model::aligned_file_reader::aligned_file_reader::AlignedFileReader>::read`
       --> diskann\src\model\aligned_file_reader\linux_aligned_file_reader.rs:221:24
        |
    221 |         let mut ring = IoUring::new(MAX_IO_CONCURRENCY as u32)?;
         */
    #[cfg(miri)]
    type AlignedReaderType = StorageProviderAlignedFileReader;

    #[cfg(all(not(miri), target_os = "linux"))]
    type AlignedReaderType = LinuxAlignedFileReader;

    #[cfg(all(not(miri), target_os = "windows"))]
    type AlignedReaderType = WindowsAlignedFileReader;

    fn build(&self) -> ANNResult<Self::AlignedReaderType> {
        #[cfg(miri)]
        return StorageProviderAlignedFileReader::new(
            &crate::storage::FileStorageProvider,
            self.file_path.as_str(),
        );

        #[cfg(all(not(miri), target_os = "windows"))]
        return WindowsAlignedFileReader::new(self.file_path.as_str());

        #[cfg(all(not(miri), target_os = "linux"))]
        return LinuxAlignedFileReader::new(self.file_path.as_str());
    }
}

impl AlignedFileReaderFactory {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_file_reader_factory_new() {
        let path = "/tmp/test.bin".to_string();
        let factory = AlignedFileReaderFactory::new(path.clone());
        assert_eq!(factory.file_path, path);
    }

    #[test]
    fn test_aligned_file_reader_factory_implements_trait() {
        // Verify that AlignedFileReaderFactory implements AlignedReaderFactory
        fn check_impl<T: AlignedReaderFactory>() {}
        check_impl::<AlignedFileReaderFactory>();
    }

    #[test]
    fn test_aligned_file_reader_factory_field_access() {
        let factory = AlignedFileReaderFactory::new("/path/to/file".to_string());
        assert_eq!(factory.file_path, "/path/to/file");
    }
}
