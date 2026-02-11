/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[allow(clippy::module_inception)]
mod aligned_file_reader;
pub use aligned_file_reader::AlignedFileReader;

pub mod aligned_reader_factory;
pub use aligned_reader_factory::AlignedReaderFactory;

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_structure() {
        // Module structure is verified at compile time
    }
}
