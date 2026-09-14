/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk-specific storage operations.
//!
//! This module contains storage utilities and types
//! specific to disk index operations.

pub mod disk_index_reader;

mod disk_index_writer;
pub use disk_index_writer::DiskIndexWriter;

pub mod quant;

pub mod api;

use std::io::{self, BufReader};

use diskann_providers::storage::StorageReadProvider;

/// Opens a buffered reader, limiting its capacity to the file length.
pub(crate) fn open_buf_reader<S: StorageReadProvider>(
    provider: &S,
    path: &str,
    capacity: usize,
) -> io::Result<BufReader<S::Reader>> {
    let capacity = (capacity as u64).min(provider.get_length(path)?) as usize;
    Ok(BufReader::with_capacity(
        capacity,
        provider.open_reader(path)?,
    ))
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    use diskann_providers::storage::{StorageWriteProvider, VirtualStorageProvider};

    use super::open_buf_reader;

    #[test]
    fn buffered_reader_capacity_is_bounded_by_file_length() -> std::io::Result<()> {
        let provider = VirtualStorageProvider::new_memory();
        for size in [0, 4, 8, 12] {
            let data = vec![42; size];
            let mut writer = provider.create_for_write("/buffered.bin")?;
            writer.write_all(&data)?;
            writer.flush()?;
            drop(writer);

            let mut reader = open_buf_reader(&provider, "/buffered.bin", 8)?;
            assert_eq!(reader.capacity(), size.min(8));
            let mut actual = Vec::new();
            reader.read_to_end(&mut actual)?;
            assert_eq!(actual, data);
        }
        assert!(open_buf_reader(&provider, "/missing.bin", 8).is_err());
        Ok(())
    }
}
