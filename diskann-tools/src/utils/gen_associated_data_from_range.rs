/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann_providers::storage::StorageWriteProvider;
use diskann_utils::io::Metadata;

use super::{CMDResult, CMDToolError};

pub fn gen_associated_data_from_range<S: StorageWriteProvider>(
    storage_provider: &S,
    associated_data_path: &str,
    start: u32,
    end: u32,
) -> CMDResult<()> {
    if end < start {
        return Err(CMDToolError {
            details: format!(
                "invalid range: end ({end}) must be greater than or equal to start ({start})"
            ),
        });
    }

    let mut file = storage_provider.create_for_write(associated_data_path)?;

    // Calculate the number of integers and the number of integers in associated data
    // Use checked arithmetic to avoid overflow when end == u32::MAX and start == 0
    let num_ints = (end - start).checked_add(1).ok_or_else(|| CMDToolError {
        details: format!("range [{start}, {end}] is too large: count overflows u32"),
    })?;
    let int_length: u32 = 1;

    // Write the number of integers and the length of each integer as little endian
    Metadata::new(num_ints, int_length)?.write(&mut file)?;

    // Write the integers from the range as little endian
    for i in start..=end {
        file.write_all(&i.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, ReadBytesExt};
    use diskann_providers::storage::{StorageReadProvider, VirtualStorageProvider};

    #[test]
    fn test_gen_associated_data_from_range() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let path = "/test_gen_associated_data_from_range.bin";

        // Generate data from range 0 to 9
        gen_associated_data_from_range(&storage_provider, path, 0, 9).unwrap();

        // Read back and verify
        let mut file = storage_provider.open_reader(path).unwrap();

        // Read metadata
        let num_ints = file.read_u32::<LittleEndian>().unwrap();
        let int_length = file.read_u32::<LittleEndian>().unwrap();

        assert_eq!(num_ints, 10);
        assert_eq!(int_length, 1);

        // Read integers
        for expected in 0u32..=9 {
            let actual = file.read_u32::<LittleEndian>().unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_gen_associated_data_from_range_single_value() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let path = "/test_gen_associated_data_single.bin";

        // Generate data for a single value
        gen_associated_data_from_range(&storage_provider, path, 42, 42).unwrap();

        let mut file = storage_provider.open_reader(path).unwrap();

        let num_ints = file.read_u32::<LittleEndian>().unwrap();
        let int_length = file.read_u32::<LittleEndian>().unwrap();

        assert_eq!(num_ints, 1);
        assert_eq!(int_length, 1);

        let value = file.read_u32::<LittleEndian>().unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_gen_associated_data_from_range_large() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let path = "/test_gen_associated_data_large.bin";

        // Generate data for range 100 to 199
        gen_associated_data_from_range(&storage_provider, path, 100, 199).unwrap();

        let mut file = storage_provider.open_reader(path).unwrap();

        let num_ints = file.read_u32::<LittleEndian>().unwrap();
        let int_length = file.read_u32::<LittleEndian>().unwrap();

        assert_eq!(num_ints, 100);
        assert_eq!(int_length, 1);

        for expected in 100u32..=199 {
            let actual = file.read_u32::<LittleEndian>().unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_gen_associated_data_from_range_end_less_than_start() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let path = "/test_gen_associated_data_invalid.bin";

        let result = gen_associated_data_from_range(&storage_provider, path, 10, 5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.details.contains("end") && err.details.contains("start"),
            "error message should mention end and start: {err}"
        );
        assert!(
            err.details.contains("5") && err.details.contains("10"),
            "error message should include the specific values 5 and 10: {err}"
        );
    }

    #[test]
    fn test_gen_associated_data_from_range_max_overflow() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let path = "/test_gen_associated_data_overflow.bin";

        // end == u32::MAX and start == 0 would make count == u32::MAX + 1, which overflows
        let result = gen_associated_data_from_range(&storage_provider, path, 0, u32::MAX);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.details.contains("overflow") || err.details.contains("too large"),
            "error message should mention overflow or too large: {err}"
        );
    }
}
