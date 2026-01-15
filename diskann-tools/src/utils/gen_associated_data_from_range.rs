/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann_providers::storage::StorageWriteProvider;
use diskann_providers::{storage::FileStorageProvider, utils::write_metadata};

use super::CMDResult;

pub fn gen_associated_data_from_range(
    storage_provider: &FileStorageProvider,
    associated_data_path: &str,
    start: u32,
    end: u32,
) -> CMDResult<()> {
    let mut file = storage_provider.create_for_write(associated_data_path)?;

    // Calculate the number of integers and the number of integers in associated data
    let num_ints = end - start + 1;
    let int_length: u32 = 1;

    // Write the number of integers and the length of each integer as little endian
    write_metadata(&mut file, num_ints, int_length)?;

    // Write the integers from the range as little endian
    for i in start..=end {
        file.write_all(&i.to_le_bytes())?;
    }

    Ok(())
}
