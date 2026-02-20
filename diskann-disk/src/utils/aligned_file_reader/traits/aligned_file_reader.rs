/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use crate::utils::aligned_file_reader::AlignedRead;

pub trait AlignedFileReader: Send + Sync {
    /// Read the data from the file by sending concurrent io requests in batches.
    fn read(&mut self, read_requests: &mut [AlignedRead<u8>]) -> ANNResult<()>;
}
