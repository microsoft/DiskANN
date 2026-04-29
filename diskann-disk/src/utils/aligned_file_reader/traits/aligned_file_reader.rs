/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use crate::utils::aligned_file_reader::{aligned_read::Alignment, AlignedRead};

pub trait AlignedFileReader: Send + Sync {
    /// Alignment requirement applied to every `AlignedRead<u8, Self::Alignment>`
    /// passed to [`Self::read`]. All three constraints — buffer pointer,
    /// buffer length, and disk offset — are checked against this value at
    /// `AlignedRead::new` time.
    ///
    /// Direct-I/O readers (O_DIRECT, `FILE_FLAG_NO_BUFFERING`) set this to the
    /// device sector size (typically 512 bytes); buffered readers use `A1`.
    /// The caller is responsible for generating offsets and buffers that
    /// satisfy this alignment — `DiskSectorGraph`, for example, requires its
    /// `block_size` to be a multiple of `Self::Alignment::VALUE`.
    type Alignment: Alignment;

    /// Read the data from the file by sending concurrent io requests in batches.
    fn read(&mut self, read_requests: &mut [AlignedRead<u8, Self::Alignment>]) -> ANNResult<()>;
}
