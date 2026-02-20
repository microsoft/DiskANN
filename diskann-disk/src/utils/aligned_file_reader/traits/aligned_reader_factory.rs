/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use super::AlignedFileReader;

pub trait AlignedReaderFactory: Send + Sync {
    type AlignedReaderType: AlignedFileReader;

    fn build(&self) -> ANNResult<Self::AlignedReaderType>;
}
