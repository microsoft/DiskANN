/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Dataset dto used for other layer, such as storage
/// N is the aligned dimension
#[derive(Debug)]
pub struct DatasetDto<'a, T> {
    /// data slice borrow from dataset
    pub data: &'a mut [T],

    /// rounded dimension
    pub rounded_dim: usize,
}
