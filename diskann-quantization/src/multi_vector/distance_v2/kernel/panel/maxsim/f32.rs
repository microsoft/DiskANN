/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::multi_vector::distance_v2::{blocks, kernel};

#[derive(Debug)]
struct BlockWithRowMajor<'a, Arch> {
    kernel: kernel::micro::f32::MaxSim<Arch>,
    a:
}

