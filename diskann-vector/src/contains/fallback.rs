/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) fn contains_simd_u32(vector: &[u32], target: u32) -> bool {
    vector.contains(&target)
}

pub(crate) fn contains_simd_u64(vector: &[u64], target: u64) -> bool {
    vector.contains(&target)
}
