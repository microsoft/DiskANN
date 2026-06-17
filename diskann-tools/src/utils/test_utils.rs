/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(not(miri))]
pub mod size_constants {
    /// The small dataset size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_DATASET_SIZE_SMALL: u64 = 101;

    /// The recommended number of dimensions for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    /// When "memory aligned" the dimensions become 64 (8*8). Setting to non-aligned value to ensure aligning works.
    pub const TEST_NUM_DIMENSIONS_RECOMMENDED: usize = 59;
}

#[cfg(miri)]
pub mod size_constants {
    /// The small dataset size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_DATASET_SIZE_SMALL: u64 = 3;

    /// The recommended number of dimensions for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    /// When "memory aligned" the dimensions become 16 (8*2). Setting to non-aligned value to ensure aligning works.
    pub const TEST_NUM_DIMENSIONS_RECOMMENDED: usize = 13;
}
