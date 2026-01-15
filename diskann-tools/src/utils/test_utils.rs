/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(not(miri))]
pub mod size_constants {
    /// The recommended dataset size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_DATASET_SIZE_RECOMMENDED: u64 = 991;

    /// The small dataset size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_DATASET_SIZE_SMALL: u64 = 101;

    /// The recommended query size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_QUERYSET_SIZE_RECOMMENDED: u64 = 101;

    /// The small query size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_QUERYSET_SIZE_SMALL: u64 = 11;

    /// The recommended number of dimensions for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    /// When "memory aligned" the dimensions become 64 (8*8). Setting to non-aligned value to ensure aligning works.
    pub const TEST_NUM_DIMENSIONS_RECOMMENDED: usize = 59;

    /// The recommended "memory aligned" number of dimensions for testing the library (64=8*8).
    pub const TEST_NUM_DIMENSIONS_RECOMMENDED_MEMORY_ALIGNED: usize = 64;

    /// The small number of dimensions for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_NUM_DIMENSIONS_SMALL: usize = 13;

    /// The small "memory aligned" number of dimensions for testing the library.
    pub const TEST_NUM_DIMENSIONS_SMALL_MEMORY_ALIGNED: usize = 16;
}

#[cfg(miri)]
pub mod size_constants {
    /// The recommended dataset size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_DATASET_SIZE_RECOMMENDED: u64 = 7;

    /// The small dataset size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_DATASET_SIZE_SMALL: u64 = 3;

    /// The recommended query size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_QUERYSET_SIZE_RECOMMENDED: u64 = 3;

    /// The small query size for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_QUERYSET_SIZE_SMALL: u64 = 1;

    /// The recommended number of dimensions for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    /// When "memory aligned" the dimensions become 16 (8*2). Setting to non-aligned value to ensure aligning works.
    pub const TEST_NUM_DIMENSIONS_RECOMMENDED: usize = 13;

    /// The recommended "memory aligned" number of dimensions for testing the library (16=8*2).
    pub const TEST_NUM_DIMENSIONS_RECOMMENDED_MEMORY_ALIGNED: usize = 16;

    /// The small number of dimensions for testing the library.
    /// A prime number is used to avoid any accidental patterns in the data.
    pub const TEST_NUM_DIMENSIONS_SMALL: usize = 7;

    /// The small "memory aligned" number of dimensions for testing the library.
    pub const TEST_NUM_DIMENSIONS_SMALL_MEMORY_ALIGNED: usize = 8;
}
