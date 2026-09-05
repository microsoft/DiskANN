/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared utilities for DiskANN crates.

#[cfg(not(target_endian = "little"))]
compile_error!("diskann-utils assumes little-endian targets");

pub mod reborrow;
pub use reborrow::{Reborrow, ReborrowMut};

pub mod future;

pub mod io;
pub mod object_pool;
pub mod sampling;

// Views
pub mod strided;
pub mod views;

mod lazystring;
pub use lazystring::LazyString;

mod internal;

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn test_data_directory() -> &'static str {
    "test_data"
}

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn test_data_root() -> std::path::PathBuf {
    workspace_root().join(test_data_directory())
}

// Test function

pub fn test_function(x: views::rowmajor::Ref<'_, u32>) -> u32 {
    use views::rowmajor::Matrix;

    let mut sum = 0;
    for w in x.window_iter(10) {
        for r in w.row_iter() {
            for i in r.iter() {
                sum += *i;
            }
        }
    }
    sum
}
