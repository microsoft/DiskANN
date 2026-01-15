/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod reborrow;
pub use reborrow::{Reborrow, ReborrowMut};

pub mod lifetime;
pub use lifetime::WithLifetime;

pub mod future;

pub mod sampling;

// Views
pub mod strided;
pub mod views;

mod lazystring;
pub use lazystring::LazyString;

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
