/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Making these `pub(crate)` means that rustdoc won't declare the exported structs as originating
// from the `generated` module.
pub(crate) mod generated;

pub mod scalar_quantization;
pub use scalar_quantization::{ProtoConversionError, ScalarQuantizer};

mod storage;
pub use storage::{ProtoStorageError, load, save};
