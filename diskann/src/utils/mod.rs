/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod object_pool;

pub mod async_tools;
pub use async_tools::VectorIdBoxSlice;

#[allow(clippy::module_inception)]
pub mod utils;
pub use utils::*;

pub mod vector_repr;
pub use vector_repr::VectorRepr;

mod vector_id;
pub use vector_id::VectorId;
