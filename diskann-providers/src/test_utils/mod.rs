/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod search_utils;
#[cfg(test)]
pub use search_utils::is_match;
pub use search_utils::{assert_top_k_exactly_match, groundtruth};
