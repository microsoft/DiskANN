/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::{NonZeroU32, NonZeroUsize};

use super::to_nonzero_usize;

/// An experimental addition to index construction that will retry the search and prune
/// phase if an insufficient number of candidates are discovered.
///
/// The algorithm works as follows:
///
/// 1. After search and prune, inspect the number candidates for the inserted point's
///    adjacency list. If it is below `retry_if_candidates_shorter_than`, double the search
///    window size and try again.
///
/// 2. Repeat this process for `max_retries` attempts, growing the search list size again
///    on each attempt.
///
/// 3. On the last attempt, if `saturate_on_last_attempt` is enabled, then saturate the
///    candidate adjacency list after pruning to increase its degree.
#[derive(Debug, Clone, PartialEq)]
pub struct InsertRetry {
    /// The maximum number of attempts.
    max_retries: NonZeroU32,

    /// Retry if the post-pruned adjacency list is below this threshold.
    ///
    /// This is relative to the `max_degree` of the parent configuration.
    retry_if_candidates_shorter_than: NonZeroU32,

    /// Force graph saturation on the last attempt.
    saturate_on_last_attempt: bool,
}

impl InsertRetry {
    pub fn new(
        max_retries: NonZeroU32,
        retry_if_candidates_shorter_than: NonZeroU32,
        saturate_on_last_attempt: bool,
    ) -> Self {
        Self {
            max_retries,
            retry_if_candidates_shorter_than,
            saturate_on_last_attempt,
        }
    }

    pub fn max_retries(&self) -> NonZeroUsize {
        to_nonzero_usize!(self.max_retries)
    }

    pub fn retry_if_candidates_shorter_than(&self) -> NonZeroUsize {
        to_nonzero_usize!(self.retry_if_candidates_shorter_than)
    }

    pub fn saturate_on_last_attempt(&self) -> bool {
        self.saturate_on_last_attempt
    }

    pub fn should_saturate(&self, attempt: usize) -> bool {
        // The subtraction will not underflow because `max_retries` is non-zero.
        self.saturate_on_last_attempt() && (attempt == (self.max_retries().get() - 1))
    }

    pub fn should_retry(&self, attempt: usize, num_candidates: usize) -> bool {
        // The subtraction will not underflow because `max_retries` is non-zero.
        attempt != (self.max_retries().get() - 1)
            && num_candidates < self.retry_if_candidates_shorter_than().get()
    }
}
