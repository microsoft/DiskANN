/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult, utils::IntoUsize};
use std::num::NonZeroUsize;

/// Represents a range of start points for an index.
/// The range includes `start` and excludes `end`.
/// `start` is the first valid point, and `end - 1` is the last valid point.
pub struct StartPoints {
    start: u32,
    end: u32,
}

impl StartPoints {
    pub fn new(valid_points: u32, frozen_points: NonZeroUsize) -> ANNResult<Self> {
        Ok(Self {
            start: valid_points,
            end: match valid_points.checked_add(frozen_points.get() as u32) {
                Some(end) => end,
                None => {
                    return Err(ANNError::log_index_error(
                        "Sum of valid points and frozen points exceeds u32::MAX",
                    ));
                }
            },
        })
    }

    pub fn range(&self) -> std::ops::Range<u32> {
        self.start..self.end
    }

    pub fn len(&self) -> usize {
        (self.end - self.start).into_usize()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn start(&self) -> u32 {
        self.start
    }

    pub fn end(&self) -> u32 {
        self.end
    }
}
