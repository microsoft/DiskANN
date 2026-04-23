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

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    #[test]
    fn new_creates_correct_range() {
        // valid_points of ten with five frozen points gives range 10..15
        let sp = StartPoints::new(10, NonZeroUsize::new(5).unwrap())
            .expect("should construct without overflow");
        let r = sp.range().collect::<Vec<_>>();
        assert_eq!(r, vec![10, 11, 12, 13, 14]);
        assert_eq!(sp.end(), 15);
    }

    #[test]
    fn new_returns_error_on_overflow() {
        // valid_points at u32::MAX plus one frozen point must overflow
        let max = u32::MAX;
        let res = StartPoints::new(max, NonZeroUsize::new(1).unwrap());
        assert!(res.is_err(), "expected an error when sum exceeds u32::MAX");
        if let Err(err) = res {
            let msg = err.to_string();
            assert!(
                msg.contains("Sum of valid points and frozen points exceeds u32::MAX"),
                "unexpected error message: {}",
                msg
            );
        }
    }

    #[test]
    fn len_and_is_empty() {
        let sp = StartPoints::new(0, NonZeroUsize::new(3).unwrap()).unwrap();
        assert_eq!(sp.len(), 3);
        assert!(!sp.is_empty());
        assert_eq!(sp.start(), 0);
        assert_eq!(sp.end(), 3);
    }

    #[test]
    fn single_frozen_point() {
        let sp = StartPoints::new(100, NonZeroUsize::new(1).unwrap()).unwrap();
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.range().collect::<Vec<_>>(), vec![100]);
    }
}
