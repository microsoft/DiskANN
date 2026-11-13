/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! It's a bitvector!

use diskann::utils::IntoUsize;

use crate::num::IdLimit;

const BUCKET_SIZE: usize = 32;

/// Like `Vec<bool>`, but with bits instead of bytes.
#[derive(Debug)]
pub(crate) struct BitVector {
    buckets: Vec<u32>,

    // Invariant:
    // * If `buckets.len() == 0` then `len == 0`.
    // * Otherwise, `32 * (buckets.len() - 1) < len <= 32 * buckets.len()`.
    id_limit: IdLimit,
}

impl BitVector {
    /// Construct a new [`BitVector`] with all entries set to `false`.
    pub(crate) fn new(id_limit: IdLimit) -> Self {
        let num_buckets = id_limit.value().into_usize().div_ceil(BUCKET_SIZE);
        Self {
            buckets: vec![0; num_buckets],
            id_limit,
        }
    }

    /// Set entry `i` to `true`, returning whether the entry changed.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.id_limit().value()`.
    #[expect(clippy::panic, reason = "out-of-bounds indexing panics")]
    pub(crate) fn set_true(&mut self, i: u32) -> bool {
        if i < self.id_limit().value() {
            let (index, bit) = Self::split(i);

            // SAFETY: `i` is in-bounds.
            let bucket = unsafe { self.buckets.get_unchecked_mut(index) };
            let previous = *bucket;

            *bucket |= bit;
            (previous & bit) == 0
        } else {
            panic!("index {i} is out-of-bounds");
        }
    }

    /// Set entry `i` to `false`, returning whether the entry changed.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.id_limit().value()`.
    #[cfg(test)]
    fn set_false(&mut self, i: u32) -> bool {
        if i < self.id_limit().value() {
            let (index, bit) = Self::split(i);

            // SAFETY: `i` is in-bounds.
            let bucket = unsafe { self.buckets.get_unchecked_mut(index) };
            let previous = *bucket;

            *bucket &= !bit;
            (previous & bit) != 0
        } else {
            panic!("index {i} is out-of-bounds");
        }
    }

    /// Get the value of entry `i` if `i` is in bounds.
    ///
    /// Returns `false` if `i >= self.len()`.
    pub(crate) fn get(&self, i: u32) -> bool {
        if i < self.id_limit().value() {
            let (index, bit) = Self::split(i);

            // SAFETY: `i` is in-bounds.
            (*unsafe { self.buckets.get_unchecked(index) } & bit) != 0
        } else {
            false
        }
    }

    /// Return the [`IdLimit`] of the [`BitVector`].
    pub(crate) fn id_limit(&self) -> IdLimit {
        self.id_limit
    }

    fn split(i: u32) -> (usize, u32) {
        (i.into_usize() / 32, 1 << (i % 32))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the vector starts with all entries set to `false` and that
    /// [`BitVector::get`] returns `false` for out-of-bounds accesses.
    #[test]
    fn test_constructor_and_oob_indexing() {
        for id_limit in [0, 1, 31, 32, 33, 63, 64, 65].map(IdLimit::new) {
            let bits = BitVector::new(id_limit);

            assert_eq!(bits.id_limit(), id_limit);
            for i in 0..id_limit.value() {
                assert!(!bits.get(i), "id_limit = {id_limit}, index = {i}");
            }
            assert!(!bits.get(id_limit.value()));
            assert!(!bits.get(u32::MAX));
        }
    }

    #[test]
    fn test_setters() {
        let mut bits = BitVector::new(IdLimit::new(65));
        let mut expected = [false; 65];

        for i in [0, 31, 32, 63, 64] {
            assert!(bits.set_true(i));
            assert!(!bits.set_true(i));
            expected[i as usize] = true;

            for (j, value) in expected.iter().enumerate() {
                assert_eq!(bits.get(j as u32), *value, "set {i}, read {j}");
            }
        }

        for i in [32, 0, 64, 31, 63] {
            assert!(bits.set_false(i));
            assert!(!bits.set_false(i));
            expected[i as usize] = false;

            for (j, value) in expected.iter().enumerate() {
                assert_eq!(bits.get(j as u32), *value, "clear {i}, read {j}");
            }
        }

        for i in 0..bits.id_limit().value() {
            assert!(bits.set_true(i));
            assert!(!bits.set_true(i));
            assert!(bits.get(i));
        }

        for i in 0..bits.id_limit().value() {
            assert!(bits.get(i));
        }

        for i in 0..bits.id_limit().value() {
            assert!(bits.set_false(i));
            assert!(!bits.set_false(i));
            assert!(!bits.get(i));
        }

        for i in 0..bits.id_limit().value() {
            assert!(!bits.get(i));
        }
    }

    #[test]
    #[should_panic(expected = "index 33 is out-of-bounds")]
    fn set_true_panics_past_length() {
        BitVector::new(IdLimit::new(33)).set_true(33);
    }

    #[test]
    #[should_panic(expected = "index 33 is out-of-bounds")]
    fn set_false_panics_past_length() {
        BitVector::new(IdLimit::new(33)).set_false(33);
    }
}
