// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! MaxSim and Chamfer distance types for multi-vector representations.

use thiserror::Error;

/// Error type for [`MaxSim`] operations.
#[derive(Clone, Debug, Copy, Error)]
pub enum MaxSimError {
    #[error("Trying to access score in index {0} for output of size {1}")]
    IndexOutOfBounds(usize, usize),
    #[error("Scores buffer length cannot be 0")]
    BufferLengthIsZero,
    #[error("Invalid buffer length {0} for query size {0}")]
    InvalidBufferLength(usize, usize),
}

////////////
// MaxSim //
////////////

/// Computes per-query-vector maximum similarities to document vectors.
///
/// For each query vector `qᵢ`, finds the maximum similarity (minimum negated
/// inner product) to any document vector:
///
/// ```text
/// scores[i] = minⱼ -IP(qᵢ, dⱼ)
/// ```
///
/// Implements `DistanceFnMut` for various matrix types
/// (e.g., [`MatRef<Standard<f32>>`](crate::multi_vector::MatRef)).
///
/// # Usage
/// - Create with [`MaxSim::new`], providing a mutable scores buffer.
/// - Call `DistanceFnMut::evaluate` with query and document matrices.
/// - Read results from the scores buffer.
#[derive(Debug)]
pub struct MaxSim<'a> {
    pub(crate) scores: &'a mut [f32],
}

impl<'a> MaxSim<'a> {
    /// Creates a new [`MaxSim`] with the provided scores buffer.
    ///
    /// # Errors
    /// Returns an error if `scores` is empty.
    pub fn new(scores: &'a mut [f32]) -> Result<Self, MaxSimError> {
        if scores.is_empty() {
            return Err(MaxSimError::BufferLengthIsZero);
        }
        Ok(Self { scores })
    }

    /// Returns the number of score slots in the buffer.
    #[inline]
    pub fn size(&self) -> usize {
        self.scores.len()
    }

    /// Returns the score at index `i`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> Result<f32, MaxSimError> {
        self.scores
            .get(i)
            .copied()
            .ok_or_else(|| MaxSimError::IndexOutOfBounds(i, self.size()))
    }

    /// Sets the score at index `i`.
    #[inline(always)]
    pub fn set(&mut self, i: usize, x: f32) -> Result<(), MaxSimError> {
        let size = self.size();

        let s = self
            .scores
            .get_mut(i)
            .ok_or(MaxSimError::IndexOutOfBounds(i, size))?;

        *s = x;
        Ok(())
    }

    /// Returns a mutable reference to the internal buffer of scores.
    ///
    /// This is useful for implementations external to crate as well as
    /// optimized implementations to access the buffer if needed.
    pub fn scores_mut(&mut self) -> &mut [f32] {
        self.scores
    }
}

/////////////
// Chamfer //
/////////////

/// Asymmetric Chamfer distance for multi-vector similarity.
///
/// Computes the sum of per-query-vector maximum similarities:
///
/// ```text
/// Chamfer(Q, D) = Σᵢ minⱼ -IP(qᵢ, dⱼ)
/// ```
///
/// Implements [`PureDistanceFunction`](diskann_vector::PureDistanceFunction)
/// for matrix view types.
#[derive(Debug, Clone, Copy)]
pub struct Chamfer;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test fixture providing common buffer sizes for testing
    struct TestFixture {
        buffer: Vec<f32>,
    }

    impl TestFixture {
        fn new(size: usize) -> Self {
            Self {
                buffer: vec![0.0; size],
            }
        }

        fn with_values(values: &[f32]) -> Self {
            Self {
                buffer: values.to_vec(),
            }
        }

        fn max_sim(&mut self) -> Result<MaxSim<'_>, MaxSimError> {
            MaxSim::new(&mut self.buffer)
        }
    }

    mod max_sim_new {
        use super::*;

        #[test]
        fn fails_with_empty_buffer() {
            let mut buffer: Vec<f32> = vec![];
            let result = MaxSim::new(&mut buffer);
            assert!(matches!(result, Err(MaxSimError::BufferLengthIsZero)));
        }

        #[test]
        fn returns_correct_size() {
            let sizes = [1, 2, 5, 100, 1000];
            for size in sizes {
                let mut fixture = TestFixture::new(size);
                let mut max_sim = fixture.max_sim().unwrap();
                assert_eq!(max_sim.size(), size, "size mismatch for buffer of {}", size);

                let scores = max_sim.scores_mut();
                assert_eq!(scores.len(), max_sim.size());
            }
        }
    }

    mod max_sim_get {
        use super::*;

        #[test]
        fn returns_value_at_valid_index() {
            let mut fixture = TestFixture::with_values(&[1.0, 2.0, 3.0]);
            let max_sim = fixture.max_sim().unwrap();

            assert_eq!(max_sim.get(0).unwrap(), 1.0);
            assert_eq!(max_sim.get(1).unwrap(), 2.0);
            assert_eq!(max_sim.get(2).unwrap(), 3.0);
        }

        #[test]
        fn fails_at_out_of_bounds_index() {
            let mut fixture = TestFixture::new(3);
            let max_sim = fixture.max_sim().unwrap();

            let result = max_sim.get(3);
            assert!(matches!(result, Err(MaxSimError::IndexOutOfBounds(3, 3))));

            let result = max_sim.get(100);
            assert!(matches!(result, Err(MaxSimError::IndexOutOfBounds(100, 3))));
        }
    }

    mod max_sim_set {
        use super::*;

        #[test]
        fn sets_value_at_valid_index() {
            let mut fixture = TestFixture::new(3);
            let mut max_sim = fixture.max_sim().unwrap();

            max_sim.set(0, 10.0).unwrap();
            max_sim.set(1, 20.0).unwrap();
            max_sim.set(2, 30.0).unwrap();

            assert_eq!(max_sim.get(0).unwrap(), 10.0);
            assert_eq!(max_sim.get(1).unwrap(), 20.0);
            assert_eq!(max_sim.get(2).unwrap(), 30.0);
        }

        #[test]
        fn fails_at_out_of_bounds_index() {
            let mut fixture = TestFixture::new(3);
            let mut max_sim = fixture.max_sim().unwrap();

            let result = max_sim.set(3, 999.0);
            assert!(matches!(result, Err(MaxSimError::IndexOutOfBounds(3, 3))));
        }

        #[test]
        fn overwrites_existing_value() {
            let mut fixture = TestFixture::with_values(&[1.0, 2.0, 3.0]);
            let mut max_sim = fixture.max_sim().unwrap();

            max_sim.set(1, 99.0).unwrap();

            assert_eq!(max_sim.get(0).unwrap(), 1.0); // unchanged
            assert_eq!(max_sim.get(1).unwrap(), 99.0); // changed
            assert_eq!(max_sim.get(2).unwrap(), 3.0); // unchanged
        }

        #[test]
        fn handles_special_float_values() {
            let mut fixture = TestFixture::new(4);
            let mut max_sim = fixture.max_sim().unwrap();

            max_sim.set(0, f32::INFINITY).unwrap();
            max_sim.set(1, f32::NEG_INFINITY).unwrap();
            max_sim.set(2, f32::NAN).unwrap();
            max_sim.set(3, -0.0).unwrap();

            assert_eq!(max_sim.get(0).unwrap(), f32::INFINITY);
            assert_eq!(max_sim.get(1).unwrap(), f32::NEG_INFINITY);
            assert!(max_sim.get(2).unwrap().is_nan());
            assert!(max_sim.get(3).unwrap().is_sign_negative());
        }

        #[test]
        fn writes_through_to_underlying_buffer() {
            let mut buffer = vec![0.0f32; 3];
            {
                let mut max_sim = MaxSim::new(&mut buffer).unwrap();
                max_sim.set(0, 1.0).unwrap();
                max_sim.set(1, 2.0).unwrap();
            }
            // After MaxSim is dropped, buffer reflects the changes
            assert_eq!(buffer, vec![1.0, 2.0, 0.0]);
        }
    }
}
