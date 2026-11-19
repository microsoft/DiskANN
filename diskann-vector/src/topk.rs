/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Candidate storage and scalar sorted insertion for TopK results.
//!
//! Nearest sets are sorted by nondecreasing distance, with [`Candidate::EMPTY`]
//! in unused slots. Architecture-specific batching, SIMD widths, and mask
//! traversal belong to the kernels that consume this representation.

/// The ID of an output slot that holds no candidate.
pub const UNASSIGNED: u32 = u32::MAX;

/// One slot of a nearest set: an input-local candidate ID and its distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub local_idx: u32,
    pub distance: f32,
}

impl Candidate {
    /// An unfilled slot. Every distance that can enter a nearest set is nearer.
    pub const EMPTY: Self = Self::new(UNASSIGNED, f32::INFINITY);

    pub const fn new(local_idx: u32, distance: f32) -> Self {
        Self {
            local_idx,
            distance,
        }
    }

    /// Return `true` if this slot holds a candidate.
    pub const fn is_assigned(self) -> bool {
        self.local_idx != UNASSIGNED
    }
}

impl Default for Candidate {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Insert a candidate into a nearest set and return the new k-th distance.
///
/// The set must be nonempty, sorted by nondecreasing distance, and the candidate
/// must be strictly nearer than its last slot. Its ID must not be [`UNASSIGNED`].
/// A candidate goes after existing candidates with an equal distance.
/// Arrays and slices both retain their concrete type at the call site.
#[inline(always)]
pub fn insert_sorted<Nearest>(nearest: &mut Nearest, candidate: Candidate) -> f32
where
    Nearest: AsMut<[Candidate]> + ?Sized,
{
    let nearest = nearest.as_mut();
    let last = nearest.len() - 1;
    let mut slot = last;
    while slot > 0 && candidate.distance < nearest[slot - 1].distance {
        nearest[slot] = nearest[slot - 1];
        slot -= 1;
    }
    nearest[slot] = candidate;
    nearest[last].distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_supports_arrays_slices_and_stable_ties() {
        let mut nearest = [Candidate::EMPTY; 3];
        insert_sorted(&mut nearest, Candidate::new(5, -0.0));
        insert_sorted(nearest.as_mut_slice(), Candidate::new(6, 0.0));
        insert_sorted(&mut nearest, Candidate::new(7, -1.0));
        assert_eq!(
            nearest,
            [
                Candidate::new(7, -1.0),
                Candidate::new(5, -0.0),
                Candidate::new(6, 0.0),
            ]
        );
        assert_eq!(nearest[1].distance.to_bits(), (-0.0f32).to_bits());
        assert_eq!(nearest[2].distance.to_bits(), 0.0f32.to_bits());
    }
}
