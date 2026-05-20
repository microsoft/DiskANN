/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Compile-time-sized top-K tracker for fused SIMD distance + selection loops.
//!
//! The tracker is a `[(u32, f32); K]` sorted ascending by distance. It is
//! designed to live in registers across a hot loop: callers pre-fill the array
//! with `(u32::MAX, f32::MAX)` and call [`topk_insert`] for each candidate
//! that beats the current worst entry. Suited for K ≤ 16 (limit of the
//! current `MAX_FANOUT` use in PiPNN partition).

/// Insert `(idx, dist)` into a sorted top-K tracker if it beats the current
/// worst entry. `top[..K]` is kept sorted ascending by distance;
/// `threshold_idx = K - 1` is the worst slot.
///
/// Caller is expected to initialize `top` with sentinel values so the first
/// `K` candidates are always accepted:
/// ```ignore
/// let mut top: [(u32, f32); K] = [(u32::MAX, f32::MAX); K];
/// for (idx, dist) in candidates {
///     topk_insert(&mut top, K - 1, idx, dist);
/// }
/// ```
///
/// The bubble-up is O(K) worst-case (16 swaps for K=16), but typically
/// 0–1 swap once the tracker has filled, since incoming `dist < top[K-1].1`
/// means only the worst slot is displaced.
#[inline(always)]
pub fn topk_insert<const K: usize>(
    top: &mut [(u32, f32); K],
    threshold_idx: usize,
    idx: u32,
    dist: f32,
) {
    if dist >= top[threshold_idx].1 {
        return;
    }
    top[threshold_idx] = (idx, dist);
    let mut t = threshold_idx;
    while t > 0 && top[t].1 < top[t - 1].1 {
        top.swap(t, t - 1);
        t -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_accepts_first_k() {
        let mut top: [(u32, f32); 3] = [(u32::MAX, f32::MAX); 3];
        topk_insert(&mut top, 2, 7, 3.0);
        topk_insert(&mut top, 2, 8, 1.0);
        topk_insert(&mut top, 2, 9, 2.0);
        assert_eq!(top, [(8, 1.0), (9, 2.0), (7, 3.0)]);
    }

    #[test]
    fn rejects_when_worse_than_worst() {
        let mut top: [(u32, f32); 3] = [(8, 1.0), (9, 2.0), (7, 3.0)];
        topk_insert(&mut top, 2, 11, 5.0); // worse than 3.0
        assert_eq!(top, [(8, 1.0), (9, 2.0), (7, 3.0)]);
    }

    #[test]
    fn displaces_worst_and_resorts() {
        let mut top: [(u32, f32); 3] = [(8, 1.0), (9, 2.0), (7, 3.0)];
        topk_insert(&mut top, 2, 11, 0.5);
        assert_eq!(top, [(11, 0.5), (8, 1.0), (9, 2.0)]);
    }

    #[test]
    fn k_one() {
        let mut top: [(u32, f32); 1] = [(u32::MAX, f32::MAX)];
        topk_insert(&mut top, 0, 5, 2.0);
        topk_insert(&mut top, 0, 6, 1.0);
        topk_insert(&mut top, 0, 7, 3.0); // rejected
        assert_eq!(top, [(6, 1.0)]);
    }
}
