/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Lossy `f32` ↔ `u16` (bf16) conversion for compact distance storage.
//!
//! `bf16` is the upper 16 bits of an IEEE-754 `f32`: same exponent, mantissa
//! truncated from 23 to 7 bits. For non-negative values, `bf16` bit ordering
//! matches `f32` ordering, so a packed `bf16` can be compared as `u16` to give
//! the correct distance order — useful for keeping a top-k tracker as compact
//! 8-byte entries.

/// Convert `f32` → bf16 by truncating the lower 16 mantissa bits.
#[inline(always)]
pub fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Reconstruct `f32` from a bf16, zero-filling the lower mantissa bits.
#[inline(always)]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bf16_exact_values() {
        // bf16 has 7 mantissa bits. Pick f32 values whose lower 16 mantissa
        // bits are zero so the truncation is lossless.
        for &x in &[0.0_f32, 1.0, 2.0, 0.5, 0.25, 4.0, -1.0, -0.5] {
            let back = bf16_to_f32(f32_to_bf16(x));
            assert_eq!(back, x, "exact bf16 roundtrip failed for {}", x);
        }
    }

    #[test]
    fn truncation_is_within_relative_tolerance() {
        // Values that aren't bf16-exact still round to within ~2^-7 relative.
        use std::f32::consts::{E, PI};
        for &x in &[1e-10_f32, 1e10, PI, E] {
            let back = bf16_to_f32(f32_to_bf16(x));
            let rel = ((back - x) / x).abs();
            assert!(rel <= 0.01, "{} → {}: rel error {} > 1%", x, back, rel);
        }
    }

    #[test]
    fn ordering_preserved_for_non_negative() {
        // For non-negative f32, bf16 (as u16) preserves ordering.
        let xs: [f32; 6] = [0.0, 1e-10, 0.1, 1.0, 10.0, 1e10];
        let bs: Vec<u16> = xs.iter().map(|&x| f32_to_bf16(x)).collect();
        for w in bs.windows(2) {
            assert!(w[0] <= w[1], "bf16 ordering broken: {} > {}", w[0], w[1]);
        }
    }
}
