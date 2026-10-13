/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Nibble expansion, independent of query packing and metadata.

pub(super) const BYTES: usize = 32;

use diskann_wide::{Architecture, arch::Scalar};

pub(super) trait Decoder: Architecture {
    fn block(self, packed: &[u8; BYTES], low: &mut [u8; BYTES], high: &mut [u8; BYTES]);

    #[expect(
        clippy::expect_used,
        reason = "a full decoder block contains exactly BYTES bytes"
    )]
    #[inline(always)]
    fn unpack(self, packed: &[u8], low: &mut [u8; BYTES], high: &mut [u8; BYTES]) {
        assert!(packed.len() <= BYTES, "packed block exceeds decoder width");
        if packed.len() == BYTES {
            self.block(packed.try_into().expect("full decoder block"), low, high);
        } else {
            let mut tail = [0; BYTES];
            tail[..packed.len()].copy_from_slice(packed);
            self.block(&tail, low, high);
        }
    }
}

impl Decoder for Scalar {
    #[inline(always)]
    fn block(self, packed: &[u8; BYTES], low: &mut [u8; BYTES], high: &mut [u8; BYTES]) {
        for (i, &byte) in packed.iter().enumerate() {
            low[i] = byte & 15;
            high[i] = byte >> 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use diskann_wide::{
        SIMDVector,
        arch::x86_64::{V3, V4},
    };

    impl Decoder for V3 {
        #[inline(always)]
        fn block(self, packed: &[u8; BYTES], low: &mut [u8; BYTES], high: &mut [u8; BYTES]) {
            diskann_wide::alias!(words = <V3>::u32x8);
            diskann_wide::alias!(bytes = <V3>::u8x32);
            // Native u32 shifts avoid AVX2's emulated per-byte shifts.
            let mask = words::splat(self, 0x0f0f_0f0f);
            // SAFETY: packed spans 32 bytes; SIMD loads do not require alignment.
            let codes = unsafe { words::load_simd(self, packed.as_ptr().cast()) };
            let lows = bytes::from_underlying(self, (codes & mask).to_underlying());
            let highs = bytes::from_underlying(self, ((codes >> 4) & mask).to_underlying());
            // SAFETY: Both output channels span 32 writable bytes.
            unsafe {
                lows.store_simd(low.as_mut_ptr());
                highs.store_simd(high.as_mut_ptr());
            }
        }
    }

    impl Decoder for V4 {
        #[inline(always)]
        fn block(self, packed: &[u8; BYTES], low: &mut [u8; BYTES], high: &mut [u8; BYTES]) {
            self.retarget().block(packed, low, high);
        }
        #[inline(always)]
        fn unpack(self, packed: &[u8], low: &mut [u8; BYTES], high: &mut [u8; BYTES]) {
            assert!(packed.len() <= BYTES, "packed block exceeds decoder width");
            diskann_wide::alias!(bytes = <V4>::u8x32);
            // SAFETY: Only the available bytes are read; the other lanes are zero.
            let codes = unsafe { bytes::load_simd_first(self, packed.as_ptr(), packed.len()) };
            self.retarget().block(&codes.to_array(), low, high);
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl Decoder for diskann_wide::arch::aarch64::Neon {
    #[inline(always)]
    fn block(self, packed: &[u8; BYTES], low: &mut [u8; BYTES], high: &mut [u8; BYTES]) {
        use diskann_wide::SIMDVector;
        diskann_wide::alias!(bytes = <diskann_wide::arch::aarch64::Neon>::u8x16);
        let mask = bytes::splat(self, 15);
        for offset in [0, 16] {
            // SAFETY: Each load and store stays within its 32-byte channel.
            unsafe {
                let codes = bytes::load_simd(self, packed.as_ptr().add(offset));
                (codes & mask).store_simd(low.as_mut_ptr().add(offset));
                ((codes >> 4) & mask).store_simd(high.as_mut_ptr().add(offset));
            }
        }
    }
}
