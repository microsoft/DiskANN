/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Four-bit codes only: no canonical headers, compensation, or kernel packing parameters.

use diskann_wide::{Architecture, arch::Scalar};

pub(super) const BLOCK_BYTES: usize = 32;

pub(super) trait Decoder: Architecture {
    /// Expand each byte into separate low/high channels, without choosing a column order.
    fn unpack_block(
        self,
        packed: &[u8; BLOCK_BYTES],
        low: &mut [u8; BLOCK_BYTES],
        high: &mut [u8; BLOCK_BYTES],
    );

    /// Expand up to one block of bytes, treating bytes beyond the input as zero.
    #[inline]
    fn unpack_tail(self, packed: &[u8], low: &mut [u8; BLOCK_BYTES], high: &mut [u8; BLOCK_BYTES]) {
        let mut tail = [0; BLOCK_BYTES];
        tail[..packed.len()].copy_from_slice(packed);
        self.unpack_block(&tail, low, high);
    }
}

impl Decoder for Scalar {
    #[inline(always)]
    fn unpack_block(
        self,
        packed: &[u8; BLOCK_BYTES],
        low: &mut [u8; BLOCK_BYTES],
        high: &mut [u8; BLOCK_BYTES],
    ) {
        for (i, &value) in packed.iter().enumerate() {
            low[i] = value & 0x0F;
            high[i] = value >> 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use diskann_wide::{
        SIMDVector,
        arch::x86_64::{V3, V4},
    };

    impl Decoder for V3 {
        #[inline(always)]
        fn unpack_block(
            self,
            packed: &[u8; BLOCK_BYTES],
            low: &mut [u8; BLOCK_BYTES],
            high: &mut [u8; BLOCK_BYTES],
        ) {
            self.run_inline(
                #[inline]
                || {
                    diskann_wide::alias!(u32s = <V3>::u32x8);
                    diskann_wide::alias!(u8s = <V3>::u8x32);
                    // AVX2 byte shifts are emulated; shift native u32 lanes then mask.
                    let mask = u32s::splat(self, 0x0f0f_0f0f);
                    // SAFETY: The input contains 32 bytes; load_simd is unaligned.
                    let codes = unsafe { u32s::load_simd(self, packed.as_ptr().cast()) };
                    let low_codes = u8s::from_underlying(self, (codes & mask).to_underlying());
                    let high_codes =
                        u8s::from_underlying(self, ((codes >> 4) & mask).to_underlying());
                    // SAFETY: Each output channel contains exactly 32 writable bytes.
                    unsafe {
                        low_codes.store_simd(low.as_mut_ptr());
                        high_codes.store_simd(high.as_mut_ptr());
                    }
                },
            );
        }
    }

    impl Decoder for V4 {
        #[inline(always)]
        fn unpack_block(
            self,
            packed: &[u8; BLOCK_BYTES],
            low: &mut [u8; BLOCK_BYTES],
            high: &mut [u8; BLOCK_BYTES],
        ) {
            self.retarget().unpack_block(packed, low, high);
        }

        #[inline(always)]
        fn unpack_tail(
            self,
            packed: &[u8],
            low: &mut [u8; BLOCK_BYTES],
            high: &mut [u8; BLOCK_BYTES],
        ) {
            assert!(packed.len() <= 32, "packed tail exceeds one block");
            self.run_inline(
                #[inline]
                || {
                    diskann_wide::alias!(u8s = <V4>::u8x32);
                    // SAFETY: Only packed.len() bytes are accessed; masked-off lanes
                    // are zeroed, including when the input ends at an allocation boundary.
                    let codes =
                        unsafe { u8s::load_simd_first(self, packed.as_ptr(), packed.len()) };
                    self.retarget().unpack_block(&codes.to_array(), low, high);
                },
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl Decoder for diskann_wide::arch::aarch64::Neon {
    #[inline(always)]
    fn unpack_block(
        self,
        packed: &[u8; BLOCK_BYTES],
        low: &mut [u8; BLOCK_BYTES],
        high: &mut [u8; BLOCK_BYTES],
    ) {
        self.run_inline(
            #[inline]
            || {
                use diskann_wide::SIMDVector;
                diskann_wide::alias!(u8s = <diskann_wide::arch::aarch64::Neon>::u8x16);
                let mask = u8s::splat(self, 15);
                for offset in [0, 16] {
                    // SAFETY: Each 16-byte load/store stays inside its fixed-size
                    // slice; load_simd and store_simd have no alignment requirements.
                    unsafe {
                        let codes = u8s::load_simd(self, packed.as_ptr().add(offset));
                        (codes & mask).store_simd(low.as_mut_ptr().add(offset));
                        ((codes >> 4) & mask).store_simd(high.as_mut_ptr().add(offset));
                    }
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::layout::{BLOCK, Layout, tests::position};
    use super::*;

    fn check<A: Decoder>(arch: A) {
        let packed = core::array::from_fn(|i| (i * 73 + 19) as u8);
        let mut low = [0xff; BLOCK_BYTES];
        let mut high = [0xff; BLOCK_BYTES];
        arch.unpack_block(&packed, &mut low, &mut high);
        assert_eq!(low, packed.map(|v| v & 15));
        assert_eq!(high, packed.map(|v| v >> 4));
        for dim in (1..64).chain(super::super::layout::tests::DIMS.iter().copied()) {
            for pattern in [Some(0), Some(255), Some(0x5a), Some(0xa5), Some(0x73), None] {
                // An offset of one exercises unaligned input; the final nibble remains
                // nonzero for odd dimensions. Sentinels guard both output boundaries.
                let bytes: Vec<_> = (0..dim.div_ceil(2) + 1)
                    .map(|i| pattern.unwrap_or_else(|| (i.wrapping_mul(73) ^ (i / 32)) as u8))
                    .collect();
                let codes = &bytes[1..];
                let layout = Layout::new::<8, 4, 6>(dim).unwrap();
                let k = dim.next_multiple_of(BLOCK);
                let mut output = vec![0xfe; k + 2];
                layout.decode_row(arch, codes, &mut output[1..k + 1]);
                let mut expected = vec![0; k];
                for d in 0..dim {
                    expected[position(d)] = (codes[d / 2] >> (4 * (d % 2))) & 15;
                }
                assert_eq!(&output[1..k + 1], expected, "dim={dim}");
                assert_eq!(output[0], 0xfe);
                assert_eq!(output[k + 1], 0xfe);
                let mut scalar = vec![0xfd; k];
                layout.decode_row(Scalar::new(), codes, &mut scalar);
                assert_eq!(scalar, expected);
            }
        }
    }

    #[test]
    fn scalar() {
        check(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_decoders() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            check(arch);
        }
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
            check(arch);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check(arch);
        }
    }

    #[test]
    #[should_panic(expected = "packed code length mismatch")]
    fn rejects_inexact_input() {
        Layout::new::<8, 4, 6>(9)
            .unwrap()
            .decode_row(Scalar::new(), &[0; 6], &mut [0; 64]);
    }

    #[test]
    #[should_panic(expected = "decoded row length mismatch")]
    fn rejects_inexact_output() {
        Layout::new::<8, 4, 6>(9)
            .unwrap()
            .decode_row(Scalar::new(), &[0; 5], &mut [0; 63]);
    }
}
