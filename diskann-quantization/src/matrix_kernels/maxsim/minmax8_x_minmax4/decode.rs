/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Four-bit codes only: no canonical headers, compensation, or kernel packing parameters.

use diskann_wide::{Architecture, arch::Scalar};

use super::layout::EvenOdd;

pub(super) trait Decoder: Architecture {
    /// Write all 32 low nibbles followed by all 32 high nibbles, in input byte order.
    /// The capability token enables the required ISA.
    fn decode_block(self, packed: &[u8; 32], output: &mut [u8; 64]);

    /// Decode up to 32 packed bytes, treating bytes beyond the input as zero.
    #[inline]
    fn decode_tail(self, packed: &[u8], output: &mut [u8; 64]) {
        let mut tail = [0; 32];
        tail[..packed.len()].copy_from_slice(packed);
        self.decode_block(&tail, output);
    }

    /// All output bytes are overwritten, including dimension padding.
    #[expect(
        clippy::expect_used,
        reason = "chunk lengths match the fixed layout block sizes"
    )]
    #[inline]
    fn decode(self, packed: &[u8], dim: usize, output: &mut [u8]) {
        assert_eq!(packed.len(), dim.div_ceil(2), "packed code length mismatch");
        assert_eq!(
            output.len(),
            EvenOdd::padded(dim),
            "decoded row length mismatch"
        );
        let full = dim / EvenOdd::BLOCK;
        for (source, dest) in packed[..full * EvenOdd::PACKED_BYTES]
            .as_chunks::<32>()
            .0
            .iter()
            .zip(output[..full * EvenOdd::BLOCK].as_chunks_mut::<64>().0)
        {
            self.decode_block(source, dest);
        }
        if !dim.is_multiple_of(EvenOdd::BLOCK) {
            let source = &packed[full * EvenOdd::PACKED_BYTES..];
            let output: &mut [u8; 64] = (&mut output[full * EvenOdd::BLOCK..])
                .try_into()
                .expect("one tail block");
            self.decode_tail(source, output);
            if !dim.is_multiple_of(2) {
                // The final high nibble is outside D even when its byte is in bounds.
                output[EvenOdd::PACKED_BYTES + source.len() - 1] = 0;
            }
        }
    }
}

impl Decoder for Scalar {
    #[inline(always)]
    fn decode_block(self, packed: &[u8; 32], output: &mut [u8; 64]) {
        for (i, &value) in packed.iter().enumerate() {
            output[i] = value & 0x0F;
            output[EvenOdd::PACKED_BYTES + i] = value >> 4;
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
        fn decode_block(self, packed: &[u8; 32], output: &mut [u8; 64]) {
            self.run_inline(
                #[inline]
                || {
                    diskann_wide::alias!(u32s = <V3>::u32x8);
                    diskann_wide::alias!(u8s = <V3>::u8x32);
                    // AVX2 byte shifts are emulated; shift native u32 lanes then mask.
                    let mask = u32s::splat(self, 0x0f0f_0f0f);
                    // SAFETY: The input contains 32 bytes; load_simd is unaligned.
                    let codes = unsafe { u32s::load_simd(self, packed.as_ptr().cast()) };
                    let low = u8s::from_underlying(self, (codes & mask).to_underlying());
                    let high = u8s::from_underlying(self, ((codes >> 4) & mask).to_underlying());
                    // SAFETY: Each output half contains exactly 32 writable bytes.
                    unsafe {
                        low.store_simd(output.as_mut_ptr());
                        high.store_simd(output.as_mut_ptr().add(32));
                    }
                },
            );
        }
    }

    impl Decoder for V4 {
        #[inline(always)]
        fn decode_block(self, packed: &[u8; 32], output: &mut [u8; 64]) {
            self.retarget().decode_block(packed, output);
        }

        #[inline(always)]
        fn decode_tail(self, packed: &[u8], output: &mut [u8; 64]) {
            assert!(packed.len() <= 32, "packed tail exceeds one block");
            self.run_inline(
                #[inline]
                || {
                    diskann_wide::alias!(u8s = <V4>::u8x32);
                    // SAFETY: Only packed.len() bytes are accessed; masked-off lanes
                    // are zeroed, including when the input ends at an allocation boundary.
                    let codes =
                        unsafe { u8s::load_simd_first(self, packed.as_ptr(), packed.len()) };
                    self.retarget().decode_block(&codes.to_array(), output);
                },
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl Decoder for diskann_wide::arch::aarch64::Neon {
    #[inline(always)]
    fn decode_block(self, packed: &[u8; 32], output: &mut [u8; 64]) {
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
                        (codes & mask).store_simd(output.as_mut_ptr().add(offset));
                        ((codes >> 4) & mask).store_simd(output.as_mut_ptr().add(32 + offset));
                    }
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<A: Decoder>(arch: A) {
        for dim in (1..64).chain(super::super::layout::tests::DIMS.iter().copied()) {
            for pattern in [Some(0), Some(255), Some(0x5a), Some(0xa5), Some(0x73), None] {
                // An offset of one exercises unaligned input; the final nibble remains
                // nonzero for odd dimensions. Sentinels guard both output boundaries.
                let bytes: Vec<_> = (0..dim.div_ceil(2) + 1)
                    .map(|i| pattern.unwrap_or_else(|| (i.wrapping_mul(73) ^ (i / 32)) as u8))
                    .collect();
                let codes = &bytes[1..];
                let k = dim.next_multiple_of(EvenOdd::BLOCK);
                let mut output = vec![0xfe; k + 2];
                arch.decode(codes, dim, &mut output[1..k + 1]);
                let mut expected = vec![0; k];
                for d in 0..dim {
                    expected[EvenOdd::position(d)] = (codes[d / 2] >> (4 * (d % 2))) & 15;
                }
                assert_eq!(&output[1..k + 1], expected, "dim={dim}");
                assert_eq!(output[0], 0xfe);
                assert_eq!(output[k + 1], 0xfe);
                let mut scalar = vec![0xfd; k];
                Scalar::new().decode(codes, dim, &mut scalar);
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
        Scalar::new().decode(&[0; 6], 9, &mut [0; 64]);
    }

    #[test]
    #[should_panic(expected = "decoded row length mismatch")]
    fn rejects_inexact_output() {
        Scalar::new().decode(&[0; 5], 9, &mut [0; 63]);
    }
}
