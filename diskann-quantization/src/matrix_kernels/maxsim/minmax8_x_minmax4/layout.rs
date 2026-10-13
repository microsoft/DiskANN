/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared order and borrowed panels for MinMax query packing and document decoding.
//! Each 64-dimensional block stores even coordinates followed by odd coordinates.
//! Missing coordinates are zero-padded before applying the block-transposed PACK layout.

use super::decode::{self, Decoder};
use crate::{
    matrix_kernels::{
        blocks::packed,
        num::{DimK, value_or_one},
        ptr::Slice,
    },
    minmax::MinMaxCompensation,
    multi_vector::{BlockTransposedRef, Overflow},
};

const BLOCK: usize = 2 * decode::BYTES;
pub(crate) const MAX_DIM: usize = u32::MAX as usize / (255 * 15);

#[derive(Debug)]
pub(crate) struct Layout {
    dim: usize,
    padded: usize,
}

impl Layout {
    pub(crate) fn new(dim: usize) -> Result<Self, Overflow> {
        let padded = dim
            .checked_next_multiple_of(BLOCK)
            .ok_or_else(|| Overflow::for_type::<u8>(1, dim))?;
        Ok(Self { dim, padded })
    }
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
    pub(crate) fn padded_dim(&self) -> usize {
        self.padded
    }
    pub(super) fn k(&self) -> DimK {
        DimK::new(value_or_one(self.padded))
    }
    pub(crate) fn position(d: usize) -> usize {
        d / BLOCK * BLOCK + d % 2 * decode::BYTES + d % BLOCK / 2
    }

    #[expect(
        clippy::expect_used,
        reason = "padded blocks contain two complete decoder channels"
    )]
    #[inline(always)]
    pub(super) fn decode_row(&self, arch: impl Decoder, packed: &[u8], output: &mut [u8]) {
        assert_eq!(packed.len(), self.dim.div_ceil(2));
        assert_eq!(output.len(), self.padded);
        let (blocks, tail) = packed.as_chunks::<{ decode::BYTES }>();
        let (decoded, decoded_tail) = output.split_at_mut(blocks.len() * BLOCK);
        for (source, block) in blocks.iter().zip(decoded.as_chunks_mut::<BLOCK>().0) {
            let (low, high) = block.split_at_mut(decode::BYTES);
            arch.block(
                source,
                low.try_into().expect("low channel"),
                high.try_into().expect("high channel"),
            );
        }
        if !tail.is_empty() {
            let (low, high) = decoded_tail.split_at_mut(decode::BYTES);
            arch.unpack(
                tail,
                low.try_into().expect("low channel"),
                high.try_into().expect("high channel"),
            );
        }
        if !self.dim.is_multiple_of(2) {
            output[Self::position(self.dim)] = 0;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AView<'a, const MR: usize, const PACK: usize> {
    values: packed::View<'a, u8, MR, PACK>,
    compensation: &'a [[MinMaxCompensation; MR]],
    rows: usize,
    layout: &'a Layout,
}

impl<'a, const MR: usize, const PACK: usize> AView<'a, MR, PACK> {
    pub(crate) fn new(
        values: BlockTransposedRef<'a, u8, MR, PACK>,
        compensation: &'a [[MinMaxCompensation; MR]],
        layout: &'a Layout,
    ) -> Option<Self> {
        const { assert!(BLOCK.is_multiple_of(PACK)) };
        assert_eq!(values.ncols(), layout.padded_dim(), "query layout mismatch");
        assert_eq!(
            values.num_blocks(),
            compensation.len(),
            "query metadata mismatch"
        );
        Some(Self {
            values: packed::View::from_block_transposed(values)?,
            compensation,
            rows: values.nrows(),
            layout,
        })
    }

    pub(super) fn rows(self) -> usize {
        self.rows
    }
    pub(super) fn layout(self) -> &'a Layout {
        self.layout
    }
    #[inline(always)]
    pub(super) fn visit_panels(self, mut visit: impl FnMut(APanel<'_, MR, PACK>, usize)) {
        // SAFETY: Each packed panel has matching metadata and zero row/column padding.
        unsafe {
            self.values.visit_panels(self.layout.k(), |values, block| {
                visit(
                    APanel {
                        values,
                        compensation: &self.compensation[block],
                        valid_rows: (self.rows - block * MR).min(MR),
                        layout: self.layout,
                    },
                    block * MR,
                );
            });
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct APanel<'a, const MR: usize, const PACK: usize> {
    values: packed::Panel<'a, u8, MR, PACK>,
    compensation: &'a [MinMaxCompensation; MR],
    valid_rows: usize,
    layout: &'a Layout,
}

impl<const MR: usize, const PACK: usize> APanel<'_, MR, PACK> {
    pub(super) fn layout(&self) -> &Layout {
        self.layout
    }

    pub(super) fn valid_rows(&self) -> usize {
        self.valid_rows
    }
    pub(super) fn compensation(&self) -> &[MinMaxCompensation; MR] {
        self.compensation
    }

    /// # Safety
    ///
    /// group < K / PACK. Query padding makes every group a complete MR x PACK patch.
    pub(super) unsafe fn group(&self, group: usize) -> &[[u8; PACK]; MR] {
        // SAFETY: The caller provides an in-bounds group; each spans MR * PACK elements.
        let patch: Slice<'_, u8> = unsafe {
            self.values
                .as_ptr()
                .add(self.values.pack_stride() * group)
                .truncate(self.values.pack_stride())
        };
        // SAFETY: This span contains exactly MR * PACK consecutive bytes.
        unsafe { &*patch.as_ptr().cast::<[[u8; PACK]; MR]>() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_decode(arch: impl Decoder) {
        for dim in 0..=129 {
            let layout = Layout::new(dim).unwrap();
            let packed = vec![0xa5; dim.div_ceil(2)];
            let mut output = vec![0xfe; layout.padded + 2];
            layout.decode_row(arch, &packed, &mut output[1..layout.padded + 1]);
            let mut expected = vec![0; layout.padded];
            for block in (0..dim).step_by(BLOCK) {
                let count = (dim - block).min(BLOCK);
                expected[block..block + count.div_ceil(2)].fill(5);
                expected[block + decode::BYTES..block + decode::BYTES + count / 2].fill(10);
            }
            assert_eq!(&output[1..layout.padded + 1], expected, "dim={dim}");
            assert_eq!((output[0], output[layout.padded + 1]), (0xfe, 0xfe));
        }
        assert!(Layout::new(usize::MAX).is_err());
    }

    #[test]
    fn all_partial_blocks_ignore_unused_nibbles_and_overwrite_padding() {
        check_decode(diskann_wide::arch::Scalar::new());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_decode(arch);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                check_decode(arch);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_decode(arch);
        }
    }
}
