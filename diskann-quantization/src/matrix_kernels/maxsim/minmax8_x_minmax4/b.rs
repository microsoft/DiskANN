/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! B-side unpacking, scratch preparation, and synchronized value/compensation panels.

use super::layout::{BLOCK, Layout, PACKED_BLOCK};
use crate::{
    matrix_kernels::{
        blocks::unpacked,
        bounds::{self, Bound},
        num::{Bytes, Elements},
        ptr::Slice,
    },
    minmax::{MinMaxCompensation, MinMaxMeta},
    multi_vector::MatRef,
};
use diskann_wide::{Architecture, arch::Scalar};
use std::num::NonZeroUsize;

#[inline(always)]
fn unpack_row(layout: &Layout, arch: impl UnpackNibbles, packed: &[u8], output: &mut [u8]) {
    assert_eq!(packed.len(), layout.dim().div_ceil(2));
    assert_eq!(output.len(), layout.padded_dim());
    let (blocks, tail) = packed.as_chunks::<PACKED_BLOCK>();
    let (unpacked, unpacked_tail) = output.split_at_mut(blocks.len() * BLOCK);
    for (source, block) in blocks.iter().zip(unpacked.as_chunks_mut::<BLOCK>().0) {
        let (low, high) = block.split_at_mut(PACKED_BLOCK);
        arch.unpack(source, low, high);
    }
    if !tail.is_empty() {
        let (low, high) = unpacked_tail.split_at_mut(PACKED_BLOCK);
        arch.unpack(tail, low, high);
    }
    if !layout.dim().is_multiple_of(2) {
        output[Layout::position(layout.dim())] = 0;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RawRows<'a> {
    doc: MatRef<'a, MinMaxMeta<4>>,
    start: usize,
    rows: usize,
}

impl<'a> RawRows<'a> {
    pub(crate) fn new(doc: MatRef<'a, MinMaxMeta<4>>) -> Self {
        Self {
            doc,
            start: 0,
            rows: doc.num_vectors(),
        }
    }
    pub(super) fn dim(self) -> usize {
        self.doc.repr().intrinsic_dim()
    }
    pub(super) fn rows(self) -> usize {
        self.rows
    }
    #[inline(always)]
    pub(super) fn visit_tiles(self, rows: NonZeroUsize, mut visit: impl FnMut(RawRows<'_>)) {
        for offset in (0..self.rows).step_by(rows.get()) {
            visit(Self {
                doc: self.doc,
                start: self.start + offset,
                rows: (self.rows - offset).min(rows.get()),
            });
        }
    }
}

#[derive(Default)]
pub(super) struct Scratch {
    values: Vec<u8>,
    compensation: Vec<MinMaxCompensation>,
}

impl Scratch {
    /// Working-set bytes per prepared row, including K padding and compensation.
    pub(super) fn row_bytes(layout: &Layout) -> Bytes {
        Bytes::new(layout.padded_dim() + std::mem::size_of::<MinMaxCompensation>())
    }

    #[expect(
        clippy::expect_used,
        reason = "tile extents are nonzero and rows retain canonical provenance"
    )]
    #[inline(always)]
    pub(super) fn prepare<'a>(
        &'a mut self,
        arch: impl UnpackNibbles,
        raw: RawRows<'_>,
        layout: &'a Layout,
    ) -> BTile<'a> {
        assert_eq!(raw.dim(), layout.dim(), "document dimension mismatch");
        let k = layout.k();
        let rows = NonZeroUsize::new(raw.rows).expect("nonempty B tile");
        let len = raw
            .rows
            .checked_mul(k.value().get())
            .expect("B scratch extent overflow");
        if self.values.len() < len {
            self.values.resize(len, 0);
        }
        if self.compensation.len() < raw.rows {
            self.compensation
                .resize(raw.rows, MinMaxCompensation::default());
        }
        let values = &mut self.values[..len];
        let compensation = &mut self.compensation[..raw.rows];
        for (i, (output, metadata)) in values
            .chunks_exact_mut(k.value().get())
            .zip(compensation.iter_mut())
            .enumerate()
        {
            let source = raw
                .doc
                .get_row(raw.start + i)
                .expect("canonical row in bounds");
            *metadata = source.meta();
            unpack_row(layout, arch, source.vector().as_slice(), output);
        }
        BTile {
            // SAFETY: Preparation completely writes rows * K values and one metadata per row.
            values: unsafe { unpacked::View::new(Slice::new(values), rows, k) },
            compensation,
            layout,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct BTile<'a> {
    values: unpacked::View<'a, u8>,
    compensation: &'a [MinMaxCompensation],
    layout: &'a Layout,
}

impl BTile<'_> {
    pub(super) fn layout(&self) -> &Layout {
        self.layout
    }
    #[inline(always)]
    pub(super) fn visit_panels<const N: usize>(
        &self,
        mut visit: impl FnMut(BPanel<'_, N>),
    ) -> Option<Remainder<'_, N>> {
        const { assert!(N > 0) };
        let metadata = self.compensation.as_chunks::<N>().0;
        // SAFETY: Scratch::prepare binds the unpacked stride and value/metadata row counts.
        let remainder = unsafe {
            self.values.visit_panels(
                self.layout.k(),
                |values: unpacked::Panel<'_, u8, N>, start| {
                    visit(BPanel {
                        values,
                        compensation: &metadata[start / N],
                        layout: self.layout,
                    });
                },
            )
        };
        remainder.map(|values| Remainder {
            compensation: &self.compensation[values.start()..],
            values,
            layout: self.layout,
        })
    }
}

pub(super) struct Remainder<'a, const CAP: usize> {
    values: unpacked::Remainder<'a, u8, CAP>,
    compensation: &'a [MinMaxCompensation],
    layout: &'a Layout,
}

impl<const CAP: usize> Remainder<'_, CAP> {
    #[expect(
        clippy::expect_used,
        reason = "the metadata remainder has the same extent as values"
    )]
    pub(super) fn panel<const N: usize>(&self) -> Option<BPanel<'_, N>> {
        let values = self.values.try_as_panel::<N>()?;
        Some(BPanel {
            values,
            compensation: self.compensation.try_into().expect("exact metadata panel"),
            layout: self.layout,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct BPanel<'a, const N: usize> {
    values: unpacked::Panel<'a, u8, N>,
    compensation: &'a [MinMaxCompensation; N],
    layout: &'a Layout,
}

impl<const N: usize> BPanel<'_, N> {
    pub(super) fn compensation(&self) -> &[MinMaxCompensation; N] {
        self.compensation
    }

    /// # Safety
    ///
    /// row < N and group < K / PACK. Layout preparation completely writes K bytes per row.
    pub(super) unsafe fn group<const PACK: usize>(
        &self,
        row: usize,
        group: usize,
    ) -> Slice<'_, u8> {
        const { assert!(PACK > 0) };
        bounds::check_lt!(Bound::new(row), N, "document row out of bounds");
        bounds::check_lt!(
            Bound::new(group),
            self.layout.k().value().get() / PACK,
            "document group out of bounds"
        );
        // SAFETY: The caller selects a complete group inside one unpacked row.
        unsafe {
            self.values
                .as_ptr()
                .add(self.values.stride(self.layout.k()) * row + Elements::new(group * PACK))
                .truncate(Elements::new(PACK))
        }
    }
}

pub(super) trait UnpackNibbles: Architecture {
    /// Expand low/high nibbles into equally sized channels and zero-fill unused output.
    /// Channel lengths must be at least `packed.len()`; no SIMD-width alignment is required.
    fn unpack(self, packed: &[u8], low: &mut [u8], high: &mut [u8]);
}

#[inline(always)]
fn assert_unpack_lengths(packed: &[u8], low: &[u8], high: &[u8]) {
    assert_eq!(low.len(), high.len(), "unpack channel length mismatch");
    assert!(packed.len() <= low.len(), "unpack output is too short");
}

impl UnpackNibbles for Scalar {
    #[inline(always)]
    fn unpack(self, packed: &[u8], low: &mut [u8], high: &mut [u8]) {
        assert_unpack_lengths(packed, low, high);
        for (i, (low, high)) in low.iter_mut().zip(high).enumerate() {
            let byte = packed.get(i).copied().unwrap_or(0);
            *low = byte & 15;
            *high = byte >> 4;
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

    #[inline(always)]
    fn unpack(
        arch: V3,
        packed: &[u8],
        low: &mut [u8],
        high: &mut [u8],
        load: impl Fn(&[u8]) -> <V3 as Architecture>::u32x8,
    ) {
        assert_unpack_lengths(packed, low, high);
        diskann_wide::alias!(words = <V3>::u32x8);
        diskann_wide::alias!(bytes = <V3>::u8x32);
        let mask = words::splat(arch, 0x0f0f_0f0f);
        for ((source, low), high) in packed
            .chunks(32)
            .zip(low.chunks_mut(32))
            .zip(high.chunks_mut(32))
        {
            let codes = load(source);
            let lows = bytes::from_underlying(arch, (codes & mask).to_underlying());
            let highs = bytes::from_underlying(arch, ((codes >> 4) & mask).to_underlying());
            // SAFETY: The channels have equal lengths. Complete chunks admit full stores;
            // partial chunks are written only up to their actual length.
            unsafe {
                if low.len() == 32 {
                    lows.store_simd(low.as_mut_ptr());
                    highs.store_simd(high.as_mut_ptr());
                } else {
                    lows.store_simd_first(low.as_mut_ptr(), low.len());
                    highs.store_simd_first(high.as_mut_ptr(), high.len());
                }
            }
        }
        let written = packed.len().next_multiple_of(32).min(low.len());
        if written < low.len() {
            low[written..].fill(0);
            high[written..].fill(0);
        }
    }

    impl UnpackNibbles for V3 {
        #[inline(always)]
        fn unpack(self, packed: &[u8], low: &mut [u8], high: &mut [u8]) {
            diskann_wide::alias!(words = <V3>::u32x8);
            unpack(
                self,
                packed,
                low,
                high,
                #[inline(always)]
                |source| {
                    if source.len() == 32 {
                        // SAFETY: The complete input chunk contains 32 readable bytes.
                        unsafe { words::load_simd(self, source.as_ptr().cast()) }
                    } else {
                        let mut tail = [0; 32];
                        tail[..source.len()].copy_from_slice(source);
                        // SAFETY: The local buffer contains 32 initialized bytes.
                        unsafe { words::load_simd(self, tail.as_ptr().cast()) }
                    }
                },
            );
        }
    }

    impl UnpackNibbles for V4 {
        #[inline(always)]
        fn unpack(self, packed: &[u8], low: &mut [u8], high: &mut [u8]) {
            diskann_wide::alias!(bytes = <V4>::u8x32);
            diskann_wide::alias!(words = <V3>::u32x8);
            let arch = self.retarget();
            unpack(
                arch,
                packed,
                low,
                high,
                #[inline(always)]
                |source| {
                    if source.len() == 32 {
                        // SAFETY: The complete input chunk contains 32 readable bytes.
                        unsafe { words::load_simd(arch, source.as_ptr().cast()) }
                    } else {
                        // SAFETY: Read only the available bytes; the other lanes are zero.
                        let codes =
                            unsafe { bytes::load_simd_first(self, source.as_ptr(), source.len()) };
                        // SAFETY: The SIMD array contains 32 initialized bytes.
                        unsafe { words::load_simd(arch, codes.to_array().as_ptr().cast()) }
                    }
                },
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl UnpackNibbles for diskann_wide::arch::aarch64::Neon {
    #[inline(always)]
    fn unpack(self, packed: &[u8], low: &mut [u8], high: &mut [u8]) {
        assert_unpack_lengths(packed, low, high);
        use diskann_wide::SIMDVector;
        diskann_wide::alias!(bytes = <diskann_wide::arch::aarch64::Neon>::u8x16);
        let mask = bytes::splat(self, 15);
        for ((source, low), high) in packed
            .chunks(16)
            .zip(low.chunks_mut(16))
            .zip(high.chunks_mut(16))
        {
            // SAFETY: Access only the available input/output bytes of each chunk.
            unsafe {
                let codes = bytes::load_simd_first(self, source.as_ptr(), source.len());
                if low.len() == 16 {
                    (codes & mask).store_simd(low.as_mut_ptr());
                    ((codes >> 4) & mask).store_simd(high.as_mut_ptr());
                } else {
                    (codes & mask).store_simd_first(low.as_mut_ptr(), low.len());
                    ((codes >> 4) & mask).store_simd_first(high.as_mut_ptr(), high.len());
                }
            }
        }

        let written = packed.len().next_multiple_of(16).min(low.len());
        if written < low.len() {
            low[written..].fill(0);
            high[written..].fill(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_bytes_include_unpacked_padding_and_compensation() {
        use crate::multi_vector::{Defaulted, Mat};

        for dim in [1, 63, 64, 65, 250, 256] {
            let layout = Layout::new(dim).unwrap();
            for rows in [1, 3] {
                let docs = Mat::new(MinMaxMeta::<4>::new(rows, dim), Defaulted).unwrap();
                let mut scratch = Scratch::default();
                scratch.prepare(Scalar::new(), RawRows::new(docs.as_view()), &layout);
                let bytes = std::mem::size_of_val(scratch.values.as_slice())
                    + std::mem::size_of_val(scratch.compensation.as_slice());
                assert_eq!(Scratch::row_bytes(&layout).value() * rows, bytes);
                assert!(Scratch::row_bytes(&layout).value() > docs.repr().ncols());
            }
        }
    }

    fn check(arch: impl UnpackNibbles) {
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
            let input: Vec<_> = (0..len).map(|i| (i * 37 + 0xf3) as u8).collect();
            for padding in [0, 1, 31] {
                let extent = len + padding;
                let mut low = vec![0xa5; extent + 2];
                let mut high = low.clone();
                arch.unpack(&input, &mut low[1..extent + 1], &mut high[1..extent + 1]);
                for i in 0..extent {
                    let byte = input.get(i).copied().unwrap_or(0);
                    assert_eq!(low[i + 1], byte & 15, "len={len}, padding={padding}, i={i}");
                    assert_eq!(
                        high[i + 1],
                        byte >> 4,
                        "len={len}, padding={padding}, i={i}"
                    );
                }
                assert_eq!(
                    (low[0], low[extent + 1], high[0], high[extent + 1]),
                    (0xa5, 0xa5, 0xa5, 0xa5)
                );
            }
        }
    }

    #[test]
    fn unpacking_is_independent_of_layout_and_register_width() {
        check(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check(arch);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                check(arch);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check(arch);
        }
    }

    #[test]
    #[should_panic(expected = "unpack channel length mismatch")]
    fn mismatched_channels_are_rejected() {
        Scalar::new().unpack(&[0x12], &mut [0; 1], &mut [0; 2]);
    }

    #[test]
    #[should_panic(expected = "unpack output is too short")]
    fn short_channels_are_rejected() {
        Scalar::new().unpack(&[0x12, 0x34], &mut [0; 1], &mut [0; 1]);
    }

    fn check_unpack(arch: impl UnpackNibbles) {
        for dim in 0..=129 {
            let layout = Layout::new(dim).unwrap();
            let mut packed: Vec<_> = (0..dim.div_ceil(2))
                .map(|i| (i * 37 + 0xa5) as u8)
                .collect();
            if dim % 2 == 1 {
                *packed.last_mut().unwrap() |= 0xf0;
            }
            let padded = layout.padded_dim();
            let mut output = vec![0xfe; padded + 2];
            unpack_row(&layout, arch, &packed, &mut output[1..padded + 1]);
            let mut expected = vec![0; padded];
            for block in (0..dim).step_by(BLOCK) {
                let count = (dim - block).min(BLOCK);
                for i in 0..count.div_ceil(2) {
                    expected[block + i] = packed[block / 2 + i] & 15;
                }
                for i in 0..count / 2 {
                    expected[block + PACKED_BLOCK + i] = packed[block / 2 + i] >> 4;
                }
            }
            assert_eq!(&output[1..padded + 1], expected, "dim={dim}");
            assert_eq!((output[0], output[padded + 1]), (0xfe, 0xfe));
        }
        assert!(Layout::new(usize::MAX).is_err());
    }

    #[test]
    fn all_partial_blocks_ignore_unused_nibbles_and_overwrite_padding() {
        check_unpack(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_unpack(arch);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                check_unpack(arch);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_unpack(arch);
        }
    }
}
