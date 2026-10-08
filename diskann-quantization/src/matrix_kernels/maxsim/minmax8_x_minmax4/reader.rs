/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Canonical document rows and decoded panels with synchronized compensation metadata.

use std::num::NonZeroUsize;

use crate::{
    matrix_kernels::{
        blocks::unpacked,
        num::{DimK, Elements},
        ptr::Slice,
    },
    minmax::{DataRef, MinMaxCompensation, MinMaxMeta},
    multi_vector::MatRef,
};

use super::{decode::Decoder, layout::EvenOdd};

#[derive(Clone, Copy)]
pub(crate) struct MinMax4Rows<'a> {
    doc: MatRef<'a, MinMaxMeta<4>>,
    start: usize,
    len: usize,
}

impl<'a> MinMax4Rows<'a> {
    pub(crate) fn new(doc: MatRef<'a, MinMaxMeta<4>>) -> Self {
        Self {
            doc,
            start: 0,
            len: doc.num_vectors(),
        }
    }

    pub(super) fn dim(self) -> usize {
        self.doc.repr().intrinsic_dim()
    }

    pub(super) fn rows(self) -> usize {
        self.len
    }

    pub(super) fn visit_tiles(self, rows: NonZeroUsize, mut visit: impl FnMut(Self)) {
        for offset in (0..self.len).step_by(rows.get()) {
            visit(Self {
                doc: self.doc,
                start: self.start + offset,
                len: (self.len - offset).min(rows.get()),
            });
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "visit_tiles preserves the source row bounds"
    )]
    fn iter(&self) -> impl ExactSizeIterator<Item = DataRef<'_, 4>> {
        (self.start..self.start + self.len).map(|row| {
            self.doc
                .get_row(row)
                .expect("document tile row out of bounds")
        })
    }
}

/// Scratch for a single B tile. Resizing and slicing always apply to both buffers.
pub(super) struct BScratch {
    values: Vec<u8>,
    compensation: Vec<MinMaxCompensation>,
    dim: usize,
    k: DimK,
}

impl BScratch {
    pub(super) fn new(dim: usize, k: DimK, rows: NonZeroUsize) -> Self {
        assert_eq!(
            k.value().get(),
            EvenOdd::padded(dim),
            "scratch padded dimension mismatch",
        );
        Self {
            values: vec![0; rows.get() * k.value().get()],
            compensation: vec![MinMaxCompensation::default(); rows.get()],
            dim,
            k,
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "the driver visits only nonempty document tiles"
    )]
    pub(super) fn decode(&mut self, arch: impl Decoder, doc: MinMax4Rows<'_>) -> BTile<'_> {
        assert_eq!(doc.dim(), self.dim, "document dimension mismatch");
        assert!(
            doc.rows() <= self.compensation.len(),
            "document tile exceeds scratch: {} rows, capacity {}",
            doc.rows(),
            self.compensation.len(),
        );
        let rows = doc.rows();
        let values = &mut self.values[..rows * self.k.value().get()];
        let meta = &mut self.compensation[..rows];
        for ((row, output), meta) in doc
            .iter()
            .zip(values.chunks_exact_mut(self.k.value().get()))
            .zip(meta.iter_mut())
        {
            *meta = row.meta();
            arch.decode(row.vector().as_slice(), self.dim, output);
        }
        let rows = NonZeroUsize::new(rows).expect("empty document tile");
        BTile {
            // SAFETY: Every row has K decoded bytes and matching metadata.
            values: unsafe { unpacked::View::new(Slice::new(values), rows, self.k) },
            compensation: meta,
            dim: self.dim,
            k: self.k,
        }
    }
}

/// Decoded documents in the query's dimension order. Only [`BScratch`] can construct this view.
#[derive(Clone, Copy)]
pub(super) struct BTile<'a> {
    values: unpacked::View<'a, u8>,
    compensation: &'a [MinMaxCompensation],
    dim: usize,
    k: DimK,
}

impl BTile<'_> {
    #[inline(always)]
    pub(super) fn visit_panels<const NR: usize>(
        &self,
        mut visit: impl FnMut(BPanel<'_, NR>),
    ) -> Option<BRemainder<'_, NR>> {
        let meta = self.compensation.as_chunks::<NR>().0;
        // SAFETY: The value view and K are constructed together by BScratch.
        let remainder = unsafe {
            self.values.visit_panels(
                self.k,
                |values: unpacked::Panel<'_, u8, NR>, start: usize| {
                    visit(BPanel {
                        values,
                        compensation: &meta[start / NR],
                    });
                },
            )
        };
        remainder.map(|values| BRemainder {
            compensation: &self.compensation[values.start()..],
            values,
        })
    }

    pub(super) fn dim(self) -> usize {
        self.dim
    }

    pub(super) fn k(self) -> DimK {
        self.k
    }

    #[cfg(test)]
    pub(super) fn as_slices(&self) -> (&[u8], &[MinMaxCompensation]) {
        // SAFETY: The stored dimension and view originate from the same scratch allocation.
        (
            unsafe { self.values.as_std_slice(self.k) },
            self.compensation,
        )
    }
}

pub(super) struct BRemainder<'a, const NR: usize> {
    values: unpacked::Remainder<'a, u8, NR>,
    compensation: &'a [MinMaxCompensation],
}

impl<const NR: usize> BRemainder<'_, NR> {
    #[inline(always)]
    pub(super) fn try_as_panel<const N: usize>(&self) -> Option<BPanel<'_, N>> {
        let values = self.values.try_as_panel()?;
        Some(BPanel {
            values,
            compensation: &self.compensation.as_chunks::<N>().0[0],
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct BPanel<'a, const NR: usize> {
    values: unpacked::Panel<'a, u8, NR>,
    compensation: &'a [MinMaxCompensation; NR],
}

impl<'a, const NR: usize> BPanel<'a, NR> {
    #[cfg(test)]
    pub(super) fn from_test_values(
        values: &'a [u8],
        compensation: &'a [MinMaxCompensation; NR],
        k: DimK,
    ) -> Self {
        assert_eq!(values.len(), NR * k.value().get());
        assert!(
            k.value()
                .get()
                .is_multiple_of(super::layout::EvenOdd::BLOCK)
        );
        assert!(values.iter().all(|&v| v < 16));
        Self {
            // SAFETY: The test supplies NR complete rows with the specified dimension.
            values: unsafe { unpacked::Panel::new(Slice::new(values), k) },
            compensation,
        }
    }

    pub(super) fn compensation(self) -> &'a [MinMaxCompensation; NR] {
        self.compensation
    }

    /// # Safety
    ///
    /// `k` must equal the panel's contraction dimension, `PACK` must divide `k`,
    /// `row < NR`, and `group < k / PACK`.
    #[inline(always)]
    pub(super) unsafe fn group<const PACK: usize>(
        self,
        k: DimK,
        row: usize,
        group: usize,
    ) -> [u8; PACK] {
        let stride = self.values.stride(k);
        // SAFETY: Both dimensions are multiples of PACK, and caller supplies a valid group.
        unsafe {
            self.values
                .as_ptr()
                .add(stride * row + Elements::new(group * PACK))
                .truncate(Elements::new(PACK))
                .as_ptr()
                .cast::<[u8; PACK]>()
                .read_unaligned()
        }
    }
}
