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

use super::{decode::Decoder, layout::RowLayout};

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
pub(super) struct BScratch<'a> {
    values: Vec<u8>,
    compensation: Vec<MinMaxCompensation>,
    layout: RowLayout<'a>,
}

impl<'a> BScratch<'a> {
    pub(super) fn new(layout: RowLayout<'a>, rows: NonZeroUsize) -> Self {
        assert!(
            rows.get() <= layout.max_b_rows(),
            "scratch exceeds validated byte budget"
        );
        Self {
            values: vec![0; rows.get() * layout.k().value().get()],
            compensation: vec![MinMaxCompensation::default(); rows.get()],
            layout,
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "the driver visits only nonempty document tiles"
    )]
    pub(super) fn decode(&mut self, arch: impl Decoder, doc: MinMax4Rows<'_>) -> BTile<'_> {
        assert_eq!(doc.dim(), self.layout.dim(), "document dimension mismatch");
        assert!(
            doc.rows() <= self.compensation.len(),
            "document tile exceeds scratch: {} rows, capacity {}",
            doc.rows(),
            self.compensation.len(),
        );
        let rows = doc.rows();
        let k = self.layout.k();
        let values = &mut self.values[..rows * k.value().get()];
        let meta = &mut self.compensation[..rows];
        for ((row, output), meta) in doc
            .iter()
            .zip(values.chunks_exact_mut(k.value().get()))
            .zip(meta.iter_mut())
        {
            *meta = row.meta();
            self.layout
                .decode_row(arch, row.vector().as_slice(), output);
        }
        let rows = NonZeroUsize::new(rows).expect("empty document tile");
        BTile {
            // SAFETY: Every row has K decoded bytes and matching metadata.
            values: unsafe { unpacked::View::new(Slice::new(values), rows, k) },
            compensation: meta,
            layout: self.layout,
        }
    }
}

/// Decoded documents in the query's dimension order. Only [`BScratch`] can construct this view.
#[derive(Clone, Copy)]
pub(super) struct BTile<'a> {
    values: unpacked::View<'a, u8>,
    compensation: &'a [MinMaxCompensation],
    layout: RowLayout<'a>,
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
                self.k(),
                |values: unpacked::Panel<'_, u8, NR>, start: usize| {
                    visit(BPanel {
                        values,
                        compensation: &meta[start / NR],
                        layout: self.layout,
                    });
                },
            )
        };
        remainder.map(|values| BRemainder {
            compensation: &self.compensation[values.start()..],
            values,
            layout: self.layout,
        })
    }

    pub(super) fn dim(self) -> usize {
        self.layout.dim()
    }

    pub(super) fn k(self) -> DimK {
        self.layout.k()
    }

    #[cfg(test)]
    pub(super) fn as_slices(&self) -> (&[u8], &[MinMaxCompensation]) {
        // SAFETY: The stored dimension and view originate from the same scratch allocation.
        (
            unsafe { self.values.as_std_slice(self.k()) },
            self.compensation,
        )
    }
}

pub(super) struct BRemainder<'a, const NR: usize> {
    values: unpacked::Remainder<'a, u8, NR>,
    compensation: &'a [MinMaxCompensation],
    layout: RowLayout<'a>,
}

impl<const NR: usize> BRemainder<'_, NR> {
    #[inline(always)]
    pub(super) fn try_as_panel<const N: usize>(&self) -> Option<BPanel<'_, N>> {
        let values = self.values.try_as_panel()?;
        Some(BPanel {
            values,
            compensation: &self.compensation.as_chunks::<N>().0[0],
            layout: self.layout,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct BPanel<'a, const NR: usize> {
    values: unpacked::Panel<'a, u8, NR>,
    compensation: &'a [MinMaxCompensation; NR],
    layout: RowLayout<'a>,
}

impl<'a, const NR: usize> BPanel<'a, NR> {
    #[cfg(test)]
    pub(super) fn from_test_values(
        values: &'a [u8],
        compensation: &'a [MinMaxCompensation; NR],
        layout: RowLayout<'a>,
    ) -> Self {
        let k = layout.k();
        assert_eq!(values.len(), NR * k.value().get());
        assert!(values.iter().all(|&v| v < 16));
        Self {
            // SAFETY: The test supplies NR complete rows with the specified dimension.
            values: unsafe { unpacked::Panel::new(Slice::new(values), k) },
            compensation,
            layout,
        }
    }

    pub(super) fn compensation(self) -> &'a [MinMaxCompensation; NR] {
        self.compensation
    }

    /// # Safety
    ///
    /// `PACK` must divide the layout's K, `row < NR`, and `group < K / PACK`.
    #[inline(always)]
    pub(super) unsafe fn group<const PACK: usize>(self, row: usize, group: usize) -> [u8; PACK] {
        let stride = self.values.stride(self.layout.k());
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
