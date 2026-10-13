/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Canonical B tiles and prepared panels whose values and compensation cannot drift apart.

use super::{decode::Decoder, layout::Layout};
use crate::{
    matrix_kernels::{blocks::unpacked, num::Elements, ptr::Slice},
    minmax::{MinMaxCompensation, MinMaxMeta},
    multi_vector::MatRef,
};
use std::num::NonZeroUsize;

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
    #[expect(
        clippy::expect_used,
        reason = "tile extents are nonzero and rows retain canonical provenance"
    )]
    #[inline(always)]
    pub(super) fn prepare<'a>(
        &'a mut self,
        arch: impl Decoder,
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
            layout.decode_row(arch, source.vector().as_slice(), output);
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
        // SAFETY: Scratch::prepare binds the decoded stride and value/metadata row counts.
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
    pub(super) unsafe fn group<const PACK: usize>(&self, row: usize, group: usize) -> [u8; PACK] {
        // SAFETY: The caller selects a complete group inside one decoded row.
        unsafe {
            self.values
                .as_ptr()
                .add(self.values.stride(self.layout.k()) * row + Elements::new(group * PACK))
                .truncate(Elements::new(PACK))
                .as_ptr()
                .cast::<[u8; PACK]>()
                .read_unaligned()
        }
    }
}
