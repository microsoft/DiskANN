/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Adapt canonical MinMax4 rows to decoded tiles, preserving metadata row order.

use std::num::NonZeroUsize;

use crate::{
    matrix_kernels::{blocks::unpacked, num::DimK, ptr::Slice},
    minmax::{DataRef, MinMaxCompensation, MinMaxMeta},
    multi_vector::MatRef,
};

use super::{decode::Decoder, layout::EvenOdd64Layout};

/// Borrowed MinMax4 document rows, including packed codes and compensation metadata.
#[derive(Clone, Copy)]
pub(crate) struct MinMax4Rows<'a> {
    values: unpacked::View<'a, u8>,
    stride: DimK,
    dim: usize,
}

impl<'a> MinMax4Rows<'a> {
    #[expect(
        clippy::expect_used,
        reason = "canonical representation supplies a nonzero valid row stride"
    )]
    pub(crate) fn new(doc: MatRef<'a, MinMaxMeta<4>>) -> Option<Self> {
        let rows = NonZeroUsize::new(doc.num_vectors())?;
        let stride = DimK::new(NonZeroUsize::new(doc.repr().ncols()).expect("canonical header"));
        let len = rows
            .get()
            .checked_mul(stride.value().get())
            .expect("canonical extent overflow");
        // SAFETY: MatRef owns the provenance for exactly rows * canonical stride bytes.
        let bytes = unsafe { std::slice::from_raw_parts(doc.as_raw_ptr(), len) };
        Some(Self {
            // SAFETY: The validated representation provides the exact shape of bytes.
            values: unsafe { unpacked::View::new(Slice::new(bytes), rows, stride) },
            stride,
            dim: doc.repr().intrinsic_dim(),
        })
    }

    pub(super) fn dim(self) -> usize {
        self.dim
    }

    pub(super) fn rows(self) -> NonZeroUsize {
        self.values.extent()
    }

    pub(super) fn visit_tiles(self, rows: NonZeroUsize, mut visit: impl FnMut(MinMax4Rows<'_>)) {
        // SAFETY: All tiles inherit the row stride and canonical format from this reader.
        unsafe {
            self.values.visit_sub_views(rows, self.stride, |values, _| {
                visit(MinMax4Rows {
                    values,
                    stride: self.stride,
                    dim: self.dim,
                });
            });
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "decoded tiles require nonzero K and representable scratch"
    )]
    pub(super) fn decode<'b>(
        self,
        arch: impl Decoder,
        layout: EvenOdd64Layout,
        values: &'b mut [u8],
        meta: &'b mut [MinMaxCompensation],
    ) -> BTile<'b> {
        assert_eq!(self.dim, layout.dim(), "document dimension mismatch");
        let k = layout.padded();
        let dimension = DimK::new(NonZeroUsize::new(k).expect("nonempty decoded dimension"));
        assert_eq!(
            values.len(),
            self.rows()
                .get()
                .checked_mul(k)
                .expect("decoded tile extent overflow"),
            "decoded B value scratch must contain exactly rows * K elements",
        );
        assert_eq!(
            meta.len(),
            self.rows().get(),
            "decoded B metadata scratch must match rows"
        );
        // SAFETY: The stride is retained from the canonical input.
        let bytes = unsafe { self.values.as_std_slice(self.stride) };
        for ((row, output), meta) in bytes
            .chunks_exact(self.stride.value().get())
            .zip(values.chunks_exact_mut(k))
            .zip(meta.iter_mut())
        {
            // SAFETY: Every row has the canonical byte count for dim.
            let data = unsafe { DataRef::<4>::from_canonical_unchecked(row, self.dim) };
            *meta = data.meta();
            let vector = data.vector();
            // SAFETY: The canonical vector exposes exactly ceil(D / 2) code bytes.
            let codes =
                unsafe { std::slice::from_raw_parts(vector.as_ptr(), self.dim.div_ceil(2)) };
            arch.decode(codes, layout, output);
        }
        BTile {
            // SAFETY: The scratch shape was checked and every byte was initialized.
            values: unsafe { unpacked::View::new(Slice::new(values), self.rows(), dimension) },
            meta: Slice::new(meta),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct BTile<'a> {
    pub(super) values: unpacked::View<'a, u8>,
    pub(super) meta: Slice<'a, MinMaxCompensation>,
}
