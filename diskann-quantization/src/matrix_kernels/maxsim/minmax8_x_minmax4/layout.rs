/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared layout contract for MinMax A and B.
//! Each 64-dimensional block stores even coordinates followed by odd coordinates.
//! Missing coordinates are zero-padded before applying the block-transposed PACK layout.
//! This is a prepared-data format, not an ISA register width.

use crate::{
    matrix_kernels::num::{DimK, value_or_one},
    multi_vector::Overflow,
};

// V4 benchmarks on Xeon 8370C (MR16/NR8/PACK8, 22 shapes with 16/64 query and
// document rows, selected dimensions 63-257) compared 32/64/128-dimensional blocks.
// Versus 64, geometric-mean latency was ~1.7% lower at 32 and ~5.8% higher at 128.
// Keep the AVX2-friendly shared format rather than specialize V4 for a small gain;
// this is a measured tradeoff on that CPU, not a universally optimal block size.
pub(super) const BLOCK: usize = 64;
pub(super) const PACKED_BLOCK: usize = BLOCK / 2;
pub(crate) const MAX_DIM: usize = u32::MAX as usize / (255 * 15);

/// Original dimension and physical K for the fixed prepared-data format.
/// Equal layouts are interchangeable; they do not identify a particular allocation.
#[derive(Debug, PartialEq, Eq)]
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
        d / BLOCK * BLOCK + d % 2 * PACKED_BLOCK + d % BLOCK / 2
    }
}
