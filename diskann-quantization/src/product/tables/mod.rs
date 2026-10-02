/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod basic;
mod transposed;

pub mod lookup;

#[cfg(test)]
pub(super) mod test;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DotAndNorm {
    dot: f32,
    square_norm: f32,
}

impl DotAndNorm {
    pub const fn new(dot: f32, square_norm: f32) -> Self {
        Self { dot, square_norm }
    }

    pub fn dot(&self) -> f32 {
        self.dot
    }

    pub fn square_norm(&self) -> f32 {
        self.square_norm
    }

    pub fn finish_cosine(&self, query_norm: f32) -> diskann_vector::SimilarityScore<f32> {
        use diskann_vector::SimilarityScore;

        if self.square_norm < f32::MIN_POSITIVE || query_norm < f32::MIN_POSITIVE {
            SimilarityScore::new(1.0)
        } else {
            let v = self.dot / (self.square_norm.sqrt() * query_norm);
            SimilarityScore::new(1.0 - (-1.0f32).max(1.0f32.min(v)))
        }
    }
}

impl std::ops::Add for DotAndNorm {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            dot: self.dot + rhs.dot,
            square_norm: self.square_norm + rhs.square_norm,
        }
    }
}

/////////////
// Exports //
/////////////

pub use basic::{BasicTable, BasicTableBase, BasicTableView, TableCompressionError};
pub use transposed::TransposedTable;
