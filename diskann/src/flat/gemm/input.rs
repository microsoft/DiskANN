/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{conversion::CastFromSlice, distance::Metric};

use crate::{ANNError, ANNResult, utils::VectorRepr};

/// Opt in to value-preserving FP32 packing for dense flat search.
///
/// Implementations must preserve the distance semantics of VectorRepr; a codec
/// whose original scorer uses approximate distances should keep its visitor instead.
pub trait GemmElement: VectorRepr + Copy + 'static {
    /// Whether packing requires a copy, used by the automatic execution policy.
    const NEEDS_PACKING: bool;

    /// Borrow FP32 input or pack it into reusable scratch space. The returned
    /// slice must contain exactly one value-preserving FP32 element per input.
    fn pack<'a>(input: &'a [Self], scratch: &'a mut Vec<f32>) -> &'a [f32];

    /// Preserve type-specific metric semantics before converting the elements.
    fn effective_metric(metric: Metric) -> Metric {
        metric
    }
}

impl GemmElement for f32 {
    const NEEDS_PACKING: bool = false;
    fn pack<'a>(input: &'a [Self], _scratch: &'a mut Vec<f32>) -> &'a [f32] {
        input
    }
}

impl GemmElement for half::f16 {
    const NEEDS_PACKING: bool = true;
    fn pack<'a>(input: &'a [Self], scratch: &'a mut Vec<f32>) -> &'a [f32] {
        scratch.resize(input.len(), 0.0);
        scratch.as_mut_slice().cast_from_slice(input);
        scratch
    }
}

macro_rules! integer_element {
    ($t:ty) => {
        impl GemmElement for $t {
            const NEEDS_PACKING: bool = true;
            fn pack<'a>(input: &'a [Self], scratch: &'a mut Vec<f32>) -> &'a [f32] {
                scratch.clear();
                scratch.extend(input.iter().map(|&x| f32::from(x)));
                scratch
            }
            fn effective_metric(metric: Metric) -> Metric {
                match metric {
                    Metric::CosineNormalized => Metric::Cosine,
                    other => other,
                }
            }
        }
    };
}
integer_element!(i8);
integer_element!(u8);

/// Rows to visit, in scan order. Repeated rows remain repeated candidates, as in
/// the visitor API. Filtering and ties refer to this order, not to ID magnitude.
#[derive(Debug, Clone, Copy, Default)]
pub enum Selection<'a> {
    #[default]
    All,
    Range {
        start: usize,
        end: usize,
    },
    Rows(&'a [usize]),
}

#[derive(Debug, Clone, Copy)]
enum Ids<'a, I> {
    Explicit(&'a [I]),
    Implicit(fn(usize) -> I),
}

/// A borrowed database with an explicit row-to-ID mapping and optional selection.
#[derive(Debug, Clone, Copy)]
pub struct Dataset<'a, T, I> {
    pub(super) data: rowmajor::Ref<'a, T>,
    ids: Ids<'a, I>,
    selection: Selection<'a>,
    pub(super) norms: Option<&'a [f32]>,
}

impl<'a, T, I: Copy> Dataset<'a, T, I> {
    /// # Errors
    /// Returns an error unless there is exactly one ID per database row.
    #[track_caller]
    pub fn new(data: rowmajor::Ref<'a, T>, ids: &'a [I]) -> ANNResult<Self> {
        if ids.len() != data.nrows() {
            return Err(ANNError::message("database row and ID counts differ"));
        }
        Ok(Self {
            data,
            ids: Ids::Explicit(ids),
            selection: Selection::All,
            norms: None,
        })
    }

    /// # Errors
    /// Returns an error for an invalid range or a row outside the database.
    #[track_caller]
    pub fn select(mut self, selection: Selection<'a>) -> ANNResult<Self> {
        let valid = match selection {
            Selection::All => true,
            Selection::Range { start, end } => start <= end && end <= self.data.nrows(),
            Selection::Rows(rows) => rows.iter().all(|&row| row < self.data.nrows()),
        };
        if !valid {
            return Err(ANNError::message(
                "candidate selection is outside the database",
            ));
        }
        self.selection = selection;
        Ok(self)
    }

    /// Reuse squared norms belonging to this exact database snapshot. Values must
    /// be computed on its FP32 representation, in original database row order.
    ///
    /// # Errors
    /// Returns an error if the norm count differs from the database row count.
    #[track_caller]
    pub fn with_squared_norms(mut self, norms: &'a [f32]) -> ANNResult<Self> {
        if norms.len() != self.data.nrows() {
            return Err(ANNError::message("database row and norm counts differ"));
        }
        self.norms = Some(norms);
        Ok(self)
    }

    pub(super) fn len(&self) -> usize {
        match self.selection {
            Selection::All => self.data.nrows(),
            Selection::Range { start, end } => end - start,
            Selection::Rows(rows) => rows.len(),
        }
    }

    pub(super) fn row_index(&self, ordinal: usize) -> usize {
        match self.selection {
            Selection::All => ordinal,
            Selection::Range { start, .. } => start + ordinal,
            Selection::Rows(rows) => rows[ordinal],
        }
    }

    pub(super) fn id(&self, ordinal: usize) -> I {
        let row = self.row_index(ordinal);
        match self.ids {
            Ids::Explicit(ids) => ids[row],
            Ids::Implicit(map) => map(row),
        }
    }

    pub(super) fn tile<'b>(&'b self, start: usize, len: usize, scratch: &'b mut Vec<T>) -> &'b [T]
    where
        T: Copy,
    {
        let dim = self.data.ncols();
        match self.selection {
            Selection::All | Selection::Range { .. } => {
                let row = self.row_index(start);
                &self.data.as_slice()[row * dim..(row + len) * dim]
            }
            Selection::Rows(_) => {
                scratch.clear();
                for ordinal in start..start + len {
                    scratch.extend_from_slice(self.data.row(self.row_index(ordinal)));
                }
                scratch
            }
        }
    }
}

impl<'a> Dataset<'a, f32, u32> {
    pub(super) fn sequential(data: rowmajor::Ref<'a, f32>) -> Self {
        Self {
            data,
            ids: Ids::Implicit(|row| row as u32),
            selection: Selection::All,
            norms: None,
        }
    }
}

/// A synchronous, stable membership decision for a query and candidate ID.
/// Async metadata access belongs in the source, before submitting the block.
pub trait CandidateFilter<I>: Sync {
    /// May be true only when every pair is eligible, allowing checks to be skipped.
    const ALL: bool = false;
    fn is_member(&self, query: usize, id: I) -> bool;
}

impl<I, F: Fn(usize, I) -> bool + Sync> CandidateFilter<I> for F {
    fn is_member(&self, query: usize, id: I) -> bool {
        self(query, id)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AllCandidates;

impl<I> CandidateFilter<I> for AllCandidates {
    const ALL: bool = true;
    fn is_member(&self, _query: usize, _id: I) -> bool {
        true
    }
}
