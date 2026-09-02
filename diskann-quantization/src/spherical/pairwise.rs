/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prepared symmetric one-bit scoring for spherical RaBitQ data vectors.
//!
//! [`Pairwise1Bit`] scores one canonical data row against another row or a
//! contiguous target panel. The caller supplies the architecture selected for the
//! enclosing graph build. Packed-bit and compensation kernels use that same witness.

use diskann_utils::views::MatrixView;
use diskann_vector::{Norm, norm::FastL2NormSquared};
use diskann_wide::{Architecture, SIMDCast, SIMDPopcount, SIMDSumTree, SIMDVector, arch::Scalar};

#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use super::{DataMetaF32, DataRef, SphericalQuantizer, SupportedMetric, vectors};

/// Failure while preparing or evaluating spherical one-bit pair scores.
#[derive(Debug, thiserror::Error)]
pub enum PairwiseError {
    /// Quantizer output dimension cannot be represented by this scorer.
    #[error("one-bit encoded dimension {0} exceeds the u32 score limit")]
    DimensionTooLarge(usize),
    /// Quantizer constants required by compensation are not finite.
    #[error("spherical one-bit quantizer contains non-finite compensation constants")]
    NonFiniteQuantizer,
    /// One canonical source or target row has the wrong byte length.
    #[error("invalid {role} row length: expected {expected}, got {actual}")]
    InvalidRowLength {
        /// Row role used for diagnostics.
        role: &'static str,
        /// Canonical byte length required by the quantizer.
        expected: usize,
        /// Supplied byte length.
        actual: usize,
    },
    /// Target panel columns do not match one canonical row.
    #[error("invalid target panel width: expected {expected}, got {actual}")]
    InvalidTargetWidth {
        /// Canonical row byte length.
        expected: usize,
        /// Supplied panel column count.
        actual: usize,
    },
    /// Output score count does not match target panel rows.
    #[error("invalid score count: expected {expected}, got {actual}")]
    InvalidScoreCount {
        /// Required number of scores.
        expected: usize,
        /// Supplied score slice length.
        actual: usize,
    },
    /// Canonical metadata is malformed or non-finite.
    #[error("invalid spherical one-bit metadata in {role} row {row}")]
    InvalidMetadata {
        /// Row role used for diagnostics.
        role: &'static str,
        /// Row position within that role.
        row: usize,
    },
    /// Compensation produced a score that cannot be ranked.
    #[error("spherical one-bit score for target {target} is not finite")]
    NonFiniteScore {
        /// Target row position within the panel.
        target: usize,
    },
    /// Prepared metadata does not cover a requested target range.
    #[error("prepared panel has {prepared} rows but scoring requires {required}")]
    InvalidPreparedPanel {
        /// Number of rows successfully prepared in worker scratch.
        prepared: usize,
        /// Exclusive target end required by this score call.
        required: usize,
    },
    /// Worker scratch could not grow to the requested panel size.
    #[error("failed to reserve {additional} values for {buffer}")]
    Allocation {
        /// Scratch buffer being grown.
        buffer: &'static str,
        /// Additional element capacity requested.
        additional: usize,
    },
}

/// Reusable worker-owned scratch for [`Pairwise1Bit::score_panel`].
#[derive(Debug, Default)]
pub struct Pairwise1BitScratch {
    metadata: Vec<DataMetaF32>,
    raw_inner_products: Vec<u32>,
    source_words: Vec<u64>,
    transposed_words: Vec<u64>,
    target_corrections: Vec<f32>,
    target_metric_terms: Vec<f32>,
    target_bit_sums: Vec<f32>,
    transposed_stride: usize,
    transposed_word_count: usize,
    prepared_rows: usize,
}

impl Pairwise1BitScratch {
    /// Construct empty scratch without allocating.
    pub const fn new() -> Self {
        Self {
            metadata: Vec::new(),
            raw_inner_products: Vec::new(),
            source_words: Vec::new(),
            transposed_words: Vec::new(),
            target_corrections: Vec::new(),
            target_metric_terms: Vec::new(),
            target_bit_sums: Vec::new(),
            transposed_stride: 0,
            transposed_word_count: 0,
            prepared_rows: 0,
        }
    }

    /// Grow scratch for one target panel while retaining prior capacity.
    ///
    /// # Errors
    ///
    /// Returns [`PairwiseError::Allocation`] if either buffer cannot reserve the
    /// requested target count.
    pub fn prepare(&mut self, panel_rows: usize) -> Result<(), PairwiseError> {
        self.prepared_rows = 0;
        resize(
            "pairwise metadata",
            &mut self.metadata,
            panel_rows,
            DataMetaF32::default(),
        )?;
        resize(
            "pairwise raw inner products",
            &mut self.raw_inner_products,
            panel_rows,
            0,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct MetricParameters {
    squared_shift_norm: f32,
}

#[derive(Debug)]
struct PanelCall<'a> {
    source: &'a [u8],
    source_metadata: Option<DataMetaF32>,
    targets: MatrixView<'a, u8>,
    first_target: usize,
    scores: &'a mut [f32],
    scratch: &'a mut Pairwise1BitScratch,
    encoded_dim: usize,
    row_bytes: usize,
    parameters: MetricParameters,
}

/// Prepared symmetric scorer for canonical spherical `Data<1>` rows.
///
/// The handle stores no row or scratch borrow. It is immutable and may be shared
/// across workers; each worker must own its [`Pairwise1BitScratch`].
#[derive(Clone, Copy, Debug)]
pub struct Pairwise1Bit {
    encoded_dim: usize,
    row_bytes: usize,
    parameters: MetricParameters,
    metric: SupportedMetric,
}

impl Pairwise1Bit {
    /// Prepare metric and runtime-architecture-specific one-bit scoring.
    ///
    /// # Errors
    ///
    /// Returns [`PairwiseError`] if the encoded dimension or quantizer
    /// compensation constants cannot produce finite representable scores.
    pub fn new(quantizer: &SphericalQuantizer) -> Result<Self, PairwiseError> {
        let encoded_dim = quantizer.output_dim();
        if encoded_dim > u32::MAX as usize {
            return Err(PairwiseError::DimensionTooLarge(encoded_dim));
        }
        let bit_bytes = encoded_dim
            .checked_add(7)
            .ok_or(PairwiseError::DimensionTooLarge(encoded_dim))?
            / 8;
        let row_bytes = bit_bytes
            .checked_add(std::mem::size_of::<super::DataMeta>())
            .ok_or(PairwiseError::DimensionTooLarge(encoded_dim))?;
        debug_assert_eq!(row_bytes, DataRef::<1>::canonical_bytes(encoded_dim));

        let squared_shift_norm = FastL2NormSquared.evaluate(quantizer.shift());
        if !squared_shift_norm.is_finite() {
            return Err(PairwiseError::NonFiniteQuantizer);
        }
        Ok(Self {
            encoded_dim,
            row_bytes,
            parameters: MetricParameters { squared_shift_norm },
            metric: quantizer.metric(),
        })
    }

    /// Return the transformed dimension represented by each encoded row.
    pub const fn encoded_dim(&self) -> usize {
        self.encoded_dim
    }

    /// Return the canonical byte length required for each encoded row.
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Compute one final ascending metric score.
    ///
    /// Both inputs must use the canonical back-metadata layout produced by the
    /// quantizer used to construct this scorer.
    ///
    /// # Errors
    ///
    /// Returns [`PairwiseError`] for malformed rows, invalid metadata, or a
    /// non-finite compensated score.
    pub fn score_pair<A: PairwiseSIMDSchema>(
        &self,
        arch: A,
        source: &[u8],
        target: &[u8],
    ) -> Result<f32, PairwiseError> {
        validate_row(source, self.row_bytes, "source")?;
        validate_row(target, self.row_bytes, "target")?;
        let source_metadata = decode_row(arch, source, self.encoded_dim, "source", 0)?;
        let target_metadata = decode_row(arch, target, self.encoded_dim, "target", 0)?;
        let targets = MatrixView::try_from(target, 1, self.row_bytes).map_err(|error| {
            PairwiseError::InvalidRowLength {
                role: "target",
                expected: self.row_bytes,
                actual: error.into_inner().len(),
            }
        })?;
        let mut raw = [0u32; 1];
        A::raw_panel(arch, source, targets, self.encoded_dim, &mut raw);
        let score = match self.metric {
            SupportedMetric::SquaredL2 => SquaredL2::distance_from_raw(
                raw[0],
                source_metadata,
                target_metadata,
                self.encoded_dim as f32,
                self.parameters,
            ),
            SupportedMetric::InnerProduct => InnerProduct::distance_from_raw(
                raw[0],
                source_metadata,
                target_metadata,
                self.encoded_dim as f32,
                self.parameters,
            ),
            SupportedMetric::Cosine => Cosine::distance_from_raw(
                raw[0],
                source_metadata,
                target_metadata,
                self.encoded_dim as f32,
                self.parameters,
            ),
        };
        if score.is_finite() {
            Ok(score)
        } else {
            Err(PairwiseError::NonFiniteScore { target: 0 })
        }
    }

    /// Decode and retain metadata for a target panel in worker scratch.
    ///
    /// Call once when many sources score against the same panel, then call
    /// [`Self::score_prepared_panel`] for each source. Packed bits remain in the
    /// caller's canonical matrix; only compact compensation metadata is cached.
    ///
    /// # Errors
    ///
    /// Returns [`PairwiseError`] for invalid panel width, malformed metadata, or
    /// scratch allocation failure. Failure leaves no prepared panel active.
    pub fn prepare_panel<A: PairwiseSIMDSchema>(
        &self,
        arch: A,
        targets: MatrixView<'_, u8>,
        scratch: &mut Pairwise1BitScratch,
    ) -> Result<(), PairwiseError> {
        if targets.ncols() != self.row_bytes {
            return Err(PairwiseError::InvalidTargetWidth {
                expected: self.row_bytes,
                actual: targets.ncols(),
            });
        }
        scratch.prepare(targets.nrows())?;
        for (row, (encoded, metadata)) in targets
            .row_iter()
            .zip(scratch.metadata.iter_mut())
            .enumerate()
        {
            *metadata = decode_row(arch, encoded, self.encoded_dim, "target", row)?;
        }
        A::prepare_panel(arch, targets, self.encoded_dim, scratch);
        scratch.prepared_rows = targets.nrows();
        Ok(())
    }

    /// Score one source against rows whose metadata is already prepared.
    ///
    /// `targets` is the canonical row range beginning at `first_target` in the
    /// panel passed to [`Self::prepare_panel`]. `scores.len()` must equal
    /// `targets.nrows()`.
    ///
    /// # Errors
    ///
    /// Returns [`PairwiseError`] for shape mismatch, an unprepared target range,
    /// malformed source metadata, or any non-finite output score.
    pub fn score_prepared_panel<A: PairwiseSIMDSchema>(
        &self,
        arch: A,
        source: &[u8],
        first_target: usize,
        targets: MatrixView<'_, u8>,
        scores: &mut [f32],
        scratch: &mut Pairwise1BitScratch,
    ) -> Result<(), PairwiseError> {
        score_panel_for_metric(
            arch,
            self.metric,
            PanelCall {
                source,
                source_metadata: None,
                targets,
                first_target,
                scores,
                scratch,
                encoded_dim: self.encoded_dim,
                row_bytes: self.row_bytes,
                parameters: self.parameters,
            },
        )
    }

    /// Score a source whose metadata is part of the prepared panel.
    ///
    /// Leaf self-joins use this entry to reuse metadata decoded by
    /// [`Self::prepare_panel`]. `source_index` identifies the source row in that
    /// prepared panel; target rows may be any prepared subrange.
    ///
    /// # Errors
    ///
    /// Returns [`PairwiseError`] for an invalid source index or target range.
    pub fn score_prepared_panel_from_prepared_source<A: PairwiseSIMDSchema>(
        &self,
        arch: A,
        source: &[u8],
        source_index: usize,
        first_target: usize,
        targets: MatrixView<'_, u8>,
        scores: &mut [f32],
        scratch: &mut Pairwise1BitScratch,
    ) -> Result<(), PairwiseError> {
        let source_metadata = scratch.metadata.get(source_index).copied().ok_or(
            PairwiseError::InvalidPreparedPanel {
                prepared: scratch.prepared_rows,
                required: source_index.saturating_add(1),
            },
        )?;
        score_panel_for_metric(
            arch,
            self.metric,
            PanelCall {
                source,
                source_metadata: Some(source_metadata),
                targets,
                first_target,
                scores,
                scratch,
                encoded_dim: self.encoded_dim,
                row_bytes: self.row_bytes,
                parameters: self.parameters,
            },
        )
    }

    /// Score one encoded source against every row in a target panel.
    ///
    /// This one-shot convenience method prepares target metadata, then delegates
    /// to [`Self::score_prepared_panel`]. Repeated-source callers should prepare
    /// once explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`PairwiseError`] for shape mismatch, malformed metadata,
    /// scratch allocation failure, or any non-finite output score.
    pub fn score_panel<A: PairwiseSIMDSchema>(
        &self,
        arch: A,
        source: &[u8],
        targets: MatrixView<'_, u8>,
        scores: &mut [f32],
        scratch: &mut Pairwise1BitScratch,
    ) -> Result<(), PairwiseError> {
        self.prepare_panel(arch, targets, scratch)?;
        self.score_prepared_panel(arch, source, 0, targets, scores, scratch)
    }
}

trait PairwiseMetric: Send + Sync + 'static {
    const IS_L2: bool;

    fn distance_from_raw(
        raw_inner_product: u32,
        source: DataMetaF32,
        target: DataMetaF32,
        dim: f32,
        parameters: MetricParameters,
    ) -> f32;
}

struct SquaredL2;
struct InnerProduct;
struct Cosine;

impl PairwiseMetric for SquaredL2 {
    const IS_L2: bool = true;

    #[inline(always)]
    fn distance_from_raw(
        raw_inner_product: u32,
        source: DataMetaF32,
        target: DataMetaF32,
        dim: f32,
        _: MetricParameters,
    ) -> f32 {
        let corrected =
            vectors::corrected_component_from_raw::<1>(raw_inner_product, source, target, dim);
        vectors::squared_l2_from_corrected(source, target, corrected)
    }
}

impl PairwiseMetric for InnerProduct {
    const IS_L2: bool = false;

    #[inline(always)]
    fn distance_from_raw(
        raw_inner_product: u32,
        source: DataMetaF32,
        target: DataMetaF32,
        dim: f32,
        parameters: MetricParameters,
    ) -> f32 {
        let corrected =
            vectors::corrected_component_from_raw::<1>(raw_inner_product, source, target, dim);
        -vectors::inner_product_from_corrected(
            source,
            target,
            corrected,
            parameters.squared_shift_norm,
        )
    }
}

impl PairwiseMetric for Cosine {
    const IS_L2: bool = false;

    #[inline(always)]
    fn distance_from_raw(
        raw_inner_product: u32,
        source: DataMetaF32,
        target: DataMetaF32,
        dim: f32,
        parameters: MetricParameters,
    ) -> f32 {
        let corrected =
            vectors::corrected_component_from_raw::<1>(raw_inner_product, source, target, dim);
        1.0 - vectors::inner_product_from_corrected(
            source,
            target,
            corrected,
            parameters.squared_shift_norm,
        )
    }
}

fn score_panel_for_metric<A: PairwiseSIMDSchema>(
    arch: A,
    metric: SupportedMetric,
    call: PanelCall<'_>,
) -> Result<(), PairwiseError> {
    match metric {
        SupportedMetric::SquaredL2 => score_panel_for::<A, SquaredL2>(arch, call),
        SupportedMetric::InnerProduct => score_panel_for::<A, InnerProduct>(arch, call),
        SupportedMetric::Cosine => score_panel_for::<A, Cosine>(arch, call),
    }
}

fn score_panel_for<A, M>(arch: A, call: PanelCall<'_>) -> Result<(), PairwiseError>
where
    A: PairwiseSIMDSchema,
    M: PairwiseMetric,
{
    validate_row(call.source, call.row_bytes, "source")?;
    if call.targets.ncols() != call.row_bytes {
        return Err(PairwiseError::InvalidTargetWidth {
            expected: call.row_bytes,
            actual: call.targets.ncols(),
        });
    }
    if call.scores.len() != call.targets.nrows() {
        return Err(PairwiseError::InvalidScoreCount {
            expected: call.targets.nrows(),
            actual: call.scores.len(),
        });
    }

    let required = call.first_target.saturating_add(call.targets.nrows());
    if required > call.scratch.prepared_rows {
        return Err(PairwiseError::InvalidPreparedPanel {
            prepared: call.scratch.prepared_rows,
            required,
        });
    }

    let source = match call.source_metadata {
        Some(metadata) => metadata,
        None => decode_row(arch, call.source, call.encoded_dim, "source", 0)?,
    };
    let Pairwise1BitScratch {
        metadata,
        raw_inner_products,
        source_words,
        transposed_words,
        target_corrections,
        target_metric_terms,
        target_bit_sums,
        transposed_stride,
        transposed_word_count,
        ..
    } = &mut *call.scratch;
    if M::IS_L2
        && A::score_l2_prepared_panel(
            arch,
            call.source,
            source,
            call.encoded_dim,
            call.scores,
            source_words,
            transposed_words,
            *transposed_stride,
            *transposed_word_count,
            call.first_target,
            target_corrections,
            target_metric_terms,
            target_bit_sums,
        )
    {
        return Ok(());
    }
    A::raw_prepared_panel(
        arch,
        call.source,
        call.targets,
        call.encoded_dim,
        &mut raw_inner_products[..call.targets.nrows()],
        source_words,
        transposed_words,
        *transposed_stride,
        *transposed_word_count,
        call.first_target,
    );
    for ((score, &raw), &metadata) in call
        .scores
        .iter_mut()
        .zip(&raw_inner_products[..call.targets.nrows()])
        .zip(&metadata[call.first_target..required])
    {
        let value = M::distance_from_raw(
            raw,
            source,
            metadata,
            call.encoded_dim as f32,
            call.parameters,
        );
        // Finite f16 metadata, raw <= encoded_dim <= u32::MAX, and a
        // finite shift norm bound every intermediate far below f32 overflow.
        // Keep the assertion as an executable check of that validation proof
        // without paying a per-pair classification branch in release builds.
        debug_assert!(value.is_finite());
        *score = value;
    }
    Ok(())
}

fn validate_row(row: &[u8], expected: usize, role: &'static str) -> Result<(), PairwiseError> {
    if row.len() == expected {
        Ok(())
    } else {
        Err(PairwiseError::InvalidRowLength {
            role,
            expected,
            actual: row.len(),
        })
    }
}

fn decode_row<A: Architecture>(
    arch: A,
    row: &[u8],
    encoded_dim: usize,
    role: &'static str,
    index: usize,
) -> Result<DataMetaF32, PairwiseError> {
    let data = DataRef::<1>::from_canonical_back(row, encoded_dim).map_err(|_| {
        PairwiseError::InvalidRowLength {
            role,
            expected: DataRef::<1>::canonical_bytes(encoded_dim),
            actual: row.len(),
        }
    })?;
    let metadata = data.meta().to_full(arch);
    validate_metadata(metadata, encoded_dim, role, index)?;
    Ok(metadata)
}

fn validate_metadata(
    metadata: DataMetaF32,
    encoded_dim: usize,
    role: &'static str,
    row: usize,
) -> Result<(), PairwiseError> {
    if metadata.inner_product_correction.is_finite()
        && metadata.metric_specific.is_finite()
        && metadata.bit_sum.is_finite()
        && metadata.bit_sum >= 0.0
        && metadata.bit_sum <= encoded_dim as f32
    {
        Ok(())
    } else {
        Err(PairwiseError::InvalidMetadata { role, row })
    }
}

/// Static one-bit kernel schema for an architecture selected by the caller.
#[doc(hidden)]
pub trait PairwiseSIMDSchema: Architecture {
    fn prepare_panel(
        _arch: Self,
        _targets: MatrixView<'_, u8>,
        _encoded_dim: usize,
        _scratch: &mut Pairwise1BitScratch,
    ) {
    }

    fn raw_panel(
        arch: Self,
        source: &[u8],
        targets: MatrixView<'_, u8>,
        encoded_dim: usize,
        output: &mut [u32],
    );

    #[allow(clippy::too_many_arguments)]
    fn score_l2_prepared_panel(
        _arch: Self,
        _source: &[u8],
        _source_metadata: DataMetaF32,
        _encoded_dim: usize,
        _scores: &mut [f32],
        _source_words: &mut [u64],
        _transposed: &[u64],
        _stride: usize,
        _words: usize,
        _first_target: usize,
        _target_corrections: &[f32],
        _target_metric_terms: &[f32],
        _target_bit_sums: &[f32],
    ) -> bool {
        false
    }

    fn raw_prepared_panel(
        arch: Self,
        source: &[u8],
        targets: MatrixView<'_, u8>,
        encoded_dim: usize,
        output: &mut [u32],
        _source_words: &mut [u64],
        _transposed: &[u64],
        _stride: usize,
        _words: usize,
        _first_target: usize,
    ) {
        Self::raw_panel(arch, source, targets, encoded_dim, output);
    }
}

impl PairwiseSIMDSchema for Scalar {
    #[inline(always)]
    fn raw_panel(
        _: Self,
        source: &[u8],
        targets: MatrixView<'_, u8>,
        encoded_dim: usize,
        output: &mut [u32],
    ) {
        raw_panel_unrolled::<1>(source, targets, encoded_dim, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl PairwiseSIMDSchema for V3 {
    #[inline(always)]
    fn raw_panel(
        arch: Self,
        source: &[u8],
        targets: MatrixView<'_, u8>,
        encoded_dim: usize,
        output: &mut [u32],
    ) {
        #[cfg(miri)]
        raw_panel_unrolled::<4>(source, targets, encoded_dim, output);

        #[cfg(not(miri))]
        arch.run(move || raw_panel_v3(source, targets, encoded_dim, output));
    }
}

#[cfg(target_arch = "x86_64")]
impl PairwiseSIMDSchema for V4 {
    fn prepare_panel(
        _: Self,
        targets: MatrixView<'_, u8>,
        encoded_dim: usize,
        scratch: &mut Pairwise1BitScratch,
    ) {
        prepare_panel_v4(targets, encoded_dim, scratch);
    }

    #[inline(always)]
    fn raw_panel(
        arch: Self,
        source: &[u8],
        targets: MatrixView<'_, u8>,
        encoded_dim: usize,
        output: &mut [u32],
    ) {
        #[cfg(miri)]
        raw_panel_unrolled::<8>(source, targets, encoded_dim, output);

        #[cfg(not(miri))]
        arch.run(move || raw_panel_v4(arch, source, targets, encoded_dim, output));
    }

    #[cfg(not(miri))]
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn score_l2_prepared_panel(
        arch: Self,
        source: &[u8],
        source_metadata: DataMetaF32,
        encoded_dim: usize,
        scores: &mut [f32],
        source_words: &mut [u64],
        transposed: &[u64],
        stride: usize,
        words: usize,
        first_target: usize,
        target_corrections: &[f32],
        target_metric_terms: &[f32],
        target_bit_sums: &[f32],
    ) -> bool {
        arch.run(move || {
            score_panel_v4_l2(
                arch,
                source,
                source_metadata,
                encoded_dim,
                scores,
                source_words,
                transposed,
                stride,
                words,
                first_target,
                target_corrections,
                target_metric_terms,
                target_bit_sums,
            );
        });
        true
    }

    #[cfg(not(miri))]
    #[inline(always)]
    fn raw_prepared_panel(
        arch: Self,
        source: &[u8],
        _targets: MatrixView<'_, u8>,
        encoded_dim: usize,
        output: &mut [u32],
        source_words: &mut [u64],
        transposed: &[u64],
        stride: usize,
        words: usize,
        first_target: usize,
    ) {
        arch.run(move || {
            raw_panel_v4_transposed(
                arch,
                source,
                encoded_dim,
                output,
                source_words,
                transposed,
                stride,
                words,
                first_target,
            );
        });
    }
}

#[cfg(target_arch = "aarch64")]
impl PairwiseSIMDSchema for Neon {
    #[inline(always)]
    fn raw_panel(
        _: Self,
        source: &[u8],
        targets: MatrixView<'_, u8>,
        encoded_dim: usize,
        output: &mut [u32],
    ) {
        // Initial AArch64 path keeps four independent scalar accumulators. A
        // NEON CNT implementation should replace this only with benchmark proof.
        raw_panel_unrolled::<4>(source, targets, encoded_dim, output);
    }
}

#[cfg(target_arch = "x86_64")]
fn prepare_panel_v4(
    targets: MatrixView<'_, u8>,
    encoded_dim: usize,
    scratch: &mut Pairwise1BitScratch,
) {
    // A SIMD load can start at the final real target. Three extra panels keep
    // each four-panel tile inside the prepared slabs.
    scratch.transposed_stride = targets.nrows().saturating_add(31).next_multiple_of(32);
    scratch.transposed_word_count = encoded_dim.div_ceil(64);
    scratch
        .source_words
        .resize(scratch.transposed_word_count, 0);
    scratch
        .transposed_words
        .resize(scratch.transposed_stride * scratch.transposed_word_count, 0);
    scratch.transposed_words.fill(0);
    scratch
        .target_corrections
        .resize(scratch.transposed_stride, 0.0);
    scratch
        .target_metric_terms
        .resize(scratch.transposed_stride, 0.0);
    scratch
        .target_bit_sums
        .resize(scratch.transposed_stride, 0.0);
    scratch.target_corrections.fill(0.0);
    scratch.target_metric_terms.fill(0.0);
    scratch.target_bit_sums.fill(0.0);
    for (target, metadata) in scratch.metadata.iter().copied().enumerate() {
        scratch.target_corrections[target] = metadata.inner_product_correction;
        scratch.target_metric_terms[target] = metadata.metric_specific;
        scratch.target_bit_sums[target] = metadata.bit_sum;
    }

    let bit_bytes = encoded_dim.div_ceil(8);
    let tail = encoded_dim % 64;
    let tail_mask = if tail == 0 {
        u64::MAX
    } else {
        (1u64 << tail) - 1
    };
    for (target, row) in targets.row_iter().enumerate() {
        for word in 0..scratch.transposed_word_count {
            let offset = word * 8;
            let remaining = bit_bytes.saturating_sub(offset).min(8);
            let mut value = if remaining == 8 {
                // SAFETY: This word is inside the validated packed-bit prefix.
                unsafe { std::ptr::read_unaligned(row.as_ptr().add(offset).cast::<u64>()) }
            } else {
                read_partial_word(&row[offset..offset + remaining])
            };
            if word + 1 == scratch.transposed_word_count {
                value &= tail_mask;
            }
            scratch.transposed_words[word * scratch.transposed_stride + target] = value;
        }
    }
}

#[inline(always)]
fn raw_panel_unrolled<const PANEL: usize>(
    source: &[u8],
    targets: MatrixView<'_, u8>,
    encoded_dim: usize,
    output: &mut [u32],
) {
    debug_assert_eq!(targets.nrows(), output.len());
    let bit_bytes = encoded_dim.div_ceil(8);
    let source = &source[..bit_bytes];
    for (panel, output_panel) in output.chunks_mut(PANEL).enumerate() {
        let first = panel * PANEL;
        let count = output_panel.len();
        let mut accumulators = [0u32; PANEL];
        let full_words = encoded_dim / 64;
        for word in 0..full_words {
            let offset = word * 8;
            // SAFETY: a complete 64-bit word lies within `bit_bytes`; unaligned
            // reads are required because canonical rows have byte alignment.
            let source_word =
                unsafe { std::ptr::read_unaligned(source.as_ptr().add(offset).cast::<u64>()) };
            for (lane, accumulator) in accumulators[..count].iter_mut().enumerate() {
                let target = targets.row(first + lane);
                // SAFETY: target rows have the same validated canonical width,
                // and this word is inside their packed-bit prefix.
                let target_word =
                    unsafe { std::ptr::read_unaligned(target.as_ptr().add(offset).cast::<u64>()) };
                *accumulator += (source_word & target_word).count_ones();
            }
        }
        let consumed_bits = full_words * 64;
        if consumed_bits < encoded_dim {
            let offset = consumed_bits / 8;
            for (lane, accumulator) in accumulators[..count].iter_mut().enumerate() {
                *accumulator += raw_tail(
                    &source[offset..],
                    &targets.row(first + lane)[offset..bit_bytes],
                    encoded_dim - consumed_bits,
                );
            }
        }
        output_panel.copy_from_slice(&accumulators[..count]);
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline(always)]
fn raw_panel_v3(
    source: &[u8],
    targets: MatrixView<'_, u8>,
    encoded_dim: usize,
    output: &mut [u32],
) {
    debug_assert_eq!(targets.nrows(), output.len());
    let bit_bytes = encoded_dim.div_ceil(8);
    let source = &source[..bit_bytes];
    let full_words = encoded_dim / 64;
    let full_panels = output.len() / 4;

    for panel in 0..full_panels {
        let first = panel * 4;
        let target0 = targets.row(first);
        let target1 = targets.row(first + 1);
        let target2 = targets.row(first + 2);
        let target3 = targets.row(first + 3);
        let mut accumulator0 = 0u32;
        let mut accumulator1 = 0u32;
        let mut accumulator2 = 0u32;
        let mut accumulator3 = 0u32;

        for word in 0..full_words {
            let offset = word * 8;
            // SAFETY: every pointer names one complete 64-bit word inside a
            // validated canonical row. Unaligned access is required by layout.
            let source_word =
                unsafe { std::ptr::read_unaligned(source.as_ptr().add(offset).cast::<u64>()) };
            // SAFETY: same complete-word and canonical-row preconditions as source.
            let word0 =
                unsafe { std::ptr::read_unaligned(target0.as_ptr().add(offset).cast::<u64>()) };
            // SAFETY: same complete-word and canonical-row preconditions as source.
            let word1 =
                unsafe { std::ptr::read_unaligned(target1.as_ptr().add(offset).cast::<u64>()) };
            // SAFETY: same complete-word and canonical-row preconditions as source.
            let word2 =
                unsafe { std::ptr::read_unaligned(target2.as_ptr().add(offset).cast::<u64>()) };
            // SAFETY: same complete-word and canonical-row preconditions as source.
            let word3 =
                unsafe { std::ptr::read_unaligned(target3.as_ptr().add(offset).cast::<u64>()) };
            accumulator0 += (source_word & word0).count_ones();
            accumulator1 += (source_word & word1).count_ones();
            accumulator2 += (source_word & word2).count_ones();
            accumulator3 += (source_word & word3).count_ones();
        }

        let consumed_bits = full_words * 64;
        if consumed_bits < encoded_dim {
            let offset = consumed_bits / 8;
            let remaining = encoded_dim - consumed_bits;
            accumulator0 += raw_tail(&source[offset..], &target0[offset..bit_bytes], remaining);
            accumulator1 += raw_tail(&source[offset..], &target1[offset..bit_bytes], remaining);
            accumulator2 += raw_tail(&source[offset..], &target2[offset..bit_bytes], remaining);
            accumulator3 += raw_tail(&source[offset..], &target3[offset..bit_bytes], remaining);
        }
        output[first..first + 4].copy_from_slice(&[
            accumulator0,
            accumulator1,
            accumulator2,
            accumulator3,
        ]);
    }

    for (target, result) in (full_panels * 4..targets.nrows()).zip(&mut output[full_panels * 4..]) {
        *result = raw_pair_v3(source, targets.row(target), encoded_dim);
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline(always)]
fn raw_pair_v3(source: &[u8], target: &[u8], encoded_dim: usize) -> u32 {
    let bit_bytes = encoded_dim.div_ceil(8);
    let full_words = encoded_dim / 64;
    let mut result = 0u32;
    for word in 0..full_words {
        let offset = word * 8;
        // SAFETY: both pointers name complete words inside canonical bit prefixes.
        let source_word =
            unsafe { std::ptr::read_unaligned(source.as_ptr().add(offset).cast::<u64>()) };
        // SAFETY: same complete-word precondition as source.
        let target_word =
            unsafe { std::ptr::read_unaligned(target.as_ptr().add(offset).cast::<u64>()) };
        result += (source_word & target_word).count_ones();
    }
    let consumed_bits = full_words * 64;
    if consumed_bits < encoded_dim {
        let offset = consumed_bits / 8;
        result += raw_tail(
            &source[offset..bit_bytes],
            &target[offset..bit_bytes],
            encoded_dim - consumed_bits,
        );
    }
    result
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[allow(clippy::too_many_arguments)]
fn score_panel_v4_l2<A>(
    arch: A,
    source: &[u8],
    source_metadata: DataMetaF32,
    encoded_dim: usize,
    scores: &mut [f32],
    source_words: &mut [u64],
    transposed: &[u64],
    stride: usize,
    words: usize,
    first_target: usize,
    target_corrections: &[f32],
    target_metric_terms: &[f32],
    target_bit_sums: &[f32],
) where
    A: Architecture,
    A::u64x8: SIMDPopcount + SIMDCast<f32, Cast = A::f32x8>,
{
    pack_source_words(source, encoded_dim, &mut source_words[..words]);
    let source_correction = A::f32x8::splat(arch, source_metadata.inner_product_correction);
    let source_metric = A::f32x8::splat(arch, source_metadata.metric_specific);
    let source_bit_sum = A::f32x8::splat(arch, source_metadata.bit_sum);
    let half = A::f32x8::splat(arch, 0.5);
    let quarter_dimension = A::f32x8::splat(arch, encoded_dim as f32 * 0.25);
    let two = A::f32x8::splat(arch, 2.0);

    macro_rules! write_scores {
        ($counts:expr, $first:expr, $output:expr) => {{
            let raw = $counts.cast::<f32>();
            // SAFETY: The prepared metadata slabs include 31 padded targets.
            let target_correction =
                unsafe { A::f32x8::load_simd(arch, target_corrections.as_ptr().add($first)) };
            // SAFETY: The same padded-slab invariant applies to this load.
            let target_metric =
                unsafe { A::f32x8::load_simd(arch, target_metric_terms.as_ptr().add($first)) };
            // SAFETY: The same padded-slab invariant applies to this load.
            let target_bit_sum =
                unsafe { A::f32x8::load_simd(arch, target_bit_sums.as_ptr().add($first)) };
            let centered = raw - half * (source_bit_sum + target_bit_sum) + quarter_dimension;
            let corrected = source_correction * target_correction * centered;
            let distances = source_metric + target_metric - two * corrected;
            let output: &mut [f32] = $output;
            // SAFETY: `output` owns its first `output.len().min(8)` elements.
            unsafe { distances.store_simd_first(output.as_mut_ptr(), output.len()) };
        }};
    }

    let full_tiles = scores.len() / 32;
    for tile in 0..full_tiles {
        let first = first_target + tile * 32;
        let mut count0 = A::u64x8::default(arch);
        let mut count1 = A::u64x8::default(arch);
        let mut count2 = A::u64x8::default(arch);
        let mut count3 = A::u64x8::default(arch);
        for (word, &source_word) in source_words[..words].iter().enumerate() {
            let source_lanes = A::u64x8::splat(arch, source_word);
            // SAFETY: Four complete panels stay inside the padded transposed slab.
            let targets = unsafe { transposed.as_ptr().add(word * stride + first) };
            // SAFETY: Each pointer names eight contiguous prepared target words.
            count0 = count0
                + (source_lanes & unsafe { A::u64x8::load_simd(arch, targets) }).popcount_simd();
            count1 = count1
                + (source_lanes & unsafe { A::u64x8::load_simd(arch, targets.add(8)) })
                    .popcount_simd();
            count2 = count2
                + (source_lanes & unsafe { A::u64x8::load_simd(arch, targets.add(16)) })
                    .popcount_simd();
            count3 = count3
                + (source_lanes & unsafe { A::u64x8::load_simd(arch, targets.add(24)) })
                    .popcount_simd();
        }
        let output = &mut scores[tile * 32..tile * 32 + 32];
        write_scores!(count0, first, &mut output[..8]);
        write_scores!(count1, first + 8, &mut output[8..16]);
        write_scores!(count2, first + 16, &mut output[16..24]);
        write_scores!(count3, first + 24, &mut output[24..]);
    }

    let consumed = full_tiles * 32;
    for (panel, output) in scores[consumed..].chunks_mut(8).enumerate() {
        let first = first_target + consumed + panel * 8;
        let mut counts = A::u64x8::default(arch);
        for (word, &source_word) in source_words[..words].iter().enumerate() {
            let source_lanes = A::u64x8::splat(arch, source_word);
            // SAFETY: This complete panel stays inside the padded transposed slab.
            let targets = unsafe {
                A::u64x8::load_simd(arch, transposed.as_ptr().add(word * stride + first))
            };
            counts = counts + (source_lanes & targets).popcount_simd();
        }
        write_scores!(counts, first, output);
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[allow(clippy::too_many_arguments)]
fn raw_panel_v4_transposed<A>(
    arch: A,
    source: &[u8],
    encoded_dim: usize,
    output: &mut [u32],
    source_words: &mut [u64],
    transposed: &[u64],
    stride: usize,
    words: usize,
    first_target: usize,
) where
    A: Architecture,
    A::u64x8: SIMDPopcount,
{
    pack_source_words(source, encoded_dim, &mut source_words[..words]);
    for (panel, output_panel) in output.chunks_mut(8).enumerate() {
        let first = first_target + panel * 8;
        let mut counts = A::u64x8::default(arch);
        for (word, &source_word) in source_words[..words].iter().enumerate() {
            let source_lanes = A::u64x8::splat(arch, source_word);
            // SAFETY: This panel stays inside the padded transposed slab.
            let targets = unsafe {
                A::u64x8::load_simd(arch, transposed.as_ptr().add(word * stride + first))
            };
            counts = counts + (source_lanes & targets).popcount_simd();
        }
        let mut lanes = [0u64; 8];
        // SAFETY: `lanes` contains exactly eight writable values.
        unsafe { counts.store_simd(lanes.as_mut_ptr()) };
        for (destination, count) in output_panel.iter_mut().zip(lanes) {
            *destination = count as u32;
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline(always)]
fn raw_panel_v4<A>(
    arch: A,
    source: &[u8],
    targets: MatrixView<'_, u8>,
    encoded_dim: usize,
    output: &mut [u32],
) where
    A: Architecture,
    A::u64x8: SIMDPopcount,
{
    debug_assert_eq!(targets.nrows(), output.len());
    let bit_bytes = encoded_dim.div_ceil(8);
    let source = &source[..bit_bytes];
    let full_blocks = encoded_dim / 512;
    let block_bits = full_blocks * 512;
    let remaining_words = (encoded_dim - block_bits) / 64;
    let consumed_bits = block_bits + remaining_words * 64;

    for (panel, output_panel) in output.chunks_mut(8).enumerate() {
        let first = panel * 8;
        let mut accumulators = [0u32; 8];
        for block in 0..full_blocks {
            let offset = block * 64;
            // SAFETY: This complete block is inside the packed-bit prefix.
            let source_block =
                unsafe { A::u64x8::load_simd(arch, source.as_ptr().add(offset).cast::<u64>()) };
            for (lane, accumulator) in accumulators[..output_panel.len()].iter_mut().enumerate() {
                let target = targets.row(first + lane);
                // SAFETY: Target rows have the same validated packed width.
                let target_block =
                    unsafe { A::u64x8::load_simd(arch, target.as_ptr().add(offset).cast::<u64>()) };
                *accumulator += (source_block & target_block).popcount_simd().sum_tree() as u32;
            }
        }
        if remaining_words != 0 {
            let offset = block_bits / 8;
            // SAFETY: Only complete words inside the packed prefix are enabled.
            let source_block = unsafe {
                A::u64x8::load_simd_first(
                    arch,
                    source.as_ptr().add(offset).cast::<u64>(),
                    remaining_words,
                )
            };
            for (lane, accumulator) in accumulators[..output_panel.len()].iter_mut().enumerate() {
                let target = targets.row(first + lane);
                // SAFETY: Target rows share the source packed width.
                let target_block = unsafe {
                    A::u64x8::load_simd_first(
                        arch,
                        target.as_ptr().add(offset).cast::<u64>(),
                        remaining_words,
                    )
                };
                *accumulator += (source_block & target_block).popcount_simd().sum_tree() as u32;
            }
        }
        if consumed_bits < encoded_dim {
            let offset = consumed_bits / 8;
            for (lane, accumulator) in accumulators[..output_panel.len()].iter_mut().enumerate() {
                *accumulator += raw_tail(
                    &source[offset..],
                    &targets.row(first + lane)[offset..bit_bytes],
                    encoded_dim - consumed_bits,
                );
            }
        }
        output_panel.copy_from_slice(&accumulators[..output_panel.len()]);
    }
}

#[inline(always)]
fn read_partial_word(bytes: &[u8]) -> u64 {
    bytes.iter().enumerate().fold(0, |word, (byte, &value)| {
        word | (u64::from(value) << (byte * 8))
    })
}

#[inline(always)]
fn pack_source_words(source: &[u8], encoded_dim: usize, output: &mut [u64]) {
    let bit_bytes = encoded_dim.div_ceil(8);
    debug_assert_eq!(output.len(), encoded_dim.div_ceil(64));
    for (word, destination) in output.iter_mut().enumerate() {
        let offset = word * 8;
        let remaining = bit_bytes.saturating_sub(offset).min(8);
        *destination = if remaining == 8 {
            // SAFETY: This word is inside the validated packed-bit prefix.
            unsafe { std::ptr::read_unaligned(source.as_ptr().add(offset).cast::<u64>()) }
        } else {
            read_partial_word(&source[offset..offset + remaining])
        };
    }
    if let (Some(last), tail @ 1..=63) = (output.last_mut(), encoded_dim % 64) {
        *last &= (1u64 << tail) - 1;
    }
}

#[inline(always)]
fn raw_tail(source: &[u8], target: &[u8], dimensions: usize) -> u32 {
    let full_bytes = dimensions / 8;
    let mut result = source[..full_bytes]
        .iter()
        .zip(&target[..full_bytes])
        .map(|(&left, &right)| (left & right).count_ones())
        .sum();
    let remaining = dimensions % 8;
    if remaining != 0 {
        let mask = (1u8 << remaining) - 1;
        result += (source[full_bytes] & target[full_bytes] & mask).count_ones();
    }
    result
}

fn resize<T: Clone>(
    buffer: &'static str,
    values: &mut Vec<T>,
    len: usize,
    value: T,
) -> Result<(), PairwiseError> {
    let additional = len.saturating_sub(values.len());
    values
        .try_reserve(additional)
        .map_err(|_| PairwiseError::Allocation { buffer, additional })?;
    values.resize(len, value);
    Ok(())
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Pairwise1Bit>();
};

#[cfg(test)]
mod tests {
    use diskann_utils::views::{Matrix, MatrixView};
    use diskann_wide::arch::Scalar;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{
        AsFunctor, CompressIntoWith,
        algorithms::{TransformKind, transforms::TargetDim},
        alloc::{GlobalAllocator, ScopedAllocator},
        distances,
        spherical::{
            CompensatedCosine, CompensatedIP, CompensatedSquaredL2, DataMut, DataRef, PreScale,
        },
    };

    use super::*;

    fn data(dim: usize, rows: usize) -> Matrix<f32> {
        let mut rng = StdRng::seed_from_u64(0x5241_4249_5451_1000 ^ dim as u64);
        let mut values = Matrix::new(0.0, rows, dim);
        for (row_index, row) in values.row_iter_mut().enumerate() {
            for (column, value) in row.iter_mut().enumerate() {
                *value = rng.random_range(-2.0f32..2.0)
                    + (row_index as f32 + 1.0) * 0.03125
                    + column as f32 * 0.000_976_562_5;
            }
        }
        values
    }

    fn train_and_encode(
        metric: SupportedMetric,
        values: MatrixView<'_, f32>,
    ) -> (SphericalQuantizer, Matrix<u8>) {
        let mut rng = StdRng::seed_from_u64(0x5241_4249_5451_2000 ^ values.ncols() as u64);
        let quantizer = SphericalQuantizer::train(
            values,
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            metric,
            PreScale::ReciprocalMeanNorm,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap();
        let dim = quantizer.output_dim();
        let row_bytes = DataRef::<1>::canonical_bytes(dim);
        let mut encoded = Matrix::new(0u8, values.nrows(), row_bytes);
        for (source, target) in values.row_iter().zip(encoded.row_iter_mut()) {
            let target = DataMut::<1>::from_canonical_back_mut(target, dim).unwrap();
            quantizer
                .compress_into_with(source, target, ScopedAllocator::global())
                .unwrap();
        }
        (quantizer, encoded)
    }

    fn oracle(quantizer: &SphericalQuantizer, left: &[u8], right: &[u8]) -> f32 {
        let dim = quantizer.output_dim();
        let left = DataRef::<1>::from_canonical_back(left, dim).unwrap();
        let right = DataRef::<1>::from_canonical_back(right, dim).unwrap();
        match quantizer.metric() {
            SupportedMetric::SquaredL2 => {
                let distance: CompensatedSquaredL2 = quantizer.as_functor();
                let result: distances::Result<f32> =
                    diskann_wide::arch::dispatch2(distance, left, right);
                result.unwrap()
            }
            SupportedMetric::InnerProduct => {
                let distance: CompensatedIP = quantizer.as_functor();
                let result: distances::Result<f32> =
                    diskann_wide::arch::dispatch2(distance, left, right);
                result.unwrap()
            }
            SupportedMetric::Cosine => {
                let distance: CompensatedCosine = quantizer.as_functor();
                let result: distances::Result<f32> =
                    diskann_wide::arch::dispatch2(distance, left, right);
                result.unwrap()
            }
        }
    }

    fn assert_close(actual: f32, expected: f32, context: &str) {
        let tolerance = 2.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }

    fn scorer_for_arch<A: PairwiseSIMDSchema>(
        quantizer: &SphericalQuantizer,
        _: A,
    ) -> Pairwise1Bit {
        Pairwise1Bit::new(quantizer).unwrap()
    }

    #[test]
    fn pair_and_panel_match_existing_data_distance_at_dimension_boundaries() {
        let dimensions = [1, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1_536];
        let metrics = [
            SupportedMetric::SquaredL2,
            SupportedMetric::InnerProduct,
            SupportedMetric::Cosine,
        ];
        for metric in metrics {
            for dim in dimensions {
                let values = data(dim, 11);
                let (quantizer, encoded) = train_and_encode(metric, values.as_view());
                let scorer = Pairwise1Bit::new(&quantizer).unwrap();
                let mut scratch = Pairwise1BitScratch::new();
                let mut scores = vec![0.0; encoded.nrows()];
                scorer
                    .score_panel(
                        Scalar::new(),
                        encoded.row(7),
                        encoded.as_view(),
                        &mut scores,
                        &mut scratch,
                    )
                    .unwrap();
                for (target, &score) in scores.iter().enumerate() {
                    let expected = oracle(&quantizer, encoded.row(7), encoded.row(target));
                    assert_close(
                        score,
                        expected,
                        &format!("metric={metric:?}, dim={dim}, target={target}"),
                    );
                    assert_close(
                        scorer
                            .score_pair(Scalar::new(), encoded.row(7), encoded.row(target))
                            .unwrap(),
                        expected,
                        &format!("pair metric={metric:?}, dim={dim}, target={target}"),
                    );
                }
            }
        }
    }

    #[test]
    fn panel_widths_cross_micro_panel_boundaries() {
        let values = data(129, 17);
        let (quantizer, encoded) = train_and_encode(SupportedMetric::SquaredL2, values.as_view());
        let scorer = Pairwise1Bit::new(&quantizer).unwrap();
        let mut scratch = Pairwise1BitScratch::new();
        for width in [0, 1, 3, 4, 5, 7, 8, 9, 16, 17] {
            let targets = encoded.subview(0..width).unwrap();
            let mut scores = vec![0.0; width];
            scorer
                .score_panel(
                    Scalar::new(),
                    encoded.row(16),
                    targets,
                    &mut scores,
                    &mut scratch,
                )
                .unwrap();
            for (target, &score) in scores.iter().enumerate() {
                assert_close(
                    score,
                    oracle(&quantizer, encoded.row(16), encoded.row(target)),
                    &format!("width={width}, target={target}"),
                );
            }
        }
    }

    #[test]
    fn prepared_source_metadata_matches_one_shot_scoring() {
        let values = data(1536, 17);
        let (quantizer, encoded) = train_and_encode(SupportedMetric::SquaredL2, values.as_view());
        let scorer = Pairwise1Bit::new(&quantizer).unwrap();
        let targets = encoded.subview(0..13).unwrap();
        let mut scratch = Pairwise1BitScratch::new();
        scorer
            .prepare_panel(Scalar::new(), encoded.as_view(), &mut scratch)
            .unwrap();
        let mut actual = vec![0.0; targets.nrows()];
        scorer
            .score_prepared_panel_from_prepared_source(
                Scalar::new(),
                encoded.row(16),
                16,
                0,
                targets,
                &mut actual,
                &mut scratch,
            )
            .unwrap();
        for (target, &score) in actual.iter().enumerate() {
            assert_close(
                score,
                oracle(&quantizer, encoded.row(16), encoded.row(target)),
                &format!("prepared source target={target}"),
            );
        }
    }

    #[test]
    fn scalar_and_available_x86_backends_agree() {
        let values = data(513, 12);
        let (quantizer, encoded) = train_and_encode(SupportedMetric::Cosine, values.as_view());
        let scalar = scorer_for_arch(&quantizer, Scalar::new());
        let mut scalar_scratch = Pairwise1BitScratch::new();
        let mut expected = vec![0.0; encoded.nrows()];
        scalar
            .score_panel(
                Scalar::new(),
                encoded.row(9),
                encoded.as_view(),
                &mut expected,
                &mut scalar_scratch,
            )
            .unwrap();

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = V3::new_checked() {
                let scorer = scorer_for_arch(&quantizer, arch);
                let mut scratch = Pairwise1BitScratch::new();
                let mut actual = vec![0.0; encoded.nrows()];
                scorer
                    .score_panel(
                        arch,
                        encoded.row(9),
                        encoded.as_view(),
                        &mut actual,
                        &mut scratch,
                    )
                    .unwrap();
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    assert_close(actual, expected, &format!("V3 target={index}"));
                }
            }
            if let Some(arch) = V4::new_checked() {
                let scorer = scorer_for_arch(&quantizer, arch);
                let mut scratch = Pairwise1BitScratch::new();
                let mut actual = vec![0.0; encoded.nrows()];
                scorer
                    .score_panel(
                        arch,
                        encoded.row(9),
                        encoded.as_view(),
                        &mut actual,
                        &mut scratch,
                    )
                    .unwrap();
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    assert_close(actual, expected, &format!("V4 target={index}"));
                }
            }
        }
    }

    #[test]
    fn malformed_shapes_metadata_and_non_finite_scores_are_rejected() {
        let values = data(65, 4);
        let (quantizer, mut encoded) =
            train_and_encode(SupportedMetric::SquaredL2, values.as_view());
        let scorer = Pairwise1Bit::new(&quantizer).unwrap();
        let mut scratch = Pairwise1BitScratch::new();
        let mut scores = vec![0.0; encoded.nrows()];

        assert!(matches!(
            scorer.score_pair(Scalar::new(), &encoded.row(0)[1..], encoded.row(1)),
            Err(PairwiseError::InvalidRowLength { role: "source", .. })
        ));
        assert!(matches!(
            scorer.score_panel(
                Scalar::new(),
                encoded.row(0),
                MatrixView::try_from(encoded.as_slice(), 1, encoded.as_slice().len()).unwrap(),
                &mut scores,
                &mut scratch,
            ),
            Err(PairwiseError::InvalidTargetWidth { .. })
        ));
        assert!(matches!(
            scorer.score_panel(
                Scalar::new(),
                encoded.row(0),
                encoded.as_view(),
                &mut scores[..3],
                &mut scratch,
            ),
            Err(PairwiseError::InvalidScoreCount { .. })
        ));

        let row_bytes = encoded.ncols();
        let metadata_offset = row_bytes - std::mem::size_of::<super::super::DataMeta>();
        encoded.row_mut(1)[metadata_offset..metadata_offset + 2]
            .copy_from_slice(&half::f16::INFINITY.to_bits().to_ne_bytes());
        assert!(matches!(
            scorer.score_pair(Scalar::new(), encoded.row(0), encoded.row(1)),
            Err(PairwiseError::InvalidMetadata {
                role: "target",
                row: 0
            })
        ));
    }

    #[test]
    fn unused_tail_bits_do_not_change_scores() {
        let values = data(65, 3);
        let (quantizer, mut encoded) =
            train_and_encode(SupportedMetric::InnerProduct, values.as_view());
        let scorer = Pairwise1Bit::new(&quantizer).unwrap();
        let before = scorer
            .score_pair(Scalar::new(), encoded.row(0), encoded.row(1))
            .unwrap();
        // Dimension 65 uses one bit in the final byte; canonical bit distances
        // must ignore the other seven storage bits.
        encoded.row_mut(1)[8] |= 0b1111_1110;
        let after = scorer
            .score_pair(Scalar::new(), encoded.row(0), encoded.row(1))
            .unwrap();
        assert_eq!(before.to_bits(), after.to_bits());
    }

    #[test]
    fn zero_dimension_training_is_rejected() {
        let values = Matrix::<f32>::new(0.0, 2, 0);
        let mut rng = StdRng::seed_from_u64(1);
        assert!(
            SphericalQuantizer::train(
                values.as_view(),
                TransformKind::DoubleHadamard {
                    target_dim: TargetDim::Same,
                },
                SupportedMetric::SquaredL2,
                PreScale::ReciprocalMeanNorm,
                &mut rng,
                GlobalAllocator,
            )
            .is_err()
        );
    }
}
