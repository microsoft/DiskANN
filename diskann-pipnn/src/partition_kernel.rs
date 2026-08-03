/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prepared distance and top-k kernels for partition assignment.
//!
//! The caller computes a row-major `points · leadersᵀ` tile with GEMM, then
//! passes it to a [`PartitionKernel`] prepared once for the build metric. Kernel
//! preparation selects the runtime architecture and concrete metric type once;
//! repeated stripes call a direct `diskann-wide` function pointer with no ISA or
//! metric branch in the row loop.
//!
//! L2 deliberately omits the point norm because it is constant across every
//! leader in one row. Cosine consumes squared point norms and leader norms. NaN
//! distances are not rankable, and equal distances retain leader scan order.

use std::marker::PhantomData;

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::{
    arch::{self, Dispatched2, FTarget2},
    lifetime::AddLifetime,
    Architecture, SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector,
};

use crate::kernel_metric::{erase_metric, EraseMetric, KernelMetric, ScaleKind};

/// Maximum number of leaders retained for one point.
///
/// Supported PiPNN partition fanouts fit within 16. Keeping this as a fixed
/// stack tracker bounds per-row stack use and code size; larger requests are
/// rejected rather than silently truncated.
pub const MAX_PARTITION_FANOUT: usize = 16;

type TopK = [(u32, f32); MAX_PARTITION_FANOUT];

/// Metric-specific normalization inputs for one partition tile.
#[derive(Clone, Copy, Debug)]
pub enum PartitionScales<'a> {
    /// L2 needs only squared leader norms; the point norm cannot affect ranking.
    L2 {
        /// Squared norm for every leader column.
        leader_squared_norms: &'a [f32],
    },
    /// Unnormalized cosine needs squared point norms and leader norms.
    Cosine {
        /// Squared norm for every point row.
        row_squared_norms: &'a [f32],
        /// Norm for every leader column.
        leader_norms: &'a [f32],
    },
    /// Normalized cosine and inner product need no normalization inputs.
    None,
}

/// One row-major point-by-leader dot-product tile.
#[derive(Clone, Copy, Debug)]
pub struct PartitionTopK<'a> {
    /// Point rows by leader columns.
    pub dots: MatrixView<'a, f32>,
    /// Normalization inputs matching the prepared metric.
    pub scales: PartitionScales<'a>,
}

/// Validation error returned by [`PartitionKernel::nearest_leaders`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PartitionKernelError {
    /// A declared matrix shape overflowed `usize`.
    #[error("{buffer} shape {rows} x {cols} overflows usize")]
    ShapeOverflow {
        /// Name of the buffer whose shape overflowed.
        buffer: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },
    /// The output matrix does not match the input row count.
    #[error(
        "invalid output shape: expected {expected_rows} rows, got {actual_rows} rows and {actual_cols} columns"
    )]
    InvalidOutputShape {
        /// Required row count.
        expected_rows: usize,
        /// Supplied row count.
        actual_rows: usize,
        /// Supplied column count.
        actual_cols: usize,
    },
    /// A metric-specific scale slice has the wrong length.
    #[error("invalid {buffer} length: expected {expected}, got {actual}")]
    InvalidBufferLength {
        /// Name of the invalid scale buffer.
        buffer: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// Scale inputs do not match the metric used to prepare the kernel.
    #[error("partition scales do not match prepared {expected} metric")]
    InvalidScales {
        /// Expected scale layout.
        expected: &'static str,
    },
    /// The requested fanout cannot be represented by the fixed top-k tracker.
    #[error(
        "invalid fanout {fanout}: must not exceed {leaders} leaders or kernel maximum {maximum}"
    )]
    InvalidFanout {
        /// Requested number of leaders per row.
        fanout: usize,
        /// Available leader count.
        leaders: usize,
        /// Kernel maximum.
        maximum: usize,
    },
    /// Leader positions cannot be represented as `u32`.
    #[error("leader count {0} exceeds the u32 position limit")]
    TooManyLeaders(usize),
    /// A row did not contain enough rankable distances to fill its output.
    #[error("row {row} has fewer than {fanout} rankable leader distances")]
    InsufficientRankableDistances {
        /// Zero-based row position in the input tile.
        row: usize,
        /// Requested number of leader positions.
        fanout: usize,
    },
}

#[derive(Debug)]
struct PartitionInput;

impl AddLifetime for PartitionInput {
    type Of<'a> = PartitionTopK<'a>;
}

#[derive(Debug)]
struct PartitionOutput;

impl AddLifetime for PartitionOutput {
    type Of<'a> = MutMatrixView<'a, u32>;
}

type PartitionFn = Dispatched2<Result<(), PartitionKernelError>, PartitionInput, PartitionOutput>;

/// A partition kernel prepared for one metric and the current CPU.
///
/// Construct this once with [`PartitionKernel::new`] and reuse it for every
/// point stripe. The handle is a direct function pointer and is `Copy`, `Send`,
/// and `Sync`.
#[derive(Clone, Copy, Debug)]
pub struct PartitionKernel {
    run: PartitionFn,
}

impl PartitionKernel {
    /// Prepare a partition kernel for `metric` and the current CPU.
    pub fn new(metric: Metric) -> Self {
        diskann_wide::arch::dispatch1_no_features(PreparePartition, metric)
    }

    /// Select the nearest leader positions for every input row.
    ///
    /// `output.nrows()` must equal `input.dots.nrows()`; its column count is the
    /// requested fanout. Results are ordered by ascending distance. For L2, the
    /// score omits the point norm because it cannot affect within-row ranking.
    pub fn nearest_leaders(
        &self,
        input: PartitionTopK<'_>,
        output: MutMatrixView<'_, u32>,
    ) -> Result<(), PartitionKernelError> {
        self.run.call(input, output)
    }
}

struct PreparePartition;

impl<A> arch::Target1<A, PartitionKernel, Metric> for PreparePartition
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(self, arch: A, metric: Metric) -> PartitionKernel {
        erase_metric(metric, BuildPartition(arch))
    }
}

struct BuildPartition<A>(A);

impl<A> EraseMetric for BuildPartition<A>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    type Output = PartitionKernel;

    fn erase<M: KernelMetric>(self) -> Self::Output {
        PartitionKernel {
            run: self.0.dispatch2::<
                PartitionEntry<M>,
                Result<(), PartitionKernelError>,
                PartitionInput,
                PartitionOutput,
            >(),
        }
    }
}

struct PartitionEntry<M>(PhantomData<M>);

impl<A, M> FTarget2<A, Result<(), PartitionKernelError>, PartitionTopK<'_>, MutMatrixView<'_, u32>>
    for PartitionEntry<M>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
{
    fn run(
        arch: A,
        input: PartitionTopK<'_>,
        mut output: MutMatrixView<'_, u32>,
    ) -> Result<(), PartitionKernelError> {
        let scales = validate::<M>(input, &output)?;
        let fanout = output.ncols();
        if fanout == 0 || input.dots.nrows() == 0 {
            return Ok(());
        }

        process_rows::<A::f32x16, M>(arch, input.dots, scales, fanout, output.as_mut_slice());
        if let Some(row) = output
            .as_slice()
            .chunks_exact(fanout)
            .position(|leaders| leaders[fanout - 1] == u32::MAX)
        {
            return Err(PartitionKernelError::InsufficientRankableDistances { row, fanout });
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ScaleSlices<'a> {
    rows: &'a [f32],
    leaders: &'a [f32],
}

fn validate<'a, M: KernelMetric>(
    input: PartitionTopK<'a>,
    output: &MutMatrixView<'_, u32>,
) -> Result<ScaleSlices<'a>, PartitionKernelError> {
    let rows = input.dots.nrows();
    let leaders = input.dots.ncols();
    let fanout = output.ncols();

    let dots_len = checked_area("dot-product tile", rows, leaders)?;
    check_length("dot-product tile", input.dots.as_slice().len(), dots_len)?;
    let output_len = checked_area("output", output.nrows(), fanout)?;
    check_length("output", output.as_slice().len(), output_len)?;

    if output.nrows() != rows {
        return Err(PartitionKernelError::InvalidOutputShape {
            expected_rows: rows,
            actual_rows: output.nrows(),
            actual_cols: output.ncols(),
        });
    }
    if leaders > u32::MAX as usize {
        return Err(PartitionKernelError::TooManyLeaders(leaders));
    }
    if fanout > MAX_PARTITION_FANOUT || fanout > leaders {
        return Err(PartitionKernelError::InvalidFanout {
            fanout,
            leaders,
            maximum: MAX_PARTITION_FANOUT,
        });
    }

    let scales = match (M::METRIC, input.scales) {
        (
            Metric::L2,
            PartitionScales::L2 {
                leader_squared_norms,
            },
        ) => ScaleSlices {
            rows: &[],
            leaders: leader_squared_norms,
        },
        (
            Metric::Cosine,
            PartitionScales::Cosine {
                row_squared_norms,
                leader_norms,
            },
        ) => ScaleSlices {
            rows: row_squared_norms,
            leaders: leader_norms,
        },
        (Metric::CosineNormalized | Metric::InnerProduct, PartitionScales::None) => ScaleSlices {
            rows: &[],
            leaders: &[],
        },
        (Metric::L2, _) => return Err(PartitionKernelError::InvalidScales { expected: "L2" }),
        (Metric::Cosine, _) => {
            return Err(PartitionKernelError::InvalidScales { expected: "cosine" });
        }
        (Metric::CosineNormalized, _) => {
            return Err(PartitionKernelError::InvalidScales {
                expected: "normalized cosine",
            });
        }
        (Metric::InnerProduct, _) => {
            return Err(PartitionKernelError::InvalidScales {
                expected: "inner product",
            });
        }
    };

    check_length(
        "row scales",
        scales.rows.len(),
        expected_scale_len(M::PARTITION_ROW_SCALE, rows),
    )?;
    check_length(
        "leader scales",
        scales.leaders.len(),
        expected_scale_len(M::PARTITION_LEADER_SCALE, leaders),
    )?;
    Ok(scales)
}

const fn expected_scale_len(kind: ScaleKind, count: usize) -> usize {
    if kind.is_some() {
        count
    } else {
        0
    }
}

fn checked_area(
    buffer: &'static str,
    rows: usize,
    cols: usize,
) -> Result<usize, PartitionKernelError> {
    rows.checked_mul(cols)
        .ok_or(PartitionKernelError::ShapeOverflow { buffer, rows, cols })
}

fn check_length(
    buffer: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), PartitionKernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PartitionKernelError::InvalidBufferLength {
            buffer,
            expected,
            actual,
        })
    }
}

fn process_rows<F, M>(
    arch: F::Arch,
    dots: MatrixView<'_, f32>,
    scales: ScaleSlices<'_>,
    fanout: usize,
    output: &mut [u32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let leaders = dots.ncols();
    for (row, (dot_row, output_row)) in dots
        .as_slice()
        .chunks_exact(leaders)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        let row_scale = if M::PARTITION_ROW_SCALE.is_some() {
            M::PARTITION_ROW_SCALE.transform(scales.rows[row])
        } else {
            0.0
        };
        let row_scale_vector = F::splat(arch, row_scale);
        let mut top = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
        let full = leaders / F::LANES * F::LANES;

        for base in (0..full).step_by(F::LANES) {
            // SAFETY: `base + F::LANES <= full <= dot_row.len()`.
            let dots = unsafe { F::load_simd(arch, dot_row.as_ptr().add(base)) };
            let leader_scales = if M::PARTITION_LEADER_SCALE.is_some() {
                // SAFETY: validation requires one leader scale per dot-product column.
                unsafe { F::load_simd(arch, scales.leaders.as_ptr().add(base)) }
            } else {
                F::default(arch)
            };
            insert_lanes(
                M::partition_distance(arch, dots, row_scale_vector, leader_scales),
                base,
                &mut top,
                fanout,
            );
        }

        for (leader, &dot) in dot_row.iter().enumerate().skip(full) {
            let leader_scale = if M::PARTITION_LEADER_SCALE.is_some() {
                M::PARTITION_LEADER_SCALE.transform(scales.leaders[leader])
            } else {
                0.0
            };
            insert_topk(
                &mut top,
                fanout,
                leader as u32,
                M::partition_distance_scalar(dot, row_scale, leader_scale),
            );
        }
        copy_ids(&top, output_row);
    }
}

#[cfg(test)]
fn process_rows_scalar<M: KernelMetric>(
    dots: MatrixView<'_, f32>,
    scales: ScaleSlices<'_>,
    fanout: usize,
    output: &mut [u32],
) {
    let leaders = dots.ncols();
    for (row, (dot_row, output_row)) in dots
        .as_slice()
        .chunks_exact(leaders)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        let row_scale = if M::PARTITION_ROW_SCALE.is_some() {
            M::PARTITION_ROW_SCALE.transform(scales.rows[row])
        } else {
            0.0
        };
        let mut top = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
        for (leader, &dot) in dot_row.iter().enumerate() {
            let leader_scale = if M::PARTITION_LEADER_SCALE.is_some() {
                M::PARTITION_LEADER_SCALE.transform(scales.leaders[leader])
            } else {
                0.0
            };
            insert_topk(
                &mut top,
                fanout,
                leader as u32,
                M::partition_distance_scalar(dot, row_scale, leader_scale),
            );
        }
        copy_ids(&top, output_row);
    }
}

fn insert_lanes<F>(distances: F, base: usize, top: &mut TopK, fanout: usize)
where
    F: SIMDVector<Scalar = f32> + SIMDPartialOrd,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let threshold = F::splat(distances.arch(), top[fanout - 1].1);
    let eligible = distances.lt_simd(threshold);
    if eligible.none() {
        return;
    }

    let values = distances.to_array();
    let values = values.as_ref();
    let mut lanes = u64::from(eligible.bitmask().to_underlying());
    while lanes != 0 {
        let lane = lanes.trailing_zeros() as usize;
        lanes &= lanes - 1;
        insert_topk(top, fanout, (base + lane) as u32, values[lane]);
    }
}

#[inline(always)]
fn insert_topk(top: &mut TopK, fanout: usize, leader: u32, distance: f32) {
    let threshold = fanout - 1;
    if distance.partial_cmp(&top[threshold].1) != Some(std::cmp::Ordering::Less) {
        return;
    }

    top[threshold] = (leader, distance);
    let mut position = threshold;
    while position > 0 && top[position].1 < top[position - 1].1 {
        top.swap(position, position - 1);
        position -= 1;
    }
}

fn copy_ids(top: &TopK, output: &mut [u32]) {
    for (destination, &(leader, _)) in output.iter_mut().zip(top) {
        *destination = leader;
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel_metric::{Cosine, CosineNormalized, InnerProduct, KernelMetric, L2};

    use super::*;

    fn data(metric: Metric, leaders: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dots = (0..2 * leaders)
            .map(|index| (((index * 13 + 7) % 29) as f32 - 14.0) * 0.125)
            .collect();
        let row_scales = if metric == Metric::Cosine {
            vec![0.0, 16.0]
        } else {
            Vec::new()
        };
        let leader_scales = match metric {
            Metric::L2 => (0..leaders)
                .map(|leader| ((leader + 1) as f32).powi(2))
                .collect(),
            Metric::Cosine => (0..leaders)
                .map(|leader| {
                    if leader == 0 {
                        0.0
                    } else {
                        (leader + 1) as f32
                    }
                })
                .collect(),
            Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
        };
        (dots, row_scales, leader_scales)
    }

    fn input<'a>(
        metric: Metric,
        dots: &'a [f32],
        rows: usize,
        leaders: usize,
        row_scales: &'a [f32],
        leader_scales: &'a [f32],
    ) -> PartitionTopK<'a> {
        let scales = match metric {
            Metric::L2 => PartitionScales::L2 {
                leader_squared_norms: leader_scales,
            },
            Metric::Cosine => PartitionScales::Cosine {
                row_squared_norms: row_scales,
                leader_norms: leader_scales,
            },
            Metric::CosineNormalized | Metric::InnerProduct => PartitionScales::None,
        };
        PartitionTopK {
            dots: MatrixView::try_from(dots, rows, leaders).unwrap(),
            scales,
        }
    }

    fn scalar<M: KernelMetric>(input: PartitionTopK<'_>, fanout: usize, output: &mut [u32]) {
        let scales = match input.scales {
            PartitionScales::L2 {
                leader_squared_norms,
            } => ScaleSlices {
                rows: &[],
                leaders: leader_squared_norms,
            },
            PartitionScales::Cosine {
                row_squared_norms,
                leader_norms,
            } => ScaleSlices {
                rows: row_squared_norms,
                leaders: leader_norms,
            },
            PartitionScales::None => ScaleSlices {
                rows: &[],
                leaders: &[],
            },
        };
        process_rows_scalar::<M>(input.dots, scales, fanout, output);
    }

    fn scalar_for_metric(
        metric: Metric,
        input: PartitionTopK<'_>,
        fanout: usize,
        output: &mut [u32],
    ) {
        match metric {
            Metric::L2 => scalar::<L2>(input, fanout, output),
            Metric::Cosine => scalar::<Cosine>(input, fanout, output),
            Metric::CosineNormalized => scalar::<CosineNormalized>(input, fanout, output),
            Metric::InnerProduct => scalar::<InnerProduct>(input, fanout, output),
        }
    }

    fn assert_scalar_reference_matches_prepared_dispatch(metric: Metric) {
        // Leader count controls SIMD chunking. Exercise both sides of 4-, 8-, and
        // 16-lane boundaries, then a second 16-lane chunk.
        for leaders in [2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let (dots, row_scales, leader_scales) = data(metric, leaders);
            let input = input(metric, &dots, 2, leaders, &row_scales, &leader_scales);
            let kernel = PartitionKernel::new(metric);
            for fanout in [1, 2, 6, MAX_PARTITION_FANOUT] {
                if fanout > leaders {
                    continue;
                }
                let mut expected = vec![u32::MAX; 2 * fanout];
                kernel
                    .nearest_leaders(
                        input,
                        MutMatrixView::try_from(expected.as_mut_slice(), 2, fanout).unwrap(),
                    )
                    .unwrap();

                let mut actual = vec![u32::MAX; 2 * fanout];
                scalar_for_metric(metric, input, fanout, &mut actual);
                assert_eq!(
                    actual, expected,
                    "{metric:?}, leaders={leaders}, k={fanout}"
                );
            }
        }
    }

    #[test]
    fn l2_scalar_reference_matches_prepared_dispatch_at_lane_boundaries() {
        assert_scalar_reference_matches_prepared_dispatch(Metric::L2);
    }

    #[test]
    fn cosine_scalar_reference_matches_prepared_dispatch_at_lane_boundaries() {
        assert_scalar_reference_matches_prepared_dispatch(Metric::Cosine);
    }

    #[test]
    fn normalized_cosine_scalar_reference_matches_prepared_dispatch_at_lane_boundaries() {
        assert_scalar_reference_matches_prepared_dispatch(Metric::CosineNormalized);
    }

    #[test]
    fn inner_product_scalar_reference_matches_prepared_dispatch_at_lane_boundaries() {
        assert_scalar_reference_matches_prepared_dispatch(Metric::InnerProduct);
    }

    #[test]
    fn scalar_distance_matches_metric_contract() {
        assert_eq!(L2::partition_distance_scalar(2.0, 0.0, 9.0), 5.0);
        assert_eq!(
            CosineNormalized::partition_distance_scalar(0.25, 0.0, 0.0),
            0.75
        );
        assert_eq!(InnerProduct::partition_distance_scalar(3.0, 0.0, 0.0), -3.0);
        assert_eq!(Cosine::partition_distance_scalar(4.0, 2.0, 4.0), 0.5);
        assert_eq!(Cosine::partition_distance_scalar(4.0, 0.0, 4.0), 1.0);
        assert!(Cosine::partition_distance_scalar(1.0, f32::NAN, 1.0).is_nan());
    }

    #[test]
    fn cosine_special_norms_match_scalar_and_prepared_dispatch() {
        let leaders = 17;
        let dots = vec![1.0; 4 * leaders];
        let row_scales = [0.0, f32::MIN_POSITIVE / 2.0, f32::MIN_POSITIVE, f32::NAN];
        let mut leader_scales = vec![1.0; leaders];
        leader_scales[..4].copy_from_slice(&[
            0.0,
            f32::MIN_POSITIVE.sqrt() / 2.0,
            f32::MIN_POSITIVE.sqrt(),
            f32::NAN,
        ]);
        let input = input(
            Metric::Cosine,
            &dots,
            row_scales.len(),
            leaders,
            &row_scales,
            &leader_scales,
        );
        let mut expected = vec![u32::MAX; row_scales.len() * 2];
        scalar::<Cosine>(input, 2, &mut expected);
        let mut actual = vec![u32::MAX; row_scales.len() * 2];
        PartitionKernel::new(Metric::Cosine)
            .nearest_leaders(
                input,
                MutMatrixView::try_from(actual.as_mut_slice(), row_scales.len(), 2).unwrap(),
            )
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(&actual[..4], &[0, 1, 0, 1]);
        assert_eq!(&actual[6..], &[0, 1]);
    }

    #[test]
    fn matrix_area_overflow_is_rejected_before_kernel_access() {
        assert_eq!(
            checked_area("dot-product tile", usize::MAX, 2),
            Err(PartitionKernelError::ShapeOverflow {
                buffer: "dot-product tile",
                rows: usize::MAX,
                cols: 2,
            })
        );
    }

    #[test]
    fn scalar_topk_orders_candidates_and_preserves_ties() {
        let mut top = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
        for (leader, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 1.0)] {
            insert_topk(&mut top, 4, leader, distance);
        }
        insert_topk(&mut top, 4, 5, f32::NAN);

        assert_eq!(top[..4], [(1, 1.0), (4, 1.0), (3, 2.0), (2, 3.0)]);
    }
}
