use crate::{ANNError, utils::VectorRepr};
use diskann_quantization::{
    CompressIntoWith,
    algorithms::{TransformKind, transforms::TargetDim},
    alloc::{GlobalAllocator, ScopedAllocator},
    spherical::{
        DataMut, Pairwise1Bit, Pairwise1BitScratch, PreScale, SphericalQuantizer, SupportedMetric,
    },
};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric;
use rand::{SeedableRng, rngs::StdRng, seq::index};
use rayon::prelude::*;

use super::leaf_kernel::LeafNeighbor;

const TRAINING_ROWS: usize = 1_000;
const SAMPLE_STREAM: u64 = 0x5241_4249_5451_0001;
const TRAIN_STREAM: u64 = 0x5241_4249_5451_0002;

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("RaBitQ1 requires non-empty data")]
    Empty,
    #[error("RaBitQ1 shape overflow")]
    Shape,
    #[error("RaBitQ1 conversion failed for point {point}")]
    Conversion {
        point: usize,
        #[source]
        source: ANNError,
    },
    #[error("RaBitQ1 compression failed for point {point}")]
    Compression {
        point: usize,
        #[source]
        source: diskann_quantization::spherical::CompressionError,
    },
    #[error("RaBitQ1 canonical row failed for point {point}")]
    Canonical { point: usize },
    #[error("RaBitQ1 point {point} is outside {points} rows")]
    Point { point: u32, points: usize },
    #[error("RaBitQ1 training failed: {0}")]
    Training(#[from] diskann_quantization::spherical::TrainError),
    #[error("RaBitQ1 plan failed: {0}")]
    Plan(#[from] diskann_quantization::alloc::AllocatorError),
}

pub(super) struct Store {
    rows: Matrix<u8>,
    self_scores: Vec<f32>,
    pairwise: Pairwise1Bit,
    metric: Metric,
    row_bytes: usize,
    dimensions: usize,
}

impl Store {
    pub(super) fn train<T>(
        data: MatrixView<'_, T>,
        metric: Metric,
        seed: u64,
    ) -> Result<Self, Error>
    where
        T: VectorRepr + Send + Sync,
    {
        if data.nrows() == 0 || data.ncols() == 0 {
            return Err(Error::Empty);
        }
        let count = data.nrows().min(TRAINING_ROWS);
        let ids = if data.nrows() <= TRAINING_ROWS {
            (0..count).collect()
        } else {
            let mut rng = StdRng::seed_from_u64(seed ^ SAMPLE_STREAM);
            let mut ids = index::sample(&mut rng, data.nrows(), count).into_vec();
            ids.sort_unstable();
            ids
        };
        let mut training = Matrix::new(0.0f32, count, data.ncols());
        for (&point, output) in ids.iter().zip(training.row_iter_mut()) {
            T::as_f32_into(data.row(point), output).map_err(|source| Error::Conversion {
                point,
                source: source.into(),
            })?;
        }
        let mut rng = StdRng::seed_from_u64(seed ^ TRAIN_STREAM);
        let quantizer = SphericalQuantizer::train(
            training.as_view(),
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            supported_metric(metric),
            PreScale::ReciprocalMeanNorm,
            &mut rng,
            GlobalAllocator,
        )?;
        let plan = diskann_quantization::spherical::iface::Impl::<1>::new(quantizer)?;
        let pairwise = Pairwise1Bit::new(plan.quantizer());
        let dimensions = pairwise.encoded_dim();
        let row_bytes = pairwise.row_bytes();
        data.nrows().checked_mul(row_bytes).ok_or(Error::Shape)?;
        let mut rows = Matrix::new(0u8, data.nrows(), row_bytes);
        let mut self_scores = vec![0.0f32; data.nrows()];
        rows.as_mut_slice()
            .par_chunks_mut(row_bytes)
            .zip(self_scores.par_iter_mut())
            .zip(data.as_slice().par_chunks_exact(data.ncols()))
            .enumerate()
            .try_for_each_init(
                || vec![0.0f32; data.ncols()],
                |scratch, (point, ((encoded, self_score), source))| {
                    T::as_f32_into(source, scratch).map_err(|source| Error::Conversion {
                        point,
                        source: source.into(),
                    })?;
                    *self_score =
                        exact_self_score(scratch, metric).ok_or(Error::Canonical { point })?;
                    let target = DataMut::<1>::from_canonical_back_mut(encoded, dimensions)
                        .map_err(|_| Error::Canonical { point })?;
                    plan.quantizer()
                        .compress_into_with(scratch.as_ref(), target, ScopedAllocator::global())
                        .map_err(|source| Error::Compression { point, source })
                },
            )?;
        Ok(Self {
            rows,
            self_scores,
            pairwise,
            metric,
            row_bytes,
            dimensions,
        })
    }

    pub(super) fn points(&self) -> usize {
        self.rows.nrows()
    }
    pub(super) fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    pub(super) fn row(&self, point: u32) -> Result<&[u8], Error> {
        self.rows.get_row(point as usize).ok_or(Error::Point {
            point,
            points: self.rows.nrows(),
        })
    }
    pub(super) fn self_score(&self, point: u32) -> Result<f32, Error> {
        self.self_scores
            .get(point as usize)
            .copied()
            .ok_or(Error::Point {
                point,
                points: self.rows.nrows(),
            })
    }
    pub(super) fn gather(&self, ids: &[u32], output: &mut Vec<u8>) -> Result<(), Error> {
        let len = ids.len().checked_mul(self.row_bytes).ok_or(Error::Shape)?;
        output.resize(len, 0);
        for (&id, target) in ids
            .iter()
            .zip(output[..len].chunks_exact_mut(self.row_bytes))
        {
            target.copy_from_slice(self.row(id)?);
        }
        Ok(())
    }
    pub(super) fn prepare_panel<A: super::simd::PiPNNSIMDSchema>(
        &self,
        arch: A,
        rows: MatrixView<'_, u8>,
        scratch: &mut Pairwise1BitScratch,
    ) {
        self.pairwise.prepare_panel(arch, rows, scratch);
    }
    pub(super) fn score_prepared<A: super::simd::PiPNNSIMDSchema>(
        &self,
        arch: A,
        source: u32,
        self_target: Option<usize>,
        targets: MatrixView<'_, u8>,
        scores: &mut [f32],
        scratch: &mut Pairwise1BitScratch,
    ) -> Result<(), Error> {
        self.pairwise
            .score_prepared_panel(arch, self.row(source)?, 0, targets, scores, scratch);
        if let Some(target) = self_target {
            scores[target] = self.self_score(source)?;
        }
        Ok(())
    }
    pub(super) fn rank_leaf<A>(
        &self,
        arch: A,
        rows: MatrixView<'_, u8>,
        k: usize,
        output: &mut [LeafNeighbor],
        scores: &mut Vec<f32>,
        worst: &mut Vec<f32>,
        scratch: &mut Pairwise1BitScratch,
    ) where
        A: super::simd::PiPNNSIMDSchema,
    {
        let points = rows.nrows();
        scores.resize(points, 0.0);
        worst.resize(points, f32::INFINITY);
        worst.fill(f32::INFINITY);
        output.fill(LeafNeighbor::default());
        self.prepare_panel(arch, rows, scratch);
        for source in 1..points {
            let targets = rows.subview(0..source).unwrap();
            self.pairwise.score_prepared_panel_from_prepared_source(
                arch,
                rows.row(source),
                source,
                0,
                targets,
                &mut scores[..source],
                scratch,
            );
            super::leaf_kernel::rank_final_score_row(
                arch,
                source,
                &scores[..source],
                output,
                k,
                worst,
            );
        }
    }
    pub(super) fn score_pair<A: super::simd::PiPNNSIMDSchema>(
        &self,
        arch: A,
        left: u32,
        right: u32,
    ) -> Result<f32, Error> {
        if left == right {
            self.self_score(left)
        } else {
            Ok(self
                .pairwise
                .score_pair(arch, self.row(left)?, self.row(right)?))
        }
    }
}

#[inline(always)]
fn insert_neighbor(neighbors: &mut [LeafNeighbor], target: u32, distance: f32) -> f32 {
    let last = neighbors.len() - 1;
    if distance.partial_cmp(&neighbors[last].distance) != Some(std::cmp::Ordering::Less) {
        return neighbors[last].distance;
    }
    neighbors[last] = LeafNeighbor { target, distance };
    let mut slot = last;
    while slot > 0 && neighbors[slot].distance < neighbors[slot - 1].distance {
        neighbors.swap(slot, slot - 1);
        slot -= 1;
    }
    neighbors[last].distance
}

fn supported_metric(metric: Metric) -> SupportedMetric {
    match metric {
        Metric::L2 => SupportedMetric::SquaredL2,
        Metric::InnerProduct => SupportedMetric::InnerProduct,
        Metric::Cosine | Metric::CosineNormalized => SupportedMetric::Cosine,
    }
}

fn exact_self_score(values: &[f32], metric: Metric) -> Option<f32> {
    let squared_norm = values.iter().try_fold(0.0f32, |sum, value| {
        let next = sum + value * value;
        next.is_finite().then_some(next)
    })?;
    Some(match metric {
        Metric::L2 => 0.0,
        Metric::InnerProduct => -squared_norm,
        Metric::Cosine | Metric::CosineNormalized => {
            if squared_norm == 0.0 {
                1.0
            } else {
                0.0
            }
        }
    })
}
