/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use anyhow::{anyhow, Context};
use diskann_benchmark_runner::{
    files::InputFile, utils::datatype::DataType, CheckDeserialization, Checker,
};
use serde::{Deserialize, Serialize};

use crate::{
    inputs::{as_input, Example, Input},
    utils::{datafiles::ConvertingLoad, SimilarityMeasure},
};

const PRINT_WIDTH: usize = 18;
macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>PRINT_WIDTH$}: {}", $field, $($expr)*)
    }
}

//////////////
// Registry //
//////////////

as_input!(Spherical);
as_input!(Product);
as_input!(MinMax);

pub(super) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    registry.register(Input::<Spherical>::new())?;
    registry.register(Input::<Product>::new())?;
    registry.register(Input::<MinMax>::new())?;
    Ok(())
}

////////////
// Search //
////////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SearchValues {
    pub(crate) recall_k: Vec<usize>,
    pub(crate) recall_n: Vec<usize>,
}

impl CheckDeserialization for SearchValues {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Ensure that both `recall_k` and `recall_n` are non-empty.
        if self.recall_k.is_empty() {
            return Err(anyhow!("field `recall_k` cannot be empty"));
        }

        if self.recall_n.is_empty() {
            return Err(anyhow!("field `recall_n` cannot be empty"));
        }

        // Sort `recall_k` and `recall_n`.
        self.recall_k.sort_unstable();
        self.recall_k.dedup();

        self.recall_n.sort_unstable();
        self.recall_n.dedup();

        // Ensure that there is at least one valid combination of `recall_k` and `recall_n`.
        // Also check that both are non-empty.
        let min_recall_k = match self.recall_k.first() {
            None => {
                return Err(anyhow!("field `recall_k` cannot be empty"));
            }
            Some(recall_k) => recall_k,
        };

        let max_recall_n = match self.recall_n.last() {
            None => {
                return Err(anyhow!("field `recall_n` cannot be empty"));
            }
            Some(recall_n) => recall_n,
        };

        if min_recall_k > max_recall_n {
            return Err(anyhow!(
                "minimum `recall_k` value ({}) must be less than the maximum `recall_n` ({})",
                min_recall_k,
                max_recall_n,
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    // Enable sweeping threads
    pub(crate) num_threads: NonZeroUsize,
    pub(crate) recalls: SearchValues,
}

impl CheckDeserialization for SearchPhase {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Check the validity of the input files.
        self.queries.check_deserialization(checker)?;
        self.groundtruth.check_deserialization(checker)?;
        self.recalls.check_deserialization(checker)?;
        Ok(())
    }
}

impl Example for SearchPhase {
    fn example() -> Self {
        const NUM_THREADS: NonZeroUsize = NonZeroUsize::new(8).unwrap();

        let recalls = SearchValues {
            recall_k: vec![10, 20, 30, 40],
            recall_n: vec![10, 20, 30, 40],
        };

        Self {
            queries: InputFile::new("path/to/queries"),
            groundtruth: InputFile::new("path/to/groundtruth"),
            num_threads: NUM_THREADS,
            recalls,
        }
    }
}

////////////////////////////////
// Transforms related methods //
///////////////////////////////
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TargetDim {
    Same,
    Natural,
    Override(NonZeroUsize),
}

impl std::fmt::Display for TargetDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Same => write!(f, "same_dim"),
            Self::Natural => write!(f, "natural"),
            Self::Override(dim) => write!(f, "{}", dim.get()),
        }
    }
}

impl From<TargetDim> for diskann_quantization::algorithms::transforms::TargetDim {
    fn from(dim: TargetDim) -> Self {
        match dim {
            TargetDim::Same => Self::Same,
            TargetDim::Natural => Self::Natural,
            TargetDim::Override(dim) => Self::Override(dim),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TransformKind {
    PaddingHadamard(TargetDim),
    RandomRotation(TargetDim),
    DoubleHadamard(TargetDim),
    Null,
}

impl std::fmt::Display for TransformKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PaddingHadamard(target) => write!(f, "padding_hadamard({})", target),
            Self::RandomRotation(target) => write!(f, "random_rotation({})", target),
            Self::DoubleHadamard(target) => write!(f, "double_hadamard({})", target),
            Self::Null => write!(f, "null_transform"),
        }
    }
}

impl From<&TransformKind> for diskann_quantization::algorithms::transforms::TransformKind {
    fn from(kind: &TransformKind) -> Self {
        match kind {
            TransformKind::PaddingHadamard(target) => {
                diskann_quantization::algorithms::transforms::TransformKind::PaddingHadamard {
                    target_dim: (*target).into(),
                }
            }
            TransformKind::RandomRotation(target) => {
                diskann_quantization::algorithms::transforms::TransformKind::RandomRotation {
                    target_dim: (*target).into(),
                }
            }
            TransformKind::DoubleHadamard(target) => {
                diskann_quantization::algorithms::transforms::TransformKind::DoubleHadamard {
                    target_dim: (*target).into(),
                }
            }
            TransformKind::Null => {
                diskann_quantization::algorithms::transforms::TransformKind::Null
            }
        }
    }
}
//////////////////////////////////
// Product Quantization Methods //
//////////////////////////////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Product {
    pub(crate) data: InputFile,
    pub(crate) data_type: DataType,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) compression_threads: NonZeroUsize,
    pub(crate) search: SearchPhase,
    pub(crate) seed: u64,
    pub(crate) num_pq_chunks: NonZeroUsize,
    pub(crate) num_pq_centers: NonZeroUsize,
}

impl Product {
    pub(crate) const fn tag() -> &'static str {
        "exhaustive-product-quantization"
    }
}

impl CheckDeserialization for Product {
    fn check_deserialization(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.data.check_deserialization(checker)?;
        self.search.check_deserialization(checker)?;

        // Chcck that provided data type is compatible with `f32`.
        f32::check_converting_load(self.data_type)?;

        let num_centers = self.num_pq_centers.get();
        if num_centers > 256 {
            return Err(anyhow!(
                "Number of PQ Centers ({}) cannot exceed 256",
                num_centers
            ));
        }

        Ok(())
    }
}

impl Example for Product {
    fn example() -> Self {
        const NUM_PQ_CHUNKS: NonZeroUsize = NonZeroUsize::new(128).unwrap();
        const NUM_PQ_CENTERS: NonZeroUsize = NonZeroUsize::new(256).unwrap();
        const COMPRESSION_THREADS: NonZeroUsize = NonZeroUsize::new(8).unwrap();

        Self {
            data: InputFile::new("path/to/data"),
            data_type: DataType::Float32,
            distance: SimilarityMeasure::SquaredL2,
            compression_threads: COMPRESSION_THREADS,
            search: SearchPhase::example(),
            seed: 0x6cae32c479ac3407,
            num_pq_chunks: NUM_PQ_CHUNKS,
            num_pq_centers: NUM_PQ_CENTERS,
        }
    }
}

impl std::fmt::Display for Product {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Product Quantization Exhaustive Search")?;
        write_field!(f, "data", self.data.display())?;
        write_field!(f, "data type", self.data_type)?;
        write_field!(f, "distance", self.distance)?;
        write_field!(f, "seed", self.seed)?;
        write_field!(f, "PQ Chunks", self.num_pq_chunks.get())?;
        write_field!(f, "PQ Centers", self.num_pq_centers.get())?;
        Ok(())
    }
}

//////////////////////////////////////////
// Spherical-quantization-based methods //
//////////////////////////////////////////

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SphericalQuery {
    SameAsData,
    FourBitTransposed,
    ScalarQuantized,
    FullPrecision,
}

impl std::fmt::Display for SphericalQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::SameAsData => "same_as_data",
            Self::FourBitTransposed => "four_bit_transposed",
            Self::ScalarQuantized => "scalar_quantized",
            Self::FullPrecision => "full_precision",
        };
        write!(f, "{}", st)
    }
}

impl From<SphericalQuery> for diskann_quantization::spherical::iface::QueryLayout {
    fn from(value: SphericalQuery) -> Self {
        match value {
            SphericalQuery::SameAsData => Self::SameAsData,
            SphericalQuery::FourBitTransposed => Self::FourBitTransposed,
            SphericalQuery::ScalarQuantized => Self::ScalarQuantized,
            SphericalQuery::FullPrecision => Self::FullPrecision,
        }
    }
}

/// Check the compatibility between the number of bits used for compression and the kind
/// of the query.
pub(super) fn check_compatibility(num_bits: usize, query: SphericalQuery) -> anyhow::Result<()> {
    use SphericalQuery::{FourBitTransposed, FullPrecision, SameAsData, ScalarQuantized};

    match num_bits {
        1 => match query {
            SameAsData | FourBitTransposed | FullPrecision => Ok(()),
            ScalarQuantized => Err(anyhow::anyhow!(
                "Normal scalar quantization is not compatible with 1-bit data.\
                 Use \"four_bit_transpose\" instead"
            )),
        },
        2 | 4 | 8 => match query {
            SameAsData | ScalarQuantized | FullPrecision => Ok(()),
            FourBitTransposed => Err(anyhow::anyhow!(
                "Bit transposed (\"{}\") queries are not compatible with {}-bit data. \
                 Use \"scalar_quantized\" instead",
                FourBitTransposed,
                num_bits
            )),
        },
        x => Err(anyhow::anyhow!(
            "{} bits are not supported for spherical quantization",
            x
        )),
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PreScale {
    None,
    Some(f32),
    ReciprocalMeanNorm,
}

impl TryFrom<PreScale> for diskann_quantization::spherical::PreScale {
    type Error = anyhow::Error;
    fn try_from(value: PreScale) -> Result<Self, Self::Error> {
        let v = match value {
            PreScale::None => Self::None,
            PreScale::Some(v) => Self::Some(diskann_quantization::num::Positive::new(v)?),
            PreScale::ReciprocalMeanNorm => Self::ReciprocalMeanNorm,
        };
        Ok(v)
    }
}

impl std::fmt::Display for PreScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "no scaling"),
            Self::Some(v) => write!(f, "pre-scale({})", v),
            Self::ReciprocalMeanNorm => write!(f, "reciprocal mean norm scaling"),
        }
    }
}

impl CheckDeserialization for PreScale {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        if let Self::Some(v) = self {
            if *v <= 0.0 {
                anyhow::bail!("pre-scaling {} must be positive", v);
            }

            if !v.is_finite() {
                anyhow::bail!("pre-scaling {} must be finite", v);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Spherical {
    pub(crate) data: InputFile,
    pub(crate) data_type: DataType,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) compression_threads: NonZeroUsize,
    pub(crate) search: SearchPhase,
    pub(crate) query_layouts: Vec<SphericalQuery>,
    pub(crate) seed: u64,
    pub(crate) transform_kind: TransformKind,
    pub(crate) num_bits: NonZeroUsize,
    pub(crate) pre_scale: PreScale,
}

impl Spherical {
    pub(crate) const fn tag() -> &'static str {
        "exhaustive-spherical-quantization"
    }
}

impl CheckDeserialization for Spherical {
    fn check_deserialization(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.data.check_deserialization(checker)?;
        self.search.check_deserialization(checker)?;

        // Chcck that provided data type is compatible with `f32`.
        f32::check_converting_load(self.data_type)?;

        // Check query plan.
        for (i, layout) in self.query_layouts.iter().enumerate() {
            check_compatibility(self.num_bits.get(), *layout).with_context(|| {
                format!(
                    "while validating query layout {} of {}",
                    i + 1,
                    self.query_layouts.len()
                )
            })?;
        }

        self.pre_scale.check_deserialization(checker)?;
        Ok(())
    }
}

impl Example for Spherical {
    fn example() -> Self {
        const NUM_BITS: NonZeroUsize = NonZeroUsize::new(1).unwrap();
        const COMPRESSION_THREADS: NonZeroUsize = NonZeroUsize::new(8).unwrap();

        Self {
            data: InputFile::new("path/to/data"),
            data_type: DataType::Float32,
            distance: SimilarityMeasure::SquaredL2,
            compression_threads: COMPRESSION_THREADS,
            search: SearchPhase::example(),
            query_layouts: vec![
                SphericalQuery::SameAsData,
                SphericalQuery::FourBitTransposed,
            ],
            seed: 0x6cae32c479ac3407,
            transform_kind: TransformKind::PaddingHadamard(TargetDim::Same),
            num_bits: NUM_BITS,
            pre_scale: PreScale::Some(1.0),
        }
    }
}

impl std::fmt::Display for Spherical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Spherical Quantization Exhaustive Search")?;
        write_field!(f, "data", self.data.display())?;
        write_field!(f, "data type", self.data_type)?;
        write_field!(f, "distance", self.distance)?;
        write_field!(f, "seed", self.seed)?;
        write_field!(f, "transform", self.transform_kind)?;
        write_field!(f, "num bits", self.num_bits)?;
        write_field!(f, "pre scale", self.pre_scale)?;
        Ok(())
    }
}

///////////////////////////////////////
// MinMax-quantization-based methods //
///////////////////////////////////////

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MinMaxQuery {
    SameAsData,
    FullPrecision,
}

impl std::fmt::Display for MinMaxQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::SameAsData => "same_as_data",
            Self::FullPrecision => "full_precision",
        };
        write!(f, "{}", st)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MinMax {
    pub(crate) data: InputFile,
    pub(crate) data_type: DataType,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) search: SearchPhase,
    pub(crate) query_layouts: Vec<MinMaxQuery>,
    pub(crate) num_bits: NonZeroUsize,
    pub(crate) transform_kind: TransformKind,
    pub(crate) seed: u64, //for transform
    pub(crate) scale: f32,
}

impl MinMax {
    pub(crate) const fn tag() -> &'static str {
        "exhaustive-minmax-quantization"
    }
}

impl CheckDeserialization for MinMax {
    fn check_deserialization(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.data.check_deserialization(checker)?;
        self.search.check_deserialization(checker)?;

        // Chcck that provided data type is compatible with `f32`.
        f32::check_converting_load(self.data_type)?;

        if self.scale <= 0.0 {
            return Err(anyhow::anyhow!(
                "Grid scale parameter for minmax must be >= 0.0, got {}",
                self.scale
            ));
        }

        Ok(())
    }
}

impl Example for MinMax {
    fn example() -> Self {
        const NUM_BITS: NonZeroUsize = NonZeroUsize::new(4).unwrap();

        Self {
            data: InputFile::new("path/to/data"),
            data_type: DataType::Float32,
            distance: SimilarityMeasure::SquaredL2,
            search: SearchPhase::example(),
            query_layouts: vec![MinMaxQuery::SameAsData, MinMaxQuery::FullPrecision],
            num_bits: NUM_BITS,
            transform_kind: TransformKind::DoubleHadamard(TargetDim::Same),
            seed: 0x6cae32c479ac3407,
            scale: 1.0,
        }
    }
}

impl std::fmt::Display for MinMax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MinMax Quantization Exhaustive Search")?;
        write_field!(f, "data", self.data.display())?;
        write_field!(f, "data type", self.data_type)?;
        write_field!(f, "distance", self.distance)?;
        write_field!(f, "num_bits", self.num_bits)?;
        write_field!(f, "transform", self.transform_kind)?;
        write_field!(f, "seed", self.seed)?;
        write_field!(f, "scale", self.scale)?;
        Ok(())
    }
}
