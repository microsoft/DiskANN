/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::mem;

use diskann_providers::model::{IndexConfiguration, GRAPH_SLACK_FACTOR};
use tracing::info;

use crate::{data_model::GraphDataType, disk_index_build_parameter::BYTES_IN_GB, QuantizationType};
/// Overhead factor for RAM estimation during index build (10% buffer).
const OVERHEAD_FACTOR: f64 = 1.1f64;

/// Estimate RAM usage in bytes for building an index.
#[inline]
pub(super) fn estimate_build_index_ram_usage(
    num_points: u64,
    dim: u64,
    datasize: u64,
    graph_degree: u64,
    build_quantization_type: &QuantizationType,
) -> f64 {
    let graph_size =
        (num_points * graph_degree * mem::size_of::<u32>() as u64) as f64 * GRAPH_SLACK_FACTOR;

    let single_vec_size = match *build_quantization_type {
        QuantizationType::FP => dim.next_multiple_of(8u64) * datasize,
        // We can skip PQ pivots data as it is very small(~3MB) for even large datasets like OAI-3072.
        QuantizationType::PQ { num_chunks } => num_chunks as u64,
        // `+ std::mem::size_of::<f32>()` for f32 compensation metadata for the scalar quantizer.
        QuantizationType::SQ { nbits, .. } => {
            (nbits as u64 * dim).div_ceil(8) + std::mem::size_of::<f32>() as u64
        }
    };

    OVERHEAD_FACTOR * (graph_size + (single_vec_size * num_points) as f64)
}

pub(in crate::build::builder) enum IndexBuildStrategy {
    OneShot,
    Merged,
}

pub(in crate::build::builder) fn determine_build_strategy<Data: GraphDataType>(
    index_configuration: &IndexConfiguration,
    index_build_ram_limit_in_bytes: f64,
    build_quantization_type: &QuantizationType,
) -> IndexBuildStrategy {
    let estimated_index_ram_in_bytes = estimate_build_index_ram_usage(
        index_configuration.max_points as u64,
        index_configuration.dim as u64,
        mem::size_of::<Data::VectorDataType>() as u64,
        index_configuration.config.max_degree().get() as u64,
        build_quantization_type,
    );

    info!(
        "Estimated index RAM usage: {} GB, index_build_ram_limit={} GB",
        estimated_index_ram_in_bytes / BYTES_IN_GB,
        index_build_ram_limit_in_bytes / BYTES_IN_GB
    );

    if estimated_index_ram_in_bytes >= index_build_ram_limit_in_bytes {
        info!(
            "Insufficient memory budget for index build in one shot, index_build_ram_limit={} GB estimated_index_ram={} GB",
            index_build_ram_limit_in_bytes / BYTES_IN_GB,
            estimated_index_ram_in_bytes / BYTES_IN_GB,
        );
        IndexBuildStrategy::Merged
    } else {
        info!(
            "Full index fits in RAM budget, should consume at most {} GBs, so building in one shot",
            estimated_index_ram_in_bytes / BYTES_IN_GB
        );
        IndexBuildStrategy::OneShot
    }
}

#[cfg(test)]
mod ram_estimation_tests {
    use rstest::rstest;

    use super::*;
    use crate::QuantizationType;

    #[rstest]
    #[case(QuantizationType::FP)]
    #[case(QuantizationType::PQ { num_chunks: 15 })]
    #[case(QuantizationType::SQ { nbits: 1, standard_deviation: None })]
    fn test_estimate_build_index_ram_usage(#[case] build_quantization_type: QuantizationType) {
        let num_points = 1000;
        let dim = 128;
        let size_of_t = std::mem::size_of::<f32>() as u64;
        let graph_degree = 50;

        let single_vec_size = match build_quantization_type {
            QuantizationType::FP => dim * size_of_t,
            QuantizationType::PQ { num_chunks } => num_chunks as u64,
            QuantizationType::SQ { nbits, .. } => {
                (nbits as u64 * dim).div_ceil(8) + std::mem::size_of::<f32>() as u64
            }
        };
        let mut expected_ram_usage = (num_points as f64)
            * (graph_degree as f64)
            * (std::mem::size_of::<u32>() as f64)
            * GRAPH_SLACK_FACTOR
            + (num_points * single_vec_size) as f64;
        expected_ram_usage *= OVERHEAD_FACTOR;

        let actual_ram_usage = estimate_build_index_ram_usage(
            num_points,
            dim,
            size_of_t,
            graph_degree,
            &build_quantization_type,
        );

        assert_eq!(actual_ram_usage, expected_ram_usage);
    }
}
