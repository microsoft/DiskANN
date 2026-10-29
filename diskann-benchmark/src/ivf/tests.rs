/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use anyhow::Result;
use diskann::{
    ivf::{IVFIndex, SearchResult},
    provider::DefaultContext,
};
use diskann_benchmark_runner::utils::datatype::DataType;
use diskann_utils::views::rowmajor::Matrix;
use tempfile::NamedTempFile;

use super::{build, data, provider::InMemProvider};
#[cfg(not(feature = "ivf-experimental-compute"))]
use crate::inputs::ivf::SplitMode;
use crate::inputs::ivf::{Build, DataFormat};

fn build_config() -> Build {
    Build {
        batch_size: NonZeroUsize::new(8).unwrap(),
        split_threshold: 4,
        iterations: 5,
        ..Build::default()
    }
}

fn ids(result: &SearchResult<u32>) -> Vec<u32> {
    result.neighbors.iter().map(|hit| *hit.id()).collect()
}

async fn check_search_against_flat(
    index: &IVFIndex<InMemProvider>,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> Result<()> {
    let nlists = index.provider().lists.len();
    for query in queries {
        let truth = data::flat_top_k(base, query, k);
        let mut previous_scanned = 0;
        let mut previous_hits = 0;
        for nprobe in 1..=nlists {
            let result = index.search(&DefaultContext, query, k, nprobe).await?;
            let found = ids(&result);
            let hits = found.iter().filter(|id| truth.contains(id)).count();
            assert!(result.candidates_scanned >= previous_scanned && hits >= previous_hits);
            previous_scanned = result.candidates_scanned;
            previous_hits = hits;
            if nprobe == nlists {
                assert_eq!(found, truth);
                assert_eq!(result.candidates_scanned, base.len());
            }
        }
    }
    Ok(())
}

#[test]
fn benchmark_provider_matches_flat_after_build() -> Result<()> {
    diskann_benchmark_core::tokio::runtime(1)?.block_on(async {
        let base: Vec<Vec<f32>> = (0..48).map(|i| vec![i as f32, (i % 7) as f32]).collect();
        let config = build_config();
        let (index, _) = build::build(&base, &config, &mut || Ok(())).await?;
        assert!(index.build_metrics().split_calls > 0);
        check_search_against_flat(
            &index,
            &base,
            &[vec![0.4, 1.1], vec![20.0, -0.5], vec![49.0, 2.1]],
            3,
        )
        .await?;
        Ok(())
    })
}

#[test]
fn invalid_build_configuration() -> Result<()> {
    let config = build_config();
    config.validate()?;
    #[cfg(not(feature = "ivf-parallel"))]
    {
        let unsupported = Build {
            num_threads: NonZeroUsize::new(4).unwrap(),
            ..config.clone()
        };
        assert!(unsupported.validate().is_err());
    }
    #[cfg(not(feature = "ivf-experimental-compute"))]
    {
        let unsupported = Build {
            split_mode: SplitMode::ProjectionGemm,
            ..config.clone()
        };
        assert!(unsupported.validate().is_err());
    }
    assert!(Build {
        split_threshold: 1,
        ..config
    }
    .validate()
    .is_err());
    Ok(())
}

#[test]
fn retained_modes_drive_benchmark_build() -> Result<()> {
    diskann_benchmark_core::tokio::runtime(1)?.block_on(async {
        let (base, queries) = data::synthetic();
        let base = &base[..1024];
        #[cfg(not(feature = "ivf-experimental-compute"))]
        let modes = ["baseline"];
        #[cfg(feature = "ivf-experimental-compute")]
        let modes = ["baseline", "projection-gemm"];
        for mode in modes {
            let config: Build = serde_json::from_value(serde_json::json!({
                "batch_size": 256, "split_threshold": 32, "iterations": 5, "split_mode": mode,
            }))?;
            config.validate()?;
            let mut checkpoints = 0;
            let (index, result) = build::build(base, &config, &mut || {
                checkpoints += 1;
                Ok(())
            })
            .await?;
            assert_eq!(checkpoints, 4);
            assert_eq!(result.base_count, base.len());
            assert!(result.split_calls > 0);
            assert_eq!(result.projection_points > 0, mode == "projection-gemm");
            assert_eq!(result.gemm_calls > 0, mode == "projection-gemm");
            assert_eq!(result.numeric_fallbacks, 0);
            check_search_against_flat(&index, base, &queries[..3], 3).await?;
        }
        Ok(())
    })
}

#[test]
fn file_inputs_validate_shapes_and_truth_ids() -> Result<()> {
    let file = NamedTempFile::new()?;
    let path = file.path();
    let row: Vec<u8> = [1i32.to_le_bytes(), 1.5f32.to_le_bytes()].concat();
    std::fs::write(path, &row)?;
    assert_eq!(
        data::read_vectors(path, DataFormat::Fvecs, DataType::Float32)?,
        vec![vec![1.5]]
    );
    std::fs::write(path, &row[..7])?;
    assert!(data::read_vectors(path, DataFormat::Fvecs, DataType::Float32).is_err());
    let mixed = [row, 2i32.to_le_bytes().to_vec(), vec![0; 8]].concat();
    std::fs::write(path, mixed)?;
    assert!(data::read_vectors(path, DataFormat::Fvecs, DataType::Float32).is_err());
    let mut binary = [2u32.to_le_bytes(), 2u32.to_le_bytes()].concat();
    for bits in [0x3c00u16, 0xc100, 0x0000, 0x4200] {
        binary.extend(bits.to_le_bytes());
    }
    std::fs::write(path, &binary)?;
    assert_eq!(
        data::read_vectors(path, DataFormat::Bin, DataType::Float16)?,
        vec![vec![1.0, -2.5], vec![0.0, 3.0]]
    );
    assert!(data::read_vectors(path, DataFormat::Bin, DataType::Float32).is_err());
    std::fs::write(path, &binary[..binary.len() - 1])?;
    assert!(data::read_vectors(path, DataFormat::Bin, DataType::Float16).is_err());
    let mut truth = [2u32.to_le_bytes(), 3u32.to_le_bytes()].concat();
    for id in [0u32, 2, 1, 2, 1, 0] {
        truth.extend(id.to_le_bytes());
    }
    std::fs::write(path, &truth)?;
    let loaded = data::read_truth(path, 2, 3, 2)?;
    assert_eq!((loaded.nrows(), loaded.ncols()), (2, 2));
    assert_eq!(loaded.as_slice(), [0, 2, 2, 1]);
    assert!(data::read_truth(path, 1, 3, 2).is_err());
    truth[12..16].copy_from_slice(&0u32.to_le_bytes());
    std::fs::write(path, &truth)?;
    assert!(data::read_truth(path, 2, 3, 2).is_err());
    Ok(())
}
