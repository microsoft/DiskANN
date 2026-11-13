/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use anyhow::Result;
use diskann::{
    ivf::{
        region::Update,
        traits::{BuildAccessor, BuildStrategy},
    },
    provider::DefaultContext,
};
use diskann_benchmark_runner::utils::datatype::DataType;
use diskann_utils::views::rowmajor::Matrix;
use tempfile::NamedTempFile;

use super::{
    build::{self, BuiltIndex},
    data, metric,
    provider::InMemProvider,
    search::{validate_full_probe, SearchResult},
    search_provider::SearchProvider,
};
use crate::{
    inputs::ivf::{Build, DataFormat, SearchPhase},
    utils::SimilarityMeasure,
};

fn config() -> Build {
    Build {
        batch_size: NonZeroUsize::new(128).unwrap(),
        split_threshold: NonZeroUsize::new(64).unwrap(),
        iterations: NonZeroUsize::new(5).unwrap(),
        record_layout: true,
        ..Build::default()
    }
}

fn sample() -> (Arc<data::Rows>, data::Rows) {
    let (mut base, mut queries) = data::synthetic();
    base.truncate(768);
    queries.truncate(12);
    (Arc::new(base), queries)
}

async fn check_search(
    index: &BuiltIndex,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    distance: SimilarityMeasure,
) -> Result<()> {
    let provider = SearchProvider::new(&*index.provider.read()?, distance);
    let count = index.provider.read()?.lists.len();
    for query in queries {
        let truth = data::flat_top_k(base, query, 31, distance);
        let full = index
            .search(&provider, query, NonZeroUsize::new(31).unwrap(), count)
            .await?;
        validate_full_probe(&full, &truth, base, query, distance)?;
        assert_eq!(full.candidates_scanned, base.len());
        assert_eq!(full.lists_scanned, count);
        let partial = index
            .search(&provider, query, NonZeroUsize::new(31).unwrap(), 1)
            .await?;
        assert!(partial.candidates_scanned <= full.candidates_scanned);
    }
    Ok(())
}

fn state(index: &BuiltIndex) -> Vec<(u32, Vec<u32>, Vec<u32>)> {
    index
        .provider
        .read()
        .unwrap()
        .lists
        .iter()
        .map(|(&id, list)| {
            (
                id,
                list.centroid.iter().map(|x| x.to_bits()).collect(),
                list.members.clone(),
            )
        })
        .collect()
}

#[test]
fn build_and_search_cover_splits_and_optional_reassignment() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let (base, queries) = sample();
        for k in [2, 3] {
            for reassign_count in [0, 8] {
                let config = Build {
                    split_k: NonZeroUsize::new(k).unwrap(),
                    reassign_count,
                    ..config()
                };
                let mut checkpoints = 0;
                let (index, result) = build::build(base.clone(), &config, &mut || {
                    checkpoints += 1;
                    Ok(())
                })
                .await?;
                assert_eq!(checkpoints, base.len().div_ceil(config.batch_size.get()));
                assert_eq!(result.trajectory.len(), checkpoints);
                for (batch, layout) in result.trajectory.iter().enumerate() {
                    assert_eq!(
                        layout.inserted,
                        ((batch + 1) * config.batch_size.get()).min(base.len())
                    );
                }
                assert_eq!(result.lists, 1 + result.split_parents * (k - 1));
                assert!(result.split_calls > 0 && result.split_calls <= checkpoints);
                check_search(&index, &base, &queries, SimilarityMeasure::SquaredL2).await?;
            }
        }
        Ok(())
    })
}

#[test]
fn dropped_build_abandons_routing_and_finished_updates() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let (base, _) = sample();
        let provider = InMemProvider::new(base.clone())?;
        let before = build::layout(&*provider.read()?);
        {
            let mut transaction = provider.build_accessor(&DefaultContext)?;
            transaction.route(0..32, |_, _| {}).await?;
            let (_region, mut update) = transaction.fill(std::iter::once(0)).await?;
            update.retire(0)?;
            update.create(&base[1], &(0..32).collect::<Vec<_>>())?;
            update.finish().await?;
        }
        assert_eq!(before, build::layout(&*provider.read()?));
        let mut transaction = provider.build_accessor(&DefaultContext)?;
        transaction.route(0..32, |_, _| {}).await?;
        transaction.finish().await?;
        assert_eq!(provider.read()?.inserted, 32);
        Ok(())
    })
}

#[test]
fn configuration_rejects_unknown_or_invalid_build_options() -> Result<()> {
    for input in [
        r#"{"split_k":1}"#,
        r#"{"iterations":0}"#,
        r#"{"batch_size":0}"#,
        r#"{"split_threshold":0}"#,
        r#"{"split_k":4,"split_threshold":2}"#,
        r#"{"seed":7}"#,
        r#"{"split_mode":"projection-gemm"}"#,
        r#"{"kmeans_mode":"exact"}"#,
        r#"{"reassignment_mode":"batched"}"#,
        r#"{"num_threads":4}"#,
    ] {
        let parsed = serde_json::from_str::<Build>(input);
        assert!(parsed.is_err() || parsed?.validate().is_err());
    }
    Ok(())
}

#[test]
fn scan_target_configuration_accepts_lists_and_legacy_scalars() -> Result<()> {
    for (input, expected) in [
        (
            r#"{"nprobe":[],"target_scan_ratios":[0.02,0.04,0.06,0.08]}"#,
            vec![0.02, 0.04, 0.06, 0.08],
        ),
        (r#"{"nprobe":[],"target_scan_ratio":0.08}"#, vec![0.08]),
        (r#"{"nprobe":[1],"target_scan_ratio":null}"#, Vec::new()),
    ] {
        let search: SearchPhase = serde_json::from_str(input)?;
        assert_eq!(search.target_scan_ratios, expected);
        let serialized = serde_json::to_value(&search)?;
        assert_eq!(
            serialized["target_scan_ratios"],
            serde_json::json!(expected)
        );
        assert!(serialized.get("target_scan_ratio").is_none());
        search.validate()?;
    }
    for input in [
        r#"{"nprobe":[],"target_scan_ratios":[]}"#,
        r#"{"target_scan_ratios":[0.02,0.0]}"#,
        r#"{"target_scan_ratios":[0.02,1.01]}"#,
        r#"{"target_scan_ratios":[-0.1]}"#,
        r#"{"target_scan_ratios":[null]}"#,
        r#"{"target_scan_ratio":0.08,"target_scan_ratios":[0.04]}"#,
    ] {
        let parsed = serde_json::from_str::<SearchPhase>(input);
        assert!(parsed.is_err() || parsed?.validate().is_err());
    }
    Ok(())
}

#[test]
fn scan_calibration_matches_measured_candidate_work() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let (base, queries) = sample();
        let base: Arc<data::Rows> = Arc::new(
            base.iter()
                .map(|v| metric::prepare_centroid(SimilarityMeasure::CosineNormalized, v))
                .collect(),
        );
        let queries: Vec<_> = queries
            .iter()
            .map(|v| metric::prepare_centroid(SimilarityMeasure::CosineNormalized, v))
            .collect();
        let (index, result) = build::build(base.clone(), &config(), &mut || Ok(())).await?;
        for distance in [
            SimilarityMeasure::SquaredL2,
            SimilarityMeasure::InnerProduct,
            SimilarityMeasure::Cosine,
            SimilarityMeasure::CosineNormalized,
        ] {
            let provider = SearchProvider::new(&*index.provider.read()?, distance);
            let mut counts = Vec::new();
            for nprobe in 1..=result.lists {
                let mut count = 0;
                for query in &queries {
                    count += index
                        .search(&provider, query, NonZeroUsize::MIN, nprobe)
                        .await?
                        .candidates_scanned;
                }
                counts.push(count);
            }
            let ratios = [0.08, 0.02, 0.04, 0.06, 0.08, 0.5, 1.0];
            let probes = provider.calibrate(&queries, &ratios)?;
            assert_eq!(probes.len(), ratios.len());
            for (ratio, actual) in ratios.into_iter().zip(probes) {
                let target = ratio * base.len() as f64 * queries.len() as f64;
                let expected = counts
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        ((**a as f64 - target).abs()).total_cmp(&(**b as f64 - target).abs())
                    })
                    .unwrap()
                    .0
                    + 1;
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    })
}

#[test]
fn full_probe_oracle_accepts_exact_ties_but_rejects_wrong_results() -> Result<()> {
    use diskann::neighbor::Neighbor;
    let base = vec![vec![-1.0], vec![1.0], vec![2.0]];
    let make = |hits: &[(u32, f32)]| SearchResult {
        neighbors: hits.iter().map(|&(id, d)| Neighbor::new(id, d)).collect(),
        candidates_scanned: 3,
        lists_scanned: 1,
    };
    assert!(!validate_full_probe(
        &make(&[(1, 1.0), (0, 1.0)]),
        &[0, 1],
        &base,
        &[0.0],
        SimilarityMeasure::SquaredL2,
    )?);
    assert!(!validate_full_probe(
        &make(&[(1, 1.0)]),
        &[0],
        &base,
        &[0.0],
        SimilarityMeasure::SquaredL2,
    )?);
    for hits in [
        vec![(1, 1.0), (1, 1.0)],
        vec![(0, 1.0), (2, 4.0)],
        vec![(0, 1.0), (99, 1.0)],
        vec![(0, 1.0), (1, f32::from_bits(1.0f32.to_bits() + 1))],
    ] {
        assert!(validate_full_probe(
            &make(&hits),
            &[0, 1],
            &base,
            &[0.0],
            SimilarityMeasure::SquaredL2
        )
        .is_err());
    }
    assert!(validate_full_probe(
        &make(&[(2, 4.0), (0, 1.0)]),
        &[0, 2],
        &base,
        &[0.0],
        SimilarityMeasure::SquaredL2
    )
    .is_err());
    Ok(())
}

#[test]
fn cosine_centroids_are_normalized_before_list_selection() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let base: Arc<data::Rows> = Arc::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]]);
        let (index, _) = build::build(base.clone(), &config(), &mut || Ok(())).await?;
        let mut transaction = index.provider.build_accessor(&DefaultContext)?;
        let (_, mut update) = transaction.fill(std::iter::once(0)).await?;
        update.retire(0)?;
        update.create(&[0.1, 0.0], &[0])?;
        update.create(&[0.8, 0.6], &[1])?;
        update.create(&[0.0, 0.0], &[2])?;
        update.finish().await?;
        transaction.finish().await?;
        let before = state(&index);
        for (distance, expected) in [
            (SimilarityMeasure::SquaredL2, 1),
            (SimilarityMeasure::InnerProduct, 1),
            (SimilarityMeasure::Cosine, 0),
            (SimilarityMeasure::CosineNormalized, 0),
        ] {
            let provider = SearchProvider::new(&*index.provider.read()?, distance);
            let result = index
                .search(&provider, &[1.0, 0.0], NonZeroUsize::MIN, 1)
                .await?;
            assert_eq!(*result.neighbors[0].id(), expected);
            assert_eq!(result.candidates_scanned, 1);
            let truth = data::flat_top_k(&base, &[1.0, 0.0], 3, distance);
            let full = index
                .search(&provider, &[1.0, 0.0], NonZeroUsize::new(3).unwrap(), 3)
                .await?;
            validate_full_probe(&full, &truth, &base, &[1.0, 0.0], distance)?;
            assert!(full.neighbors.iter().all(|hit| hit.distance().is_finite()));
            assert_eq!(
                state(&index),
                before,
                "search preparation changed stored means"
            );
        }
        Ok(())
    })
}

#[test]
fn posting_scores_and_flat_oracle_use_the_selected_metric() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let base: Arc<data::Rows> = Arc::new(vec![vec![2.0, 0.0], vec![0.8, 0.6], vec![-1.0, 0.0]]);
        let (index, _) = build::build(base.clone(), &config(), &mut || Ok(())).await?;
        for (distance, expected, score) in [
            (SimilarityMeasure::SquaredL2, vec![1, 0, 2], 0.4),
            (SimilarityMeasure::InnerProduct, vec![0, 1, 2], -2.0),
            (SimilarityMeasure::Cosine, vec![0, 1, 2], 0.0),
        ] {
            let provider = SearchProvider::new(&*index.provider.read()?, distance);
            let full = index
                .search(&provider, &[1.0, 0.0], NonZeroUsize::new(3).unwrap(), 1)
                .await?;
            assert_eq!(
                full.neighbors
                    .iter()
                    .map(|hit| *hit.id())
                    .collect::<Vec<_>>(),
                expected
            );
            assert!((full.neighbors[0].distance() - score).abs() < 1.0e-6);
            assert_eq!(data::flat_top_k(&base, &[1.0, 0.0], 3, distance), expected);
            validate_full_probe(&full, &expected, &base, &[1.0, 0.0], distance)?;
        }
        Ok(())
    })
}

#[test]
fn centroid_preparation_handles_zero_and_finite_extremes() -> Result<()> {
    for input in [
        vec![3.0, 4.0],
        vec![f32::MAX, f32::MAX],
        vec![f32::from_bits(1), 0.0],
    ] {
        let prepared = metric::prepare_centroid(SimilarityMeasure::CosineNormalized, &input);
        let squared_norm: f64 = prepared.iter().map(|&x| f64::from(x).powi(2)).sum();
        assert!((squared_norm - 1.0).abs() < 1.0e-6);
        for distance in [
            SimilarityMeasure::SquaredL2,
            SimilarityMeasure::InnerProduct,
        ] {
            assert_eq!(metric::prepare_centroid(distance, &input), input);
        }
    }
    let zero = metric::prepare_centroid(SimilarityMeasure::CosineNormalized, &[0.0, 0.0]);
    assert_eq!(zero, vec![0.0, 0.0]);
    assert_eq!(
        metric::evaluate(SimilarityMeasure::CosineNormalized, &[1.0, 0.0], &zero),
        1.0
    );
    assert_eq!(
        metric::evaluate(
            SimilarityMeasure::CosineNormalized,
            &[1.0, 0.0],
            &[0.0, 1.0]
        ),
        1.0
    );
    assert_eq!(
        metric::evaluate(
            SimilarityMeasure::CosineNormalized,
            &[1.0, 0.0],
            &[-1.0, 0.0]
        ),
        2.0
    );
    metric::validate_normalized(&[vec![1.0002, 0.0]], "base")?;
    for row in [vec![0.0, 0.0], vec![0.1, 0.0], vec![f32::NAN, 0.0]] {
        assert!(metric::validate_normalized(&[row], "base").is_err());
    }
    Ok(())
}

#[test]
fn fbv8_configuration_selects_normalized_cosine() -> Result<()> {
    let input: serde_json::Value =
        serde_json::from_str(include_str!("../../perf_test_inputs/ivf-fbv8.json"))?;
    let search: SearchPhase =
        serde_json::from_value(input["jobs"][0]["content"]["search"].clone())?;
    assert_eq!(search.distance, SimilarityMeasure::CosineNormalized);
    assert_eq!(
        serde_json::from_str::<SearchPhase>("{}")?.distance,
        SimilarityMeasure::SquaredL2
    );
    for input in [
        r#"{"distance":"cosnie"}"#,
        r#"{"search_distance":"cosine"}"#,
    ] {
        assert!(serde_json::from_str::<SearchPhase>(input).is_err());
    }
    Ok(())
}

#[test]
fn normalized_cosine_search_matches_flat_truth_after_splits() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let distance = SimilarityMeasure::CosineNormalized;
        let (base, queries) = sample();
        let base: Arc<data::Rows> = Arc::new(
            base.iter()
                .map(|v| metric::prepare_centroid(distance, v))
                .collect(),
        );
        let queries: Vec<_> = queries
            .iter()
            .map(|v| metric::prepare_centroid(distance, v))
            .collect();
        let (index, _) = build::build(base.clone(), &config(), &mut || Ok(())).await?;
        check_search(&index, &base, &queries, distance).await
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
