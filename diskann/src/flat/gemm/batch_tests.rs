/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};

use super::*;
use crate::{
    graph::{
        SearchOutputBuffer,
        glue::{CopyIds, SearchPostProcess},
    },
    provider::HasId,
};

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}
fn pool() -> Arc<ThreadPool> {
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    )
}

fn oracle<T: GemmElement>(
    data: Dataset<'_, T, u64>,
    query: &[T],
    q: usize,
    k: usize,
    metric: Metric,
    filter: &impl CandidateFilter<u64>,
) -> Vec<(usize, u64, f32)> {
    let computer = T::query_distance(query, metric);
    let mut scores: Vec<_> = (0..data.len())
        .filter_map(|i| {
            let id = data.id(i);
            if !filter.is_member(q, id) {
                return None;
            }
            let score = computer.evaluate_similarity(data.data.row(data.row_index(i)));
            (!score.is_nan()).then_some((i, id, score))
        })
        .collect();
    scores.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap().then_with(|| b.0.cmp(&a.0)));
    scores.truncate(k);
    scores
}

fn types_and_metrics<T: GemmElement>(convert: impl Fn(f32) -> T) {
    let pool = pool();
    for dim in [0, 7, 33] {
        let data = rowmajor::Owned::from_fn(113, dim, |rc| {
            convert(((rc.row * 11 + rc.col * 5) % 29) as f32 - 14.0)
        });
        let queries = rowmajor::Owned::from_fn(4, dim, |rc| {
            convert(((rc.row * 7 + rc.col * 13) % 19) as f32 - 9.0)
        });
        let ids: Vec<_> = (0..113).map(|i| (1u64 << 40) + (113 - i) * 31).collect();
        let selected: Vec<_> = (0..109).map(|i| i * 31 % 113).collect();
        let dataset = Dataset::new(data.as_view(), &ids)
            .unwrap()
            .select(Selection::Rows(&selected))
            .unwrap();
        let filter = |q, id| q != 3 && (q == 0 || id % 3 != 0);
        for metric in [
            Metric::InnerProduct,
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
        ] {
            for engine in [Engine::Direct, Engine::Gemm, Engine::Auto] {
                for k in [1, 100, 200] {
                    let options = BatchOptions {
                        engine,
                        gemm_threshold: 1,
                        gemm: GemmOptions {
                            max_workers: nz(3),
                            query_block_size: nz(2),
                            database_block_size: nz(37),
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    let actual = knn_search(
                        dataset,
                        queries.as_view(),
                        nz(k),
                        metric,
                        &filter,
                        &options,
                        &pool,
                    )
                    .unwrap();
                    for (q, result) in actual.iter().enumerate() {
                        let expected = oracle(dataset, queries.row(q), q, k, metric, &filter);
                        assert_eq!(result.neighbors.len(), expected.len());
                        for (neighbor, (_, _, expected_score)) in
                            result.neighbors.iter().zip(&expected)
                        {
                            assert!(filter(q, *neighbor.id()));
                            let tolerance = 3e-5 * expected_score.abs().max(1.0);
                            assert!(
                                (*neighbor.distance() - expected_score).abs() <= tolerance,
                                "metric={metric}, engine={engine:?}, dim={dim}, q={q}, actual={}, expected={expected_score}",
                                neighbor.distance()
                            );
                            let original = ids.iter().position(|id| id == neighbor.id()).unwrap();
                            let score = T::query_distance(queries.row(q), metric)
                                .evaluate_similarity(data.row(original));
                            assert!(
                                (*neighbor.distance() - score).abs() <= 3e-5 * score.abs().max(1.0)
                            );
                        }
                        assert_eq!(
                            result.stats.eligible_candidates,
                            selected.iter().filter(|&&i| filter(q, ids[i])).count() as u64
                        );
                        assert_eq!(result.stats.result_count, result.neighbors.len() as u64);
                        if engine == Engine::Direct {
                            assert_eq!(result.stats.cmps, result.stats.eligible_candidates);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn native_types_metrics_and_filtered_candidates() {
    types_and_metrics::<f32>(|x| x);
    types_and_metrics::<half::f16>(half::f16::from_f32);
    types_and_metrics::<i8>(|x| x as i8);
    types_and_metrics::<u8>(|x| (x + 14.0) as u8);
}

#[test]
fn ties_follow_selection_order_across_partitions() {
    let data = rowmajor::Owned::from_element(9, 3, 1.0);
    let queries = rowmajor::Owned::from_element(2, 3, 1.0);
    let ids = [90u64, 70, 50, 30, 10, 80, 60, 40, 20];
    let rows = [6, 2, 7, 0, 4, 1, 8, 3, 5];
    let data = Dataset::new(data.as_view(), &ids)
        .unwrap()
        .select(Selection::Rows(&rows))
        .unwrap();
    for workers in [1, 3] {
        for threshold in [0, 100] {
            let options = BatchOptions {
                engine: Engine::Gemm,
                gemm: GemmOptions {
                    max_workers: nz(workers),
                    database_block_size: nz(2),
                    collection: SearchOptions {
                        reservoir_min_k: threshold,
                    },
                    ..Default::default()
                },
                ..Default::default()
            };
            for k in [1, 4] {
                let result = knn_search(
                    data,
                    queries.as_view(),
                    nz(k),
                    Metric::InnerProduct,
                    &AllCandidates,
                    &options,
                    &pool(),
                )
                .unwrap();
                let expected: Vec<_> = rows.iter().rev().take(k).map(|&i| ids[i]).collect();
                for row in result {
                    assert_eq!(
                        row.neighbors.iter().map(|n| *n.id()).collect::<Vec<_>>(),
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn l2_cancellation_cosine_zero_and_cached_norms() {
    let data = rowmajor::Owned::try_from_data(
        vec![1e10f32, 1e10, 0.0, 0.0, 3.0, 4.0].into_boxed_slice(),
        3,
        2,
    )
    .unwrap();
    let ids = [9u64, 1, 7];
    let queries = rowmajor::Owned::try_from_data(
        vec![1e10f32 + 1024.0, 1e10, 0.0, 0.0, 3.0, 4.0].into_boxed_slice(),
        3,
        2,
    )
    .unwrap();
    let opts = BatchOptions {
        engine: Engine::Gemm,
        ..Default::default()
    };
    let dataset = Dataset::new(data.as_view(), &ids).unwrap();
    let direct = BatchOptions {
        engine: Engine::Direct,
        ..opts
    };
    let result = knn_search(
        dataset,
        queries.as_view(),
        nz(3),
        Metric::L2,
        &AllCandidates,
        &opts,
        &pool(),
    )
    .unwrap();
    let expected = knn_search(
        dataset,
        queries.as_view(),
        nz(3),
        Metric::L2,
        &AllCandidates,
        &direct,
        &pool(),
    )
    .unwrap();
    assert_eq!(
        result[0].neighbors[0].as_tuple(),
        expected[0].neighbors[0].as_tuple()
    );
    assert!(result[0].stats.recomputations > 0);
    let norms = [2e20, 0.0, 25.0];
    for metric in [Metric::Cosine, Metric::CosineNormalized] {
        let a = knn_search(
            dataset,
            queries.as_view(),
            nz(3),
            metric,
            &AllCandidates,
            &opts,
            &pool(),
        )
        .unwrap();
        let b = knn_search(
            dataset.with_squared_norms(&norms).unwrap(),
            queries.as_view(),
            nz(3),
            metric,
            &AllCandidates,
            &opts,
            &pool(),
        )
        .unwrap();
        for (a, b) in a.iter().zip(&b) {
            for (a, b) in a.neighbors.iter().zip(&b.neighbors) {
                assert_eq!(a.as_tuple(), b.as_tuple());
            }
        }
        assert!(a[1].neighbors.iter().all(|n| *n.distance() == 1.0));
    }
    // Non-normalized input under the normalized hint must not be clamped.
    let result = knn_search(
        dataset,
        queries.as_view(),
        nz(1),
        Metric::CosineNormalized,
        &AllCandidates,
        &opts,
        &pool(),
    )
    .unwrap();
    assert!(*result[2].neighbors[0].distance() < 0.0);
}

#[test]
fn validation_empty_ranges_and_non_finite_scores() {
    let data = rowmajor::Owned::from_element(3, 1, 1.0);
    assert!(Dataset::new(data.as_view(), &[1u64]).is_err());
    let ids = [1u64, 2, 3];
    let dataset = Dataset::new(data.as_view(), &ids).unwrap();
    assert!(dataset.select(Selection::Rows(&[3])).is_err());
    assert!(
        dataset
            .select(Selection::Range { start: 2, end: 1 })
            .is_err()
    );
    assert!(dataset.with_squared_norms(&[1.0]).is_err());
    let opts = BatchOptions::default();
    let bad = BatchOptions {
        min_gemm_density: f32::NAN,
        ..opts
    };
    assert!(
        knn_search(
            dataset,
            data.as_view(),
            nz(1),
            Metric::L2,
            &AllCandidates,
            &bad,
            &pool()
        )
        .is_err()
    );
    let empty = dataset
        .select(Selection::Range { start: 1, end: 1 })
        .unwrap();
    let result = knn_search(
        empty,
        data.as_view(),
        nz(5),
        Metric::L2,
        &AllCandidates,
        &opts,
        &pool(),
    )
    .unwrap();
    assert!(
        result
            .iter()
            .all(|r| r.neighbors.is_empty() && r.stats.cmps == 0)
    );
    let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
    let data = rowmajor::Owned::from_fn(values.len(), 1, |rc| values[rc.row]);
    let queries = rowmajor::Owned::from_element(1, 1, 1.0);
    let ids: Vec<_> = (0..values.len() as u64).collect();
    let dataset = Dataset::new(data.as_view(), &ids).unwrap();
    for metric in [
        Metric::InnerProduct,
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
    ] {
        let a = knn_search(
            dataset,
            queries.as_view(),
            nz(10),
            metric,
            &AllCandidates,
            &BatchOptions {
                engine: Engine::Gemm,
                ..opts
            },
            &pool(),
        )
        .unwrap();
        let b = knn_search(
            dataset,
            queries.as_view(),
            nz(10),
            metric,
            &AllCandidates,
            &BatchOptions {
                engine: Engine::Direct,
                ..opts
            },
            &pool(),
        )
        .unwrap();
        assert_eq!(
            a[0].neighbors
                .iter()
                .map(|n| n.as_tuple())
                .collect::<Vec<_>>(),
            b[0].neighbors
                .iter()
                .map(|n| n.as_tuple())
                .collect::<Vec<_>>()
        );
    }
}

struct DelayedSource {
    inner: MemorySource<f32, u64>,
    reads: usize,
    posts: usize,
    fail: bool,
}
impl HasId for DelayedSource {
    type Id = u64;
}
impl BatchSource for DelayedSource {
    type Element = f32;
    type Error = ANNError;
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
    async fn next_block(&mut self, max: NonZeroUsize) -> ANNResult<Option<OwnedBlock<f32, u64>>> {
        tokio::task::yield_now().await;
        self.reads += 1;
        if self.fail && self.reads == 2 {
            return Err(ANNError::message("injected read failure"));
        }
        Ok(self.inner.next_block(max).await.unwrap())
    }
}

struct Hydrate;
impl SearchPostProcess<DelayedSource, &[f32], u128> for Hydrate {
    type Error = ANNError;
    async fn post_process<I, B>(
        &self,
        source: &mut DelayedSource,
        _query: &[f32],
        candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        I: Iterator<Item = Neighbor<u64>> + Send,
        B: SearchOutputBuffer<u128> + Send + ?Sized,
    {
        tokio::task::yield_now().await;
        source.posts += 1;
        Ok(output.extend(
            candidates
                .filter(|n| n.id() % 2 == 0)
                .map(|n| Neighbor::new((1u128 << 80) + u128::from(*n.id()), *n.distance())),
        ))
    }
}

fn source() -> DelayedSource {
    let data = Arc::new(rowmajor::Owned::from_fn(17, 7, |rc| {
        (rc.row + rc.col) as f32
    }));
    let ids: Arc<[u64]> = (0..17).rev().collect::<Vec<_>>().into();
    DelayedSource {
        inner: MemorySource::new(OwnedBlock::new(data, ids).unwrap()),
        reads: 0,
        posts: 0,
        fail: false,
    }
}

#[tokio::test]
async fn streaming_postprocessing_and_read_failure() {
    let queries = Arc::new(rowmajor::Owned::from_element(3, 7, 1.0));
    let opts = BatchOptions {
        read_block_size: nz(4),
        engine: Engine::Gemm,
        gemm: GemmOptions {
            database_block_size: nz(4),
            max_workers: nz(3),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut source = source();
    let mut output: Vec<Vec<Neighbor<u128>>> = (0..3).map(|_| Vec::new()).collect();
    let stats = knn_search_batch(
        &mut source,
        queries.clone(),
        nz(5),
        Metric::L2,
        Arc::new(|q, id| q != 2 && id != 16),
        &opts,
        pool(),
        Hydrate,
        &mut output,
    )
    .await
    .unwrap();
    assert_eq!(source.posts, 3);
    assert_eq!(source.reads, 6);
    assert_eq!(source.inner.storage().get_row(0).unwrap().0, 16);
    assert!(source.inner.storage().get_row(17).is_none());
    assert_eq!(stats[0].eligible_candidates, 16);
    assert_eq!(stats[2].eligible_candidates, 0);
    for (out, stats) in output.iter().zip(&stats) {
        assert_eq!(stats.result_count as usize, out.len());
    }
    assert_eq!(
        output[0].iter().map(|n| *n.id()).collect::<Vec<_>>(),
        vec![(1u128 << 80) + 14, (1u128 << 80) + 12]
    );
    let mut source = super::batch_tests::source();
    source.fail = true;
    let mut output: Vec<Vec<Neighbor<u64>>> = (0..3).map(|_| Vec::new()).collect();
    let error = knn_search_batch(
        &mut source,
        queries,
        nz(5),
        Metric::L2,
        Arc::new(AllCandidates),
        &opts,
        pool(),
        CopyIds,
        &mut output,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("injected read failure"));
    assert!(output.iter().all(Vec::is_empty));
    assert_eq!(source.posts, 0);
}

#[tokio::test]
async fn streaming_ties_and_worker_panics() {
    use futures_util::FutureExt;
    let data = Arc::new(rowmajor::Owned::from_element(11, 3, 1.0));
    let ids: Arc<[u64]> = vec![8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13].into();
    let queries = Arc::new(rowmajor::Owned::from_element(1, 3, 1.0));
    let opts = BatchOptions {
        read_block_size: nz(4),
        engine: Engine::Gemm,
        gemm: GemmOptions {
            database_block_size: nz(4),
            max_workers: nz(3),
            ..Default::default()
        },
        ..Default::default()
    };
    let block = OwnedBlock::new(data, ids).unwrap();
    assert!(block.clone().with_squared_norms(vec![].into()).is_err());
    let block = block.with_squared_norms(vec![3.0; 11].into()).unwrap();
    let mut source = MemorySource::new(block.clone());
    let mut output = vec![Vec::<Neighbor<u64>>::new()];
    knn_search_batch(
        &mut source,
        queries.clone(),
        nz(4),
        Metric::Cosine,
        Arc::new(AllCandidates),
        &opts,
        pool(),
        CopyIds,
        &mut output,
    )
    .await
    .unwrap();
    assert_eq!(
        output[0].iter().map(|n| *n.id()).collect::<Vec<_>>(),
        [13, 5, 12, 4]
    );
    let mut source = MemorySource::new(block);
    output[0].clear();
    let result = std::panic::AssertUnwindSafe(knn_search_batch(
        &mut source,
        queries,
        nz(4),
        Metric::L2,
        Arc::new(|_: usize, _: u64| -> bool { panic!("injected worker panic") }),
        &opts,
        pool(),
        CopyIds,
        &mut output,
    ))
    .catch_unwind()
    .await;
    assert!(result.is_err());
    assert!(output[0].is_empty());
}

#[test]
fn automatic_dispatch_accounts_for_filter_density() {
    let data = rowmajor::Owned::from_element(64, 32, 1.0);
    let queries = rowmajor::Owned::from_element(4, 32, 1.0);
    let ids: Vec<u64> = (0..64).collect();
    let dataset = Dataset::new(data.as_view(), &ids).unwrap();
    let options = BatchOptions {
        gemm_threshold: 1,
        min_gemm_density: 0.5,
        ..Default::default()
    };
    let dense = knn_search(
        dataset,
        queries.as_view(),
        nz(2),
        Metric::InnerProduct,
        &AllCandidates,
        &options,
        &pool(),
    )
    .unwrap();
    assert!(
        dense
            .iter()
            .all(|r| r.stats.gemm_tiles == 1 && r.stats.direct_tiles == 0)
    );
    let sparse = knn_search(
        dataset,
        queries.as_view(),
        nz(2),
        Metric::InnerProduct,
        &|_: usize, id: u64| id == 5,
        &options,
        &pool(),
    )
    .unwrap();
    assert!(
        sparse
            .iter()
            .all(|r| r.stats.gemm_tiles == 0 && r.stats.cmps == 1)
    );
    let masked = knn_search(
        dataset,
        queries.as_view(),
        nz(2),
        Metric::InnerProduct,
        &|_: usize, id: u64| id == 5,
        &BatchOptions {
            engine: Engine::Gemm,
            ..options
        },
        &pool(),
    )
    .unwrap();
    assert!(
        masked
            .iter()
            .all(|r| r.stats.cmps == 64 && r.stats.eligible_candidates == 1)
    );
}

#[test]
fn floating_scores_agree_with_f64_reference() {
    let data = rowmajor::Owned::from_fn(41, 257, |rc| ((rc.row * 131 + rc.col * 37) as f32).sin());
    let queries = rowmajor::Owned::from_fn(3, 257, |rc| ((rc.row * 93 + rc.col * 17) as f32).cos());
    let ids: Vec<_> = (0..41u64).collect();
    let dataset = Dataset::new(data.as_view(), &ids).unwrap();
    for metric in [
        Metric::InnerProduct,
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
    ] {
        let results = knn_search(
            dataset,
            queries.as_view(),
            nz(41),
            metric,
            &AllCandidates,
            &BatchOptions {
                engine: Engine::Gemm,
                ..Default::default()
            },
            &pool(),
        )
        .unwrap();
        for (q, result) in results.iter().enumerate() {
            for n in &result.neighbors {
                let query = queries.row(q);
                let vector = data.row(*n.id() as usize);
                let dot: f64 = query
                    .iter()
                    .zip(vector)
                    .map(|(&x, &y)| f64::from(x) * f64::from(y))
                    .sum();
                let expected = match metric {
                    Metric::InnerProduct => -dot,
                    Metric::L2 => query
                        .iter()
                        .zip(vector)
                        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                        .sum(),
                    Metric::CosineNormalized => 1.0 - dot,
                    Metric::Cosine => {
                        let qn: f64 = query.iter().map(|&x| f64::from(x).powi(2)).sum();
                        let xn: f64 = vector.iter().map(|&x| f64::from(x).powi(2)).sum();
                        1.0 - dot / (qn * xn).sqrt()
                    }
                };
                assert!(
                    (f64::from(*n.distance()) - expected).abs() < 5e-5 * expected.abs().max(1.0)
                );
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn cpu_work_does_not_block_the_async_executor() {
    use std::sync::{
        Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    let entered = Arc::new(tokio::sync::Notify::new());
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let once = Arc::new(AtomicBool::new(false));
    let worker_entered = entered.clone();
    let worker_gate = gate.clone();
    let worker_once = once.clone();
    let filter = Arc::new(move |_: usize, _: u64| {
        if !worker_once.swap(true, Ordering::SeqCst) {
            worker_entered.notify_one();
            let (lock, condvar) = &*worker_gate;
            let (released, timeout) = condvar
                .wait_timeout_while(
                    lock.lock().unwrap(),
                    std::time::Duration::from_secs(3),
                    |released| !*released,
                )
                .unwrap();
            assert!(
                *released && !timeout.timed_out(),
                "CPU job blocked the async executor"
            );
        }
        true
    });
    let release = tokio::spawn(async move {
        entered.notified().await;
        let (lock, condvar) = &*gate;
        *lock.lock().unwrap() = true;
        condvar.notify_one();
    });
    let mut source = source();
    let queries = Arc::new(rowmajor::Owned::from_element(1, 7, 1.0));
    let mut output = vec![Vec::<Neighbor<u64>>::new()];
    knn_search_batch(
        &mut source,
        queries,
        nz(3),
        Metric::L2,
        filter,
        &BatchOptions::default(),
        pool(),
        CopyIds,
        &mut output,
    )
    .await
    .unwrap();
    release.await.unwrap();
    assert_eq!(output[0].len(), 3);
}

#[tokio::test]
async fn cancellation_retains_worker_storage_and_stops_output() {
    use futures_util::FutureExt;
    use std::sync::{
        Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let entered = Arc::new(tokio::sync::Notify::new());
    let once = Arc::new(AtomicBool::new(false));
    let worker_gate = gate.clone();
    let worker_entered = entered.clone();
    let worker_once = once.clone();
    let filter = Arc::new(move |_: usize, _: u64| {
        if !worker_once.swap(true, Ordering::SeqCst) {
            worker_entered.notify_one();
            let (lock, cv) = &*worker_gate;
            let _guard = cv
                .wait_timeout_while(
                    lock.lock().unwrap(),
                    std::time::Duration::from_secs(3),
                    |b| !*b,
                )
                .unwrap();
        }
        true
    });
    let data = Arc::new(rowmajor::Owned::from_element(8, 3, 1.0));
    let weak = Arc::downgrade(&data);
    let block = OwnedBlock::new(data, (0..8u64).collect::<Vec<_>>().into()).unwrap();
    let mut source = MemorySource::new(block);
    let queries = Arc::new(rowmajor::Owned::from_element(1, 3, 1.0));
    let mut output = vec![Vec::<Neighbor<u64>>::new()];
    let options = BatchOptions::default();
    let mut future = Box::pin(knn_search_batch(
        &mut source,
        queries,
        nz(2),
        Metric::InnerProduct,
        filter,
        &options,
        pool(),
        CopyIds,
        &mut output,
    ));
    assert!(future.as_mut().now_or_never().is_none());
    entered.notified().await;
    drop(future);
    drop(source);
    assert!(output[0].is_empty());
    assert!(
        weak.upgrade().is_some(),
        "in-flight job must retain the block"
    );
    let (lock, cv) = &*gate;
    *lock.lock().unwrap() = true;
    cv.notify_one();
}

struct InvalidSource {
    block: Option<OwnedBlock<f32, u64>>,
    dim: usize,
}
impl HasId for InvalidSource {
    type Id = u64;
}
impl BatchSource for InvalidSource {
    type Element = f32;
    type Error = crate::error::Infallible;
    fn dimensions(&self) -> usize {
        self.dim
    }
    async fn next_block(&mut self, _: NonZeroUsize) -> BlockResult<f32, u64, Self::Error> {
        Ok(self.block.take())
    }
}

#[tokio::test]
async fn source_shape_errors_and_empty_queries() {
    let queries = Arc::new(rowmajor::Owned::from_element(1, 3, 1.0));
    let options = BatchOptions {
        read_block_size: nz(4),
        gemm: GemmOptions {
            database_block_size: nz(4),
            ..Default::default()
        },
        ..Default::default()
    };
    for (rows, dim) in [(0, 3), (5, 3), (2, 4)] {
        let block = OwnedBlock::new(
            Arc::new(rowmajor::Owned::from_element(rows, dim, 1.0)),
            (0..rows as u64).collect::<Vec<_>>().into(),
        )
        .unwrap();
        let mut source = InvalidSource {
            block: Some(block),
            dim: 3,
        };
        let mut output = vec![vec![Neighbor::new(99u64, 42.0)]];
        let err = knn_search_batch(
            &mut source,
            queries.clone(),
            nz(2),
            Metric::L2,
            Arc::new(AllCandidates),
            &options,
            pool(),
            CopyIds,
            &mut output,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("block shape"));
        assert_eq!(output[0][0].as_tuple(), (99, 42.0));
    }
    let mut source = source();
    let empty_queries = Arc::new(rowmajor::Owned::from_element(0, 7, 0.0));
    let mut output: Vec<Vec<Neighbor<u64>>> = Vec::new();
    let stats = knn_search_batch(
        &mut source,
        empty_queries,
        nz(1),
        Metric::L2,
        Arc::new(AllCandidates),
        &options,
        pool(),
        CopyIds,
        &mut output,
    )
    .await
    .unwrap();
    assert!(stats.is_empty());
    assert_eq!(source.reads, 0);
    let err = knn_search_batch(
        &mut source,
        queries,
        nz(1),
        Metric::L2,
        Arc::new(AllCandidates),
        &options,
        pool(),
        CopyIds,
        &mut output,
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("dimensions differ"));
    assert_eq!(source.reads, 0);
}

struct FailPostProcess;
impl SearchPostProcess<DelayedSource, &[f32]> for FailPostProcess {
    type Error = ANNError;
    async fn post_process<I, B>(
        &self,
        source: &mut DelayedSource,
        _: &[f32],
        candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        I: Iterator<Item = Neighbor<u64>> + Send,
        B: SearchOutputBuffer<u64> + Send + ?Sized,
    {
        source.posts += 1;
        if source.posts == 2 {
            return Err(ANNError::message("injected post-processing failure"));
        }
        Ok(output.extend(candidates))
    }
}

#[tokio::test]
async fn postprocessing_failure_is_propagated_after_complete_scan() {
    let mut source = source();
    let queries = Arc::new(rowmajor::Owned::from_element(2, 7, 1.0));
    let mut output = vec![Vec::<Neighbor<u64>>::new(), Vec::new()];
    let error = knn_search_batch(
        &mut source,
        queries,
        nz(2),
        Metric::L2,
        Arc::new(AllCandidates),
        &BatchOptions::default(),
        pool(),
        FailPostProcess,
        &mut output,
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected post-processing failure")
    );
    assert_eq!(source.reads, 2);
    assert_eq!(output[0].len(), 2);
    assert!(output[1].is_empty());
}

struct ContextId;
impl SearchPostProcess<DelayedSource, u64> for ContextId {
    type Error = ANNError;
    async fn post_process<I, B>(
        &self,
        source: &mut DelayedSource,
        context: u64,
        candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        I: Iterator<Item = Neighbor<u64>> + Send,
        B: SearchOutputBuffer<u64> + Send + ?Sized,
    {
        source.posts += 1;
        Ok(output.extend(candidates.map(|n| Neighbor::new(context, *n.distance()))))
    }
}

#[tokio::test]
async fn caller_context_is_preserved_for_identical_query_vectors() {
    let queries = Arc::new(rowmajor::Owned::from_element(2, 7, 1.0));
    let mut source = source();
    let mut output = vec![Vec::<Neighbor<u64>>::new(), Vec::new()];
    let options = BatchOptions::default();
    assert!(
        knn_search_batch_with_contexts(
            &mut source,
            queries.clone(),
            nz(1),
            Metric::L2,
            Arc::new(AllCandidates),
            &options,
            pool(),
            ContextId,
            &[77],
            &mut output
        )
        .await
        .is_err()
    );
    assert_eq!(source.reads, 0);
    knn_search_batch_with_contexts(
        &mut source,
        queries,
        nz(1),
        Metric::L2,
        Arc::new(AllCandidates),
        &options,
        pool(),
        ContextId,
        &[77, 88],
        &mut output,
    )
    .await
    .unwrap();
    assert_eq!(*output[0][0].id(), 77);
    assert_eq!(*output[1][0].id(), 88);
    assert_eq!(source.posts, 2);
}
