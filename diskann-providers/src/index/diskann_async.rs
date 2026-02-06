/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{Config, DiskANNIndex},
    utils::VectorRepr,
};
use diskann_utils::future::AsyncFriendly;

use crate::model::{
    self,
    graph::provider::async_::{
        common::{CreateDeleteProvider, CreateVectorStore, NoDeletes, NoStore},
        inmem::{
            CreateFullPrecision, DefaultProvider, DefaultProviderParameters, DefaultQuant,
            FullPrecisionProvider,
        },
    },
};

/////////////////////////
// Helper Constructors //
/////////////////////////

#[cfg(test)]
pub(crate) fn simplified_builder(
    l_search: usize,
    pruned_degree: usize,
    metric: diskann_vector::distance::Metric,
    dim: usize,
    max_points: usize,
    modify: impl FnOnce(&mut diskann::graph::config::Builder),
) -> ANNResult<(Config, DefaultProviderParameters)> {
    let config = diskann::graph::config::Builder::new_with(
        pruned_degree,
        diskann::graph::config::MaxDegree::default_slack(),
        l_search,
        metric.into(),
        modify,
    )
    .build()?;

    let params = DefaultProviderParameters {
        max_points,
        frozen_points: diskann::utils::ONE,
        metric,
        dim,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: config.max_degree_u32().get(),
    };

    Ok((config, params))
}

pub fn train_pq<Pool>(
    data: diskann_utils::views::MatrixView<f32>,
    num_pq_chunks: usize,
    rng: &mut dyn rand::RngCore,
    pool: Pool,
) -> ANNResult<model::pq::FixedChunkPQTable>
where
    Pool: crate::utils::AsThreadPool,
{
    let dim = data.ncols();
    let pivot_args = model::GeneratePivotArguments::new(
        data.nrows(),
        data.ncols(),
        model::pq::NUM_PQ_CENTROIDS,
        num_pq_chunks,
        5,
        false,
    )?;
    let mut centroid = vec![0.0; dim];
    let mut offsets = vec![0; num_pq_chunks + 1];
    let mut full_pivot_data = vec![0.0; model::pq::NUM_PQ_CENTROIDS * dim];

    model::pq::generate_pq_pivots_from_membuf(
        &pivot_args,
        data.as_slice(),
        &mut centroid,
        &mut offsets,
        &mut full_pivot_data,
        rng,
        &mut (false),
        pool,
    )?;

    model::pq::FixedChunkPQTable::new(
        dim,
        full_pivot_data.into(),
        centroid.into(),
        offsets.into(),
        None,
    )
}

pub type MemoryIndex<T, D = NoDeletes> = Arc<DiskANNIndex<FullPrecisionProvider<T, NoStore, D>>>;

pub type QuantMemoryIndex<T, Q, D = NoDeletes> = Arc<DiskANNIndex<FullPrecisionProvider<T, Q, D>>>;

pub type PQMemoryIndex<T, D = NoDeletes> = QuantMemoryIndex<T, DefaultQuant, D>;

pub type QuantOnlyIndex<Q, D = NoDeletes> = DiskANNIndex<DefaultProvider<NoStore, Q, D>>;

pub fn new_index<T, D>(
    config: Config,
    params: DefaultProviderParameters,
    deleter: D,
) -> ANNResult<MemoryIndex<T, D::Target>>
where
    T: VectorRepr,
    D: CreateDeleteProvider,
    D::Target: AsyncFriendly,
{
    let fp_precursor = CreateFullPrecision::new(params.dim, params.prefetch_cache_line_level);
    let data_provider = DefaultProvider::new_empty(params, fp_precursor, NoStore, deleter)?;
    Ok(Arc::new(DiskANNIndex::new(config, data_provider, None)))
}

pub fn new_quant_index<T, Q, D>(
    config: Config,
    params: DefaultProviderParameters,
    quant: Q,
    deleter: D,
) -> ANNResult<QuantMemoryIndex<T, Q::Target, D::Target>>
where
    T: VectorRepr,
    Q: CreateVectorStore,
    Q::Target: AsyncFriendly,
    D: CreateDeleteProvider,
    D::Target: AsyncFriendly,
{
    let fp_precursor = CreateFullPrecision::new(params.dim, params.prefetch_cache_line_level);
    let data_provider = DefaultProvider::new_empty(params, fp_precursor, quant, deleter)?;
    Ok(Arc::new(DiskANNIndex::new(config, data_provider, None)))
}

pub fn new_quant_only_index<Q, D>(
    config: Config,
    params: DefaultProviderParameters,
    quant: Q,
    deleter: D,
) -> ANNResult<QuantOnlyIndex<Q::Target, D::Target>>
where
    Q: CreateVectorStore,
    Q::Target: AsyncFriendly,
    D: CreateDeleteProvider,
    D::Target: AsyncFriendly,
{
    let data = DefaultProvider::new_empty(params, NoStore, quant, deleter)?;
    Ok(DiskANNIndex::new(config, data, None))
}

///////////
// Tests //
///////////

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        collections::HashSet,
        marker::PhantomData,
        num::{NonZeroU32, NonZeroUsize},
        sync::{Arc, Mutex},
    };

    use crate::storage::VirtualStorageProvider;
    use diskann::{
        graph::{
            self, AdjacencyList, ConsolidateKind, InplaceDeleteMethod, RangeSearchParams,
            SearchParams, StartPointStrategy,
            config::IntraBatchCandidates,
            glue::{AsElement, InplaceDeleteStrategy, InsertStrategy, SearchStrategy, aliases},
            index::{PartitionedNeighbors, QueryLabelProvider, QueryVisitDecision},
            search_output_buffer,
        },
        neighbor::Neighbor,
        provider::{
            AsNeighbor, AsNeighborMut, BuildQueryComputer, DataProvider, DefaultContext, Delete,
            ExecutionContext, Guard, NeighborAccessor, NeighborAccessorMut, SetElement,
        },
        utils::{IntoUsize, ONE, async_tools::VectorIdBoxSlice},
    };
    use diskann_quantization::scalar::train::ScalarQuantizationParameters;
    use diskann_utils::views::Matrix;
    use diskann_vector::{
        DistanceFunction, PureDistanceFunction,
        distance::{Metric, SquaredL2},
    };
    use rand::{distr::Distribution, rngs::StdRng, seq::SliceRandom};
    use rstest::rstest;

    use super::*;
    use crate::{
        model::graph::provider::{
            async_::{
                TableDeleteProviderAsync,
                common::{FullPrecision, Hybrid, NoDeletes, Quantized, TableBasedDeletes},
                inmem::{self, DefaultQuant, SetStartPoints},
            },
            layers::BetaFilter,
        },
        test_utils::{
            assert_range_results_exactly_match, assert_top_k_exactly_match, groundtruth, is_match,
        },
        utils::{self, VectorDataIterator, create_rnd_from_seed_in_tests, file_util},
    };

    // Callbacks for use with `simplified_builder`.
    fn no_modify(_: &mut diskann::graph::config::Builder) {}

    /////////////////////////////////////////
    // Tests from the original async index //
    /////////////////////////////////////////

    /// Convert an iterator of vectors into a single Matrix. All elements in `data` must
    /// have the same length, otherwise this function panics.
    pub(crate) fn squish<'a, To, T, Itr>(data: Itr, dim: usize) -> diskann_utils::views::Matrix<To>
    where
        To: Clone + Default,
        T: Clone + Into<To> + 'a,
        Itr: ExactSizeIterator<Item = &'a Vec<T>> + 'a,
    {
        // Assume that all the vectors in `data` have the same length.
        // If they don't, `copy_from_slice` will panic, so we're double checking.
        let mut mat = diskann_utils::views::Matrix::new(To::default(), data.len(), dim);
        std::iter::zip(mat.row_iter_mut(), data).for_each(|(output, input)| {
            assert_eq!(
                input.len(),
                dim,
                "all elements in data must have the same length"
            );
            std::iter::zip(output.iter_mut(), input.iter()).for_each(|(o, i)| {
                *o = i.clone().into();
            });
        });
        mat
    }

    pub(crate) struct PagedSearch<T> {
        query: Vec<T>,
        groundtruth: Vec<Neighbor<u32>>,
    }

    impl<T> PagedSearch<T> {
        pub(crate) fn new(query: Vec<T>, groundtruth: Vec<Neighbor<u32>>) -> Self {
            Self { query, groundtruth }
        }
    }

    pub(crate) async fn populate_data<DP, Ctx, T>(provider: &DP, context: &Ctx, source: &[Vec<T>])
    where
        Ctx: ExecutionContext,
        DP: DataProvider<Context = Ctx, InternalId = u32, ExternalId = u32> + SetElement<[T]>,
    {
        for (i, v) in source.iter().enumerate() {
            let guard = provider.set_element(context, &(i as u32), v).await.unwrap();
            assert_eq!(
                guard.id(),
                i as u32,
                "populate_data only works properly for providers with the identity mapping"
            );
            guard.complete().await;
        }
    }

    pub(crate) async fn populate_graph<NA>(accessor: &mut NA, source: &[AdjacencyList<u32>])
    where
        NA: AsNeighborMut<Id = u32>,
    {
        for (i, v) in source.iter().enumerate() {
            accessor.set_neighbors(i as u32, v).await.unwrap();
        }
    }

    // Grid generators for different types //
    pub(crate) trait GenerateGrid: Sized {
        /// Generate a synthetic dataset that is a hypercube of point beginning at the
        /// origin and ending at `[size - 1; dim]`.
        ///
        /// This is generally implemented for 1, 3, and 4 dimensions.
        ///
        /// Callers may assume the following about the generated grid:
        ///
        /// 1. The origin will be at position 0.
        /// 2. The terminal point `[size - 1; dim]` will be at the last position.
        /// 3. All points in the grid will exist, generating `dim ^ size` total points.
        fn generate_grid(dim: usize, size: usize) -> Vec<Vec<Self>>;
    }

    impl GenerateGrid for f32 {
        fn generate_grid(dim: usize, size: usize) -> Vec<Vec<Self>> {
            match dim {
                1 => utils::generate_1d_grid_vectors_f32(size as u32),
                3 => utils::generate_3d_grid_vectors_f32(size as u32),
                4 => utils::generate_4d_grid_vectors_f32(size as u32),
                _ => panic!("{}-dimensions is not support for grid-generation", size),
            }
        }
    }

    impl GenerateGrid for i8 {
        fn generate_grid(dim: usize, size: usize) -> Vec<Vec<Self>> {
            match dim {
                1 => utils::generate_1d_grid_vectors_i8(size.try_into().unwrap()),
                3 => utils::generate_3d_grid_vectors_i8(size.try_into().unwrap()),
                4 => utils::generate_4d_grid_vectors_i8(size.try_into().unwrap()),
                _ => panic!("{}-dimensions is not support for grid-generation", size),
            }
        }
    }

    impl GenerateGrid for u8 {
        fn generate_grid(dim: usize, size: usize) -> Vec<Vec<Self>> {
            match dim {
                1 => utils::generate_1d_grid_vectors_u8(size.try_into().unwrap()),
                3 => utils::generate_3d_grid_vectors_u8(size.try_into().unwrap()),
                4 => utils::generate_4d_grid_vectors_u8(size.try_into().unwrap()),
                _ => panic!("{}-dimensions is not support for grid-generation", size),
            }
        }
    }

    #[derive(Debug)]
    struct SearchParameters<Ctx> {
        context: Ctx,
        search_l: usize,
        search_k: usize,
        to_check: usize,
    }

    /// Check the contents of a single search for the query.
    ///
    /// # Arguments
    async fn test_search<DP, S, Q, Checker>(
        index: &DiskANNIndex<DP>,
        parameters: &SearchParameters<DP::Context>,
        strategy: S,
        query: &Q,
        mut checker: Checker,
    ) where
        DP: DataProvider<InternalId = u32>,
        S: SearchStrategy<DP, Q>,
        Q: std::fmt::Debug + Sync + ?Sized,
        Checker: FnMut(usize, (u32, f32)) -> Result<(), Box<dyn std::fmt::Display>>,
    {
        let mut ids = vec![0; parameters.search_k];
        let mut distances = vec![0.0; parameters.search_k];
        let mut result_output_buffer =
            search_output_buffer::IdDistance::new(&mut ids, &mut distances);
        index
            .search(
                &strategy,
                &parameters.context,
                query,
                &SearchParams::new_default(parameters.search_k, parameters.search_l).unwrap(),
                &mut result_output_buffer,
            )
            .await
            .unwrap();

        // Loop over the requested number of results to check, invoking the checker closure.
        //
        // If the checker closure detects an error, embed that error in a more descriptive
        // formatted panic.
        for i in 0..parameters.to_check {
            println!("{ids:?}");
            if let Err(message) = checker(i, (ids[i], distances[i])) {
                panic!(
                    "Check failed for result {} with error: {}. Query = {:?}. Result: ({}, {})",
                    i, message, query, ids[i], distances[i]
                );
            }
        }
    }

    /// Check the contents of a single search for the query.
    ///
    /// # Arguments
    async fn test_multihop_search<DP, S, Q, Checker>(
        index: &DiskANNIndex<DP>,
        parameters: &SearchParameters<DP::Context>,
        strategy: &S,
        query: &Q,
        mut checker: Checker,
        filter: &dyn QueryLabelProvider<DP::InternalId>,
    ) where
        DP: DataProvider<InternalId = u32>,
        S: SearchStrategy<DP, Q>,
        Q: std::fmt::Debug + Sync + ?Sized,
        Checker: FnMut(usize, (u32, f32)) -> Result<(), Box<dyn std::fmt::Display>>,
    {
        let mut ids = vec![0; parameters.search_k];
        let mut distances = vec![0.0; parameters.search_k];
        let mut result_output_buffer =
            search_output_buffer::IdDistance::new(&mut ids, &mut distances);
        index
            .multihop_search(
                strategy,
                &parameters.context,
                query,
                &SearchParams::new_default(parameters.search_k, parameters.search_l).unwrap(),
                &mut result_output_buffer,
                filter,
            )
            .await
            .unwrap();

        // Loop over the requested number of results to check, invoking the checker closure.
        //
        // If the checker closure detects an error, embed that error in a more descriptive
        // formatted panic.
        for i in 0..parameters.to_check {
            println!("{ids:?}");
            if let Err(message) = checker(i, (ids[i], distances[i])) {
                panic!(
                    "Check failed for result {} with error: {}. Query = {:?}. Result: ({}, {})",
                    i, message, query, ids[i], distances[i]
                );
            }
        }
    }

    async fn test_paged_search<DP, S, Q>(
        index: &DiskANNIndex<DP>,
        strategy: S,
        parameters: &SearchParameters<DP::Context>,
        query: &Q,
        groundtruth: &mut Vec<Neighbor<u32>>,
        max_candidates: usize,
    ) where
        DP: DataProvider<InternalId = u32>,
        S: SearchStrategy<DP, Q> + 'static,
        Q: std::fmt::Debug + Send + Sync + ?Sized,
    {
        assert!(max_candidates <= groundtruth.len());
        let mut state = index
            .start_paged_search(strategy, &parameters.context, query, parameters.search_l)
            .await
            .unwrap();

        let mut buffer = vec![Neighbor::<u32>::default(); parameters.search_k];
        let mut iter = 0;
        let mut seen = 0;
        while !groundtruth.is_empty() {
            let count = index
                .next_search_results::<S, Q>(
                    &parameters.context,
                    &mut state,
                    parameters.search_k,
                    &mut buffer,
                )
                .await
                .unwrap();
            for (i, b) in buffer.iter().enumerate().take(count) {
                let m = is_match(groundtruth, *b, 0.01);
                match m {
                    None => {
                        let last = groundtruth.len();
                        let start = last - last.min(10);

                        panic!(
                            "Remaining Groundtruth: {:?}\n\
                             Could not match: {:?} on iteration {}, position {}.\n\
                             Remaining entries: {:?}",
                            &groundtruth[start..],
                            b,
                            iter,
                            i,
                            &buffer[i..],
                        );
                    }
                    Some(j) => groundtruth.remove(j),
                };

                // Check stopping point.
                seen += 1;
                if seen == max_candidates {
                    return;
                }
            }
            iter += 1;
        }
    }

    pub(crate) async fn check_grid_search<DP, T, FS, QS>(
        index: &DiskANNIndex<DP>,
        vectors: &[Vec<T>],
        paged_queries: &[PagedSearch<T>],
        full_strategy: FS,
        quant_strategy: QS,
    ) where
        DP: DataProvider<InternalId = u32, Context = DefaultContext>,
        FS: SearchStrategy<DP, [T]> + Clone + 'static,
        QS: SearchStrategy<DP, [T]> + Clone + 'static,
        T: Default + Clone + Send + Sync + std::fmt::Debug,
    {
        // Assume all vectors have the same length.
        let dim = vectors[0].len();
        // Subtract 1 to compensate for the start point.
        let num_points = vectors.len();

        // This tests full precision and quantized searches.
        //
        // This first test checks that we can traverse the entire graph because the
        // all-zeros query is as far from the entry point as possible.
        let query = vec![T::default(); dim];
        let parameters = SearchParameters {
            context: DefaultContext,
            search_l: 10,
            // Since we are looking at one of the corners of the grid, we retrieve
            // `dim + 1` points. The closest neighbor should have 0 distance, while the
            // next `dim` entries should have an L2 distace of 1.
            search_k: dim + 1,
            // We can check all `dim + 1` entries.
            to_check: dim + 1,
        };

        let checker = |position, (id, distance)| -> Result<(), Box<dyn std::fmt::Display>> {
            if position == 0 {
                if id != 0 {
                    return Err(Box::new("expected the nearest neighbor to be 0"));
                }
                if distance != 0.0 {
                    return Err(Box::new("expected the nearest distance to be 0"));
                }
            } else if distance != 1.0 {
                return Err(Box::new(
                    "expected corner query close neighbor to have distance 1.0",
                ));
            }
            Ok(())
        };

        // Full Precisision
        test_search(
            index,
            &parameters,
            full_strategy.clone(),
            query.as_slice(),
            checker,
        )
        .await;

        // Quantized
        test_search(
            index,
            &parameters,
            quant_strategy.clone(),
            query.as_slice(),
            checker,
        )
        .await;

        // Make sure the start point does not appear in the output.
        let query = vectors.last().unwrap();
        let parameters = SearchParameters {
            to_check: 1,
            ..parameters
        };

        // Make sure the expected nearest distance is accurate.
        assert_eq!(vectors.len(), num_points);

        let checker = |position, (id, distance)| -> Result<(), Box<dyn std::fmt::Display>> {
            assert_eq!(position, 0);
            if id as usize == num_points - 1 {
                return Err(Box::new("start point should not be returned"));
            }
            if id as usize != num_points - 2 {
                return Err(Box::new(format!(
                    "expected {} as the nearest id",
                    num_points - 2
                )));
            }
            if distance != dim as f32 {
                return Err(Box::new(format!("nearest distance should be {}", dim)));
            }
            Ok(())
        };

        // Full Precision
        test_search(
            index,
            &parameters,
            full_strategy.clone(),
            query.as_slice(),
            checker,
        )
        .await;

        // Quantized
        test_search(
            index,
            &parameters,
            quant_strategy.clone(),
            query.as_slice(),
            checker,
        )
        .await;

        // Paged Search
        let parameters = SearchParameters {
            context: DefaultContext,
            search_l: 10,
            // Since we are looking at one of the corners of the grid, we retrieve
            // `dim + 1` points. The closest neighbor should have 0 distance, while the
            // next `dim` entries should have an L2 distace of 1.
            search_k: dim + 1,
            // We can check all `dim + 1` entries.
            to_check: dim + 1,
        };

        // Check paged searches.
        for paged in paged_queries {
            let mut gt = paged.groundtruth.clone();
            let max_candidates = gt.len();
            test_paged_search(
                index,
                full_strategy.clone(),
                &parameters,
                &paged.query,
                &mut gt,
                max_candidates,
            )
            .await;

            let mut gt = paged.groundtruth.clone();
            test_paged_search(
                index,
                quant_strategy.clone(),
                &parameters,
                &paged.query,
                &mut gt,
                max_candidates,
            )
            .await
        }
    }

    #[rstest]
    #[case(1, 100)]
    #[case(3, 7)]
    #[case(4, 5)]
    #[tokio::test]
    async fn grid_search(#[case] dim: usize, #[case] grid_size: usize) {
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, no_modify).unwrap();

        let mut adjacency_lists = match dim {
            1 => utils::generate_1d_grid_adj_list(grid_size as u32),
            3 => utils::genererate_3d_grid_adj_list(grid_size as u32),
            4 => utils::generate_4d_grid_adj_list(grid_size as u32),
            _ => panic!("Unsupported number of dimensions"),
        };
        let mut vectors = f32::generate_grid(dim, grid_size);

        assert_eq!(adjacency_lists.len(), num_points);
        assert_eq!(vectors.len(), num_points);

        // Append an additional item to the input vectors for the start point.
        adjacency_lists.push((num_points as u32 - 1).into());
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();

        let index = new_quant_index::<f32, _, _>(config, parameters, table, NoDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_data(&index.data_provider, &DefaultContext, &vectors).await;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        // The corpus of actual vectors consists of all but the last point, which we use
        // as the start point.
        //
        // So, when we compute the corpus used during groundtruth generation, we take all
        // but this last point.
        let corpus: diskann_utils::views::Matrix<f32> =
            squish(vectors.iter().take(num_points), dim);

        let mut paged_tests = Vec::new();

        // Test with the zero query.
        let query = vec![0.0; dim];
        let gt = groundtruth(corpus.as_view(), &query, |a, b| SquaredL2::evaluate(a, b));
        paged_tests.push(PagedSearch::new(query, gt));

        // Test with the start point to ensure it is filtered out.
        let query = vectors.last().unwrap();
        let gt = groundtruth(corpus.as_view(), query, |a, b| SquaredL2::evaluate(a, b));
        paged_tests.push(PagedSearch::new(query.clone(), gt));

        check_grid_search(
            &index,
            &vectors,
            &paged_tests,
            FullPrecision,
            Hybrid::new(None),
        )
        .await;
    }

    const IBC_NONE: IntraBatchCandidates = IntraBatchCandidates::None;
    const IBC_ALL: IntraBatchCandidates = IntraBatchCandidates::All;

    #[rstest]
    #[tokio::test]
    async fn grid_search_with_build<T>(
        #[values(PhantomData::<f32>, PhantomData::<i8>, PhantomData::<u8>)] _v: PhantomData<T>,
        #[values((1, 100), (3, 7), (4, 5))] dim_and_size: (usize, usize),
        #[values(IBC_NONE, IBC_ALL)] intra_batch_candidates: IntraBatchCandidates,
    ) where
        T: VectorRepr + GenerateGrid + Into<f32>,
    {
        let dim = dim_and_size.0;
        let grid_size = dim_and_size.1;

        let l = 10;

        // NOTE: Be careful changing `max_degree`. It needs to be high enough that the
        // graph is navigable, but low enough that the batch parallel handling inside
        // `multi_insert` is needed for the multi-insert graph to be navigable.
        //
        // With the current configured values, removing the other elements in the batch
        // from the visited set during `multi_insert` results in a graph failure.
        let max_degree = 2 * dim;
        let minibatch_par = 10;

        let max_fp_vecs_per_prune = Some(2);
        let hybrid = Hybrid::new(max_fp_vecs_per_prune);

        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, |p| {
                p.max_minibatch_par(minibatch_par)
                    .intra_batch_candidates(intra_batch_candidates);
            })
            .unwrap();

        let mut vectors = T::generate_grid(dim, grid_size);
        assert_eq!(vectors.len(), num_points);

        // This is a little subtle, but we need `vectors` to contain the start point as
        // its last element, but we **don't** want to include it in the index build.
        //
        // This basically means that we need to be careful with out index initialization.
        vectors.push(vec![
            <T as num_traits::FromPrimitive>::from_usize(grid_size)
                .unwrap();
            dim
        ]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();

        // Initialize an index for a new round of building.
        let init_index = || {
            let index = new_quant_index::<T, _, _>(
                config.clone(),
                parameters.clone(),
                table.clone(),
                NoDeletes,
            )
            .unwrap();
            index
                .provider()
                .set_start_points(std::iter::once(vectors.last().unwrap().as_slice()))
                .unwrap();
            index
        };

        // Build with full-precision single insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            for (i, v) in vectors.iter().take(num_points).enumerate() {
                index
                    .insert(FullPrecision, &ctx, &(i as u32), v.as_slice())
                    .await
                    .unwrap();
            }

            check_grid_search(&index, &vectors, &[], FullPrecision, hybrid).await;
        }

        // Build with quantized single insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            for (i, v) in vectors.iter().take(num_points).enumerate() {
                index
                    .insert(hybrid, &ctx, &(i as u32), v.as_slice())
                    .await
                    .unwrap();
            }

            check_grid_search(&index, &vectors, &[], FullPrecision, hybrid).await;
        }

        // Build with full-precision multi-insert
        {
            let index = init_index();
            let ctx = DefaultContext;

            let mut itr = vectors
                .iter()
                .take(num_points)
                .enumerate()
                .map(|(id, v)| VectorIdBoxSlice::new(id as u32, v.as_slice().into()));

            // Partition by `max_minibatch_par`.
            loop {
                let v: Vec<_> = itr.by_ref().take(2 * minibatch_par).collect();
                if v.is_empty() {
                    break;
                }

                index
                    .multi_insert(FullPrecision, &ctx, v.into())
                    .await
                    .unwrap();
            }

            check_grid_search(&index, &vectors, &[], FullPrecision, hybrid).await;
        }

        // Build with quantized multi-insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            let batch: Box<[_]> = vectors
                .iter()
                .take(num_points)
                .enumerate()
                .map(|(id, v)| VectorIdBoxSlice::new(id as u32, v.as_slice().into()))
                .collect();

            index.multi_insert(hybrid, &ctx, batch).await.unwrap();

            check_grid_search(&index, &vectors, &[], FullPrecision, hybrid).await;
        }
    }

    ///////////////////
    // Sphere Search //
    ///////////////////

    trait GenerateSphericalData: Sized {
        /// Generate spherically distributed data with a single start point appended to the
        /// end.
        fn generate_spherical(
            num: usize,
            dim: usize,
            radius: f32,
            rng: &mut StdRng,
        ) -> Vec<Vec<Self>>;
    }

    macro_rules! impl_generate_spherical_data {
        ($T:ty) => {
            impl GenerateSphericalData for $T {
                fn generate_spherical(
                    num: usize,
                    dim: usize,
                    radius: f32,
                    rng: &mut StdRng,
                ) -> Vec<Vec<Self>> {
                    use crate::utils::math_util;

                    let mut vectors =
                        math_util::generate_vectors_with_norm::<$T>(num, dim, radius, rng).unwrap();
                    assert_eq!(vectors.len(), num);

                    let mut start_point = vec![<$T>::default(); dim];
                    start_point[0] = radius as $T;
                    vectors.push(start_point);
                    vectors
                }
            }
        };
    }

    impl_generate_spherical_data!(f32);
    impl_generate_spherical_data!(i8);
    impl_generate_spherical_data!(u8);

    struct SphericalTest {
        num: usize,
        dim: usize,
        radius: f32,
        num_queries: usize,
    }

    async fn test_spherical_data_impl<T, S>(
        strategy: S,
        metric: Metric,
        params: SphericalTest,
        rng: &mut StdRng,
    ) where
        T: VectorRepr + GenerateSphericalData + Into<f32>,
        S: InsertStrategy<FullPrecisionProvider<T, DefaultQuant>, [T]>
            + SearchStrategy<FullPrecisionProvider<T, DefaultQuant>, [T]>
            + Clone
            + 'static,
        rand::distr::StandardUniform: Distribution<T>,
    {
        // Unpack arguments.
        let SphericalTest {
            num,
            dim,
            radius,
            num_queries,
        } = params;

        let ctx = &DefaultContext;
        let l_search = 10;

        let (config, params) =
            simplified_builder(l_search, 3 * dim, metric, dim, num, no_modify).unwrap();

        let data = T::generate_spherical(num, dim, radius, rng);
        let table = {
            let train_data: diskann_utils::views::Matrix<f32> = squish(data.iter(), dim);
            train_pq(train_data.as_view(), 2.min(dim), rng, 1usize).unwrap()
        };

        let index = new_quant_index::<T, _, _>(config, params, table, NoDeletes).unwrap();
        index
            .provider()
            .set_start_points(std::iter::once(data[num].as_slice()))
            .unwrap();
        for (i, v) in data.iter().take(num).enumerate() {
            index
                .insert(strategy.clone(), ctx, &(i as u32), v.as_slice())
                .await
                .unwrap();
        }

        let distribution = rand::distr::StandardUniform {};
        let data = squish::<T, T, _>(data.iter().take(num), dim);
        let distance = T::distance(metric, None);

        let parameters = SearchParameters {
            context: DefaultContext,
            search_l: 20,
            search_k: 10,
            to_check: 10,
        };

        for _ in 0..num_queries {
            let query: Vec<T> = (0..dim).map(|_| distribution.sample(rng)).collect();
            let mut gt = groundtruth(data.as_view(), &query, |a, b| {
                distance.evaluate_similarity(a, b)
            });

            let checker = |position, (id, distance)| -> Result<(), Box<dyn std::fmt::Display>> {
                let expected: Neighbor<u32> = gt[gt.len() - 1 - position];
                if id != expected.id {
                    // We can allow it if the distance is the same.
                    if distance == expected.distance {
                        Ok(())
                    } else {
                        Err(Box::new(format!(
                            "expected neighbor {:?}, but found {}",
                            expected, id
                        )))
                    }
                } else if distance != expected.distance {
                    Err(Box::new(format!(
                        "expected neighbor {:?}, but found {}",
                        expected, distance
                    )))
                } else {
                    Ok(())
                }
            };

            // Direct search.
            test_search(
                &index,
                &parameters,
                strategy.clone(),
                query.as_slice(),
                checker,
            )
            .await;

            // Paged Search.
            test_paged_search(
                &index,
                strategy.clone(),
                &parameters,
                query.as_slice(),
                &mut gt,
                3 * parameters.search_k,
            )
            .await;
        }
    }

    const PF32: PhantomData<f32> = PhantomData;
    const PU8: PhantomData<u8> = PhantomData;
    const PI8: PhantomData<i8> = PhantomData;

    #[rstest]
    #[case(PF32, FullPrecision, Metric::L2, 100, 4, 1.5)]
    #[case(PF32, Hybrid::new(Some(6)), Metric::L2, 100, 4, 1.5)]
    #[case(PF32, FullPrecision, Metric::InnerProduct, 93, 5, 543.5)]
    #[case(PF32, Hybrid::new(Some(8)), Metric::InnerProduct, 93, 5, 543.3)]
    #[case(PF32, FullPrecision, Metric::Cosine, 77, 7, 2.5)]
    #[case(PF32, Hybrid::new(Some(32)), Metric::Cosine, 77, 7, 2.5)]
    #[case(PU8, FullPrecision, Metric::L2, 100, 7, 43.0)]
    #[case(PU8, FullPrecision, Metric::Cosine, 93, 5, 46.0)]
    #[case(PU8, FullPrecision, Metric::InnerProduct, 77, 6, 47.0)]
    #[case(PI8, FullPrecision, Metric::L2, 100, 7, 43.0)]
    #[case(PI8, FullPrecision, Metric::Cosine, 93, 5, 46.0)]
    #[case(PI8, FullPrecision, Metric::InnerProduct, 77, 6, 47.0)]
    #[tokio::test]
    async fn test_sphere_search<T, S>(
        #[case] ty: PhantomData<T>,
        #[case] strategy: S,
        #[case] metric: Metric,
        #[case] num: usize,
        #[case] dim: usize,
        #[case] radius: f32,
    ) where
        T: VectorRepr + GenerateSphericalData + Into<f32>,
        S: InsertStrategy<FullPrecisionProvider<T, DefaultQuant>, [T]>
            + SearchStrategy<FullPrecisionProvider<T, DefaultQuant>, [T]>
            + Clone
            + 'static,
        rand::distr::StandardUniform: Distribution<T>,
    {
        use std::hash::{DefaultHasher, Hash, Hasher};

        // Construct the RNG seed by hashing all the arguments.
        let rng = &mut {
            let mut s = DefaultHasher::new();
            ty.hash(&mut s);
            num.hash(&mut s);
            dim.hash(&mut s);
            create_rnd_from_seed_in_tests(s.finish())
        };

        let num_queries = 4;
        test_spherical_data_impl::<T, _>(
            strategy,
            metric,
            SphericalTest {
                num,
                dim,
                radius,
                num_queries,
            },
            rng,
        )
        .await;
    }

    ////////////////////
    // Beta Filtering //
    ////////////////////

    // We test beta-filtering by reusing grid search and creating a filter that accepts even
    // numbered candidates but not odd numbered candidates.
    //
    // Much of the existing checking machinery can be reused. We just need to supply a
    // slightly modified groundtruth list.
    #[derive(Debug)]
    struct EvenFilter;

    impl QueryLabelProvider<u32> for EvenFilter {
        fn is_match(&self, id: u32) -> bool {
            id.is_multiple_of(2)
        }
    }

    async fn test_beta_filtering(
        filter: Arc<dyn QueryLabelProvider<u32>>,
        dim: usize,
        grid_size: usize,
    ) {
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, no_modify).unwrap();

        let mut adjacency_lists = utils::genererate_3d_grid_adj_list(grid_size as u32);
        let mut vectors = f32::generate_grid(dim, grid_size);

        assert_eq!(adjacency_lists.len(), num_points);
        assert_eq!(vectors.len(), num_points);

        // Append an additional item to the input vectors for the start point.
        adjacency_lists.push((num_points as u32 - 1).into());
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();

        let index = new_quant_index::<f32, _, _>(config, parameters, table, NoDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_data(&index.data_provider, &DefaultContext, &vectors).await;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        let beta = 0.5;

        let corpus: diskann_utils::views::Matrix<f32> =
            squish(vectors.iter().take(num_points), dim);
        let query = vec![grid_size as f32; dim];

        // The strategy we use here for checking is that we pull in a lot of neighbors and
        // then walk through the list, verifying monotonicity and that the filter was
        // applied properly.
        let parameters = SearchParameters {
            context: DefaultContext,
            search_l: 40,
            search_k: 20,
            to_check: 20,
        };

        // Compute the raw groundtruth, then recalculate using `beta` applied to the
        // even indices.
        let gt = {
            let mut gt = groundtruth(corpus.as_view(), &query, |a, b| SquaredL2::evaluate(a, b));
            for n in gt.iter_mut() {
                if filter.is_match(n.id) {
                    n.distance *= beta;
                }
            }
            gt.sort_unstable_by(|a, b| a.cmp(b).reverse());
            gt
        };

        // Clone the base groundtruth so we don't need to recompute every time.
        let mut gt_clone = gt.clone();
        let strategy = BetaFilter::new(FullPrecision, filter.clone(), beta);
        test_search(
            &index,
            &parameters,
            strategy.clone(),
            query.as_slice(),
            |_, (id, distance)| -> Result<(), Box<dyn std::fmt::Display>> {
                if let Some(position) = is_match(&gt_clone, Neighbor::new(id, distance), 0.0) {
                    gt_clone.remove(position);
                    Ok(())
                } else {
                    if id.into_usize() == num_points + 1 {
                        return Err(Box::new("The start point should not be returned"));
                    }
                    Err(Box::new("mismatch"))
                }
            },
        )
        .await;

        let paged_parameters = SearchParameters {
            search_k: 10,
            search_l: 40,
            ..parameters
        };
        // Because of how Beta filtering interacts with search, we tend to lose accuracy
        // as we get deep in the paged search stack. So, this limits how far we look.
        //
        // In debug mode, an underflow will cause a panic, so we can be sure that we always
        // test for 100 candidates.
        test_paged_search(
            &index,
            strategy,
            &paged_parameters,
            query.as_slice(),
            &mut gt.clone(),
            60,
        )
        .await;
    }

    #[tokio::test]
    async fn test_even_filtering_beta() {
        let filter = Arc::new(EvenFilter);
        test_beta_filtering(filter, 3, 7).await;
    }

    /////////////////////////
    // Multi-Hop Filtering //
    /////////////////////////

    async fn test_multihop_filtering(
        filter: &dyn QueryLabelProvider<u32>,
        dim: usize,
        grid_size: usize,
    ) {
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, no_modify).unwrap();

        let mut adjacency_lists = utils::genererate_3d_grid_adj_list(grid_size as u32);
        let mut vectors = f32::generate_grid(dim, grid_size);

        assert_eq!(adjacency_lists.len(), num_points);
        assert_eq!(vectors.len(), num_points);

        // Append an additional item to the input vectors for the start point.
        adjacency_lists.push((num_points as u32 - 1).into());
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();

        let index = new_quant_index::<f32, _, _>(config, parameters, table, NoDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_data(&index.data_provider, &DefaultContext, &vectors).await;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        let corpus: diskann_utils::views::Matrix<f32> =
            squish(vectors.iter().take(num_points), dim);
        let query = vec![grid_size as f32; dim];

        // The strategy we use here for checking is that we pull in a lot of neighbors and
        // then walk through the list, verifying monotonicity and that the filter was
        // applied properly.
        let parameters = SearchParameters {
            context: DefaultContext,
            search_l: 40,
            search_k: 20,
            to_check: 20,
        };

        // Compute the raw groundtruth, then screen out any points that don't match the filter
        let gt = {
            let mut gt = groundtruth(corpus.as_view(), &query, |a, b| SquaredL2::evaluate(a, b));
            gt.retain(|n| filter.is_match(n.id));
            gt.sort_unstable_by(|a, b| a.cmp(b).reverse());
            gt
        };

        // Clone the base groundtruth so we don't need to recompute every time.
        let mut gt_clone = gt.clone();
        let strategy = FullPrecision;

        test_multihop_search(
            &index,
            &parameters,
            &strategy.clone(),
            query.as_slice(),
            |_, (id, distance)| -> Result<(), Box<dyn std::fmt::Display>> {
                if let Some(position) = is_match(&gt_clone, Neighbor::new(id, distance), 0.0) {
                    gt_clone.remove(position);
                    Ok(())
                } else {
                    if id.into_usize() == num_points + 1 {
                        return Err(Box::new("The start point should not be returned"));
                    }
                    Err(Box::new("mismatch"))
                }
            },
            filter,
        )
        .await;
    }

    #[tokio::test]
    async fn test_even_filtering_multihop() {
        test_multihop_filtering(&EvenFilter, 3, 7).await;
    }

    /// Metrics tracked by [`CallbackFilter`] for test validation.
    #[derive(Debug, Clone, Default)]
    struct CallbackMetrics {
        /// Total number of callback invocations.
        total_visits: usize,
        /// Number of candidates that were rejected.
        rejected_count: usize,
        /// Number of candidates that had distance adjusted.
        adjusted_count: usize,
        /// All visited candidate IDs in order.
        visited_ids: Vec<u32>,
    }

    #[derive(Debug)]
    struct CallbackFilter {
        blocked: u32,
        adjusted: u32,
        adjustment_factor: f32,
        metrics: Mutex<CallbackMetrics>,
    }

    impl CallbackFilter {
        fn new(blocked: u32, adjusted: u32, adjustment_factor: f32) -> Self {
            Self {
                blocked,
                adjusted,
                adjustment_factor,
                metrics: Mutex::new(CallbackMetrics::default()),
            }
        }

        fn hits(&self) -> Vec<u32> {
            self.metrics
                .lock()
                .expect("callback metrics mutex should not be poisoned")
                .visited_ids
                .clone()
        }

        fn metrics(&self) -> CallbackMetrics {
            self.metrics
                .lock()
                .expect("callback metrics mutex should not be poisoned")
                .clone()
        }
    }

    impl QueryLabelProvider<u32> for CallbackFilter {
        fn is_match(&self, _: u32) -> bool {
            true
        }

        fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
            let mut metrics = self
                .metrics
                .lock()
                .expect("callback metrics mutex should not be poisoned");

            metrics.total_visits += 1;
            metrics.visited_ids.push(neighbor.id);

            if neighbor.id == self.blocked {
                metrics.rejected_count += 1;
                return QueryVisitDecision::Reject;
            }
            if neighbor.id == self.adjusted {
                metrics.adjusted_count += 1;
                let adjusted =
                    Neighbor::new(neighbor.id, neighbor.distance * self.adjustment_factor);
                return QueryVisitDecision::Accept(adjusted);
            }
            QueryVisitDecision::Accept(neighbor)
        }
    }

    #[tokio::test]
    async fn test_multihop_callback_enforces_filtering() {
        // Test configuration
        let dim = 3;
        let grid_size: usize = 5;
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, no_modify).unwrap();

        let mut adjacency_lists = utils::genererate_3d_grid_adj_list(grid_size as u32);
        let mut vectors = f32::generate_grid(dim, grid_size);

        adjacency_lists.push((num_points as u32 - 1).into());
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim),
            &mut create_rnd_from_seed_in_tests(0xdd81b895605c73d4),
            1usize,
        )
        .unwrap();

        let index = new_quant_index::<f32, _, _>(config, parameters, table, NoDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_data(&index.data_provider, &DefaultContext, &vectors).await;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        let corpus: diskann_utils::views::Matrix<f32> =
            squish(vectors.iter().take(num_points), dim);
        let query = vec![grid_size as f32; dim];

        let parameters = SearchParameters {
            context: DefaultContext,
            search_l: 40,
            search_k: 20,
            to_check: 10,
        };

        let mut ids = vec![0; parameters.search_k];
        let mut distances = vec![0.0; parameters.search_k];
        let mut result_output_buffer =
            search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        let blocked = (num_points - 2) as u32;
        let adjusted = (num_points - 1) as u32;

        // Compute baseline groundtruth for validation
        let mut baseline_gt =
            groundtruth(corpus.as_view(), &query, |a, b| SquaredL2::evaluate(a, b));
        baseline_gt.sort_unstable_by(|a, b| a.cmp(b).reverse());

        assert!(
            baseline_gt.iter().any(|n| n.id == blocked),
            "blocked candidate must exist in groundtruth"
        );

        let baseline_adjusted_distance = baseline_gt
            .iter()
            .find(|n| n.id == adjusted)
            .expect("adjusted node should exist in groundtruth")
            .distance;

        let filter = CallbackFilter::new(blocked, adjusted, 0.5);

        let stats = index
            .multihop_search(
                &FullPrecision,
                &parameters.context,
                query.as_slice(),
                &SearchParams::new_default(parameters.search_k, parameters.search_l).unwrap(),
                &mut result_output_buffer,
                &filter,
            )
            .await
            .unwrap();

        // Retrieve callback metrics for detailed validation
        let callback_metrics = filter.metrics();

        // Validate search statistics
        assert!(
            stats.result_count >= parameters.to_check as u32,
            "expected at least {} results, got {}",
            parameters.to_check,
            stats.result_count
        );

        // Validate callback was invoked and tracked the blocked candidate
        assert!(
            callback_metrics.total_visits > 0,
            "callback should have been invoked at least once"
        );
        assert!(
            filter.hits().contains(&blocked),
            "callback must evaluate the blocked candidate (visited {} candidates)",
            callback_metrics.total_visits
        );
        assert_eq!(
            callback_metrics.rejected_count, 1,
            "exactly one candidate (blocked={}) should be rejected",
            blocked
        );

        // Validate blocked candidate is excluded from results
        let produced = stats.result_count as usize;
        let inspected = produced.min(parameters.to_check);
        assert!(
            !ids.iter().take(inspected).any(|&id| id == blocked),
            "blocked candidate {} should not appear in final results (found in: {:?})",
            blocked,
            &ids[..inspected]
        );

        // Validate distance adjustment was applied
        assert!(
            callback_metrics.adjusted_count >= 1,
            "adjusted candidate {} should have been visited",
            adjusted
        );

        let adjusted_idx = ids
            .iter()
            .take(inspected)
            .position(|&id| id == adjusted)
            .expect("adjusted candidate should be present in results");
        let expected_distance = baseline_adjusted_distance * 0.5;
        assert!(
            (distances[adjusted_idx] - expected_distance).abs() < 1e-5,
            "callback should adjust distances before ranking: \
             expected {:.6}, got {:.6} (baseline: {:.6}, factor: 0.5)",
            expected_distance,
            distances[adjusted_idx],
            baseline_adjusted_distance
        );

        // Log metrics for debugging/review
        println!(
            "test_multihop_callback_enforces_filtering metrics:\n\
             - total callback visits: {}\n\
             - rejected count: {}\n\
             - adjusted count: {}\n\
             - search hops: {}\n\
             - search comparisons: {}\n\
             - result count: {}",
            callback_metrics.total_visits,
            callback_metrics.rejected_count,
            callback_metrics.adjusted_count,
            stats.hops,
            stats.cmps,
            stats.result_count
        );
    }

    //////////////
    // Deletion //
    //////////////

    async fn setup_inplace_delete_test() -> Arc<TestIndex> {
        let dim = 1;
        let (config, parameters) = simplified_builder(
            10,         // l_search
            3,          // max_degree
            Metric::L2, // metric
            dim,        // dim
            5,          // max_points
            no_modify,
        )
        .unwrap();

        let pqtable = model::pq::FixedChunkPQTable::new(
            dim,
            Box::new([0.0]),
            Box::new([0.0]),
            Box::new([0, 1]),
            None,
        )
        .unwrap();

        let index =
            new_quant_index::<f32, _, _>(config, parameters, pqtable, TableBasedDeletes).unwrap();
        let mut neighbor_accessor = index.provider().neighbors();
        // build graph
        let adjacency_lists = [
            AdjacencyList::from_iter_untrusted([2, 3]),
            AdjacencyList::from_iter_untrusted([2, 3]),
            AdjacencyList::from_iter_untrusted([1, 4]),
            AdjacencyList::from_iter_untrusted([2, 4]),
            AdjacencyList::from_iter_untrusted([1, 3]),
        ];
        populate_graph(&mut neighbor_accessor, &adjacency_lists).await;

        index
    }

    #[tokio::test]
    async fn test_return_refs_to_deleted_vertex() {
        let index = setup_inplace_delete_test().await;

        // Expected outcome:
        // * Index 0 is unchanged because it doesn't contain an edge to 1
        // * Index 2's adjacency list should be changed to remove index 1.
        // * Index 4's adjacency list should be changed to remove index 1.
        //
        // Indices 2 and 4 should be returned.

        let candidates: Vec<u32> = vec![0, 2, 4];

        let ret_list = index
            .return_refs_to_deleted_vertex(&mut index.provider().neighbors(), 1, &candidates)
            .await
            .unwrap();

        // Check that the return list contains only candidates 2 and 4.
        assert_eq!(&ret_list, &[2, 4]);
    }

    #[tokio::test]
    async fn test_is_any_neighbor_deleted() {
        let dim = 1;
        let (config, parameters) = simplified_builder(
            10,         // l_search
            3,          // max_degree
            Metric::L2, // metric
            dim,        // dim
            5,          // max_points
            no_modify,
        )
        .unwrap();

        let pqtable = model::pq::FixedChunkPQTable::new(
            dim,
            Box::new([0.0]),
            Box::new([0.0]),
            Box::new([0, 1]),
            None,
        )
        .unwrap();

        let index =
            new_quant_index::<f32, _, _>(config, parameters, pqtable, TableBasedDeletes).unwrap();
        let mut neighbor_accessor = index.provider().neighbors();
        //build graph
        let adjacency_lists = [
            AdjacencyList::from_iter_untrusted([2, 3, 1]),
            AdjacencyList::from_iter_untrusted([2, 3, 4]),
            AdjacencyList::from_iter_untrusted([0, 1, 4]),
            AdjacencyList::from_iter_untrusted([2, 4, 0]),
            AdjacencyList::from_iter_untrusted([0, 3, 2]),
        ];

        let ctx = DefaultContext;
        populate_graph(&mut neighbor_accessor, &adjacency_lists).await;

        // delete id number 3
        // FIXME: Provider an interface at the index level!.
        index
            .data_provider
            .delete(&ctx, &3_u32)
            .await
            .expect("Error in delete");

        // expected outcome: adjacency lists 0, 1, 4 should return true
        // adjacency lists 2, 3 should return false

        let neighbor_accessor = &mut index.provider().neighbors();
        let msg = "Error in is_any_neighbor_deleted";
        assert!(
            (index.is_any_neighbor_deleted(&ctx, neighbor_accessor, 0))
                .await
                .expect(msg)
        );
        assert!(
            (index.is_any_neighbor_deleted(&ctx, neighbor_accessor, 1))
                .await
                .expect(msg)
        );
        assert!(
            !(index.is_any_neighbor_deleted(&ctx, neighbor_accessor, 2))
                .await
                .expect(msg)
        );
        assert!(
            !(index.is_any_neighbor_deleted(&ctx, neighbor_accessor, 3))
                .await
                .expect(msg)
        );
        assert!(
            (index.is_any_neighbor_deleted(&ctx, neighbor_accessor, 4))
                .await
                .expect(msg)
        );
    }

    #[tokio::test]
    async fn test_drop_deleted_neighbors() {
        let dim = 1;
        let (config, parameters) = simplified_builder(
            10,         // l_search
            3,          // max_degree
            Metric::L2, // metric
            dim,        // dim
            5,          // max_points
            no_modify,
        )
        .unwrap();

        let pqtable = model::pq::FixedChunkPQTable::new(
            dim,
            Box::new([0.0]),
            Box::new([0.0]),
            Box::new([0, 1]),
            None,
        )
        .unwrap();

        let index =
            new_quant_index::<f32, _, _>(config, parameters, pqtable, TableBasedDeletes).unwrap();

        //build graph
        let adjacency_lists = [
            AdjacencyList::from_iter_untrusted([2, 3, 1]),
            AdjacencyList::from_iter_untrusted([2, 3, 4]),
            AdjacencyList::from_iter_untrusted([0, 1, 4]),
            AdjacencyList::from_iter_untrusted([2, 4, 0]),
            AdjacencyList::from_iter_untrusted([0, 3, 2]),
        ];

        let neighbor_accessor = &mut index.provider().neighbors();
        let ctx = DefaultContext;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        // delete id number 3
        // FIXME: Provider an interface at the index level!.
        index
            .data_provider
            .delete(&ctx, &3_u32)
            .await
            .expect("Error in delete");

        let drop_msg = "Error in drop_deleted_neighbors";
        let adj_msg = "Error in get_neighbors";

        // call drop_deleted_neighbors on vertex 0 with check_delete = false
        // expected outcome: deleted neighbor is dropped

        index
            .drop_deleted_neighbors(&ctx, neighbor_accessor, 0, false)
            .await
            .expect(drop_msg);

        let mut list0 = AdjacencyList::new();
        neighbor_accessor
            .get_neighbors(0, &mut list0)
            .await
            .expect(adj_msg);
        list0.sort();
        assert_eq!(&*list0, &[1, 2]);

        // call drop_deleted_neighbors on vertex 1 with check_delete = true
        // expected outcome: deleted neighbor is not dropped

        index
            .drop_deleted_neighbors(&ctx, neighbor_accessor, 1, true)
            .await
            .expect(drop_msg);

        let mut list1_before_drop = AdjacencyList::new();
        neighbor_accessor
            .get_neighbors(1, &mut list1_before_drop)
            .await
            .expect(adj_msg);
        list1_before_drop.sort();
        assert_eq!(&*list1_before_drop, &[2, 3, 4]);

        // drop vertex 3's adjacency list

        index
            .drop_adj_list(neighbor_accessor, 3)
            .await
            .expect("Error in drop_adj_list");

        // call drop_deleted_neighbors on vertex 1 with check_delete = true
        // expected outcome: deleted neighbor is dropped

        index
            .drop_deleted_neighbors(&ctx, neighbor_accessor, 1, true)
            .await
            .expect(drop_msg);

        let mut list1_after_drop = AdjacencyList::new();
        neighbor_accessor
            .get_neighbors(1, &mut list1_after_drop)
            .await
            .expect(adj_msg);
        list1_after_drop.sort();
        assert_eq!(&*list1_after_drop, &[2, 4]);
    }

    #[tokio::test]
    async fn test_get_undeleted_neighbors() {
        // create small index instance
        let dim = 1;
        let (config, parameters) = simplified_builder(
            10,         // l_search
            3,          // max_degree
            Metric::L2, // metric
            dim,        // dim
            5,          // max_points
            no_modify,
        )
        .unwrap();

        let pqtable = model::pq::FixedChunkPQTable::new(
            dim,
            Box::new([0.0]),
            Box::new([0.0]),
            Box::new([0, 1]),
            None,
        )
        .unwrap();

        let index =
            new_quant_index::<f32, _, _>(config, parameters, pqtable, TableBasedDeletes).unwrap();

        // build graph
        let adjacency_lists = [
            AdjacencyList::from_iter_untrusted([2, 3, 1]),
            AdjacencyList::from_iter_untrusted([2, 3, 4]),
            AdjacencyList::from_iter_untrusted([0, 1, 4]),
            AdjacencyList::from_iter_untrusted([2, 4, 0]),
            AdjacencyList::from_iter_untrusted([0, 3, 2]),
        ];

        let neighbor_accessor = &mut index.provider().neighbors();
        let ctx = DefaultContext;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        // delete id number 3
        index
            .data_provider
            .delete(&DefaultContext, &3_u32)
            .await
            .expect("Error in delete");

        // we'll check vertices 0 and 2
        {
            let PartitionedNeighbors {
                mut undeleted,
                mut deleted,
            } = index
                .get_undeleted_neighbors(&ctx, neighbor_accessor, 0)
                .await
                .expect("Error in get_undeleted_neighbors");
            undeleted.sort();
            assert_eq!(&undeleted, &[1, 2]);
            deleted.sort();
            assert_eq!(&deleted, &[3]);

            let PartitionedNeighbors { undeleted, deleted } = index
                .get_undeleted_neighbors(&ctx, neighbor_accessor, 2)
                .await
                .expect("Error in deleted");
            assert!(undeleted.len() == 3);
            assert!(deleted.is_empty());
        }

        // delete id number 2
        index
            .data_provider
            .delete(&DefaultContext, &2_u32)
            .await
            .expect("Error in delete");

        // we'll check vertices 0, 2, and 3
        {
            let PartitionedNeighbors {
                mut undeleted,
                mut deleted,
            } = index
                .get_undeleted_neighbors(&ctx, neighbor_accessor, 0)
                .await
                .expect("Error in get_undeleted_neighbors");
            undeleted.sort();
            assert_eq!(&undeleted, &[1]);
            deleted.sort();
            assert_eq!(&deleted, &[2, 3]);

            let PartitionedNeighbors { undeleted, deleted } = index
                .get_undeleted_neighbors(&ctx, neighbor_accessor, 2)
                .await
                .expect("Error in get_undeleted_neighbors");
            assert!(undeleted.len() == 3);
            assert!(deleted.is_empty());

            let PartitionedNeighbors {
                mut undeleted,
                mut deleted,
            } = index
                .get_undeleted_neighbors(&ctx, neighbor_accessor, 3)
                .await
                .expect("Error in get_undeleted_neighbors");
            undeleted.sort();
            assert_eq!(&undeleted, &[0, 4]);
            deleted.sort();
            assert_eq!(&deleted, &[2]);
        }
    }

    #[tokio::test]
    async fn test_inplace_delete_2d() {
        test_inplace_delete_2d_impl(FullPrecision).await;
        test_inplace_delete_2d_impl(Hybrid::new(None)).await;
    }

    async fn test_inplace_delete_2d_impl<S>(strategy: S)
    where
        S: InplaceDeleteStrategy<TestProvider>
            + for<'a> SearchStrategy<TestProvider, S::DeleteElement<'a>>
            + Sync
            + std::clone::Clone,
    {
        // create small index instance
        let dim = 2;
        let (config, parameters) = simplified_builder(
            10,         // l_search
            4,          // max_degree
            Metric::L2, // metric
            dim,        // dim
            4,          // max_points
            no_modify,
        )
        .unwrap();

        let pqtable = model::pq::FixedChunkPQTable::new(
            dim,
            Box::new([0.0, 0.0]),
            Box::new([0.0, 0.0]),
            Box::new([0, 2]),
            None,
        )
        .unwrap();

        let index =
            new_quant_index::<f32, _, _>(config, parameters, pqtable, TableBasedDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();

        let start_point: &[f32] = &[0.5, 0.5];

        index
            .provider()
            .set_start_points(std::iter::once(start_point))
            .unwrap();

        // vectors are the four corners of a square, with the start point in the middle
        // the middle point forms an edge to each corner, while corners form an edge
        // to their opposite vertex vertically as well as the middle
        let vectors = [
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let adjacency_lists = [
            AdjacencyList::from_iter_untrusted([4, 1]),
            AdjacencyList::from_iter_untrusted([4, 0]),
            AdjacencyList::from_iter_untrusted([4, 3]),
            AdjacencyList::from_iter_untrusted([4, 2]),
            AdjacencyList::from_iter_untrusted([0, 1, 2, 3]),
        ];

        let ctx = DefaultContext;
        populate_graph(neighbor_accessor, &adjacency_lists).await;
        populate_data(&index.data_provider, &ctx, &vectors).await;

        index
            .inplace_delete(
                strategy,
                &ctx,
                &3, // id to delete
                3,  // num_to_replace
                InplaceDeleteMethod::VisitedAndTopK {
                    k_value: 4,
                    l_value: 10,
                },
            )
            .await
            .unwrap();

        // Check that the vertex was marked as deleted.
        assert!(
            index
                .data_provider
                .status_by_internal_id(&ctx, 3)
                .await
                .unwrap()
                .is_deleted()
        );

        let neighbor_accessor = &mut index.provider().neighbors();

        // expected outcome:
        // vertex 4 (the start point) has its edge to 3 deleted
        // vertex 2 (the other point with edge pointing to 3) should have its edge to point 3 deleted,
        // and replaced with edges to points 0 and 1
        // vertices 0 and 1 should add an edge pointing to 2.
        // vertex 3 should be dropped
        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(4, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 1, 2]);
        }

        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(2, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 1, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(0, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[1, 2, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(1, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 2, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(3, &mut list).await.unwrap();
            assert!(list.is_empty());
        }
    }

    #[tokio::test]
    async fn test_consolidate_deletes_2d() {
        // create small index instance
        let dim = 2;
        let (config, parameters) = simplified_builder(
            10,         // l_search
            4,          // max_degree
            Metric::L2, // metric
            dim,        // dim
            4,          // max_points
            no_modify,
        )
        .unwrap();

        let pqtable = model::pq::FixedChunkPQTable::new(
            dim,
            Box::new([0.0, 0.0]),
            Box::new([0.0, 0.0]),
            Box::new([0, 2]),
            None,
        )
        .unwrap();

        let index =
            new_quant_index::<f32, _, _>(config, parameters, pqtable, TableBasedDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();

        let start_point: &[f32] = &[0.5, 0.5];

        index
            .provider()
            .set_start_points(std::iter::once(start_point))
            .unwrap();

        // vectors are the four corners of a square, with the start point in the middle
        // the middle point forms an edge to each corner, while corners form an edge
        // to their opposite vertex vertically and horizontally as well as the middle
        let vectors = [
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let adjacency_lists = [
            AdjacencyList::from_iter_untrusted([4, 1, 2]), // point 0
            AdjacencyList::from_iter_untrusted([4, 0, 3]), // point 1
            AdjacencyList::from_iter_untrusted([4, 3, 0]), // point 2
            AdjacencyList::from_iter_untrusted([4, 2, 1]), // point 3
            AdjacencyList::from_iter_untrusted([0, 1, 2, 3]), // point 4, start point
        ];

        let ctx = DefaultContext;
        populate_graph(neighbor_accessor, &adjacency_lists).await;
        populate_data(&index.data_provider, &ctx, &vectors).await;

        let starting_point_ids = index.provider().starting_points().unwrap();
        assert!(starting_point_ids.contains(&4));
        assert!(starting_point_ids.len() == 1);

        // delete id number 3
        index
            .data_provider
            .delete(&ctx, &3_u32)
            .await
            .expect("Error in delete");

        for vector_id in 0..5 {
            index
                .consolidate_vector(&FullPrecision, &ctx, vector_id as u32)
                .await
                .expect("Error in consolidate_vector");
        }

        let neighbor_accessor = &mut index.provider().neighbors();
        // expected outcome:
        // vertex 0 should be unchanged
        // vertex 1 (a point with edge pointing to 3) should have its edge to point 3 deleted,
        // and replaced with an edge to point 2
        // vertex 2 (a point with edge pointing to 3) should have its edge to point 3 deleted,
        // and replaced with an edge to point 1
        // vertex 4 (the start point) has its edge to 3 deleted
        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(0, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[1, 2, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(1, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 2, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(2, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 1, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            neighbor_accessor.get_neighbors(4, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 1, 2]);
        }
    }

    const SIFTSMALL: &str = "/test_data/sift/siftsmall_learn_256pts.fbin";

    #[rstest]
    #[tokio::test]
    async fn test_sift_build_and_search<S>(
        #[values(FullPrecision, Hybrid::new(None))] build_strategy: S,
        #[values(1, 10)] batchsize: usize,
    ) where
        S: InsertStrategy<TestProvider, [f32]> + Clone + Send + Sync,
        for<'a> aliases::InsertPruneAccessor<'a, S, TestProvider, [f32]>: AsElement<&'a [f32]>,
        S::PruneStrategy: Clone,
    {
        let ctx = &DefaultContext;
        let parameters = InitParams {
            l_build: 64,
            max_degree: 16,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(batchsize).unwrap(),
        };

        let (index, data) = init_from_file(
            build_strategy.clone(),
            parameters,
            SIFTSMALL,
            8,
            StartPointStrategy::RandomSamples {
                nsamples: ONE,
                seed: 0xe058c9c57864dd1e,
            },
        )
        .await;

        let starting_points = index.provider().starting_points().unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        // There should be one more reachable node than points in the dataset to account for
        // the start point.
        assert_eq!(
            index
                .count_reachable_nodes(&starting_points, neighbor_accessor)
                .await
                .unwrap(),
            data.nrows() + 1,
        );

        let top_k = 10;
        let search_l = 32;
        let mut ids = vec![0; top_k];
        let mut distances = vec![0.0; top_k];

        // Here, we use elements of the dataset to search the dataset itself.
        //
        // We do this for each query, computing the expected ground truth and verifying
        // that our simple graph search matches.
        //
        // Because this dataset is small, we can expect exact equality.
        for (q, query) in data.row_iter().enumerate() {
            let gt = groundtruth(data.as_view(), query, |a, b| SquaredL2::evaluate(a, b));
            {
                let mut result_output_buffer =
                    search_output_buffer::IdDistance::new(&mut ids, &mut distances);
                // Full Precision Search.
                index
                    .search(
                        &FullPrecision,
                        ctx,
                        query,
                        &SearchParams::new_default(top_k, search_l).unwrap(),
                        &mut result_output_buffer,
                    )
                    .await
                    .unwrap();
            }
            assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);

            {
                let mut result_output_buffer =
                    search_output_buffer::IdDistance::new(&mut ids, &mut distances);
                // Quantized Search
                index
                    .search(
                        &Hybrid::new(None),
                        ctx,
                        query,
                        &SearchParams::new_default(top_k, search_l).unwrap(),
                        &mut result_output_buffer,
                    )
                    .await
                    .unwrap();
            }
            assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_sift_build_and_range_search<S>(
        #[values(FullPrecision, Hybrid::new(None))] build_strategy: S,
        #[values(1, 10)] batchsize: usize,
        #[values((-2.0,-1.0), (-1.0, 0.0), (40000.0,50000.0), (50000.0,75000.0))] radii: (f32, f32),
    ) where
        S: InsertStrategy<TestProvider, [f32]> + Clone + Send + Sync,
        for<'a> aliases::InsertPruneAccessor<'a, S, TestProvider, [f32]>: AsElement<&'a [f32]>,
        S::PruneStrategy: Clone,
    {
        let ctx = &DefaultContext;
        let parameters = InitParams {
            l_build: 64,
            max_degree: 16,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(batchsize).unwrap(),
        };

        let (index, data) = init_from_file(
            build_strategy.clone(),
            parameters,
            SIFTSMALL,
            8,
            StartPointStrategy::RandomSamples {
                nsamples: ONE,
                seed: 0xe058c9c57864dd1e,
            },
        )
        .await;

        let starting_l_value = 32;
        let lower_l_value = 4;

        let radius = radii.1;
        let inner_radius = radii.0;

        // Here, we use elements of the dataset to search the dataset itself.
        //
        // We do this for each query, computing the expected ground truth and verifying
        // that our simple graph search matches.
        //
        // Because this dataset is small, we can expect exact equality expect for the
        // case where we use a lower initial beam, which will trigger more two-round searches.

        for (q, query) in data.row_iter().enumerate() {
            let gt = groundtruth(data.as_view(), query, |a, b| SquaredL2::evaluate(a, b));
            {
                // Full Precision Search.
                let (_, ids, _) = index
                    .range_search(
                        &FullPrecision,
                        ctx,
                        query,
                        &RangeSearchParams::new_default(starting_l_value, radius).unwrap(),
                    )
                    .await
                    .unwrap();

                assert_range_results_exactly_match(q, &gt, &ids, radius, None);
            }

            {
                // Quantized Search
                let (_, ids, _) = index
                    .range_search(
                        &Hybrid::new(None),
                        ctx,
                        query,
                        &RangeSearchParams::new_default(starting_l_value, radius).unwrap(),
                    )
                    .await
                    .unwrap();

                assert_range_results_exactly_match(q, &gt, &ids, radius, None);
            }

            {
                // Test with an inner radius

                assert!(inner_radius <= radius);
                let (_, ids, _) = index
                    .range_search(
                        &FullPrecision,
                        ctx,
                        query,
                        &RangeSearchParams::new(
                            None,
                            starting_l_value,
                            None,
                            radius,
                            Some(inner_radius),
                            1.0,
                            1.0,
                        )
                        .unwrap(),
                    )
                    .await
                    .unwrap();

                assert_range_results_exactly_match(q, &gt, &ids, radius, Some(inner_radius));
            }

            {
                // Test with a lower initial beam to trigger more two-round searches
                // We don't expect results to exactly match here
                let (_, ids, _) = index
                    .range_search(
                        &FullPrecision,
                        ctx,
                        query,
                        &RangeSearchParams::new_default(lower_l_value, radius).unwrap(),
                    )
                    .await
                    .unwrap();

                // check that ids don't have duplicates
                let mut ids_set = std::collections::HashSet::new();
                for id in &ids {
                    assert!(ids_set.insert(*id));
                }
            }
        }
    }

    ///////////////////////////
    // Scalar Build & Search //
    ///////////////////////////

    async fn init_and_build_index_from_file<C, B, DP>(
        file: &str,
        create_fn: C,
        build_fn: B,
    ) -> (Arc<DiskANNIndex<DP>>, Arc<Matrix<f32>>)
    where
        C: FnOnce(Arc<Matrix<f32>>, &[f32]) -> Arc<DiskANNIndex<DP>>,
        B: AsyncFnOnce(Arc<DiskANNIndex<DP>>, Arc<Matrix<f32>>),
        DP: DataProvider<Context = DefaultContext, ExternalId = u32>
            + diskann::provider::SetElement<[f32]>,
    {
        let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let storage = VirtualStorageProvider::new_overlay(workspace_root);
        let (data_vec, npoints, dim) = file_util::load_bin(&storage, file, 0).unwrap();
        let data =
            Arc::new(Matrix::<f32>::try_from(data_vec.into_boxed_slice(), npoints, dim).unwrap());

        let rng = &mut create_rnd_from_seed_in_tests(0xe058c9c57864dd1e);
        let random_index = rand::Rng::random_range(rng, 0..data.nrows());
        let start_point = data.row(random_index);

        let index = create_fn(data.clone(), start_point);
        build_fn(index.clone(), data.clone()).await;

        (index, data)
    }

    async fn build_using_single_insert<DP>(index: Arc<DiskANNIndex<DP>>, data: Arc<Matrix<f32>>)
    where
        DP: DataProvider<Context = DefaultContext, ExternalId = u32>
            + diskann::provider::SetElement<[f32]>,
        Quantized: InsertStrategy<DP, [f32]> + Clone + Send + Sync,
    {
        let ctx = &DefaultContext;
        for (i, vector) in data.row_iter().enumerate() {
            index
                .insert(Quantized, ctx, &(i as u32), vector)
                .await
                .unwrap()
        }
    }

    macro_rules! scalar_quant_test {
        ($name:ident, $nbits:literal, $search_l:literal) => {
            #[tokio::test]
            async fn $name() {
                let ctx = &DefaultContext;
                let parameters = InitParams {
                    l_build: 64,
                    max_degree: 16,
                    metric: Metric::L2,
                    batchsize: NonZeroUsize::new(1).unwrap(),
                };

                let create_fn = |data: Arc<Matrix<f32>>, start_point: &[f32]| {
                    let quantizer = ScalarQuantizationParameters::default().train(data.as_view());
                    let (config, params) =
                        parameters.materialize(data.nrows(), data.ncols()).unwrap();
                    let index = new_quant_index::<f32, _, _>(
                        config,
                        params,
                        inmem::WithBits::<$nbits>::new(quantizer),
                        NoDeletes,
                    )
                    .unwrap();
                    index
                        .provider()
                        .set_start_points(std::iter::once(start_point))
                        .unwrap();
                    index
                };
                let (index, data) =
                    init_and_build_index_from_file(SIFTSMALL, create_fn, build_using_single_insert)
                        .await;

                let neighbor_accessor = &mut index.provider().neighbors();
                // There should be one more reachable node than points in the dataset to account for
                // the start point.
                assert_eq!(
                    index
                        .count_reachable_nodes(
                            &index.provider().starting_points().unwrap(),
                            neighbor_accessor
                        )
                        .await
                        .unwrap(),
                    data.nrows() + 1,
                );

                let top_k = 8;
                let search_l = $search_l; // Keep higher L to be able to get top K correctly for Scalar quantization for small(100) dim data
                let mut ids = vec![0; top_k];
                let mut distances = vec![0.0; top_k];

                // Here, we use elements of the dataset to search the dataset itself.
                //
                // We do this for each query, computing the expected ground truth and verifying
                // that our simple graph search matches.
                //
                // Because this dataset is small, we can expect exact equality.
                for (q, query) in data.row_iter().enumerate() {
                    let gt = groundtruth(data.as_view(), query, |a, b| SquaredL2::evaluate(a, b));
                    {
                        let mut result_output_buffer =
                            search_output_buffer::IdDistance::new(&mut ids, &mut distances);
                        // Full Precision Search.
                        index
                            .search(
                                &FullPrecision,
                                ctx,
                                query,
                                &SearchParams::new_default(top_k, search_l).unwrap(),
                                &mut result_output_buffer,
                            )
                            .await
                            .unwrap();
                    }
                    assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);

                    {
                        let mut result_output_buffer =
                            search_output_buffer::IdDistance::new(&mut ids, &mut distances);
                        // Quantized Search
                        index
                            .search(
                                &Quantized,
                                ctx,
                                query,
                                &SearchParams::new_default(top_k, search_l).unwrap(),
                                &mut result_output_buffer,
                            )
                            .await
                            .unwrap();
                    }
                    assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);
                }
            }
        };
    }

    scalar_quant_test!(test_sift_build_and_search_scalar_q_1bit, 1, 130);
    scalar_quant_test!(test_sift_build_and_search_scalar_q_4bit, 4, 20);
    scalar_quant_test!(test_sift_build_and_search_scalar_q_8bit, 8, 20);
    scalar_quant_test!(test_sift_build_and_search_scalar_q_7bit, 7, 20);

    macro_rules! scalar_only_test {
        ($name:ident, $nbits:literal) => {
            #[tokio::test]
            async fn $name() {
                let ctx = &DefaultContext;
                let parameters = InitParams {
                    l_build: 64,
                    max_degree: 16,
                    metric: Metric::L2,
                    batchsize: NonZeroUsize::new(1).unwrap(),
                };

                let create_fn = |data: Arc<Matrix<f32>>, start_point: &[f32]| {
                    let quantizer = ScalarQuantizationParameters::default().train(data.as_view());
                    let (config, params) =
                        parameters.materialize(data.nrows(), data.ncols()).unwrap();
                    let index = Arc::new(
                        new_quant_only_index(
                            config,
                            params,
                            inmem::WithBits::<$nbits>::new(quantizer),
                            NoDeletes,
                        )
                        .unwrap(),
                    );
                    index
                        .provider()
                        .set_start_points(std::iter::once(start_point))
                        .unwrap();
                    index
                };
                let (index, data) =
                    init_and_build_index_from_file(SIFTSMALL, create_fn, build_using_single_insert)
                        .await;

                let neighbor_accessor = &mut index.provider().neighbors();
                // There should be one more reachable node than points in the dataset to account for
                // the start point.
                assert_eq!(
                    index
                        .count_reachable_nodes(
                            &index.provider().starting_points().unwrap(),
                            neighbor_accessor
                        )
                        .await
                        .unwrap(),
                    data.nrows() + 1,
                );

                let top_k = 10;
                let mut ids = vec![0; top_k];
                let mut distances = vec![0.0; top_k];

                // Here, we use elements of the dataset to search the dataset itself.
                //
                // We do this for each query, computing the expected ground truth and verifying
                // that our simple graph search matches.
                //
                // Because this dataset is small, we can expect exact equality.
                for (q, query) in data.row_iter().enumerate() {
                    {
                        let mut result_output_buffer =
                            search_output_buffer::IdDistance::new(&mut ids, &mut distances);
                        // Quantized Search
                        index
                            .search(
                                &Quantized,
                                ctx,
                                query,
                                &SearchParams::new_default(top_k, top_k).unwrap(),
                                &mut result_output_buffer,
                            )
                            .await
                            .unwrap();
                    }

                    // Easy assert as there is no reranking for this small(100) dim data.
                    assert!(ids.contains(&(q as u32)));
                }
            }
        };
    }

    scalar_only_test!(test_sift_quant_only_build_and_search_scalar_1bit, 1);
    scalar_only_test!(test_sift_quant_only_build_and_search_scalar_4bit, 4);
    scalar_only_test!(test_sift_quant_only_build_and_search_scalar_8bit, 8);
    scalar_only_test!(test_sift_quant_only_build_and_search_scalar_7bit, 7);

    //////////////////////////////
    // Spherical Build & Search //
    //////////////////////////////

    #[tokio::test]
    async fn test_sift_build_and_search_spherical() {
        let ctx = &DefaultContext;
        let parameters = InitParams {
            l_build: 64,
            max_degree: 16,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(1).unwrap(),
        };

        let rng = &mut create_rnd_from_seed_in_tests(0x56870bccb0c44b66);

        let create_fn = |data: Arc<Matrix<f32>>, start_point: &[f32]| {
            let quantizer = diskann_quantization::spherical::SphericalQuantizer::train(
                data.as_view(),
                diskann_quantization::algorithms::transforms::TransformKind::PaddingHadamard {
                    target_dim: diskann_quantization::algorithms::transforms::TargetDim::Natural,
                },
                Metric::L2.try_into().unwrap(),
                diskann_quantization::spherical::PreScale::ReciprocalMeanNorm,
                rng,
                diskann_quantization::alloc::GlobalAllocator,
            )
            .unwrap();

            let (config, params) = parameters.materialize(data.nrows(), data.ncols()).unwrap();

            let index = new_quant_index::<f32, _, _>(
                config,
                params,
                diskann_quantization::spherical::iface::Impl::<1>::new(quantizer).unwrap(),
                NoDeletes,
            )
            .unwrap();

            index
                .provider()
                .set_start_points(std::iter::once(start_point))
                .unwrap();
            index
        };

        let build_fn = async |index: Arc<DiskANNIndex<_>>, data: Arc<Matrix<f32>>| {
            let ctx = &DefaultContext;
            let strategy = inmem::spherical::Quantized::build();
            for (i, vector) in data.row_iter().enumerate() {
                index
                    .insert(strategy, ctx, &(i as u32), vector)
                    .await
                    .unwrap()
            }
        };

        let (index, data) = init_and_build_index_from_file(SIFTSMALL, create_fn, build_fn).await;
        let neighbor_accessor = &mut index.provider().neighbors();
        // There should be one more reachable node than points in the dataset to account for
        // the start point.
        assert_eq!(
            index
                .count_reachable_nodes(
                    &index.provider().starting_points().unwrap(),
                    neighbor_accessor
                )
                .await
                .unwrap(),
            data.nrows() + 1,
        );

        let top_k = 5;
        let search_l = 80;
        let mut ids = vec![0; top_k];
        let mut distances = vec![0.0; top_k];

        // Here, we use elements of the dataset to search the dataset itself.
        //
        // We do this for each query, computing the expected ground truth and verifying
        // that our simple graph search matches.
        //
        // Because this dataset is small, we can expect exact equality.
        for (q, query) in data.row_iter().enumerate() {
            let gt = groundtruth(data.as_view(), query, |a, b| SquaredL2::evaluate(a, b));

            // Full Precision Search.
            let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            index
                .search(
                    &FullPrecision,
                    ctx,
                    query,
                    &SearchParams::new_default(top_k, search_l).unwrap(),
                    &mut output,
                )
                .await
                .unwrap();
            assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);

            // Quantized Search
            let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            let strategy = inmem::spherical::Quantized::search(
                diskann_quantization::spherical::iface::QueryLayout::FourBitTransposed,
            );

            index
                .search(
                    &strategy,
                    ctx,
                    query,
                    &SearchParams::new_default(top_k, search_l).unwrap(),
                    &mut output,
                )
                .await
                .unwrap();
            assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);
        }

        // Ensure that the query computer used for insertion uses the `SameAsData` layout.
        let strategy = inmem::spherical::Quantized::build();
        let accessor = strategy.search_accessor(index.provider(), ctx).unwrap();
        let computer = accessor.build_query_computer(data.row(0)).unwrap();
        assert_eq!(
            computer.layout(),
            diskann_quantization::spherical::iface::QueryLayout::SameAsData
        );
    }

    ///////////////////////////////////
    // Spherical only Build & Search //
    ///////////////////////////////////

    #[tokio::test]
    async fn test_sift_spherical_only_build_and_search_() {
        let ctx = &DefaultContext;
        let rng = &mut create_rnd_from_seed_in_tests(0x56870bccb0c44b66);

        let create_fn = |data: Arc<Matrix<f32>>, start_points: &[f32]| {
            let quantizer = diskann_quantization::spherical::SphericalQuantizer::train(
                data.as_view(),
                diskann_quantization::algorithms::transforms::TransformKind::PaddingHadamard {
                    target_dim: diskann_quantization::algorithms::transforms::TargetDim::Natural,
                },
                Metric::L2.try_into().unwrap(),
                diskann_quantization::spherical::PreScale::ReciprocalMeanNorm,
                rng,
                diskann_quantization::alloc::GlobalAllocator,
            )
            .unwrap();

            let (config, params) =
                simplified_builder(64, 16, Metric::L2, data.ncols(), data.nrows(), no_modify)
                    .unwrap();

            let index = new_quant_only_index(
                config,
                params,
                diskann_quantization::spherical::iface::Impl::<1>::new(quantizer).unwrap(),
                NoDeletes,
            )
            .unwrap();
            index
                .provider()
                .set_start_points(std::iter::once(start_points))
                .unwrap();
            Arc::new(index)
        };

        let build_fn = async |index: Arc<DiskANNIndex<_>>, data: Arc<Matrix<f32>>| {
            let ctx = &DefaultContext;
            let strategy = inmem::spherical::Quantized::build();
            for (i, vector) in data.row_iter().enumerate() {
                index
                    .insert(strategy, ctx, &(i as u32), vector)
                    .await
                    .unwrap()
            }
        };

        let (index, data) = init_and_build_index_from_file(SIFTSMALL, create_fn, build_fn).await;

        let neighbor_accessor = &mut index.provider().neighbors();
        // There should be one more reachable node than points in the dataset to account for
        // the start point.
        assert_eq!(
            index
                .count_reachable_nodes(
                    &index.provider().starting_points().unwrap(),
                    neighbor_accessor
                )
                .await
                .unwrap(),
            data.nrows() + 1,
        );

        let top_k = 5;
        let search_l = 80;
        let mut ids = vec![0; top_k];
        let mut distances = vec![0.0; top_k];

        // Here, we use elements of the dataset to search the dataset itself.
        //
        // We do this for each query, computing the expected ground truth and verifying
        // that our simple graph search matches.
        //
        // Because this dataset is small, we can expect exact equality.
        for (q, query) in data.row_iter().enumerate() {
            // Quantized Search
            let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            let strategy = inmem::spherical::Quantized::search(
                diskann_quantization::spherical::iface::QueryLayout::FourBitTransposed,
            );

            index
                .search(
                    &strategy,
                    ctx,
                    query,
                    &SearchParams::new_default(top_k, search_l).unwrap(),
                    &mut output,
                )
                .await
                .unwrap();

            // Easy assert as there is no reranking for this small(100) dim data.
            assert!(ids.contains(&(q as u32)));
        }

        // Ensure that the query computer used for insertion uses the `SameAsData` layout.
        let strategy = inmem::spherical::Quantized::build();
        let accessor = <inmem::spherical::Quantized as SearchStrategy<
            DefaultProvider<NoStore, inmem::spherical::SphericalStore>,
            [f32],
            _,
        >>::search_accessor(&strategy, index.provider(), ctx)
        .unwrap();
        let computer = accessor.build_query_computer(data.row(0)).unwrap();
        assert_eq!(
            computer.layout(),
            diskann_quantization::spherical::iface::QueryLayout::SameAsData
        );
    }

    //////////////////////////////
    /// PQ only Build & Search ///
    //////////////////////////////

    #[tokio::test]
    async fn test_sift_pq_only_build_and_search() {
        let ctx = &DefaultContext;
        let create_fn = |data: Arc<Matrix<f32>>, start_points: &[f32]| {
            let pq_table = train_pq(
                data.as_view(),
                32,
                &mut create_rnd_from_seed_in_tests(0xe3c52ef001bc7ade),
                1,
            )
            .unwrap();

            let (config, parameters) =
                simplified_builder(64, 16, Metric::L2, data.ncols(), data.nrows(), no_modify)
                    .unwrap();

            let index =
                Arc::new(new_quant_only_index(config, parameters, pq_table, NoDeletes).unwrap());
            index
                .provider()
                .set_start_points(std::iter::once(start_points))
                .unwrap();
            index
        };
        let (index, data) =
            init_and_build_index_from_file(SIFTSMALL, create_fn, build_using_single_insert).await;

        let neighbor_accessor = &mut index.provider().neighbors();
        // There should be one more reachable node than points in the dataset to account for
        // the start point.
        assert_eq!(
            index
                .count_reachable_nodes(
                    &index.provider().starting_points().unwrap(),
                    neighbor_accessor
                )
                .await
                .unwrap(),
            data.nrows() + 1,
        );

        let top_k = 10;
        let search_l = 32;
        let mut ids = vec![0; top_k];
        let mut distances = vec![0.0; top_k];

        // Here, we use elements of the dataset to search the dataset itself.
        //
        // We do this for each query, computing the expected ground truth and verifying
        // that our simple graph search matches.
        //
        // Because this dataset is small, we can expect exact equality.
        for (q, query) in data.row_iter().enumerate() {
            let gt = groundtruth(data.as_view(), query, |a, b| SquaredL2::evaluate(a, b));

            let mut result_output_buffer =
                search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            // Full Precision Search.
            index
                .search(
                    &Quantized,
                    ctx,
                    query,
                    &SearchParams::new_default(top_k, search_l).unwrap(),
                    &mut result_output_buffer,
                )
                .await
                .unwrap();

            assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);
        }
    }

    async fn check_graph_for_self_loops_or_duplicates<NA, Itr>(accessor: &mut NA, itr: Itr)
    where
        NA: AsNeighbor<Id = u32>,
        Itr: Iterator<Item = u32>,
    {
        for id in itr {
            let mut adj_list = AdjacencyList::new();
            accessor
                .get_neighbors(id, &mut adj_list)
                .await
                .expect("Error in get_neighbors");
            assert!(!adj_list.contains(id));
            let len_before_dedup = adj_list.len();

            let mut adj_list: Vec<_> = adj_list.into();
            adj_list.sort();
            adj_list.dedup();
            assert_eq!(adj_list.len(), len_before_dedup);
        }
    }

    type TestProvider =
        FullPrecisionProvider<f32, DefaultQuant, TableDeleteProviderAsync, DefaultContext>;
    type TestIndex = DiskANNIndex<TestProvider>;

    /// Parameters for initializing an index during the build process.
    #[derive(Debug, Clone, Copy)]
    pub struct InitParams {
        /// The search budget used during construction (L_build parameter).
        pub l_build: usize,
        /// The maximum degree for nodes in the graph.
        pub max_degree: usize,
        /// The distance metric to use.
        pub metric: Metric,
        /// The batch size for insertion operations.
        pub batchsize: NonZeroUsize,
    }

    impl InitParams {
        /// Create index configuration and provider parameters from these initialization parameters.
        pub fn materialize(
            &self,
            npoints: usize,
            dim: usize,
        ) -> ANNResult<(Config, DefaultProviderParameters)> {
            simplified_builder(
                self.l_build,
                self.max_degree,
                self.metric,
                dim,
                npoints,
                |builder| {
                    builder.max_minibatch_par(self.batchsize.into());
                },
            )
        }
    }

    /// Build an index by inserting vectors from a file.
    ///
    /// This function reads vectors from the specified file and inserts them into the index
    /// using the provided insertion strategy. It supports different strategies for selecting
    /// start points and handles both single and batch insertion modes.
    pub async fn build_index<S, U, V, D>(
        index: &Arc<DiskANNIndex<DefaultProvider<U, V, D>>>,
        strategy: S,
        parameters: InitParams,
        file: &str,
        start_strategy: StartPointStrategy,
        train_data: diskann_utils::views::MatrixView<'_, f32>,
    ) where
        DefaultProvider<U, V, D>: DataProvider<ExternalId = u32, Context = DefaultContext>
            + SetElement<[f32]>
            + SetStartPoints<[f32]>,
        S: InsertStrategy<DefaultProvider<U, V, D>, [f32]> + Clone + Send + Sync,
        for<'a> aliases::InsertPruneAccessor<'a, S, DefaultProvider<U, V, D>, [f32]>:
            AsElement<&'a [f32]>,
        S::PruneStrategy: Clone,
    {
        let ctx = &DefaultContext;
        let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let storage = VirtualStorageProvider::new_overlay(workspace_root);

        let mut iter = VectorDataIterator::<_, crate::model::graph::traits::AdHoc<f32>>::new(
            file, None, &storage,
        )
        .unwrap();

        let start_vectors: Matrix<f32> = start_strategy.compute(train_data).unwrap();

        index
            .provider()
            .set_start_points(start_vectors.row_iter())
            .unwrap();

        let batchsize: usize = parameters.batchsize.into();
        if batchsize == 1 {
            for (i, (vector, _)) in iter.enumerate() {
                index
                    .insert(strategy.clone(), ctx, &(i as u32), &vector)
                    .await
                    .unwrap()
            }
        } else {
            let mut i: u32 = 0;
            while let Some(data) = iter.next_n(batchsize) {
                let pairs: Box<[_]> = data
                    .iter()
                    .map(|(v, _)| {
                        let r = VectorIdBoxSlice::new(i, v.clone());
                        i += 1;
                        r
                    })
                    .collect();

                index
                    .multi_insert(strategy.clone(), ctx, pairs)
                    .await
                    .unwrap();
            }
        }
    }

    async fn init_from_file<S>(
        strategy: S,
        parameters: InitParams,
        file: &str,
        num_pq_chunks: usize,
        startpoint: StartPointStrategy,
    ) -> (Arc<TestIndex>, diskann_utils::views::Matrix<f32>)
    where
        S: InsertStrategy<TestProvider, [f32]> + Clone + Send + Sync,
        for<'a> aliases::InsertPruneAccessor<'a, S, TestProvider, [f32]>: AsElement<&'a [f32]>,
        S::PruneStrategy: Clone,
    {
        let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let storage = VirtualStorageProvider::new_overlay(workspace_root);
        let (train_data, npoints, dim) = file_util::load_bin(&storage, file, 0).unwrap();

        let train_data_view =
            diskann_utils::views::MatrixView::try_from(&train_data, npoints, dim).unwrap();

        let table = train_pq(
            train_data_view,
            num_pq_chunks,
            &mut create_rnd_from_seed_in_tests(0xe3c52ef001bc7ade),
            1,
        )
        .unwrap();

        let (config, params) = parameters.materialize(npoints, dim).unwrap();
        let index = new_quant_index(config, params, table, TableBasedDeletes).unwrap();

        build_index(
            &index,
            strategy,
            parameters,
            file,
            startpoint,
            train_data_view,
        )
        .await;

        (index, train_data_view.to_owned())
    }

    #[rstest]
    #[tokio::test]
    async fn inplace_delete_on_sift<S>(
        #[values(FullPrecision, Hybrid::new(None))] strategy: S,
        #[values(20, 100)] points_to_delete: u32,
        #[values(
            InplaceDeleteMethod::VisitedAndTopK{k_value:5, l_value:10},
            InplaceDeleteMethod::TwoHopAndOneHop,
            InplaceDeleteMethod::OneHop,
        )]
        delete_method: InplaceDeleteMethod,
    ) where
        S: InsertStrategy<TestProvider, [f32]>
            + SearchStrategy<TestProvider, [f32]>
            + for<'a> InplaceDeleteStrategy<TestProvider, DeleteElement<'a> = [f32]>
            + Clone
            + Sync,
        for<'a> aliases::InsertPruneAccessor<'a, S, TestProvider, [f32]>: AsElement<&'a [f32]>,
        <S as InsertStrategy<TestProvider, [f32]>>::PruneStrategy: Clone,
    {
        let ctx = &DefaultContext;
        let parameters = InitParams {
            l_build: 10,
            max_degree: 32,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(1).unwrap(),
        };

        let (index, data) = init_from_file(
            strategy.clone(),
            parameters,
            SIFTSMALL,
            8,
            StartPointStrategy::RandomSamples {
                nsamples: ONE,
                seed: 0x440f42ab05085ba2,
            },
        )
        .await;

        // perform inplace deletes
        let num_to_replace = 3;
        for id in 0..points_to_delete {
            index
                .inplace_delete(strategy.clone(), ctx, &id, num_to_replace, delete_method)
                .await
                .unwrap();
        }

        //check that each deleted point is present in the delete provider
        for id in 0..points_to_delete {
            assert!(
                index
                    .data_provider
                    .status_by_external_id(ctx, &id)
                    .await
                    .unwrap()
                    .is_deleted()
            );
        }

        // drop deleted neighbors from every point in the index
        let num_start_points = index
            .provider()
            .starting_points()
            .expect("Error in get_starting_point_ids")
            .len();

        let neighbor_accessor = &mut index.provider().neighbors();
        for id in 0..data.nrows() + num_start_points {
            index
                .drop_deleted_neighbors(ctx, neighbor_accessor, id.try_into().unwrap(), false)
                .await
                .unwrap();
        }

        // check that no edges to a deleted vertex exist in the graph
        for id in points_to_delete.into_usize()..data.nrows() + num_start_points {
            assert!(
                !(index.is_any_neighbor_deleted(ctx, neighbor_accessor, id.try_into().unwrap()))
                    .await
                    .expect("Error in is_any_neighbor_deleted")
            );
        }

        // check that each deleted point has a length-zero adjacency list
        let mut adj_list = AdjacencyList::new();
        for id in 0..points_to_delete {
            neighbor_accessor
                .get_neighbors(id, &mut adj_list)
                .await
                .expect("Error in get_neighbors");
            assert!(adj_list.is_empty());
        }

        // check that the graph has no self-loops or repeated vertices
        check_graph_for_self_loops_or_duplicates(
            neighbor_accessor,
            (&index.data_provider).into_iter(),
        )
        .await;
    }

    #[rstest]
    #[tokio::test]
    async fn multi_inplace_delete_on_sift<S>(
        #[values(FullPrecision, Hybrid::new(None))] strategy: S,
        #[values(20, 100)] points_to_delete: u32,
        #[values(
            InplaceDeleteMethod::VisitedAndTopK{k_value:5, l_value:10},
            InplaceDeleteMethod::TwoHopAndOneHop
        )]
        delete_method: InplaceDeleteMethod,
    ) where
        S: InsertStrategy<TestProvider, [f32]>
            + SearchStrategy<TestProvider, [f32]>
            + for<'a> InplaceDeleteStrategy<TestProvider, DeleteElement<'a> = [f32]>
            + Clone
            + Sync,
        for<'a> aliases::InsertPruneAccessor<'a, S, TestProvider, [f32]>: AsElement<&'a [f32]>,
        <S as InsertStrategy<TestProvider, [f32]>>::PruneStrategy: Clone,
    {
        let ctx = &DefaultContext;
        let parameters = InitParams {
            l_build: 10,
            max_degree: 32,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(1).unwrap(),
        };

        let (index, data) = init_from_file(
            strategy.clone(),
            parameters,
            SIFTSMALL,
            8,
            StartPointStrategy::RandomSamples {
                nsamples: ONE,
                seed: 0x440f42ab05085ba2,
            },
        )
        .await;

        // perform inplace deletes
        let num_to_replace = 3;

        let ids: Vec<u32> = (0..points_to_delete).collect();
        // perform inplace deletes

        let ids = Arc::new(ids.as_slice());

        index
            .multi_inplace_delete(
                strategy,
                ctx,
                (&**ids).into(),
                num_to_replace,
                delete_method,
            )
            .await
            .unwrap();

        //check that each deleted point is present in the delete provider
        for id in 0..points_to_delete {
            assert!(
                index
                    .data_provider
                    .status_by_external_id(ctx, &id)
                    .await
                    .unwrap()
                    .is_deleted()
            );
        }

        // drop deleted neighbors from every point in the index
        let num_start_points = index
            .data_provider
            .starting_points()
            .expect("Error in get_starting_point_ids")
            .len();

        let neighbor_accessor = &mut index.provider().neighbors();
        for id in 0..data.nrows() + num_start_points {
            index
                .drop_deleted_neighbors(ctx, neighbor_accessor, id.try_into().unwrap(), false)
                .await
                .unwrap();
        }

        // check that no edges to a deleted vertex exist in the graph
        for id in points_to_delete.into_usize()..data.nrows() + num_start_points {
            assert!(
                !(index.is_any_neighbor_deleted(ctx, neighbor_accessor, id.try_into().unwrap()))
                    .await
                    .expect("Error in is_any_neighbor_deleted")
            );
        }

        // check that each deleted point has a length-zero adjacency list
        let mut adj_list = AdjacencyList::new();
        for id in 0..points_to_delete {
            neighbor_accessor
                .get_neighbors(id, &mut adj_list)
                .await
                .expect("Error in get_neighbors");
            assert!(adj_list.is_empty());
        }

        // check that the graph has no self-loops or repeated vertices
        check_graph_for_self_loops_or_duplicates(
            neighbor_accessor,
            (&index.data_provider).into_iter(),
        )
        .await;
    }

    #[rstest]
    #[tokio::test]
    async fn test_sift_256_vectors_with_consolidate_deletes(
        #[values(20, 100)] points_to_delete: u32,
    ) {
        let ctx = &DefaultContext;
        let parameters = InitParams {
            l_build: 10,
            max_degree: 32,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(1).unwrap(),
        };

        let (index, data) = init_from_file(
            FullPrecision,
            parameters,
            SIFTSMALL,
            8,
            StartPointStrategy::RandomSamples {
                nsamples: ONE,
                seed: 0x440f42ab05085ba2,
            },
        )
        .await;

        for id in 0..points_to_delete {
            index
                .data_provider
                .delete(ctx, &id)
                .await
                .expect("Error in delete");
        }

        // check that each deleted point is present in the delete provider
        for id in 0..points_to_delete {
            assert!(
                index
                    .data_provider
                    .status_by_external_id(ctx, &id)
                    .await
                    .unwrap()
                    .is_deleted()
            );
        }

        // perform consolidation
        let num_start_points = index
            .provider()
            .starting_points()
            .expect("Error in get_starting_point_ids")
            .len();

        let total_points = data.nrows() + num_start_points;
        for id in 0..total_points {
            index
                .consolidate_vector(&FullPrecision, ctx, id.try_into().unwrap())
                .await
                .expect("Error in consolidate_vector");
        }

        let neighbor_accessor = &mut index.provider().neighbors();
        // check that no edges to a deleted vertex exist in the non-deleted vertices
        for id in points_to_delete.into_usize()..total_points {
            assert!(
                !(index.is_any_neighbor_deleted(ctx, neighbor_accessor, id.try_into().unwrap()))
                    .await
                    .expect("Error in is_any_neighbor_deleted")
            );
        }

        // check that the graph has no self-loops or repeated vertices
        check_graph_for_self_loops_or_duplicates(
            neighbor_accessor,
            (&index.data_provider).into_iter(),
        )
        .await;
    }

    #[tokio::test]
    async fn test_final_prune() {
        let ctx = &DefaultContext;
        let max_degree = 32;
        let parameters = InitParams {
            l_build: 15,
            max_degree,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(1).unwrap(),
        };

        let (index, _) = init_from_file(
            FullPrecision,
            parameters,
            SIFTSMALL,
            8,
            StartPointStrategy::RandomSamples {
                nsamples: ONE,
                seed: 0x986ce825cbe015e9,
            },
        )
        .await;

        let neighbor_accessor = &mut index.provider().neighbors();
        // check that we have an unpruned graph
        let stats = index.get_degree_stats(neighbor_accessor).await.unwrap();
        assert!(stats.max_degree.into_usize() > max_degree);

        // prune graph and check that max_degree is respected
        index
            .prune_range(&FullPrecision, ctx, 0..256)
            .await
            .unwrap();
        let stats = index.get_degree_stats(neighbor_accessor).await.unwrap();
        assert!(stats.max_degree.into_usize() <= max_degree);
    }

    #[rstest]
    #[tokio::test]
    async fn test_replace_sift_256_vectors_with_quant_vectors(
        #[values(None, Some(32))] max_fp_vecs_per_prune: Option<usize>,
        #[values(1, 3)] insert_minibatch_size: usize,
    ) {
        let ctx = &DefaultContext;
        let parameters = InitParams {
            l_build: 35,
            max_degree: 32,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(insert_minibatch_size).unwrap(),
        };

        let (index, data) = init_from_file(
            Hybrid::new(max_fp_vecs_per_prune),
            parameters,
            SIFTSMALL,
            32,
            StartPointStrategy::RandomSamples {
                nsamples: ONE,
                seed: 0x812b98835db95971,
            },
        )
        .await;

        let mut indices: Vec<_> = (0..data.nrows()).collect();

        // Randomize the vectors
        let rng = &mut create_rnd_from_seed_in_tests(0x7dc205fcda38d3a3);
        indices.shuffle(rng);
        let mut queries = diskann_utils::views::Matrix::new(0.0, data.nrows(), data.ncols());
        std::iter::zip(queries.row_iter_mut(), indices.iter()).for_each(|(row, i)| {
            row.copy_from_slice(data.row(*i));
        });

        for (pos, query) in queries.row_iter().enumerate() {
            index
                .insert(
                    Hybrid::new(max_fp_vecs_per_prune),
                    ctx,
                    &(pos as u32),
                    query,
                )
                .await
                .unwrap();
        }

        // Check reachability of all nodes
        assert_eq!(
            index
                .count_reachable_nodes(
                    &index.provider().starting_points().unwrap(),
                    &mut index.provider().neighbors()
                )
                .await
                .unwrap(),
            data.nrows() + 1,
        );

        // Check searchability.
        let top_k = 4;
        let search_l = 40;
        let mut ids = vec![0; top_k];
        let mut distances = vec![0.0; top_k];

        for (q, query) in queries.row_iter().enumerate() {
            let gt = groundtruth(queries.as_view(), query, |a, b| SquaredL2::evaluate(a, b));
            let mut result_output_buffer =
                search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            // Full Precision Search.
            index
                .search(
                    &Hybrid::new(max_fp_vecs_per_prune),
                    ctx,
                    query,
                    &SearchParams::new_default(top_k, search_l).unwrap(),
                    &mut result_output_buffer,
                )
                .await
                .unwrap();

            println!(
                "gt = {:?}, ids = {:?}, distance = {:?}",
                &gt[gt.len() - 2 * top_k..],
                ids,
                distances
            );
            assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);
        }
    }

    // Check exact equality between the two-level index using only the full-precision
    // portion and the one-level index.
    async fn test_one_level_index_same_as_two_level_impl(batchsize: NonZeroUsize) {
        let parameters = InitParams {
            l_build: 64,
            max_degree: 16,
            metric: Metric::L2,
            batchsize,
        };

        let start_point = StartPointStrategy::RandomSamples {
            nsamples: ONE,
            seed: 0xe058c9c57864dd1e,
        };

        // This is the two level index.
        let (quant_index, data) =
            init_from_file(FullPrecision, parameters, SIFTSMALL, 8, start_point).await;

        // Next, we initialize and populate the one-level index in the same way.
        let (config, params) = parameters.materialize(data.nrows(), data.ncols()).unwrap();
        let full_index = new_index(config, params, TableBasedDeletes).unwrap();

        build_index(
            &full_index,
            FullPrecision,
            parameters,
            SIFTSMALL,
            start_point,
            data.as_view(),
        )
        .await;

        // Check that the adjacency lists formed for the two indexes are the same.
        let iter = (&quant_index.data_provider).into_iter();
        let mut from_quant = AdjacencyList::new();
        let mut from_full = AdjacencyList::new();
        for id in iter {
            quant_index
                .data_provider
                .neighbors()
                .get_neighbors(id, &mut from_quant)
                .await
                .unwrap();

            full_index
                .data_provider
                .neighbors()
                .get_neighbors(id, &mut from_full)
                .await
                .unwrap();

            from_quant.sort();
            from_full.sort();
            assert_eq!(from_quant, from_full);
        }
    }

    #[tokio::test]
    async fn test_one_level_index_same_as_two_level() {
        test_one_level_index_same_as_two_level_impl(NonZeroUsize::new(1).unwrap()).await;
        test_one_level_index_same_as_two_level_impl(NonZeroUsize::new(10).unwrap()).await;
    }

    /////////////////////////////
    // Flaky Provider Handling //
    /////////////////////////////

    // This test uses a "Flaky" accessor that spuriously fails with non-critical errors to
    // check that such errors are not propagated by DiskANN.
    #[tokio::test]
    async fn test_flaky_build() {
        let parameters = InitParams {
            l_build: 64,
            max_degree: 16,
            metric: Metric::L2,
            batchsize: NonZeroUsize::new(1).unwrap(),
        };

        let start_point = StartPointStrategy::RandomSamples {
            nsamples: ONE,
            seed: 0xb4de0a1298a86eea,
        };

        // This is the two level index.
        let (index, data) = init_from_file(
            inmem::test::Flaky::new(9),
            parameters,
            SIFTSMALL,
            8,
            start_point,
        )
        .await;

        // There should be one more reachable node than points in the dataset to account for
        // the start point.
        let neighbor_accessor = &mut index.provider().neighbors();
        assert_eq!(
            index
                .count_reachable_nodes(
                    &index.provider().starting_points().unwrap(),
                    neighbor_accessor
                )
                .await
                .unwrap(),
            data.nrows() + 1,
        );

        let top_k = 10;
        let search_l = 32;
        let mut ids = vec![0; top_k];
        let mut distances = vec![0.0; top_k];

        // Here, we use elements of the dataset to search the dataset itself.
        //
        // We do this for each query, computing the expected ground truth and verifying
        // that our simple graph search matches.
        //
        // Because this dataset is small, we can expect exact equality.
        let ctx = &DefaultContext;
        for (q, query) in data.row_iter().enumerate() {
            let gt = groundtruth(data.as_view(), query, |a, b| SquaredL2::evaluate(a, b));
            let mut result_output_buffer =
                search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            // Full Precision Search.
            index
                .search(
                    &FullPrecision,
                    ctx,
                    query,
                    &SearchParams::new_default(top_k, search_l).unwrap(),
                    &mut result_output_buffer,
                )
                .await
                .unwrap();

            assert_top_k_exactly_match(q, &gt, &ids, &distances, top_k);
        }
    }

    // This test uses a "Flaky" accessor that spuriously fails with non-critical errors to
    // check that such errors are not propagated by DiskANN.
    #[tokio::test]
    async fn test_flaky_consolidate() {
        // What we need to do is populate a graph with an element that has an adjacency list
        // that exceeds the configured maximum degree.
        //
        // We then need to try to consolidate that element and ensure that retrieval of
        // that element's data results in a transient error.

        // create small index instance
        let dim = 2;
        let (config, parameters) = simplified_builder(
            10,         // l_search
            4,          // max_degree
            Metric::L2, // metric
            dim,        // dim
            10,         // max_points
            no_modify,
        )
        .unwrap();

        let pqtable = model::pq::FixedChunkPQTable::new(
            dim,
            Box::new([0.0, 0.0]),
            Box::new([0.0, 0.0]),
            Box::new([0, 2]),
            None,
        )
        .unwrap();

        let index =
            new_quant_index::<f32, _, _>(config, parameters, pqtable, TableBasedDeletes).unwrap();

        let start_point: &[f32] = &[0.5, 0.5];

        index
            .provider()
            .set_start_points(std::iter::once(start_point))
            .unwrap();

        // vectors are the four corners of a square, with the start point in the middle
        // the middle point forms an edge to each corner, while corners form an edge
        // to their opposite vertex vertically and horizontally as well as the middle
        let vectors = [
            vec![0.0, 0.0], // point 0
            vec![0.0, 1.0], // point 1
            vec![1.0, 0.0], // point 2
            vec![1.0, 1.0], // point 3
            vec![2.0, 2.0], // point 4
            vec![0.0, 2.0], // point 5
            vec![2.0, 0.0], // point 6
        ];
        let adjacency_lists = [
            AdjacencyList::from_iter_untrusted([1, 2, 3, 4, 5]), // point 0
            AdjacencyList::from_iter_untrusted([4, 0, 3, 6]),    // point 1
            AdjacencyList::from_iter_untrusted([4, 3, 0, 6]),    // point 2
            AdjacencyList::from_iter_untrusted([4, 2, 1, 6]),    // point 3
            AdjacencyList::from_iter_untrusted([0, 1, 2, 3, 6]), // point 4
            AdjacencyList::from_iter_untrusted([0, 1, 2, 5, 6]), // point 5
            AdjacencyList::from_iter_untrusted([0, 1, 2, 5, 3]), // point 6 -- start point
        ];

        let ctx = &DefaultContext;
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_graph(neighbor_accessor, &adjacency_lists).await;
        populate_data(&index.data_provider, ctx, &vectors).await;

        let r = index
            .consolidate_vector(&inmem::test::SuperFlaky, ctx, 0)
            .await
            .unwrap();
        assert_eq!(r, ConsolidateKind::FailedVectorRetrieval);
    }

    async fn create_retry_saturated_index(
        retry: NonZeroU32,
        saturated: bool,
    ) -> ANNResult<MemoryIndex<f32>> {
        let (config, params) = simplified_builder(5, 10, Metric::L2, 3, 1001, |builder| {
            builder.insert_retry(graph::config::experimental::InsertRetry::new(
                retry,
                NonZeroU32::new(20).unwrap(),
                saturated,
            ));
        })
        .unwrap();

        let index = new_index::<f32, _>(config, params, NoDeletes).unwrap();
        let mut id_counter = 1;
        let start = vec![0.0, 0.0, 0.0];
        index
            .provider()
            .set_start_points(vec![start.as_slice()].into_iter())
            .unwrap();

        for x in 1..11 {
            for y in 1..11 {
                for z in 1..11 {
                    let vec = vec![x as f32, y as f32, z as f32];
                    index
                        .insert(FullPrecision, &DefaultContext, &id_counter.clone(), &vec)
                        .await?;
                    id_counter += 1;
                }
            }
        }
        Ok(index)
    }

    #[tokio::test]
    async fn test_saturate_index() {
        let index_sat = create_retry_saturated_index(NonZeroU32::new(1).unwrap(), true)
            .await
            .unwrap();
        let mut accessor_sat = inmem::FullAccessor::new(index_sat.provider());
        let res_sat = index_sat.get_degree_stats(&mut accessor_sat).await.unwrap();

        let index_unsat = create_retry_saturated_index(NonZeroU32::new(1).unwrap(), false)
            .await
            .unwrap();
        let mut accessor_unsat = inmem::FullAccessor::new(index_unsat.provider());
        let res_unsat = index_sat
            .get_degree_stats(&mut accessor_unsat)
            .await
            .unwrap();
        assert!(
            res_sat.avg_degree > res_unsat.avg_degree,
            "Saturated index should have higher average degree than the unsaturated index"
        );
    }

    #[tokio::test]
    async fn test_retry_index() {
        let index_sat = create_retry_saturated_index(NonZeroU32::new(3).unwrap(), false)
            .await
            .unwrap();
        let mut accessor_sat = inmem::FullAccessor::new(index_sat.provider());
        let res_sat = index_sat.get_degree_stats(&mut accessor_sat).await.unwrap();

        let index_unsat = create_retry_saturated_index(NonZeroU32::new(1).unwrap(), false)
            .await
            .unwrap();
        let mut accessor_unsat = inmem::FullAccessor::new(index_unsat.provider());
        let res_unsat = index_sat
            .get_degree_stats(&mut accessor_unsat)
            .await
            .unwrap();
        assert!(
            res_sat.avg_degree > res_unsat.avg_degree,
            "Saturated index should have higher average degree than the unsaturated index"
        );
    }

    #[cfg(feature = "experimental_diversity_search")]
    #[tokio::test]
    async fn test_inmemory_search_diversity_search() {
        use diskann::neighbor::AttributeValueProvider;
        use rand::Rng;
        use std::collections::HashMap;

        // Simple test attribute provider
        #[derive(Debug, Clone)]
        struct TestAttributeProvider {
            attributes: HashMap<u32, u32>,
        }
        impl TestAttributeProvider {
            fn new() -> Self {
                Self {
                    attributes: HashMap::new(),
                }
            }
            fn insert(&mut self, id: u32, attribute: u32) {
                self.attributes.insert(id, attribute);
            }
        }
        impl diskann::provider::HasId for TestAttributeProvider {
            type Id = u32;
        }

        impl AttributeValueProvider for TestAttributeProvider {
            type Value = u32;

            fn get(&self, id: Self::Id) -> Option<Self::Value> {
                self.attributes.get(&id).copied()
            }
        }

        // Create test data (256 vectors of 128 dimensions)
        let dim = 128;
        let num_points = 256;
        let mut data_vectors = Vec::new();

        // Generate simple test data
        let mut rng = create_rnd_from_seed_in_tests(42);
        for _ in 0..num_points {
            let vec: Vec<f32> = (0..dim).map(|_| rng.random_range(0.0..1.0)).collect();
            data_vectors.push(vec);
        }

        // Create in-memory index using simplified_builder pattern
        let l_build = 50;
        let max_degree = 32;
        let (config, parameters) =
            simplified_builder(l_build, max_degree, Metric::L2, dim, num_points, no_modify)
                .unwrap();

        let index = new_index::<f32, _>(config, parameters, NoDeletes).unwrap();

        // Set start points - use the first vector as start point
        index
            .provider()
            .set_start_points(std::iter::once(data_vectors[0].as_slice()))
            .unwrap();

        // Insert data into index to build the graph
        for (i, vec) in data_vectors.iter().enumerate() {
            index
                .insert(FullPrecision, &DefaultContext, &(i as u32), vec.as_slice())
                .await
                .unwrap();
        }

        // Create attribute provider with labels (1 to 5)
        let mut attribute_provider = TestAttributeProvider::new();
        for i in 0..num_points {
            let label = ((i % 5) + 1) as u32;
            attribute_provider.insert(i as u32, label);
        }
        // Also add attribute for the start point (ID = num_points = 256)
        // Start points are stored at indices starting from max_points
        attribute_provider.insert(num_points as u32, 1);
        // Wrap in Arc once to avoid cloning the HashMap later
        let attribute_provider = std::sync::Arc::new(attribute_provider);

        // Perform diversity search on a query vector
        let query = vec![0.5f32; dim];
        let return_list_size = 10;
        let search_list_size = 20;
        let diverse_results_k = 1;

        let mut indices = vec![0u32; return_list_size];
        let mut distances = vec![0f32; return_list_size];
        let mut result_output_buffer =
            diskann::graph::IdDistance::new(&mut indices, &mut distances);

        let diverse_params = diskann::graph::DiverseSearchParams::new(
            0, // diverse_attribute_id
            diverse_results_k,
            attribute_provider.clone(),
        );

        let search_params = diskann::graph::SearchParams::new(
            return_list_size,
            search_list_size,
            None, // beam_width
        )
        .unwrap();

        use diskann::graph::search::record::NoopSearchRecord;
        let mut search_record = NoopSearchRecord::new();

        let result = index
            .diverse_search_experimental(
                &FullPrecision,
                &DefaultContext,
                &query,
                &search_params,
                &diverse_params,
                &mut result_output_buffer,
                &mut search_record,
            )
            .await;

        assert!(result.is_ok(), "Expected diversity search to succeed");
        let stats = result.unwrap();

        // Verify results
        assert!(
            stats.result_count as usize <= return_list_size,
            "Expected result count to be <= {}",
            return_list_size
        );
        assert!(
            stats.result_count > 0,
            "Expected to get some search results"
        );

        // Print search results with their attributes
        println!("\n=== In-Memory Diversity Search Results ===");
        println!("Query: [0.5f32; {}]", dim);
        println!("diverse_results_k: {}", diverse_results_k);
        println!("Total results: {}\n", stats.result_count);
        println!("{:<10} {:<15} {:<10}", "Vertex ID", "Distance", "Label");
        println!("{}", "-".repeat(35));
        for i in 0..stats.result_count as usize {
            let attribute_value = attribute_provider.get(indices[i]).unwrap_or(0);
            println!(
                "{:<10} {:<15.2} {:<10}",
                indices[i], distances[i], attribute_value
            );
        }

        // Verify that distances are non-negative and sorted
        for i in 0..(stats.result_count as usize).saturating_sub(1) {
            assert!(distances[i] >= 0.0, "Expected non-negative distance");
            assert!(
                distances[i] <= distances[i + 1],
                "Expected distances to be sorted in ascending order"
            );
        }

        // Verify diversity: Check that we have diverse attribute values in the results
        let mut attribute_counts = HashMap::new();
        for item in indices.iter().take(stats.result_count as usize) {
            if let Some(attribute_value) = attribute_provider.get(*item) {
                *attribute_counts.entry(attribute_value).or_insert(0) += 1;
            }
        }

        // Print attribute distribution
        println!("\n=== Attribute Distribution ===");
        let mut sorted_attrs: Vec<_> = attribute_counts.iter().collect();
        sorted_attrs.sort_by_key(|(k, _)| *k);
        for (attribute_value, count) in &sorted_attrs {
            println!(
                "Label {}: {} occurrences (max allowed: {})",
                attribute_value, count, diverse_results_k
            );
        }
        println!("Total unique labels: {}", attribute_counts.len());
        println!("================================\n");

        // Verify diversity constraints
        for (attribute_value, count) in &attribute_counts {
            println!(
                "Assert: Label {} has {} occurrences (max: {})",
                attribute_value, count, diverse_results_k
            );
            assert!(
                *count <= diverse_results_k,
                "Attribute value {} appears {} times, which exceeds diverse_results_k of {}",
                attribute_value,
                count,
                diverse_results_k
            );
        }

        // Verify that we have multiple different attribute values (diversity)
        println!(
            "Assert: Found {} unique labels (expected at least 2)",
            attribute_counts.len()
        );
        assert!(
            attribute_counts.len() >= 2,
            "Expected at least 2 different attribute values for diversity, got {}",
            attribute_counts.len()
        );
    }

    /////////////////////////////////////
    // Multi-Hop Callback Edge Cases   //
    /////////////////////////////////////

    /// Filter that rejects all candidates via on_visit callback.
    /// Used to test the fallback behavior when all candidates are rejected.
    #[derive(Debug)]
    struct RejectAllFilter {
        allowed_in_results: HashSet<u32>,
    }

    impl RejectAllFilter {
        fn only<I: IntoIterator<Item = u32>>(ids: I) -> Self {
            Self {
                allowed_in_results: ids.into_iter().collect(),
            }
        }
    }

    impl QueryLabelProvider<u32> for RejectAllFilter {
        fn is_match(&self, vec_id: u32) -> bool {
            self.allowed_in_results.contains(&vec_id)
        }

        fn on_visit(&self, _neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
            QueryVisitDecision::Reject
        }
    }

    /// Filter that tracks visit order and can terminate early.
    #[derive(Debug)]
    struct TerminatingFilter {
        target: u32,
        hits: Mutex<Vec<u32>>,
    }

    impl TerminatingFilter {
        fn new(target: u32) -> Self {
            Self {
                target,
                hits: Mutex::new(Vec::new()),
            }
        }

        fn hits(&self) -> Vec<u32> {
            self.hits
                .lock()
                .expect("mutex should not be poisoned")
                .clone()
        }
    }

    impl QueryLabelProvider<u32> for TerminatingFilter {
        fn is_match(&self, vec_id: u32) -> bool {
            vec_id == self.target
        }

        fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
            self.hits
                .lock()
                .expect("mutex should not be poisoned")
                .push(neighbor.id);
            if neighbor.id == self.target {
                QueryVisitDecision::Terminate
            } else {
                QueryVisitDecision::Accept(neighbor)
            }
        }
    }

    #[tokio::test]
    async fn test_multihop_reject_all_returns_zero_results() {
        // When on_visit rejects all candidates, the search should return zero results
        // because rejected candidates don't get added to the frontier.
        let dim = 3;
        let grid_size: usize = 4;
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, no_modify).unwrap();

        let mut adjacency_lists = utils::genererate_3d_grid_adj_list(grid_size as u32);
        let mut vectors = f32::generate_grid(dim, grid_size);

        adjacency_lists.push((num_points as u32 - 1).into());
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim),
            &mut create_rnd_from_seed_in_tests(0x1234567890abcdef),
            1usize,
        )
        .unwrap();

        let index = new_quant_index::<f32, _, _>(config, parameters, table, NoDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_data(&index.data_provider, &DefaultContext, &vectors).await;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        let query = vec![grid_size as f32; dim];

        let mut ids = vec![0; 10];
        let mut distances = vec![0.0; 10];
        let mut result_output_buffer =
            search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        // Allow only the first start point (0) in results via is_match,
        // but reject everything via on_visit
        let filter = RejectAllFilter::only([0_u32]);

        let stats = index
            .multihop_search(
                &FullPrecision,
                &DefaultContext,
                query.as_slice(),
                &SearchParams::new_default(10, 20).unwrap(),
                &mut result_output_buffer,
                &filter,
            )
            .await
            .unwrap();

        // When all candidates are rejected via on_visit, result_count should be 0
        // because rejected candidates are not added to the search frontier
        assert_eq!(
            stats.result_count, 0,
            "rejecting all via on_visit should result in zero results"
        );
    }

    #[tokio::test]
    async fn test_multihop_early_termination() {
        // Test that Terminate causes the search to stop early
        let dim = 3;
        let grid_size: usize = 5;
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, no_modify).unwrap();

        let mut adjacency_lists = utils::genererate_3d_grid_adj_list(grid_size as u32);
        let mut vectors = f32::generate_grid(dim, grid_size);

        adjacency_lists.push((num_points as u32 - 1).into());
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim),
            &mut create_rnd_from_seed_in_tests(0xfedcba0987654321),
            1usize,
        )
        .unwrap();

        let index = new_quant_index::<f32, _, _>(config, parameters, table, NoDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_data(&index.data_provider, &DefaultContext, &vectors).await;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        let query = vec![grid_size as f32; dim];

        let mut ids = vec![0; 10];
        let mut distances = vec![0.0; 10];
        let mut result_output_buffer =
            search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        // Target a point in the middle of the grid
        let target = (num_points / 2) as u32;
        let filter = TerminatingFilter::new(target);

        let stats = index
            .multihop_search(
                &FullPrecision,
                &DefaultContext,
                query.as_slice(),
                &SearchParams::new_default(10, 40).unwrap(),
                &mut result_output_buffer,
                &filter,
            )
            .await
            .unwrap();

        let hits = filter.hits();

        // The search should have terminated after finding the target
        assert!(
            hits.contains(&target),
            "search should have visited the target"
        );
        assert!(
            stats.result_count >= 1,
            "should have at least one result (the target)"
        );
    }

    #[tokio::test]
    async fn test_multihop_distance_adjustment_affects_ranking() {
        // Test that distance adjustments in on_visit affect the final ranking
        let dim = 3;
        let grid_size: usize = 4;
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, no_modify).unwrap();

        let mut adjacency_lists = utils::genererate_3d_grid_adj_list(grid_size as u32);
        let mut vectors = f32::generate_grid(dim, grid_size);

        adjacency_lists.push((num_points as u32 - 1).into());
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim),
            &mut create_rnd_from_seed_in_tests(0xabcdef1234567890),
            1usize,
        )
        .unwrap();

        let index = new_quant_index::<f32, _, _>(config, parameters, table, NoDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_data(&index.data_provider, &DefaultContext, &vectors).await;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        let query = vec![0.0; dim]; // Query at origin

        // First, run without adjustment to get baseline
        let mut baseline_ids = vec![0; 10];
        let mut baseline_distances = vec![0.0; 10];
        let mut baseline_buffer =
            search_output_buffer::IdDistance::new(&mut baseline_ids, &mut baseline_distances);

        let baseline_stats = index
            .multihop_search(
                &FullPrecision,
                &DefaultContext,
                query.as_slice(),
                &SearchParams::new_default(10, 20).unwrap(),
                &mut baseline_buffer,
                &EvenFilter, // Just filter to even IDs
            )
            .await
            .unwrap();

        // Now run with a filter that boosts a specific far-away point
        let boosted_point = (num_points - 2) as u32; // A point far from origin
        let filter = CallbackFilter::new(u32::MAX, boosted_point, 0.01); // Shrink its distance

        let mut adjusted_ids = vec![0; 10];
        let mut adjusted_distances = vec![0.0; 10];
        let mut adjusted_buffer =
            search_output_buffer::IdDistance::new(&mut adjusted_ids, &mut adjusted_distances);

        let adjusted_stats = index
            .multihop_search(
                &FullPrecision,
                &DefaultContext,
                query.as_slice(),
                &SearchParams::new_default(10, 20).unwrap(),
                &mut adjusted_buffer,
                &filter,
            )
            .await
            .unwrap();

        // Both searches should return results
        assert!(
            baseline_stats.result_count > 0,
            "baseline should have results"
        );
        assert!(
            adjusted_stats.result_count > 0,
            "adjusted should have results"
        );

        // If the boosted point was visited and adjusted, it should appear earlier
        // in the adjusted results than in the baseline (or appear when it didn't before)
        let boosted_in_baseline = baseline_ids
            .iter()
            .take(baseline_stats.result_count as usize)
            .position(|&id| id == boosted_point);
        let boosted_in_adjusted = adjusted_ids
            .iter()
            .take(adjusted_stats.result_count as usize)
            .position(|&id| id == boosted_point);

        // The distance adjustment should have some effect if the point was visited
        if filter.hits().contains(&boosted_point) {
            assert!(
                boosted_in_adjusted.is_some(),
                "boosted point should appear in adjusted results when visited"
            );
            if let (Some(baseline_pos), Some(adjusted_pos)) =
                (boosted_in_baseline, boosted_in_adjusted)
            {
                assert!(
                    adjusted_pos <= baseline_pos,
                    "boosted point should rank equal or better after distance reduction"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_multihop_terminate_stops_traversal() {
        // Test that Terminate (without accept) stops traversal immediately
        #[derive(Debug)]
        struct TerminateAfterN {
            max_visits: usize,
            visits: Mutex<usize>,
        }

        impl TerminateAfterN {
            fn new(max_visits: usize) -> Self {
                Self {
                    max_visits,
                    visits: Mutex::new(0),
                }
            }

            fn visit_count(&self) -> usize {
                *self.visits.lock().unwrap()
            }
        }

        impl QueryLabelProvider<u32> for TerminateAfterN {
            fn is_match(&self, _: u32) -> bool {
                true
            }

            fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
                let mut visits = self.visits.lock().unwrap();
                *visits += 1;
                if *visits >= self.max_visits {
                    QueryVisitDecision::Terminate
                } else {
                    QueryVisitDecision::Accept(neighbor)
                }
            }
        }

        let dim = 3;
        let grid_size: usize = 5;
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let (config, parameters) =
            simplified_builder(l, max_degree, Metric::L2, dim, num_points, no_modify).unwrap();

        let mut adjacency_lists = utils::genererate_3d_grid_adj_list(grid_size as u32);
        let mut vectors = f32::generate_grid(dim, grid_size);

        adjacency_lists.push((num_points as u32 - 1).into());
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim),
            &mut create_rnd_from_seed_in_tests(0x9876543210fedcba),
            1usize,
        )
        .unwrap();

        let index = new_quant_index::<f32, _, _>(config, parameters, table, NoDeletes).unwrap();
        let neighbor_accessor = &mut index.provider().neighbors();
        populate_data(&index.data_provider, &DefaultContext, &vectors).await;
        populate_graph(neighbor_accessor, &adjacency_lists).await;

        let query = vec![grid_size as f32; dim];

        let mut ids = vec![0; 10];
        let mut distances = vec![0.0; 10];
        let mut result_output_buffer =
            search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        let max_visits = 5;
        let filter = TerminateAfterN::new(max_visits);

        let _stats = index
            .multihop_search(
                &FullPrecision,
                &DefaultContext,
                query.as_slice(),
                &SearchParams::new_default(10, 100).unwrap(), // Large L to ensure we'd visit more without termination
                &mut result_output_buffer,
                &filter,
            )
            .await
            .unwrap();

        // The search should have stopped after max_visits
        assert!(
            filter.visit_count() <= max_visits + 10, // Allow some slack for beam expansion
            "search should have terminated early, got {} visits",
            filter.visit_count()
        );
    }
}
