/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Deref, sync::Arc};

use diskann::{ANNError, ANNResult, utils::object_pool::ObjectPool};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

// Concrete implementations
use super::{cosine::DirectCosine, innerproduct::TableIP, l2::TableL2};
use crate::model::pq::fixed_chunk_pq_table::FixedChunkPQTable;

/// A quantized computation that works for multiple distance functions.
#[derive(Debug)]
pub enum QueryComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    L2(TableL2<T>),
    IP(TableIP<T>),
    Cosine(DirectCosine<T>),
}

impl<T> QueryComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    /// Create a new query computer implementing the `PreprocessedDistanceFunction` API.
    ///
    /// The returned object will implement the requested distance according to the provided
    /// `Metric.
    ///
    /// Currently supported values for `metric` are:
    ///
    /// * `Metric::L2`
    /// * `Metric::InnerProduct`
    /// * `Metric::Cosine`
    /// * `Metric::CosineNormalized - partially supported as it is currently computed using L2`
    ///
    /// # Notes on CosineNormalized
    ///
    /// This is a temporary fix made with the following rationale: Our implementation of
    /// CosineNormalized yielding a similarity score for vectors `x` and `y` can be computed as
    /// follows:
    /// ```text
    /// s = 1 - <x, y> / (||x|| * ||y||)
    /// ```
    /// where `<x, y>` denotes inner product and `||x||` is the L2 norm. When x and y are
    /// normalized, this simplified to
    /// ```text
    /// s = 1 - <x, y>
    /// ```
    /// The squared L2 distance can be computed as follows:
    /// ```text
    /// s = ||x||^2 + ||y||^2 - 2<x, y>
    /// ```
    /// When vectors are normalized, this becomes
    /// ```text
    /// s = 2 - 2<x, y> = 2 * (1 - <x, y>)
    /// ```
    /// In other words, the similarity score for the squared L2 distance in an ideal world is
    /// 2 times that for cosine similarity. Therefore, squared L2 may serves as a stand-in for
    /// cosine normalized as ordering is preserved.
    ///
    /// Even though PQ does not necessarily preserve the norms of compressed vectors, using L2
    /// for Cosine Normalized seems to work well enough in practice to work as a temporary fix.
    pub fn new<U>(
        table: T,
        metric: Metric,
        query: &[U],
        pool: Option<Arc<ObjectPool<Vec<f32>>>>,
    ) -> ANNResult<Self>
    where
        U: Into<f32> + Copy,
    {
        let result = match metric {
            Metric::L2 => Self::L2(TableL2::new(table, query, pool)?),
            Metric::InnerProduct => Self::IP(TableIP::new(table, query, pool)?),
            Metric::Cosine => Self::Cosine(DirectCosine::new(table, query)?),
            Metric::CosineNormalized => Self::L2(TableL2::new(table, query, pool)?),
        };
        Ok(result)
    }
}

impl<T> PreprocessedDistanceFunction<&[u8], f32> for QueryComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    fn evaluate_similarity(&self, changing: &[u8]) -> f32 {
        match self {
            QueryComputer::L2(f) => PreprocessedDistanceFunction::evaluate_similarity(f, changing),
            QueryComputer::IP(f) => PreprocessedDistanceFunction::evaluate_similarity(f, changing),
            QueryComputer::Cosine(f) => {
                PreprocessedDistanceFunction::evaluate_similarity(f, changing)
            }
        }
    }
}

impl<T> PreprocessedDistanceFunction<&Vec<u8>, f32> for QueryComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    fn evaluate_similarity(&self, changing: &Vec<u8>) -> f32 {
        self.evaluate_similarity(changing.as_slice())
    }
}

impl<T> PreprocessedDistanceFunction<&&[u8], f32> for QueryComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    fn evaluate_similarity(&self, changing: &&[u8]) -> f32 {
        let changing: &[u8] = changing;
        self.evaluate_similarity(changing)
    }
}

/// Pre-dispatched distance functions for the `FixedChunkPQTable`.
#[derive(Debug)]
pub struct VTable {
    pub distance_fn: fn(&FixedChunkPQTable, &[f32], &[u8]) -> f32,
    pub distance_fn_qq: fn(&FixedChunkPQTable, &[u8], &[u8]) -> f32,
}

impl Clone for VTable {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for VTable {}

impl VTable {
    pub fn new(distance: Metric) -> Self {
        let distance_fn: fn(&FixedChunkPQTable, &[f32], &[u8]) -> f32 = match distance {
            Metric::L2 => FixedChunkPQTable::l2_distance,
            Metric::Cosine => FixedChunkPQTable::cosine_distance,
            Metric::InnerProduct => FixedChunkPQTable::inner_product,
            Metric::CosineNormalized => FixedChunkPQTable::cosine_normalized_distance,
        };

        let distance_fn_qq: fn(&FixedChunkPQTable, &[u8], &[u8]) -> f32 = match distance {
            Metric::L2 => FixedChunkPQTable::qq_l2_distance,
            Metric::Cosine => FixedChunkPQTable::qq_cosine_distance,
            Metric::InnerProduct => FixedChunkPQTable::qq_ip_distance,
            Metric::CosineNormalized => FixedChunkPQTable::qq_cosine_distance,
        };

        Self {
            distance_fn,
            distance_fn_qq,
        }
    }
}

/// A distance computer for computing random access distances.
/// That is, distances where preprocessing is too expensive to be worth it.
///
/// This object is capable of computing both fullprecision-quant distances and quant-quant
/// distances.
///
/// # Internals (subject to change)
///
/// Internally, a mini v-table is kept to invoke the correct distance function.
#[derive(Debug)]
pub struct DistanceComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    table: T,
    vtable: VTable,
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum DistanceComputerConstructionError {
    #[error("random access computer does not support OPQ")]
    OPQNotSupported,
}

impl From<DistanceComputerConstructionError> for ANNError {
    #[track_caller]
    fn from(value: DistanceComputerConstructionError) -> ANNError {
        ANNError::log_pq_error(value)
    }
}

impl<T> DistanceComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    /// Create a new distance computer implementing the `DistanceFunction` API for
    /// full-precision/quant and quant/quant combinations.
    ///
    /// The returned object will implement the requested distance according to the provided
    /// `Metric.
    ///
    /// Currently supported values for `metric` are:
    ///
    /// * `Metric::L2`
    /// * `Metric::InnerProduct`
    /// * `Metric::Cosine`
    /// * `Metric::CosineNormalized`
    ///
    pub fn new(table: T, distance: Metric) -> Result<Self, DistanceComputerConstructionError> {
        // Check for OPQ usage - bail if it is enabled.
        if table.has_opq() {
            return Err(DistanceComputerConstructionError::OPQNotSupported);
        }

        Ok(Self {
            table,
            vtable: VTable::new(distance),
        })
    }
}

const INVALID_PQ_DIMENSION: &str = "invalid PQ dimension";

/// Perform a comparison between a full-precision vector and quantized vector.
impl<T> DistanceFunction<&[f32], &[u8], f32> for DistanceComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, fp: &[f32], q: &[u8]) -> f32 {
        assert_eq!(
            q.len(),
            self.table.get_num_chunks(),
            "{}",
            INVALID_PQ_DIMENSION
        );
        (self.vtable.distance_fn)(&self.table, fp, q)
    }
}

/// Perform a comparison between two quantized vectors.
impl<T> DistanceFunction<&[u8], &[u8], f32> for DistanceComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, q0: &[u8], q1: &[u8]) -> f32 {
        let num_pq_chunks = self.table.get_num_chunks();
        assert_eq!(q0.len(), num_pq_chunks, "{}", INVALID_PQ_DIMENSION);
        assert_eq!(q1.len(), num_pq_chunks, "{}", INVALID_PQ_DIMENSION);
        (self.vtable.distance_fn_qq)(&self.table, q0, q1)
    }
}

/// Perform a comparison between a full-precision vector and quantized vector.
impl<T> DistanceFunction<&[f32], &&[u8], f32> for DistanceComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, fp: &[f32], q: &&[u8]) -> f32 {
        let q: &[u8] = q;
        self.evaluate_similarity(fp, q)
    }
}

impl<T> DistanceFunction<&[f32], &Vec<u8>, f32> for DistanceComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, fp: &[f32], q: &Vec<u8>) -> f32 {
        self.evaluate_similarity(fp, q.as_slice())
    }
}

/// Perform a comparison between two quantized vectors.
impl<T> DistanceFunction<&&[u8], &&[u8], f32> for DistanceComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, q0: &&[u8], q1: &&[u8]) -> f32 {
        let q0: &[u8] = q0;
        let q1: &[u8] = q1;
        self.evaluate_similarity(q0, q1)
    }
}

/// Perform a comparison between two quantized vectors.
impl<T> DistanceFunction<&Vec<u8>, &Vec<u8>, f32> for DistanceComputer<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, q0: &Vec<u8>, q1: &Vec<u8>) -> f32 {
        self.evaluate_similarity(q0.as_slice(), q1.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use approx::assert_relative_eq;
    use diskann_vector::{
        Half, Norm, PureDistanceFunction,
        distance::{Cosine, CosineNormalized, InnerProduct, SquaredL2},
        norm::FastL2Norm,
    };
    use rand::SeedableRng;
    use rstest::rstest;

    use super::{
        super::test_utils::{self, TestDistribution},
        *,
    };

    // A wrapper for the `DistanceComputer` that enables it to behave like a
    // `PreprocessedDistanceFunction`.
    //
    // This lets us reuse the testing infrastructure for the `QueryComputer`.
    struct PreprocessedWrapper<T>
    where
        T: Deref<Target = FixedChunkPQTable>,
    {
        table: DistanceComputer<T>,
        query: Vec<f32>,
    }

    impl<T> PreprocessedDistanceFunction<&[u8], f32> for PreprocessedWrapper<T>
    where
        T: Deref<Target = FixedChunkPQTable>,
    {
        fn evaluate_similarity(&self, x: &[u8]) -> f32 {
            self.table.evaluate_similarity(&*self.query, x)
        }
    }

    ////////
    // L2 //
    ////////

    #[rstest]
    fn test_l2<T>(
        #[values(PhantomData::<f32>, PhantomData::<Half>, PhantomData::<i8>, PhantomData::<u8>)]
        _marker: PhantomData<T>,
        #[values(false, true)] use_opq: bool,
    ) where
        T: Into<f32> + TestDistribution,
    {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x83aa68de5765b565);
        for dim in [50, 51] {
            for pq_chunks in [8, 19, 50] {
                for num_pivots in [10, 200, 256] {
                    if pq_chunks > dim {
                        continue;
                    }

                    let config = test_utils::TableConfig {
                        dim,
                        pq_chunks,
                        num_pivots,
                        start_value: 0.0,
                        use_opq,
                    };

                    let table = test_utils::seed_pivot_table(config);
                    let num_trials = 100;

                    let errors = test_utils::RelativeAndAbsolute {
                        relative: 6e-7,
                        absolute: 0.0,
                    };

                    test_utils::test_l2_inner(
                        |table: &FixedChunkPQTable, query: &[T]| {
                            QueryComputer::new(table, Metric::L2, query, None).unwrap()
                        },
                        &table,
                        num_trials,
                        config,
                        &mut rng,
                        errors,
                    );

                    if !use_opq {
                        test_utils::test_l2_inner(
                            |table: &FixedChunkPQTable, query: &[T]| PreprocessedWrapper {
                                table: DistanceComputer::new(table, Metric::L2).unwrap(),
                                query: query.iter().map(|i| <T as Into<f32>>::into(*i)).collect(),
                            },
                            &table,
                            num_trials,
                            config,
                            &mut rng,
                            errors,
                        );
                    }
                }
            }
        }
    }

    //////////////////
    // InnerProduct //
    //////////////////

    #[rstest]
    #[case(PhantomData::<f32>)]
    #[case(PhantomData::<Half>)]
    #[case(PhantomData::<i8>)]
    #[case(PhantomData::<u8>)]
    fn test_innerproduct<T>(#[case] _marker: PhantomData<T>)
    where
        T: Into<f32> + TestDistribution,
    {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc392d773dc8de593);
        for dim in [12, 15, 128] {
            for pq_chunks in [2, 5, 15] {
                for num_pivots in [16, 58, 256] {
                    if pq_chunks > dim {
                        continue;
                    }

                    let config = test_utils::TableConfig {
                        dim,
                        pq_chunks,
                        num_pivots,
                        start_value: 0.0,
                        use_opq: false,
                    };

                    let table = test_utils::seed_pivot_table(config);
                    let num_trials = 100;

                    let errors = test_utils::RelativeAndAbsolute {
                        relative: 6.0e-4,
                        absolute: 5.0e-3,
                    };

                    test_utils::test_ip_inner(
                        |table: &FixedChunkPQTable, query: &[T]| {
                            QueryComputer::new(table, Metric::InnerProduct, query, None).unwrap()
                        },
                        &table,
                        num_trials,
                        config,
                        &mut rng,
                        errors,
                    );

                    test_utils::test_ip_inner(
                        |table: &FixedChunkPQTable, query: &[T]| PreprocessedWrapper {
                            table: DistanceComputer::new(table, Metric::InnerProduct).unwrap(),
                            query: query.iter().map(|i| <T as Into<f32>>::into(*i)).collect(),
                        },
                        &table,
                        num_trials,
                        config,
                        &mut rng,
                        errors,
                    );
                }
            }
        }
    }

    ////////////
    // Cosine //
    ////////////

    #[rstest]
    #[case(PhantomData::<f32>)]
    #[case(PhantomData::<Half>)]
    #[case(PhantomData::<i8>)]
    #[case(PhantomData::<u8>)]
    fn test_cosine<T>(#[case] _marker: PhantomData<T>)
    where
        T: Into<f32> + TestDistribution,
    {
        // RNG
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc33529acbe474958);
        let num_trials = 20;

        for dim in [64, 117, 128] {
            for pq_chunks in [2, 5, 15] {
                for num_pivots in [20, 77, 256] {
                    if pq_chunks > dim {
                        continue;
                    }
                    // Table and computer creation.
                    let config = test_utils::TableConfig {
                        dim,
                        pq_chunks,
                        num_pivots,
                        start_value: 0.0,
                        use_opq: false,
                    };
                    let table = test_utils::seed_pivot_table(config);
                    let errors = test_utils::RelativeAndAbsolute {
                        relative: 2.0e-7,
                        absolute: 0.0,
                    };

                    test_utils::test_cosine_inner(
                        |table: &FixedChunkPQTable, query: &[T]| {
                            QueryComputer::new(table, Metric::Cosine, query, None).unwrap()
                        },
                        &table,
                        num_trials,
                        config,
                        &mut rng,
                        errors,
                    );

                    test_utils::test_cosine_inner(
                        |table: &FixedChunkPQTable, query: &[T]| PreprocessedWrapper {
                            table: DistanceComputer::new(table, Metric::Cosine).unwrap(),
                            query: query.iter().map(|i| <T as Into<f32>>::into(*i)).collect(),
                        },
                        &table,
                        num_trials,
                        config,
                        &mut rng,
                        errors,
                    );
                }
            }
        }
    }

    ///////////////////////////
    // Quant-Quant Distances //
    ///////////////////////////

    fn normalize(x: &mut [f32]) {
        let norm: f32 = (FastL2Norm).evaluate(&*x);
        x.iter_mut().for_each(|i| *i /= norm);
    }

    #[rstest]
    #[case(20, 7)]
    #[case(200, 7)]
    #[case(20, 20)]
    fn test_quant_quant_distances(#[case] dim: usize, #[case] pq_chunks: usize) {
        let config = test_utils::TableConfig {
            dim,
            pq_chunks,
            num_pivots: 20,
            start_value: 0.0,
            use_opq: false,
        };

        let table = test_utils::seed_pivot_table(config);

        let num_trials = 100;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xd49347d0d870ab83);
        for _ in 0..num_trials {
            let code0 =
                test_utils::generate_random_code(config.num_pivots, config.pq_chunks, &mut rng);
            let code1 =
                test_utils::generate_random_code(config.num_pivots, config.pq_chunks, &mut rng);

            let mut v0 = test_utils::generate_expected_vector(
                &code0,
                table.get_chunk_offsets(),
                config.start_value,
            );
            let mut v1 = test_utils::generate_expected_vector(
                &code1,
                table.get_chunk_offsets(),
                config.start_value,
            );

            let squared_l2 = DistanceComputer::new(&table, Metric::L2).unwrap();
            let expected: f32 = SquaredL2::evaluate(&*v0, &*v1);
            assert_eq!(squared_l2.evaluate_similarity(&*code0, &*code1), expected);

            let inner_product = DistanceComputer::new(&table, Metric::InnerProduct).unwrap();
            let expected: f32 = InnerProduct::evaluate(&*v0, &*v1);
            assert_eq!(
                inner_product.evaluate_similarity(&*code0, &*code1),
                expected,
            );

            let cosine = DistanceComputer::new(&table, Metric::Cosine).unwrap();
            let sim: f32 = cosine.evaluate_similarity(&*code0, &*code1);
            assert!(0.0 <= sim);
            assert!(sim <= 2.0);

            let expected: f32 = Cosine::evaluate(&*v0, &*v1);
            assert_eq!(sim, expected);

            normalize(&mut v0);
            normalize(&mut v1);

            let cosine_normalized =
                DistanceComputer::new(&table, Metric::CosineNormalized).unwrap();
            let expected: f32 = CosineNormalized::evaluate(&*v0, &*v1);
            assert_relative_eq!(
                cosine_normalized.evaluate_similarity(&*code0, &*code1),
                expected,
                max_relative = 4.0e-6,
            );
        }
    }

    #[test]
    fn test_construction_failure_on_opq() {
        let table = FixedChunkPQTable::new(
            2,
            Box::new([0.0; 2 * 2]),
            Box::new([0.0, 0.0]),
            Box::new([0, 1, 2]),
            Some(Box::new([0.0; 2 * 2])),
        )
        .unwrap();

        let v = DistanceComputer::new(&table, Metric::L2);
        assert!(v.is_err());
        assert_eq!(
            v.unwrap_err().to_string(),
            "random access computer does not support OPQ"
        );
    }
}
