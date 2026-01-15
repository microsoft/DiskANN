/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, Mutex};

use arc_swap::{ArcSwap, Guard};
use diskann::{ANNError, ANNResult};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::model::{
    FixedChunkPQTable,
    pq::{distance::multi, generate_pq_data_from_pivots_from_membuf},
};

/// The discriminant type for PQ vector versions.
type VersionId = u8;
type VersionedPQVector = multi::VersionedPQVector<VersionId>;
type TableType = Arc<FixedChunkPQTable>;
type MultiTable = multi::MultiTable<TableType, VersionId>;
type QueryComputer = multi::MultiQueryComputer<TableType, VersionId>;
type DistanceComputer = multi::MultiDistanceComputer<TableType, VersionId>;

/// A provider that has two PQ schemas.
pub struct TestMultiPQProviderAsync {
    max_vectors: usize,
    num_start_points: usize,
    quant_vectors: Vec<ArcSwap<VersionedPQVector>>,
    table_new: Arc<FixedChunkPQTable>,
    /// The secondary schema to use. If `None`, then only one schema is used and all
    /// vectors added to/provided by this class will be assigned to table1.
    table_old: Option<Arc<FixedChunkPQTable>>,
    metric: Metric,
    /// A ratio between 0.0 and 1.0 deciding which schema vectors are assigned to.
    split: f64,
    rng: Mutex<StdRng>,
}

impl TestMultiPQProviderAsync {
    pub fn new(
        metric: Metric,
        max_vectors: usize,
        num_start_points: usize,
        table_new: FixedChunkPQTable,
        table_old: Option<FixedChunkPQTable>,
        split: f64,
        seed: u64,
    ) -> Self {
        let quant_vectors = (0..max_vectors + num_start_points)
            .map(|_| ArcSwap::new(Arc::new(VersionedPQVector::new(Vec::new(), 0))))
            .collect();

        Self {
            max_vectors,
            num_start_points,
            quant_vectors,
            table_new: Arc::new(table_new),
            table_old: table_old.map(Arc::new),
            metric,
            split,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn num_pq_chunks(&self) -> usize {
        self.table_new.get_num_chunks()
    }

    pub fn multi_table(&self) -> Result<MultiTable, multi::EqualVersionsError> {
        match &self.table_old {
            None => Ok(MultiTable::one(self.table_new.clone(), 1)),
            Some(table_old) => MultiTable::two(self.table_new.clone(), table_old.clone(), 2, 1),
        }
    }

    pub fn get_query_computer<T>(&self, query: &[T]) -> ANNResult<NoneToInfinity<QueryComputer>>
    where
        T: Copy + Into<f32>,
    {
        let table = self.multi_table().map_err(|err| {
            ANNError::log_index_error(format_args!("Table consruction failed with: {}", err))
        })?;
        Ok(NoneToInfinity(QueryComputer::new(
            table,
            self.metric,
            query,
        )?))
    }

    pub fn get_distance_computer(&self) -> ANNResult<NoneToInfinity<DistanceComputer>> {
        let table = self.multi_table().map_err(|err| {
            ANNError::log_index_error(format_args!("Table consruction failed with: {}", err))
        })?;
        Ok(NoneToInfinity(DistanceComputer::new(table, self.metric)?))
    }

    pub fn get_vector(&self, id: usize) -> ANNResult<Guard<Arc<VersionedPQVector>>> {
        match self.quant_vectors.get(id) {
            Some(vector) => Ok(vector.load()),
            None => Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            )),
        }
    }

    pub fn set_vector<T>(&self, id: usize, v: &[T]) -> ANNResult<()>
    where
        T: Copy + Into<f32>,
    {
        if id >= self.max_vectors + self.num_start_points {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            ));
        }
        if v.len() != self.table_new.get_dim() {
            return Err(ANNError::log_index_error(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }

        let vector_f32: Vec<f32> = v.iter().map(|&x| x.into()).collect::<Vec<f32>>();

        // Determine the table and version to use for this vector.
        //
        // If only one table is used, than use that table.
        //
        // Otherwise, generate a random number between 0 and 1 to determine which table
        // this vector will get assigned to.
        let (table, version): (_, u8) = match &self.table_old {
            None => (&self.table_new, 1),
            Some(table_old) => {
                let v: f64 = {
                    let mut guard = self.rng.lock().map_err(|_| {
                        ANNError::log_lock_poison_error("in multi provider".to_string())
                    })?;
                    guard.random()
                };
                if v <= self.split {
                    (table_old, 1)
                } else {
                    (&self.table_new, 2)
                }
            }
        };

        let mut quant_vector: Vec<u8> = vec![0; table.get_num_chunks()];
        if generate_pq_data_from_pivots_from_membuf(
            &vector_f32,
            table.get_pq_table(),
            table.get_num_centers(),
            Some(table.get_centroids()),
            table.get_chunk_offsets(),
            &mut quant_vector,
        )
        .is_err()
        {
            return Err(ANNError::log_index_error("Error in generating PQ data."));
        }

        let new = Arc::new(VersionedPQVector::new(quant_vector, version));
        self.quant_vectors[id].swap(new);
        Ok(())
    }
}

/// A distance adaptor that converts the `None`s returned by the multi distance
/// providers into infinities.
///
/// This effectively ignores vectors for which the ID's are not recognized.
#[repr(transparent)]
pub struct NoneToInfinity<T>(T);

impl<B, T> DistanceFunction<&[f32], &Guard<Arc<B>>, f32> for NoneToInfinity<T>
where
    T: for<'a, 'b> DistanceFunction<&'a [f32], &'a B, Option<f32>>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, x: &[f32], y: &Guard<Arc<B>>) -> f32 {
        self.0.evaluate_similarity(x, &**y).unwrap_or(f32::INFINITY)
    }
}

impl<A, B, T> DistanceFunction<&Guard<Arc<A>>, &Guard<Arc<B>>, f32> for NoneToInfinity<T>
where
    T: for<'a, 'b> DistanceFunction<&'a A, &'b B, Option<f32>>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, x: &Guard<Arc<A>>, y: &Guard<Arc<B>>) -> f32 {
        self.0
            .evaluate_similarity(&**x, &**y)
            .unwrap_or(f32::INFINITY)
    }
}

impl<A, T> PreprocessedDistanceFunction<&Guard<Arc<A>>, f32> for NoneToInfinity<T>
where
    T: for<'a> PreprocessedDistanceFunction<&'a A, Option<f32>>,
{
    #[inline(always)]
    fn evaluate_similarity(&self, x: &Guard<Arc<A>>) -> f32 {
        self.0.evaluate_similarity(&**x).unwrap_or(f32::INFINITY)
    }
}

/// Testing Stratetgy:
///
/// We use the test utils in `pq` to generate pivot tables with known quantities.
///
/// We can then feed in vectors specifically constructed to match known schemas, so we can
/// determine what distances should be.
#[cfg(test)]
mod tests {
    use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

    use super::*;
    use crate::model::pq::distance::test_utils;

    ////////////////
    // One Schema //
    ////////////////

    /// An auxiliary helper function is used to avoid writing
    /// `<Provider as QuantVecProviderAsync<DefaultContext, Data>` for every single method.
    ///
    /// This function is closely coupled with `test_single_schema`.
    fn test_single_schema_as_qvpa(
        provider: &TestMultiPQProviderAsync,
        config: &test_utils::TableConfig,
    ) {
        let table = test_utils::seed_pivot_table(*config);
        let generate_expected_vector = |v: &[u8]| {
            test_utils::generate_expected_vector(v, table.get_chunk_offsets(), config.start_value)
        };

        assert_eq!(provider.num_pq_chunks(), config.pq_chunks);

        // Test vector assignment.
        let v0: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
        let v1: Vec<u8> = vec![5, 4, 3, 2, 1, 0];
        let v2: Vec<u8> = vec![0, 1, 0, 1, 0, 1];
        let v3: Vec<u8> = vec![3, 4, 0, 3, 2, 5];
        let v4: Vec<u8> = vec![4, 4, 4, 4, 4, 4];

        let test_vec: Vec<f32> = vec![1.5; config.dim];
        let distance_computer = provider.get_distance_computer().unwrap();
        let query_computer = provider.get_query_computer(&test_vec).unwrap();

        let vecs = [v0.clone(), v1, v2, v3, v4];
        for (i, v) in vecs.iter().enumerate() {
            let expected = generate_expected_vector(v);
            provider.set_vector(i, &expected).unwrap();

            let output = provider.get_vector(i).unwrap();
            assert_eq!(output.version(), &1);
            assert_eq!(output.data(), v);

            let expected_distance: f32 = SquaredL2::evaluate(&*test_vec, &*expected);
            assert_eq!(
                distance_computer.evaluate_similarity(&*test_vec, &output),
                expected_distance
            );
            assert_eq!(
                query_computer.evaluate_similarity(&output),
                expected_distance
            );
        }

        // Test that providing PQ vectors with an invalid version returns infinity
        let invalid_vector = ArcSwap::new(Arc::new(VersionedPQVector::new(v0.clone(), 100)));
        assert_eq!(
            distance_computer.evaluate_similarity(&*test_vec, &invalid_vector.load()),
            f32::INFINITY
        );
        assert_eq!(
            query_computer.evaluate_similarity(&invalid_vector.load()),
            f32::INFINITY
        );
        assert_eq!(
            distance_computer.evaluate_similarity(&invalid_vector.load(), &invalid_vector.load()),
            f32::INFINITY
        );

        // Errors
        assert!(provider.set_vector(10, &test_vec).is_err());
        assert!(provider.get_vector(10).is_err());

        // Make sure that we can detect mis-sized vectors and if we do, the underlying
        // data is not changed.
        let mut too_long = test_vec.clone();
        too_long.push(1.0);
        assert!(provider.set_vector(0, &too_long).is_err());
        assert_eq!(provider.get_vector(0).unwrap().data(), v0);
    }

    // One Schema.
    #[test]
    fn test_single_schema() {
        let config = test_utils::TableConfig {
            dim: 20,
            pq_chunks: 6,
            num_pivots: 16,
            start_value: 1.0,
            use_opq: false,
        };

        let provider = TestMultiPQProviderAsync::new(
            Metric::L2,
            4,
            1,
            test_utils::seed_pivot_table(config),
            None,
            0.0,
            0,
        );

        test_single_schema_as_qvpa(&provider, &config);
    }

    /////////////////
    // Two Schemas //
    /////////////////

    /// An auxiliary helper function is used to avoid writing
    /// `<Provider as QuantVecProviderAsync<DefaultContext, Data>` for every single method.
    ///
    /// This function is closely coupled with `test_double_schema`.
    fn test_double_schema_as_qvpa(
        provider: &TestMultiPQProviderAsync,
        config_new: &test_utils::TableConfig,
        config_old: &test_utils::TableConfig,
    ) {
        let table_new = test_utils::seed_pivot_table(*config_new);
        let table_old = test_utils::seed_pivot_table(*config_old);
        let generate_expected_vector = |v: &[u8], new: bool| {
            if new {
                test_utils::generate_expected_vector(
                    v,
                    table_new.get_chunk_offsets(),
                    config_new.start_value,
                )
            } else {
                test_utils::generate_expected_vector(
                    v,
                    table_old.get_chunk_offsets(),
                    config_old.start_value,
                )
            }
        };

        assert_eq!(provider.num_pq_chunks(), config_new.pq_chunks);

        // Test vector assignment.
        // NOTE: As long as the encodings are in `[1, num_pq_chunks - 1)` - reconstruction
        // should be exact for both schemas.
        assert_eq!(config_new.pq_chunks, 6);
        assert_eq!(config_old.pq_chunks, 6);
        assert_eq!(config_new.start_value, 2.0);
        assert_eq!(config_old.start_value, 1.0);

        // These vectors here all use encodings for the old PQ schema.
        // To get the encoding for the new PQ schema - subtract 1 from each component.
        let v0: Vec<u8> = vec![1, 1, 2, 3, 4, 4];
        let v1: Vec<u8> = vec![5, 4, 3, 2, 1, 1];
        let v2: Vec<u8> = vec![1, 1, 1, 1, 1, 1];
        let v3: Vec<u8> = vec![3, 4, 1, 3, 2, 4];
        let v4: Vec<u8> = vec![4, 4, 4, 4, 4, 4];

        assert_eq!(config_old.dim, config_new.dim);
        let test_vec: Vec<f32> = vec![1.5; config_new.dim];
        let distance_computer = provider.get_distance_computer().unwrap();
        let query_computer = provider.get_query_computer(&test_vec).unwrap();

        // Track whether or not we've seen the old and new versions.
        let mut old_seen = false;
        let mut new_seen = false;
        let vecs = [v0.clone(), v1, v2, v3, v4];
        for (i, v) in vecs.iter().enumerate() {
            provider
                .set_vector(i, &generate_expected_vector(v, false))
                .unwrap();

            let output = provider.get_vector(i).unwrap();

            let version = *output.version();
            let encoding = if version == 1 {
                old_seen = true;
                v.clone()
            } else if version == 2 {
                new_seen = true;
                v.iter().map(|i| i - 1).collect::<Vec<u8>>()
            } else {
                panic!("Unexpected version: {version}");
            };
            assert_eq!(output.data(), encoding);

            let expected = if version == 1 {
                generate_expected_vector(&encoding, false)
            } else {
                generate_expected_vector(&encoding, true)
            };

            let expected_distance: f32 = SquaredL2::evaluate(&*test_vec, &*expected);
            assert_eq!(
                distance_computer.evaluate_similarity(&*test_vec, &output),
                expected_distance
            );
            assert_eq!(
                query_computer.evaluate_similarity(&output),
                expected_distance
            );
        }

        assert!(old_seen);
        assert!(new_seen);

        // Test that providing PQ vectors with an invalid version returns infinity
        let invalid_vector = ArcSwap::new(Arc::new(VersionedPQVector::new(v0.clone(), 100)));
        assert_eq!(
            distance_computer.evaluate_similarity(&*test_vec, &invalid_vector.load()),
            f32::INFINITY
        );
        assert_eq!(
            query_computer.evaluate_similarity(&invalid_vector.load()),
            f32::INFINITY
        );
        assert_eq!(
            distance_computer.evaluate_similarity(&invalid_vector.load(), &invalid_vector.load()),
            f32::INFINITY
        );
    }

    #[test]
    fn test_double_schema() {
        let config_old = test_utils::TableConfig {
            dim: 20,
            pq_chunks: 6,
            num_pivots: 16,
            start_value: 1.0,
            use_opq: false,
        };

        let config_new = test_utils::TableConfig {
            dim: 20,
            pq_chunks: 6,
            num_pivots: 16,
            start_value: 2.0,
            use_opq: false,
        };

        let provider = TestMultiPQProviderAsync::new(
            Metric::L2,
            4,
            1,
            test_utils::seed_pivot_table(config_new),
            Some(test_utils::seed_pivot_table(config_old)),
            0.5,
            0x4644c5bcfe4f985f,
        );

        test_double_schema_as_qvpa(&provider, &config_new, &config_old);
    }
}
