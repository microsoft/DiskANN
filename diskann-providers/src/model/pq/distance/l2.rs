/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Deref, sync::Arc};

use diskann::{
    ANNResult,
    utils::object_pool::{self, ObjectPool, PoolOption},
};
use diskann_vector::PreprocessedDistanceFunction;

use super::common::get_lookup_table_size;
use crate::model::pq::fixed_chunk_pq_table::{FixedChunkPQTable, pq_dist_lookup_single};

////////
// L2 //
////////

/// A `diskann_vector::PreprocessedDistanceFunction` for table-based lookups
/// using L2.
#[derive(Debug)]
pub struct TableL2<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    /// Pre-computed distances between a query and the pivots.
    /// This table is laid out in row major order like
    /// ```ignore
    /// d00 d01 d02 ... d0N // row0
    /// d10 d11 d12 ... d1N // row1
    /// ...
    /// dK0 dK1 dK2 ... dKN // rowK
    /// ```
    /// where `dAB` is the distance between the `A`th chunk of the query vector and the `B`th
    /// centroid for that chunk.
    ///
    /// The parameter `N` is the number of PQ chunks and is obtained from the parent table.
    lookup_table: PoolOption<Vec<f32>>,

    /// The number of centers per chunk in the lookup table (dimension `N+1` in the table
    /// above).
    num_centers: usize,

    /// The parent table for the pivots and other metadata regarding the PQ Schema.
    parent: T,
}

impl<T> TableL2<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    pub(crate) fn new<U>(
        parent: T,
        query: &[U],
        pool: Option<Arc<ObjectPool<Vec<f32>>>>,
    ) -> ANNResult<Self>
    where
        U: Into<f32> + Copy,
    {
        let mut object = Self::new_unpopulated(parent, pool);
        object.populate(query)?;
        Ok(object)
    }

    fn new_unpopulated(parent: T, pool: Option<Arc<ObjectPool<Vec<f32>>>>) -> Self {
        let vec_size = get_lookup_table_size(&parent);
        Self {
            lookup_table: match pool {
                Some(p) => PoolOption::pooled(&p, object_pool::Undef::new(vec_size)),
                None => PoolOption::non_pooled_create(object_pool::Undef::new(vec_size)),
            },
            num_centers: parent.get_num_centers(),
            parent,
        }
    }

    fn populate<U: Into<f32> + Copy>(&mut self, query: &[U]) -> ANNResult<()> {
        // Ensure that the query has the expected length.
        //
        // Alignment means that the size of `query` gets increased ...
        // This makes is VERY hard to do error checking on dimension propagation.
        assert!(self.parent.get_dim() <= query.len());
        let mut local_query: Vec<f32> = query.iter().map(|x| (*x).into()).collect();

        // This function does the following:
        // 1. Centers the data (if the centorid is non-zero).
        // 2. Applies the OPQ transformation matrix (if it exists).
        self.parent.preprocess_query(&mut local_query);

        // Compute the partial distances into the lookup-table.
        self.parent
            .populate_chunk_distances(&local_query, &mut self.lookup_table)
    }

    /// Compute the distance between a PQ code that the query provided to the most recent
    /// call to `preprocess`.
    ///
    /// The query itself is not needed as everything we need from the query has been encoded
    /// in the lookup table.
    ///
    /// # Panics
    ///
    /// Panics if `code.len()` is not equal to the parent table's number of PQ chunks.
    fn evaluate(&self, code: &[u8]) -> f32 {
        let expected = self.parent.get_num_chunks();
        assert_eq!(
            expected,
            code.len(),
            "PQ code must have {} entries",
            expected
        );
        pq_dist_lookup_single(code, &self.lookup_table, self.num_centers)
    }
}

impl<T> PreprocessedDistanceFunction<&[u8], f32> for TableL2<T>
where
    T: Deref<Target = FixedChunkPQTable>,
{
    fn evaluate_similarity(&self, changing: &[u8]) -> f32 {
        self.evaluate(changing)
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use diskann_vector::Half;
    use rand::SeedableRng;
    use rstest::rstest;

    use super::{
        super::test_utils::{self, TestDistribution},
        *,
    };

    #[rstest]
    fn test_l2<T>(
        #[values(PhantomData::<f32>, PhantomData::<Half>, PhantomData::<i8>, PhantomData::<u8>)]
        _marker: PhantomData<T>,
        #[values(false, true)] use_opq: bool,
    ) where
        T: Into<f32> + TestDistribution,
    {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for dim in [12, 17, 100, 101] {
            for pq_chunks in [1, 17, 19, 20] {
                for num_pivots in [10, 127, 256] {
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
                    let num_trials = 10;

                    // RNG

                    let errors = test_utils::RelativeAndAbsolute {
                        relative: 5e-7,
                        absolute: 0.0,
                    };

                    // Basic `TableL2`
                    test_utils::test_l2_inner(
                        |table: &FixedChunkPQTable, query: &[T]| {
                            TableL2::new(table, query, None).unwrap()
                        },
                        &table,
                        num_trials,
                        config,
                        &mut rng,
                        errors,
                    )
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "PQ code must have 3 entries")]
    fn panic_on_too_long_vector() {
        let config = test_utils::TableConfig {
            dim: 10,
            pq_chunks: 3,
            num_pivots: 4,
            start_value: 0.0,
            use_opq: false,
        };

        let table = test_utils::seed_pivot_table(config);
        let query = vec![0.0; config.dim];
        let computer = TableL2::new(&table, &query, None).unwrap();

        let code = vec![0, 0, 0, 0];
        computer.evaluate_similarity(&code);
    }

    #[test]
    #[should_panic(expected = "the len is 4 but the index is 4")]
    fn panic_on_out_of_bounds_entry() {
        let config = test_utils::TableConfig {
            dim: 10,
            pq_chunks: 3,
            num_pivots: 4,
            start_value: 0.0,
            use_opq: false,
        };

        let table = test_utils::seed_pivot_table(config);
        let query = vec![0.0; config.dim];
        let computer = TableL2::new(&table, &query, None).unwrap();

        // Entry `4` is out-of-bounds.
        let code = vec![0, 4, 0];
        computer.evaluate_similarity(&code);
    }
}
