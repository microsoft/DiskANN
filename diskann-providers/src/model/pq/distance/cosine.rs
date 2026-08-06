/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_vector::PreprocessedDistanceFunction;

use crate::model::pq::fixed_chunk_pq_table::FixedChunkPQTable;

////////////
// Cosine //
////////////

/// A `PreprocessedDistanceFunction` for Cosine Similarity
#[derive(Debug)]
pub struct DirectCosine<'a> {
    /// Pre-converted query
    query: Vec<f32>,

    /// The parent table for the pivots and other meta-data regarding the PQ Schema.
    parent: &'a FixedChunkPQTable,
}

impl<'a> DirectCosine<'a> {
    /// Caller must ensure `query.len() == parent.get_dim()` (validated by `QueryComputer::new`).
    pub(crate) fn new(parent: &'a FixedChunkPQTable, query: &[f32]) -> ANNResult<Self> {
        let mut object = Self::new_unpopulated(parent);
        object.populate(query)?;
        Ok(object)
    }

    fn new_unpopulated(parent: &'a FixedChunkPQTable) -> Self {
        Self {
            query: vec![0.0f32; parent.get_dim()],
            parent,
        }
    }

    fn populate(&mut self, query: &[f32]) -> ANNResult<()> {
        // Stash a copy of the query so subsequent `evaluate` calls can reuse it
        // without converting on every call.
        self.query.copy_from_slice(query);
        Ok(())
    }

    /// Compute the distance between a PQ code that the query.
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

        // Just call the associated method on the parent.
        self.parent.cosine_distance(&(self.query), code)
    }
}

impl PreprocessedDistanceFunction<&[u8], f32> for DirectCosine<'_> {
    fn evaluate_similarity(&self, changing: &[u8]) -> f32 {
        self.evaluate(changing)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::{super::test_utils, *};

    #[test]
    fn test_cosine() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc33529acbe474958);
        let num_trials = 20;

        for dim in [64, 117, 128] {
            for pq_chunks in [2, 5, 15] {
                for num_pivots in [10, 127, 256] {
                    if pq_chunks > dim {
                        continue;
                    }
                    // Table and computer creation.
                    let config = test_utils::TableConfig {
                        dim,
                        pq_chunks,
                        num_pivots,
                        start_value: 0.0,
                    };
                    let table = test_utils::seed_pivot_table(config);
                    let errors = test_utils::RelativeAndAbsolute {
                        relative: 2.0e-7,
                        absolute: 0.0,
                    };

                    // DirectCosine
                    test_utils::test_cosine_inner(
                        |table: &FixedChunkPQTable, query: &[f32]| {
                            DirectCosine::new(table, query).unwrap()
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

    #[test]
    #[should_panic(expected = "PQ code must have 3 entries")]
    fn panic_on_too_long_vector() {
        let config = test_utils::TableConfig {
            dim: 10,
            pq_chunks: 3,
            num_pivots: 4,
            start_value: 0.0,
        };

        let table = test_utils::seed_pivot_table(config);
        let query = vec![0.0; config.dim];
        let computer = DirectCosine::new(&table, &query).unwrap();

        let code = vec![0, 0, 0, 0];
        computer.evaluate_similarity(&code);
    }

    #[test]
    #[should_panic]
    fn panic_on_out_of_bounds_entry() {
        let config = test_utils::TableConfig {
            dim: 10,
            pq_chunks: 3,
            num_pivots: 4,
            start_value: 0.0,
        };

        let table = test_utils::seed_pivot_table(config);
        let query = vec![0.0; config.dim];
        let computer = DirectCosine::new(&table, &query).unwrap();

        // Entry `4` is out-of-bounds.
        let code = vec![0, 4, 0];
        computer.evaluate_similarity(&code);
    }
}
