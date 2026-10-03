/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod pivots;
mod table;

pub use table::TransposedTable;

///////////
// Tests //
///////////

/// These tests check the distance formulation as a result of pre-processing in the transposed
/// table. They ensure we have an end-to-end working example of full distance calculations.
///
/// The tests are broken into metric specific tests, mainly so they can run more efficiently
/// in parallel as there are a decent number of cases that must be covered.
#[cfg(test)]
mod tests {
    use super::*;

    use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
    use diskann_vector::{Norm, norm::FastL2Norm};

    use crate::{
        distances,
        product::tables::{
            lookup::{self, DotAndNorm},
            test::{self as table_test, DistanceTestTable, QueryLike},
        },
        test_util::Check,
    };

    /// Common cases between all distances.
    fn cases() -> &'static [Case] {
        const CASES: &[Case] = &[
            // dim, chunks, pivots, start
            Case::new(1, 1, 1, 0.0, Check::exact()),
            Case::new(3, 3, 2, -1.0, Check::exact()),
            Case::new(7, 3, 15, -7.0, Check::exact()),
            Case::new(8, 4, 16, -8.0, Check::exact()),
            Case::new(13, 5, 17, -11.0, Check::exact()),
            Case::new(32, 4, 32, -16.0, Check::exact()),
            Case::new(37, 7, 33, -20.0, Check::exact()),
            Case::new(17, 5, 256, -128.0, Check::exact()),
        ];
        CASES
    }

    #[derive(Debug, Clone, Copy)]
    struct Case {
        dim: usize,
        chunks: usize,
        pivots: usize,
        start: f32,
        check: Check,
    }

    impl Case {
        const fn new(dim: usize, chunks: usize, pivots: usize, start: f32, check: Check) -> Self {
            Self {
                dim,
                chunks,
                pivots,
                start,
                check,
            }
        }
    }

    fn run_test(
        cases: &[Case],
        create: &dyn Fn(&TransposedTable) -> Box<dyn QueryLike + '_>,
        reference: &dyn Fn(&[f32], &[f32]) -> f32,
        ctx: &dyn std::fmt::Display,
    ) {
        let (num_queries, num_trials) = if cfg!(miri) {
            // The driver will run some directed tests even if there are no regular random
            // trials.
            (1, 0)
        } else {
            (10, 10)
        };

        for Case {
            dim,
            chunks,
            pivots,
            start,
            check,
        } in cases.iter().copied()
        {
            let driver = DistanceTestTable::new(dim, chunks, pivots, start);
            let basic = driver.basic_table();
            let transposed =
                TransposedTable::from_parts(basic.view_pivots(), basic.view_offsets().to_owned())
                    .unwrap();

            let mut dut = create(&transposed);

            driver.drive_query_like(
                num_queries,
                num_trials,
                &mut driver.rng(0xc0ffee),
                check,
                reference,
                &mut *dut,
                format_args!(
                    "[{}] transposed table - dim = {}, chunks = {}, pivots = {}",
                    ctx, dim, chunks, pivots
                ),
            );
        }
    }

    // L2
    #[test]
    fn test_l2() {
        #[derive(Debug)]
        struct Dut<'a> {
            table: &'a TransposedTable,
            lut: rowmajor::Owned<f32>,
        }

        impl QueryLike for Dut<'_> {
            fn preprocess(&mut self, query: &[f32]) {
                self.table
                    .process_into::<distances::SquaredL2, f32>(query, self.lut.as_view_mut())
            }

            fn evaluate(&mut self, code: &[u8]) -> f32 {
                lookup::lookup_single(lookup::Sum, self.lut.as_view(), code).unwrap()
            }
        }

        run_test(
            cases(),
            &|table: &TransposedTable| {
                let lut = rowmajor::Owned::from_element(table.nchunks(), table.ncenters(), 0.0f32);
                Box::new(Dut { table, lut })
            },
            &table_test::squared_l2,
            &"squared l2",
        );
    }

    // IP
    #[test]
    fn test_ip() {
        #[derive(Debug)]
        struct Dut<'a> {
            table: &'a TransposedTable,
            lut: rowmajor::Owned<f32>,
        }

        impl QueryLike for Dut<'_> {
            fn preprocess(&mut self, query: &[f32]) {
                self.table
                    .process_into::<distances::InnerProduct, f32>(query, self.lut.as_view_mut())
            }

            fn evaluate(&mut self, code: &[u8]) -> f32 {
                lookup::lookup_single(lookup::Sum, self.lut.as_view(), code).unwrap()
            }
        }

        run_test(
            cases(),
            &|table: &TransposedTable| {
                let lut = rowmajor::Owned::from_element(table.nchunks(), table.ncenters(), 0.0f32);
                Box::new(Dut { table, lut })
            },
            &table_test::inner_product,
            &"inner product",
        );
    }

    // Cosine
    #[test]
    fn test_cosine() {
        #[derive(Debug)]
        struct Dut<'a> {
            table: &'a TransposedTable,
            lut: rowmajor::Owned<DotAndNorm>,
            query_norm: f32,
        }

        impl QueryLike for Dut<'_> {
            fn preprocess(&mut self, query: &[f32]) {
                self.query_norm = (FastL2Norm).evaluate(query);
                self.table
                    .process_into::<distances::Cosine, DotAndNorm>(query, self.lut.as_view_mut())
            }

            fn evaluate(&mut self, code: &[u8]) -> f32 {
                let partial = lookup::lookup_single(lookup::Sum, self.lut.as_view(), code).unwrap();
                partial.finish_cosine(self.query_norm).into_inner()
            }
        }

        run_test(
            cases(),
            &|table: &TransposedTable| {
                let lut = rowmajor::Owned::from_element(
                    table.nchunks(),
                    table.ncenters(),
                    DotAndNorm::default(),
                );
                Box::new(Dut {
                    table,
                    lut,
                    query_norm: 0.0f32,
                })
            },
            &table_test::cosine,
            &"cosine",
        );
    }
}
