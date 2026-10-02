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
            lookup,
            DotAndNorm,
            test::{DistanceTestTable, Preprocessed},
        },
        test_util::Check,
    };

    // L2
    #[test]
    fn test_l2() {
        let cases = [
            // dim, chunks, pivots, start, check
            (1, 1, 1, 0.0, Check::exact()),
            (3, 3, 2, -1.0, Check::exact()),
            (7, 3, 15, -7.0, Check::exact()),
            (8, 4, 16, -8.0, Check::exact()),
            (13, 5, 17, -11.0, Check::exact()),
            (32, 4, 32, -16.0, Check::exact()),
            (37, 7, 33, -20.0, Check::exact()),
            (17, 5, 256, -128.0, Check::exact()),
        ];

        let (num_queries, num_trials) = if cfg!(miri) {
            // The driver will run some directed tests even if there are no regular random
            // trials.
            (1, 0)
        } else {
            (10, 10)
        };

        #[derive(Debug)]
        struct Dut<'a> {
            table: &'a TransposedTable,
            lut: rowmajor::Owned<f32>,
        }

        impl Preprocessed for Dut<'_> {
            fn preprocess(&mut self, query: &[f32]) {
                self.table
                    .process_into::<distances::SquaredL2, f32>(query, self.lut.as_view_mut())
            }

            fn evaluate(&mut self, code: &[u8]) -> f32 {
                lookup::lookup_single(lookup::Sum, self.lut.as_view(), code).unwrap()
            }
        }

        for (dim, chunks, pivots, start, check) in cases {
            let driver = DistanceTestTable::new(dim, chunks, pivots, start);
            let basic = driver.basic_table();
            let transposed =
                TransposedTable::from_parts(basic.view_pivots(), basic.view_offsets().to_owned())
                    .unwrap();

            let lut =
                rowmajor::Owned::from_element(transposed.nchunks(), transposed.ncenters(), 0.0f32);

            let mut dut = Dut {
                table: &transposed,
                lut,
            };

            driver.check_squared_l2(
                num_queries,
                num_trials,
                &mut driver.rng(0xc0ffee),
                check,
                &mut dut,
                format_args!(
                    "transposed table - dim = {}, chunks = {}, pivots = {}",
                    dim, chunks, pivots
                ),
            );
        }
    }

    // IP
    #[test]
    fn test_ip() {
        let cases = [
            // dim, chunks, pivots, start, check
            (1, 1, 1, 0.0, Check::exact()),
            (3, 3, 2, -1.0, Check::exact()),
            (7, 3, 15, -7.0, Check::exact()),
            (8, 4, 16, -8.0, Check::exact()),
            (13, 5, 17, -11.0, Check::exact()),
            (32, 4, 32, -16.0, Check::exact()),
            (37, 7, 33, -20.0, Check::exact()),
            (17, 5, 256, -128.0, Check::exact()),
        ];

        let (num_queries, num_trials) = if cfg!(miri) {
            // The driver will run some directed tests even if there are no regular random
            // trials.
            (1, 0)
        } else {
            (10, 10)
        };

        #[derive(Debug)]
        struct Dut<'a> {
            table: &'a TransposedTable,
            lut: rowmajor::Owned<f32>,
        }

        impl Preprocessed for Dut<'_> {
            fn preprocess(&mut self, query: &[f32]) {
                self.table
                    .process_into::<distances::InnerProduct, f32>(query, self.lut.as_view_mut())
            }

            fn evaluate(&mut self, code: &[u8]) -> f32 {
                lookup::lookup_single(lookup::Sum, self.lut.as_view(), code).unwrap()
            }
        }

        for (dim, chunks, pivots, start, check) in cases {
            let driver = DistanceTestTable::new(dim, chunks, pivots, start);
            let basic = driver.basic_table();
            let transposed =
                TransposedTable::from_parts(basic.view_pivots(), basic.view_offsets().to_owned())
                    .unwrap();

            let lut =
                rowmajor::Owned::from_element(transposed.nchunks(), transposed.ncenters(), 0.0f32);

            let mut dut = Dut {
                table: &transposed,
                lut,
            };

            driver.check_inner_product(
                num_queries,
                num_trials,
                &mut driver.rng(0xc0ffee),
                check,
                &mut dut,
                format_args!(
                    "transposed table - dim = {}, chunks = {}, pivots = {}",
                    dim, chunks, pivots
                ),
            );
        }
    }

    // Cosine
    #[test]
    fn test_cosine() {
        let cases = [
            // dim, chunks, pivots, start, check
            (1, 1, 1, 0.0, Check::exact()),
            (3, 3, 2, -1.0, Check::exact()),
            (7, 3, 15, -7.0, Check::exact()),
            (8, 4, 16, -8.0, Check::exact()),
            (13, 5, 17, -11.0, Check::exact()),
            (32, 4, 32, -16.0, Check::exact()),
            (37, 7, 33, -20.0, Check::exact()),
            (17, 5, 256, -128.0, Check::exact()),
        ];

        let (num_queries, num_trials) = if cfg!(miri) {
            // The driver will run some directed tests even if there are no regular random
            // trials.
            (1, 0)
        } else {
            (10, 10)
        };

        #[derive(Debug)]
        struct Dut<'a> {
            table: &'a TransposedTable,
            lut: rowmajor::Owned<DotAndNorm>,
            query_norm: f32,
        }

        impl Preprocessed for Dut<'_> {
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

        for (dim, chunks, pivots, start, check) in cases {
            let driver = DistanceTestTable::new(dim, chunks, pivots, start);
            let basic = driver.basic_table();
            let transposed =
                TransposedTable::from_parts(basic.view_pivots(), basic.view_offsets().to_owned())
                    .unwrap();

            let lut =
                rowmajor::Owned::from_element(transposed.nchunks(), transposed.ncenters(), DotAndNorm::default());

            let mut dut = Dut {
                table: &transposed,
                lut,
                query_norm: 0.0,
            };

            driver.check_cosine(
                num_queries,
                num_trials,
                &mut driver.rng(0xc0ffee),
                check,
                &mut dut,
                format_args!(
                    "transposed table - dim = {}, chunks = {}, pivots = {}",
                    dim, chunks, pivots
                ),
            );
        }
    }
}
