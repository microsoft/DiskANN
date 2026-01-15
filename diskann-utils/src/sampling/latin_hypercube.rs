/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::views::{Matrix, MatrixView};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Return multiple rows sampled using Latin Hypercube Sampling in `data` that aproximetely uniformly distributed.
/// This makes the assumtion that the data is uniformly distributed.
pub trait SampleLatinHyperCube: Sized + Copy + Default {
    fn sample_latin_hypercube(
        data: MatrixView<Self>,
        num_samples: usize,
        seed: Option<u64>,
    ) -> Matrix<Self>;
}

impl<T: Sized + Copy + Default> SampleLatinHyperCube for T {
    fn sample_latin_hypercube(
        data: MatrixView<Self>,
        num_samples: usize,
        seed: Option<u64>,
    ) -> Matrix<Self> {
        let nrows = data.nrows();
        let ncols = data.ncols();
        if ncols == 0 || nrows == 0 {
            return Matrix::new(T::default(), num_samples, ncols);
        }

        let seed = seed.unwrap_or(0xaf2f5fa0b5161acf);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut result: Matrix<Self> = Matrix::new(T::default(), num_samples, ncols);

        // sample a random partitions down the diagonal
        for (s, res) in result.row_iter_mut().enumerate() {
            for (idx, val) in res.iter_mut().enumerate() {
                let step = nrows / num_samples;
                let value = data
                    .get_row(rng.random_range(s * step..(s + 1) * step))
                    .unwrap()
                    .get(idx)
                    .unwrap();
                *val = *value;
            }
        }

        // shuffle the dimensions between the vectors for random sampling
        for start_idx in 0..num_samples {
            for dim_idx in 0..ncols {
                let swap_idx = rng.random_range(0..num_samples);
                let swap = result[(start_idx, dim_idx)];
                result[(start_idx, dim_idx)] = result[(swap_idx, dim_idx)];
                result[(swap_idx, dim_idx)] = swap;
            }
        }

        result
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use crate::views::{Init, Matrix};
    use diskann_vector::conversion::CastFromSlice;
    use half::f16;
    use rand::{
        distr::{Distribution, StandardUniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;

    fn example_dataset() -> Matrix<f32> {
        let data: Vec<f32> = vec![
            // row 0
            0.203688,
            0.841956,
            0.855665,
            0.801917,
            0.754536,
            // row 1
            0.312881,
            0.217382,
            0.0644115,
            0.348708,
            0.999495,
            // row 2
            0.657741,
            0.914681,
            0.555228,
            0.13253,
            0.118615,
            // row 3
            0.356464,
            0.207449,
            0.452471,
            0.925219,
            0.508498,
            // row 4
            0.749786,
            0.90786,
            0.129618,
            0.597719,
            0.000622153,
            // row 5 -- this is the medoid
            0.569517,
            0.435447,
            0.558136,
            0.480974,
            0.711425,
            // row 6
            0.896353,
            0.275053,
            0.0427179,
            0.660916,
            0.464851,
            // row 7
            0.558689,
            0.596543,
            0.740983,
            0.122136,
            0.453822,
            // row 8
            0.526895,
            0.492643,
            0.0951115,
            0.495487,
            0.446127,
            // row 9
            0.454093,
            0.160239,
            0.924585,
            0.901708,
            0.329328,
        ];

        Matrix::<f32>::try_from(data.into(), 10, 5).unwrap()
    }

    fn example_dataset_u8() -> Matrix<u8> {
        let data: Vec<u8> = vec![
            52, 215, 218, 204, 192, // row 0
            79, 55, 16, 89, 255, // row 1
            167, 233, 141, 33, 30, // row 2
            91, 53, 115, 236, 130, // row 3
            191, 232, 33, 152, 1, // row 4
            145, 111, 142, 122, 181, // row 5 -- this is the medoid
        ];

        Matrix::<u8>::try_from(data.into(), 6, 5).unwrap()
    }

    // This is a test for the i8 function. Each entry is between -128 and 127.
    fn example_dataset_i8() -> Matrix<i8> {
        let data: Vec<i8> = vec![
            -76, 87, 90, 76, 64, // row 0
            -49, -73, -112, -39, 127, // row 1
            39, 105, 13, -95, -98, // row 2
            -37, -75, -13, 108, 2, // row 3
            -37, -75, -13, 108, 2, // row 4
            17, -17, 14, -6, 53, // row 5 -- this is the medoid
        ];

        Matrix::<i8>::try_from(data.into(), 6, 5).unwrap()
    }

    fn test_for_type<T>(data: Matrix<T>)
    where
        T: SampleLatinHyperCube + PartialEq + std::fmt::Debug + Display,
        StandardUniform: Distribution<T>,
    {
        // No Rows
        let x = Matrix::<T>::new(T::default(), 0, 10);
        assert_eq!(
            T::sample_latin_hypercube(x.as_view(), 1, None),
            Matrix::<T>::new(T::default(), 1, x.ncols())
        );

        // No Cols0
        let x = Matrix::<T>::new(T::default(), 1, 0);
        assert_eq!(
            T::sample_latin_hypercube(x.as_view(), 1, None),
            Matrix::<T>::new(T::default(), 1, x.ncols())
        );

        let mut rng: StdRng = StdRng::seed_from_u64(0xaf2f5fa0b5161acf);

        // One row
        let dist = StandardUniform;
        for dim in 1..20 {
            let x = Matrix::<T>::new(Init(|| dist.sample(&mut rng)), 1, dim);
            assert_eq!(
                T::sample_latin_hypercube(x.as_view(), 1, None),
                Matrix::<T>::try_from(x.row(0).to_vec().into_boxed_slice(), 1, dim).unwrap()
            );
        }

        // Example dataset
        let starts = T::sample_latin_hypercube(data.as_view(), 2, None);
        for s in starts.row_iter() {
            for (col, &val) in s.iter().enumerate() {
                let col_vals: Vec<T> = (0..data.nrows())
                    .map(|row| {
                        *data
                            .get_row(row)
                            .expect("Row must exist")
                            .get(col)
                            .expect("Column must exist")
                    })
                    .collect();
                assert!(
                    col_vals.contains(&val),
                    "Value {} in column {} not found in data",
                    val,
                    col
                );
            }
        }
    }

    #[test]
    fn test_f32() {
        test_for_type(example_dataset())
    }

    #[test]
    fn test_f16() {
        let data = example_dataset();
        let mut data_f16 = Matrix::<f16>::new(f16::default(), data.nrows(), data.ncols());
        data_f16.as_mut_slice().cast_from_slice(data.as_slice());
        test_for_type(data_f16);
    }

    #[test]
    fn test_u8() {
        test_for_type(example_dataset_u8());
    }

    #[test]
    fn test_i8() {
        test_for_type(example_dataset_i8());
    }
}
