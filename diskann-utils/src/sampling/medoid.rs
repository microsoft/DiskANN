/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::views::MatrixView;
use diskann_vector::{conversion::CastFromSlice, distance::SquaredL2, PureDistanceFunction};
use half::f16;

/// Return the row in `data` that is closest to the medoid of all rows.
pub trait ComputeMedoid: Sized {
    fn compute_medoid(data: MatrixView<Self>) -> Vec<Self>;
}

impl ComputeMedoid for f32 {
    fn compute_medoid(data: MatrixView<Self>) -> Vec<Self> {
        if data.ncols() == 0 {
            return vec![];
        }

        let mut sum = vec![0.0f64; data.ncols()];
        data.row_iter().for_each(|r| {
            std::iter::zip(sum.iter_mut(), r.iter()).for_each(|(o, i)| {
                let i: f64 = (*i).into();
                *o += i;
            });
        });

        let m: Vec<f32> = sum
            .iter()
            .map(|s| (s / data.nrows() as f64) as f32)
            .collect();

        let mut min_dist: f32 = f32::MAX;
        let mut medoid = None;
        data.row_iter().for_each(|r| {
            let d = SquaredL2::evaluate(m.as_slice(), r);
            if d < min_dist {
                min_dist = d;
                medoid = Some(r);
            }
        });

        medoid
            .map(|x| x.into())
            .unwrap_or(vec![0.0f32; data.ncols()])
    }
}

impl ComputeMedoid for f16 {
    fn compute_medoid(data: MatrixView<Self>) -> Vec<Self> {
        if data.ncols() == 0 {
            return vec![];
        }

        let mut sum = vec![0.0f64; data.ncols()];
        let mut buffer = vec![0.0f32; data.ncols()];
        data.row_iter().for_each(|r| {
            buffer.cast_from_slice(r);
            std::iter::zip(sum.iter_mut(), buffer.iter()).for_each(|(o, i)| {
                let i: f64 = (*i).into();
                *o += i;
            });
        });

        std::iter::zip(buffer.iter_mut(), sum.iter()).for_each(|(o, i)| {
            *o = (*i / data.nrows() as f64) as f32;
        });

        let mut min_dist: f32 = f32::MAX;
        let mut medoid = None;
        data.row_iter().for_each(|r| {
            let d = SquaredL2::evaluate(buffer.as_slice(), r);
            if d < min_dist {
                min_dist = d;
                medoid = Some(r);
            }
        });

        medoid
            .map(|x| x.into())
            .unwrap_or(vec![f16::default(); data.ncols()])
    }
}

impl ComputeMedoid for u8 {
    fn compute_medoid(data: MatrixView<Self>) -> Vec<Self> {
        if data.ncols() == 0 {
            return vec![];
        }

        let mut sum = vec![0.0f64; data.ncols()];
        data.row_iter().for_each(|r| {
            std::iter::zip(sum.iter_mut(), r.iter()).for_each(|(o, i)| {
                let i: f64 = (*i).into();
                *o += i;
            });
        });

        let m: Vec<f32> = sum
            .iter()
            .map(|s| (s / data.nrows() as f64) as f32)
            .collect();

        let mut min_dist: f32 = f32::MAX;
        let mut medoid = None;
        let mut as_float = vec![0.0f32; data.ncols()];
        data.row_iter().for_each(|r| {
            std::iter::zip(as_float.iter_mut(), r.iter())
                .for_each(|(dst, src)| *dst = (*src).into());
            let d = SquaredL2::evaluate(m.as_slice(), &*as_float);
            if d < min_dist {
                min_dist = d;
                medoid = Some(r);
            }
        });

        medoid.map(|x| x.into()).unwrap_or(vec![0u8; data.ncols()])
    }
}

impl ComputeMedoid for i8 {
    fn compute_medoid(data: MatrixView<Self>) -> Vec<Self> {
        if data.ncols() == 0 {
            return vec![];
        }

        let mut sum = vec![0.0f64; data.ncols()];
        data.row_iter().for_each(|r| {
            std::iter::zip(sum.iter_mut(), r.iter()).for_each(|(o, i)| {
                let i: f64 = (*i).into();
                *o += i;
            });
        });

        let m: Vec<f32> = sum
            .iter()
            .map(|s| (s / data.nrows() as f64) as f32)
            .collect();

        let mut min_dist: f32 = f32::MAX;
        let mut medoid = None;
        let mut as_float = vec![0.0f32; data.ncols()];
        data.row_iter().for_each(|r| {
            std::iter::zip(as_float.iter_mut(), r.iter())
                .for_each(|(dst, src)| *dst = (*src).into());
            let d = SquaredL2::evaluate(m.as_slice(), &*as_float);
            if d < min_dist {
                min_dist = d;
                medoid = Some(r);
            }
        });

        medoid.map(|x| x.into()).unwrap_or(vec![0i8; data.ncols()])
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use crate::views::{Init, Matrix};
    use rand::{
        distr::{Distribution, StandardUniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;

    fn example_dataset() -> (Matrix<f32>, Vec<f32>) {
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

        let data = Matrix::<f32>::try_from(data.into(), 10, 5).unwrap();
        let expected: Vec<f32> = data.row(5).into();
        (data, expected)
    }

    #[test]
    fn test_f32() {
        // No Rows
        let x = Matrix::<f32>::new(0.0f32, 0, 10);
        assert_eq!(f32::compute_medoid(x.as_view()), vec![0.0; x.ncols()]);

        // No Cols
        let x = Matrix::<f32>::new(0.0f32, 10, 0);
        assert_eq!(f32::compute_medoid(x.as_view()), Vec::<f32>::new());

        let mut rng = StdRng::seed_from_u64(0xaf2f5fa0b5161acf);

        // One row
        let dist = StandardUniform;
        for dim in 1..20 {
            let x = Matrix::<f32>::new(Init(|| dist.sample(&mut rng)), 1, dim);
            assert_eq!(&*f32::compute_medoid(x.as_view()), x.row(0));
        }

        // Example dataset
        let (data, expected) = example_dataset();
        let m = f32::compute_medoid(data.as_view());
        assert_eq!(m, expected);
    }

    #[test]
    fn test_f16() {
        // No Rows
        let x = Matrix::<f16>::new(f16::default(), 0, 10);
        assert_eq!(
            f16::compute_medoid(x.as_view()),
            vec![f16::default(); x.ncols()]
        );

        // No Cols
        let x = Matrix::<f16>::new(f16::default(), 10, 0);
        assert_eq!(f16::compute_medoid(x.as_view()), Vec::<f16>::new());

        let mut rng = StdRng::seed_from_u64(0x88e2f7096fc9b90e);

        // One row
        let dist = StandardUniform;
        for dim in 1..20 {
            let x = Matrix::<f16>::new(Init(|| f16::from_f32(dist.sample(&mut rng))), 1, dim);
            assert_eq!(&*f16::compute_medoid(x.as_view()), x.row(0));
        }

        // Example dataset
        let (data, expected) = example_dataset();
        let mut data_f16 = Matrix::<f16>::new(f16::default(), data.nrows(), data.ncols());
        data_f16.as_mut_slice().cast_from_slice(data.as_slice());

        let mut expected_f16 = vec![f16::default(); expected.len()];
        expected_f16.cast_from_slice(expected.as_slice());

        let m = f16::compute_medoid(data_f16.as_view());
        assert_eq!(m, expected_f16);
    }

    fn example_dataset_u8() -> (Matrix<u8>, Vec<u8>) {
        let data: Vec<u8> = vec![
            52, 215, 218, 204, 192, // row 0
            79, 55, 16, 89, 255, // row 1
            167, 233, 141, 33, 30, // row 2
            91, 53, 115, 236, 130, // row 3
            191, 232, 33, 152, 1, // row 4
            145, 111, 142, 122, 181, // row 5 -- this is the medoid
        ];

        let data = Matrix::<u8>::try_from(data.into(), 6, 5).unwrap();
        let expected: Vec<u8> = data.row(5).into();
        (data, expected)
    }

    #[test]
    fn test_u8() {
        // No Rows
        let x = Matrix::<u8>::new(0u8, 0, 10);
        assert_eq!(u8::compute_medoid(x.as_view()), vec![0u8; x.ncols()]);

        // No Cols
        let x = Matrix::<u8>::new(0u8, 10, 0);
        assert_eq!(u8::compute_medoid(x.as_view()), Vec::<u8>::new());
        let mut rng = StdRng::seed_from_u64(0x8f2f5fa0b5161acf);

        // One row
        let dist = StandardUniform;
        for dim in 1..20 {
            let x = Matrix::<u8>::new(Init(|| dist.sample(&mut rng)), 1, dim);
            assert_eq!(&*u8::compute_medoid(x.as_view()), x.row(0));
        }

        // Example dataset
        let (data, expected) = example_dataset_u8();
        let m = u8::compute_medoid(data.as_view());
        assert_eq!(m, expected);
    }

    // This is a test for the i8 medoid function. Each entry is between -128 and 127.
    fn example_dataset_i8() -> (Matrix<i8>, Vec<i8>) {
        let data: Vec<i8> = vec![
            -76, 87, 90, 76, 64, // row 0
            -49, -73, -112, -39, 127, // row 1
            39, 105, 13, -95, -98, // row 2
            -37, -75, -13, 108, 2, // row 3
            -37, -75, -13, 108, 2, // row 4
            17, -17, 14, -6, 53, // row 5 -- this is the medoid
        ];

        let data = Matrix::<i8>::try_from(data.into(), 6, 5).unwrap();
        let expected: Vec<i8> = data.row(5).into();
        (data, expected)
    }

    #[test]
    fn test_i8() {
        // No Rows
        let x = Matrix::<i8>::new(0i8, 0, 10);
        assert_eq!(i8::compute_medoid(x.as_view()), vec![0i8; x.ncols()]);

        // No Cols
        let x = Matrix::<i8>::new(0i8, 10, 0);
        assert_eq!(i8::compute_medoid(x.as_view()), Vec::<i8>::new());

        let mut rng = StdRng::seed_from_u64(0x8f2f5fa0b5161acf);

        // One row
        let dist = StandardUniform;
        for dim in 1..20 {
            let x = Matrix::<i8>::new(Init(|| dist.sample(&mut rng)), 1, dim);
            assert_eq!(&*i8::compute_medoid(x.as_view()), x.row(0));
        }

        // Example dataset
        let (data, expected) = example_dataset_i8();
        let m = i8::compute_medoid(data.as_view());
        assert_eq!(m, expected);
    }
}
