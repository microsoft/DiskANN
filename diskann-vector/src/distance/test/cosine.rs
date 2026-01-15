/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(test)]
mod distance_test {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{
        distance::{test::FullPrecisionDistance, Cosine, CosineNormalized, Metric},
        test_util::no_vector_compare_f16_as_f64,
        Half, PureDistanceFunction,
    };

    #[repr(C, align(32))]
    struct F32Slice112([f32; 112]);
    #[repr(C, align(32))]
    struct F32Slice256([f32; 256]);
    #[repr(C, align(32))]
    struct F16Slice112([Half; 112]);

    fn get_turing_test_data() -> (F32Slice112, F32Slice112) {
        let mut a_slice = F32Slice112([0.0; 112]);
        let mut b_slice = F32Slice112([0.0; 112]);
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..112 {
            a_slice.0[i] = rng.random_range(-1.0..1.0);
            b_slice.0[i] = rng.random_range(-1.0..1.0);
        }
        (a_slice, b_slice)
    }

    fn get_turing_test_data_f16() -> (F16Slice112, F16Slice112) {
        let (a_slice, b_slice) = get_turing_test_data();
        let a_data = a_slice.0.iter().map(|x| Half::from_f32(*x));
        let b_data = b_slice.0.iter().map(|x| Half::from_f32(*x));

        (
            F16Slice112(a_data.collect::<Vec<Half>>().try_into().unwrap()),
            F16Slice112(b_data.collect::<Vec<Half>>().try_into().unwrap()),
        )
    }
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dist_cosine_float_turing() {
        // two vectors are allocated in the contiguous heap memory
        let (a_slice, b_slice) = get_turing_test_data();
        let distance = <[f32; 112] as FullPrecisionDistance<f32, 112>>::distance_compare(
            &a_slice.0,
            &b_slice.0,
            Metric::Cosine,
        );

        let got: f32 = Cosine::evaluate(&a_slice.0, &b_slice.0);
        assert_abs_diff_eq!(distance, got, epsilon = 1e-5);
    }

    #[test]
    fn test_dist_l2_f16_turing() {
        // two vectors are allocated in the contiguous heap memory
        let (a_slice, b_slice) = get_turing_test_data_f16();
        let distance = <[Half; 112] as FullPrecisionDistance<Half, 112>>::distance_compare(
            &a_slice.0,
            &b_slice.0,
            Metric::L2,
        );

        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f16_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-4f64
        );
    }

    #[test]
    fn test_dist_cosine_f16_turing() {
        // two vectors are allocated in the contiguous heap memory
        let (a_slice, b_slice) = get_turing_test_data_f16();
        let distance = <[Half; 112] as FullPrecisionDistance<Half, 112>>::distance_compare(
            &a_slice.0,
            &b_slice.0,
            Metric::Cosine,
        );

        // Note the variance between the full 32 bit precision and the 16 bit precision
        let got: f32 = Cosine::evaluate(&a_slice.0, &b_slice.0);
        assert_abs_diff_eq!(distance, got, epsilon = 1e-5);
    }

    #[test]
    fn cosine_distance_test() {
        #[repr(C, align(32))]
        struct Vector32ByteAligned {
            v: [f32; 512],
        }

        for seed in 0..100 {
            let mut vec1 = Box::new(Vector32ByteAligned { v: [0.0; 512] });

            let mut rng = StdRng::seed_from_u64(seed);
            for i in 0..512 {
                vec1.v[i] = rng.random_range(-1.0..1.0);
            }

            let mut vec2 = F32Slice256([0.0; 256]);
            let mut vec3 = F32Slice256([0.0; 256]);

            for j in 0..256 {
                vec2.0[j] = vec1.v[j];
            }
            for z in 256..512 {
                vec3.0[z - 256] = vec1.v[z];
            }

            let distance = compare::<f32, 256>(256, Metric::Cosine, &vec1.v);
            let expected: f32 = Cosine::evaluate(&vec2.0, &vec3.0);
            assert_abs_diff_eq!(distance, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_dist_cosine_normalized_float_turing() {
        // two vectors are allocated in the contiguous heap memory
        let (a_slice, b_slice) = get_turing_test_data();
        let distance = <[f32; 112] as FullPrecisionDistance<f32, 112>>::distance_compare(
            &a_slice.0,
            &b_slice.0,
            Metric::CosineNormalized,
        );

        let got: f32 = CosineNormalized::evaluate(&a_slice.0, &b_slice.0);
        assert_abs_diff_eq!(distance, got, epsilon = 1e-5);
    }

    #[test]
    fn test_dist_cosine_normalized_f16_turing() {
        // two vectors are allocated in the contiguous heap memory
        let (a_slice, b_slice) = get_turing_test_data_f16();
        let distance = <[Half; 112] as FullPrecisionDistance<Half, 112>>::distance_compare(
            &a_slice.0,
            &b_slice.0,
            Metric::CosineNormalized,
        );

        // Note the variance between the full 32 bit precision and the 16 bit precision
        let got: f32 = CosineNormalized::evaluate(&a_slice.0, &b_slice.0);
        assert_abs_diff_eq!(distance, got, epsilon = 1e-5);
    }

    #[test]
    fn cosine_normalized_distance_test() {
        #[repr(C, align(32))]
        struct Vector32ByteAligned {
            v: [f32; 512],
        }

        for seed in 0..100 {
            let mut vec1 = Box::new(Vector32ByteAligned { v: [0.0; 512] });

            let mut rng = StdRng::seed_from_u64(seed);
            for i in 0..512 {
                vec1.v[i] = rng.random_range(-1.0..1.0);
            }

            let mut vec2 = F32Slice256([0.0; 256]);
            let mut vec3 = F32Slice256([0.0; 256]);

            for j in 0..256 {
                vec2.0[j] = vec1.v[j];
            }
            for z in 256..512 {
                vec3.0[z - 256] = vec1.v[z];
            }

            let distance = compare::<f32, 256>(256, Metric::CosineNormalized, &vec1.v);
            let expected: f32 = CosineNormalized::evaluate(&vec2.0, &vec3.0);
            assert_abs_diff_eq!(distance, expected, epsilon = 1e-5);
        }
    }

    fn compare<T, const N: usize>(dim: usize, metric: Metric, v: &[f32]) -> f32
    where
        for<'a> [T; N]: FullPrecisionDistance<T, N>,
    {
        let a_ptr = v.as_ptr();
        let b_ptr = unsafe { a_ptr.add(dim) };

        let a_ref =
            <&[f32; N]>::try_from(unsafe { std::slice::from_raw_parts(a_ptr, dim) }).unwrap();
        let b_ref =
            <&[f32; N]>::try_from(unsafe { std::slice::from_raw_parts(b_ptr, dim) }).unwrap();

        <[f32; N]>::distance_compare(a_ref, b_ref, metric)
    }
}
