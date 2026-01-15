/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(test)]
mod e2e_test {
    use rand::{distr::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

    use crate::{
        distance::{Cosine, CosineNormalized, InnerProduct, SquaredL2},
        PureDistanceFunction,
    };

    #[test]
    /// Assert that all three distance functions return the same order
    /// for a set of vectors. (L2, Cosine, Cosinenormalized)
    fn all_distance_comparers_have_same_order() {
        let vec = prepare_random_aligned_vectors();
        let (order_1, order_2, order_3, order_4) = calculate_distance_order(&vec);

        assert_eq!(order_1, order_2, "1 vs 2");
        assert_eq!(order_1, order_3, "1 vs 3");
        assert_eq!(order_1, order_4, "1 vs 4");
    }

    // Function to calculate the distance order for a set of vectors using each distance function.
    fn calculate_distance_order(
        vectors: &[Vector32ByteAligned],
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut order_1 = (0..vectors.len()).collect::<Vec<usize>>();
        let mut order_2 = order_1.clone();
        let mut order_3 = order_1.clone();
        let mut order_4 = order_1.clone();

        // Sort the indices based on the distances using each distance function.
        order_1.sort_unstable_by(|&a, &b| {
            let x: f32 = SquaredL2::evaluate(&vectors[a].v, &vectors[0].v);
            let y: f32 = SquaredL2::evaluate(&vectors[b].v, &vectors[0].v);
            x.partial_cmp(&y).unwrap()
        });

        order_2.sort_unstable_by(|&a, &b| {
            let x: f32 = InnerProduct::evaluate(&vectors[a].v, &vectors[0].v);
            let y: f32 = InnerProduct::evaluate(&vectors[b].v, &vectors[0].v);
            x.partial_cmp(&y).unwrap()
        });

        order_3.sort_unstable_by(|&a, &b| {
            let x: f32 = CosineNormalized::evaluate(&vectors[a].v, &vectors[0].v);
            let y: f32 = CosineNormalized::evaluate(&vectors[b].v, &vectors[0].v);
            x.partial_cmp(&y).unwrap()
        });

        order_4.sort_unstable_by(|&a, &b| {
            let x: f32 = Cosine::evaluate(&vectors[a].v, &vectors[0].v);
            let y: f32 = Cosine::evaluate(&vectors[b].v, &vectors[0].v);
            x.partial_cmp(&y).unwrap()
        });

        (order_1, order_2, order_3, order_4)
    }

    #[repr(C, align(32))]
    struct Vector32ByteAligned {
        v: [f32; 104],
    }

    // make sure the vector is 104-bit (32 bytes) aligned required by AVX2
    fn prepare_random_aligned_vectors() -> Vec<Vector32ByteAligned> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let seed = rng.random::<u8>();
        let seed: [u8; 32] = [seed; 32]; // Constant seed of a random number.
        println!("seed: {:?}", seed);
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let range = Uniform::new(-100.0, 100.0).unwrap();

        let mut vec = Vec::with_capacity(10);
        for _ in 0..10 {
            let mut a = [0.0; 104];
            for elem in a.iter_mut() {
                *elem = range.sample(&mut rng) as f32;
            }
            normalize_vector(&mut a);
            vec.push(Vector32ByteAligned { v: a });
        }

        vec
    }

    fn normalize_vector(vector: &mut [f32]) {
        let magnitude_squared: f32 = vector.iter().map(|&x| x * x).sum();
        let magnitude = magnitude_squared.sqrt();

        if magnitude != 0.0 {
            for elem in vector.iter_mut() {
                *elem /= magnitude;
            }
        }
    }
}
