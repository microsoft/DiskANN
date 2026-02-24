/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmarks comparing inner product kernels:
//!
//! - `USlice<8> × USlice<4>` (heterogeneous 8-bit × 4-bit integer IP)
//! - `&[f32] × USlice<4>` (float × 4-bit IP)
//! - `&[f32] × &[f32]` (float × float IP)
//!
//! All three compute an inner product over the same logical dimension but with different
//! data representations. The benchmark measures raw kernel throughput for representative
//! embedding dimensions.

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
        use criterion::{criterion_group, criterion_main, Criterion, black_box};
        use diskann_quantization::bits::{BoxedBitSlice, Unsigned};
        use diskann_quantization::distances::InnerProduct as QuantInnerProduct;
        use diskann_utils::Reborrow;
        use diskann_vector::{PureDistanceFunction, MathematicalValue};
        use diskann_vector::distance::InnerProduct as VecInnerProduct;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use rand_distr::Uniform;
        use std::time::Duration;

        /// Dimensions to benchmark. Covers small (64), medium (256), and large (768)
        /// embeddings that are representative of real-world vector search workloads.
        const DIMS: &[usize] = &[64, 256, 768];

        /// Number of vector pairs per dimension. Enough to amortize per-iteration overhead.
        const COUNT: usize = 512;

        fn benchmark_u8x4_ip(c: &mut Criterion) {
            let mut group = c.benchmark_group("ip/u8x4");
            let mut rng = StdRng::seed_from_u64(0xbe4c_0084);

            for &dim in DIMS {
                // Pre-allocate all vector pairs.
                let dist_8bit = Uniform::new_inclusive(0u8, 255u8).unwrap();
                let dist_4bit = Uniform::new_inclusive(0i64, 15i64).unwrap();

                let xs: Vec<BoxedBitSlice<8, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<8, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_8bit) as i64).unwrap();
                        }
                        bs
                    })
                    .collect();

                let ys: Vec<BoxedBitSlice<4, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<4, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_4bit)).unwrap();
                        }
                        bs
                    })
                    .collect();

                group.bench_function(format!("dim={dim}"), |b| {
                    b.iter(|| {
                        for i in 0..COUNT {
                            let _ = black_box(
                                QuantInnerProduct::evaluate(
                                    black_box(xs[i].reborrow()),
                                    black_box(ys[i].reborrow()),
                                ),
                            );
                        }
                    });
                });
            }
            group.finish();
        }

        fn benchmark_f32x4_ip(c: &mut Criterion) {
            let mut group = c.benchmark_group("ip/f32x4");
            let mut rng = StdRng::seed_from_u64(0xf32a_0004);

            for &dim in DIMS {
                let dist_f32 = Uniform::new(-1.0f32, 1.0f32).unwrap();
                let dist_4bit = Uniform::new_inclusive(0i64, 15i64).unwrap();

                let xs: Vec<Vec<f32>> = (0..COUNT)
                    .map(|_| (0..dim).map(|_| rng.sample(dist_f32)).collect())
                    .collect();

                let ys: Vec<BoxedBitSlice<4, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<4, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_4bit)).unwrap();
                        }
                        bs
                    })
                    .collect();

                group.bench_function(format!("dim={dim}"), |b| {
                    b.iter(|| {
                        for i in 0..COUNT {
                            let _ = black_box(
                                QuantInnerProduct::evaluate(
                                    black_box(xs[i].as_slice()),
                                    black_box(ys[i].reborrow()),
                                ),
                            );
                        }
                    });
                });
            }
            group.finish();
        }

        fn benchmark_f32xf32_ip(c: &mut Criterion) {
            let mut group = c.benchmark_group("ip/f32xf32");
            let mut rng = StdRng::seed_from_u64(0xf32f_32ff);

            for &dim in DIMS {
                let dist_f32 = Uniform::new(-1.0f32, 1.0f32).unwrap();

                let xs: Vec<Vec<f32>> = (0..COUNT)
                    .map(|_| (0..dim).map(|_| rng.sample(dist_f32)).collect())
                    .collect();

                let ys: Vec<Vec<f32>> = (0..COUNT)
                    .map(|_| (0..dim).map(|_| rng.sample(dist_f32)).collect())
                    .collect();

                group.bench_function(format!("dim={dim}"), |b| {
                    b.iter(|| {
                        for i in 0..COUNT {
                            let r: MathematicalValue<f32> =
                                VecInnerProduct::evaluate(
                                    black_box(xs[i].as_slice()),
                                    black_box(ys[i].as_slice()),
                                );
                            black_box(r);
                        }
                    });
                });
            }
            group.finish();
        }

        fn benchmark_u8x8_ip(c: &mut Criterion) {
            let mut group = c.benchmark_group("ip/u8x8");
            let mut rng = StdRng::seed_from_u64(0xa8a8_0088);

            for &dim in DIMS {
                let dist_8bit = Uniform::new_inclusive(0u8, 255u8).unwrap();

                let xs: Vec<BoxedBitSlice<8, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<8, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_8bit) as i64).unwrap();
                        }
                        bs
                    })
                    .collect();

                let ys: Vec<BoxedBitSlice<8, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<8, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_8bit) as i64).unwrap();
                        }
                        bs
                    })
                    .collect();

                group.bench_function(format!("dim={dim}"), |b| {
                    b.iter(|| {
                        for i in 0..COUNT {
                            let _ = black_box(
                                QuantInnerProduct::evaluate(
                                    black_box(xs[i].reborrow()),
                                    black_box(ys[i].reborrow()),
                                ),
                            );
                        }
                    });
                });
            }
            group.finish();
        }

        fn benchmark_u4x4_ip(c: &mut Criterion) {
            let mut group = c.benchmark_group("ip/u4x4");
            let mut rng = StdRng::seed_from_u64(0x4444_0044);

            for &dim in DIMS {
                let dist_4bit = Uniform::new_inclusive(0i64, 15i64).unwrap();

                let xs: Vec<BoxedBitSlice<4, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<4, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_4bit)).unwrap();
                        }
                        bs
                    })
                    .collect();

                let ys: Vec<BoxedBitSlice<4, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<4, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_4bit)).unwrap();
                        }
                        bs
                    })
                    .collect();

                group.bench_function(format!("dim={dim}"), |b| {
                    b.iter(|| {
                        for i in 0..COUNT {
                            let _ = black_box(
                                QuantInnerProduct::evaluate(
                                    black_box(xs[i].reborrow()),
                                    black_box(ys[i].reborrow()),
                                ),
                            );
                        }
                    });
                });
            }
            group.finish();
        }

        fn benchmark_u8x2_ip(c: &mut Criterion) {
            let mut group = c.benchmark_group("ip/u8x2");
            let mut rng = StdRng::seed_from_u64(0x82c4_0022);

            for &dim in DIMS {
                let dist_8bit = Uniform::new_inclusive(0u8, 255u8).unwrap();
                let dist_2bit = Uniform::new_inclusive(0i64, 3i64).unwrap();

                let xs: Vec<BoxedBitSlice<8, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<8, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_8bit) as i64).unwrap();
                        }
                        bs
                    })
                    .collect();

                let ys: Vec<BoxedBitSlice<2, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<2, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_2bit)).unwrap();
                        }
                        bs
                    })
                    .collect();

                group.bench_function(format!("dim={dim}"), |b| {
                    b.iter(|| {
                        for i in 0..COUNT {
                            let _ = black_box(
                                QuantInnerProduct::evaluate(
                                    black_box(xs[i].reborrow()),
                                    black_box(ys[i].reborrow()),
                                ),
                            );
                        }
                    });
                });
            }
            group.finish();
        }

        fn benchmark_f32x2_ip(c: &mut Criterion) {
            let mut group = c.benchmark_group("ip/f32x2");
            let mut rng = StdRng::seed_from_u64(0xf32a_0002);

            for &dim in DIMS {
                let dist_f32 = Uniform::new(-1.0f32, 1.0f32).unwrap();
                let dist_2bit = Uniform::new_inclusive(0i64, 3i64).unwrap();

                let xs: Vec<Vec<f32>> = (0..COUNT)
                    .map(|_| (0..dim).map(|_| rng.sample(dist_f32)).collect())
                    .collect();

                let ys: Vec<BoxedBitSlice<2, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<2, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_2bit)).unwrap();
                        }
                        bs
                    })
                    .collect();

                group.bench_function(format!("dim={dim}"), |b| {
                    b.iter(|| {
                        for i in 0..COUNT {
                            let _ = black_box(
                                QuantInnerProduct::evaluate(
                                    black_box(xs[i].as_slice()),
                                    black_box(ys[i].reborrow()),
                                ),
                            );
                        }
                    });
                });
            }
            group.finish();
        }

        fn benchmark_f32x1_ip(c: &mut Criterion) {
            let mut group = c.benchmark_group("ip/f32x1");
            let mut rng = StdRng::seed_from_u64(0xf32a_0001);

            for &dim in DIMS {
                let dist_f32 = Uniform::new(-1.0f32, 1.0f32).unwrap();
                let dist_1bit = Uniform::new_inclusive(0i64, 1i64).unwrap();

                let xs: Vec<Vec<f32>> = (0..COUNT)
                    .map(|_| (0..dim).map(|_| rng.sample(dist_f32)).collect())
                    .collect();

                let ys: Vec<BoxedBitSlice<1, Unsigned>> = (0..COUNT)
                    .map(|_| {
                        let mut bs = BoxedBitSlice::<1, Unsigned>::new_boxed(dim);
                        for j in 0..dim {
                            bs.set(j, rng.sample(dist_1bit)).unwrap();
                        }
                        bs
                    })
                    .collect();

                group.bench_function(format!("dim={dim}"), |b| {
                    b.iter(|| {
                        for i in 0..COUNT {
                            let _ = black_box(
                                QuantInnerProduct::evaluate(
                                    black_box(xs[i].as_slice()),
                                    black_box(ys[i].reborrow()),
                                ),
                            );
                        }
                    });
                });
            }
            group.finish();
        }

        criterion_group!(
            name = benches;
            config = Criterion::default()
                .sample_size(500)
                .warm_up_time(Duration::from_secs(2))
                .measurement_time(Duration::from_secs(5))
                .nresamples(50_000);
            targets =
                benchmark_u8x8_ip,
                benchmark_u8x4_ip,
                benchmark_u8x2_ip,
                benchmark_u4x4_ip,
                benchmark_f32x4_ip,
                benchmark_f32x2_ip,
                benchmark_f32x1_ip,
                benchmark_f32xf32_ip,
        );
        criterion_main!(benches);
    } else {
        fn main() {} // no benchmarks for non-SIMD hardware
    }
}
