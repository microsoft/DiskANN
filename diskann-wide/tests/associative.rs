/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// This integration test demonstrates that different SIMD tiling strategies and
// implementation strategies can yield different results for the same operation due to
// differences in associative summation of floating point numbers.
use diskann_wide::*;

alias!(f32s = f32x8);

// The number of lanes to use in a SIMD register.
const LANES: usize = f32s::LANES;

/// This is a basic implementation of inner product using a single 8-wide accumulation
/// register.
fn innerproduct_no_unroll(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let len = a.len();
    let a = a.as_ptr();
    let b = b.as_ptr();

    let mut sum = f32s::default(ARCH);
    let trips = len / LANES;
    let remainder = len % LANES;
    for i in 0..trips {
        // SAFETY: By loop construction, `a.add(LANES * (i + 1) - 1)` is always in-bounds.
        let wa = unsafe { f32s::load_simd(ARCH, a.add(LANES * i)) };
        // SAFETY: By loop construction, `b.add(LANES * (i + 1) - 1)` is always in-bounds.
        let wb = unsafe { f32s::load_simd(ARCH, b.add(LANES * i)) };
        sum = wa.mul_add_simd(wb, sum);
    }

    // Handle and remaining using predicated loads.
    if remainder != 0 {
        // SAFETY: By loop construction, `a.add(LANES * trips)` is always in-bounds.
        let wa = unsafe { f32s::load_simd_first(ARCH, a.add(trips * LANES), remainder) };
        // SAFETY: By loop construction, `b.add(LANES * trips)` is always in-bounds.
        let wb = unsafe { f32s::load_simd_first(ARCH, b.add(trips * LANES), remainder) };
        sum = wa.mul_add_simd(wb, sum);
    }

    sum.sum_tree()
}

/// The `no_unroll` implementation can suffer from a lack of instruction level parallelism
/// because the single accumulation register becomes a serial bottleneck.
///
/// Therefore, it can be helpful to use multiple accumulation registers to make better use
/// of the underlying hardware.
///
/// In this case, the order in which partial products are accumulated is different than
/// in the `no_unroll` case. Effectively, this implementation uses a 32-wide register for
/// accumulation rather than an 8-wide register.
fn innerproduct_4x_unroll(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    const UNROLL: usize = 4;
    let len = a.len();
    let a = a.as_ptr();
    let b = b.as_ptr();

    const ELEMENTS_PER_UNROLLED_TRIP: usize = LANES * UNROLL;
    let unrolled_trips = len / ELEMENTS_PER_UNROLLED_TRIP;
    let next_after_unrolled = unrolled_trips * ELEMENTS_PER_UNROLLED_TRIP;

    let full_width_epilogues = (len - (unrolled_trips * ELEMENTS_PER_UNROLLED_TRIP)) / LANES;

    let remainder = len % LANES;

    // Unrolled trip.
    let mut sum = if unrolled_trips != 0 {
        let mut s0 = f32s::default(ARCH);
        let mut s1 = f32s::default(ARCH);
        let mut s2 = f32s::default(ARCH);
        let mut s3 = f32s::default(ARCH);

        for i in 0..unrolled_trips {
            // Unroll 0
            // SAFETY: By construction, `ELEMENTS_PER_UNROLLED_TRIP * i + 4 * LANES - 1` is
            // always in-bounds, so this load is safe.
            let wa = unsafe { f32s::load_simd(ARCH, a.add(ELEMENTS_PER_UNROLLED_TRIP * i)) };
            // SAFETY: By construction, `ELEMENTS_PER_UNROLLED_TRIP * i + 4 * LANES - 1` is
            // always in-bounds, so this load is safe.
            let wb = unsafe { f32s::load_simd(ARCH, b.add(ELEMENTS_PER_UNROLLED_TRIP * i)) };
            s0 = wa.mul_add_simd(wb, s0);

            // Unroll 1
            // SAFETY: By construction, `ELEMENTS_PER_UNROLLED_TRIP * i + 4 * LANES - 1` is
            // always in-bounds, so this load is safe.
            let wa =
                unsafe { f32s::load_simd(ARCH, a.add(ELEMENTS_PER_UNROLLED_TRIP * i + LANES)) };
            // SAFETY: By construction, `ELEMENTS_PER_UNROLLED_TRIP * i + 4 * LANES - 1` is
            // always in-bounds, so this load is safe.
            let wb =
                unsafe { f32s::load_simd(ARCH, b.add(ELEMENTS_PER_UNROLLED_TRIP * i + LANES)) };
            s1 = wa.mul_add_simd(wb, s1);

            // Unroll 2
            // SAFETY: By construction, `ELEMENTS_PER_UNROLLED_TRIP * i + 4 * LANES - 1` is
            // always in-bounds, so this load is safe.
            let wa =
                unsafe { f32s::load_simd(ARCH, a.add(ELEMENTS_PER_UNROLLED_TRIP * i + 2 * LANES)) };
            // SAFETY: By construction, `ELEMENTS_PER_UNROLLED_TRIP * i + 4 * LANES - 1` is
            // always in-bounds, so this load is safe.
            let wb =
                unsafe { f32s::load_simd(ARCH, b.add(ELEMENTS_PER_UNROLLED_TRIP * i + 2 * LANES)) };
            s2 = wa.mul_add_simd(wb, s2);

            // Unroll 3
            // SAFETY: By construction, `ELEMENTS_PER_UNROLLED_TRIP * i + 4 * LANES - 1` is
            // always in-bounds, so this load is safe.
            let wa =
                unsafe { f32s::load_simd(ARCH, a.add(ELEMENTS_PER_UNROLLED_TRIP * i + 3 * LANES)) };
            // SAFETY: By construction, `ELEMENTS_PER_UNROLLED_TRIP * i + 4 * LANES - 1` is
            // always in-bounds, so this load is safe.
            let wb =
                unsafe { f32s::load_simd(ARCH, b.add(ELEMENTS_PER_UNROLLED_TRIP * i + 3 * LANES)) };
            s3 = wa.mul_add_simd(wb, s3);
        }
        (s0 + s1) + (s2 + s3)
    } else {
        f32s::default(ARCH)
    };

    for i in 0..full_width_epilogues {
        // SAFETY: By construction, `next_after_unrolled + LANES * full_width_epilogues - 1`
        // is in-bounds, so this load is safe.
        let wa = unsafe { f32s::load_simd(ARCH, a.add(next_after_unrolled + i * LANES)) };
        // SAFETY: By construction, `next_after_unrolled + LANES * full_width_epilogues - 1`
        // is in-bounds, so this load is safe.
        let wb = unsafe { f32s::load_simd(ARCH, b.add(next_after_unrolled + i * LANES)) };
        sum = wa.mul_add_simd(wb, sum);
    }

    // Handle and remaining using predicated loads.
    if remainder != 0 {
        // SAFETY: We've checked that the remainder is nonzero and `trips * LANES < len`.
        let wa = unsafe { f32s::load_simd_first(ARCH, a.add(len - remainder), remainder) };
        // SAFETY: We've checked that the remainder is nonzero and `trips * LANES < len`.
        let wb = unsafe { f32s::load_simd_first(ARCH, b.add(len - remainder), remainder) };
        sum = wa.mul_add_simd(wb, sum);
    }

    sum.sum_tree()
}

/// The two previous implementations add together positive and negative partial products
/// with reckless abandon.
///
/// Another strategy is to send all of the positive partial products to one accumulator and
/// all the negative partial products to another.
///
/// This *can* avoid some cancelation of small partial products.
fn innerproduct_switching(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let len = a.len();
    let a = a.as_ptr();
    let b = b.as_ptr();

    let mut sum_positive = f32s::default(ARCH);
    let mut sum_negative = f32s::default(ARCH);
    let zero = f32s::default(ARCH);

    let trips = len / LANES;
    let remainder = len % LANES;
    for i in 0..trips {
        // SAFETY: By loop construction, `a.add(LANES * (i + 1) - 1)` is always in-bounds.
        let wa = unsafe { f32s::load_simd(ARCH, a.add(LANES * i)) };
        // SAFETY: By loop construction, `b.add(LANES * (i + 1) - 1)` is always in-bounds.
        let wb = unsafe { f32s::load_simd(ARCH, b.add(LANES * i)) };

        // Multiply the two values together and then send the partial sums to their
        // respective accumulators.
        let partial = wa * wb;
        let ge_mask = partial.ge_simd(zero);
        sum_positive = sum_positive + ge_mask.select(partial, zero);
        sum_negative = sum_negative + ge_mask.select(zero, partial);
    }

    // Handle and remaining using predicated loads.
    if remainder != 0 {
        // SAFETY: We've checked that the remainder is nonzero and `trips * LANES < len`.
        let wa = unsafe { f32s::load_simd_first(ARCH, a.add(trips * LANES), remainder) };
        // SAFETY: We've checked that the remainder is nonzero and `trips * LANES < len`.
        let wb = unsafe { f32s::load_simd_first(ARCH, b.add(trips * LANES), remainder) };

        let partial = wa * wb;

        let ge_mask = partial.ge_simd(zero);
        sum_positive = sum_positive + ge_mask.select(partial, zero);
        sum_negative = sum_negative + ge_mask.select(zero, partial);
    }

    // Only one occurance of a positive number plus a negative number.
    sum_positive.sum_tree() + sum_negative.sum_tree()
}

/// Finally, lets explore a sequential implementation that promotes and accumulates in `f64`.
fn innerproduct_f64_nosimd(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    std::iter::zip(x.iter(), y.iter()).fold(0.0f64, |acc, (&ix, &iy)| {
        let ix: f64 = ix.into();
        let iy: f64 = iy.into();

        ix.mul_add(iy, acc)
    }) as f32
}

/// Even though all implementation are computing the same function - the way they compute
/// it varies.
///
/// This lead to slight differences in the computed result.
///
/// Now - why might accumulation widths differ between architectures?
///
/// * x86 processors with AVX2 can have 16 architectural registers with a width up to 256-bits.
/// * x86 processorw with AVX-512 have 32 architectural registers with a width up to 512-bits
///   plus a separate mask register file, which may influence an implementation.
/// * Arm processors with Neon have 32 architectural registers with a width of 128-bits.
/// * Arm processors with SVE have register widths that aren't even known at compile time.
///
/// These differences in micro-architecture mean the best implementation *can* depend on the
/// underlying machine.
#[test]
fn demonstrate_difference() {
    let (x, y) = get_x_and_y();

    let no_unroll = innerproduct_no_unroll(&x, &y);
    let unroll_4x = innerproduct_4x_unroll(&x, &y);
    let switching = innerproduct_switching(&x, &y);
    let from_f64 = innerproduct_f64_nosimd(&x, &y);

    println!("No Unroll: {no_unroll}");
    println!("Unroll 4x: {unroll_4x}");
    println!("Switching: {switching}");
    println!("From F64: {from_f64}");

    assert_eq!(no_unroll, -2.4482543);
    assert_eq!(unroll_4x, -2.4482546);
    assert_eq!(switching, -2.4482555);
    assert_eq!(from_f64, -2.4482548);
}

/// Return two sample vectors to demonstrate the difference implementation strategy can
/// have on the final result.
fn get_x_and_y() -> (Vec<f32>, Vec<f32>) {
    let x = vec![
        1.0587195,
        -0.03812505,
        0.54363215,
        -0.5245157,
        -1.1216756,
        -0.96697515,
        -0.6023211,
        -0.7890334,
        0.71664125,
        0.93929476,
        -1.0040216,
        0.0009801295,
        -0.22976695,
        -0.122815445,
        -0.2199377,
        1.3805063,
        -0.20443898,
        -0.35089707,
        0.561345,
        1.0442268,
        1.1165859,
        -0.60838705,
        -0.2206766,
        0.73438275,
        0.5554171,
        -1.1783961,
        -0.5537273,
        -0.31328288,
        -0.8012327,
        -0.68842053,
        0.92414194,
        -1.6065389,
        0.5011909,
        -0.60476923,
        1.0768179,
        1.0933931,
        -0.8196649,
        -0.7924749,
        0.1986087,
        1.1199812,
        0.59062964,
        0.8188963,
        1.5737948,
        0.7905916,
        0.4158534,
        0.07332524,
        -2.3650591,
    ];

    let y = vec![
        0.42544082,
        -0.74533653,
        -0.3201682,
        0.24189381,
        0.81574935,
        0.92732084,
        0.49251068,
        -0.09835721,
        -0.26167348,
        -0.18099633,
        -1.5926921,
        0.57866585,
        0.67955637,
        -0.25137362,
        1.3598162,
        -1.6068382,
        -0.97026235,
        0.7185971,
        -0.124563806,
        0.022480685,
        -0.41827333,
        -0.2020838,
        -0.10343805,
        1.6951554,
        0.22812097,
        0.8407058,
        1.2321464,
        0.895014,
        1.3540926,
        -0.28383157,
        1.3259526,
        -0.9737379,
        0.5235545,
        0.5777137,
        0.2526582,
        -1.2532156,
        -0.24522695,
        0.76966465,
        -0.5180771,
        0.02358619,
        -0.10248907,
        0.1371764,
        0.56763655,
        -1.0281502,
        -1.9113115,
        -0.9008785,
        -0.975584,
    ];

    (x, y)
}
