/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(target_arch = "x86_64")]
use diskann_wide::{SIMDMask, SIMDPartialOrd, SIMDVector};

// Experimental kernels. The linear kernel reproduces the existing queue scan.
#[inline]
#[cfg(target_arch = "x86_64")]
pub(super) fn linear(distances: &[f32], target: f32) -> usize {
    diskann_wide::alias!(f32s = f32x8);
    let target = f32s::splat(diskann_wide::ARCH, target);
    let mut index = 0;
    while index + 16 <= distances.len() {
        // SAFETY: Both eight-lane loads are within the slice.
        let search = unsafe { f32s::load_simd(diskann_wide::ARCH, distances.as_ptr().add(index)) };
        let offset1 = search.ge_simd(target).first();
        // SAFETY: The loop condition covers these eight lanes.
        let search =
            unsafe { f32s::load_simd(diskann_wide::ARCH, distances.as_ptr().add(index + 8)) };
        let offset2 = search.ge_simd(target).first();
        match (offset1, offset2) {
            (Some(offset), _) => return index + offset,
            (None, Some(offset)) => return index + 8 + offset,
            _ => (),
        }
        index += 16;
    }
    if index + 8 <= distances.len() {
        // SAFETY: The condition covers these eight lanes.
        let search = unsafe { f32s::load_simd(diskann_wide::ARCH, distances.as_ptr().add(index)) };
        if let Some(offset) = search.ge_simd(target).first() {
            return index + offset;
        }
        index += 8;
    }
    if index < distances.len() {
        // SAFETY: Only the remaining initialized lanes are loaded.
        let search = unsafe {
            f32s::load_simd_first(
                diskann_wide::ARCH,
                distances.as_ptr().add(index),
                distances.len() - index,
            )
        };
        if let Some(offset) = search.ge_simd(target).first() {
            return index + offset;
        }
    }
    distances.len()
}

#[inline]
#[cfg(not(target_arch = "x86_64"))]
pub(super) fn linear(distances: &[f32], target: f32) -> usize {
    distances
        .iter()
        .position(|distance| *distance >= target)
        .unwrap_or(distances.len())
}

#[inline]
pub(super) fn binary(distances: &[f32], target: f32) -> usize {
    // insert/batch_insert filter NaNs; remove also calls lower_bound.
    if target.is_nan() {
        return distances.len();
    }
    distances.partition_point(|distance| *distance < target)
}

#[inline]
pub(super) fn hybrid<const THRESHOLD: usize, const PREFIX: usize>(
    distances: &[f32],
    target: f32,
) -> usize {
    if distances.len() <= THRESHOLD {
        return linear(distances, target);
    }
    // Keep near-front insertions cheap, especially the minimum of a large batch.
    let prefix = linear(&distances[..PREFIX], target);
    if prefix < PREFIX {
        prefix
    } else {
        PREFIX + binary(&distances[PREFIX..], target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_bounds_preserve_ties_and_numeric_edges() {
        for len in [
            0, 1, 7, 8, 15, 16, 17, 20, 40, 100, 127, 128, 129, 200, 255, 256, 257, 500, 511, 512,
            513, 1000, 1023, 1024, 1025, 2000, 8192,
        ] {
            let mut distances: Vec<_> = (0..len)
                .map(|i| (i / 3) as f32 - len as f32 / 6.0)
                .collect();
            if len >= 4 {
                distances[0] = f32::NEG_INFINITY;
                distances[len - 1] = f32::INFINITY;
            }
            let mut targets = distances.clone();
            targets.extend([
                f32::NAN,
                f32::NEG_INFINITY,
                -0.0,
                0.0,
                f32::INFINITY,
                -10000.0,
                10000.0,
            ]);
            targets.extend(distances.windows(2).map(|w| (w[0] + w[1]) * 0.5));
            for target in targets {
                let expected = distances
                    .iter()
                    .position(|distance| *distance >= target)
                    .unwrap_or(len);
                assert_eq!(linear(&distances, target), expected, "{len} {target}");
                assert_eq!(binary(&distances, target), expected, "{len} {target}");
                assert_eq!(
                    hybrid::<128, 16>(&distances, target),
                    expected,
                    "{len} {target}"
                );
                assert_eq!(
                    hybrid::<256, 16>(&distances, target),
                    expected,
                    "{len} {target}"
                );
                assert_eq!(
                    hybrid::<512, 16>(&distances, target),
                    expected,
                    "{len} {target}"
                );
                assert_eq!(
                    hybrid::<1024, 16>(&distances, target),
                    expected,
                    "{len} {target}"
                );
                assert_eq!(
                    hybrid::<512, 256>(&distances, target),
                    expected,
                    "{len} {target}"
                );
            }
        }
        for distances in [[-0.0, 0.0], [0.0, -0.0]] {
            for target in [-0.0, 0.0] {
                assert_eq!(linear(&distances, target), 0);
                assert_eq!(binary(&distances, target), 0);
                assert_eq!(hybrid::<128, 16>(&distances, target), 0);
            }
        }
    }
}
