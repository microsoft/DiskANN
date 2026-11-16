/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rand::{Rng, distr::Open01};

pub(super) fn revive_underused_clusters(
    mut centers: rowmajor::Mut<'_, f32>,
    masses: &mut [f64],
    rng: &mut (impl Rng + ?Sized),
) -> usize {
    let threshold = 0.1 * masses.iter().sum::<f64>() / masses.len() as f64;
    let underused: Vec<_> = masses
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| (m < threshold).then_some(i))
        .collect();
    if underused.is_empty() {
        return 0;
    }
    // Gumbel top-k samples donors in proportion to mass, without replacement.
    let mut donors: Vec<_> = masses
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m >= threshold && m > 0.0)
        .map(|(i, &m)| {
            let uniform: f64 = rng.sample(Open01);
            (m.ln() - (-uniform.ln()).ln(), i)
        })
        .collect();
    donors.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let splits = underused.len().min(donors.len());
    for (target, (_, source)) in underused.into_iter().zip(donors) {
        let half_mass = masses[source] * 0.5;
        masses[source] = half_mass;
        masses[target] = half_mass;
        for coordinate in 0..centers.ncols() {
            let value = centers.row(source)[coordinate];
            let delta = if coordinate.is_multiple_of(2) {
                1e-4
            } else {
                -1e-4
            };
            centers.row_mut(source)[coordinate] = value * (1.0 - delta);
            centers.row_mut(target)[coordinate] = value * (1.0 + delta);
        }
        if centers.row(source) == centers.row(target) {
            let value = centers.row(target)[0];
            let next = value.next_up();
            centers.row_mut(target)[0] = if next.is_finite() {
                next
            } else {
                value.next_down()
            };
        }
    }
    splits
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn assert_near(got: f64, expected: f64, tolerance: f64) {
        assert!(
            (got - expected).abs() <= tolerance,
            "expected {expected}, got {got}, tolerance {tolerance}"
        );
    }

    #[test]
    fn revival_uses_distinct_donors_and_conserves_active_mass() {
        let mut centers = rowmajor::Owned::from_fn(5, 3, |rc| (10 * rc.row + rc.col + 1) as f32);
        let original = centers.clone();
        let mut masses = vec![8.0, 4.0, 0.0, 0.0, 0.0];
        assert_eq!(
            revive_underused_clusters(
                centers.as_view_mut(),
                &mut masses,
                &mut StdRng::seed_from_u64(11)
            ),
            2
        );
        assert_eq!(&masses[..2], &[4.0, 2.0]);
        assert_eq!(masses.iter().sum::<f64>(), 12.0);
        assert_eq!(masses[4], 0.0);
        assert_eq!(centers.row(4), original.row(4));
        assert!((masses[2] == 4.0 && masses[3] == 2.0) || (masses[2] == 2.0 && masses[3] == 4.0));
        for (target, &mass) in masses.iter().enumerate().take(4).skip(2) {
            let source = if mass == 4.0 { 0 } else { 1 };
            for coordinate in 0..3 {
                assert_near(
                    f64::from(centers.row(source)[coordinate])
                        + f64::from(centers.row(target)[coordinate]),
                    2.0 * f64::from(original.row(source)[coordinate]),
                    4e-6,
                );
            }
        }
    }
    #[test]
    fn revival_separates_zero_centers_and_skips_active_clusters() {
        let mut centers = rowmajor::Owned::from_element(2, 3, 0.0);
        let mut masses = vec![2.0, 0.0];
        assert_eq!(
            revive_underused_clusters(
                centers.as_view_mut(),
                &mut masses,
                &mut StdRng::seed_from_u64(1)
            ),
            1
        );
        assert_eq!(masses, [1.0, 1.0]);
        assert_eq!(centers.row(0), &[0.0, 0.0, 0.0]);
        assert_eq!(centers.row(1), &[0.0_f32.next_up(), 0.0, 0.0]);
        let previous = centers.clone();
        assert_eq!(
            revive_underused_clusters(
                centers.as_view_mut(),
                &mut masses,
                &mut StdRng::seed_from_u64(1)
            ),
            0
        );
        assert_eq!(centers.as_slice(), previous.as_slice());
    }
}
