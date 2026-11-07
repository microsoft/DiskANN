/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Consume complete dot tiles into sorted nearest sets instead of max reductions.

use crate::matrix_kernels::blocks::unpacked::PanelVisitor;
use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    driver,
    maxsim::packed_f32_x_unpacked_f32::{ExtraWide, Params, dot_tile, visit_tiles},
    num::{Bytes, DimK},
};
use diskann_vector::{
    distance::Metric,
    topk::{Candidate, insert_sorted},
};
use diskann_wide::{Architecture, SIMDFloat, SIMDMask, arch::Scalar};

diskann_wide::alias!(f32x4<A> = f32x4);
diskann_wide::alias!(f32x8<A> = f32x8);
diskann_wide::alias!(f32x16<A> = f32x16);

/// Checked working-set accounting, including nearest sets and L2 norms.
pub(crate) fn params(
    cache: Cache,
    dimensions: usize,
    width: usize,
    mr: usize,
    nr: usize,
    metric: Metric,
) -> Option<Params> {
    let norm = if metric == Metric::L2 {
        size_of::<f32>()
    } else {
        0
    };
    let codes = dimensions.checked_mul(size_of::<f32>())?;
    let row = codes
        .checked_add(width.checked_mul(size_of::<Candidate>())?)?
        .checked_add(size_of::<f32>())?
        .checked_add(norm)?;
    let a = mr.checked_mul(row)?;
    let b = codes.checked_add(norm)?;
    let b_group = nr.checked_mul(b)?;
    let budget = cache.l1().get().saturating_sub(a).max(1);
    nr.checked_mul(budget.div_ceil(b_group))?;
    if a > isize::MAX as usize || b_group > isize::MAX as usize {
        return None;
    }
    Some(Params::new(cache, Bytes::new(a), Bytes::new(b), nr))
}

/// Output and thresholds span the whole B scan, not an individual tile.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, f32, MR>,
    b: unpacked::View<'a, f32>,
    c: Rows<'a>,
    b_norms: &'a [f32],
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// # Safety
    ///
    /// Panels have contraction dimension `k`. `c.width` is positive,
    /// `c.c` has exactly `c.valid * c.width` slots, and `c.valid` occupies
    /// exactly the A panels. `c.kth` covers the padded A rows. L2 norms in
    /// `c.a_norms` and `b_norms` cover padded A rows and all B rows.
    /// B row IDs fit below `u32::MAX`. Blocking was checked by [`params`].
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f32>,
        c: Rows<'a>,
        b_norms: &'a [f32],
        k: DimK,
        params: Params,
    ) -> Self {
        Self {
            arch,
            a,
            b,
            c,
            b_norms,
            k,
            params,
        }
    }
}

/// The logical C rows touched by one tile, with padded A metadata.
pub(crate) struct Rows<'a> {
    c: &'a mut [Candidate],
    kth: &'a mut [f32],
    a_norms: &'a [f32],
    metric: Metric,
    width: usize,
    valid: usize,
}

impl<'a> Rows<'a> {
    pub(crate) fn new(
        c: &'a mut [Candidate],
        kth: &'a mut [f32],
        a_norms: &'a [f32],
        metric: Metric,
        width: usize,
    ) -> Self {
        Self {
            valid: c.len() / width,
            c,
            kth,
            a_norms,
            metric,
            width,
        }
    }
}

pub(crate) trait Kernel<const MR: usize>: ExtraWide<MR> + Architecture {
    fn offer<const K: usize>(self, dots: Self::Wide, rows: &mut Rows<'_>, b_norm: f32, id: u32);
}

#[inline(always)]
#[expect(
    clippy::expect_used,
    reason = "fixed widths are selected before the tile scan"
)]
fn offer<V, const N: usize, const K: usize>(
    arch: V::Arch,
    dot: V,
    rows: &mut Rows<'_>,
    b_norm: f32,
    id: u32,
) where
    V: SIMDFloat<Scalar = f32, ConstLanes = diskann_wide::Const<N>>,
    u64: From<<<V::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    assert_eq!(V::LANES, N);
    // SAFETY: The driver provides padded thresholds covering all native lanes.
    let thresholds = unsafe { V::load_simd(arch, rows.kth.as_ptr()) };
    let distances = match rows.metric {
        Metric::InnerProduct => V::splat(arch, -1.0) * dot,
        Metric::L2 => {
            // SAFETY: L2 A norms have the same padded extent as the thresholds.
            let a_norm = unsafe { V::load_simd(arch, rows.a_norms.as_ptr()) };
            (a_norm + V::splat(arch, b_norm)) - V::splat(arch, 2.0) * dot
        }
        _ => unreachable!("the TopK factory validates the metric"),
    };
    let mask = u64::from(distances.lt_simd(thresholds).bitmask().to_underlying())
        & ((1u64 << rows.valid) - 1);
    diskann_vector::topk::for_each_eligible::<V, N>(distances, mask, |lane, distance| {
        let nearest = &mut rows.c[lane * rows.width..(lane + 1) * rows.width];
        let candidate = Candidate::new(id, distance);
        rows.kth[lane] = if K == 0 {
            insert_sorted(nearest, candidate)
        } else {
            let nearest: &mut [Candidate; K] = nearest.try_into().expect("fixed TopK width");
            insert_sorted(nearest, candidate)
        };
    });
}

macro_rules! kernel {
    ($arch:ty, $mr:literal, $lanes:literal, $vector:ty) => {
        impl Kernel<$mr> for $arch {
            #[inline(always)]
            fn offer<const K: usize>(
                self,
                dots: Self::Wide,
                rows: &mut Rows<'_>,
                b_norm: f32,
                id: u32,
            ) {
                for (group, dot) in dots.into_iter().enumerate() {
                    let first = group * $lanes;
                    let valid = rows.valid.saturating_sub(first).min($lanes);
                    if valid == 0 {
                        continue;
                    }
                    let mut group_rows = Rows {
                        c: &mut rows.c[first * rows.width..],
                        kth: &mut rows.kth[first..],
                        a_norms: if rows.metric == Metric::L2 {
                            &rows.a_norms[first..]
                        } else {
                            &[]
                        },
                        metric: rows.metric,
                        width: rows.width,
                        valid,
                    };
                    offer::<$vector, $lanes, K>(self, dot, &mut group_rows, b_norm, id);
                }
            }
        }
    };
}

kernel!(Scalar, 8, 4, f32x4<Scalar>);
#[cfg(target_arch = "x86_64")]
kernel!(
    diskann_wide::arch::x86_64::V3,
    16,
    8,
    f32x8<diskann_wide::arch::x86_64::V3>
);
#[cfg(target_arch = "x86_64")]
kernel!(
    diskann_wide::arch::x86_64::V4,
    32,
    16,
    f32x16<diskann_wide::arch::x86_64::V4>
);
#[cfg(target_arch = "aarch64")]
kernel!(
    diskann_wide::arch::aarch64::Neon,
    8,
    4,
    f32x4<diskann_wide::arch::aarch64::Neon>
);

struct Visitor<'a, A, const MR: usize, const K: usize> {
    arch: A,
    a: packed::Panel<'a, f32, MR>,
    rows: Rows<'a>,
    b_norms: &'a [f32],
    b_base: usize,
    k: DimK,
}

impl<A: Kernel<MR>, const MR: usize, const NR: usize, const K: usize>
    unpacked::PanelVisitor<f32, NR> for Visitor<'_, A, MR, K>
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, f32, NR>, start: usize) {
        // SAFETY: All panel views come from the validated driver with dimension `k`.
        let dots = unsafe { dot_tile(self.arch, self.a, b, self.k) };
        for (column, dot) in dots.into_iter().enumerate() {
            let id = self.b_base + start + column;
            let norm = if self.rows.metric == Metric::L2 {
                self.b_norms[id]
            } else {
                0.0
            };
            self.arch.offer::<K>(dot, &mut self.rows, norm, id as u32);
        }
    }
}

struct Entry<'a, 'b, A, const MR: usize, const NR: usize>(&'a mut Driver<'b, A, MR, NR>);

impl<A: Kernel<MR>, const MR: usize, const NR: usize, const K: usize>
    unpacked::PanelVisitor<f32, NR> for &mut Visitor<'_, A, MR, K>
{
    #[inline(always)]
    fn visit(&mut self, panel: unpacked::Panel<'_, f32, NR>, start: usize) {
        (**self).visit(panel, start);
    }
}

impl<A: Kernel<MR>, const MR: usize, const NR: usize> driver::Drive for Driver<'_, A, MR, NR> {
    fn drive(&mut self) {
        self.arch.run(Entry(self));
    }
}

impl<A: Kernel<MR>, const MR: usize, const NR: usize> diskann_wide::arch::Target<A, ()>
    for Entry<'_, '_, A, MR, NR>
{
    #[inline(always)]
    fn run(self, _: A) {
        match self.0.c.width {
            1 => self.0.scan::<1>(),
            2 => self.0.scan::<2>(),
            3 => self.0.scan::<3>(),
            8 => self.0.scan::<8>(),
            10 => self.0.scan::<10>(),
            _ => self.0.scan::<0>(),
        }
    }
}

impl<A: Kernel<MR>, const MR: usize, const NR: usize> Driver<'_, A, MR, NR> {
    #[inline(always)]
    fn scan<const K: usize>(&mut self) {
        let arch = self.arch;
        let k = self.k;
        let width = self.c.width;
        let rows = self.c.valid;
        let metric = self.c.metric;
        let c = &mut *self.c.c;
        let kth = &mut *self.c.kth;
        let a_norms = self.c.a_norms;
        let b_norms = self.b_norms;
        // SAFETY: The constructor requires matching contraction dimensions.
        unsafe {
            visit_tiles(
                self.a,
                self.b,
                self.params,
                k,
                #[inline(always)]
                |a_panel, b_view, a_block, b_base| {
                    let first = a_block * MR;
                    let valid_rows = (rows - first).min(MR);
                    let mut visitor = Visitor::<A, MR, K> {
                        arch,
                        a: a_panel,
                        rows: Rows {
                            c: &mut c[first * width..(first + valid_rows) * width],
                            kth: &mut kth[first..first + MR],
                            a_norms: if metric == Metric::L2 {
                                &a_norms[first..first + MR]
                            } else {
                                &[]
                            },
                            metric,
                            width,
                            valid: valid_rows,
                        },
                        b_norms,
                        b_base,
                        k,
                    };
                    if let Some(tail) = b_view.visit_panels::<NR>(k, &mut visitor) {
                        let start = tail.start();
                        if let Some(panel) = tail.try_as_panel::<1>() {
                            visitor.visit(panel, start);
                        }
                        if NR == 6 {
                            if let Some(panel) = tail.try_as_panel::<2>() {
                                visitor.visit(panel, start);
                            }
                            if let Some(panel) = tail.try_as_panel::<3>() {
                                visitor.visit(panel, start);
                            }
                            if let Some(panel) = tail.try_as_panel::<4>() {
                                visitor.visit(panel, start);
                            }
                            if let Some(panel) = tail.try_as_panel::<5>() {
                                visitor.visit(panel, start);
                            }
                        }
                    }
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::multi_vector::BlockTransposed;
    use diskann_vector::{Norm, norm::FastL2NormSquared};

    #[cfg(target_arch = "aarch64")]
    use diskann_wide::arch::aarch64::Neon;
    #[cfg(target_arch = "x86_64")]
    use diskann_wide::arch::x86_64::{V3, V4};

    fn check_top_k_blocks<A: Kernel<MR>, const MR: usize, const NR: usize>(arch: A) {
        let rows = 2 * MR + 1;
        let dim = 17;
        let count = 19;
        let q: Vec<_> = (0..rows * dim)
            .map(|i| 18.0 + (i / dim) as f32 / 32.0)
            .collect();
        let d: Vec<_> = (0..count * dim).map(|i| (i / dim) as f32).collect();
        let q_view = diskann_utils::views::rowmajor::Ref::try_from_data(&q[..], rows, dim).unwrap();
        let d_view =
            diskann_utils::views::rowmajor::Ref::try_from_data(&d[..], count, dim).unwrap();
        let packed = BlockTransposed::<f32, MR>::from_matrix_view(q_view);
        let mut q_norms: Vec<f32> = q
            .chunks_exact(dim)
            .map(|r| FastL2NormSquared.evaluate(r))
            .collect();
        q_norms.resize(packed.padded_nrows(), 0.0);
        let d_norms: Vec<f32> = d
            .chunks_exact(dim)
            .map(|r| FastL2NormSquared.evaluate(r))
            .collect();
        for metric in [Metric::InnerProduct, Metric::L2] {
            for width in [1, 2, 3, 4, 8, 10, 21] {
                for b_block in [1, 5, 6, 7] {
                    let mut output = vec![Candidate::EMPTY; rows * width];
                    let mut kth = vec![f32::INFINITY; packed.padded_nrows()];
                    let k = DimK::new(NonZeroUsize::new(dim).unwrap());
                    // SAFETY: Views come from checked storage; norms/thresholds are padded,
                    // output has rows * width slots, and count fits the candidate ID range.
                    let mut driver = unsafe {
                        Driver::<A, MR, NR>::new(
                            arch,
                            packed::View::from_block_transposed(packed.as_view()).unwrap(),
                            unpacked::View::from_matrix_view(d_view).unwrap(),
                            Rows::new(&mut output, &mut kth, &q_norms, metric, width),
                            &d_norms,
                            k,
                            Params {
                                a_panels_in_l2: NonZeroUsize::new(1).unwrap(),
                                b_cols_in_l1: NonZeroUsize::new(b_block).unwrap(),
                            },
                        )
                    };
                    driver::Drive::drive(&mut driver);
                    for (i, row) in output.chunks_exact(width).enumerate() {
                        let x = f64::from(q[i * dim]);
                        let mut expected: Vec<_> = (0..count)
                            .map(|id| {
                                let y = id as f64;
                                let distance = if metric == Metric::L2 {
                                    dim as f64 * (x - y).powi(2)
                                } else {
                                    -(dim as f64 * x * y)
                                };
                                Candidate::new(id as u32, distance as f32)
                            })
                            .collect();
                        expected.sort_by(|a, b| a.distance.total_cmp(&b.distance));
                        expected.truncate(width);
                        expected.resize(width, Candidate::EMPTY);
                        assert_eq!(row, expected, "row={i}, k={width}, B block={b_block}");
                    }
                }
            }
        }
    }

    #[test]
    fn top_k_accumulates_across_cache_blocks_with_global_ids() {
        check_top_k_blocks::<_, 8, 2>(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = V3::new_checked() {
                check_top_k_blocks::<_, 16, 6>(arch);
            }
            if let Some(arch) = V4::new_checked() {
                check_top_k_blocks::<_, 32, 6>(arch);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = Neon::new_checked() {
            check_top_k_blocks::<_, 8, 6>(arch);
        }
    }

    #[test]
    fn top_k_working_set_arithmetic_is_checked() {
        let cache = Cache::detect();
        assert!(params(cache, 17, 10, 32, 6, Metric::L2).is_some());
        assert!(params(cache, usize::MAX, 1, 32, 6, Metric::L2).is_none());
        assert!(params(cache, 1, usize::MAX, 32, 6, Metric::InnerProduct).is_none());
        assert!(params(cache, isize::MAX as usize / 4, 1, 32, 6, Metric::L2).is_none());
    }
}
