/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Consume complete dot tiles into sorted nearest sets instead of max reductions.
//! Contiguous cutoffs permit SIMD rejection without reading strided candidate rows.

use std::num::NonZeroUsize;

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::topk::{Candidate, insert_sorted};
use diskann_wide::{Architecture, SIMDFloat, SIMDMask, arch::Scalar};

diskann_wide::alias!(f32x4<A> = f32x4);
diskann_wide::alias!(f32x8<A> = f32x8);
diskann_wide::alias!(f32x16<A> = f32x16);

use crate::{
    matrix_kernels::{
        Cache,
        blocks::{packed, unpacked},
        maxsim::packed_f32_x_unpacked_f32::{ExtraWide, Params, dot_tile, visit_tiles},
        num::{Bytes, DimK},
    },
    multi_vector::BlockTransposedRef,
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Scoring<N> {
    InnerProduct,
    L2(N),
}

#[derive(Clone, Copy)]
pub(crate) struct Norms<'a> {
    pub a: &'a [f32],
    pub b: &'a [f32],
}

impl Scoring<Norms<'_>> {
    #[inline]
    pub(crate) fn distance(self, i: usize, j: usize, dot: f32) -> f32 {
        match self {
            Self::InnerProduct => -dot,
            Self::L2(n) => (n.a[i] + n.b[j]) - 2.0 * dot,
        }
    }

    pub(crate) fn params<const MR: usize, const NR: usize>(
        self,
        cache: Cache,
        dim: usize,
        width: usize,
    ) -> Option<Params> {
        let norm = if matches!(self, Self::L2(_)) {
            size_of::<f32>()
        } else {
            0
        };
        let codes = dim.checked_mul(size_of::<f32>())?;
        let row = codes
            .checked_add(width.checked_mul(size_of::<Candidate>())?)?
            .checked_add(size_of::<f32>())?
            .checked_add(norm)?;
        Params::try_new(
            cache,
            Bytes::new(MR.checked_mul(row)?),
            Bytes::new(codes.checked_add(norm)?),
            NR,
        )
    }
}

pub(crate) trait Kernel<const MR: usize>: ExtraWide<MR> + Architecture {
    fn offer(
        self,
        column: Self::Wide,
        rows: &mut SelectionRows<'_>,
        score: Scoring<(&[f32], f32)>,
        id: u32,
    );
}

/// Candidate rows and their contiguous cutoffs are initialized and updated together.
/// Subviews retain a whole number of nonempty rows and padding only in cutoffs.
/// Each logical cutoff equals its candidate row's last distance; padded cutoffs stay infinite.
pub(crate) struct SelectionRows<'a> {
    candidates: &'a mut [Candidate],
    cutoffs: &'a mut [f32],
    width: NonZeroUsize,
    valid: usize,
}

impl<'a> SelectionRows<'a> {
    #[expect(
        clippy::expect_used,
        reason = "a nonempty TopK scan has positive width"
    )]
    fn new(mut output: rowmajor::Mut<'a, Candidate>, cutoffs: &'a mut [f32]) -> Self {
        let width = NonZeroUsize::new(output.ncols()).expect("nonempty TopK scan");
        let valid = output.nrows();
        assert!(cutoffs.len() >= valid, "cutoffs cover all logical rows");
        output.as_mut_slice().fill(Candidate::EMPTY);
        cutoffs.fill(f32::INFINITY);
        Self {
            candidates: output.into_mut_slice(),
            cutoffs,
            width,
            valid,
        }
    }

    #[inline(always)]
    fn sub_rows(&mut self, first: usize, padded_rows: usize) -> SelectionRows<'_> {
        let valid = (self.valid - first).min(padded_rows);
        let width = self.width.get();
        SelectionRows {
            candidates: &mut self.candidates[first * width..(first + valid) * width],
            cutoffs: &mut self.cutoffs[first..first + padded_rows],
            width: self.width,
            valid,
        }
    }

    #[inline(always)]
    fn insert(&mut self, lane: usize, candidate: Candidate) {
        let width = self.width.get();
        let row = &mut self.candidates[lane * width..(lane + 1) * width];
        self.cutoffs[lane] = insert_sorted(row, candidate);
    }
}

#[inline(always)]
fn offer<V, const N: usize>(
    arch: V::Arch,
    dot: V,
    rows: &mut SelectionRows<'_>,
    score: Scoring<(&[f32], f32)>,
    id: u32,
) where
    V: SIMDFloat<Scalar = f32, ConstLanes = diskann_wide::Const<N>>,
    diskann_wide::Const<N>: diskann_wide::SupportedLaneCount,
    u64: From<<<V::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    const { assert!(N > 0 && N <= u64::BITS as usize) };
    assert_eq!(rows.cutoffs.len(), N);
    // SAFETY: sub_rows provides exactly N initialized cutoffs, including padded lanes.
    let thresholds = unsafe { V::load_simd(arch, rows.cutoffs.as_ptr()) };
    let distances = match score {
        Scoring::InnerProduct => V::splat(arch, -1.0) * dot,
        Scoring::L2((norms, b)) => {
            assert_eq!(norms.len(), N);
            // SAFETY: the checked norm slice covers all N lanes, including padding.
            let a = unsafe { V::load_simd(arch, norms.as_ptr()) };
            (a + V::splat(arch, b)) - V::splat(arch, 2.0) * dot
        }
    };
    let mut mask = u64::from(distances.lt_simd(thresholds).bitmask().to_underlying())
        & (u64::MAX >> (u64::BITS as usize - rows.valid));
    if mask == 0 {
        return;
    }
    let distances = distances.to_array();
    while mask != 0 {
        let lane = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        rows.insert(lane, Candidate::new(id, distances[lane]));
    }
}

macro_rules! kernel {
    ($arch:ty, $mr:literal, $lanes:literal, $vector:ty) => {
        impl Kernel<$mr> for $arch {
            #[inline(always)]
            fn offer(
                self,
                column: Self::Wide,
                rows: &mut SelectionRows<'_>,
                score: Scoring<(&[f32], f32)>,
                id: u32,
            ) {
                for (group, dot) in column.into_iter().enumerate() {
                    let first = group * $lanes;
                    if first >= rows.valid {
                        break;
                    }
                    let mut group_rows = rows.sub_rows(first, $lanes);
                    let score = match score {
                        Scoring::InnerProduct => Scoring::InnerProduct,
                        Scoring::L2((a, b)) => Scoring::L2((&a[first..first + $lanes], b)),
                    };
                    offer::<$vector, $lanes>(self, dot, &mut group_rows, score, id);
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

/// The output is a nonempty-width view of the logical rows in this A panel.
struct Tile<'a, A, const MR: usize> {
    arch: A,
    a: packed::Panel<'a, f32, MR>,
    selection: SelectionRows<'a>,
    score: Scoring<Norms<'a>>,
    first_b: usize,
    dim: DimK,
}

impl<A: Kernel<MR>, const MR: usize, const NR: usize> unpacked::PanelVisitor<f32, NR>
    for &mut Tile<'_, A, MR>
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, f32, NR>, start: usize) {
        // SAFETY: compute derives both panels from matrices with the same checked dim.
        let dots = unsafe { dot_tile(self.arch, self.a, b, self.dim) };
        let first_b = self.first_b + start;
        for (j, column) in dots.into_iter().enumerate() {
            let id = first_b + j;
            let score = match self.score {
                Scoring::InnerProduct => Scoring::InnerProduct,
                Scoring::L2(norms) => Scoring::L2((norms.a, norms.b[id])),
            };
            self.arch
                .offer(column, &mut self.selection, score, id as u32);
        }
    }
}

/// Nonempty scan. Check the logical shapes before erasing dimensions in the panel views.
#[expect(
    clippy::expect_used,
    reason = "nonempty checked matrices and in-bounds panel ranges"
)]
pub(crate) fn compute<A: Kernel<MR>, const MR: usize, const NR: usize>(
    arch: A,
    a: BlockTransposedRef<'_, f32, MR>,
    b: rowmajor::Ref<'_, f32>,
    output: rowmajor::Mut<'_, Candidate>,
    score: Scoring<Norms<'_>>,
    params: Params,
) {
    const { assert!(NR != 0) };
    assert_eq!(a.ncols(), b.ncols());
    assert_eq!(a.nrows(), output.nrows());
    assert_eq!(output.nrows().div_ceil(MR), a.num_blocks());
    assert!(b.nrows() <= u32::MAX as usize);
    if let Scoring::L2(norms) = score {
        assert_eq!(norms.a.len(), a.nrows(), "one norm per A row");
        assert_eq!(norms.b.len(), b.nrows(), "one norm per B row");
    }
    let dim = DimK::new(NonZeroUsize::new(a.ncols()).expect("nonempty dimension"));
    let padded_norms;
    let score = match score {
        Scoring::L2(norms) if norms.a.len() != a.padded_nrows() => {
            padded_norms = {
                let mut padded = norms.a.to_vec();
                padded.resize(a.padded_nrows(), 0.0);
                padded
            };
            Scoring::L2(Norms {
                a: &padded_norms,
                b: norms.b,
            })
        }
        _ => score,
    };
    let mut cutoffs = vec![0.0; a.padded_nrows()];
    let mut selection = SelectionRows::new(output, &mut cutoffs);
    let a = packed::View::from_block_transposed(a).expect("nonempty query");
    let b = unpacked::View::from_matrix_view(b).expect("nonempty documents");

    arch.run(
        #[inline(always)]
        || {
            // SAFETY: dim is the common contraction dimension of a and b. All visited
            // subviews (including the remainder) preserve it. Only logical rows reach selection.
            unsafe {
                visit_tiles(
                    a,
                    b,
                    params,
                    dim,
                    #[inline(always)]
                    |a, b, block, first_b| {
                        let first_a = block * MR;
                        let score = match score {
                            Scoring::InnerProduct => Scoring::InnerProduct,
                            Scoring::L2(norms) => Scoring::L2(Norms {
                                a: &norms.a[first_a..first_a + MR],
                                b: norms.b,
                            }),
                        };
                        let mut tile = Tile {
                            arch,
                            a,
                            selection: selection.sub_rows(first_a, MR),
                            score,
                            first_b,
                            dim,
                        };
                        if let Some(tail) = b.visit_panels::<NR>(dim, &mut tile) {
                            tile.first_b += tail.start();
                            let complete =
                                tail.into_view().visit_panels::<1>(dim, &mut tile).is_none();
                            debug_assert!(complete);
                        }
                    },
                );
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::BlockTransposed;
    use diskann_vector::{Norm, norm::FastL2NormSquared};
    use diskann_wide::arch::Scalar;

    #[test]
    fn selection_subviews_keep_cutoffs_in_sync_and_ignore_padding() {
        let mut output = rowmajor::Owned::from_element(5, 2, Candidate::new(99, -10.0));
        let mut cutoffs = [0.0; 8];
        {
            let mut rows = SelectionRows::new(output.as_view_mut(), &mut cutoffs);
            {
                let mut tail = rows.sub_rows(4, 4);
                tail.insert(0, Candidate::new(1, 3.0));
                tail.insert(0, Candidate::new(2, 1.0));
                tail.insert(0, Candidate::new(3, 2.0));
                assert_eq!(
                    tail.cutoffs,
                    &[2.0, f32::INFINITY, f32::INFINITY, f32::INFINITY]
                );
                assert_eq!(
                    tail.candidates,
                    &[Candidate::new(2, 1.0), Candidate::new(3, 2.0)]
                );
            }
            assert!(rows.candidates[..8].iter().all(|c| *c == Candidate::EMPTY));
            let mut first = rows.sub_rows(0, 4);
            first.insert(1, Candidate::new(4, -1.0));
            assert_eq!(first.cutoffs[1], f32::INFINITY);
        }
        assert_eq!(cutoffs[4], 2.0);
        assert!(cutoffs[..4].iter().all(|&x| x == f32::INFINITY));
    }

    fn check_blocks<A: Kernel<MR>, const MR: usize, const NR: usize>(arch: A) {
        let rows = 2 * MR + 1;
        let dim = if cfg!(miri) { 2 } else { 17 };
        let count = 19;
        let q = rowmajor::Owned::from_fn(rows, dim, |pos| 18.0 + pos.row as f32 / 32.0);
        let d = rowmajor::Owned::from_fn(count, dim, |pos| pos.row as f32);
        let packed = BlockTransposed::<f32, MR>::try_from_matrix_view(q.as_view()).unwrap();
        let norms = |row: &[f32]| FastL2NormSquared.evaluate(row);
        let a: Vec<_> = q.rows().map(norms).collect();
        let b: Vec<_> = d.rows().map(norms).collect();
        // Under the interpreter retain a partial A panel, B remainders and multiple B blocks.
        let (widths, blocks): (&[usize], &[usize]) = if cfg!(miri) {
            (&[3], &[7])
        } else {
            (&[1, 2, 3, 4, 8, 10, 21], &[1, 5, 6, 7])
        };
        for score in [Scoring::InnerProduct, Scoring::L2(Norms { a: &a, b: &b })] {
            for &width in widths {
                for &b_block in blocks {
                    let mut out = rowmajor::Owned::from_element(rows, width, Candidate::EMPTY);
                    compute::<_, MR, NR>(
                        arch,
                        packed.as_view(),
                        d.as_view(),
                        out.as_view_mut(),
                        score,
                        Params {
                            a_panels_in_l2: NonZeroUsize::new(1).unwrap(),
                            b_cols_in_l1: NonZeroUsize::new(b_block).unwrap(),
                        },
                    );
                    for (i, row) in out.rows().enumerate() {
                        let x = f64::from(q.row(i)[0]);
                        let mut expected: Vec<_> = (0..count)
                            .map(|j| {
                                let y = j as f64;
                                let dist = match score {
                                    Scoring::InnerProduct => -(dim as f64 * x * y),
                                    Scoring::L2(_) => dim as f64 * (x - y).powi(2),
                                };
                                Candidate::new(j as u32, dist as f32)
                            })
                            .collect();
                        expected.sort_by(|a, b| a.distance.total_cmp(&b.distance));
                        expected.resize(width, Candidate::EMPTY);
                        assert_eq!(row, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn top_k_accumulates_across_cache_blocks_with_global_ids() {
        check_blocks::<_, 8, 2>(Scalar::new());
        // A non-production width exercises multi-column remainders independently of ISA.
        check_blocks::<_, 8, 4>(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_blocks::<_, 16, 6>(arch);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
                check_blocks::<_, 32, 6>(arch);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_blocks::<_, 8, 6>(arch);
        }
    }

    #[test]
    #[should_panic(expected = "one norm per A row")]
    fn top_k_rejects_missing_norm_for_partial_panel() {
        let a = BlockTransposed::<f32, 8>::try_new(9, 1).unwrap();
        let b = rowmajor::Ref::column_vector(&[1.0f32]);
        let mut output = rowmajor::Owned::from_element(9, 2, Candidate::EMPTY);
        let score = Scoring::L2(Norms {
            a: &[0.0; 8],
            b: &[1.0],
        });
        let params = score.params::<8, 2>(Cache::detect(), 1, 2).unwrap();
        compute::<_, 8, 2>(
            Scalar::new(),
            a.as_view(),
            b,
            output.as_view_mut(),
            score,
            params,
        );
    }

    #[test]
    fn top_k_working_set_arithmetic_is_checked() {
        let cache = Cache::detect();
        let score = Scoring::InnerProduct;
        assert!(score.params::<32, 6>(cache, 17, 10).is_some());
        assert!(score.params::<1, 1>(cache, usize::MAX, 1).is_none());
        assert!(score.params::<1, 1>(cache, 1, usize::MAX).is_none());
        assert!(
            score
                .params::<1, 1>(cache, isize::MAX as usize / 4, 1)
                .is_none()
        );
    }
}
