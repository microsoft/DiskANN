/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Consume complete dot tiles into sorted nearest sets instead of max reductions.

use std::num::NonZeroUsize;

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::topk::{Candidate, insert_sorted};
use diskann_wide::{Architecture, SIMDVector};

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
    fn unpack(column: Self::Wide) -> [f32; MR];
}

macro_rules! kernel {
    ($arch:ty, $mr:literal) => {
        impl Kernel<$mr> for $arch {
            #[inline(always)]
            fn unpack(column: Self::Wide) -> [f32; $mr] {
                let groups = column.map(SIMDVector::to_array);
                std::array::from_fn(|i| groups.as_flattened()[i])
            }
        }
    };
}

kernel!(diskann_wide::arch::Scalar, 8);
#[cfg(target_arch = "x86_64")]
kernel!(diskann_wide::arch::x86_64::V3, 16);
#[cfg(target_arch = "x86_64")]
kernel!(diskann_wide::arch::x86_64::V4, 32);
#[cfg(target_arch = "aarch64")]
kernel!(diskann_wide::arch::aarch64::Neon, 8);

/// The output is a nonempty-width view of the logical rows in this A panel.
struct Tile<'a, A, const MR: usize> {
    arch: A,
    a: packed::Panel<'a, f32, MR>,
    output: rowmajor::Mut<'a, Candidate>,
    width: NonZeroUsize,
    score: Scoring<Norms<'a>>,
    first_a: usize,
    first_b: usize,
    dim: DimK,
}

impl<A: Kernel<MR>, const MR: usize, const NR: usize> unpacked::PanelVisitor<f32, NR>
    for &mut Tile<'_, A, MR>
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, f32, NR>, start: usize) {
        // SAFETY: compute derives both panels from matrices with the same checked dim.
        let dots = unsafe { dot_tile(self.arch, self.a, b, self.dim) }.map(A::unpack);
        let last = self.width.get() - 1;
        let first_b = self.first_b + start;
        for (i, row) in self.output.rows_mut().enumerate() {
            let mut kth = row[last].distance;
            for (j, col) in dots.iter().enumerate() {
                let id = first_b + j;
                let distance = self.score.distance(self.first_a + i, id, col[i]);
                if distance < kth {
                    kth = insert_sorted(row, Candidate::new(id as u32, distance));
                }
            }
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
    mut output: rowmajor::Mut<'_, Candidate>,
    score: Scoring<Norms<'_>>,
    params: Params,
) {
    const { assert!(NR != 0) };
    assert_eq!(a.ncols(), b.ncols());
    assert_eq!(a.nrows(), output.nrows());
    assert_eq!(output.nrows().div_ceil(MR), a.num_blocks());
    let width = NonZeroUsize::new(output.ncols()).expect("nonempty TopK scan");
    assert!(b.nrows() <= u32::MAX as usize);
    if let Scoring::L2(norms) = score {
        assert_eq!(norms.a.len(), a.nrows(), "one norm per A row");
        assert_eq!(norms.b.len(), b.nrows(), "one norm per B row");
    }
    let dim = DimK::new(NonZeroUsize::new(a.ncols()).expect("nonempty dimension"));
    let a = packed::View::from_block_transposed(a).expect("nonempty query");
    let b = unpacked::View::from_matrix_view(b).expect("nonempty documents");

    output.as_mut_slice().fill(Candidate::EMPTY);
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
                        let end = (first_a + MR).min(output.nrows());
                        let mut tile = Tile {
                            arch,
                            a,
                            // A row subview preserves the parent's exact, nonzero width.
                            output: output.subview_mut(first_a..end).expect("valid A panel"),
                            width,
                            score,
                            first_a,
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
