/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{
    strided::StridedView,
    views::{MatrixView, MutMatrixView},
};
use diskann_wide::{SIMDMulAdd, SIMDSumTree, SIMDVector};

/// Compute the squared L2 norm of the argument.
pub(crate) fn square_norm(x: &[f32]) -> f32 {
    let px: *const f32 = x.as_ptr();
    let len = x.len();

    diskann_wide::alias!(f32s = f32x8);

    let mut i = 0;
    let mut s = f32s::default(diskann_wide::ARCH);

    // The number of 32-bit blocks over the underlying slice.
    if i + 32 <= len {
        let mut s0 = f32s::default(diskann_wide::ARCH);
        let mut s1 = f32s::default(diskann_wide::ARCH);
        let mut s2 = f32s::default(diskann_wide::ARCH);
        let mut s3 = f32s::default(diskann_wide::ARCH);
        while i + 32 <= len {
            // SAFETY: The memory range `[i, i + 32)` is valid by the loop bounds.
            let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i)) };
            s0 = vx.mul_add_simd(vx, s0);

            // SAFETY: The memory range `[i, i + 32)` is valid by the loop bounds.
            let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i + 8)) };
            s1 = vx.mul_add_simd(vx, s1);

            // SAFETY: The memory range `[i, i + 32)` is valid by the loop bounds.
            let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i + 16)) };
            s2 = vx.mul_add_simd(vx, s2);

            // SAFETY: The memory range `[i, i + 32)` is valid by the loop bounds.
            let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i + 24)) };
            s3 = vx.mul_add_simd(vx, s3);

            i += 32;
        }

        s = (s0 + s1) + (s2 + s3)
    }

    while i + 8 <= len {
        // SAFETY: The memory range `[i, i + 8)` is valid by the loop bounds.
        let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i)) };
        s = vx.mul_add_simd(vx, s);
        i += 8;
    }

    let remainder = len - i;
    if remainder != 0 {
        // SAFETY: The pointer add is valid because `i < len` (strict inequality), so the
        // base pointer belongs to the memory owned by `x`.
        //
        // Furthermore, the load is valid for the first `remainder` items.
        let vx = unsafe { f32s::load_simd_first(diskann_wide::ARCH, px.add(i), remainder) };
        s = vx.mul_add_simd(vx, s);
    }

    s.sum_tree()
}

////////////////////
// BlockTranspose //
////////////////////

/// A representation of 2D data consisting of blocks of tranposes.
///
/// The generic parameter `N` denotes how many rows are in a block.
///
/// For example, if the original data is in a row major layout like the following:
/// ```text
/// a0 a1 a2 a3 ... aK
/// b0 b1 b2 b3 ... bK
/// c0 c1 c2 c3 ... cK
/// d0 d1 d2 d3 ... dK
/// e0 e1 e2 e3 ... eK
/// ```
/// and the blocking parameter `N = 3`, then the blocked-transpose layout (still row major)
/// will be as follows:
/// ```text
///           Group Size (3)
///            <---------->
///
///            +----------+    ^
///            | a0 b0 c0 |    |
///            | a1 b1 c1 |    |
///            | a2 b2 c2 |    | Block Size (K + 1)
///  Block 0   | a3 b3 c3 |    |
///  (Full)    | ...      |    |
///            | aK bK cK |    |
///            +----------+    v
///            +----------+
///            | d0 e0 XX |
///            | d1 e1 XX |
///            | d2 e2 XX |
///  Block 1   | d3 e3 XX |
///  (Partial) | ...      |
///            | dK eK XX |
///            +----------+
/// ```
/// Note the following characteristics:
///
/// * The same dimension of different source rows are store contiguously (this helps with
///   SIMD algorithms).
///
/// * Subsequent groups of the following dimensions are also stored contiguously.
///
/// * Blocks are stored contiguously so all the entire `BlockTranspose` consists of a single
///   allocation.
///
/// * Allocation is done at a block-level of granularity with the last block only partially
///   filled if the number of rows does not evenly divide the block-size.
///
///   Padding is done as indicated in the diagram. SIMD algorithms are free to load entire
///   rows provided any bookkeeping tracks the partially full status.
#[derive(Debug)]
pub struct BlockTranspose<const N: usize> {
    data: Box<[f32]>,
    block_size: usize,
    /// How many blocks are completely filled with data.
    full_blocks: usize,
    /// The total number of data rows stored in this representation.
    nrows: usize,
}

impl<const N: usize> BlockTranspose<N> {
    /// Construct a new `BlockTranspose` sized to contain a matrix of size `nrows x ncols`.
    ///
    /// Data will be zero initialized.
    pub fn new_matrix(nrows: usize, ncols: usize) -> Self {
        let block_size = ncols;
        let full_blocks = nrows / N;
        let remainder = nrows - full_blocks * N;

        let num_blocks = if remainder == 0 {
            full_blocks
        } else {
            full_blocks + 1
        };

        Self {
            data: vec![0.0; N * block_size * num_blocks].into(),
            block_size,
            full_blocks,
            nrows,
        }
    }

    /// Return the number of rows of data stored in `self`.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Return the number of columns in each rows of the data in `self`.
    pub fn ncols(&self) -> usize {
        self.block_size
    }

    /// Return the number of **physical** rows in each data block.
    ///
    /// Conceptually, this is the same as the number of columns.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return the length of each row in a block.
    ///
    /// Conceptually, this is the number of source-data rows stored in a block.
    pub fn group_size(&self) -> usize {
        N
    }

    /// Return the length of each row in a block.
    pub const fn const_group_size() -> usize {
        N
    }

    /// Return the number of completely full data blocks.
    ///
    /// This will always be equal to `self.nrows() / self.group_size()`.
    pub fn full_blocks(&self) -> usize {
        self.full_blocks
    }

    /// Return the total number of data blocks including any partially full terminal block.
    ///
    /// This will always be equal to:
    /// `crate::utils::div_round_up(self.nrows(), self.group_size())`
    pub fn num_blocks(&self) -> usize {
        if self.remainder() == 0 {
            self.full_blocks()
        } else {
            self.full_blocks() + 1
        }
    }

    /// Return the number of elements in the last partially full block.
    ///
    /// A return value of 0 indicates that all blocks are full.
    pub fn remainder(&self) -> usize {
        self.nrows % N
    }

    /// Return a pointer to the beginning of `block`.
    ///
    /// The caller may assume that for the returned pointer `ptr`,
    /// `[ptr, ptr + self.block_stride())` points to valid memory, even for the remainder
    /// block.
    ///
    /// # Safety
    ///
    /// Block must be in-bounds (i.e., `block < self.num_blocks()`).
    pub unsafe fn block_ptr_unchecked(&self, block: usize) -> *const f32 {
        debug_assert!(block < self.num_blocks());
        // SAFETY: If we assume `block < self.num_blocks()`, then
        //
        // 1. Our base pointer was allocated in the first place, so this computed offset
        //    must fit within an `isize`.
        // 2. This pointer (and an offset `self.block_stride()`) higher all live within
        //    a single allocated object.
        self.data.as_ptr().add(self.block_offset(block))
    }

    /// Return a pointer to the start of data segment.
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// The linear offset of the beginning of `block`.
    fn block_offset(&self, block: usize) -> usize {
        self.block_stride() * block
    }

    /// The number of elements of type `f32` in each block (i.e., the spacing between the
    /// starts of blocks).
    fn block_stride(&self) -> usize {
        N * self.block_size
    }

    /// Return a view over a full `block`.
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.full_blocks()`.
    #[allow(clippy::expect_used)]
    pub fn block(&self, block: usize) -> MatrixView<'_, f32> {
        assert!(block < self.full_blocks());
        let offset = self.block_offset(block);
        let stride = self.block_stride();
        let block_size = self.block_size();
        MatrixView::try_from(&self.data[offset..offset + stride], block_size, N)
            .expect("base data should have been sized correctly")
    }

    /// Return a view over the remainder block, or `None` if there is no remainder block.
    #[allow(clippy::expect_used)]
    pub fn remainder_block(&self) -> Option<MatrixView<'_, f32>> {
        if self.remainder() == 0 {
            None
        } else {
            let offset = self.block_offset(self.full_blocks());
            let stride = self.block_stride();
            Some(
                MatrixView::try_from(&self.data[offset..offset + stride], self.block_size, N)
                    .expect("base data should have been sized correctly"),
            )
        }
    }

    /// Return a mutable view over a full `block`.
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.full_blocks()`.
    #[allow(clippy::expect_used)]
    pub fn block_mut(&mut self, block: usize) -> MutMatrixView<'_, f32> {
        assert!(block < self.full_blocks());
        let offset = self.block_offset(block);
        let stride = self.block_stride();
        let block_size = self.block_size();
        MutMatrixView::try_from(&mut self.data[offset..offset + stride], block_size, N)
            .expect("base data should have been sized correctly")
    }

    /// Return a mutable view over the remainder block, or `None` if there is no remainder block.
    #[allow(clippy::expect_used)]
    pub fn remainder_block_mut(&mut self) -> Option<MutMatrixView<'_, f32>> {
        if self.remainder() == 0 {
            None
        } else {
            let offset = self.block_offset(self.full_blocks());
            let stride = self.block_stride();
            let block_size = self.block_size();
            Some(
                MutMatrixView::try_from(&mut self.data[offset..offset + stride], block_size, N)
                    .expect("base data should have been sized correctly"),
            )
        }
    }

    //////////////////
    // Constructors //
    //////////////////

    /// Construct a copy of `v` inside a BlockTranspose.
    pub fn from_strided(v: StridedView<'_, f32>) -> Self {
        let mut data = BlockTranspose::<N>::new_matrix(v.nrows(), v.ncols());

        // Pack full blocks
        let full_blocks = data.full_blocks();
        for block_index in 0..full_blocks {
            let mut block = data.block_mut(block_index);
            for col in 0..v.ncols() {
                for row in 0..N {
                    block[(col, row)] = v[(N * block_index + row, col)]
                }
            }
        }

        // Pack remainder
        let remaining_rows = data.remainder();
        if let Some(mut block) = data.remainder_block_mut() {
            for col in 0..v.ncols() {
                for row in 0..remaining_rows {
                    block[(col, row)] = v[(N * full_blocks + row, col)]
                }
            }
        }

        data
    }

    /// Construct a copy of `v` inside a BlockTranspose.
    pub fn from_matrix_view(v: MatrixView<'_, f32>) -> Self {
        Self::from_strided(v.into())
    }
}

impl<const N: usize> std::ops::Index<(usize, usize)> for BlockTranspose<N> {
    type Output = f32;

    /// Return a reference the the element at the logical `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.nrows()` or `col >= self.ncols()`.
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        let block = row / N;
        let offset = row % N;
        &self.data[self.block_offset(block) + col * N + offset]
    }
}

#[cfg(test)]
mod tests {
    use diskann_utils::{lazy_format, views::Matrix};
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        Rng, SeedableRng,
    };

    use super::*;
    use crate::utils::div_round_up;

    /////////////////
    // Square Norm //
    /////////////////

    fn square_norm_reference(x: &[f32]) -> f32 {
        x.iter().map(|&i| i * i).sum()
    }

    fn test_square_norm_impl<R: Rng>(
        dim: usize,
        ntrials: usize,
        relative_error: f32,
        absolute_error: f32,
        rng: &mut R,
    ) {
        let distribution = Uniform::<f32>::new(-1.0, 1.0).unwrap();
        let mut x: Vec<f32> = vec![0.0; dim];
        for trial in 0..ntrials {
            x.iter_mut().for_each(|i| *i = distribution.sample(rng));
            let expected = square_norm_reference(&x);
            let got = square_norm(&x);

            let this_absolute_error = (expected - got).abs();
            let this_relative_error = this_absolute_error / expected.abs();

            let absolute_ok = this_absolute_error <= absolute_error;
            let relative_ok = this_relative_error <= relative_error;

            if !absolute_ok && !relative_ok {
                panic!(
                    "recieved abolute/relative errors of {}/{} when the bounds were {}/{}\n\
                     dim = {}, trial = {} of {}",
                    this_absolute_error,
                    this_relative_error,
                    absolute_error,
                    relative_error,
                    dim,
                    trial,
                    ntrials,
                )
            }
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const NTRIALS: usize = 1;
            const MAX_DIM: usize = 80;
        } else {
            const NTRIALS: usize = 100;
            const MAX_DIM: usize = 128;
        }
    }

    #[test]
    fn test_square_norm() {
        let mut rng = StdRng::seed_from_u64(0x71d00ad8c7105273);
        for dim in 0..MAX_DIM {
            let relative_error = 8.0e-7;
            let absolute_error = 1.0e-5;

            test_square_norm_impl(dim, NTRIALS, relative_error, absolute_error, &mut rng);
        }
    }

    /////////////////////
    // Block Transpose //
    /////////////////////

    fn test_block_transpose<const N: usize>(nrows: usize, ncols: usize) {
        let context = lazy_format!("N = {}, nrows = {}, ncols = {}", N, nrows, ncols);

        // Create initial data with the following layout:
        //       0         1         2 ...   ncols-1
        //   ncols   ncols+1   ncols+2     2*ncols-1
        // 2*ncols 2*ncols+1 2*ncols+2     3*ncols-1
        // ...
        let mut data = Matrix::new(0.0, nrows, ncols);
        data.as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, d)| *d = i as f32);

        let mut transpose = BlockTranspose::<N>::from_matrix_view(data.as_view());

        // Make sure the public methods return their advertised methods.
        assert_eq!(transpose.nrows(), nrows, "{}", context);
        assert_eq!(transpose.ncols(), ncols, "{}", context);
        assert_eq!(transpose.block_size(), ncols, "{}", context);
        assert_eq!(transpose.group_size(), N, "{}", context);
        assert_eq!(transpose.full_blocks(), nrows / N, "{}", context);
        assert_eq!(
            transpose.num_blocks(),
            div_round_up(nrows, N),
            "{}",
            context
        );
        assert_eq!(transpose.remainder(), nrows % N, "{}", context);

        // Check regular row-column based indexing.
        for row in 0..nrows {
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    transpose[(row, col)],
                    "failed for (row, col) = ({}, {})",
                    row,
                    col
                );
            }
        }

        // Check indexing on the block level.
        for b in 0..transpose.full_blocks() {
            let block = transpose.block(b);
            assert_eq!(block.nrows(), ncols);
            assert_eq!(block.ncols(), N);

            // Are the contents correct?
            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    assert_eq!(
                        block[(i, j)],
                        data[(N * b + j, i)],
                        "failed in block {}, row {}, col {} -- {}",
                        b,
                        i,
                        j,
                        context
                    );
                }
            }

            // Make sure the pointer API is correct.
            // SAFETY: The loop bounds above ensure `b < transpose.num_blocks()`.
            let ptr = unsafe { transpose.block_ptr_unchecked(b) };
            assert_eq!(ptr, block.as_slice().as_ptr());

            // Construct a mutable version and zero it.
            let mut block_mut = transpose.block_mut(b);
            assert_eq!(ptr, block_mut.as_slice().as_ptr());
            assert_eq!(block_mut.nrows(), ncols);
            assert_eq!(block_mut.ncols(), N);
            block_mut.as_mut_slice().fill(0.0);
        }

        let expected_remainder = nrows % N;
        if expected_remainder != 0 {
            let b = transpose.full_blocks();
            let block = transpose.remainder_block().unwrap();
            assert_eq!(block.nrows(), ncols);
            assert_eq!(block.ncols(), N);

            // Are the contents correct?
            for i in 0..block.nrows() {
                for j in 0..expected_remainder {
                    assert_eq!(
                        block[(i, j)],
                        data[(N * b + j, i)],
                        "failed in block {}, row {}, col {} -- {}",
                        b,
                        i,
                        j,
                        context
                    );
                }
            }

            // Make sure the pointer API is correct.
            // SAFETY: The loop bounds above ensure `b < transpose.num_blocks()`.
            let ptr = unsafe { transpose.block_ptr_unchecked(b) };
            assert_eq!(ptr, block.as_slice().as_ptr());

            // Construct a mutable version and zero it.
            let mut block_mut = transpose.remainder_block_mut().unwrap();
            assert_eq!(ptr, block_mut.as_slice().as_ptr());
            assert_eq!(block_mut.nrows(), ncols);
            assert_eq!(block_mut.ncols(), N);
            block_mut.as_mut_slice().fill(0.0);
        } else {
            assert!(transpose.remainder_block().is_none());
            assert!(transpose.remainder_block_mut().is_none());
        }

        // Check that the inner state is now zeroed.
        assert!(transpose.data.iter().all(|i| *i == 0.0));
    }

    #[test]
    fn test_block_transpose_16() {
        let row_range = if cfg!(miri) { 127..128 } else { 0..128 };
        let column_range = if cfg!(miri) { 4..5 } else { 0..5 };

        for nrows in row_range {
            for ncols in column_range.clone() {
                test_block_transpose::<16>(nrows, ncols);
            }
        }
    }

    #[test]
    fn test_block_transpose_8() {
        let row_range = if cfg!(miri) { 127..128 } else { 0..128 };
        let column_range = if cfg!(miri) { 4..5 } else { 0..5 };

        for nrows in row_range {
            for ncols in column_range.clone() {
                test_block_transpose::<8>(nrows, ncols);
            }
        }
    }
}
