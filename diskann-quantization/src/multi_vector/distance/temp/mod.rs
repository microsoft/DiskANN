/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Gist: Currently, this file just implements the kernel for operations on an `A-Tile` and
//! a `B-Tile`. A more end-to-end solution will also need to iterate over tiles and construct
//! this tile iteration so that A-Tiles fit within the L2 cache and B-Tiles fit within the L1
//! cache.
//!
//!

use std::marker::PhantomData;

use diskann_wide::{ARCH, SIMDMulAdd, SIMDMinMax, SIMDVector, arch::x86_64::V3};

use crate::{
    alloc::{AllocatorCore, Poly},
    bits::{Dynamic, Length, Static},
    multi_vector::{BlockTransposedRef, MatRef, Standard},
};

pub fn test_function(
    a: BlockTransposedRef<'_, f32, 16>,
    b: MatRef<'_, Standard<f32>>,
    acc: &mut DotAcc<16, 4, crate::alloc::ScopedAllocator<'_>>,
    buf: &mut [f32],
) {
    assert_eq!(a.ncols(), b.vector_dim());
    assert_eq!(buf.len(), a.nrows());

    let mut callback = |acc: &mut DotAcc<_, _, _>, index: APanelIndex| {
        let lower = 16 * index.value();
        let upper = (lower + 16).min(a.nrows());
        let count = upper - lower;

        buf[lower..upper].copy_from_slice(&acc.max_sim(ARCH, b.num_vectors())[..count]);
    };

    tile_loop::<
        V3,
        Dot,
        BlockTransposedRef<'_, f32, 16>,
        MatRef<'_, Standard<f32>>,
        DotAcc<16, 4, crate::alloc::ScopedAllocator<'_>>,
    >(ARCH, Dot, a, b, acc, a.ncols(), &mut callback)
}

///////////////
// Iterators //
///////////////

pub(crate) trait TailIterator: ExactSizeIterator {
    type Tail;
    fn tail(self) -> Option<Self::Tail>;
}

/// Some iterators are constructed such that a tail element is impossible (e.g. via padding).
///
/// In these situations, [`NoTail`] can be used to dispatch to no-op micro-kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) enum NoTail {}

impl Iterator for NoTail {
    type Item = NoTail;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl ExactSizeIterator for NoTail {}

impl TailIterator for NoTail {
    type Tail = Self;
    fn tail(self) -> Option<Self> {
        None
    }
}

pub(crate) trait AsTailIterator<'a> {
    type Item;

    type Tail;

    type Iter: TailIterator<Item = Self::Item, Tail = Self::Tail>;

    fn as_tail_iter(&'a mut self) -> Self::Iter;
}

///////////////////
// TileIteration //
///////////////////

/// Perform the following logical dimensional partition:
/// ```text
/// +-----------------+      +-----------------+      +---------+
/// | b00 b01 b02 b03 |      | b04 b05 b06 b07 |      | b08 b09 |
/// | b10 b11 b12 b13 |  +-->| b14 b15 b16 b17 |  +-->| b18 b19 |
/// | b20 b21 b22 b23 |  |   | b25 b25 b26 b27 |  |   | b29 b29 |
/// | b31 b31 b32 b33 |  |   | b34 b35 b36 b37 |  |   | b38 b39 |
/// +-----------------+  |   +-----------------+  |   +---------+
///          |           |            |           |        |
///          V           |            V           |        V
/// +-----------------+  |   +-----------------+  |   +---------+
/// | b40 b41 b42 b43 |  |   | b44 b45 b46 b47 |  |   | b48 b49 |
/// | b50 b51 b52 b53 |  |   | b54 b55 b56 b57 |  |   | b58 b59 |
/// | b60 b61 b62 b63 |  |   | b64 b65 b66 b67 |  |   | b68 b69 |
/// | b70 b71 b72 b73 |  |   | b74 b75 b76 b77 |  |   | b78 b79 |
/// +-----------------+  |   +-----------------+  |   +---------+
///          |           |            |           |        |
///          V           |            V           |        V
/// +-----------------+  |   +-----------------+  |   +---------+
/// | b80 b81 b82 b83 |  |   | b84 b85 b86 b87 |  |   | b88 b89 |
/// | b90 b91 b92 b93 |--+   | b94 b95 b96 b97 |--+   | b98 b99 |
/// +-----------------+      +-----------------+      +---------+
/// ```
pub(crate) trait ColumnPaneled {
    type Panel: Copy;
    type PanelTail: Copy;

    type ColumnKIter: TailIterator<Item = Self::Panel, Tail = Self::PanelTail>;
    type Iter: ExactSizeIterator<Item = Self::ColumnKIter>;

    fn column_paneled(&self, k: usize) -> Self::Iter;
}

/// Perform the following logical dimensional partition:
/// ```text
/// +-----------------+      +-----------------+      +---------+
/// | a00 a01 a02 a03 |      | a04 a05 a06 a07 |      | a08 a09 |
/// | a10 a11 a12 a13 |----->| a14 a15 a16 a17 |----->| a18 a19 |
/// | a20 a21 a22 a23 |      | a25 a25 a26 a27 |      | a29 a29 |
/// | a31 a31 a32 a33 |      | a34 a35 a36 a37 |      | a38 a39 |
/// +-----------------+      +-----------------+      +---------+
///                                                        |
///          +---------------------------------------------+
///          V
/// +-----------------+      +-----------------+      +---------+
/// | a40 a41 a42 a43 |      | a44 a45 a46 a47 |      | a48 a49 |
/// | a50 a51 a52 a53 |----->| a54 a55 a56 a57 |----->| a58 a59 |
/// | a60 a61 a62 a63 |      | a64 a65 a66 a67 |      | a68 a69 |
/// | a70 a71 a72 a73 |      | a74 a75 a76 a77 |      | a78 a79 |
/// +-----------------+      +-----------------+      +---------+
///                                                        |
///          +---------------------------------------------+
///          V
/// +-----------------+      +-----------------+      +---------+
/// | a80 a81 a82 a83 |----->| a84 a85 a86 a87 |----->| a88 a89 |
/// | a90 a91 a92 a93 |      | a94 a95 a96 a97 |      | a98 a99 |
/// +-----------------+      +-----------------+      +---------+
/// ```
pub(crate) trait RowPaneled {
    type Panel: Copy;

    type PanelTail: Copy;

    type RowKIter: ExactSizeIterator<Item = Self::Panel>;

    type TailRowKIter: ExactSizeIterator<Item = Self::PanelTail>;

    type Iter: TailIterator<Item = Self::RowKIter, Tail = Self::TailRowKIter>;

    fn row_paneled(&self, k: usize) -> Self::Iter;
}

//////////////////
// Accumulation //
//////////////////

pub(crate) trait Accumulator<Arch, A, B>: Copy {
    type Item<'a>;
    type Tail<'a>;

    type Accumulator: for<'a> AsTailIterator<'a, Item = Self::Item<'a>, Tail = Self::Tail<'a>>;

    fn accumulator(&self, a: &A, b: &B) -> Self::Accumulator;
}

pub(crate) trait Kernel<Arch, A, B, Acc> {
    fn op<const OVERWRITE: bool>(&self, arch: Arch, a: A, b: B, accumulator: Acc);
}

//////////
// Loop //
//////////

// ///////////
// // Tiled //
// ///////////
//
// pub(crate) trait Tile<'a, Bound = &'a Self> {
//     type Item: Copy;
// }
//
// pub(crate) trait Tiled: for<'a> Tile<'a> {
//     fn next(&mut self) -> Option<<Self as Tile<'_>>::Item>;
// }
//
// pub(crate) trait AsTiled<'a> {
//     type Tiled: Tiled;
//     fn as_tiled(&'a mut self) -> Self::Tiled;
// }
//
// fn do_the_thing<Arch, K, A, B, Acc>(
//     arch: Arch,
//     kernel: K,
//     mut a: A,
//     mut b: B,
//     split: usize,
//     acc: &mut Acc,
// ) where
//     A: Tiled,
//     B: for<'a> AsTiled<'a>,
//     K: Copy,
//     Arch: Copy,
//     GenericBarrier: for<'a, 'b, 'c> LoopNest<
//             Arch,
//             K,
//             <A as Tile<'a>>::Item,
//             <<B as AsTiled<'b>>::Tiled as Tile<'c>>::Item,
//             Acc,
//         >,
// {
//     let barrier = GenericBarrier::new(split);
//
//     while let Some(a_tile) = a.next() {
//         let mut b_tiled = b.as_tiled();
//         while let Some(b_tile) = b_tiled.next() {
//             barrier.loop_nest(arch, kernel, a_tile, b_tile, acc)
//         }
//     }
// }
//
// pub(crate) trait LoopNest<Arch, K, A, B, Acc>: Copy {
//     fn loop_nest(self, arch: Arch, kernel: K, a: A, b: B, acc: &mut Acc);
// }
//
// #[derive(Debug, Clone, Copy)]
// struct GenericBarrier {
//     fracture: usize,
// }
//
// impl GenericBarrier {
//     pub(crate) fn new(fracture: usize) -> Self {
//         Self { fracture }
//     }
// }
//
// impl<Arch, K, A, B, Acc> LoopNest<Arch, K, A, B, Acc> for GenericBarrier
// where
//     A: IterateAcrossRows,
//     B: ColumnPaneled,
//     Acc: for<'a> AsTailIterator<'a>,
//     K: for<'b> Kernel<Arch, A::Block, B::Panel, <Acc as AsTailIterator<'b>>::Item>
//         + for<'b> Kernel<Arch, A::Block, B::PanelTail, <Acc as AsTailIterator<'b>>::Tail>
//         + for<'b> Kernel<Arch, A::TailBlock, B::Panel, <Acc as AsTailIterator<'b>>::Item>
//         + for<'b> Kernel<Arch, A::TailBlock, B::PanelTail, <Acc as AsTailIterator<'b>>::Tail>
//         + Copy,
//     Arch: Copy,
// {
//     #[inline(always)]
//     fn loop_nest(self, arch: Arch, kernel: K, a: A, b: B, acc: &mut Acc) {
//         tile_loop(arch, kernel, a, b, acc, self.fracture)
//     }
// }

/// The `apanel` within the greater tile.
///
/// This is a number beginning at zero that increments by one for every item yielded
/// from [`RowPaneled::Iter`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct APanelIndex(usize);

impl APanelIndex {
    /// The base for of the "A" multi-vector.
    pub(crate) fn value(&self) -> usize {
        self.0
    }
}

#[inline(always)]
fn tile_loop<Arch, K, A, B, Acc>(
    arch: Arch,
    kernel: K,
    a: A,
    b: B,
    acc: &mut Acc,
    sz: usize,
    finish: &mut dyn FnMut(&mut Acc, APanelIndex),
)
where
    A: RowPaneled,
    B: ColumnPaneled,
    Acc: for<'a> AsTailIterator<'a>,
    K: for<'b> Kernel<Arch, A::Panel, B::Panel, <Acc as AsTailIterator<'b>>::Item>
        + for<'b> Kernel<Arch, A::Panel, B::PanelTail, <Acc as AsTailIterator<'b>>::Tail>
        + for<'b> Kernel<Arch, A::PanelTail, B::Panel, <Acc as AsTailIterator<'b>>::Item>
        + for<'b> Kernel<Arch, A::PanelTail, B::PanelTail, <Acc as AsTailIterator<'b>>::Tail>
        + Copy,
    Arch: Copy,
{
    let mut a = a.row_paneled(sz);

    let mut i = 0;
    for mut a_blocks in a.by_ref() {
        let mut b = b.column_paneled(sz);
        if let (Some(a_block), Some(b_cols)) = (a_blocks.next(), b.next()) {
            run_kernel::<true, _, _, _, _, _>(arch, kernel, a_block, b_cols, acc);
        }

        for (a_block, b_cols) in std::iter::zip(a_blocks, b) {
            run_kernel::<false, _, _, _, _, _>(arch, kernel, a_block, b_cols, acc);
        }

        finish(acc, APanelIndex(i));
        i += 1;
    }

    if let Some(mut a_blocks) = a.tail() {
        let mut b = b.column_paneled(sz);

        if let (Some(a_block), Some(b_cols)) = (a_blocks.next(), b.next()) {
            run_kernel::<true, _, _, _, _, _>(arch, kernel, a_block, b_cols, acc);
        }

        for (a_block, b_cols) in std::iter::zip(a_blocks, b) {
            run_kernel::<false, _, _, _, _, _>(arch, kernel, a_block, b_cols, acc);
        }

        finish(acc, APanelIndex(i));
    }
}

#[inline(always)]
fn run_kernel<'a, const OVERWRITE: bool, Arch, K, A, B, Acc>(
    arch: Arch,
    kernel: K,
    a: A,
    mut b: B,
    acc: &'a mut Acc,
) where
    A: Copy,
    B: TailIterator,
    Acc: AsTailIterator<'a>,
    K: Kernel<Arch, A, B::Item, Acc::Item> + Kernel<Arch, A, B::Tail, Acc::Tail>,
    Arch: Copy,
{
    let mut acc = acc.as_tail_iter();

    for (b, acc) in std::iter::zip(b.by_ref(), acc.by_ref()) {
        kernel.op::<OVERWRITE>(arch, a, b, acc);
    }

    match (b.tail(), acc.tail()) {
        (Some(b_tail), Some(acc_tail)) => {
            kernel.op::<OVERWRITE>(arch, a, b_tail, acc_tail);
        }
        (None, None) => {}
        _ => panic!("desync"),
    }
}

////////////////////
// BlockTranspose //
////////////////////

impl<'a, const N: usize> RowPaneled for BlockTransposedRef<'a, f32, N, 1> {
    type Panel = BlockTransposePanel<'a, N>;
    type PanelTail = NoTail;
    type RowKIter = BlockTransposeRowIter<'a, N>;
    type TailRowKIter = NoTail;
    type Iter = BlockTransposeIter<'a, N>;

    fn row_paneled(&self, chunk_size: usize) -> Self::Iter {
        debug_assert!(chunk_size <= self.ncols());

        let as_slice = self.as_slice();

        BlockTransposeIter {
            ptr: as_slice.as_ptr(),
            end: unsafe { as_slice.as_ptr().add(as_slice.len()) },
            ncols: self.ncols(),
            nsubcols: chunk_size,
            _lifetime: PhantomData,
        }
    }
}

/// An iterator over blocks within an overall transpose.
///
/// # Safety Invariants
///
/// The region of memory in `[ptr, end)` must always be valid.
#[derive(Debug)]
pub(crate) struct BlockTransposeIter<'a, const N: usize> {
    /// The base of the region of memory.
    ptr: *const f32,

    /// The block transpose iterator requires that the number of rows is a multiple of the
    /// block size `N`.
    ///
    /// The pointer `end` points to one-past the last element.
    end: *const f32,

    /// The number of columns in each row.
    ncols: usize,

    /// The number of columns to process at a time in the micro-kernel.
    nsubcols: usize,

    /// The lifetime of the borrow.
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, const N: usize> BlockTransposeIter<'a, N> {
    /// The number of remaining elements.
    fn remaining_elements(&self) -> usize {
        // SAFETY: We maintain the invariant that `self.ptr` is always less then `self.end`.
        unsafe { self.end.offset_from_unsigned(self.ptr) }
    }

    /// The number of elements in each block.
    fn block_stride(&self) -> usize {
        N * self.ncols
    }
}

impl<'a, const N: usize> Iterator for BlockTransposeIter<'a, N> {
    type Item = BlockTransposeRowIter<'a, N>;
    fn next(&mut self) -> Option<Self::Item> {
        let block_stride = self.block_stride();

        // If there is space for a
        if self.remaining_elements() >= block_stride {
            let next = unsafe { self.ptr.add(block_stride) };
            let iter = BlockTransposeRowIter {
                ptr: self.ptr,
                end: next,
                nsubcols: self.nsubcols,
                _lifetime: PhantomData,
            };

            self.ptr = next;

            Some(iter)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_elements() / self.block_stride();
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for BlockTransposeIter<'_, N> {}

impl<const N: usize> TailIterator for BlockTransposeIter<'_, N> {
    type Tail = NoTail;
    fn tail(self) -> Option<Self::Tail> {
        None
    }
}

/// The block transpose row-iter partitions a "Block" in the parent transpose into multiple
/// `BlockTransposePanels` - traversing across the entire block.
#[derive(Debug)]
pub(crate) struct BlockTransposeRowIter<'a, const N: usize> {
    /// The base pointer for this block.
    ptr: *const f32,

    /// One past the end of this block.
    end: *const f32,

    /// The number of columns within each panel.
    nsubcols: usize,

    _lifetime: PhantomData<&'a ()>,
}

impl<'a, const N: usize> BlockTransposeRowIter<'a, N> {
    fn remaining_cols(&self) -> usize {
        // SAFETY: We maintain the invariant that `self.end >= self.ptr`.
        (unsafe { self.end.offset_from_unsigned(self.ptr) }) / N
    }

    /// Return the number of rows contained within each yielded [`BlockTransposePanel`].
    fn nrows(&self) -> usize {
        N
    }
}

impl<'a, const N: usize> Iterator for BlockTransposeRowIter<'a, N> {
    type Item = BlockTransposePanel<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        let remaining_cols = self.remaining_cols();
        let nsubcols = remaining_cols.min(self.nsubcols);

        if nsubcols == 0 {
            None
        } else {
            let next = unsafe { self.ptr.add(nsubcols * N) };
            let panel = BlockTransposePanel {
                ptr: self.ptr,
                nsubcols,
                _lifetime: PhantomData,
            };

            self.ptr = next;
            Some(panel)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_cols().div_ceil(self.nsubcols);
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for BlockTransposeRowIter<'_, N> {}

/// A block transpose panel consists of potentially a subset of columns (called "subcols")
/// of a block in the overall matrix.
///
/// It's *very* important to note that the number of subcolumns may be *less* than the total
/// number of columns in the matrix.
///
/// ```text
/// +-------------------------------------------+
/// | a00 a10 a20 a30 ... aN0 --+               |
/// | a01 a11 a21 a31 ... aN1   +--- "subcols"  |
/// | a02 a12 a22 a32 ... aN2 --+               |  Block 0
/// | ...                 ...                   |
/// | a0K a1K a2K a3K ... aNK                   |
/// +-------------------------------------------+
/// | b00 b01 b02 b03 ... b0N                   |
/// | b10 b11 b12 b13 ... b13                   |
/// | b20 b21 b22 b23 ... b2N                   |  Block 1
/// | ...                 ...                   |
/// | bK0 bK1 bK2 bK3 ... bKN                   |
/// +-------------------------------------------+
///  ...
/// ```
///
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockTransposePanel<'a, const N: usize> {
    /// The base pointer of the panel.
    ptr: *const f32,

    /// The number of columns in this panel.
    nsubcols: usize,

    _lifetime: PhantomData<&'a ()>,
}

impl<const N: usize> BlockTransposePanel<'_, N> {
    fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row < N && col < self.nsubcols {
            Some(unsafe { self.ptr.add(col * N + row).read() })
        } else {
            None
        }
    }

    fn nrows(&self) -> usize {
        N
    }

    fn ncols(&self) -> usize {
        self.nsubcols
    }
}

//////////////
// RowMajor //
//////////////

impl<'a> ColumnPaneled for MatRef<'a, Standard<f32>> {
    type Panel = RowMajorPanel<'a, 4, Static<4>>;
    type PanelTail = RowMajorPanel<'a, 4, Dynamic>;
    type ColumnKIter = RowMajorColumnIter<'a, 4>;
    type Iter = RowMajorIter<'a, 4>;

    fn column_paneled(&self, chunk_size: usize) -> Self::Iter {
        RowMajorIter {
            ptr: self.as_ptr(),
            remaining_cols: self.repr().ncols(),
            nrows: self.repr().nrows(),
            ncols: self.repr().ncols(),
            nsubcols: chunk_size,
            _lifetime: PhantomData,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RowMajorIter<'a, const N: usize> {
    ptr: *const f32,
    remaining_cols: usize,
    nrows: usize,
    ncols: usize,
    nsubcols: usize,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, const N: usize> Iterator for RowMajorIter<'a, N> {
    type Item = RowMajorColumnIter<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_cols == 0 {
            None
        } else {
            let cols = self.nsubcols.min(self.remaining_cols);
            let ptr = self.ptr;
            self.remaining_cols -= cols;
            self.ptr = unsafe { self.ptr.add(cols) };

            let iter = RowMajorColumnIter {
                ptr,
                remaining_rows: self.nrows,
                nsubcols: cols,
                rowstride: self.ncols,
                _lifetime: PhantomData,
            };

            Some(iter)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_cols.div_ceil(self.nsubcols);
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for RowMajorIter<'_, N> {}

/// A "down-column" iterator for a row-major matrix.
///
/// Not necessarily ideal from a memory-efficiency stand-point, but works as a proof-of-concept.
///
/// The parameter `N` is the number of rows belonging to the main blocks.
///
/// ```text
/// a00 a01 a02 a03 a04 .... a0K
/// a10 a11 a12 a13 a14 .... a1K
/// a20 a21 a22 a23 a24 .... a2K
///
/// ```
#[derive(Debug)]
pub(crate) struct RowMajorColumnIter<'a, const N: usize> {
    /// The current base pointer.
    ptr: *const f32,
    remaining_rows: usize,
    nsubcols: usize,
    rowstride: usize,
    _lifetime: PhantomData<&'a ()>,
}

impl<const N: usize> RowMajorColumnIter<'_, N> {
    /// Return the number of columns spanned by this iterator.
    ///
    /// Note that this may just be a subset of the columns in the parent matrix.
    fn panel_cols(&self) -> usize {
        self.nsubcols
    }

    /// Return the number of rows contained in each full panel.
    fn panel_rows(&self) -> usize {
        N
    }
}

impl<'a, const N: usize> Iterator for RowMajorColumnIter<'a, N> {
    type Item = RowMajorPanel<'a, N, Static<N>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_rows >= N {
            let panel = RowMajorPanel {
                ptr: self.ptr,
                nsubcols: self.nsubcols,
                rowstride: self.rowstride,
                nrows: Static,
                _lifetime: PhantomData,
            };

            // Because the base pointer for the column iterator is potentially offset from
            // the base of the allocation, we can run into issues with `ptr::add` when the
            // number of rows is evenly divisible by `N`.
            //
            // Since the validity of the iterator is tracked via `self.remaining_rows`, we
            // use `wrapping_add` here.
            self.ptr = self.ptr.wrapping_add(N * self.rowstride);
            self.remaining_rows -= N;

            Some(panel)
        } else {
            None
        }
    }
}

impl<const N: usize> ExactSizeIterator for RowMajorColumnIter<'_, N> {}

impl<'a, const N: usize> TailIterator for RowMajorColumnIter<'a, N> {
    type Tail = RowMajorPanel<'a, N, Dynamic>;
    fn tail(self) -> Option<Self::Tail> {
        debug_assert!(
            self.remaining_rows < N,
            "tail iterator was consumed too early"
        );

        let nrows = self.remaining_rows.min(N.saturating_sub(1));
        if nrows == 0 {
            None
        } else {
            let panel = RowMajorPanel {
                ptr: self.ptr,
                nsubcols: self.nsubcols,
                rowstride: self.rowstride,
                nrows: Dynamic(nrows),
                _lifetime: PhantomData,
            };

            Some(panel)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RowMajorPanel<'a, const N: usize, L>
where
    L: Length,
{
    ptr: *const f32,
    nsubcols: usize,
    rowstride: usize,
    nrows: L,
    _lifetime: PhantomData<&'a ()>,
}

impl<const N: usize, L> RowMajorPanel<'_, N, L>
where
    L: Length,
{
    fn rowstride(&self) -> usize {
        self.rowstride
    }

    fn nrows(&self) -> usize {
        self.nrows.value()
    }

    fn ncols(&self) -> usize {
        self.nsubcols
    }

    fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row < self.nrows.value() && col < self.nsubcols {
            Some(unsafe { self.ptr.add(self.rowstride * row + col).read() })
        } else {
            None
        }
    }
}

//////////////////
// Accumulators //
//////////////////

#[derive(Debug)]
pub(crate) struct DotAcc<const AROWS: usize, const BROWS: usize, A>
where
    A: AllocatorCore,
{
    data: Poly<[f32], A>,
    rows: usize,
}

impl<const BROWS: usize, A> DotAcc<16, BROWS, A>
where
    A: AllocatorCore,
{
    fn max_sim(&self, arch: V3, maxrows: usize) -> [f32; 16] {
        assert!(maxrows <= self.rows);
        let (chunks, []) = self.data.as_chunks::<16>() else {
            panic!("invalid!");
        };

        diskann_wide::alias!(f32s = <V3>::f32x16);
        let mut acc = f32s::splat(arch, f32::NEG_INFINITY);

        for c in chunks.iter().take(maxrows) {
            let this = f32s::from_array(arch, *c);
            acc = acc.max_simd(this);
        }

        (f32s::default(arch) - acc).to_array()
    }
}

impl<'a, const AROWS: usize, const BROWS: usize, A> AsTailIterator<'a> for DotAcc<AROWS, BROWS, A>
where
    A: AllocatorCore,
{
    type Item = DotAccBlock<'a, AROWS, BROWS, Static<BROWS>>;
    type Tail = DotAccBlock<'a, AROWS, BROWS, Dynamic>;

    type Iter = DotAccIter<'a, AROWS, BROWS>;

    fn as_tail_iter(&'a mut self) -> Self::Iter {
        DotAccIter {
            ptr: self.data.as_mut_ptr(),
            remaining_rows: self.rows,
            _lifetime: PhantomData,
        }
    }
}

#[derive(Debug)]
pub(crate) struct DotAccIter<'a, const AROWS: usize, const BROWS: usize> {
    ptr: *mut f32,
    remaining_rows: usize,
    _lifetime: PhantomData<&'a mut ()>,
}

#[derive(Debug)]
pub(crate) struct DotAccBlock<'a, const AROWS: usize, const BROWS: usize, L: Length> {
    ptr: *mut f32,
    active_rows: L,
    _lifetime: PhantomData<&'a mut ()>,
}

impl<'a, const AROWS: usize, const BROWS: usize> Iterator for DotAccIter<'a, AROWS, BROWS> {
    type Item = DotAccBlock<'a, AROWS, BROWS, Static<BROWS>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_rows >= BROWS {
            let block = DotAccBlock {
                ptr: self.ptr,
                active_rows: Static,
                _lifetime: PhantomData,
            };

            self.ptr = unsafe { self.ptr.add(BROWS * AROWS) };
            self.remaining_rows -= BROWS;

            Some(block)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_rows.div_ceil(BROWS);
        (remaining, Some(remaining))
    }
}

impl<const AROWS: usize, const BROWS: usize> ExactSizeIterator for DotAccIter<'_, AROWS, BROWS> {}

impl<'a, const AROWS: usize, const BROWS: usize> TailIterator for DotAccIter<'a, AROWS, BROWS> {
    type Tail = DotAccBlock<'a, AROWS, BROWS, Dynamic>;
    fn tail(self) -> Option<Self::Tail> {
        debug_assert!(self.remaining_rows < BROWS);

        if self.remaining_rows == 0 {
            None
        } else {
            let block = DotAccBlock {
                ptr: self.ptr,
                active_rows: Dynamic(self.remaining_rows.min(BROWS.saturating_sub(1))),
                _lifetime: PhantomData,
            };
            Some(block)
        }
    }
}

/////////////
// Kernels //
/////////////

#[derive(Debug, Clone, Copy)]
struct Dot;

impl Kernel<V3, NoTail, RowMajorPanel<'_, 4, Static<4>>, DotAccBlock<'_, 16, 4, Static<4>>>
    for Dot
{
    #[inline(always)]
    fn op<const OVERWRITE: bool>(
        &self,
        _arch: V3,
        _a: NoTail,
        _b: RowMajorPanel<'_, 4, Static<4>>,
        _accumulator: DotAccBlock<'_, 16, 4, Static<4>>,
    ) {
    }
}

impl Kernel<V3, NoTail, RowMajorPanel<'_, 4, Dynamic>, DotAccBlock<'_, 16, 4, Dynamic>> for Dot {
    #[inline(always)]
    fn op<const OVERWRITE: bool>(
        &self,
        _arch: V3,
        _a: NoTail,
        _b: RowMajorPanel<'_, 4, Dynamic>,
        _accumulator: DotAccBlock<'_, 16, 4, Dynamic>,
    ) {
    }
}

impl
    Kernel<
        V3,
        BlockTransposePanel<'_, 16>,
        RowMajorPanel<'_, 4, Static<4>>,
        DotAccBlock<'_, 16, 4, Static<4>>,
    > for Dot
{
    #[inline(always)]
    fn op<const OVERWRITE: bool>(
        &self,
        arch: V3,
        a: BlockTransposePanel<'_, 16>,
        b: RowMajorPanel<'_, 4, Static<4>>,
        accumulator: DotAccBlock<'_, 16, 4, Static<4>>,
    ) {
        debug_assert_eq!(a.nsubcols, b.nsubcols);
        unsafe {
            microkernel::<OVERWRITE, 4>(
                arch,
                a.ptr,
                b.ptr,
                b.rowstride(),
                a.nsubcols,
                accumulator.ptr,
            )
        };
    }
}

impl
    Kernel<
        V3,
        BlockTransposePanel<'_, 16>,
        RowMajorPanel<'_, 4, Dynamic>,
        DotAccBlock<'_, 16, 4, Dynamic>,
    > for Dot
{
    #[inline(always)]
    fn op<const OVERWRITE: bool>(
        &self,
        arch: V3,
        a: BlockTransposePanel<'_, 16>,
        b: RowMajorPanel<'_, 4, Dynamic>,
        accumulator: DotAccBlock<'_, 16, 4, Dynamic>,
    ) {
        if cfg!(debug_assertions) {
            assert_eq!(a.nsubcols, b.nsubcols);
            assert_eq!(b.nrows.value(), accumulator.active_rows.value());
            assert!(b.nrows.value() < 4);
        }

        let subcols = a.nsubcols.min(b.nsubcols);
        let nrows = b.nrows.value().min(accumulator.active_rows.value());

        match nrows {
            0 => {}
            1 => unsafe {
                microkernel::<OVERWRITE, 1>(
                    arch,
                    a.ptr,
                    b.ptr,
                    b.rowstride(),
                    subcols,
                    accumulator.ptr,
                )
            },
            2 => unsafe {
                microkernel::<OVERWRITE, 2>(
                    arch,
                    a.ptr,
                    b.ptr,
                    b.rowstride(),
                    subcols,
                    accumulator.ptr,
                )
            },
            _ => unsafe {
                microkernel::<OVERWRITE, 3>(
                    arch,
                    a.ptr,
                    b.ptr,
                    b.rowstride(),
                    subcols,
                    accumulator.ptr,
                )
            },
        }
    }
}

#[inline(always)]
unsafe fn microkernel<const OVERWRITE: bool, const UNROLL: usize>(
    arch: V3,
    a: *const f32,
    b: *const f32,
    b_row_stride: usize,
    k: usize,
    buf: *mut f32,
) {
    diskann_wide::alias!(f32s = <V3>::f32x8);

    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| b_row_stride * i);

    let a_stride = 2 * f32s::LANES;
    let a_stride_half = f32s::LANES;

    for i in 0..k {
        unsafe {
            let a0 = f32s::load_simd(arch, a.add(a_stride * i));
            let a1 = f32s::load_simd(arch, a.add(a_stride * i + a_stride_half));

            for j in 0..UNROLL {
                let bj = f32s::splat(arch, b.add(i + offsets[j]).read_unaligned());
                p0[j] = a0.mul_add_simd(bj, p0[j]);
                p1[j] = a1.mul_add_simd(bj, p1[j]);
            }
        }
    }

    const BUF_STRIDE: usize = 16;
    for j in 0..UNROLL {
        if const { OVERWRITE } {
            unsafe {
                p0[j].store_simd(buf.add(j * BUF_STRIDE));
                p1[j].store_simd(buf.add(j * BUF_STRIDE + a_stride_half));
            }
        } else {
            let acc0 = unsafe { f32s::load_simd(arch, buf.add(j * BUF_STRIDE)) };
            let acc1 = unsafe { f32s::load_simd(arch, buf.add(j * BUF_STRIDE + a_stride_half)) };

            unsafe {
                (acc0 + p0[j]).store_simd(buf.add(j * BUF_STRIDE));
                (acc1 + p1[j]).store_simd(buf.add(j * BUF_STRIDE + a_stride_half));
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_utils::views::{Init, Matrix};

    use crate::multi_vector::{BlockTransposed, Mat};

    //----------------//
    // BlockTranspose //
    //----------------//

    fn test_row_paneled_inner(nrows: usize, ncols: usize) {
        let raw = {
            let mut i = 0;
            Matrix::<f32>::new(
                Init(|| {
                    let j = i;
                    i += 1;
                    j as f32
                }),
                nrows,
                ncols,
            )
        };

        let block_transpose = BlockTransposed::<f32, 16>::from_matrix_view(raw.as_view());

        let ks = [
            1,
            ncols.div_ceil(3),
            ncols.div_ceil(2),
            ncols.saturating_sub(1),
            ncols,
        ];
        for k in ks {
            if k == 0 {
                continue;
            }

            let mut itr = block_transpose.as_view().row_paneled(k);

            let mut base_row = 0;
            for mut row_k_iter in itr.by_ref() {
                let mut base_col = 0;
                for panel in row_k_iter.by_ref() {
                    let subcols = panel.ncols();
                    assert!(subcols <= k);
                    assert_eq!(panel.nrows(), 16);

                    for c in 0..panel.ncols() {
                        for r in 0..panel.nrows() {
                            // Make sure the data access is vaild.
                            let got = panel.get(r, c).unwrap();

                            // Don't index out-of-bounds
                            if base_row + r >= nrows {
                                break;
                            }

                            let expected = raw[(base_row + r, base_col + c)];
                            assert_eq!(got, expected);
                        }
                    }

                    base_col += subcols;
                }

                assert_eq!(base_col, ncols);

                base_row += row_k_iter.nrows();
            }

            assert!(itr.tail().is_none());
            assert_eq!(base_row, nrows.next_multiple_of(16));
        }
    }

    #[test]
    fn test_row_paneled() {
        for nrows in [1, 2, 4, 8, 10, 16, 20, 32, 40] {
            for ncols in [1, 2, 4, 7, 10] {
                test_row_paneled_inner(nrows, ncols);
            }
        }
    }

    //----------//
    // RowMajor //
    //----------//

    fn test_row_major_iter_inner(nrows: usize, ncols: usize) {
        println!("nrows = {nrows}, ncols = {ncols}");
        let mat = {
            let mut i = 0;
            Mat::from_fn(Standard::<f32>::new(nrows, ncols).unwrap(), || {
                let j = i;
                i += 1;
                j as f32
            })
        };

        let ks = [
            1,
            ncols.div_ceil(3),
            ncols.div_ceil(2),
            ncols.saturating_sub(1),
            ncols,
        ];

        for k in ks {
            if k == 0 {
                continue;
            }

            let itr = mat.as_view().column_paneled(k);

            let mut base_col = 0;
            for mut col_k_iter in itr {
                let panel_rows = col_k_iter.panel_rows();
                let panel_cols = col_k_iter.panel_cols();

                let expected_cols = if base_col + k > ncols {
                    ncols - base_col
                } else {
                    k
                };

                assert_eq!(panel_cols, expected_cols);

                let mut base_row = 0;
                for panel in col_k_iter.by_ref() {
                    assert_eq!(panel.ncols(), expected_cols);
                    for r in 0..panel.nrows() {
                        for c in 0..panel.ncols() {
                            let got = panel.get(r, c).unwrap();
                            let expected = mat.get_row(base_row + r).unwrap()[base_col + c];
                            assert_eq!(got, expected);
                        }
                    }

                    base_row += panel.nrows();
                }

                let remainder_expected = !nrows.is_multiple_of(panel_rows);

                match (remainder_expected, col_k_iter.tail()) {
                    (true, Some(panel)) => {
                        assert_eq!(panel.ncols(), expected_cols);
                        for r in 0..panel.nrows() {
                            for c in 0..panel.ncols() {
                                let got = panel.get(r, c).unwrap();
                                let expected = mat.get_row(base_row + r).unwrap()[base_col + c];
                                assert_eq!(got, expected);
                            }
                        }

                        base_row += panel.nrows();
                    }
                    (false, None) => {}
                    (true, None) => panic!("No remainder was yielded when one was expected"),
                    (false, Some(_)) => panic!("A remainder was yielded when none was expected"),
                }

                assert_eq!(base_row, nrows);
                base_col += panel_cols;
            }

            assert_eq!(base_col, ncols);
        }
    }

    #[test]
    fn test_row_major_paneled() {
        for nrows in [1, 2, 4, 8, 10, 16, 20, 32, 40] {
            for ncols in [1, 2, 4, 7, 10] {
                test_row_major_iter_inner(nrows, ncols);
            }
        }
    }
}
