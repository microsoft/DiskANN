/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::marker::PhantomData;

use diskann_wide::{SIMDMulAdd, SIMDVector, arch::x86_64::V3};

use crate::{
    alloc::{AllocatorCore, Poly},
    bits::{Dynamic, Length, Static},
    multi_vector::{BlockTransposedRef, MatRef, Standard},
};

pub fn test_function(
    a: BlockTransposedRef<'_, f32, 16>,
    b: MatRef<'_, Standard<f32>>,
    acc: &mut DotAcc<16, 4, crate::alloc::GlobalAllocator>,
) {
    assert_eq!(a.ncols(), b.vector_dim());

    do_loop::<
        V3,
        Dot,
        BlockTransposedRef<'_, f32, 16>,
        MatRef<'_, Standard<f32>>,
        DotAcc<16, 4, crate::alloc::GlobalAllocator>,
    >(diskann_wide::ARCH, Dot, a, b, acc, a.ncols())
}

///////////
// Tiled //
///////////

pub(crate) trait Tile<'a, Bound = &'a Self> {
    type Item: Copy;
}

pub(crate) trait Tiled: for<'a> Tile<'a> {
    fn next(&mut self) -> Option<<Self as Tile<'_>>::Item>;
}

pub(crate) trait AsTiled<'a> {
    type Tiled: Tiled;
    fn as_tiled(&'a mut self) -> Self::Tiled;
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
//         do_loop(arch, kernel, a, b, acc, self.fracture)
//     }
// }

#[inline(always)]
fn do_loop<Arch, K, A, B, Acc>(arch: Arch, kernel: K, a: A, b: B, acc: &mut Acc, sz: usize)
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

    for mut a_blocks in a.by_ref() {
        let mut b = b.column_paneled(sz);
        if let (Some(a_block), Some(b_cols)) = (a_blocks.next(), b.next()) {
            do_kernel::<true, _, _, _, _, _>(arch, kernel, a_block, b_cols, acc);
        }

        for (a_block, b_cols) in std::iter::zip(a_blocks, b) {
            do_kernel::<false, _, _, _, _, _>(arch, kernel, a_block, b_cols, acc);
        }

        // post process on `acc`
    }

    if let Some(mut a_blocks) = a.tail() {
        let mut b = b.column_paneled(sz);

        if let (Some(a_block), Some(b_cols)) = (a_blocks.next(), b.next()) {
            do_kernel::<true, _, _, _, _, _>(arch, kernel, a_block, b_cols, acc);
        }

        for (a_block, b_cols) in std::iter::zip(a_blocks, b) {
            do_kernel::<false, _, _, _, _, _>(arch, kernel, a_block, b_cols, acc);
        }
    }
}

#[inline(always)]
fn do_kernel<'a, const OVERWRITE: bool, Arch, K, A, B, Acc>(
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
                nsubcols: self.nsubcols,
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

            self.ptr = unsafe { self.ptr.add(N * self.rowstride) };
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

            self.ptr = unsafe { self.ptr.add(BROWS) };
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
        arch: V3,
        a: NoTail,
        b: RowMajorPanel<'_, 4, Static<4>>,
        accumulator: DotAccBlock<'_, 16, 4, Static<4>>,
    ) {
    }
}

impl Kernel<V3, NoTail, RowMajorPanel<'_, 4, Dynamic>, DotAccBlock<'_, 16, 4, Dynamic>> for Dot {
    #[inline(always)]
    fn op<const OVERWRITE: bool>(
        &self,
        arch: V3,
        a: NoTail,
        b: RowMajorPanel<'_, 4, Dynamic>,
        accumulator: DotAccBlock<'_, 16, 4, Dynamic>,
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
        assert_eq!(a.nsubcols, b.nsubcols);
        unsafe { microkernel::<OVERWRITE, 4>(arch, a.ptr, b.ptr, a.nsubcols, accumulator.ptr) };
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
        assert_eq!(a.nsubcols, b.nsubcols);
        assert_eq!(b.nrows.value(), accumulator.active_rows.value());

        match b.nrows.value() {
            1 => unsafe {
                microkernel::<OVERWRITE, 1>(arch, a.ptr, b.ptr, a.nsubcols, accumulator.ptr)
            },
            2 => unsafe {
                microkernel::<OVERWRITE, 2>(arch, a.ptr, b.ptr, a.nsubcols, accumulator.ptr)
            },
            3 => unsafe {
                microkernel::<OVERWRITE, 3>(arch, a.ptr, b.ptr, a.nsubcols, accumulator.ptr)
            },
            _ => unreachable!("invalid value"),
        }
    }
}

#[inline(always)]
unsafe fn microkernel<const OVERWRITE: bool, const UNROLL: usize>(
    arch: V3,
    a: *const f32,
    b: *const f32,
    k: usize,
    buf: *mut f32,
) {
    diskann_wide::alias!(f32s = <V3>::f32x8);

    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

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
