/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_wide::arch::Scalar;

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use crate::multi_vector::distance_v2::{blocks, kernel, num::AllColumns, ptr::MutSlice};

#[derive(Debug)]
pub(crate) struct BlockWithRowMajor<'a, A, const NA: usize, const NB: usize> {
    pub(crate) kernel: kernel::micro::MaxSim<A>,
    pub(crate) a: blocks::fixed::FullBlockTranspose<'a, f32, NA>,
    pub(crate) b: blocks::dynamic::RowMajor<'a, f32>,
    pub(crate) c: MutSlice<'a, f32>,
    pub(crate) cols: AllColumns,
}

trait TailDispatch {
    fn tail_dispatch(&mut self);
}

macro_rules! tail_dispatch {
    ($arch:ty, $na:literal, $nb: literal, [ $($ns:literal),+ $(,)? ]) => {
        impl TailDispatch for BlockWithRowMajor<'_, $arch, $na, $nb> {
            #[inline]
            fn tail_dispatch(&mut self) {
                let last = $nb * (self.b.nrows() / $nb);
                let remainder = self.b.nrows() - last;

                // Repeitition Pattern.
                $(
                    const { assert!($ns < $nb) };
                    if remainder == $ns {
                        kernel::micro::Kernel::kernel(
                            &self.kernel,
                            self.a,
                            self.b.block::<$ns>(self.cols, last),
                            self.cols.value(),
                            self.c.reborrow(),
                        );
                    }
                )+
            }
        }
    }
}

tail_dispatch!(Scalar, 8, 2, [1]);
tail_dispatch!(V3, 16, 4, [1, 2, 3]);
tail_dispatch!(V4, 16, 4, [1, 2, 3]);

impl<Arch, const NA: usize, const NB: usize> kernel::panel::Kernel
    for BlockWithRowMajor<'_, Arch, NA, NB>
where
    for<'a> kernel::micro::MaxSim<Arch>: kernel::micro::Kernel<
        blocks::fixed::FullBlockTranspose<'a, f32, NA>,
        blocks::fixed::FullRowMajor<'a, f32, NB>,
        MutSlice<'a, f32>,
    >,
    Self: TailDispatch,
{
    fn run(&mut self) {
        let blocks = self.b.nrows() / NB;

        for i in 0..blocks {
            kernel::micro::Kernel::kernel(
                &self.kernel,
                self.a,
                self.b.block::<NB>(self.cols, NB * i),
                self.cols.value(),
                self.c.reborrow(),
            );
        }

        self.tail_dispatch();
    }
}
