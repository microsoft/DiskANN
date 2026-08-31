/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use half::f16;

use crate::multi_vector::distance_v2::{
    Cache,
    blocks::{packed, unpacked},
    bounds,
    kernel::{Drive, PanelKernel, maxsim::MaxSim},
    num::{DimK, value_or_one},
    ptr::{MutSlice, Slice},
    util::{Convert, Converter},
};

use super::f32::{BlockWithRowMajor, Params};

pub(crate) struct PackedXUnpacked<'a, A, const MR: usize, const NR: usize> {
    kernel: MaxSim<A>,
    a: packed::View<'a, f32, MR>,
    b: unpacked::View<'a, f16>,
    c: MutSlice<'a, f32>,
    k: DimK,
    b_converted: Vec<f32>,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> PackedXUnpacked<'a, A, MR, NR> {
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f16>,
        c: MutSlice<'a, f32>,
        k: DimK,
        cache: Cache,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(b.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(
            c.len(),
            a.extent(),
            "output slice must have one entry for every row in `a`",
        );

        // Pick the number of A-panels to process at a time so the working set is within
        // the L2 cache.
        let a_panel_bytes = a.block_stride(k).bytes();
        let a_panels_in_l2 = value_or_one(cache.l2().get() / a_panel_bytes);

        // Pick the number of B-panels to process to the `B` working set plus a single
        // panel of `A` fits in the L1 cache.
        let b_budget = cache.l1().get().saturating_sub(a_panel_bytes);
        let b_cols_in_l1 = value_or_one(NR * (b_budget / (NR * b.stride(k).bytes())));

        let params = Params {
            a_panels_in_l2,
            b_cols_in_l1,
        };

        Self {
            kernel: MaxSim::new(arch),
            a,
            b,
            c,
            k,
            b_converted: vec![0.0f32; (b.stride(k) * b_cols_in_l1.get()).value()],
            params,
        }
    }
}

impl<A, const MR: usize, const NR: usize> Drive for PackedXUnpacked<'_, A, MR, NR>
where
    A: Copy,
    Converter<A>: Convert<f32, f16>,
    for<'a> BlockWithRowMajor<'a, A, MR, NR>: PanelKernel,
{
    #[inline(never)]
    fn drive(&mut self) {
        let on_a_panels = |a_panels: packed::View<'_, f32, MR>, a_block_base| {
            let on_b_panels = |b_panels: unpacked::View<'_, f16>, _| {
                // Convert `f16` to `f32`.
                let b_flat = unsafe { b_panels.as_std_slice(self.k) };
                let b_converted = &mut self.b_converted[..b_flat.len()];
                Converter::new(self.kernel.arch()).convert(b_converted, b_flat);

                let b_panels_converted = unsafe {
                    unpacked::View::new(Slice::new(b_converted), b_panels.extent(), self.k)
                };

                let panel_kernel = |a_panel: packed::Panel<'_, f32, MR>, a_block_offset| {
                    let mut c = unsafe {
                        self.c
                            .subslice(MR * (a_block_base + a_block_offset), bounds::Bound::new(MR))
                    };

                    let c = unsafe { c.as_array::<MR>() };

                    let mut kernel = unsafe {
                        BlockWithRowMajor::new(self.kernel, a_panel, b_panels_converted, c, self.k)
                    };

                    kernel.panel_kernel();
                };

                unsafe {
                    a_panels.visit_panels(self.k, panel_kernel);
                }
            };

            unsafe {
                self.b
                    .visit_sub_views(self.params.b_cols_in_l1, self.k, on_b_panels);
            }
        };

        unsafe {
            self.a
                .visit_sub_views(self.params.a_panels_in_l2, self.k, on_a_panels)
        };
    }
}
