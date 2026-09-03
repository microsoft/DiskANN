/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This closely follows the implementation in [`super::packed_f32_x_unpacked_f32`].
//!
//! The only addition is that the sub-views of `b` are converted from `f16` to `f32` when
//! before any panel-kernel operation.

use diskann_wide::arch::Architecture;
use half::f16;

use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    bounds, driver,
    num::DimK,
    ptr::{MutSlice, Slice},
    util::{self, Convert, Converter},
};

use super::packed_f32_x_unpacked_f32::{PanelKernel, Params};

/// A driver for prepacked by unpacked "maxsim" computations.
///
/// Results are returned directly in `c`.
///
/// Note that class invariant (2) allows the physical length of the output to be less than
/// the packed extent of `a` as long as it resides within the last physical block of `a`.
///
/// This allows the kernel to write directly into an output buffer when `a` is not logically
/// filled. `a` being *physically* filled is still a requirement.
///
/// # Class Invariants
///
/// 1. `a.k()` and `b.k()` must be equal to `k`.
/// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, f32, MR>,
    b: unpacked::View<'a, f16>,
    c: &'a mut [f32],
    k: DimK,
    b_converted: Vec<f32>,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// Prepare for a maxsim on `a` and `b` with the results stored directly into `c`.
    ///
    /// `c` does not any specific initial value.
    ///
    /// # Safety
    ///
    /// 1. `a.k()` and `b.k()` must be equal to `k`.
    /// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f16>,
        c: &'a mut [f32],
        k: DimK,
        cache: Cache,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(b.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupiy exactly the packed A blocks",
        );

        let params = Params::new(
            cache,
            a.block_stride(k).bytes(),
            b.stride(k).cast::<f32>().bytes(),
            NR,
        );

        // SAFETY: Inherited from caller.
        unsafe { Self::new_inner(arch, a, b, c, k, params) }
    }

    /// # Safety
    ///
    /// 1. `a.k()` and `b.k()` must be equal to `k`.
    /// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
    unsafe fn new_inner(
        arch: A,
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f16>,
        c: &'a mut [f32],
        k: DimK,
        params: Params,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(b.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupiy exactly the packed A blocks",
        );

        Self {
            arch,
            a,
            b,
            c,
            k,
            b_converted: vec![0.0f32; (b.stride(k) * params.b_cols_in_l1.get()).value()],
            params,
        }
    }
}

impl<A, const MR: usize, const NR: usize> driver::Drive for Driver<'_, A, MR, NR>
where
    A: util::LoadStore<f32, MR> + Architecture,
    Converter<A>: Convert<f32, f16>,
    for<'a> PanelKernel<'a, A, MR, NR>: driver::PanelKernel,
{
    #[inline(never)]
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                // Pre-fill `c`.
                self.c.fill(f32::NEG_INFINITY);

                // We allow `c` to be slightly under-filled.
                //
                // These variables track if under-fill is happening.
                let remainder = self.c.len() % MR;
                let last_a_block = self.a.blocks().get() - 1;

                let mut c = MutSlice::new(self.c);

                let on_a_panels = |a_panels: packed::View<'_, f32, MR>, a_block_base| {
                    let on_b_panels = |b_panels: unpacked::View<'_, f16>, _| {
                        // Convert `f16` to `f32`.
                        //
                        // SAFETY: Class invariant - `self.b.k()` is equal to `self.k`.
                        let b_flat = unsafe { b_panels.as_std_slice(self.k) };
                        let b_converted = &mut self.b_converted[..b_flat.len()];
                        Converter::new(self.arch).convert(b_converted, b_flat);

                        // SAFETY: `b_converted` has length `b_panels.extent() * self.k`.
                        let b_panels_converted = unsafe {
                            unpacked::View::new(Slice::new(b_converted), b_panels.extent(), self.k)
                        };

                        let panel_kernel = |a_panel: packed::Panel<'_, f32, MR>, a_block_offset| {
                            // If we are in the very last block and we need to sub-fill, do that.
                            // Otherwise, reference the output in place.
                            let a_block = a_block_base + a_block_offset;
                            let handling_tail = a_block == last_a_block && remainder != 0;

                            let bound =
                                bounds::Bound::from_fn(
                                    || if handling_tail { remainder } else { MR },
                                );

                            // SAFETY: By class invariant,
                            //
                            // `MR * (self.a.blocks() - 1) < c.len() <= MR * self.a.blocks()`.
                            //
                            // From the visitor, `a_block <= self.a.blocks()`.
                            let mut region = unsafe { c.subslice(MR * a_block, bound) };
                            let c = if handling_tail {
                                util::LoadStore::<f32, MR>::load(
                                    self.arch,
                                    // SAFETY: `region` as length exactly `remainder`.
                                    unsafe { region.as_std_slice(remainder) },
                                )
                            } else {
                                // SAFETY: `region` has length exactly `MR`.
                                unsafe { *region.as_array::<MR>() }
                            };

                            // Run the kernel
                            //
                            // SAFETY: By class invariant, `a_panel.k()` and
                            // `b_panels_converted.k()` are equal to `self.k`.
                            let mut kernel = unsafe {
                                PanelKernel::new(self.arch, a_panel, b_panels_converted, c, self.k)
                            };

                            driver::PanelKernel::panel_kernel(&mut kernel);

                            let c_final = kernel.take();

                            // Put back `C`.
                            if handling_tail {
                                util::LoadStore::<f32, MR>::store(
                                    self.arch,
                                    c_final,
                                    // SAFETY: `region` has length exactly `remainder`.
                                    unsafe { region.as_std_mut_slice(remainder) },
                                );
                            } else {
                                // SAFETY: `region` has length exactly `MR`.
                                unsafe { *region.as_array::<MR>() = c_final };
                            }
                        };

                        // SAFETY: By class invariant, `a_panels.k() == self.k`.
                        unsafe {
                            a_panels.visit_panels(self.k, panel_kernel);
                        }
                    };

                    // SAFETY: By class invariant, `self.b.k() == self.k`.
                    unsafe {
                        self.b
                            .visit_sub_views(self.params.b_cols_in_l1, self.k, on_b_panels);
                    }
                };

                // SAFETY: By class invariant, `self.a.k() == self.k`.
                unsafe {
                    self.a
                        .visit_sub_views(self.params.a_panels_in_l2, self.k, on_a_panels)
                };
            },
        );
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroUsize;

    use diskann_wide::arch::Scalar;
    use rand::{SeedableRng, rngs::StdRng};

    #[cfg(target_arch = "x86_64")]
    use diskann_wide::arch::x86_64::{V3, V4};

    #[cfg(target_arch = "aarch64")]
    use diskann_wide::arch::aarch64::Neon;

    use crate::{matrix_kernels::maxsim, multi_vector::BlockTransposed};

    fn test_driver<A, const MR: usize, const NR: usize>(arch: A, rng: &mut impl rand::Rng)
    where
        A: Copy,
        for<'a> Driver<'a, A, MR, NR>: driver::Drive,
    {
        let cases = maxsim::test::packed_x_unpacked_test_dims(MR, NR);
        for case in cases {
            let maxsim::test::TestDims {
                a_panels_per_tile,
                total_a_rows,
                b_cols_per_tile,
                total_b_cols,
                k,
            } = case.clone();

            let k = DimK::new(NonZeroUsize::new(k).unwrap());

            let (ref_a, ref_b, ref_c) =
                maxsim::test::generate(total_a_rows, k.value().get(), total_b_cols, rng);

            // Massage the input data in the form needed by the kernel.
            let a_bt = BlockTransposed::<f32, MR>::from_matrix_view(ref_a.as_view());
            let b = ref_b.map(|v| diskann_wide::cast_f32_to_f16(*v)).transpose();

            let mut c = vec![f32::NAN; a_bt.nrows()];

            // SAFETY: Test builds will verify the bounds we passed.
            let mut driver = unsafe {
                Driver::new_inner(
                    arch,
                    packed::View::from_block_transposed(a_bt.as_view()).unwrap(),
                    unpacked::View::from_matrix_view(b.as_view()).unwrap(),
                    &mut c,
                    k,
                    Params {
                        a_panels_in_l2: NonZeroUsize::new(a_panels_per_tile).unwrap(),
                        b_cols_in_l1: NonZeroUsize::new(b_cols_per_tile).unwrap(),
                    },
                )
            };

            driver::Drive::drive(&mut driver);
            assert_eq!(ref_c, c, "setup: {:?}", case);
        }
    }

    macro_rules! test_driver {
        (
            $fn:ident,
            $arch:expr,
            $seed:literal,
            $(
                (
                    $MR:literal, $NR:literal
                )
            ),+ $(,)?
        ) => {
            #[test]
            fn $fn() {
                if let Some(arch) = $arch {
                    let mut rng = StdRng::seed_from_u64($seed);

                    $(test_driver::<_, $MR, $NR>(arch, &mut rng);)+
                }
            }
        }
    }

    test_driver!(
        test_driver_scalar,
        Some(Scalar::new()),
        0x2c03eb9ee51d30c3,
        (8, 2),
    );

    #[cfg(target_arch = "x86_64")]
    test_driver!(
        test_driver_v3,
        V3::new_checked(),
        0x2c03eb9ee51d30c3,
        (16, 6),
    );

    #[cfg(target_arch = "x86_64")]
    test_driver!(
        test_driver_v4,
        V4::new_checked_miri(),
        0x2c03eb9ee51d30c3,
        (16, 6),
        (32, 6),
    );

    #[cfg(target_arch = "aarch64")]
    test_driver!(
        test_driver_neon,
        Neon::new_checked(),
        0x2c03eb9ee51d30c3,
        (8, 6),
    );
}
