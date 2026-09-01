/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use half::f16;

use crate::multi_vector::distance::v2::{
    Cache,
    blocks::{packed, unpacked},
    bounds,
    kernel::{self, maxsim::MaxSim},
    num::DimK,
    ptr::{MutSlice, Slice},
    util::{Convert, Converter},
};

use super::packed_f32_x_unpacked_f32::{PanelKernel, Params};

pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    kernel: MaxSim<A>,
    a: packed::View<'a, f32, MR>,
    b: unpacked::View<'a, f16>,
    c: MutSlice<'a, f32>,
    k: DimK,
    b_converted: Vec<f32>,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
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

        let params = Params::new(
            cache,
            a.block_stride(k).bytes(),
            b.stride(k).cast::<f32>().bytes(),
            NR,
        );

        unsafe { Self::new_inner(arch, a, b, c, k, params) }
    }

    unsafe fn new_inner(
        arch: A,
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f16>,
        c: MutSlice<'a, f32>,
        k: DimK,
        params: Params,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(b.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(
            c.len(),
            a.extent(),
            "output slice must have one entry for every row in `a`",
        );

        Self {
            kernel: MaxSim::new(arch),
            a,
            b,
            c,
            k,
            b_converted: vec![0.0f32; (b.stride(k) * params.b_cols_in_l1.get()).value()],
            params,
        }
    }
}

impl<A, const MR: usize, const NR: usize> kernel::Drive for Driver<'_, A, MR, NR>
where
    A: Copy,
    Converter<A>: Convert<f32, f16>,
    for<'a> PanelKernel<'a, A, MR, NR>: kernel::PanelKernel,
{
    #[inline(never)]
    fn drive(&mut self) {
        // SAFETY: Class invariant - the length of `self.c` must be equal to `self.a.extent()`
        unsafe { self.c.as_std_mut_slice(self.a.extent().get()) }.fill(f32::NEG_INFINITY);

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
                        PanelKernel::new(self.kernel, a_panel, b_panels_converted, c, self.k)
                    };

                    kernel::PanelKernel::panel_kernel(&mut kernel);
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

    use crate::multi_vector::{BlockTransposed, distance::v2::kernel::maxsim};

    fn test_driver<A, const MR: usize, const NR: usize>(arch: A, rng: &mut impl rand::Rng)
    where
        A: Copy,
        for<'a> Driver<'a, A, MR, NR>: kernel::Drive,
    {
        // (a-panels-per-tile, a-rows, b-cols-per-tile, b-cols, k)
        let cases = [
            (1, MR * 1, 1, 1, 1),           // Smallest valid setup
            (1, MR * 3, 1, 3, 1),           // Unit advancement, no reuse.
            (2, MR * 2, 2 * NR, 2 * NR, 3), // Values a direct multiple of the blocking.
            (2, MR * 3, 2 * NR, NR, 3),     //
            (2, MR * 1, 2 * NR, 2 * NR + 1, 3),
            (2, MR * 3, 2 * NR, 2 * NR + 1, 5),
            (2, MR * 5, 2 * NR, 4 * NR + 1, 1),
        ];

        for case in cases {
            let (a_panels_per_tile, a_rows, b_cols_per_tile, b_cols, k) = case;

            let k = DimK::new(NonZeroUsize::new(k).unwrap());

            let (ref_a, ref_b, ref_c) =
                maxsim::test::generate(a_rows, k.value().get(), b_cols, rng);

            // Massage the input data in the form needed by the kernel.
            let a_bt = BlockTransposed::<f32, MR>::from_matrix_view(ref_a.as_view());
            let b = ref_b.map(|v| diskann_wide::cast_f32_to_f16(*v)).transpose();

            let mut c = vec![f32::NAN; a_bt.padded_nrows()];

            let mut driver = unsafe {
                Driver::new_inner(
                    arch,
                    packed::View::from_block_transposed(a_bt.as_view()),
                    unpacked::View::from_matrix_view(b.as_view()),
                    MutSlice::new(&mut c),
                    k,
                    Params {
                        a_panels_in_l2: NonZeroUsize::new(a_panels_per_tile).unwrap(),
                        b_cols_in_l1: NonZeroUsize::new(b_cols_per_tile).unwrap(),
                    },
                )
            };

            kernel::Drive::drive(&mut driver);

            assert_eq!(
                ref_c, c,
                "a_panels_per_tile: {}, a_rows: {}, b_cols_per_tile: {}, b_cols: {}, k: {:?}",
                a_panels_per_tile, a_rows, b_cols_per_tile, b_cols, k,
            );
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

    test_driver!(
        test_driver_v3,
        V3::new_checked(),
        0x2c03eb9ee51d30c3,
        (16, 4),
        (16, 6),
    );

    test_driver!(
        test_driver_v4,
        V4::new_checked_miri(),
        0x2c03eb9ee51d30c3,
        (16, 4),
    );
}
