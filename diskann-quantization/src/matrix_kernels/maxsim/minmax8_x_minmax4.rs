// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! MinMax8 by MinMax4 MaxSim over the existing packed/unpacked panel views.
//!
//! A is block-transposed in architecture-specific contraction groups. B remains in its
//! canonical packed MinMax4 layout. Small B tiles can be expanded once into scratch and
//! reused across query panels; both representations use the same panel traversal.
//! Only the original dimension participates in compensation; padded groups are zero.
//! The driver borrows all inputs and bounds its document scratch on the stack.
//! Scalar, V3, and Neon use four-byte even/odd groups. V4 uses contiguous eight-byte
//! groups so one broadcast feeds two adjacent VNNI lanes per query row.

use diskann_wide::{Architecture, SIMDMinMax, SIMDVector, arch::Scalar};

use crate::{
    matrix_kernels::{
        blocks::{packed, unpacked},
        bounds::{self, Bound},
        driver,
        num::{DimK, Elements, value_or_one},
        ptr::{MutSlice, Slice},
        util,
    },
    minmax::{Data, DataRef, MinMaxCompensation},
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Grouped<const N: usize>(pub(crate) [u8; N]);

impl<const N: usize> Default for Grouped<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> Grouped<N> {
    pub(crate) fn count(dim: usize) -> usize {
        const { assert!(N == 4 || N == 8) };
        dim.div_ceil(8) * (8 / N)
    }

    pub(crate) fn from_query(values: &[u8], group: usize) -> Self {
        assert!(group < Self::count(values.len()));
        Self(core::array::from_fn(|lane| {
            let dim = if N == 4 {
                group / 2 * 8 + 2 * lane + group % 2
            } else {
                group * 8 + lane
            };
            values.get(dim).copied().unwrap_or(0)
        }))
    }

    /// Expand one group from a canonical packed MinMax4 row.
    ///
    /// # Safety
    ///
    /// `values` contains exactly `dim.div_ceil(2)` bytes and
    /// `group < Self::count(dim)`.
    #[inline(always)]
    unsafe fn from_packed(values: Slice<'_, u8>, group: usize, dim: usize) -> Self {
        bounds::check_eq!(values.len(), dim.div_ceil(2));
        bounds::check_lt!(Bound::new(group), Self::count(dim));

        let groups_per_block = 8 / N;
        let block = group / groups_per_block;
        let within_block = group % groups_per_block;
        let byte_start = block * 4;
        let dimensions = (dim - block * 8).min(8);
        let byte_count = dimensions.div_ceil(2);
        // SAFETY: `group` is within the number of eight-dimensional blocks.
        let packed = unsafe {
            values
                .add(Elements::new(byte_start))
                .truncate(Elements::new(byte_count))
        };

        if dimensions == 8 {
            // SAFETY: A complete block tracks exactly four bytes.
            let source = u32::from_le(unsafe { packed.as_ptr().cast::<u32>().read_unaligned() });
            if N == 4 {
                let expanded = if within_block == 0 {
                    (source & 0x0f0f_0f0f).to_le_bytes()
                } else {
                    ((source >> 4) & 0x0f0f_0f0f).to_le_bytes()
                };
                return Self(core::array::from_fn(|lane| expanded[lane]));
            }

            #[cfg(all(target_arch = "x86_64", not(miri)))]
            {
                use std::arch::x86_64::_pdep_u64;

                // SAFETY: Grouped<8> is only used by V4, which provides BMI2.
                let expanded =
                    unsafe { _pdep_u64(u64::from(source), 0x0f0f_0f0f_0f0f_0f0f) }.to_le_bytes();
                return Self(core::array::from_fn(|lane| expanded[lane]));
            }
        }

        // SAFETY: `packed` tracks exactly the bytes in the final partial block.
        let packed = unsafe { packed.as_std_slice(byte_count) };
        Self(core::array::from_fn(|lane| {
            let dimension = if N == 4 {
                2 * lane + within_block
            } else {
                lane
            };
            if dimension >= dimensions {
                0
            } else {
                (packed[dimension / 2] >> (4 * (dimension % 2))) & 15
            }
        }))
    }
}

impl<const N: usize> std::ops::Deref for Grouped<N> {
    type Target = [u8; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> std::ops::DerefMut for Grouped<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Structure-of-arrays coefficients for one complete query panel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryCompensation<const MR: usize> {
    pub(crate) scale: [f32; MR],
    pub(crate) bias: [f32; MR],
    pub(crate) scaled_sum: [f32; MR],
}

impl<const MR: usize> Default for QueryCompensation<MR> {
    fn default() -> Self {
        Self {
            scale: [0.0; MR],
            bias: [0.0; MR],
            scaled_sum: [0.0; MR],
        }
    }
}

/// `k` counts contraction groups, not original dimensions.
pub(crate) struct Driver<'a, A, const N: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, Grouped<N>, MR>,
    a_meta: &'a [QueryCompensation<MR>],
    b: unpacked::View<'a, u8>,
    c: &'a mut [f32],
    dim: DimK,
    k: DimK,
    b_stride: DimK,
}

impl<'a, A, const N: usize, const MR: usize, const NR: usize> Driver<'a, A, N, MR, NR> {
    /// # Safety
    ///
    /// * A uses the `Grouped<N>` packing for `dim`, including zero padding.
    /// * `a_meta.len() == a.blocks()` and `c.len().div_ceil(MR) == a.blocks()`.
    /// * B contains canonical MinMax4 rows of dimension `dim`.
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, Grouped<N>, MR>,
        a_meta: &'a [QueryCompensation<MR>],
        b: unpacked::View<'a, u8>,
        c: &'a mut [f32],
        dim: DimK,
    ) -> Self {
        // SAFETY: Inherited from caller.
        unsafe { Self::new_inner(arch, a, a_meta, b, c, dim) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn new_inner(
        arch: A,
        a: packed::View<'a, Grouped<N>, MR>,
        a_meta: &'a [QueryCompensation<MR>],
        b: unpacked::View<'a, u8>,
        c: &'a mut [f32],
        dim: DimK,
    ) -> Self {
        let k = DimK::new(value_or_one(Grouped::<N>::count(dim.value().get())));
        let b_stride = DimK::new(value_or_one(Data::<4>::canonical_bytes(dim.value().get())));
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), b_stride);
        bounds::check_eq!(Bound::new(a_meta.len()), a.blocks());
        bounds::check_eq!(Bound::new(c.len().div_ceil(MR)), a.blocks());
        Self {
            arch,
            a,
            a_meta,
            b,
            c,
            dim,
            k,
            b_stride,
        }
    }
}

impl<A, const N: usize, const MR: usize, const NR: usize> Driver<'_, A, N, MR, NR> {
    #[inline(always)]
    fn drive_b<B: BSource<N>>(&mut self, b: B, b_stride: DimK)
    where
        A: Architecture + util::LoadStore<f32, MR> + ExtraWide<N, MR>,
        for<'a> PanelKernel<'a, A, N, MR, NR, B>: driver::PanelKernel,
    {
        let output_rows = self.c.len();
        let mut c = MutSlice::new(self.c);
        let a_meta = Slice::new(self.a_meta);
        let on_panel = |a: packed::Panel<'_, Grouped<N>, MR>, block: usize| {
            let valid_rows = (output_rows - block * MR).min(MR);
            // SAFETY: The output occupies exactly the packed A blocks.
            let mut region = unsafe { c.subslice(block * MR, Bound::new(valid_rows)) };
            // SAFETY: The region was truncated to `valid_rows` above.
            let output = unsafe { region.as_std_mut_slice(valid_rows) };
            let scores = util::LoadStore::<f32, MR>::load(self.arch, output);
            // SAFETY: Every packed A block has exactly one metadata entry.
            let query = unsafe { a_meta.add(Elements::new(block)).as_unit().as_ref() };
            // SAFETY: The driver validates all shared dimensions, and b_stride is the
            // stride of the canonical or decoded B view supplied by the caller.
            let mut panel = unsafe {
                PanelKernel::new(
                    self.arch, a, query, b, scores, self.k, self.dim, b_stride, valid_rows,
                )
            };
            driver::PanelKernel::panel_kernel(&mut panel);
            util::LoadStore::<f32, MR>::store(self.arch, panel.c, output);
        };
        // SAFETY: A was validated against k on construction.
        unsafe { self.a.visit_panels(self.k, on_panel) };
    }

    // Keep the bounded scratch frame off the direct path for single-panel queries.
    #[inline(never)]
    fn drive_tiled(&mut self, tile_rows: usize)
    where
        A: Architecture + util::LoadStore<f32, MR> + ExtraWide<N, MR>,
        for<'a, 'b> PanelKernel<'a, A, N, MR, NR, DecodedB<'b, N>>: driver::PanelKernel,
    {
        self.arch.run(
            #[inline]
            || {
                let groups = self.k.value().get();
                let mut values = [Grouped::<N>::default(); B_TILE_GROUPS];
                let mut metadata = [MinMaxCompensation::default(); B_TILE_ROWS];
                let b = self.b;
                // SAFETY: The driver validates B's canonical stride. Each tile fits
                // in both scratch arrays, including its final partial extent.
                unsafe {
                    b.visit_sub_views(value_or_one(tile_rows), self.b_stride, |tile, _| {
                        let rows = tile.extent().get();
                        let decoded = DecodedB::new(
                            tile,
                            self.b_stride,
                            self.dim,
                            self.k,
                            &mut values[..rows * groups],
                            &mut metadata[..rows],
                        );
                        self.drive_b(decoded, self.k);
                    });
                }
            },
        );
    }
}

const B_TILE_GROUPS: usize = 2048;
const B_TILE_ROWS: usize = 32;

impl<A, const N: usize, const MR: usize, const NR: usize> driver::Drive for Driver<'_, A, N, MR, NR>
where
    A: Architecture + util::LoadStore<f32, MR> + ExtraWide<N, MR>,
    for<'a, 'b> PanelKernel<'a, A, N, MR, NR, unpacked::View<'b, u8>>: driver::PanelKernel,
    for<'a, 'b> PanelKernel<'a, A, N, MR, NR, DecodedB<'b, N>>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                self.c.fill(f32::MAX);
                let groups = self.k.value().get();
                let tile_rows = (B_TILE_GROUPS / groups).min(B_TILE_ROWS) / NR * NR;
                if self.a.blocks().get() == 1 || tile_rows == 0 {
                    self.drive_b(self.b, self.b_stride);
                    return;
                }

                self.drive_tiled(tile_rows);
            },
        );
    }
}

struct PanelKernel<'a, A, const N: usize, const MR: usize, const NR: usize, B> {
    arch: A,
    a: packed::Panel<'a, Grouped<N>, MR>,
    query: &'a QueryCompensation<MR>,
    b: B,
    c: [f32; MR],
    k: DimK,
    dim: DimK,
    b_stride: DimK,
    valid_rows: usize,
}

impl<'a, A, const N: usize, const MR: usize, const NR: usize, B: BSource<N>>
    PanelKernel<'a, A, N, MR, NR, B>
{
    /// # Safety
    ///
    /// A has k grouped columns and B contains canonical or decoded MinMax4 rows for dim.
    /// Only `valid_rows` query rows are logically present.
    #[allow(clippy::too_many_arguments)]
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, Grouped<N>, MR>,
        query: &'a QueryCompensation<MR>,
        b: B,
        c: [f32; MR],
        k: DimK,
        dim: DimK,
        b_stride: DimK,
        valid_rows: usize,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.view().k(), b_stride);
        bounds::check_eq!(Bound::new(Grouped::<N>::count(dim.value().get())), k);
        bounds::check_le!(Bound::new(valid_rows), MR);
        Self {
            arch,
            a,
            query,
            b,
            c,
            k,
            dim,
            b_stride,
            valid_rows,
        }
    }
}

struct Visitor<'a, A, const N: usize, const MR: usize, B> {
    arch: A,
    a: packed::Panel<'a, Grouped<N>, MR>,
    query: &'a QueryCompensation<MR>,
    c: &'a mut [f32; MR],
    k: DimK,
    dim: DimK,
    b_stride: DimK,
    valid_rows: usize,
    source: B,
}

impl<A, const N: usize, const MR: usize, const NR: usize, B> unpacked::PanelVisitor<B::Value, NR>
    for Visitor<'_, A, N, MR, B>
where
    A: Architecture + ExtraWide<N, MR>,
    B: BSource<N>,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, B::Value, NR>, start: usize) {
        // SAFETY: The visitor receives complete rows from the validated source.
        let b = unsafe { self.source.panel(&b, start, self.b_stride, self.dim) };
        let mut micro = MicroKernel {
            arch: self.arch,
            a: self.a,
            query: self.query,
            b,
            c: self.c,
            k: self.k,
            dim: self.dim,
            valid_rows: self.valid_rows,
        };
        driver::MicroKernel::micro_kernel(&mut micro);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $n:literal, $mr:literal, $nr:literal, [$($tail:literal),+]) => {
        impl<B: BSource<$n>> driver::PanelKernel
            for PanelKernel<'_, $arch, $n, $mr, $nr, B>
        {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let visitor = Visitor {
                    arch: self.arch, a: self.a, query: self.query, c: &mut self.c,
                    k: self.k, dim: self.dim, b_stride: self.b_stride,
                    valid_rows: self.valid_rows,
                    source: self.b,
                };
                let b = self.b.view();
                // SAFETY: The constructor checks B's row stride.
                let remainder = unsafe { b.visit_panels::<$nr>(self.b_stride, visitor) };
                if let Some(remainder) = remainder {
                    $(
                        if let Some(panel) = remainder.try_as_panel::<$tail>() {
                            let mut visitor = Visitor {
                                arch: self.arch, a: self.a, query: self.query, c: &mut self.c,
                                k: self.k, dim: self.dim, b_stride: self.b_stride,
                                valid_rows: self.valid_rows,
                                source: self.b,
                            };
                            unpacked::PanelVisitor::visit(&mut visitor, panel, remainder.start());
                        }
                    )+
                }
            }
        }
    };
}

#[derive(Clone, Copy)]
struct BPanel<'a, const NR: usize, T = u8> {
    values: [Slice<'a, T>; NR],
    meta: [MinMaxCompensation; NR],
}

impl<'a, const NR: usize> BPanel<'a, NR> {
    /// # Safety
    ///
    /// `panel` contains NR canonical MinMax4 rows of dimension `dim`.
    unsafe fn new(panel: unpacked::Panel<'a, u8, NR>, stride: DimK, dim: DimK) -> Self {
        bounds::check_eq!(panel.k(), stride);
        let dim = dim.value().get();
        let packed_len = dim.div_ceil(2);
        let base = panel.as_ptr();
        let row_stride = panel.stride(stride);
        let rows: [(Slice<'a, u8>, MinMaxCompensation); NR] = core::array::from_fn(|row| {
            // SAFETY: `panel` contains NR complete canonical rows.
            let bytes = unsafe {
                base.add(row_stride * row)
                    .truncate(row_stride)
                    .as_std_slice(stride.value().get())
            };
            // SAFETY: The driver validated the canonical row stride and dimension.
            let data = unsafe { DataRef::<4>::from_canonical_unchecked(bytes, dim) };
            let vector = data.vector();
            // SAFETY: The vector owns exactly `ceil(dim / 2)` packed bytes.
            let values = unsafe {
                Slice::from_raw(
                    std::ptr::NonNull::new_unchecked(vector.as_ptr().cast_mut()),
                    Bound::new(packed_len),
                )
            };
            (values, data.meta())
        });
        Self {
            values: rows.map(|(values, _)| values),
            meta: rows.map(|(_, meta)| meta),
        }
    }
}

trait BSource<const N: usize>: Copy {
    type Value: LoadBGroup<N>;

    fn view(&self) -> unpacked::View<'_, Self::Value>;

    /// # Safety
    ///
    /// The panel contains NR rows beginning at start in this source, with the
    /// supplied stride and original dimension.
    unsafe fn panel<'a, const NR: usize>(
        self,
        panel: &'a unpacked::Panel<'_, Self::Value, NR>,
        start: usize,
        stride: DimK,
        dim: DimK,
    ) -> BPanel<'a, NR, Self::Value>;
}

impl<const N: usize> BSource<N> for unpacked::View<'_, u8> {
    type Value = u8;

    fn view(&self) -> unpacked::View<'_, u8> {
        *self
    }

    #[inline(always)]
    unsafe fn panel<'a, const NR: usize>(
        self,
        panel: &'a unpacked::Panel<'_, u8, NR>,
        _: usize,
        stride: DimK,
        dim: DimK,
    ) -> BPanel<'a, NR> {
        // SAFETY: Inherited from the source contract.
        unsafe { BPanel::new(*panel, stride, dim) }
    }
}

#[derive(Clone, Copy)]
struct DecodedB<'a, const N: usize> {
    values: unpacked::View<'a, Grouped<N>>,
    meta: Slice<'a, MinMaxCompensation>,
}

impl<'a, const N: usize> DecodedB<'a, N> {
    /// # Safety
    ///
    /// B contains canonical MinMax4 rows of dimension dim and stride b_stride.
    /// k equals Grouped<N>::count(dim), and the backend supports this grouping.
    unsafe fn new(
        b: unpacked::View<'_, u8>,
        b_stride: DimK,
        dim: DimK,
        k: DimK,
        values: &'a mut [Grouped<N>],
        meta: &'a mut [MinMaxCompensation],
    ) -> Self {
        let rows = b.extent();
        let groups = k.value().get();
        let dim = dim.value().get();
        assert_eq!(values.len(), rows.get() * groups);
        assert_eq!(meta.len(), rows.get());
        bounds::check_eq!(Bound::new(groups), Grouped::<N>::count(dim));
        // SAFETY: The caller supplies the validated canonical stride.
        let bytes = unsafe { b.as_std_slice(b_stride) };
        for ((row, output), meta) in bytes
            .chunks_exact(b_stride.value().get())
            .zip(values.chunks_exact_mut(groups))
            .zip(meta.iter_mut())
        {
            // SAFETY: Each row has the canonical stride and dimension.
            let data = unsafe { DataRef::<4>::from_canonical_unchecked(row, dim) };
            *meta = data.meta();
            let vector = data.vector();
            // SAFETY: The vector retains exactly ceil(dim / 2) packed bytes.
            let source = unsafe {
                Slice::from_raw(
                    std::ptr::NonNull::new_unchecked(vector.as_ptr().cast_mut()),
                    Bound::new(dim.div_ceil(2)),
                )
            };
            for (group, output) in output.iter_mut().enumerate() {
                // SAFETY: Every group is within the validated dimension.
                *output = unsafe { Grouped::from_packed(source, group, dim) };
            }
        }
        Self {
            // SAFETY: values contains exactly rows * k initialized groups.
            values: unsafe { unpacked::View::new(Slice::new(values), rows, k) },
            meta: Slice::new(meta),
        }
    }
}

impl<const N: usize> BSource<N> for DecodedB<'_, N> {
    type Value = Grouped<N>;

    fn view(&self) -> unpacked::View<'_, Self::Value> {
        self.values
    }

    #[inline(always)]
    unsafe fn panel<'a, const NR: usize>(
        self,
        panel: &'a unpacked::Panel<'_, Self::Value, NR>,
        start: usize,
        stride: DimK,
        _: DimK,
    ) -> BPanel<'a, NR, Self::Value> {
        let row_stride = panel.stride(stride);
        let values = core::array::from_fn(|row| {
            // SAFETY: The fixed-size panel contains NR complete grouped rows.
            unsafe { panel.as_ptr().add(row_stride * row).truncate(row_stride) }
        });
        let meta = core::array::from_fn(|row| {
            // SAFETY: The source provides metadata for every row in this panel.
            unsafe { *self.meta.add(Elements::new(start + row)).as_unit().as_ref() }
        });
        BPanel { values, meta }
    }
}

trait LoadBGroup<const N: usize>: Copy {
    /// # Safety
    ///
    /// Values contains a complete row for dim, and group is within its grouped extent.
    unsafe fn splat<A: ExtraWide<N, MR>, const MR: usize>(
        arch: A,
        values: Slice<'_, Self>,
        group: usize,
        dim: usize,
    ) -> A::Splat;
}

impl<const N: usize> LoadBGroup<N> for u8 {
    #[inline(always)]
    unsafe fn splat<A: ExtraWide<N, MR>, const MR: usize>(
        arch: A,
        values: Slice<'_, Self>,
        group: usize,
        dim: usize,
    ) -> A::Splat {
        // SAFETY: Inherited from the complete packed-row contract.
        unsafe { arch.unpack(values, group, dim) }
    }
}

impl<const N: usize> LoadBGroup<N> for Grouped<N> {
    #[inline(always)]
    unsafe fn splat<A: ExtraWide<N, MR>, const MR: usize>(
        arch: A,
        values: Slice<'_, Self>,
        group: usize,
        dim: usize,
    ) -> A::Splat {
        bounds::check_eq!(values.len(), Grouped::<N>::count(dim));
        // SAFETY: Group is within the initialized grouped row.
        arch.splat(unsafe { *values.add(Elements::new(group)).as_unit().as_ref() })
    }
}

struct MicroKernel<'a, A, const N: usize, const MR: usize, const NR: usize, T> {
    arch: A,
    a: packed::Panel<'a, Grouped<N>, MR>,
    query: &'a QueryCompensation<MR>,
    b: BPanel<'a, NR, T>,
    c: &'a mut [f32; MR],
    k: DimK,
    dim: DimK,
    valid_rows: usize,
}

impl<A, const N: usize, const MR: usize, const NR: usize, T> driver::MicroKernel
    for MicroKernel<'_, A, N, MR, NR, T>
where
    A: Architecture + ExtraWide<N, MR>,
    T: LoadBGroup<N>,
{
    #[inline(always)]
    fn micro_kernel(&mut self) {
        self.arch.run_inline(
            #[inline]
            || {
                bounds::check_eq!(self.a.k(), self.k);
                // SAFETY: The panel constructor establishes the common contraction
                // dimension and the number of valid query rows.
                let acc = unsafe {
                    self.arch
                        .contract(self.a, &self.b, self.k, self.dim, self.valid_rows)
                };
                // SAFETY: The metadata span contains exactly NR entries.
                unsafe {
                    self.arch
                        .reduce(acc, self.query, Slice::new(&self.b.meta), self.dim, self.c)
                };
            },
        );
    }
}

/// Register layout and instruction selection are private to each implementation.
trait ExtraWide<const N: usize, const MR: usize>: Copy {
    type Query: Copy;
    type Splat: Copy;
    type Accumulator: Copy;

    /// # Safety
    ///
    /// `values` contains exactly MR groups and `valid_rows <= MR`.
    unsafe fn load(self, values: Slice<'_, Grouped<N>>, valid_rows: usize) -> Self::Query;
    fn zero(self) -> Self::Accumulator;
    fn splat(self, value: Grouped<N>) -> Self::Splat;

    /// Expand one packed document group into the backend's broadcast representation.
    ///
    /// # Safety
    ///
    /// `values` contains exactly `dim.div_ceil(2)` packed MinMax4 bytes and
    /// `group < Grouped::<N>::count(dim)`.
    #[inline(always)]
    unsafe fn unpack(self, values: Slice<'_, u8>, group: usize, dim: usize) -> Self::Splat {
        // SAFETY: Inherited from the trait contract.
        self.splat(unsafe { Grouped::<N>::from_packed(values, group, dim) })
    }

    fn dot(self, a: Self::Query, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator;

    /// # Safety
    ///
    /// Both panels have contraction dimension `k` and `valid_rows <= MR`.
    #[inline(always)]
    unsafe fn contract<const NR: usize, T: LoadBGroup<N>>(
        self,
        a: packed::Panel<'_, Grouped<N>, MR>,
        b: &BPanel<'_, NR, T>,
        k: DimK,
        dim: DimK,
        valid_rows: usize,
    ) -> [Self::Accumulator; NR] {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(Bound::new(Grouped::<N>::count(dim.value().get())), k);
        bounds::check_le!(Bound::new(valid_rows), MR);
        let mut acc = [self.zero(); NR];
        let ap = a.as_ptr();
        for i in 0..k.value().get() {
            // SAFETY: A has k complete MR-element groups.
            let a = unsafe {
                self.load(
                    ap.add(Elements::new(i * MR)).truncate(Elements::new(MR)),
                    valid_rows,
                )
            };
            for (j, acc) in acc.iter_mut().enumerate() {
                // SAFETY: The B panel contains one complete packed row per accumulator.
                let b = unsafe { T::splat(self, b.values[j], i, dim.value().get()) };
                *acc = self.dot(a, b, *acc);
            }
        }
        acc
    }

    /// # Safety
    ///
    /// `docs` contains exactly NR metadata entries, one per accumulator.
    unsafe fn reduce<const NR: usize>(
        self,
        acc: [Self::Accumulator; NR],
        query: &QueryCompensation<MR>,
        docs: Slice<'_, MinMaxCompensation>,
        dim: DimK,
        scores: &mut [f32; MR],
    );
}

impl ExtraWide<4, 8> for Scalar {
    type Query = [[u32; 8]; 4];
    type Splat = [u32; 4];
    type Accumulator = [u32; 8];

    #[inline(always)]
    unsafe fn load(self, values: Slice<'_, Grouped<4>>, _: usize) -> Self::Query {
        // SAFETY: The trait contract requires exactly eight initialized groups.
        let values = unsafe { values.as_std_slice(8) };
        core::array::from_fn(|d| core::array::from_fn(|row| u32::from(values[row][d])))
    }
    #[inline(always)]
    fn zero(self) -> Self::Accumulator {
        [0; 8]
    }
    #[inline(always)]
    fn splat(self, value: Grouped<4>) -> Self::Splat {
        value.map(u32::from)
    }
    #[inline(always)]
    fn dot(self, a: Self::Query, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator {
        core::array::from_fn(|i| {
            let dot = a[0][i] * b[0] + a[1][i] * b[1] + a[2][i] * b[2] + a[3][i] * b[3];
            acc[i].wrapping_add(dot)
        })
    }
    #[inline(always)]
    unsafe fn reduce<const NR: usize>(
        self,
        acc: [Self::Accumulator; NR],
        query: &QueryCompensation<8>,
        docs: Slice<'_, MinMaxCompensation>,
        dim: DimK,
        scores: &mut [f32; 8],
    ) {
        // SAFETY: The trait contract requires exactly NR metadata entries.
        let docs = unsafe { docs.as_std_slice(NR) };
        for (acc, doc) in acc.into_iter().zip(docs) {
            for i in 0..8 {
                let similarity = query.scale[i] * doc.a * acc[i] as f32
                    + query.scaled_sum[i] * doc.b
                    + doc.n * query.bias[i]
                    + query.bias[i] * doc.b * dim.value().get() as f32;
                scores[i] = scores[i].min(-similarity);
            }
        }
    }
}

panel_kernel!(Scalar, 4, 8, 6, [1, 2, 3, 4, 5]);

// Kept separate from contraction so metadata and floating point constraints do not
// leak through ExtraWide's opaque register types.
macro_rules! compensate {
    ($arch:expr, $float:ty, $lanes:literal, $parts:literal, $acc:expr, $convert:expr, $query:expr, $docs:expr, $dim:expr, $scores:expr) => {{
        let query = $query;
        let acc = $acc;
        let docs = $docs;
        bounds::check_eq!(docs.len(), acc.len());
        let mut scores = MutSlice::new($scores);
        let scale = Slice::new(&query.scale);
        let bias = Slice::new(&query.bias);
        let sum = Slice::new(&query.scaled_sum);
        for i in 0..$parts {
            let start = i * $lanes;
            // SAFETY: The backend provides exactly MR / lanes vectors. All metadata
            // and scores have MR elements; each access is narrowed to one SIMD vector.
            unsafe {
                let scale = <$float>::load_simd(
                    $arch,
                    scale
                        .add(Elements::new(start))
                        .truncate(Elements::new($lanes))
                        .as_ptr(),
                );
                let bias = <$float>::load_simd(
                    $arch,
                    bias.add(Elements::new(start))
                        .truncate(Elements::new($lanes))
                        .as_ptr(),
                );
                let sum = <$float>::load_simd(
                    $arch,
                    sum.add(Elements::new(start))
                        .truncate(Elements::new($lanes))
                        .as_ptr(),
                );
                let mut output = scores.subslice(start, Bound::new($lanes));
                let mut best = <$float>::load_simd($arch, output.as_ptr());
                for (j, acc) in acc.iter().enumerate() {
                    let doc = docs.add(Elements::new(j)).as_unit().as_ref();
                    let raw = ($convert)(*acc, i);
                    let mut similarity = (scale * <$float>::splat($arch, doc.a)) * raw;
                    similarity = similarity + sum * <$float>::splat($arch, doc.b);
                    similarity = similarity + <$float>::splat($arch, doc.n) * bias;
                    similarity = similarity
                        + (bias * <$float>::splat($arch, doc.b))
                            * <$float>::splat($arch, $dim.value().get() as f32);
                    best = best.min_simd_standard(<$float>::default($arch) - similarity);
                }
                best.store_simd(output.as_mut_ptr());
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use diskann_wide::{
        SIMDDotProduct,
        arch::x86_64::{V3, V4},
    };

    impl ExtraWide<4, 16> for V3 {
        type Query = (
            <V3 as Architecture>::u8x32,
            Option<<V3 as Architecture>::u8x32>,
        );
        type Splat = <V3 as Architecture>::i8x32;
        type Accumulator = [<V3 as Architecture>::i32x8; 2];

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Grouped<4>>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 16);
            // SAFETY: Each half contains eight groups, or 32 bytes.
            unsafe {
                let lo =
                    SIMDVector::load_simd(self, values.truncate(Elements::new(8)).as_ptr().cast());
                let hi = (rows > 8).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(8))
                            .truncate(Elements::new(8))
                            .as_ptr()
                            .cast(),
                    )
                });
                (lo, hi)
            }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn splat(self, value: Grouped<4>) -> Self::Splat {
            diskann_wide::alias!(u32s = <V3>::u32x8);
            Self::Splat::from_underlying(
                self,
                u32s::splat(self, u32::from_le_bytes(*value)).to_underlying(),
            )
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            use std::arch::x86_64::_mm256_maddubs_epi16;
            diskann_wide::alias!(i16s = <V3>::i16x16);
            let dot = |a: <V3 as Architecture>::u8x32, acc: <V3 as Architecture>::i32x8| {
                // SAFETY: V3 provides AVX2. B contains unsigned nibbles, so each
                // pair sum is at most 2 * 255 * 15 and cannot saturate.
                let products = i16s::from_underlying(self, unsafe {
                    _mm256_maddubs_epi16(a.to_underlying(), b.to_underlying())
                });
                acc.dot_simd(products, i16s::splat(self, 1))
            };
            acc[0] = dot(a.0, acc[0]);
            if let Some(hi) = a.1 {
                acc[1] = dot(hi, acc[1]);
            }
            acc
        }
        #[inline(always)]
        unsafe fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<16>,
            docs: Slice<'_, MinMaxCompensation>,
            dim: DimK,
            scores: &mut [f32; 16],
        ) {
            diskann_wide::alias!(floats = <V3>::f32x8);
            let convert = |acc: Self::Accumulator, part: usize| {
                floats::from_array(self, acc[part].to_array().map(|x| x as u32 as f32))
            };
            compensate!(self, floats, 8, 2, acc, convert, query, docs, dim, scores);
        }
    }

    impl ExtraWide<8, 16> for V4 {
        type Query = (
            <V4 as Architecture>::u8x64,
            Option<<V4 as Architecture>::u8x64>,
        );
        type Splat = <V4 as Architecture>::i8x64;
        type Accumulator = [<V4 as Architecture>::i32x16; 2];

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Grouped<8>>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 16);
            // SAFETY: Each half contains eight groups, or 64 bytes.
            unsafe {
                let lo =
                    SIMDVector::load_simd(self, values.truncate(Elements::new(8)).as_ptr().cast());
                let hi = (rows > 8).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(8))
                            .truncate(Elements::new(8))
                            .as_ptr()
                            .cast(),
                    )
                });
                (lo, hi)
            }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn splat(self, value: Grouped<8>) -> Self::Splat {
            #[cfg(miri)]
            {
                Self::Splat::from_array(self, core::array::from_fn(|lane| value[lane % 8] as i8))
            }

            #[cfg(not(miri))]
            {
                diskann_wide::alias!(u64s = <V4>::u64x8);
                Self::Splat::from_underlying(
                    self,
                    u64s::splat(self, u64::from_le_bytes(*value)).to_underlying(),
                )
            }
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            acc[0] = acc[0].dot_simd(a.0, b);
            if let Some(hi) = a.1 {
                acc[1] = acc[1].dot_simd(hi, b);
            }
            acc
        }
        #[inline(always)]
        unsafe fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<16>,
            docs: Slice<'_, MinMaxCompensation>,
            dim: DimK,
            scores: &mut [f32; 16],
        ) {
            diskann_wide::alias!(floats = <V4>::f32x8);
            let convert = |acc: Self::Accumulator, part: usize| {
                #[cfg(miri)]
                {
                    let values = acc[part].to_array();
                    floats::from_array(
                        self,
                        core::array::from_fn(|i| {
                            values[2 * i].wrapping_add(values[2 * i + 1]) as u32 as f32
                        }),
                    )
                }

                #[cfg(not(miri))]
                {
                    use std::arch::x86_64::{
                        _mm512_add_epi32, _mm512_cvtepi64_epi32, _mm512_srli_epi64,
                    };
                    diskann_wide::alias!(i32s = <V4>::i32x8);

                    let value = acc[part].to_underlying();
                    // SAFETY: V4 provides AVX-512F and AVX-512DQ.
                    let pairs = unsafe { _mm512_add_epi32(value, _mm512_srli_epi64::<32>(value)) };
                    // SAFETY: V4 provides AVX-512F and AVX-512DQ.
                    let reduced =
                        i32s::from_underlying(self, unsafe { _mm512_cvtepi64_epi32(pairs) });
                    floats::from_array(self, reduced.to_array().map(|x| x as u32 as f32))
                }
            };
            compensate!(self, floats, 8, 2, acc, convert, query, docs, dim, scores);
        }
    }

    panel_kernel!(V3, 4, 16, 8, [1, 2, 3, 4, 5, 6, 7]);
    panel_kernel!(V4, 8, 16, 8, [1, 2, 3, 4, 5, 6, 7]);
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use diskann_wide::{SIMDDotProduct, arch::aarch64::Neon};

    impl ExtraWide<4, 8> for Neon {
        type Query = (
            <Neon as Architecture>::u8x16,
            Option<<Neon as Architecture>::u8x16>,
        );
        type Splat = <Neon as Architecture>::u8x16;
        type Accumulator = [<Neon as Architecture>::u32x4; 2];

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Grouped<4>>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 8);
            // SAFETY: Each half contains four groups, or 16 bytes.
            unsafe {
                let lo =
                    SIMDVector::load_simd(self, values.truncate(Elements::new(4)).as_ptr().cast());
                let hi = (rows > 4).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(4))
                            .truncate(Elements::new(4))
                            .as_ptr()
                            .cast(),
                    )
                });
                (lo, hi)
            }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn splat(self, value: Grouped<4>) -> Self::Splat {
            Self::Splat::from_array(self, core::array::from_fn(|i| value[i % 4]))
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            acc[0] = acc[0].dot_simd(a.0, b);
            if let Some(hi) = a.1 {
                acc[1] = acc[1].dot_simd(hi, b);
            }
            acc
        }
        #[inline(always)]
        unsafe fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<8>,
            docs: Slice<'_, MinMaxCompensation>,
            dim: DimK,
            scores: &mut [f32; 8],
        ) {
            diskann_wide::alias!(floats = <Neon>::f32x4);
            let convert = |acc: Self::Accumulator, part: usize| {
                floats::from_array(self, acc[part].to_array().map(|x| x as f32))
            };
            compensate!(self, floats, 4, 2, acc, convert, query, docs, dim, scores);
        }
    }

    panel_kernel!(Neon, 4, 8, 8, [1, 2, 3, 4, 5, 6, 7]);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{minmax::DataMutRef, multi_vector::BlockTransposed};
    use diskann_utils::views::Matrix;
    use std::num::NonZeroUsize;

    #[test]
    fn even_odd_groups() {
        for dim in 1..=8 {
            let values: Vec<_> = (0..dim as u8).map(|i| i + 1).collect();
            for group in 0..Grouped::<4>::count(dim) {
                let got = Grouped::<4>::from_query(&values, group);
                let parity = group % 2;
                for i in 0..4 {
                    assert_eq!(got[i], values.get(2 * i + parity).copied().unwrap_or(0));
                }
            }
        }
    }

    fn dimension(value: usize) -> DimK {
        DimK::new(NonZeroUsize::new(value).unwrap())
    }

    fn check_decoded_b<const N: usize>() {
        for dim in (1..=33).chain([249, 250, 256, 257]) {
            for rows in [1, 7, 8, 9, 31, 32, 33] {
                if cfg!(miri) && !(matches!(dim, 1 | 8 | 9) && rows <= 8) {
                    continue;
                }
                let b = documents(rows, dim);
                let groups = Grouped::<N>::count(dim);
                let mut values = vec![Grouped::<N>::default(); rows * groups];
                let mut meta = vec![MinMaxCompensation::default(); rows];
                // SAFETY: The fixture contains canonical rows and exact scratch extents.
                let decoded = unsafe {
                    DecodedB::new(
                        unpacked::View::from_matrix_view(b.as_view()).unwrap(),
                        dimension(Data::<4>::canonical_bytes(dim)),
                        dimension(dim),
                        dimension(groups),
                        &mut values,
                        &mut meta,
                    )
                };
                // SAFETY: The decoder initialized rows * groups elements.
                let values = unsafe { decoded.values.as_std_slice(dimension(groups)) };
                for row in 0..rows {
                    // SAFETY: The decoder retained one metadata entry per row.
                    let meta = unsafe { decoded.meta.add(Elements::new(row)).as_unit().as_ref() };
                    assert_eq!(meta.a, (row % 4 + 1) as f32 * 0.25);
                    assert_eq!(meta.b, (row % 3) as f32 - 1.0);
                    assert_eq!(meta.n, (row % 5) as f32 * 0.5);
                    for group in 0..groups {
                        for (lane, &value) in values[row * groups + group].iter().enumerate() {
                            let d = if N == 4 {
                                group / 2 * 8 + 2 * lane + group % 2
                            } else {
                                group * 8 + lane
                            };
                            let expected = if d < dim {
                                ((row * 7 + d * 3 + 1) % 16) as u8
                            } else {
                                0
                            };
                            assert_eq!(value, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn decoded_b_groups_and_metadata() {
        check_decoded_b::<4>();
    }

    fn documents(rows: usize, dim: usize) -> Matrix<u8> {
        let stride = Data::<4>::canonical_bytes(dim);
        let mut bytes = Matrix::new(0_u8, rows, stride);
        for (i, bytes) in bytes.as_mut_slice().chunks_exact_mut(stride).enumerate() {
            let mut row = DataMutRef::<4>::from_canonical_front_mut(bytes, dim).unwrap();
            row.set_meta(MinMaxCompensation {
                a: (i % 4 + 1) as f32 * 0.25,
                b: (i % 3) as f32 - 1.0,
                n: (i % 5) as f32 * 0.5,
                dim: dim as u32,
                ..Default::default()
            });
            for d in 0..dim {
                row.vector_mut()
                    .set(d, ((i * 7 + d * 3 + 1) % 16) as i64)
                    .unwrap();
            }
        }
        bytes
    }

    #[test]
    fn invalid_driver_bounds_are_detected() {
        use crate::matrix_kernels::test_util::panic_message_for;

        let values = BlockTransposed::<Grouped<4>, 8>::new(8, 4);
        let docs = documents(2, 9);
        for (dim, metadata_rows, output_rows) in [(8, 1, 8), (9, 0, 8), (9, 1, 0), (9, 1, 9)] {
            let _ = panic_message_for(|| {
                let metadata = vec![QueryCompensation::default(); metadata_rows];
                let mut scores = vec![0.0; output_rows];
                // SAFETY: In test builds new_inner checks every supplied size relationship
                // before creating or dereferencing any derived spans.
                let _ = unsafe {
                    Driver::<_, 4, 8, 6>::new_inner(
                        Scalar::new(),
                        packed::View::from_block_transposed(values.as_view()).unwrap(),
                        &metadata,
                        unpacked::View::from_matrix_view(docs.as_view()).unwrap(),
                        &mut scores,
                        dimension(dim),
                    )
                };
            });
        }
    }

    fn check_driver<A, const N: usize, const MR: usize, const NR: usize>(arch: A)
    where
        A: Architecture + ExtraWide<N, MR> + util::LoadStore<f32, MR>,
        for<'a> Driver<'a, A, N, MR, NR>: driver::Drive,
    {
        let scratch_dimension = 8 * (B_TILE_GROUPS / NR / (8 / N));
        let cases = super::super::test::packed_x_unpacked_test_dims(MR, NR)
            .into_iter()
            .map(|c| (c.total_a_rows, c.total_b_cols, c.k))
            .chain((1..=NR).flat_map(|n| (1..=17).map(move |dim| (MR + 1, n + NR, dim))))
            .chain([
                (MR + 1, 32, 1),
                (MR + 1, 33, 1),
                (MR + 1, 16, 256),
                (MR + 1, 17, 256),
                (MR + 1, 16, 257),
                (MR + 1, B_TILE_ROWS - 1, 250),
                (MR + 1, B_TILE_ROWS, 250),
                (MR + 1, B_TILE_ROWS + 1, 250),
                (MR + 1, 2 * NR + 1, scratch_dimension),
                (MR + 1, 2 * NR + 1, scratch_dimension + 1),
            ]);
        for (rows, cols, dim) in cases {
            if cfg!(miri)
                && !(rows <= MR + 1 && cols <= NR + 1 && matches!(dim, 1 | 9)
                    || dim == 1 && matches!(cols, 32 | 33))
            {
                continue;
            }
            let k = Grouped::<N>::count(dim);
            let mut grouped = Matrix::new(Grouped::<N>::default(), rows, k);
            let mut query = vec![QueryCompensation::<MR>::default(); rows.div_ceil(MR)];
            for row in 0..rows {
                let values: Vec<_> = (0..dim)
                    .map(|d| ((row * 17 + d * 3 + 1) % 256) as u8)
                    .collect();
                for group in 0..k {
                    grouped.as_mut_slice()[row * k + group] =
                        Grouped::<N>::from_query(&values, group);
                }
                let q = &mut query[row / MR];
                q.scale[row % MR] = (row % 3 + 1) as f32 * 0.5;
                q.bias[row % MR] = (row % 5) as f32 - 2.0;
                q.scaled_sum[row % MR] = (row % 7) as f32;
            }
            let a = BlockTransposed::<_, MR>::from_matrix_view(grouped.as_view());
            let b = documents(cols, dim);
            let expected: Vec<f32> = (0..rows)
                .map(|i| {
                    let q = &query[i / MR];
                    let lane = i % MR;
                    let mut best = f32::MAX;
                    for j in 0..cols {
                        let doc = DataRef::<4>::from_canonical_front(b.row(j), dim)
                            .unwrap()
                            .meta();
                        let raw: u32 = (0..dim)
                            .map(|d| {
                                ((i * 17 + d * 3 + 1) % 256) as u32
                                    * ((j * 7 + d * 3 + 1) % 16) as u32
                            })
                            .sum();
                        let similarity = q.scale[lane] * doc.a * raw as f32
                            + q.scaled_sum[lane] * doc.b
                            + doc.n * q.bias[lane]
                            + q.bias[lane] * doc.b * dim as f32;
                        best = best.min(-similarity);
                    }
                    best
                })
                .collect();
            let mut scores = vec![12345.0; rows + 2];
            // SAFETY: The fixture supplies matching packed groups, metadata and output size.
            let mut driver = unsafe {
                Driver::<_, N, MR, NR>::new_inner(
                    arch,
                    packed::View::from_block_transposed(a.as_view()).unwrap(),
                    &query,
                    unpacked::View::from_matrix_view(b.as_view()).unwrap(),
                    &mut scores[1..rows + 1],
                    dimension(dim),
                )
            };
            driver::Drive::drive(&mut driver);
            driver::Drive::drive(&mut driver);
            assert_eq!(scores[0], 12345.0);
            assert_eq!(scores[rows + 1], 12345.0);
            assert_eq!(&scores[1..rows + 1], expected, "({rows},{cols},{dim})");
        }
    }

    fn check_registers<A, const N: usize, const MR: usize>(arch: A)
    where
        A: Architecture + ExtraWide<N, MR>,
    {
        arch.run_inline(|| {
            for rows in 1..=MR {
                for pattern in 0..3 {
                    let b = Grouped(core::array::from_fn(|lane| match pattern {
                        0 => 0,
                        1 => 15,
                        _ => [1, 7, 3, 15, 2, 9, 4, 12][lane],
                    }));
                    let values: [Grouped<N>; MR] = core::array::from_fn(|i| {
                        Grouped(core::array::from_fn(|j| {
                            if i.is_multiple_of(2) {
                                255
                            } else {
                                (i * 11 + j * 13) as u8
                            }
                        }))
                    });
                    // SAFETY: The span has exactly MR groups and rows <= MR.
                    let a = unsafe { arch.load(Slice::new(&values), rows) };
                    let mut acc = arch.zero();
                    for _ in 0..3 {
                        acc = arch.dot(a, arch.splat(b), acc);
                    }
                    let query = QueryCompensation {
                        scale: [0.5; MR],
                        bias: [-2.0; MR],
                        scaled_sum: [7.0; MR],
                    };
                    let doc = MinMaxCompensation {
                        a: 0.25,
                        b: 3.0,
                        n: 11.0,
                        ..Default::default()
                    };
                    let mut scores = [f32::MAX; MR];
                    let group_dim = N;
                    // SAFETY: One accumulator has exactly one metadata entry.
                    unsafe {
                        arch.reduce(
                            [acc],
                            &query,
                            Slice::new(&[doc]),
                            dimension(group_dim),
                            &mut scores,
                        )
                    };
                    for i in 0..rows {
                        let raw: u32 = values[i]
                            .iter()
                            .zip(b.iter())
                            .map(|(&a, &b)| u32::from(a) * u32::from(b))
                            .sum::<u32>()
                            * 3;
                        let expected = -(0.5 * 0.25 * raw as f32
                            + 7.0 * 3.0
                            + 11.0 * -2.0
                            + -2.0 * 3.0 * group_dim as f32);
                        assert_eq!(scores[i], expected);
                    }
                    let previous = scores;
                    let nan = MinMaxCompensation { a: f32::NAN, ..doc };
                    // SAFETY: One accumulator has exactly one metadata entry.
                    unsafe {
                        arch.reduce(
                            [acc],
                            &query,
                            Slice::new(&[nan]),
                            dimension(group_dim),
                            &mut scores,
                        )
                    };
                    assert_eq!(
                        scores, previous,
                        "NaN compensation must not discard prior scores"
                    );
                }
            }
        });
    }

    #[test]
    fn scalar_driver_and_registers() {
        check_driver::<_, 4, 8, 6>(Scalar::new());
        check_registers::<_, 4, 8>(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            check_driver::<_, 4, 16, 8>(arch);
            check_registers::<_, 4, 16>(arch);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
            arch.run_inline(check_decoded_b::<8>);
            check_driver::<_, 8, 16, 8>(arch);
            check_registers::<_, 8, 16>(arch);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_driver::<_, 4, 8, 8>(arch);
            check_registers::<_, 4, 8>(arch);
        }
    }
}
