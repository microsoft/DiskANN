// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f16 MaxSim.
//!
//! There is no f16 leaf: f16 widens to f32 and reuses the f32 pipeline. Both sides widen a
//! tile at a time into a buffer the walk reuses, which is what the lending [`TileWalk`]
//! exists for — the whole query never has to be staged at once, and the staged copy stays
//! inside the cache level its tile was sized for.

use diskann_vector::conversion::SliceCast;
use diskann_wide::Architecture;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::V3;
use diskann_wide::arch::{Scalar, Target2};

use super::leaves::scalar::{A_PANEL as SC_A, B_PANEL as SC_B};
#[cfg(target_arch = "x86_64")]
use super::leaves::v3::{A_PANEL as V3_A, B_PANEL as V3_B};
use super::tiles::{Cursor, DocTile, QueryTile};
use super::{Plan, TileAt, TileBudget, TileWalk, float};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

/// Stages one source tile at a time as f32.
struct Widen<'a, Arch> {
    arch: Arch,
    cursor: Cursor<'a, half::f16>,
    buf: Vec<f32>,
    k: usize,
}

impl<'a, Arch: Architecture> Widen<'a, Arch>
where
    SliceCast<f32, half::f16>: for<'x> Target2<Arch, (), &'x mut [f32], &'x [half::f16]>,
{
    /// # Panics
    ///
    /// Panics if `k` is zero — the entry guards that case before any walk is built.
    fn new(arch: Arch, src: &'a [half::f16], k: usize, stride: usize) -> Self {
        assert!(k > 0, "widening walk requires a non-empty contraction");
        let cursor = Cursor::new(src, stride);
        let buf = vec![0.0f32; cursor.widest()];
        Self {
            arch,
            cursor,
            buf,
            k,
        }
    }

    fn next(&mut self) -> Option<&[f32]> {
        let arch = self.arch;
        let src = self.cursor.next()?;
        let len = src.len();
        arch.run2(SliceCast::new(), &mut self.buf[..len], src);
        Some(&self.buf[..len])
    }
}

/// Widens the padded storage of a block-transposed f16 query.
///
/// Widening is element-wise, so it preserves the block-transposed permutation.
struct QueryWiden<'a, Arch, const AR: usize>(Widen<'a, Arch>);

impl<'a, Arch: Architecture, const AR: usize> QueryWiden<'a, Arch, AR>
where
    SliceCast<f32, half::f16>: for<'x> Target2<Arch, (), &'x mut [f32], &'x [half::f16]>,
{
    fn new(arch: Arch, view: BlockTransposedRef<'a, half::f16, AR>, a_panels: usize) -> Self {
        let k = view.padded_ncols();
        Self(Widen::new(arch, view.as_slice(), k, a_panels * AR * k))
    }
}

impl<'t, Arch, const AR: usize> TileAt<'t> for QueryWiden<'_, Arch, AR> {
    type Tile = QueryTile<'t, f32, AR>;
}

impl<Arch: Architecture, const AR: usize> TileWalk for QueryWiden<'_, Arch, AR>
where
    SliceCast<f32, half::f16>: for<'x> Target2<Arch, (), &'x mut [f32], &'x [half::f16]>,
{
    fn next(&mut self) -> Option<QueryTile<'_, f32, AR>> {
        let k = self.0.k;
        self.0.next().map(|data| QueryTile::new(data, k))
    }

    fn reset(&mut self) {
        self.0.cursor.reset();
    }
}

/// Widens a row-major f16 doc matrix.
struct DocWiden<'a, Arch, const BR: usize>(Widen<'a, Arch>);

impl<'a, Arch: Architecture, const BR: usize> DocWiden<'a, Arch, BR>
where
    SliceCast<f32, half::f16>: for<'x> Target2<Arch, (), &'x mut [f32], &'x [half::f16]>,
{
    fn new(arch: Arch, docs: MatRef<'a, Standard<half::f16>>, b_panels: usize) -> Self {
        let k = docs.vector_dim();
        Self(Widen::new(arch, docs.as_slice(), k, b_panels * BR * k))
    }
}

impl<'t, Arch, const BR: usize> TileAt<'t> for DocWiden<'_, Arch, BR> {
    type Tile = DocTile<'t, f32, BR>;
}

impl<Arch: Architecture, const BR: usize> TileWalk for DocWiden<'_, Arch, BR>
where
    SliceCast<f32, half::f16>: for<'x> Target2<Arch, (), &'x mut [f32], &'x [half::f16]>,
{
    fn next(&mut self) -> Option<DocTile<'_, f32, BR>> {
        let k = self.0.k;
        self.0.next().map(|data| DocTile::new(data, k))
    }

    fn reset(&mut self) {
        self.0.cursor.reset();
    }
}

// ── Entry ────────────────────────────────────────────────────────

/// The f16 MaxSim entry — the f32 pipeline behind widening walks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MaxIpF16;

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target3<
        V3,
        (),
        BlockTransposedRef<'_, half::f16, V3_A>,
        MatRef<'_, Standard<half::f16>>,
        &mut [f32],
    > for MaxIpF16
{
    #[inline(always)]
    fn run(
        self,
        arch: V3,
        query: BlockTransposedRef<'_, half::f16, V3_A>,
        docs: MatRef<'_, Standard<half::f16>>,
        state: &mut [f32],
    ) {
        float::run(
            arch,
            docs.num_vectors(),
            query.padded_ncols(),
            TileBudget::default(),
            state,
            |plan: Plan<V3_A, V3_B>| {
                (
                    QueryWiden::new(arch, query, plan.a_panels),
                    DocWiden::new(arch, docs, plan.b_panels),
                )
            },
        );
    }
}

impl
    diskann_wide::arch::Target3<
        Scalar,
        (),
        BlockTransposedRef<'_, half::f16, SC_A>,
        MatRef<'_, Standard<half::f16>>,
        &mut [f32],
    > for MaxIpF16
{
    #[inline(always)]
    fn run(
        self,
        arch: Scalar,
        query: BlockTransposedRef<'_, half::f16, SC_A>,
        docs: MatRef<'_, Standard<half::f16>>,
        state: &mut [f32],
    ) {
        float::run(
            arch,
            docs.num_vectors(),
            query.padded_ncols(),
            TileBudget::default(),
            state,
            |plan: Plan<SC_A, SC_B>| {
                (
                    QueryWiden::new(arch, query, plan.a_panels),
                    DocWiden::new(arch, docs, plan.b_panels),
                )
            },
        );
    }
}
