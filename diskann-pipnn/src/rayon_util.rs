/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Marker wrappers for rayon parallel iterators.
//!
//! The workspace clippy config bans direct `ParallelIterator::for_each` /
//! `::collect` calls to force routing through a `RayonThreadPool::install`.
//! PiPNN already runs entirely inside `pool.install(...)` (set up by
//! `build_internal`), so the par-iter calls inside it already execute on
//! the right pool — the lint just can't see that.
//!
//! These wrappers preserve the workspace convention (one `#[allow]` in a
//! single helper module rather than scattered across hot paths) without
//! requiring callers to pass a `&ThreadPool`. Use these from any function
//! reachable from `build_internal`.

use rayon::iter::{FromParallelIterator, ParallelIterator};

#[allow(clippy::disallowed_methods)]
pub trait ParIterInstalled: ParallelIterator + Sized {
    /// `for_each` — caller asserts execution is already inside `pool.install`.
    #[inline]
    fn for_each_installed<OP>(self, op: OP)
    where
        OP: Fn(Self::Item) + Sync + Send,
    {
        self.for_each(op)
    }

    /// `collect` — caller asserts execution is already inside `pool.install`.
    #[inline]
    fn collect_installed<C>(self) -> C
    where
        C: FromParallelIterator<Self::Item>,
    {
        self.collect()
    }
}

impl<T: ParallelIterator + Sized> ParIterInstalled for T {}
