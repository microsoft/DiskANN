/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Display;

pub(crate) trait Dimensionality: Display {}

/// Dynamic dimensionality (distance function not specialized on dimensionality).
#[derive(Clone, Copy)]
pub struct Dynamic(usize);

impl Dimensionality for Dynamic {}
impl Display for Dynamic {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "dynamic<{}>", self.0)
    }
}

/// Static dimensionality (specialized on a compile-time dimension).
#[derive(Clone, Copy, Default, Debug)]
pub struct Static<const N: usize>;

impl<const N: usize> Dimensionality for Static<N> {}
impl<const N: usize> Display for Static<N> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "static<{}>", N)
    }
}

impl<const N: usize> Static<N> {
    pub const fn get() -> usize {
        N
    }

    /// Return the `Dynamic` equivalent of the `Static`.
    pub fn dynamic(&self) -> Dynamic {
        Dynamic(N)
    }
}

impl<const N: usize> From<Static<N>> for usize {
    fn from(_: Static<N>) -> Self {
        Static::<N>::get()
    }
}

/// Create a name for benchmark functions within a group that has a standard format for
/// easier processing by tooling.
///
/// The expected format is:
/// ```ignore
/// dimtype<dim>-[aligned/unaligned]
/// ```
/// where
/// * `dimtype`: The dimensionality specialization of implementation.
///   - `dynamic`: The length of the vectors being processed is not provided at compile time.
///   - `static`: The length of the vectors is provided a compile-time
///
/// * `dim`: The actual vector dimension.
///
/// * `aligned/unaligned`: Whether the vector pointers provided are *all* 32-byte aligned
///   or not.
#[allow(dead_code)] // benchmark_iai doesn't use this ...
pub(crate) fn format_benchmark<Dim>(dim: Dim, aligned: bool) -> String
where
    Dim: Dimensionality,
{
    let align = if aligned { "aligned" } else { "unaligned" };
    format!("{}-{}", dim, align)
}
