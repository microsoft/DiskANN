/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Implementation of "maxsim" kernels.
//!
//! Let:
//!
//! * `A` be a `M x K` matrix.
//! * `B` be a `K x N` matrix.
//!
//! The "maxsim" result `C` is a `M`-dimensional vector where
//!
//! ```text
//! C[i] = max(j in 0..N, dot(A[i, :], B[:, j]))
//! ```

pub(crate) mod minmax8_x_minmax4;
pub(crate) mod packed_f32_x_unpacked_f16;
pub(crate) mod packed_f32_x_unpacked_f32;

#[cfg(test)]
mod test;
