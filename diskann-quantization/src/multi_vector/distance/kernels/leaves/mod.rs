/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-ISA micro-kernels.
//!
//! Panel geometry is fixed per ISA, not derived from a lane count. The scalar leaf is
//! deliberately narrower than `2 × LANES` would suggest, because its "lanes" are a
//! loop, not silicon.
//!
//! Every leaf here **stores** its products into a slot and leaves the reduction to
//! `max_into_rows`. That reduction is not negligible work when the strip is wide and
//! shallow, so it runs several independent max chains rather than serializing on the
//! latency of a single one.

pub(super) mod scalar;
#[cfg(target_arch = "x86_64")]
pub(super) mod v3;

/// Independent max chains in `max_into_rows`: enough to keep a multi-cycle max off its own
/// critical path, few enough that the chains stay in registers.
const WAYS: usize = 4;
