// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Per-ISA micro-kernels.
//!
//! Panel geometry is fixed per ISA rather than derived from a lane count: the scalar leaf
//! is deliberately narrower than `2 × LANES` would suggest, because its "lanes" are a
//! loop, not silicon.
//!
//! Every leaf here **stores** its products into a slot and leaves the reduction to
//! `fold_columns`, which folds along independent max chains so the reduction does not
//! serialize on its own latency — it is not negligible work when the strip is wide and
//! shallow.

pub(super) mod scalar;
#[cfg(target_arch = "x86_64")]
pub(super) mod v3;

/// Independent max chains in `fold_columns`: enough to keep a multi-cycle max off its own
/// critical path, few enough that the chains stay in registers.
const WAYS: usize = 4;
