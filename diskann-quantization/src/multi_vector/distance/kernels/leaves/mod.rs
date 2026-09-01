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
//! Every leaf completes the inner products for one panel pair and folds them directly
//! into the running per-A-row maximum.

pub(super) mod scalar;
#[cfg(target_arch = "x86_64")]
pub(super) mod v3;
