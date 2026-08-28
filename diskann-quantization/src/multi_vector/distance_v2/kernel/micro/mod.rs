/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

mod maxsim;

pub(super) use maxsim::MaxSim;

pub(super) trait Kernel<A, B, C> {
    fn kernel(&self, a: A, b: B, cols: NonZeroUsize, c: C);
}
