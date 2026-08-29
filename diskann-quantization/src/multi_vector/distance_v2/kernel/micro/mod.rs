/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

mod maxsim;

pub(crate) use maxsim::MaxSim;

pub(crate) trait Kernel<A, B, C> {
    fn kernel(&self, a: A, b: B, cols: NonZeroUsize, c: C);
}
