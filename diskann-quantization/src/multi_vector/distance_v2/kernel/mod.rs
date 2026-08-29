/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// pub(crate) mod micro;
// pub(crate) mod panel;

pub(crate) mod maxsim;

pub(crate) trait Kernel {
    fn run(&mut self);
}

pub(crate) trait MicroKernel<A, B, C> {
    fn kernel(&self, a: A, b: B, cols: std::num::NonZeroUsize, c: C);
}
