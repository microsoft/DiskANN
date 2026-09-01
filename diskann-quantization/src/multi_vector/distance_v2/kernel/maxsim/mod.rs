/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod packed_f32_x_unpacked_f16;
pub(crate) mod packed_f32_x_unpacked_f32;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MaxSim<A>(A);

impl<A> MaxSim<A> {
    pub(crate) fn new(arch: A) -> Self {
        Self(arch)
    }

    fn arch(self) -> A {
        self.0
    }
}
