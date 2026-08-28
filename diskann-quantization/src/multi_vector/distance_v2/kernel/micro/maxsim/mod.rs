/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod f32;

#[derive(Debug, Clone, Copy)]
pub(super) struct MaxSim<A>(A);

impl<A> MaxSim<A> {
    fn new(arch: A) -> Self {
        Self(arch)
    }
}

