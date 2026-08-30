/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::multi_vector::distance_v2::num::DimK;

pub(crate) mod maxsim;

pub(crate) trait PanelKernel {
    fn panel_kernel(&mut self);
}

pub(crate) trait MicroKernel<A, B, C> {
    fn kernel(&self, a: A, b: B, k: DimK, c: C);
}
