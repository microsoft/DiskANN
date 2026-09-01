/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) trait Drive {
    fn drive(&mut self);
}

pub(super) trait PanelKernel {
    fn panel_kernel(&mut self);
}

pub(super) trait MicroKernel {
    fn micro_kernel(&mut self);
}

