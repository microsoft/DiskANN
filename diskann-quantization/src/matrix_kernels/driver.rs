/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Drive a top level matrix kernel.
pub(crate) trait Drive {
    fn drive(&mut self);
}

/// A panel-level operation. Typically invoked multiple times in [`Drive`].
pub(super) trait PanelKernel {
    fn panel_kernel(&mut self);
}

/// A micro-kernel operation. Typically invoked multiple times in [`PanelKernel`].
pub(super) trait MicroKernel {
    fn micro_kernel(&mut self);
}
