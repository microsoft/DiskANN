/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use std::fmt::Debug;

use crate::num::IdLimit;

use super::Lifecycle;

pub(crate) trait PluginConfig: Debug {
    type Plugin: Plugin;
    fn build(self, id_limit: IdLimit) -> ANNResult<Self::Plugin>;
}

pub(crate) trait Plugin: Debug + 'static {
    type Slot<'a>: Slot;
    fn id_limit(&self) -> IdLimit;

    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Self::Slot<'_>;
    unsafe fn reclaim(&self, i: u32, _: Lifecycle);
    unsafe fn retire(&self, i: u32, _: Lifecycle);
}

pub(crate) trait Slot: Debug {
    fn publish(self, _: Lifecycle);
    fn freeze(self, _: Lifecycle);
    fn abort(self, _: Lifecycle);
}

