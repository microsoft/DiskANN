/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

#[derive(Debug, Clone, Copy, Default, Error)]
#[error(transparent)]
#[repr(transparent)]
// A wrapper type to bridge errors from one type to another.
// We need this to convert errors from other crates into ANNError for interop
pub struct Bridge<T>(pub T);

impl<T> Bridge<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

pub trait BridgeErr<T, E> {
    fn bridge_err(self) -> Result<T, Bridge<E>>;
}

impl<T, E> BridgeErr<T, E> for Result<T, E> {
    fn bridge_err(self) -> Result<T, Bridge<E>> {
        self.map_err(Bridge)
    }
}
