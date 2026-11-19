/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// A `bool` that is marked as must use, allowing it to be embedded inside [`Result`]s or
/// [`std::future::Future`]s without losing it's must-use-ness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
#[must_use = "this result must be used"]
pub struct Bool(bool);

impl Bool {
    #[inline]
    pub fn new(value: bool) -> Self {
        Self(value)
    }

    #[inline]
    pub fn get(self) -> bool {
        self.0
    }
}

impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        Self(value)
    }
}

impl From<Bool> for bool {
    fn from(value: Bool) -> bool {
        value.get()
    }
}
