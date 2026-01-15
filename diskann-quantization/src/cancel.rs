/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Utilities to support cancelation of long-running operations.

use std::sync::atomic::{AtomicBool, Ordering};

/// Provides a means for cancelling long-running operations.
///
/// Reasonable implementation of this trait should ensure that once `should_cancel` returns
/// `true`, all future calls to `should_cancel` should **also** return `true`.
pub trait Cancelation {
    fn should_cancel(&self) -> bool;
}

/// A light-weight cancelation token based on an `AtomicBool`.
pub struct AtomicCancelation<'a>(&'a AtomicBool);

impl<'a> AtomicCancelation<'a> {
    pub fn new(val: &'a AtomicBool) -> Self {
        Self(val)
    }
}

impl Cancelation for AtomicCancelation<'_> {
    fn should_cancel(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// A no-op cancelation token.
pub struct DontCancel;

impl Cancelation for DontCancel {
    fn should_cancel(&self) -> bool {
        false
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_cancelation() {
        let x = AtomicBool::new(false);
        let y = AtomicCancelation::new(&x);
        assert!(!y.should_cancel());
        x.store(true, Ordering::Relaxed);
        assert!(y.should_cancel());
    }

    #[test]
    fn test_no_cancelation() {
        let x = DontCancel;
        assert!(!x.should_cancel());
    }
}
