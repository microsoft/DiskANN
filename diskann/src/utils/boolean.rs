/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// A must-use boolean that warns when discarded, including after being extracted from
/// a [`Result`] or returned by a [`std::future::Future`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
#[must_use = "this value must be used"]
pub struct Bool(bool);

impl Bool {
    /// Create a new [`Bool`].
    #[inline]
    pub const fn new(value: bool) -> Self {
        Self(value)
    }

    /// Return the contained value.
    #[inline]
    pub const fn get(self) -> bool {
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool() {
        let v = Bool::new(true);
        assert!(v.get());
        assert!(bool::from(v));

        let v = Bool::from(true);
        assert!(v.get());
        assert!(bool::from(v));

        let v = Bool::new(false);
        assert!(!v.get());
        assert!(!bool::from(v));

        let v = Bool::from(false);
        assert!(!v.get());
        assert!(!bool::from(v));
    }
}
