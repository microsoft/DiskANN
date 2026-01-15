/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::{Display, Error, Formatter};

/// A struct used to lazily defer creation of custom async logging messages until we know
/// that the message is actually needed.
///
/// # Context
///
/// Logging in the async context explicitly requires passing of a context pointer to enable
/// CDB to determine the source of error message. To that end, a custom logging function is
/// used.
///
/// The `LazyString` captures a lambda that constructs the logging message and implements
/// `std::fmt::Display`, allowing string formatting to only be performed once we know a
/// message needs to be logged.
pub struct LazyString<F>(F)
where
    F: Fn(&mut Formatter<'_>) -> Result<(), Error>;

impl<F> LazyString<F>
where
    F: Fn(&mut Formatter<'_>) -> Result<(), Error>,
{
    /// Construct a new `LazyString` around the provided lambda.
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

impl<F> Display for LazyString<F>
where
    F: Fn(&mut Formatter<'_>) -> Result<(), Error>,
{
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        (self.0)(f)
    }
}

/// A macro that behaves like `format!` but constructs a `diskann::utils::LazyString`
/// to enable deferred evaluation of the error message.
///
/// Invoking this macro has the following equivalence:
/// ```ignore
/// let a: f32 = 10.5;
/// let b: usize = 20;
/// // Macro form
/// let lazy_from_macro = lazy_format("This is a test. A = {}, B = {}", a, b);
///
/// // Direct form
/// let lazy_direct = crate::utils::LazyString::new(|f: &mut std::fmt::Formatter<'_>| {
///     write!(f, "ihis is a test. A = {}, B = {}", a, b)
/// });
/// ```
#[macro_export]
macro_rules! lazy_format {
    ($($arg:tt)*) => {
        // Must be a full path and only available inside `DiskANN`.
        $crate::LazyString::new(|f: &mut std::fmt::Formatter<'_>| {
            write!(f, $($arg)*)
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_lazy_string() {
        let x: f32 = 10.5;
        let y: usize = 20;

        let lazy = LazyString::new(|f: &mut std::fmt::Formatter| {
            write!(f, "Lazy Message: x = {x}, y = {y}")
        });
        assert_eq!(lazy.to_string(), "Lazy Message: x = 10.5, y = 20");

        let lazy = lazy_format!("Lazy Message: x = {x}, y = {y}");
        assert_eq!(lazy.to_string(), "Lazy Message: x = 10.5, y = 20");
    }
}
