/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::{Display, Error, Formatter};

/// A macro that behaves like `format!` but constructs a [`LazyString`] to defer string
/// formatting if [`Display`] is unused.
///
/// ```rust
/// use diskann_utils::lazy_format;
///
/// let a: f32 = 10.5;
/// let b: usize = 20;
///
/// let lazy_string = lazy_format!("This is a test. A = {}, B = {}", a, b);
/// assert_eq!(lazy_string.to_string(), "This is a test. A = 10.5, B = 20");
///
/// // Formatting of captured members is deferred until the created `LazyString` is formatted.
/// #[derive(Default)]
/// struct Formatted(std::cell::Cell<bool>);
///
/// impl Formatted {
///     fn was_formatted(&self) -> bool {
///         self.0.get()
///     }
///
///     fn mark_as_formatted(&self) {
///         self.0.set(true)
///     }
/// }
///
/// impl std::fmt::Display for Formatted {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
///         if self.was_formatted() {
///             f.write_str("yes")
///         } else {
///             self.mark_as_formatted();
///             f.write_str("not yet")
///         }
///     }
/// }
///
/// let f = Formatted::default();
/// let lazy = lazy_format!("Was this formatted: {}", f);
///
/// assert!(!f.was_formatted(), "string formatting should be deferred");
/// assert_eq!(lazy.to_string(), "Was this formatted: not yet");
///
/// assert!(f.was_formatted());
/// assert_eq!(lazy.to_string(), "Was this formatted: yes");
/// ```
#[macro_export]
macro_rules! lazy_format {
    ($($arg:tt)*) => {
        $crate::LazyString::new(|f: &mut std::fmt::Formatter<'_>| {
            write!(f, $($arg)*)
        })
    }
}

/// A struct used to lazily defer string formatting until needed. This is used to implement
/// [`lazy_format`]: a lazy version of the standard `format!` macro.
///
/// See [`lazy_format`] for usage.
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
