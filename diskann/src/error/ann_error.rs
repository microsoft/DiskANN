/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    alloc::LayoutError,
    array::TryFromSliceError,
    fmt::{Debug, Display},
    io,
    num::TryFromIntError,
    sync::mpsc,
};

use crate::always_escalate;

/// Convenience alias for a `Result<T, ANNError>`.
pub type ANNResult<T> = Result<T, ANNError>;

/// Common error type shared through DiskANN.
///
/// This type disambiguates the runtime origin of errors using the `kind()` enum. Third
/// party implementations of DiskANN plugin types like provider can use `kind()` and the
/// downcasting API to throw custom errors from low in the callstack and retrieve those
/// errors higher in the stack.
/// ```rust
/// use diskann::{ANNError, error::ErrorContext};
/// use thiserror::Error;
///
/// // A custom error type used by a third-party.
/// #[derive(Debug, Error)]
/// #[error("custom error: {0}")]
/// struct CustomError(usize);
///
/// // A low-level function that returns an error.
/// fn errors() -> Result<(), ANNError> {
///     Err(ANNError::new(CustomError(42)))
/// }
///
/// // A function that propagates an error, adding context.
/// fn propagates_with_context() -> Result<(), ANNError> {
///     errors().context("propagated")
/// }
///
/// // Call a function that returns a contextual error.
/// let err = propagates_with_context().unwrap_err();
///
/// // The formatted error will contain the base error and all contexts.
/// let message = err.to_string();
/// assert!(message.contains("custom error: 42"));
/// assert!(message.contains("propagated"));
///
/// // If we know the concrete error type, we can downcast the error.
/// let downcasted = err.downcast_ref::<CustomError>().unwrap();
/// assert_eq!(downcasted.0, 42);
/// ```
///
/// # Backtraces
///
/// Backtraces will be obtained upon the first construction of an `ANNError` if the
/// environment variable `RUST_BACKTRACE=1` is set.
///
/// Backtrace collection adds a time overhead to error collection.
///
/// # Properties
///
/// `ANNError` has the following properties to support efficiency:
///
/// * `std::mem::size_of::<ANNError>() == 8`: The struct is 8 bytes. This allows it to be
///   returned in registers rather than on the stack.
/// * `std::mem::size_of::<Option<ANNError>>() == 8`: The struct can use Rust's niche
///   optimization.
#[derive(Debug)]
pub struct ANNError {
    error: anyhow::Error,
}

impl ANNError {
    /// Construct a new `ANNError` encapsulating `err`.
    ///
    /// Errors constructed this way can be retrieved using downcasting.
    /// ```rust
    /// use diskann::{ANNError};
    /// use std::env::VarError;
    ///
    /// let err = ANNError::new(
    ///     VarError::NotPresent,
    /// );
    ///
    /// let retrieved: VarError = err.downcast::<VarError>().unwrap();
    /// ```
    ///
    /// # Attributes
    ///
    /// - `track_caller`: Internally, the type `err` is embedded inside a `Located` struct,
    ///   recording the file and line of creation. The `[track_caller]` attribute allows
    ///   for precise recording of the caller.
    ///
    /// - `inline(never)`: To keep the happy-path cost as minimal as possible, this function
    ///   is marked as `[inline(never)]` to outline error handling code.
    #[track_caller]
    #[inline(never)]
    pub fn new<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            error: anyhow::Error::new(Located::new(err)),
        }
    }

    /// Construct a new `ANNError` with the provided error message.
    ///
    /// # Note
    ///
    /// Errors constructed this way are not necessarily recoverable by using the
    /// downcasting API.
    ///
    /// # Attributes
    ///
    /// - `track_caller`: Internally, the type `err` is embeded inside a `Located` struct,
    ///   recording the file and line of creation. The `[track_caller]` attribute allows
    ///   for precise recording of the caller.
    ///
    /// - `inline(never)`: To keep the happy-path cost as minimal as possible, this function
    ///   is marked as `[inline(never)]` to outline error handling code.
    #[track_caller]
    #[inline(never)]
    pub fn message<D>(display: D) -> Self
    where
        D: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    {
        Self {
            error: anyhow::Error::msg(Located::new(display)),
        }
    }

    #[must_use]
    pub fn is<E>(&self) -> bool
    where
        E: Display + Debug + Send + Sync + 'static,
    {
        self.error.is::<E>()
    }

    /// Attempt to downcast the error object to a concrete type.
    pub fn downcast<E>(self) -> Result<E, Self>
    where
        E: Display + Debug + Send + Sync + 'static,
    {
        match self.error.downcast::<E>() {
            Ok(value) => Ok(value),
            Err(error) => match error.downcast::<Located<E>>() {
                Ok(value) => Ok(value.err),
                Err(error) => Err(Self { error }),
            },
        }
    }

    /// Attempt to downcast the error object by reference.
    pub fn downcast_ref<E>(&self) -> Option<&E>
    where
        E: Display + Debug + Send + Sync + 'static,
    {
        match self.error.downcast_ref::<E>() {
            Some(err) => Some(err),
            None => self.error.downcast_ref::<Located<E>>().map(|e| &e.err),
        }
    }

    /// Attempt to downcast the error object by reference.
    pub fn downcast_mut<E>(&mut self) -> Option<&mut E>
    where
        E: Display + Debug + Send + Sync + 'static,
    {
        // We need to do a double-check with `anyhow::Error::is` instead of
        // an early return straight from `downcast_mut` due to
        // NLL: https://github.com/rust-lang/rust/issues/51826
        if self.error.is::<E>() {
            self.error.downcast_mut::<E>()
        } else {
            self.error.downcast_mut::<Located<E>>().map(|e| &mut e.err)
        }
    }

    /// Attach the context to `Self` and return a new error.
    #[track_caller]
    #[inline(never)]
    pub fn context<C>(self, context: C) -> Self
    where
        C: Display + Debug + Send + Sync + 'static,
    {
        Self {
            error: self.error.context(Located::new(context)),
        }
    }
}

impl Display for ANNError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        // Use the debug format `{:?}` for `anyhow::Error` to get the source chain as well
        // as a stack trace.
        write!(formatter, "ANNError\n\n{:?}", self.error)
    }
}

impl std::error::Error for ANNError {
    // Don't implement `source` because we print the whole source chain in our `Display`
    // implementation.
}

always_escalate!(ANNError);

#[macro_export]
macro_rules! convert_error {
    ($T:ty) => {
        impl From<$T> for $crate::ANNError {
            #[track_caller]
            fn from(e: $T) -> $crate::ANNError {
                $crate::ANNError::new(e)
            }
        }
    };
}

impl From<std::convert::Infallible> for ANNError {
    #[track_caller]
    fn from(_: std::convert::Infallible) -> Self {
        unreachable!("Infallible is an unconstructible type");
    }
}

convert_error!(io::Error);
convert_error!(LayoutError);
convert_error!(TryFromIntError);
convert_error!(TryFromSliceError);
convert_error!(diskann_utils::io::ReadBinError);
convert_error!(diskann_utils::io::SaveBinError);
convert_error!(diskann_utils::views::TryFromErrorLight);

// Convert from `mpsc::SendError` to `ANNError`
impl<T> From<mpsc::SendError<T>> for ANNError
where
    T: Send + Sync + 'static,
{
    #[track_caller]
    fn from(err: mpsc::SendError<T>) -> Self {
        ANNError::new(err)
    }
}

impl<T, U> From<diskann_utils::io::MetadataError<T, U>> for ANNError
where
    T: std::error::Error + Send + Sync + 'static,
    U: std::error::Error + Send + Sync + 'static,
{
    #[track_caller]
    fn from(err: diskann_utils::io::MetadataError<T, U>) -> Self {
        ANNError::new(err)
    }
}

impl<T> From<diskann_utils::views::TryFromError<T>> for ANNError
where
    T: diskann_utils::views::DenseData,
{
    #[track_caller]
    fn from(err: diskann_utils::views::TryFromError<T>) -> Self {
        Self::from(err.as_static())
    }
}

/// An internal wrapper for error types that also tracks the file and line information
/// for where the error was first converted and where context was propagated.
#[derive(Debug)]
struct Located<T>
where
    T: Debug,
{
    err: T,
    location: &'static std::panic::Location<'static>,
}

impl<T> Located<T>
where
    T: Debug,
{
    #[track_caller]
    fn new(err: T) -> Self {
        Self {
            err,
            location: std::panic::Location::caller(),
        }
    }
}

impl<T> Display for Located<T>
where
    T: Display + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{} -- ({}:{})",
            self.err,
            self.location.file(),
            self.location.line()
        )
    }
}

impl<T> std::error::Error for Located<T>
where
    T: std::error::Error + Debug,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.err.source()
    }
}

//////////////////
// ErrorContext //
//////////////////

/// Add context to a returned error that will be included in the source chain.
/// ```rust
/// use diskann::{ANNError, error::ErrorContext};
///
/// fn fn_a() -> Result<(), ANNError> {
///     Err(ANNError::message("thrown by function A"))
/// }
///
/// fn fn_b() -> Result<(), ANNError> {
///     fn_a().context("propagated by function B")
/// }
///
/// fn fn_c() -> Result<(), ANNError> {
///     fn_b().with_context(|| "propagated by function C")
/// }
///
/// // Format the final error message.
/// let message = fn_c().unwrap_err().to_string();
///
/// // Ensure that is has all the propagation reports.
/// assert!(message.contains("thrown by function A"));
/// assert!(message.contains("propagated by function B"));
/// assert!(message.contains("propagated by function C"));
/// ```
pub trait ErrorContext<T> {
    /// Attach the provided context to the error part of the result.
    fn context<C>(self, context: C) -> Result<T, ANNError>
    where
        C: Display + Debug + Send + Sync + 'static;

    /// Attach the provided context to the error part of the result.
    ///
    /// The function `f` will only be evaluated if `self` is an `Err`.
    fn with_context<F, C>(self, f: F) -> Result<T, ANNError>
    where
        C: Display + Debug + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    ANNError: From<E>,
{
    #[track_caller]
    fn context<C>(self, context: C) -> Result<T, ANNError>
    where
        C: Display + Debug + Send + Sync + 'static,
    {
        match self {
            Ok(value) => Ok(value),
            Err(error) => Err(ANNError::from(error).context(context)),
        }
    }

    #[track_caller]
    fn with_context<F, C>(self, f: F) -> Result<T, ANNError>
    where
        C: Display + Debug + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Ok(value) => Ok(value),
            Err(error) => Err(ANNError::from(error).context(f())),
        }
    }
}

/// Convert compatible types into `ANNResult`.
///
/// This trait enables conversion `Result<T, E: Into<ANNError>>` into `Result<T, ANNError>`,
/// allowing associated error types to express an `Into<ANNError>` bound while mostly
/// maintaining compatibility with the "?" operator.
pub trait IntoANNResult<T> {
    fn into_ann_result(self) -> Result<T, ANNError>;
}

impl<T, E> IntoANNResult<T> for Result<T, E>
where
    E: Into<ANNError>,
{
    #[inline(always)]
    #[track_caller]
    fn into_ann_result(self) -> Result<T, ANNError> {
        match self {
            Ok(v) => Ok(v),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod ann_result_test {
    use super::*;

    #[test]
    fn ann_err_is_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}
        assert_send_and_sync::<ANNError>();
    }

    // Check that the error type fits within 16-bytes and is available for niche
    // optimization.
    //
    // This is important to keep `Results` within 16-bytes so they can be returned in
    // registers.
    #[test]
    fn check_struct_size() {
        assert_eq!(std::mem::size_of::<ANNError>(), 8);
        assert_eq!(std::mem::size_of::<Option<ANNError>>(), 8);
        assert_eq!(std::mem::size_of::<Result<f32, ANNError>>(), 16);
    }

    #[derive(Debug, Clone)]
    struct SampleError {
        value: usize,
    }

    impl SampleError {
        fn new(value: usize) -> Self {
            Self { value }
        }
    }

    impl Display for SampleError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(f, "SampleError {{ {} }}", self.value)
        }
    }

    impl std::error::Error for SampleError {}
    convert_error!(SampleError);

    #[derive(Debug, Clone)]
    struct SampleChainedError {
        value: usize,
        source: SampleError,
    }

    impl SampleChainedError {
        fn new(value: usize, source: SampleError) -> Self {
            Self { value, source }
        }
    }

    impl Display for SampleChainedError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(f, "SampleChainedError {{ {} }}", self.value)
        }
    }

    impl std::error::Error for SampleChainedError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.source)
        }
    }

    #[test]
    fn check_downcasting() {
        let err = SampleError::new(10);
        let base_error = err.to_string();
        {
            let mut ann = ANNError::from(err.clone());

            // Make sure the error message is properly contained inside the larger error.
            assert!(format!("{}", ann).contains(&base_error));

            // Can we downcast by reference?
            let r = ann.downcast_ref::<SampleError>().unwrap();
            assert_eq!(r.value, 10);

            // Can we downcast by mutable reference and have the result stick?
            let r = ann.downcast_mut::<SampleError>().unwrap();
            r.value = 100;

            let r = ann.downcast_ref::<SampleError>().unwrap();
            assert_eq!(r.value, 100);

            // Consume by downcasting.
            let r = ann.downcast::<SampleError>().unwrap();
            assert_eq!(r.value, 100);
        }

        {
            // Make sure downcasting works even if embedded inside of contexts.
            let mut ann = ANNError::from(err.clone())
                .context("some context here")
                .context("more context");

            let formatted = ann.to_string();
            assert!(formatted.contains(&base_error));
            assert!(formatted.contains("some context here"));
            assert!(formatted.contains("more context"));

            // Can we downcast by reference?
            let r = ann.downcast_ref::<SampleError>().unwrap();
            assert_eq!(r.value, 10);

            // Can we downcast by mutable reference and have the result stick?
            let r = ann.downcast_mut::<SampleError>().unwrap();
            r.value = 100;

            let r = ann.downcast_ref::<SampleError>().unwrap();
            assert_eq!(r.value, 100);

            // Consume by downcasting.
            let r = ann.downcast::<SampleError>().unwrap();
            assert_eq!(r.value, 100);
        }

        // Failing paths.
        {
            // Make sure downcasting works even if embedded inside of contexts.
            let ann = ANNError::from(err.clone())
                .context("some context here")
                .context("more context");

            println!("{}", ann);

            let formatted = ann.to_string();

            // If we get the wrong type, make sure we return the original value.
            let mut ann = ann.downcast::<usize>().unwrap_err();
            assert_eq!(formatted, ann.to_string());

            assert!(ann.downcast_ref::<usize>().is_none());
            assert!(ann.downcast_mut::<usize>().is_none());
        }
    }

    // Context Chaining
    #[test]
    fn context_chaining() {
        let sample = SampleError::new(5).to_string();

        fn err() -> Result<usize, ANNError> {
            Err(ANNError::new(SampleError::new(5)))
        }

        fn ok() -> Result<usize, ANNError> {
            Ok(77)
        }

        // Context is applied properly.
        {
            let propagates = || err().context("with context");
            let chained = propagates().unwrap_err();
            let message = chained.to_string();
            assert!(message.contains("with context"), "got: {}", message);
            assert!(message.contains(&sample), "got: {}", message);
            assert_eq!(chained.downcast_ref::<SampleError>().unwrap().value, 5);
        }

        // Context not applied if okay.
        {
            let propagates = || ok().context("with context");
            let fine = propagates().unwrap();
            assert_eq!(fine, 77);
        }

        // With context is applied properly.
        {
            let mut called = false;
            let mut propagates = || {
                err().with_context(|| {
                    assert!(!called);
                    called = true;
                    "with context"
                })
            };
            let chained = propagates().unwrap_err();
            assert!(called);
            let message = chained.to_string();
            assert!(message.contains("with context"), "got: {}", message);
            assert!(message.contains(&sample), "got: {}", message);
            assert_eq!(chained.downcast_ref::<SampleError>().unwrap().value, 5);
        }

        // With context not applied if okay.
        {
            let propagates = || ok().with_context(|| -> ! { panic!("should not be called") });
            let fine = propagates().unwrap();
            assert_eq!(fine, 77);
        }
    }

    // Test the full formatting with line numbers.
    #[test]
    fn full_formatting() {
        let sample = SampleError::new(5);
        let file = file!();

        let l0 = line!() + 1;
        let err = ANNError::from(sample);
        let l1 = line!() + 1;
        let err = err.context("some context");
        let l2 = line!() + 1;
        let err = err.context("more context");

        let expected = format!(
            "ANNError

more context -- ({}:{})

Caused by:
    0: some context -- ({}:{})
    1: SampleError {{ {} }} -- ({}:{})",
            file, l2, file, l1, 5, file, l0
        );

        let got = err.to_string();
        assert!(
            got.starts_with(&expected),
            "got:\n{}\n\nexpected:\n{}",
            got,
            expected
        );
    }

    // Test the full formatting with line numbers.
    #[test]
    fn full_formatting_with_cause() {
        let sample = SampleChainedError::new(10, SampleError::new(5));
        let file = file!();

        let l0 = line!() + 1;
        let err = ANNError::new(sample);
        let l1 = line!() + 1;
        let err = err.context("some context");
        let l2 = line!() + 1;
        let err = err.context("more context");

        let expected = format!(
            "ANNError

more context -- ({}:{})

Caused by:
    0: some context -- ({}:{})
    1: SampleChainedError {{ 10 }} -- ({}:{})
    2: SampleError {{ 5 }}",
            file, l2, file, l1, file, l0
        );

        let got = err.to_string();
        assert!(
            got.starts_with(&expected),
            "got:\n{}\n\nexpected:\n{}",
            got,
            expected
        );
    }

    #[test]
    fn full_formatting_with_cause_no_context() {
        let sample = SampleChainedError::new(10, SampleError::new(5));
        let file = file!();

        let l0 = line!() + 1;
        let err = ANNError::new(sample);

        let expected = format!(
            "ANNError

SampleChainedError {{ 10 }} -- ({}:{})

Caused by:
    SampleError {{ 5 }}",
            file, l0
        );

        let got = err.to_string();
        assert!(
            got.starts_with(&expected),
            "got:\n{}\n\nexpected:\n{}",
            got,
            expected
        );
    }
}
