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

/// Error wrapper for values that should be carried by `ANNError` through their
/// `Display` implementation.
#[derive(Debug)]
pub struct DisplayError<D>(D);

impl<D> DisplayError<D> {
    pub fn new(display: D) -> Self {
        Self(display)
    }
}

impl<D> Display for DisplayError<D>
where
    D: Display,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        Display::fmt(&self.0, formatter)
    }
}

impl<D> std::error::Error for DisplayError<D> where D: Display + Debug + Send + Sync + 'static {}

/// Convenience alias for a `Result<T, ANNError>`.
pub type ANNResult<T> = Result<T, ANNError>;

/// Common error type shared through DiskANN.
///
/// This type disambiguates the runtime origin of errors using the `kind()` enum. Third
/// party implementations of DiskANN plugin types like provider can use `kind()` and the
/// downcasting API to throw custom errors from low in the callstack and retrieve those
/// errors higher in the stack.
/// ```rust
/// use diskann::{ANNError, ANNErrorKind, error::ErrorContext};
/// use thiserror::Error;
///
/// // A custom error type used by a third-party.
/// #[derive(Debug, Error)]
/// #[error("custom error: {0}")]
/// struct CustomError(usize);
///
/// // A low-level function that returns an error.
/// fn errors() -> Result<(), ANNError> {
///     Err(ANNError::new(ANNErrorKind::Tagged(100), CustomError(42)))
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
/// // If we retrieve the `ANNErrorKind`, we can recognize that it belongs to a third-party
/// // plugin.
/// assert_eq!(err.kind(), ANNErrorKind::Tagged(100));
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
/// * `std::mem::size_of::<ANNError>() == 16`: The struct is 16 bytes. This allows it to be
///   returned in registers rather than on the stack.
/// * `std::mem::size_of::<Option<ANNError>>() == 16`: The struct can use Rust's niche
///   optimization.
#[derive(Debug)]
pub struct ANNError {
    kind: ANNErrorKind,
    error: anyhow::Error,
}

impl ANNError {
    /// Construct a new `ANNError` encapsulating `err`.
    ///
    /// Errors constructed this way can be retrieved using downcasting.
    /// ```rust
    /// use diskann::{ANNError, ANNErrorKind};
    /// use std::env::VarError;
    ///
    /// let err = ANNError::new(
    ///     ANNErrorKind::IndexError,
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
    pub fn new<E>(kind: ANNErrorKind, err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind,
            error: anyhow::Error::new(Located::new(err)),
        }
    }

    /// Construct a new `ANNError` encapsulating `err` tagged with `ANNErrorKind::Opaque`.
    ///
    /// Errors constructed this way can be retrieved using downcasting.
    /// ```rust
    /// use diskann::{ANNError, ANNErrorKind};
    /// use std::env::VarError;
    ///
    /// let err = ANNError::opaque(VarError::NotPresent);
    ///
    /// assert_eq!(err.kind(), ANNErrorKind::Opaque);
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
    pub fn opaque<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind: ANNErrorKind::Opaque,
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
    pub fn message<D>(kind: ANNErrorKind, display: D) -> Self
    where
        D: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    {
        Self {
            kind,
            error: anyhow::Error::msg(Located::new(display)),
        }
    }

    /// Construct a new `ANNError` from a display value.
    ///
    /// Prefer a concrete error type with [`ANNError::new`] when the callsite can describe
    /// a precise failure. This helper is useful when adapting existing display-only error
    /// paths away from the legacy `log_*` constructors.
    #[track_caller]
    #[inline(never)]
    pub fn from_display<D>(kind: ANNErrorKind, display: D) -> Self
    where
        D: Display + Debug + Send + Sync + 'static,
    {
        Self::new(kind, DisplayError::new(display))
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
                Err(error) => Err(Self {
                    kind: self.kind,
                    error,
                }),
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
            kind: self.kind,
            error: self.error.context(Located::new(context)),
        }
    }

    /// Return the kind of the originally constructed error.
    pub fn kind(&self) -> ANNErrorKind {
        self.kind
    }
}

impl Display for ANNError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        // Use the debug format `{:?}` for `anyhow::Error` to get the source chain as well
        // as a stack trace.
        write!(formatter, "ANNError: {:?}\n\n{:?}", self.kind, self.error)
    }
}

impl std::error::Error for ANNError {
    // Don't implement `source` because we print the whole source chain in our `Display`
    // implementation.
}

always_escalate!(ANNError);

impl From<std::convert::Infallible> for ANNError {
    #[track_caller]
    fn from(_: std::convert::Infallible) -> Self {
        unreachable!("Infallible is an unconstructible type");
    }
}

// Convert from `io::Error` to `ANNError`
impl From<io::Error> for ANNError {
    #[track_caller]
    fn from(err: io::Error) -> Self {
        ANNError::new(ANNErrorKind::IOError, err)
    }
}

// Convert from `mpsc::SendError` to `ANNError`
impl<T> From<mpsc::SendError<T>> for ANNError
where
    T: Send + Sync + 'static,
{
    #[track_caller]
    fn from(err: mpsc::SendError<T>) -> Self {
        ANNError::new(ANNErrorKind::IOSendError, err)
    }
}

// Convert from `LayoutError` to `ANNError`
impl From<LayoutError> for ANNError {
    #[track_caller]
    fn from(err: LayoutError) -> Self {
        ANNError::new(ANNErrorKind::MemoryAllocLayoutError, err)
    }
}

// Convert from `TryFromIntError` to `ANNError`
impl From<TryFromIntError> for ANNError {
    #[track_caller]
    fn from(err: TryFromIntError) -> Self {
        ANNError::new(ANNErrorKind::TryFromIntError, err)
    }
}

// Convert from `TryFromSliceError` to `ANNError`
impl From<TryFromSliceError> for ANNError {
    #[track_caller]
    fn from(err: TryFromSliceError) -> Self {
        ANNError::new(ANNErrorKind::TryFromSliceError, err)
    }
}

impl From<diskann_utils::io::ReadBinError> for ANNError {
    #[track_caller]
    fn from(err: diskann_utils::io::ReadBinError) -> Self {
        ANNError::new(ANNErrorKind::IOError, err)
    }
}

impl From<diskann_utils::io::SaveBinError> for ANNError {
    #[track_caller]
    fn from(err: diskann_utils::io::SaveBinError) -> Self {
        ANNError::new(ANNErrorKind::IOError, err)
    }
}

impl<T, U> From<diskann_utils::io::MetadataError<T, U>> for ANNError
where
    T: std::error::Error + Send + Sync + 'static,
    U: std::error::Error + Send + Sync + 'static,
{
    #[track_caller]
    fn from(err: diskann_utils::io::MetadataError<T, U>) -> Self {
        ANNError::new(ANNErrorKind::IOError, err)
    }
}

impl From<diskann_utils::views::TryFromErrorLight> for ANNError {
    #[track_caller]
    fn from(err: diskann_utils::views::TryFromErrorLight) -> Self {
        ANNError::new(ANNErrorKind::DimensionMismatchError, err)
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
/// use diskann::{ANNError, ANNErrorKind, error::ErrorContext};
///
/// fn fn_a() -> Result<(), ANNError> {
///     Err(ANNError::message(ANNErrorKind::IndexError, "thrown by function A"))
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

// /// An internal macro for creating opaque, adhoc errors to help when debugging.
// macro_rules! ann_error {
//     ($($arg:tt)+) => {{
//         ANNError::message(ANNErrorKind::Opaque, format!($($args)+))
//     }};
// }
//
// pub(crate) use ann_error;

//////////////////
// ANNErrorKind //
//////////////////

/// DiskANN error kinds used to tag a returned error.
///
/// Third-party implementations of DiskANN components (for example, custom implementations
/// of providers), can use the `Tagged` alternative to tag the error type for later
/// inspection. This mechanism can be used in coordination with the downcast API of
/// `ANNError` to retrieve the source error.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ANNErrorKind {
    /// The error originiated within DiskANN.
    DiskANN(DiskANNError),
    /// An error with a tagged type to help with recovery. Provider implementations may
    /// choose to tag their errors if useful.
    Tagged(u32),
    /// An opaque error with no tag.
    ///
    /// This can be used by provider implementations that do not care to tag their errors.
    Opaque,
}

macro_rules! define_alias {
    ($name:ident) => {
        #[allow(non_upper_case_globals)]
        pub const $name: ANNErrorKind = ANNErrorKind::DiskANN(DiskANNError::$name);
    };
}

impl ANNErrorKind {
    // Aliases - this is to maintain compatibility with an earlier version of the error type.
    define_alias!(IndexError);
    define_alias!(IndexConfigError);
    define_alias!(TryFromIntError);
    define_alias!(DimensionMismatchError);
    define_alias!(FileNotFoundError);
    define_alias!(FileHandleError);
    define_alias!(AsyncIOThreadError);
    define_alias!(GroundTruthError);
    define_alias!(IOError);
    define_alias!(IOSendError);
    define_alias!(MemoryAllocLayoutError);
    define_alias!(LockPoisonError);
    define_alias!(DiskIOAlignmentError);
    define_alias!(PQError);
    define_alias!(OPQError);
    define_alias!(KMeansError);
    define_alias!(TryFromSliceError);
    define_alias!(AdjacencyListConversionError);
    define_alias!(SerdeError);
    define_alias!(GetVertexDataError);
    define_alias!(ParseSliceError);
    define_alias!(ThreadPoolError);
    define_alias!(NoExpectedNormError);
    define_alias!(UnexpectedCheckpoint);
    define_alias!(InvalidOperation);
    define_alias!(PagedSearchError);
    define_alias!(AsyncError);
    define_alias!(AsyncShutdownError);
    define_alias!(RustRuntimeError);
    define_alias!(AsyncIndexError);
    define_alias!(ProfilerError);
    define_alias!(PQSchemaRegistrationError);
    define_alias!(InvalidFileFormatError);
    define_alias!(StartPointComputeError);
    define_alias!(SQError);
    define_alias!(BuildInterrupted);

    // Legacy Linux specific error.
    define_alias!(PushError);
}

/// A Internal errors yielded by DiskANN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiskANNError {
    /// Index construction and search error
    IndexError,

    /// Index configuration error
    IndexConfigError,

    /// Integer conversion error
    TryFromIntError,

    /// Dimension mismatch error
    DimensionMismatchError,

    /// File does not exist
    FileNotFoundError,

    /// Error with the file handle
    FileHandleError,

    /// Error with async IO threading
    AsyncIOThreadError,

    /// Error with ground-truth
    GroundTruthError,

    /// IO error
    IOError,

    /// IO SendError
    IOSendError,

    /// Layout error in memory allocation
    MemoryAllocLayoutError,

    /// PoisonError which can be returned whenever a lock is acquired
    /// Both Mutexes and RwLocks are poisoned whenever a thread fails while the lock is held
    LockPoisonError,

    /// DiskIOAlignmentError which can be returned when calling windows API CreateFileA for the disk index file fails.
    DiskIOAlignmentError,

    // PQ construction error
    // Error happened when we construct PQ pivot or PQ compressed table
    PQError,

    // OPQ construction error (deprecated — kept to preserve variant numbering)
    OPQError,

    // K-means error
    // Error happened when we run k-means clustering
    KMeansError,

    /// Array conversion error
    TryFromSliceError,

    /// Array conversion error
    AdjacencyListConversionError,

    /// Array conversion error
    SerdeError,

    /// Error when we try to get the vertex data from vertex provider.
    GetVertexDataError,

    /// Error when we try to parse a slice to an object.
    ParseSliceError,

    ThreadPoolError,

    NoExpectedNormError,

    // Error when the checkpoint record is expected.
    UnexpectedCheckpoint,

    // Generic invalid operation error.
    InvalidOperation,

    PagedSearchError,

    AsyncError,

    AsyncShutdownError,

    RustRuntimeError,

    AsyncIndexError,

    ProfilerError,

    // Errors related to PQ schema registration.
    PQSchemaRegistrationError,

    // Error when file format doesn't match expectations
    InvalidFileFormatError,

    /// Error when index build process is intentionally interrupted
    ///
    /// This is not a true error, but a controlled interruption signal used to gracefully
    /// exit from multi-level function calls in the build process.
    BuildInterrupted,

    // Error when computing start point from data
    StartPointComputeError,

    // SQ construction error
    // Error happened when we build the SQ index
    SQError,

    /// Linux io-uring error when pushing a task into the submission ring
    PushError,
}

#[cfg(test)]
mod ann_result_test {
    use std::{alloc::Layout, array::TryFromSliceError, io};

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
        assert_eq!(std::mem::size_of::<ANNError>(), 16);
        assert_eq!(std::mem::size_of::<Option<ANNError>>(), 16);
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

    impl From<SampleError> for ANNError {
        #[track_caller]
        fn from(value: SampleError) -> ANNError {
            ANNError::new(ANNErrorKind::Tagged(0), value)
        }
    }

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
            assert_eq!(ann.kind(), ANNErrorKind::Tagged(0));

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

    // Opaque
    #[test]
    fn test_opaque_constructor() {
        let err = SampleError::new(50);
        let ann = ANNError::opaque(err.clone());

        assert_eq!(ann.kind(), ANNErrorKind::Opaque);
        assert!(ann.to_string().contains(&err.to_string()));
    }

    // Context Chaining
    #[test]
    fn context_chaining() {
        let sample = SampleError::new(5).to_string();

        fn err() -> Result<usize, ANNError> {
            Err(ANNError::new(ANNErrorKind::Tagged(42), SampleError::new(5)))
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
            assert_eq!(chained.kind(), ANNErrorKind::Tagged(42));
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
            assert_eq!(chained.kind(), ANNErrorKind::Tagged(42));
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
            "ANNError: Tagged(0)

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
        let err = ANNError::new(ANNErrorKind::Tagged(0), sample);
        let l1 = line!() + 1;
        let err = err.context("some context");
        let l2 = line!() + 1;
        let err = err.context("more context");

        let expected = format!(
            "ANNError: Tagged(0)

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
        let err = ANNError::new(ANNErrorKind::Tagged(0), sample);

        let expected = format!(
            "ANNError: Tagged(0)

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

    #[test]
    fn test_io_error_to_ann_error() {
        let io_err = io::Error::other("test error");
        let expected = io_err.to_string();
        let ann_err: ANNError = ANNError::from(io_err);
        assert_eq!(ann_err.kind(), ANNErrorKind::IOError);
        assert!(ann_err.to_string().contains(&expected));
    }

    #[test]
    fn test_send_error_to_ann_error() {
        let send_err = mpsc::SendError(());
        let expected = send_err.to_string();
        let ann_err: ANNError = send_err.into();
        assert_eq!(ann_err.kind(), ANNErrorKind::IOSendError);
        assert!(ann_err.to_string().contains(&expected));
    }

    #[test]
    fn test_layout_error_to_ann_error() {
        let layout_err = Layout::from_size_align(1, 0).unwrap_err();
        let ann_err: ANNError = layout_err.into();
        assert_eq!(ann_err.kind(), ANNErrorKind::MemoryAllocLayoutError);
    }

    #[test]
    fn test_try_from_int_error_to_ann_error() {
        let err = u8::try_from(1_000usize).unwrap_err();
        let ann_err: ANNError = err.into();
        assert_eq!(ann_err.kind(), ANNErrorKind::TryFromIntError);
    }

    #[test]
    fn test_try_from_slice_error_to_ann_error() {
        let slice: &[u8] = &[1, 2, 3];
        let slice_err: Result<[u8; 4], TryFromSliceError> = slice.try_into();
        let err = slice_err.unwrap_err();
        let ann_err: ANNError = err.into();
        assert_eq!(ann_err.kind(), ANNErrorKind::TryFromSliceError);
    }

    #[test]
    fn test_display_ann_error() {
        let err = ANNErrorKind::IndexError;
        assert_eq!(format!("{:?}", err), "DiskANN(IndexError)");
    }

    #[test]
    fn test_invaild_file_format_error() {
        let err_msg = String::from("Invalid file format error");
        let ann_err = ANNError::from_display(ANNErrorKind::InvalidFileFormatError, err_msg.clone());
        assert_eq!(ann_err.kind(), ANNErrorKind::InvalidFileFormatError);
    }

    #[test]
    fn test_build_interrupted() {
        let message = "BuildIndicesOnShards";
        let ann_err = ANNError::from_display(ANNErrorKind::BuildInterrupted, message);
        assert_eq!(ann_err.kind(), ANNErrorKind::BuildInterrupted);
    }

    #[test]
    fn from_read_bin_error() {
        let err = diskann_utils::io::ReadBinError::SizeMismatch {
            expected: 100,
            available: 50,
            npoints: 10,
            ndims: 5,
            type_size: 2,
        };
        let ann_err = ANNError::from(err);
        assert_eq!(ann_err.kind(), ANNErrorKind::IOError);
    }

    #[test]
    fn from_save_bin_error() {
        let err = diskann_utils::io::SaveBinError::DimensionOverflow { nrows: 1, ncols: 1 };
        let ann_err = ANNError::from(err);
        assert_eq!(ann_err.kind(), ANNErrorKind::IOError);
    }

    #[test]
    fn from_metadata_error() {
        let err = diskann_utils::io::Metadata::new(u64::MAX, 1u32).unwrap_err();
        let ann_err = ANNError::from(err);
        assert_eq!(ann_err.kind(), ANNErrorKind::IOError);
    }

    #[test]
    fn from_try_from_error_light() {
        let data: &[f32] = &[1.0, 2.0, 3.0];
        let light = diskann_utils::views::MatrixView::try_from(data, 2, 2)
            .unwrap_err()
            .as_static();
        let ann_err = ANNError::from(light);
        assert_eq!(ann_err.kind(), ANNErrorKind::DimensionMismatchError);
    }

    #[test]
    fn from_try_from_error() {
        let data: &[f32] = &[1.0, 2.0, 3.0];
        let err = diskann_utils::views::MatrixView::try_from(data, 2, 2).unwrap_err();
        let ann_err = ANNError::from(err);
        assert_eq!(ann_err.kind(), ANNErrorKind::DimensionMismatchError);
    }
}
