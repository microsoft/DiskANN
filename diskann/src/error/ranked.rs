/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-Modal Error Handling
//!
//! # Background
//!
//! Sometime, data providers can experience spurious failures during operations such as vector
//! retrieval.
//!
//! A primary culprit is access to deleted vectors (if vectors are eagerly deleted rather
//! than deferred until graph clean up), which may returns an error but is more or
//! less expected and should not take down the DiskANN algorithm.
//!
//! However, not all classes of errors can be ignored in this way, such as errors indicating
//! that DiskANN should abort processing.
//!
//! Furthermore, DiskANN should not be in the habit of ignoring all errors from providers as
//! many users may want such errors to be propagated.
//!
//! This means we need a multi-level error handling mechanism that achieves several design
//! goals:
//!
//! 1. Disambiguate between "transient" errors (those which are expected to resolve eventually
//!    or can be ignored) and "hard" errors (those that should stop the current task and be
//!    propagated) from user error types.
//!
//! 2. Require algorithmic code to explicitly acknowledge or escalate "transient" errors with
//!    minimal probability of silently dropping an error or implicitly escalating.
//!
//!    In this context:
//!
//!    * "acknowledge": Notify the error that it has been observed and that the algorithm
//!      is choosing to continue.
//!
//!    * "escalate": Notify the error that is has been observed and is being escalated by
//!      the algorithm into a hard error.
//!
//! 3. Provide light-weight callback mechanisms for error acknowledgement or escalation
//!    to enable telemetry in user types.
//!
//!    Practically, this means that user error types can use diagnostic logging to record
//!    acknowledgements or escalations to assist in debugging efforts.
//!
//! 4. Ideally be minimal or no overhead for error types that are never transient.
//!
//! # How Does This Work?
//!
//! User error types (those implemented by the [`DataProvider`] and (eventually)
//! [`NeighborProviderAsync`]) are expected to implement [`ToRanked`], which converts the
//! error type to a [`RankedError`] consisting of a `Transient` and full `Error` alternatives.
//! The [`RankedError::Transient`] alternative must implement [`TransientError`], which
//! provides callback mechanisms for transient error acknowledgement or escalation.
//! These callback are provided with a concrete reason for the decision to aid in debugging
//! or program telemetry.
//!
//! The trait [`ErrorExt`] is then defined which takes `Result`s of such [`ToRanked`] types
//! and provides the `acknowledge`/`escalate` interface to these `Result`s.
//!
//! So, how does this satisfy our goals:
//!
//! 1. User types explicitly mark themselves as either "transient" or "hard" when constructing
//!    the [`RankedError`] enum through [`ToRanked`].
//!
//! 2. Generic algorithmic code should constrain error types to be just [`ToRanked`]. This
//!    means that the `?` operator will not apply by default, requiring explicit handling
//!    through either [`ErrorExt::acknowledge`] or [`ErrorExt::escalate`].
//!
//! 3. The methods in [`ErrorExt`] require reasons for the decision, which are forwarded
//!    the implementation of [`TransientError`] and therefore made available to user types.
//!
//! Now, what about low-overhead?
//!
//! User types `T` that wish to **always** escalate can use the [`diskann::always_escalate`]
//! macro. This works be defining the `Transient` alternative of [`ToRanked`] to be the
//! unconstructable enum [`diskann::error::NeverTransient`]. This ensures at compile time
//! that the [`RankedError::Transient`] alternative is unreachable and therefore the layout of
//! `RankedError<NeverTransient, T>` is the same as `T` and all the transient error handling
//! goes away.

use std::fmt::{Debug, Display};

use crate::{ANNError, ANNResult};

pub trait TransientError<T>: Sized + std::fmt::Debug + Send + Sync {
    /// Consume self, acknowledging the transient error but proceeding with program logic.
    ///
    /// This method accepts a parameter describing the reason for the acknowledgement.
    fn acknowledge<D>(self, why: D)
    where
        D: Display;

    /// Consume self, acknowledging the transient error but proceeding with program logic.
    ///
    /// The closure `why` provides a deferred method for evaluating the acknowledgement
    /// reason and can be used by error types that do not log acknowledgements to avoid
    /// evaluating the reason all-together.
    ///
    /// # Track Caller
    ///
    /// The provided implementation is annotated with `track_caller`.
    #[track_caller]
    fn acknowledge_with<F, D>(self, why: F)
    where
        F: FnOnce() -> D,
        D: Display,
    {
        self.acknowledge(why())
    }

    /// Report to `self` that transient errors are not acceptable in the current context
    /// and that the transient error is being upgraded to a full error.
    ///
    /// This method accepts a parameter describing the reason for the escalation.
    fn escalate<D>(self, why: D) -> T
    where
        D: Display;

    /// Report to `self` that transient errors are not acceptable in the current context
    /// and that the transient error is being upgraded to a full error.
    ///
    /// The closure `why` provides a deferred method for evaluating the acknowledgement
    /// reason and can be used by error types that do not log acknowledgements to avoid
    /// evaluating the reason all-together.
    ///
    /// # Track Caller
    ///
    /// The provided implementation is annotated with `track_caller`.
    #[track_caller]
    fn escalate_with<F, D>(self, why: F) -> T
    where
        F: FnOnce() -> D,
        D: Display,
    {
        self.escalate(why())
    }
}

/// Allow conversion from an error type to a [`RankedError`].
///
/// This trait bound is applied to providers that may return transient or non-critical
/// failures. The [`RankedError`] allows DiskANN algorithmic logic to determine whether
/// an error can be suppressed safely or needs to be escalated.
///
/// * See also: [`always_escalate!`], [`ErrorExt`].
pub trait ToRanked {
    type Transient: TransientError<Self::Error>;
    type Error: Into<ANNError> + std::fmt::Debug + Send + Sync;

    /// Convert `self` into a `RankedError`.
    fn to_ranked(self) -> RankedError<Self::Transient, Self::Error>;

    /// Construct `Self` from its transient variant.
    fn from_transient(transient: Self::Transient) -> Self;

    /// Construct `Self` from its error variant.
    fn from_error(error: Self::Error) -> Self;
}

/// An error type consisting of a transient (non-critical) error and an unignorable error.
#[must_use]
#[derive(Debug)]
pub enum RankedError<R, E>
where
    R: TransientError<E>,
{
    /// This error can be ignored if the allowed by the caller code. Such errors should
    /// be acknowledged (see [`TransientError::acknowledge`] before being dropped, but
    /// aren't necessarily required to be.
    Transient(R),

    /// This error cannot be ignored and should be propagated. The caller should **never**
    /// suppress values in this alternative.
    Error(E),
}

impl<R, E> ToRanked for RankedError<R, E>
where
    R: TransientError<E>,
    E: Into<ANNError> + std::fmt::Debug + Send + Sync,
{
    type Transient = R;
    type Error = E;

    fn to_ranked(self) -> Self {
        self
    }

    fn from_transient(transient: <Self as ToRanked>::Transient) -> Self {
        Self::Transient(transient)
    }

    fn from_error(error: <Self as ToRanked>::Error) -> Self {
        Self::Error(error)
    }
}

/// A zero-sized type that is unconstructable.
///
/// This is used as the [`TransientError`] type for error types using the
/// [`always_escalate!`] macro to opt-out of transient error handling.
#[derive(Debug)]
pub enum NeverTransient {}

/// Mark the type `T` as "Always Escalating". This will implement `ToRanked for $T` in a
/// way that:
///
/// 1. Ensures `to_ranked` **always** returns [`RankedError::Error`].
///
/// 2. Defines the "transient" portion of `ToRanked` in such a way that it is impossible to
///    instantiate. This means that the layout of [`RankedError`] is identical to the layout
///    of `$T` and allows the methods in [`ErrorExt`] to optimize out all transient error
///    handling.
#[macro_export]
macro_rules! always_escalate {
    ($T:ty) => {
        impl $crate::error::TransientError<$T> for $crate::error::NeverTransient {
            fn acknowledge<D>(self, _: D)
            where
                D: std::fmt::Display,
            {
                unreachable!("NeverTransient is an unconstructable type");
            }
            fn acknowledge_with<F, D>(self, _: F)
            where
                F: FnOnce() -> D,
                D: std::fmt::Display,
            {
                unreachable!("NeverTransient is an unconstructable type");
            }
            fn escalate<D>(self, _: D) -> $T
            where
                D: std::fmt::Display,
            {
                unreachable!("NeverTransient is an unconstructable type");
            }
            fn escalate_with<F, D>(self, _: F) -> $T
            where
                F: FnOnce() -> D,
                D: std::fmt::Display,
            {
                unreachable!("NeverTransient is an unconstructable type");
            }
        }

        impl $crate::error::ToRanked for $T {
            type Transient = $crate::error::NeverTransient;
            type Error = Self;

            fn to_ranked(self) -> $crate::error::RankedError<Self::Transient, Self::Error> {
                $crate::error::RankedError::Error(self)
            }

            fn from_transient(_: $crate::error::NeverTransient) -> Self {
                unreachable!("NeverTransient is an unconstructable type");
            }

            fn from_error(error: Self) -> Self {
                error
            }
        }
    };
}

/// An infallible error type so that the compiler knows this error can
/// never be thrown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Infallible {}

impl From<Infallible> for ANNError {
    fn from(_: Infallible) -> Self {
        // This is unreachable but we need to implement it to keep the compiler happy when
        // the error type for an associated type must implement `Into<ANNError>` and is
        // `Infallible`. E.g. `VectorRepr` impl for f32.
        unreachable!()
    }
}

impl std::fmt::Display for Infallible {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unreachable!()
    }
}

impl std::error::Error for Infallible {}

impl Infallible {
    /// Match result containing `Infallible` error type.
    ///
    /// This function is a utility for unwrapping a `Result` that has an `Infallible` error type.
    /// Since `Infallible` represents a type that cannot be instantiated, any `Result<T, Infallible>`
    /// is guaranteed to be `Ok(T)`. This function safely extracts the value from such a result.
    ///
    /// # Arguments
    ///
    /// * `x` - A result with an infallible error type.
    ///
    /// # Returns
    ///
    /// The inner value `T` from the `Ok` variant.
    ///
    /// # Example
    ///
    /// ```
    /// use diskann::error::Infallible;
    ///
    /// let result: Result<i32, Infallible> = Ok(42);
    /// let value = Infallible::match_infallible(result);
    /// assert_eq!(value, 42);
    /// ```
    pub fn match_infallible<T>(x: Result<T, Infallible>) -> T {
        x.unwrap_or_else(|inf| match inf {})
    }
}

always_escalate!(Infallible);

/// Provide explicit error handling for compatible result types.
///
/// This trait requires the caller to differentiate between callsites that can handle
/// transient errors and those that cannot.
///
/// Explicit acknowledgement or escalation of transient errors is accompanied by a reason
/// why to assist with telementry.
pub trait ErrorExt<T> {
    /// Acknowledge and drop transient errors while propagating critical errors.
    ///
    /// * If `self` is not an error, return `Ok(Some(v))`.
    /// * If `self` is a transient error, acknowledge it with the provided reason and return
    ///   `Ok(None)`.
    /// * Otherwise, return an error.
    fn allow_transient<D>(self, why: D) -> ANNResult<Option<T>>
    where
        D: Display;

    /// Acknowledge and drop transient errors while propagating critical errors.
    ///
    /// * If `self` is not an error, return `Ok(Some(v))`.
    /// * If `self` is a transient error, acknowledge it with the provided reason and return
    ///   `Ok(None)`.
    /// * Otherwise, return an error.
    fn allow_transient_with<F, D>(self, why: F) -> ANNResult<Option<T>>
    where
        F: FnOnce() -> D,
        D: Display;

    /// Escalate transient errors into full errors.
    ///
    /// * If `self` is not an error, return `Ok(v)`.
    /// * If `self` is a transient error, escalate with the provided reason and return the
    ///   escalated error.
    /// * Otherwise, return the critical error.
    fn escalate<D>(self, why: D) -> ANNResult<T>
    where
        D: Display;

    /// Escalate transient errors into full errors.
    ///
    /// * If `self` is not an error, return `Ok(v)`.
    /// * If `self` is a transient error, escalate with the provided reason and return the
    ///   escalated error.
    /// * Otherwise, return the critical error.
    fn escalate_with<F, D>(self, why: F) -> ANNResult<T>
    where
        F: FnOnce() -> D,
        D: Display;
}

impl<T, E> ErrorExt<T> for Result<T, E>
where
    E: ToRanked,
{
    #[track_caller]
    fn allow_transient<D>(self, why: D) -> ANNResult<Option<T>>
    where
        D: Display,
    {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(err) => match err.to_ranked() {
                RankedError::Transient(transient) => {
                    transient.acknowledge(why);
                    Ok(None)
                }
                RankedError::Error(err) => Err(err.into()),
            },
        }
    }

    #[track_caller]
    fn allow_transient_with<F, D>(self, why: F) -> ANNResult<Option<T>>
    where
        F: FnOnce() -> D,
        D: Display,
    {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(err) => match err.to_ranked() {
                RankedError::Transient(transient) => {
                    transient.acknowledge_with(why);
                    Ok(None)
                }
                RankedError::Error(err) => Err(err.into()),
            },
        }
    }

    #[track_caller]
    fn escalate<D>(self, why: D) -> ANNResult<T>
    where
        D: Display,
    {
        match self {
            Ok(v) => Ok(v),
            Err(err) => match err.to_ranked() {
                RankedError::Transient(transient) => Err(transient.escalate(why).into()),
                RankedError::Error(err) => Err(err.into()),
            },
        }
    }

    #[track_caller]
    fn escalate_with<F, D>(self, why: F) -> ANNResult<T>
    where
        F: FnOnce() -> D,
        D: Display,
    {
        match self {
            Ok(v) => Ok(v),
            Err(err) => match err.to_ranked() {
                RankedError::Transient(transient) => Err(transient.escalate_with(why).into()),
                RankedError::Error(err) => Err(err.into()),
            },
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use thiserror::Error;

    use super::*;

    // Check that the layout of Ranked "Always Escalate" types `T` is the same as the layout
    // for `T`
    #[derive(Debug, Clone, Copy, Error)]
    #[error("generic error message: {0}")]
    struct AlwaysEscalate(usize);

    impl From<AlwaysEscalate> for ANNError {
        fn from(value: AlwaysEscalate) -> ANNError {
            ANNError::log_index_error(value)
        }
    }

    always_escalate!(AlwaysEscalate);

    #[test]
    fn test_always_escalate() {
        // The goal here is to ensure that the ranked version of a type that has opted in
        // to `always_escalate!` has the same layout (i.e., same size) as the error type
        // itself.
        //
        // The goal is to make ranking always-escalate errors
        assert_eq!(
            std::mem::size_of::<RankedError<NeverTransient, AlwaysEscalate>>(),
            std::mem::size_of::<AlwaysEscalate>()
        );

        let r = AlwaysEscalate(10).to_ranked();
        assert!(matches!(r, RankedError::Error(AlwaysEscalate(10))));
    }

    #[derive(Debug, Error)]
    #[error(
        "Bomb: value = {}, ack = {}, escalated = {}",
        value,
        acknowledged,
        escalated
    )]
    struct Bomb<'a> {
        messages: &'a Mutex<Vec<(String, u32)>>,
        acknowledged: bool,
        escalated: bool,
        value: u64,
    }

    impl<'a> Bomb<'a> {
        fn new(messages: &'a Mutex<Vec<(String, u32)>>, value: u64) -> Self {
            Self {
                messages,
                acknowledged: false,
                escalated: false,
                value,
            }
        }
    }

    impl Drop for Bomb<'_> {
        fn drop(&mut self) {
            if !self.acknowledged && !self.escalated {
                panic!("Bomb error was neither acknowledged nor escalated");
            }
            if self.acknowledged && self.escalated {
                panic!("Bomb error was both acknowledged and escalated");
            }
        }
    }

    #[derive(Debug, Error)]
    #[error("Disarmed: value = {}", value)]
    struct Disarmed<'a> {
        messages: &'a Mutex<Vec<(String, u32)>>,
        value: u64,
    }

    impl<'a> Disarmed<'a> {
        fn new(messages: &'a Mutex<Vec<(String, u32)>>, value: u64) -> Self {
            Self { messages, value }
        }
    }

    impl<'a> TransientError<Disarmed<'a>> for Bomb<'a> {
        #[track_caller]
        fn acknowledge<D>(mut self, why: D)
        where
            D: Display,
        {
            self.acknowledged = true;
            let mut v = self.messages.lock().unwrap();
            let location = std::panic::Location::caller();
            v.push((format!("acknowledged: {}", why), location.line()))
        }

        #[track_caller]
        fn escalate<D>(mut self, why: D) -> Disarmed<'a>
        where
            D: Display,
        {
            self.escalated = true;
            let mut v = self.messages.lock().unwrap();
            let location = std::panic::Location::caller();
            v.push((format!("escalated: {}", why), location.line()));

            Disarmed {
                messages: self.messages,
                value: self.value,
            }
        }
    }

    impl From<Disarmed<'_>> for ANNError {
        #[track_caller]
        fn from(value: Disarmed<'_>) -> ANNError {
            ANNError::log_index_error(&value)
        }
    }

    struct MaybeTransient<'a> {
        messages: &'a Mutex<Vec<(String, u32)>>,
        value: u64,
        transient: bool,
    }

    impl<'a> MaybeTransient<'a> {
        fn new(messages: &'a Mutex<Vec<(String, u32)>>, value: u64, transient: bool) -> Self {
            Self {
                messages,
                value,
                transient,
            }
        }
    }

    impl<'a> ToRanked for MaybeTransient<'a> {
        type Transient = Bomb<'a>;
        type Error = Disarmed<'a>;

        fn to_ranked(self) -> RankedError<Self::Transient, Self::Error> {
            if self.transient {
                RankedError::Transient(Bomb::new(self.messages, self.value))
            } else {
                RankedError::Error(Disarmed::new(self.messages, self.value))
            }
        }

        fn from_transient(mut transient: Bomb<'a>) -> Self {
            transient.acknowledged = true;
            Self::new(transient.messages, transient.value, true)
        }

        fn from_error(error: Disarmed<'a>) -> Self {
            Self::new(error.messages, error.value, false)
        }
    }

    #[test]
    fn to_ranked_idempotent() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        let v = MaybeTransient::new(&messages, 10, true).to_ranked();
        assert!(matches!(v, RankedError::Transient(..)));

        // Disarm the bomb
        match v.to_ranked() {
            RankedError::Transient(v) => v.acknowledge(""),
            _ => panic!("wrong variant"),
        }

        let v = MaybeTransient::new(&messages, 10, false).to_ranked();
        assert!(matches!(v, RankedError::Error(..)));
        let v = v.to_ranked();
        assert!(matches!(v, RankedError::Error(..)));
    }

    #[test]
    fn error_ext_allow_transient_ok() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Non-Error Path.
        let v: usize = Result::<usize, MaybeTransient<'_>>::Ok(10)
            .allow_transient("hello")
            .unwrap()
            .unwrap();
        assert_eq!(v, 10);
        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn error_ext_allow_transient_with_ok() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Non-Error Path.
        let v: usize = Result::<usize, MaybeTransient<'_>>::Ok(10)
            .allow_transient_with(|| -> &str {
                panic!("this should not be called!");
            })
            .unwrap()
            .unwrap();
        assert_eq!(v, 10);
        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn error_ext_escalate_ok() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Non-Error Path.
        let v: usize = Result::<usize, MaybeTransient<'_>>::Ok(10)
            .escalate("hello")
            .unwrap();
        assert_eq!(v, 10);
        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn error_ext_escalate_with_ok() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Non-Error Path.
        let v: usize = Result::<usize, MaybeTransient<'_>>::Ok(10)
            .escalate_with(|| -> &str {
                panic!("this should not be called");
            })
            .unwrap();

        assert_eq!(v, 10);
        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn error_ext_allow_transient_transient() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Error path - transient
        let why = "foo";
        let line = line!();
        assert!(
            Result::<usize, MaybeTransient<'_>>::Err(MaybeTransient::new(&messages, 10, true))
                .allow_transient(why)
                .unwrap()
                .is_none()
        );

        let m = messages.lock().unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].1, line + 3);
        assert_eq!(m[0].0, format!("acknowledged: {}", why));
    }

    #[test]
    fn error_ext_allow_transient_with_transient() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Error path - transient
        let why = "foo";
        let mut called: bool = false;
        let line = line!();
        assert!(
            Result::<usize, MaybeTransient<'_>>::Err(MaybeTransient::new(&messages, 10, true))
                .allow_transient_with(|| {
                    called = true;
                    why
                })
                .unwrap()
                .is_none()
        );

        assert!(called);
        let m = messages.lock().unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].1, line + 3);
        assert_eq!(m[0].0, format!("acknowledged: {}", why));
    }

    #[test]
    fn error_ext_escalate_transient() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Error path - transient
        let why = "foo";
        let line = line!();
        assert!(
            Result::<usize, MaybeTransient<'_>>::Err(MaybeTransient::new(&messages, 10, true))
                .escalate(why)
                .is_err()
        );

        let m = messages.lock().unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].1, line + 3);
        assert_eq!(m[0].0, format!("escalated: {}", why));
    }

    #[test]
    fn error_ext_escalate_with_transient() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Error path - transient
        let why = "foo";
        let mut called: bool = false;
        let line = line!();
        assert!(
            Result::<usize, MaybeTransient<'_>>::Err(MaybeTransient::new(&messages, 10, true))
                .escalate_with(|| {
                    called = true;
                    why
                })
                .is_err()
        );

        assert!(called);
        let m = messages.lock().unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].1, line + 3);
        assert_eq!(m[0].0, format!("escalated: {}", why));
    }

    #[test]
    fn error_ext_allow_transient_error() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Error path - error
        let why = "foo";
        assert!(
            Result::<usize, MaybeTransient<'_>>::Err(MaybeTransient::new(&messages, 10, false))
                .allow_transient(why)
                .is_err()
        );

        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn error_ext_allow_transient_with_error() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Error path - error
        assert!(
            Result::<usize, MaybeTransient<'_>>::Err(MaybeTransient::new(&messages, 10, false))
                .allow_transient_with(|| -> &str {
                    panic!("should not be called");
                })
                .is_err()
        );

        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn error_ext_escalate_error() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Error path - error
        let why = "foo";
        assert!(
            Result::<usize, MaybeTransient<'_>>::Err(MaybeTransient::new(&messages, 10, false))
                .escalate(why)
                .is_err()
        );

        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn error_ext_escalate_with_error() {
        let messages = Mutex::new(Vec::<(String, u32)>::new());

        // Error path - error
        assert!(
            Result::<usize, MaybeTransient<'_>>::Err(MaybeTransient::new(&messages, 10, false))
                .escalate_with(|| -> &str {
                    panic!("should not be called");
                })
                .is_err()
        );

        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn test_infallible() {
        // Test that match_infallible can extract values from Ok results
        let result: Result<i32, Infallible> = Ok(42);
        let value = Infallible::match_infallible(result);
        assert_eq!(value, 42);

        // Test with different types
        let result: Result<String, Infallible> = Ok("hello".to_string());
        let value = Infallible::match_infallible(result);
        assert_eq!(value, "hello");

        // Test that Infallible converts to ANNError (though this should never happen in practice)
        // We can't actually construct an Infallible value to test this directly since it's unconstructable
        // But we can verify the From implementation exists by checking the type constraint
        fn _test_infallible_into_ann_error(_: Infallible) -> ANNError {
            ANNError::log_index_error("This should never be called")
        }
    }
}
