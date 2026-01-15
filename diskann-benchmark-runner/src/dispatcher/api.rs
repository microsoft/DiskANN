/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::{self, Display, Formatter};

/// Successful matches from [`DispatchRule`] will return `MatchScores`.
///
/// A lower numerical value indicates a better match for purposes of overload resolution.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MatchScore(pub u32);

impl Display for MatchScore {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "success ({})", self.0)
    }
}

/// Successful matches from [`DispatchRule`] will return `FailureScores`.
///
/// A lower numerical value indicates a better match, which can help when compiling a
/// list of considered and rejected candidates.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FailureScore(pub u32);

impl Display for FailureScore {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "fail ({})", self.0)
    }
}

/// A version of [`FailureScore`] that contains the score as well as the reason for the
/// failure.
pub struct TaggedFailureScore<'a> {
    pub(crate) score: u32,
    pub(crate) why: Box<dyn std::fmt::Display + 'a>,
}

impl TaggedFailureScore<'_> {
    /// Return the failure score for `Self`.
    pub fn score(&self) -> FailureScore {
        FailureScore(self.score)
    }
}

impl fmt::Debug for TaggedFailureScore<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaggedFailureScore")
            .field("score", &self.score)
            .field("why", &self.why.to_string())
            .finish()
    }
}

impl Display for TaggedFailureScore<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.why)
    }
}

/// The primary trait for conducting dispatch matches from the type `From`.
pub trait DispatchRule<From>: Sized {
    /// Errors that can occur during `convert`.
    type Error: std::fmt::Debug + std::fmt::Display + 'static;

    /// Attempt to match the value `From` to the type represented by `Self`.
    ///
    /// If `from` has a compatible value, return `Ok(score)` where `score` attempts to
    /// describe how good of a fit `from` is to allow for overload resolution.
    ///
    /// If `from` is incompatible, return `Err(score)`
    fn try_match(from: &From) -> Result<MatchScore, FailureScore>;

    /// Perform the actual conversion.
    ///
    /// It is expected that this method will only be called when `try_match(&from)` returns
    /// success. An error type can be returned due to either:
    ///
    /// 1. `try_match` returning `Ok()` erroneously due to an incorrect implementation.
    /// 2. `from` originally looked like a match, but broke some invariant of `Self`s
    ///    constructor.
    fn convert(from: From) -> Result<Self, Self::Error>;

    //////////////////////
    // Provided Methods //
    //////////////////////

    /// Write a description of the dispatch rule and outcome to the formatter.
    ///
    /// If `from.is_none()`, then a description of `Self` should be provided.
    ///
    /// Otherwise, the implementation should provide a description of the dispatching logic
    /// (success or failure) for the argument.
    fn description(f: &mut Formatter<'_>, _from: Option<&From>) -> fmt::Result {
        write!(f, "<no description>")
    }

    /// The equivalent of `try_match` but returns a reason for a failed score.
    ///
    /// This allows emission of diagnostics for method mismatches.
    ///
    /// The provided implementation of this method calls [`Self::description(_, Some(from))`].
    fn try_match_verbose<'a>(from: &'a From) -> Result<MatchScore, TaggedFailureScore<'a>>
    where
        Self: 'a,
    {
        match Self::try_match(from) {
            Ok(score) => Ok(score),
            Err(score) => Err(TaggedFailureScore {
                score: score.0,
                why: Box::new(Why::<From, Self>::new(from)),
            }),
        }
    }
}

/// A helper struct to help dscribe the reason for a match failure.
#[derive(Debug, Clone, Copy)]
pub struct Why<'a, From, To> {
    from: &'a From,
    _to: std::marker::PhantomData<To>,
}

impl<'a, From, To> Why<'a, From, To> {
    pub fn new(from: &'a From) -> Self {
        Self {
            from,
            _to: std::marker::PhantomData,
        }
    }
}

impl<From, To> std::fmt::Display for Why<'_, From, To>
where
    To: DispatchRule<From>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        To::description(f, Some(self.from))
    }
}

/// A helper struct to retrieve the empty description from a [`DispatchRule`].
#[derive(Debug, Clone, Copy)]
pub struct Description<From, To> {
    _from: std::marker::PhantomData<From>,
    _to: std::marker::PhantomData<To>,
}

impl<From, To> Description<From, To> {
    pub fn new() -> Self {
        Self {
            _from: std::marker::PhantomData,
            _to: std::marker::PhantomData,
        }
    }
}

impl<From, To> Default for Description<From, To> {
    fn default() -> Self {
        Self::new()
    }
}

impl<From, To> std::fmt::Display for Description<From, To>
where
    To: DispatchRule<From>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        To::description(f, None)
    }
}

/////////////////////////////
// Blanket Implementations //
/////////////////////////////

/// A score assigned to implicit matches either via the identity transformation or through
/// mut-ref to ref conversion.
pub const IMPLICIT_MATCH_SCORE: MatchScore = MatchScore(100000);

impl<T: Sized> DispatchRule<T> for T {
    type Error = std::convert::Infallible;

    fn try_match(_from: &T) -> Result<MatchScore, FailureScore> {
        Ok(IMPLICIT_MATCH_SCORE)
    }

    fn convert(from: T) -> Result<T, Self::Error> {
        Ok(from)
    }

    fn description(f: &mut Formatter<'_>, from: Option<&T>) -> fmt::Result {
        match from {
            None => write!(f, "{}", std::any::type_name::<T>()),
            Some(_) => write!(f, "identity match"),
        }
    }
}

// Allow mutable references to be forwarded to const-references.
impl<'a, T: Sized> DispatchRule<&'a mut T> for &'a T {
    type Error = std::convert::Infallible;

    fn try_match(_from: &&'a mut T) -> Result<MatchScore, FailureScore> {
        Ok(IMPLICIT_MATCH_SCORE)
    }

    fn convert(from: &'a mut T) -> Result<&'a T, Self::Error> {
        Ok(from)
    }

    fn description(f: &mut Formatter<'_>, from: Option<&&'a mut T>) -> fmt::Result {
        match from {
            None => write!(f, "&{}", std::any::type_name::<T>()),
            Some(_) => write!(f, "identity match"),
        }
    }
}

/// # Lifetime Mapping
///
/// The types in signatures for dispatches need to be `'static` due to Rust.
/// However, it is nice to allow objects with lifetimes to cross the dispatcher boundary.
///
/// The `Map` trait facilitates this by allowing `'static` types to have an optional
/// lifetime attached as a generic associated type.
///
/// This associated type is that is what is actually given to dispatcher methods.
///
/// ## Example
///
/// To pass a `Vec` across a dispatcher boundary, we can use the [`Type`] helper:
///
/// ```
/// use diskann_benchmark_runner::dispatcher::{Dispatcher1, Type};
///
/// let mut d = Dispatcher1::<&'static str, Type<Vec<f32>>>::new();
/// d.register::<_, Type<Vec<f32>>>("method",  |_: Vec<f32>| "called");
/// assert_eq!(d.call(vec![1.0]), Some("called"));
/// ```
///
/// This is a bit tedious to write every time, so instead types can implement [`Map`] for
/// themselves:
///
/// ```
/// use diskann_benchmark_runner::{self_map, dispatcher::{Dispatcher1}};
///
/// struct MyNum(f32);
/// self_map!(MyNum);
///
/// // Now, `MyNum` can be used directly in dispatcher signatures.
/// let mut d = Dispatcher1::<f32, MyNum>::new();
/// d.register::<_, MyNum>("method", |n: MyNum| n.0);
/// assert_eq!(d.call(MyNum(0.0)), Some(0.0));
/// ```
///
/// ## See Also:
///
/// * [`Ref`]: Mapping References
/// * [`MutRef`]: Mapping Mutable References
/// * [`Type`]: Mapper for generic types
/// * [`crate::self_map!`]: Allow types to represent themselves in dispatcher signatures.
///
pub trait Map: 'static {
    /// The actual type provided to the dispatcher, with an optional additional lifetime.
    type Type<'a>;
}

/// Allow references to cross dispatcher boundaries as shown in the following example:
///
/// ```
/// use diskann_benchmark_runner::dispatcher::{Dispatcher1, Ref};
///
/// let mut d = Dispatcher1::<*const f32, Ref<[f32]>>::new();
/// d.register::<_, Ref<[f32]>>("method", |data: &[f32]| data.as_ptr());
///
/// let v = vec![1.0, 2.0];
/// assert_eq!(d.call(&v), Some(v.as_ptr()));
/// ```
pub struct Ref<T: ?Sized + 'static>(std::marker::PhantomData<T>);

impl<T: ?Sized> Map for Ref<T> {
    type Type<'a> = &'a T;
}

/// Allow mutable references to cross dispatcher boundaries as shown below.
///
/// ```
/// use diskann_benchmark_runner::dispatcher::{Dispatcher1, MutRef};
///
/// let mut d = Dispatcher1::<(), MutRef<Vec<f32>>>::new();
/// d.register::<_, MutRef<Vec<f32>>>("method", |v: &mut Vec<f32>| v.push(0.0));
///
/// let mut v = Vec::new();
/// d.call(&mut v).unwrap();
/// assert_eq!(&v, &[0.0]);
/// ```
pub struct MutRef<T: ?Sized + 'static>(std::marker::PhantomData<T>);
impl<T: ?Sized> Map for MutRef<T> {
    type Type<'a> = &'a mut T;
}

pub struct Type<T: 'static>(std::marker::PhantomData<T>);
impl<T> Map for Type<T> {
    type Type<'a> = T;
}

#[macro_export]
macro_rules! self_map {
    ($($type:tt)*) => {
        impl $crate::dispatcher::Map for $($type)* {
            type Type<'a> = $($type)*;
        }
    }
}

self_map!(bool);
self_map!(usize);
self_map!(u8);
self_map!(u16);
self_map!(u32);
self_map!(u64);
self_map!(u128);
self_map!(i8);
self_map!(i16);
self_map!(i32);
self_map!(i64);
self_map!(i128);
self_map!(String);
self_map!(f32);
self_map!(f64);

/// Reasons for a method call mismatch.
///
/// The name of the associated method can be queried using `self.method()` and reasons
/// are obtained in `self.mismatches()`.
pub struct ArgumentMismatch<'a, const N: usize> {
    pub(crate) method: &'a str,
    pub(crate) mismatches: [Option<Box<dyn std::fmt::Display + 'a>>; N],
}

impl<'a, const N: usize> ArgumentMismatch<'a, N> {
    /// Return the name of the associated method.
    pub fn method(&self) -> &str {
        self.method
    }

    /// Return a slice of reasons for method match failure.
    ///
    /// The returned slice contains one entry per argument. An entry is `None` if that
    /// argument matched the input value.
    ///
    /// If the argument did not match the input value, then the corresponding
    /// [`std::fmt::Display`] object can be used to retrieve the reason.
    pub fn mismatches(&self) -> &[Option<Box<dyn std::fmt::Display + 'a>>; N] {
        &self.mismatches
    }
}

/// Return the signature for an argument type.
pub struct Signature(pub(crate) fn(&mut Formatter<'_>) -> std::fmt::Result);

impl std::fmt::Display for Signature {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        (self.0)(f)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_score() {
        let x = MatchScore(10);
        let y = MatchScore(20);
        assert!(x < y);
        assert!(x <= y);
        assert!(x <= x);
        assert!(x == x);
        assert!(x != y);

        assert!(y == y);
        assert!(y != x);
        assert!(y > x);
        assert!(y >= x);

        assert_eq!(x.to_string(), "success (10)");
    }

    #[test]
    fn test_fail_score() {
        let x = FailureScore(10);
        let y = FailureScore(20);
        assert!(x < y);
        assert!(x <= y);
        assert!(x <= x);
        assert!(x == x);
        assert!(x != y);

        assert!(y == y);
        assert!(y != x);
        assert!(y > x);
        assert!(y >= x);

        assert_eq!(x.to_string(), "fail (10)");
    }

    #[test]
    fn test_tagged_failure() {
        let tagged = TaggedFailureScore {
            score: 10,
            why: Box::new(20),
        };

        assert_eq!(tagged.score(), FailureScore(10));

        // Formatted goes through the inner formatter.
        assert_eq!(tagged.to_string(), "20");

        assert_eq!(
            format!("{:?}", tagged),
            "TaggedFailureScore { score: 10, why: \"20\" }"
        );
    }

    enum TestEnum {
        A,
        B,
    }

    struct TestType;

    impl DispatchRule<TestEnum> for TestType {
        type Error = std::convert::Infallible;
        fn try_match(x: &TestEnum) -> Result<MatchScore, FailureScore> {
            match x {
                TestEnum::A => Ok(MatchScore(10)),
                TestEnum::B => Err(FailureScore(20)),
            }
        }

        fn convert(x: TestEnum) -> Result<Self, Self::Error> {
            assert!(matches!(x, TestEnum::A));
            Ok(TestType)
        }

        fn description(f: &mut Formatter<'_>, from: Option<&TestEnum>) -> fmt::Result {
            match from {
                None => write!(f, "TestEnum::A"),
                Some(value) => match value {
                    TestEnum::A => write!(f, "success"),
                    TestEnum::B => write!(f, "expected TestEnum::B"),
                },
            }
        }
    }

    #[test]
    fn test_dispatch_helpers() {
        let desc = Description::<TestEnum, TestType>::default().to_string();
        assert_eq!(desc, "TestEnum::A");

        let a = TestEnum::A;
        let why = Why::<_, TestType>::new(&a).to_string();
        assert_eq!(why, "success");

        let b = TestEnum::B;
        let why = Why::<_, TestType>::new(&b).to_string();
        assert_eq!(why, "expected TestEnum::B");

        let result = TestType::try_match_verbose(&a).unwrap();
        assert_eq!(result, MatchScore(10));

        let result = TestType::try_match_verbose(&b).unwrap_err();
        assert_eq!(result.score(), FailureScore(20));
        assert_eq!(result.to_string(), "expected TestEnum::B");

        TestType::convert(TestEnum::A).unwrap();
    }

    #[test]
    fn test_implicit_conversions() {
        // Identity
        let x = f32::try_match(&0.0f32).unwrap();
        assert_eq!(x, IMPLICIT_MATCH_SCORE);

        let x = f32::convert(0.0f32).unwrap();
        assert_eq!(x, 0.0f32);

        let x = <&f32>::try_match(&&mut 0.0f32).unwrap();
        assert_eq!(x, IMPLICIT_MATCH_SCORE);

        let mut x: f32 = 10.0;
        let x = <&f32>::convert(&mut x).unwrap();
        assert_eq!(*x, 10.0);

        assert_eq!(Description::<f32, f32>::new().to_string(), "f32");
        assert_eq!(Why::<f32, f32>::new(&0.0f32).to_string(), "identity match");

        assert_eq!(Description::<&mut f32, &f32>::new().to_string(), "&f32");
        assert_eq!(
            Why::<&mut f32, &f32>::new(&&mut 0.0f32).to_string(),
            "identity match"
        );
    }

    #[test]
    #[should_panic]
    fn convert_panics() {
        let _ = TestType::convert(TestEnum::B);
    }
}
