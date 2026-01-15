/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Dynamic Dispatcher
//!
//! This crate implements a family of generic structures supporting pull-driven generic
//! multiple dispatch.
//!
//! In other words, it allows functions to be registered in a central location, enables
//! value-to-type lifting of arguments, overload resolution, and utilities for inspecting
//! failures for diagnostic reporting.
//!
//! # Quick Example
//!
//! Suppose we have a small collection of operations for which we would like to specialize
//! for some type `T`, which we may wish to alter from time to time.
//!
//! Furthermore, suppose this operation can returns a `String`.
//!
//! We can do that fairly easily:
//! ```
//! use diskann_benchmark_runner::dispatcher::{self, examples::{DataType, Type}};
//!
//! // A dynamic dispatcher that takes 1 argument of type `DataType` and returns a `String`.
//! let mut d = dispatcher::Dispatcher1::<String, DataType>::new();
//!
//! // We can register two methods with the dispatcher.
//! d.register::<_, Type<f32>>("method-a", |_: Type<f32>| "called method A".to_string());
//! d.register::<_, Type<f64>>("method-b", |_: Type<f64>| "called method B".to_string());
//!
//! // We can now verify that these methods are reachable.
//! assert_eq!(&d.call(DataType::Float32).unwrap(), "called method A");
//! assert_eq!(&d.call(DataType::Float64).unwrap(), "called method B");
//!
//! // If we try to call the dispatcher with a unregistered value for `DataType`, we
//! // get `None` as a result.
//! assert!(d.call(DataType::UInt8).is_none());
//!
//! // But now suppose that we can implement a generic method, taking *all* data types.
//! //
//! // We can register that method and call it.
//! d.register::<_, DataType>("generic", |_: DataType| "called generic method".to_string());
//! assert_eq!(&d.call(DataType::UInt8).unwrap(), "called generic method");
//!
//! // However, more specific methods will be called if available.
//! assert_eq!(&d.call(DataType::Float32).unwrap(), "called method A");
//!
//! // This is not order dependent.
//! //
//! // If we register yet another method, this time specialized for `UInt8`, it will get
//! // called when applicable.
//! d.register::<_, Type<u8>>("method-c", |_: Type<u8>| "called method C".to_string());
//! assert_eq!(&d.call(DataType::UInt8).unwrap(), "called method C");
//! ```

mod api;
mod dispatch;

pub mod examples;

pub use api::{
    ArgumentMismatch, Description, DispatchRule, FailureScore, Map, MatchScore, MutRef, Ref,
    Signature, TaggedFailureScore, Type, Why, IMPLICIT_MATCH_SCORE,
};

//////////////////////
// Dispatch Related //
//////////////////////

pub use dispatch::{
    Dispatch1, Dispatch2, Dispatch3, Dispatcher1, Dispatcher2, Dispatcher3, Method1, Method2,
    Method3,
};

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use super::*;

    use crate::self_map;

    ///////////
    // Types //
    ///////////

    struct TestType<T> {
        _phantom: PhantomData<T>,
    }

    impl<T> TestType<T> {
        fn new() -> Self {
            Self {
                _phantom: PhantomData,
            }
        }
    }

    impl<T: 'static> Map for TestType<T> {
        type Type<'a> = Self;
    }

    #[derive(Debug)]
    enum TypeEnum {
        Float32,
        Int8,
        UInt8,
    }

    self_map!(TypeEnum);

    impl DispatchRule<TypeEnum> for TestType<f32> {
        type Error = std::convert::Infallible;

        fn try_match(from: &TypeEnum) -> Result<MatchScore, FailureScore> {
            match from {
                TypeEnum::Float32 => Ok(MatchScore(0)),
                _ => Err(FailureScore(0)),
            }
        }

        fn convert(from: TypeEnum) -> Result<Self, Self::Error> {
            assert!(Self::try_match(&from).is_ok());
            Ok(Self::new())
        }
    }

    impl DispatchRule<TypeEnum> for TestType<i8> {
        type Error = std::convert::Infallible;

        fn try_match(from: &TypeEnum) -> Result<MatchScore, FailureScore> {
            match from {
                TypeEnum::Int8 => Ok(MatchScore(0)),
                _ => Err(FailureScore(0)),
            }
        }

        fn convert(from: TypeEnum) -> Result<Self, Self::Error> {
            assert!(Self::try_match(&from).is_ok());
            Ok(Self::new())
        }
    }

    //////////////
    // UnaryOps //
    //////////////

    enum UnaryOp {
        Square,
        Double,
        DoesNotExist,
    }

    struct Square;
    struct Double;

    self_map!(UnaryOp);
    self_map!(Square);
    self_map!(Double);

    impl DispatchRule<UnaryOp> for Square {
        type Error = std::convert::Infallible;

        fn try_match(from: &UnaryOp) -> Result<MatchScore, FailureScore> {
            match from {
                UnaryOp::Square => Ok(MatchScore(0)),
                _ => Err(FailureScore(0)),
            }
        }

        fn convert(from: UnaryOp) -> Result<Self, Self::Error> {
            assert!(Self::try_match(&from).is_ok());
            Ok(Self)
        }
    }

    impl DispatchRule<UnaryOp> for Double {
        type Error = std::convert::Infallible;

        fn try_match(from: &UnaryOp) -> Result<MatchScore, FailureScore> {
            match from {
                UnaryOp::Double => Ok(MatchScore(0)),
                _ => Err(FailureScore(0)),
            }
        }

        fn convert(from: UnaryOp) -> Result<Self, Self::Error> {
            assert!(Self::try_match(&from).is_ok());
            Ok(Self)
        }
    }

    ///////////////////
    // Test Routines //
    ///////////////////

    #[test]
    fn test_empty_description() {
        assert_eq!(
            Description::<UnaryOp, Double>::new().to_string(),
            "<no description>"
        );
        assert_eq!(
            Why::<UnaryOp, Double>::new(&UnaryOp::Double).to_string(),
            "<no description>"
        );
    }

    #[test]
    fn test_dispatch1() {
        let mut dispatcher = Dispatcher1::<&'static str, TypeEnum>::new();
        dispatcher.register::<_, TestType<f32>>("method1", |_: TestType<f32>| "float32");
        dispatcher.register::<_, TestType<i8>>("method2", |_: TestType<i8>| "int8");

        assert_eq!(dispatcher.call(TypeEnum::Int8), Some("int8"));
        assert_eq!(dispatcher.call(TypeEnum::Float32), Some("float32"));
        assert_eq!(dispatcher.call(TypeEnum::UInt8), None);

        let mut dispatcher = Dispatcher1::<(), MutRef<[f32]>>::new();
        dispatcher.register::<_, MutRef<[f32]>>("method1", |_: &mut [f32]| println!("hello world"));
    }

    #[test]
    fn test_dispatch2() {
        let mut dispatcher = Dispatcher2::<u64, UnaryOp, u64>::new();
        dispatcher.register::<_, Square, u64>("square", |_: Square, x: u64| x * x);
        dispatcher.register::<_, Double, u64>("double", |_: Double, x: u64| 2 * x);

        assert_eq!(dispatcher.call(UnaryOp::Square, 10).unwrap(), 100);
        assert_eq!(dispatcher.call(UnaryOp::Double, 10).unwrap(), 20);
        assert_eq!(dispatcher.call(UnaryOp::DoesNotExist, 0), None);

        let mut dispatcher = Dispatcher2::<(), UnaryOp, MutRef<u64>>::new();
        dispatcher.register::<_, Square, MutRef<u64>>("square", |_: Square, x: &mut u64| *x *= *x);
        dispatcher.register::<_, Double, MutRef<u64>>("double", |_: Double, x: &mut u64| *x *= 2);

        let mut x: u64 = 10;
        dispatcher.call(UnaryOp::Square, &mut x).unwrap();
        assert_eq!(x, 100);

        dispatcher.call(UnaryOp::Double, &mut x).unwrap();
        assert_eq!(x, 200);
    }
}
