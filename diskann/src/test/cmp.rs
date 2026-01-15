/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

use crate::{ANNError, ANNResult};

/// A variation of [`PartialEq`] that provides diagnostics if two values are not equal.
///
/// The diagnostic chain is reported as an error chain inside an [`ANNError`].
///
/// Primitive types like integers and floating point numbers should return a descriptive
/// error when they do not compare equal.
///
/// Indexed types like [`Vec`] and tuples and aggregate types like `struct`s should invoke
/// [`VerboseEq`] recursively on each index/lfield. On the first mismatch,
/// [`ANNError::context`] should be used to record the index of the mismatch while preserving
/// the current error chain.
///
/// Following these guidelines will ensure that the full path to the mismatching entry is
/// preserved and reported even for deeply nested data structures.
///
/// See also: [`crate::test::cmp::verbose_eq`] and [`crate::test::cmp::assert_eq_verbose`].
pub(crate) trait VerboseEq {
    fn verbose_eq(&self, other: &Self) -> ANNResult<()>;
}

/// A macro helper to implement [`VerboseEq`] without quite needing a procedural macro.
///
/// ## Example
///
/// ```ignore
/// use crate::test::cmp::verbose_eq;
///
/// struct MyStruct {
///     a: usize,
///     b: usize,
/// }
///
/// verbose_eq!(MyStruct { a, b });
/// ```
///
/// For the time being, this requires duplicating the field names. However, this macro
/// does guard against changes to `MyStruct` being silently ignored by emitting a compile
/// error if the lists differ.
macro_rules! verbose_eq {
    ($struct:path { $($fields:ident),+ $(,)? }) => {
        impl $crate::test::cmp::VerboseEq for $struct {
            #[inline(never)]
            #[track_caller]
            fn verbose_eq(&self, other: &Self) -> $crate::ANNResult<()> {
                // Parameter unpacking like this is what guarantees that the fields supplied
                // to the macro align with the fields of the actual struct.
                let $struct { $($fields),+ } = self;

                // Recursively check each field - returning the first error with the name
                // of the mismatching field.
                $(
                    if let Err(err) = ($fields).verbose_eq(&other.$fields) {
                        return Err(err.context($crate::test::cmp::Field(stringify!($fields))));
                    }
                )+

                Ok(())
            }
        }
    }
}

pub(crate) use verbose_eq;

/// A macro that behaves like the standard library [`assert_eq!`], but uses [`VerboseEq`]
/// to provide more detailed information in the event of a mismatch.
macro_rules! assert_eq_verbose {
    ($left:expr, $right:expr $(,)?) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                use $crate::test::cmp::VerboseEq;
                if let Err(err) = (*left_val).verbose_eq(&*right_val) {
                    panic!(
                        "Assert failed with message\n\n{}\n\nLHS: {:?}\nRHS: {:?}",
                        err, &*left_val, &*right_val,
                    );
                }
            }
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                use $crate::test::cmp::VerboseEq;
                if let Err(err) = (*left_val).verbose_eq(&*right_val) {
                    panic!(
                        "Assert failed with message\n\n{}\n\nLHS: {:?}\nRHS: {:?}\n\n{}",
                        err, &*left_val, &*right_val, format_args!($($arg)+),
                    );
                }
            }
        }
    };
}

pub(crate) use assert_eq_verbose;

////////////////////
// Implementation //
////////////////////

/// Display implementation for recording a field mismatch.
#[derive(Debug)]
pub(crate) struct Field(pub(crate) &'static str);

impl std::fmt::Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "field \"{}\"", self.0)
    }
}

/// Error implementation for leaf type mismatch.
#[derive(Debug, Error)]
#[error("LHS {} is not equal to RHS {}", self.0, self.1)]
struct NotEq<T>(T, T)
where
    T: std::fmt::Display;

macro_rules! impl_via_partial_eq {
    ($T:ty) => {
        impl $crate::test::cmp::VerboseEq for $T {
            fn verbose_eq(&self, other: &Self) -> $crate::ANNResult<()> {
                if self != other {
                    Err($crate::ANNError::opaque(NotEq(self.clone(), other.clone())))
                } else {
                    Ok(())
                }
            }
        }
    };
    ($($Ts:ty),* $(,)?) => {
        $(impl_via_partial_eq!($Ts);)*
    }
}

impl_via_partial_eq!(
    u8, u16, u32, u64, i8, i16, i32, i64, usize, f32, f64, String,
);

macro_rules! impl_tuple {
    ($N:literal, { $($Ts:ident),+ $(,)? }, { $($Is:tt),+ $(,)? }) => {
        impl<$($Ts,)*> $crate::test::cmp::VerboseEq for ($($Ts,)+)
        where $($Ts: $crate::test::cmp::VerboseEq,)+
        {
            #[inline(never)]
            fn verbose_eq(&self, other: &Self) -> $crate::ANNResult<()> {
                $(
                    if let Err(err) = (self.$Is).verbose_eq(&other.$Is) {
                        return Err(err.context(Index {
                            failed: $Is,
                            total: $N,
                        }))
                    }
                )+

                Ok(())
            }
        }
    }
}

impl_tuple!(1, { T0 }, { 0 });
impl_tuple!(2, { T0, T1 }, { 0, 1 });
impl_tuple!(3, { T0, T1, T2 }, { 0, 1, 2 });

#[derive(Debug, Error)]
#[error("LHS vector has length {} while RHS has {}", self.0, self.1)]
struct UnequalLengths(usize, usize);

#[derive(Debug)]
struct Index {
    failed: usize,
    total: usize,
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "first mismatch on index {} of {}",
            self.failed, self.total
        )
    }
}

impl<T> VerboseEq for Vec<T>
where
    T: VerboseEq,
{
    #[inline(never)]
    fn verbose_eq(&self, other: &Self) -> ANNResult<()> {
        use crate::error::ErrorContext;

        if self.len() != other.len() {
            return Err(ANNError::opaque(UnequalLengths(self.len(), other.len())));
        }

        self.iter()
            .zip(other.iter())
            .enumerate()
            .try_for_each(|(i, (lhs, rhs))| {
                lhs.verbose_eq(rhs).with_context(|| Index {
                    failed: i,
                    total: self.len(),
                })
            })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test::assert_message_contains;

    // Built-in Types
    #[test]
    fn test_builtin() {
        // u8
        assert!(0u8.verbose_eq(&0u8).is_ok());
        assert!(0u8.verbose_eq(&1u8).is_err());

        // u16
        assert!(0u16.verbose_eq(&0u16).is_ok());
        assert!(0u16.verbose_eq(&1u16).is_err());

        // u32
        assert!(0u32.verbose_eq(&0u32).is_ok());
        assert!(0u32.verbose_eq(&1u32).is_err());

        // u64
        assert!(0u64.verbose_eq(&0u64).is_ok());
        assert!(0u64.verbose_eq(&1u64).is_err());

        // i8
        assert!(0i8.verbose_eq(&0i8).is_ok());
        assert!(0i8.verbose_eq(&1i8).is_err());

        // i16
        assert!(0i16.verbose_eq(&0i16).is_ok());
        assert!(0i16.verbose_eq(&1i16).is_err());

        // i32
        assert!(0i32.verbose_eq(&0i32).is_ok());
        assert!(0i32.verbose_eq(&1i32).is_err());

        // u64
        assert!(0u64.verbose_eq(&0u64).is_ok());
        assert!(0u64.verbose_eq(&1u64).is_err());

        // usize
        assert!(0usize.verbose_eq(&0usize).is_ok());
        assert!(0usize.verbose_eq(&1usize).is_err());

        // f32
        assert!(0f32.verbose_eq(&0f32).is_ok());
        assert!(0f32.verbose_eq(&1f32).is_err());

        // f32
        assert!(0f32.verbose_eq(&0f32).is_ok());
        assert!(0f32.verbose_eq(&1f32).is_err());

        // String
        {
            let a = "hello".to_string();
            let b = "world".to_string();
            assert!(a.verbose_eq(&a).is_ok());
            assert!(a.verbose_eq(&b).is_err());
        }
    }

    #[test]
    fn tuple_length_1() {
        let x = (1usize,);
        let y = (2usize,);

        assert!(x.verbose_eq(&x).is_ok());
        assert!(x.verbose_eq(&y).is_err());

        let msg = x.verbose_eq(&y).unwrap_err().to_string();
        assert_message_contains!(msg, "first mismatch on index 0 of 1");
        assert_message_contains!(msg, "LHS 1 is not equal to RHS 2");
    }

    #[test]
    fn tuple_length_2() {
        let x = 1usize;
        let y = 2usize;

        // Happy paths
        assert!((x, y).verbose_eq(&(x, y)).is_ok());
        assert!((y, x).verbose_eq(&(y, x)).is_ok());

        // Mismatch First
        let msg = (x, y).verbose_eq(&(y, x)).unwrap_err().to_string();
        assert_message_contains!(msg, "first mismatch on index 0 of 2");
        assert_message_contains!(msg, "LHS 1 is not equal to RHS 2");

        // Mismatch Second
        let msg = (x, y).verbose_eq(&(x, x)).unwrap_err().to_string();
        assert_message_contains!(msg, "first mismatch on index 1 of 2");
        assert_message_contains!(msg, "LHS 2 is not equal to RHS 1");
    }

    #[test]
    fn tuple_length_3() {
        let x = 1usize;
        let y = 2usize;
        let z = 3usize;

        // Happy paths
        assert!((x, y, z).verbose_eq(&(x, y, z)).is_ok());
        assert!((y, z, x).verbose_eq(&(y, z, x)).is_ok());

        // Mismatch First
        let msg = (x, y, z).verbose_eq(&(y, x, z)).unwrap_err().to_string();
        assert_message_contains!(msg, "first mismatch on index 0 of 3");
        assert_message_contains!(msg, "LHS 1 is not equal to RHS 2");

        // Mismatch Second
        let msg = (x, y, z).verbose_eq(&(x, x, z)).unwrap_err().to_string();
        assert_message_contains!(msg, "first mismatch on index 1 of 3");
        assert_message_contains!(msg, "LHS 2 is not equal to RHS 1");

        // Mismatch Third
        let msg = (x, y, z).verbose_eq(&(x, y, y)).unwrap_err().to_string();
        assert_message_contains!(msg, "first mismatch on index 2 of 3");
        assert_message_contains!(msg, "LHS 3 is not equal to RHS 2");
    }

    #[test]
    fn test_vector() {
        // Happy Path
        {
            let x = vec![1, 2, 3];
            let y = vec![1, 2, 3];
            assert!(x.verbose_eq(&y).is_ok());
        }

        // Mismatched lengths
        {
            let x = vec![1, 2];
            let y = vec![1, 2, 3];
            let msg = x.verbose_eq(&y).unwrap_err().to_string();
            assert_message_contains!(msg, "LHS vector has length 2 while RHS has 3");

            let msg = y.verbose_eq(&x).unwrap_err().to_string();
            assert_message_contains!(msg, "LHS vector has length 3 while RHS has 2");
        }

        // Mismatching entries
        {
            let x = vec![1, 2, 3];
            let y = vec![1, 2, 2];
            let msg = x.verbose_eq(&y).unwrap_err().to_string();
            assert_message_contains!(msg, "first mismatch on index 2 of 3");
            assert_message_contains!(msg, "LHS 3 is not equal to RHS 2");
        }
    }

    #[test]
    fn test_macro() {
        #[derive(Debug, Clone)]
        struct A {
            string: String,
            value: usize,
        }

        impl A {
            fn new(string: String, value: usize) -> Self {
                Self { string, value }
            }
        }

        verbose_eq!(A { string, value });

        #[derive(Debug, Clone)]
        struct B {
            a: A,
            value: usize,
        }

        impl B {
            fn new(a: A, value: usize) -> Self {
                Self { a, value }
            }
        }

        verbose_eq!(B { a, value });

        {
            let lhs = A::new("hello".into(), 20);
            let rhs1 = A::new("world".into(), 20);
            let rhs2 = A::new("hello".into(), 10);

            assert!(lhs.verbose_eq(&lhs).is_ok());

            let msg = lhs.verbose_eq(&rhs1).unwrap_err().to_string();
            assert_message_contains!(msg, "field \"string\"");
            assert_message_contains!(msg, "LHS hello is not equal to RHS world");

            let msg = lhs.verbose_eq(&rhs2).unwrap_err().to_string();
            assert_message_contains!(msg, "field \"value\"");
            assert_message_contains!(msg, "LHS 20 is not equal to RHS 10");
        }

        {
            let a_lhs = A::new("hello".into(), 20);
            let a_rhs1 = A::new("world".into(), 20);
            let a_rhs2 = A::new("hello".into(), 10);

            let lhs = B::new(a_lhs.clone(), 10);
            let rhs_1 = B::new(a_rhs1, 10);
            let rhs_2 = B::new(a_rhs2, 10);
            let rhs_3 = B::new(a_lhs, 25);

            assert!(lhs.verbose_eq(&lhs).is_ok());

            let msg = lhs.verbose_eq(&rhs_1).unwrap_err().to_string();
            assert_message_contains!(msg, "field \"a\"");
            assert_message_contains!(msg, "field \"string\"");
            assert_message_contains!(msg, "LHS hello is not equal to RHS world");

            let msg = lhs.verbose_eq(&rhs_2).unwrap_err().to_string();
            assert_message_contains!(msg, "field \"a\"");
            assert_message_contains!(msg, "field \"value\"");
            assert_message_contains!(msg, "LHS 20 is not equal to RHS 10");

            let msg = lhs.verbose_eq(&rhs_3).unwrap_err().to_string();
            assert_message_contains!(msg, "field \"value\"");
            assert_message_contains!(msg, "LHS 10 is not equal to RHS 25");
        }
    }

    #[test]
    fn test_assert_happy_path() {
        assert_eq_verbose!(2usize, 2usize);
        assert_eq_verbose!(2usize, 2usize, "some context: {}", 10);
    }

    #[test]
    #[should_panic(expected = "Assert failed with message")]
    fn test_assert_verbose_eq_no_context() {
        assert_eq_verbose!(1usize, 2usize);
    }

    #[test]
    #[should_panic(expected = "Assert failed with message")]
    fn test_assert_verbose_eq() {
        assert_eq_verbose!(1usize, 2usize, "some context: {}, {}", "a", 10);
    }
}
