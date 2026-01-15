/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Formatter;

use super::{
    ArgumentMismatch, DispatchRule, FailureScore, Map, MatchScore, Signature, TaggedFailureScore,
};

/// Return `Some` if all the entries in `Input` are `Ok(MatchScore)`.
///
/// Otherwise, return None
fn coalesce<const N: usize>(
    input: &[Result<MatchScore, FailureScore>; N],
) -> Option<[MatchScore; N]> {
    let mut output = [MatchScore(0); N];
    for i in 0..N {
        output[i] = match input[i] {
            Ok(score) => score,
            Err(_) => return None,
        }
    }
    Some(output)
}

/// Return `true` if all values in `inpu` are `Ok`.
fn all_match<const N: usize, T>(input: &[Result<MatchScore, T>; N]) -> bool {
    input.iter().all(|i| matches!(i, Ok(MatchScore(_))))
}

/// A method match along with a tagged failure score.
///
/// This is used as part of the match failure debugging process.
struct TaggedMatch<'a, const N: usize> {
    method: &'a str,
    score: [Result<MatchScore, TaggedFailureScore<'a>>; N],
}

impl<'a, const N: usize> From<TaggedMatch<'a, N>> for ArgumentMismatch<'a, N> {
    fn from(value: TaggedMatch<'a, N>) -> Self {
        ArgumentMismatch {
            method: value.method,
            mismatches: value.score.map(|r| match r {
                Ok(_) => None,
                Err(tagged) => Some(tagged.why),
            }),
        }
    }
}

/// An ordered priority queue that keeps track of the "closest" mismatches.
struct Queue<'a, const N: usize> {
    buffer: Vec<TaggedMatch<'a, N>>,
    max_methods: usize,
}

impl<'a, const N: usize> Queue<'a, N> {
    fn new(max_methods: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_methods),
            max_methods,
        }
    }

    fn finish(self) -> Vec<ArgumentMismatch<'a, N>> {
        self.buffer.into_iter().map(|m| m.into()).collect()
    }

    /// Insert `r` into the queue in sorted order.
    ///
    /// Returns `Err(())` if all entries in `r` are matches. This provies a means for
    /// algorithms reporting errors to detect if in fact the collection of arguments
    /// are dispatchable and debugging is not actually needed.
    fn push(&mut self, y: TaggedMatch<'a, N>) -> Result<(), ()> {
        use std::cmp::Ordering;

        if all_match(&y.score) {
            return Err(());
        }

        // Now we get the fun part of ranking methods.
        // We rank first on `MatchScore`, then on `FailureScore`.
        let lt = |x: &TaggedMatch<'a, N>| {
            for i in 0..N {
                let xi = &x.score[i];
                let yi = &y.score[i];
                match xi {
                    Ok(MatchScore(x_score)) => match yi {
                        Ok(MatchScore(y_score)) => match x_score.cmp(y_score) {
                            Ordering::Equal => {}
                            strict => return strict,
                        },
                        Err(_) => {
                            return Ordering::Less;
                        }
                    },
                    Err(TaggedFailureScore { score: x_score, .. }) => match yi {
                        Ok(_) => {
                            return Ordering::Greater;
                        }
                        Err(TaggedFailureScore { score: y_score, .. }) => {
                            match x_score.cmp(y_score) {
                                Ordering::Equal => {}
                                strict => return strict,
                            }
                        }
                    },
                }
            }
            Ordering::Equal
        };

        // `binary_search_by` will always return an index that will allow the key to be
        // placed in sorted order.
        //
        // We do not care if the method is present or not, we just want the index.
        let i = match self.buffer.binary_search_by(lt) {
            Ok(i) => i,
            Err(i) => i,
        };

        if self.buffer.len() == self.max_methods {
            // No need to insert, it's greater than our worst match so far.
            if i > self.buffer.len() {
                return Ok(());
            }
            self.buffer.insert(i, y);
            self.buffer.truncate(self.max_methods);
        } else {
            self.buffer.insert(i, y);
        }
        Ok(())
    }
}

pub trait Sealed {}

macro_rules! implement_dispatch {
    ($trait:ident,
     $method:ident,
     $dispatcher:ident,
     $N:literal,
     { $($T:ident )+ },
     { $($x:ident )+ },
     { $($A:ident )+ },
     { $($lf:lifetime )+ }
    ) => {
        /// A dispatchable method.
        ///
        /// # Macro Expansion
        ///
        /// Generates the code below:
        /// ```text
        /// pub trait DispatcherN<R, T0, T1, ...>
        /// where
        ///     T0: Map,
        ///     T1: Map,
        ///     ...,
        /// {
        ///     fn try_match(&self, x0: &T0::Type<'_>, x1: &T1::Type<'_>, ...);
        ///
        ///     fn call(&self, x0: T0::Type<'_>, x1: T1::Type<'_), ...) -> R;
        ///
        ///     fn signatures(&self) -> [Signature; N];
        ///
        ///     fn try_match_verbose<'a, 'a0, 'a1, ...>(
        ///         &'a self,
        ///         x0: &'a T0::Type<'a0>,
        ///         x1: &'a T1::Type<'a1>,
        ///         ...
        ///     ) -> [Result<MatchScore, TaggedFailureScore<'a>>; N]
        ///     where
        ///         'a0: 'a,
        ///         'a1: 'a,
        ///         ...;
        /// }
        /// ```
        pub trait $trait<R, $($T,)*>: Sealed
        where
            $($T: Map,)*
        {
            /// Invoke [`DispatchRule::try_match`] on each argument/type pair where the type
            /// comes from the backend method.
            ///
            /// Return all results.
            fn try_match(&self, $($x: &$T::Type<'_>,)*) -> [Result<MatchScore, FailureScore>; $N];

            /// Invoke this method with the given types, invoking [`DispatchRule::convert`]
            /// on each argument to the target types of the backend method.
            ///
            /// This function is only safe to call if [`Self::try_match`] returns a success.
            /// Calling this method incorrectly may panic.
            ///
            /// # Panics
            ///
            /// Panics if any call to [`DispatchRule::convert`] fails.
            fn call(&self, $($x: $T::Type<'_>,)*) -> R;

            /// Return the signatures for each back-end argument type.
            fn signatures(&self) -> [Signature; $N];

            /// The equivalent of [`Self::try_match`], but using the
            /// [`DispatchRule::try_match_verbose`] interface.
            ///
            /// This provides a method for inspecting the reason for match failures.
            fn try_match_verbose<'a, $($lf,)*>(
                &'a self,
                $($x: &'a $T::Type<$lf>,)*
            ) -> [Result<MatchScore, TaggedFailureScore<'a>>; $N]
            where
                $($lf: 'a,)*;
        }

        /// # Macro Expansion
        ///
        /// ```text
        /// pub struct MethodN<R, A0, A1, ...>
        /// where
        ///     A0: Map,
        ///     A1: Map,
        ///     ...,
        /// {
        ///     f: Box<dyn for<'a0, 'a1, ...> Fn(A0::Type<'a0>, A1::Type<'a1>, ...) -> R>,
        ///     _types: std::marker::PhantomData<(A0, A1, ...)>,
        /// }
        /// ```
        pub struct $method<R, $($A,)*>
        where
            $($A: Map,)*
        {
            f: Box<dyn for<$($lf,)*> Fn($($A::Type<$lf>,)*) -> R>,
            _types: std::marker::PhantomData<($($A,)*)>,
        }

        /// # Macro Expansion
        ///
        /// ```text
        /// impl <R, A0, A1, ...> MethodN<R, A0, A1, ...>
        /// where
        ///     R: 'static,
        ///     A0: Map,
        ///     A1: Map,
        ///     ...,
        /// {
        ///     pub fn new<F>(f: F) -> Self
        ///     where
        ///         F: for<'a0, 'a1, ...> Fn(A0::Type<'a0>, A1::Type<'a1>, ...) -> R + 'static,
        ///     {
        ///         Self {
        ///             f: Box::new(f),
        ///             _types: std::marker::PhantomData,
        ///         }
        ///     }
        /// }
        /// ```
        impl<R, $($A,)*> $method<R, $($A,)*>
        where
            $($A: Map,)*
        {
            fn new<F>(f: F) -> Self
            where
                F: for<$($lf,)*> Fn($($A::Type<$lf>,)*) -> R + 'static,
            {
                Self {
                    f: Box::new(f),
                    _types: std::marker::PhantomData,
                }
            }
        }

        impl<R, $($A,)*> Sealed for $method<R, $($A,)*>
        where
            $($A: Map,)*
        {}

        impl<R, $($T,)* $($A,)*> $trait<R, $($T,)*> for $method<R, $($A,)*>
        where
            $($T: Map,)*
            $($A: Map,)*
            $(for<'a> $A::Type<'a>: DispatchRule<$T::Type<'a>>,)*
        {
            fn try_match(&self, $($x: &$T::Type<'_>,)*) -> [Result<MatchScore, FailureScore>; $N] {
                // Splat out all the pair-wise `try_match`es.
                [$($A::Type::try_match($x),)*]
            }

            fn call(&self, $($x: $T::Type<'_>,)*) -> R {
                // Convert and unwrap all pair-wise matches.
                (self.f)($($A::Type::convert($x).unwrap(),)*)
            }

            fn signatures(&self) -> [Signature; $N] {
                // The strategy here involves decaying a stateless lambda to a function
                // pointer, and generating one such lambda for each input type.
                //
                // Note that we need to couple it with its corresponding dispatch type
                // to ensure we get routed to the correct description.
                [
                    $(Signature(|f: &mut Formatter<'_>| {
                        $A::Type::description(f, None::<&$T::Type<'_>>)
                    }),)*
                ]
            }

            fn try_match_verbose<'a, $($lf,)*>(
                &self,
                $($x: &'a $T::Type<$lf>,)*
            ) -> [Result<MatchScore, TaggedFailureScore<'a>>; $N]
            where
                $($lf: 'a,)*
            {
                // Simply construct an array by calling `try_match_verbose` on each pair.
                [$($A::Type::try_match_verbose($x),)*]
            }
        }

        /// A central dispatcher for multi-method overloading.
        pub struct $dispatcher<R, $($T,)*>
        where
            R: 'static,
            $($T: Map,)*
        {
            pub(super) methods: Vec<(String, Box<dyn $trait<R, $($T,)*>>)>,
        }

        impl<R, $($T,)*> Default for $dispatcher<R, $($T,)*>
        where
            R: 'static,
            $($T: Map,)*
        {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<R, $($T,)*> $dispatcher<R, $($T,)*>
        where
            R: 'static,
            $($T: Map,)*
        {
            /// Construct a new, empty dispatcher.
            pub fn new() -> Self {
                Self { methods: Vec::new() }
            }

            /// Register the new named method with the dispatcher.
            pub fn register<F, $($A,)*>(&mut self, name: impl Into<String>, f: F)
            where
                $($A: Map,)*
                $(for<'a> $A::Type<'a>: DispatchRule<$T::Type<'a>>,)*
                F: for<$($lf,)*> Fn($($A::Type<$lf>,)*) -> R + 'static,
            {
                let method = $method::<R, $($A,)*>::new(f);
                self.methods.push((name.into(), Box::new(method)))
            }

            /// Try to invoke the best fitting method with the given arguments.
            ///
            /// If no such method can be found, returns `None`.
            pub fn call(&self, $($x: $T::Type<'_>,)*) -> Option<R> {
                let mut method: Option<(&_, [MatchScore; $N])> = None;
                self.methods.iter().for_each(|m| {
                    match coalesce(&(m.1.try_match($(&$x,)*))) {
                        // Valid match
                        Some(score) => match method.as_mut() {
                            Some(method) => {
                                if score < method.1 {
                                    *method = (m, score)
                                }
                            }
                            None => {
                                method.replace((m, score));
                            }
                        },
                        None => {}
                    }
                });

                // Invoke the best method
                method.map(|(m, _)| m.1.call($($x,)*))
            }

            /// Return an iterator to the methods registered in this dispatcher.
            pub fn methods(
                &self
            ) -> impl ExactSizeIterator<Item = &(String, Box<dyn $trait<R, $($T,)*>>)> {
                self.methods.iter()
            }

            /// Query whether the combination of values has a valid matching method without
            /// trying to invoke that method.
            pub fn has_match(&self, $($x: &$T::Type<'_>,)*) -> bool {
                for m in self.methods.iter() {
                    if all_match(&m.1.try_match($(&$x,)*)) {
                        return true;
                    }
                }
                return false;
            }

            /// Check if a back-end method exists for the arguments.
            ///
            /// If so, returns `Ok(())`.
            ///
            /// Otherwise, returns a vector of `ArgumentMismatch` for the up-to
            /// `max_methods` closest methods.
            ///
            /// In this context, "closeness" is defined by first comparing match or failure
            /// scores for argument 0, followed by argument 1 if equal and so on.
            pub fn debug<'a, $($lf,)*>(
                &'a self,
                max_methods: usize,
                $($x: &'a $T::Type<$lf>,)*
            ) -> Result<(), Vec<ArgumentMismatch<'a, $N>>>
            where
                $($lf: 'a,)*
            {
                let mut methods = Queue::new(max_methods);
                for m in self.methods.iter() {
                    let t = TaggedMatch {
                        method: &m.0,
                        score: m.1.try_match_verbose($($x,)*),
                    };
                    match methods.push(t) {
                        Ok(()) => {},
                        Err(()) => return Ok(()),
                    }
                }
                Err(methods.finish())
            }
        }
    }
}

implement_dispatch!(Dispatch1, Method1, Dispatcher1, 1, { T0 }, { x0 }, { A0 }, { 'a0 });
implement_dispatch!(
    Dispatch2, Method2, Dispatcher2, 2,
    { T0 T1 }, { x0 x1 }, { A0 A1 }, { 'a0 'a1 }
);
implement_dispatch!(
    Dispatch3, Method3, Dispatcher3, 3,
    { T0 T1 T2 }, { x0 x1 x2 }, { A0 A1 A2 }, { 'a0 'a1 'a2 }
);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    struct Num<const N: usize>;

    impl<const N: usize> Map for Num<N> {
        type Type<'a> = Self;
    }

    impl<const N: usize> DispatchRule<usize> for Num<N> {
        type Error = std::convert::Infallible;

        // For testing purposes, we accept values within 2 of `N`, but with decreasing
        // precedence.
        fn try_match(from: &usize) -> Result<MatchScore, FailureScore> {
            let diff = from.abs_diff(N);
            if diff <= 2 {
                Ok(MatchScore(diff as u32))
            } else {
                Err(FailureScore(diff as u32))
            }
        }

        fn convert(from: usize) -> Result<Self, Self::Error> {
            assert!(from.abs_diff(N) <= 2);
            Ok(Self)
        }

        fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&usize>) -> std::fmt::Result {
            match from {
                None => write!(f, "{}", N),
                Some(value) => {
                    let diff = value.abs_diff(N);
                    match diff {
                        0 => write!(f, "success: exact match"),
                        1 => write!(f, "success: off by 1"),
                        2 => write!(f, "success: off by 2"),
                        x => write!(f, "error: off by {}", x),
                    }
                }
            }
        }
    }

    ////////////////
    // Dispatch 1 //
    ////////////////

    #[test]
    fn test_dispatch_1() {
        let mut x = Dispatcher1::<usize, usize>::default();
        x.register::<_, Num<0>>("method 0", |_| 0);
        x.register::<_, Num<3>>("method 3", |_| 3);
        x.register::<_, Num<5>>("method 5", |_| 5);
        x.register::<_, Num<8>>("method 8", |_| 8);

        {
            let methods: Vec<_> = x.methods().collect();
            assert_eq!(methods.len(), 4);
            assert_eq!(methods[0].0, "method 0");
            assert_eq!(methods[0].1.signatures()[0].to_string(), "0");

            assert_eq!(methods[1].0, "method 3");
            assert_eq!(methods[1].1.signatures()[0].to_string(), "3");
        }

        // Test that dispatching works properly.
        assert_eq!(x.call(0), Some(0));
        assert_eq!(x.call(1), Some(0));
        assert_eq!(x.call(2), Some(3));
        assert_eq!(x.call(3), Some(3));
        assert_eq!(x.call(4), Some(3));
        assert_eq!(x.call(5), Some(5));
        assert_eq!(x.call(6), Some(5));
        assert_eq!(x.call(7), Some(8));
        assert_eq!(x.call(8), Some(8));
        assert_eq!(x.call(11), None);

        for i in 0..11 {
            assert!(x.has_match(&i));
        }
        for i in 11..20 {
            assert!(!x.has_match(&i));
        }

        // Make sure `Debug` works.
        assert!(x.debug(3, &10).is_ok());

        let mismatches = x.debug(3, &11).unwrap_err();
        assert_eq!(mismatches.len(), 3);

        // Method 8 is the closest.
        assert_eq!(mismatches[0].method(), "method 8");
        assert_eq!(
            mismatches[0].mismatches()[0].as_ref().unwrap().to_string(),
            "error: off by 3"
        );

        // Method 5 is next.
        assert_eq!(mismatches[1].method(), "method 5");
        assert_eq!(
            mismatches[1].mismatches()[0].as_ref().unwrap().to_string(),
            "error: off by 6"
        );

        // Method 3 is next.
        assert_eq!(mismatches[2].method(), "method 3");
        assert_eq!(
            mismatches[2].mismatches()[0].as_ref().unwrap().to_string(),
            "error: off by 8"
        );

        // Make sure that if we request more than the total number of methods that it is
        // capped.
        assert_eq!(x.debug(10, &20).unwrap_err().len(), 4);
    }

    ////////////////
    // Dispatch 2 //
    ////////////////

    #[test]
    fn test_dispatch_2() {
        let mut x = Dispatcher2::<usize, usize, usize>::default();

        x.register::<_, Num<10>, Num<10>>("method 0", |_, _| 0);
        x.register::<_, Num<10>, Num<13>>("method 1", |_, _| 1);
        x.register::<_, Num<13>, Num<12>>("method 3", |_, _| 3);
        x.register::<_, Num<12>, Num<10>>("method 2", |_, _| 2);

        {
            let methods: Vec<_> = x.methods().collect();
            assert_eq!(methods.len(), 4);
            assert_eq!(methods[0].0, "method 0");
            assert_eq!(methods[0].1.signatures()[0].to_string(), "10");
            assert_eq!(methods[0].1.signatures()[1].to_string(), "10");

            assert_eq!(methods[1].0, "method 1");
            assert_eq!(methods[1].1.signatures()[0].to_string(), "10");
            assert_eq!(methods[1].1.signatures()[1].to_string(), "13");
        }

        // This is where things get weird.
        assert_eq!(x.call(10, 10), Some(0)); // Match method 0
        assert_eq!(x.call(10, 11), Some(0)); // Match method 0
        assert_eq!(x.call(10, 12), Some(1)); // Match method 1
        assert_eq!(x.call(11, 12), Some(1)); // Match method 1
        assert_eq!(x.call(12, 12), Some(2)); // Match method 2
        assert_eq!(x.call(13, 12), Some(3)); // Match method 3

        // Check error handling.
        {
            assert!(x.call(10, 7).is_none());
            let m = x.debug(3, &9, &7).unwrap_err();
            // The closest hit is method 0, followed by method 1.
            assert_eq!(m[0].method(), "method 0");
            assert_eq!(m[1].method(), "method 1");
            assert_eq!(m[2].method(), "method 2");

            let mismatches = m[0].mismatches();
            // The first argument is a match - the second argument is a mismatch.
            assert!(mismatches[0].is_none());
            assert_eq!(
                mismatches[1].as_ref().unwrap().to_string(),
                "error: off by 3"
            );

            let mismatches = m[2].mismatches();
            assert_eq!(
                mismatches[0].as_ref().unwrap().to_string(),
                "error: off by 3"
            );
            assert_eq!(
                mismatches[1].as_ref().unwrap().to_string(),
                "error: off by 3"
            );
        }

        // Try again, but this time from the other direction.
        {
            let m = x.debug(4, &16, &12).unwrap_err();
            assert_eq!(m[0].method(), "method 3");
            assert_eq!(m[1].method(), "method 2");
            assert_eq!(m[2].method(), "method 1");
        }
    }
}
