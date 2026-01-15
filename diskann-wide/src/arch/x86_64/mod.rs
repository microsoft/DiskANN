/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::atomic::{AtomicU64, Ordering};

use super::{Scalar, Target, Target1, Target2, Target3};

// Internal helpers
mod algorithms;
mod common;
mod macros;

pub mod v3;
pub mod v4;

pub use v3::V3;
pub use v4::V4;

////////////////////////////
// Architecture Selection //
////////////////////////////

cfg_if::cfg_if! {
    if #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512cd",
        target_feature = "avx512dq",
        target_feature = "avx512vl",
        target_feature = "avx512vnni",
        // target_feature = "avx512bitalg",
        // target_feature = "avx512vpopcntdq",
    ))] {
        pub type Current = V4;

        pub const fn current() -> Current {
            // SAFETY: This function is gated by a CFG guard for the features needed by V4.
            unsafe { V4::new() }
        }
    } else if #[cfg(all(
        target_feature = "avx2",
        target_feature = "bmi1",
        target_feature = "bmi2",
        target_feature = "f16c",
        target_feature = "fma",
        target_feature = "lzcnt",
        target_feature = "movbe",
        target_feature = "xsave",
        not(doc),
    ))] {
        pub type Current = V3;

        pub const fn current() -> Current {
            // SAFETY: This function is gated by a CFG guard for the features needed by V3.
            unsafe { V3::new() }
        }
    } else {
        /// The type of the [`crate::Architecture`] most compatible with the program's
        /// compilation target.
        pub type Current = Scalar;

        /// Return the [`crate::Architecture`] most compatible with the program's compilation
        /// target.
        pub const fn current() -> Current {
            Scalar::new()
        }
    }
}

// We cache a single enum and use it to indicate the version with the following meaning:
//
// 0: Uninitialized
// 1: V3
// 2 and above: Scalar
static ARCH_NUMBER: AtomicU64 = AtomicU64::new(ARCH_UNINITIALIZED);

// NOTE: Architecture must be properly nested in ascending order so compatibility checks
// can be done with a `>=` comparison.
const ARCH_UNINITIALIZED: u64 = 0;
const ARCH_SCALAR: u64 = 1;
const ARCH_V3: u64 = 2;
const ARCH_V4: u64 = 3;

macro_rules! get_or_set_architecture {
    () => {{
        use std::sync::atomic::Ordering;

        let mut version = $crate::arch::x86_64::ARCH_NUMBER.load(Ordering::Relaxed);
        if version == $crate::arch::x86_64::ARCH_UNINITIALIZED {
            version = $crate::arch::x86_64::resolve_architecture();
        }
        version
    }};
}

pub(super) use get_or_set_architecture;

fn arch_number() -> u64 {
    if is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("lzcnt")
        && is_x86_feature_detected!("movbe")
        && is_x86_feature_detected!("xsave")
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512cd")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni")
        // && is_x86_feature_detected!("avx512bitalg")
        // && is_x86_feature_detected!("avx512vpopcntdq")
        {
            ARCH_V4
        } else {
            ARCH_V3
        }
    } else {
        ARCH_SCALAR
    }
}

#[inline(never)]
fn resolve_architecture() -> u64 {
    let arch = arch_number();
    ARCH_NUMBER.store(arch, Ordering::Relaxed);
    arch
}

macro_rules! impl_dispatch {
    (
        $name:ident,
        $resolve:ident,
        $name_no_features:ident,
        $resolve_no_features:ident,
        $target:ident,
        $method:ident,
        { $($x:ident )* },
        { $($A:ident )* }
    ) => {
        /// Dispatch the target-compatible functor to the most advanced architecture
        /// supported by the current machine. This function will instantiate the
        /// `Target::run` method by applying the target features associated with the
        /// dynamically selected architecture.
        ///
        /// If you want to efficiently run `f` on the highest compatible architecture
        /// **without** applying target features, see
        #[doc = concat!(stringify!($name_no_features), ".")]
        #[inline]
        pub fn $name<T, R, $($A,)*>(f: T, $($x: $A,)*) -> R
        where
            T: $target<V4, R, $($A,)*>
               + $target<V3, R, $($A,)*>
               + $target<Scalar, R, $($A,)*>,
        {
            let version = ARCH_NUMBER.load(Ordering::Relaxed);
            if version == ARCH_UNINITIALIZED {
                $resolve(f, $($x,)*)
            } else if version == ARCH_V4 {
                // SAFETY: Architecture resolution has determined that the current machine
                // is V4 compatible.
                let arch = unsafe { V4::new() };

                // SAFETY: Since we are V4 compatible, it is safe to call the `run_with*`
                // methods on the architecture because we know the required features exist.
                unsafe { arch.$method(f, $($x,)*) }
            } else if version == ARCH_V3 {
                // SAFETY: Architecture resolution has determined that the current machine
                // is V3 compatible.
                let arch = unsafe { V3::new() };

                // SAFETY: Since we are V3 compatible, it is safe to call the `run_with*`
                // methods on the architecture because we know the required features exist.
                unsafe { arch.$method(f, $($x,)*) }
            } else {
                f.run(Scalar::new(), $($x,)*)
            }
        }

        /// Dispatch the target-compatible functor to the most advanced architecture
        /// supported by the current machine.
        ///
        /// This function *will not* apply the associated target features for the
        /// architecture when invoking [`Target::run`].
        #[inline]
        pub fn $name_no_features<T, R, $($A,)*>(f: T, $($x: $A,)*) -> R
        where
            T: $target<V4, R, $($A,)*>
               + $target<V3, R, $($A,)*>
               + $target<Scalar, R, $($A,)*>,
        {
            let version = ARCH_NUMBER.load(Ordering::Relaxed);
            if version == ARCH_UNINITIALIZED {
                $resolve_no_features(f, $($x,)*)
            } else if version == ARCH_V4 {
                // SAFETY: Architecture resolution has determined that the current machine
                // is V4 compatible.
                let arch = unsafe { V4::new() };
                f.run(arch, $($x,)*)
            } else if version == ARCH_V3 {
                // SAFETY: Architecture resolution has determined that the current machine
                // is V3 compatible.
                let arch = unsafe { V3::new() };
                f.run(arch, $($x,)*)
            } else {
                f.run(Scalar::new(), $($x,)*)
            }
        }

        // What's happening here is subtle: basically, we are trying to ensure that the
        // machinery required to handle the version selection has as minimal runtime impact
        // on the happy path.
        //
        // If the call to `resolve_architecture` lives inside the main dispatching method,
        // LLVM does not reliably avoid pushing arguments onto the stack just to pop them
        // immediately, just in the off-chance that we need to perform architecture
        // resolution.
        //
        // Using an explicitly outlined copy of the main dispatch function and using
        // recursion has the following impact on the dispatch call site:
        //
        // * The ABI of all called methods is the same, so LLVM completely avoids messing
        //   with the stack.
        //
        // * Modeling this as a function call means that our argument types do not need to
        //   be `Copy` to call a variadic resolution method.
        //
        // * Any register saving needed to set up for architecture resolution is delegated
        //   entirely to the outlined slow path.
        #[inline(never)]
        fn $resolve<T, R, $($A,)*>(f: T, $($x: $A,)*) -> R
        where
            T: $target<V4, R, $($A,)*>
               + $target<V3, R, $($A,)*>
               + $target<Scalar, R, $($A,)*>,
        {
            resolve_architecture();

            // We're kind of in a rough situation when deciding whether or not to block
            // inlining of this method.
            //
            // Since the parent resolution function is marked as `inline`, blocking inlining
            // of the recursive call means that the entry point basically needs to be
            // compiled twice.
            //
            // If it is inlined ... it's *still* compiled twice, but at least we end up
            // fewer overall functions.
            $name(f, $($x,)*)
        }

        /// See the documentation for `resolve`.
        #[inline(never)]
        fn $resolve_no_features<T, R, $($A,)*>(f: T, $($x: $A,)*) -> R
        where
            T: $target<V4, R, $($A,)*>
               + $target<V3, R, $($A,)*>
               + $target<Scalar, R, $($A,)*>,
        {
            resolve_architecture();
            $name_no_features(f, $($x,)*)
        }
    }
}

impl_dispatch!(
    dispatch,
    dispatch_resolve,
    dispatch_no_features,
    dispatch_resolve_no_features,
    Target,
    run_with,
    {},
    {}
);
impl_dispatch!(
    dispatch1,
    dispatch_resolve1,
    dispatch1_no_features,
    dispatch_resolve1_no_features,
    Target1,
    run_with_1,
    { x0 },
    { T0 }
);
impl_dispatch!(
    dispatch2,
    dispatch_resolve2,
    dispatch2_no_features,
    dispatch_resolve2_no_features,
    Target2,
    run_with_2,
    { x0 x1 },
    { T0 T1 }
);
impl_dispatch!(
    dispatch3,
    dispatch_resolve3,
    dispatch3_no_features,
    dispatch_resolve3_no_features,
    Target3,
    run_with_3,
    { x0 x1 x2 },
    { T0 T1 T2 }
);

#[cfg(test)]
static TEST_ARCH_NUMBER: AtomicU64 = AtomicU64::new(ARCH_UNINITIALIZED);

/// Return the architecture number to use for crate-level testing purposes.
///
/// If the environment variable `WIDE_TEST_MIN_ARCH` is not set, this defaults to
/// [`arch_number()`]. Otherwise, it will use the configured architecture with the following
/// mapping for the variable values:
///
/// * `all`: Run the highest architecture
/// * `x86-64-v4`: V4
/// * `x86-64-v3`: V3
/// * `scalar`: Scalar
#[cfg(test)]
#[inline(never)]
pub(super) fn test_arch_number() -> u64 {
    // Get and cache the test number of needed.
    let mut requested = TEST_ARCH_NUMBER.load(Ordering::Relaxed);
    if requested == ARCH_UNINITIALIZED {
        requested = match crate::get_test_arch() {
            Some(arch) => {
                if arch == "all" || arch == "x86-64-v4" {
                    ARCH_V4
                } else if arch == "x86-64-v3" {
                    ARCH_V3
                } else if arch == "scalar" {
                    ARCH_SCALAR
                } else {
                    panic!("Unrecognized test architecture: \"{arch}\"");
                }
            }
            None => arch_number(),
        };
        TEST_ARCH_NUMBER.store(requested, Ordering::Relaxed);
    };
    requested
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Architecture;

    struct TestOp;

    impl Target<Scalar, &'static str> for TestOp {
        fn run(self, _: Scalar) -> &'static str {
            "scalar"
        }
    }

    impl Target1<Scalar, String, &str> for TestOp {
        fn run(self, _: Scalar, x0: &str) -> String {
            format!("scalar: {}", x0)
        }
    }

    impl Target2<Scalar, String, &str, &str> for TestOp {
        fn run(self, _: Scalar, x0: &str, x1: &str) -> String {
            format!("scalar: {}, {}", x0, x1)
        }
    }

    impl Target3<Scalar, String, &str, &str, &str> for TestOp {
        fn run(self, _: Scalar, x0: &str, x1: &str, x2: &str) -> String {
            format!("scalar: {}, {}, {}", x0, x1, x2)
        }
    }

    impl Target<V3, &'static str> for TestOp {
        fn run(self, _: V3) -> &'static str {
            "v3"
        }
    }

    impl Target1<V3, String, &str> for TestOp {
        fn run(self, _: V3, x0: &str) -> String {
            format!("v3: {}", x0)
        }
    }

    impl Target2<V3, String, &str, &str> for TestOp {
        fn run(self, _: V3, x0: &str, x1: &str) -> String {
            format!("v3: {}, {}", x0, x1)
        }
    }

    impl Target3<V3, String, &str, &str, &str> for TestOp {
        // Set `inline(never)` to avoid getting unsuitable target features.
        #[inline(never)]
        fn run(self, _: V3, x0: &str, x1: &str, x2: &str) -> String {
            format!("v3: {}, {}, {}", x0, x1, x2)
        }
    }

    impl Target<V4, &'static str> for TestOp {
        fn run(self, _: V4) -> &'static str {
            "v4"
        }
    }

    impl Target1<V4, String, &str> for TestOp {
        fn run(self, _: V4, x0: &str) -> String {
            format!("v4: {}", x0)
        }
    }

    impl Target2<V4, String, &str, &str> for TestOp {
        fn run(self, _: V4, x0: &str, x1: &str) -> String {
            format!("v4: {}, {}", x0, x1)
        }
    }

    impl Target3<V4, String, &str, &str, &str> for TestOp {
        // Set `inline(never)` to avoid getting unsuitable target features.
        #[inline(never)]
        fn run(self, _: V4, x0: &str, x1: &str, x2: &str) -> String {
            format!("v4: {}, {}, {}", x0, x1, x2)
        }
    }

    // These tests reach directly into the dispatch mechanism.
    //
    // There should only be a single test (this one) that does this, and all other tests
    // involving dispatch should either be configured to work properly regarless of the
    // backend architecture, or be run in their own process.
    #[test]
    fn test_dispatch() {
        // Test that `new_checked` works properly.
        // Test that `new_checked` works properly.
        ARCH_NUMBER.store(ARCH_V4, Ordering::Relaxed);
        assert!(V4::new_checked().is_some());
        assert!(V3::new_checked().is_some());

        ARCH_NUMBER.store(ARCH_V3, Ordering::Relaxed);
        assert!(V4::new_checked().is_none());
        assert!(V3::new_checked().is_some());

        ARCH_NUMBER.store(ARCH_SCALAR, Ordering::Relaxed);
        assert!(V4::new_checked().is_none());
        assert!(V3::new_checked().is_none());

        // Now that we have the scalar version, ensure that `Scalar` is the dispatch target.
        assert_eq!(dispatch(TestOp), "scalar");
        assert_eq!(dispatch1(TestOp, "foo"), "scalar: foo");
        assert_eq!(dispatch2(TestOp, "foo", "bar"), "scalar: foo, bar");
        assert_eq!(
            dispatch3(TestOp, "foo", "bar", "baz"),
            "scalar: foo, bar, baz",
        );

        assert_eq!(dispatch_no_features(TestOp), "scalar");
        assert_eq!(dispatch1_no_features(TestOp, "foo"), "scalar: foo");
        assert_eq!(
            dispatch2_no_features(TestOp, "foo", "bar"),
            "scalar: foo, bar"
        );
        assert_eq!(
            dispatch3_no_features(TestOp, "foo", "bar", "baz"),
            "scalar: foo, bar, baz",
        );

        // V3
        ARCH_NUMBER.store(ARCH_V3, Ordering::Relaxed);
        assert_eq!(dispatch(TestOp), "v3");
        assert_eq!(dispatch1(TestOp, "foo"), "v3: foo");
        assert_eq!(dispatch2(TestOp, "foo", "bar"), "v3: foo, bar");
        assert_eq!(dispatch3(TestOp, "foo", "bar", "baz"), "v3: foo, bar, baz",);

        assert_eq!(dispatch_no_features(TestOp), "v3");
        assert_eq!(dispatch1_no_features(TestOp, "foo"), "v3: foo");
        assert_eq!(dispatch2_no_features(TestOp, "foo", "bar"), "v3: foo, bar");
        assert_eq!(
            dispatch3_no_features(TestOp, "foo", "bar", "baz"),
            "v3: foo, bar, baz",
        );

        // V4
        ARCH_NUMBER.store(ARCH_V4, Ordering::Relaxed);
        assert_eq!(dispatch(TestOp), "v4");
        assert_eq!(dispatch1(TestOp, "foo"), "v4: foo");
        assert_eq!(dispatch2(TestOp, "foo", "bar"), "v4: foo, bar");
        assert_eq!(dispatch3(TestOp, "foo", "bar", "baz"), "v4: foo, bar, baz",);

        assert_eq!(dispatch_no_features(TestOp), "v4");
        assert_eq!(dispatch1_no_features(TestOp, "foo"), "v4: foo");
        assert_eq!(dispatch2_no_features(TestOp, "foo", "bar"), "v4: foo, bar");
        assert_eq!(
            dispatch3_no_features(TestOp, "foo", "bar", "baz"),
            "v4: foo, bar, baz",
        );

        // Recovery
        ARCH_NUMBER.store(ARCH_UNINITIALIZED, Ordering::Relaxed);
        // Dispatching should recover from an uninitialized arch number.
        let _ = dispatch(TestOp);

        ARCH_NUMBER.store(ARCH_UNINITIALIZED, Ordering::Relaxed);
        // Dispatching should recover from an uninitialized arch number.
        let _ = dispatch_no_features(TestOp);
    }

    #[test]
    fn test_run() {
        if let Some(arch) = V3::new_checked_uncached() {
            let mut x = 10;
            let y: &str = arch.run(|| {
                x += 10;
                "foo"
            });
            assert_eq!(x, 20);
            assert_eq!(y, "foo");
        }
    }
}
