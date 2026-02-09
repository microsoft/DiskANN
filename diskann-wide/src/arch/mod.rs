/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Traits and functions supporting multi-architecture applications.
//!
//! Many SIMD instructions are micro-architecture specific, meaning that only a subset of
//! CPUs found in the wild can support SIMD accelerated algorithms. This module provides
//! tools for writing SIMD algorithms supporting multiple architectures and provides a
//! light-weight runtime dispatching service to select the most appropriate implementation
//! at run time.
//!
//! The example code below demonstrates a multi-versioned `X = X + Y` kernel:
//! ```rust
//! use diskann_wide::arch::{Target2, dispatch2};
//!
//! // A zero-sized type that we can use to implement a trait.
//! struct Add;
//!
//! impl<A: diskann_wide::Architecture> Target2<A, (), &mut [f32], &[f32]> for Add {
//!     #[inline]
//!     fn run(self, _: A, dst: &mut [f32], src: &[f32]) {
//!         std::iter::zip(dst.iter_mut(), src.iter()).for_each(|(d, s)| *d += *s);
//!     }
//! }
//!
//! fn add(dst: &mut [f32], src: &[f32]) {
//!     dispatch2(Add, dst, src)
//! }
//!
//! let mut dst = vec![1.0, 2.0, 3.0];
//! add(&mut dst, &[2.0, 3.0, 4.0]);
//! assert_eq!(dst, &[3.0, 5.0, 7.0]);
//! ```
//!
//! Lets break down what's happening.
//!
//! The function [`dispatch2`] (suffixed with "2" because it takes two arguments, more on
//! this later) takes the struct `Add`, which is required to implement [`Target2`] for all
//! supported micro-architecture levels supported by `wide` for compilation target CPU.
//!
//! It then will determine at run time the features supported by the current CPU and invoke
//! `Add::run` with the best architecture. The above example does not use any explicit SIMD
//! and is generic with respect to the [`Architecture`]. This still allows the compiler to
//! perform auto-vectorization for different platforms, which can still result in a speed-up.
//!
//! ## Mechanics of Dispatching
//!
//! Run time architecture detection happens only once (modulo race conditions) and once
//! resolved involves an atomic load and a branch.
//!
//! ## Variadic traits and ABI
//!
//! The traits like [`Target1`] or [`dispatch2`] are suffixed by the number of additional
//! arguments that they take. This is important for cases where we want the calling
//! convention of the dispatched-to function to match the calling-convention of the
//! dispatcher function. In this case, Rust will invoke the dispatched-to function using a
//! jump instead of a function call.
//!
//! For example, if we had implemented the add method above without these variadics:
//! ```rust
//! use diskann_wide::arch::{Target, dispatch};
//!
//! struct AddV2<'a>(&'a mut [f32], &'a [f32]);
//!
//! impl<A: diskann_wide::Architecture> Target<A, ()> for AddV2<'_> {
//!     #[inline]
//!     fn run(self, _: A) {
//!         std::iter::zip(self.0.iter_mut(), self.1.iter()).for_each(|(d, s)| *d += *s);
//!     }
//! }
//!
//! #[inline(never)]
//! fn add_v2(dst: &mut [f32], src: &[f32]) {
//!     dispatch(AddV2(dst, src))
//! }
//!
//! let mut dst = vec![1.0, 2.0, 3.0];
//! add_v2(&mut dst, &[2.0, 3.0, 4.0]);
//! assert_eq!(dst, &[3.0, 5.0, 7.0]);
//! ```
//! then the function still works, but the assembly code goes from something that looks like
//! this
//! ```asm
//!        mov rax, qword ptr [rip + diskann_wide::arch::x86_64::ARCH_NUMBER@GOTPCREL]
//!        mov rax, qword ptr [rax]
//!        cmp rax, 1
//!        je diskann_wide::arch::x86_64::V3::run_with_2
//!        test rax, rax
//!        je diskann_wide::arch::x86_64::dispatch_resolve2
//!        <scalar-code>
//! ```
//! where there are no stack writes and the `V3` compatible code is reached via `jmp` to
//! something that looks like
//! ```asm
//!        sub rsp, 56
//!        mov rax, qword ptr fs:[40]
//!        mov qword ptr [rsp + 48], rax
//!        mov rax, qword ptr [rip + diskann_wide::arch::x86_64::ARCH_NUMBER@GOTPCREL]
//!        mov rax, qword ptr [rax]
//!        test rax, rax
//!        je .LBB14_3
//!        cmp rax, 1
//!        jne .LBB14_2
//!        mov qword ptr [rsp + 8], rdi
//!        mov qword ptr [rsp + 16], 3
//!        lea rax, [rip + .L__unnamed_8]
//!        mov qword ptr [rsp + 24], rax
//!        mov qword ptr [rsp + 32], 3
//!        lea rax, [rsp + 7]
//!        mov qword ptr [rsp + 40], rax
//!        lea rdi, [rsp + 8]
//!        call diskann_wide::arch::x86_64::V3::run_with
//!        jmp .LBB14_5
//!.LBB14_3:
//!        mov qword ptr [rsp + 8], rdi
//!        mov qword ptr [rsp + 16], 3
//!        lea rax, [rip + .L__unnamed_8]
//!        mov qword ptr [rsp + 24], rax
//!        mov qword ptr [rsp + 32], 3
//!        lea rdi, [rsp + 8]
//!        call diskann_wide::arch::x86_64::dispatch_resolve::<loop_example::AddV2, ()>
//!        jmp .LBB14_5
//!        <scalar-code>
//! ```
//! Notice some unconditional stack writes, stack preparation for calling the `V3` compatible
//! code, and a `call` to run the code.
//!
//! What's happening is that Rust will not inline functions annotated with
//! `target_feature(enable = "feature")]` into an incompatible context. Since `AddV3` exceeds
//! 16 bytes, it must be passed on the stack (on Linux at least). Therefore, we have extra
//! overhead of stack preparation to call the dispatch target.
//!
//! ## Function Pointer API
//!
//! The previous section discussed performing a dynamic dispatch at a single call site, but
//! you need to pay the (admittedly small) overhead every time this function is called.
//! Another approach would be to perform dispatch a single time by obtaining a function pointer
//! to the dispatched function and then calling through that function pointer.
//!
//! This is a little tricky for two reasons (and a whole host of less obvious reasons).
//!
//! Reason 1: Functions with additional `target_features` cannot be inlined. This means the
//! simple approach of
//! ```rust
//! use diskann_wide::{Architecture, arch::{Scalar, Target, Target2, dispatch2}};
//!
//! // A zero-sized type that we can use to implement a trait.
//! struct Add;
//!
//! impl<A: Architecture> Target2<A, (), &mut [f32], &[f32]> for Add {
//!     #[inline]
//!     fn run(self, _: A, dst: &mut [f32], src: &[f32]) {
//!         std::iter::zip(dst.iter_mut(), src.iter()).for_each(|(d, s)| *d += *s);
//!     }
//! }
//!
//! impl<A: Architecture> Target<A, fn(A, &mut [f32], &[f32])> for Add {
//!    fn run(self, arch: A) -> fn(A, &mut [f32], &[f32]) {
//!        // Create a non-capturing closure that invokes `arch.run2`.
//!        //
//!        // The invocation of `arch.run2` will apply the necessary target features and
//!        // the non-capturing closure can be coerced into a function pointer.
//!        let f = |arch: A, dst: &mut [f32], src: &[f32]| arch.run2(Add, dst, src);
//!        f
//!    }
//! }
//!
//! let f: fn(Scalar, &mut [f32], &[f32]) = (Scalar).run(Add);
//!
//! let mut dst = vec![1.0, 2.0, 3.0];
//! f(Scalar, &mut dst, &[2.0, 3.0, 4.0]);
//! assert_eq!(dst, &[3.0, 5.0, 7.0]);
//! ```
//! would likely generate code that looks something like:
//! ```asm
//! .section .text.<<diskann_wide::Add as diskann_wide::arch::Target<_, _>::run::{closure#0} /* snip */>>
//!        .p2align        4
//!.type   <<diskann_wide::Add as _>::call_once,@function
//!<<diskann_wide::Add as _>>::call_once:
//!        .cfi_startproc
//!        jmp <diskann_wide::arch::Scalar>::run_with_2::<diskann_wide::Add, &mut [f32], &[f32], ()>
//! ```
//! The body is simply an unconditional jump to the actual implementation precisely because
//! the actual implementation cannot be inlined into the body of the closure we coerced into
//! a function pointer. Unfortunately, the same applies to most other ways one would try to
//! create a function pointer to the dispatched-to function.
//!
//! The consequence of this is that we need to take an **unsafe** function pointer so we
//! can dispatch call directly to the implementation.
//!
//! Reason 2: Even if the above approach worked, the [`Architecture`] is sill present in the
//! signature of the `fn`, meaning we haven't really hidden the micro-architecture
//! information.
//!
//! With that in mind, the current solution looks like the following.
//! ```rust
//! use diskann_wide::{
//!     Architecture,
//!     arch::{self, Dispatched2},
//!     lifetime::{Ref, Mut},
//! };
//!
//! struct Add;
//!
//! // Note the use of `FTarget` instead of `Target`. That is because the implementation is
//! // simply an associated function instead of a method.
//! impl<A: Architecture> arch::FTarget2<A, (), &mut [f32], &[f32]> for Add {
//!     #[inline]
//!     fn run(_: A, dst: &mut [f32], src: &[f32]) {
//!         std::iter::zip(dst.iter_mut(), src.iter()).for_each(|(d, s)| *d += *s);
//!     }
//! }
//!
//! // The `Dispatched2` struct is a slightly magical wrapper around a function pointer,
//! // returning the unit type `()` and taking two arguments; one `&mut [f32]` and
//! // the other `&[f32]`.
//! //
//! // The need for `Mut` and `Ref` is described below.
//! type FnPtr = Dispatched2<(), Mut<[f32]>, Ref<[f32]>>;
//!
//! impl<A: Architecture> arch::Target<A, FnPtr> for Add {
//!    fn run(self, arch: A) -> FnPtr {
//!        arch.dispatch2::<Self, (), Mut<[f32]>, Ref<[f32]>>()
//!    }
//! }
//!
//! let f: FnPtr = diskann_wide::arch::dispatch(Add);
//! let mut dst = vec![1.0, 2.0, 3.0];
//!
//! // Invoke the function pointer
//! f.call(&mut dst, &[2.0, 3.0, 4.0]);
//! assert_eq!(dst, &[3.0, 5.0, 7.0]);
//! ```
//! The resulting function pointer (though "safely unsafe"), successfully hides that
//! dispatched micro-architecture and when called will always go directly to the
//! implementation rather than needing the trampoline.
//!
//! The [`Architecture`] methods
//! * [`Architecture::dispatch1`], [`Architecture::dispatch2`], [`Architecture::dispatch3`]
//!
//! Will produce function pointers of the annotated arities
//!
//! * [`Dispatched1`], [`Dispatched2`], and [`Dispatched3`]
//!
//! and are accompanied by the generator traits
//!
//! * [`FTarget1`], [`FTarget2`], and [`FTarget3`]
//!
//! ### Lifetime Annotations and Limitations
//!
//! One thing to note in the above example is the use of [`crate::lifetime::Mut`] and
//! [`crate::lifetime::Ref`] in the invocation of [`Architecture::dispatch2`]. This is an
//! unfortunate limitation of the Rust compiler at the moment when it comes to inferring
//! lifetimes of function pointers.
//!
//! For example, the following does not compile
//! ```compile_fail
//! pub struct Example;
//!
//! trait Run<T, U> {
//!     fn run(x: T, y: U);
//! }
//!
//! impl Example {
//!     fn run<F, T, U>(self, x: T, y: U)
//!     where
//!         F: Run<T, U>,
//!     {
//!         F::run(x, y)
//!     }
//! }
//!
//! struct Add;
//!
//! impl Run<&mut [f32], &[f32]> for Add {
//!     #[inline]
//!     fn run(dst: &mut [f32], src: &[f32]) {
//!         std::iter::zip(dst.iter_mut(), src.iter()).for_each(|(d, s)| *d += *s);
//!     }
//! }
//!
//! // This fails to compile! :(
//! pub fn make() -> fn(Example, &mut [f32], &[f32]) {
//!     let f = Example::run::<Add, &mut [f32], &[f32]>;
//!     f
//! }
//! ```
//! Fails to compile with the following error message
//! ```text
//! error[E0308]: mismatched types
//!   --> <source>:27:5
//!    |
//! 25 | pub fn make() -> fn(Example, &mut [f32], &[f32]) {
//!    |                  ------------------------------- expected `for<'a, 'b> fn(Example, &'a mut [f32], &'b [f32])` because of return type
//! 26 |     let f = Example::run::<Add, &mut [f32], &[f32]>;
//! 27 |     f
//!    |     ^ one type is more general than the other
//!    |
//!    = note: expected fn pointer `for<'a, 'b> fn(Example, &'a mut _, &'b _)`
//!                  found fn item `fn(Example, &mut _, &_) {Example::run::<Add, &mut [f32], &[f32]>}`
//! ```
//! The [`crate::lifetime::AddLifetime`] trait is the only solution the author found at time
//! of writing that both avoids the trampoline the closure-like approaches induce and the above
//! lifetime error. This leads to a few practical limitations:
//!
//! * Types passed through the function pointer interface can have at most a single lifetime.
//! * The lifetimes of types passed through the function pointer interface must all be disjoint.
//! * The return type cannot have a lifetime.
//!
//! Practically, these limitations are acceptable because micro-architecture dispatching is
//! almost always the result of doing mathematical manipulation on primitive types and thus
//! the lifetimes of the associated types are generally not complicated.
//!
//! ## Obtaining the [`Current`] Architecture
//!
//! When Rust crates are compiled, they can be provided with a target CPU. The function
//! [`current`], the type [`Current`], and the constant [`crate::ARCH`] are all populated with
//! the best matching wide [`Architecture`] selected at compile time.
//!
//! ## Hierarchies
//!
//! Each [`Architecture`] exposes a [`Level`] via [`Architecture::level()`] that
//! can be used to compare capabilities without instantiating the architecture.
//!
//! ### X86
//!
//! * [`x86_64::V4`]: Supporting AVX-512 (and AVX2 and lower).
//! * [`x86_64::V3`]: Supporting AVX2 and lower.
//! * [`Scalar`]: Fallback architecture.
//!
//! The ordering is `Scalar` < `V3` < `V4`.
//!
//! ### Arm
//!
//! Currently, Arm support is limited to [`Scalar`].

use half::f16;

use crate::{
    Const, SIMDCast, SIMDDotProduct, SIMDFloat, SIMDMask, SIMDSelect, SIMDSigned, SIMDSumTree,
    SIMDUnsigned, SIMDVector, SplitJoin, lifetime::AddLifetime,
};

pub(crate) mod emulated;

/// An [`Architecture`] that implements all operation as scalar loops, relying on the
/// compiler for optimization.
pub use emulated::Scalar;

/// An opaque representation of an [`Architecture`]'s capability level.
///
/// `Level` allows comparing the relative capabilities of different architectures
/// without requiring an instance of the architecture type. This is useful for
/// compile-time checks against [`crate::ARCH`] where constructing architecture
/// types like [`x86_64::V3`] would require `unsafe`.
///
/// Levels are totally ordered within an ISA family, with greater values indicating
/// more capable instruction sets. [`Scalar`] is always the lowest level.
///
/// # Examples
///
/// Checking if the compile-time architecture meets a minimum capability:
///
/// ```
/// #[cfg(target_arch = "x86_64")]
/// use diskann_wide::{Architecture, arch};
///
/// // Check at compile time whether we were built with AVX2+ support.
/// #[cfg(target_arch = "x86_64")]
/// let _meets_v3 = arch::Current::level() >= arch::x86_64::V3::level();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Level(LevelInner);

impl Level {
    const fn scalar() -> Self {
        Self(LevelInner::Scalar)
    }
}

cfg_if::cfg_if! {
    if #[cfg(any(target_arch = "x86_64", doc))] {
        // Delegate to the architecture selection within the `x86_64` module.
        pub mod x86_64;

        use x86_64::LevelInner;

        pub use x86_64::current;
        pub use x86_64::Current;

        pub use x86_64::dispatch;
        pub use x86_64::dispatch1;
        pub use x86_64::dispatch2;
        pub use x86_64::dispatch3;

        pub use x86_64::dispatch_no_features;
        pub use x86_64::dispatch1_no_features;
        pub use x86_64::dispatch2_no_features;
        pub use x86_64::dispatch3_no_features;

        impl Level {
            const fn v3() -> Self {
                Self(LevelInner::V3)
            }

            const fn v4() -> Self {
                Self(LevelInner::V4)
            }
        }
    } else {
        pub type Current = Scalar;

        // There is only one architecture present in this mode.
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        enum LevelInner {
            Scalar,
        }

        pub const fn current() -> Current {
            Scalar::new()
        }

        /// Run the target functor.
        ///
        /// In scalar mode, this does nothing special.
        pub fn dispatch<T, R>(f: T) -> R
        where T: Target<Scalar, R> {
            f.run(Scalar::new())
        }

        /// Run the target functor.
        ///
        /// In scalar mode, this does nothing special.
        pub fn dispatch1<T, T0, R>(f: T, x0: T0) -> R
        where T: Target1<Scalar, R, T0> {
            f.run(Scalar::new(), x0)
        }

        /// Run the target functor.
        ///
        /// In scalar mode, this does nothing special.
        pub fn dispatch2<T, T0, T1, R>(f: T, x0: T0, x1: T1) -> R
        where T: Target2<Scalar, R, T0, T1> {
            f.run(Scalar::new(), x0, x1)
        }

        /// Run the target functor.
        ///
        /// In scalar mode, this does nothing special.
        pub fn dispatch3<T, T0, T1, T2, R>(f: T, x0: T0, x1: T1, x2: T2) -> R
        where T: Target3<Scalar, R, T0, T1, T2> {
            f.run(Scalar::new(), x0, x1, x2)
        }

        /// Run the target functor.
        ///
        /// In scalar mode, this does nothing special.
        pub fn dispatch_no_features<T, R>(f: T) -> R
        where T: Target<Scalar, R> {
            f.run(Scalar::new())
        }

        /// Run the target functor.
        ///
        /// In scalar mode, this does nothing special.
        pub fn dispatch1_no_features<T, T0, R>(f: T, x0: T0) -> R
        where T: Target1<Scalar, R, T0> {
            f.run(Scalar::new(), x0)
        }

        /// Run the target functor.
        ///
        /// In scalar mode, this does nothing special.
        pub fn dispatch2_no_features<T, T0, T1, R>(f: T, x0: T0, x1: T1) -> R
        where T: Target2<Scalar, R, T0, T1> {
            f.run(Scalar::new(), x0, x1)
        }

        /// Run the target functor.
        ///
        /// In scalar mode, this does nothing special.
        pub fn dispatch3_no_features<T, T0, T1, T2, R>(f: T, x0: T0, x1: T1, x2: T2) -> R
        where T: Target3<Scalar, R, T0, T1, T2> {
            f.run(Scalar::new(), x0, x1, x2)
        }
    }
}

mod sealed {
    pub trait Sealed: std::fmt::Debug + Copy + PartialEq + Send + Sync + 'static {}
}

pub(crate) use sealed::Sealed;

macro_rules! vector {
    ($me:ident: <$self:ident, $T:ty, $N:literal, $mask:ident> + $($rest:tt)*) => {
        type $me: SIMDVector<Arch = $self, Scalar = $T, ConstLanes = Const<$N>, Mask = Self::$mask> + $($rest)*;
    }
}

#[allow(non_camel_case_types)]
pub trait Architecture: sealed::Sealed {
    // mask types
    type mask_f16x8: SIMDMask;
    type mask_f16x16: SIMDMask;

    type mask_f32x4: SIMDMask + SIMDSelect<Self::f32x4>;
    type mask_f32x8: SIMDMask + SIMDSelect<Self::f32x8>;
    type mask_f32x16: SIMDMask + SIMDSelect<Self::f32x16>;

    type mask_i8x16: SIMDMask;
    type mask_i8x32: SIMDMask;
    type mask_i8x64: SIMDMask;

    type mask_i16x8: SIMDMask;
    type mask_i16x16: SIMDMask;
    type mask_i16x32: SIMDMask;

    type mask_i32x4: SIMDMask;
    type mask_i32x8: SIMDMask + From<Self::mask_f32x8> + SIMDSelect<Self::i32x8>;
    type mask_i32x16: SIMDMask + SIMDSelect<Self::i32x16>;

    type mask_u8x16: SIMDMask;
    type mask_u8x32: SIMDMask;
    type mask_u8x64: SIMDMask;

    type mask_u32x4: SIMDMask;
    type mask_u32x8: SIMDMask + From<Self::mask_f32x8>;
    type mask_u32x16: SIMDMask + SIMDSelect<Self::u32x16>;
    type mask_u64x2: SIMDMask;
    type mask_u64x4: SIMDMask;

    /////////////////
    //-- vectors --//
    /////////////////

    // floats
    vector!(
        f16x8: <Self, f16, 8, mask_f16x8>
        + SIMDCast<f32, Cast = Self::f32x8>
    );
    vector!(
        f16x16: <Self, f16, 16, mask_f16x16>
        + SplitJoin<Halved = Self::f16x8>
        + SIMDCast<f32, Cast = Self::f32x16>
    );

    vector!(
        f32x4: <Self, f32, 4, mask_f32x4>
        + SIMDFloat
        + SIMDSumTree
    );
    vector!(
        f32x8: <Self, f32, 8, mask_f32x8>
        + SIMDFloat
        + SIMDSumTree
        + SIMDCast<f16, Cast = Self::f16x8>
        + SplitJoin<Halved = Self::f32x4>
        + From<Self::f16x8>
    );
    vector!(
        f32x16: <Self, f32, 16, mask_f32x16>
        + SIMDFloat
        + SplitJoin<Halved = Self::f32x8>
        + SIMDSumTree
        + From<Self::f16x16>
    );

    // signed-integer
    vector!(
        i8x16: <Self, i8, 16, mask_i8x16>
        + SIMDSigned
    );
    vector!(
        i8x32: <Self, i8, 32, mask_i8x32>
        + SIMDSigned
    );
    vector!(
        i8x64: <Self, i8, 64, mask_i8x64>
        + SIMDSigned
    );

    vector!(
        i16x8: <Self, i16, 8, mask_i16x8>
        + SIMDSigned
    );
    vector!(
        i16x16: <Self, i16, 16, mask_i16x16>
        + SIMDSigned
        + SplitJoin<Halved = Self::i16x8>
        + From<Self::i8x16>
        + From<Self::u8x16>
    );
    vector!(
        i16x32: <Self, i16, 32, mask_i16x32>
        + SIMDSigned
        + SplitJoin<Halved = Self::i16x16>
        + From<Self::i8x32>
        + From<Self::u8x32>
    );

    vector!(
        i32x4: <Self, i32, 4, mask_i32x4>
        + SIMDSigned
    );
    vector!(
        i32x8: <Self, i32, 8, mask_i32x8>
        + SIMDSigned
        + SIMDSumTree
        + SplitJoin<Halved = Self::i32x4>
        + SIMDDotProduct<Self::i16x16>
        + SIMDDotProduct<Self::u8x32, Self::i8x32>
        + SIMDDotProduct<Self::i8x32, Self::u8x32>
        + SIMDCast<f32, Cast = Self::f32x8>
    );
    vector!(
        i32x16: <Self, i32, 16, mask_i32x16>
        + SIMDSigned
        + SIMDSumTree
        + SplitJoin<Halved = Self::i32x8>
        + SIMDDotProduct<Self::u8x64, Self::i8x64>
        + SIMDDotProduct<Self::i8x64, Self::u8x64>
    );

    // unsigned-integer
    vector!(
        u8x16: <Self, u8, 16, mask_u8x16>
        + SIMDUnsigned
    );
    vector!(
        u8x32: <Self, u8, 32, mask_u8x32>
        + SIMDUnsigned
    );

    vector!(
        u8x64: <Self, u8, 64, mask_u8x64>
        + SIMDUnsigned
    );

    vector!(
        u32x4: <Self, u32, 4, mask_u32x4>
        + SIMDUnsigned
    );
    vector!(
        u32x8: <Self, u32, 8, mask_u32x8>
        + SplitJoin<Halved = Self::u32x4>
        + SIMDUnsigned
        + SIMDSumTree
    );
    vector!(
        u32x16: <Self, u32, 16, mask_u32x16>
        + SIMDUnsigned
        + SIMDSumTree
        + SplitJoin<Halved = Self::u32x8>
    );

    vector!(
        u64x2: <Self, u64, 2, mask_u64x2>
        + SIMDUnsigned
    );
    vector!(
        u64x4: <Self, u64, 4, mask_u64x4>
        + SplitJoin<Halved = Self::u64x2>
        + SIMDUnsigned
    );

    //---------//
    // Methods //
    //---------//

    /// Return an opaque [`Level`] representing the capabilities of this architecture.
    ///
    /// Levels that compare greater represent architectures that are more capable.
    ///
    /// # Examples
    ///
    /// ```
    /// use diskann_wide::{Architecture, arch};
    ///
    /// // Scalar is the baseline — every other architecture compares greater.
    /// assert_eq!(arch::Scalar::level(), arch::Scalar::level());
    ///
    /// #[cfg(target_arch = "x86_64")]
    /// assert!(arch::Scalar::level() < arch::x86_64::V3::level());
    /// ```
    fn level() -> Level;

    /// Run the provided closure targeting this architecture.
    ///
    /// This function is always safe to call, but the function `f` likely needs to be
    /// inlined into `run` in for the correct target features to be applied.
    fn run<F, R>(self, f: F) -> R
    where
        F: Target<Self, R>;

    /// Run the provided closure targeting this architecture with an inlining hint.
    ///
    /// This function is always safe to call, but the function `f` likely needs to be
    /// inlined into `run` in for the correct target features to be applied.
    ///
    /// Note that although an inline hint is applied, it is not a guaranteed that this call
    /// will be inlined due to the interaction of `target_features`. If you really need `F`
    /// to be inlined, you can call its `Target` method directly, but care must be taken
    /// because this will not reapply `target_features`.
    fn run_inline<F, R>(self, f: F) -> R
    where
        F: Target<Self, R>;

    /// Run the provided closure targeting this architecture with an additional argument.
    ///
    /// This function is always safe to call, but the function `f` likely needs to be
    /// inlined into `run` in for the correct target features to be applied.
    fn run1<F, T0, R>(self, f: F, x0: T0) -> R
    where
        F: Target1<Self, R, T0>;

    /// Run the provided closure targeting this architecture with an additional argument and
    /// an inlining hint.
    ///
    /// This function is always safe to call, but the function `f` likely needs to be
    /// inlined into `run` in for the correct target features to be applied.
    ///
    /// Note that although an inline hint is applied, it is not a guaranteed that this call
    /// will be inlined due to the interaction of `target_features`. If you really need `F`
    /// to be inlined, you can call its `Target1` method directly, but care must be taken
    /// because this will not reapply `target_features`.
    fn run1_inline<F, T0, R>(self, f: F, x0: T0) -> R
    where
        F: Target1<Self, R, T0>;

    /// Run the provided closure targeting this architecture with two additional arguments.
    ///
    /// This function is always safe to call, but the function `f` likely needs to be
    /// inlined into `run` in for the correct target features to be applied.
    fn run2<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
    where
        F: Target2<Self, R, T0, T1>;

    /// Run the provided closure targeting this architecture with two additional arguments
    /// and an inlining hint.
    ///
    /// This function is always safe to call, but the function `f` likely needs to be
    /// inlined into `run` in for the correct target features to be applied.
    ///
    /// Note that although an inline hint is applied, it is not a guaranteed that this call
    /// will be inlined due to the interaction of `target_features`. If you really need `F`
    /// to be inlined, you can call its `Target2` method directly, but care must be taken
    /// because this will not reapply `target_features`.
    fn run2_inline<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
    where
        F: Target2<Self, R, T0, T1>;

    /// Run the provided closure targeting this architecture with three additional arguments.
    ///
    /// This function is always safe to call, but the function `f` likely needs to be
    /// inlined into `run` in for the correct target features to be applied.
    fn run3<F, T0, T1, T2, R>(self, f: F, x0: T0, x1: T1, x2: T2) -> R
    where
        F: Target3<Self, R, T0, T1, T2>;

    /// Run the provided closure targeting this architecture with three additional arguments
    /// and an inlining hint.
    ///
    /// This function is always safe to call, but the function `f` likely needs to be
    /// inlined into `run` in for the correct target features to be applied.
    ///
    /// Note that although an inline hint is applied, it is not a guaranteed that this call
    /// will be inlined due to the interaction of `target_features`. If you really need `F`
    /// to be inlined, you can call its `Target3` method directly, but care must be taken
    /// because this will not reapply `target_features`.
    fn run3_inline<F, T0, T1, T2, R>(self, f: F, x0: T0, x1: T1, x2: T2) -> R
    where
        F: Target3<Self, R, T0, T1, T2>;

    /// Return a function pointer invoking [`FTarget1::run`] with `self` as the architecture.
    /// ```
    /// use diskann_wide::{Architecture, arch::FTarget1};
    ///
    /// struct Square;
    ///
    /// impl<A: Architecture> FTarget1<A, f32, f32> for Square
    /// {
    ///     fn run(_: A, x: f32) -> f32 {
    ///         x * x
    ///     }
    /// }
    ///
    /// let f = (diskann_wide::ARCH).dispatch1::<Square, f32, f32>();
    /// assert_eq!(f.call(10.0), 100.0);
    /// ```
    fn dispatch1<F, R, T0>(self) -> Dispatched1<R, T0>
    where
        T0: AddLifetime,
        F: for<'a> FTarget1<Self, R, T0::Of<'a>>;

    /// Return a function pointer invoking [`FTarget2::run`] with `self` as the architecture.
    /// ```
    /// use diskann_wide::{
    ///     Architecture,
    ///     arch::FTarget2,
    ///     lifetime::{Mut, Ref},
    /// };
    ///
    /// struct Copy;
    ///
    /// // Copy a slice and return the number of elements.
    /// impl<A: Architecture> FTarget2<A, usize, &mut [f32], &[f32]> for Copy
    /// {
    ///     fn run(_: A, dst: &mut [f32], src: &[f32]) -> usize {
    ///         dst.copy_from_slice(src);
    ///         src.len()
    ///     }
    /// }
    ///
    /// let f = (diskann_wide::ARCH).dispatch2::<Copy, usize, Mut<[f32]>, Ref<[f32]>>();
    /// let src = [1.0, 2.0, 3.0];
    /// let mut dst = [0.0f32; 3];
    /// assert_eq!(f.call(&mut dst, &src), 3);
    /// assert_eq!(src, dst);
    /// ```
    fn dispatch2<F, R, T0, T1>(self) -> Dispatched2<R, T0, T1>
    where
        T0: AddLifetime,
        T1: AddLifetime,
        F: for<'a, 'b> FTarget2<Self, R, T0::Of<'a>, T1::Of<'b>>;

    /// Return a function pointer invoking [`FTarget3::run`] with `self` as the architecture.
    /// ```
    /// use diskann_wide::{
    ///     Architecture,
    ///     arch::FTarget3,
    ///     lifetime::Ref,
    /// };
    ///
    /// struct Sum;
    ///
    /// // Return the sum of the three arguments.
    /// impl<A: Architecture> FTarget3<A, f32, &f32, &f32, &f32> for Sum
    /// {
    ///     fn run(_: A, x: &f32, y: &f32, z: &f32) -> f32 {
    ///         x + y + z
    ///     }
    /// }
    ///
    /// let f = (diskann_wide::ARCH).dispatch3::<Sum, f32, Ref<f32>, Ref<f32>, Ref<f32>>();
    /// assert_eq!(f.call(&1.0, &2.0, &3.0), 6.0);
    /// ```
    fn dispatch3<F, R, T0, T1, T2>(self) -> Dispatched3<R, T0, T1, T2>
    where
        T0: AddLifetime,
        T1: AddLifetime,
        T2: AddLifetime,
        F: for<'a, 'b, 'c> FTarget3<Self, R, T0::Of<'a>, T1::Of<'b>, T2::Of<'c>>;
}

/// A functor that targets a particular architecture, accepting no additional arguments.
pub trait Target<A, R>
where
    A: Architecture,
{
    /// Run the operation with the provided `Architecture`.
    fn run(self, arch: A) -> R;
}

/// A functor that targets a particular architecture, accepting one additional arguments.
pub trait Target1<A, R, T0>
where
    A: Architecture,
{
    /// Run the operation with the provided `Architecture`.
    fn run(self, arch: A, x0: T0) -> R;
}

/// A functor that targets a particular architecture, accepting two additional arguments.
pub trait Target2<A, R, T0, T1>
where
    A: Architecture,
{
    /// Run the operation with the provided `Architecture`.
    fn run(self, arch: A, x0: T0, x1: T1) -> R;
}

/// A functor that targets a particular architecture, accepting three additional arguments.
pub trait Target3<A, R, T0, T1, T2>
where
    A: Architecture,
{
    /// Run the operation with the provided `Architecture`.
    fn run(self, arch: A, x0: T0, x1: T1, x2: T2) -> R;
}

/// A variation of [`Target1`] that uses an associated function instead of a method.
///
/// This is useful used in the function pointer API.
pub trait FTarget1<A, R, T0>
where
    A: Architecture,
{
    fn run(arch: A, x0: T0) -> R;
}

/// A variation of [`Target2`] that uses an associated function instead of a method.
///
/// This is useful used in the function pointer API.
pub trait FTarget2<A, R, T0, T1>
where
    A: Architecture,
{
    fn run(arch: A, x0: T0, x1: T1) -> R;
}

/// A variation of [`Target3`] that uses an associated function instead of a method.
///
/// This is useful used in the function pointer API.
pub trait FTarget3<A, R, T0, T1, T2>
where
    A: Architecture,
{
    fn run(arch: A, x0: T0, x1: T1, x2: T2) -> R;
}

/// Run the closure with code-generated for the specified architecture.
///
/// Note that if the body of the closure is not inlined, this will likely have no effect.
impl<A, R, F> Target<A, R> for F
where
    A: Architecture,
    F: FnOnce() -> R,
{
    #[inline]
    fn run(self, _: A) -> R {
        (self)()
    }
}

/// Run the closure with code-generated for the specified architecture.
///
/// Note that if the body of the closure is not inlined, this will likely have no effect.
impl<A, R, T0, F> Target1<A, R, T0> for F
where
    A: Architecture,
    F: FnOnce(T0) -> R,
{
    #[inline]
    fn run(self, _: A, x0: T0) -> R {
        (self)(x0)
    }
}

/// Run the closure with code-generated for the specified architecture.
///
/// Note that if the body of the closure is not inlined, this will likely have no effect.
impl<A, R, T0, T1, F> Target2<A, R, T0, T1> for F
where
    A: Architecture,
    F: FnOnce(T0, T1) -> R,
{
    #[inline]
    fn run(self, _: A, x0: T0, x1: T1) -> R {
        (self)(x0, x1)
    }
}

/// Run the closure with code-generated for the specified architecture.
///
/// Note that if the body of the closure is not inlined, this will likely have no effect.
impl<A, R, T0, T1, T2, F> Target3<A, R, T0, T1, T2> for F
where
    A: Architecture,
    F: FnOnce(T0, T1, T2) -> R,
{
    #[inline]
    fn run(self, _: A, x0: T0, x1: T1, x2: T2) -> R {
        (self)(x0, x1, x2)
    }
}

/// A hidden architecture for use in the function pointer API.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct Hidden;

const _ASSERT_ZST: () = assert!(
    std::mem::size_of::<Hidden>() == 0,
    "Hidden **must** be zero sized"
);

const _ASSERT_ALIGNED: () = assert!(
    std::mem::align_of::<Hidden>() == 1,
    "Hidden **must** be alignment 1"
);

macro_rules! dispatched {
    ($name:ident, { $($Ts:ident )* }, { $($xs:ident )* }, { $($lt:lifetime )* }) => {
        /// A function pointer that calls directly into a micro-architecture optimized
        /// function, returning a value of type `R` and accepting the speficied number of
        /// arguments.
        ///
        /// Arguments are mapped using the [`AddLifetime`] trait to enable passing structs
        /// with up to a single non-`'static` lifetime parameter.
        ///
        /// This type is guaranteed:
        ///
        /// * To have the same size as a regular function pointer.
        /// * To have the same ABI as a regular function pointer.
        /// * To use the null-pointer optimization - so `Option<Self>` has the same size as
        ///   `Self`.
        #[derive(Debug)]
        #[repr(transparent)]
        pub struct $name<R, $($Ts,)*>
        where
            $($Ts: AddLifetime,)*
        {
            f: for<$($lt,)*> unsafe fn(Hidden, $($Ts::Of<$lt>,)*) -> R,
        }

        impl<R, $($Ts,)*> $name<R, $($Ts,)*>
        where
            $($Ts: AddLifetime,)*
        {
            /// Construct a new safe instance of `Self` around the raw function pointer.
            ///
            /// # Safety
            ///
            /// The caller asserts that the runtime implementation of `f` is safe to call
            /// on the current CPU target.
            ///
            /// Usually, this means that the runtime CPU has the target features required
            /// for the destination of `f`.
            unsafe fn new(f: unsafe fn(Hidden, $($Ts::Of<'_>,)*) -> R) -> Self {
                Self { f }
            }

            /// Invoke the function with the lifetime-annotated arguments and return the
            /// result.
            ///
            /// The below example demonstrates the behavior for [`Dispatched1`], but the
            /// same logic applies to all variadic instances.
            #[inline(always)]
            pub fn call(self, $($xs: $Ts::Of<'_>,)*) -> R {
                // SAFETY: The constructor of `Dispatched` asserts the call is safe.
                unsafe { (self.f)(Hidden, $($xs,)*) }
            }
        }

        impl<R, $($Ts,)*> Clone for $name<R, $($Ts,)*>
        where
            $($Ts: AddLifetime,)*
        {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<R, $($Ts,)*> Copy for $name<R, $($Ts,)*>
        where
            $($Ts: AddLifetime,)*
        {
        }
    }
}

dispatched!(Dispatched1, { T0 }, { x0 }, { 'a0 });
dispatched!(Dispatched2, { T0 T1 }, { x0 x1 }, { 'a0 'a1 });
dispatched!(Dispatched3, { T0 T1 T2 }, { x0 x1 x2 }, { 'a0 'a1 'a2 });

/// This macro stamps out the function-pointer tranmute trick we use to type-erase
/// architecture in the function-pointer API.
macro_rules! hide {
    ($name:ident, $dispatched:ident, { $($Ts:ident )* }) => {
        /// Construct a new instance of [`Self`] from the raw function pointer.
        ///
        /// # Safety
        ///
        /// Internally, we will transmute the zero-sized type `A` to another hidden type
        /// to erase the architecture information of the dispatched function pointer.
        ///
        /// This function will not compile if `A` is not zero sized nor has an alignment
        /// of 1.
        ///
        /// We can do this because Rust guarantees that zero sized types are ABI
        /// compatible.
        ///
        /// The caller must ensure that winking into existance and instance of `A` is
        /// a safe operation. For [`Architectures`], this means that the requirements
        /// of `A::new()` are uphelf.
        ///
        /// Put plainly:
        ///
        /// 1. The runtime CPU must support the target features required by `A`.
        /// 2. `A` **must** be a zero-sized type with an alignment of 1.
        unsafe fn $name<A, R, $($Ts,)*>(
            f: unsafe fn(A, $($Ts::Of<'_>,)*) -> R
        ) -> $dispatched<R, $($Ts,)*>
        where
            $($Ts: AddLifetime,)*
        {
            // Check that `A` is a zero sized type.
            const {
                assert!(
                    std::mem::size_of::<A>() == 0,
                    "A must be zero sized to be ABI compatible with `Hidden`"
                )
            };

            // Check that `A` has alignment 1.
            const {
                assert!(
                    std::mem::align_of::<A>() == 1,
                    "A must have an alignment of 1 to be ABI compatible with `Hidden`"
                )
            };

            // SAFETY: The transmute is safe because `Hidden` and `A` are both zero
            // sized types with alignment 1, which are ABI compatible.
            //
            // All the rest of the arguments are untouched.
            //
            // The caller asserts that it is safe to call this function pointer.
            let f = unsafe {
                std::mem::transmute::<
                    unsafe fn(A, $($Ts::Of<'_>,)*) -> R,
                    unsafe fn(Hidden, $($Ts::Of<'_>,)*) -> R
                >(f)
            };

            // SAFETY: The caller asserts that it is safe to call `f`.
            unsafe { $dispatched::new(f) }
        }
    }
}

hide!(hide1, Dispatched1, { T0 });
hide!(hide2, Dispatched2, { T0 T1 });
hide!(hide3, Dispatched3, { T0 T1 T2 });

// Macros to help implement architectures.

macro_rules! maskdef {
    ($mask:ident = $repr:ty) => {
        type $mask = <$repr as SIMDVector>::Mask;
    };
    ($($mask:ident = $repr:ty),+ $(,)?) => {
        $($crate::arch::maskdef!($mask = $repr);)+
    };
    () => {
        $crate::arch::maskdef!(
            mask_f16x8 = f16x8,
            mask_f16x16 = f16x16,

            mask_f32x4 = f32x4,
            mask_f32x8 = f32x8,
            mask_f32x16 = f32x16,

            mask_i8x16 = i8x16,
            mask_i8x32 = i8x32,
            mask_i8x64 = i8x64,

            mask_i16x8 = i16x8,
            mask_i16x16 = i16x16,
            mask_i16x32 = i16x32,

            mask_i32x4 = i32x4,
            mask_i32x8 = i32x8,
            mask_i32x16 = i32x16,

            mask_u8x16 = u8x16,
            mask_u8x32 = u8x32,
            mask_u8x64 = u8x64,

            mask_u32x4 = u32x4,
            mask_u32x8 = u32x8,
            mask_u32x16 = u32x16,

            mask_u64x2 = u64x2,
            mask_u64x4 = u64x4,
        );
    };
}

macro_rules! typedef {
    () => {
        $crate::arch::typedef!(
            f16x8,
            f16x16,

            f32x4,
            f32x8,
            f32x16,

            i8x16,
            i8x32,
            i8x64,

            i16x8,
            i16x16,
            i16x32,

            i32x4,
            i32x8,
            i32x16,

            u8x16,
            u8x32,
            u8x64,

            u32x4,
            u32x8,
            u32x16,

            u64x2,
            u64x4,
        );
    };
    ($repr:ident) => {
        type $repr = $repr;
    };
    ($($repr:ident),+ $(,)?) => {
        $($crate::arch::typedef!($repr);)+
    };
}

pub(crate) use maskdef;
pub(crate) use typedef;

///////////
// Tests //
///////////

// All tests here **must** run successfully under Miri.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifetime::{Mut, Ref};

    struct TestOp;

    // Returns a static string.
    impl<A> Target<A, &'static str> for TestOp
    where
        A: Architecture,
    {
        fn run(self, _: A) -> &'static str {
            "hello world"
        }
    }

    // Simply add all elements in the array and return the result.
    impl<A> Target1<A, f32, &[f32]> for TestOp
    where
        A: Architecture,
    {
        fn run(self, _: A, x: &[f32]) -> f32 {
            x.iter().sum()
        }
    }

    impl<A> FTarget1<A, f32, &[f32]> for TestOp
    where
        A: Architecture,
    {
        fn run(arch: A, x: &[f32]) -> f32 {
            <_ as Target1<_, _, _>>::run(Self, arch, x)
        }
    }

    // Both perform a sum and copy the results into the destination.
    impl<A> Target2<A, f32, &mut [f32], &[f32]> for TestOp
    where
        A: Architecture,
    {
        fn run(self, _: A, x: &mut [f32], y: &[f32]) -> f32 {
            x.copy_from_slice(y);
            y.iter().sum()
        }
    }

    impl<A> FTarget2<A, f32, &mut [f32], &[f32]> for TestOp
    where
        A: Architecture,
    {
        fn run(arch: A, x: &mut [f32], y: &[f32]) -> f32 {
            <_ as Target2<_, _, _, _>>::run(TestOp, arch, x, y)
        }
    }

    impl<A> Target3<A, f32, &mut [f32], &[f32], f32> for TestOp
    where
        A: Architecture,
    {
        fn run(self, _: A, x: &mut [f32], y: &[f32], z: f32) -> f32 {
            assert_eq!(x.len(), y.len());
            x.iter_mut().zip(y.iter()).for_each(|(d, s)| *d = *s + z);
            y.iter().sum()
        }
    }

    impl<A> FTarget3<A, f32, &mut [f32], &[f32], f32> for TestOp
    where
        A: Architecture,
    {
        fn run(arch: A, x: &mut [f32], y: &[f32], z: f32) -> f32 {
            <_ as Target3<_, _, _, _, _>>::run(TestOp, arch, x, y, z)
        }
    }

    //------------------//
    // Target Interface //
    //------------------//

    #[test]
    fn zero_arg_target() {
        let expected = "hello world";
        assert_eq!((Scalar).run(TestOp), expected);
        assert_eq!((Scalar).run_inline(TestOp), expected);

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V3::new_checked_uncached() {
            assert_eq!(arch.run(TestOp), expected);
            assert_eq!(arch.run_inline(TestOp), expected);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V4::new_checked_miri() {
            assert_eq!(arch.run(TestOp), expected);
            assert_eq!(arch.run_inline(TestOp), expected);
        }
    }

    #[test]
    fn one_arg_target() {
        let src = [1.0f32, 2.0f32, 3.0f32];
        let sum: f32 = src.iter().sum();

        assert_eq!((Scalar).run1(TestOp, &src), sum);
        assert_eq!((Scalar).run1_inline(TestOp, &src), sum);

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V3::new_checked_uncached() {
            assert_eq!(arch.run1(TestOp, &src), sum);
            assert_eq!(arch.run1_inline(TestOp, &src), sum);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V4::new_checked_miri() {
            assert_eq!(arch.run1(TestOp, &src), sum);
            assert_eq!(arch.run1_inline(TestOp, &src), sum);
        }
    }

    #[test]
    fn two_arg_target() {
        let src = [1.0f32, 2.0f32, 3.0f32];
        let sum: f32 = src.iter().sum();

        macro_rules! gen_test {
            ($arch:ident) => {{
                let mut dst = [0.0f32; 3];
                assert_eq!($arch.run2(TestOp, &mut dst, &src), sum);
                assert_eq!(dst, src);
            }

            {
                let mut dst = [0.0f32; 3];
                assert_eq!($arch.run2_inline(TestOp, &mut dst, &src), sum);
                assert_eq!(dst, src);
            }};
        }

        gen_test!(Scalar);

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V3::new_checked_uncached() {
            gen_test!(arch);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V4::new_checked_miri() {
            gen_test!(arch);
        }
    }

    #[test]
    fn three_arg_target() {
        let src = [1.0f32, 2.0f32, 3.0f32];
        let sum: f32 = src.iter().sum();
        let offset = 10.0f32;
        let expected = [11.0f32, 12.0f32, 13.0f32];

        macro_rules! gen_test {
            ($arch:ident) => {{
                let mut dst = [0.0f32; 3];
                assert_eq!($arch.run3(TestOp, &mut dst, &src, offset), sum);
                assert_eq!(dst, expected);
            }

            {
                let mut dst = [0.0f32; 3];
                assert_eq!($arch.run3_inline(TestOp, &mut dst, &src, offset), sum);
                assert_eq!(dst, expected);
            }};
        }

        gen_test!(Scalar);

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V3::new_checked_uncached() {
            gen_test!(arch);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V4::new_checked_miri() {
            gen_test!(arch);
        }
    }

    //----------------------------//
    // Function Pointer Interface //
    //----------------------------//

    #[test]
    fn one_arg_function_pointer() {
        let src = [1.0f32, 2.0f32, 3.0f32];
        let sum: f32 = src.iter().sum();

        type FnPtr = Dispatched1<f32, Ref<[f32]>>;

        assert_eq!(std::mem::size_of::<FnPtr>(), std::mem::size_of::<fn()>());
        assert_eq!(
            std::mem::size_of::<Option<FnPtr>>(),
            std::mem::size_of::<fn()>()
        );

        {
            let f: FnPtr = (Scalar).dispatch1::<TestOp, f32, Ref<[f32]>>();
            assert_eq!(f.call(&src), sum);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V3::new_checked_uncached() {
            let f: FnPtr = arch.dispatch1::<TestOp, f32, Ref<[f32]>>();
            assert_eq!(f.call(&src), sum);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V4::new_checked_miri() {
            let f: FnPtr = arch.dispatch1::<TestOp, f32, Ref<[f32]>>();
            assert_eq!(f.call(&src), sum);
        }
    }

    #[test]
    fn two_arg_function_pointer() {
        let src = [1.0f32, 2.0f32, 3.0f32];
        let sum: f32 = src.iter().sum();

        type FnPtr = Dispatched2<f32, Mut<[f32]>, Ref<[f32]>>;

        assert_eq!(std::mem::size_of::<FnPtr>(), std::mem::size_of::<fn()>());
        assert_eq!(
            std::mem::size_of::<Option<FnPtr>>(),
            std::mem::size_of::<fn()>()
        );

        {
            let mut dst = [0.0f32; 3];
            let f: FnPtr = (Scalar).dispatch2::<TestOp, f32, Mut<[f32]>, Ref<[f32]>>();
            assert_eq!(f.call(&mut dst, &src), sum);
            assert_eq!(dst, src);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V3::new_checked_uncached() {
            let mut dst = [0.0f32; 3];
            let f: FnPtr = arch.dispatch2::<TestOp, f32, Mut<[f32]>, Ref<[f32]>>();
            assert_eq!(f.call(&mut dst, &src), sum);
            assert_eq!(dst, src);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V4::new_checked_miri() {
            let mut dst = [0.0f32; 3];
            let f: FnPtr = arch.dispatch2::<TestOp, f32, Mut<[f32]>, Ref<[f32]>>();
            assert_eq!(f.call(&mut dst, &src), sum);
            assert_eq!(dst, src);
        }
    }

    #[test]
    fn three_arg_function_pointer() {
        let src = [1.0f32, 2.0f32, 3.0f32];
        let sum: f32 = src.iter().sum();
        let offset = 10.0f32;
        let expected = [11.0f32, 12.0f32, 13.0f32];

        type FnPtr = Dispatched3<f32, Mut<[f32]>, Ref<[f32]>, f32>;

        assert_eq!(std::mem::size_of::<FnPtr>(), std::mem::size_of::<fn()>());
        assert_eq!(
            std::mem::size_of::<Option<FnPtr>>(),
            std::mem::size_of::<fn()>()
        );

        {
            let mut dst = [0.0f32; 3];
            let f: FnPtr = (Scalar).dispatch3::<TestOp, f32, Mut<[f32]>, Ref<[f32]>, f32>();
            assert_eq!(f.call(&mut dst, &src, offset), sum);
            assert_eq!(dst, expected);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V3::new_checked_uncached() {
            let mut dst = [0.0f32; 3];
            let f: FnPtr = arch.dispatch3::<TestOp, f32, Mut<[f32]>, Ref<[f32]>, f32>();
            assert_eq!(f.call(&mut dst, &src, offset), sum);
            assert_eq!(dst, expected);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = x86_64::V4::new_checked_miri() {
            let mut dst = [0.0f32; 3];
            let f: FnPtr = arch.dispatch3::<TestOp, f32, Mut<[f32]>, Ref<[f32]>, f32>();
            assert_eq!(f.call(&mut dst, &src, offset), sum);
            assert_eq!(dst, expected);
        }
    }
}
