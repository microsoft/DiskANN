/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Formatting utilities for error chains.

use std::{cell::UnsafeCell, marker::PhantomData, mem::MaybeUninit};

/// Format the entire error chain for `err` by first calling `err.to_string()` and then
/// by walking the error's
/// [source tree](https://doc.rust-lang.org/std/error/trait.Error.html#method.source).
pub fn format<E>(err: &E) -> String
where
    E: std::error::Error + ?Sized,
{
    // Cast wrap the walking of the source chain into something that behaves like an
    // iterator.
    struct SourceIterator<'a>(Option<&'a (dyn std::error::Error + 'static)>);
    impl<'a> Iterator for SourceIterator<'a> {
        type Item = &'a (dyn std::error::Error + 'static);
        fn next(&mut self) -> Option<Self::Item> {
            let current = self.0;
            self.0 = match current {
                Some(current) => current.source(),
                None => None,
            };
            current
        }
    }

    // Get the base message from the error.
    let mut message = err.to_string();
    // Walk the source chain, formatting each
    for source in SourceIterator(err.source()) {
        message.push_str("\n    caused by: ");
        message.push_str(&source.to_string());
    }
    message
}

/// An implementation of `Box<dyn std::error::Error>` that stores the error payload inline,
/// avoiding dynamic memory allocation. This has several practical drawbacks:
///
/// 1. The size of the error payload must be at most `N` bytes.
/// 2. The alignment of the error payload must be at most 8 bytes.
///
/// Both of these contraints are verified using post-monomorphization errors.
///
/// # Example
///
/// ```
/// use diskann_quantization::error::InlineError;
///
/// let base_error = u32::try_from(u64::MAX).unwrap_err();
/// let mut error = InlineError::<8>::new(base_error);
/// assert_eq!(error.to_string(), base_error.to_string());
///
/// // Change the dynamic type of the contained error.
/// error = InlineError::new(Box::new(base_error));
/// assert_eq!(error.to_string(), base_error.to_string());
/// ```
#[repr(C)]
pub struct InlineError<const N: usize = 16> {
    // We place the vtable first to enable the niche-optimization.
    vtable: &'static ErrorVTable,

    // NOTE: We need to use `MaybeUninit` instead of `u8` to maintain the provenance of
    // any pointers/references stored in the payload.
    //
    // Additionally, the `UnsafeCell` is needed because the payload may have interior
    // mutability as a side-effect of API calls.
    object: UnsafeCell<[MaybeUninit<u8>; N]>,
}

// SAFETY: We only allow error payloads that are `Send`.
unsafe impl<const N: usize> Send for InlineError<N> {}

// SAFETY: We only allow error payloads that are `Sync`.
unsafe impl<const N: usize> Sync for InlineError<N> {}

impl<const N: usize> InlineError<N> {
    /// Construct a new `InlineError` around `error`.
    ///
    /// Fails to compile if:
    ///
    /// 1. `std::mem::align_of::<T>() > 8`: Objects of type `T` must be compatible with the
    ///    inline storage buffer.
    /// 2. `std::mem::size_of::<T>() > N`: Objects of type `T` must fit within a buffer of
    ///    size `N`.
    pub fn new<T>(error: T) -> Self
    where
        T: std::error::Error + Send + Sync + 'static,
    {
        const { assert!(std::mem::size_of::<T>() <= N, "error type is too big") };
        const {
            assert!(
                std::mem::align_of::<T>() <= std::mem::align_of::<&'static ErrorVTable>(),
                "error type has alignment stricter than 8"
            )
        };

        let mut this = Self {
            vtable: &ErrorVTable {
                debug: error_debug::<T>,
                display: error_display::<T>,
                source: error_source::<T>,
                drop: error_drop::<T>,
            },
            object: UnsafeCell::new([MaybeUninit::uninit(); N]),
        };

        // SAFETY: We have const assertions that the size and alignment of `T` are
        // compatible with the buffer we created.
        //
        // Additionally, the memory we are writing to does not have a valid object stored,
        // so using `ptr::write` will not leak memory.
        unsafe { this.object.get_mut().as_mut_ptr().cast::<T>().write(error) };

        this
    }

    // Return the base pointer of the inline storage in a type that propagates the lifetime
    // of `self`. This allows the `.source()` implementation to propagate the correct
    // lifetime.
    fn ptr_ref(&self) -> Ref<'_> {
        Ref {
            ptr: self.object.get().cast::<MaybeUninit<u8>>(),
            _lifetime: PhantomData,
        }
    }
}

impl<const N: usize> Drop for InlineError<N> {
    fn drop(&mut self) {
        // SAFETY: The constructor invariants of `InlineError` ensure that the vtable method
        // is safe to call.
        //
        // Since the only place where the `drop` function is called is in the implementation
        // of `Drop` for `InlineError`, we are guaranteed that the underlying object is
        // valid.
        unsafe { (self.vtable.drop)(self.object.get().cast::<MaybeUninit<u8>>()) }
    }
}

impl<const N: usize> std::fmt::Display for InlineError<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // SAFETY: The constructor invariants of `InlineError` ensure that the vtable method
        // is safe to call.
        unsafe { (self.vtable.display)(self.object.get().cast::<MaybeUninit<u8>>(), f) }
    }
}

impl<const N: usize> std::fmt::Debug for InlineError<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InlineError<{}> {{ object: ", N)?;
        // SAFETY: The constructor invariants of `InlineError` ensure that the vtable method
        // is safe to call.
        unsafe { (self.vtable.debug)(self.object.get().cast::<MaybeUninit<u8>>(), f) }?;
        write!(f, ", vtable: {:?} }}", self.vtable)
    }
}

impl<const N: usize> std::error::Error for InlineError<N> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // SAFETY: The constructor invariants of `InlineError` ensure that the vtable method
        // is safe to call.
        unsafe { (self.vtable.source)(self.ptr_ref()) }
    }
}

#[derive(Debug)]
struct ErrorVTable {
    debug: unsafe fn(*const MaybeUninit<u8>, &mut std::fmt::Formatter<'_>) -> std::fmt::Result,
    display: unsafe fn(*const MaybeUninit<u8>, &mut std::fmt::Formatter<'_>) -> std::fmt::Result,
    source: unsafe fn(Ref<'_>) -> Option<&(dyn std::error::Error + 'static)>,
    drop: unsafe fn(*mut MaybeUninit<u8>),
}

// SAFETY: `object` must point to a valid object of type `T`.
unsafe fn error_debug<T>(
    object: *const MaybeUninit<u8>,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result
where
    T: std::fmt::Debug,
{
    // SAFETY: Required of caller.
    unsafe { &*object.cast::<T>() }.fmt(f)
}

// SAFETY: `object` must point to a valid object of type `T`.
unsafe fn error_display<T>(
    object: *const MaybeUninit<u8>,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result
where
    T: std::fmt::Display,
{
    // SAFETY: Required of caller.
    unsafe { &*object.cast::<T>() }.fmt(f)
}

// SAFETY: A valid instance of type `T` must be stored in `object` beginning at the start
// of the slice. Note that this implies that `std::mem::size_of::<T>() <= object.len()` and
// that the start of the slice is properly aligned.
unsafe fn error_source<T>(object: Ref<'_>) -> Option<&(dyn std::error::Error + 'static)>
where
    T: std::error::Error + 'static,
{
    // SAFETY: Required of caller.
    unsafe { &*object.ptr.cast::<T>() }.source()
}

// A pointer with a tagged lifetime.
struct Ref<'a> {
    ptr: *const MaybeUninit<u8>,
    _lifetime: PhantomData<&'a MaybeUninit<u8>>,
}

// SAFETY: `object` must point to a valid object of type `T`. As a side effect, the
// pointed-to object will be dropped.
unsafe fn error_drop<T>(object: *mut MaybeUninit<u8>) {
    // SAFETY: Required of caller.
    unsafe { std::ptr::drop_in_place::<T>(object.cast::<T>()) }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    use thiserror::Error;

    use super::*;

    #[derive(Error, Debug)]
    #[error("error A")]
    struct ErrorA;

    #[derive(Error, Debug)]
    #[error("error B with val {val}")]
    struct ErrorB<Inner: std::error::Error> {
        val: usize,
        #[source]
        source: Inner,
    }

    #[derive(Error, Debug)]
    #[error("error C with message {message}")]
    struct ErrorC<Inner: std::error::Error> {
        message: String,
        /// `thiserror` automatically marks this as the error source.
        source: Inner,
    }

    #[test]
    fn test_formatting() {
        // No Nesting
        let message = format(&ErrorA);
        assert_eq!(message, "error A");

        // One Level of Nesting
        let error = ErrorB {
            val: 10,
            source: ErrorA,
        };

        let expected = "error B with val 10\n    caused by: error A";
        assert_eq!(format(&error), expected);

        // Multiple Levels of Nesting
        let error = ErrorC {
            message: "Hello World".to_string(),
            source: error,
        };
        let expected = "error C with message Hello World\n    \
                        caused by: error B with val 10\n    \
                        caused by: error A";
        assert_eq!(format(&error), expected);
    }

    ///////////
    // Error //
    ///////////

    #[derive(Debug, Error)]
    #[error("zero sized error")]
    struct ZeroSizedError;

    #[derive(Debug, Error)]
    #[error("error with drop: {}", self.0.load(Ordering::Relaxed))]
    struct ErrorWithDrop(Arc<AtomicUsize>);

    impl Drop for ErrorWithDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[derive(Debug, Error)]
    #[error("error with source")]
    struct ErrorWithSource(#[from] ZeroSizedError);

    // This tests (using Miri) that it's safe to contain error types with interior mutability.
    struct ErrorWithInteriorMutability(Mutex<usize>);

    impl std::fmt::Debug for ErrorWithInteriorMutability {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let current = {
                let mut guard = self.0.lock().unwrap();
                let current = *guard;
                *guard += 1;
                current
            };

            write!(f, "{}", current)
        }
    }

    impl std::fmt::Display for ErrorWithInteriorMutability {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let current = {
                let mut guard = self.0.lock().unwrap();
                let current = *guard;
                *guard += 1;
                current
            };

            write!(f, "{}", current)
        }
    }

    impl std::error::Error for ErrorWithInteriorMutability {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            *self.0.lock().unwrap() += 1;
            None
        }
    }

    #[test]
    fn sizes_and_offsets() {
        let ref_size = std::mem::size_of::<&'static ()>();
        let ref_align = std::mem::align_of::<&'static ()>();

        assert_eq!(std::mem::offset_of!(InlineError<0>, object), ref_size);
        assert_eq!(std::mem::offset_of!(InlineError<8>, object), ref_size);
        assert_eq!(std::mem::offset_of!(InlineError<16>, object), ref_size);

        assert_eq!(std::mem::size_of::<InlineError<0>>(), ref_size);
        assert_eq!(std::mem::size_of::<Option<InlineError<0>>>(), ref_size);
        assert_eq!(std::mem::align_of::<InlineError<0>>(), ref_align);
        assert_eq!(std::mem::align_of::<Option<InlineError<0>>>(), ref_align);

        assert_eq!(std::mem::size_of::<InlineError<8>>(), ref_size + 8);
        assert_eq!(std::mem::size_of::<Option<InlineError<8>>>(), ref_size + 8);
        assert_eq!(std::mem::align_of::<InlineError<8>>(), ref_align);
        assert_eq!(std::mem::align_of::<Option<InlineError<8>>>(), ref_align);

        assert_eq!(std::mem::size_of::<InlineError<16>>(), ref_size + 16);
        assert_eq!(
            std::mem::size_of::<Option<InlineError<16>>>(),
            ref_size + 16
        );
        assert_eq!(std::mem::align_of::<InlineError<16>>(), ref_align);
        assert_eq!(std::mem::align_of::<Option<InlineError<16>>>(), ref_align);
    }

    #[test]
    fn inline_error_zst() {
        use std::error::Error;

        let error = InlineError::<0>::new(ZeroSizedError);
        assert_eq!(
            std::mem::size_of_val(&error),
            8,
            "expected 8 bytes for the payload and 0-bytes for the vtable"
        );
        assert_eq!(error.to_string(), "zero sized error");

        let debug = format!("{:?}", error);
        assert!(
            debug.starts_with(&format!("InlineError<0> {{ object: {:?}", ZeroSizedError)),
            "debug message: {}",
            debug
        );

        assert!(error.source().is_none());

        // Move it into a box. This is mainly a Miri tests.
        let _ = Box::new(error);
    }

    #[test]
    fn inline_error_with_drop() {
        use std::error::Error;

        let count = Arc::new(AtomicUsize::new(10));
        let mut error = InlineError::<8>::new(ErrorWithDrop(count.clone()));
        assert_eq!(
            std::mem::size_of_val(&error),
            16,
            "expected 8 bytes for the payload and 8-bytes for the vtable"
        );
        assert_eq!(error.to_string(), "error with drop: 10");
        assert!(error.source().is_none());

        // Move it into a box. This is mainly a Miri tests.
        error = InlineError::new(ZeroSizedError);
        assert_eq!(error.to_string(), "zero sized error");

        assert_eq!(count.load(Ordering::Relaxed), 11, "failed to run \"drop\"");
    }

    #[test]
    fn inline_error_with_interior_mutability() {
        use std::error::Error;

        let error = InlineError::<16>::new(ErrorWithInteriorMutability(Mutex::new(0)));
        assert_eq!(
            std::mem::size_of_val(&error),
            24,
            "expected 16 bytes for the payload and 8-bytes for the vtable"
        );
        assert_eq!(error.to_string(), "0");
        let debug = format!("{:?}", error);
        assert!(debug.contains("object: 1"), "got {}", debug);
        assert_eq!(error.to_string(), "2");

        let debug = format!("{:?}", error);
        assert!(debug.contains("object: 3"), "got {}", debug);

        assert!(error.source().is_none());
        assert_eq!(error.to_string(), "5");
    }

    #[test]
    fn inline_error_with_source() {
        use std::error::Error;

        let error = InlineError::<8>::new(ErrorWithSource(ZeroSizedError));
        assert_eq!(
            std::mem::size_of_val(&error),
            16,
            "expected 8 bytes for the payload and 8-bytes for the vtable"
        );
        assert_eq!(error.to_string(), "error with source");
        assert_eq!(error.source().unwrap().to_string(), "zero sized error");

        // Move it into a box. This is mainly a Miri tests.
        let _ = Box::new(error);
    }
}
