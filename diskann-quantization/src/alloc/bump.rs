/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cell::UnsafeCell,
    ptr::NonNull,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use super::{AlignedAllocator, AllocatorCore, AllocatorError, Poly};
use crate::num::PowerOfTwo;

/// An [`AllocatorCore`] that pre-allocates a large buffer of memory and then satisfies
/// allocation requests from that buffer.
///
/// Note that the memory for allocations made through this allocator and its clones will not
/// be freed until all clones have been dropped.
///
/// Memory allocation through this page is thread safe.
#[derive(Debug, Clone)]
pub struct BumpAllocator {
    inner: Arc<BumpAllocatorInner>,
}

#[derive(Debug)]
struct BumpAllocatorInner {
    buffer: Poly<UnsafeCell<[u8]>, AlignedAllocator>,
    head: AtomicUsize,
}

// SAFETY: Allocation and deallocation are thread-safe, so `BumpAllocatorInner` can be sent
// between threads.
unsafe impl Send for BumpAllocatorInner {}

// SAFETY: Allocation and deallocation are thread-safe, so `BumpAllocatorInner` can be shared
// between threads.
unsafe impl Sync for BumpAllocatorInner {}

// Interior mutation only occurs in a section of code that should not panic. So even if
// we're unwinding around a `&BumpAllocatorInner`, we should not break invariants.
impl std::panic::RefUnwindSafe for BumpAllocatorInner {}

impl BumpAllocator {
    /// Construct a new [`BumpAllocator`] with room for `capacity` bytes. The base pointer
    /// for the allocator will be aligned to at least `alignment` bytes.
    ///
    /// Returns an error if `alignment` is not a power of two or if an error occurs during
    /// memory allocation.
    pub fn new(capacity: usize, alignment: PowerOfTwo) -> Result<Self, AllocatorError> {
        let allocator = AlignedAllocator::new(alignment);
        let buffer = Poly::<[u8], _>::new_uninit_slice(capacity.max(1), allocator)?;
        let (ptr, alloc) = Poly::into_raw(buffer);

        // SAFETY: The layout for `UnsafeCell<T>` is the same as the layout for `T`, so
        // casting from `[u8]` to `UnsafeCell<[u8]>` is safe.
        //
        // It is safe to cast away `MaybeUninit` because `u8` has not padding and is valid
        // for all bit patterns.
        //
        // Finally, it is safe to construct `NonNull` because `ptr` is already `NonNull`.
        let buffer = unsafe {
            Poly::from_raw(
                NonNull::new_unchecked(ptr.as_ptr() as *mut UnsafeCell<[u8]>),
                alloc,
            )
        };

        Ok(Self {
            inner: std::sync::Arc::new(BumpAllocatorInner {
                buffer,
                head: Default::default(),
            }),
        })
    }

    /// Return the capacity this allocator was created with.
    pub fn capacity(&self) -> usize {
        self.inner.buffer.get().len()
    }

    /// Return a pointer to the base of the buffer behind this allocator.
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.buffer.get().cast::<u8>().cast_const()
    }
}

/// Given a `base` address and a current `offset` from that base, compute a `new_offset`
/// such that the range spanned by `[base + offset, base + new_offset)` has sufficient room
/// to fulfill the allocation request in `layout`.
fn next(base: usize, offset: usize, layout: std::alloc::Layout) -> Option<usize> {
    let p = PowerOfTwo::from_align(&layout);
    p.arg_checked_next_multiple_of(base + offset)
        .map(|x| x - base)
        .and_then(|x| x.checked_add(layout.size()))
}

// SAFETY: The implementation of `BumpAllocator` ensures that upon success
//
// 1. Allocations provided from the buffer are properly aligned, regardless of the current
//    state of the `head` pointer.
// 2. The allocation is always of the requested size.
//
// If both of these cannot be satisfied without running off the end of the page, an error
// is returned.
unsafe impl AllocatorCore for BumpAllocator {
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocatorError> {
        // Get the base pointer as an integer for alignment calculations.
        let base = self.as_ptr() as usize;

        // Return the new head that ensures we have room for the allocation defined by
        // `layout` starting from the current head.
        let compute_next = |head: usize| -> Result<usize, AllocatorError> {
            let new_head = next(base, head, layout).ok_or(AllocatorError)?;
            if new_head > self.capacity() {
                Err(AllocatorError)
            } else {
                Ok(new_head)
            }
        };

        // Spin until we successfully update the `head` pointer. Successful update indicates
        // that we own the span of memory between `old_head` and `new_head` and can provide
        // that for the allocation.
        let mut old_head = self.inner.head.load(Ordering::Relaxed);
        let mut new_head = compute_next(old_head)?;
        loop {
            match self.inner.head.compare_exchange(
                old_head,
                new_head,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(h) => {
                    old_head = h;
                    new_head = compute_next(h)?;
                }
            }
        }

        // SAFETY: `old_head` is guaranteed to be within the range of the buffer allocation.
        let ptr = unsafe { self.as_ptr().add(old_head) };

        // SAFETY: The computation of `new_head` ensures that we have space to do this
        // alignment.
        let ptr =
            unsafe { ptr.add(PowerOfTwo::from_align(&layout).arg_align_offset(ptr as usize)) };

        // SAFETY: The computation of `new_head` ensures that we have space to construct
        // a slice of this length after alignment.
        NonNull::new(std::ptr::slice_from_raw_parts_mut(
            ptr.cast_mut(),
            layout.size(),
        ))
        .ok_or(AllocatorError)
    }

    // No work to do in deallocation - dropping the reference count for the bump allocator
    // is sufficient.
    unsafe fn deallocate(&self, _ptr: NonNull<[u8]>, _layout: std::alloc::Layout) {}
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;
    use crate::alloc::Poly;

    ///////////////////
    // BumpAllocator //
    ///////////////////

    #[test]
    fn test_bump_allocator() {
        let allocator = BumpAllocator::new(128, PowerOfTwo::new(1).unwrap()).unwrap();
        let mut a = Poly::new(0usize, allocator.clone()).unwrap();
        let mut b = Poly::new(1usize, allocator.clone()).unwrap();
        let mut c = Poly::new(2usize, allocator.clone()).unwrap();

        *b = 5;
        *a = 10;
        *c = 87;
        *a = 20;

        assert_eq!(*b, 5);
    }

    #[test]
    fn poly_new_with_allocates_first() {
        let allocator = BumpAllocator::new(128, PowerOfTwo::new(64).unwrap()).unwrap();

        struct Nested {
            inner: Poly<[usize], BumpAllocator>,
            value: f32,
        }

        let poly = Poly::<Nested, _>::new_with(
            |a| -> Result<_, AllocatorError> {
                Ok(Nested {
                    inner: Poly::from_iter(0..10, a)?,
                    value: 10.0,
                })
            },
            allocator.clone(),
        )
        .unwrap();

        // Ensure that `poly` was initialized properly.
        assert!(poly.inner.iter().enumerate().all(|(i, v)| i == *v));
        assert_eq!(poly.value, 10.0);

        // Ensure that `poly` was allocated before `poly.inner`.
        let base = allocator.as_ptr();
        assert_eq!(base, Poly::as_ptr(&poly).cast::<u8>());
        assert_eq!(
            base.wrapping_add(32),
            Poly::as_ptr(&poly.inner).cast::<u8>()
        );
    }

    fn values<T: Default>(alloc: BumpAllocator, seed: u64) {
        let mut buf = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);

        let index_dist = Uniform::new(0, 10).unwrap();

        while let Ok(poly) = Poly::new(T::default(), alloc.clone()) {
            buf.push(poly);
            if buf.len() == 10 {
                buf.remove(index_dist.sample(&mut rng));
            }
        }
    }

    fn slices<T: Default>(alloc: BumpAllocator, seed: u64) {
        let mut buf = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);

        let dist = Uniform::new(0, 10).unwrap();

        while let Ok(poly) = Poly::from_iter(
            (0..dist.sample(&mut rng)).map(|_| T::default()),
            alloc.clone(),
        ) {
            buf.push(poly);
            if buf.len() == 10 {
                buf.remove(dist.sample(&mut rng));
            }
        }
    }

    fn stress_test_impl() {
        let alloc = BumpAllocator::new(4096, PowerOfTwo::new(1).unwrap()).unwrap();

        let c0 = alloc.clone();
        let c1 = alloc.clone();
        let c2 = alloc.clone();
        let c3 = alloc.clone();
        let handles = [
            std::thread::spawn(move || values::<u8>(c0, 0xa7c0b68e3ece66f7)),
            std::thread::spawn(move || values::<String>(c1, 0x72f0fbcaaefbc884)),
            std::thread::spawn(move || slices::<u16>(c2, 0x447a846ceb3eeda9)),
            std::thread::spawn(move || slices::<String>(c3, 0xd34c7cbedaf165ad)),
        ];

        for h in handles.into_iter() {
            h.join().unwrap();
        }
    }

    #[test]
    fn stress_test() {
        let trials = if cfg!(miri) { 3 } else { 100 };

        for _ in 0..trials {
            stress_test_impl();
        }
    }
}
