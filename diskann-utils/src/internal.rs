/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ptr::NonNull;

pub(crate) fn slice_to_nonnull<T>(s: &[T]) -> NonNull<T> {
    // SAFETY: slices are guaranteed to have non-null base pointers.
    unsafe { std::ptr::NonNull::new_unchecked(s.as_ptr().cast_mut()) }
}

pub(crate) fn mut_slice_to_nonnull<T>(s: &mut [T]) -> NonNull<T> {
    // SAFETY: slices are guaranteed to have non-null base pointers.
    unsafe { std::ptr::NonNull::new_unchecked(s.as_mut_ptr()) }
}

pub(crate) fn box_to_nonnull<T>(b: Box<[T]>) -> NonNull<T> {
    let ptr = Box::into_raw(b).cast::<T>();
    // SAFETY: boxes are guaranteed to have non-null base poihnters.
    unsafe { NonNull::new_unchecked(ptr) }
}

pub(crate) unsafe fn nonnull_to_box<T>(p: NonNull<T>, len: usize) -> Box<[T]> {
    let slice = std::ptr::slice_from_raw_parts_mut(p.as_ptr(), len);
    unsafe { Box::from_raw(slice) }
}
