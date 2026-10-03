/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) fn slice_to_nonnull<T>(s: &[T]) -> std::ptr::NonNull<T> {
    // SAFETY: slices are guaranteed to have non-null base pointers.
    unsafe { std::ptr::NonNull::new_unchecked(s.as_ptr().cast_mut()) }
}
