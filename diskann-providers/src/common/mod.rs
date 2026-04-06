/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub use diskann_quantization::alloc::{AlignedSlice, aligned_slice};

/// Creates a new [`AlignedSlice`] with the given capacity and alignment.
/// The allocated memory is set to `T::default()`.
///
/// # Error
///
/// Returns an `IndexError` if the alignment is not a power of two or if the layout is invalid.
pub fn aligned_alloc<T: Default>(
    capacity: usize,
    alignment: usize,
) -> diskann::ANNResult<AlignedSlice<T>> {
    use diskann::ANNError;
    use diskann_quantization::num::PowerOfTwo;
    let alignment = PowerOfTwo::new(alignment).map_err(ANNError::log_index_error)?;
    aligned_slice(capacity, alignment).map_err(ANNError::log_index_error)
}

mod minmax_repr;
pub use minmax_repr::{MinMax4, MinMax8, MinMaxElement};

mod ignore_lock_poison;
pub use ignore_lock_poison::IgnoreLockPoison;
