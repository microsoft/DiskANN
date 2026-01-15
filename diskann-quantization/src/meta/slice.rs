/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use thiserror::Error;

use crate::{
    alloc::{AllocatorCore, AllocatorError, Poly},
    num::PowerOfTwo,
    ownership::{Mut, Owned, Ref},
};

/// A wrapper for a traditional Rust slice that provides the addition of arbitrary metadata.
///
/// # Examples
///
/// The `Slice` has several named variants that should be used instead of `Slice` directly:
/// * [`PolySlice`]: An owning, independently allocated `Slice`.
/// * [`SliceMut`]: A mutable, reference-like type.
/// * [`SliceRef`]: A const, reference-like type.
///
/// ```
/// use diskann_quantization::{
///     alloc::GlobalAllocator,
///     meta::slice,
///     bits::Unsigned,
/// };
///
/// use diskann_utils::{Reborrow, ReborrowMut};
///
/// #[derive(Debug, Default, Clone, Copy, PartialEq)]
/// struct Metadata {
///     value: f32,
/// }
///
/// // Create a new heap-allocated Vector for 4-bit compressions capable of
/// // holding 3 elements.
/// //
/// // In this case, the associated m
/// let mut v = slice::PolySlice::new_in(3, GlobalAllocator).unwrap();
///
/// // We can inspect the underlying bitslice.
/// let data = v.vector();
/// assert_eq!(&data, &[0, 0, 0]);
/// assert_eq!(*v.meta(), Metadata::default(), "expected default metadata value");
///
/// // If we want, we can mutably borrow the bitslice and mutate its components.
/// let mut data = v.vector_mut();
/// assert_eq!(data.len(), 3);
/// data[0] = 1;
/// data[1] = 2;
/// data[2] = 3;
///
/// // Setting the underlying compensation will be visible in the original allocation.
/// *v.meta_mut() = Metadata { value: 10.5 };
///
/// // Check that the changes are visible.
/// assert_eq!(v.meta().value, 10.5);
/// assert_eq!(&v.vector(), &[1, 2, 3]);
/// ```
///
/// ## Constructing a `SliceMut` From Components
///
/// The following example shows how to assemble a `SliceMut` from raw parts.
/// ```
/// use diskann_quantization::meta::slice;
///
/// // For exposition purposes, we will use a slice of `u8` and `f32` as the metadata.
/// let mut data = vec![0u8; 4];
/// let mut metadata: f32 = 0.0;
/// {
///     let mut v = slice::SliceMut::new(data.as_mut_slice(), &mut metadata);
///
///     // Through `v`, we can set all the components in `slice` and the compensation.
///     *v.meta_mut() = 123.4;
///     let mut data = v.vector_mut();
///     data[0] = 1;
///     data[1] = 2;
///     data[2] = 3;
///     data[3] = 4;
/// }
///
/// // Now we can check that the changes made internally are visible.
/// assert_eq!(&data, &[1, 2, 3, 4]);
/// assert_eq!(metadata, 123.4);
/// ```
///
/// ## Canonical Layout
///
/// When the slice element type `T` and metadata type `M` are both
/// [`bytemuck::Pod`](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html), [`SliceRef`]
/// and [`SliceMut`] support layout canonicalization, where a raw slice can be used as the
/// backing store for such vectors, enabling inline storage.
///
/// The layout is specified by:
///
/// * A base alignment of the maximum alignments of `T` and `M`.
/// * The first `M` bytes contain the metadata.
/// * Padding if necessary to reach the alignment of `T`.
/// * The values of type `T` stored contiguously.
///
/// The canonical layout needs the following properties:
///
/// * `T: bytemuck::Pod` and `M: bytemuck::Pod: For safely storing and retrieving.
/// * The length for a vector with `N` dimensions must be equal to the value returned
///   from [`SliceRef::canonical_bytes`].
/// * The **alignment** of the base pointer must be equal to [`SliceRef::canonical_align()`].
///
/// The following functions can be used to construct slices from raw slices:
///
/// * [`SliceRef::from_canonical`]
/// * [`SliceMut::from_canonical_mut`]
///
/// An example is shown below.
/// ```rust
/// use diskann_quantization::{
///     alloc::{AlignedAllocator, Poly},
///     meta::slice,
///     num::PowerOfTwo,
/// };
///
/// let dim = 3;
///
/// // Since we don't control the alignment of the returned pointer, we need to oversize it.
/// let bytes = slice::SliceRef::<u16, f32>::canonical_bytes(dim);
/// let align = slice::SliceRef::<u16, f32>::canonical_align();
/// let mut data = Poly::broadcast(
///     0u8,
///     bytes,
///     AlignedAllocator::new(align)
/// ).unwrap();
///
/// // Construct a mutable compensated vector over the slice.
/// let mut v = slice::SliceMut::<u16, f32>::from_canonical_mut(&mut data, dim).unwrap();
/// *v.meta_mut() = 1.0;
/// v.vector_mut().copy_from_slice(&[1, 2, 3]);
///
/// // Reconstruct a constant CompensatedVector.
/// let cv = slice::SliceRef::<u16, f32>::from_canonical(&data, dim).unwrap();
/// assert_eq!(*cv.meta(), 1.0);
/// assert_eq!(&cv.vector(), &[1, 2, 3]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Slice<T, M> {
    slice: T,
    meta: M,
}

// Use the maximum alignment of `T` and `M` to ensure that no runtime padding is needed.
//
// For example, if `T` had a stricter alignment than `M` and we required an alignment of
// `M`, then the number of padding bytes necessary would depend on the runtime alignment
// of `M`, which is pretty useless for a storage format.
const fn canonical_align<T, M>() -> PowerOfTwo {
    let m_align = PowerOfTwo::alignment_of::<M>();
    let t_align = PowerOfTwo::alignment_of::<T>();

    // Poor man's `const`-compatible `max`.
    if m_align.raw() > t_align.raw() {
        m_align
    } else {
        t_align
    }
}

// The number of bytes required for the metadata prefix. This will consist of the bytes
// required for `M` as well as any padding to obtain an alignment of `T`.
//
// If `M` is a zero-sized type, then the return value is zero. This works because the base
// alignment is at least the alignment of `T`, so no padding is necessary.
const fn canonical_metadata_bytes<T, M>() -> usize {
    let m_size = std::mem::size_of::<M>();
    if m_size == 0 {
        0
    } else {
        m_size.next_multiple_of(std::mem::align_of::<T>())
    }
}

// A simple computation consisting of the bytes for the metadata, followed by the bytes
// needed for the slice itself.
const fn canonical_bytes<T, M>(count: usize) -> usize {
    canonical_metadata_bytes::<T, M>() + std::mem::size_of::<T>() * count
}

impl<T, M> Slice<T, M> {
    /// Construct a new `Slice` over the components.
    pub fn new<U>(slice: T, meta: U) -> Self
    where
        U: Into<M>,
    {
        Self {
            slice,
            meta: meta.into(),
        }
    }

    /// Return the metadata value for this vector.
    pub fn meta(&self) -> &M::Target
    where
        M: Deref,
    {
        &self.meta
    }

    /// Get a mutable reference to the metadata component.
    pub fn meta_mut(&mut self) -> &mut M::Target
    where
        M: DerefMut,
    {
        &mut self.meta
    }
}

impl<T, M, U, V> Slice<T, M>
where
    T: Deref<Target = [U]>,
    M: Deref<Target = V>,
{
    /// Return the number of dimensions of in the slice
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Return whether or not the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Borrow the data slice.
    pub fn vector(&self) -> &[U] {
        &self.slice
    }

    /// Borrow the integer compressed vector.
    pub fn vector_mut(&mut self) -> &mut [U]
    where
        T: DerefMut,
    {
        &mut self.slice
    }

    /// Return the necessary alignment for the base pointer required for
    /// [`SliceRef::from_canonical`] and [`SliceMut::from_canonical_mut`].
    ///
    /// The return value is guaranteed to be a power of two.
    pub const fn canonical_align() -> PowerOfTwo {
        canonical_align::<U, V>()
    }

    /// Return the number of bytes required to store `count` elements plus metadata in a
    /// canonical layout.
    ///
    /// See: [`SliceRef::from_canonical`], [`SliceMut::from_canonical_mut`].
    pub const fn canonical_bytes(count: usize) -> usize {
        canonical_bytes::<U, V>(count)
    }
}

impl<T, A, M> Slice<Poly<[T], A>, Owned<M>>
where
    A: AllocatorCore,
    T: Default,
    M: Default,
{
    /// Create a new owned `VectorBase` with its metadata default initialized.
    pub fn new_in(len: usize, allocator: A) -> Result<Self, AllocatorError> {
        Ok(Self {
            slice: Poly::from_iter((0..len).map(|_| T::default()), allocator)?,
            meta: Owned::default(),
        })
    }
}

/// A reference to a slice and associated metadata.
pub type SliceRef<'a, T, M> = Slice<&'a [T], Ref<'a, M>>;

/// A mutable reference to a slice and associated metadata.
pub type SliceMut<'a, T, M> = Slice<&'a mut [T], Mut<'a, M>>;

/// An owning slice and associated metadata.
pub type PolySlice<T, M, A> = Slice<Poly<[T], A>, Owned<M>>;

//////////////////////
// Canonical Layout //
//////////////////////

#[derive(Debug, Error, PartialEq, Clone, Copy)]
pub enum NotCanonical {
    #[error("expected a slice length of {0} bytes but instead got {1} bytes")]
    WrongLength(usize, usize),
    #[error("expected a base pointer alignment of at least {0}")]
    NotAligned(usize),
}

impl<'a, T, M> SliceRef<'a, T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    /// Construct an instance of `Self` viewing `data` as the canonical layout for a vector.
    /// The canonical layout is as follows:
    ///
    /// * `std::mem::size_of::<T>().max(std::mem::size_of::<M>())` for the metadata.
    /// * Necessary additional padding to achieve the alignment requirements for `T`.
    /// * `std::mem::size_of::<T>() * dim` for the slice.
    ///
    /// Returns an error if:
    ///
    /// * `data` is not aligned to `Self::canonical_align()`.
    /// * `data.len() != `Self::canonical_bytes(dim)`.
    pub fn from_canonical(data: &'a [u8], dim: usize) -> Result<Self, NotCanonical> {
        let expected_align = Self::canonical_align().raw();
        let expected_len = Self::canonical_bytes(dim);

        if !(data.as_ptr() as usize).is_multiple_of(expected_align) {
            Err(NotCanonical::NotAligned(expected_align))
        } else if data.len() != expected_len {
            Err(NotCanonical::WrongLength(expected_len, data.len()))
        } else {
            // SAFETY: We have checked both the length and alignment of `data`.
            Ok(unsafe { Self::from_canonical_unchecked(data, dim) })
        }
    }

    /// Construct a `VectorRef` from the raw data.
    ///
    /// # Safety
    ///
    /// * `data.as_ptr()` must be aligned to `Self::canonical_align()`.
    /// * `data.len()` must be equal to `Self::canonical_bytes(dim)`.
    ///
    /// This invariant is checked in debug builds and will panic if not satisfied.
    pub unsafe fn from_canonical_unchecked(data: &'a [u8], dim: usize) -> Self {
        debug_assert_eq!(data.len(), Self::canonical_bytes(dim));
        let offset = canonical_metadata_bytes::<T, M>();

        // SAFETY: The length pre-condition of this function implies that the offset region
        // `[offset, offset + size_of::<T>() * dim]` is valid for reading.
        //
        // Additionally, the alignment requirment of the base pointer ensures that after
        // applying `offset`, we still have proper alignment for `T`.
        //
        // The `bytemuck::Pod` bound ensures we don't have malformed types after the type cast.
        let slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().add(offset).cast::<T>(), dim) };

        // SAFETY: The pointer is valid and non-null because `data` is a slice, its length
        // must be at least `std::mem::size_of::<M>()` (from the length precondition for
        // this function).
        //
        // The alignemnt pre-condition ensures that the pointer is suitable aligned.
        //
        // THe `bytemuck::Pod` bound ensures that the resulting type is valid.
        let meta =
            unsafe { Ref::new(NonNull::new_unchecked(data.as_ptr().cast_mut()).cast::<M>()) };
        Self { slice, meta }
    }
}

impl<'a, T, M> SliceMut<'a, T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    /// Construct an instance of `Self` viewing `data` as the canonical layout for a vector.
    /// The canonical layout is as follows:
    ///
    /// * `std::mem::size_of::<T>().max(std::mem::size_of::<M>())` for the metadata.
    /// * Necessary additional padding to achieve the alignment requirements for `T`.
    /// * `std::mem::size_of::<T>() * dim` for the slice.
    ///
    /// Returns an error if:
    ///
    /// * `data` is not aligned to `Self::canonical_align()`.
    /// * `data.len() != `Self::canonical_bytes(dim)`.
    pub fn from_canonical_mut(data: &'a mut [u8], dim: usize) -> Result<Self, NotCanonical> {
        let expected_align = Self::canonical_align().raw();
        let expected_len = Self::canonical_bytes(dim);

        if !(data.as_ptr() as usize).is_multiple_of(expected_align) {
            return Err(NotCanonical::NotAligned(expected_align));
        } else if data.len() != expected_len {
            return Err(NotCanonical::WrongLength(expected_len, data.len()));
        }

        let offset = canonical_metadata_bytes::<T, M>();

        // SAFETY: `offset < expected_len` and `data.len() == expected_len`, so `offset`
        // is a valid interior offset for `data`.
        let (meta, slice) = unsafe { data.split_at_mut_unchecked(offset) };

        // SAFETY: `data.as_ptr()` when offset by `offset` will have an alignment suitable
        // for type `T`.
        //
        // We have checked that `data.len() == expected_len`, which implies that the region
        // of memory between `offset` and `data.len()` covers exactly `size_of::<T>() * dim`
        // bytes.
        //
        // The `bytemuck::Pod` requirement on `T` ensures the resulting values are valid.
        let slice = unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<T>(), dim) };

        // SAFETY: `data.as_ptr()` has an alignemnt of at least that required by `M`.
        //
        // Since `data` is a slice, its base pointer is `NonNull`.
        //
        // The `bytemuck::Pod` requirement ensures we have a valid instance.
        let meta = unsafe { Mut::new(NonNull::new_unchecked(meta.as_mut_ptr()).cast::<M>()) };

        Ok(Self { slice, meta })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;
    use crate::{
        alloc::{AlignedAllocator, GlobalAllocator},
        num::PowerOfTwo,
    };

    ////////////////////////
    // Compensated Vector //
    ////////////////////////

    #[derive(Default, Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    struct Metadata {
        a: u32,
        b: u32,
    }

    impl Metadata {
        fn new(a: u32, b: u32) -> Metadata {
            Self { a, b }
        }
    }

    #[test]
    fn test_vector() {
        let len = 20;
        let mut base = PolySlice::<f32, Metadata, _>::new_in(len, GlobalAllocator).unwrap();

        assert_eq!(base.len(), len);
        assert_eq!(*base.meta(), Metadata::default());
        assert!(!base.is_empty());

        // Ensure that if we reborrow mutably that changes are visible.
        {
            *base.meta_mut() = Metadata::new(1, 2);
            let v = base.vector_mut();

            assert_eq!(v.len(), len);
            v.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
        }

        // Are the changes visible?
        {
            let expected_metadata = Metadata::new(1, 2);
            assert_eq!(*base.meta(), expected_metadata);
            assert_eq!(base.len(), len);
            let v = base.vector();
            v.iter().enumerate().for_each(|(i, v)| {
                assert_eq!(*v, i as f32);
            })
        }
    }

    //////////////////////
    // Canonicalization //
    //////////////////////

    // A test zero-sized type with non-strict alignment.
    #[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    struct Zst;

    #[expect(clippy::infallible_try_from)]
    impl TryFrom<usize> for Zst {
        type Error = std::convert::Infallible;
        fn try_from(_: usize) -> Result<Self, Self::Error> {
            Ok(Self)
        }
    }

    // A test zero-sized type with a strict alignment.
    #[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C, align(16))]
    struct ZstAligned;

    #[expect(clippy::infallible_try_from)]
    impl TryFrom<usize> for ZstAligned {
        type Error = std::convert::Infallible;
        fn try_from(_: usize) -> Result<Self, Self::Error> {
            Ok(Self)
        }
    }

    fn check_canonicalization<T, M>(
        dim: usize,
        align: usize,
        slope: usize,
        offset: usize,
        ntrials: usize,
        rng: &mut StdRng,
    ) where
        T: bytemuck::Pod + TryFrom<usize, Error: Debug> + Debug + PartialEq,
        M: bytemuck::Pod + TryFrom<usize, Error: Debug> + Debug + PartialEq,
    {
        let bytes = SliceRef::<T, M>::canonical_bytes(dim);

        assert_eq!(
            bytes,
            slope * dim + offset,
            "computed bytes did not match the expected formula"
        );

        let expected_align = std::mem::align_of::<T>().max(std::mem::align_of::<M>());
        assert_eq!(SliceRef::<T, M>::canonical_align().raw(), align);
        assert_eq!(SliceRef::<T, M>::canonical_align().raw(), expected_align);

        let mut buffer = Poly::broadcast(
            0u8,
            bytes + expected_align,
            AlignedAllocator::new(PowerOfTwo::new(expected_align).unwrap()),
        )
        .unwrap();

        // Expected metadata and vector encoding.
        let mut expected = vec![usize::default(); dim];
        let dist = Uniform::new(0, 255).unwrap();

        for _ in 0..ntrials {
            let m: usize = dist.sample(rng);
            expected.iter_mut().for_each(|i| *i = dist.sample(rng));
            {
                let mut v =
                    SliceMut::<T, M>::from_canonical_mut(&mut buffer[..bytes], dim).unwrap();
                *v.meta_mut() = m.try_into().unwrap();

                assert_eq!(v.vector().len(), dim);
                assert_eq!(v.vector_mut().len(), dim);
                std::iter::zip(v.vector_mut().iter_mut(), expected.iter_mut()).for_each(
                    |(v, e)| {
                        *v = (*e).try_into().unwrap();
                    },
                );
            }

            // Make sure the reconstruction is valid.
            {
                let v = SliceRef::<T, M>::from_canonical(&buffer[..bytes], dim).unwrap();
                assert_eq!(*v.meta(), m.try_into().unwrap());

                assert_eq!(v.vector().len(), dim);
                std::iter::zip(v.vector().iter(), expected.iter()).for_each(|(v, e)| {
                    assert_eq!(*v, (*e).try_into().unwrap());
                });
            }
        }

        // Length Errors
        {
            for len in 0..bytes {
                // Too short
                let err =
                    SliceMut::<T, M>::from_canonical_mut(&mut buffer[..len], dim).unwrap_err();
                assert!(matches!(err, NotCanonical::WrongLength(_, _)));

                // Too short
                let err = SliceRef::<T, M>::from_canonical(&buffer[..len], dim).unwrap_err();
                assert!(matches!(err, NotCanonical::WrongLength(_, _)));
            }

            // Too long
            let err =
                SliceMut::<T, M>::from_canonical_mut(&mut buffer[..bytes + 1], dim).unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            let err = SliceRef::<T, M>::from_canonical(&buffer[..bytes + 1], dim).unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));
        }

        // Alignment
        {
            for offset in 1..expected_align {
                let err =
                    SliceMut::<T, M>::from_canonical_mut(&mut buffer[offset..offset + bytes], dim)
                        .unwrap_err();
                assert!(matches!(err, NotCanonical::NotAligned(_)));

                let err = SliceRef::<T, M>::from_canonical(&buffer[offset..offset + bytes], dim)
                    .unwrap_err();
                assert!(matches!(err, NotCanonical::NotAligned(_)));
            }
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const MAX_DIM: usize = 10;
            const TRIALS_PER_DIM: usize = 1;
        } else {
            const MAX_DIM: usize = 256;
            const TRIALS_PER_DIM: usize = 20;
        }
    }

    macro_rules! test_canonical {
        ($name:ident, $M:ty, $T:ty, $align:literal, $slope:literal, $offset:literal, $seed:literal) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);
                for dim in 0..MAX_DIM {
                    check_canonicalization::<$T, $M>(
                        dim,
                        $align,
                        $slope,
                        $offset,
                        TRIALS_PER_DIM,
                        &mut rng,
                    );
                }
            }
        };
    }

    test_canonical!(canonical_u8_u32, u8, u32, 4, 4, 4, 0x60884b7a4ca28f49);
    test_canonical!(canonical_u32_u8, u32, u8, 4, 1, 4, 0x874aa5d8f40ec5ef);
    test_canonical!(canonical_u32_u32, u32, u32, 4, 4, 4, 0x516c550e7be19acc);

    test_canonical!(canonical_zst_u32, Zst, u32, 4, 4, 0, 0x908682ebda7c0fb9);
    test_canonical!(canonical_u32_zst, u32, Zst, 4, 0, 4, 0xf223385881819c1c);

    test_canonical!(
        canonical_zstaligned_u32,
        ZstAligned,
        u32,
        16,
        4,
        0,
        0x1811ee0fd078a173
    );
    test_canonical!(
        canonical_u32_zstaligned,
        u32,
        ZstAligned,
        16,
        0,
        16,
        0x6c9a67b09c0b6c0f
    );
}
