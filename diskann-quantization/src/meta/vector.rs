/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ptr::NonNull;

use diskann_utils::{Reborrow, ReborrowMut};
use thiserror::Error;

use crate::{
    alloc::{AllocatorCore, AllocatorError, GlobalAllocator, Poly},
    bits::{
        AsMutPtr, AsPtr, BitSlice, BitSliceBase, Dense, MutBitSlice, MutSlicePtr,
        PermutationStrategy, Representation, SlicePtr,
    },
    ownership::{CopyMut, CopyRef, Mut, Owned, Ref},
};

/// A wrapper for [`BitSliceBase`] that provides the addition of arbitrary metadata.
///
/// # Examples
///
/// The `VectorBase` has several named variants that are commonly used:
/// * [`Vector`]: An owning, independently allocated `VectorBase`.
/// * [`VectorMut`]: A mutable, reference-like type to a `VectorBase`.
/// * [`VectorRef`]: A const, reference-like type to a `VectorBase`.
///
/// ```
/// use diskann_quantization::{
///     meta::{Vector, VectorMut, VectorRef},
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
/// let mut v = Vector::<4, Unsigned, Metadata>::new_boxed(3);
///
/// // We can inspect the underlying bitslice.
/// let bitslice = v.vector();
/// assert_eq!(bitslice.get(0).unwrap(), 0);
/// assert_eq!(bitslice.get(1).unwrap(), 0);
/// assert_eq!(v.meta(), Metadata::default(), "expected default metadata value");
///
/// // If we want, we can mutably borrow the bitslice and mutate its components.
/// let mut bitslice = v.vector_mut();
/// bitslice.set(0, 1).unwrap();
/// bitslice.set(1, 2).unwrap();
/// bitslice.set(2, 3).unwrap();
///
/// assert!(bitslice.set(3, 4).is_err(), "out-of-bounds access");
///
/// // Get the underlying pointer for comparison.
/// let ptr = bitslice.as_ptr();
///
/// // Vectors can be converted to a generalized reference.
/// let mut v_ref = v.reborrow_mut();
///
/// // The generalized reference preserves the underlying pointer.
/// assert_eq!(v_ref.vector().as_ptr(), ptr);
/// let mut bitslice = v_ref.vector_mut();
/// bitslice.set(0, 10).unwrap();
///
/// // Setting the underlying compensation will be visible in the original allocation.
/// v_ref.set_meta(Metadata { value: 10.5 });
///
/// // Check that the changes are visible.
/// assert_eq!(v.meta().value, 10.5);
/// assert_eq!(v.vector().get(0).unwrap(), 10);
///
/// // Finally, the immutable ref also maintains pointer compatibility.
/// let v_ref = v.reborrow();
/// assert_eq!(v_ref.vector().as_ptr(), ptr);
/// ```
///
/// ## Constructing a `VectorMut` From Components
///
/// The following example shows how to assemble a `VectorMut` from raw memory.
/// ```
/// use diskann_quantization::{bits::{Unsigned, MutBitSlice}, meta::VectorMut};
///
/// // Start with 2 bytes of memory. We will impose a 4-bit scalar quantization on top of
/// // these 2 bytes.
/// let mut data = vec![0u8; 2];
/// let mut metadata: f32 = 0.0;
/// {
///     // First, we need to construct a bit-slice over the data.
///     // This will check that it is sized properly for 4, 4-bit values.
///     let mut slice = MutBitSlice::<4, Unsigned>::new(data.as_mut_slice(), 4).unwrap();
///
///     // Next, we construct the `VectorMut`.
///     let mut v = VectorMut::new(slice, &mut metadata);
///
///     // Through `v`, we can set all the components in `slice` and the compensation.
///     v.set_meta(123.4);
///     let mut from_v = v.vector_mut();
///     from_v.set(0, 1).unwrap();
///     from_v.set(1, 2).unwrap();
///     from_v.set(2, 3).unwrap();
///     from_v.set(3, 4).unwrap();
/// }
///
/// // Now we can check that the changes made internally are visible.
/// assert_eq!(&data, &[0x21, 0x43]);
/// assert_eq!(metadata, 123.4);
/// ```
///
/// ## Canonical Layout
///
/// When the metadata type `T` is
/// [`bytemuck::Pod`](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html), [`VectorRef`]
/// and [`VectorMut`] support layout canonicalization, where a raw slice can be used as the
/// backing store for such vectors, enabling inline storage.
///
/// There are two supported schems for the canonical layout, depending on whether the
/// metadata is located at the beginning of the slice or at the end of the slice.
///
/// If the metadata is at the front, then the layout consists of a slice `&[u8]` where the
/// first `std::mem::size_of::<T>()` bytes are the metadata and the remainder compose the
/// [`BitSlice`] codes.
///
/// If the metadata is at the back, , then the layout consists of a slice `&[u8]` where the
/// last `std::mem::size_of::<T>()` bytes are the metadata and the prefix is the
/// [`BitSlice`] codes.
///
/// The canonical layout needs the following properties:
///
/// * `T: bytemuck::Pod`: For safely storing and retrieving.
/// * The length for a vector with `N` dimensions must be equal to the value returne from
///   [`Vector::canonical_bytes`].
///
/// The following functions can be used to construct [`VectorBase`]s from raw slices:
///
/// * [`VectorRef::from_canonical_front`]
/// * [`VectorRef::from_canonical_back`]
/// * [`VectorMut::from_canonical_front_mut`]
/// * [`VectorMut::from_canonical_back_mut`]
///
/// An example is shown below.
/// ```rust
/// use diskann_quantization::{bits, meta::{Vector, VectorRef, VectorMut}};
///
/// type CVRef<'a, const NBITS: usize> = VectorRef<'a, NBITS, bits::Unsigned, f32>;
/// type MutCV<'a, const NBITS: usize> = VectorMut<'a, NBITS, bits::Unsigned, f32>;
///
/// let dim = 3;
///
/// // Since we don't control the alignment of the returned pointer, we need to oversize it.
/// let bytes = CVRef::<4>::canonical_bytes(dim);
/// let mut data: Box<[u8]> = (0..bytes).map(|_| u8::default()).collect();
///
/// // Construct a mutable compensated vector over the slice.
/// let mut mut_cv = MutCV::<4>::from_canonical_front_mut(&mut data, dim).unwrap();
/// mut_cv.set_meta(1.0);
/// let mut v = mut_cv.vector_mut();
/// v.set(0, 1).unwrap();
/// v.set(1, 2).unwrap();
/// v.set(2, 3).unwrap();
///
/// // Reconstruct a constant CompensatedVector.
/// let cv = CVRef::<4>::from_canonical_front(&data, dim).unwrap();
/// assert_eq!(cv.meta(), 1.0);
/// let v = cv.vector();
/// assert_eq!(v.get(0).unwrap(), 1);
/// assert_eq!(v.get(1).unwrap(), 2);
/// assert_eq!(v.get(2).unwrap(), 3);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VectorBase<const NBITS: usize, Repr, Ptr, T, Perm = Dense>
where
    Ptr: AsPtr<Type = u8>,
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
{
    bits: BitSliceBase<NBITS, Repr, Ptr, Perm>,
    meta: T,
}

impl<const NBITS: usize, Repr, Ptr, T, Perm> VectorBase<NBITS, Repr, Ptr, T, Perm>
where
    Ptr: AsPtr<Type = u8>,
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
{
    /// Return the number of bytes required for the underlying `BitSlice`.
    pub fn slice_bytes(count: usize) -> usize {
        BitSliceBase::<NBITS, Repr, Ptr, Perm>::bytes_for(count)
    }

    /// Return the number of bytes required for the canonical representation of a
    /// `Vector`.
    ///
    /// See: [`VectorRef::from_canonical_back`], [`VectorMut::from_canonical_back_mut`].
    pub fn canonical_bytes(count: usize) -> usize
    where
        T: CopyRef,
        T::Target: bytemuck::Pod,
    {
        Self::slice_bytes(count) + std::mem::size_of::<T::Target>()
    }

    /// Construct a new `VectorBase` over the bit-slice.
    pub fn new<M>(bits: BitSliceBase<NBITS, Repr, Ptr, Perm>, meta: M) -> Self
    where
        M: Into<T>,
    {
        Self {
            bits,
            meta: meta.into(),
        }
    }

    /// Return the number of dimensions of in the vector.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Return whether or not the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Return the metadata value for this vector.
    pub fn meta(&self) -> T::Target
    where
        T: CopyRef,
    {
        self.meta.copy_ref()
    }

    /// Borrow the integer compressed vector.
    pub fn vector(&self) -> BitSlice<'_, NBITS, Repr, Perm> {
        self.bits.reborrow()
    }

    /// Mutably borrow the integer compressed vector.
    pub fn vector_mut(&mut self) -> MutBitSlice<'_, NBITS, Repr, Perm>
    where
        Ptr: AsMutPtr,
    {
        self.bits.reborrow_mut()
    }

    /// Get a mutable reference to the metadata component.
    ///
    /// In addition to a mutable reference, this also requires `Ptr: AsMutPtr` to prevent
    /// accidental misuse where the `VectorBase` is mutable but the underlying
    /// `BitSlice` is not.
    pub fn set_meta(&mut self, value: T::Target)
    where
        Ptr: AsMutPtr,
        T: CopyMut,
    {
        self.meta.copy_mut(value)
    }
}

impl<const NBITS: usize, Repr, Perm, T>
    VectorBase<NBITS, Repr, Poly<[u8], GlobalAllocator>, Owned<T>, Perm>
where
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    T: Default,
{
    /// Create a new owned `VectorBase` with its metadata default initialized.
    pub fn new_boxed(len: usize) -> Self {
        Self {
            bits: BitSliceBase::new_boxed(len),
            meta: Owned::default(),
        }
    }
}

impl<const NBITS: usize, Repr, Perm, T, A> VectorBase<NBITS, Repr, Poly<[u8], A>, Owned<T>, Perm>
where
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    T: Default,
    A: AllocatorCore,
{
    /// Create a new owned `VectorBase` with its metadata default initialized.
    pub fn new_in(len: usize, allocator: A) -> Result<Self, AllocatorError> {
        Ok(Self {
            bits: BitSliceBase::new_in(len, allocator)?,
            meta: Owned::default(),
        })
    }
}

/// A borrowed `Vector`.
///
/// See: [`VectorBase`].
pub type VectorRef<'a, const NBITS: usize, Repr, T, Perm = Dense> =
    VectorBase<NBITS, Repr, SlicePtr<'a, u8>, Ref<'a, T>, Perm>;

/// A mutably borrowed `Vector`.
///
/// See: [`VectorBase`].
pub type VectorMut<'a, const NBITS: usize, Repr, T, Perm = Dense> =
    VectorBase<NBITS, Repr, MutSlicePtr<'a, u8>, Mut<'a, T>, Perm>;

/// An owning `VectorBase`.
///
/// See: [`VectorBase`].
pub type Vector<const NBITS: usize, Repr, T, Perm = Dense> =
    VectorBase<NBITS, Repr, Poly<[u8], GlobalAllocator>, Owned<T>, Perm>;

/// An owning `VectorBase`.
///
/// See: [`VectorBase`].
pub type PolyVector<const NBITS: usize, Repr, T, Perm, A> =
    VectorBase<NBITS, Repr, Poly<[u8], A>, Owned<T>, Perm>;

// Reborrow
impl<'this, const NBITS: usize, Repr, Ptr, T, Perm> Reborrow<'this>
    for VectorBase<NBITS, Repr, Ptr, T, Perm>
where
    Ptr: AsPtr<Type = u8>,
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    T: CopyRef + Reborrow<'this, Target = Ref<'this, <T as CopyRef>::Target>>,
{
    type Target = VectorRef<'this, NBITS, Repr, <T as CopyRef>::Target, Perm>;

    fn reborrow(&'this self) -> Self::Target {
        Self::Target {
            bits: self.bits.reborrow(),
            meta: self.meta.reborrow(),
        }
    }
}

// ReborrowMut
impl<'this, const NBITS: usize, Repr, Ptr, T, Perm> ReborrowMut<'this>
    for VectorBase<NBITS, Repr, Ptr, T, Perm>
where
    Ptr: AsMutPtr<Type = u8>,
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    T: CopyMut + ReborrowMut<'this, Target = Mut<'this, <T as CopyRef>::Target>>,
{
    type Target = VectorMut<'this, NBITS, Repr, <T as CopyRef>::Target, Perm>;

    fn reborrow_mut(&'this mut self) -> Self::Target {
        Self::Target {
            bits: self.bits.reborrow_mut(),
            meta: self.meta.reborrow_mut(),
        }
    }
}

//////////////////////
// Canonical Layout //
//////////////////////

#[derive(Debug, Error, PartialEq, Clone, Copy)]
pub enum NotCanonical {
    #[error("expected a slice length of {0} bytes but instead got {1} bytes")]
    WrongLength(usize, usize),
}

impl<'a, const NBITS: usize, Repr, T, Perm> VectorRef<'a, NBITS, Repr, T, Perm>
where
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    T: bytemuck::Pod,
{
    /// Construct an instance of `Self` viewing `data` as the canonical layout for a vector.
    /// The canonical layout is as follows:
    ///
    /// * `std::mem::size_of::<T>()` for the metadata coefficient.
    /// * `Self::slice_bytes(dim)` for the underlying bit-slice.
    ///
    /// Returns an error if `data.len() != `Self::canonical_bytes`.
    pub fn from_canonical_front(data: &'a [u8], dim: usize) -> Result<Self, NotCanonical> {
        let expected = Self::canonical_bytes(dim);
        if data.len() != expected {
            Err(NotCanonical::WrongLength(expected, data.len()))
        } else {
            // SAFETY: We have checked both the length and alignment of `data`.
            Ok(unsafe { Self::from_canonical_unchecked(data, dim) })
        }
    }

    /// Construct an instance of `Self` viewing `data` as the canonical layout for a vector.
    /// The back canonical layout is as follows:
    ///
    /// * `Self::slice_bytes(dim)` for the underlying bit-slice.
    /// * `std::mem::size_of::<T>()` for the metadata coefficient.
    ///
    /// Returns an error if `data.len() != `Self::canonical_bytes`.
    pub fn from_canonical_back(data: &'a [u8], dim: usize) -> Result<Self, NotCanonical> {
        let expected = Self::canonical_bytes(dim);
        if data.len() != expected {
            Err(NotCanonical::WrongLength(expected, data.len()))
        } else {
            // SAFETY: We have checked both the length and alignment of `data`.
            Ok(unsafe { Self::from_canonical_back_unchecked(data, dim) })
        }
    }

    /// Construct a `VectorRef` from the raw data.
    ///
    /// # Safety
    ///
    /// * `data.len()` must be equal to `Self::canonical_bytes(dim)`.
    ///
    /// This invariant is checked in debug builds and will panic if not satisfied.
    pub unsafe fn from_canonical_unchecked(data: &'a [u8], dim: usize) -> Self {
        debug_assert_eq!(data.len(), Self::canonical_bytes(dim));

        // SAFETY: `BitSlice` has no alignment requirements, but the length precondition
        // for this function (i.e., `data.len() == Self::canonical_bytes(dim)`) implies
        // that `Self::slice_bytes(dim)` is valid beginning at an offset of
        // `std::mem::size_of::<T>()`.
        let bits =
            unsafe { BitSlice::new_unchecked(data.get_unchecked(std::mem::size_of::<T>()..), dim) };

        // SAFETY: The pointer is valid and non-null because `data` is a slice, its length
        // must be at least `std::mem::size_of::<T>()` (from the length precondition for
        // this function).
        let meta =
            unsafe { Ref::new(NonNull::new_unchecked(data.as_ptr().cast_mut()).cast::<T>()) };
        Self { bits, meta }
    }

    /// Construct a `VectorRef` from the raw data.
    ///
    /// # Safety
    ///
    /// * `data.len()` must be equal to `Self::canonical_bytes(dim)`.
    ///
    /// This invariant is checked in debug builds and will panic if not satisfied.
    pub unsafe fn from_canonical_back_unchecked(data: &'a [u8], dim: usize) -> Self {
        debug_assert_eq!(data.len(), Self::canonical_bytes(dim));
        // SAFETY: The caller asserts that
        // `data.len() == Self::canonical_bytes(dim) >= std::mem::size_of::<T>()`.
        let (data, meta) =
            unsafe { data.split_at_unchecked(data.len() - std::mem::size_of::<T>()) };

        // SAFETY: `BitSlice` has no alignment requirements, but the length precondition
        // for this function (i.e., `data.len() == Self::canonical_bytes(dim)`) implies
        // that `Self::slice_bytes(dim)` is valid beginning at an offset of
        // `std::mem::size_of::<T>()`.
        let bits = unsafe { BitSlice::new_unchecked(data, dim) };

        // SAFETY: The pointer is valid and non-null because `data` is a slice, its length
        // must be at least `std::mem::size_of::<T>()` (from the length precondition for
        // this function).
        let meta =
            unsafe { Ref::new(NonNull::new_unchecked(meta.as_ptr().cast_mut()).cast::<T>()) };
        Self { bits, meta }
    }
}

impl<'a, const NBITS: usize, Repr, T, Perm> VectorMut<'a, NBITS, Repr, T, Perm>
where
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    T: bytemuck::Pod,
{
    /// Construct an instance of `Self` viewing `data` as the canonical layout for a vector.
    /// The canonical layout is as follows:
    ///
    /// * `std::mem::size_of::<T>()` for the metadata coefficient.
    /// * `Self::slice_bytes(dim)` for the underlying bit-slice.
    ///
    /// Returns an error if `data.len() != `Self::canonical_bytes`.
    pub fn from_canonical_front_mut(data: &'a mut [u8], dim: usize) -> Result<Self, NotCanonical> {
        let expected = Self::canonical_bytes(dim);
        let bytes = data.len();
        let (front, back) = match data.split_at_mut_checked(std::mem::size_of::<T>()) {
            Some(v) => v,
            None => {
                return Err(NotCanonical::WrongLength(expected, bytes));
            }
        };

        let bits =
            MutBitSlice::new(back, dim).map_err(|_| NotCanonical::WrongLength(expected, bytes))?;

        // SAFETY: `split_at_mut_checked` was successful, so `front` points to a valid
        // slice of `std::mem::size_of::<T>()` bytes. Further, we have verified that the
        // base pointer for `front` is properly aligned to `std::mem::align_of::<T>()`, so
        // we can safely construct a reference to a `T` from the pointer returned by
        // `front.as_ptr_mut()`.
        let meta = unsafe { Mut::new(NonNull::new_unchecked(front.as_mut_ptr()).cast::<T>()) };
        Ok(Self { bits, meta })
    }

    /// Construct an instance of `Self` viewing `data` as the canonical layout for a vector.
    /// The back canonical layout is as follows:
    ///
    /// * `Self::slice_bytes(dim)` for the underlying bit-slice.
    /// * `std::mem::size_of::<T>()` for the metadata coefficient.
    ///
    /// Returns an error if `data.len() != `Self::canonical_bytes`.
    pub fn from_canonical_back_mut(data: &'a mut [u8], dim: usize) -> Result<Self, NotCanonical> {
        let len = data.len();
        let expected = || Self::canonical_bytes(dim);
        let (front, back) = match data.split_at_mut_checked(Self::slice_bytes(dim)) {
            Some(v) => v,
            None => {
                return Err(NotCanonical::WrongLength(expected(), len));
            }
        };

        if back.len() != std::mem::size_of::<T>() {
            return Err(NotCanonical::WrongLength(expected(), len));
        }

        // SAFETY: Since `split_at_mut_checked` was successful, we know that the underlying
        // slice is the correct size.
        let bits = unsafe { MutBitSlice::new_unchecked(front, dim) };

        // SAFETY: `split_at_mut_checked` was successful and `back` was checked for lenght,
        // so `back` points to a valid slice of `std::mem::size_of::<T>()` bytes.
        let meta = unsafe { Mut::new(NonNull::new_unchecked(back.as_mut_ptr()).cast::<T>()) };
        Ok(Self { bits, meta })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::{Reborrow, ReborrowMut};
    use rand::{
        distr::{Distribution, StandardUniform, Uniform},
        rngs::StdRng,
        Rng, SeedableRng,
    };

    use super::*;
    use crate::bits::{BoxedBitSlice, Representation, Unsigned};

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
        let mut base = Vector::<7, Unsigned, Metadata>::new_boxed(len);
        assert_eq!(base.len(), len);
        assert_eq!(base.meta(), Metadata::default());
        assert!(!base.is_empty());
        // Ensure that if we reborrow mutably that changes are visible.
        {
            let mut rb = base.reborrow_mut();
            assert_eq!(rb.len(), len);
            rb.set_meta(Metadata::new(1, 2));
            let mut v = rb.vector_mut();

            assert_eq!(v.len(), len);
            for i in 0..v.len() {
                v.set(i, i as i64).unwrap();
            }
        }

        // Are the changes visible?
        let expected_metadata = Metadata::new(1, 2);
        assert_eq!(base.meta(), expected_metadata);
        assert_eq!(base.len(), len);
        let v = base.vector();
        for i in 0..v.len() {
            assert_eq!(v.get(i).unwrap(), i as i64);
        }

        // Are the changes still visible if we reborrow?
        {
            let rb = base.reborrow();
            assert_eq!(rb.len(), len);
            assert_eq!(rb.meta(), expected_metadata);
            let v = rb.vector();
            for i in 0..v.len() {
                assert_eq!(v.get(i).unwrap(), i as i64);
            }
        }
    }

    #[test]
    fn test_compensated_mut() {
        let len = 30;
        let mut v = BoxedBitSlice::<7, Unsigned>::new_boxed(len);
        let mut m = Metadata::default();

        // borrowed duration
        let mut vector = VectorMut::new(v.reborrow_mut(), &mut m);
        assert_eq!(vector.len(), len);
        vector.set_meta(Metadata::new(200, 5));
        for i in 0..vector.len() {
            vector.vector_mut().set(i, i as i64).unwrap();
        }

        // ensure changes are visible
        assert_eq!(m.a, 200);
        assert_eq!(m.b, 5);
        for i in 0..len {
            assert_eq!(v.get(i).unwrap(), i as i64);
        }
    }

    //////////////////////
    // Canonicalization //
    //////////////////////

    type TestVectorRef<'a, const NBITS: usize> = VectorRef<'a, NBITS, Unsigned, Metadata>;
    type TestVectorMut<'a, const NBITS: usize> = VectorMut<'a, NBITS, Unsigned, Metadata>;

    fn check_canonicalization<const NBITS: usize, R>(dim: usize, ntrials: usize, rng: &mut R)
    where
        Unsigned: Representation<NBITS>,
        R: Rng,
    {
        let bytes = TestVectorRef::<NBITS>::canonical_bytes(dim);
        assert_eq!(
            bytes,
            std::mem::size_of::<Metadata>() + BitSlice::<NBITS, Unsigned>::bytes_for(dim)
        );

        let mut buffer_front = vec![u8::default(); bytes + std::mem::size_of::<Metadata>() + 1];
        let mut buffer_back = vec![u8::default(); bytes + std::mem::size_of::<Metadata>() + 1];

        // Expected metadata and vector encoding.
        let mut expected = vec![i64::default(); dim];

        let uniform = Uniform::try_from(Unsigned::domain_const::<NBITS>()).unwrap();

        for _ in 0..ntrials {
            let offset = Uniform::new(0, std::mem::size_of::<Metadata>())
                .unwrap()
                .sample(rng);
            let a: u32 = StandardUniform.sample(rng);
            let b: u32 = StandardUniform.sample(rng);

            expected.iter_mut().for_each(|i| *i = uniform.sample(rng));
            {
                let set = |mut cv: TestVectorMut<NBITS>| {
                    cv.set_meta(Metadata::new(a, b));
                    let mut vector = cv.vector_mut();
                    for (i, e) in expected.iter().enumerate() {
                        vector.set(i, *e).unwrap();
                    }
                };

                // Front
                let cv = TestVectorMut::<NBITS>::from_canonical_front_mut(
                    &mut buffer_front[offset..offset + bytes],
                    dim,
                )
                .unwrap();
                set(cv);

                // Back
                let cv = TestVectorMut::<NBITS>::from_canonical_back_mut(
                    &mut buffer_back[offset..offset + bytes],
                    dim,
                )
                .unwrap();
                set(cv);
            }

            // Make sure the reconstruction is valid.
            {
                let check = |cv: TestVectorRef<NBITS>| {
                    assert_eq!(cv.meta(), Metadata::new(a, b));
                    let vector = cv.vector();
                    for (i, e) in expected.iter().enumerate() {
                        assert_eq!(vector.get(i).unwrap(), *e);
                    }
                };

                let cv = TestVectorRef::<NBITS>::from_canonical_front(
                    &buffer_front[offset..offset + bytes],
                    dim,
                )
                .unwrap();
                check(cv);

                let cv = TestVectorRef::<NBITS>::from_canonical_back(
                    &buffer_back[offset..offset + bytes],
                    dim,
                )
                .unwrap();
                check(cv);
            }
        }

        // Check Errors - Mut
        {
            // Too short
            let err = TestVectorMut::<NBITS>::from_canonical_front_mut(
                &mut buffer_front[..bytes - 1],
                dim,
            )
            .unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            let err =
                TestVectorMut::<NBITS>::from_canonical_back_mut(&mut buffer_back[..bytes - 1], dim)
                    .unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            // Empty
            let err = TestVectorMut::<NBITS>::from_canonical_front_mut(&mut [], dim).unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            let err = TestVectorMut::<NBITS>::from_canonical_back_mut(&mut [], dim).unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            // Too long
            let err = TestVectorMut::<NBITS>::from_canonical_front_mut(
                &mut buffer_front[..bytes + 1],
                dim,
            )
            .unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            let err =
                TestVectorMut::<NBITS>::from_canonical_back_mut(&mut buffer_back[..bytes + 1], dim)
                    .unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));
        }

        // Check Errors - Const
        {
            // Too short
            let err = TestVectorRef::<NBITS>::from_canonical_front(&buffer_front[..bytes - 1], dim)
                .unwrap_err();
            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            let err = TestVectorRef::<NBITS>::from_canonical_back(&buffer_back[..bytes - 1], dim)
                .unwrap_err();
            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            // Empty
            let err = TestVectorRef::<NBITS>::from_canonical_front(&[], dim).unwrap_err();
            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            let err = TestVectorRef::<NBITS>::from_canonical_back(&[], dim).unwrap_err();
            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            // Too long
            let err = TestVectorRef::<NBITS>::from_canonical_front(&buffer_front[..bytes + 1], dim)
                .unwrap_err();
            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            let err = TestVectorRef::<NBITS>::from_canonical_back(&buffer_back[..bytes + 1], dim)
                .unwrap_err();
            assert!(matches!(err, NotCanonical::WrongLength(_, _)));
        }
    }

    fn check_canonicalization_zst<const NBITS: usize, R>(dim: usize, ntrials: usize, rng: &mut R)
    where
        Unsigned: Representation<NBITS>,
        R: Rng,
    {
        let bytes = VectorRef::<NBITS, Unsigned, ()>::canonical_bytes(dim);
        assert_eq!(bytes, BitSlice::<NBITS, Unsigned>::bytes_for(dim));

        let max_offset = 10;
        let mut buffer_front = vec![u8::default(); bytes + max_offset];
        let mut buffer_back = vec![u8::default(); bytes + max_offset];

        // Expected metadata and vector encoding.
        let mut expected = vec![i64::default(); dim];

        let uniform = Uniform::try_from(Unsigned::domain_const::<NBITS>()).unwrap();

        for _ in 0..ntrials {
            let offset = Uniform::new(0, max_offset).unwrap().sample(rng);
            expected.iter_mut().for_each(|i| *i = uniform.sample(rng));
            {
                let set = |mut cv: VectorMut<NBITS, Unsigned, ()>| {
                    cv.set_meta(());
                    let mut vector = cv.vector_mut();
                    for (i, e) in expected.iter().enumerate() {
                        vector.set(i, *e).unwrap();
                    }
                };

                let cv = VectorMut::<NBITS, Unsigned, ()>::from_canonical_front_mut(
                    &mut buffer_front[offset..offset + bytes],
                    dim,
                )
                .unwrap();
                set(cv);

                let cv = VectorMut::<NBITS, Unsigned, ()>::from_canonical_back_mut(
                    &mut buffer_back[offset..offset + bytes],
                    dim,
                )
                .unwrap();
                set(cv);
            }

            // Make sure the reconstruction is valid.
            {
                let check = |cv: VectorRef<NBITS, Unsigned, ()>| {
                    let vector = cv.vector();
                    for (i, e) in expected.iter().enumerate() {
                        assert_eq!(vector.get(i).unwrap(), *e);
                    }
                };

                let cv = VectorRef::<NBITS, Unsigned, ()>::from_canonical_front(
                    &buffer_front[offset..offset + bytes],
                    dim,
                )
                .unwrap();
                check(cv);

                let cv = VectorRef::<NBITS, Unsigned, ()>::from_canonical_back(
                    &buffer_back[offset..offset + bytes],
                    dim,
                )
                .unwrap();
                check(cv);
            }
        }

        // Check Errors - Mut
        {
            // Too short
            if dim >= 1 {
                let err = VectorMut::<NBITS, Unsigned, ()>::from_canonical_front_mut(
                    &mut buffer_front[..bytes - 1],
                    dim,
                )
                .unwrap_err();
                assert!(matches!(err, NotCanonical::WrongLength(_, _)));

                let err = VectorMut::<NBITS, Unsigned, ()>::from_canonical_back_mut(
                    &mut buffer_back[..bytes - 1],
                    dim,
                )
                .unwrap_err();
                assert!(matches!(err, NotCanonical::WrongLength(_, _)));
            }

            // Empty
            if dim >= 1 {
                let err = VectorMut::<NBITS, Unsigned, ()>::from_canonical_front_mut(&mut [], dim)
                    .unwrap_err();
                assert!(matches!(err, NotCanonical::WrongLength(_, _)));

                let err = VectorMut::<NBITS, Unsigned, ()>::from_canonical_back_mut(&mut [], dim)
                    .unwrap_err();
                assert!(matches!(err, NotCanonical::WrongLength(_, _)));
            }

            // Too long
            {
                let err = VectorMut::<NBITS, Unsigned, ()>::from_canonical_front_mut(
                    &mut buffer_front[..bytes + 1],
                    dim,
                )
                .unwrap_err();

                assert!(matches!(err, NotCanonical::WrongLength(_, _)));

                let err = VectorMut::<NBITS, Unsigned, ()>::from_canonical_back_mut(
                    &mut buffer_back[..bytes + 1],
                    dim,
                )
                .unwrap_err();

                assert!(matches!(err, NotCanonical::WrongLength(_, _)));
            }
        }

        // Check Errors - Const
        {
            // Too short
            if dim >= 1 {
                let err = VectorRef::<NBITS, Unsigned, ()>::from_canonical_front(
                    &buffer_front[..bytes - 1],
                    dim,
                )
                .unwrap_err();

                assert!(matches!(err, NotCanonical::WrongLength(_, _)));

                let err = VectorRef::<NBITS, Unsigned, ()>::from_canonical_back(
                    &buffer_back[..bytes - 1],
                    dim,
                )
                .unwrap_err();

                assert!(matches!(err, NotCanonical::WrongLength(_, _)));
            }

            // Too long
            let err = VectorMut::<NBITS, Unsigned, ()>::from_canonical_front_mut(
                &mut buffer_front[..bytes + 1],
                dim,
            )
            .unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));

            let err = VectorMut::<NBITS, Unsigned, ()>::from_canonical_back_mut(
                &mut buffer_back[..bytes + 1],
                dim,
            )
            .unwrap_err();

            assert!(matches!(err, NotCanonical::WrongLength(_, _)));
        }

        // Check Errors - Const
        {
            // Too short
            if dim >= 1 {
                let err =
                    VectorRef::<NBITS, Unsigned, ()>::from_canonical_front(&[], dim).unwrap_err();

                assert!(matches!(err, NotCanonical::WrongLength(_, _)));

                let err =
                    VectorRef::<NBITS, Unsigned, ()>::from_canonical_back(&[], dim).unwrap_err();

                assert!(matches!(err, NotCanonical::WrongLength(_, _)));
            }

            // Too long
            {
                let err = VectorRef::<NBITS, Unsigned, ()>::from_canonical_front(
                    &buffer_front[..bytes + 1],
                    dim,
                )
                .unwrap_err();

                assert!(matches!(err, NotCanonical::WrongLength(_, _)));

                let err = VectorRef::<NBITS, Unsigned, ()>::from_canonical_back(
                    &buffer_back[..bytes + 1],
                    dim,
                )
                .unwrap_err();

                assert!(matches!(err, NotCanonical::WrongLength(_, _)));
            }
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            // The max dim does not need to be as high for `CompensatedVectors` because they
            // defer their distance function implementation to `BitSlice`, which is more
            // heavily tested.
            const MAX_DIM: usize = 37;
            const TRIALS_PER_DIM: usize = 1;
        } else {
            const MAX_DIM: usize = 256;
            const TRIALS_PER_DIM: usize = 20;
        }
    }

    macro_rules! test_canonical {
        ($name:ident, $nbits:literal, $seed:literal) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);
                for dim in 0..MAX_DIM {
                    check_canonicalization::<$nbits, _>(dim, TRIALS_PER_DIM, &mut rng);
                    check_canonicalization_zst::<$nbits, _>(dim, TRIALS_PER_DIM, &mut rng);
                }
            }
        };
    }

    test_canonical!(canonical_8bit, 8, 0xe64518a00ee99e2f);
    test_canonical!(canonical_7bit, 7, 0x3907123f8c38def2);
    test_canonical!(canonical_6bit, 6, 0xeccaeb83965ff6a1);
    test_canonical!(canonical_5bit, 5, 0x9691fe59e49bfb96);
    test_canonical!(canonical_4bit, 4, 0xc4d3e9bc699a7e6f);
    test_canonical!(canonical_3bit, 3, 0x8a01b2ccdca8fb2b);
    test_canonical!(canonical_2bit, 2, 0x3a07429e8184b67f);
    test_canonical!(canonical_1bit, 1, 0x93fddb26059c115c);
}
