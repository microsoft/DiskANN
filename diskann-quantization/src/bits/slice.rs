/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, ops::RangeInclusive, ptr::NonNull};

use diskann_utils::{Reborrow, ReborrowMut};
use thiserror::Error;

use super::{
    length::{Dynamic, Length},
    packing,
    ptr::{AsMutPtr, AsPtr, MutSlicePtr, Precursor, SlicePtr},
};
use crate::{
    alloc::{AllocatorCore, AllocatorError, GlobalAllocator, Poly},
    utils,
};

//////////////////////
// Retrieval Traits //
//////////////////////

/// Representation of `NBITS` bit numbers in the associated domain.
pub trait Representation<const NBITS: usize> {
    /// The type of the domain accepted by this representation.
    type Domain: Iterator<Item = i64>;

    /// Encode `value` into the lower order bits of a byte. Returns the encoded value on
    /// success, or an `EncodingError` if the value is unencodable.
    fn encode(value: i64) -> Result<u8, EncodingError>;

    /// Encode `value` into the lower order bits of a byte without checking if `value`
    /// is encodable. This function is not marked as unsafe because in-and-of itself, it
    /// won't cause memory safety issues.
    ///
    /// This may panic in debug mode when `value` is outside of this representation's
    /// domain.
    fn encode_unchecked(value: i64) -> u8;

    /// Decode a previously encoded value. The result will be in the range
    /// `[Self::MIN, Self::MAX]`.
    ///
    /// # Panics
    ///
    /// May panic in debug builds if `raw` is not a valid pattern emitted by `encode`.
    fn decode(raw: u8) -> i64;

    /// Check whether or not the argument is in the domain.
    fn check(value: i64) -> bool;

    /// Return an iterator over the domain of representable values.
    fn domain() -> Self::Domain;
}

#[derive(Debug, Error, Clone, Copy)]
#[error("value {} is not in the encodable range of {}", got, domain)]
pub struct EncodingError {
    got: i64,
    // Question: Why is this a ref-ref??
    //
    // Answer: I have a personal vendetta to keep this struct within 16-bytes with a
    // niche-optimization. A `&'static src` is 16 bytes in and of itself. But a
    // `&'static &'static str`, now *that's* just 8 bytes.
    domain: &'static &'static str,
}

impl EncodingError {
    fn new(got: i64, domain: &'static &'static str) -> Self {
        Self { got, domain }
    }
}

//////////////
// Unsigned //
//////////////

/// Storage unsigned integers in slices.
///
/// For a bit count of `NBITS`, the `Unsigned` type can store unsigned integers in
/// the range `[0, 2^NBITS - 1]`.
#[derive(Debug, Clone, Copy)]
pub struct Unsigned;

impl Unsigned {
    /// Return the dynamic range of an `Unsigned` encoding for `NBITS`.
    pub const fn domain_const<const NBITS: usize>() -> std::ops::RangeInclusive<i64> {
        0..=2i64.pow(NBITS as u32) - 1
    }

    #[allow(clippy::panic)]
    const fn domain_str(nbits: usize) -> &'static &'static str {
        match nbits {
            8 => &"[0, 255]",
            7 => &"[0, 127]",
            6 => &"[0, 63]",
            5 => &"[0, 31]",
            4 => &"[0, 15]",
            3 => &"[0, 7]",
            2 => &"[0, 3]",
            1 => &"[0, 1]",
            _ => panic!("unimplemented"),
        }
    }
}

macro_rules! repr_unsigned {
    ($N:literal) => {
        impl Representation<$N> for Unsigned {
            type Domain = RangeInclusive<i64>;

            fn encode(value: i64) -> Result<u8, EncodingError> {
                if !<Self as Representation<$N>>::check(value) {
                    // Even with the macro gymnastics - we still have to manually inline
                    // this computation :(
                    let domain = Self::domain_str($N);
                    Err(EncodingError::new(value, domain))
                } else {
                    Ok(<Self as Representation<$N>>::encode_unchecked(value))
                }
            }

            fn encode_unchecked(value: i64) -> u8 {
                debug_assert!(<Self as Representation<$N>>::check(value));
                value as u8
            }

            fn decode(raw: u8) -> i64 {
                // Feed through the value un-modified.
                let raw: i64 = raw.into();
                debug_assert!(<Self as Representation<$N>>::check(raw));
                raw
            }

            fn check(value: i64) -> bool {
                <Self as Representation<$N>>::domain().contains(&value)
            }

            fn domain() -> Self::Domain {
                Self::domain_const::<$N>()
            }
        }
    };
    ($N:literal, $($Ns:literal),+) => {
        repr_unsigned!($N);
        $(repr_unsigned!($Ns);)+
    };
}

repr_unsigned!(1, 2, 3, 4, 5, 6, 7, 8);

////////////
// Binary //
////////////

/// A 1-bit binary quantization mapping `-1` to `0` and `1` to `1`.
#[derive(Debug, Clone, Copy)]
pub struct Binary;

impl Representation<1> for Binary {
    type Domain = std::array::IntoIter<i64, 2>;

    fn encode(value: i64) -> Result<u8, EncodingError> {
        if !Self::check(value) {
            const DOMAIN: &str = "{-1, 1}";
            Err(EncodingError::new(value, &DOMAIN))
        } else {
            Ok(Self::encode_unchecked(value))
        }
    }

    fn encode_unchecked(value: i64) -> u8 {
        debug_assert!(Self::check(value));
        // The use of `clamp` here is a quick way of sending `-1` to `0` and `1` to `1`.
        value.clamp(0, 1) as u8
    }

    fn decode(raw: u8) -> i64 {
        // Raw is either 0 or 1. We want to map it to -1 or 1.
        // We can do this by multiplying by 2 and subtracting 1.
        let raw: i64 = raw.into();
        (raw << 1) - 1
    }

    fn check(value: i64) -> bool {
        value == -1 || value == 1
    }

    /// Return the domain of the encoding.
    ///
    /// The domain is the set `{-1, 1}`.
    fn domain() -> Self::Domain {
        [-1, 1].into_iter()
    }
}

////////////////////////
// Permutation Traits //
////////////////////////

/// A enable the dimensions within a BitSlice to be permuted in an arbitrary way.
///
/// # Safety
///
/// This provides the computation for the number of bytes required to store a given number
/// of `NBITS` bit-packed values. Improper implementation will result in out-of-bounds
/// accesses being made.
///
/// The following must hold:
///
/// For all counts values `c`, let `b = Self::bytes(c)` be requested number of bytes for `c`
/// for this permutation strategy and `s` be a slice of bytes with length `c`. Then, for all
/// `i < c`, `Self::pack(s, i, _)` and `Self::unpack(s, i)` must only access `s` in-bounds.
///
/// This implementation must be such that unsafe code can rely on this property holding.
pub unsafe trait PermutationStrategy<const NBITS: usize> {
    /// Return the number of bytes required to store `count` values of with `NBITS`.
    fn bytes(count: usize) -> usize;

    /// Pack the lower `NBITS` bits of `value` into `s` at logical index `i`.
    ///
    /// # Safety
    ///
    /// This is a tricky function to call with several subtle requirements.
    ///
    /// * Let `s` be a slice of length `c` where `c = Self::bytes(b)` for some `b`. Then
    ///   this function is safe to call if `i` is in `[0, b)`.
    unsafe fn pack(s: &mut [u8], i: usize, value: u8);

    /// Unpack the value stored at logical index `i` and return it as the lower `NBITS` bits
    /// in the return value.
    ///
    /// # Safety
    ///
    /// This is a tricky function to call with several subtle requirements.
    ///
    /// * Let `s` be a slice of length `c` where `c = Self::bytes(b)` for some `b`. Then
    ///   this function is safe to call if `i` is in `[0, b)`.
    unsafe fn unpack(s: &[u8], i: usize) -> u8;
}

/// The identity permutation strategy.
///
/// All values are densly packed.
#[derive(Debug, Clone, Copy)]
pub struct Dense;

impl Dense {
    fn bytes<const NBITS: usize>(count: usize) -> usize {
        utils::div_round_up(NBITS * count, 8)
    }
}

/// Safety: For all `0 <= i < count`, `NBITS * i <= 8 * ceil((NBITS * count) / 8)`.
unsafe impl<const NBITS: usize> PermutationStrategy<NBITS> for Dense {
    fn bytes(count: usize) -> usize {
        Self::bytes::<NBITS>(count)
    }

    unsafe fn pack(data: &mut [u8], i: usize, encoded: u8) {
        let bitaddress = NBITS * i;

        let bytestart = bitaddress / 8;
        let bytestop = (bitaddress + NBITS - 1) / 8;
        let bitstart = bitaddress - 8 * bytestart;
        debug_assert!(bytestop < data.len());

        if bytestart == bytestop {
            // SAFETY: This is safe for the following:
            // ```
            // data.len() >= ceil(NBITS * i / 8)        from `pack`'s safety requirements.
            //            >= floor(NBITS * i / 8)
            //            = bytestart
            //
            // Since we are only reading one byte - this is in-bounds.
            // ```
            let raw = unsafe { data.as_ptr().add(bytestart).read() };
            let packed = packing::pack_u8::<NBITS>(raw, encoded, bitstart);

            // SAFETY: See previous argument for in-bounds access.
            // For writing, we are the only writers in this function and we have a mutable
            // reference to `data`.
            unsafe { data.as_mut_ptr().add(bytestart).write(packed) };
        } else {
            // SAFETY: This is safe for the following reason:
            // ```
            // data.len() >= ceil(NBITS * i / 8)        from `pack`'s safety requirements.
            //            = bytestop
            //            = bytestart + 1
            // ```
            // Therefore, it is safe to read 2-bytes starting at `bytestart`.
            let raw = unsafe { data.as_ptr().add(bytestart).cast::<u16>().read_unaligned() };
            let packed = packing::pack_u16::<NBITS>(raw, encoded, bitstart);

            // SAFETY: See previous argument for in-bounds access.
            // For writing, we are the only writers in this function and we have a mutable
            // reference to `data`.
            unsafe {
                data.as_mut_ptr()
                    .add(bytestart)
                    .cast::<u16>()
                    .write_unaligned(packed)
            };
        }
    }

    unsafe fn unpack(data: &[u8], i: usize) -> u8 {
        let bitaddress = NBITS * i;

        let bytestart = bitaddress / 8;
        let bytestop = (bitaddress + NBITS - 1) / 8;
        debug_assert!(bytestop < data.len());
        if bytestart == bytestop {
            // SAFETY: See the safety argument in `pack` for in-bounds.
            let raw = unsafe { data.as_ptr().add(bytestart).read() };
            packing::unpack_u8::<NBITS>(raw, bitaddress - 8 * bytestart)
        } else {
            // SAFETY: See the safety argument in `pack` for in-bounds.
            let raw = unsafe { data.as_ptr().add(bytestart).cast::<u16>().read_unaligned() };
            packing::unpack_u16::<NBITS>(raw, bitaddress - 8 * bytestart)
        }
    }
}

/// A layout specialized for performing multi-bit operations with 1-bit scalar quantization.
///
/// The layout provided by this struct is as follows. Assume we are compressing `N` bit data.
/// Then, the store the data in blocks of `64 * N` bits (where 64 comes from the native CPU
/// word size).
///
/// Each block can contain 64 values, stored in `N` 64-bit words. The 0th bit of each value
/// is stored in word 0, the 1st bit is stored in word 1, etc.
///
/// # Partially Filled Blocks
///
/// This strategy always requests data in blocks. For partially filled blocks, the lower
/// bits in the last block will be used.
#[derive(Debug, Clone, Copy)]
pub struct BitTranspose;

/// Safety: We ask for bytes in multiples of 32. Furthermore, the accesses to the packed
/// data in `pack` and `unpack` use checked accesses, so out-of-bounds reads will panic.
unsafe impl PermutationStrategy<4> for BitTranspose {
    fn bytes(count: usize) -> usize {
        32 * utils::div_round_up(count, 64)
    }

    unsafe fn pack(data: &mut [u8], i: usize, encoded: u8) {
        // Compute the byte-address of the block containing `i`.
        let block_start = 32 * (i / 64);
        // Compute the offset within the block to find the first byte containing `i`.
        let byte_start = block_start + (i % 64) / 8;
        // Finally, compute the bit within the byte that we are interested in.
        let bit = i % 8;

        let mask: u8 = 0x1 << bit;
        for p in 0..4 {
            let mut v = data[byte_start + 8 * p];
            v = (v & !mask) | (((encoded >> p) & 0x1) << bit);
            data[byte_start + 8 * p] = v;
        }
    }

    unsafe fn unpack(data: &[u8], i: usize) -> u8 {
        // Compute the byte-address of the block containing `i`.
        let block_start = 32 * (i / 64);
        // Compute the offset within the block to find the first byte containing `i`.
        let byte_start = block_start + (i % 64) / 8;
        // Finally, compute the bit within the byte that we are interested in.
        let bit = i % 8;

        let mut output: u8 = 0;
        for p in 0..4 {
            let v = data[byte_start + 8 * p];
            output |= ((v >> bit) & 0x1) << p
        }
        output
    }
}

////////////
// Errors //
////////////

#[derive(Debug, Error, Clone, Copy)]
#[error("input span has length {got} bytes but expected {expected}")]
pub struct ConstructionError {
    got: usize,
    expected: usize,
}

#[derive(Debug, Error, Clone, Copy)]
#[error("index {index} exceeds the maximum length of {len}")]
pub struct IndexOutOfBounds {
    index: usize,
    len: usize,
}

impl IndexOutOfBounds {
    fn new(index: usize, len: usize) -> Self {
        Self { index, len }
    }
}

#[derive(Debug, Error, Clone, Copy)]
#[error("error setting index in bitslice")]
#[non_exhaustive]
pub enum SetError {
    IndexError(#[from] IndexOutOfBounds),
    EncodingError(#[from] EncodingError),
}

#[derive(Debug, Error, Clone, Copy)]
#[error("error getting index in bitslice")]
pub enum GetError {
    IndexError(#[from] IndexOutOfBounds),
}

//////////////
// BitSlice //
//////////////

/// A generalized representation for packed small bit integer encodings over a contiguous
/// span of memory.
///
/// Think of this as a Rust slice, but supporting integer elements with fewer than 8-bits.
/// The borrowed representations [`BitSlice`] and [`MutBitSlice`] consist of just a pointer
/// and a length and are therefore just 16-bytes in size and amenable to the niche
/// optimization.
///
/// # Parameters
///
/// * `NBITS`: The number of bits occupied by each entry in the vector.
///
/// * `Repr`: The storage representation for each collection of 8-bits. This representation
///   defines the domain of the encoding (i.e., range of realized values) as well as how
///   this domain is mapped into `NBITS` bits.
///
/// * `Ptr`: The storage type for the contiguous memory. Possible representations are:
///   - `diskann_quantization::bits::SlicePtr<'_, u8>`: For immutable views.
///   - `diskann_quantization::bits::MutSlicePtr<'_, u8>`: For mutable views.
///   - `Box<[u8]>`: For standalone vectors.
///
/// * `Perm`: By default, this type uses a dense storage strategy where the least significant
///   bit of the value at index `i` occurs directly after the most significant bit of
///   the value at index `i-1`.
///
///   Different permutations can be used to enable faster distance computations between
///   compressed vectors and full-precision vectors by enabling faster SIMD unpacking.
///
/// * `Len`: The representation for the length of the vector. This may only be one of the
///   two families of types:
///   - `diskann_quantization::bits::Dynamic`: For instances with a run-time length.
///   - `diskann_quantization::bits::Static<N>`: For instances with a compile-time known length
///     of `N`.
///
/// # Examples
///
/// ## Canonical Bit Slice
///
/// The canonical `BitSlice` stores unsigned integers of `NBITS` densely in memory.
/// That is, for a type `BitSliceBase<3, Unsigned, _>`, the layout is as follows:
/// ```text
/// |<--LSB-- byte 0 --MSB--->|<--LSB-- byte 1 --MSB--->|
/// | a0 a1 a2 b0 b1 b2 c0 c1 | c2 d0 d1 d2 e0 e1 e2 f0 |
/// |<-- A -->|<-- B ->|<--- C -->|<-- D ->|<-- E ->|<- F
/// ```
/// An example is shown below:
///
/// ```rust
/// use diskann_quantization::bits::{BoxedBitSlice, Unsigned};
/// // Create a new boxed bit-slice with capacity for 10 dimensions.
/// let mut x = BoxedBitSlice::<3, Unsigned>::new_boxed(10);
/// assert_eq!(x.len(), 10);
/// // The number of bytes in the canonical representation is computed by
/// // ceil((len * NBITS) / 8);
/// assert_eq!(x.bytes(), 4);
///
/// // Assign values.
/// x.set(0, 1).unwrap(); // assign the value 1 to index 0
/// x.set(1, 5).unwrap(); // assign the value 5 to index 1
/// assert_eq!(x.get(0).unwrap(), 1); // retrieve the value at index 0
/// assert_eq!(x.get(1).unwrap(), 5); // retrieve the value at index 1
///
/// // Assigning out-of-bounds will result in an error.
/// let err = x.set(1, 10).unwrap_err();
/// assert!(matches!(diskann_quantization::bits::SetError::EncodingError, err));
/// // The old value is left untouched.
/// assert_eq!(x.get(1).unwrap(), 5);
///
/// // `BoxedBitSlice` allows itself to be consumed, returning the underlying storage.
/// let y = x.into_inner();
/// assert_eq!(y.len(), BoxedBitSlice::<3, Unsigned>::bytes_for(10));
/// ```
///
/// The above example demonstrates a boxed bit slice - a type that owns its underlying
/// memory. However, this is not always ergonomic when interfacing with data stores.
/// For this, the viewing interface can be used.
/// ```rust
/// use diskann_quantization::bits::{MutBitSlice, Unsigned};
///
/// let mut x: Vec<u8> = vec![0; 4];
/// let mut slice = MutBitSlice::<3, Unsigned>::new(x.as_mut_slice(), 10).unwrap();
/// assert_eq!(slice.len(), 10);
/// assert_eq!(slice.bytes(), 4);
///
/// // The slice reference behaves just like boxed slice.
/// slice.set(0, 5).unwrap();
/// assert_eq!(slice.get(0).unwrap(), 5);
///
/// // Note - if the number of bytes required for the provided dimensions does not match
/// // the length of the provided span, than slice construction will return an error.
/// let err = MutBitSlice::<3, Unsigned>::new(x.as_mut_slice(), 11).unwrap_err();
/// assert_eq!(err.to_string(), "input span has length 4 bytes but expected 5");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BitSliceBase<const NBITS: usize, Repr, Ptr, Perm = Dense, Len = Dynamic>
where
    Repr: Representation<NBITS>,
    Ptr: AsPtr<Type = u8>,
    Perm: PermutationStrategy<NBITS>,
    Len: Length,
{
    ptr: Ptr,
    len: Len,
    repr: PhantomData<Repr>,
    packing: PhantomData<Perm>,
}

impl<const NBITS: usize, Repr, Ptr, Perm, Len> BitSliceBase<NBITS, Repr, Ptr, Perm, Len>
where
    Repr: Representation<NBITS>,
    Ptr: AsPtr<Type = u8>,
    Perm: PermutationStrategy<NBITS>,
    Len: Length,
{
    /// Check that NBITS is in the interval [1, 8].
    const _CHECK: () = assert!(NBITS > 0 && NBITS <= 8);

    /// Return the exact number of bytes required to store `count` values.
    pub fn bytes_for(count: usize) -> usize {
        Perm::bytes(count)
    }

    /// Return a new `BitSlice` over the data behind `ptr`.
    ///
    /// # Safety
    ///
    /// It's the callers responsibility to ensure that all the invariants required for
    /// `std::slice::from_raw_parts(ptr.as_ptr(), len.value)` hold.
    unsafe fn new_unchecked_internal(ptr: Ptr, len: Len) -> Self {
        Self {
            ptr,
            len,
            repr: PhantomData,
            packing: PhantomData,
        }
    }

    /// Construct a new `BitSlice` without checking preconditions.
    ///
    /// # Safety
    ///
    /// Requires the following to avoid undefined behavior:
    ///
    /// * `precursor.precursor_len() == Self::bytes_for(<Count as Into<Len>>::into(count).value())`.
    ///
    /// This is checked in debug builds.
    pub unsafe fn new_unchecked<Pre, Count>(precursor: Pre, count: Count) -> Self
    where
        Count: Into<Len>,
        Pre: Precursor<Ptr>,
    {
        let count: Len = count.into();
        debug_assert_eq!(precursor.precursor_len(), Self::bytes_for(count.value()));
        Self::new_unchecked_internal(precursor.precursor_into(), count)
    }

    /// Construct a new `BitSlice` from the `precursor` capable of holding `count` encoded
    /// elements of size `NBITS.
    ///
    /// # Requirements
    ///
    /// The number of bytes pointed to by the precursor must be equal to the number of bytes
    /// required by the layout. That is:
    ///
    /// * `precursor.precursor_len() == Self::bytes_for(<Count as Into<Len>>::into(count).value())`.
    pub fn new<Pre, Count>(precursor: Pre, count: Count) -> Result<Self, ConstructionError>
    where
        Count: Into<Len>,
        Pre: Precursor<Ptr>,
    {
        // Allow callers to pass in `usize` as the count when using dynamic
        let count: Len = count.into();

        // Make sure that the slice has the correct length.
        if precursor.precursor_len() != Self::bytes_for(count.value()) {
            Err(ConstructionError {
                got: precursor.precursor_len(),
                expected: Self::bytes_for(count.value()),
            })
        } else {
            // SAFETY: We have checked that `precursor` has the correct number of bytes.
            // The only implementations of `Precursor` are those we defined for slices, so we
            // don't have to worry about downstream users inserting their own, incorrectly
            // implemented implementation.
            Ok(unsafe { Self::new_unchecked(precursor, count) })
        }
    }

    /// Return the number of elements contained in the slice.
    pub fn len(&self) -> usize {
        self.len.value()
    }

    /// Return whether or not the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of bytes occupied by this slice.
    pub fn bytes(&self) -> usize {
        Self::bytes_for(self.len())
    }

    /// Return the value at logical index `i`.
    pub fn get(&self, i: usize) -> Result<i64, GetError> {
        if i >= self.len() {
            Err(IndexOutOfBounds::new(i, self.len()).into())
        } else {
            // SAFETY: We've performed the bounds check.
            Ok(unsafe { self.get_unchecked(i) })
        }
    }

    /// Return the value at logical index `i`.
    ///
    /// # Safety
    ///
    /// Argument `i` must be in bounds: `0 <= i < self.len()`.
    pub unsafe fn get_unchecked(&self, i: usize) -> i64 {
        debug_assert!(i < self.len());
        debug_assert_eq!(self.as_slice().len(), Perm::bytes(self.len()));

        // SAFETY: We maintain the invariant that
        // `self.as_slice().len() == Perm::bytes(self.len())`.
        //
        // So, `i < self.len()` implies we uphold the safety requirements of `unpack`.
        Repr::decode(unsafe { Perm::unpack(self.as_slice(), i) })
    }

    /// Encode and assign `value` to logical index `i`.
    pub fn set(&mut self, i: usize, value: i64) -> Result<(), SetError>
    where
        Ptr: AsMutPtr<Type = u8>,
    {
        if i >= self.len() {
            return Err(IndexOutOfBounds::new(i, self.len()).into());
        }

        let encoded = Repr::encode(value)?;

        // SAFETY: We've performed the bounds check.
        unsafe { self.set_unchecked(i, encoded) }
        Ok(())
    }

    /// Assign `value` to logical index `i`.
    ///
    /// # Safety
    ///
    /// Argument `i` must be in bounds: `0 <= i < self.len()`.
    pub unsafe fn set_unchecked(&mut self, i: usize, encoded: u8)
    where
        Ptr: AsMutPtr<Type = u8>,
    {
        debug_assert!(i < self.len());
        debug_assert_eq!(self.as_slice().len(), Perm::bytes(self.len()));

        // SAFETY: We maintain the invariant that
        // `self.as_slice().len() == Perm::bytes(self.len())`.
        //
        // So, `i < self.len()` implies we uphold the safety requirements of `unpack`.
        unsafe { Perm::pack(self.as_mut_slice(), i, encoded) }
    }

    /// Return the domain of acceptable values.
    pub fn domain(&self) -> Repr::Domain {
        Repr::domain()
    }

    pub(crate) fn as_slice(&self) -> &'_ [u8] {
        // SAFETY: This class has the invariant that the backing storage must be initialized
        // and exist in a single allocation containing at least
        // `[self.ptr.as_ptr(), self.ptr_ptr() + self.bytes())`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.bytes()) }
    }

    /// Return a pointer to the beginning of the memory associated with this slice.
    ///
    /// # NOTE
    ///
    /// The memory span underlying this instances is valid for `self.bytes()`, not
    /// necessarily `self.len()`.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// This function is very easy to use incorrectly and hence is crate-local.
    pub(super) fn as_mut_slice(&mut self) -> &'_ mut [u8]
    where
        Ptr: AsMutPtr,
    {
        // SAFETY: This class has the invariant that the backing storage must be initialized
        // and exist in a single allocation containing at least
        // `[self.ptr.as_ptr(), self.ptr_ptr() + self.bytes())`.
        //
        // A mutable reference to self with `Ptr: AsMutPtr` attests to the fact that we
        // have an exclusive borrow over the underlying memory.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_mut_ptr(), self.bytes()) }
    }

    /// This function is very easy to use incorrectly and hence is private.
    fn as_mut_ptr(&mut self) -> *mut u8
    where
        Ptr: AsMutPtr,
    {
        self.ptr.as_mut_ptr()
    }
}

impl<const NBITS: usize, Repr, Perm, Len>
    BitSliceBase<NBITS, Repr, Poly<[u8], GlobalAllocator>, Perm, Len>
where
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    Len: Length,
{
    /// Construct a new owning `BitSlice` capable of holding `Count` logical values.
    /// The slice is initialized in a valid but undefined state.
    ///
    /// # Example
    ///
    /// ```
    /// use diskann_quantization::bits::{BoxedBitSlice, Unsigned};
    /// let mut x = BoxedBitSlice::<3, Unsigned>::new_boxed(4);
    /// x.set(0, 0).unwrap();
    /// x.set(1, 2).unwrap();
    /// x.set(2, 4).unwrap();
    /// x.set(3, 6).unwrap();
    ///
    /// assert_eq!(x.get(0).unwrap(), 0);
    /// assert_eq!(x.get(1).unwrap(), 2);
    /// assert_eq!(x.get(2).unwrap(), 4);
    /// assert_eq!(x.get(3).unwrap(), 6);
    /// ```
    pub fn new_boxed<Count>(count: Count) -> Self
    where
        Count: Into<Len>,
    {
        let count: Len = count.into();
        let bytes = Self::bytes_for(count.value());
        let storage: Box<[u8]> = (0..bytes).map(|_| 0).collect();

        // SAFETY: We've ensured that the backing storage has the correct number of bytes
        // as required by the count and PermutationStrategy.
        //
        // Since this is owned storage, we do not need to worry about capturing lifetimes.
        unsafe { Self::new_unchecked(Poly::from(storage), count) }
    }
}

impl<const NBITS: usize, Repr, Perm, Len, A> BitSliceBase<NBITS, Repr, Poly<[u8], A>, Perm, Len>
where
    Repr: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    Len: Length,
    A: AllocatorCore,
{
    /// Construct a new owning `BitSlice` capable of holding `Count` logical values using
    /// the provided allocator.
    ///
    /// The slice is initialized in a valid but undefined state.
    ///
    /// # Example
    ///
    /// ```
    /// use diskann_quantization::{
    ///     alloc::GlobalAllocator,
    ///     bits::{BoxedBitSlice, Unsigned}
    /// };
    /// let mut x = BoxedBitSlice::<3, Unsigned>::new_in(4, GlobalAllocator).unwrap();
    /// x.set(0, 0).unwrap();
    /// x.set(1, 2).unwrap();
    /// x.set(2, 4).unwrap();
    /// x.set(3, 6).unwrap();
    ///
    /// assert_eq!(x.get(0).unwrap(), 0);
    /// assert_eq!(x.get(1).unwrap(), 2);
    /// assert_eq!(x.get(2).unwrap(), 4);
    /// assert_eq!(x.get(3).unwrap(), 6);
    /// ```
    pub fn new_in<Count>(count: Count, allocator: A) -> Result<Self, AllocatorError>
    where
        Count: Into<Len>,
    {
        let count: Len = count.into();
        let bytes = Self::bytes_for(count.value());
        let storage = Poly::broadcast(0, bytes, allocator)?;

        // SAFETY: We've ensured that the backing storage has the correct number of bytes
        // as required by the count and PermutationStrategy.
        //
        // Since this is owned storage, we do not need to worry about capturing lifetimes.
        Ok(unsafe { Self::new_unchecked(storage, count) })
    }

    /// Consume `self` and return the boxed allocation.
    pub fn into_inner(self) -> Poly<[u8], A> {
        self.ptr
    }
}

/// The layout for `N`-bit integers that references a raw underlying slice.
pub type BitSlice<'a, const N: usize, Repr, Perm = Dense, Len = Dynamic> =
    BitSliceBase<N, Repr, SlicePtr<'a, u8>, Perm, Len>;

/// The layout for `N`-bit integers that mutable references a raw underlying slice.
pub type MutBitSlice<'a, const N: usize, Repr, Perm = Dense, Len = Dynamic> =
    BitSliceBase<N, Repr, MutSlicePtr<'a, u8>, Perm, Len>;

/// The layout for `N`-bit integers that own the underlying slice.
pub type PolyBitSlice<const N: usize, Repr, A, Perm = Dense, Len = Dynamic> =
    BitSliceBase<N, Repr, Poly<[u8], A>, Perm, Len>;

/// The layout for `N`-bit integers that own the underlying slice.
pub type BoxedBitSlice<const N: usize, Repr, Perm = Dense, Len = Dynamic> =
    PolyBitSlice<N, Repr, GlobalAllocator, Perm, Len>;

///////////////////////////////
// Special Cased Conversions //
///////////////////////////////

impl<'a, Ptr> From<&'a BitSliceBase<8, Unsigned, Ptr>> for &'a [u8]
where
    Ptr: AsPtr<Type = u8>,
{
    fn from(slice: &'a BitSliceBase<8, Unsigned, Ptr>) -> Self {
        // SAFETY: The original pointer must have been obtained from a slice of the
        // appropriate length.
        //
        // Furthermore, the layout of this type of slice is guaranteed to be identical
        // to the layout of a `[u8]`.
        unsafe { std::slice::from_raw_parts(slice.as_ptr(), slice.len()) }
    }
}

impl<'this, const NBITS: usize, Repr, Ptr, Perm, Len> Reborrow<'this>
    for BitSliceBase<NBITS, Repr, Ptr, Perm, Len>
where
    Repr: Representation<NBITS>,
    Ptr: AsPtr<Type = u8>,
    Perm: PermutationStrategy<NBITS>,
    Len: Length,
{
    type Target = BitSlice<'this, NBITS, Repr, Perm, Len>;

    fn reborrow(&'this self) -> Self::Target {
        let ptr: *const u8 = self.as_ptr();
        debug_assert!(!ptr.is_null());

        // Safety: `AsPtr` may never return null pointers.
        // The `cast_mut()` is safe because `SlicePtr` does not provide a way of retrieving
        // a mutable pointer.
        let nonnull = unsafe { NonNull::new_unchecked(ptr.cast_mut()) };

        // Safety: By struct invariant,
        // `[self.ptr(), self.ptr() + Self::bytes_for(self.len()))` is a valid slice, so
        // the returned object will also uphold these invariants.
        //
        // The returned struct will not outlive `&'this self`, so we've attached the
        // proper lifetime.
        let ptr = unsafe { SlicePtr::new_unchecked(nonnull) };

        Self::Target {
            ptr,
            len: self.len,
            repr: PhantomData,
            packing: PhantomData,
        }
    }
}

impl<'this, const NBITS: usize, Repr, Ptr, Perm, Len> ReborrowMut<'this>
    for BitSliceBase<NBITS, Repr, Ptr, Perm, Len>
where
    Repr: Representation<NBITS>,
    Ptr: AsMutPtr<Type = u8>,
    Perm: PermutationStrategy<NBITS>,
    Len: Length,
{
    type Target = MutBitSlice<'this, NBITS, Repr, Perm, Len>;

    fn reborrow_mut(&'this mut self) -> Self::Target {
        let ptr: *mut u8 = self.as_mut_ptr();
        debug_assert!(!ptr.is_null());

        // Safety: `AsMutPtr` may never return null pointers.
        let nonnull = unsafe { NonNull::new_unchecked(ptr) };

        // Safety: By struct invariant,
        // `[self.ptr(), self.ptr() + Self::bytes_for(self.len()))` is a valid slice, so
        // the returned object will also uphold these invariants.
        //
        // The returned struct will not outlive `&'this mut self`, so we've attached the
        // proper lifetime.
        //
        // Exclusive ownership is attested by both `AsMutPtr` and the mutable refernce
        // to self.
        let ptr = unsafe { MutSlicePtr::new_unchecked(nonnull) };

        Self::Target {
            ptr,
            len: self.len,
            repr: PhantomData,
            packing: PhantomData,
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        seq::{IndexedRandom, SliceRandom},
        Rng, SeedableRng,
    };

    use super::*;
    use crate::{bits::Static, test_util::AlwaysFails};

    ////////////
    // Errors //
    ////////////

    const BOUNDS: &str = "special bounds";

    #[test]
    fn test_encoding_error() {
        assert_eq!(std::mem::size_of::<EncodingError>(), 16);
        assert_eq!(
            std::mem::size_of::<Option<EncodingError>>(),
            16,
            "expected EncodingError to have the niche optimization"
        );
        let err = EncodingError::new(7, &BOUNDS);
        assert_eq!(
            err.to_string(),
            "value 7 is not in the encodable range of special bounds"
        );
    }

    // Check that a type is `Send` and `Sync`.
    fn assert_send_and_sync<T: Send + Sync>(_x: &T) {}

    ////////////
    // Binary //
    ////////////

    #[test]
    fn test_binary_repr() {
        assert_eq!(Binary::encode(-1).unwrap(), 0);
        assert_eq!(Binary::encode(1).unwrap(), 1);
        assert_eq!(Binary::decode(0), -1);
        assert_eq!(Binary::decode(1), 1);

        assert!(Binary::check(-1));
        assert!(Binary::check(1));
        assert!(!Binary::check(0));
        assert!(!Binary::check(-2));
        assert!(!Binary::check(2));

        let domain: Vec<_> = Binary::domain().collect();
        assert_eq!(domain, &[-1, 1]);
    }

    ///////////
    // Sizes //
    ///////////

    #[test]
    fn test_sizes() {
        assert_eq!(std::mem::size_of::<BitSlice<'static, 8, Unsigned>>(), 16);
        assert_eq!(std::mem::size_of::<MutBitSlice<'static, 8, Unsigned>>(), 16);

        // Ensure the borrowed slices are eligible for niche optimization.
        assert_eq!(
            std::mem::size_of::<Option<BitSlice<'static, 8, Unsigned>>>(),
            16
        );
        assert_eq!(
            std::mem::size_of::<Option<MutBitSlice<'static, 8, Unsigned>>>(),
            16
        );

        assert_eq!(
            std::mem::size_of::<BitSlice<'static, 8, Unsigned, Dense, Static<128>>>(),
            8
        );
    }

    ///////////////////
    // General Tests //
    ///////////////////

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const MAX_DIM: usize = 160;
            const FUZZ_ITERATIONS: usize = 1;
        } else if #[cfg(debug_assertions)] {
            const MAX_DIM: usize = 128;
            const FUZZ_ITERATIONS: usize = 10;
        } else {
            const MAX_DIM: usize = 256;
            const FUZZ_ITERATIONS: usize = 100;
        }
    }

    fn test_send_and_sync<const NBITS: usize, Repr, Perm>()
    where
        Repr: Representation<NBITS> + Send + Sync,
        Perm: PermutationStrategy<NBITS> + Send + Sync,
    {
        let mut x = BoxedBitSlice::<NBITS, Repr, Perm>::new_boxed(1);
        assert_send_and_sync(&x);
        assert_send_and_sync(&x.reborrow());
        assert_send_and_sync(&x.reborrow_mut());
    }

    fn test_empty<const NBITS: usize, Repr, Perm>()
    where
        Repr: Representation<NBITS>,
        Perm: PermutationStrategy<NBITS>,
    {
        let base: &mut [u8] = &mut [];
        let mut slice = MutBitSlice::<NBITS, Repr, Perm>::new(base, 0).unwrap();
        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());

        {
            let reborrow = slice.reborrow();
            assert_eq!(reborrow.len(), 0);
            assert!(reborrow.is_empty());
        }

        {
            let reborrow = slice.reborrow_mut();
            assert_eq!(reborrow.len(), 0);
            assert!(reborrow.is_empty());
        }
    }

    // times, ensuring that values are preserved.
    fn test_construction_errors<const NBITS: usize, Repr, Perm>()
    where
        Repr: Representation<NBITS>,
        Perm: PermutationStrategy<NBITS>,
    {
        let len: usize = 10;
        let bytes = Perm::bytes(len);

        // Construction errors for Boxes
        let box_big = Poly::broadcast(0u8, bytes + 1, GlobalAllocator).unwrap();
        let box_small = Poly::broadcast(0u8, bytes - 1, GlobalAllocator).unwrap();
        let box_right = Poly::broadcast(0u8, bytes, GlobalAllocator).unwrap();

        let result = BoxedBitSlice::<NBITS, Repr, Perm>::new(box_big, len);
        match result {
            Err(ConstructionError { got, expected }) => {
                assert_eq!(got, bytes + 1);
                assert_eq!(expected, bytes);
            }
            _ => panic!("shouldn't have reached here!"),
        };

        let result = BoxedBitSlice::<NBITS, Repr, Perm>::new(box_small, len);
        match result {
            Err(ConstructionError { got, expected }) => {
                assert_eq!(got, bytes - 1);
                assert_eq!(expected, bytes);
            }
            _ => panic!("shouldn't have reached here!"),
        };

        let mut base = BoxedBitSlice::<NBITS, Repr, Perm>::new(box_right, len).unwrap();
        let ptr = base.as_ptr();
        assert_eq!(base.len(), len);

        // Successful mutable reborrow and borrow.
        {
            // Use reborrow
            let borrowed = base.reborrow_mut();
            assert_eq!(borrowed.as_ptr(), ptr);
            assert_eq!(borrowed.len(), len);

            // Go through a slice.
            let borrowed = MutBitSlice::<NBITS, Repr, Perm>::new(base.as_mut_slice(), len).unwrap();
            assert_eq!(borrowed.as_ptr(), ptr);
            assert_eq!(borrowed.len(), len);
        }

        // Successful mutable borrow.
        {
            // Try constructing from an oversized slice.
            let mut oversized = vec![0; bytes + 1];
            let result = MutBitSlice::<NBITS, Repr, Perm>::new(oversized.as_mut_slice(), len);
            match result {
                Err(ConstructionError { got, expected }) => {
                    assert_eq!(got, bytes + 1);
                    assert_eq!(expected, bytes);
                }
                _ => panic!("shouldn't have reached here!"),
            };

            let mut undersized = vec![0; bytes - 1];
            let result = MutBitSlice::<NBITS, Repr, Perm>::new(undersized.as_mut_slice(), len);
            match result {
                Err(ConstructionError { got, expected }) => {
                    assert_eq!(got, bytes - 1);
                    assert_eq!(expected, bytes);
                }
                _ => panic!("shouldn't have reached here!"),
            };
        }

        // Successful const borrow and reborrow.
        {
            // Use reborrow
            let borrowed = base.reborrow();
            assert_eq!(borrowed.as_ptr(), ptr);
            assert_eq!(borrowed.len(), len);

            // Go through a slice.
            let borrowed = BitSlice::<NBITS, Repr, Perm>::new(base.as_slice(), len).unwrap();
            assert_eq!(borrowed.as_ptr(), ptr);
            assert_eq!(borrowed.len(), len);

            // Go through a mutable slice.
            let borrowed = BitSlice::<NBITS, Repr, Perm>::new(base.as_mut_slice(), len).unwrap();
            assert_eq!(borrowed.as_ptr(), ptr);
            assert_eq!(borrowed.len(), len);
        }

        // Successful mutable borrow.
        {
            // Try constructing from an oversized slice.
            let mut oversized = vec![0; bytes + 1];
            let result = BitSlice::<NBITS, Repr, Perm>::new(oversized.as_mut_slice(), len);
            match result {
                Err(ConstructionError { got, expected }) => {
                    assert_eq!(got, bytes + 1);
                    assert_eq!(expected, bytes);
                }
                _ => panic!("shouldn't have reached here!"),
            };

            let result = BitSlice::<NBITS, Repr, Perm>::new(oversized.as_slice(), len);
            match result {
                Err(ConstructionError { got, expected }) => {
                    assert_eq!(got, bytes + 1);
                    assert_eq!(expected, bytes);
                }
                _ => panic!("shouldn't have reached here!"),
            };

            // Try constructing from an undersized slice.
            let mut undersized = vec![0; bytes - 1];
            let result = BitSlice::<NBITS, Repr, Perm>::new(undersized.as_mut_slice(), len);
            match result {
                Err(ConstructionError { got, expected }) => {
                    assert_eq!(got, bytes - 1);
                    assert_eq!(expected, bytes);
                }
                _ => panic!("shouldn't have reached here!"),
            };

            let result = BitSlice::<NBITS, Repr, Perm>::new(undersized.as_slice(), len);
            match result {
                Err(ConstructionError { got, expected }) => {
                    assert_eq!(got, bytes - 1);
                    assert_eq!(expected, bytes);
                }
                _ => panic!("shouldn't have reached here!"),
            };
        }
    }

    // This series of tests writes to all indices in the vector in random orders multiple
    // times, ensuring that values are preserved.
    fn run_overwrite_test<const NBITS: usize, Perm, Len, R>(
        base: &mut BoxedBitSlice<NBITS, Unsigned, Perm, Len>,
        num_iterations: usize,
        rng: &mut R,
    ) where
        Unsigned: Representation<NBITS, Domain = RangeInclusive<i64>>,
        Len: Length,
        Perm: PermutationStrategy<NBITS>,
        R: Rng,
    {
        let mut expected: Vec<i64> = vec![0; base.len()];
        let mut indices: Vec<usize> = (0..base.len()).collect();
        for i in 0..base.len() {
            base.set(i, 0).unwrap();
        }

        for i in 0..base.len() {
            assert_eq!(base.get(i).unwrap(), 0, "failed to initialize bit vector");
        }

        let domain = base.domain();
        assert_eq!(domain, 0..=2i64.pow(NBITS as u32) - 1);
        let distribution = Uniform::new_inclusive(*domain.start(), *domain.end()).unwrap();

        for iter in 0..num_iterations {
            // Shuffle insertion order.
            indices.shuffle(rng);

            // Insert random values.
            for &i in indices.iter() {
                let value = distribution.sample(rng);
                expected[i] = value;
                base.set(i, value).unwrap();
            }

            // Make sure values are preserved.
            for (i, &expect) in expected.iter().enumerate() {
                let value = base.get(i).unwrap();
                assert_eq!(
                    value, expect,
                    "retrieval failed on iteration {iter} at index {i}"
                );
            }

            // Make sure the reborrowed version matches.
            let borrowed = base.reborrow();
            for (i, &expect) in expected.iter().enumerate() {
                let value = borrowed.get(i).unwrap();
                assert_eq!(
                    value, expect,
                    "reborrow retrieval failed on iteration {iter} at index {i}"
                );
            }
        }
    }

    fn run_overwrite_binary_test<Perm, Len, R>(
        base: &mut BoxedBitSlice<1, Binary, Perm, Len>,
        num_iterations: usize,
        rng: &mut R,
    ) where
        Len: Length,
        Perm: PermutationStrategy<1>,
        R: Rng,
    {
        let mut expected: Vec<i64> = vec![0; base.len()];
        let mut indices: Vec<usize> = (0..base.len()).collect();
        for i in 0..base.len() {
            base.set(i, -1).unwrap();
        }

        for i in 0..base.len() {
            assert_eq!(base.get(i).unwrap(), -1, "failed to initialize bit vector");
        }

        let distribution: [i64; 2] = [-1, 1];

        for iter in 0..num_iterations {
            // Shuffle insertion order.
            indices.shuffle(rng);

            // Insert random values.
            for &i in indices.iter() {
                let value = distribution.choose(rng).unwrap();
                expected[i] = *value;
                base.set(i, *value).unwrap();
            }

            // Make sure values are preserved.
            for (i, &expect) in expected.iter().enumerate() {
                let value = base.get(i).unwrap();
                assert_eq!(
                    value, expect,
                    "retrieval failed on iteration {iter} at index {i}"
                );
            }

            // Make sure the reborrowed version matches.
            let borrowed = base.reborrow();
            for (i, &expect) in expected.iter().enumerate() {
                let value = borrowed.get(i).unwrap();
                assert_eq!(
                    value, expect,
                    "reborrow retrieval failed on iteration {iter} at index {i}"
                );
            }
        }
    }

    //////////////////////
    // Unsigned - Dense //
    //////////////////////

    fn test_unsigned_dense<const NBITS: usize, Len, R>(
        len: Len,
        minimum: i64,
        maximum: i64,
        rng: &mut R,
    ) where
        Unsigned: Representation<NBITS, Domain = RangeInclusive<i64>>,
        Dense: PermutationStrategy<NBITS>,
        Len: Length,
        R: Rng,
    {
        test_send_and_sync::<NBITS, Unsigned, Dense>();
        test_empty::<NBITS, Unsigned, Dense>();
        test_construction_errors::<NBITS, Unsigned, Dense>();
        assert_eq!(Unsigned::domain_const::<NBITS>(), Unsigned::domain(),);

        match PolyBitSlice::<NBITS, Unsigned, _, Dense, Len>::new_in(len, AlwaysFails) {
            Ok(_) => {
                if len.value() != 0 {
                    panic!("zero sized allocations don't require an allocator");
                }
            }
            Err(AllocatorError) => {
                if len.value() == 0 {
                    panic!("allocation should have failed");
                }
            }
        }

        let mut base =
            PolyBitSlice::<NBITS, Unsigned, _, Dense, Len>::new_in(len, GlobalAllocator).unwrap();
        assert_eq!(
            base.len(),
            len.value(),
            "BoxedBitSlice returned the incorrect length"
        );

        let expected_bytes = BitSlice::<'static, NBITS, Unsigned>::bytes_for(len.value());
        assert_eq!(
            base.bytes(),
            expected_bytes,
            "BoxedBitSlice has the incorrect number of bytes"
        );

        // Check that the minimum and maximum values reported by the struct are correct.
        assert_eq!(base.domain(), minimum..=maximum);

        if len.value() == 0 {
            return;
        }

        let ptr = base.as_ptr();

        // Now that we know the length is non-zero, we can try testing the interface.
        // Setting the lowest index should always work.
        {
            let mut borrowed = base.reborrow_mut();

            // Make sure the pointer is preserved.
            assert_eq!(
                borrowed.as_ptr(),
                ptr,
                "pointer was not preserved during borrowing!"
            );
            assert_eq!(
                borrowed.len(),
                len.value(),
                "borrowing did not preserve length!"
            );

            borrowed.set(0, 0).unwrap();
            assert_eq!(borrowed.get(0).unwrap(), 0);

            borrowed.set(0, 1).unwrap();
            assert_eq!(borrowed.get(0).unwrap(), 1);

            borrowed.set(0, 0).unwrap();
            assert_eq!(borrowed.get(0).unwrap(), 0);

            // Setting to an invalid value should yield an error.
            let result = borrowed.set(0, minimum - 1);
            assert!(matches!(result, Err(SetError::EncodingError { .. })));

            let result = borrowed.set(0, maximum + 1);
            assert!(matches!(result, Err(SetError::EncodingError { .. })));

            // Make sure an out-of-bounds access is caught.
            let result = borrowed.set(borrowed.len(), 0);
            assert!(matches!(result, Err(SetError::IndexError { .. })));

            // Ensure that getting out-of-bounds is an error.
            let result = borrowed.get(borrowed.len());
            assert!(matches!(result, Err(GetError::IndexError { .. })));
        }

        {
            // Reconsturct the mutable borrow directly through a slice.
            let borrowed =
                MutBitSlice::<NBITS, Unsigned, Dense, Len>::new(base.as_mut_slice(), len).unwrap();

            assert_eq!(
                borrowed.as_ptr(),
                ptr,
                "pointer was not preserved during borrowing!"
            );
            assert_eq!(
                borrowed.len(),
                len.value(),
                "borrowing did not preserve length!"
            );
        }

        {
            let borrowed = base.reborrow();

            // Make sure the pointer is preserved.
            assert_eq!(
                borrowed.as_ptr(),
                ptr,
                "pointer was not preserved during borrowing!"
            );

            assert_eq!(
                borrowed.len(),
                len.value(),
                "borrowing did not preserve length!"
            );

            // Ensure that getting out-of-bounds is an error.
            let result = borrowed.get(borrowed.len());
            assert!(matches!(result, Err(GetError::IndexError { .. })));
        }

        {
            // Reconsturct the mutable borrow directly through a slice.
            let borrowed =
                BitSlice::<NBITS, Unsigned, Dense, Len>::new(base.as_slice(), len).unwrap();

            assert_eq!(
                borrowed.as_ptr(),
                ptr,
                "pointer was not preserved during borrowing!"
            );
            assert_eq!(
                borrowed.len(),
                len.value(),
                "borrowing did not preserve length!"
            );
        }

        {
            // Reconsturct the mutable borrow directly through a slice.
            let borrowed =
                BitSlice::<NBITS, Unsigned, Dense, Len>::new(base.as_mut_slice(), len).unwrap();

            assert_eq!(
                borrowed.as_ptr(),
                ptr,
                "pointer was not preserved during borrowing!"
            );
            assert_eq!(
                borrowed.len(),
                len.value(),
                "borrowing did not preserve length!"
            );
        }

        // Now we begin the testing loop.
        run_overwrite_test(&mut base, FUZZ_ITERATIONS, rng);
    }

    macro_rules! generate_unsigned_test {
        ($name:ident, $NBITS:literal, $MIN:literal, $MAX:literal, $SEED:literal) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($SEED);
                for dim in 0..MAX_DIM {
                    test_unsigned_dense::<$NBITS, Dynamic, _>(dim.into(), $MIN, $MAX, &mut rng);
                }
            }
        };
    }

    generate_unsigned_test!(test_unsigned_8bit, 8, 0, 0xff, 0xc652f2a1018f442b);
    generate_unsigned_test!(test_unsigned_7bit, 7, 0, 0x7f, 0xb732e59fec6d6c9c);
    generate_unsigned_test!(test_unsigned_6bit, 6, 0, 0x3f, 0x35d9380d0a318f21);
    generate_unsigned_test!(test_unsigned_5bit, 5, 0, 0x1f, 0xfb09895183334304);
    generate_unsigned_test!(test_unsigned_4bit, 4, 0, 0x0f, 0x38dfcf9e82c33f48);
    generate_unsigned_test!(test_unsigned_3bit, 3, 0, 0x07, 0xf9a94c8c749ee26c);
    generate_unsigned_test!(test_unsigned_2bit, 2, 0, 0x03, 0xbba03db62cecf4cf);
    generate_unsigned_test!(test_unsigned_1bit, 1, 0, 0x01, 0x54ea2a07d7c67f37);

    #[test]
    fn test_binary_dense() {
        let mut rng = StdRng::seed_from_u64(0xb3c95e8e19d3842e);
        for len in 0..MAX_DIM {
            #[cfg(miri)]
            if len != MAX_DIM - 1 {
                continue;
            }

            test_send_and_sync::<1, Binary, Dense>();
            test_empty::<1, Binary, Dense>();
            test_construction_errors::<1, Binary, Dense>();

            // Create a boxed base.
            let mut base = BoxedBitSlice::<1, Binary>::new_boxed(len);
            assert_eq!(
                base.len(),
                len,
                "BoxedBitSlice returned the incorrect length"
            );

            assert_eq!(base.bytes(), len.div_ceil(8));

            let bytes = BitSlice::<'static, 1, Binary>::bytes_for(len);
            assert_eq!(
                bytes,
                len.div_ceil(8),
                "BoxedBitSlice has the incorrect number of bytes"
            );

            if len == 0 {
                continue;
            }

            // Setting to an invalid value should yield an error.
            let result = base.set(0, 0);
            assert!(matches!(result, Err(SetError::EncodingError { .. })));

            // Make sure an out-of-bounds access is caught.
            let result = base.set(base.len(), -1);
            assert!(matches!(result, Err(SetError::IndexError { .. })));

            // Ensure that getting out-of-bounds is an error.
            let result = base.get(base.len());
            assert!(matches!(result, Err(GetError::IndexError { .. })));

            // Now we begin the testing loop.
            run_overwrite_binary_test(&mut base, FUZZ_ITERATIONS, &mut rng);
        }
    }

    #[test]
    fn test_4bit_bit_transpose() {
        let mut rng = StdRng::seed_from_u64(0xb3c95e8e19d3842e);
        for len in 0..MAX_DIM {
            #[cfg(miri)]
            if len != MAX_DIM - 1 {
                continue;
            }

            test_send_and_sync::<4, Unsigned, BitTranspose>();
            test_empty::<4, Unsigned, BitTranspose>();
            test_construction_errors::<4, Unsigned, BitTranspose>();

            // Create a boxed base.
            let mut base = BoxedBitSlice::<4, Unsigned, BitTranspose>::new_boxed(len);
            assert_eq!(
                base.len(),
                len,
                "BoxedBitSlice returned the incorrect length"
            );

            assert_eq!(base.bytes(), 32 * len.div_ceil(64));

            let bytes = BitSlice::<'static, 4, Unsigned, BitTranspose>::bytes_for(len);
            assert_eq!(
                bytes,
                32 * len.div_ceil(64),
                "BoxedBitSlice has the incorrect number of bytes"
            );

            if len == 0 {
                continue;
            }

            // Setting to an invalid value should yield an error.
            let result = base.set(0, -1);
            assert!(matches!(result, Err(SetError::EncodingError { .. })));

            // Make sure an out-of-bounds access is caught.
            let result = base.set(base.len(), -1);
            assert!(matches!(result, Err(SetError::IndexError { .. })));

            // Ensure that getting out-of-bounds is an error.
            let result = base.get(base.len());
            assert!(matches!(result, Err(GetError::IndexError { .. })));

            // Now we begin the testing loop.
            run_overwrite_test(&mut base, FUZZ_ITERATIONS, &mut rng);
        }
    }
}
