/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{
    arch,
    bitmask::{BitMask, FromInt},
    constant::{Const, SupportedLaneCount},
};

/// Rust currently lacks the ability to use a trait's associated constants as constraints
/// or constant parameters to other type definitions within the same trait.
///
/// The use of the `Const` type moves the const generic into the type domain, allowing us
/// to properly constrain and define other associated items.
///
/// In particular, we currently can't do nice things like:
/// ```ignore
/// trait MyTrait {
///     const MY_CONSTANT: usize;
///     type ArrayType = [f32; Self::MY_CONSTANT];
///                            ----------------- Currently unsupported.
/// }
/// ```
///
/// See:
/// - <https://users.rust-lang.org/t/limitation-of-associated-const-in-traits/73491/2>
/// - <https://github.com/rust-lang/rust/issues/76560>
pub trait ArrayType<T>: SupportedLaneCount {
    type Type;
}

/// Map scalar + lengths to arrays.
impl<T, const N: usize> ArrayType<T> for Const<N>
where
    Const<N>: SupportedLaneCount,
{
    type Type = [T; N];
}

/// Stable Rust does not allow expressions involving compile-time computation with
/// const generic parameters: <http://github.com/rust-lang/rust/issues/76560s>
///
/// This makes is difficult to go a const parameter defining the number of SIMD lanes
/// to an appropriately sized mask.
///
/// This helper trait provides a level of indirection to map SIMD representations to
/// the associated bitmask.
pub trait BitMaskType<A: arch::Sealed>: SupportedLaneCount {
    type Type; // should always be a `BitMask`.
}

impl<A, const N: usize> BitMaskType<A> for Const<N>
where
    Const<N>: SupportedLaneCount,
    A: arch::Sealed,
{
    type Type = BitMask<N, A>;
}

/// Convert `Self` to the SIMD type `T`. This is mainly useful when implementing fallback
/// operations through [`crate::Emulated`] to restore the original SIMD type.
pub trait AsSIMD<T>: Copy
where
    T: SIMDVector,
{
    fn as_simd(self, arch: T::Arch) -> T;
}

/// A logical mask for SIMD operations.
///
/// The representation of this type varies between architectures and micro-architectures.
/// For example:
///
/// * On AVX 2 systems, a SIMD mask for type/length pairs `(T, N)` consists of a SIMD
///   register of an unsigned integer with the same size of `T` and length `N`.
///
///   The semantics of such registers are to allow operations in lanes where the top-most
///   bit is set to 1.
///
/// * On AVX-512 systems, the story is much simpler as the masks used in that instruction
///   set are simply the correponsing bit mask.
///
///   So a mask for 8-wide operations is simply an 8-bit unsigned integer.
///
/// * Emulated systems should use a bit-mask for the most compact representation.
pub trait SIMDMask: Copy + std::fmt::Debug {
    /// The architecture type this struct belongs to.
    type Arch: arch::Sealed;

    /// The type of the underlying intrinsic.
    type Underlying: Copy + std::fmt::Debug;

    /// The bitmask associated with the logical mask.
    type BitMask: SIMDMask<Arch = Self::Arch> + Into<Self> + From<Self>;

    /// The number of lanes in the bitmask.
    const LANES: usize;

    /// Whether or not this mask implementation is a bit mask.
    const ISBITS: bool;

    /// Return the architecture object associated with this vector.
    fn arch(self) -> Self::Arch;

    /// Retrieve the underlying type.
    /// This will always be an unsigned integer of the minimum width required to contain
    /// `LANES` bits.
    fn to_underlying(self) -> Self::Underlying;

    /// Construct the mask from the underlying type.
    fn from_underlying(arch: Self::Arch, value: Self::Underlying) -> Self;

    /// Return `true` if lane `i` is set and `false` otherwise.
    ///
    /// This method is unchecked, but safe in the sense that if `i >= LANES` false will
    /// always be returned. No out of bounds access will be made, but no error indication
    /// will be provided.
    fn get_unchecked(&self, i: usize) -> bool;

    /// Efficiently construct a new mask with the first `i` bits set and the remainder
    /// set to zero.
    ///
    /// If `i >= LANES` then all bits will be set.
    fn keep_first(arch: Self::Arch, i: usize) -> Self;

    /// Return the first set index in the mask or `None` if no entries are set.
    fn first(&self) -> Option<usize> {
        self.bitmask().first()
    }

    //////////////////////////////
    // Provided Implementations //
    //////////////////////////////

    /// Return the associated BitMask for this Mask.
    fn bitmask(self) -> Self::BitMask {
        <Self::BitMask as From<Self>>::from(self)
    }

    /// Return `true` if lane `i` is set and `false` otherwise. Returns an empty `Option`
    /// if the index `i` is out-of-bounds.
    fn get(&self, i: usize) -> Option<bool> {
        if i >= Self::LANES {
            None
        } else {
            Some(self.get_unchecked(i))
        }
    }

    /// Construct a mask based on the result of invoking `f` once each element in the range
    /// `0..Self::LANES` in order.
    ///
    /// In the returned mask `m`, `m.get(0)` corresponds to the value of `f(0)`. Similarly,
    /// `m.get(1)` corresponds to `f(1)` etc.
    #[inline(always)]
    fn from_fn<F>(arch: Self::Arch, f: F) -> Self
    where
        F: FnMut(usize) -> bool,
    {
        // Recurse to BitMask.
        Self::BitMask::from_fn(arch, f).into()
    }

    /// Return `true` if any lane in the mask is set. Otherwise, return `false.
    #[inline(always)]
    fn any(self) -> bool {
        // Recurse to BitMask.
        <Self::BitMask as From<Self>>::from(self).any()
    }

    /// Return `true` if all lanes in the mask are set. Otherwise, return `false`.
    #[inline(always)]
    fn all(self) -> bool {
        // Recurse to BitMask.
        <Self::BitMask as From<Self>>::from(self).all()
    }

    /// Return `true` if all lanes in the mask are set. Otherwise, return `false`.
    #[inline(always)]
    fn none(self) -> bool {
        !self.any()
    }

    /// Return the number of lanes that evaluate to `true`.
    #[inline(always)]
    fn count(self) -> usize {
        // Recurse to BitMask.
        <Self::BitMask as From<Self>>::from(self).count()
    }
}

/// A trait representing minimal behavior for a SIMD-like vector.
///
/// A SIMDVector can be thought of as a homogeneous array `[T; N]` (with potentially
/// stricter alignment requirements) that generally behave for arithmetic purposes like
/// scalars in the sense that if
/// ```ignore
/// fn add(a: V, b: V) -> V
/// where V: SIMDVector {
///     a + b
/// }
/// ```
/// will have the same semantics of broadcasting the `+` operation across all lanes in the
/// vector.
pub trait SIMDVector: Copy + std::fmt::Debug {
    /// The architecture this vector belongs to.
    type Arch: arch::Sealed;

    /// The type of each element in the vector.
    type Scalar: Copy + std::fmt::Debug;

    /// The underlying representation.
    type Underlying: Copy;

    /// The number of lanes in the vector.
    const LANES: usize;

    /// The value of `LANES` but in the type domain so we can use it to constrain other
    /// aspects of this trait.
    ///
    /// Should be the type `Const<Self::LANES>`.
    type ConstLanes: ArrayType<Self::Scalar> + BitMaskType<Self::Arch>;

    /// The expanded logical mask representation.
    /// This may-or-may-not actually be a bitmask, but should be easily convertible to and
    /// from a bitmask.
    type Mask: SIMDMask<Arch = Self::Arch>
        + From<<Self::ConstLanes as BitMaskType<Self::Arch>>::Type>
        + Into<<Self::ConstLanes as BitMaskType<Self::Arch>>::Type>;

    /// Whether or not this is an emulated vector.
    ///
    /// Emulated vectors are backed by Rust arrays and use scalar loops to implement
    /// arithmetic operations.
    const EMULATED: bool;

    /// Return the architecture object associated with this vector.
    ///
    /// # NOTE
    ///
    /// This is safe because construction of `self` serves as the witness that we are on
    /// a compatible architecture.
    fn arch(self) -> Self::Arch;

    /// Return the default value for the type. This is always the numberic 0 for the
    /// associated scalar type.
    fn default(arch: Self::Arch) -> Self;

    /// Return the underlying type.
    fn to_underlying(self) -> Self::Underlying;

    /// Construct from the underlying type.
    fn from_underlying(arch: Self::Arch, repr: Self::Underlying) -> Self;

    /// Retrieve the contents as an array.
    fn to_array(self) -> <Self::ConstLanes as ArrayType<Self::Scalar>>::Type;

    /// Construct from the associated array.
    ///
    /// The argument `arch` provides a "proof of compatibility" as `A` can only be safely
    /// instantiated when all the requirements for the architecture are met.
    fn from_array(arch: Self::Arch, x: <Self::ConstLanes as ArrayType<Self::Scalar>>::Type)
    -> Self;

    /// Broadcast the provided scalar across all lanes.
    ///
    /// The argument `arch` provides a "proof of compatibility" as `A` can only be safely
    /// instantiated when all the requirements for the architecture are met.
    fn splat(arch: Self::Arch, value: Self::Scalar) -> Self;

    /// Return the number of lanes in this vector.
    fn num_lanes() -> usize {
        Self::LANES
    }

    /// Load `<Self as SIMDVector>::LANES` number of elements starting at the provided
    /// pointer.
    ///
    /// The alignment of `ptr` must be the same as `<Self as SIMDVector>::Scalar`, but does
    /// not need to be stricter.
    ///
    /// # Safety
    ///
    /// A contiguous read of `<Self as SIMDVector>::LANES` must touch valid memory.
    unsafe fn load_simd(arch: Self::Arch, ptr: *const <Self as SIMDVector>::Scalar) -> Self;

    /// Load `<Self as SIMDVector>::LANES` number of elements starting at the provided
    /// pointer.
    ///
    /// The alignment of `ptr` must be the same as `<Self as SIMDVector>::Scalar`, but does
    /// not need to be stricter.
    ///
    /// Entries in the mask that evaluate to `false` will not be accessed.
    /// This makes it safe to use this function with lanes masked out that would otherwise
    /// cross a page boundary or otherwise cause an out-of-bounds read.
    ///
    /// # Safety
    ///
    /// Offsets from the `ptr` where the mask evaluates to true must be dereferencable to
    /// the underlying scalar type.
    unsafe fn load_simd_masked_logical(
        arch: Self::Arch,
        ptr: *const <Self as SIMDVector>::Scalar,
        mask: <Self as SIMDVector>::Mask,
    ) -> Self;

    /// The same as `load_simd_masked_logical` but taking a BitMask instead.
    ///
    /// No load attempt will be made to lanes that are masked out.
    ///
    /// # Safety
    ///
    /// Offsets from the `ptr` where the mask evaluates to true must be dereferencable to
    /// the underlying scalar type. For implementations using the provided default, the
    /// conversion from the bitmask to the actual mask must be correct.
    #[inline(always)]
    unsafe fn load_simd_masked(
        arch: Self::Arch,
        ptr: *const <Self as SIMDVector>::Scalar,
        mask: <<Self as SIMDVector>::ConstLanes as BitMaskType<Self::Arch>>::Type,
    ) -> Self {
        // SAFETY: Bitmasks must be convertible to their corresponding logical mask.
        // When the logical mask **is** a bitbask, this is a no-op.
        unsafe { Self::load_simd_masked_logical(arch, ptr, mask.into()) }
    }

    /// The same as `load_simd_masked_logical`, but potentially specialized for situations
    /// where it is known that some number of first elements will be accessed.
    ///
    /// If `first` is greater than or equal to the number of lanes, then all lanes will be
    /// loaded.
    ///
    /// # Safety
    ///
    /// A contiguous read of `first.min(<Self as SIMDVector>::LANES)` must be valid.
    #[inline(always)]
    unsafe fn load_simd_first(
        arch: Self::Arch,
        ptr: *const <Self as SIMDVector>::Scalar,
        first: usize,
    ) -> Self {
        // SAFETY: The implementation of `SIMDMask` must be correct.
        unsafe {
            Self::load_simd_masked_logical(
                arch,
                ptr,
                <Self as SIMDVector>::Mask::keep_first(arch, first),
            )
        }
    }

    /// Store `<Self as SIMDVector>::LANES` number of elements contiguously starting at the
    /// provided pointer.
    ///
    /// The alignment of `ptr` must be the same as `<Self as SIMDVector>::Scalar`, but does
    /// not need to be stricter.
    ///
    /// # Safety
    ///
    /// The pointed-to memory must adhere to Rust's exclusive reference rules.
    ///
    /// A contiguous store of `<Self as SIMDVector>::LANES` must touch valid memory.
    unsafe fn store_simd(self, ptr: *mut <Self as SIMDVector>::Scalar);

    /// Store `<Self as SIMDVector>::LANES` number of elements starting at the provided
    /// pointer.
    ///
    /// The alignment of `ptr` must be the same as `<Self as SIMDVector>::Scalar`, but does
    /// not need to be stricter.
    ///
    /// Entries in the mask that evaluate to `false` will not be accessed.
    /// This makes it safe to use this function with lanes masked out that would otherwise
    /// cross a page boundary or otherwise cause an out-of-bounds write.
    ///
    /// # Safety
    ///
    /// The pointed-to memory must adhere to Rust's exclusive reference rules.
    ///
    /// Offsets from the `ptr` where the mask evaluates to true must be mutably
    /// dereferencable to the underlying scalar type.
    unsafe fn store_simd_masked_logical(
        self,
        ptr: *mut <Self as SIMDVector>::Scalar,
        mask: <Self as SIMDVector>::Mask,
    );

    /// The same as `load_simd_masked_logical` but taking a BitMask instead.
    ///
    /// No store attempt will be made to lanes that are masked out.
    ///
    /// # Safety
    ///
    /// The pointed-to memory must adhere to Rust's exclusive reference rules.
    ///
    /// Offsets from the `ptr` where the mask evaluates to true must be mutably
    /// dereferencable to the underlying scalar type.
    ///
    /// For implementations using the provided default, the conversion from the bitmask to
    /// the actual mask must be correct.
    #[inline(always)]
    unsafe fn store_simd_masked(
        self,
        ptr: *mut <Self as SIMDVector>::Scalar,
        mask: <<Self as SIMDVector>::ConstLanes as BitMaskType<Self::Arch>>::Type,
    ) {
        // SAFETY: Bitmasks must be convertible to their corresponding logical mask.
        // When the logical mask **is** a bitbask, this is a no-op.
        unsafe { self.store_simd_masked_logical(ptr, mask.into()) }
    }

    /// The same as `store_simd_masked_logical`, but potentially specialized for situations
    /// where it is known that some number of first elements will be accessed.
    ///
    /// If `first` is greater than or equal to the number of lanes, then all lanes will be
    /// written.
    ///
    /// # Safety
    ///
    /// The pointed-to memory must adhere to Rust's exclusive reference rules.
    ///
    /// A contiguous write of `first.min(<Self as SIMDVector>::LANES)` must be valid.
    #[inline(always)]
    unsafe fn store_simd_first(self, ptr: *mut <Self as SIMDVector>::Scalar, first: usize) {
        // SAFETY: The implementation of `SIMDMask` must be correct.
        unsafe {
            self.store_simd_masked_logical(
                ptr,
                <Self as SIMDVector>::Mask::keep_first(self.arch(), first),
            )
        }
    }

    /// Perform a numeric cast on each element, returning a new SIMD vector.
    ///
    /// See also: [`SIMDCast`].
    #[inline(always)]
    fn cast<T>(self) -> <Self as SIMDCast<T>>::Cast
    where
        Self: SIMDCast<T>,
    {
        self.simd_cast()
    }
}

/// Efficiently perform the operation
/// ```ignore
/// self * rhs + accumulator
/// ```
/// with the following semantics dependant on the associated scalar type.
///
/// * floating point: Perform a fused multiply-add, implementing the operation with only a
///   single rounding instance.
///
/// * integer: Perform the multiplication followed by the accumulation. Both binary
///   operations will be performed using wrap-around arithmetic.
pub trait SIMDMulAdd {
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self;
}

/// Efficiently retrieve the pairwise minimum or maximum for the two arguments.
///
/// Each function comes in two flavors:
///
/// * Standard (suffixed): Compute the minimum or maximum in a way that is equivalent to
///   Rust's built-in minimum or maximum functions.
///
///   When the scalar type is integral, the behavior is unambiguous.
///
///   When the scalar type is a floating point and one value of a pair is NaN, the other
///   value is returned. When the result is zero, either a positive or a negative zero can
///   be returned.
///
/// * Fast (unsuffixed): Compute the minimum or maximum using the fastest possible method
///   on the given architecture with non-standard NaN handing.
///
///   When the scalar type is integral, the behavior is the same as the standard
///   implementations.
///
///   When the scalar type is a floating point, the implementation is allowed to differ
///   with respect to NaN handling. That is, when one of the arguments is NaN, the
///   implementation is allowed to return **either** the other argument (like the standard
///   implementation) or NaN. Like the standard implementation, if the result is zero, then
///   a zero of either sign can be returned.
///
///   This method should be preferred when precise NaN handling is not needed as it can be
///   more efficient.
pub trait SIMDMinMax: Sized {
    /// Return the pairwise minimum of `self` and `rhs`, subject to looser NaN handling.
    fn min_simd(self, rhs: Self) -> Self;

    /// Return the pairwise minimum of `self` and `rhs` as if by applying the standard
    /// library's `min` method for the scalar type.
    #[inline(always)]
    fn min_simd_standard(self, rhs: Self) -> Self {
        self.min_simd(rhs)
    }

    /// Return the pairwise maximum of `self` and `rhs`, subject to looser NaN handling.
    fn max_simd(self, rhs: Self) -> Self;

    /// Return the pairwise maximum of `self` and `rhs` as if by applying the standard
    /// library's `max` method for the scalar type.
    #[inline(always)]
    fn max_simd_standard(self, rhs: Self) -> Self {
        self.max_simd(rhs)
    }
}

/// Take the absolute value of each lane.
///
/// # Notes
///
/// For signed integer types T, this works as expected for all values except for `T::MIN`,
/// in which case `T::MIN` is returned. This keeps the behavior in line with hardware
/// intrinsics.
///
/// A correct answer can be retrieved by casting the result to the equivalent unsigned
/// integer.
pub trait SIMDAbs {
    fn abs_simd(self) -> Self;
}

/// A SIMD equivalent of `std::cmp::PartialEq`.
///
/// Instead of a boolean, return `Self::Mask` containin the result of the element-wise
/// comparison of the two vectors.
pub trait SIMDPartialEq: SIMDVector {
    /// SIMD equivalent of `std::cmp::PartialEq::eq`, applying the latter trait to each
    /// lane-wise pair of elements in `self` and `other`.
    fn eq_simd(self, other: Self) -> Self::Mask;

    /// SIMD equivalent of `std::cmp::PartialEq::neq`, applying the latter trait to each
    /// lane-wise pair of elements in `self` and `other`.
    fn ne_simd(self, other: Self) -> Self::Mask;
}

/// A SIMD equaivalent of `std::cmp::PartialOrd`.
///
/// Instead of a boolean, return `Self::Mask` containing the result of the element-wise
/// comparisons of the two vectors.
pub trait SIMDPartialOrd: SIMDVector {
    /// SIMD equivalent of `std::cmp::PartialOrd::lt`.
    fn lt_simd(self, other: Self) -> Self::Mask;

    /// SIMD equivalent of `std::cmp::PartialOrd::le`.
    fn le_simd(self, other: Self) -> Self::Mask;

    //////////////////////
    // Provided Methods //
    //////////////////////

    /// SIMD equivalent of `std::cmp::PartialOrd::gt`.
    ///
    /// Types are free to override the provided method if a more efficient implementation
    /// is possible.
    #[inline(always)]
    fn gt_simd(self, other: Self) -> Self::Mask {
        other.lt_simd(self)
    }

    /// SIMD equivalent of `std::cmp::PartialOrd::ge`.
    ///
    /// Types are free to override the provided method if a more efficient implementation
    /// is possible.
    #[inline(always)]
    fn ge_simd(self, other: Self) -> Self::Mask {
        other.le_simd(self)
    }
}

/// Perform a pairwise reducing sum of all lanes in the vector and return the result as a
/// scalar.
///
/// For example, the summing pattern for a vector of 8 elements is as follows:
/// ```text
/// let v0 = [x0, x1, x2, x3, x4, x5, x6, x7];
/// let v1 = [v0[0] + v0[4], v0[1] + v0[5], v0[2] + v0[6], v0[3] + v0[7]];
/// let v2 = [v1[0] + v1[2], v1[1] + v1[3]];
/// v2[0] + v2[1]
/// ```
pub trait SIMDSumTree: SIMDVector {
    fn sum_tree(self) -> <Self as SIMDVector>::Scalar;
}

/// A vectorized "if else".
pub trait SIMDSelect<V: SIMDVector>: SIMDMask {
    fn select(self, x: V, y: V) -> V;
}

/// Optimized dot-product style accumulation.
///
/// This tries to match against intrinsics like:
///
/// * `_mm256_madd_epi16`
/// * `_mm256_dpbusd_epi32`
///
/// The gist is to perform element-wise multiplication between left and right, promoting the
/// result to the element-type of `Self`, adding adjacent entries
///
/// # Precise Enumeration of Semantics for Implementations
///
/// The semantics depend on the source and destination type, but are intended to be the same
/// for each type combination across architectures.
///
/// ## `SIMDDotProduct<i16x16> for i32x8`
///
/// 1. Perform multiplication as `i16x16 x i16x16` as if converting each lane to `i32`,
///    resulting in effectively `i32x16`. No overflow can happen.
/// 2. Add together adjacent pairs in the resulting `i32x16` to yield `i32x8`. Again, this
///    step cannot overflow.
/// 3. Add the resulting `i32x8` into `Self`, returning the result.
///
/// ## `SIMDDotProduct<u8x32, i8x32> for i32x8`
///
/// 1. Perform multiplication as `i32x32 x i32x32` as if converting each lane to `i32`,
///    resulting in effectively `i32x32`. No overflow can happen.
/// 2. Sum together consecutive groups of 4 in the resulting `i32x32` to yield `i32x8`.
/// 3. Add the resulting `i342x8` into `Self`.
///
/// The same applies when the order of `u8x32` and `i8x32` are swapped and for types that
/// are twice as wide.
///
/// The main goal of this function is to hit VNNI instructions like `_mm512_dpbusd_epi32`
/// that can do the whole operation in a single go on the `V4` architecture. Use of this
/// instruction is not recommended on non-`V4` architectures.
pub trait SIMDDotProduct<L: SIMDVector, R: SIMDVector = L> {
    /// Element wise multiply each component of `left` and `right`, promoting the
    /// intermediate results to a higher precision.
    ///
    /// Then, horizontally add together groups of the accumulated values and add the
    /// resulting sums to `self`.
    ///
    /// The size of the group depends on the relative number of lanes in `Self` and `Source`.
    ///
    /// However, it is required that `Self::num_lanes()` evenly divides `Source::num_lanes()`
    /// so that the size of each group is uniform.
    fn dot_simd(self, left: L, right: R) -> Self;
}

/// Perform a bit-cast from one SIMD type to another.
pub trait SIMDReinterpret<To: SIMDVector>: SIMDVector {
    fn reinterpret_simd(self) -> To;
}

/// Perform a numeric cast on the scalar type.
///
/// Unlike `From`, this conversion is allowed to be lossy, with similar semantics to
/// numeric casts in scalar Rust.
///
/// This is meant to model Rust's numeric conversion with the "as" operator.
pub trait SIMDCast<T>: SIMDVector {
    /// The [`SIMDVector`] type of the result.
    type Cast: SIMDVector<Scalar = T, ConstLanes = Self::ConstLanes>;
    /// Perform the cast.
    fn simd_cast(self) -> Self::Cast;
}

/// A roll-up of traits required for SIMD floating point types.
pub trait SIMDFloat:
    SIMDVector
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Sub<Output = Self>
    + SIMDMulAdd
    + SIMDMinMax
    + SIMDPartialEq
    + SIMDPartialOrd
{
}

impl<T> SIMDFloat for T where
    T: SIMDVector
        + std::ops::Add<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::ops::Sub<Output = Self>
        + SIMDMulAdd
        + SIMDMinMax
        + SIMDPartialEq
        + SIMDPartialOrd
{
}

/// A roll-up of traits required for SIMD integer types.
pub trait SIMDUnsigned:
    SIMDVector
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Shr<Output = Self>
    + std::ops::Shl<Output = Self>
    + std::ops::Shr<Self::Scalar, Output = Self>
    + std::ops::Shl<Self::Scalar, Output = Self>
    + SIMDMulAdd
    + SIMDPartialEq
    + SIMDPartialOrd
{
}

impl<T> SIMDUnsigned for T where
    T: SIMDVector
        + std::ops::Add<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::BitAnd<Output = Self>
        + std::ops::BitOr<Output = Self>
        + std::ops::BitXor<Output = Self>
        + std::ops::Shr<Output = Self>
        + std::ops::Shl<Output = Self>
        + std::ops::Shr<Self::Scalar, Output = Self>
        + std::ops::Shl<Self::Scalar, Output = Self>
        + SIMDMulAdd
        + SIMDPartialEq
        + SIMDPartialOrd
{
}

pub trait SIMDSigned: SIMDUnsigned + SIMDAbs {}
impl<T> SIMDSigned for T where T: SIMDUnsigned + SIMDAbs {}

// Since it is so difficult to work directly with generic integers, resort to using a macro
// to stamp out implementations of `SIMDMask` for `BitMask`.
//
// The argument `submask` is a bit-pattern to apply to the underlying type and is to mask
// out upper-bits of the representation for 2 and 4 bit masks.
macro_rules! impl_simd_mask_for_bitmask {
    ($N:literal, $repr:ty, $submask:expr) => {
        impl<A: arch::Sealed> SIMDMask for BitMask<$N, A> {
            type Arch = A;
            type Underlying = $repr;
            type BitMask = Self;
            const ISBITS: bool = true;
            const LANES: usize = $N;

            #[inline(always)]
            fn arch(self) -> A {
                self.get_arch()
            }

            #[inline(always)]
            fn to_underlying(self) -> Self::Underlying {
                self.0
            }

            #[inline(always)]
            fn from_underlying(arch: A, value: Self::Underlying) -> Self {
                Self::from_int(arch, value)
            }

            #[inline(always)]
            fn keep_first(arch: A, i: usize) -> Self {
                // Ensure that providing a value that is too big still yields sensible
                // results.
                let i = i.min(Self::LANES);

                // Handle 64-bit integers properly.
                // It is expected that the compiler will be able to optimize out this branch
                // for non-64-bit types.
                if Self::LANES == 64 && i == 64 {
                    return Self::from_underlying(arch, Self::Underlying::MAX);
                }

                let one: u64 = 1;
                // "as" conversion in Rust performs truncation on the integers.
                Self::from_underlying(arch, ((one << i) - one) as Self::Underlying)
            }

            #[inline(always)]
            fn get_unchecked(&self, i: usize) -> bool {
                if i >= Self::LANES {
                    false
                } else {
                    (self.0 >> i) % 2 == 1
                }
            }

            #[inline(always)]
            fn first(&self) -> Option<usize> {
                let count = self.0.trailing_zeros() as usize;
                if count >= Self::LANES {
                    None
                } else {
                    Some(count)
                }
            }

            // End of recursion functions
            fn from_fn<F>(arch: A, mut f: F) -> Self
            where
                F: FnMut(usize) -> bool,
            {
                let mut x: $repr = 0;
                for i in 0..Self::LANES {
                    if f(i) {
                        x |= (1 << i);
                    }
                }
                Self::from_underlying(arch, x)
            }

            #[inline(always)]
            fn any(self) -> bool {
                self.0 != 0
            }

            #[inline(always)]
            fn all(self) -> bool {
                let v: u64 = self.0.into();

                // We again need to handle 64-wide masks differently.
                if $N == 64 {
                    v == u64::MAX
                } else {
                    v == (1 << $N) - 1
                }
            }

            #[inline(always)]
            fn count(self) -> usize {
                // We keep the invariant that all constructors of `BitMask` must zero out
                // upper bits (for 2 and 4 width BitMasks).
                self.0.count_ones() as usize
            }
        }

        // Transformation from bitmask to integer.
        impl From<BitMask<$N>> for $repr {
            fn from(value: BitMask<$N>) -> Self {
                value.to_underlying()
            }
        }
    };
}

// Stamp out a bunch of implementations.
impl_simd_mask_for_bitmask!(1, u8, 0x1);
impl_simd_mask_for_bitmask!(2, u8, 0x3);
impl_simd_mask_for_bitmask!(4, u8, 0xf);
impl_simd_mask_for_bitmask!(8, u8, u8::MAX);
impl_simd_mask_for_bitmask!(16, u16, u16::MAX);
impl_simd_mask_for_bitmask!(32, u32, u32::MAX);
impl_simd_mask_for_bitmask!(64, u64, u64::MAX);

#[cfg(test)]
mod test_traits {
    use rand::{
        SeedableRng,
        distr::{Distribution, StandardUniform},
        rngs::StdRng,
    };

    use super::*;
    use crate::{
        ARCH, arch,
        splitjoin::{LoHi, SplitJoin},
        test_utils,
    };

    // Allow unsigned 128-bit integers to be converted to narrow types.
    trait FromU128 {
        fn from_(value: u128) -> Self;
    }

    impl FromU128 for u8 {
        fn from_(value: u128) -> Self {
            value as u8
        }
    }
    impl FromU128 for u16 {
        fn from_(value: u128) -> Self {
            value as u16
        }
    }
    impl FromU128 for u32 {
        fn from_(value: u128) -> Self {
            value as u32
        }
    }
    impl FromU128 for u64 {
        fn from_(value: u128) -> Self {
            value as u64
        }
    }

    /// Test that bitmasks faithfully implement the trait `SIMDMask`.
    ///
    /// Since conversion between `u128` and arbitrary generic parameters `T` are now
    /// allowed, we take a conversion function to do this for us with all the known type
    /// information.
    fn test_bitmask_impl<const N: usize, T>()
    where
        Const<N>: SupportedLaneCount, // this value of `N` has a bitmask representation.
        T: std::fmt::Debug + std::cmp::Eq + FromU128 + From<BitMask<N, arch::Current>>,
        BitMask<N, arch::Current>: SIMDMask<Arch = arch::Current, Underlying = T>,
    {
        const MAXLEN: usize = 64;
        assert_eq!(N, BitMask::<N, arch::Current>::LANES);

        // The bit-mask corresponding to all lanes.
        let one = 1_u128;

        let all: u128 = (one << N) - one;

        for i in 0..=MAXLEN {
            let mask = BitMask::<N, arch::Current>::keep_first(arch::current(), i);

            let expected: u128 = ((one << i) - one) & all;

            // Cannot use "as" since T is not known to be a primitive type ...
            assert_eq!(mask.to_underlying(), T::from_(expected));
            assert_eq!(T::from_(expected), mask.into());
            for j in 0..=MAXLEN {
                let b = mask.get_unchecked(j);
                let o = mask.get(j);

                let expected: bool = j < i;
                if j < N {
                    assert_eq!(b, expected);
                    assert_eq!(o.unwrap(), expected);
                } else {
                    assert!(!b);
                    assert!(o.is_none());
                }
            }

            // Check reductions.
            if i == 0 {
                assert!(!mask.any());
                assert!(!mask.all());
                assert!(mask.none());
            } else if i >= N {
                assert!(mask.any());
                assert!(mask.all());
                assert!(!mask.none());
            } else {
                assert!(mask.any());
                assert!(!mask.all());
                assert!(!mask.none());
            }
        }
    }

    #[test]
    fn test_bitmask() {
        test_bitmask_impl::<1, u8>();
        test_bitmask_impl::<2, u8>();
        test_bitmask_impl::<4, u8>();
        test_bitmask_impl::<8, u8>();
        test_bitmask_impl::<16, u16>();
        test_bitmask_impl::<32, u32>();
        test_bitmask_impl::<64, u64>();
    }

    fn test_bitmask_splitjoin_impl<const N: usize, const NHALF: usize>(ntrials: usize, seed: u64)
    where
        Const<N>: SupportedLaneCount,
        Const<NHALF>: SupportedLaneCount,
        BitMask<N, arch::Current>:
            SIMDMask<Arch = arch::Current> + SplitJoin<Halved = BitMask<NHALF, arch::Current>>,
        BitMask<NHALF, arch::Current>: SIMDMask<Arch = arch::Current>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..ntrials {
            let base = BitMask::<N>::from_fn(ARCH, |_| StandardUniform {}.sample(&mut rng));
            let LoHi { lo, hi } = base.split();

            for i in 0..NHALF {
                assert_eq!(base.get(i).unwrap(), lo.get(i).unwrap());
            }

            for i in 0..NHALF {
                assert_eq!(base.get(i + NHALF).unwrap(), hi.get(i).unwrap());
            }

            let joined = BitMask::<N>::join(LoHi::new(lo, hi));
            bitmasks_equal(base, joined);
        }
    }

    #[test]
    fn test_bitmask_splitjoin() {
        test_bitmask_splitjoin_impl::<2, 1>(100, 0xcbdbdca310caec88);
        test_bitmask_splitjoin_impl::<4, 2>(100, 0x9c8b9b6c70d941c5);
        test_bitmask_splitjoin_impl::<8, 4>(100, 0xc81a25918b683d39);
        test_bitmask_splitjoin_impl::<16, 8>(50, 0xad045b437c3fa0cc);
        test_bitmask_splitjoin_impl::<32, 16>(50, 0xe710ccdbbd329c77);
        test_bitmask_splitjoin_impl::<64, 32>(25, 0xd6697e3c534fc134);
    }

    // Explicit tests to ensure that upper bits are masked out during construction of
    // 2 and 4 bit masks.
    #[test]
    fn test_zeroing() {
        let b = BitMask::<2>::from_underlying(arch::current(), 0xff);
        assert_eq!(b.to_underlying(), 0x3);
        assert_eq!(b.count(), 2);

        let b = BitMask::<4>::from_underlying(arch::current(), 0xff);
        assert_eq!(b.to_underlying(), 0xf);
        assert_eq!(b.count(), 4);
    }

    fn bitmasks_equal<const N: usize>(x: BitMask<N, arch::Current>, y: BitMask<N, arch::Current>)
    where
        Const<N>: SupportedLaneCount,
        BitMask<N, arch::Current>: SIMDMask,
    {
        assert_eq!(x.0, y.0);
    }

    // A helper macro to run a BitMask through the SIMDMask test routines.
    macro_rules! test_simdmask {
        ($N:literal) => {
            paste::paste! {
                #[test]
                fn [<test_simd_mask_ $N>]() {
                    let arch = arch::current();
                    test_utils::mask::test_keep_first::<BitMask<$N, arch::Current>, $N, _, _>(
                        arch,
                        bitmasks_equal
                    );
                    test_utils::mask::test_from_fn::<BitMask<$N, arch::Current>, $N, _, _>(
                        arch,
                        bitmasks_equal
                    );
                    test_utils::mask::test_reductions::<BitMask<$N, arch::Current>, $N, _, _>(
                        arch,
                        bitmasks_equal
                    );
                    test_utils::mask::test_first::<BitMask<$N, arch::Current>, $N, _, _>(
                        arch,
                        bitmasks_equal
                    );
                }
            }
        };
    }

    test_simdmask!(2);
    test_simdmask!(4);
    test_simdmask!(8);
    test_simdmask!(16);
    test_simdmask!(32);
    test_simdmask!(64);
}
