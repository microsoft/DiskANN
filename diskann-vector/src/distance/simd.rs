/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::convert::AsRef;

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

#[cfg(not(target_arch = "aarch64"))]
use diskann_wide::SIMDDotProduct;
use diskann_wide::{
    arch::Scalar, Architecture, Const, Constant, Emulated, SIMDAbs, SIMDMulAdd, SIMDSumTree,
    SIMDVector,
};

use crate::Half;

/// A helper trait to allow integer to f32 conversion (which may be lossy).
pub trait LossyF32Conversion: Copy {
    fn as_f32_lossy(self) -> f32;
}

impl LossyF32Conversion for f32 {
    fn as_f32_lossy(self) -> f32 {
        self
    }
}

impl LossyF32Conversion for i32 {
    fn as_f32_lossy(self) -> f32 {
        self as f32
    }
}

cfg_if::cfg_if! {
    if #[cfg(miri)] {
        fn force_eval(_x: f32) {}
    } else if #[cfg(target_arch = "x86_64")] {
        use std::arch::asm;

        /// Force the evaluation of the argument, preventing the compiler from reordering the
        /// computation of `x` behind a condition.
        ///
        /// In the context of Cosine similarity, this can help code generation for
        /// static-dimensional kernels
        #[inline(always)]
        fn force_eval(x: f32) {
            // SAFETY: This function executes no instructions. As such, it satisfies the long
            // list of requirements for inline assembly.
            //
            // See: https://doc.rust-lang.org/reference/inline-assembly.html#rules-for-inline-assembly
            unsafe {
                asm!(
                    // Assembly comment to "use" the argument
                    "/* {0} */",
                    // Use an `xmm_reg` since LLVM almost always uses such a register for
                    // scalar floating point
                    in(xmm_reg) x,
                    // Explanation:
                    // * `nostack`: This function does not touch the stack, so the compiler
                    //   does not need to worry that the stack gets messed up.
                    // * `nomem`: This function does not touch memory. The compiler doesn't
                    //   have to reload any values.
                    // * `preserves_flags`: This function preserves architectural condition
                    //   flags. We can make this guarantee because this function literally
                    //   does nothing.
                    options(nostack, nomem, preserves_flags)
                )
            }
        }
    } else {
        // Fallback implementation.
        fn force_eval(_x: f32) {}
    }
}

/// A utility struct to help with SIMD loading.
///
/// The main loop of SIMD kernels consists of various tilings of loads and arithmetic.
/// Outside of the epilogue, these loads are all full-width vector loads.
///
/// To aid in defining different tilings, this struct takes the base pointers for left and
/// right hand pointers and provides a `load` method to extract full vectors for both
/// the left and right-hand sides.
///
/// This works in conjunction with the [`SIMDSchema`] to help write unrolled loops.
#[derive(Debug, Clone, Copy)]
pub struct Loader<Schema, Left, Right, A>
where
    Schema: SIMDSchema<Left, Right, A>,
    A: Architecture,
{
    arch: A,
    schema: Schema,
    left: *const Left,
    right: *const Right,
    len: usize,
}

impl<Schema, Left, Right, A> Loader<Schema, Left, Right, A>
where
    Schema: SIMDSchema<Left, Right, A>,
    A: Architecture,
{
    /// Construct a new loader for the left and right hand pointers.
    ///
    /// Requires that the memory ranges `[left, left + len)` and `[right, right + len)` are
    /// both valid, where `len` is the *number* of the elements of type `T` and `U`.
    #[inline(always)]
    fn new(arch: A, schema: Schema, left: *const Left, right: *const Right, len: usize) -> Self {
        Self {
            arch,
            schema,
            left,
            right,
            len,
        }
    }

    /// Return the underlying architecture.
    #[inline(always)]
    fn arch(&self) -> A {
        self.arch
    }

    /// Return the SIMD Schema.
    #[inline(always)]
    fn schema(&self) -> Schema {
        self.schema
    }

    /// Load full width vectors for the left and right hand memory spans.
    ///
    /// This loads a [`SIMDSchema::SIMDWidth`] chunk of data using the following formula:
    ///
    /// ```text
    /// // The number of elements in an unrolled [`MainLoop`].
    /// let simd_width = Schema::SIMDWidth::value();
    /// let block_size = simd_width * Schema::Main::BLOCK_SIZE;
    ///
    /// load(px + block_size * block + simd_width * offset);
    /// ```
    ///
    /// # Safety
    ///
    /// Requires that the following memory addresses are in-bounds (i.e., the highest
    /// read address is at an offset less than `len`):
    ///
    /// ```text
    /// [
    ///     px + block_size * block + simd_width * offset,
    ///     px + block_size * block + simd_width * (offset + 1)
    /// )
    ///
    /// [
    ///     py + block_size * block + simd_width * offset,
    ///     py + block_size * block + simd_width * (offset + 1)
    /// )
    /// ```
    ///
    /// This invariant is checked in debug builds.
    #[inline(always)]
    unsafe fn load(&self, block: usize, offset: usize) -> (Schema::Left, Schema::Right) {
        let stride = Schema::SIMDWidth::value();
        let block_stride = stride * Schema::Main::BLOCK_SIZE;
        let offset = block_stride * block + stride * offset;

        debug_assert!(
            offset + stride <= self.len,
            "length = {}, offset = {}",
            self.len,
            offset
        );

        (
            Schema::Left::load_simd(self.arch, self.left.add(offset)),
            Schema::Right::load_simd(self.arch, self.right.add(offset)),
        )
    }
}

/// A representation of the main unrolled-loop for SIMD kernels.
pub trait MainLoop {
    /// The effective number of unrolling (in terms of SIMD vectors) performed by this
    /// kernel. For example, if `BLOCK_SIZE = 4` and the SIMD width is 8, than each iteration
    /// of the main loop will process `4 * 8 = 32` elements.
    ///
    /// This parameter will be used to compute the number of full-width epilogues that need
    /// to be executed.
    const BLOCK_SIZE: usize;

    /// Perform the main unrolled loops of a SIMD kernel. This loop is expected to process
    /// all elements in the range `[0, trip_count * S::get_simd_width() * Self::BLOCK_SIZE)`
    /// and return an accumulator consisting of the result.
    ///
    /// # Arguments
    ///
    /// * `loader`: A SIMD loader to emit loads to the two source spans.
    /// * `trip_count`: The number of blocks of size `BLOCK_SIZE` to process. A single "trip"
    ///   will process `S::get_simd_width() * Self::BLOCK_SIZE` elements. So, computation of
    ///   `trip_count` should be computed as:
    ///   ```math
    ///   let trip_count = len / (S::get_simd_width() * <_ as MainLoop>::BLOCK_SIZE);
    ///   ```
    /// * `epilogues`: The number of `S::get_simd_width()` vectors remaining after all the
    ///   main blocks have been processed. This is guaranteed to be less than
    ///   `Self::BLOCK_SIZE`.
    ///
    /// # Safety
    ///
    /// All elements in the accessed range must be valid. The memory addresses touched are
    ///
    /// ```text
    /// let block_size = Self::BLOCK_SIZE;
    /// let simd_width = S::get_simd_width();
    /// [
    ///     loader.left,
    ///     loader.left + trip_count * simd_width + block_size + epilogues * simd_width
    /// )
    /// [
    ///     loader.right,
    ///     loader.right + trip_count * simd_width + block_size + epilogues * simd_width
    /// )
    /// ```
    ///
    /// The `loader` will ensure that all accesses are in-bounds in debug builds.
    unsafe fn main<S, L, R, A>(
        loader: &Loader<S, L, R, A>,
        trip_count: usize,
        epilogues: usize,
    ) -> S::Accumulator
    where
        A: Architecture,
        S: SIMDSchema<L, R, A>;
}
/// An inner loop implementation strategy using 1 parallel instances of the schema
/// accumulator with a manual inner loop unroll of 1.
pub struct Strategy1x1;

/// An inner loop implementation strategy using 2 parallel instances of the schema
/// accumulator with a manual inner loop unroll of 1.
pub struct Strategy2x1;

/// An inner loop implementation strategy using 4 parallel instances of the schema
/// accumulator with a manual inner loop unroll of 1.
pub struct Strategy4x1;

/// An inner loop implementation strategy using 4 parallel instances of the schema
/// accumulator with a manual inner loop unroll of 2.
pub struct Strategy4x2;

/// An inner loop implementation strategy using 2 parallel instances of the schema
/// accumulator with a manual inner loop unroll of 4.
pub struct Strategy2x4;

impl MainLoop for Strategy1x1 {
    const BLOCK_SIZE: usize = 1;

    #[inline(always)]
    unsafe fn main<S, L, R, A>(
        loader: &Loader<S, L, R, A>,
        trip_count: usize,
        _epilogues: usize,
    ) -> S::Accumulator
    where
        A: Architecture,
        S: SIMDSchema<L, R, A>,
    {
        let arch = loader.arch();
        let schema = loader.schema();

        let mut s0 = schema.init(arch);
        for i in 0..trip_count {
            s0 = schema.accumulate_tuple(s0, loader.load(i, 0));
        }

        s0
    }
}

impl MainLoop for Strategy2x1 {
    const BLOCK_SIZE: usize = 2;

    #[inline(always)]
    unsafe fn main<S, L, R, A>(
        loader: &Loader<S, L, R, A>,
        trip_count: usize,
        epilogues: usize,
    ) -> S::Accumulator
    where
        A: Architecture,
        S: SIMDSchema<L, R, A>,
    {
        let arch = loader.arch();
        let schema = loader.schema();

        let mut s0 = schema.init(arch);
        let mut s1 = schema.init(arch);

        for i in 0..trip_count {
            s0 = schema.accumulate_tuple(s0, loader.load(i, 0));
            s1 = schema.accumulate_tuple(s1, loader.load(i, 1));
        }

        let mut s = schema.combine(s0, s1);
        if epilogues != 0 {
            s = schema.accumulate_tuple(s, loader.load(trip_count, 0));
        }

        s
    }
}

impl MainLoop for Strategy4x1 {
    const BLOCK_SIZE: usize = 4;

    #[inline(always)]
    unsafe fn main<S, L, R, A>(
        loader: &Loader<S, L, R, A>,
        trip_count: usize,
        epilogues: usize,
    ) -> S::Accumulator
    where
        A: Architecture,
        S: SIMDSchema<L, R, A>,
    {
        let arch = loader.arch();
        let schema = loader.schema();

        let mut s0 = schema.init(arch);
        let mut s1 = schema.init(arch);
        let mut s2 = schema.init(arch);
        let mut s3 = schema.init(arch);

        for i in 0..trip_count {
            s0 = schema.accumulate_tuple(s0, loader.load(i, 0));
            s1 = schema.accumulate_tuple(s1, loader.load(i, 1));
            s2 = schema.accumulate_tuple(s2, loader.load(i, 2));
            s3 = schema.accumulate_tuple(s3, loader.load(i, 3));
        }

        if epilogues >= 1 {
            s0 = schema.accumulate_tuple(s0, loader.load(trip_count, 0));
        }

        if epilogues >= 2 {
            s1 = schema.accumulate_tuple(s1, loader.load(trip_count, 1));
        }

        if epilogues >= 3 {
            s2 = schema.accumulate_tuple(s2, loader.load(trip_count, 2));
        }

        schema.combine(schema.combine(s0, s1), schema.combine(s2, s3))
    }
}

impl MainLoop for Strategy4x2 {
    const BLOCK_SIZE: usize = 4;

    #[inline(always)]
    unsafe fn main<S, L, R, A>(
        loader: &Loader<S, L, R, A>,
        trip_count: usize,
        epilogues: usize,
    ) -> S::Accumulator
    where
        A: Architecture,
        S: SIMDSchema<L, R, A>,
    {
        let arch = loader.arch();
        let schema = loader.schema();

        let mut s0 = schema.init(arch);
        let mut s1 = schema.init(arch);
        let mut s2 = schema.init(arch);
        let mut s3 = schema.init(arch);

        for i in 0..(trip_count / 2) {
            let j = 2 * i;
            s0 = schema.accumulate_tuple(s0, loader.load(j, 0));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 1));
            s2 = schema.accumulate_tuple(s2, loader.load(j, 2));
            s3 = schema.accumulate_tuple(s3, loader.load(j, 3));

            s0 = schema.accumulate_tuple(s0, loader.load(j, 4));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 5));
            s2 = schema.accumulate_tuple(s2, loader.load(j, 6));
            s3 = schema.accumulate_tuple(s3, loader.load(j, 7));
        }

        if !trip_count.is_multiple_of(2) {
            // Will not underflow because `trip_count` is odd.
            let j = trip_count - 1;
            s0 = schema.accumulate_tuple(s0, loader.load(j, 0));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 1));
            s2 = schema.accumulate_tuple(s2, loader.load(j, 2));
            s3 = schema.accumulate_tuple(s3, loader.load(j, 3));
        }

        if epilogues >= 1 {
            s0 = schema.accumulate_tuple(s0, loader.load(trip_count, 0));
        }

        if epilogues >= 2 {
            s1 = schema.accumulate_tuple(s1, loader.load(trip_count, 1));
        }

        if epilogues >= 3 {
            s2 = schema.accumulate_tuple(s2, loader.load(trip_count, 2));
        }

        schema.combine(schema.combine(s0, s1), schema.combine(s2, s3))
    }
}

impl MainLoop for Strategy2x4 {
    const BLOCK_SIZE: usize = 4;

    /// The implementation here has a global unroll of 4, but the unroll factor of the main
    /// loop is actually 8.
    ///
    /// There is a single peeled iteration at the end that handles the last group of 4
    /// if needed.
    #[inline(always)]
    unsafe fn main<S, L, R, A>(
        loader: &Loader<S, L, R, A>,
        trip_count: usize,
        epilogues: usize,
    ) -> S::Accumulator
    where
        A: Architecture,
        S: SIMDSchema<L, R, A>,
    {
        let arch = loader.arch();
        let schema = loader.schema();

        let mut s0 = schema.init(arch);
        let mut s1 = schema.init(arch);

        for i in 0..(trip_count / 2) {
            let j = 2 * i;
            s0 = schema.accumulate_tuple(s0, loader.load(j, 0));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 1));
            s0 = schema.accumulate_tuple(s0, loader.load(j, 2));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 3));

            s0 = schema.accumulate_tuple(s0, loader.load(j, 4));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 5));
            s0 = schema.accumulate_tuple(s0, loader.load(j, 6));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 7));
        }

        if !trip_count.is_multiple_of(2) {
            let j = trip_count - 1;
            s0 = schema.accumulate_tuple(s0, loader.load(j, 0));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 1));
            s0 = schema.accumulate_tuple(s0, loader.load(j, 2));
            s1 = schema.accumulate_tuple(s1, loader.load(j, 3));
        }

        if epilogues >= 1 {
            s0 = schema.accumulate_tuple(s0, loader.load(trip_count, 0));
        }

        if epilogues >= 2 {
            s1 = schema.accumulate_tuple(s1, loader.load(trip_count, 1));
        }

        if epilogues >= 3 {
            s0 = schema.accumulate_tuple(s0, loader.load(trip_count, 2));
        }

        schema.combine(s0, s1)
    }
}

/// An interface trait for SIMD operations.
///
/// Patterns like unrolling, pointer arithmetic, and epilogue handling are common across
/// many different combinations of left and right hand types for distance computations.
///
/// This higher level handling is delegated to functions like `simd_op`, which in turn
/// uses a `SIMDSchema` to customize the mechanics of loading and accumulation.
pub trait SIMDSchema<T, U, A: Architecture = diskann_wide::arch::Current>: Copy {
    /// The desired SIMD read width.
    /// Reads from the input slice will be use this stride when accessing memory.
    type SIMDWidth: Constant<Type = usize>;

    /// The type used to represent partial accumulated values.
    type Accumulator: std::ops::Add<Output = Self::Accumulator> + std::fmt::Debug + Copy;

    /// The type used for the left-hand side.
    type Left: SIMDVector<Arch = A, Scalar = T, ConstLanes = Self::SIMDWidth>;

    /// The type used for the right-hand side.
    type Right: SIMDVector<Arch = A, Scalar = U, ConstLanes = Self::SIMDWidth>;

    /// The final return type.
    /// This is often `f32` for complete distance functions, but need not always be.
    type Return;

    /// The implementation of the main loop.
    type Main: MainLoop;

    /// Initialize an empty (identity) accumulator.
    fn init(&self, arch: A) -> Self::Accumulator;

    /// Perform an accumulation.
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator;

    /// Combine two independent accumulators (allows for unrolling).
    #[inline(always)]
    fn combine(&self, x: Self::Accumulator, y: Self::Accumulator) -> Self::Accumulator {
        x + y
    }

    /// A supplied trait for dealing with non-full-width epilogues.
    /// Often, masked based loading will do the right thing, but for architectures like AVX2
    /// that have limited support for masking 8 and 16-bit operations, using a scalar
    /// fallback may just be better.
    ///
    /// This provides a customization point to enable a scalar fallback.
    ///
    /// # Safety
    ///
    /// * Both pointers `x` and `y` must point to memory.
    /// * It must be safe to read `len` contiguous items of type `T` starting at `x` and
    ///   `len` contiguous items of type `U` starting at `y`.
    ///
    /// The following guarantee is made:
    ///
    /// * No read will be emitted to memory locations at and after `x.add(len)` and
    ///   `y.add(len)`.
    #[inline(always)]
    unsafe fn epilogue(
        &self,
        arch: A,
        x: *const T,
        y: *const U,
        len: usize,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        // SAFETY: Performing this read is safe by the safety preconditions of `epilogue`.
        // Guarentee: The load implementation must be correct.
        let a = Self::Left::load_simd_first(arch, x, len);

        // SAFETY: Performing this read is safe by the safety preconditions of `epilogue`.
        // Guarentee: The load implementation must be correct.
        let b = Self::Right::load_simd_first(arch, y, len);
        self.accumulate(a, b, acc)
    }

    /// Perform a reduction on the accumulator to yield the final result.
    ///
    /// This will be called at the end of distance processing.
    fn reduce(&self, x: Self::Accumulator) -> Self::Return;

    /// !! Do not extend this function !!
    ///
    /// Due to limitations on how associated constants can be used, we need a function
    /// to access the SIMD width and rely on the compiler to constant propagate the result.
    #[inline(always)]
    fn get_simd_width() -> usize {
        Self::SIMDWidth::value()
    }

    /// !! Do not extend this function !!
    ///
    /// Due to limitations on how associated constants can be used, we need a function
    /// to access the unroll factor of the main loop and rely on the compiler to constant
    /// propagate the result.
    #[inline(always)]
    fn get_main_bocksize() -> usize {
        Self::Main::BLOCK_SIZE
    }

    /// A helper method to access [`Self::accumulate`] in a way that is immediately
    /// compatible with [`Loader::load`].
    #[doc(hidden)]
    #[inline(always)]
    fn accumulate_tuple(
        &self,
        acc: Self::Accumulator,
        (x, y): (Self::Left, Self::Right),
    ) -> Self::Accumulator {
        self.accumulate(x, y, acc)
    }
}

/// In some contexts - it can be beneficial to begin a computation on one pair of slices and
/// then store intermediate state for resumption on another pair of slices.
///
/// A good example of this is direct-computation of PQ distances where different chunks need
/// to be gathered and partially accumulated before the final reduction.
///
/// The `ResumableSchema` provides a relatively straight-forward way of achieving this.
pub trait ResumableSIMDSchema<T, U, A = diskann_wide::arch::Current>: Copy
where
    A: Architecture,
{
    // The associated type for this function that is non-reentrant.
    type NonResumable: SIMDSchema<T, U, A> + Default;
    type FinalReturn;

    fn init(arch: A) -> Self;
    fn combine_with(&self, other: <Self::NonResumable as SIMDSchema<T, U, A>>::Accumulator)
        -> Self;
    fn sum(&self) -> Self::FinalReturn;
}

#[derive(Debug, Clone, Copy)]
pub struct Resumable<T>(T);

impl<T> Resumable<T> {
    pub fn new(val: T) -> Self {
        Self(val)
    }

    pub fn consume(self) -> T {
        self.0
    }
}

impl<T, U, R, A> SIMDSchema<T, U, A> for Resumable<R>
where
    A: Architecture,
    R: ResumableSIMDSchema<T, U, A>,
{
    type SIMDWidth = <R::NonResumable as SIMDSchema<T, U, A>>::SIMDWidth;
    type Accumulator = <R::NonResumable as SIMDSchema<T, U, A>>::Accumulator;
    type Left = <R::NonResumable as SIMDSchema<T, U, A>>::Left;
    type Right = <R::NonResumable as SIMDSchema<T, U, A>>::Right;
    type Return = Self;
    type Main = <R::NonResumable as SIMDSchema<T, U, A>>::Main;

    fn init(&self, arch: A) -> Self::Accumulator {
        R::NonResumable::default().init(arch)
    }

    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        R::NonResumable::default().accumulate(x, y, acc)
    }

    fn combine(&self, x: Self::Accumulator, y: Self::Accumulator) -> Self::Accumulator {
        R::NonResumable::default().combine(x, y)
    }

    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        Self(self.0.combine_with(x))
    }
}

#[inline(never)]
#[allow(clippy::panic)]
fn emit_length_error(xlen: usize, ylen: usize) -> ! {
    panic!(
        "lengths must be equal, instead got: xlen = {}, ylen = {}",
        xlen, ylen
    )
}

/// A SIMD executor for binary ops using the provided `SIMDSchema`.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
#[inline(always)]
pub fn simd_op<L, R, S, T, U, A>(schema: &S, arch: A, x: T, y: U) -> S::Return
where
    A: Architecture,
    T: AsRef<[L]>,
    U: AsRef<[R]>,
    S: SIMDSchema<L, R, A>,
{
    let x: &[L] = x.as_ref();
    let y: &[R] = y.as_ref();

    let len = x.len();

    // The two lengths of the vectors must be the same.
    // Eventually - it will probably be worth looking into various wrapper functions for
    // `simd_op` that perform this checking, but for now, consider providing two
    // different-length slices as a hard program bug.
    //
    // N.B.: Redirect through `emit_length_error` to keep code generation as clean as
    // possible.
    if len != y.len() {
        emit_length_error(len, y.len());
    }
    let px = x.as_ptr();
    let py = y.as_ptr();

    // N.B.: Due to limitations in Rust's handling of const generics (and outer type
    // parameters), we cannot just reach into `S` and pull out the constant SIMDWidth.
    //
    // Instead, we need to go through a helper function. Since associated functions cannot
    // be marked as `const`, we cannot require that the extracted width is evaluated at
    // compile time.
    //
    // HOWEVER, compilers are very good at optimizing these kinds of patterns and
    // recognizing that this value is indeed constant and optimizing accordingly.
    let simd_width: usize = S::get_simd_width();
    let unroll: usize = S::get_main_bocksize();

    let trip_count = len / (simd_width * unroll);
    let epilogues = (len - simd_width * unroll * trip_count) / simd_width;

    // Create a loader that (in debug mode) will check that all of our full-width accesses
    // are in-bounds.
    let loader: Loader<S, L, R, A> = Loader::new(arch, *schema, px, py, len);

    // SAFETY: The value of `trip_count`  and `epilogues` so
    // `[0, trip_count * simd_width * unroll + epilogues * simd_width)` is in-bounds,
    // satifying the requirements of `main`.
    let mut s0 = unsafe { <S as SIMDSchema<L, R, A>>::Main::main(&loader, trip_count, epilogues) };

    let remainder = len % simd_width;
    if remainder != 0 {
        let i = len - remainder;

        // SAFETY: We have ensured that the lengths of the two inputs are the same.
        //
        // Furthermore, preceding computations on the induction variable mean that the
        // remaining memory must be valid.
        s0 = unsafe { schema.epilogue(arch, px.add(i), py.add(i), remainder, s0) };
    }

    schema.reduce(s0)
}

/////
///// L2 Implementations
/////

// A pure L2 distance function that provides a final reduction.
#[derive(Debug, Default, Clone, Copy)]
pub struct L2;

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<f32, f32, V4> for L2 {
    type SIMDWidth = Const<8>;
    type Accumulator = <V4 as Architecture>::f32x8;
    type Left = <V4 as Architecture>::f32x8;
    type Right = <V4 as Architecture>::f32x8;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let c = x - y;
        c.mul_add_simd(c, acc)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<f32, f32, V3> for L2 {
    type SIMDWidth = Const<8>;
    type Accumulator = <V3 as Architecture>::f32x8;
    type Left = <V3 as Architecture>::f32x8;
    type Right = <V3 as Architecture>::f32x8;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let c = x - y;
        c.mul_add_simd(c, acc)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

impl SIMDSchema<f32, f32, Scalar> for L2 {
    type SIMDWidth = Const<4>;
    type Accumulator = Emulated<f32, 4>;
    type Left = Emulated<f32, 4>;
    type Right = Emulated<f32, 4>;
    type Return = f32;
    type Main = Strategy2x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        // Don't assume the presence of FMA.
        let c = x - y;
        (c * c) + acc
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        arch: Scalar,
        x: *const f32,
        y: *const f32,
        len: usize,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let mut s: f32 = 0.0;
        for i in 0..len {
            // SAFETY: The range `[x, x.add(len))` is valid for reads.
            let vx = unsafe { x.add(i).read() };
            // SAFETY: The range `[y, y.add(len))` is valid for reads.
            let vy = unsafe { y.add(i).read() };
            let d = vx - vy;
            s += d * d;
        }
        acc + Self::Accumulator::from_array(arch, [s, 0.0, 0.0, 0.0])
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<Half, Half, V4> for L2 {
    type SIMDWidth = Const<8>;
    type Accumulator = <V4 as Architecture>::f32x8;
    type Left = <V4 as Architecture>::f16x8;
    type Right = <V4 as Architecture>::f16x8;
    type Return = f32;
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(f32s = <V4>::f32x8);

        let x: f32s = x.into();
        let y: f32s = y.into();

        let c = x - y;
        c.mul_add_simd(c, acc)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<Half, Half, V3> for L2 {
    type SIMDWidth = Const<8>;
    type Accumulator = <V3 as Architecture>::f32x8;
    type Left = <V3 as Architecture>::f16x8;
    type Right = <V3 as Architecture>::f16x8;
    type Return = f32;
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(f32s = <V3>::f32x8);

        let x: f32s = x.into();
        let y: f32s = y.into();

        let c = x - y;
        c.mul_add_simd(c, acc)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

impl SIMDSchema<Half, Half, Scalar> for L2 {
    type SIMDWidth = Const<1>;
    type Accumulator = Emulated<f32, 1>;
    type Left = Emulated<Half, 1>;
    type Right = Emulated<Half, 1>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Self::Accumulator = x.into();
        let y: Self::Accumulator = y.into();

        let c = x - y;
        acc + (c * c)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.to_array()[0]
    }
}

impl<A> SIMDSchema<f32, Half, A> for L2
where
    A: Architecture,
{
    type SIMDWidth = Const<8>;
    type Accumulator = A::f32x8;
    type Left = A::f32x8;
    type Right = A::f16x8;
    type Return = f32;
    type Main = Strategy4x2;

    #[inline(always)]
    fn init(&self, arch: A) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let y: A::f32x8 = y.into();
        let c = x - y;
        c.mul_add_simd(c, acc)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<i8, i8, V4> for L2 {
    type SIMDWidth = Const<32>;
    type Accumulator = <V4 as Architecture>::i32x16;
    type Left = <V4 as Architecture>::i8x32;
    type Right = <V4 as Architecture>::i8x32;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V4>::i16x32);

        let x: i16s = x.into();
        let y: i16s = y.into();
        let c = x - y;
        acc.dot_simd(c, c)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree().as_f32_lossy()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<i8, i8, V3> for L2 {
    type SIMDWidth = Const<16>;
    type Accumulator = <V3 as Architecture>::i32x8;
    type Left = <V3 as Architecture>::i8x16;
    type Right = <V3 as Architecture>::i8x16;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V3>::i16x16);

        let x: i16s = x.into();
        let y: i16s = y.into();
        let c = x - y;
        acc.dot_simd(c, c)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree().as_f32_lossy()
    }
}

impl SIMDSchema<i8, i8, Scalar> for L2 {
    type SIMDWidth = Const<4>;
    type Accumulator = Emulated<i32, 4>;
    type Left = Emulated<i8, 4>;
    type Right = Emulated<i8, 4>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Self::Accumulator = x.into();
        let y: Self::Accumulator = y.into();
        let c = x - y;
        acc + c * c
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.to_array().into_iter().sum::<i32>().as_f32_lossy()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        arch: Scalar,
        x: *const i8,
        y: *const i8,
        len: usize,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let mut s: i32 = 0;
        for i in 0..len {
            // SAFETY: The range `[x, x.add(len))` is valid for reads.
            let vx: i32 = unsafe { x.add(i).read() }.into();
            // SAFETY: The range `[y, y.add(len))` is valid for reads.
            let vy: i32 = unsafe { y.add(i).read() }.into();
            let d = vx - vy;
            s += d * d;
        }
        acc + Self::Accumulator::from_array(arch, [s, 0, 0, 0])
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<u8, u8, V4> for L2 {
    type SIMDWidth = Const<32>;
    type Accumulator = <V4 as Architecture>::i32x16;
    type Left = <V4 as Architecture>::u8x32;
    type Right = <V4 as Architecture>::u8x32;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V4>::i16x32);

        let x: i16s = x.into();
        let y: i16s = y.into();
        let c = x - y;
        acc.dot_simd(c, c)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree().as_f32_lossy()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<u8, u8, V3> for L2 {
    type SIMDWidth = Const<16>;
    type Accumulator = <V3 as Architecture>::i32x8;
    type Left = <V3 as Architecture>::u8x16;
    type Right = <V3 as Architecture>::u8x16;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V3>::i16x16);

        let x: i16s = x.into();
        let y: i16s = y.into();
        let c = x - y;
        acc.dot_simd(c, c)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree().as_f32_lossy()
    }
}

impl SIMDSchema<u8, u8, Scalar> for L2 {
    type SIMDWidth = Const<4>;
    type Accumulator = Emulated<i32, 4>;
    type Left = Emulated<u8, 4>;
    type Right = Emulated<u8, 4>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Self::Accumulator = x.into();
        let y: Self::Accumulator = y.into();
        let c = x - y;
        acc + c * c
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.to_array().into_iter().sum::<i32>().as_f32_lossy()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        arch: Scalar,
        x: *const u8,
        y: *const u8,
        len: usize,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let mut s: i32 = 0;
        for i in 0..len {
            // SAFETY: The range `[x, x.add(len))` is valid for reads.
            let vx: i32 = unsafe { x.add(i).read() }.into();
            // SAFETY: The range `[y, y.add(len))` is valid for reads.
            let vy: i32 = unsafe { y.add(i).read() }.into();
            let d = vx - vy;
            s += d * d;
        }
        acc + Self::Accumulator::from_array(arch, [s, 0, 0, 0])
    }
}

// A L2 distance function that defers a final reduction, allowing for a distance
// computation to take place across multiple slice pairs.
#[derive(Clone, Copy, Debug)]
pub struct ResumableL2<A = diskann_wide::arch::Current>
where
    A: Architecture,
    L2: SIMDSchema<f32, f32, A>,
{
    acc: <L2 as SIMDSchema<f32, f32, A>>::Accumulator,
}

impl<A> ResumableSIMDSchema<f32, f32, A> for ResumableL2<A>
where
    A: Architecture,
    L2: SIMDSchema<f32, f32, A, Return = f32>,
{
    type NonResumable = L2;
    type FinalReturn = f32;

    #[inline(always)]
    fn init(arch: A) -> Self {
        Self { acc: L2.init(arch) }
    }

    #[inline(always)]
    fn combine_with(&self, other: <L2 as SIMDSchema<f32, f32, A>>::Accumulator) -> Self {
        Self {
            acc: self.acc + other,
        }
    }

    #[inline(always)]
    fn sum(&self) -> f32 {
        L2.reduce(self.acc)
    }
}

/////
///// IP Implementations
/////

// A pure IP distance function that provides a final reduction.
#[derive(Clone, Copy, Debug, Default)]
pub struct IP;

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<f32, f32, V4> for IP {
    type SIMDWidth = Const<8>;
    type Accumulator = <V4 as Architecture>::f32x8;
    type Left = <V4 as Architecture>::f32x8;
    type Right = <V4 as Architecture>::f32x8;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        x.mul_add_simd(y, acc)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<f32, f32, V3> for IP {
    type SIMDWidth = Const<8>;
    type Accumulator = <V3 as Architecture>::f32x8;
    type Left = <V3 as Architecture>::f32x8;
    type Right = <V3 as Architecture>::f32x8;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        x.mul_add_simd(y, acc)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

impl SIMDSchema<f32, f32, Scalar> for IP {
    type SIMDWidth = Const<4>;
    type Accumulator = Emulated<f32, 4>;
    type Left = Emulated<f32, 4>;
    type Right = Emulated<f32, 4>;
    type Return = f32;
    type Main = Strategy2x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        x * y + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        arch: Scalar,
        x: *const f32,
        y: *const f32,
        len: usize,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let mut s: f32 = 0.0;
        for i in 0..len {
            // SAFETY: The range `[x, x.add(len))` is valid for reads.
            let vx = unsafe { x.add(i).read() };
            // SAFETY: The range `[y, y.add(len))` is valid for reads.
            let vy = unsafe { y.add(i).read() };
            s += vx * vy;
        }
        acc + Self::Accumulator::from_array(arch, [s, 0.0, 0.0, 0.0])
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<Half, Half, V4> for IP {
    type SIMDWidth = Const<8>;
    type Accumulator = <V4 as Architecture>::f32x8;
    type Left = <V4 as Architecture>::f16x8;
    type Right = <V4 as Architecture>::f16x8;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(f32s = <V4>::f32x8);

        let x: f32s = x.into();
        let y: f32s = y.into();
        x.mul_add_simd(y, acc)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<Half, Half, V3> for IP {
    type SIMDWidth = Const<8>;
    type Accumulator = <V3 as Architecture>::f32x8;
    type Left = <V3 as Architecture>::f16x8;
    type Right = <V3 as Architecture>::f16x8;
    type Return = f32;
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(f32s = <V3>::f32x8);

        let x: f32s = x.into();
        let y: f32s = y.into();
        x.mul_add_simd(y, acc)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

impl SIMDSchema<Half, Half, Scalar> for IP {
    type SIMDWidth = Const<1>;
    type Accumulator = Emulated<f32, 1>;
    type Left = Emulated<Half, 1>;
    type Right = Emulated<Half, 1>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Self::Accumulator = x.into();
        let y: Self::Accumulator = y.into();
        x * y + acc
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.to_array()[0]
    }
}

impl<A> SIMDSchema<f32, Half, A> for IP
where
    A: Architecture,
{
    type SIMDWidth = Const<8>;
    type Accumulator = A::f32x8;
    type Left = A::f32x8;
    type Right = A::f16x8;
    type Return = f32;
    type Main = Strategy4x2;

    #[inline(always)]
    fn init(&self, arch: A) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let y: A::f32x8 = y.into();
        x.mul_add_simd(y, acc)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<i8, i8, V4> for IP {
    type SIMDWidth = Const<32>;
    type Accumulator = <V4 as Architecture>::i32x16;
    type Left = <V4 as Architecture>::i8x32;
    type Right = <V4 as Architecture>::i8x32;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V4>::i16x32);

        let x: i16s = x.into();
        let y: i16s = y.into();
        acc.dot_simd(x, y)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree().as_f32_lossy()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<i8, i8, V3> for IP {
    type SIMDWidth = Const<16>;
    type Accumulator = <V3 as Architecture>::i32x8;
    type Left = <V3 as Architecture>::i8x16;
    type Right = <V3 as Architecture>::i8x16;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V3>::i16x16);

        let x: i16s = x.into();
        let y: i16s = y.into();
        acc.dot_simd(x, y)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree().as_f32_lossy()
    }
}

impl SIMDSchema<i8, i8, Scalar> for IP {
    type SIMDWidth = Const<1>;
    type Accumulator = Emulated<i32, 1>;
    type Left = Emulated<i8, 1>;
    type Right = Emulated<i8, 1>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Self::Accumulator = x.into();
        let y: Self::Accumulator = y.into();
        x * y + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.to_array().into_iter().sum::<i32>().as_f32_lossy()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        _arch: Scalar,
        _x: *const i8,
        _y: *const i8,
        _len: usize,
        _acc: Self::Accumulator,
    ) -> Self::Accumulator {
        unreachable!("The SIMD width is 1, so there should be no epilogue")
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<u8, u8, V4> for IP {
    type SIMDWidth = Const<32>;
    type Accumulator = <V4 as Architecture>::i32x16;
    type Left = <V4 as Architecture>::u8x32;
    type Right = <V4 as Architecture>::u8x32;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V4>::i16x32);

        let x: i16s = x.into();
        let y: i16s = y.into();
        acc.dot_simd(x, y)
    }

    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree().as_f32_lossy()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<u8, u8, V3> for IP {
    type SIMDWidth = Const<16>;
    type Accumulator = <V3 as Architecture>::i32x8;
    type Left = <V3 as Architecture>::u8x16;
    type Right = <V3 as Architecture>::u8x16;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V3>::i16x16);

        // NOTE: Promotiving to `i16` rather than `u16` to hit specialized AVX2
        // instructions on x86 hardware.
        let x: i16s = x.into();
        let y: i16s = y.into();
        acc.dot_simd(x, y)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree().as_f32_lossy()
    }
}

impl SIMDSchema<u8, u8, Scalar> for IP {
    type SIMDWidth = Const<1>;
    type Accumulator = Emulated<i32, 1>;
    type Left = Emulated<u8, 1>;
    type Right = Emulated<u8, 1>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Self::Accumulator = x.into();
        let y: Self::Accumulator = y.into();
        x * y + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.to_array().into_iter().sum::<i32>().as_f32_lossy()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        _arch: Scalar,
        _x: *const u8,
        _y: *const u8,
        _len: usize,
        _acc: Self::Accumulator,
    ) -> Self::Accumulator {
        unreachable!("The SIMD width is 1, so there should be no epilogue")
    }
}

// An IP distance function that defers a final reduction.
#[derive(Clone, Copy, Debug)]
pub struct ResumableIP<A = diskann_wide::arch::Current>
where
    A: Architecture,
    IP: SIMDSchema<f32, f32, A>,
{
    acc: <IP as SIMDSchema<f32, f32, A>>::Accumulator,
}

impl<A> ResumableSIMDSchema<f32, f32, A> for ResumableIP<A>
where
    A: Architecture,
    IP: SIMDSchema<f32, f32, A, Return = f32>,
{
    type NonResumable = IP;
    type FinalReturn = f32;

    #[inline(always)]
    fn init(arch: A) -> Self {
        Self { acc: IP.init(arch) }
    }

    #[inline(always)]
    fn combine_with(&self, other: <IP as SIMDSchema<f32, f32, A>>::Accumulator) -> Self {
        Self {
            acc: self.acc + other,
        }
    }

    #[inline(always)]
    fn sum(&self) -> f32 {
        IP.reduce(self.acc)
    }
}

/////////////////////////////////
// Stateless Cosine Similarity //
/////////////////////////////////

/// Accumulator of partial products for a full cosine distance computation (where
/// the norms of both the query and the dataset vector are computed on the fly).
#[derive(Debug, Clone, Copy)]
pub struct FullCosineAccumulator<T> {
    normx: T,
    normy: T,
    xy: T,
}

impl<T> FullCosineAccumulator<T>
where
    T: SIMDVector
        + SIMDSumTree
        + SIMDMulAdd
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>,
    T::Scalar: LossyF32Conversion,
{
    #[inline(always)]
    pub fn new(arch: T::Arch) -> Self {
        // SAFETY: Zero initializing a SIMD vector is safe.
        let zero = T::default(arch);
        Self {
            normx: zero,
            normy: zero,
            xy: zero,
        }
    }

    #[inline(always)]
    pub fn add_with(&self, x: T, y: T) -> Self {
        // SAFETY: Arithmetic on valid arguments is valid.
        FullCosineAccumulator {
            normx: x.mul_add_simd(x, self.normx),
            normy: y.mul_add_simd(y, self.normy),
            xy: x.mul_add_simd(y, self.xy),
        }
    }

    #[inline(always)]
    pub fn add_with_unfused(&self, x: T, y: T) -> Self {
        // SAFETY: Arithmetic on valid arguments is valid.
        FullCosineAccumulator {
            normx: x * x + self.normx,
            normy: y * y + self.normy,
            xy: x * y + self.xy,
        }
    }

    #[inline(always)]
    pub fn sum(&self) -> f32 {
        let normx = self.normx.sum_tree().as_f32_lossy();
        let normy = self.normy.sum_tree().as_f32_lossy();

        // Evaluate the denominator early and use `force_eval`.
        // This will allow the long `sqrt` to be overlapped with some other instructions
        // rather than waiting at the end of the function.
        //
        // There is some worry of subnormal numbers, but we're optimizing for the common
        // case where norms are reasonable values.
        let denominator = normx.sqrt() * normy.sqrt();
        let prod = self.xy.sum_tree().as_f32_lossy();

        // Force the final products to be completely computed before the range check.
        //
        // This prevents LLVM from trying to compute `normy` or `prod` *after* the check
        // to `normx`, which causes it to spill heavily to the stack.
        //
        // Unfortunately, this results in a reduction pattern that appears to be slightly
        // slower on AMD or Windows.
        force_eval(denominator);
        force_eval(prod);

        // This basically checks if either norm is subnormal and if so, we treat the vector
        // as having norm zero.
        //
        // The reason to do this rather than checking `denominator` directly is to have
        // consistent behavior when one vector has a small norm (i.e., always treat it as
        // zero) rather than potentially changing behavior when the other vector has a very
        // large norm to compensate.
        if normx < f32::MIN_POSITIVE || normy < f32::MIN_POSITIVE {
            return 0.0;
        }

        let v = prod / denominator;
        (-1.0f32).max(1.0f32.min(v))
    }

    /// Compute the L2 distance from the partial products rather than the cosine similarity.
    #[inline(always)]
    pub fn sum_as_l2(&self) -> f32 {
        let normx = self.normx.sum_tree().as_f32_lossy();
        let normy = self.normy.sum_tree().as_f32_lossy();
        let xy = self.xy.sum_tree().as_f32_lossy();
        normx + normy - (xy + xy)
    }
}

impl<T> std::ops::Add for FullCosineAccumulator<T>
where
    T: std::ops::Add<Output = T>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        FullCosineAccumulator {
            normx: self.normx + other.normx,
            normy: self.normy + other.normy,
            xy: self.xy + other.xy,
        }
    }
}

/// A pure Cosine Similarity function that provides a final reduction.
#[derive(Default, Clone, Copy)]
pub struct CosineStateless;

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<f32, f32, V4> for CosineStateless {
    type SIMDWidth = Const<16>;
    type Accumulator = FullCosineAccumulator<<V4 as Architecture>::f32x16>;
    type Left = <V4 as Architecture>::f32x16;
    type Right = <V4 as Architecture>::f32x16;
    type Return = f32;

    // Cosine accumulators are pretty large, so only use 2 parallel accumulator with a
    // hefty unroll factor.
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        acc.add_with(x, y)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, acc: Self::Accumulator) -> Self::Return {
        acc.sum()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<f32, f32, V3> for CosineStateless {
    type SIMDWidth = Const<8>;
    type Accumulator = FullCosineAccumulator<<V3 as Architecture>::f32x8>;
    type Left = <V3 as Architecture>::f32x8;
    type Right = <V3 as Architecture>::f32x8;
    type Return = f32;

    // Cosine accumulators are pretty large, so only use 2 parallel accumulator with a
    // hefty unroll factor.
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        acc.add_with(x, y)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, acc: Self::Accumulator) -> Self::Return {
        acc.sum()
    }
}

impl SIMDSchema<f32, f32, Scalar> for CosineStateless {
    type SIMDWidth = Const<4>;
    type Accumulator = FullCosineAccumulator<Emulated<f32, 4>>;
    type Left = Emulated<f32, 4>;
    type Right = Emulated<f32, 4>;
    type Return = f32;

    type Main = Strategy2x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        acc.add_with_unfused(x, y)
    }

    #[inline(always)]
    fn reduce(&self, acc: Self::Accumulator) -> Self::Return {
        acc.sum()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<Half, Half, V4> for CosineStateless {
    type SIMDWidth = Const<16>;
    type Accumulator = FullCosineAccumulator<<V4 as Architecture>::f32x16>;
    type Left = <V4 as Architecture>::f16x16;
    type Right = <V4 as Architecture>::f16x16;
    type Return = f32;
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(f32s = <V4>::f32x16);

        let x: f32s = x.into();
        let y: f32s = y.into();
        acc.add_with(x, y)
    }

    #[inline(always)]
    fn reduce(&self, acc: Self::Accumulator) -> Self::Return {
        acc.sum()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<Half, Half, V3> for CosineStateless {
    type SIMDWidth = Const<8>;
    type Accumulator = FullCosineAccumulator<<V3 as Architecture>::f32x8>;
    type Left = <V3 as Architecture>::f16x8;
    type Right = <V3 as Architecture>::f16x8;
    type Return = f32;
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(f32s = <V3>::f32x8);

        let x: f32s = x.into();
        let y: f32s = y.into();
        acc.add_with(x, y)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, acc: Self::Accumulator) -> Self::Return {
        acc.sum()
    }
}

impl SIMDSchema<Half, Half, Scalar> for CosineStateless {
    type SIMDWidth = Const<1>;
    type Accumulator = FullCosineAccumulator<Emulated<f32, 1>>;
    type Left = Emulated<Half, 1>;
    type Right = Emulated<Half, 1>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Emulated<f32, 1> = x.into();
        let y: Emulated<f32, 1> = y.into();
        acc.add_with_unfused(x, y)
    }

    #[inline(always)]
    fn reduce(&self, acc: Self::Accumulator) -> Self::Return {
        acc.sum()
    }
}
impl<A> SIMDSchema<f32, Half, A> for CosineStateless
where
    A: Architecture,
{
    type SIMDWidth = Const<8>;
    type Accumulator = FullCosineAccumulator<A::f32x8>;
    type Left = A::f32x8;
    type Right = A::f16x8;
    type Return = f32;
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: A) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let y: A::f32x8 = y.into();
        acc.add_with(x, y)
    }

    #[inline(always)]
    fn reduce(&self, acc: Self::Accumulator) -> Self::Return {
        acc.sum()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<i8, i8, V4> for CosineStateless {
    type SIMDWidth = Const<32>;
    type Accumulator = FullCosineAccumulator<<V4 as Architecture>::i32x16>;
    type Left = <V4 as Architecture>::i8x32;
    type Right = <V4 as Architecture>::i8x32;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V4>::i16x32);

        let x: i16s = x.into();
        let y: i16s = y.into();

        FullCosineAccumulator {
            normx: acc.normx.dot_simd(x, x),
            normy: acc.normy.dot_simd(y, y),
            xy: acc.xy.dot_simd(x, y),
        }
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<i8, i8, V3> for CosineStateless {
    type SIMDWidth = Const<16>;
    type Accumulator = FullCosineAccumulator<<V3 as Architecture>::i32x8>;
    type Left = <V3 as Architecture>::i8x16;
    type Right = <V3 as Architecture>::i8x16;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V3>::i16x16);

        let x: i16s = x.into();
        let y: i16s = y.into();

        FullCosineAccumulator {
            normx: acc.normx.dot_simd(x, x),
            normy: acc.normy.dot_simd(y, y),
            xy: acc.xy.dot_simd(x, y),
        }
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum()
    }
}

impl SIMDSchema<i8, i8, Scalar> for CosineStateless {
    type SIMDWidth = Const<4>;
    type Accumulator = FullCosineAccumulator<Emulated<i32, 4>>;
    type Left = Emulated<i8, 4>;
    type Right = Emulated<i8, 4>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Emulated<i32, 4> = x.into();
        let y: Emulated<i32, 4> = y.into();
        acc.add_with(x, y)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        arch: Scalar,
        x: *const i8,
        y: *const i8,
        len: usize,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let mut xy: i32 = 0;
        let mut xx: i32 = 0;
        let mut yy: i32 = 0;

        for i in 0..len {
            // SAFETY: The range `[x, x.add(len))` is valid for reads.
            let vx: i32 = unsafe { x.add(i).read() }.into();
            // SAFETY: The range `[y, y.add(len))` is valid for reads.
            let vy: i32 = unsafe { y.add(i).read() }.into();

            xx += vx * vx;
            xy += vx * vy;
            yy += vy * vy;
        }

        acc + FullCosineAccumulator {
            normx: Emulated::from_array(arch, [xx, 0, 0, 0]),
            normy: Emulated::from_array(arch, [yy, 0, 0, 0]),
            xy: Emulated::from_array(arch, [xy, 0, 0, 0]),
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<u8, u8, V4> for CosineStateless {
    type SIMDWidth = Const<32>;
    type Accumulator = FullCosineAccumulator<<V4 as Architecture>::i32x16>;
    type Left = <V4 as Architecture>::u8x32;
    type Right = <V4 as Architecture>::u8x32;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V4>::i16x32);

        let x: i16s = x.into();
        let y: i16s = y.into();

        FullCosineAccumulator {
            normx: acc.normx.dot_simd(x, x),
            normy: acc.normy.dot_simd(y, y),
            xy: acc.xy.dot_simd(x, y),
        }
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<u8, u8, V3> for CosineStateless {
    type SIMDWidth = Const<16>;
    type Accumulator = FullCosineAccumulator<<V3 as Architecture>::i32x8>;
    type Left = <V3 as Architecture>::u8x16;
    type Right = <V3 as Architecture>::u8x16;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        diskann_wide::alias!(i16s = <V3>::i16x16);

        let x: i16s = x.into();
        let y: i16s = y.into();

        FullCosineAccumulator {
            normx: acc.normx.dot_simd(x, x),
            normy: acc.normy.dot_simd(y, y),
            xy: acc.xy.dot_simd(x, y),
        }
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum()
    }
}

impl SIMDSchema<u8, u8, Scalar> for CosineStateless {
    type SIMDWidth = Const<4>;
    type Accumulator = FullCosineAccumulator<Emulated<i32, 4>>;
    type Left = Emulated<u8, 4>;
    type Right = Emulated<u8, 4>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::new(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Emulated<i32, 4> = x.into();
        let y: Emulated<i32, 4> = y.into();
        acc.add_with(x, y)
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        arch: Scalar,
        x: *const u8,
        y: *const u8,
        len: usize,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let mut xy: i32 = 0;
        let mut xx: i32 = 0;
        let mut yy: i32 = 0;

        for i in 0..len {
            // SAFETY: The range `[x, x.add(len))` is valid for reads.
            let vx: i32 = unsafe { x.add(i).read() }.into();
            // SAFETY: The range `[y, y.add(len))` is valid for reads.
            let vy: i32 = unsafe { y.add(i).read() }.into();

            xx += vx * vx;
            xy += vx * vy;
            yy += vy * vy;
        }

        acc + FullCosineAccumulator {
            normx: Emulated::from_array(arch, [xx, 0, 0, 0]),
            normy: Emulated::from_array(arch, [yy, 0, 0, 0]),
            xy: Emulated::from_array(arch, [xy, 0, 0, 0]),
        }
    }
}

/// A resumable cosine similarity computation.
#[derive(Debug, Clone, Copy)]
pub struct ResumableCosine<A = diskann_wide::arch::Current>(
    <CosineStateless as SIMDSchema<f32, f32, A>>::Accumulator,
)
where
    A: Architecture,
    CosineStateless: SIMDSchema<f32, f32, A>;

impl<A> ResumableSIMDSchema<f32, f32, A> for ResumableCosine<A>
where
    A: Architecture,
    CosineStateless: SIMDSchema<f32, f32, A, Return = f32>,
{
    type NonResumable = CosineStateless;
    type FinalReturn = f32;

    #[inline(always)]
    fn init(arch: A) -> Self {
        Self(CosineStateless.init(arch))
    }

    #[inline(always)]
    fn combine_with(
        &self,
        other: <CosineStateless as SIMDSchema<f32, f32, A>>::Accumulator,
    ) -> Self {
        Self(self.0 + other)
    }

    #[inline(always)]
    fn sum(&self) -> f32 {
        CosineStateless.reduce(self.0)
    }
}

/////
///// L1 Norm Implementations
/////

// ==================================================================================================
// NOTE: L1Norm IS A LOGICAL UNARY OPERATION
// --------------------------------------------------------------------------------------------------
// Although wired through the generic binary 'SIMDSchema'/'simd_op' infrastructure (which expects
// two input slices of equal length), 'L1Norm' conceptually computes: sum_i |x_i|
// The right-hand operand is completely ignored and exists ONLY to satisfy the shared execution
// machinery (loop tiling, epilogue handling, etc.).
// ==================================================================================================

// A pure L1 norm function that provides a final reduction.
#[derive(Clone, Copy, Debug, Default)]
pub struct L1Norm;

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<f32, f32, V4> for L1Norm {
    type SIMDWidth = Const<16>;
    type Accumulator = <V4 as Architecture>::f32x16;
    type Left = <V4 as Architecture>::f32x16;
    type Right = <V4 as Architecture>::f32x16;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        _y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        x.abs_simd() + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<f32, f32, V3> for L1Norm {
    type SIMDWidth = Const<8>;
    type Accumulator = <V3 as Architecture>::f32x8;
    type Left = <V3 as Architecture>::f32x8;
    type Right = <V3 as Architecture>::f32x8;
    type Return = f32;
    type Main = Strategy4x1;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        _y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        x.abs_simd() + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

impl SIMDSchema<f32, f32, Scalar> for L1Norm {
    type SIMDWidth = Const<4>;
    type Accumulator = Emulated<f32, 4>;
    type Left = Emulated<f32, 4>;
    type Right = Emulated<f32, 4>;
    type Return = f32;
    type Main = Strategy2x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        _y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        x.abs_simd() + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        arch: Scalar,
        x: *const f32,
        _y: *const f32,
        len: usize,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let mut s: f32 = 0.0;
        for i in 0..len {
            // SAFETY: The range `[x, x.add(len))` is valid for reads.
            let vx = unsafe { x.add(i).read() };
            s += vx.abs();
        }
        acc + Self::Accumulator::from_array(arch, [s, 0.0, 0.0, 0.0])
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<Half, Half, V4> for L1Norm {
    type SIMDWidth = Const<8>;
    type Accumulator = <V4 as Architecture>::f32x8;
    type Left = <V4 as Architecture>::f16x8;
    type Right = <V4 as Architecture>::f16x8;
    type Return = f32;
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V4) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        _y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: <V4 as Architecture>::f32x8 = x.into();
        x.abs_simd() + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDSchema<Half, Half, V3> for L1Norm {
    type SIMDWidth = Const<8>;
    type Accumulator = <V3 as Architecture>::f32x8;
    type Left = <V3 as Architecture>::f16x8;
    type Right = <V3 as Architecture>::f16x8;
    type Return = f32;
    type Main = Strategy2x4;

    #[inline(always)]
    fn init(&self, arch: V3) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        _y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: <V3 as Architecture>::f32x8 = x.into();
        x.abs_simd() + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.sum_tree()
    }
}

impl SIMDSchema<Half, Half, Scalar> for L1Norm {
    type SIMDWidth = Const<1>;
    type Accumulator = Emulated<f32, 1>;
    type Left = Emulated<Half, 1>;
    type Right = Emulated<Half, 1>;
    type Return = f32;
    type Main = Strategy1x1;

    #[inline(always)]
    fn init(&self, arch: Scalar) -> Self::Accumulator {
        Self::Accumulator::default(arch)
    }

    #[inline(always)]
    fn accumulate(
        &self,
        x: Self::Left,
        _y: Self::Right,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        let x: Self::Accumulator = x.into();
        x.abs_simd() + acc
    }

    // Perform a final reduction.
    #[inline(always)]
    fn reduce(&self, x: Self::Accumulator) -> Self::Return {
        x.to_array()[0]
    }

    #[inline(always)]
    unsafe fn epilogue(
        &self,
        _arch: Scalar,
        _x: *const Half,
        _y: *const Half,
        _len: usize,
        _acc: Self::Accumulator,
    ) -> Self::Accumulator {
        unreachable!("The SIMD width is 1, so there should be no epilogue")
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::LazyLock};

    use approx::assert_relative_eq;
    use diskann_wide::{arch::Target1, ARCH};
    use half::f16;
    use rand::{distr::StandardUniform, rngs::StdRng, Rng, SeedableRng};
    use rand_distr;

    use super::*;
    use crate::{distance::reference, norm::LInfNorm, test_util};

    ///////////////////////
    // Cosine Norm Check //
    ///////////////////////

    fn cosine_norm_check_impl<A>(arch: A)
    where
        A: diskann_wide::Architecture,
        CosineStateless:
            SIMDSchema<f32, f32, A, Return = f32> + SIMDSchema<Half, Half, A, Return = f32>,
    {
        // Zero - f32
        {
            let x: [f32; 2] = [0.0, 0.0];
            let y: [f32; 2] = [0.0, 1.0];
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, x),
                0.0,
                "when both vectors are zero, similarity should be zero",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, y),
                0.0,
                "when one vector is zero, similarity should be zero",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, y, x),
                0.0,
                "when one vector is zero, similarity should be zero",
            );
        }

        // Subnormal - f32
        {
            let x: [f32; 4] = [0.0, 0.0, 2.938736e-39f32, 0.0];
            let y: [f32; 4] = [0.0, 0.0, 1.0, 0.0];
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, x),
                0.0,
                "when both vectors are almost zero, similarity should be zero",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, y),
                0.0,
                "when one vector is almost zero, similarity should be zero",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, y, x),
                0.0,
                "when one vector is almost zero, similarity should be zero",
            );
        }

        // Small - f32
        {
            let x: [f32; 4] = [0.0, 0.0, 1.0842022e-19f32, 0.0];
            let y: [f32; 4] = [0.0, 0.0, 1.0, 0.0];
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, x),
                1.0,
                "cosine-stateless should handle vectors this small",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, y),
                1.0,
                "cosine-stateless should handle vectors this small",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, y, x),
                1.0,
                "cosine-stateless should handle vectors this small",
            );
        }

        let cvt = diskann_wide::cast_f32_to_f16;

        // Zero - f16
        {
            let x: [Half; 2] = [Half::default(), Half::default()];
            let y: [Half; 2] = [Half::default(), cvt(1.0)];
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, x),
                0.0,
                "when both vectors are zero, similarity should be zero",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, y),
                0.0,
                "when one vector is zero, similarity should be zero",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, y, x),
                0.0,
                "when one vector is zero, similarity should be zero",
            );
        }

        // Subnormal - f16
        {
            let x: [Half; 4] = [
                Half::default(),
                Half::default(),
                Half::MIN_POSITIVE_SUBNORMAL,
                Half::default(),
            ];
            let y: [Half; 4] = [Half::default(), Half::default(), cvt(1.0), Half::default()];
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, x),
                1.0,
                "when both vectors are almost zero, similarity should be zero",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, x, y),
                1.0,
                "when one vector is almost zero, similarity should be zero",
            );
            assert_eq!(
                simd_op(&CosineStateless {}, arch, y, x),
                1.0,
                "when one vector is almost zero, similarity should be zero",
            );

            // Grab a range of floating point numbers whose squares cover the range of
            // our target threshold.
            //
            // Ensure that all combinations of values within this critical range to not
            // result in a misrounding.
            let threshold = f32::MIN_POSITIVE;
            let bound = 50;
            let values = {
                let mut down = threshold;
                let mut up = threshold;
                for _ in 0..bound {
                    down = down.next_down();
                    up = up.next_up();
                }
                assert!(down > 0.0);
                let min = down.sqrt();
                let max = up.sqrt();
                let mut v = min;
                let mut values = Vec::new();
                while v <= max {
                    values.push(v);
                    v = v.next_up();
                }
                values
            };

            let mut lo = 0;
            let mut hi = 0;
            for i in values.iter() {
                for j in values.iter() {
                    let s: f32 = simd_op(&CosineStateless {}, arch, [*i], [*j]);
                    if i * i < threshold || j * j < threshold {
                        lo += 1;
                        assert_eq!(s, 0.0, "failed for i = {}, j = {}", i, j);
                    } else {
                        hi += 1;
                        assert_eq!(s, 1.0, "failed for i = {}, j = {}", i, j);
                    }
                }
            }
            assert_ne!(lo, 0);
            assert_ne!(hi, 0);
        }
    }

    #[test]
    fn cosine_norm_check() {
        cosine_norm_check_impl::<diskann_wide::arch::Current>(diskann_wide::arch::current());
        cosine_norm_check_impl::<diskann_wide::arch::Scalar>(diskann_wide::arch::Scalar::new());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn cosine_norm_check_x86_64() {
        if let Some(arch) = V3::new_checked() {
            cosine_norm_check_impl::<V3>(arch);
        }

        if let Some(arch) = V4::new_checked_miri() {
            cosine_norm_check_impl::<V4>(arch);
        }
    }

    ////////////
    // Schema //
    ////////////

    // Chunk the left and right hand slices and compute the result using a resumable function.
    fn test_resumable<T, L, R, A>(arch: A, x: &[L], y: &[R], chunk_size: usize) -> f32
    where
        A: Architecture,
        T: ResumableSIMDSchema<L, R, A, FinalReturn = f32>,
    {
        let mut acc = Resumable(<T as ResumableSIMDSchema<L, R, A>>::init(arch));
        let iter = std::iter::zip(x.chunks(chunk_size), y.chunks(chunk_size));
        for (a, b) in iter {
            acc = simd_op(&acc, arch, a, b);
        }
        acc.0.sum()
    }

    fn stress_test_with_resumable<
        A: Architecture,
        O: Default + SIMDSchema<f32, f32, A, Return = f32>,
        T: ResumableSIMDSchema<f32, f32, A, NonResumable = O, FinalReturn = f32>,
        Rand: Rng,
    >(
        arch: A,
        reference: fn(&[f32], &[f32]) -> f32,
        dim: usize,
        epsilon: f32,
        max_relative: f32,
        rng: &mut Rand,
    ) {
        // Pick chunk sizes that exercise combinations of the unrolled loops.
        let chunk_divisors: Vec<usize> = vec![1, 2, 3, 4, 16, 54, 64, 65, 70, 77];
        let checker = test_util::AdHocChecker::<f32, f32>::new(|a: &[f32], b: &[f32]| {
            let expected = reference(a, b);
            let got = simd_op(&O::default(), arch, a, b);
            println!("dim = {}", dim);
            assert_relative_eq!(
                expected,
                got,
                epsilon = epsilon,
                max_relative = max_relative,
            );

            if dim == 0 {
                return;
            }

            for d in &chunk_divisors {
                let chunk_size = dim / d + (!dim.is_multiple_of(*d) as usize);
                let chunked = test_resumable::<T, f32, f32, _>(arch, a, b, chunk_size);
                assert_relative_eq!(chunked, got, epsilon = epsilon, max_relative = max_relative);
            }
        });

        test_util::test_distance_function(
            checker,
            rand_distr::Normal::new(0.0, 10.0).unwrap(),
            rand_distr::Normal::new(0.0, 10.0).unwrap(),
            dim,
            10,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stress_test<L, R, DistLeft, DistRight, O, Rand, A>(
        arch: A,
        reference: fn(&[L], &[R]) -> f32,
        left_dist: DistLeft,
        right_dist: DistRight,
        dim: usize,
        epsilon: f32,
        max_relative: f32,
        rng: &mut Rand,
    ) where
        L: test_util::CornerCases,
        R: test_util::CornerCases,
        DistLeft: test_util::GenerateRandomArguments<L>,
        DistRight: test_util::GenerateRandomArguments<R>,
        O: Default + SIMDSchema<L, R, A, Return = f32>,
        Rand: Rng,
        A: Architecture,
    {
        let checker = test_util::Checker::<L, R, f32>::new(
            |x: &[L], y: &[R]| simd_op(&O::default(), arch, x, y),
            reference,
            |got, expected| {
                assert_relative_eq!(
                    expected,
                    got,
                    epsilon = epsilon,
                    max_relative = max_relative
                );
            },
        );

        let trials = if cfg!(miri) { 0 } else { 10 };

        test_util::test_distance_function(checker, left_dist, right_dist, dim, trials, rng);
    }

    fn stress_test_linf<L, Dist, Rand, A>(
        arch: A,
        reference: fn(&[L]) -> f32,
        dist: Dist,
        dim: usize,
        epsilon: f32,
        max_relative: f32,
        rng: &mut Rand,
    ) where
        L: test_util::CornerCases + Copy,
        Dist: Clone + test_util::GenerateRandomArguments<L>,
        Rand: Rng,
        A: Architecture,
        LInfNorm: for<'a> Target1<A, f32, &'a [L]>,
    {
        let checker = test_util::Checker::<L, L, f32>::new(
            |x: &[L], _y: &[L]| (LInfNorm).run(arch, x),
            |x: &[L], _y: &[L]| reference(x),
            |got, expected| {
                assert_relative_eq!(
                    expected,
                    got,
                    epsilon = epsilon,
                    max_relative = max_relative
                );
            },
        );

        println!("checking {dim}");
        test_util::test_distance_function(checker, dist.clone(), dist, dim, 10, rng);
    }

    /////////
    // f32 //
    /////////

    macro_rules! float_test {
        ($name:ident,
         $impl:ty,
         $resumable:ident,
         $reference:path,
         $eps:literal,
         $relative:literal,
         $seed:literal,
         $upper:literal,
         $($arch:tt)*
        ) => {
            #[test]
            fn $name() {
                if let Some(arch) = $($arch)* {
                    let mut rng = StdRng::seed_from_u64($seed);
                    for dim in 0..$upper {
                        stress_test_with_resumable::<_, $impl, $resumable<_>, StdRng>(
                            arch,
                            |l, r| $reference(l, r).into_inner(),
                            dim,
                            $eps,
                            $relative,
                            &mut rng,
                        );
                    }
                }
            }
        }
    }

    //----//
    // L2 //
    //----//

    float_test!(
        test_l2_f32_current,
        L2,
        ResumableL2,
        reference::reference_squared_l2_f32_mathematical,
        1e-5,
        1e-5,
        0xf149c2bcde660128,
        64,
        Some(diskann_wide::ARCH)
    );

    float_test!(
        test_l2_f32_scalar,
        L2,
        ResumableL2,
        reference::reference_squared_l2_f32_mathematical,
        1e-5,
        1e-5,
        0xf149c2bcde660128,
        64,
        Some(diskann_wide::arch::Scalar)
    );

    #[cfg(target_arch = "x86_64")]
    float_test!(
        test_l2_f32_x86_64_v3,
        L2,
        ResumableL2,
        reference::reference_squared_l2_f32_mathematical,
        1e-5,
        1e-5,
        0xf149c2bcde660128,
        256,
        V3::new_checked()
    );

    #[cfg(target_arch = "x86_64")]
    float_test!(
        test_l2_f32_x86_64_v4,
        L2,
        ResumableL2,
        reference::reference_squared_l2_f32_mathematical,
        1e-5,
        1e-5,
        0xf149c2bcde660128,
        256,
        V4::new_checked_miri()
    );

    //----//
    // IP //
    //----//

    float_test!(
        test_ip_f32_current,
        IP,
        ResumableIP,
        reference::reference_innerproduct_f32_mathematical,
        2e-4,
        1e-3,
        0xb4687c17a9ea9866,
        64,
        Some(diskann_wide::ARCH)
    );

    float_test!(
        test_ip_f32_scalar,
        IP,
        ResumableIP,
        reference::reference_innerproduct_f32_mathematical,
        2e-4,
        1e-3,
        0xb4687c17a9ea9866,
        64,
        Some(diskann_wide::arch::Scalar)
    );

    #[cfg(target_arch = "x86_64")]
    float_test!(
        test_ip_f32_x86_64_v3,
        IP,
        ResumableIP,
        reference::reference_innerproduct_f32_mathematical,
        2e-4,
        1e-3,
        0xb4687c17a9ea9866,
        256,
        V3::new_checked()
    );

    #[cfg(target_arch = "x86_64")]
    float_test!(
        test_ip_f32_x86_64_v4,
        IP,
        ResumableIP,
        reference::reference_innerproduct_f32_mathematical,
        2e-4,
        1e-3,
        0xb4687c17a9ea9866,
        256,
        V4::new_checked_miri()
    );

    //--------//
    // Cosine //
    //--------//

    float_test!(
        test_cosine_f32_current,
        CosineStateless,
        ResumableCosine,
        reference::reference_cosine_f32_mathematical,
        1e-5,
        1e-5,
        0xe860e9dc65f38bb8,
        64,
        Some(diskann_wide::ARCH)
    );

    float_test!(
        test_cosine_f32_scalar,
        CosineStateless,
        ResumableCosine,
        reference::reference_cosine_f32_mathematical,
        1e-5,
        1e-5,
        0xe860e9dc65f38bb8,
        64,
        Some(diskann_wide::arch::Scalar)
    );

    #[cfg(target_arch = "x86_64")]
    float_test!(
        test_cosine_f32_x86_64_v3,
        CosineStateless,
        ResumableCosine,
        reference::reference_cosine_f32_mathematical,
        1e-5,
        1e-5,
        0xe860e9dc65f38bb8,
        256,
        V3::new_checked()
    );

    #[cfg(target_arch = "x86_64")]
    float_test!(
        test_cosine_f32_x86_64_v4,
        CosineStateless,
        ResumableCosine,
        reference::reference_cosine_f32_mathematical,
        1e-5,
        1e-5,
        0xe860e9dc65f38bb8,
        256,
        V4::new_checked_miri()
    );

    /////////
    // f16 //
    /////////

    macro_rules! half_test {
        ($name:ident,
         $impl:ty,
         $reference:path,
         $eps:literal,
         $relative:literal,
         $seed:literal,
         $upper:literal,
         $($arch:tt)*
        ) => {
            #[test]
            fn $name() {
                if let Some(arch) = $($arch)* {
                    let mut rng = StdRng::seed_from_u64($seed);
                    for dim in 0..$upper {
                        stress_test::<
                            Half,
                            Half,
                            rand_distr::Normal<f32>,
                            rand_distr::Normal<f32>,
                            $impl,
                            StdRng,
                            _
                        >(
                            arch,
                            |l, r| $reference(l, r).into_inner(),
                            rand_distr::Normal::new(0.0, 10.0).unwrap(),
                            rand_distr::Normal::new(0.0, 10.0).unwrap(),
                            dim,
                            $eps,
                            $relative,
                            &mut rng
                        );
                    }
                }
            }
        }
    }

    //----//
    // L2 //
    //----//

    half_test!(
        test_l2_f16_current,
        L2,
        reference::reference_squared_l2_f16_mathematical,
        1e-5,
        1e-5,
        0x87ca6f1051667500,
        64,
        Some(diskann_wide::ARCH)
    );

    half_test!(
        test_l2_f16_scalar,
        L2,
        reference::reference_squared_l2_f16_mathematical,
        1e-5,
        1e-5,
        0x87ca6f1051667500,
        64,
        Some(diskann_wide::arch::Scalar)
    );

    #[cfg(target_arch = "x86_64")]
    half_test!(
        test_l2_f16_x86_64_v3,
        L2,
        reference::reference_squared_l2_f16_mathematical,
        1e-5,
        1e-5,
        0x87ca6f1051667500,
        256,
        V3::new_checked()
    );

    #[cfg(target_arch = "x86_64")]
    half_test!(
        test_l2_f16_x86_64_v4,
        L2,
        reference::reference_squared_l2_f16_mathematical,
        1e-5,
        1e-5,
        0x87ca6f1051667500,
        256,
        V4::new_checked_miri()
    );

    //----//
    // IP //
    //----//

    half_test!(
        test_ip_f16_current,
        IP,
        reference::reference_innerproduct_f16_mathematical,
        2e-4,
        2e-4,
        0x5909f5f20307ccbe,
        64,
        Some(diskann_wide::ARCH)
    );

    half_test!(
        test_ip_f16_scalar,
        IP,
        reference::reference_innerproduct_f16_mathematical,
        2e-4,
        2e-4,
        0x5909f5f20307ccbe,
        64,
        Some(diskann_wide::arch::Scalar)
    );

    #[cfg(target_arch = "x86_64")]
    half_test!(
        test_ip_f16_x86_64_v3,
        IP,
        reference::reference_innerproduct_f16_mathematical,
        2e-4,
        2e-4,
        0x5909f5f20307ccbe,
        256,
        V3::new_checked()
    );

    #[cfg(target_arch = "x86_64")]
    half_test!(
        test_ip_f16_x86_64_v4,
        IP,
        reference::reference_innerproduct_f16_mathematical,
        2e-4,
        2e-4,
        0x5909f5f20307ccbe,
        256,
        V4::new_checked_miri()
    );

    //--------//
    // Cosine //
    //--------//

    half_test!(
        test_cosine_f16_current,
        CosineStateless,
        reference::reference_cosine_f16_mathematical,
        1e-5,
        1e-5,
        0x41dda34655f05ef6,
        64,
        Some(diskann_wide::ARCH)
    );

    half_test!(
        test_cosine_f16_scalar,
        CosineStateless,
        reference::reference_cosine_f16_mathematical,
        1e-5,
        1e-5,
        0x41dda34655f05ef6,
        64,
        Some(diskann_wide::arch::Scalar)
    );

    #[cfg(target_arch = "x86_64")]
    half_test!(
        test_cosine_f16_x86_64_v3,
        CosineStateless,
        reference::reference_cosine_f16_mathematical,
        1e-5,
        1e-5,
        0x41dda34655f05ef6,
        256,
        V3::new_checked()
    );

    #[cfg(target_arch = "x86_64")]
    half_test!(
        test_cosine_f16_x86_64_v4,
        CosineStateless,
        reference::reference_cosine_f16_mathematical,
        1e-5,
        1e-5,
        0x41dda34655f05ef6,
        256,
        V4::new_checked_miri()
    );

    /////////////
    // Integer //
    /////////////

    macro_rules! int_test {
        (
            $name:ident,
            $T:ty,
            $impl:ty,
            $reference:path,
            $seed:literal,
            $upper:literal,
            { $($arch:tt)* }
        ) => {
            #[test]
            fn $name() {
                if let Some(arch) = $($arch)* {
                    let mut rng = StdRng::seed_from_u64($seed);
                    for dim in 0..$upper {
                        stress_test::<$T, $T, _, _, $impl, _, _>(
                            arch,
                            |l, r| $reference(l, r).into_inner(),
                            StandardUniform,
                            StandardUniform,
                            dim,
                            0.0,
                            0.0,
                            &mut rng,
                        )
                    }
                }
            }
        }
    }

    //----//
    // U8 //
    //----//

    int_test!(
        test_l2_u8_current,
        u8,
        L2,
        reference::reference_squared_l2_u8_mathematical,
        0x945bdc37d8279d4b,
        128,
        { Some(ARCH) }
    );

    int_test!(
        test_l2_u8_scalar,
        u8,
        L2,
        reference::reference_squared_l2_u8_mathematical,
        0x74c86334ab7a51f9,
        128,
        { Some(diskann_wide::arch::Scalar) }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_l2_u8_x86_64_v3,
        u8,
        L2,
        reference::reference_squared_l2_u8_mathematical,
        0x74c86334ab7a51f9,
        256,
        { V3::new_checked() }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_l2_u8_x86_64_v4,
        u8,
        L2,
        reference::reference_squared_l2_u8_mathematical,
        0x74c86334ab7a51f9,
        320,
        { V4::new_checked_miri() }
    );

    int_test!(
        test_ip_u8_current,
        u8,
        IP,
        reference::reference_innerproduct_u8_mathematical,
        0xcbe0342c75085fd5,
        64,
        { Some(ARCH) }
    );

    int_test!(
        test_ip_u8_scalar,
        u8,
        IP,
        reference::reference_innerproduct_u8_mathematical,
        0x888e07fc489e773f,
        64,
        { Some(diskann_wide::arch::Scalar) }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_ip_u8_x86_64_v3,
        u8,
        IP,
        reference::reference_innerproduct_u8_mathematical,
        0x888e07fc489e773f,
        256,
        { V3::new_checked() }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_ip_u8_x86_64_v4,
        u8,
        IP,
        reference::reference_innerproduct_u8_mathematical,
        0x888e07fc489e773f,
        320,
        { V4::new_checked_miri() }
    );

    int_test!(
        test_cosine_u8_current,
        u8,
        CosineStateless,
        reference::reference_cosine_u8_mathematical,
        0x96867b6aff616b28,
        64,
        { Some(ARCH) }
    );

    int_test!(
        test_cosine_u8_scalar,
        u8,
        CosineStateless,
        reference::reference_cosine_u8_mathematical,
        0xcc258c9391733211,
        64,
        { Some(diskann_wide::arch::Scalar) }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_cosine_u8_x86_64_v3,
        u8,
        CosineStateless,
        reference::reference_cosine_u8_mathematical,
        0xcc258c9391733211,
        256,
        { V3::new_checked() }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_cosine_u8_x86_64_v4,
        u8,
        CosineStateless,
        reference::reference_cosine_u8_mathematical,
        0xcc258c9391733211,
        320,
        { V4::new_checked_miri() }
    );

    //----//
    // I8 //
    //----//

    int_test!(
        test_l2_i8_current,
        i8,
        L2,
        reference::reference_squared_l2_i8_mathematical,
        0xa60136248cd3c2f0,
        64,
        { Some(ARCH) }
    );

    int_test!(
        test_l2_i8_scalar,
        i8,
        L2,
        reference::reference_squared_l2_i8_mathematical,
        0x3e8bada709e176be,
        64,
        { Some(diskann_wide::arch::Scalar) }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_l2_i8_x86_64_v3,
        i8,
        L2,
        reference::reference_squared_l2_i8_mathematical,
        0x3e8bada709e176be,
        256,
        { V3::new_checked() }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_l2_i8_x86_64_v4,
        i8,
        L2,
        reference::reference_squared_l2_i8_mathematical,
        0x3e8bada709e176be,
        320,
        { V4::new_checked_miri() }
    );

    int_test!(
        test_ip_i8_current,
        i8,
        IP,
        reference::reference_innerproduct_i8_mathematical,
        0xe8306104740509e1,
        64,
        { Some(ARCH) }
    );

    int_test!(
        test_ip_i8_scalar,
        i8,
        IP,
        reference::reference_innerproduct_i8_mathematical,
        0x8a263408c7b31d85,
        64,
        { Some(diskann_wide::arch::Scalar) }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_ip_i8_x86_64_v3,
        i8,
        IP,
        reference::reference_innerproduct_i8_mathematical,
        0x8a263408c7b31d85,
        256,
        { V3::new_checked() }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_ip_i8_x86_64_v4,
        i8,
        IP,
        reference::reference_innerproduct_i8_mathematical,
        0x8a263408c7b31d85,
        320,
        { V4::new_checked_miri() }
    );

    int_test!(
        test_cosine_i8_current,
        i8,
        CosineStateless,
        reference::reference_cosine_i8_mathematical,
        0x818c210190701e4b,
        64,
        { Some(ARCH) }
    );

    int_test!(
        test_cosine_i8_scalar,
        i8,
        CosineStateless,
        reference::reference_cosine_i8_mathematical,
        0x2d077bed2629b18e,
        64,
        { Some(diskann_wide::arch::Scalar) }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_cosine_i8_x86_64_v3,
        i8,
        CosineStateless,
        reference::reference_cosine_i8_mathematical,
        0x2d077bed2629b18e,
        256,
        { V3::new_checked() }
    );

    #[cfg(target_arch = "x86_64")]
    int_test!(
        test_cosine_i8_x86_64_v4,
        i8,
        CosineStateless,
        reference::reference_cosine_i8_mathematical,
        0x2d077bed2629b18e,
        320,
        { V4::new_checked_miri() }
    );

    //////////
    // LInf //
    //////////

    macro_rules! linf_test {
        ($name:ident,
         $T:ty,
         $reference:path,
         $eps:literal,
         $relative:literal,
         $seed:literal,
         $upper:literal,
         $($arch:tt)*
        ) => {
            #[test]
            fn $name() {
                if let Some(arch) = $($arch)* {
                    let mut rng = StdRng::seed_from_u64($seed);
                    for dim in 0..$upper {
                        stress_test_linf::<$T, _, StdRng, _>(
                            arch,
                            |l| $reference(l).into_inner(),
                            rand_distr::Normal::new(-10.0, 10.0).unwrap(),
                            dim,
                            $eps,
                            $relative,
                            &mut rng,
                        );
                    }
                }
            }
        }
    }

    linf_test!(
        test_linf_f32_scalar,
        f32,
        reference::reference_linf_f32_mathematical,
        1e-6,
        1e-6,
        0xf149c2bcde660128,
        256,
        Some(Scalar::new())
    );

    #[cfg(target_arch = "x86_64")]
    linf_test!(
        test_linf_f32_v3,
        f32,
        reference::reference_linf_f32_mathematical,
        1e-6,
        1e-6,
        0xf149c2bcde660128,
        256,
        V3::new_checked()
    );

    #[cfg(target_arch = "x86_64")]
    linf_test!(
        test_linf_f32_v4,
        f32,
        reference::reference_linf_f32_mathematical,
        1e-6,
        1e-6,
        0xf149c2bcde660128,
        256,
        V4::new_checked_miri()
    );

    linf_test!(
        test_linf_f16_scalar,
        f16,
        reference::reference_linf_f16_mathematical,
        1e-6,
        1e-6,
        0xf149c2bcde660128,
        256,
        Some(Scalar::new())
    );

    #[cfg(target_arch = "x86_64")]
    linf_test!(
        test_linf_f16_v3,
        f16,
        reference::reference_linf_f16_mathematical,
        1e-6,
        1e-6,
        0xf149c2bcde660128,
        256,
        V3::new_checked()
    );

    #[cfg(target_arch = "x86_64")]
    linf_test!(
        test_linf_f16_v4,
        f16,
        reference::reference_linf_f16_mathematical,
        1e-6,
        1e-6,
        0xf149c2bcde660128,
        256,
        V4::new_checked_miri()
    );

    ////////////////
    // Miri Tests //
    ////////////////

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum DataType {
        Float32,
        Float16,
        UInt8,
        Int8,
    }

    trait AsDataType {
        fn as_data_type() -> DataType;
    }

    impl AsDataType for f32 {
        fn as_data_type() -> DataType {
            DataType::Float32
        }
    }

    impl AsDataType for f16 {
        fn as_data_type() -> DataType {
            DataType::Float16
        }
    }

    impl AsDataType for u8 {
        fn as_data_type() -> DataType {
            DataType::UInt8
        }
    }

    impl AsDataType for i8 {
        fn as_data_type() -> DataType {
            DataType::Int8
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Arch {
        Scalar,
        #[expect(non_camel_case_types)]
        X86_64_V3,
        #[expect(non_camel_case_types)]
        X86_64_V4,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct Key {
        arch: Arch,
        left: DataType,
        right: DataType,
    }

    impl Key {
        fn new(arch: Arch, left: DataType, right: DataType) -> Self {
            Self { arch, left, right }
        }
    }

    static MIRI_BOUNDS: LazyLock<HashMap<Key, usize>> = LazyLock::new(|| {
        use Arch::{Scalar, X86_64_V3, X86_64_V4};
        use DataType::{Float16, Float32, Int8, UInt8};

        [
            (Key::new(Scalar, Float32, Float32), 64),
            (Key::new(X86_64_V3, Float32, Float32), 256),
            (Key::new(X86_64_V4, Float32, Float32), 256),
            (Key::new(Scalar, Float16, Float16), 64),
            (Key::new(X86_64_V3, Float16, Float16), 256),
            (Key::new(X86_64_V4, Float16, Float16), 256),
            (Key::new(Scalar, Float32, Float16), 64),
            (Key::new(X86_64_V3, Float32, Float16), 256),
            (Key::new(X86_64_V4, Float32, Float16), 256),
            (Key::new(Scalar, UInt8, UInt8), 64),
            (Key::new(X86_64_V3, UInt8, UInt8), 256),
            (Key::new(X86_64_V4, UInt8, UInt8), 320),
            (Key::new(Scalar, Int8, Int8), 64),
            (Key::new(X86_64_V3, Int8, Int8), 256),
            (Key::new(X86_64_V4, Int8, Int8), 320),
        ]
        .into_iter()
        .collect()
    });

    macro_rules! test_bounds {
        (
            $function:ident,
            $left:ty,
            $left_ex:expr,
            $right:ty,
            $right_ex:expr
        ) => {
            #[test]
            fn $function() {
                let left: $left = $left_ex;
                let right: $right = $right_ex;

                let left_type = <$left>::as_data_type();
                let right_type = <$right>::as_data_type();

                // Scalar
                {
                    let max = MIRI_BOUNDS[&Key::new(Arch::Scalar, left_type, right_type)];
                    for dim in 0..max {
                        let left: Vec<$left> = vec![left; dim];
                        let right: Vec<$right> = vec![right; dim];

                        let arch = diskann_wide::arch::Scalar;
                        simd_op(&L2, arch, left.as_slice(), right.as_slice());
                        simd_op(&IP, arch, left.as_slice(), right.as_slice());
                        simd_op(&CosineStateless, arch, left.as_slice(), right.as_slice());
                    }
                }

                #[cfg(target_arch = "x86_64")]
                if let Some(arch) = V3::new_checked() {
                    let max = MIRI_BOUNDS[&Key::new(Arch::X86_64_V3, left_type, right_type)];
                    for dim in 0..max {
                        let left: Vec<$left> = vec![left; dim];
                        let right: Vec<$right> = vec![right; dim];

                        simd_op(&L2, arch, left.as_slice(), right.as_slice());
                        simd_op(&IP, arch, left.as_slice(), right.as_slice());
                        simd_op(&CosineStateless, arch, left.as_slice(), right.as_slice());
                    }
                }

                #[cfg(target_arch = "x86_64")]
                if let Some(arch) = V4::new_checked_miri() {
                    let max = MIRI_BOUNDS[&Key::new(Arch::X86_64_V4, left_type, right_type)];
                    for dim in 0..max {
                        let left: Vec<$left> = vec![left; dim];
                        let right: Vec<$right> = vec![right; dim];

                        simd_op(&L2, arch, left.as_slice(), right.as_slice());
                        simd_op(&IP, arch, left.as_slice(), right.as_slice());
                        simd_op(&CosineStateless, arch, left.as_slice(), right.as_slice());
                    }
                }
            }
        };
    }

    test_bounds!(miri_test_bounds_f32xf32, f32, 1.0f32, f32, 2.0f32);
    test_bounds!(
        miri_test_bounds_f16xf16,
        f16,
        diskann_wide::cast_f32_to_f16(1.0f32),
        f16,
        diskann_wide::cast_f32_to_f16(2.0f32)
    );
    test_bounds!(
        miri_test_bounds_f32xf16,
        f32,
        1.0f32,
        f16,
        diskann_wide::cast_f32_to_f16(2.0f32)
    );
    test_bounds!(miri_test_bounds_u8xu8, u8, 1u8, u8, 1u8);
    test_bounds!(miri_test_bounds_i8xi8, i8, 1i8, i8, 1i8);
}
