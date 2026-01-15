/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A collection of tools for working with generalized references and scrounging through types.

use sealed::{BoundTo, Sealed};

/// An hybrid combination of reference covariance and borrowing for generalized reference
/// types.
/// ```
/// use diskann_utils::Reborrow;
/// let mut base = vec![1usize, 3usize];
///
/// let borrowed = base.reborrow();
/// assert_eq!(borrowed[0], 1);
/// assert_eq!(borrowed[1], 3);
/// ```
///
/// # Notes
///
/// The extra hidden generic parameter is an implementation of
/// <https://sabrinajewson.orgblog/the-better-alternative-to-lifetime-gats/> and allows
/// [HRTB](https://doc.rust-lang.org/nomicon/hrtb.html) to work properly with shortened
/// borrows.
pub trait Reborrow<'this, Lifetime: Sealed = BoundTo<&'this Self>> {
    type Target;

    /// Borrow `self` into a generalized reference type and reborrow
    fn reborrow(&'this self) -> Self::Target;
}

/// An hybrid combination of reference covariance and borrowing for generalized reference
/// types.
/// ```ignore
/// use diskann_utils::ReborrowMut;
/// let mut base = vec[0, 0];
/// {
///     let mut borrowed = base.reborrow_mut();
///     borrowed[0] = 1;
///     borrowed[1] = 3;
/// }
///
/// assert_eq!(&*base, &*[1, 3]);
/// ```
pub trait ReborrowMut<'this, Lifetime: Sealed = BoundTo<&'this Self>> {
    type Target;

    /// Mutably borrow `self` into a generalized reference type and reborrow.
    fn reborrow_mut(&'this mut self) -> Self::Target;
}

// Custom Impls
impl<'short, T> Reborrow<'short> for &T
where
    T: ?Sized,
{
    type Target = &'short T;
    fn reborrow(&'short self) -> Self::Target {
        self
    }
}

impl<'short, T> ReborrowMut<'short> for &mut T
where
    T: ?Sized,
{
    type Target = &'short mut T;
    fn reborrow_mut(&'short mut self) -> Self::Target {
        self
    }
}

impl<'this, T> Reborrow<'this> for Vec<T> {
    type Target = &'this [T];
    fn reborrow(&'this self) -> Self::Target {
        self
    }
}

impl<'this, T> ReborrowMut<'this> for Vec<T> {
    type Target = &'this mut [T];
    fn reborrow_mut(&'this mut self) -> Self::Target {
        self
    }
}

impl<'this, T> Reborrow<'this> for Box<T>
where
    T: ?Sized,
{
    type Target = &'this T;
    fn reborrow(&'this self) -> Self::Target {
        self
    }
}

impl<'this, T> ReborrowMut<'this> for Box<T>
where
    T: ?Sized,
{
    type Target = &'this mut T;
    fn reborrow_mut(&'this mut self) -> Self::Target {
        self
    }
}

impl<'short, T> Reborrow<'short> for std::borrow::Cow<'_, T>
where
    T: std::borrow::ToOwned + ?Sized,
{
    type Target = &'short T;
    fn reborrow(&'short self) -> Self::Target {
        self
    }
}

impl<'short> Reborrow<'short> for String {
    type Target = &'short str;
    fn reborrow(&'short self) -> Self::Target {
        self
    }
}

impl<'short> ReborrowMut<'short> for String {
    type Target = &'short mut str;
    fn reborrow_mut(&'short mut self) -> Self::Target {
        self
    }
}

///////////
// Lower //
///////////

/// For some problems, neither [`std::ops::Deref`] nor [`Reborrow`] alone are sufficient.
/// This can occur in cases where
///
/// 1. A `?Sized` without lifetimes type is desired for a clean API.
/// 2. A container cannot implement `Deref` to a target type.
///
/// As a concrete example, consider the following:
/// ```rust
/// use diskann_utils::Reborrow;
///
/// // A type participating in generalized references.
/// enum ThisOrThat {
///     This(Vec<f32>),
///     That(Vec<f64>),
/// }
///
/// enum ThisOrThatView<'a> {
///     This(&'a [f32]),
///     That(&'a [f64]),
/// }
///
/// impl<'a> Reborrow<'a> for ThisOrThat {
///     type Target = ThisOrThatView<'a>;
///     fn reborrow(&'a self) -> Self::Target {
///         match self {
///             Self::This(v) => ThisOrThatView::This(&*v),
///             Self::That(v) => ThisOrThatView::That(&*v),
///         }
///     }
/// }
///
/// // We have a function taking a `?Sized` type.
/// fn takes_unsized<T: ?Sized>(_: &T) {}
///
/// fn calls_unsized<'a, U, T: ?Sized>(x: &'a U) {
///     // How do I call `takes_unsized`?
/// }
/// ```
/// The question we're trying to answer is: how do we make a trait that allows both
///
/// * `calls_unsized::<'a, Vec<f32>, [f32]>`
/// * `calls_unsized::<'a, ThisOrThat, ThisOrThatView<'a>>`
///
/// Observe that `Vec<f32>` can use `Deref` to get to `[f32]`, but no such implementation
/// is possible for `ThisOrThat`.
///
/// # `Lower`
///
/// The [`Lower`] trait solves this problem by first lowering `U` to a proxy type that is
/// then dereferencable to `T`. To solve the above example:
///
/// ```rust
/// # use diskann_utils::Reborrow;
///
/// # // A type participating in generalized references.
/// # enum ThisOrThat {
/// #     This(Vec<f32>),
/// #     That(Vec<f64>),
/// # }
///
/// # enum ThisOrThatView<'a> {
/// #     This(&'a [f32]),
/// #     That(&'a [f64]),
/// # }
///
/// # impl<'a> Reborrow<'a> for ThisOrThat {
/// #     type Target = ThisOrThatView<'a>;
/// #     fn reborrow(&'a self) -> Self::Target {
/// #         match self {
/// #             Self::This(v) => ThisOrThatView::This(&*v),
/// #             Self::That(v) => ThisOrThatView::That(&*v),
/// #         }
/// #     }
/// # }
///
/// # // We have a function taking a `?Sized` type.
/// # fn takes_unsized<T: ?Sized>(_: &T) {}
///
/// use diskann_utils::reborrow::Lower;
///
/// fn calls_unsized<'a, U, T: ?Sized>(x: &'a U)
/// where
///     U: Lower<'a, T>,
/// {
///     takes_unsized::<T>(&x.lower())
/// }
///
/// impl<'short> Lower<'short, ThisOrThatView<'short>> for ThisOrThat {
///     type Proxy = diskann_utils::reborrow::Place<ThisOrThatView<'short>>;
///     fn lower(&'short self) -> Self::Proxy {
///         diskann_utils::reborrow::Place(self.reborrow())
///     }
/// }
///
/// fn with_vec(x: &Vec<f32>) {
///     calls_unsized::<Vec<f32>, [f32]>(x)
/// }
///
/// fn with_this_or_that<'a>(x: &'a ThisOrThat) {
///     calls_unsized::<'a, ThisOrThat, ThisOrThatView<'a>>(x)
/// }
/// ```
/// Note the use of [`diskann_utils::reborrow::Place`] to create a type that dereferences to
/// the contained value.
pub trait Lower<'short, T: ?Sized, Lifetime: Sealed = BoundTo<&'short Self>> {
    /// The type of the proxy that dereferences to the target.
    type Proxy: std::ops::Deref<Target = T>;

    /// Cheaply lower `self` to its proxy type.
    fn lower(&'short self) -> Self::Proxy;
}

impl<'a, T> Lower<'a, [T]> for Vec<T> {
    type Proxy = &'a [T];
    fn lower(&'a self) -> Self::Proxy {
        self
    }
}

impl<'a, T> Lower<'a, T> for Box<T>
where
    T: ?Sized,
{
    type Proxy = &'a T;
    fn lower(&'a self) -> Self::Proxy {
        self
    }
}

impl<'a, T> Lower<'a, T> for std::borrow::Cow<'_, T>
where
    T: std::borrow::ToOwned + ?Sized,
{
    type Proxy = &'a T;
    fn lower(&'a self) -> Self::Proxy {
        self
    }
}

/// An async-friendly version of [`Lower`] that further constrains the proxy to be `Send` and
/// `Sync`.
///
/// This trait is automatically implemented for types implementing [`Lower`] whose proxies
/// are already `Send` and `Sync`.
///
/// # Notes
///
/// As of time of writing, adding constraints like
/// ```text
/// T: Lower<'a, T, Proxy: Send + Sync>
/// ```
/// is not sufficient to attach `Send` and `Sync` bounds to the proxy.
pub trait AsyncLower<'short, T: ?Sized, Lifetime: Sealed = BoundTo<&'short Self>> {
    type Proxy: std::ops::Deref<Target = T> + Send + Sync;
    fn async_lower(&'short self) -> Self::Proxy;
}

impl<'short, T, U> AsyncLower<'short, T> for U
where
    T: ?Sized,
    U: Lower<'short, T, Proxy: Send + Sync>,
{
    type Proxy = <Self as Lower<'short, T>>::Proxy;
    fn async_lower(&'short self) -> Self::Proxy {
        self.lower()
    }
}

////////////
// Helper //
////////////

/// A container for types `T` providing an implementation `Deref<Target = T>` as well as
/// reborrowing functionality mapped to `T as Deref` and `T as DerefMut`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Place<T>(pub T);

impl<T> std::ops::Deref for Place<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Place<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<'this, T> Reborrow<'this> for Place<T>
where
    T: std::ops::Deref,
{
    type Target = &'this T::Target;
    fn reborrow(&'this self) -> Self::Target {
        self
    }
}

impl<'this, T> ReborrowMut<'this> for Place<T>
where
    T: std::ops::DerefMut,
{
    type Target = &'this mut T::Target;
    fn reborrow_mut(&'this mut self) -> Self::Target {
        self
    }
}

/// A container that reborrows by cloning the contained value.
///
/// Note that [`ReborrowMut`] is not implemented for this type.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Cloned<T>(pub T)
where
    T: Clone;

impl<T> std::ops::Deref for Cloned<T>
where
    T: Clone,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Cloned<T>
where
    T: Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'this, T> Reborrow<'this> for Cloned<T>
where
    T: Clone,
{
    type Target = Self;

    fn reborrow(&'this self) -> Self::Target {
        self.clone()
    }
}

/// A container that reborrows by copying the contained value.
///
/// Note that [`ReborrowMut`] is not implemented for this type.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Copied<T>(pub T)
where
    T: Copy;

impl<T> std::ops::Deref for Copied<T>
where
    T: Copy,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Copied<T>
where
    T: Copy,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'this, T> Reborrow<'this> for Copied<T>
where
    T: Copy,
{
    type Target = Self;

    fn reborrow(&'this self) -> Self::Target {
        *self
    }
}

macro_rules! trivial_reborrow {
    ($T:ty) => {
        impl<'a> Reborrow<'a> for $T {
            type Target = Self;

            fn reborrow(&'a self) -> Self {
                *self
            }
        }
    };
    ($($T:ty),* $(,)?) => {
        $(trivial_reborrow!($T);)*
    }
}

trivial_reborrow!(half::f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);

/// Helper traits for the [HRTB]() trick.
mod sealed {
    pub trait Sealed: Sized {}
    pub struct BoundTo<T>(T);
    impl<T> Sealed for BoundTo<T> {}
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hrtb_reborrow<T>(_x: T)
    where
        T: for<'a> Reborrow<'a>,
    {
    }

    fn test_hrtb_reborrow_mut<T>(_x: T)
    where
        T: for<'a> ReborrowMut<'a>,
    {
    }

    fn test_reborrow_constrained<T>(x: T) -> String
    where
        T: for<'a> Reborrow<'a, Target: std::fmt::Debug>,
    {
        format!("{:?}", x.reborrow())
    }

    #[test]
    fn hrbt_reborrow() {
        let x: &[usize] = &[10];
        test_hrtb_reborrow(x);
    }

    #[test]
    fn hrbt_reborrow_mut() {
        let x: &mut [usize] = &mut [10];
        test_hrtb_reborrow_mut(x);
    }

    #[test]
    fn reborrow_constrained() {
        let x: &[usize] = &[10];
        let s = test_reborrow_constrained(x);
        assert_eq!(s, "[10]");
    }

    ////////////////////////
    // Reborrow built-ins //
    ////////////////////////

    #[test]
    fn test_slice() {
        let x: &[usize] = &[1, 2, 3];
        let ptr = x.as_ptr();
        let len = x.len();

        let y: &[usize] = x.reborrow();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());
    }

    #[test]
    fn test_mut_slice() {
        let mut x: &mut [usize] = &mut [0, 0, 0];
        let ptr = x.as_ptr();
        let len = x.len();

        let y: &mut [usize] = x.reborrow_mut();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());
        y[0] = 1;
        y[1] = 2;
        y[2] = 3;

        assert_eq!(x, [1, 2, 3]);
    }

    #[test]
    fn test_vec() {
        let x: Vec<usize> = vec![1, 2, 3];
        let ptr = x.as_ptr();
        let len = x.len();

        let y: &[usize] = x.reborrow();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());
    }

    #[test]
    fn test_vec_mut() {
        let mut x: Vec<usize> = vec![0, 0, 0];
        let ptr = x.as_ptr();
        let len = x.len();

        let y: &mut [usize] = x.reborrow_mut();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());
        y[0] = 1;
        y[1] = 2;
        y[2] = 3;

        assert_eq!(x, [1, 2, 3]);
    }

    #[test]
    fn test_box() {
        let x: Box<[usize]> = Box::new([1, 2, 3]);
        let ptr = x.as_ptr();
        let len = x.len();

        let y: &[usize] = x.reborrow();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());
    }

    #[test]
    fn test_box_mut() {
        let mut x: Box<[usize]> = Box::new([0, 0, 0]);
        let ptr = x.as_ptr();
        let len = x.len();

        let y: &mut [usize] = x.reborrow_mut();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());
        y[0] = 1;
        y[1] = 2;
        y[2] = 3;

        assert_eq!(&*x, [1, 2, 3]);
    }

    #[test]
    fn test_cow() {
        let x = &[1, 2, 3];
        let ptr = x.as_ptr();
        let len = x.len();
        let cow = std::borrow::Cow::<[usize]>::Borrowed(x);

        let y: &[usize] = cow.reborrow();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());

        let cow = cow.into_owned();
        let ptr = cow.as_ptr();
        let len = cow.len();

        let y: &[usize] = cow.reborrow();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());
    }

    #[test]
    fn test_string() {
        let mut x = String::from("hello world");
        let ptr = x.as_ptr();
        let len = x.len();

        let y: &str = x.reborrow();
        assert_eq!(y, x);
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());

        let y: &mut str = x.reborrow_mut();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(len, y.len());
        y.make_ascii_uppercase();

        assert_eq!(x, "HELLO WORLD");
    }

    /////////////////////
    // Lower Built-Ins //
    /////////////////////

    #[test]
    fn test_lower_vec() {
        let v = vec![1, 2, 3];
        let ptr = v.as_ptr();
        let slice: &[usize] = v.lower();
        assert_eq!(ptr, slice.as_ptr());
        assert_eq!(slice, [1, 2, 3]);

        let slice: &[usize] = v.async_lower();
        assert_eq!(ptr, slice.as_ptr());
        assert_eq!(slice, [1, 2, 3]);
    }

    #[test]
    fn test_lower_box() {
        let v: Box<[usize]> = Box::new([1, 2, 3]);
        let ptr = v.as_ptr();
        let slice: &[usize] = v.lower();
        assert_eq!(ptr, slice.as_ptr());
        assert_eq!(slice, [1, 2, 3]);

        let slice: &[usize] = v.async_lower();
        assert_eq!(ptr, slice.as_ptr());
        assert_eq!(slice, [1, 2, 3]);
    }

    #[test]
    fn test_lower_cow() {
        let x = &[1, 2, 3];
        let ptr = x.as_ptr();
        let cow = std::borrow::Cow::<[usize]>::Borrowed(x);

        let y: &[usize] = cow.lower();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(y, [1, 2, 3]);

        let cow = cow.into_owned();
        let ptr = cow.as_ptr();

        let y: &[usize] = cow.lower();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(y, [1, 2, 3]);

        let y: &[usize] = cow.async_lower();
        assert_eq!(ptr, y.as_ptr());
        assert_eq!(y, [1, 2, 3]);
    }

    // A type participating in generalized references.
    enum ThisOrThat {
        This(Vec<f32>),
        That(Vec<f64>),
    }

    #[allow(dead_code)]
    enum ThisOrThatView<'a> {
        This(&'a [f32]),
        That(&'a [f64]),
    }

    impl<'a> Reborrow<'a> for ThisOrThat {
        type Target = ThisOrThatView<'a>;
        fn reborrow(&'a self) -> Self::Target {
            match self {
                Self::This(v) => ThisOrThatView::This(v),
                Self::That(v) => ThisOrThatView::That(v),
            }
        }
    }

    // We have a function taking a `?Sized` type.
    fn takes_unsized<T: ?Sized>(_x: &T) -> bool {
        true
    }

    fn calls_unsized<'a, U, T: ?Sized>(x: &'a U) -> bool
    where
        U: Lower<'a, T>,
    {
        takes_unsized::<T>(&x.lower())
    }

    impl<'short> Lower<'short, ThisOrThatView<'short>> for ThisOrThat {
        type Proxy = Place<ThisOrThatView<'short>>;
        fn lower(&'short self) -> Self::Proxy {
            Place(self.reborrow())
        }
    }

    fn with_vec(x: &Vec<f32>) -> bool {
        calls_unsized::<Vec<f32>, [f32]>(x)
    }

    fn with_this_or_that<'a>(x: &'a ThisOrThat) -> bool {
        calls_unsized::<'a, ThisOrThat, ThisOrThatView<'a>>(x)
    }

    #[test]
    fn test_lower_example() {
        assert!(with_vec(&vec![1.0, 2.0, 3.0]));
        assert!(with_this_or_that(&ThisOrThat::This(vec![2.0, 3.0, 4.0])));
        assert!(with_this_or_that(&ThisOrThat::That(vec![2.0, 3.0, 4.0])));
    }

    ///////////
    // Place //
    ///////////

    #[test]
    fn test_place() {
        let mut x: Place<Box<[usize]>> = Place(Box::new([0, 0, 0]));
        // DerefMut through `Box`.
        x[0] = 1;
        x[1] = 2;
        x[2] = 3;

        assert_eq!(&**x, [1, 2, 3]);
        assert_eq!(x.reborrow(), [1, 2, 3]);

        *x = Box::new([2, 3, 4]);
        assert_eq!(&**x, [2, 3, 4]);
        assert_eq!(x.reborrow(), [2, 3, 4]);

        let y = x.reborrow_mut();
        y[0] = 10;
        y[1] = 20;
        y[2] = 30;
        assert_eq!(&**x, [10, 20, 30]);
        assert_eq!(x.reborrow(), [10, 20, 30]);
    }

    /////////////
    // Helpers //
    /////////////

    #[test]
    fn test_cloned() {
        let mut x = Cloned(10);
        assert_eq!(*x, 10);

        let y: Cloned<usize> = x.reborrow();
        assert_eq!(x, y);

        // Test derive copy;
        let z = x;
        assert_eq!(*z, 10);

        // DerefMut
        *x = 50;
        assert_eq!(*x, 50);
        assert_ne!(x, y);
    }

    #[test]
    fn test_copied() {
        let mut x = Copied(10);
        assert_eq!(*x, 10);

        let y: Copied<usize> = x.reborrow();
        assert_eq!(x, y);

        // Test derive copy;
        let z = x;
        assert_eq!(*z, 10);

        // DerefMut
        *x = 50;
        assert_eq!(*x, 50);
        assert_ne!(x, y);
    }

    //////////////////////
    // Trivial Reborrow //
    //////////////////////

    fn reborrow_to_self<T>(x: T) -> T
    where
        for<'a> T: Reborrow<'a, Target = T>,
    {
        x.reborrow()
    }

    #[test]
    fn trivial_reborrows() {
        assert_eq!(
            reborrow_to_self::<half::f16>(Default::default()),
            Default::default()
        );
        assert_eq!(reborrow_to_self::<f32>(1.0f32), 1.0);
        assert_eq!(reborrow_to_self::<f64>(1.0f64), 1.0);

        assert_eq!(reborrow_to_self::<i8>(1), 1);
        assert_eq!(reborrow_to_self::<i16>(1), 1);
        assert_eq!(reborrow_to_self::<i32>(1), 1);
        assert_eq!(reborrow_to_self::<i64>(1), 1);

        assert_eq!(reborrow_to_self::<u8>(1), 1);
        assert_eq!(reborrow_to_self::<u16>(1), 1);
        assert_eq!(reborrow_to_self::<u32>(1), 1);
        assert_eq!(reborrow_to_self::<u64>(1), 1);
    }
}
