/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[derive(Debug, Clone, Copy)]
pub(super) struct Folder;

impl Folder {
    #[inline]
    pub(super) fn fold<const N: usize, T, F>(x: [T; N], f: F) -> T
    where
        Self: Fold<N>,
        F: Fn(T, T) -> T,
    {
        (Self).__fold(x, f)
    }
}

pub(super) trait Fold<const N: usize> {
    fn __fold<T, F>(self, x: [T; N], f: F) -> T
    where
        F: Fn(T, T) -> T;
}

impl Fold<1> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 1], _f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0] = x;
        a0
    }
}

impl Fold<2> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 2], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1] = x;
        f(a0, a1)
    }
}

impl Fold<3> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 3], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2] = x;
        f(f(a0, a1), a2)
    }
}

impl Fold<4> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 4], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2, a3] = x;
        f(f(a0, a1), f(a2, a3))
    }
}

impl Fold<5> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 5], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2, a3, a4] = x;
        self.__fold([f(a0, a1), f(a2, a3), a4], f)
    }
}

impl Fold<6> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 6], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2, a3, a4, a5] = x;
        self.__fold([f(a0, a1), f(a2, a3), f(a4, a5)], f)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_fold() {
        fn max(x: usize, y: usize) -> usize {
            x.max(y)
        }

        // One
        assert_eq!(Folder::fold([1], max), 1);
        assert_eq!(Folder::fold([2], max), 2);
        assert_eq!(Folder::fold([3], max), 3);

        // Two
        assert_eq!(Folder::fold([0, 10], max), 10);
        assert_eq!(Folder::fold([10, 0], max), 10);

        // Three
        assert_eq!(Folder::fold([0, 0, 10], max), 10);
        assert_eq!(Folder::fold([0, 10, 0], max), 10);
        assert_eq!(Folder::fold([10, 0, 0], max), 10);

        // Four
        assert_eq!(Folder::fold([0, 0, 0, 10], max), 10);
        assert_eq!(Folder::fold([0, 0, 10, 0], max), 10);
        assert_eq!(Folder::fold([0, 10, 0, 0], max), 10);
        assert_eq!(Folder::fold([10, 0, 0, 0], max), 10);
    }
}
