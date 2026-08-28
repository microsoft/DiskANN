/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(super) trait Fold<T> {
    fn fold<F>(self, f: F) -> T
    where
        F: Fn(T, T) -> T;
}

impl<T> Fold<T> for [T; 1] {
    fn fold<F>(self, _f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0] = self;

        a0
    }
}

impl<T> Fold<T> for [T; 2] {
    fn fold<F>(self, f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1] = self;
        f(a0, a1)
    }
}

impl<T> Fold<T> for [T; 3] {
    fn fold<F>(self, f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2] = self;
        f(f(a0, a1), a2)
    }
}

impl<T> Fold<T> for [T; 4] {
    fn fold<F>(self, f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2, a3] = self;
        f(f(a0, a1), f(a2, a3))
    }
}
