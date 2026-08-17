/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::mem::MaybeUninit;

#[derive(Debug)]
pub(crate) struct StackBuffer<T: Copy, const N: usize>([MaybeUninit<T>; N]);

impl<T, const N: usize> StackBuffer<T, N>
where
    T: Copy,
{
    pub(crate) fn new() -> Self {
        Self(core::array::from_fn(|_| MaybeUninit::uninit()))
    }

    pub(crate) fn as_mut_slice(&mut self) -> StackSlice<'_, T> {
        StackSlice(&mut self.0)
    }
}

#[derive(Debug)]
pub(crate) struct StackSlice<'a, T: Copy>(&'a mut [MaybeUninit<T>]);

pub(crate) trait Chunked<T>: std::fmt::Debug
where
    T: Copy,
{
    fn next<'a>(&'a mut self, buffer: StackSlice<'a, T>) -> &'a [T];
}

#[derive(Debug)]
pub(crate) struct Iter<I>(pub(crate) I);

impl<I> Chunked<I::Item> for Iter<I>
where
    I: Iterator + std::fmt::Debug,
    I::Item: Copy,
{
    fn next<'a>(&'a mut self, buffer: StackSlice<'a, I::Item>) -> &'a [I::Item] {
        let raw = buffer.0;

        let count = std::iter::zip(raw.iter_mut(), self.0.by_ref())
            .map(|(dst, src)| {
                dst.write(src);
            })
            .count();

        unsafe { raw[..count].assume_init_ref() }
    }
}
