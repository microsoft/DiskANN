/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::arch::{Target, Target2, dispatch, dispatch2};

// A zero-sized type that we can use to implement a trait.
struct Add;

impl<A: diskann_wide::Architecture> Target2<A, (), &mut [f32], &[f32]> for Add {
    #[inline(always)]
    fn run(self, _: A, dst: &mut [f32], src: &[f32]) {
        std::iter::zip(dst.iter_mut(), src.iter()).for_each(|(d, s)| *d += *s);
    }
}

#[inline(never)]
fn add(dst: &mut [f32], src: &[f32]) {
    dispatch2(Add, dst, src)
}

struct AddV2<'a>(&'a mut [f32], &'a [f32]);

impl<A: diskann_wide::Architecture> Target<A, ()> for AddV2<'_> {
    #[inline]
    fn run(self, _: A) {
        std::iter::zip(self.0.iter_mut(), self.1.iter()).for_each(|(d, s)| *d += *s);
    }
}

#[inline(never)]
fn add_v2(dst: &mut [f32], src: &[f32]) {
    dispatch(AddV2(dst, src))
}

#[test]
fn test_add() {
    let mut dst = vec![1.0, 2.0, 3.0];
    add(&mut dst, &[2.0, 3.0, 4.0]);
    assert_eq!(dst, &[3.0, 5.0, 7.0]);

    let mut dst = vec![1.0, 2.0, 3.0];
    add_v2(&mut dst, &[2.0, 3.0, 4.0]);
    assert_eq!(dst, &[3.0, 5.0, 7.0]);
}
