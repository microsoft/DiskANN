/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// An iterator chain that implements [`ExactSizeIterator`] by assuming the
/// combined length of `A` and `B` does not overflow `usize`.
///
/// [`std::iter::Chain`] deliberately does not implement `ExactSizeIterator`
/// because `a.len() + b.len()` can theoretically overflow. In our domain,
/// both sides are small (neighbor lists, candidate pools) so overflow is
/// impossible.
#[derive(Debug, Clone)]
pub(crate) struct Chain<A, B> {
    a: A,
    b: B,
}

impl<A, B> Chain<A, B> {
    fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

pub(crate) fn chain<T, U>(a: T, b: U) -> Chain<T::IntoIter, U::IntoIter>
where
    T: IntoIterator,
    U: IntoIterator<Item = T::Item>,
{
    Chain::new(a.into_iter(), b.into_iter())
}

impl<A, B> Iterator for Chain<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().or_else(|| self.b.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lo, _) = self.a.size_hint();
        let (b_lo, _) = self.b.size_hint();
        let len = a_lo.checked_add(b_lo).expect("Chain length overflow");
        (len, Some(len))
    }
}

impl<A, B> ExactSizeIterator for Chain<A, B>
where
    A: ExactSizeIterator,
    B: ExactSizeIterator<Item = A::Item>,
{
    fn len(&self) -> usize {
        self.a
            .len()
            .checked_add(self.b.len())
            .expect("Chain length overflow")
    }
}
