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
        #[expect(
            clippy::expect_used,
            reason = "internally - we should never even get close"
        )]
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
        #[expect(
            clippy::expect_used,
            reason = "internally - we should never even get close"
        )]
        self.a
            .len()
            .checked_add(self.b.len())
            .expect("Chain length overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::chain;

    #[test]
    fn empty_both() {
        let c = chain(std::iter::empty::<i32>(), std::iter::empty::<i32>());
        assert_eq!(c.len(), 0);
        assert_eq!(c.collect::<Vec<_>>(), Vec::<i32>::new());
    }

    #[test]
    fn empty_left() {
        let c = chain(std::iter::empty::<i32>(), [1, 2, 3]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn empty_right() {
        let c = chain([1, 2, 3], std::iter::empty::<i32>());
        assert_eq!(c.len(), 3);
        assert_eq!(c.collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn both_non_empty() {
        let c = chain([1, 2], [3, 4, 5]);
        assert_eq!(c.len(), 5);
        assert_eq!(c.collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn len_decreases_as_consumed() {
        let mut c = chain([10, 20], [30]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.next(), Some(10));
        assert_eq!(c.len(), 2);
        assert_eq!(c.next(), Some(20));
        assert_eq!(c.len(), 1);
        assert_eq!(c.next(), Some(30));
        assert_eq!(c.len(), 0);
        assert_eq!(c.next(), None);
    }

    #[test]
    fn clone_is_independent() {
        let c = chain([1, 2], [3]);
        let collected_clone = c.clone().collect::<Vec<_>>();
        let collected_orig = c.collect::<Vec<_>>();
        assert_eq!(collected_clone, collected_orig);
    }

    #[test]
    fn size_hint_matches_len() {
        let c = chain([1, 2, 3], [4, 5]);
        let (lo, hi) = c.size_hint();
        assert_eq!(lo, 5);
        assert_eq!(hi, Some(5));
        assert_eq!(c.len(), 5);
    }
}
