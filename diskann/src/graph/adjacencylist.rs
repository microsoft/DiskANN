/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ops::{Deref, DerefMut, Index};

use diskann_vector::contains::ContainsSimd;

/// Represents the out neighbors of a vertex.
///
/// The methods on this type are meant to help ensure that all items in the list are unique.
/// However, unsafe code must not rely on this guarantee.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct AdjacencyList<I> {
    edges: Vec<I>,
}

impl<I> AdjacencyList<I> {
    /// Construct a new empty adjacency list.
    ///
    /// This method does not allocate.
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Construct a new empty adjacency list with capacity to store at least `capacity`
    /// elements without reallocating.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            edges: Vec::with_capacity(capacity),
        }
    }

    /// Return the number of elements that can be store in the [`AdjacencyList`] without
    /// reallocating.
    pub fn capacity(&self) -> usize {
        self.edges.capacity()
    }

    /// Return the last `count` items in the list. This can be useful when combined with
    /// [`Self::insert`] to obtain the newly added edges.
    pub fn last(&self, count: usize) -> Option<&[I]> {
        self.len().checked_sub(count).map(|start| {
            // SAFETY: We've checked that `start <= self.len()`.
            unsafe { self.edges.get_unchecked(start..) }
        })
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&I) -> bool,
    {
        self.edges.retain(f)
    }

    /// Clear the slice.
    pub fn clear(&mut self) {
        self.edges.clear();
    }

    /// Shrink the list to at most `len` items long. If the current length is less than
    /// or equal to `len`, no change is made.
    pub fn truncate(&mut self, len: usize) {
        self.edges.truncate(len)
    }
}

impl<I> AdjacencyList<I>
where
    I: Copy + std::fmt::Debug,
{
    /// Append an element to the back of a collection if it does not already exist.
    ///
    /// Return `true` if `i` was added. Otherwise, return `false`.
    ///
    /// # Panics
    ///
    /// Panics if `Vec::push` would panic while inserting `i`.
    pub fn push(&mut self, i: I) -> bool
    where
        I: ContainsSimd,
    {
        if !self.contains(i) {
            self.edges.push(i);
            true
        } else {
            false
        }
    }

    /// Append all elements of `is` that are not already in the list. Duplicates within `is`
    /// will be removed.
    ///
    /// Return the number of elements inserted.
    ///
    /// # Panics
    ///
    /// Panics if `Vec::push` would panic while inserting `i` for any `i` in `is`.
    pub fn extend_from_slice(&mut self, is: &[I]) -> usize
    where
        I: ContainsSimd,
    {
        is.iter().filter(|&i| self.push(*i)).count()
    }

    /// Check if the slice contains the given node.
    pub fn contains(&self, i: I) -> bool
    where
        I: ContainsSimd,
    {
        I::contains_simd(self, i)
    }

    /// Sort the contents of the adjacency list. This internally uses [`Vec::sort_unstable`].
    pub fn sort(&mut self)
    where
        I: Ord,
    {
        self.edges.sort_unstable()
    }

    /// Construct a new [`AdjacencyList`] from the iterator where we trust that all items
    /// yielded by the iterator are unique.
    ///
    /// The order of elements yielded from `itr` is preserved.
    pub fn from_iter_unique<Itr>(itr: Itr) -> Self
    where
        Itr: UniqueIter<Item = I>,
    {
        Self {
            edges: itr.collect(),
        }
    }

    /// Resize the underlying storage to `capacity` elements and return a guard allowing
    /// full mutable access to the resized span.
    ///
    /// The caller is responsible for ensuring the finalized slice is free of duplicates
    /// and calling [`ResizeGuard::finish`] when complete.
    ///
    /// ```
    /// use diskann::graph::AdjacencyList;
    ///
    /// let mut list = AdjacencyList::<u32>::new();
    /// assert_eq!(list.len(), 0);
    ///
    /// let mut guard = list.resize(4);
    /// assert_eq!(guard.len(), 4);
    /// guard[0] = 1;
    /// guard[1] = 2;
    /// guard[2] = 3;
    ///
    /// // We only populated three elements - let the guard know this.
    /// guard.finish(3);
    ///
    /// // After this, the original list will be configured.
    /// assert_eq!(&*list, &[1, 2, 3]);
    /// ```
    pub fn resize(&mut self, capacity: usize) -> ResizeGuard<'_, I>
    where
        I: Default + ContainsSimd,
    {
        self.edges.resize(capacity, I::default());
        ResizeGuard(self)
    }

    //-----------//
    // Untrusted //
    //-----------//

    /// Construct a new [`AdjacencyList`] from an iterator where we cannot statically prove
    /// that all elements are unique.
    ///
    /// This method will ensure that the items in the returned list are unique according
    /// to the `Ord` implementation.
    ///
    /// The order of elements yielded from `itr` is not preserved.
    pub fn from_iter_untrusted<Itr>(itr: Itr) -> Self
    where
        Itr: IntoIterator<Item = I>,
        I: Ord,
    {
        let mut edges: Vec<_> = itr.into_iter().collect();
        edges.sort_unstable();
        edges.dedup();
        Self { edges }
    }

    //---------//
    // Trusted //
    //---------//

    /// Overwrite the contents of `self` with the contents of `is` with the assumption that
    /// the contents of `is` are all unique.
    ///
    /// Uniqueness will be checked in debug builds but not in release builds.
    pub fn overwrite_trusted(&mut self, is: &[I])
    where
        I: Clone + ContainsSimd,
    {
        self.clear();
        self.edges.extend_from_slice(is);
        self.debug_check_uniqueness();
    }

    /// Apply the function `f` to each element in the adjacency list.
    ///
    /// The caller asserts that after `f` has been applied to each element, the result will
    /// be unique (provided the contents are already unique).
    ///
    /// Uniqueness will be checked in debug builds but not in release builds.
    pub fn remap_trusted<F>(&mut self, f: F)
    where
        F: FnMut(&mut I),
        I: ContainsSimd,
    {
        self.edges.iter_mut().for_each(f);
        self.debug_check_uniqueness();
    }

    //-------//
    // Check //
    //-------//

    /// Return `true` if all elements in `self` appear to be unique.
    pub fn all_unique(&self) -> bool
    where
        I: ContainsSimd,
    {
        let mut other = AdjacencyList::new();
        other.extend_from_slice(self) == self.len()
    }

    fn debug_check_uniqueness(&self)
    where
        I: ContainsSimd,
    {
        // This is not a particularly fast implementation of duplicate check in that it's
        // runtime is `O(len^2)`. However, this is only meant to run in debug mode and
        // allows us to bypass requiring `Eq + Hash` (for hash tables) or `Ord` for sorting
        // and de-duplicating.
        #[cfg(debug_assertions)]
        #[allow(clippy::panic)]
        if !self.all_unique() {
            panic!("duplicate items detected: {:?}", self);
        }
    }
}

impl<I, Idx> Index<Idx> for AdjacencyList<I>
where
    Idx: std::slice::SliceIndex<[I]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.edges[index]
    }
}

impl<I> Default for AdjacencyList<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I> Deref for AdjacencyList<I> {
    type Target = [I];

    fn deref(&self) -> &Self::Target {
        &self.edges
    }
}

/// Convert the `AdjacencyList` into a `Vec` without allocating.
impl<I> From<AdjacencyList<I>> for Vec<I> {
    fn from(list: AdjacencyList<I>) -> Self {
        list.edges
    }
}

/// Allow construction of a new adjacency list from a single element.
impl<I> From<I> for AdjacencyList<I> {
    fn from(value: I) -> Self {
        Self { edges: vec![value] }
    }
}

//////////////////
// Resize Guard //
//////////////////

/// A guard providing mutable access to the contents of an [`AdjacencyList`].
///
/// When finished, [`ResizeGuard::finish`] should be called to properly indicate completion.
/// Otherwise, the contained [`AdjacencyList`] will be cleared.
///
/// # Correct Usage
///
/// ```
/// use diskann::graph::AdjacencyList;
///
/// let mut list = AdjacencyList::<u32>::new();
/// assert_eq!(list.len(), 0);
///
/// let mut guard = list.resize(4);
/// assert_eq!(guard.len(), 4);
/// guard[0] = 1;
/// guard[1] = 2;
/// guard[2] = 3;
///
/// // We only populated three elements - let the guard know this.
/// guard.finish(3);
///
/// // After this, the original list will be configured.
/// assert_eq!(&*list, &[1, 2, 3]);
/// ```
///
/// # When `finish` is not called.
///
/// ```
/// use diskann::graph::AdjacencyList;
///
/// let mut list = AdjacencyList::<u32>::from_iter_untrusted([1, 2, 3]);
/// assert_eq!(list.len(), 3);
///
/// // Drop the guard without calling `finish`.
/// let guard = list.resize(4);
/// std::mem::drop(guard);
///
/// // The parent list is empty.
/// assert!(list.is_empty());
/// ```
#[derive(Debug)]
pub struct ResizeGuard<'a, I>(&'a mut AdjacencyList<I>)
where
    I: Copy + ContainsSimd;

impl<'a, I> ResizeGuard<'a, I>
where
    I: Copy + ContainsSimd + std::fmt::Debug,
{
    /// Consume the guard, truncating the underlying [`AdjacencyList`] to `at_most` length.
    ///
    /// The caller should ensure that the underlying slice only contains unique items.
    /// This is checked in debug builds but not in release builds.
    pub fn finish(self, at_most: usize) {
        self.0.truncate(at_most);
        self.0.debug_check_uniqueness();
        std::mem::forget(self);
    }
}

impl<I> Deref for ResizeGuard<'_, I>
where
    I: Copy + ContainsSimd,
{
    type Target = [I];

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<I> DerefMut for ResizeGuard<'_, I>
where
    I: Copy + ContainsSimd,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.edges
    }
}

impl<'a, I> Drop for ResizeGuard<'a, I>
where
    I: Copy + ContainsSimd,
{
    fn drop(&mut self) {
        self.0.clear()
    }
}

/////////////////
// Unique Iter //
/////////////////

/// A marker trait for iterators that we trust to not yield duplicate items.
///
/// This trait is not marked as `unsafe` and therefore, generic unsafe code may not rely on
/// this guarantee.
pub trait UniqueIter: Iterator {}

/// Sets contain unique items.
impl<K> UniqueIter for std::collections::hash_set::Iter<'_, K> {}

/// Sets contain unique items.
impl<K> UniqueIter for std::collections::hash_set::IntoIter<K> {}

/// Collections with only one item are trivially unique.
impl<T> UniqueIter for std::iter::Once<T> {}

/// Sets contain unique items.
impl<K> UniqueIter for hashbrown::hash_set::IntoIter<K> {}

/// Ranges contain unique items.
impl UniqueIter for std::ops::Range<u32> {}
impl UniqueIter for std::ops::Range<u64> {}

/// Copies of unique items are still unique.
///
/// Note that this this does not necessarily hold for `Clone` because malicious
/// implementations of `Clone` could modify the value.
impl<I> UniqueIter for std::iter::Copied<I>
where
    I: UniqueIter,
    std::iter::Copied<I>: Iterator,
{
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::{
        SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };

    use super::*;

    //---------------------//
    // Vector-like Methods //
    //---------------------//

    #[test]
    fn test_new() {
        let x = AdjacencyList::<u32>::new();
        assert_eq!(x.len(), 0);
        assert!(x.is_empty());
        assert_eq!(x.capacity(), 0);
    }

    #[test]
    fn test_with_capacity() {
        for cap in [0, 1, 2, 5, 10, 100] {
            let mut x = AdjacencyList::<u32>::with_capacity(cap);
            assert_eq!(x.len(), 0);
            assert!(x.is_empty());
            assert!(
                x.capacity() >= cap,
                "got {}, expected at least {}",
                x.capacity(),
                cap
            );

            // Push `cap` items to the queue - ensure that a reallocation didn't happen.
            let ptr = x.as_ptr();
            for i in 0..cap {
                assert!(x.push(i.try_into().unwrap()));
            }
            assert_eq!(x.len(), cap);
            assert_eq!(ptr, x.as_ptr());
        }
    }

    #[test]
    fn test_last() {
        let x = AdjacencyList::<u32>::from_iter_unique(0..10);
        for i in 0..=10 {
            let last = x.last(i).unwrap();
            let expected: Vec<_> = ((10 - i) as u32..10).collect();
            assert_eq!(last, &*expected);
        }
        for i in 11..15 {
            assert!(x.last(i).is_none(), "failed for length {}", i);
        }
    }

    #[test]
    fn test_retain() {
        // Safe to call on an empty list.
        let mut x = AdjacencyList::<u32>::new();
        x.retain(|_| false);
        assert!(x.is_empty());

        // Can clear all items.
        let mut x = AdjacencyList::<u32>::from_iter_unique(0..10);
        x.retain(|_| false);
        assert!(x.is_empty());

        // Can clear no items.
        let mut x = AdjacencyList::<u32>::from_iter_unique(0..10);
        x.retain(|_| true);
        assert_eq!(x.len(), 10);
        assert_eq!(&*x, &*((0..10).collect::<Vec<u32>>()));

        // Clear some items.
        let mut x = AdjacencyList::<u32>::from_iter_unique(0..10);
        x.retain(|i| i % 2 == 0);
        assert_eq!(x.len(), 5);
        assert_eq!(&*x, &[0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_clear() {
        let mut x = AdjacencyList::<u32>::from_iter_unique(0..10);
        assert!(x.all_unique());
        let cap = x.capacity();
        assert_eq!(x.len(), 10);
        x.clear();
        assert!(x.is_empty());
        assert_eq!(x.capacity(), cap, "capacity should remain unchanged");
    }

    #[test]
    fn test_truncate() {
        let mut x = AdjacencyList::<u32>::from_iter_unique(0..10);
        assert!(x.all_unique());
        let ptr = x.as_ptr();
        for i in 0..10 {
            let len = 10 - i;
            x.truncate(len);
            assert_eq!(x.len(), len);
            assert_eq!(ptr, x.as_ptr(), "truncating should not reallocate");
            assert_eq!(&*x, &*((0..len as u32).collect::<Vec<_>>()));
        }
    }

    #[test]
    fn test_to_vec() {
        let x = AdjacencyList::<u32>::from_iter_unique(0..10);
        let ptr = x.as_ptr();

        // The underlying vector should be exactly the same.
        let y: Vec<u32> = x.into();
        assert_eq!(&*y, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(y.as_ptr(), ptr);
    }

    //---------------------------------//
    // Adjacency-list Specific Methods //
    //---------------------------------//

    #[test]
    fn test_push_directed() {
        let mut x = AdjacencyList::<u32>::new();
        assert!(x.push(10));
        assert_eq!(&*x, &[10]);

        assert!(!x.push(10));
        assert_eq!(&*x, &[10]);

        assert!(x.push(0));
        assert_eq!(&*x, &[10, 0]);

        assert!(x.push(12));
        assert_eq!(&*x, &[10, 0, 12]);

        assert!(!x.push(0));
        assert_eq!(&*x, &[10, 0, 12]);

        assert!(!x.push(12));
        assert_eq!(&*x, &[10, 0, 12]);

        x.sort();
        assert_eq!(&*x, &[0, 10, 12]);
    }

    fn test_push_fuzz_impl(domain: Uniform<u32>, ntrials: usize, rng: &mut StdRng) {
        let mut set = HashSet::new();
        let mut list = AdjacencyList::new();

        for _ in 0..ntrials {
            let v = domain.sample(rng);
            let should_insert = set.insert(v);
            let inserted = list.push(v);

            assert_eq!(should_insert, inserted);
            assert_eq!(set.len(), list.len());
            if inserted {
                assert_eq!(list[list.len() - 1], v);
            }
        }
    }

    #[test]
    fn test_push_fuzz() {
        let mut rng = StdRng::seed_from_u64(0x50e02da44abc56c3);
        let domain = Uniform::new(0, 100).unwrap();
        for _ in 0..10 {
            test_push_fuzz_impl(domain, 200, &mut rng);
        }
    }

    #[test]
    fn test_extend_from_slice() {
        let mut x = AdjacencyList::from_iter_untrusted([1, 2, 3, 4]);
        assert!(x.contains(1));
        assert!(!x.contains(5));
        assert!(!x.contains(9));

        assert_eq!(x.extend_from_slice(&[1, 5, 9]), 2);
        assert_eq!(&*x, &[1, 2, 3, 4, 5, 9]);

        fn some(y: &[u32]) -> Option<&[u32]> {
            Some(y)
        }

        assert_eq!(x.last(0), some(&[]));
        assert_eq!(x.last(1), some(&[9]));
        assert_eq!(x.last(2), some(&[5, 9]));

        // Allow the inserted list to have repeates.
        assert_eq!(x.extend_from_slice(&[1, 10, 9, 10, 8]), 2);
        assert_eq!(&*x, &[1, 2, 3, 4, 5, 9, 10, 8]);
        assert_eq!(x.last(2), some(&[10, 8]));

        assert_eq!(x.extend_from_slice(&[]), 0);
        assert_eq!(&*x, &[1, 2, 3, 4, 5, 9, 10, 8]);
    }

    fn test_extend_from_slice_fuzz_impl(
        domain: Uniform<u32>,
        length_distribution: Uniform<usize>,
        ntrials: usize,
        rng: &mut StdRng,
    ) {
        let mut set = HashSet::new();
        let mut list = AdjacencyList::new();

        for _ in 0..ntrials {
            let len = length_distribution.sample(rng);

            // The list of candidates to insert - may contain repeats.
            let to_insert: Vec<_> = (0..len).map(|_| domain.sample(rng)).collect();

            // Use the HashSet to obtain the exact IDs that *should* be inserted.
            let should_be_inserted: Vec<u32> = to_insert
                .iter()
                .copied()
                .filter(|i| set.insert(*i))
                .collect();

            let num_inserted = list.extend_from_slice(&to_insert);

            assert_eq!(num_inserted, should_be_inserted.len());
            assert_eq!(list.last(num_inserted).unwrap(), &*should_be_inserted);
            assert_eq!(set.len(), list.len());
        }
    }

    #[test]
    fn test_extend_from_slice_fuzz() {
        let mut rng = StdRng::seed_from_u64(0x50e02da44abc56c3);
        let domain = Uniform::new(0, 100).unwrap();
        let length_distribution = Uniform::new(0, 10).unwrap();
        for _ in 0..10 {
            test_extend_from_slice_fuzz_impl(domain, length_distribution, 50, &mut rng);
        }
    }

    #[test]
    fn test_from_iter_untrusted() {
        let x = AdjacencyList::<u32>::from_iter_untrusted([]);
        assert!(x.is_empty());

        let x = AdjacencyList::<u32>::from_iter_untrusted([1]);
        assert_eq!(&*x, &[1]);

        let x = AdjacencyList::<u32>::from_iter_untrusted([2, 1]);
        assert_eq!(&*x, &[1, 2]);

        let x = AdjacencyList::<u32>::from_iter_untrusted([1, 2, 1]);
        assert_eq!(&*x, &[1, 2]);
    }

    #[test]
    fn test_overwrite() {
        let mut x = AdjacencyList::<u32>::from_iter_unique(0..10);
        x.overwrite_trusted(&[]);
        assert!(x.is_empty());

        let mut x = AdjacencyList::<u32>::from_iter_unique(0..10);
        x.overwrite_trusted(&[10, 2, 3, 4]);
        assert_eq!(&*x, &[10, 2, 3, 4]);

        let mut x = AdjacencyList::<u32>::new();
        x.overwrite_trusted(&[4, 3, 10, 9]);
        assert_eq!(&*x, &[4, 3, 10, 9]);
    }

    #[test]
    fn test_remap() {
        let mut x = AdjacencyList::<u32>::from_iter_unique(0..10);
        x.remap_trusted(|i| *i += 1);
        assert_eq!(&*x, &*((1..11).collect::<Vec<u32>>()));
    }

    #[test]
    fn test_resize() {
        let mut x = AdjacencyList::<u32>::new();
        {
            let mut guard = x.resize(4);
            assert_eq!(guard.len(), 4);
            guard[0] = 1;
            guard[1] = 2;
            guard[2] = 3;
            guard.finish(3);
        }
        assert_eq!(&*x, &[1, 2, 3]);

        {
            let _guard = x.resize(10);
            // Let `drop` run.
        }
        assert!(x.is_empty());

        // Too long of a length is okay.
        {
            let mut guard = x.resize(3);
            guard.copy_from_slice(&[3, 2, 1]);
            guard.finish(10);
        }
        assert_eq!(&*x, &[3, 2, 1]);

        // Explicitly setting to 0 is okay.
        {
            let guard = x.resize(10);
            guard.finish(0);
        }
        assert!(x.is_empty());
    }
}
