/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::Set;
use diskann::ANNResult;
use roaring::{RoaringBitmap, RoaringTreemap};

///////////////////////////////////////////////////
// Macro to implement Set for roaring containers //
///////////////////////////////////////////////////

macro_rules! impl_set_for_roaring {
    ($ty:ty, $elem:ty) => {
        impl Set<$elem> for $ty {
            fn empty_set() -> Self {
                <$ty>::new()
            }

            fn intersection(&self, other: &Self) -> Self {
                self & other
            }

            fn union(&self, other: &Self) -> Self {
                self | other
            }

            fn insert(&mut self, value: &$elem) -> ANNResult<bool> {
                // Return true if element was newly inserted, false if already present
                Ok(<$ty>::insert(self, *value))
            }

            fn remove(&mut self, value: &$elem) -> ANNResult<bool> {
                Ok(<$ty>::remove(self, *value))
            }

            fn contains(&self, value: &$elem) -> ANNResult<bool> {
                Ok(<$ty>::contains(self, *value))
            }

            fn clear(&mut self) -> ANNResult<()> {
                <$ty>::clear(self);
                Ok(())
            }

            fn len(&self) -> ANNResult<usize> {
                Ok(self.len() as usize)
            }

            fn is_empty(&self) -> ANNResult<bool> {
                Ok(self.is_empty())
            }
        }
    };
}

impl_set_for_roaring!(RoaringBitmap, u32);
impl_set_for_roaring!(RoaringTreemap, u64);

#[cfg(test)]
mod tests {
    use super::{RoaringBitmap, RoaringTreemap, Set};

    fn run_set_suite<T, S>()
    where
        S: Set<T>,
        T: From<u32> + Copy + Eq,
    {
        // empty
        let empty = S::empty_set();
        assert_eq!(empty.len().unwrap(), 0);
        assert!(empty.is_empty().unwrap());

        // insert and contains
        let mut a = S::empty_set();
        let v1: T = 1u32.into();
        let v2: T = 2u32.into();
        let v3: T = 3u32.into();
        let v4: T = 4u32.into();
        let v5: T = 5u32.into();

        assert!(a.insert(&v1).unwrap());
        assert!(a.insert(&v2).unwrap());
        assert!(a.insert(&v3).unwrap());
        // duplicate insert should return false since element already exists
        assert!(!a.insert(&v3).unwrap());

        assert_eq!(a.len().unwrap(), 3);
        assert!(a.contains(&v1).unwrap());
        assert!(a.contains(&v2).unwrap());
        assert!(a.contains(&v3).unwrap());
        assert!(!a.contains(&v4).unwrap());

        // is_empty on non-empty set
        assert!(!a.is_empty().unwrap());

        // remove
        assert!(a.remove(&v2).unwrap()); // should return true - element was present
        assert!(!a.remove(&v2).unwrap()); // should return false - element no longer present
        assert!(!a.remove(&v4).unwrap()); // should return false - element was never present
        assert!(!a.contains(&v2).unwrap());
        assert_eq!(a.len().unwrap(), 2);
        // add it back for following tests
        assert!(a.insert(&v2).unwrap()); // should return true - element being newly inserted

        // union
        let mut b = S::empty_set();
        assert!(b.insert(&v3).unwrap());
        assert!(b.insert(&v4).unwrap());
        assert!(b.insert(&v5).unwrap());

        let u = a.union(&b);
        assert_eq!(u.len().unwrap(), 5);
        for v in [v1, v2, v3, v4, v5] {
            assert!(u.contains(&v).unwrap());
        }

        // intersection
        let i = a.intersection(&b);
        assert_eq!(i.len().unwrap(), 1);
        assert!(i.contains(&v3).unwrap());
        assert!(!i.contains(&v1).unwrap());
        assert!(!i.contains(&v4).unwrap());

        // clear
        let mut c = a.clone();
        assert!(c.len().unwrap() > 0);
        c.clear().unwrap();
        assert_eq!(c.len().unwrap(), 0);

        // iteration over owned set
        let mut count = 0usize;
        for _ in a {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn roaring_bitmap_u32() {
        run_set_suite::<u32, RoaringBitmap>();
    }

    #[test]
    fn roaring_treemap_u64() {
        run_set_suite::<u64, RoaringTreemap>();
    }
}
