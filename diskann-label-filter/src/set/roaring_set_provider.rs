/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{Set, SetProvider};
use diskann::ANNResult;
use diskann_utils::future::AsyncFriendly;
use roaring::{RoaringBitmap, RoaringTreemap};
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

///Hasher that is specialized for integer keys and simply
/// uses the value of the key as the hash. This is ONLY
/// for use by the Roaring hashmaps.
#[derive(Default)]
struct IdentityHasher {
    hash: u64,
}

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    // ---- integer specializations ----
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.hash = i as u64;
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.hash = i as u64;
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.hash = i as u64;
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        // fold 128 bits into 64 bits deterministically
        self.hash = ((i >> 64) as u64) ^ (i as u64);
    }

    // ---- fallback path ----
    fn write(&mut self, bytes: &[u8]) {
        // Using a FNV-1a algorithm, but this will never
        // come into play.
        let mut h = 0xcbf29ce484222325u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.hash = h;
    }
}

/// A convenient type alias to plug directly into `HashMap`.
type BuildIdentityHasher = BuildHasherDefault<IdentityHasher>;

/// Set Provider implementation for RoaringBitmap based sets.
/// This struct assumes a dense representation of vectors, that is,
/// vector ids are contiguous and there are few gaps between any two ids.
#[derive(Default)]
pub struct RoaringSetProvider<Key> {
    index: HashMap<Key, RoaringBitmap, BuildIdentityHasher>,
}

impl<Key> RoaringSetProvider<Key>
where
    Key: Eq + std::hash::Hash + Clone + AsyncFriendly,
{
    pub fn new() -> Self {
        Self {
            index: HashMap::with_hasher(BuildIdentityHasher::default()),
        }
    }
}

/// Macro to implement SetProvider for roaring containers
macro_rules! impl_set_provider {
    ($provider:ty, $key:ident, $value:ty, $set:ty) => {
        impl<$key> SetProvider<$key, $value> for $provider
        where
            $key: Eq + std::hash::Hash + Clone + AsyncFriendly,
        {
            type S = $set;

            fn get(&'_ self, id: &$key) -> ANNResult<Option<Cow<'_, Self::S>>> {
                match self.index.get(id) {
                    Some(s) => Ok(Some(Cow::Borrowed(s))),
                    None => Ok(None),
                }
            }

            fn count(&self) -> ANNResult<usize> {
                Ok(self.index.len())
            }

            fn exists(&self, id: &Key) -> ANNResult<bool> {
                Ok(self.index.contains_key(id))
            }

            fn insert(&mut self, id: &$key, value: &$value) -> ANNResult<bool> {
                let set_of_id = self
                    .index
                    .entry(id.clone())
                    .or_insert_with(<Self::S as Set<$value>>::empty_set);
                <Self::S as Set<$value>>::insert(set_of_id, value)
            }

            fn insert_values(&mut self, id: &$key, value: &[$value]) -> ANNResult<bool> {
                let set_of_id = self
                    .index
                    .entry(id.clone())
                    .or_insert_with(<Self::S as Set<$value>>::empty_set);
                let mut all_inserted = true;
                for v in value {
                    all_inserted = all_inserted && <Self::S as Set<$value>>::insert(set_of_id, v)?;
                }
                Ok(all_inserted)
            }

            fn delete(&mut self, key: &$key) -> ANNResult<bool> {
                Ok(self.index.remove(key).is_some())
            }

            fn delete_from_set(&mut self, key: &$key, value: &$value) -> ANNResult<bool> {
                if let Some(set) = self.index.get_mut(key) {
                    <Self::S as Set<$value>>::remove(set, value)
                } else {
                    Ok(false)
                }
            }

            fn clear(&mut self) -> ANNResult<()> {
                Ok(self.index.clear())
            }
        }
    };
}

// RoaringBitmap provider with u32 values
impl_set_provider!(RoaringSetProvider<Key>, Key, u32, RoaringBitmap);

/// RoaringTreemap backed provider storing sets of u64.
#[derive(Default)]
pub struct RoaringTreemapSetProvider<Key> {
    index: HashMap<Key, RoaringTreemap, BuildIdentityHasher>,
}

impl<Key> RoaringTreemapSetProvider<Key>
where
    Key: Eq + std::hash::Hash + Clone + AsyncFriendly,
{
    pub fn new() -> Self {
        Self {
            index: HashMap::with_hasher(BuildIdentityHasher::default()),
        }
    }
}

// RoaringTreemap provider with u64 values
impl_set_provider!(RoaringTreemapSetProvider<Key>, Key, u64, RoaringTreemap);

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn test_roaring_set_provider_basic() {
        let mut provider: RoaringSetProvider<u32> = RoaringSetProvider::new();

        // initial state
        assert_eq!(provider.count().unwrap(), 0usize);

        // get on empty provider
        let set = provider.get(&1).unwrap();
        assert!(set.is_none());

        // insertion
        assert!(provider.insert(&1, &10).unwrap());
        assert!(provider.insert(&1, &20).unwrap());
        assert!(provider.insert(&2, &30).unwrap());

        // count after insertions
        assert_eq!(provider.count().unwrap(), 2usize);

        // get after insertions
        let set1 = provider.get(&1).unwrap().unwrap();
        assert_eq!(set1.len(), 2);
        assert!(set1.contains(10));
        assert!(set1.contains(20));

        let set2 = provider.get(&2).unwrap().unwrap();
        assert_eq!(set2.len(), 1);
        assert!(set2.contains(30));

        // non existent key
        assert!(provider.get(&3).unwrap().is_none());
    }

    #[test]
    fn test_roaring_set_provider_delete() {
        let mut provider: RoaringSetProvider<u32> = RoaringSetProvider::new();

        provider.insert(&1, &10).unwrap();
        provider.insert(&1, &20).unwrap();
        provider.insert(&2, &30).unwrap();

        // delete_from_set
        assert!(provider.delete_from_set(&1, &10).unwrap());

        // verify deletion
        let set1 = provider.get(&1).unwrap().unwrap();
        assert_eq!(set1.len(), 1);
        assert!(!set1.contains(10));
        assert!(set1.contains(20));

        // delete non existent value returns false
        assert!(!provider.delete_from_set(&1, &100).unwrap());

        // delete_from_set on non existent key returns false
        assert!(!provider.delete_from_set(&3, &10).unwrap());

        // delete entire entry
        assert!(provider.delete(&1).unwrap());
        assert_eq!(provider.count().unwrap(), 1usize);

        assert!(provider.get(&1).unwrap().is_none());

        // delete non existent entry returns false
        assert!(!provider.delete(&3).unwrap());
    }

    #[test]
    fn test_roaring_set_provider_cow() {
        let mut provider: RoaringSetProvider<u32> = RoaringSetProvider::new();

        provider.insert(&1, &10).unwrap();
        provider.insert(&1, &20).unwrap();

        // Borrowed when key exists
        let set1 = provider.get(&1).unwrap().unwrap();
        assert!(matches!(set1, Cow::Borrowed(_)));
        //if key doesn't exist, then get() returns None.
    }

    #[test]
    fn test_roaring_treemap_provider_basic() {
        let mut provider: RoaringTreemapSetProvider<u32> = RoaringTreemapSetProvider::new();

        // initial state
        assert_eq!(provider.count().unwrap(), 0usize);

        // get on empty provider
        assert!(provider.get(&1).unwrap().is_none());

        // insertion with u64 values
        let big_value: u64 = 1 << 40;

        assert!(provider.insert(&1, &10).unwrap());
        assert!(provider.insert(&1, &big_value).unwrap());
        assert!(provider.insert(&2, &30).unwrap());

        // count after insertions
        assert_eq!(provider.count().unwrap(), 2usize);

        // get after insertions
        let set1 = provider.get(&1).unwrap().unwrap();
        assert_eq!(set1.len(), 2);
        assert!(set1.contains(10));
        assert!(set1.contains(big_value));

        let set2 = provider.get(&2).unwrap().unwrap();
        assert_eq!(set2.len(), 1);
        assert!(set2.contains(30));
    }

    #[test]
    fn test_roaring_treemap_provider_delete() {
        let mut provider: RoaringTreemapSetProvider<u32> = RoaringTreemapSetProvider::new();

        let big_value: u64 = 1 << 40;

        provider.insert(&1, &10).unwrap();
        provider.insert(&1, &big_value).unwrap();
        provider.insert(&2, &30).unwrap();

        // delete_from_set
        assert!(provider.delete_from_set(&1, &10).unwrap());

        // verify the value was deleted
        let set1 = provider.get(&1).unwrap().unwrap();
        assert_eq!(set1.len(), 1);
        assert!(!set1.contains(10));
        assert!(set1.contains(big_value));

        // delete entire entry
        assert!(provider.delete(&1).unwrap());
        assert_eq!(provider.count().unwrap(), 1usize);

        // deleted entry returns empty set
        assert!(provider.get(&1).unwrap().is_none());
    }
}
