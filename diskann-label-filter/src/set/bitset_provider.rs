/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::Set;
use diskann::ANNResult;
use diskann_utils::future::AsyncFriendly;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Hasher that uses the integer value directly as the hash.
/// Identical to the one in roaring_set_provider but kept local to avoid
/// exposing internal details.
#[derive(Default)]
struct IdentityHasher {
    hash: u64,
}

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

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
        self.hash = ((i >> 64) as u64) ^ (i as u64);
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = 0xcbf29ce484222325u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.hash = h;
    }
}

type BuildIdentityHasher = BuildHasherDefault<IdentityHasher>;

/// A compact bitset wrapper that implements `Set<u64>`.
///
/// Attribute IDs are small sequential integers assigned by `AttributeEncoder`,
/// making a bit-vector an ideal representation.
#[derive(Clone, Default, Debug)]
pub struct LabelBitSet(bit_set::BitSet);

impl LabelBitSet {
    pub fn new() -> Self {
        Self(bit_set::BitSet::new())
    }

    /// Iterate over the elements as `u64`.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.0.iter().map(|v| v as u64)
    }

    pub fn contains(&self, value: u64) -> bool {
        self.0.contains(value as usize)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl IntoIterator for LabelBitSet {
    type Item = u64;
    type IntoIter = std::vec::IntoIter<u64>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().map(|v| v as u64).collect::<Vec<u64>>().into_iter()
    }
}

impl Set<u64> for LabelBitSet {
    fn empty_set() -> Self {
        Self::new()
    }

    fn intersection(&self, other: &Self) -> Self {
        let mut result = self.0.clone();
        result.intersect_with(&other.0);
        Self(result)
    }

    fn union(&self, other: &Self) -> Self {
        let mut result = self.0.clone();
        result.union_with(&other.0);
        Self(result)
    }

    fn insert(&mut self, value: &u64) -> ANNResult<bool> {
        let idx = *value as usize;
        let was_absent = !self.0.contains(idx);
        self.0.insert(idx);
        Ok(was_absent)
    }

    fn remove(&mut self, value: &u64) -> ANNResult<bool> {
        let idx = *value as usize;
        let was_present = self.0.contains(idx);
        self.0.remove(idx);
        Ok(was_present)
    }

    fn contains(&self, value: &u64) -> ANNResult<bool> {
        Ok(self.0.contains(*value as usize))
    }

    fn clear(&mut self) -> ANNResult<()> {
        self.0.clear();
        Ok(())
    }

    fn len(&self) -> ANNResult<usize> {
        Ok(self.0.len())
    }

    fn is_empty(&self) -> ANNResult<bool> {
        Ok(self.0.is_empty())
    }
}

/// BitSet-backed provider storing sets of `u64` keyed by `Key`.
pub struct BitSetProvider<Key> {
    index: HashMap<Key, LabelBitSet, BuildIdentityHasher>,
}

impl<Key> BitSetProvider<Key>
where
    Key: Eq + std::hash::Hash + Clone + AsyncFriendly,
{
    pub fn new() -> Self {
        Self {
            index: HashMap::with_hasher(BuildIdentityHasher::default()),
        }
    }

    pub fn get(&self, id: &Key) -> ANNResult<Option<Cow<'_, LabelBitSet>>> {
        match self.index.get(id) {
            Some(s) => Ok(Some(Cow::Borrowed(s))),
            None => Ok(None),
        }
    }

    pub fn count(&self) -> ANNResult<usize> {
        Ok(self.index.len())
    }

    pub fn exists(&self, id: &Key) -> ANNResult<bool> {
        Ok(self.index.contains_key(id))
    }

    pub fn insert(&mut self, id: &Key, value: &u64) -> ANNResult<bool> {
        let set_of_id = self
            .index
            .entry(id.clone())
            .or_insert_with(LabelBitSet::new);
        <LabelBitSet as Set<u64>>::insert(set_of_id, value)
    }

    pub fn delete(&mut self, key: &Key) -> ANNResult<bool> {
        Ok(self.index.remove(key).is_some())
    }

    pub fn delete_from_set(&mut self, key: &Key, value: &u64) -> ANNResult<bool> {
        if let Some(set) = self.index.get_mut(key) {
            <LabelBitSet as Set<u64>>::remove(set, value)
        } else {
            Ok(false)
        }
    }

    pub fn clear(&mut self) -> ANNResult<()> {
        Ok(self.index.clear())
    }
}
