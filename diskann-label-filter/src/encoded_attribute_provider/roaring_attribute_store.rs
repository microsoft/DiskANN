/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    attribute::Attribute,
    encoded_attribute_provider::{
        attribute_encoder::AttributeEncoder, encoded_attribute_accessor::EncodedAttributeAccessor,
    },
    set::{roaring_set_provider::RoaringTreemapSetProvider, SetProvider},
    traits::attribute_store::AttributeStore,
};
use diskann::{
    utils::{IntoUsize, VectorId},
    ANNError, ANNErrorKind, ANNResult,
};
use diskann_utils::future::AsyncFriendly;
use std::sync::{Arc, RwLock};

pub(crate) struct RoaringAttributeStore<IT>
where
    IT: VectorId + IntoUsize + AsyncFriendly,
{
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    index: Arc<RwLock<RoaringTreemapSetProvider<IT>>>,
    inv_index: Arc<RwLock<RoaringTreemapSetProvider<u64>>>,
}

impl<IT> RoaringAttributeStore<IT>
where
    IT: VectorId + IntoUsize,
{
    #[allow(
        dead_code,
        reason = "This will be invoked by callers when they create a document provider."
    )]
    pub fn new() -> Self {
        Self {
            attribute_map: Arc::new(RwLock::new(AttributeEncoder::new())),
            index: Arc::new(RwLock::new(RoaringTreemapSetProvider::<IT>::new())),
            inv_index: Arc::new(RwLock::new(RoaringTreemapSetProvider::<u64>::new())),
        }
    }

    #[cfg(test)]
    pub fn get_index(&self) -> Arc<RwLock<RoaringTreemapSetProvider<IT>>> {
        self.index.clone()
    }

    pub(crate) fn attribute_map(&self) -> Arc<RwLock<AttributeEncoder>> {
        self.attribute_map.clone()
    }
}

impl<IT> AttributeStore<IT> for RoaringAttributeStore<IT>
where
    IT: VectorId + IntoUsize,
{
    type AT = u64;
    type Accessor = EncodedAttributeAccessor<RoaringTreemapSetProvider<IT>>;
    type StoreError = ANNError;

    fn attribute_accessor(&self) -> Result<Self::Accessor, Self::StoreError> {
        Ok(EncodedAttributeAccessor::new(self.index.clone()))
    }

    /// Delete the attributes of a vector represented by the vec_id from the store.
    /// Returns "Result" because we may make this a trait going forward, so even
    /// though this implementation will simply return Ok().
    ///
    fn delete(&self, vec_id: &IT) -> ANNResult<bool>
    where
        IT: VectorId + IntoUsize,
    {
        let vec_id_u64 = (*vec_id).into_usize() as u64;
        let mut deleted = true;

        // Acquire locks in consistent order: index first, then inv_index
        let mut index_guard = self.index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on index",
            )
        })?;
        let mut inv_index_guard = self.inv_index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on inv_index",
            )
        })?;

        let existing_set = match index_guard.get(vec_id)? {
            Some(set) => set,
            None => {
                return Ok(false);
            } //we are in good shape even if the vector id doesn't exist.
        };

        // At this point we have already checked that the id exists in the index.
        // Therefore any failures in delete_from_set() or delete() are logical errors.
        // So we will flag them as such.

        // delete the id from the inverted index.
        for attr_id in existing_set.iter() {
            deleted = deleted && inv_index_guard.delete_from_set(&attr_id, &vec_id_u64)?;
        }
        if !deleted {
            return Err(ANNError::message(
                ANNErrorKind::IndexError,
                "Failed to delete id from the inverted index.",
            ));
        }

        //delete the id from the index.
        deleted = index_guard.delete(vec_id)?; //we know deleted is true so far.
        if deleted {
            Ok(true)
        } else {
            Err(ANNError::message(
                ANNErrorKind::IndexError,
                "Failed to delete id from the index.",
            ))
        }
    }

    fn id_exists(&self, vec_id: &IT) -> ANNResult<bool> {
        let index_guard = self.index.read().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on the label index.",
            )
        })?;
        index_guard.exists(vec_id)
    }

    fn set_element(&self, vec_id: &IT, attributes: &[Attribute]) -> ANNResult<bool>
    where
        IT: VectorId + IntoUsize,
    {
        let id_u64: u64 = (*vec_id).into_usize() as u64;

        //For now, we assume that it is an error if a point has zero attributes.
        if attributes.is_empty() {
            return Err(ANNError::message(
                ANNErrorKind::Opaque,
                "A vector must have atleast one attribute.",
            ));
        }

        // Acquire locks in consistent order: attribute_map, index, inv_index
        let mut attr_map_guard = self.attribute_map.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on attribute_map",
            )
        })?;
        let mut index_guard = self.index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on index",
            )
        })?;
        let mut inv_index_guard = self.inv_index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on inv_index",
            )
        })?;

        // Update the inverted index.
        // Delete all instances of id from the inv_index for the old labels.
        if let Some(set) = index_guard.get(vec_id)? {
            for attr_id in set.iter() {
                //delete_from_set() returns false if the attr_id or id_u64 don't exist. It
                //doesn't make a difference, so we ignore the return value.
                let _ = inv_index_guard.delete_from_set(&attr_id, &id_u64)?;
            }
        };

        // Delete existing entries in the label index
        index_guard.delete(vec_id)?; //returns false if vec_id doesn't exist, but that don't matter to us.

        // Insert entries for the new attributes in the inv_index and index
        for attr in attributes {
            let attr_id = attr_map_guard.insert(attr);
            inv_index_guard.insert(&attr_id, &id_u64)?;
            index_guard.insert(vec_id, &attr_id)?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::AttributeValue;

    fn attr(name: &str, value: &str) -> Attribute {
        Attribute::from_value(name, AttributeValue::String(value.to_owned()))
    }

    #[test]
    fn set_element_then_id_exists_and_delete() {
        let store = RoaringAttributeStore::<u32>::new();
        let attrs = vec![attr("color", "red"), attr("size", "large")];

        // Insert a vector with two attributes.
        assert!(store.set_element(&7u32, &attrs).unwrap());
        assert!(store.id_exists(&7u32).unwrap());

        // A different id does not exist.
        assert!(!store.id_exists(&8u32).unwrap());

        // The attribute map should now contain the inserted attributes.
        {
            let map = store.attribute_map();
            let guard = map.read().unwrap();
            assert!(guard.get(&attr("color", "red")).is_some());
            assert!(guard.get(&attr("size", "large")).is_some());
        }

        // Deleting the vector removes it.
        assert!(store.delete(&7u32).unwrap());
        assert!(!store.id_exists(&7u32).unwrap());
    }

    #[test]
    fn set_element_with_no_attributes_is_error() {
        let store = RoaringAttributeStore::<u32>::new();
        assert!(store.set_element(&1u32, &[]).is_err());
    }

    #[test]
    fn delete_missing_id_returns_false() {
        let store = RoaringAttributeStore::<u32>::new();
        assert!(!store.delete(&99u32).unwrap());
    }

    #[test]
    fn set_element_twice_updates_existing_labels() {
        let store = RoaringAttributeStore::<u32>::new();

        assert!(store.set_element(&3u32, &[attr("k", "v1")]).unwrap());
        // Re-setting the same id exercises the "delete old labels" path.
        assert!(store.set_element(&3u32, &[attr("k", "v2")]).unwrap());
        assert!(store.id_exists(&3u32).unwrap());
    }

    #[test]
    fn attribute_accessor_is_constructible() {
        let store = RoaringAttributeStore::<u32>::new();
        assert!(store.attribute_accessor().is_ok());
    }
}
