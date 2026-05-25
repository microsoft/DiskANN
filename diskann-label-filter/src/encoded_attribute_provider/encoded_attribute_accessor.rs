/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use crate::{set::bitset_provider::{BitSetProvider, LabelBitSet}, traits::attribute_accessor::AttributeAccessor};
use diskann::utils::VectorId;
use diskann::ANNError;
use std::borrow::Cow;
use std::sync::{Arc, RwLock};

/// Implementation of a AttributeAccessor where attributes are mapped to
/// u64s for efficient matching. Is a read-only view of attribute data that
/// the MappedAttributeProvider manages. In this, labels are represented as
/// Set<u64>.
///
/// Type Parameters:
///     IdType: The vector id type used as keys in the BitSetProvider.
///
/// Assumptions:
///     Queries do not have relational operators.
pub struct EncodedAttributeAccessor<IdType> {
    //Ideally, we don't want to expose internal details of the provider like
    // locking to an external class like the accessor. However, this is the
    // simplest way to maintain sanity in a multi-threaded environment.
    locked_attr_index: Arc<RwLock<BitSetProvider<IdType>>>,
}

impl<IdType> EncodedAttributeAccessor<IdType> {
    pub fn new(locked_attribute_index: Arc<RwLock<BitSetProvider<IdType>>>) -> Self {
        Self {
            locked_attr_index: locked_attribute_index,
        }
    }

    fn invoke_visitor<F, R>(
        &self,
        vec_id: IdType,
        visitor: F,
        attr_index: std::sync::RwLockReadGuard<'_, BitSetProvider<IdType>>,
    ) -> Result<R, ANNError>
    where
        IdType: VectorId,
        F: FnOnce(IdType, Option<Cow<'_, LabelBitSet>>) -> R,
    {
        match attr_index.get(&vec_id) {
            Ok(s) => Ok(visitor(vec_id, s)),
            Err(e) => Err(e.context("Failed to get set from attr_index.")),
        }
    }

    fn invoke_visitor_for_each<F>(
        &self,
        vec_id: IdType,
        visitor: &mut F,
        attr_index: &std::sync::RwLockReadGuard<'_, BitSetProvider<IdType>>,
    ) -> Result<(), ANNError>
    where
        IdType: VectorId,
        F: FnMut(IdType, Option<Cow<'_, LabelBitSet>>),
    {
        match attr_index.get(&vec_id) {
            Ok(s) => {
                visitor(vec_id, s);
                Ok(())
            }
            Err(e) => Err(e.context("Failed to get a set from the attr_index.")),
        }
    }
}

impl<IdType> AttributeAccessor<IdType> for EncodedAttributeAccessor<IdType>
where
    IdType: VectorId,
{
    type AT = u64;

    type SetType = LabelBitSet;

    fn visit_labels_of_point<F, R>(&mut self, vec_id: IdType, visitor: F) -> Result<R, ANNError>
    where
        F: FnOnce(IdType, Option<Cow<'_, Self::SetType>>) -> R,
    {
        match self.locked_attr_index.read() {
            Ok(attr_index) => self.invoke_visitor(vec_id, visitor, attr_index),
            Err(poison_error) => {
                //since we know that visitor will not modify state, even if it panics, the
                //attr_index will be in good shape, so we can clear the poison.
                //Writes should not panic anyway.
                let guard = poison_error.into_inner();
                self.invoke_visitor(vec_id, visitor, guard)
            }
        }
    }

    fn visit_labels_of_points<I, F>(&mut self, ids: I, mut visitor: F) -> Result<(), ANNError>
    where
        I: IntoIterator<Item = IdType>,
        F: FnMut(IdType, Option<Cow<'_, Self::SetType>>),
    {
        match self.locked_attr_index.read() {
            Ok(attr_index) => {
                for id in ids {
                    self.invoke_visitor_for_each(id, &mut visitor, &attr_index)?;
                }
                Ok(())
            }
            Err(poison_error) => {
                //If the lock is poisoned, we can clear it and try the operation, because
                //visitors are not modifying state and we control that writes will not panic.
                let attr_index = poison_error.into_inner();
                for id in ids {
                    self.invoke_visitor_for_each(id, &mut visitor, &attr_index)?;
                }
                Ok(())
            }
        }
    }
}
