/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use crate::{set::SetProvider, traits::attribute_accessor::AttributeAccessor};
use diskann::utils::VectorId;
use diskann::ANNError;
use std::borrow::Cow;
use std::sync::{Arc, RwLock};

/// Type alias for the SetType used in MappedAttributeAccessor
type AccessorSetType<IdType, SP> =
    <EncodedAttributeAccessor<SP> as AttributeAccessor<IdType>>::SetType;

/// Implementation of a AttributeAccessor where attributes are mapped to
/// u64s for efficient matching. Is a read-only view of attribute data that
/// the MappedAttributeProvider manages. In this, labels are represented as
/// Set<u64>.
///
/// Type Parameters:
///     SP: Store containing attributes per point. In theory this can be
/// any KV store.
///
/// Assumptions:
///     Queries do not have relational operators.
pub struct EncodedAttributeAccessor<SP> {
    //Ideally, we don't want to expose internal details of the provider like
    // locking to an external class like the accessor. However, this is the
    // simplest way to maintain sanity in a multi-threaded environment.
    locked_attr_index: Arc<RwLock<SP>>,
}

impl<SP> EncodedAttributeAccessor<SP> {
    pub fn new(locked_attribute_index: Arc<RwLock<SP>>) -> Self {
        Self {
            locked_attr_index: locked_attribute_index,
        }
    }

    fn invoke_visitor<IdType, F, R>(
        &self,
        vec_id: IdType,
        visitor: F,
        attr_index: std::sync::RwLockReadGuard<'_, SP>,
    ) -> Result<R, ANNError>
    where
        IdType: VectorId,
        SP: SetProvider<IdType, u64>,
        F: FnOnce(IdType, Option<Cow<'_, AccessorSetType<IdType, SP>>>) -> R,
    {
        match attr_index.get(&vec_id) {
            Ok(s) => Ok(visitor(vec_id, s)),
            Err(e) => Err(e.context("Failed to get set from attr_index.")),
        }
    }

    fn invoke_visitor_for_each<IdType, F>(
        &self,
        vec_id: IdType,
        visitor: &mut F,
        attr_index: &std::sync::RwLockReadGuard<'_, SP>,
    ) -> Result<(), ANNError>
    where
        IdType: VectorId,
        SP: SetProvider<IdType, u64>,
        F: FnMut(IdType, Option<Cow<'_, AccessorSetType<IdType, SP>>>),
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

impl<IdType, SP> AttributeAccessor<IdType> for EncodedAttributeAccessor<SP>
where
    IdType: VectorId,
    SP: SetProvider<IdType, u64>,
{
    type AT = u64;

    type SetType = SP::S;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set::roaring_set_provider::RoaringTreemapSetProvider;

    fn provider_with_labels() -> Arc<RwLock<RoaringTreemapSetProvider<u32>>> {
        let mut sp = RoaringTreemapSetProvider::<u32>::new();
        // id 1 has labels {100, 200}; id 5 has label {300}.
        sp.insert(&1u32, &100u64).unwrap();
        sp.insert(&1u32, &200u64).unwrap();
        sp.insert(&5u32, &300u64).unwrap();
        Arc::new(RwLock::new(sp))
    }

    #[test]
    fn visit_labels_of_point_present_and_absent() {
        let mut accessor = EncodedAttributeAccessor::new(provider_with_labels());

        // Present id -> Some set with both labels.
        let count = accessor
            .visit_labels_of_point(1u32, |id, set| {
                assert_eq!(id, 1u32);
                let set = set.expect("expected a set for id 1");
                assert!(set.contains(100u64));
                assert!(set.contains(200u64));
                set.len()
            })
            .unwrap();
        assert_eq!(count, 2u64);

        // Absent id -> None.
        let is_none = accessor
            .visit_labels_of_point(42u32, |_id, set| set.is_none())
            .unwrap();
        assert!(is_none);
    }

    #[test]
    fn visit_labels_of_points_iterates_all() {
        let mut accessor = EncodedAttributeAccessor::new(provider_with_labels());

        let mut seen: Vec<(u32, bool)> = Vec::new();
        accessor
            .visit_labels_of_points([1u32, 5u32, 7u32], |id, set| {
                seen.push((id, set.is_some()));
            })
            .unwrap();

        assert_eq!(seen, vec![(1u32, true), (5u32, true), (7u32, false)]);
    }

    #[test]
    fn visit_recovers_from_poisoned_lock() {
        let provider = provider_with_labels();

        // Poison the lock by panicking while holding the write guard.
        let p2 = provider.clone();
        let _ = std::thread::spawn(move || {
            let mut sp = p2.write().unwrap();
            sp.insert(&9u32, &900u64).unwrap();
            panic!("intentional panic to poison the lock");
        })
        .join();
        assert!(provider.is_poisoned());

        // The accessor should still be able to read despite poisoning.
        let mut accessor = EncodedAttributeAccessor::new(provider);
        let present = accessor
            .visit_labels_of_point(1u32, |_id, set| set.is_some())
            .unwrap();
        assert!(present);

        let mut count = 0usize;
        accessor
            .visit_labels_of_points([1u32, 5u32], |_id, set| {
                if set.is_some() {
                    count += 1;
                }
            })
            .unwrap();
        assert_eq!(count, 2);
    }
}
