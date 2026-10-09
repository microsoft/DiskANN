/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Optional buffering for operation-local list memberships.
//!
//! Appends do not materialize stored memberships. Replacement supersedes earlier
//! appends, and retirement supersedes both. A provider can use this helper for
//! read-your-writes behavior and consume it when publishing its build operation.
//! It is not the algorithm's commit protocol; providers may implement that behavior
//! with their own storage or transactions instead.

use hashbrown::{HashMap, hash_map::Entry};

use crate::{ANNError, ANNResult, utils::VectorId};

/// A pending change relative to a list's stored membership.
#[derive(Debug)]
pub enum Membership<I> {
    Append(Vec<I>),
    Replace(Vec<I>),
    Retire,
}

/// Membership state without vector storage or centroid-ID allocation.
///
/// Callers supply stored sizes and members, and validate that list IDs exist.
#[derive(Debug, Default)]
pub struct PendingLists<I: VectorId, L: VectorId> {
    changes: HashMap<L, Membership<I>>,
}

impl<I: VectorId, L: VectorId> PendingLists<I, L> {
    pub fn append(&mut self, list: L, members: Vec<I>) -> ANNResult<()> {
        match self.changes.entry(list) {
            Entry::Vacant(entry) => {
                entry.insert(Membership::Append(members));
            }
            Entry::Occupied(mut entry) => match entry.get_mut() {
                Membership::Append(current) | Membership::Replace(current) => {
                    current.extend(members);
                }
                Membership::Retire => return Err(retired(list)),
            },
        }
        Ok(())
    }

    pub fn set(&mut self, list: L, members: Vec<I>) -> ANNResult<()> {
        if self.is_retired(list) {
            return Err(retired(list));
        }
        self.changes.insert(list, Membership::Replace(members));
        Ok(())
    }

    pub fn retire(&mut self, list: L) -> ANNResult<()> {
        if self.is_retired(list) {
            return Err(retired(list));
        }
        self.changes.insert(list, Membership::Retire);
        Ok(())
    }

    pub fn is_retired(&self, list: L) -> bool {
        matches!(self.changes.get(&list), Some(Membership::Retire))
    }

    pub fn list_size(&self, list: L, stored_size: usize) -> ANNResult<usize> {
        match self.changes.get(&list) {
            None => Ok(stored_size),
            Some(Membership::Append(members)) => stored_size
                .checked_add(members.len())
                .ok_or_else(|| ANNError::message(format!("IVF list {list} size overflows usize"))),
            Some(Membership::Replace(members)) => Ok(members.len()),
            Some(Membership::Retire) => Err(retired(list)),
        }
    }

    /// Borrow current member IDs without packing stored and appended IDs together.
    pub fn members<'a>(
        &'a self,
        list: L,
        stored: &'a [I],
    ) -> ANNResult<impl Iterator<Item = I> + Clone + 'a> {
        let (head, tail): (&[I], &[I]) = match self.changes.get(&list) {
            None => (stored, &[]),
            Some(Membership::Append(members)) => (stored, members),
            Some(Membership::Replace(members)) => (members, &[]),
            Some(Membership::Retire) => return Err(retired(list)),
        };
        Ok(head.iter().chain(tail).copied())
    }

    /// Consume pending state. The iteration order is unspecified.
    pub fn into_changes(self) -> impl Iterator<Item = (L, Membership<I>)> {
        self.changes.into_iter()
    }
}

fn retired<L: VectorId>(list: L) -> ANNError {
    ANNError::message(format!(
        "IVF list {list} is retired in this build operation"
    ))
}
