/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::AttributeType;
use crate::set::Set;
use diskann::utils::VectorId;
use diskann::ANNError;
use std::borrow::Cow;

/// An `AttributeAccessor` provides read access to the labels associated with points
/// (identified by their vector id).
///
/// This trait is designed to work with both in-memory and disk-based
/// representations of labels. To avoid forcing unnecessary cloning,
/// methods return a [`Cow`] so implementations can either borrow
/// existing sets or allocate new ones depending on their backend.
///
/// ## Type parameters
/// - `IdType`: identifier type for the point, typically a vector id.
///   Must satisfy [`VectorId`], [`PrimInt`], and [`NumCast`] so it can
///   interoperate with the indexing layer.
///
/// ## Associated types
/// - `AttributeType`: type of the individual labels (e.g. `u32` or `String`).
/// - `SetType`: concrete [`Set`] implementation used to store labels.
///
/// ## Notes
/// Why do we create a separate trait instead of implementing Accessor? The
/// reason is that Accessor comes with its own set of required traits which
/// we do not want to enforce on every client that acts as an attribute
/// accessor. For integration with the rest of the code, we will implement
/// a DocumentProvider that will provide an DocumentAccessor which in turn
/// will return a Document == {vector + attributes} to any where in the code
/// that requires it.
pub trait AttributeAccessor<IdType>: Send + Sync
where
    IdType: VectorId,
{
    /// Element type stored inside the label set. This could be
    /// a key-value pair or just a u32 if we "hash" the kvps into
    /// a number.
    type AT: AttributeType;

    /// Concrete set type that stores the labels for each point.
    type SetType: Set<Self::AT>;

    /// Process the attributes of a single point.
    ///
    /// Parameters:
    /// vec_id: Id of the vector whose labels need to be processed
    /// func: Function that does the processing.
    ///
    /// Returns result of "F" or ANNError if some error occurred
    /// in retrieving the set.
    ///
    fn visit_labels_of_point<F, R>(&mut self, vec_id: IdType, func: F) -> Result<R, ANNError>
    where
        F: FnOnce(IdType, Option<Cow<'_, Self::SetType>>) -> R;

    /// Visit attributes for a collection of point ids.
    ///
    /// Takes an iterator of ids and a callback function that receives
    /// each `(id, set)` pair. This design allows for efficient
    /// batch traversal without forcing the caller to materialize all
    /// sets in memory.
    fn visit_labels_of_points<I, F>(&mut self, ids: I, func: F) -> Result<(), ANNError>
    where
        I: IntoIterator<Item = IdType>,
        F: FnMut(IdType, Option<Cow<'_, Self::SetType>>);
}
