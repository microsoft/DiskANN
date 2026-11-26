/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

///////////
// HasId //
///////////

/// Indicate an association with an id type.
pub trait HasId {
    type Id: crate::utils::VectorId;
}

impl<T> HasId for &T
where
    T: HasId,
{
    type Id = T::Id;
}

impl<T> HasId for &mut T
where
    T: HasId,
{
    type Id = T::Id;
}

#[cfg(any(test, feature = "testing"))]
macro_rules! has_id {
    ($T:ty = $Id:ty) => {
        impl $crate::HasId for $T {
            type Id = $Id;
        }
    };
}

#[cfg(any(test, feature = "testing"))]
pub(crate) use has_id;
