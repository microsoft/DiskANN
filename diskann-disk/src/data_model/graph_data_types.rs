/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::hash::Hash;

use diskann::utils::{VectorId, VectorRepr};
use serde::{Deserialize, Serialize};

/// `GraphDataType` is a trait used for specifying the properties of a graph.
/// For each of the associated type, it describes the traits bound required for ANN algorithm to use that type.
/// Noted that the trait bounds are define on the associated type of the this trait. because these are the requirements for a type to be used in the graph.
///
/// # Associated Types
///
/// * `AssociatedDataType`: This type is for any associated data carried by the graph nodes.
///
///   * `Serialize` & `Deserialize`: These traits are necessary for the same reasons as above,
///     namely to allow instances of this type to be saved and loaded.
///
///   * `Send` & `Sync`: These traits are again necessary for multithreaded access.
pub trait GraphDataType: Send + Sync + 'static {
    type VectorDataType: VectorRepr;

    type AssociatedDataType: Sized
        + Serialize
        + for<'de> Deserialize<'de>
        + Send
        + Sync
        + Clone
        + Default
        + std::cmp::Eq
        + std::cmp::PartialEq
        + Hash
        + Copy;

    // FromPrimitive and ToPrimitive are used for converting between VectorIdType and u32/u64.
    type VectorIdType: VectorId;
}

/// An adhoc `GraphDataType` for implementations that only need the `VectorDataType`
/// and `VectorIdType`.
///
/// This type defaults to using `u32` for the ID type for extra convenience.
pub struct AdHoc<T, I = u32> {
    data: std::marker::PhantomData<T>,
    id: std::marker::PhantomData<I>,
}

impl<T, I> GraphDataType for AdHoc<T, I>
where
    T: VectorRepr,
    I: VectorId + 'static,
{
    type VectorDataType = T;
    type AssociatedDataType = ();
    type VectorIdType = I;
}
