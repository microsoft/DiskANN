/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::marker::PhantomData;

use diskann::utils::VectorRepr;
use diskann_providers::model::graph::traits::GraphDataType;

pub(super) struct GraphData<T>(PhantomData<T>);

impl<T> GraphDataType for GraphData<T>
where
    T: VectorRepr,
{
    type VectorIdType = u32;
    type VectorDataType = T;
    type AssociatedDataType = ();
}
