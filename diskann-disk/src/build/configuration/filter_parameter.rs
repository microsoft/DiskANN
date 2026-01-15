/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_providers::model::graph::traits::GraphDataType;

pub type AssociatedDataFilter<Data> =
    Box<dyn Fn(&<Data as GraphDataType>::AssociatedDataType) -> bool>;

pub fn default_associated_data_filter<Data: GraphDataType>() -> AssociatedDataFilter<Data> {
    Box::new(|_| true)
}

pub type VectorFilter<'a, Data> =
    Box<dyn Fn(&<Data as GraphDataType>::VectorIdType) -> bool + Send + Sync + 'a>;

pub fn default_vector_filter<Data: GraphDataType>() -> VectorFilter<'static, Data> {
    Box::new(|_| true)
}
