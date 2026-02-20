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

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_providers::test_utils::graph_data_type_utils::GraphDataF32VectorUnitData;

    type TestGraphData = GraphDataF32VectorUnitData;

    #[test]
    fn test_default_associated_data_filter_returns_true_for_all() {
        let filter = default_associated_data_filter::<TestGraphData>();
        // Test that the default filter always returns true
        assert!(filter(&()));
        assert!(filter(&()));
    }

    #[test]
    fn test_default_vector_filter_returns_true_for_all() {
        let filter = default_vector_filter::<TestGraphData>();
        // Test that the default filter always returns true for any vector ID
        assert!(filter(&0));
        assert!(filter(&1));
        assert!(filter(&999));
        assert!(filter(&u32::MAX));
    }
}
