/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use half::f16;

#[test]
fn integer_normalized_cosine_uses_unnormalized_cosine() {
    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        let expected = if metric == Metric::CosineNormalized {
            Metric::Cosine
        } else {
            metric
        };
        assert_eq!(effective_metric::<u8>(metric), expected);
        assert_eq!(effective_metric::<i8>(metric), expected);
        assert_eq!(effective_metric::<f32>(metric), metric);
        assert_eq!(effective_metric::<f16>(metric), metric);
    }
}
