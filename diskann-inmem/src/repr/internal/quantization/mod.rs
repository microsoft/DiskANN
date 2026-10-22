/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(in crate::repr) mod rerank;
pub(in crate::repr) use rerank::{Rerank, Reranker};

/// Distance metric used for quantization.
#[derive(Debug, Clone, Copy)]
pub(in crate::repr) enum Metric {
    SquaredL2,
    InnerProduct,
    Cosine,
}

impl Metric {
    #[cfg(test)]
    fn all() -> [Self; 3] {
        [Self::SquaredL2, Self::InnerProduct, Self::Cosine]
    }
}

impl Metric {
    pub(in crate::repr) fn as_vector_metric(&self) -> diskann_vector::distance::Metric {
        use diskann_vector::distance::Metric as VMetric;

        match self {
            Self::SquaredL2 => VMetric::L2,
            Self::InnerProduct => VMetric::InnerProduct,
            Self::Cosine => VMetric::Cosine,
        }
    }
}

impl From<diskann_vector::distance::Metric> for Metric {
    #[inline]
    fn from(m: diskann_vector::distance::Metric) -> Metric {
        use diskann_vector::distance::Metric as VMetric;

        match m {
            VMetric::L2 => Self::SquaredL2,
            VMetric::InnerProduct => Self::InnerProduct,
            VMetric::Cosine | VMetric::CosineNormalized => Self::Cosine,
        }
    }
}

impl From<Metric> for diskann_quantization::spherical::SupportedMetric {
    #[inline]
    fn from(m: Metric) -> diskann_quantization::spherical::SupportedMetric {
        match m {
            Metric::SquaredL2 => Self::SquaredL2,
            Metric::InnerProduct => Self::InnerProduct,
            Metric::Cosine => Self::Cosine,
        }
    }
}

impl From<diskann_quantization::spherical::SupportedMetric> for Metric {
    #[inline]
    fn from(m: diskann_quantization::spherical::SupportedMetric) -> Metric {
        use diskann_quantization::spherical::SupportedMetric;

        match m {
            SupportedMetric::SquaredL2 => Self::SquaredL2,
            SupportedMetric::InnerProduct => Self::InnerProduct,
            SupportedMetric::Cosine => Self::Cosine,
        }
    }
}

impl From<Metric> for diskann_quantization::product::tables::padded::Metric {
    #[inline]
    fn from(m: Metric) -> diskann_quantization::product::tables::padded::Metric {
        match m {
            Metric::SquaredL2 => Self::SquaredL2,
            Metric::InnerProduct => Self::InnerProduct,
            Metric::Cosine => Self::Cosine,
        }
    }
}
