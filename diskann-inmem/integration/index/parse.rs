/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    num::{NonZeroU32, NonZeroUsize},
    path::Path,
};

use diskann::graph::search as dgs;
use diskann_benchmark_runner::{Checker, files::InputFile};
use diskann_vector::distance::Metric;

use super::object;
use crate::support::datatype::{self, DataType};

mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case", deny_unknown_fields)]
    pub(super) enum SerdeMetric {
        L2,
        InnerProduct,
        Cosine,
    }

    impl From<SerdeMetric> for Metric {
        fn from(m: SerdeMetric) -> Self {
            match m {
                SerdeMetric::L2 => Metric::L2,
                SerdeMetric::InnerProduct => Metric::InnerProduct,
                SerdeMetric::Cosine => Metric::Cosine,
            }
        }
    }

    impl TryFrom<Metric> for SerdeMetric {
        type Error = anyhow::Error;
        fn try_from(m: Metric) -> anyhow::Result<Self> {
            match m {
                Metric::L2 => Ok(SerdeMetric::L2),
                Metric::InnerProduct => Ok(SerdeMetric::InnerProduct),
                Metric::Cosine => Ok(SerdeMetric::Cosine),
                Metric::CosineNormalized => anyhow::bail!("cosine normalized is not supported"),
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case", deny_unknown_fields)]
    pub(super) enum Preprocess {
        Halve,
        Floor,
    }

    impl From<Preprocess> for datatype::Preprocess {
        fn from(op: Preprocess) -> Self {
            match op {
                Preprocess::Halve => datatype::Preprocess::Halve,
                Preprocess::Floor => datatype::Preprocess::Floor,
            }
        }
    }

    impl From<&datatype::Preprocess> for Preprocess {
        fn from(op: &datatype::Preprocess) -> Self {
            match op {
                datatype::Preprocess::Halve => Preprocess::Halve,
                datatype::Preprocess::Floor => Preprocess::Floor,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Data {
        pub(super) data: InputFile,
        pub(super) labels: InputFile,
        pub(super) queries: InputFile,
        pub(super) metric: SerdeMetric,
        pub(super) data_type: DataType,
        pub(super) preprocess: Vec<Preprocess>,
    }

    impl Data {
        pub(super) fn example() -> Self {
            Self {
                data: InputFile::new("path/to/data"),
                labels: InputFile::new("path/to/labels.jsonl"),
                queries: InputFile::new("path/to/queries"),
                metric: dto::SerdeMetric::L2,
                data_type: DataType::F32,
                preprocess: vec![],
            }
        }
    }

    //-------------------------//
    // Quantization Parameters //
    //-------------------------//

    pub(super) mod quantization {
        use super::*;

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "kebab-case", deny_unknown_fields)]
        pub(in crate::index::parse) enum Rerank {
            None,
            F16,
        }
    }

    pub(super) mod spherical {
        use super::*;

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "kebab-case", deny_unknown_fields)]
        pub(in crate::index::parse) enum Bits {
            One,
            Two,
            Four,
        }
    }

    pub(super) mod scalar {
        use super::*;

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "kebab-case", deny_unknown_fields)]
        pub(in crate::index::parse) enum Bits {
            One,
            Two,
            Four,
            Eight,
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case", deny_unknown_fields)]
    pub(super) enum Representation {
        FullPrecision {
            data_type: DataType,
        },
        Spherical {
            bits: spherical::Bits,
            rerank: quantization::Rerank,
        },
        Product {
            chunks: NonZeroUsize,
            rerank: quantization::Rerank,
        },
        Scalar {
            bits: scalar::Bits,
            override_to_cosine_normalized: bool,
            rerank: quantization::Rerank,
        },
    }

    impl Representation {
        pub(super) fn example() -> Self {
            Self::FullPrecision {
                data_type: DataType::F32,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case", deny_unknown_fields)]
    pub(super) enum Insert {
        Single,
        Multi {
            batch_size: NonZeroU32,
            parallelism: NonZeroUsize,
            intra_batch_candidates: u32,
        },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Build {
        pub(super) pruned_degree: usize,
        pub(super) max_degree: usize,
        pub(super) l_build: usize,
        pub(super) alpha: f32,
        pub(super) insert: Insert,
    }

    impl Build {
        pub(super) fn example() -> Self {
            Self {
                pruned_degree: 16,
                max_degree: 20,
                l_build: 50,
                alpha: 1.2,
                insert: Insert::Single,
            }
        }
    }

    //--------//
    // Search //
    //--------//

    // Knn //

    pub(super) mod knn {
        use super::*;

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(in crate::index::parse) struct Params {
            pub(in crate::index::parse) knn: usize,
            pub(in crate::index::parse) search_l: usize,
            #[serde(deserialize_with = "Deserialize::deserialize")]
            pub(in crate::index::parse) beam_width: Option<usize>,
        }

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(in crate::index::parse) struct Search {
            pub(in crate::index::parse) searches: Vec<Params>,
            pub(in crate::index::parse) groundtruth: InputFile,
        }

        impl Search {
            pub(in crate::index::parse) fn example() -> Self {
                Self {
                    searches: vec![
                        Params {
                            knn: 10,
                            search_l: 50,
                            beam_width: None,
                        },
                        Params {
                            knn: 10,
                            search_l: 50,
                            beam_width: Some(3),
                        },
                        Params {
                            knn: 20,
                            search_l: 100,
                            beam_width: Some(3),
                        },
                    ],
                    groundtruth: InputFile::new("path/to/groundtruth"),
                }
            }
        }
    }

    // Inline Filter //

    pub(super) mod inline_filter {
        use super::*;

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(in crate::index::parse) struct AdaptiveL {
            pub(in crate::index::parse) sample_count: usize,
            pub(in crate::index::parse) scale_factor: f64,
        }

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(in crate::index::parse) struct Params {
            pub(in crate::index::parse) knn: usize,
            pub(in crate::index::parse) search_l: usize,
            #[serde(deserialize_with = "Deserialize::deserialize")]
            pub(in crate::index::parse) beam_width: Option<usize>,
            #[serde(deserialize_with = "Deserialize::deserialize")]
            pub(in crate::index::parse) adaptive_l: Option<AdaptiveL>,
        }

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(in crate::index::parse) struct Search {
            pub(in crate::index::parse) searches: Vec<Params>,
            pub(in crate::index::parse) predicates: InputFile,
            pub(in crate::index::parse) groundtruth: InputFile,
        }

        impl Search {
            pub(in crate::index::parse) fn example() -> Self {
                Self {
                    predicates: InputFile::new("path/to/predicate.jsonl"),
                    groundtruth: InputFile::new("path/to/filtered/groundtruth"),
                    searches: vec![
                        Params {
                            knn: 10,
                            search_l: 50,
                            beam_width: None,
                            adaptive_l: None,
                        },
                        Params {
                            knn: 10,
                            search_l: 50,
                            beam_width: Some(3),
                            adaptive_l: Some(AdaptiveL {
                                sample_count: 100,
                                scale_factor: 16.0,
                            }),
                        },
                    ],
                }
            }
        }
    }

    // Multihop Filter //

    pub(super) mod multihop_filter {
        use super::*;

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(in crate::index::parse) struct Params {
            pub(in crate::index::parse) knn: usize,
            pub(in crate::index::parse) search_l: usize,
            #[serde(deserialize_with = "Deserialize::deserialize")]
            pub(in crate::index::parse) beam_width: Option<usize>,
        }

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(in crate::index::parse) struct Search {
            pub(in crate::index::parse) searches: Vec<Params>,
            pub(in crate::index::parse) predicates: InputFile,
            pub(in crate::index::parse) groundtruth: InputFile,
        }

        impl Search {
            pub(in crate::index::parse) fn example() -> Self {
                Self {
                    searches: vec![
                        Params {
                            knn: 10,
                            search_l: 50,
                            beam_width: None,
                        },
                        Params {
                            knn: 10,
                            search_l: 50,
                            beam_width: Some(3),
                        },
                        Params {
                            knn: 20,
                            search_l: 100,
                            beam_width: Some(3),
                        },
                    ],
                    predicates: InputFile::new("path/to/predicate.jsonl"),
                    groundtruth: InputFile::new("path/to/groundtruth"),
                }
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Search {
        pub(super) knn: knn::Search,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        pub(super) inline_filter: Option<inline_filter::Search>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        pub(super) multihop_filter: Option<multihop_filter::Search>,
    }

    impl Search {
        pub(super) fn example() -> Self {
            Self {
                knn: knn::Search::example(),
                inline_filter: Some(inline_filter::Search::example()),
                multihop_filter: Some(multihop_filter::Search::example()),
            }
        }
    }

    //-------//
    // Entry //
    //-------//

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index) struct Test {
        pub(super) data: Data,
        pub(super) representation: Representation,
        pub(super) build: Build,
        pub(super) search: Search,
    }

    impl Test {
        pub(super) fn example() -> Self {
            Self {
                data: Data::example(),
                representation: Representation::example(),
                build: Build::example(),
                search: Search::example(),
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index) struct PartialTest {
        pub(super) data: Data,
        pub(super) build: Build,
        pub(super) search: knn::Search,
    }

    impl PartialTest {
        pub(super) fn example() -> Self {
            Self {
                data: Data::example(),
                build: Build::example(),
                search: knn::Search::example(),
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct Data {
    pub(super) data: InputFile,
    pub(super) labels: InputFile,
    pub(super) queries: InputFile,
    pub(super) metric: Metric,
    pub(super) data_type: DataType,
    pub(super) preprocess: Vec<datatype::Preprocess>,
}

impl Data {
    fn from_raw(raw: dto::Data, checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let dto::Data {
            mut data,
            mut labels,
            mut queries,
            metric,
            data_type,
            preprocess,
        } = raw;

        if let Some(checker) = checker {
            data.resolve(checker)?;
            labels.resolve(checker)?;
            queries.resolve(checker)?;
        }

        Ok(Self {
            data,
            labels,
            queries,
            metric: metric.into(),
            data_type,
            preprocess: preprocess.into_iter().map(From::from).collect(),
        })
    }

    fn as_raw(&self) -> anyhow::Result<dto::Data> {
        Ok(dto::Data {
            data: self.data.clone(),
            labels: self.labels.clone(),
            queries: self.queries.clone(),
            metric: self.metric.try_into()?,
            data_type: self.data_type,
            preprocess: self.preprocess.iter().map(From::from).collect(),
        })
    }
}

pub(super) mod quantization {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub(in crate::index) enum Rerank {
        None,
        F16,
    }

    impl Rerank {
        pub(super) fn from_raw(raw: dto::quantization::Rerank) -> Self {
            match raw {
                dto::quantization::Rerank::None => Self::None,
                dto::quantization::Rerank::F16 => Self::F16,
            }
        }

        pub(super) fn as_raw(&self) -> dto::quantization::Rerank {
            match self {
                Self::None => dto::quantization::Rerank::None,
                Self::F16 => dto::quantization::Rerank::F16,
            }
        }
    }
}

pub(super) mod spherical {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub(in crate::index) enum Bits {
        One,
        Two,
        Four,
    }

    impl Bits {
        pub(super) fn from_raw(raw: dto::spherical::Bits) -> Self {
            match raw {
                dto::spherical::Bits::One => Self::One,
                dto::spherical::Bits::Two => Self::Two,
                dto::spherical::Bits::Four => Self::Four,
            }
        }

        pub(super) fn as_raw(&self) -> dto::spherical::Bits {
            match self {
                Self::One => dto::spherical::Bits::One,
                Self::Two => dto::spherical::Bits::Two,
                Self::Four => dto::spherical::Bits::Four,
            }
        }
    }
}

pub(super) mod scalar {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub(in crate::index) enum Bits {
        One,
        Two,
        Four,
        Eight,
    }

    impl Bits {
        pub(super) fn from_raw(raw: dto::scalar::Bits) -> Self {
            match raw {
                dto::scalar::Bits::One => Self::One,
                dto::scalar::Bits::Two => Self::Two,
                dto::scalar::Bits::Four => Self::Four,
                dto::scalar::Bits::Eight => Self::Eight,
            }
        }

        pub(super) fn as_raw(&self) -> dto::scalar::Bits {
            match self {
                Self::One => dto::scalar::Bits::One,
                Self::Two => dto::scalar::Bits::Two,
                Self::Four => dto::scalar::Bits::Four,
                Self::Eight => dto::scalar::Bits::Eight,
            }
        }
    }
}

#[derive(Debug)]
pub(super) enum Representation {
    FullPrecision {
        data_type: DataType,
    },
    Spherical {
        bits: spherical::Bits,
        rerank: quantization::Rerank,
    },
    Product {
        chunks: NonZeroUsize,
        rerank: quantization::Rerank,
    },
    // NOTE: We allow overriding to `cosine_normalized` since that is basically L2 in
    // a trench coat. We can at least still get coverage.
    Scalar {
        bits: scalar::Bits,
        override_to_cosine_normalized: bool,
        rerank: quantization::Rerank,
    },
}

impl Representation {
    fn from_raw(raw: dto::Representation) -> Self {
        match raw {
            dto::Representation::FullPrecision { data_type } => Self::FullPrecision { data_type },
            dto::Representation::Spherical { bits, rerank } => Self::Spherical {
                bits: spherical::Bits::from_raw(bits),
                rerank: quantization::Rerank::from_raw(rerank),
            },
            dto::Representation::Product { chunks, rerank } => Self::Product {
                chunks,
                rerank: quantization::Rerank::from_raw(rerank),
            },
            dto::Representation::Scalar {
                bits,
                override_to_cosine_normalized,
                rerank,
            } => Self::Scalar {
                bits: scalar::Bits::from_raw(bits),
                override_to_cosine_normalized,
                rerank: quantization::Rerank::from_raw(rerank),
            },
        }
    }

    fn as_raw(&self) -> dto::Representation {
        match self {
            Self::FullPrecision { data_type } => dto::Representation::FullPrecision {
                data_type: *data_type,
            },
            Self::Spherical { bits, rerank } => dto::Representation::Spherical {
                bits: bits.as_raw(),
                rerank: rerank.as_raw(),
            },
            Self::Product { chunks, rerank } => dto::Representation::Product {
                chunks: *chunks,
                rerank: rerank.as_raw(),
            },
            Self::Scalar {
                bits,
                override_to_cosine_normalized,
                rerank,
            } => dto::Representation::Scalar {
                bits: bits.as_raw(),
                override_to_cosine_normalized: *override_to_cosine_normalized,
                rerank: rerank.as_raw(),
            },
        }
    }
}

#[derive(Debug)]
pub(super) enum Insert {
    Single,
    Multi { batch_size: NonZeroU32 },
}

#[derive(Debug)]
pub(super) struct Build {
    pub(super) config: diskann::graph::Config,
    pub(super) insert: Insert,
}

impl Build {
    fn from_raw(raw: dto::Build, metric: Metric) -> anyhow::Result<Self> {
        use diskann::graph::config as graph_config;

        let dto::Build {
            pruned_degree,
            max_degree,
            l_build,
            alpha,
            insert,
        } = raw;

        let mut builder = graph_config::Builder::new_with(
            pruned_degree,
            graph_config::MaxDegree::new(max_degree),
            l_build,
            metric.into(),
            |b| {
                b.alpha(alpha);
            },
        );

        let insert = match insert {
            dto::Insert::Single => Insert::Single,
            dto::Insert::Multi {
                batch_size,
                parallelism,
                intra_batch_candidates,
            } => {
                let ibc = graph_config::IntraBatchCandidates::new(intra_batch_candidates);
                builder
                    .max_minibatch_par(parallelism.get())
                    .intra_batch_candidates(ibc);
                Insert::Multi { batch_size }
            }
        };

        let config = builder.build()?;
        Ok(Self { config, insert })
    }

    fn as_raw(&self) -> dto::Build {
        use diskann::graph::config::IntraBatchCandidates as IBC;

        let insert = match self.insert {
            Insert::Single => dto::Insert::Single,
            Insert::Multi { batch_size } => dto::Insert::Multi {
                batch_size,
                parallelism: self.config.max_minibatch_par(),
                intra_batch_candidates: match self.config.intra_batch_candidates() {
                    IBC::None => 0,
                    IBC::Max(m) => m.get(),
                    IBC::All => batch_size.get(),
                },
            },
        };

        dto::Build {
            pruned_degree: self.config.pruned_degree().get(),
            max_degree: self.config.max_degree().get(),
            l_build: self.config.l_build().get(),
            alpha: self.config.alpha(),
            insert,
        }
    }
}

#[derive(Debug)]
pub(super) struct Knn {
    pub(super) searches: Vec<(usize, dgs::Knn)>,
    pub(super) groundtruth: InputFile,
}

impl Knn {
    fn from_raw(raw: dto::knn::Search, checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let dto::knn::Search {
            searches,
            mut groundtruth,
        } = raw;

        if let Some(checker) = checker {
            groundtruth.resolve(checker)?;
        }

        fn make(raw: dto::knn::Params) -> anyhow::Result<(usize, dgs::Knn)> {
            let dto::knn::Params {
                knn,
                search_l,
                beam_width,
            } = raw;
            Ok((knn, dgs::Knn::new(search_l, beam_width)?))
        }

        let me = Self {
            searches: searches
                .into_iter()
                .map(make)
                .collect::<anyhow::Result<Vec<_>>>()?,
            groundtruth,
        };

        Ok(me)
    }

    fn as_raw(&self) -> dto::knn::Search {
        let searches = self
            .searches
            .iter()
            .map(|(knn, search)| dto::knn::Params {
                knn: *knn,
                search_l: search.l_value().get(),
                beam_width: Some(search.beam_width().get()),
            })
            .collect();

        dto::knn::Search {
            searches,
            groundtruth: self.groundtruth.clone(),
        }
    }
}

#[derive(Debug)]
pub(super) struct CommonFilter<'a> {
    pub(super) searches: Vec<(usize, object::Filter)>,
    pub(super) predicates: &'a Path,
    pub(super) groundtruth: &'a Path,
}

#[derive(Debug)]
pub(super) struct InlineFilter {
    searches: Vec<(usize, dgs::InlineFilterSearch)>,
    predicates: InputFile,
    groundtruth: InputFile,
}

impl InlineFilter {
    pub(super) fn common(&self) -> CommonFilter<'_> {
        CommonFilter {
            searches: self
                .searches
                .iter()
                .map(|(k, filter)| (*k, object::Filter::Inline(filter.clone())))
                .collect(),
            predicates: self.predicates.as_ref(),
            groundtruth: self.groundtruth.as_ref(),
        }
    }

    fn from_raw(
        raw: dto::inline_filter::Search,
        checker: Option<&mut Checker>,
    ) -> anyhow::Result<Self> {
        let dto::inline_filter::Search {
            searches,
            mut predicates,
            mut groundtruth,
        } = raw;

        if let Some(checker) = checker {
            predicates.resolve(checker)?;
            groundtruth.resolve(checker)?;
        }

        fn make(s: dto::inline_filter::Params) -> anyhow::Result<(usize, dgs::InlineFilterSearch)> {
            let dto::inline_filter::Params {
                knn,
                search_l,
                beam_width,
                adaptive_l,
            } = s;

            let knn_search = dgs::Knn::new(search_l, beam_width)?;
            let adaptive_l = match adaptive_l {
                Some(dto::inline_filter::AdaptiveL {
                    sample_count,
                    scale_factor,
                }) => Some(dgs::AdaptiveL::new(sample_count, scale_factor)?),
                None => None,
            };

            Ok((knn, dgs::InlineFilterSearch::new(knn_search, adaptive_l)))
        }

        let me = Self {
            searches: searches
                .into_iter()
                .map(make)
                .collect::<anyhow::Result<Vec<_>>>()?,
            predicates,
            groundtruth,
        };

        Ok(me)
    }

    fn as_raw(&self) -> dto::inline_filter::Search {
        let searches = self
            .searches
            .iter()
            .map(|(knn, search)| dto::inline_filter::Params {
                knn: *knn,
                search_l: search.inner.l_value().get(),
                beam_width: Some(search.inner.beam_width().get()),
                adaptive_l: search
                    .adaptive_l
                    .as_ref()
                    .map(|a| dto::inline_filter::AdaptiveL {
                        sample_count: a.sample_count(),
                        scale_factor: a.scale_factor(),
                    }),
            })
            .collect();

        dto::inline_filter::Search {
            searches,
            predicates: self.predicates.clone(),
            groundtruth: self.groundtruth.clone(),
        }
    }
}

#[derive(Debug)]
pub(super) struct MultihopFilter {
    searches: Vec<(usize, dgs::MultihopFilterSearch)>,
    predicates: InputFile,
    groundtruth: InputFile,
}

impl MultihopFilter {
    pub(super) fn common(&self) -> CommonFilter<'_> {
        CommonFilter {
            searches: self
                .searches
                .iter()
                .map(|(k, filter)| (*k, object::Filter::Multihop(filter.clone())))
                .collect(),
            predicates: self.predicates.as_ref(),
            groundtruth: self.groundtruth.as_ref(),
        }
    }

    fn from_raw(
        raw: dto::multihop_filter::Search,
        checker: Option<&mut Checker>,
    ) -> anyhow::Result<Self> {
        let dto::multihop_filter::Search {
            searches,
            mut predicates,
            mut groundtruth,
        } = raw;

        if let Some(checker) = checker {
            predicates.resolve(checker)?;
            groundtruth.resolve(checker)?;
        }

        fn make(
            s: dto::multihop_filter::Params,
        ) -> anyhow::Result<(usize, dgs::MultihopFilterSearch)> {
            let dto::multihop_filter::Params {
                knn,
                search_l,
                beam_width,
            } = s;

            let multihop = dgs::MultihopFilterSearch::new(dgs::Knn::new(search_l, beam_width)?);
            Ok((knn, multihop))
        }

        let me = Self {
            searches: searches
                .into_iter()
                .map(make)
                .collect::<anyhow::Result<Vec<_>>>()?,
            predicates,
            groundtruth,
        };

        Ok(me)
    }

    fn as_raw(&self) -> dto::multihop_filter::Search {
        let searches = self
            .searches
            .iter()
            .map(|(knn, search)| dto::multihop_filter::Params {
                knn: *knn,
                search_l: search.inner.l_value().get(),
                beam_width: Some(search.inner.beam_width().get()),
            })
            .collect();

        dto::multihop_filter::Search {
            searches,
            predicates: self.predicates.clone(),
            groundtruth: self.groundtruth.clone(),
        }
    }
}

#[derive(Debug)]
pub(super) struct Search {
    pub(super) knn: Knn,
    pub(super) inline_filter: Option<InlineFilter>,
    pub(super) multihop_filter: Option<MultihopFilter>,
}

impl Search {
    fn from_raw(raw: dto::Search, mut checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let dto::Search {
            knn,
            inline_filter,
            multihop_filter,
        } = raw;

        Ok(Self {
            knn: Knn::from_raw(knn, checker.as_deref_mut())?,
            inline_filter: inline_filter
                .map(|raw| InlineFilter::from_raw(raw, checker.as_deref_mut()))
                .transpose()?,
            multihop_filter: multihop_filter
                .map(|raw| MultihopFilter::from_raw(raw, checker))
                .transpose()?,
        })
    }

    fn as_raw(&self) -> dto::Search {
        dto::Search {
            knn: self.knn.as_raw(),
            inline_filter: self.inline_filter.as_ref().map(InlineFilter::as_raw),
            multihop_filter: self.multihop_filter.as_ref().map(MultihopFilter::as_raw),
        }
    }
}

//////////
// Test //
//////////

/// The input representation used for testing the full index implementation.
#[derive(Debug)]
pub(super) struct Test {
    pub(super) data: Data,
    pub(super) representation: Representation,
    pub(super) build: Build,
    pub(super) search: Search,
}

impl Test {
    fn from_raw(raw: dto::Test, mut checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let data = Data::from_raw(raw.data, checker.as_deref_mut())?;
        let representation = Representation::from_raw(raw.representation);
        let build = Build::from_raw(raw.build, data.metric)?;
        let search = Search::from_raw(raw.search, checker)?;

        Ok(Self {
            data,
            representation,
            build,
            search,
        })
    }

    fn as_raw(&self) -> anyhow::Result<dto::Test> {
        Ok(dto::Test {
            data: self.data.as_raw()?,
            representation: self.representation.as_raw(),
            build: self.build.as_raw(),
            search: self.search.as_raw(),
        })
    }
}

impl diskann_benchmark_runner::Input for Test {
    type Raw = dto::Test;

    fn tag() -> &'static str {
        "integration-test"
    }

    fn from_raw(raw: dto::Test, checker: &mut Checker) -> anyhow::Result<Self> {
        <Test>::from_raw(raw, Some(checker))
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        let raw = self.as_raw()?;
        Ok(serde_json::to_value(raw)?)
    }

    fn example() -> dto::Test {
        dto::Test::example()
    }
}

/////////////////
// PartialTest //
/////////////////

/// The input representation used for testing the partial index implementations.
#[derive(Debug)]
pub(super) struct PartialTest {
    pub(super) data: Data,
    pub(super) build: Build,
    pub(super) search: Knn,
}

impl PartialTest {
    fn from_raw(raw: dto::PartialTest, mut checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let data = Data::from_raw(raw.data, checker.as_deref_mut())?;
        let build = Build::from_raw(raw.build, data.metric)?;
        let search = Knn::from_raw(raw.search, checker)?;

        Ok(Self {
            data,
            build,
            search,
        })
    }

    fn as_raw(&self) -> anyhow::Result<dto::PartialTest> {
        Ok(dto::PartialTest {
            data: self.data.as_raw()?,
            build: self.build.as_raw(),
            search: self.search.as_raw(),
        })
    }
}

impl diskann_benchmark_runner::Input for PartialTest {
    type Raw = dto::PartialTest;

    fn tag() -> &'static str {
        "partial-integration-test"
    }

    fn from_raw(raw: dto::PartialTest, checker: &mut Checker) -> anyhow::Result<Self> {
        <PartialTest>::from_raw(raw, Some(checker))
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        let raw = self.as_raw()?;
        Ok(serde_json::to_value(raw)?)
    }

    fn example() -> dto::PartialTest {
        dto::PartialTest::example()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_parses() {
        let _ = Test::from_raw(<Test as diskann_benchmark_runner::Input>::example(), None).unwrap();
    }
}
