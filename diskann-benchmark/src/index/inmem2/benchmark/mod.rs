/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt::{Display, Formatter},
    num::NonZeroUsize,
};

use diskann_benchmark_core::{self as dbc};
use diskann_benchmark_runner::{self as dbr, files::InputFile, utils::datatype::DataType};
use diskann_vector::{self as dv};

use crate::{index::inmem2, utils::SimilarityMeasure};

mod build;
mod knn;

// mod dto {
//     use super::*;
//
//     use serde::{Deserialize, Serialize};
//
//     //--------//
//     // Search //
//     //--------//
//
//     #[derive(Debug, Serialize, Deserialize)]
//     pub(super) struct KnnSweep<O> {
//         pub(super) search_n: usize,
//         pub(super) search_l: Vec<usize>,
//         pub(super) recall_k: usize,
//         pub(super) options: O,
//     }
//
//     #[derive(Debug, Serialize, Deserialize)]
//     pub(super) struct KnnSearch<O> {
//         pub(super) queries: InputFile,
//         pub(super) groundtruth: InputFile,
//         pub(super) reps: NonZeroUsize,
//         pub(super) num_threads: Vec<NonZeroUsize>,
//         pub(super) runs: Vec<KnnSweep<O>>,
//     }
//
//     //-------//
//     // Build //
//     //-------//
//
//     #[derive(Debug, Serialize, Deserialize)]
//     pub(super) struct Data {
//         pub(super) data_type: DataType,
//         pub(super) data: InputFile,
//         pub(super) distance: SimilarityMeasure,
//     }
//
//     #[derive(Debug, Serialize, Deserialize)]
//     pub(super) struct BuildParams {
//         pub(super) pruned_degree: usize,
//         pub(super) max_degree: usize,
//         pub(super) l_build: usize,
//         pub(super) alpha: f32,
//         pub(super) num_threads: NonZeroUsize,
//     }
//
//     #[derive(Debug, Serialize, Deserialize)]
//     pub(super) struct Build<Q, O> {
//         pub(super) data: Data,
//         pub(super) build: BuildParams,
//         pub(super) search: KnnSearch<O>,
//         pub(super) quantization: Q,
//     }
// }
//
// #[derive(Debug)]
// struct KnnInstance<O> {
//     knn_params: dbc::search::graph::KnnParams,
//     recall_k: usize,
//     options: O,
// }
//
// impl<O> KnnInstance<O> {
//     fn flatten_from_raw(raw: &[dto::KnnSweep<O::Raw>]) -> anyhow::Result<Vec<Self>>
//     where
//         O: FromRaw,
//     {
//         raw.iter()
//             .flat_map(|sweep| {
//                 let dto::KnnSweep {
//                     search_n,
//                     search_l,
//                     recall_k,
//                     options,
//                 } = sweep;
//
//                 search_l.iter().map(move |search_l| -> anyhow::Result<_> {
//                     let knn_params = dbc::search::graph::KnnParams::new(*search_n, *search_l)?;
//                     Ok(KnnInstance {
//                         knn_params,
//                         recall_k: *recall_k,
//                         options: O::from_raw(options)?,
//                     })
//                 })
//             })
//             .collect()
//     }
// }
//
// impl<O> Display for KnnInstance<O>
// where
//     O: Display,
// {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "knn = {}, search_l = {}, beam_width = {}, options = {}",
//             self.recall_k,
//             self.knn_params.knn.l_value(),
//             self.knn_params.knn.beam_width(),
//             self.options,
//         )
//     }
// }
//
// #[derive(Debug)]
// struct KnnSearch<O> {
//     queries: InputFile,
//     groundtruth: InputFile,
//     reps: NonZeroUsize,
//     num_threads: Vec<NonZeroUsize>,
//     runs: Vec<KnnInstance<O>>,
// }
//
// impl<O> KnnSearch<O> {
//     fn maximum_recall_k(&self) -> usize {
//         self.runs.iter().map(|r| r.recall_k).max().unwrap_or(0)
//     }
// }
//
// impl<O> FromRawWith for KnnSearch<O>
// where
//     O: FromRaw,
// {
//     type Raw = dto::KnnSearch<O::Raw>;
//
//     fn from_raw(
//         raw: &dto::KnnSearch<O::Raw>,
//         checker: Option<&mut dbr::Checker>,
//     ) -> anyhow::Result<Self> {
//         let dto::KnnSearch {
//             queries,
//             groundtruth,
//             reps,
//             num_threads,
//             runs,
//         } = raw;
//
//         let mut queries = queries.clone();
//         let mut groundtruth = groundtruth.clone();
//
//         if let Some(checker) = checker {
//             queries.resolve(checker)?;
//             groundtruth.resolve(checker)?;
//         }
//
//         Ok(Self {
//             queries,
//             groundtruth,
//             reps: *reps,
//             num_threads: num_threads.clone(),
//             runs: KnnInstance::flatten_from_raw(runs)?,
//         })
//     }
// }
//
// impl<O> Display for KnnSearch<O>
// where
//     O: Display,
// {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let mut kv = dbr::utils::fmt::KeyValue::new();
//         kv.push("queries", &self.queries);
//         kv.push("groundtruth", &self.groundtruth);
//         kv.push("reps", &self.reps);
//
//         let num_threads = dbr::utils::fmt::Delimit::new(self.num_threads.iter(), ", ");
//         kv.push("num_threads", &num_threads);
//
//         let runs = dbr::utils::fmt::Delimit::new(self.runs.iter(), "\n").to_string();
//         kv.push("runs", &runs);
//         write!(f, "{}", kv)
//     }
// }
//
//
// #[derive(Debug)]
// struct Data {
//     data_type: DataType,
//     data: InputFile,
//     distance: dv::distance::Metric,
// }
//
// impl FromRawWith for Data {
//     type Raw = dto::Data;
//     fn from_raw(raw: &dto::Data, checker: Option<&mut dbr::Checker>) -> anyhow::Result<Self> {
//         let dto::Data {
//             data_type,
//             data,
//             distance,
//         } = raw;
//
//         let mut data = data.clone();
//         if let Some(checker) = checker {
//             data.resolve(checker)?;
//         }
//
//         Ok(Self {
//             data_type: *data_type,
//             data,
//             distance: (*distance).into(),
//         })
//     }
// }
//
// impl Display for Data {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let mut kv = dbr::utils::fmt::KeyValue::new();
//         kv.push("data_type", &self.data_type);
//         kv.push("data", &self.data);
//         kv.push("distance", &self.distance);
//         write!(f, "{}", kv)
//     }
// }
//
// #[derive(Debug)]
// struct BuildParams {
//     config: diskann::graph::Config,
//     num_threads: NonZeroUsize,
// }
//
// impl BuildParams {
//     fn from_raw(raw: &dto::BuildParams, metric: dv::distance::Metric) -> anyhow::Result<Self> {
//         let dto::BuildParams {
//             pruned_degree,
//             max_degree,
//             l_build,
//             alpha,
//             num_threads,
//         } = raw;
//
//         let config = diskann::graph::config::Builder::new_with(
//             *pruned_degree,
//             diskann::graph::config::MaxDegree::new(*max_degree),
//             *l_build,
//             metric.into(),
//             |b| {
//                 b.alpha(*alpha);
//             },
//         )
//         .build()?;
//
//         Ok(Self {
//             config,
//             num_threads: *num_threads,
//         })
//     }
// }
//
// impl Display for BuildParams {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let mut kv = dbr::utils::fmt::KeyValue::new();
//
//         let pruned_degree = self.config.pruned_degree();
//         let max_degree = self.config.max_degree();
//         let alpha = self.config.alpha();
//         let l_build = self.config.l_build();
//
//         kv.push("pruned_degree", &pruned_degree);
//         kv.push("max_degree", &max_degree);
//         kv.push("alpha", &alpha);
//         kv.push("l_build", &l_build);
//         kv.push("num_threads", &self.num_threads);
//         write!(f, "{}", kv)
//     }
// }
//
// trait AsStaticBuild {
//     const TAG: &str;
//
//     type Quantization: FromRaw;
//     type SearchOptions: FromRaw;
// }
//
// #[derive(Debug)]
// struct StaticBuild<T>
// where
//     T: AsStaticBuild,
// {
//     data: Data,
//     build: BuildParams,
//     search: KnnSearch,
//     quantization: Quantization,
//
//     // The serialized representation of the original input.
//     input: serde_json::Value,
// }
