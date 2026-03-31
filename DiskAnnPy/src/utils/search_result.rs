/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashMap;

use diskann_disk::utils::statistics::{self, QueryStatistics};
use pyo3::prelude::*;

#[pyclass]
pub struct SearchResult {
    #[pyo3(get)]
    pub ids: Vec<u32>,
    #[pyo3(get)]
    pub distances: Vec<f32>,
}

#[pyclass]
pub struct BatchSearchResultWithStats {
    #[pyo3(get)]
    pub ids: Vec<Vec<u32>>,
    #[pyo3(get)]
    pub distances: Vec<Vec<f32>>,
    #[pyo3(get)]
    pub search_stats: HashMap<String, f64>,
}

#[pyclass]
pub struct BatchRangeSearchResultWithStats {
    #[pyo3(get)]
    pub lims: Vec<usize>,
    #[pyo3(get)]
    pub ids: Vec<u32>,
    #[pyo3(get)]
    pub distances: Vec<f32>,
    #[pyo3(get)]
    pub search_stats: HashMap<String, f64>,
}

#[derive(Clone)]
pub struct SearchStats {
    pub mean_latency: Option<f64>,
    pub latency_999: Option<u128>,
    pub latency_95: Option<u128>,
    pub mean_cpus: Option<f64>,
    pub mean_io_time: Option<f64>,
    pub mean_ios: Option<f64>,
    pub mean_comps: Option<f64>,
    pub mean_hops: Option<f64>,
}

impl SearchStats {
    pub fn from_stats_slice(statistics: &[QueryStatistics]) -> Self {
        let mean_latency =
            statistics::get_mean_stats(statistics, |stats| stats.total_execution_time_us as f64);

        let latency_999 = statistics::get_percentile_stats(statistics, 0.999, |stats| {
            stats.total_execution_time_us
        });
        let latency_95 = statistics::get_percentile_stats(statistics, 0.95, |stats| {
            stats.total_execution_time_us
        });
        let mean_ios = statistics::get_mean_stats(statistics, |stats| stats.total_io_operations);
        let mean_io_time = statistics::get_mean_stats(statistics, |stats| stats.io_time_us as f64);
        let mean_cpus = statistics::get_mean_stats(statistics, |stats| stats.cpu_time_us as f64);
        let mean_comps =
            statistics::get_mean_stats(statistics, |stats| stats.total_comparisons as f64);
        let mean_hops = statistics::get_mean_stats(statistics, |stats| stats.search_hops as f64);

        SearchStats {
            mean_latency: Some(mean_latency),
            latency_999: Some(latency_999),
            latency_95: Some(latency_95),
            mean_cpus: Some(mean_cpus),
            mean_io_time: Some(mean_io_time),
            mean_ios: Some(mean_ios),
            mean_comps: Some(mean_comps),
            mean_hops: Some(mean_hops),
        }
    }

    // function supporting statistics objects from the in-memory index, which
    // are different from the static disk index
    pub fn from_stats_slice_inmem(cmps: &[u32]) -> Self {
        let mean_comps = cmps.iter().map(|v| *v as f64).sum::<f64>() / (cmps.len() as f64);

        SearchStats {
            mean_latency: None,
            latency_999: None,
            latency_95: None,
            mean_cpus: None,
            mean_io_time: None,
            mean_ios: None,
            mean_comps: Some(mean_comps),
            mean_hops: None,
        }
    }

    fn to_dict(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        if let Some(mean_latency) = self.mean_latency {
            map.insert("mean_latency".to_string(), mean_latency);
        }
        if let Some(latency_999) = self.latency_999 {
            map.insert("latency_999".to_string(), latency_999 as f64);
        }
        if let Some(latency_95) = self.latency_95 {
            map.insert("latency_95".to_string(), latency_95 as f64);
        }
        if let Some(mean_cpus) = self.mean_cpus {
            map.insert("mean_cpus".to_string(), mean_cpus);
        }
        if let Some(mean_io_time) = self.mean_io_time {
            map.insert("mean_io_time".to_string(), mean_io_time);
        }
        if let Some(mean_ios) = self.mean_ios {
            map.insert("mean_ios".to_string(), mean_ios);
        }
        if let Some(mean_comps) = self.mean_comps {
            map.insert("mean_comps".to_string(), mean_comps);
        }
        if let Some(mean_hops) = self.mean_hops {
            map.insert("mean_hops".to_string(), mean_hops);
        }
        map
    }

    pub fn stats_to_dict(statistics: &[QueryStatistics]) -> HashMap<String, f64> {
        let stats = SearchStats::from_stats_slice(statistics);
        stats.to_dict()
    }

    pub fn stats_to_dict_inmem(statistics: &[u32]) -> HashMap<String, f64> {
        let stats = SearchStats::from_stats_slice_inmem(statistics);
        stats.to_dict()
    }
}
