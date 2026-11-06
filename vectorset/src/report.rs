use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::Quantizer;

#[derive(Serialize)]
pub struct Report {
    pub date: DateTime<Utc>,

    pub num_threads: usize,
    pub quantizer: Quantizer,

    pub num_tasks: usize,
    pub pipeline_size: usize,
    pub search_repetitions: usize,
    pub degree: usize,
    pub l_build: usize,
    pub l_search: usize,
    pub k: usize,
    pub n: usize,

    pub runbook: String,
    pub dataset: HashMap<String, Vec<StepReport>>,
}

#[derive(Clone, Serialize)]
pub enum StepReport {
    Insert(OpReport),
    Delete(OpReport),
    Replace(OpReport),
    Search(SearchReport),
}

#[derive(Clone, Serialize)]
pub struct OpReport {
    pub parallelism: usize,
    pub count: usize,
    pub wall_time_s: f64,
    pub busy_time_s: f64,
    pub latency_us_mean: f64,
    pub latency_us_p90: f64,
    pub latency_us_p99: f64,
}

impl OpReport {
    pub fn throughput(&self) -> f64 {
        self.count as f64 / self.wall_time_s
    }

    pub fn utilization(&self) -> f64 {
        self.busy_time_s / (self.wall_time_s * self.parallelism as f64)
    }
}

#[derive(Clone, Serialize)]
pub struct SearchReport {
    pub op_reports: Vec<OpReport>,
    pub k: usize,
    pub n: usize,
    pub recall: f64,
}
