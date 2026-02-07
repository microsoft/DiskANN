/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Autotuner for DiskANN benchmarks.
//!
//! This tool builds on top of the benchmark framework to sweep over a subset of parameters
//! (R/max_degree, l_build, l_search/search_l, quantization bytes where applicable)
//! and identify the best configuration based on specified optimization criteria.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "autotuner")]
#[command(about = "Autotuner for DiskANN benchmarks - sweeps parameters to find optimal configuration")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run parameter sweep to find optimal configuration
    Sweep {
        /// Base configuration file (JSON) to use as template
        #[arg(short, long)]
        base_config: PathBuf,

        /// Parameter sweep specification file (JSON)
        #[arg(short, long)]
        sweep_config: PathBuf,

        /// Output directory for sweep results
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Optimization criterion: "qps", "latency", or "recall"
        #[arg(short = 'c', long, default_value = "qps")]
        criterion: String,

        /// Target recall threshold (for qps/latency optimization)
        #[arg(short, long)]
        target_recall: Option<f64>,

        /// Path to diskann-benchmark binary
        #[arg(long, default_value = "cargo")]
        benchmark_cmd: String,

        /// Additional args for benchmark command
        #[arg(long)]
        benchmark_args: Option<String>,
    },

    /// Generate example sweep configuration
    Example {
        /// Output file for example sweep configuration
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SweepConfig {
    /// Max degree (R) values to sweep
    #[serde(skip_serializing_if = "Option::is_none")]
    max_degree: Option<Vec<u32>>,

    /// l_build values to sweep
    #[serde(skip_serializing_if = "Option::is_none")]
    l_build: Option<Vec<u32>>,

    /// search_l values to sweep
    #[serde(skip_serializing_if = "Option::is_none")]
    search_l: Option<Vec<u32>>,

    /// num_pq_chunks values to sweep (for quantized indexes)
    #[serde(skip_serializing_if = "Option::is_none")]
    num_pq_chunks: Option<Vec<u32>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SweepResult {
    config_id: String,
    parameters: HashMap<String, Value>,
    metrics: BenchmarkMetrics,
    output_file: PathBuf,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct BenchmarkMetrics {
    qps: Vec<f64>,
    recall: Vec<f64>,
    latency_p50: Option<Vec<f64>>,
    latency_p90: Option<Vec<f64>>,
    latency_p99: Option<Vec<f64>>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Sweep {
            base_config,
            sweep_config,
            output_dir,
            criterion,
            target_recall,
            benchmark_cmd,
            benchmark_args,
        } => run_sweep(
            &base_config,
            &sweep_config,
            &output_dir,
            &criterion,
            target_recall,
            &benchmark_cmd,
            benchmark_args.as_deref(),
        ),
        Commands::Example { output } => generate_example(&output),
    }
}

fn generate_example(output: &Path) -> Result<()> {
    let example = SweepConfig {
        max_degree: Some(vec![16, 32, 64]),
        l_build: Some(vec![50, 75, 100]),
        search_l: Some(vec![10, 20, 30, 40, 50, 75, 100]),
        num_pq_chunks: Some(vec![8, 16, 32]),
    };

    let file = std::fs::File::create(output)
        .with_context(|| format!("Failed to create output file: {}", output.display()))?;
    serde_json::to_writer_pretty(file, &example)?;
    
    println!("Example sweep configuration written to: {}", output.display());
    println!("\nThis configuration will sweep over:");
    println!("  - max_degree: {:?}", example.max_degree.unwrap());
    println!("  - l_build: {:?}", example.l_build.unwrap());
    println!("  - search_l: {:?}", example.search_l.unwrap());
    println!("  - num_pq_chunks: {:?}", example.num_pq_chunks.unwrap());

    Ok(())
}

fn run_sweep(
    base_config: &Path,
    sweep_config: &Path,
    output_dir: &Path,
    criterion: &str,
    target_recall: Option<f64>,
    benchmark_cmd: &str,
    benchmark_args: Option<&str>,
) -> Result<()> {
    println!("Starting parameter sweep...");
    println!("Base config: {}", base_config.display());
    println!("Sweep config: {}", sweep_config.display());
    println!("Output dir: {}", output_dir.display());
    println!("Optimization criterion: {}", criterion);
    if let Some(recall) = target_recall {
        println!("Target recall: {}", recall);
    }

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;

    // Load base configuration
    let base_config_data: Value = load_json(base_config)?;

    // Load sweep configuration
    let sweep_spec: SweepConfig = load_json(sweep_config)?;

    // Generate configurations
    let configs = generate_configurations(&base_config_data, &sweep_spec)?;
    println!("Generated {} configurations to evaluate", configs.len());

    // Run benchmarks for each configuration
    let mut results = Vec::new();
    for (idx, (config_id, config_data, params)) in configs.iter().enumerate() {
        println!(
            "\n[{}/{}] Running configuration: {}",
            idx + 1,
            configs.len(),
            config_id
        );

        // Save configuration to file
        let config_file = output_dir.join(format!("config_{}.json", config_id));
        save_json(&config_file, config_data)?;

        // Run benchmark
        let output_file = output_dir.join(format!("results_{}.json", config_id));
        run_benchmark(
            benchmark_cmd,
            benchmark_args,
            &config_file,
            &output_file,
        )?;

        // Parse results
        if let Ok(metrics) = parse_benchmark_results(&output_file) {
            results.push(SweepResult {
                config_id: config_id.clone(),
                parameters: params.clone(),
                metrics,
                output_file: output_file.clone(),
            });
            println!("  ✓ Completed");
        } else {
            println!("  ✗ Failed to parse results");
        }
    }

    // Analyze results and find best configuration
    if results.is_empty() {
        anyhow::bail!("No successful benchmark runs to analyze");
    }

    let best_config = find_best_configuration(&results, criterion, target_recall)?;
    
    // Save summary
    let summary_file = output_dir.join("sweep_summary.json");
    let summary = serde_json::json!({
        "criterion": criterion,
        "target_recall": target_recall,
        "total_configs": configs.len(),
        "successful_configs": results.len(),
        "best_config": best_config,
        "all_results": results,
    });
    save_json(&summary_file, &summary)?;

    println!("\n========================================");
    println!("Sweep completed!");
    println!("========================================");
    println!("Best configuration: {}", best_config.config_id);
    println!("Parameters:");
    for (key, value) in &best_config.parameters {
        println!("  {}: {}", key, value);
    }
    println!("Metrics:");
    println!("  QPS: {:?}", best_config.metrics.qps);
    println!("  Recall: {:?}", best_config.metrics.recall);
    println!("\nSummary saved to: {}", summary_file.display());

    Ok(())
}

fn generate_configurations(
    base_config: &Value,
    sweep_spec: &SweepConfig,
) -> Result<Vec<(String, Value, HashMap<String, Value>)>> {
    let mut configs = Vec::new();

    // Get job type to determine which parameters to sweep
    let job_type = base_config
        .get("jobs")
        .and_then(|j| j.as_array())
        .and_then(|arr| arr.first())
        .and_then(|job| job.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("unknown");

    let is_pq = job_type.contains("pq");

    // Get default values from base config
    let default_max_degree = sweep_spec.max_degree.as_ref().map(|v| v[0]).unwrap_or(32);
    let default_l_build = sweep_spec.l_build.as_ref().map(|v| v[0]).unwrap_or(50);
    let default_search_l = sweep_spec.search_l.clone().unwrap_or_else(|| vec![20, 30, 40]);
    
    // Generate all combinations of build parameters
    let max_degrees = sweep_spec.max_degree.as_ref().map_or(vec![default_max_degree], |v| v.clone());
    let l_builds = sweep_spec.l_build.as_ref().map_or(vec![default_l_build], |v| v.clone());
    let pq_chunks = if is_pq {
        sweep_spec.num_pq_chunks.as_ref().map_or(vec![], |v| v.clone())
    } else {
        vec![0] // dummy value for non-PQ
    };

    let mut config_id = 0;
    for &max_degree in &max_degrees {
        for &l_build in &l_builds {
            for &num_pq_chunks in &pq_chunks {
                if !is_pq && num_pq_chunks != 0 {
                    continue;
                }

                // Create modified config
                let mut config = base_config.clone();
                let mut params = HashMap::new();

                // Update max_degree and l_build
                if let Some(jobs) = config.get_mut("jobs").and_then(|j| j.as_array_mut()) {
                    for job in jobs.iter_mut() {
                        update_build_params(job, max_degree, l_build, is_pq, num_pq_chunks)?;
                        
                        // Update search_l values - try two different paths
                        let search_phase = job.get_mut("content")
                            .and_then(|c| c.get_mut("search_phase"));
                        
                        if search_phase.is_none() {
                            if let Some(search_phase) = job.get_mut("content")
                                .and_then(|c| c.get_mut("index_operation"))
                                .and_then(|io| io.get_mut("search_phase"))
                            {
                                if let Some(runs) = search_phase.get_mut("runs").and_then(|r| r.as_array_mut()) {
                                    for run in runs.iter_mut() {
                                        run["search_l"] = serde_json::to_value(&default_search_l)?;
                                    }
                                }
                            }
                        } else if let Some(search_phase) = search_phase {
                            if let Some(runs) = search_phase.get_mut("runs").and_then(|r| r.as_array_mut()) {
                                for run in runs.iter_mut() {
                                    run["search_l"] = serde_json::to_value(&default_search_l)?;
                                }
                            }
                        }
                    }
                }

                params.insert("max_degree".to_string(), serde_json::json!(max_degree));
                params.insert("l_build".to_string(), serde_json::json!(l_build));
                params.insert("search_l".to_string(), serde_json::json!(default_search_l));
                if is_pq {
                    params.insert("num_pq_chunks".to_string(), serde_json::json!(num_pq_chunks));
                }

                let id = format!(
                    "{:04}_R{}_L{}{}",
                    config_id,
                    max_degree,
                    l_build,
                    if is_pq { format!("_PQ{}", num_pq_chunks) } else { String::new() }
                );
                configs.push((id, config, params));
                config_id += 1;
            }
        }
    }

    Ok(configs)
}

fn update_build_params(
    job: &mut Value,
    max_degree: u32,
    l_build: u32,
    is_pq: bool,
    num_pq_chunks: u32,
) -> Result<()> {
    // Try different paths based on job type
    if let Some(source) = job.get_mut("content").and_then(|c| c.get_mut("source")) {
        // async-index-build format
        if source.get("index-source").is_some() {
            source["max_degree"] = serde_json::json!(max_degree);
            source["l_build"] = serde_json::json!(l_build);
        }
        // disk-index format
        if source.get("disk-index-source").is_some() {
            source["max_degree"] = serde_json::json!(max_degree);
            source["l_build"] = serde_json::json!(l_build);
            if is_pq {
                source["num_pq_chunks"] = serde_json::json!(num_pq_chunks);
            }
        }
    }
    
    // async-index-build-pq format
    if is_pq {
        if let Some(content) = job.get_mut("content") {
            if let Some(index_op) = content.get_mut("index_operation").and_then(|io| io.get_mut("source")) {
                index_op["max_degree"] = serde_json::json!(max_degree);
                index_op["l_build"] = serde_json::json!(l_build);
            }
            content["num_pq_chunks"] = serde_json::json!(num_pq_chunks);
        }
    }

    Ok(())
}

fn run_benchmark(
    benchmark_cmd: &str,
    benchmark_args: Option<&str>,
    config_file: &Path,
    output_file: &Path,
) -> Result<()> {
    let mut cmd = if benchmark_cmd == "cargo" {
        let mut c = std::process::Command::new("cargo");
        c.arg("run");
        c.arg("--release");
        c.arg("--package");
        c.arg("diskann-benchmark");
        c.arg("--");
        c
    } else {
        std::process::Command::new(benchmark_cmd)
    };

    if let Some(args) = benchmark_args {
        for arg in args.split_whitespace() {
            cmd.arg(arg);
        }
    }

    cmd.arg("run");
    cmd.arg("--input-file");
    cmd.arg(config_file);
    cmd.arg("--output-file");
    cmd.arg(output_file);

    let output = cmd.output()
        .with_context(|| format!("Failed to run benchmark command: {}", benchmark_cmd))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Benchmark failed: {}", stderr);
    }

    Ok(())
}

fn parse_benchmark_results(output_file: &Path) -> Result<BenchmarkMetrics> {
    let data: Vec<Value> = load_json(output_file)?;
    
    // Extract metrics from first job result
    let result = data.first()
        .ok_or_else(|| anyhow::anyhow!("No results in output file"))?;

    let search_results = result
        .get("results")
        .and_then(|r: &Value| r.get("search"))
        .and_then(|s: &Value| s.get("Topk"))
        .and_then(|t: &Value| t.as_array())
        .ok_or_else(|| anyhow::anyhow!("Invalid result format"))?;

    let mut qps_values = Vec::new();
    let mut recall_values = Vec::new();

    for search_result in search_results {
        // Extract QPS (can be array or single value)
        if let Some(qps) = search_result.get("qps") {
            if let Some(qps_arr) = qps.as_array() {
                for val in qps_arr {
                    if let Some(q) = val.as_f64() {
                        qps_values.push(q);
                    }
                }
            } else if let Some(q) = qps.as_f64() {
                qps_values.push(q);
            }
        }

        // Extract recall
        if let Some(recall) = search_result.get("recall").and_then(|r: &Value| r.get("average")) {
            if let Some(r) = recall.as_f64() {
                recall_values.push(r);
            }
        }
    }

    Ok(BenchmarkMetrics {
        qps: qps_values,
        recall: recall_values,
        latency_p50: None,
        latency_p90: None,
        latency_p99: None,
    })
}

fn find_best_configuration(
    results: &[SweepResult],
    criterion: &str,
    target_recall: Option<f64>,
) -> Result<SweepResult> {
    if results.is_empty() {
        anyhow::bail!("No results to analyze");
    }

    let target_recall = target_recall.unwrap_or(0.95);

    let best = match criterion {
        "qps" => {
            // Find configuration with max QPS at target recall
            results
                .iter()
                .filter_map(|r| {
                    // Find max QPS where recall >= target
                    let max_qps = r.metrics.qps.iter()
                        .zip(&r.metrics.recall)
                        .filter(|(_, &recall)| recall >= target_recall)
                        .map(|(&qps, _)| qps)
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                    max_qps.map(|qps| (r, qps))
                })
                .max_by(|(_, qps_a), (_, qps_b)| qps_a.partial_cmp(qps_b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(r, _)| r)
        }
        "recall" => {
            // Find configuration with max recall
            results
                .iter()
                .max_by(|a, b| {
                    let max_recall_a = a.metrics.recall.iter().cloned().fold(0.0, f64::max);
                    let max_recall_b = b.metrics.recall.iter().cloned().fold(0.0, f64::max);
                    max_recall_a.partial_cmp(&max_recall_b).unwrap_or(std::cmp::Ordering::Equal)
                })
        }
        "latency" => {
            // Find configuration with min latency at target recall
            // For now, use inverse of QPS as proxy for latency
            results
                .iter()
                .filter_map(|r| {
                    let min_latency = r.metrics.qps.iter()
                        .zip(&r.metrics.recall)
                        .filter(|(_, &recall)| recall >= target_recall)
                        .map(|(&qps, _)| 1.0 / qps)
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                    min_latency.map(|latency| (r, latency))
                })
                .min_by(|(_, lat_a), (_, lat_b)| lat_a.partial_cmp(lat_b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(r, _)| r)
        }
        _ => anyhow::bail!("Unknown criterion: {}", criterion),
    };

    best.cloned()
        .ok_or_else(|| anyhow::anyhow!("No configuration meets the specified criteria"))
}

fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader)
        .with_context(|| format!("Failed to parse JSON from: {}", path.display()))
}

fn save_json<T: serde::Serialize>(path: &Path, data: &T) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create file: {}", path.display()))?;
    serde_json::to_writer_pretty(file, data)
        .with_context(|| format!("Failed to write JSON to: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sweep_config_serialization() {
        let config = SweepConfig {
            max_degree: Some(vec![16, 32, 64]),
            l_build: Some(vec![50, 100]),
            search_l: Some(vec![10, 20, 30]),
            num_pq_chunks: Some(vec![8, 16]),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SweepConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.max_degree, deserialized.max_degree);
        assert_eq!(config.l_build, deserialized.l_build);
        assert_eq!(config.search_l, deserialized.search_l);
        assert_eq!(config.num_pq_chunks, deserialized.num_pq_chunks);
    }

    #[test]
    fn test_generate_configurations() {
        let base_config = serde_json::json!({
            "search_directories": ["test_data"],
            "jobs": [{
                "type": "async-index-build",
                "content": {
                    "source": {
                        "index-source": "Build",
                        "data_type": "float32",
                        "data": "data.fbin",
                        "distance": "squared_l2",
                        "max_degree": 32,
                        "l_build": 50,
                        "alpha": 1.2,
                        "num_threads": 1
                    },
                    "search_phase": {
                        "search-type": "topk",
                        "queries": "queries.fbin",
                        "groundtruth": "gt.bin",
                        "reps": 1,
                        "num_threads": [1],
                        "runs": [{
                            "search_n": 10,
                            "search_l": [20],
                            "recall_k": 10
                        }]
                    }
                }
            }]
        });

        let sweep_config = SweepConfig {
            max_degree: Some(vec![16, 32]),
            l_build: Some(vec![50, 100]),
            search_l: Some(vec![10, 20, 30]),
            num_pq_chunks: None,
        };

        let configs = generate_configurations(&base_config, &sweep_config).unwrap();

        // Should generate 2 max_degree * 2 l_build = 4 configurations
        assert_eq!(configs.len(), 4);

        // Check that parameters are correctly set
        for (config_id, config, params) in configs {
            assert!(config_id.contains("R"));
            assert!(config_id.contains("L"));
            assert!(params.contains_key("max_degree"));
            assert!(params.contains_key("l_build"));
            assert!(params.contains_key("search_l"));

            // Verify the config has updated values
            let job = &config["jobs"][0];
            let source = &job["content"]["source"];
            let max_degree = source["max_degree"].as_u64().unwrap() as u32;
            let l_build = source["l_build"].as_u64().unwrap() as u32;

            assert!(max_degree == 16 || max_degree == 32);
            assert!(l_build == 50 || l_build == 100);
        }
    }

    #[test]
    fn test_benchmark_metrics() {
        let metrics = BenchmarkMetrics {
            qps: vec![1000.0, 2000.0, 1500.0],
            recall: vec![0.95, 0.97, 0.99],
            latency_p50: None,
            latency_p90: None,
            latency_p99: None,
        };

        // Test serialization
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: BenchmarkMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(metrics.qps, deserialized.qps);
        assert_eq!(metrics.recall, deserialized.recall);
    }

    #[test]
    fn test_find_best_configuration_qps() {
        let results = vec![
            SweepResult {
                config_id: "config1".to_string(),
                parameters: HashMap::new(),
                metrics: BenchmarkMetrics {
                    qps: vec![1000.0, 1500.0],
                    recall: vec![0.94, 0.96],
                    latency_p50: None,
                    latency_p90: None,
                    latency_p99: None,
                },
                output_file: PathBuf::from("output1.json"),
            },
            SweepResult {
                config_id: "config2".to_string(),
                parameters: HashMap::new(),
                metrics: BenchmarkMetrics {
                    qps: vec![2000.0, 2500.0],
                    recall: vec![0.96, 0.98],
                    latency_p50: None,
                    latency_p90: None,
                    latency_p99: None,
                },
                output_file: PathBuf::from("output2.json"),
            },
        ];

        let best = find_best_configuration(&results, "qps", Some(0.95)).unwrap();
        
        // config2 has higher QPS at recall >= 0.95
        assert_eq!(best.config_id, "config2");
    }

    #[test]
    fn test_find_best_configuration_recall() {
        let results = vec![
            SweepResult {
                config_id: "config1".to_string(),
                parameters: HashMap::new(),
                metrics: BenchmarkMetrics {
                    qps: vec![1000.0],
                    recall: vec![0.95],
                    latency_p50: None,
                    latency_p90: None,
                    latency_p99: None,
                },
                output_file: PathBuf::from("output1.json"),
            },
            SweepResult {
                config_id: "config2".to_string(),
                parameters: HashMap::new(),
                metrics: BenchmarkMetrics {
                    qps: vec![800.0],
                    recall: vec![0.99],
                    latency_p50: None,
                    latency_p90: None,
                    latency_p99: None,
                },
                output_file: PathBuf::from("output2.json"),
            },
        ];

        let best = find_best_configuration(&results, "recall", None).unwrap();
        
        // config2 has higher recall
        assert_eq!(best.config_id, "config2");
    }
}
