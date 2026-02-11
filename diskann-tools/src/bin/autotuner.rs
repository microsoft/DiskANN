/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Autotuner for DiskANN benchmarks.
//!
//! This tool builds on top of the benchmark framework to sweep over a subset of parameters
//! and identify the best configuration based on specified optimization criteria.
//!
//! The autotuner uses a path-based configuration system that doesn't hardcode JSON structure,
//! making it robust to changes in the benchmark framework.

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

        /// Generate example for specific benchmark type
        #[arg(long)]
        benchmark_type: Option<String>,
    },
}

/// Path-based parameter specification for sweeping
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ParameterSweep {
    /// JSON path to the parameter (e.g., "jobs.0.content.source.max_degree")
    path: String,
    /// Values to sweep over
    values: Vec<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SweepConfig {
    /// List of parameters to sweep with their paths and values
    parameters: Vec<ParameterSweep>,
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
        Commands::Example { output, benchmark_type } => generate_example(&output, benchmark_type.as_deref()),
    }
}

fn generate_example(output: &Path, benchmark_type: Option<&str>) -> Result<()> {
    let example = match benchmark_type {
        Some("pq") | Some("product-quantization") => SweepConfig {
            parameters: vec![
                ParameterSweep {
                    path: "jobs.0.content.index_operation.source.max_degree".to_string(),
                    values: vec![16, 32, 64].into_iter().map(|v| serde_json::json!(v)).collect(),
                },
                ParameterSweep {
                    path: "jobs.0.content.index_operation.source.l_build".to_string(),
                    values: vec![50, 75, 100].into_iter().map(|v| serde_json::json!(v)).collect(),
                },
                ParameterSweep {
                    path: "jobs.0.content.index_operation.search_phase.runs.0.search_l".to_string(),
                    values: vec![
                        serde_json::json!([10, 20, 30, 40, 50]),
                        serde_json::json!([20, 40, 60, 80, 100]),
                        serde_json::json!([30, 60, 90, 120, 150]),
                    ],
                },
                ParameterSweep {
                    path: "jobs.0.content.num_pq_chunks".to_string(),
                    values: vec![8, 16, 32].into_iter().map(|v| serde_json::json!(v)).collect(),
                },
            ],
        },
        Some("disk") | Some("disk-index") => SweepConfig {
            parameters: vec![
                ParameterSweep {
                    path: "jobs.0.content.source.max_degree".to_string(),
                    values: vec![16, 32, 64].into_iter().map(|v| serde_json::json!(v)).collect(),
                },
                ParameterSweep {
                    path: "jobs.0.content.source.l_build".to_string(),
                    values: vec![50, 75, 100].into_iter().map(|v| serde_json::json!(v)).collect(),
                },
                ParameterSweep {
                    path: "jobs.0.content.search_phase.search_list".to_string(),
                    values: vec![
                        serde_json::json!([10, 20, 40]),
                        serde_json::json!([20, 40, 80]),
                        serde_json::json!([30, 60, 120]),
                    ],
                },
            ],
        },
        _ => SweepConfig {
            parameters: vec![
                ParameterSweep {
                    path: "jobs.0.content.source.max_degree".to_string(),
                    values: vec![16, 32, 64].into_iter().map(|v| serde_json::json!(v)).collect(),
                },
                ParameterSweep {
                    path: "jobs.0.content.source.l_build".to_string(),
                    values: vec![50, 75, 100].into_iter().map(|v| serde_json::json!(v)).collect(),
                },
                ParameterSweep {
                    path: "jobs.0.content.search_phase.runs.0.search_l".to_string(),
                    values: vec![
                        serde_json::json!([10, 20, 30, 40, 50]),
                        serde_json::json!([20, 40, 60, 80, 100]),
                        serde_json::json!([30, 60, 90, 120, 150]),
                    ],
                },
            ],
        },
    };

    let file = std::fs::File::create(output)
        .with_context(|| format!("Failed to create output file: {}", output.display()))?;
    serde_json::to_writer_pretty(file, &example)?;
    
    println!("Example sweep configuration written to: {}", output.display());
    println!("\nThis configuration will sweep over {} parameters:", example.parameters.len());
    for param in &example.parameters {
        println!("  - {}: {} values", param.path, param.values.len());
    }
    println!("\nTo use this configuration:");
    println!("  1. Adjust the paths if your benchmark JSON structure is different");
    println!("  2. Modify the values to sweep over");
    println!("  3. Run: autotuner sweep --base-config <base.json> --sweep-config <this file> --output-dir <output>");

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
        match run_benchmark(benchmark_cmd, benchmark_args, &config_file, &output_file) {
            Ok(_) => {
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
            Err(e) => {
                println!("  ✗ Benchmark failed: {}", e);
            }
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

    // Generate all combinations of parameter values
    let combinations = generate_parameter_combinations(&sweep_spec.parameters);
    
    for (idx, param_values) in combinations.iter().enumerate() {
        let mut config = base_config.clone();
        let mut params = HashMap::new();

        // Apply each parameter override
        for (param_spec, value) in sweep_spec.parameters.iter().zip(param_values.iter()) {
            set_json_path(&mut config, &param_spec.path, value.clone())?;
            params.insert(param_spec.path.clone(), value.clone());
        }

        // Generate config ID from parameter values
        let config_id = format!("{:04}", idx);
        configs.push((config_id, config, params));
    }

    Ok(configs)
}

/// Generate all combinations of parameter values using Cartesian product
fn generate_parameter_combinations(parameters: &[ParameterSweep]) -> Vec<Vec<Value>> {
    if parameters.is_empty() {
        return vec![vec![]];
    }

    let mut result = vec![vec![]];
    
    for param in parameters {
        let mut new_result = Vec::new();
        for combination in &result {
            for value in &param.values {
                let mut new_combination = combination.clone();
                new_combination.push(value.clone());
                new_result.push(new_combination);
            }
        }
        result = new_result;
    }
    
    result
}

/// Set a value in JSON using a dot-separated path
fn set_json_path(json: &mut Value, path: &str, value: Value) -> Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    
    if parts.is_empty() {
        anyhow::bail!("Empty path");
    }

    let mut current = json;
    
    // Navigate to the parent of the target field
    for (i, &part) in parts.iter().enumerate() {
        let is_last = i == parts.len() - 1;
        
        // Check if this is an array index
        if let Ok(index) = part.parse::<usize>() {
            if let Some(array) = current.as_array_mut() {
                if index >= array.len() {
                    anyhow::bail!("Array index {} out of bounds at path {}", index, path);
                }
                if is_last {
                    array[index] = value;
                    return Ok(());
                } else {
                    current = &mut array[index];
                }
            } else {
                anyhow::bail!("Expected array at path element '{}' in path {}", part, path);
            }
        } else {
            // Object key
            if let Some(object) = current.as_object_mut() {
                if is_last {
                    object.insert(part.to_string(), value);
                    return Ok(());
                } else {
                    current = object.get_mut(part)
                        .ok_or_else(|| anyhow::anyhow!("Path '{}' not found in JSON at '{}'", path, part))?;
                }
            } else {
                anyhow::bail!("Expected object at path element '{}' in path {}", part, path);
            }
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
    fn test_set_json_path_simple() {
        let mut json = serde_json::json!({
            "a": {
                "b": 42
            }
        });

        set_json_path(&mut json, "a.b", serde_json::json!(100)).unwrap();
        assert_eq!(json["a"]["b"], 100);
    }

    #[test]
    fn test_set_json_path_array() {
        let mut json = serde_json::json!({
            "jobs": [
                {"content": {"source": {"max_degree": 32}}}
            ]
        });

        set_json_path(&mut json, "jobs.0.content.source.max_degree", serde_json::json!(64)).unwrap();
        assert_eq!(json["jobs"][0]["content"]["source"]["max_degree"], 64);
    }

    #[test]
    fn test_generate_parameter_combinations() {
        let parameters = vec![
            ParameterSweep {
                path: "a".to_string(),
                values: vec![serde_json::json!(1), serde_json::json!(2)],
            },
            ParameterSweep {
                path: "b".to_string(),
                values: vec![serde_json::json!(10), serde_json::json!(20)],
            },
        ];

        let combinations = generate_parameter_combinations(&parameters);
        assert_eq!(combinations.len(), 4); // 2 * 2 = 4
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
