# DiskANN Autotuner

The autotuner is a tool that builds on top of the DiskANN benchmark framework to automatically sweep over parameter combinations and identify the best configuration based on specified optimization criteria.

## Overview

The autotuner uses a **path-based configuration system** that doesn't hardcode JSON structure, making it robust to changes in the benchmark framework. You specify which parameters to sweep by providing JSON paths, making the tool adaptable to any benchmark configuration format.

Key features:
- **Framework-agnostic**: Works with any benchmark JSON structure
- **Flexible parameter sweeping**: Specify any JSON path to override
- **Multiple optimization criteria**: QPS, latency, or recall
- **Automatic result analysis**: Identifies best configuration based on your criteria

## Installation

Build the autotuner tool:

```bash
cargo build --release --package diskann-tools --bin autotuner
```

## Quick Start

### 1. Generate an Example Sweep Configuration

```bash
cargo run --release --package diskann-tools --bin autotuner -- example --output sweep_config.json
```

For specific benchmark types:
```bash
# For product-quantized indexes
cargo run --release --package diskann-tools --bin autotuner -- example --output sweep_config.json --benchmark-type pq

# For disk indexes
cargo run --release --package diskann-tools --bin autotuner -- example --output sweep_config.json --benchmark-type disk
```

### 2. Understanding the Sweep Configuration

The generated sweep configuration uses JSON paths to specify which parameters to sweep:

```json
{
  "parameters": [
    {
      "path": "jobs.0.content.source.max_degree",
      "values": [16, 32, 64]
    },
    {
      "path": "jobs.0.content.source.l_build",
      "values": [50, 75, 100]
    },
    {
      "path": "jobs.0.content.search_phase.runs.0.search_l",
      "values": [
        [10, 20, 30, 40, 50],
        [20, 40, 60, 80, 100],
        [30, 60, 90, 120, 150]
      ]
    }
  ]
}
```

**Path syntax:**
- Use dot notation: `jobs.0.content.source.max_degree`
- Array indices are numbers: `jobs.0` refers to first job
- The autotuner will generate all combinations of parameter values

### 3. Prepare a Base Configuration

Use an existing benchmark configuration file as your template (from `diskann-benchmark/example/`), or create your own.

### 4. Run the Parameter Sweep

```bash
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config diskann-benchmark/example/async.json \
  --sweep-config sweep_config.json \
  --output-dir ./autotuner_results \
  --criterion qps \
  --target-recall 0.95
```

## Configuration Guide

### Finding the Right JSON Paths

To find the correct paths for your benchmark configuration:

1. **Examine your base configuration** - Look at the JSON structure
2. **Identify the parameters** - Find where parameters like `max_degree`, `l_build`, `search_l` are located
3. **Write the path** - Use dot notation with array indices

**Example paths for common parameters:**

| Benchmark Type | Parameter | Path |
|----------------|-----------|------|
| async-index-build | max_degree | `jobs.0.content.source.max_degree` |
| async-index-build | l_build | `jobs.0.content.source.l_build` |
| async-index-build | search_l | `jobs.0.content.search_phase.runs.0.search_l` |
| async-index-build-pq | max_degree | `jobs.0.content.index_operation.source.max_degree` |
| async-index-build-pq | l_build | `jobs.0.content.index_operation.source.l_build` |
| async-index-build-pq | num_pq_chunks | `jobs.0.content.num_pq_chunks` |
| disk-index | max_degree | `jobs.0.content.source.max_degree` |
| disk-index | l_build | `jobs.0.content.source.l_build` |
| disk-index | search_list | `jobs.0.content.search_phase.search_list` |

### Creating Custom Sweep Configurations

You can sweep over any parameter, not just the standard ones:

```json
{
  "parameters": [
    {
      "path": "jobs.0.content.source.alpha",
      "values": [1.0, 1.2, 1.5]
    },
    {
      "path": "jobs.0.content.source.backedge_ratio",
      "values": [0.8, 1.0, 1.2]
    },
    {
      "path": "jobs.0.content.search_phase.num_threads",
      "values": [[1], [2], [4], [8]]
    }
  ]
}
```

## Command Line Options

### `autotuner sweep`

Run a parameter sweep to find the optimal configuration.

```bash
autotuner sweep [OPTIONS] --base-config <FILE> --sweep-config <FILE> --output-dir <DIR>
```

**Options:**
- `-b, --base-config <FILE>` - Base benchmark configuration JSON file (template)
- `-s, --sweep-config <FILE>` - Parameter sweep specification JSON file
- `-o, --output-dir <DIR>` - Output directory for results
- `-c, --criterion <CRITERION>` - Optimization criterion: `qps`, `latency`, or `recall` (default: `qps`)
- `-t, --target-recall <FLOAT>` - Target recall threshold for qps/latency optimization (default: 0.95)
- `--benchmark-cmd <CMD>` - Path to diskann-benchmark binary (default: `cargo`)
- `--benchmark-args <ARGS>` - Additional arguments for benchmark command

### `autotuner example`

Generate an example sweep configuration.

```bash
autotuner example --output <FILE> [--benchmark-type <TYPE>]
```

**Options:**
- `-o, --output <FILE>` - Output file for example configuration
- `--benchmark-type <TYPE>` - Generate example for specific type: `pq`, `disk`, or default

## Output

The autotuner generates the following files in the output directory:

- `config_XXXX.json` - Generated configuration for each parameter combination
- `results_XXXX.json` - Benchmark results for each configuration
- `sweep_summary.json` - Summary of all results with the best configuration highlighted

### Example Summary

The `sweep_summary.json` file contains all sweep results. Here's an example (truncated for brevity):

```json
{
  "criterion": "qps",
  "target_recall": 0.95,
  "total_configs": 9,
  "successful_configs": 9,
  "best_config": {
    "config_id": "0005",
    "parameters": {
      "jobs.0.content.source.max_degree": 32,
      "jobs.0.content.source.l_build": 75,
      "jobs.0.content.search_phase.runs.0.search_l": [10, 20, 30, 40, 50]
    },
    "metrics": {
      "qps": [12345.6, 11234.5, ...],
      "recall": [0.98, 0.96, ...]
    }
  },
  "all_results": [
    // Array of all sweep results, same format as best_config
  ]
}
```

**Note**: The `config_id` is a sequential number (e.g., "0005") representing the configuration index in the sweep. To see which parameter values correspond to which ID, check the `parameters` field.

## Examples

### Example 1: Optimize for Maximum QPS

```bash
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config diskann-benchmark/example/async.json \
  --sweep-config sweep_config.json \
  --output-dir ./results_qps \
  --criterion qps \
  --target-recall 0.95
```

### Example 2: Optimize for Minimum Latency

```bash
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config diskann-benchmark/example/async.json \
  --sweep-config sweep_config.json \
  --output-dir ./results_latency \
  --criterion latency \
  --target-recall 0.99
```

### Example 3: Optimize for Maximum Recall

```bash
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config diskann-benchmark/example/async.json \
  --sweep-config sweep_config.json \
  --output-dir ./results_recall \
  --criterion recall
```

### Example 4: Sweep Over Product-Quantized Index Parameters

```bash
# 1. Generate PQ-specific example
cargo run --release --package diskann-tools --bin autotuner -- example \
  --output pq_sweep.json \
  --benchmark-type pq

# 2. Run sweep
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config diskann-benchmark/example/product.json \
  --sweep-config pq_sweep.json \
  --output-dir ./results_pq \
  --criterion qps \
  --target-recall 0.95
```

## Adapting to Benchmark Framework Changes

The path-based design makes the autotuner robust to changes in the benchmark framework:

1. **If parameter locations change** - Simply update the paths in your sweep configuration
2. **If new parameters are added** - Add new path-value pairs to your sweep configuration
3. **If benchmark output format changes** - Only the result parsing logic needs updating (not the sweep logic)

The autotuner itself doesn't hardcode any assumptions about the benchmark JSON structure, so it remains compatible as long as:
- The benchmark accepts JSON configuration files
- The benchmark produces JSON output with QPS/recall metrics

## Tips

1. **Start with a coarse sweep**: Use a small number of parameter values to explore the space quickly
2. **Refine iteratively**: Once you identify promising regions, create a finer-grained sweep
3. **Use appropriate benchmarks**: Make sure your base configuration is appropriate for your workload
4. **Check paths carefully**: Invalid paths will cause the sweep to fail - verify with your base config
5. **Monitor resource usage**: Each configuration runs a full benchmark, which can be time-consuming

## Troubleshooting

**Problem**: "Path 'X' not found in JSON"  
**Solution**: Check that the path exists in your base configuration file. Use dot notation with array indices (e.g., `jobs.0.content`)

**Problem**: Benchmark failures during sweep  
**Solution**: Test your base configuration manually first to ensure it works before running the sweep

**Problem**: No results meet criteria  
**Solution**: Lower your target recall threshold or adjust your parameter ranges

## Design Philosophy

The autotuner is designed to be:
- **Maintainable**: No hardcoded assumptions about JSON structure
- **Flexible**: Works with any benchmark configuration format
- **Robust**: Survives changes to the benchmark framework
- **Extensible**: Easy to add new optimization criteria or parameter types

Users control what gets swept by specifying paths, not by relying on tool-specific parameter names.
