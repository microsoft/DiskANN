# DiskANN Autotuner

The autotuner is a tool that builds on top of the DiskANN benchmark framework to automatically sweep over parameter combinations and identify the best configuration based on specified optimization criteria.

## Overview

The autotuner helps optimize DiskANN indexes by automatically:
- Sweeping over key parameters: R (max_degree), l_build, l_search (search_l), and quantization bytes (num_pq_chunks)
- Running benchmarks for each parameter combination
- Analyzing results to find the best configuration based on QPS, latency, or recall

## Installation

Build the autotuner tool:

```bash
cargo build --release --package diskann-tools --bin autotuner
```

## Usage

### 1. Generate an Example Sweep Configuration

First, generate an example sweep configuration file:

```bash
cargo run --release --package diskann-tools --bin autotuner -- example --output sweep_config.json
```

This creates a JSON file with example parameter ranges:

```json
{
  "max_degree": [16, 32, 64],
  "l_build": [50, 75, 100],
  "search_l": [10, 20, 30, 40, 50, 75, 100],
  "num_pq_chunks": [8, 16, 32]
}
```

### 2. Prepare a Base Configuration

Create a base benchmark configuration file (or use an existing one from `diskann-benchmark/example/`). The autotuner will use this as a template and modify the parameters specified in the sweep configuration.

Example base configuration (`base_config.json`):

```json
{
  "search_directories": ["test_data/disk_index_search"],
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
        "backedge_ratio": 1.0,
        "num_threads": 1,
        "start_point_strategy": "medoid"
      },
      "search_phase": {
        "search-type": "topk",
        "queries": "queries.fbin",
        "groundtruth": "groundtruth.bin",
        "reps": 5,
        "num_threads": [1],
        "runs": [{
          "search_n": 10,
          "search_l": [20, 30, 40],
          "recall_k": 10
        }]
      }
    }
  }]
}
```

### 3. Run the Parameter Sweep

Run the autotuner to sweep over parameters and find the best configuration:

```bash
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config base_config.json \
  --sweep-config sweep_config.json \
  --output-dir ./autotuner_results \
  --criterion qps \
  --target-recall 0.95
```

#### Options:

- `--base-config`: Path to the base benchmark configuration (template)
- `--sweep-config`: Path to the sweep configuration (parameter ranges)
- `--output-dir`: Directory where results will be saved
- `--criterion`: Optimization criterion:
  - `qps`: Maximize queries per second (default)
  - `latency`: Minimize latency
  - `recall`: Maximize recall
- `--target-recall`: Target recall threshold for qps/latency optimization (default: 0.95)
- `--benchmark-cmd`: Path to diskann-benchmark binary (default: "cargo")
- `--benchmark-args`: Additional arguments for the benchmark command

## Output

The autotuner generates the following files in the output directory:

- `config_<id>.json`: Generated configuration for each parameter combination
- `results_<id>.json`: Benchmark results for each configuration
- `sweep_summary.json`: Summary of all results with the best configuration highlighted

### Example Summary Output:

```json
{
  "criterion": "qps",
  "target_recall": 0.95,
  "total_configs": 9,
  "successful_configs": 9,
  "best_config": {
    "config_id": "0005_R32_L75",
    "parameters": {
      "max_degree": 32,
      "l_build": 75,
      "search_l": [10, 20, 30, 40, 50, 75, 100]
    },
    "metrics": {
      "qps": [12345.6, 11234.5, ...],
      "recall": [0.98, 0.96, ...]
    }
  },
  "all_results": [...]
}
```

## Examples

### Optimize for Maximum QPS at 95% Recall

```bash
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config diskann-benchmark/example/async.json \
  --sweep-config sweep_config.json \
  --output-dir ./results_qps \
  --criterion qps \
  --target-recall 0.95
```

### Optimize for Minimum Latency at 99% Recall

```bash
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config diskann-benchmark/example/async.json \
  --sweep-config sweep_config.json \
  --output-dir ./results_latency \
  --criterion latency \
  --target-recall 0.99
```

### Optimize for Maximum Recall

```bash
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config diskann-benchmark/example/async.json \
  --sweep-config sweep_config.json \
  --output-dir ./results_recall \
  --criterion recall
```

## Parameter Descriptions

- **max_degree (R)**: The maximum degree of the graph. Higher values increase index size but can improve recall.
- **l_build**: The search queue length during index construction. Higher values improve index quality but increase build time.
- **search_l**: The search queue length during queries. Higher values improve recall but reduce throughput.
- **num_pq_chunks**: Number of product quantization chunks (for quantized indexes). Affects compression ratio and search accuracy.

## Tips

1. **Start with a coarse sweep**: Begin with a small set of parameter values to get a rough idea of the performance landscape.
2. **Refine the sweep**: Once you identify promising regions, create a more fine-grained sweep around those values.
3. **Consider the trade-offs**: Higher QPS often comes at the cost of lower recall, and vice versa.
4. **Test data size**: Use representative data sizes for your production workload.

## Notes

- The autotuner generates one configuration per combination of build parameters (max_degree, l_build, num_pq_chunks).
- All search_l values are tested for each build configuration, allowing the tool to find the best search_l given the build parameters.
- For quantized indexes (PQ, SQ, etc.), make sure to use the appropriate base configuration template.
