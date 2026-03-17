# PiPNN: Pick-in-Partitions Nearest Neighbors for DiskANN

A fast graph index builder for [DiskANN](https://github.com/microsoft/DiskANN) based on the [PiPNN algorithm](https://arxiv.org/abs/2602.21247) (Rubel et al., 2026).

PiPNN replaces Vamana's incremental beam-search insertion with a partition-then-build approach:

1. **Partition** the dataset into overlapping clusters via Randomized Ball Carving (RBC)
2. **Build** local k-NN graphs within each cluster using GEMM-based all-pairs distance
3. **Merge** edges from overlapping clusters using HashPrune (LSH-based online pruning)
4. **Prune** (optional) with RobustPrune for diversity

The output is a standard DiskANN graph file that can be loaded and searched by the existing DiskANN infrastructure.

## Results

### SIFT-1M (128d, L2, R=64)

| Builder | Build Time | Speedup | Recall@10 (L=100) |
|---------|-----------|---------|-------------------|
| DiskANN Vamana | 81.7s | 1.0x | 0.997 |
| **PiPNN** | **7.3s** | **11.2x** | **0.985** |

### Enron (384d, fp16, cosine_normalized, R=59, 1.09M vectors)

| Builder | Build Time | Speedup | Recall@1000 (L=2000) |
|---------|-----------|---------|---------------------|
| DiskANN Vamana | 78.1s | 1.0x | 0.950 |
| **PiPNN** | **25.3s** | **3.1x** | **0.947** |

Speedup scales with dataset size and is highest on lower-dimensional data where GEMM throughput dominates.

## Build

```bash
cargo build --release -p diskann-pipnn
```

For best performance on your CPU:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release -p diskann-pipnn
```

## Usage

### Build a PiPNN index and save as DiskANN graph

```bash
./target/release/pipnn-bench \
  --data <path_to_vectors.fbin> \
  --max-degree 64 \
  --c-max 2048 --c-min 1024 \
  --leaf-k 4 --fanout "8" \
  --replicas 1 --final-prune \
  --save-path <output_graph_prefix>
```

The output graph is written in DiskANN's canonical format at `<output_graph_prefix>`. Copy or symlink your data file to `<output_graph_prefix>.data` for the DiskANN benchmark loader.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-degree` | 64 | Maximum graph degree (R) |
| `--c-max` | 1024 | Maximum leaf partition size |
| `--c-min` | c_max/4 | Minimum cluster size before merging |
| `--leaf-k` | 3 | k-NN within each leaf |
| `--fanout` | "10,3" | Overlap factor per partition level (comma-separated) |
| `--replicas` | 1 | Independent partitioning passes |
| `--l-max` | 128 | HashPrune reservoir size per node |
| `--p-samp` | 0.05 | Leader sampling fraction |
| `--final-prune` | false | Apply RobustPrune after HashPrune |
| `--fp16` | false | Read input as fp16 (auto-converts to f32) |
| `--cosine` | false | Use cosine distance (for normalized vectors) |
| `--save-path` | none | Save graph in DiskANN format |

### Recommended configurations

**Low-dimensional (d <= 128):**
```bash
--c-max 2048 --c-min 1024 --leaf-k 4 --fanout "8" --p-samp 0.01 --final-prune
```

**High-dimensional (d >= 256):**
```bash
--c-max 2048 --c-min 1024 --leaf-k 5 --fanout "8" --p-samp 0.01 --final-prune
```

### Search with DiskANN benchmark

After building, search the graph using the standard DiskANN benchmark:

```bash
# Symlink your data file
ln -s <path_to_vectors.fbin> <output_graph_prefix>.data

# Create a search config (JSON)
cat > search.json << 'EOF'
{
  "search_directories": ["."],
  "jobs": [{
    "type": "async-index-build",
    "content": {
      "source": {
        "index-source": "Load",
        "data_type": "float32",
        "distance": "squared_l2",
        "load_path": "<output_graph_prefix>",
        "max_degree": 64
      },
      "search_phase": {
        "search-type": "topk",
        "queries": "<queries.fbin>",
        "groundtruth": "<groundtruth.bin>",
        "reps": 1,
        "num_threads": [1],
        "runs": [{"search_n": 10, "search_l": [100, 200], "recall_k": 10}]
      }
    }
  }]
}
EOF

cargo run --release -p diskann-benchmark -- run --input-file search.json --output-file results.json
```

## Architecture

```
diskann-pipnn/
  src/
    lib.rs          - Config and module structure
    partition.rs    - Randomized Ball Carving with fused GEMM + assignment
    leaf_build.rs   - GEMM-based all-pairs distance + bi-directed k-NN
    hash_prune.rs   - LSH-based online pruning with per-point reservoirs
    builder.rs      - Main PiPNN orchestrator
    bin/
      pipnn_bench.rs - CLI benchmark and index writer
```
