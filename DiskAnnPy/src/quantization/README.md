# Quantization

DiskAnnPy provides two quantization algorithms for compressing high-dimensional vectors:

- **MinMaxQuantizer**: Scalar quantization with configurable bit widths (1, 2, 4, or 8 bits)
- **ProductQuantizer**: Product quantization that partitions vectors into chunks and learns codebooks

## MinMaxQuantizer

### Basic Usage

```python
from diskannpy import MinMaxQuantizer
import numpy as np

# Create quantizer: bit_width, grid_scale, dimension
quantizer = MinMaxQuantizer(bit_width=8, grid_scale=1.0, dim=128)

# Compress vectors
vectors = np.random.randn(1000, 128).astype(np.float32)
compressed = quantizer.compress_batch(vectors)  # (1000, bytes_per_vector) uint8

# Search: preprocess query, then compute distances
query = np.random.randn(1, 128).astype(np.float32)
preprocessed = quantizer.preprocess(query)
distances = quantizer.distances_batch(preprocessed, compressed, metric="l2")

# Find k nearest neighbors
nearest_indices = np.argsort(distances)[:10]
```

### Optional Transforms

Transforms can improve quantization quality by spreading information across dimensions:

```python
quantizer = MinMaxQuantizer(
    bit_width=4,
    grid_scale=1.0,
    dim=128,
    transform="double_hadamard",  # "null" or "double_hadamard"
    target_behavior="same",       # "same"
    rng_seed=42                   # for reproducibility
)
```

### Properties

| Property | Description |
|----------|-------------|
| `bit_width` | Number of bits per dimension (1, 2, 4, or 8) |
| `bytes_per_vector` | Size of compressed vector in bytes |
| `dim` | Input vector dimension |
| `output_dim` | Output dimension (may differ if transform is applied) |

## ProductQuantizer

Product quantizer that partitions vectors into chunks and learns codebooks via k-means clustering. Particularly effective for high-dimensional vectors.

### Basic Usage

```python
from diskannpy import ProductQuantizer
import numpy as np

# Training data (should be representative of your dataset)
training_data = np.random.randn(10000, 128).astype(np.float32)

# Train the quantizer
quantizer = ProductQuantizer(
    training=training_data,
    num_chunks=16,       # Number of sub-vector partitions
    num_centers=256,     # Codebook size per chunk (max 256)
    lloyds_iters=10,     # K-means iterations
    seed=42              # For reproducibility
)

# Compress vectors
vectors = np.random.randn(1000, 128).astype(np.float32)
compressed = quantizer.compress_batch(vectors)  # (1000, num_chunks) uint8

# Search: preprocess query with metric, then compute distances
query = np.random.randn(1, 128).astype(np.float32)
preprocessed = quantizer.preprocess(query, metric="l2")
distances = quantizer.distances_batch(preprocessed, compressed)

# Find k nearest neighbors
nearest_indices = np.argsort(distances)[:10]
```

### Properties

| Property | Description |
|----------|-------------|
| `num_chunks` | Number of sub-vector partitions |
| `num_centers` | Codebook size per chunk (number of centroids) |
| `bytes_per_vector` | Size of compressed vector (equals num_chunks) |
| `dim` | Input vector dimension |

## Distance Metrics

Both quantizers support the following distance metrics:

| Metric | String | 
|--------|--------|
| L2 Squared | `"l2"` |
| Inner Product | `"inner_product"` | 
| Cosine | `"cosine"` | 
| Cosine Normalized | `"cosine_normalized"` | 