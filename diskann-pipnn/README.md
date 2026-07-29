# PiPNN graph construction

This crate implements the graph-construction stages from [PiPNN: Pick in Partitions for Fast and Accurate ANN Graph Construction](https://arxiv.org/html/2602.21247v1).

## Boundary

PiPNN core consumes a dense `MatrixView<T>`, graph policy, and a caller-owned Rayon pool, then returns adjacency lists for the dataset's real point IDs. It does not own start or frozen points, vector or neighbor providers, PQ, disk headers, serialization, or search. Those concerns remain in the outer in-memory and disk pipelines.

The dense view is intentional: partition assignment and leaf all-pairs kernels operate over the whole source matrix. Materializing provider state inside the algorithm would couple numerical graph construction to storage lifecycle and would require a second dataset copy. Integrations should finish PiPNN scratch before allocating or populating their searchable provider.

## Policy ownership

- `PiPNNConfig` owns partition and leaf-selection parameters: leaf bounds, sampling fraction, fanout levels, leaf `k`, and replicas.
- DiskANN graph configuration owns metric, output degree, build-L, alpha, and prune policy.
- Candidate-merging policies are separate validated options; they must not make graph policy fields redundant or silently cap the requested degree.

## Execution

A build runs partitioning, leaf construction, candidate merging, then graph finalization. All parallel work executes in the supplied pool. Per-job scratch is initialized through Rayon and is released through normal ownership when its stage completes; the core has no global thread-local buffers or cleanup broadcasts.
