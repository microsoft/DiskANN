# DiskANN

[![DiskANN Main](https://github.com/microsoft/DiskANN/actions/workflows/push-test.yml/badge.svg?branch=main)](https://github.com/microsoft/DiskANN/actions/workflows/push-test.yml)
[![PyPI version](https://img.shields.io/pypi/v/diskannpy.svg)](https://pypi.org/project/diskannpy/)
[![Downloads shield](https://pepy.tech/badge/diskannpy)](https://pepy.tech/project/diskannpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![DiskANN Paper](https://img.shields.io/badge/Paper-NeurIPS%3A_DiskANN-blue)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Arxiv%3A_Fresh--DiskANN-blue)](https://arxiv.org/abs/2105.09613)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Filtered--DiskANN-blue)](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)

> [!IMPORTANT]
> We are currently in the process of updating this repository with a new version of the code written in Rust.

DiskANN is a suite of scalable, accurate and cost-effective approximate nearest neighbor search algorithms for large-scale vector search that support real-time changes and simple filters.
This code is based on ideas from Microsoft's [DiskANN](https://aka.ms/AboutDiskANN).
The main branch now contains a rearchitected project written in Rust.

## Architectural direction

The Rust implementation is organized as a modular workspace so algorithm development, storage, and tooling can evolve independently:

- **Base & numerics**: `diskann-wide`, `diskann-vector`, `diskann-linalg`, `diskann-utils`, and `diskann-quantization`
- **Core algorithm**: `diskann`
- **Providers & storage**: `diskann-providers`, `diskann-disk`, and `diskann-label-filter`
- **Benchmarks & tools**: `diskann-benchmark*` and `diskann-tools`

## Providers

DiskANN exposes pluggable providers for graph storage, vectors, and deletes. Current provider implementations include:

- In-memory async providers (default `inmem` provider in `diskann-providers`)
- Bf-tree providers (feature `bf_tree` in `diskann-providers`)
- Caching providers (feature `bf_tree` in `diskann-providers`)
- Disk vertex providers for on-disk indices (`diskann-disk`)
- Label filter providers for metadata filtering (`diskann-label-filter`)

## Getting started

1. Install Rust using the toolchain in `rust-toolchain.toml`.
2. Build the workspace: `cargo build`.
3. Run tests: `cargo test`.
4. Explore the core API in `diskann/` and disk index support in `diskann-disk/`.
5. For benchmarking, see the links below.

## Benchmarks

- Benchmark runner and scenarios: [`diskann-benchmark/README.md`](diskann-benchmark/README.md)
- Label filter benchmarks: [`diskann-label-filter/README.md`](diskann-label-filter/README.md)

## Papers

The list of DiskANN papers is maintained in the [DiskANN Wiki](https://github.com/microsoft/DiskANN/wiki).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See [guidelines](CONTRIBUTING.md) for contributing to this project.

## Legacy C++ Code

Older C++ code is retained on the `cpp_main` branch, but is not actively developed or maintained.
The legacy C++ code was forked off from [code for NSG](https://github.com/ZJULearning/nsg) algorithm.

If you use the C++ version in your software please cite the following:

```
@misc{diskann-github,
   author = {Simhadri, Harsha Vardhan and Krishnaswamy, Ravishankar and Srinivasa, Gopal and Subramanya, Suhas Jayaram and Antonijevic, Andrija and Pryce, Dax and Kaczynski, David and Williams, Shane and Gollapudi, Siddarth and Sivashankar, Varun and Karia, Neel and Singh, Aditi and Jaiswal, Shikhar and Mahapatro, Neelam and Adams, Philip and Tower, Bryan and Patel, Yash}},
   title = {{DiskANN: Graph-structured Indices for Scalable, Fast, Fresh and Filtered Approximate Nearest Neighbor Search}},
   url = {https://github.com/Microsoft/DiskANN},
   version = {0.6.1},
   year = {2023}
}
```

> [!NOTE]
> Trademarks: This project may contain trademarks or logos for projects, products, or services.
> Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines.
> Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
> Any use of third-party trademarks or logos are subject to those third-party’s policies.
