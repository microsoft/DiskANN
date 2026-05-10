# DiskANN3: A Composable Vector Indexing Library

DiskANN3 is a composable library for bringing scalable, accurate and cost-effective vector indexing to multiple databases. It draws on research from the DiskANN project. See the [research overview](https://github.com/microsoft/DiskANN/wiki/DiskANN-Project-and-Research-Overview-(2018%E2%80%90present)) page for more details and references. 

To use DiskANN3 in your system, you would implement the `DataProvider` trait for your store to describe how index terms such as vectors, adjacency lists should be store and retrieved. DiskANN3 provides vector update and query API to users and internally uses the implementation of `DataProvider` trait to serve these requests.

This repo offers the following Provider implementations as illustrative examples: 
- In-memory providers, for maximum performance. These are volatile and not intended for use in databases. DiskANN3 + in-memory providers outperforms HNSWlib on throughput.
- Disk provider, for larger than memory support. This is intended to match the performormance of the first version of DiskANN reported in [NeurIPS'19 Paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf).
- [Garnet](https://github.com/Microsoft/Garnet)-based provider for high-throughput scale up vector search, and as an example of mapping to a k-v store.
- Bf-tree provider as an illustration of how to connect to a B-tree in your database. 

The provider for [Cosmos DB NoSQL Vector Search](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-search) is not included here but documented in the [VLDB'25 paper](https://www.vldb.org/pvldb/vol18/p5166-upreti.pdf). 

The library supports the following algorithmic features
- Real-time updates (using [IP-DiskANN](https://arxiv.org/abs/2502.13826)) that support stable recall under long update streams -- no merges, rebuilds, patches needed.
- A diverse set of distance functions and quantizers (PQ, MinMax, Scalar, Spherical) implemented for x86 and aarch64.
- Choice of memory tiers to allow operation at different price-performance points. 
- Hooks to allow attribute filters (predicate) processsing along with vector search.

## Getting Started

- Start with [diskann-benchmarks](/diskann-benchmark/README.md) to benchmark this library and its concrete implementations. This also allows you to build, store and load indices.
- To add a new backend, implement the [Provider API](/diskann/src/provider.rs) contract for your store/DB.


This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See [guidelines](CONTRIBUTING.md) for contributing to this project.

## Legacy C++ Code

[![DiskANN Main](https://github.com/microsoft/DiskANN/actions/workflows/push-test.yml/badge.svg?branch=main)](https://github.com/microsoft/DiskANN/actions/workflows/push-test.yml)
[![PyPI version](https://img.shields.io/pypi/v/diskannpy.svg)](https://pypi.org/project/diskannpy/)
[![Downloads shield](https://pepy.tech/badge/diskannpy)](https://pepy.tech/project/diskannpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Older C++ code is retained on the `cpp_main` branch, and implements the following papers, but is not actively developed or maintained.

[![DiskANN Paper](https://img.shields.io/badge/Paper-NeurIPS%3A_DiskANN-blue)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Arxiv%3A_Fresh--DiskANN-blue)](https://arxiv.org/abs/2105.09613)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Filtered--DiskANN-blue)](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)

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
