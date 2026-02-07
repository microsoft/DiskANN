# Overview of the DiskANN Project (2018–present)

## Research Ideas

DiskANN started as a research project in 2018–2019 to address the large gap between vector search algorithms in the literature and the rapidly expanding scale and feature needs in industry.

Our research, with co-authors from MSR, Microsoft product groups, CMU, UMD, MIT, IITH, and UCI, addresses the following problems—many of which push the state of the art by an order of magnitude in one or more directions:

1. The first practical, high-performance SSD-based index that could index 10× more vectors per machine than previous in-memory systems [[1]](#ref1).
2. The first papers on updating graph-structured vector indices with stable recall, either via merges [[2]](#ref2) or via in-place edits [[4]](#ref4).
3. The first paper on predicate pushdown for vector-plus-predicate queries that provide high recall and two or more orders of magnitude higher query performance [[3]](#ref3).
4. Deterministic parallel updates to the index (experiments on 192 cores) [[5]](#ref5).
5. A single logical, distributed 50-billion-point index across 1,000 machines with 6× higher efficiency than sharded indices [[8]](#ref8).
6. Investigation of out-of-distribution (OOD) queries [[16]](#ref16).
7. Indices for diverse recommendations [[17]](#ref17).
8. Adaptations of large indices for GPUs [[21]](#ref21).
9. A theoretical analysis of beam search for graph-structured vector indices [[25]](#ref25).
10. Adaptive distances for large vector search with predicates [[26]](#ref26).

Some of the ideas are surveyed in a recent bulletin [[6]](#ref6).

## Adoption

Many of these ideas are implemented in an open-source project [[12]](#ref12), and are used widely within Microsoft and industry, and have inspired hardware adaptations. A few examples include:

1. The code we wrote supports at-scale vector indices at Microsoft in Bing, Ads, Microsoft 365, Windows, and Azure databases.
2. In the PostgreSQL ecosystem, they are implemented by TimescaleDB as pgvectorscale [[14]](#ref14).
3. In the Cassandra ecosystem, DataStax (now part of IBM) implemented them as JVector [[15]](#ref15).
4. Intel re-implemented these ideas and added new quantizers as part of their Scalable Vector Search (SVS) [[27]](#ref27).
5. Redis integrated Intel SVS as part of its vector APIs [[28]](#ref28).
6. Milvus, Pinecone, Weaviate, and other vector databases have implemented or adapted these ideas.
7. Storage-only vector search by Kioxia [[19]](#ref19).
8. Intel's adaptations for Optane PMem [[20]](#ref20).
9. NVIDIA's adaptations for the cuVS library [[18]](#ref18), [[22]](#ref22).

## Benchmarks

Along the way, we realized there were few public datasets or benchmarks, so we partnered with other companies and universities to:

1. Create new datasets for large-scale vector search and its variants [[13]](#ref13).
2. Publish open-source baseline algorithms [[12]](#ref12).
3. Run two competitions at NeurIPS 2021 and NeurIPS 2023 [[9]](#ref9), [[10]](#ref10). These have been used in many theses and research papers, including those in database and ML conferences.

## Current and Future

The code for this research [[12]](#ref12) was forked many times internally and reimplemented externally, which made it hard to manage and develop new algorithms. Further, since the 2023 version of DiskANN [[12]](#ref12) was tied to specific points in the storage hierarchy and managed its own index terms, it was hard to integrate into databases, preventing it from being hardened into a highly available and durable vector database.

With this in mind, since 2023 we have rewritten DiskANN in Rust with the following goals:

1. DiskANN delegates storage of indexing terms to a host database (or key-value store or file system), which it accesses and mutates via a Provider API.
2. DiskANN is a stateless orchestrator of vector requests between users, indexers, query engines, and the storage backend.
3. DiskANN provides a minimal API (updates with or without minibatches, paginated search) and integrates into the query planner for predicate evaluation.

This allows DiskANN to be plugged into different databases or systems and to inherit the availability and durability of the host database. The host database can choose to operate DiskANN at different memory tiers suited to target cost-performance points. Our new version has been integrated with five (and counting) backends. It can also be connected to memory buffers to compete with FAISS, hnswlib, or the older "monolithic" in-memory DiskANN.

When integrated with Azure Cosmos DB for NoSQL, Microsoft's highly available geo-distributed database, this integration brings vector indexing into operational databases and is competitive with specialized serverless vector databases [[7]](#ref7). See slides from our VLDB 2025 talk here [[23]](#ref23).

For a 25-minute overview of the project, see the slides from an overview talk at VLDB 2025 [[24]](#ref24).

## References

1. <a id="ref1"></a>[Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://harsha-simhadri.org/pubs/DiskANN19.pdf)
2. <a id="ref2"></a>[FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search](https://arxiv.org/abs/2105.09613)
3. <a id="ref3"></a>[FilteredDiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)
4. <a id="ref4"></a>[In-Place Updates of a Graph Index for Streaming Approximate Nearest Neighbor Search](https://arxiv.org/abs/2502.13826)
5. <a id="ref5"></a>[ParlayANN: Scalable and Deterministic Parallel Graph-Based Approximate Nearest Neighbor Search Algorithms](https://dl.acm.org/doi/abs/10.1145/3627535.3638475)
6. <a id="ref6"></a>[The DiskANN library: Graph-Based Indices for Fast, Fresh and Filtered Vector Search](http://sites.computer.org/debull/A24sept/p20.pdf)
7. <a id="ref7"></a>[Cost-Effective, Low Latency Vector Search with Azure Cosmos DB](https://arxiv.org/pdf/2505.05885)
8. <a id="ref8"></a>[DistributedANN: Efficient Scaling of a Single DiskANN Graph Across Thousands of Computers](https://openreview.net/forum?id=6AEsfCLRm3)
9. <a id="ref9"></a>[Results of the NeurIPS'21 Challenge on Billion-Scale Approximate Nearest Neighbor Search](https://proceedings.mlr.press/v176/simhadri22a/simhadri22a.pdf)
10. <a id="ref10"></a>[Results of the Big ANN: NeurIPS'23 competition](https://arxiv.org/abs/2409.17424)
11. <a id="ref11"></a>[https://big-ann-benchmarks.com](https://big-ann-benchmarks.com)
12. <a id="ref12"></a>[https://github.com/microsoft/DiskANN](https://github.com/microsoft/DiskANN)
13. <a id="ref13"></a>[Big ANN Benchmarks dataset list](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py)
14. <a id="ref14"></a>[Timescale DB's pgvectorscale](https://github.com/timescale/pgvectorscale)
15. <a id="ref15"></a>[IBM Datastax Jvector](https://github.com/datastax/jvector)
16. <a id="ref16"></a>[OOD-DiskANN: Efficient and Scalable Graph ANNS for Out-of-Distribution Queries](https://arxiv.org/abs/2211.12850)
17. <a id="ref17"></a>[Graph-Based Algorithms for Diverse Similarity Search](https://arxiv.org/abs/2502.13336v1)
18. <a id="ref18"></a>[https://www.nvidia.com/en-us/on-demand/session/gtc25-s72905/](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72905/)
19. <a id="ref19"></a>[AiSAQ: All-in-Storage ANNS with Product Quantization for DRAM-free Information Retrieval](https://arxiv.org/abs/2404.06004)
20. <a id="ref20"></a>[Intel: Winning the NeurIPS BillionScale Approximate Nearest Neighbor Search Challenge](https://www.intel.com/content/www/us/en/developer/articles/technical/winning-neurips-billion-scale-ann-search-challenge.html)
21. <a id="ref21"></a>[BANG: Billion-Scale Approximate Nearest Neighbor Search using a Single GPU](https://github.com/NVIDIA/cuvis)
22. <a id="ref22"></a>[NVIDIA CuVS and DiskANN](https://developer.nvidia.com/blog/optimizing-vector-search-for-indexing-and-real-time-retrieval-with-nvidia-cuvs/)
23. <a id="ref23"></a>[Cosmos DB Vector Search VLDB 2025 slides](https://harsha-simhadri.org/talks/cosmosdb_vector_search_VLDB25.pptx)
24. <a id="ref24"></a>[DiskANN overview slides](https://harsha-simhadri.org/talks/diskann_overview_talk_sep2025.pptx)
25. <a id="ref25"></a>[Sort Before You Prune: Improved Worst-Case Guarantees of the DiskANN Family of Graphs](https://openreview.net/pdf?id=JnXbUKtLzz)
26. <a id="ref26"></a>[Learning Filter-Aware Distance Metrics for Nearest Neighbor Search with Multiple Filters](https://openreview.net/forum?id=dILIRHcYvC)
27. <a id="ref27"></a>[Intel Scalable Vector Search](https://github.com/intel/ScalableVectorSearch)
28. <a id="ref28"></a>[Redis SVS Vamana Index](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/#svs-vamana-index)
