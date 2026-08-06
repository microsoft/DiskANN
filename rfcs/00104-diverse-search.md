# RFC: Diverse Similarity Search Implementation in DiskANN Rust

## Overview

This RFC proposes the implementation of diverse similarity search in DiskANN Rust. Diverse search returns results that vary across one or more attribute values, preventing result sets dominated by a single attribute value.

## What is Diversity Search?

Diversity search ensures that search results are varied across specified attributes, providing better distribution and fairness in the result set.

### Use Case examples

**Example 1: Document Deduplication (enterprise search)**
When searching for a keyword in enterprise search, results may predominantly come from the same document. A diversity requirement ensures no duplicate document IDs appear in results. This is "single attribute diversity" with `diversity_k = 1` (at most one result per document). Similarly, web search should return results diverse across different websites.

**Example 2: Embedding diversity in multi-model search (enterprise search)**
Consider a multi-model dataset. The results of a search from such a multi-model dataset are expected to include results from diverse embeddings, such as from both text embeddings & image embeddings.

**Example 3: Seller Fairness (an ads platform)**
Searching for a product might return mostly results from popular sellers like Seller A, which is unfair to smaller sellers. Users benefit from seeing diverse options. This search uses "seller name" attribute diversity with `diversity_k = 3` (at most 3 products per seller). An ads platform has such diversity needs for additional scenarios, such as diversity across product attributes like color and price ranges.

**Example 4: Multi-Attribute Diversity Movie Search (external example)**
Let us consider a movie app built by an external customer using diskann. A movie search app where all results are from the same genre would be unsatisfying. Diverse results spanning genres, categories, and other attributes improve user experience. This is covered below in "Nash diversity" where multiple attributes contribute and `diversity_k` is automatically determined by result distribution.

### Diversity Types

- **Single Attribute Diversity**: Examples 1, 2, and 3, where a specific attribute and explicit `diversity_k` value are provided
- **Nash Diversity**: Example 4, involving multiple attributes with automatic distribution

## Priorities

This RFC focuses on **single attribute diversity**, which is high priority for production scenarios.

Another team is also asking for this feature. They are still using the C++ version of DiskANN. Having this feature in DiskANN Rust can be a motivating factor for them to move to the Rust version.

The design considers future Nash diversity implementation to ensure easier implementation in DiskAnn when it becomes a priority in future but it is a lower pri at this point.

## Nash Diversity and Multi-Attribute Diversity

In Nash diversity, attribute names may be optionally specified and no `diversity_k` value is provided. The algorithm derives from fairness goals in economics: a utility value is computed for each vector during search, and results are obtained using geometric mean on utility instead of arithmetic mean. This naturally produces diverse results.


The remainder of this document focuses solely on single-attribute diversity.

## Design Considerations

Diversity search must integrate seamlessly with existing search capabilities:

- **Search Strategies**: Must work with all `SearchStrategy` implementations. For example, quantized vector search with full-precision reranking, or search and reranking both in full precision
- **Paged Search**: Each page must return diverse results according to specified `diverse_params`
- **Search Variants**: Should support most, if not all, search variants. Some variants to consider:
- in-memory search
- Disk Search
- filtered search
- flat search
- range search, etc.

One way to view diversity search is saying that "This is a regular search, but just keep the results diverse". In that sense, it is just a setting in the Search Params of regular search from a client's perspective. I am attempting to keep it as that simple in my design & PRs.

Implementation will be phased across multiple PRs, but the design must accommodate diversity search in its search code path

## Design Options

Is diversity a core concept or not? It may not seem like core concept right now, but the requirements from clients are pointing in that direction. Having a clear answer to this question will help us design it as part of diskann_core in the search path, or outside of diskann as some kind of strategy wrapping. I am assuming that it is good to treat it as core concept.

### Option 1: Diversity as a Core Concept

Add `DiverseSearchParams` as an optional field in `SearchParams`. This structure contains:
- `diverse_attribute_id`: The attribute to diversify on
- `diverse_results_k`: Maximum results per attribute value

When specified, the search function creates a specialized scratch space and uses the `diverse_neighbor_queue` with post-processing. When absent, standard search is performed.

**Status**: This RFC uses Option 1 for illustration. The approach can be changed if needed.

### Option 2: DiversityStrategy Pattern

Implement diversity as a `DiversityStrategy` that wraps an inner strategy. This keeps the search path cleaner by encapsulating diversity details.

The `DiversityStrategy` constructor would accept an inner strategy and delegate to its methods. However, the diversity algorithm requires swapping `NeighborQueue` for `DiversePriorityQueue`, necessitating trait modifications to support queue selection.

**Assessment**: Initial analysis suggests this approach is more complex and less natural than Option 1.

## Single Attribute Diversity Algorithm

The algorithm is based on the publication: [Graph-based Algorithms for Diverse Similarity Search](https://www.microsoft.com/en-us/research/publication/graph-based-algorithms-for-diverse-similarity-search/)

### Algorithm Overview

The search code path remains largely unchanged, with two key modifications:

1. Use a `DiversePriorityQueue` instead of standard `NeighborQueue`
2. Apply post-processing to enforce `diversity_k` constraints

### Diverse Priority Queue

The `DiversePriorityQueue` maintains:
- **Global priority queue**: Contains all candidates
- **Local priority queues**: One per attribute value

In the seller diversity example, separate local queues track results for Seller A, Seller B, and smaller sellers. Insert and delete operations synchronize across both global and local queues, with evictions maintaining consistency.

**Capacity**:
- Global queue: `search_l` items
- Local queues: `diverse_l` items each, where `diverse_l = diversity_k * l_value / k_value`. This is to improve on recall. In the post processing step, the excess items are removed before returning to caller.

### Search Post-Processing

Since local queues may contain more items than `diversity_k`, post-processing removes excess items while preserving priority queue order.

### Build time changes
Build algorithm needs some changes to improve diverse search recall. The first phase focuses only on search with standard build. The future PRs will update this section with build code changes.

## Handling Multi-Value Attributes

The design so far assumes each vector has a single value per attribute. In reality, a single vector may map to multiple attribute values (e.g., a vector representing text present in multiple documents).

### Approach

- Treat a vector with N attribute values as N distinct entries in the queue
- If a vector maps to 3 documents, it appears 3 times in the main queue and at most once in each local queue
- Search operations (`mark_visited`, `is_visited`, etc.) delegate to `diverse_queue`, which handles this correctly
- No changes required to core search path

**Note**: Detailed design will be updated in a future revision of this RFC.

## Implementation Plan

This RFC will be implemented across multiple PRs:

### Current PR
- Extract trait from current `NeighborQueue`
- Implement `DiverseNeighborQueue`
- Implement Option 1: use appropriate queue type based on `diverse_search_params`

### Phase 1
- Integrate with Attribute providers once those PRs are merged.
- Integrate with benchmarks.
- Performance fixes: For example, use scratch pool for diverse_priority_queue. Improve QPS.

### Phase 2
- Handle vectors with multiple attribute values

### Phase 3
- Support paged search, filtered search, and other search variants

### Future Phases
- Implement build code changes to improve diversity recall
- Implement Nash diversity and other algos if they become a priority

## Credits
Thanks to Kiran and Ravi for coming up with the algorithm. Ravi made an experimental implementation in C++ with Wikipedia dataset and it was helpful in current implementation, testing and benchmarking.

## References

- [Graph-based Algorithms for Diverse Similarity Search](https://www.microsoft.com/en-us/research/publication/graph-based-algorithms-for-diverse-similarity-search/)

