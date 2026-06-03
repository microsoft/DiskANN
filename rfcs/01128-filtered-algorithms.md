# Filtered Search Algorithms in DiskANN

| | |
|---|---|
| **Authors** | Magdalen Manohar |
| **Created** | 2026-06-02 |


## Summary and Motivation

There are currently two filtered search algorithms in DiskANN: beta-filtered search and multi-hop search. Each has performance drawbacks: beta-filtered search generally struggles to achieve high recall on our existing test datasets, and while multi-hop search generally achieves higher recall and fewer distance comparisons than beta-filtered search, it has low recall on certain datasets and can sometimes explore extremely large portions of the graph before converting.

At the same time, there are three other proposed filtered search algorithms that currently exist as branches or pull requests. We need to understand the performance of each candidate and align on a smaller set of well-performing algorithms to stand behind as our filtered algorithms for DiskANN.

This RFC presents an empirical evaluation of the existing algorithms and makes recommendations to keep two algorithms and close/deprecate the other filtered search algorithms.

### Overview of Existing Filtered Algorithms

#### Inline Filtered Search

Inline filtered search is a simple baseline which I introduced to sanity-check the other filtered search algorithms. It conducts a standard graph search with the only additional step of maintaining a separate queue of every predicate-satisfying element seen so far, and returning the closest $L_{search}$ predicate-satisfying elements at the end of the search. 

The branch implementing inline filtered search is [here](https://github.com/microsoft/DiskANN/blob/users/magdalen/inline-filter/diskann/src/graph/search/inline_filter_search.rs). 

#### Beta Search

Beta search is conceptually very simple. It sets a value $\beta \in (0,1]$, and for a point $p$ encountered during a graph search that satisfies the query filter, the raw distance between the query and $p$ is multiplied by $\beta$. Thus the search is biased towards points which satisfy the filter.

The code for beta search is found [here](https://github.com/microsoft/DiskANN/blob/main/diskann-providers/src/model/graph/provider/layers/betafilter.rs).

#### Multi-hop Search

Multi-hop search augments the regular beam search with a step to gather additional candidates satisfying the filter at each visit, and it only inserts nodes satisfying the filter into the queue. During a visit, the nodes satisfying the predicate are added to the queue. The nodes that do not satisfy the predicate are expanded again, and if their neighbors satisfy the predicate, those neighbors have their distance to the query computed and are added to the exploration queue. Multi-hop differs from the other search algorithms in that it computes more label checks than distance comparisons.

The code for multi-hop search is found [here](https://github.com/microsoft/DiskANN/blob/main/diskann/src/graph/search/multihop_search.rs).

#### Two-Queue Search (PR #929)

Two-queue search maintains a queue of neighbors satisfying the filter predicate (size k*p), where p is a multiplicative factor set by the user, and a separate, unbounded size queue of the best neighbors found so far, regardless of predicate. The search proceeds as normal with the larger queue, adding any results satisfying the predicate to the filtered queue. The search terminates for one of four reasons: (1) when the closest unexplored node in the regular queue is further away from the query than the furthest node in the filter-satisfying queue, (2) when no candidates remain to visit, (3) the number of hops exceeds a user-set maximum, or (4) the QueryVisitDecision returns a termination. 

The code for two-queue search is found in [this PR](https://github.com/microsoft/DiskANN/pull/929).

#### Adaptive L Search (PR #977)

Adaptive L search runs a filtered search in the following way: for each query, it runs a standard search until the search has performed 1000 distance computations. Then, it computes what fraction of the points seen so far satisfy the filter predicate, and scales the L_search parameter up accordingly. See [these lines](https://github.com/microsoft/DiskANN/pull/977/changes#diff-0ed5dd0ab0fa4906e3aa6e0c77d6b381f2a364b4d64df85d81224f609104388eR274-R285) for the exact scaling parameters. It only performs the adaptive scaling at one point during the search, so L_search is capped at 16 times the original value.

The code for adaptive L search was originally contributed in [this PR](https://github.com/microsoft/DiskANN/pull/977). [This branch](https://github.com/microsoft/DiskANN/tree/users/magdalen/two-queue-adaptive-l) integrates it into benchmark and keeps up-to-date with the main branch.

### Goals

The goal is to align on at most two filtered search algorithms to remain in the main branch of the DiskANN repository, based on performance evaluation of current candidates.

## Benchmark Results

To avoid adding large files to the main repo, the presentation and discussion of benchmark results is contained in a [DiskANN Wiki page](https://github.com/microsoft/DiskANN/wiki/Evaluation-of-Filtered-Search-Algorithms).

## Proposal

Based on the benchmarking results and their analysis, I propose the following actions:
1. Move inline filtering to the main repo as a new filtered search algorithm, with the adaptive-L subroutine an option that can be enabled.
2. Deprecate beta-filtered search.
3. Retain multi-hop filtered search.
4. Close the PR with two-queue search.



  

