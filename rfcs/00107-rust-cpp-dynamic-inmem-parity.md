# Cpp Parity Assessment

## Motivation
We want DiskANN Rust to outperform DiskANN C++ in all scenarios. This RFC addresses conditions for certifying parity for the dynamic, in-memory DiskANN index. It is distinct from the previous RFC that focused on the disk index. Note that only the sync Rust index is considered here, because there is no async C++ index to compare to. The issue of sync vs. async parity is not addressed here.

Furthermore, this RFC does not test for parity for fully concurrent settings; rather, due to the limitations of the framework, each operation type---deletion, insertion, and search---is performed sequentially, with internal concurrency rather than different types of concurrent operations. 

# Required Process

The experimental setup for a dynamic index is a so-called runbook. A runbook is a series of insertions, deletions, and queries from a particular dataset that the dynamic index must perform. An example of a very short runbook for the dataset `random-xs` is as follows:

```
random-xs:
  max_pts: 10000
  1: 
    operation: "insert"
    start: 0
    end: 10000
  2:
    operation: "search"
  3:
    operation: "delete"
    start: 0
    end: 5000
  4:
    operation: "search"
  5:
    operation: "insert"
    start: 0
    end: 5000
  6:
    operation: "search"
```

The indices are with respect to the base dataset, and the `search` operation refers to a particular query set. Currently the same query set is always used for every search step, hence why a file is not specified in the runbook.

There are two broad approaches to comparing performance on a runbook. The first, used by the NeurIPS 2023 Big ANN Benchmarks competition, is to set a time limit and allow each competitor to select parameters for build and search that will complete within the time limit, and measure which algorithm achieves the highest recall (where recall here is averaged across all searches). The other approach, where applicable, is to set the same parameters for each algorithm, and measure which algorithm achieves both the highest recall and completes the runbook in the shortest amount of time. Since DiskANN Rust and DiskANN C++ share the same parameters, in this RFC we suggest the second approach is a better way to capture parity: standardize parameters, and measure based on time to completion and recall. 

# Required Configurations

## Machine
Standard_L8s_v3 (vCPU: 8, RAM: 64GB)

## Platform
Linux

## Runbooks

Our goal is broad coverage of scenarios. This includes both the range of datasets used in the runbooks, and how the runbooks are created. To make sure results are robust, we also want the index to reach at least a few million points in size. 

Runbook 1: final_runbook.yaml. This runbook is based on the MSTuring-30M dataset, which has 100 float dimensions and uses Euclidean distance. It is constructed by clustering the dataset into 32 clusters, then inserting points and deleting points from each cluster in five rounds. 

Runbook 2: msturing-10M_slidingwindow_runbook.yaml. This runbook is based on the MSTuring-10M dataset, which has 100 float dimensions and uses Euclidean distance. It inserts and deletes the points in the dataset in order, maintaining a sliding window of 5000000 points. 

Runbook 3: wikipedia-35M_expirationtime_runbook.yaml. This runbook is based on the Wikipedia-35M dataset, which has 768 float dimensions and uses inner product distance. It is constructed by giving each inserted point a randomly selected expiration time and deleting the point when it reaches its expiration time. 

Runbook 4: msmarco-10M_expirationtime_runbook.yaml (not yet created). This runbook is based on the MSMARCO-10M dataset, which has 768 float dimensions and uses inner product distance. It is constructed by giving each inserted point a randomly selected expiration time and deleting the point when it reaches its expiration time. 

The additional work required here is to create the msmarco-10M runbook (there is an existing 100M runbook, but it takes over 12 hours to run), and to compute and upload groundtruth for Runbook 2 and Runbook 4. 


## Parameters

For all runbooks, the consolidate routine will trigger when the max points specified in the runbook is met or exceeded by a batch of insertions. For each runbook, we will run two sets of parameters, a "low recall" choice and a "high recall" choice. Since the framework does not allow trying and recording multiple values of `L_search` during a single runbook, this choice will allow us to explore a few different recall scenarios without making the experiments overly complicated. Since the C++ Big-ANN-Benchmarks entry does not support quantization for dynamic indices, quantization is not included here.

Low Recall Parameters:
1. `R`: 32
2. `L_build`: 64
3. `alpha`: 1.2
4. `L_search`: 64

High Recall Parameters:
1. `R`: 64
2. `L_build`: 128
3. `alpha`: 1.2
4. `L_search`: 128

# Criteria
These criteria aim to capture parity in two ways: first, using the overall time to completion and average recall over all search steps. However, we would also like a somewhat stronger notion of parity; for example, if the Rust index started a runbook at 100% accuracy and then declined to 50% accuracy by the final timestep, while the C++ index stayed constant at 75% recall, this would not be a reasonable definition of parity. Thus, we also put parity constraints on the worst discrepancy allowed between any two individual search steps to avoid such situations. For this reason, we also specify a tolerance on the total time spent querying, inserting, and consolidating as well as the time for the overall runbook.

For the following metrics, the value of the Rust dynamic index must not be worse than tolerance (defined below) of the C++ version. For the dynamic case, the metrics make use of both the overall average recall and time to completion, and also some more granular metrics over individual steps and types of operation.

## Metrics:
| Metric Name       | Disparity tolerance   |
|------------|-----|
| Average Recall (over all searches)  | 2%  |
| Recall over Each Search Step | 10%  |
| Total Time to Completion of Runbook | 2% |
| Total Time Spent Querying | 5% |
| Total Time Spent Inserting | 5% |
| Total Time Spent Consolidating | 5% |
| Peak Memory Consumption | 5% |