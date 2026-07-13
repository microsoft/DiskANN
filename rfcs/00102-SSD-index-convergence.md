**Proposal** Let us make SSD Index an implementation of `diskann_async.rs`

**Abstract**
----------
SSD Index has been written as a  stand-alone struct that does not add to the
rest of the repo. As we revised the library, we have added generic index classes
(`diskann_async.rs` and `generic_index.rs`) that decouple storage from logic.
These allow users to store data in a medium of their choice including databases.
With this divergence, any feature or a new algorithm we want to add has to be
duplicated and tested for these indices as well as the SSD Index.
This introduces unrealistic development, testing and maintenance overheads.
So we propose that the SSD index, specifically the search path, be rewritten
as a special instance of `diskann_async.rs` with specific implementations of 
providers that reflect what SSD index does today. Once this is tested,
we will stop maintaining redundant code. This has been discussed before.
A previous proposal suggests implementing SSD index using `generic_index.rs`
which does not have asynchronous interfaces. We propose instead that we implement
it with `diskann_async.rs` algorithm for two reasons:
1. SSD Index involves waiting for IO which is suited to the async model and allows
 us to hide latency.
2. At the time of the earlier proposal, `diskann_async.rs` was considerably slower
than its synchronous counterpart. We have since closed this gap in most scenarios.
In fact, the new provider APIs already allow an implementation to return
completed results where possible instead of futures.  


**Exit Criteria**
-----------------
The performance of the new implementation shall be no worse than -10% compared
to the performance of SSD index as of 04/19/2024 for FBV and OpenAI embeddings
on datasets relevant to production workloads (such as email). Performance includes query latency
and QPS with 1, 4, 16, 32 threads on a machine with >=32 vCPUs and a local SSD. 
 
**Additional details needed in the algorithm**
--------------------------------------------
`diskann_async.rs` needs a few more features to help us meet exit criteria.
We commit to implementing these and additional ones as needed.
1. Beam search to explore more than one neighbor at a time and increase SSD queue depth.
2. [Completed] Prefetch quantized versions to cache during greedy search iterations.
3. [Completed] Associated data feature must be added as an optional implementation 
of the element fetching with provider interfaces to diskann_async.rs.
4. Filtered diskann must be implemented in diskann_async.rs and related files.
