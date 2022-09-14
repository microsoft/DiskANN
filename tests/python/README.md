# Integration Tests
The following tests use Python to prepare, run, verify, and tear down the rest api services.

We do make use of the built-in `unittest` library, but that's only to take advantage of test reporting purposes.

These are decidedly **not** _unit_ tests. These are end to end integration tests.



    if (pq_file_num_centroids != 256) {
      diskann::cout << "Error. Number of PQ centroids is not 256. Exitting."
                    << std::endl;
      return -1;
    }
