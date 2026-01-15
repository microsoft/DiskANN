# LinAlg

Reference data generator for Linear-Algebra tests in DiskANN.

## Installing Julia

Install Julia using the instructions outlined here: https://julialang.org/downloads/

## Generating Reference SVD Problems

Navigate via commandline to the directory containing this README and start Julia with
```
julia --project
```
If this is your first time running this package, the run the folling in the Julia REPL:
```julia
using Pkg; Pkg.instantiate()
```
Finally, generate reference problems using:
```julia
using LinAlg
LinAlg.generate_reference_tests("reference_svd_inputs.json")
```
This will create a JSON of reference problems which can be copied into `test_data` as needed.
