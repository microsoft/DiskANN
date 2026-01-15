# FloatTestValue

Generates a reference file for float16 to float32 conversions. The reference file is a text
file consisting of pairs:
```text
0x1234, <number>
```
where the first number is a 16-bit unsigned number corresponding to the bit pattern of a
float16 number and `<number>` is its floating point representation. The value in `<number>`
can take on the following special strings:

* `"neg_infinity"`: For negative infinity.
* `"infinity"`: For infinity.
* `"nan"`: For NAN.

All float16 values will be present in the file and will be ordered from lowest to highest with
all `NaN` representations at the end.

## Installing Julia

Install Julia using the instructions outlined here: https://julialang.org/downloads/

## Generating Reference File

Navigate via commandline to the directory containing this README and start Julia with
```
julia --project
```
If this is your first time running this package, the run the following in the Julia REPL:
```julia
using Pkg; Pkg.instantiate()
```
Finally, generate reference problems using:
```julia
using FloatTestValue
generate_reference_file("float16_conversion.txt")
```
This will create the text file which can be copied into `test_data` as needed.
