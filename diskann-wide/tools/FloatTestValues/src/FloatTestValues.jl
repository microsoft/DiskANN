# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

module FloatTestValues

export generate_reference_file, generate_reference_values

"""
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
"""
function generate_reference_file(file::String)
    open(io -> generate_reference_file(io), file; write = true)
end

function generate_reference_file(io::IO)
    values = generate_reference_values()
    for (f16, f32) in values
        println(io, "0x", string(f16, base = 16), ", ", f32)
    end
end

"""
    make_repr(x::Float32) -> Union{Float32, String}

Lower the 32-bit floating point number to a JSON-compatible value using the following
rules:

* If `x == -Inf`, return "neg_infinity"
* If `x == Int`, return "infinity"
* If `x == NaN`, return "nan"
* Else, return x
"""
function make_repr(x::Float32)
    if x == -Inf32
        "neg_infinity"
    elseif x == Inf32
        "infinity"
    elseif isnan(x)
        "nan"
    else
        x
    end
end

function generate_reference_values()
    # Generate all possible float16 values and sort from smallest to largest.
    all_float16_values = [reinterpret(Float16, i) for i in zero(UInt16):typemax(UInt16)]
    sort!(all_float16_values)
    [(reinterpret(UInt16, v), make_repr(convert(Float32, v))) for v in all_float16_values]
end

end # module FloatTestValues
