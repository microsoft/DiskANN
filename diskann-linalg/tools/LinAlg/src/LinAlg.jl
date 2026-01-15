#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#

module LinAlg

import JSON
import LinearAlgebra

#######################
# Top Level Functions #
#######################

"""
    generate_reference_tests(output::AbstractString, [sizes])

Generate reference SVD problems for the specified problem sizes and save the problems as
a JSON file to the provided output. Each problem will be stored as like the following:

```text
"m": (Int) The number of rows in the test matrix.
"n": (Int) The number of columns in the test matrix.
"matrix": (List[Float]) The test matrix in row major order. Will have length `m` * `n`
"singular_values": (List[Float]) The list of expected singular values in decreasing order.
    Will have length `min(m,n)`.
```

The Rust side is expected to iterate through the collection of test problems, decompose
the test matrix and compare the singular values to the expected list.
"""
function generate_reference_tests(
    output::AbstractString,
    sizes::Vector{Tuple{Int,Int}} = reference_sizes(),
)
    results = svd_reference(sizes)
    open(output; write = true) do io
        JSON.print(io, results, 4)
    end
end

"""
    reference_sizes() -> Vector{Tuple{Int, Int}}

Return a list of reference sizes containing all possible SVD dimension combinations up
to a maximum of 20.
"""
reference_sizes() = vec([(i, j) for i in 1:20, j in 1:20])

##################
# Implementation #
##################

"""
A reference problem for SVD solvers.

* `singular_values`: The expected singular values from a decomposition.
* `matrix`: The matrix to decompose.
"""
struct SVDReference
    singular_values::Vector{Float32}
    matrix::Matrix{Float32}
end

function JSON.lower(x::SVDReference)
    Dict{String,Any}(
        "m" => size(x.matrix, 1),
        "n" => size(x.matrix, 2),
        "matrix" => vec(transpose(x.matrix)),
        "singular_values" => x.singular_values
    )
end

"""
    svd_reference(sizes::Vector{Tuple{Int, Int}}) -> Vector{SVDReference}

Generate an `SVDReference` for each pair of dimensions in `sizes`.
"""
function svd_reference(sizes::Vector{Tuple{Int,Int}})
    tests = SVDReference[]
    for sz in sizes
        push!(tests, svd_reference(sz[1], sz[2]))
    end
    tests
end

"""
    svd_reference(nrows::Integer, ncols::Integer) -> SVDReference

Create a SVDReference with the given dimensions.

The strategy is to generate random orthogonal matrices `u` and `v` and a set of singular
values. The test matrix will then be computed using:
```julia
s = u * singular_values_on_diagonal * v'
```
Since singular values are uniquely determined, we expect an SVD of the generated matrix to
yield approximately the same singular values.
"""
function svd_reference(nrows::Integer, ncols::Integer)
    u = random_orthogonal_matrix(nrows)
    v = random_orthogonal_matrix(ncols)

    # Add 4 to the randomly distributed singular values to reduce the condition number
    # (ratio of largets to smallest singular value).
    #
    # This help the implementation checks to be a little more reliable.
    singular_values = abs.(randn(Float32, min(nrows, ncols))) .+ (4 * one(Float32))
    sort!(singular_values; rev = true)

    matrix = u * materialize_singular_values(singular_values, nrows, ncols) * v'
    SVDReference(singular_values, matrix)
end

"""
    materialize_singular_values(s::Vector{T}, m::Integer, n::Integer)::Matrix{T}

Materialize a full matrix with dimension `m x n` from the singular values in `s`.
The singular values will be populated along the main diagonal, with the remaining entries
set to zero.
"""
function materialize_singular_values(s::Vector{T}, m::Integer, n::Integer)::Matrix{T} where {T}
    @assert length(s) == min(m, n)
    x = zeros(T, m, n)
    for i in eachindex(s)
        x[i, i] = s[i]
    end
    x
end

"""
    random_orthogonal_matrix(n::Integer) -> Matrix{Float32}

Generate a random orthogonal matrix of size `n x n`.
"""
function random_orthogonal_matrix(n::Integer)
    # See: https://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization
    Q, R = LinearAlgebra.qr(randn(Float32, n, n))
    Q * LinearAlgebra.Diagonal(sign.(LinearAlgebra.diag(R)))
end

end # module LinAlg
