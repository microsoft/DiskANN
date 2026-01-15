# DiskANN Utils

Traits, structs, and algorithms to be shared between the `diskann` specific crates and
auxiliary crates like `quantization`.

The goal is to allow crates like `diskann` to use traits like `Reborrow` and structs
like `MatrixView` without relying on `quantization`.

Dependencies of this crate should be kept to a minimum as like `vector` and `wide`, it sits
at the very bottom of the DiskANN dependency stack.

