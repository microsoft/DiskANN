# Quantization

Refer to the crate level documentation for usage of the `quantization` crate.

This README refers to building instructions when using the `flatbuffers-build` feature of
this crate, which compiles the files in the `schemas` directory and copies the contents
into `src/flatbuffers`.

## Building with `flatbuffers-build`

The flatbuffers compiler `flatc` needs to be installed and the environment variable
`FLATC_EXE` must be defined with the path to the executable.

The executable can be downloaded from the official release pages: https://github.com/google/flatbuffers/releases/tag/v25.2.10

SHA-512 sums of the zip directories for `v25.2.10` are as follows.

* 6a20c2fc4e4e094574a0fd064f79a374eb9e6abba9e49d4543ec384b056725f6ca9f7823ba5952fcfa40e31a56a4e25baa659415d94edd69a7a978942577c579  Linux.flatc.binary.clang++-18.zip
* 8784aae9f7984fdf5685e3944787bc547ca3a8bccefa4ba33efbe73960ebb6c94c2061d251dcc00e683133e65c8f47833195e0293415bc8abbd7b5aab4419714  Windows.flatc.binary.zip

