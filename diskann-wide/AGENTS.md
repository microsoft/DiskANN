### Multi-platform support

When touching architecture-specific intrinsics, run cross-platform validation per `diskann-wide/README.md` and test:

- AVX-512 code on non-AVX-512 capable x86-64 machines.
- Aarch64 code on x86-64 machines.
- code compiled for and running on the `x86-64` CPU (no AVX/AVX2) does not execute unsupported instructions.
