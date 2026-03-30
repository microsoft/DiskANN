# diskann-wide

Low-level SIMD abstraction layer for DiskANN. Provides portable vector
operations across multiple ISA backends:

| Backend      | ISA        | Required Features                                |
|--------------|------------|--------------------------------------------------|
| **V3**       | x86-64-v3  | AVX2, FMA, F16C                                  |
| **V4**       | x86-64-v4+ | AVX-512 (F/BW/VL/DQ/VNNI/BITALG/VPOPCNTDQ/VBMI)  |
| **Neon**     | AArch64    | NEON, dot product                                |
| **Emulated** | Any        | Scalar fallback                                  |

## Testing

The default `cargo test` runs tests for the host machine's native ISA. To validate AArch64
on x86, test AVX-512 on a non-AVX-512 machine, or ensure that baseline `x86-64` code does not
execute invalid instructions, an emulator is needed.

### AVX-512 and `x86-64` Baseline (Intel SDE)

[Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html)
emulates x86 with high fidelity. We use it to run AVX-512 code paths on AVX2-only machines and
to emulate older CPUs to ensure binaries compiled for the `x86-64` baseline behave correctly.

The steps below assume a Linux environment with SDE installed. On Windows, use
`CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER` instead to point at the SDE binary.

#### Testing AVX-512

```bash
WIDE_TEST_MIN_ARCH=x86-64-v4 \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="${PATH_TO_SDE} -spr --" \
  cargo test --profile ci --package diskann-wide
```

- `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER` tells Cargo to run every test
  binary through SDE automatically.
- `WIDE_TEST_MIN_ARCH=x86-64-v4` tells `diskann-wide` tests to require V4,
  so tests that would otherwise be skipped at runtime are executed.
- `-spr` selects Sapphire Rapids, which supports all V4 target features.
  Any CPU model from Ice Lake onward is supported.

#### Ensuring Baseline Correctness

To verify that code compiled for the `x86-64` baseline doesn't accidentally hit unsupported
instructions, run under a Nehalem model:

```bash
RUSTFLAGS="-Ctarget-cpu=x86-64" \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="${PATH_TO_SDE} -nhm --" \
  cargo test --profile ci --package diskann-wide
```

SDE will abort if any AVX instruction is executed, catching accidental use of
SIMD in baseline-compiled code.

### AArch64 on x86-64 (QEMU)

QEMU user-mode emulation runs AArch64 binaries on x86 Linux hosts. This flow
is unavailable on Windows and macOS.

First, install the cross-compiler and QEMU:

```bash
sudo apt install -y gcc-aarch64-linux-gnu qemu-user
rustup target add aarch64-unknown-linux-gnu
```

Then run tests:

```bash
WIDE_TEST_MIN_ARCH=neon \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" \
  cargo test --profile ci --package diskann-wide --target aarch64-unknown-linux-gnu
```

- `-L /usr/aarch64-linux-gnu` sets the sysroot so QEMU can find the AArch64
  dynamic linker and libc.
- `.cargo/config.toml` already sets `-Ctarget-feature=+neon,+dotprod`, but
  `WIDE_TEST_MIN_ARCH=neon` double-checks that these features are applied.
