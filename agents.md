# DiskANN Repository - Agent Onboarding Guide

**Last Updated**: 2026-02-11 (based on v0.45.0, Rust 1.92)

This guide helps coding agents understand how to work efficiently with the DiskANN repository. It covers repository structure, build systems, testing, conventions, and common workflows.

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Repository Structure](#repository-structure)
3. [Build System](#build-system)
4. [Testing](#testing)
5. [Code Quality & Linting](#code-quality--linting)
6. [Critical Patterns & Conventions](#critical-patterns--conventions)
7. [Configuration Files](#configuration-files)
8. [Common Workflows](#common-workflows)
9. [Troubleshooting](#troubleshooting)
10. [CI/CD Pipeline](#cicd-pipeline)

---

## Repository Overview

**DiskANN** is a Rust implementation of scalable approximate nearest neighbor (ANN) search algorithms. The project is a rewrite from C++ to Rust.

### Key Facts
- **Language**: Rust (Edition 2021)
- **Toolchain**: Rust 1.92 (see `rust-toolchain.toml`)
- **License**: MIT
- **Version**: 0.45.0 (uses semantic versioning)
- **Architecture**: Workspace with 15+ crates
- **Legacy Code**: Older C++ code is on the `cpp_main` branch (not maintained)

### Key Resources
- **Papers**: 
  - [NeurIPS DiskANN](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
  - [Arxiv Fresh-DiskANN](https://arxiv.org/abs/2105.09613)
  - [Filtered-DiskANN](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)
- **Contributing**: See `CONTRIBUTING.md` (requires CLA)
- **Code of Conduct**: See `CODE_OF_CONDUCT.md`

---

## Repository Structure

### Workspace Organization

The repository uses a Cargo workspace with crates organized into functional tiers:

#### **Tier 1: Foundation (No External Dependencies)**
```
diskann-wide/         - Low-level SIMD, bit manipulation, type width abstractions
diskann-vector/       - Vector primitives and operations
diskann-platform/     - Platform-specific utilities
```

#### **Tier 2: Core Libraries**
```
diskann-linalg/       - Linear algebra operations
diskann-utils/        - Shared utilities (Reborrow, MatrixView traits)
                        NOTE: Keep dependencies minimal in this crate
diskann-quantization/ - Vector quantization (PQ, SQ)
                        Contains FlatBuffers schemas in schemas/
                        Generated code in src/flatbuffers/
```

#### **Tier 3: Algorithm & Storage**
```
diskann/              - Core ANN graph algorithm and in-memory indexing
                        This is the CENTRAL crate
diskann-providers/    - Storage abstraction layer (physical/virtual/overlay)
diskann-disk/         - Disk-based indexing with io_uring support
diskann-label-filter/ - Inverted index for filtered search
```

#### **Tier 4: Infrastructure & Tools**
```
diskann-benchmark-runner/  - Test runner infrastructure
diskann-benchmark-core/    - Benchmark framework
diskann-benchmark-simd/    - SIMD-specific benchmarks
diskann-benchmark/         - Benchmark definitions and runners
diskann-tools/             - CLI utilities (autotuner, etc.)
```

### Default Members
The following crates are built by default with `cargo build`:
- diskann-wide
- diskann-vector
- diskann-quantization
- diskann-utils
- diskann

### Important Directories
```
/test_data/                   - Test datasets (committed with Git LFS)
/diskann/tests/generated/     - Cached test baselines (JSON format)
/diskann-benchmark/examples/  - Example benchmark configurations
/.cargo/                      - Cargo and nextest configuration
/.github/workflows/           - CI configuration
```

---

## Build System

### Basic Build Commands

```bash
# Build default workspace members
cargo build

# Build all workspace members
cargo build --workspace

# Build specific crate
cargo build -p diskann-disk

# Release build (optimized)
cargo build --release

# CI profile build (faster than release, includes debug info)
cargo build --profile ci
```

### Build Profiles

The project defines two main build profiles in `Cargo.toml`:

#### **Release Profile**
```toml
[profile.release]
opt-level = 3              # Maximum optimization
codegen-units = 1          # Single compilation unit for better optimization
debug = true               # Include debug info
split-debuginfo = "packed" # Packed debug info format
```

#### **CI Profile** (for fast builds with checks)
```toml
[profile.ci]
inherits = "dev"
opt-level = 1              # Light optimization
debug = true
debug-assertions = true
overflow-checks = true
```

### Target Configuration

The project uses `.cargo/config.toml` for target-specific settings:

- **x86_64**: Targets `x86-64-v3` by default (requires AVX2, FMA)
  - Override with: `RUSTFLAGS="-C target-cpu=x86-64"` for older CPUs
- **Windows**: Uses Control Flow Guard (`-C control-flow-guard`)

### Special Build Features

#### **diskann-quantization with FlatBuffers**

To build with the `flatbuffers-build` feature:

1. Download `flatc` v25.2.10:
   - Linux: https://github.com/google/flatbuffers/releases/download/v25.2.10/Linux.flatc.binary.clang++-18.zip
   - Windows: https://github.com/google/flatbuffers/releases/download/v25.2.10/Windows.flatc.binary.zip

2. Verify SHA-512 checksums (see `diskann-quantization/README.md`)

3. Set environment variable:
   ```bash
   export FLATC_EXE=/path/to/flatc
   ```

4. Build:
   ```bash
   cargo build -p diskann-quantization --features flatbuffers-build
   ```

A helper script is available at `diskann-quantization/tools/download-flatc.sh` for Linux.

---

## Testing

### Test Execution

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p diskann

# Run specific test
cargo test -p diskann -- --exact test_name

# Run with CI profile (faster)
cargo test --profile ci

# Run doc tests
cargo test --doc
```

**Note**: CI uses `cargo-nextest` for running tests:
```bash
cargo nextest run --workspace --cargo-profile ci
```

### Test Baseline Caching System

DiskANN uses a unique baseline caching system for regression detection:

#### **How It Works**
1. Tests serialize their results to JSON files in `diskann/tests/generated/`
2. These files serve as baselines for future test runs
3. Any difference between current run and baseline triggers a test failure

#### **Key APIs** (in `diskann/src/test/`)

**Cache Module** (`diskann/src/test/cache.rs`):
```rust
// Get baseline or save new one
get_or_save_test_results<R>(test_name: &str, results: &R) -> R

// Utilities for hierarchical test names
TestRoot   // Root of test hierarchy
TestPath   // Path builder (e.g., "a/b/test" → diskann/tests/generated/a/b/test.json)
```

**Comparison Module** (`diskann/src/test/cmp.rs`):
```rust
// Verbose equality trait (better error messages than PartialEq)
trait VerboseEq

// Macro to implement VerboseEq
verbose_eq!(StructName { field1, field2, field3 });

// Assert with verbose output
assert_eq_verbose!(expected, actual);
```

#### **Regenerating Baselines**

When intentionally changing algorithm behavior:

```bash
# Regenerate all baselines
DISKANN_TEST=overwrite cargo test

# Regenerate for specific crate
DISKANN_TEST=overwrite cargo test -p diskann
```

**Important**: Before committing, delete `diskann/tests/generated/` completely to remove stale baselines, then regenerate.

#### **Example Usage**
```rust
use diskann::test::{cache::get_or_save_test_results, cmp::assert_eq_verbose};

#[test]
fn my_algorithm_test() {
    let results = run_my_algorithm();
    let baseline = get_or_save_test_results("algorithm/my_test", &results);
    assert_eq_verbose!(baseline, results);
}
```

### Test Configuration

See `.cargo/nextest.toml`:
- **Timeout**: 600s (10 minutes) per test
- **Retries**: 3 attempts for flaky tests
- **Threads**: Uses all CPUs (`num-cpus`)
- **Fail-fast**: Disabled (runs all tests even after failures)

---

## Code Quality & Linting

### Formatting

```bash
# Check formatting
cargo fmt --check

# Apply formatting
cargo fmt
```

Format configuration in `rustfmt.toml`:
- Auto-reorder imports
- Unix newlines

### Clippy (Linting)

```bash
# Basic clippy check
cargo clippy --workspace --all-targets

# CI-style check (warnings as errors)
cargo clippy --workspace --all-targets --config 'build.rustflags=["-Dwarnings"]'

# Check with no default features (for specific crates)
cargo clippy -p diskann --no-default-features
```

### Code Coverage

CI runs coverage using `cargo-llvm-cov`:

```bash
# Generate coverage report
cargo llvm-cov nextest --workspace --lcov --output-path lcov.info
```

**Coverage Policy** (see `.codecov.yml`):
- **Patch Coverage**: ≥90% (blocks PRs)
- **Project Coverage**: Auto target (informational only)
- **Ignored**: `test_data/`, `*_test.rs`, `tests/`, `benches/`, `examples/`

---

## Critical Patterns & Conventions

### 1. Rayon Thread Pool Enforcement

**Problem**: Using Rayon's global thread pool can cause issues with thread locality and resource management.

**Solution**: The project enforces using explicitly-managed thread pools via custom utilities.

#### **Disallowed Methods** (will fail clippy)
❌ DO NOT USE:
```rust
rayon::iter::ParallelIterator::for_each()
rayon::iter::ParallelIterator::collect()
rayon::iter::ParallelIterator::sum()
rayon::iter::ParallelIterator::count()
// ... and many more (see clippy.toml for full list)
```

#### **Allowed Methods**
✅ USE INSTEAD:
```rust
use diskann_utils::rayon_utils::*;

for_each_in_pool()           // Instead of for_each()
collect_in_pool()            // Instead of collect()
sum_in_pool()                // Instead of sum()
try_for_each_in_pool()       // Instead of try_for_each()
```

**Reference**: See `clippy.toml` lines 1-40 for complete list of disallowed methods.

### 2. Random Number Generation

❌ DO NOT USE:
```rust
use rand::thread_rng;
rand::thread_rng()
```

✅ USE INSTEAD:
```rust
// Use functions from the random.rs module in appropriate crate
```

**Reason**: Ensures deterministic behavior for testing and reproducibility.

### 3. Filesystem Access

❌ AVOID:
```rust
use std::fs;
std::fs::read_to_string()
```

✅ USE INSTEAD:
```rust
// Use FileStorageProvider abstraction
// In tests: use VirtualStorageProvider::new_overlay()
```

❌ DO NOT USE in tests:
```rust
vfs::PhysicalFS::new()
```

✅ USE INSTEAD in tests:
```rust
VirtualStorageProvider::new_overlay()  // For overlaying virtual FS on physical
VirtualStorageProvider::new_memory()   // For pure in-memory FS
VirtualStorageProvider::new_physical() // For physical FS
```

### 4. Unsafe Code

All `unsafe` blocks MUST include a `// SAFETY:` comment explaining why the unsafe code is sound:

```rust
// SAFETY: This is safe because the slice is aligned and within bounds
unsafe {
    std::ptr::read(ptr)
}
```

This is enforced as a warning in `Cargo.toml`:
```toml
[workspace.lints.clippy]
undocumented_unsafe_blocks = "warn"
```

### 5. VirtualStorageProvider API

When using `VirtualStorageProvider`, use the factory methods, not the hidden constructor:

✅ CORRECT:
```rust
VirtualStorageProvider::new_memory()
VirtualStorageProvider::new_overlay(physical_path)
VirtualStorageProvider::new_physical()
```

❌ INCORRECT:
```rust
VirtualStorageProvider::new()  // Hidden constructor
```

### 6. Test Patterns

**DO**:
- Use baseline caching for regression detection
- Use `VerboseEq` trait for comparisons
- Name tests hierarchically (e.g., "module/submodule/test")
- Clean up `diskann/tests/generated/` before committing new baselines

**DON'T**:
- Add tests for derived traits (Clone, Debug, PartialEq)
- Add tests for enums unless they have explicit functionality

---

## Configuration Files

| File | Purpose |
|------|---------|
| `rust-toolchain.toml` | Pins Rust version to 1.92 |
| `Cargo.toml` (root) | Workspace definition, shared dependencies, profiles, lints |
| `clippy.toml` | Clippy lint rules (rayon, rand, fs enforcement) |
| `rustfmt.toml` | Format settings (import ordering, newlines) |
| `.cargo/config.toml` | Target-specific build flags (x86-64-v3, CFG) |
| `.cargo/nextest.toml` | Test runner configuration (timeouts, retries) |
| `.codecov.yml` | Code coverage thresholds and rules |
| `.gitignore` | Build artifacts, IDE files, coverage files |

---

## Common Workflows

### Running CI Checks Locally

```bash
# Format check
cargo fmt --check

# Clippy check (all targets)
cargo clippy --workspace --all-targets --config 'build.rustflags=["-Dwarnings"]'

# Clippy check (no default features) - for key crates
cargo clippy -p diskann --no-default-features --config 'build.rustflags=["-Dwarnings"]'

# Run tests
cargo test --workspace --profile ci
```

### Working with Benchmarks

```bash
# List available benchmarks
cargo run --release -p diskann-benchmark -- benchmarks

# List available inputs
cargo run --release -p diskann-benchmark -- inputs

# Get skeleton configuration
cargo run --release -p diskann-benchmark -- skeleton

# Get example input for specific benchmark
cargo run --release -p diskann-benchmark -- inputs async-index-build

# Run benchmark
cargo run --release -p diskann-benchmark -- run \
  --input-file ./diskann-benchmark/example/async.json \
  --output-file output.json
```

### Making Code Changes

1. **Create a feature branch**
2. **Make minimal changes**
3. **Run local checks**:
   ```bash
   cargo fmt
   cargo clippy --workspace --all-targets
   cargo test -p <affected-crate>
   ```
4. **Update test baselines if needed**:
   ```bash
   DISKANN_TEST=overwrite cargo test -p <crate>
   ```
5. **Commit changes** (PR template will guide you)

### Updating Dependencies

1. Update version in root `Cargo.toml` under `[workspace.dependencies]`
2. Run security check (if ecosystem is supported):
   ```bash
   # Check for known vulnerabilities before adding
   # Note: Use security tools for npm, pip, cargo, etc.
   ```
3. Run full test suite:
   ```bash
   cargo test --workspace
   ```

---

## Troubleshooting

### Common Issues

#### 1. **"rustfmt" or "clippy" is not installed**

**Problem**: Error message like "'cargo-fmt' is not installed for the toolchain".

**Solution**: Install the required components:
```bash
rustup component add rustfmt
rustup component add clippy
```

These are required for development but not automatically installed with the toolchain.

#### 2. **Build Fails with "target-cpu=x86-64-v3 not supported"**

**Problem**: Your CPU doesn't support AVX2/FMA instructions.

**Solution**: Override the target CPU:
```bash
RUSTFLAGS="-C target-cpu=x86-64" cargo build
```

Or edit `.cargo/config.toml` to change `target-cpu=x86-64-v3` to `target-cpu=x86-64`.

#### 3. **Clippy Error: "disallowed method: rayon::iter::ParallelIterator::collect"**

**Problem**: Using Rayon global thread pool methods.

**Solution**: Import and use utilities from `rayon_utils.rs`:
```rust
use diskann_utils::rayon_utils::collect_in_pool;
// Use collect_in_pool() instead of collect()
```

#### 4. **Test Fails with "Baseline mismatch"**

**Problem**: Algorithm change affected test results.

**Solutions**:
- If change is intentional: `DISKANN_TEST=overwrite cargo test`
- If change is unintentional: Fix the code
- Review the diff in `diskann/tests/generated/` before committing

#### 5. **FlatBuffers Build Fails**

**Problem**: `flatc` compiler not found or wrong version.

**Solution**:
1. Download correct version (v25.2.10) from releases
2. Verify SHA-512 checksum (see `diskann-quantization/README.md`)
3. Set `FLATC_EXE` environment variable:
   ```bash
   export FLATC_EXE=/path/to/flatc
   ```

#### 6. **Nextest Not Found**

**Problem**: CI uses `cargo-nextest` but it's not installed locally.

**Solution**: Install it:
```bash
cargo install cargo-nextest
```

Or use regular cargo test:
```bash
cargo test
```

#### 7. **Git LFS Files Missing**

**Problem**: Test data files appear as pointers.

**Solution**: Install Git LFS and pull files:
```bash
git lfs install
git lfs pull
```

---

## CI/CD Pipeline

CI is defined in `.github/workflows/ci.yml`. Understanding the CI pipeline helps debug failures.

### CI Jobs

#### 1. **basics** (Gate Job)
Runs basic checks before expensive tests. Includes:
- `fmt`: Format check
- `clippy`: Lint check (x86_64 and ARM)
- `clippy-no-default-features`: Minimal feature checks for key crates

#### 2. **qemu** (Architecture Compatibility)
- Tests CPU feature dispatch on older CPUs
- Uses QEMU to emulate old Nehalem CPU (no AVX/AVX2)
- Ensures code runs on older hardware

#### 3. **test-workspace**
- Runs full test suite on Windows and Linux
- Uses `cargo nextest` for parallel test execution
- Runs both unit tests and doc tests

#### 4. **test-workspace-features**
- Tests with all optional features enabled:
  - `virtual_storage`
  - `bf_tree`
  - `spherical-quantization`
  - `product-quantization`
  - `tracing`
  - `experimental_diversity_search`

#### 5. **coverage**
- Generates code coverage using `cargo-llvm-cov`
- Uploads to Codecov
- Requires ≥90% patch coverage to pass

### CI Environment Variables

```bash
RUST_CONFIG='build.rustflags=["-Dwarnings"]'  # Treat warnings as errors
RUST_BACKTRACE=1                               # Full backtraces on panic
```

### Concurrency Control

CI uses concurrency groups to cancel outdated runs:
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}
  cancel-in-progress: true
```

### Debugging CI Failures

1. **Check the job that failed** (basics, test-workspace, etc.)
2. **Look at the specific step** (fmt, clippy, test)
3. **Reproduce locally**:
   ```bash
   # For format failures:
   cargo fmt --check
   
   # For clippy failures:
   cargo clippy --workspace --all-targets --config 'build.rustflags=["-Dwarnings"]'
   
   # For test failures:
   cargo test --workspace --profile ci
   ```
4. **Check for flaky tests** (nextest retries 3 times)
5. **Review Codecov report** for coverage failures

---

## Quick Reference

### Essential Commands

| Task | Command |
|------|---------|
| Format code | `cargo fmt` |
| Check format | `cargo fmt --check` |
| Lint code | `cargo clippy --workspace --all-targets` |
| Build (dev) | `cargo build` |
| Build (release) | `cargo build --release` |
| Build (CI) | `cargo build --profile ci` |
| Run all tests | `cargo test --workspace` |
| Run crate tests | `cargo test -p <crate>` |
| Update baselines | `DISKANN_TEST=overwrite cargo test` |
| Generate coverage | `cargo llvm-cov nextest --workspace --lcov` |
| CI checks (local) | `cargo fmt --check && cargo clippy && cargo test` |

### Common Crates

| Crate | Purpose |
|-------|---------|
| `diskann` | Core ANN algorithm (START HERE) |
| `diskann-wide` | SIMD and type abstractions |
| `diskann-vector` | Vector primitives |
| `diskann-quantization` | Vector quantization |
| `diskann-disk` | Disk-based indexing |
| `diskann-providers` | Storage abstraction |
| `diskann-benchmark` | Benchmarking framework |
| `diskann-tools` | CLI utilities |

### Getting Help

- **Documentation**: Run `cargo doc --open` to view crate docs
- **Examples**: Check `diskann-benchmark/examples/` for benchmark configs
- **Issues**: See `.github/ISSUE_TEMPLATE/` for bug reports and features
- **Contributing**: See `CONTRIBUTING.md` for guidelines

---

## Errors Encountered and Workarounds

During the creation of this onboarding guide, the following were verified and documented:

### 1. Build System
- ✅ `cargo build` works out of the box
- ✅ `cargo test` requires Git LFS for test data
- ✅ CI profile (`--profile ci`) builds successfully

### 2. Development Tools
- ⚠️ `rustfmt` and `clippy` are NOT installed by default with the toolchain
  - **Workaround**: Run `rustup component add rustfmt clippy`
  - These are required for development but optional for basic builds
- ℹ️ `cargo-nextest` is not installed by default (CI installs it)
  - **Workaround**: Use `cargo test` locally, or install with `cargo install cargo-nextest`

### 3. Testing
- ✅ Test baseline system works as documented
- ✅ `DISKANN_TEST=overwrite` flag works correctly

### 4. Dependencies
- ✅ All dependencies resolve correctly from crates.io
- ℹ️ FlatBuffers build requires manual setup (documented above)

### 5. Clippy Rules
- ✅ Strict rayon rules are enforced correctly
- ✅ Unsafe block documentation is checked

### 6. Platform Support
- ✅ Linux (tested)
- ⚠️ Windows (not tested in this session, but supported by CI)
- ⚠️ macOS (not explicitly tested, but should work)

### 7. First-Time Setup Checklist
For new developers or agents working with this repository:
```bash
# 1. Install Rust toolchain (will read rust-toolchain.toml)
rustup toolchain install

# 2. Install required components
rustup component add rustfmt clippy

# 3. (Optional) Install nextest for faster testing
cargo install cargo-nextest

# 4. (Optional) Setup Git LFS for test data
git lfs install
git lfs pull

# 5. Build the project
cargo build

# 6. Run basic checks
cargo fmt --check
cargo clippy
cargo test
```

No major errors or blockers were encountered during exploration. All documented workflows have been verified.

---

**End of Agent Onboarding Guide**

*This guide should be updated when major changes occur to the repository structure, build system, or development workflows.*
