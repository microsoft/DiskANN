# Release Process

This document describes the dry run and actual release process for publishing DiskANN crates to [crates.io](https://crates.io).

## Overview

The DiskANN workspace consists of multiple crates that are published together with synchronized version numbers. The release process is automated through GitHub Actions and is triggered by pushing a version tag. The Rust toolchain version used by the workflow is read from [`rust-toolchain.toml`](../../rust-toolchain.toml).

## Prerequisites

1. **CRATES_IO_TOKEN Secret**: A crates.io API token must be configured as a GitHub repository secret named `CRATES_IO_TOKEN`. This token should have publish permissions for all DiskANN crates.
   - To create a token: Visit [crates.io/settings/tokens](https://crates.io/settings/tokens)
   - To add the secret: Go to repository Settings → Secrets and variables → Actions → New repository secret

2. **Maintainer Access**: You must have write access to the repository and be an owner/maintainer of all the crates on crates.io.

## Testing with Dry-Run

**Always test before publishing a real release.**

### Option 1: Manual Dry-Run via GitHub Actions (Recommended)

1. **Go to GitHub Actions**
   - Navigate to: `https://github.com/microsoft/DiskANN/actions/workflows/publish.yml`
   - Click the "Run workflow" dropdown button

2. **Configure the test run**
   - **Branch**: Select your branch (e.g., `main` or a release branch)
   - **Run in dry-run mode**: Keep as `true` (default)
   - Click the green "Run workflow" button

3. **Monitor the test**
   - Watch the workflow execution in real-time
   - It will install Rust, run the full test suite, and validate all 15 crates with `cargo publish --dry-run`
   - Look for: "✅ Dry-run completed successfully!"

### Option 2: Local Testing with cargo

```bash
# Test a single crate
cargo publish --dry-run --package diskann-wide

# Test all crates in dependency order
cargo publish --dry-run --package diskann-wide
cargo publish --dry-run --package diskann-vector
cargo publish --dry-run --package diskann-platform
cargo publish --dry-run --package diskann-linalg
cargo publish --dry-run --package diskann-utils
cargo publish --dry-run --package diskann-quantization
cargo publish --dry-run --package diskann
cargo publish --dry-run --package diskann-providers
cargo publish --dry-run --package diskann-disk
cargo publish --dry-run --package diskann-label-filter
cargo publish --dry-run --package diskann-benchmark-runner
cargo publish --dry-run --package diskann-benchmark-simd
cargo publish --dry-run --package diskann-benchmark-core
cargo publish --dry-run --package diskann-tools
cargo publish --dry-run --package diskann-benchmark
```

### What Gets Tested in Dry-Run Mode

✅ **Tested:**
- Rust installation and caching
- Limited test suite execution
- Crate metadata validation
- Packaging verification
- Dependency resolution
- Publish order correctness

❌ **NOT Tested:**
- Actual publishing to crates.io
- Crate availability timing
- Registry token authentication
- Network upload reliability

### Interpreting Results

#### Success ✅

You'll see:
```
==========================================
✅ Dry-run completed successfully!
All crates passed validation.
==========================================
```

**Next step**: You're ready to create a release tag and publish for real.

#### Failure ❌

Common issues and solutions:

1. **Test failures** — Fix failing tests before releasing
2. **Invalid metadata** — Review `Cargo.toml` files; ensure all required fields are present
3. **Dependency issues** — Check for circular dependencies; verify version constraints
4. **Packaging errors** — Look for missing files; check `.gitignore` doesn't exclude required files

## Release Steps

1. **Update Version Numbers**

   Update the version in the root `Cargo.toml` workspace configuration:

   ```toml
   [workspace.package]
   version = "0.46.0"  # Update this version
   ```

   All workspace crates use `version.workspace = true`, so this single change updates all crates.

2. **Update CHANGELOG** (if one exists)

   Document the changes, new features, bug fixes, and breaking changes.

3. **Test with Dry-Run**

   Before creating the tag, commit your version update to a branch and run the dry-run workflow (see above). Verify all crates pass validation.

4. **Create and Push a Version Tag**

   ```bash
   git tag v0.46.0
   git push origin v0.46.0
   ```

   **Important**: The tag format must be `v{major}.{minor}.{patch}` (e.g., `v0.46.0`).

5. **Monitor the Release Workflow**

   - Navigate to the Actions tab in the GitHub repository
   - Find the "Publish to crates.io" workflow run
   - The workflow will verify the tag version matches `Cargo.toml`, run the test suite, publish crates in dependency order, and wait for each crate to be available before publishing dependents.

6. **Verify Publication**

   ```bash
   cargo search diskann --limit 20
   ```

### Example: Complete Pre-Release Flow

```bash
# 1. Update version locally
vim Cargo.toml  # Change version to 0.46.0

# 2. Commit to a branch (don't tag yet!)
git checkout -b release-0.46.0
git commit -am "Bump version to 0.46.0"
git push origin release-0.46.0

# 3. Run dry-run test via GitHub UI
# - Go to Actions → Publish to crates.io
# - Run workflow on release-0.46.0 branch
# - Keep dry_run=true

# 4. If successful, merge to main and tag
git checkout main
git merge release-0.46.0
git tag v0.46.0
git push origin main --tags  # This triggers the real publish
```

## Crate Dependency Order

The crates are published in the following order to respect dependencies:

| Layer | Crates |
|-------|--------|
| 1 — Base | `diskann-wide` |
| 2 — Vector and Platform | `diskann-vector`, `diskann-platform` |
| 3 — Linalg and Utils | `diskann-linalg`, `diskann-utils` |
| 4 — Quantization | `diskann-quantization` |
| 5 — Core Algorithm | `diskann` |
| 6 — Providers | `diskann-providers`, `diskann-disk`, `diskann-label-filter` |
| 7 — Benchmark Runner | `diskann-benchmark-runner` |
| 8 — Benchmark Support | `diskann-benchmark-simd`, `diskann-benchmark-core` |
| 9 — Tools | `diskann-tools` |
| 10 — Benchmark | `diskann-benchmark` |


## Pre-release Checklist

- [ ] All CI checks pass on the main branch
- [ ] Version number is updated in `Cargo.toml`
- [ ] CHANGELOG is updated (if applicable)
- [ ] Documentation is up to date
- [ ] Breaking changes are clearly documented
- [ ] All tests pass locally: `cargo test --workspace`
- [ ] Code builds without warnings: `cargo build --workspace --release`
- [ ] **Dry-run workflow test passes successfully**