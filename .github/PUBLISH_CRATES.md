# Release Process

Publishing DiskANN crates to [crates.io](https://crates.io).

## Overview

All workspace crates are published together with synchronized version numbers using `cargo publish --workspace`, which automatically resolves dependency order and waits for each crate to be indexed before publishing its dependents. The release is triggered by pushing a version tag. The Rust toolchain version is read from [`rust-toolchain.toml`](../../rust-toolchain.toml).

## Prerequisites

1. **CRATES_IO_TOKEN Secret**: A crates.io API token configured as a GitHub repository secret named `CRATES_IO_TOKEN` with publish permissions for all DiskANN crates.
   - Create a token: [crates.io/settings/tokens](https://crates.io/settings/tokens)
   - Add the secret: Repository Settings → Secrets and variables → Actions → New repository secret

2. **Maintainer Access**: Write access to the repository and owner/maintainer of all crates on crates.io.

## Dry-Run Testing

**Always test before publishing a real release.**

### Option 1: GitHub Actions (Recommended)

1. Navigate to: `https://github.com/microsoft/DiskANN/actions/workflows/publish.yml`
2. Click **Run workflow**, select your branch, keep **dry-run = true**
3. Watch the workflow — look for successful `cargo publish --workspace --dry-run`

### Option 2: Local

```bash
cargo publish --locked --workspace --dry-run
```

### What Dry-Run Tests

- Crate metadata and packaging validation
- Dependency resolution and publish ordering
- Build verification

### What It Does NOT Test

- Actual publishing, registry token auth, upload reliability

## Release Steps

1. **Update version** in root `Cargo.toml`:

   ```toml
   [workspace.package]
   version = "0.46.0"
   ```

   All workspace crates inherit this via `version.workspace = true`.

2. **Update CHANGELOG** (if applicable).

3. **Run dry-run** on a branch to validate (see above).

4. **Tag and push**:

   ```bash
   git tag v0.46.0
   git push origin v0.46.0
   ```

   Tag format: `v{major}.{minor}.{patch}`

5. **Monitor** the workflow in the Actions tab.

6. **Verify**:

   ```bash
   cargo search diskann --limit 20
   ```

### Example Pre-Release Flow

```bash
# Update version
vim Cargo.toml  # Change to 0.46.0

# Commit to a branch (don't tag yet)
git checkout -b release-0.46.0
git commit -am "Bump version to 0.46.0"
git push origin release-0.46.0

# Run dry-run via GitHub Actions UI on release-0.46.0

# If successful, merge and tag
git checkout main
git merge release-0.46.0
git tag v0.46.0
git push origin main --tags  # Triggers the real publish
```

## Pre-release Checklist

- [ ] All CI checks pass on the main branch
- [ ] Version number is updated in `Cargo.toml`
- [ ] CHANGELOG is updated (if applicable)
- [ ] Documentation is up to date
- [ ] Breaking changes are clearly documented
- [ ] All tests pass locally: `cargo test --workspace`
- [ ] Code builds without warnings: `cargo build --workspace --release`
- [ ] **Dry-run workflow test passes successfully**