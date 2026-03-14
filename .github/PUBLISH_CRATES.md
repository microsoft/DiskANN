# Release Process

Publishing DiskANN crates to [crates.io](https://crates.io).

## Overview

All workspace crates are published together with synchronized version numbers using `cargo publish --workspace`, which automatically resolves dependency order and waits for each crate to be indexed before publishing its dependents. The Rust toolchain version is read from [`rust-toolchain.toml`](../../rust-toolchain.toml).

Releases follow a pull-request workflow: bump the version on a branch, open a PR, let the dry-run check pass, merge, then tag the release via the GitHub UI.

## Prerequisites

1. **CRATES_IO_TOKEN Secret**: A crates.io API token configured as a GitHub repository secret named `CRATES_IO_TOKEN` with publish permissions for all DiskANN crates.
   - Create a token: [crates.io/settings/tokens](https://crates.io/settings/tokens)
   - Add the secret: Repository Settings → Secrets and variables → Actions → New repository secret

2. **Maintainer Access**: Write access to the repository and owner/maintainer of all crates on crates.io.

## Dry-Run Testing

A `cargo publish --workspace --dry-run` runs **automatically** as a pull-request check whenever `Cargo.toml` is changed. You can also trigger a dry-run manually:

### Manual: GitHub Actions

1. Navigate to: `https://github.com/microsoft/DiskANN/actions/workflows/publish.yml`
2. Click **Run workflow**, select your branch, keep **dry-run = true**
3. Watch the workflow — look for successful `cargo publish --workspace --dry-run`

### Manual: Local

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

1. **Create a release branch** from `main`:

   ```bash
   git checkout main && git pull
   git checkout -b release-0.46.0
   ```

2. **Update version** in root `Cargo.toml`:

   - Set `workspace.package.version`:

     ```toml
     [workspace.package]
     version = "0.46.0"
     ```

   - Update **all internal crate entries** under `[workspace.dependencies]` to match:

     ```toml
     diskann-wide = { path = "diskann-wide", version = "0.46.0" }
     diskann-vector = { path = "diskann-vector", version = "0.46.0" }
     # ... etc
     ```

   Member crates inherit `workspace.package.version` via `version.workspace = true`,
   but `[workspace.dependencies]` versions must be set explicitly (they're baked into
   published manifests for crates.io consumers). The publish workflow verifies these match.

3. **Update CHANGELOG** (if applicable).

4. **Push and open a pull request** to `main`:

   ```bash
   git commit -am "Bump version to 0.46.0"
   git push origin release-0.46.0
   ```

   Open a PR on GitHub. The **Publish to crates.io / Dry-run publish test** check runs automatically.

5. **Wait for checks** — the dry-run and CI must both pass before merge.

6. **Merge the PR** into `main`.

7. **Create a release** via the GitHub UI:
   - Go to **Releases → Draft a new release**
   - Create a new tag `v0.46.0` targeting `main`
   - Add release notes describing changes
   - Click **Publish release**

   Pushing the tag triggers the real publish workflow.

8. **Verify** the published crates — confirm the new version appears in the output:

   ```bash
   cargo search diskann --limit 20
   ```

## Pre-release Checklist

- [ ] All CI checks pass on the PR
- [ ] Version number is updated in `Cargo.toml`
- [ ] CHANGELOG is updated (if applicable)
- [ ] Documentation is up to date
- [ ] Breaking changes are clearly documented
- [ ] **Dry-run publish check passes on the PR**