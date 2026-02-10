# Release Process

This document describes the automated release process for publishing DiskANN crates to [crates.io](https://crates.io).

## Overview

The DiskANN workspace consists of multiple crates that are published together with synchronized version numbers. The release process is automated through GitHub Actions and is triggered by pushing a version tag.

## Testing the Release Process

**Before publishing a real release**, you should test the workflow in dry-run mode to ensure everything works correctly.

### Dry-Run Test

The publish workflow can be manually triggered with a dry-run mode that validates everything without actually publishing to crates.io:

1. **Navigate to GitHub Actions**
   - Go to the repository on GitHub
   - Click on the "Actions" tab
   - Select "Publish to crates.io" from the left sidebar

2. **Run Workflow Manually**
   - Click "Run workflow" button on the right
   - Select the branch (usually `main` or your release branch)
   - Keep "Run in dry-run mode" set to **true** (default)
   - Click "Run workflow"

3. **Monitor the Dry-Run**
   - Watch the workflow execution
   - It will:
     - ✓ Install Rust and dependencies
     - ✓ Run the full test suite
     - ✓ Validate all 15 crates with `cargo publish --dry-run`
     - ✓ Check the dependency order
     - ✗ NOT actually publish anything to crates.io

4. **Verify Results**
   - All steps should pass with green checkmarks
   - Look for the message: "✅ Dry-run completed successfully!"
   - Review any warnings or errors

**Tip**: Run the dry-run test before creating your release tag to catch issues early.

## Automated Publishing

### Prerequisites

1. **CRATES_IO_TOKEN Secret**: A crates.io API token must be configured as a GitHub repository secret named `CRATES_IO_TOKEN`. This token should have publish permissions for all DiskANN crates.
   - To create a token: Visit [crates.io/settings/tokens](https://crates.io/settings/tokens)
   - To add the secret: Go to repository Settings → Secrets and variables → Actions → New repository secret

2. **Maintainer Access**: You must have write access to the repository and be an owner/maintainer of all the crates on crates.io.

### Release Steps

1. **Update Version Numbers**
   
   Update the version in the root `Cargo.toml` workspace configuration:
   
   ```toml
   [workspace.package]
   version = "0.46.0"  # Update this version
   ```
   
   All workspace crates use `version.workspace = true`, so this single change updates all crates.

2. **Update CHANGELOG** (if one exists)
   
   Document the changes, new features, bug fixes, and breaking changes in the release.

3. **Test with Dry-Run** (Recommended)
   
   Before creating the tag, test the release process:
   - Commit your version update to a branch
   - Run the manual workflow in dry-run mode (see "Testing the Release Process" above)
   - Verify all crates pass validation

4. **Create and Push a Version Tag**
   
   ```bash
   # Create a tag matching the version
   git tag v0.46.0
   
   # Push the tag to trigger the release workflow
   git push origin v0.46.0
   ```
   
   **Important**: The tag format must be `v{major}.{minor}.{patch}` (e.g., `v0.46.0`).

5. **Monitor the Release Workflow**
   
   - Navigate to the Actions tab in the GitHub repository
   - Find the "Publish to crates.io" workflow run
   - Monitor the progress and check for any errors
   
   The workflow will:
   - Verify the tag version matches `Cargo.toml`
   - Run the test suite
   - Publish crates in dependency order
   - Wait for each crate to be available on crates.io before publishing dependents

5. **Verify Publication**
   
   After the workflow completes, verify that all crates are published:
   
   ```bash
   cargo search diskann --limit 20
   ```
   
   Check that the new version appears for all crates.

## Crate Dependency Order

The crates are published in the following order to respect dependencies:

1. **Layer 1 - Base**: `diskann-wide`
2. **Layer 2 - Vector and Platform**: `diskann-vector`, `diskann-platform`
3. **Layer 3 - Linalg and Utils**: `diskann-linalg`, `diskann-utils`
4. **Layer 4 - Quantization**: `diskann-quantization`
5. **Layer 5 - Core Algorithm**: `diskann`
6. **Layer 6 - Providers**: `diskann-providers`, `diskann-disk`, `diskann-label-filter`
7. **Layer 7 - Benchmark Runner**: `diskann-benchmark-runner`
8. **Layer 8 - Benchmark Support**: `diskann-benchmark-simd`, `diskann-benchmark-core`
9. **Layer 9 - Tools**: `diskann-tools`
10. **Layer 10 - Benchmark**: `diskann-benchmark`

## Troubleshooting

### Workflow Fails on Version Mismatch

**Error**: "Tag version does not match Cargo.toml version"

**Solution**: Ensure the tag version (without the 'v' prefix) exactly matches the version in `Cargo.toml`.

### Publication Fails for a Crate

**Error**: "failed to publish crate"

**Possible Causes**:
- Network issues (retry automatically handled)
- Dependency not yet available on crates.io (wait time automatically handled)
- Authentication issues (check CRATES_IO_TOKEN secret)
- Version already published (you cannot republish the same version)

**Solution**: 
- Check the workflow logs for specific error messages
- Verify the CRATES_IO_TOKEN secret is valid and has the correct permissions
- If a crate failed midway, you may need to manually publish remaining crates

### Manual Publishing

If the automated workflow fails and you need to publish manually:

```bash
# Set your crates.io token
export CARGO_REGISTRY_TOKEN="your-token-here"

# Publish in dependency order
cargo publish --package diskann-wide
cargo publish --package diskann-vector
cargo publish --package diskann-platform
cargo publish --package diskann-linalg
cargo publish --package diskann-utils
cargo publish --package diskann-quantization
cargo publish --package diskann
cargo publish --package diskann-providers
cargo publish --package diskann-disk
cargo publish --package diskann-label-filter
cargo publish --package diskann-benchmark-runner
cargo publish --package diskann-benchmark-simd
cargo publish --package diskann-benchmark-core
cargo publish --package diskann-tools
cargo publish --package diskann-benchmark
```

**Note**: Wait 30-60 seconds between each publish to ensure dependencies are available on crates.io.

## Pre-release Checklist

Before creating a release tag:

- [ ] All CI checks pass on the main branch
- [ ] Version number is updated in `Cargo.toml`
- [ ] CHANGELOG is updated (if applicable)
- [ ] Documentation is up to date
- [ ] Breaking changes are clearly documented
- [ ] All tests pass locally: `cargo test --workspace`
- [ ] Code builds without warnings: `cargo build --workspace --release`
- [ ] **Dry-run workflow test passes successfully**

## Dry-Run vs Live Mode

### Dry-Run Mode (Testing)

- **Purpose**: Validate the release process without actually publishing
- **How to trigger**: Manual workflow dispatch with `dry_run: true`
- **What it does**:
  - ✓ Runs all tests
  - ✓ Validates crate metadata and dependencies
  - ✓ Checks that all crates can be packaged
  - ✓ Verifies publish order
  - ✗ Does NOT publish to crates.io
  - ✗ Does NOT wait for crate availability
  - ✗ Does NOT require version tag
- **Use case**: Testing changes to the workflow, validating a new release before tagging

### Live Mode (Production)

- **Purpose**: Actually publish crates to crates.io
- **How to trigger**: Push a version tag (e.g., `v0.46.0`)
- **What it does**:
  - ✓ Validates tag matches version
  - ✓ Runs all tests
  - ✓ Publishes all 15 crates to crates.io
  - ✓ Waits for each crate to be available before publishing dependents
  - ✓ Creates release notes
- **Use case**: Official releases

**Recommendation**: Always run a dry-run test before pushing a release tag, especially if:
- You've modified the workflow
- It's your first time releasing
- The version number changed significantly
- Dependencies have been updated

## Security Considerations

- **Token Security**: The CRATES_IO_TOKEN should be kept secure and rotated periodically
- **Version Control**: Once a version is published to crates.io, it cannot be unpublished (only yanked)
- **Dependency Updates**: Ensure all dependencies are up to date and free of known vulnerabilities

## Rollback

If a release needs to be rolled back:

1. **Yank the Version** (if critical bug or security issue):
   
   You'll need to yank all 15 crates that were published. Use the following commands:
   
   ```bash
   # Replace 0.46.0 with the version to yank
   VERSION="0.46.0"
   
   # Yank all crates in reverse dependency order
   cargo yank --vers $VERSION diskann-benchmark
   cargo yank --vers $VERSION diskann-tools
   cargo yank --vers $VERSION diskann-benchmark-core
   cargo yank --vers $VERSION diskann-benchmark-simd
   cargo yank --vers $VERSION diskann-benchmark-runner
   cargo yank --vers $VERSION diskann-label-filter
   cargo yank --vers $VERSION diskann-disk
   cargo yank --vers $VERSION diskann-providers
   cargo yank --vers $VERSION diskann
   cargo yank --vers $VERSION diskann-quantization
   cargo yank --vers $VERSION diskann-utils
   cargo yank --vers $VERSION diskann-linalg
   cargo yank --vers $VERSION diskann-platform
   cargo yank --vers $VERSION diskann-vector
   cargo yank --vers $VERSION diskann-wide
   ```

2. **Publish a Patch Release** with the fix as soon as possible

**Note**: Yanking prevents new projects from using the version, but doesn't affect existing users who have already downloaded it.

## Questions or Issues

If you encounter issues with the release process:

1. Check the GitHub Actions workflow logs
2. Review this documentation
3. Open an issue in the repository with:
   - The tag you were trying to release
   - Relevant error messages from the workflow
   - Steps you've already tried
