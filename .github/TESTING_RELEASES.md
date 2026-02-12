# Testing the Release Workflow

This guide explains how to test the automated crate publishing workflow before actually publishing to crates.io.

## Quick Start

### Option 1: Manual Dry-Run (Recommended)

The safest way to test the release workflow:

1. **Go to GitHub Actions**
   - Navigate to: `https://github.com/microsoft/DiskANN/actions/workflows/publish.yml`
   - Click the "Run workflow" dropdown button

2. **Configure the test run**
   - **Branch**: Select your branch (e.g., `main` or a release branch)
   - **Run in dry-run mode**: Keep as `true` (default)
   - Click the green "Run workflow" button

3. **Monitor the test**
   - Watch the workflow execution in real-time
   - All steps will run, but nothing will be published
   - Look for: "✅ Dry-run completed successfully!"

### Option 2: Local Testing with cargo

Test individual crates locally:

```bash
# Test a single crate
cargo publish --dry-run --package diskann-wide

# Test all crates in order
cargo publish --dry-run --package diskann-wide
cargo publish --dry-run --package diskann-vector
cargo publish --dry-run --package diskann-platform
# ... etc
```

## What Gets Tested in Dry-Run Mode

✅ **Tested:**
- Rust installation and caching
- Full test suite execution
- Crate metadata validation
- Packaging verification
- Dependency resolution
- Publish order correctness

❌ **NOT Tested:**
- Actual publishing to crates.io
- Crate availability timing
- Registry token authentication
- Network upload reliability

## When to Run a Dry-Run Test

**Always test before:**
- Your first release
- Modifying the publish workflow
- Major version bumps
- Significant dependency changes

**Optional but recommended:**
- Minor/patch releases
- After long periods between releases

## Interpreting Results

### Success ✅

You'll see:
```
========================================
✅ Dry-run completed successfully!
All crates passed validation.
========================================
```

**Next step**: You're ready to create a release tag and publish for real.

### Failure ❌

Common issues and solutions:

1. **Test failures**
   - Fix failing tests before releasing
   - Check the test output in the workflow logs

2. **Invalid metadata**
   - Review `Cargo.toml` files for issues
   - Ensure all required fields are present

3. **Dependency issues**
   - Check for circular dependencies
   - Verify version constraints are correct

4. **Packaging errors**
   - Look for missing files or incorrect paths
   - Check `.gitignore` doesn't exclude required files

## Comparing Dry-Run vs Live Release

| Aspect | Dry-Run | Live Release |
|--------|---------|--------------|
| Trigger | Manual dispatch | Version tag push |
| Tests | ✓ Full suite | ✓ Full suite |
| Validation | ✓ All checks | ✓ All checks |
| Publishes | ✗ Simulated | ✓ Actual |
| Waits for availability | ✗ Skipped | ✓ 5min timeout |
| Version tag required | ✗ No | ✓ Yes |
| CRATES_IO_TOKEN used | ✗ No* | ✓ Yes |
| Creates release notes | ✗ No | ✓ Yes |

\* Token not required but harmless if present

## Troubleshooting

### "Workflow not found"

Make sure your branch has the updated workflow file. The dry-run feature was added in the latest version.

### "DRY_RUN variable not set"

This is expected for tag-triggered releases. The workflow automatically sets `DRY_RUN=false` for real releases.

### "Tests pass locally but fail in workflow"

- Check rust version matches: `1.92` (defined in workflow)
- Ensure all dependencies are in `Cargo.lock`
- Review LFS files if using Git LFS

## Example: Complete Pre-Release Test

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
git push origin main --tags  # This triggers real publish
```

## Need Help?

- Check the [RELEASING.md](../../RELEASING.md) guide
- Review workflow logs in GitHub Actions
- Open an issue if you find a bug in the workflow
