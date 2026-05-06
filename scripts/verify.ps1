$ErrorActionPreference = 'Stop'
# Verify: repeatable health check (assumes rustfmt + clippy installed)
cargo fmt --all --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo clippy --workspace --all-targets --config 'build.rustflags=["-Dwarnings"]'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
