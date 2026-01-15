#pwsh version of benchmark\clippy.sh

# Exit on any error
$ErrorActionPreference = "Stop"

$CLIPPY = "cargo clippy --package diskann-benchmark --no-deps --all-targets"

$SPHERICAL = "spherical-quantization"
$PRODUCT = "product-quantization"
$SCALAR = "scalar-quantization"
$MINMAX = "minmax-quantization"

# No features
Invoke-Expression "$CLIPPY --no-default-features -- -D warnings"

# One Feature
Invoke-Expression "$CLIPPY --no-default-features --features `"$SPHERICAL`" -- -D warnings"
Invoke-Expression "$CLIPPY --no-default-features --features `"$PRODUCT`" -- -D warnings"
Invoke-Expression "$CLIPPY --no-default-features --features `"$SCALAR`" -- -D warnings"
Invoke-Expression "$CLIPPY --no-default-features --features `"$MINMAX`" -- -D warnings"

# All Features
Invoke-Expression "$CLIPPY --all-features -- -D warnings"
