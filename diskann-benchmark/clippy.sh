#!/bin/bash

set -e

CLIPPY="cargo clippy --package diskann-benchmark --no-deps --all-targets"

SPHERICAL="spherical-quantization"
PRODUCT="product-quantization"
SCALAR="scalar-quantization"
MINMAX="minmax-quantization"

# No features
$CLIPPY --no-default-features -- -D warnings

# One Feature
$CLIPPY --no-default-features --features "$SPHERICAL" -- -D warnings
$CLIPPY --no-default-features --features "$PRODUCT" -- -D warnings
$CLIPPY --no-default-features --features "$SCALAR" -- -D warnings
$CLIPPY --no-default-features --features "$MINMAX" -- -D warnings

# All Features
$CLIPPY --all-features -- -D warnings
