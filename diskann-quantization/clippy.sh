#!/bin/bash

set -e

CLIPPY="cargo clippy --package diskann-quantization --no-deps --all-targets"
POSTAMBLE="-- -D warnings"

# No features
$CLIPPY --no-default-features $POSTAMBLE

# One Feature
$CLIPPY --no-default-features --features rayon $POSTAMBLE
$CLIPPY --no-default-features --features linalg $POSTAMBLE
$CLIPPY --no-default-features --features flatbuffers $POSTAMBLE

# All Features
$CLIPPY --no-default-features --features "rayon linalg flatbuffers" $POSTAMBLE

