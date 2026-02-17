#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Minimal script to extract Rust version from rust-toolchain.toml
# Usage: ./get-rust-version.sh [path-to-rust-toolchain.toml]

set -euo pipefail

# Default to rust-toolchain.toml in repo root
TOOLCHAIN_FILE="${1:-rust-toolchain.toml}"

if [[ ! -f "$TOOLCHAIN_FILE" ]]; then
    echo "Error: File $TOOLCHAIN_FILE not found" >&2
    exit 1
fi

# Extract the channel value from the TOML file
# Supports both quoted and unquoted values
# Example: channel = "1.92" or channel = 1.92
version=$(grep -E '^\s*channel\s*=' "$TOOLCHAIN_FILE" | \
          sed -E 's/^\s*channel\s*=\s*"?([^"]+)"?\s*$/\1/' | \
          tr -d ' ')

if [[ -z "$version" ]]; then
    echo "Error: Could not find 'channel' in $TOOLCHAIN_FILE" >&2
    exit 1
fi

echo "$version"
