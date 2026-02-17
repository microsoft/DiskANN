# CI Scripts

This directory contains utility scripts used by the CI workflows.

## get-rust-version.sh

A minimal script to extract the Rust toolchain version from `rust-toolchain.toml`.

### Usage

```bash
# From repository root (default)
.github/scripts/get-rust-version.sh

# With explicit path
.github/scripts/get-rust-version.sh path/to/rust-toolchain.toml
```

### Output

Prints the Rust version string to stdout (e.g., `1.92`).

### Error Handling

- Exits with code 1 if the file is not found
- Exits with code 1 if the `channel` key is not found in the TOML file

### Implementation

The script uses standard shell tools (`grep`, `sed`, `tr`) to parse the TOML file:
- Searches for lines containing `channel =`
- Extracts the value (supports both quoted and unquoted values)
- Removes whitespace and quotes

This avoids adding dependencies on TOML parsing libraries while remaining simple and maintainable.
