RUSTFLAGS="-Ctarget-feature=+neon,+dotprod,+fhm -Clinker=aarch64-linux-gnu-gcc" cargo test \
    --package "$1" \
    --target aarch64-unknown-linux-gnu
