RUSTFLAGS="-Ctarget-feature=+neon,+dotprod -Clinker=aarch64-linux-gnu-gcc" cargo test \
    --package "$1" \
    --target aarch64-unknown-linux-gnu \
    --profile ci
