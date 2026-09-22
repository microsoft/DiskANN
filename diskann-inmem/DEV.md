# Dev Docs

## Testing

Fully testing this crate requires enabling the `integration-test` feature.
The suggested command is
```
cargo test --package diskann-inmem --all-features
```
or
```
cargo nexttest run --package diskann-inmem --all-features
```

## Documentation

`diskann-inmem` uses `docsrs` to automatically tag items with their required features.
To build and preview the public docs, use the following command:
```
RUSTDOCFLAGS="--cfg docsrs -D rustdoc::all" cargo +nightly doc \
    --package diskann-inmem \
    --no-deps \
    --all-features
```
To build private docs, use this:
```
RUSTDOCFLAGS="--cfg docsrs -D rustdoc::all" cargo +nightly doc \
    --package diskann-inmem \
    --no-deps \
    --all-features \
    --document-private-items
```


