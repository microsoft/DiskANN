# DiskANN Record

This crate provides a small framework for persisting structured Rust values (`struct`, `enum`, 
`Option<T>`, `Vec<T>`, and primitives) as a  manifest (can be serialized to JSON) plus a set 
of side-car binary artifacts, and reloading them later. It is can be used by `diskann` 
providers and indexes to implement durable, consistent and backward-compatible checkpoints.

Types describe how they map to a versioned record by implementing the `save::Save` and
`load::Load` traits; the `save_fields!` and `load_fields!` macros handle the
field-by-field plumbing for plain structs. Every record carries a `Version` so loaders
can detect schema changes and either upgrade or fall back through a probing chain.

The goal is to allow crates like `diskann` to checkpoint their state without depending on
a particular serialization backend. Currently, this crate implements `Disk` and `Memory` backends
-- `Disk` for persistent storage and `Memory` for in-memory operations. 
`Memory` pipes the output of its `SaveContext` directly into the input of its `LoadContext`, so it is not compatible with other backends.
A hypothetical `ObjectStore` backend could be implemented to be compatible with `Disk` backends to support caching and other advanced features.

This crate has minimal dependencies by design.