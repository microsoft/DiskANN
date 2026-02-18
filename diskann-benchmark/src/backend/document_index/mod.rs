/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Backend benchmark implementation for document index with label filters.
//!
//! This benchmark tests the DocumentInsertStrategy which enables inserting
//! Document objects (vector + attributes) into a DiskANN index.

mod benchmark;

pub(crate) use benchmark::register_benchmarks;
