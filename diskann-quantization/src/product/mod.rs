/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Product quantization training and compression.

pub mod train;

mod tables;

/////////////
// Exports //
/////////////

// Error types
pub use tables::{
    BasicTable, BasicTableBase, BasicTableView, TableCompressionError, TransposedTable,
};
