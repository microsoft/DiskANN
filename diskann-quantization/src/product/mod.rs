/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Product quantization training and compression.

pub mod tables;
pub mod train;

/////////////
// Exports //
/////////////

// Error types
pub use tables::{
    BasicTable, BasicTableBase, BasicTableView, TableCompressionError, TransposedTable,
};
