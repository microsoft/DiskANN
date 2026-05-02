/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Product quantization training and compression.

pub mod train;

pub mod tables;

/////////////
// Exports //
/////////////

pub use tables::{
    BasicTable, BasicTableBase, BasicTableView, TableCompressionError, TransposedTable,
};
