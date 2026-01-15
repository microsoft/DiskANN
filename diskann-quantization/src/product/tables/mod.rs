/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod basic;
mod transposed;

#[cfg(test)]
pub(super) mod test;

/////////////
// Exports //
/////////////

// Error types
pub use basic::{BasicTable, BasicTableBase, BasicTableView, TableCompressionError};
pub use transposed::TransposedTable;
