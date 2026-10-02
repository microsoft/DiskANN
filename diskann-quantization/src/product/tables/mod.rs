/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod basic;
mod transposed;

pub mod lookup;

#[cfg(test)]
pub(super) mod test;

/////////////
// Exports //
/////////////

pub use basic::{BasicTable, BasicTableBase, BasicTableView, TableCompressionError};
pub use transposed::TransposedTable;
