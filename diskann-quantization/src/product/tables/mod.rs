/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod basic;
pub mod padded;
mod transposed;

#[cfg(test)]
pub(super) mod test;

/////////////
// Exports //
/////////////

pub use basic::{BasicTable, BasicTableBase, BasicTableView, TableCompressionError};
pub use padded::PaddedTable;
pub use transposed::TransposedTable;
