/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical fitting and regional assignment for IVF construction.

pub mod assignment;
pub mod clustering;
mod split;

pub use split::SplitAndAssign;
