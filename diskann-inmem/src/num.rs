/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct Align(pub usize);
