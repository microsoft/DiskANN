/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod decode;
#[path = "../minmax8_x_minmax4.rs"]
mod kernel;
pub(crate) mod layout;
pub(crate) mod reader;

pub(crate) use kernel::{Driver, QueryCompensation};
