/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(windows)]
pub mod win;

#[cfg(windows)]
pub use win::*;

#[cfg(not(windows))]
pub mod linux;

#[cfg(not(windows))]
pub use linux::*;
