/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use hashbrown::{HashMap, hash_map::Entry};

use crate::{
    repr,
};

/// A test distance that simply sums scalar floating point values.
#[derive(Debug)]
pub(super) struct TestDistance;

impl repr::internal::RawDistance for TestDistance {
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        let x: f32 = bytemuck::pod_read_unaligned(x);
        let y: f32 = bytemuck::pod_read_unaligned(y);
        Ok(x + y)
    }
}

