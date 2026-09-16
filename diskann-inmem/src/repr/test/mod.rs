/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::repr;

/// A test distance that simple sums scalar floating point values.
#[derive(Debug)]
pub(super) struct TestDistance;

impl repr::internal::RawDistance for TestDistance {
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        assert_eq!(x.len(), std::mem::size_of::<f32>());
        assert_eq!(y.len(), std::mem::size_of::<f32>());

        let x = unsafe { x.as_ptr().cast::<f32>().read_unaligned() };
        let y = unsafe { y.as_ptr().cast::<f32>().read_unaligned() };

        Ok(x + y)
    }
}

#[derive(Debug)]
pub(super) struct TestQueryDistance {
    query: f32,
}

impl TestQueryDistance {
    pub(super) fn new(query: f32) -> Self {
        Self { query }
    }
}

impl repr::internal::RawQueryDistance for TestQueryDistance {
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error> {
        assert_eq!(x.len(), std::mem::size_of::<f32>());

        let x = unsafe { x.as_ptr().cast::<f32>().read_unaligned() };

        Ok(self.query + x)
    }
}
