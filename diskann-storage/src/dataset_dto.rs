/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Lightweight data-transfer object for aligned vector datasets.
//!
//! [`DatasetDto`] is a view type that pairs a mutable slice of vector data with
//! the rounded (padded) dimension used for aligned storage layouts.

/// Dataset DTO used to pass aligned vector data between layers.
///
/// `T` is the element type (e.g. `f32`, `u8`), and `rounded_dim` records the
/// padded dimension that may be larger than the original vector dimension due
/// to alignment requirements.
#[derive(Debug)]
pub struct DatasetDto<'a, T> {
    /// Mutable borrow of the underlying data slice.
    pub data: &'a mut [T],

    /// The rounded (padded) dimension of each vector in `data`.
    pub rounded_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_construction() {
        let mut data = vec![1.0f32, 2.0, 3.0, 0.0]; // dim=3, rounded_dim=4
        let dto = DatasetDto {
            data: &mut data,
            rounded_dim: 4,
        };
        assert_eq!(dto.rounded_dim, 4);
        assert_eq!(dto.data.len(), 4);
    }

    #[test]
    fn mutation_through_dto() {
        let mut data = vec![0u8; 8];
        {
            let dto = DatasetDto {
                data: &mut data,
                rounded_dim: 4,
            };
            dto.data[0] = 42;
            dto.data[7] = 255;
        }
        assert_eq!(data[0], 42);
        assert_eq!(data[7], 255);
    }
}
