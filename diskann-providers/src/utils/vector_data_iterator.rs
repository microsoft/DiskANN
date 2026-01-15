/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io::{Read, Seek, SeekFrom},
    marker::PhantomData,
    mem::size_of,
};

use crate::storage::StorageReadProvider;
use diskann::{ANNError, ANNErrorKind, utils::read_exact_into};
use thiserror::Error;

use crate::{model::graph::traits::GraphDataType, utils::read_metadata};

/// An iterator over the vector and associated data pairs in a dataset loaded from the storage provider.
pub struct VectorDataIterator<StorageProvider: StorageReadProvider, Data: GraphDataType> {
    vector_reader: StorageProvider::Reader,
    dimension: usize,
    associated_data_reader: Option<StorageProvider::Reader>,
    associated_data_length: usize,
    num_points: usize,
    current_index: usize,
    _phantom: PhantomData<Data>,
}

impl<StorageProvider: StorageReadProvider, Data: GraphDataType>
    VectorDataIterator<StorageProvider, Data>
{
    /// Create the iterator from a vector dataset stream and an associated data stream.
    /// vector_stream format: | num_points (4 bytes) | dimension (4 bytes) | vector data 1 (dimension * size_of::<Data::VectorDataType>())) | .. | vector data N |
    /// associated_data_stream format: | num_points (4 bytes) | associated_data_length | associated data 1 (associated_data_length) | .. | associated data N |
    pub fn new(
        vector_stream: &str,
        associated_data_stream: Option<String>,
        read_provider: &StorageProvider,
    ) -> std::io::Result<VectorDataIterator<StorageProvider, Data>> {
        let mut dataset_reader = read_provider.open_reader(vector_stream)?;

        let vector_metadata = read_metadata(&mut dataset_reader)?;
        let (vector_npts, vector_dim) = (vector_metadata.npoints, vector_metadata.ndims);

        let (associated_data_reader, associated_data_length) = if let Some(associated_data_stream) =
            associated_data_stream
        {
            let mut associated_data_reader = read_provider.open_reader(&associated_data_stream)?;

            let associated_metadata = read_metadata(&mut associated_data_reader)?;
            let (num_pts, length) = (associated_metadata.npoints, associated_metadata.ndims);

            if num_pts != vector_npts {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Number of points in vector stream ({}) does not match number of points in associated data stream ({}).",
                        vector_npts, num_pts
                    ),
                ));
            }

            (Some(associated_data_reader), length)
        } else {
            (None, 0)
        };

        Ok(VectorDataIterator {
            vector_reader: dataset_reader,
            dimension: vector_dim,
            associated_data_reader,
            associated_data_length,
            num_points: vector_npts,
            current_index: 0,
            _phantom: PhantomData,
        })
    }

    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    pub fn get_num_points(&self) -> usize {
        self.num_points
    }

    #[allow(clippy::type_complexity)]
    pub fn next_n(
        &mut self,
        n: usize,
    ) -> Option<Vec<(Box<[Data::VectorDataType]>, Data::AssociatedDataType)>> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some((vector, associated_data)) = self.next() {
                result.push((vector, associated_data));
            } else {
                break;
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Efficiently skip `n` elements by using seek operations
    ///
    /// This method efficiently jumps ahead in the iterator by seeking the underlying
    /// readers instead of iterating through each element.
    fn skip_elements(&mut self, n: usize) -> Result<(), SkipElementsError> {
        // Calculate how many elements we can actually skip
        let remaining = self.num_points.saturating_sub(self.current_index);
        if n > remaining {
            // If we try to skip more elements than are left,
            // move to the end and return how many we couldn't skip
            self.current_index = self.num_points;
            return Err(SkipElementsError::TooManyElements {
                requested: n,
                available: remaining,
            });
        }

        if n == 0 {
            return Ok(());
        }

        let vector_size = self.dimension * std::mem::size_of::<Data::VectorDataType>();
        self.vector_reader
            .seek(SeekFrom::Current((n * vector_size) as i64))?;

        // If we have associated data, seek there too
        if let Some(reader) = &mut self.associated_data_reader {
            let data_size =
                self.associated_data_length * std::mem::size_of::<Data::AssociatedDataType>();
            reader.seek(SeekFrom::Current((n * data_size) as i64))?;
        }

        // Update the current index
        self.current_index += n;
        Ok(())
    }
}

impl<R: StorageReadProvider, Data: GraphDataType> Iterator for VectorDataIterator<R, Data> {
    type Item = (Box<[Data::VectorDataType]>, Data::AssociatedDataType);

    /// Returns the next vector and associated data pair in the dataset.
    fn next(&mut self) -> Option<Self::Item> {
        self.current_index += 1;
        if self.current_index > self.num_points {
            return None;
        }

        let data = read_exact_into(&mut self.vector_reader, self.dimension);
        let boxed_vector_slice = match data {
            Ok(data) => data.into_boxed_slice(),
            Err(_) => return None, // Return None if data is an Err
        };

        match &mut self.associated_data_reader {
            Some(reader) => {
                let mut associated_data_buf =
                    vec![0u8; self.associated_data_length * size_of::<Data::AssociatedDataType>()];
                if reader.read_exact(&mut associated_data_buf).is_err() {
                    return None;
                }

                match bincode::deserialize(&associated_data_buf) {
                    Ok(associated_data) => Some((boxed_vector_slice, associated_data)),
                    Err(_) => None, // Return None if deserialization fails
                }
            }

            None => {
                // If there is no associated data, return it with the default value.
                Some((boxed_vector_slice, Data::AssociatedDataType::default()))
            }
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.skip_elements(n) {
            Ok(_) => self.next(),
            Err(SkipElementsError::TooManyElements { .. }) => None, // Expected when skipping past end
            Err(SkipElementsError::IoError(_)) => None, // IO errors should terminate iterator
        }
    }
}

/// Custom error type for skipping elements in the vector iterator
#[derive(Debug, Error)]
enum SkipElementsError {
    /// Tried to skip more elements than are available
    #[error("Tried to skip {requested} elements, but only {available} are left")]
    TooManyElements { requested: usize, available: usize },

    /// IO error occurred while seeking
    #[error("IO error while skipping elements: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<SkipElementsError> for ANNError {
    fn from(err: SkipElementsError) -> Self {
        ANNError::new(ANNErrorKind::IndexError, err)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::test_utils::graph_data_type_utils::GraphDataF32VectorU32Data;
    const TEST_VECTOR_STREAM: &str = "vector";
    const TEST_ASSOCIATED_DATA_STREAM: &str = "associated_data";
    const INCORRECT_TEST_ASSOCIATED_DATA_STREAM: &str = "incorrect_associated_data";

    struct MockStorageProvider;

    impl StorageReadProvider for MockStorageProvider {
        type Reader = Cursor<Vec<u8>>;

        fn open_reader(&self, _item_identifier: &str) -> std::io::Result<Self::Reader> {
            match _item_identifier {
                TEST_VECTOR_STREAM => {
                    let mut data = Vec::with_capacity(24);
                    data.extend_from_slice(&(2_i32.to_le_bytes()));
                    data.extend_from_slice(&(2_i32.to_le_bytes()));
                    data.extend_from_slice(&(1_f32.to_le_bytes()));
                    data.extend_from_slice(&(2_f32.to_le_bytes()));
                    data.extend_from_slice(&(3_f32.to_le_bytes()));
                    data.extend_from_slice(&(4_f32.to_le_bytes()));

                    Ok(Cursor::new(data))
                }

                TEST_ASSOCIATED_DATA_STREAM => {
                    let mut data = Vec::new();
                    data.extend_from_slice(&(2_i32.to_le_bytes()));
                    data.extend_from_slice(&(1_i32.to_le_bytes()));
                    data.extend_from_slice(&(10_u32.to_le_bytes()));
                    data.extend_from_slice(&(20_u32.to_le_bytes()));
                    Ok(Cursor::new(data))
                }

                INCORRECT_TEST_ASSOCIATED_DATA_STREAM => {
                    let mut data = Vec::new();
                    data.extend_from_slice(&(3_i32.to_le_bytes()));
                    data.extend_from_slice(&(4_i32.to_le_bytes()));
                    data.extend_from_slice(&(10_u32.to_le_bytes()));
                    data.extend_from_slice(&(20_u32.to_le_bytes()));

                    Ok(Cursor::new(data))
                }

                _ => {
                    panic!("Unexpected item identifier")
                }
            }
        }

        fn get_length(&self, _item_identifier: &str) -> std::io::Result<u64> {
            Ok(20)
        }

        fn exists(&self, _item_identifier: &str) -> bool {
            matches!(
                _item_identifier,
                TEST_VECTOR_STREAM
                    | TEST_ASSOCIATED_DATA_STREAM
                    | INCORRECT_TEST_ASSOCIATED_DATA_STREAM
            )
        }
    }

    #[test]
    fn test_initialization() {
        let read_provider = MockStorageProvider;
        let iterator = VectorDataIterator::<MockStorageProvider, GraphDataF32VectorU32Data>::new(
            TEST_VECTOR_STREAM,
            Some(TEST_ASSOCIATED_DATA_STREAM.to_string()),
            &read_provider,
        )
        .unwrap();

        assert_eq!(iterator.get_dimension(), 2);
        assert_eq!(iterator.get_num_points(), 2);
    }

    #[test]
    fn test_iteration() {
        let read_provider = MockStorageProvider;
        let mut iterator =
            VectorDataIterator::<MockStorageProvider, GraphDataF32VectorU32Data>::new(
                TEST_VECTOR_STREAM,
                Some(TEST_ASSOCIATED_DATA_STREAM.to_string()),
                &read_provider,
            )
            .unwrap();

        let (vector, associated_data) = iterator.next().unwrap();
        assert_eq!(vector, vec![1_f32, 2_f32].into_boxed_slice());
        assert_eq!(associated_data, 10_u32);

        let (vector, associated_data) = iterator.next().unwrap();
        assert_eq!(vector, vec![3_f32, 4_f32].into_boxed_slice());
        assert_eq!(associated_data, 20_u32);

        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_initialization_fail_when_associated_data_has_incorrect_length() {
        let read_provider = MockStorageProvider;
        let result = VectorDataIterator::<MockStorageProvider, GraphDataF32VectorU32Data>::new(
            TEST_VECTOR_STREAM,
            Some(INCORRECT_TEST_ASSOCIATED_DATA_STREAM.to_string()),
            &read_provider,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_skip_elements_error_conversion_to_ann_error() {
        // Test conversion of TooManyElements variant
        let skip_err = SkipElementsError::TooManyElements {
            requested: 10,
            available: 5,
        };
        let ann_err = ANNError::from(skip_err);

        assert_eq!(ann_err.kind(), ANNErrorKind::IndexError);

        let err_msg = ann_err.to_string();
        assert!(err_msg.contains("Tried to skip 10 elements, but only 5 are left"));

        // Test downcasting back to the original error
        let original_err = ann_err.downcast_ref::<SkipElementsError>().unwrap();
        assert!(matches!(
            original_err,
            SkipElementsError::TooManyElements {
                requested: 10,
                available: 5,
            }
        ));

        // Test conversion of IoError variant
        let io_err =
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "unexpected end of file");
        let skip_err = SkipElementsError::IoError(io_err);
        let ann_err = ANNError::from(skip_err);

        assert_eq!(ann_err.kind(), ANNErrorKind::IndexError);

        let err_msg = ann_err.to_string();
        assert!(err_msg.contains("IO error while skipping elements"));
        assert!(err_msg.contains("unexpected end of file"));

        // Verify the error chain includes the io::Error
        let original_err = ann_err.downcast_ref::<SkipElementsError>().unwrap();
        assert!(matches!(original_err, SkipElementsError::IoError(_)));
    }

    #[test]
    fn test_next_n() {
        let read_provider = MockStorageProvider;
        let mut iterator =
            VectorDataIterator::<MockStorageProvider, GraphDataF32VectorU32Data>::new(
                TEST_VECTOR_STREAM,
                Some(TEST_ASSOCIATED_DATA_STREAM.to_string()),
                &read_provider,
            )
            .unwrap();

        let result = iterator.next_n(2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, vec![1_f32, 2_f32].into_boxed_slice());
        assert_eq!(result[0].1, 10_u32);
        assert_eq!(result[1].0, vec![3_f32, 4_f32].into_boxed_slice());
        assert_eq!(result[1].1, 20_u32);

        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_skip_and_nth() {
        let read_provider = MockStorageProvider;
        let mut iterator1 =
            VectorDataIterator::<MockStorageProvider, GraphDataF32VectorU32Data>::new(
                TEST_VECTOR_STREAM,
                Some(TEST_ASSOCIATED_DATA_STREAM.to_string()),
                &read_provider,
            )
            .unwrap();

        // Using Iterator::nth directly
        let (vector, associated_data) = iterator1.nth(1).unwrap();
        assert_eq!(vector, vec![3_f32, 4_f32].into_boxed_slice());
        assert_eq!(associated_data, 20_u32);

        // Create a fresh iterator for the skip test
        let iterator2 = VectorDataIterator::<MockStorageProvider, GraphDataF32VectorU32Data>::new(
            TEST_VECTOR_STREAM,
            Some(TEST_ASSOCIATED_DATA_STREAM.to_string()),
            &read_provider,
        )
        .unwrap();

        // Using Iterator::skip (which calls nth internally) followed by next()
        let mut iter_after_skip = iterator2.skip(1);
        let (vector, associated_data) = iter_after_skip.next().unwrap();
        assert_eq!(vector, vec![3_f32, 4_f32].into_boxed_slice());
        assert_eq!(associated_data, 20_u32);
    }

    #[test]
    fn test_nth_out_of_bounds() {
        let read_provider = MockStorageProvider;
        let mut iterator =
            VectorDataIterator::<MockStorageProvider, GraphDataF32VectorU32Data>::new(
                TEST_VECTOR_STREAM,
                Some(TEST_ASSOCIATED_DATA_STREAM.to_string()),
                &read_provider,
            )
            .unwrap();

        // Try to get an element beyond the end
        assert!(iterator.nth(3).is_none());

        // Iterator should be at the end
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_nth_zero() {
        let read_provider = MockStorageProvider;
        let mut iterator =
            VectorDataIterator::<MockStorageProvider, GraphDataF32VectorU32Data>::new(
                TEST_VECTOR_STREAM,
                Some(TEST_ASSOCIATED_DATA_STREAM.to_string()),
                &read_provider,
            )
            .unwrap();

        // nth(0) should be equivalent to next()
        #[allow(clippy::iter_nth_zero)]
        let (vector, associated_data) = iterator.nth(0).unwrap();
        assert_eq!(vector, vec![1_f32, 2_f32].into_boxed_slice());
        assert_eq!(associated_data, 10_u32);
    }
}
