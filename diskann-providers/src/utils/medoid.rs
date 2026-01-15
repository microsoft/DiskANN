/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::storage::StorageReadProvider;
use diskann::{ANNError, ANNResult, error::IntoANNResult, utils::VectorRepr};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use rand::Rng;
use rand_distr::{Distribution, StandardUniform};
use tracing::info;

use crate::{
    model::graph::traits::AdHoc,
    utils::{
        VectorDataIterator, load_metadata_from_file,
        sampling::{SampleVectorReader, SamplingDensity},
    },
};

/// Suggested maximum sample size for medoid calculation
pub const MAX_MEDOID_SAMPLE_SIZE: usize = 50_000;

/// Calculate the centroid (mean vector) from a vector iterator
fn calculate_centroid<T, Iter>(
    iter: Iter,
    dimension: usize,
    num_points: usize,
) -> ANNResult<Vec<f32>>
where
    T: VectorRepr + Copy,
    Iter: Iterator<Item = (Box<[T]>, ())>,
{
    let mut result = vec![0.0_f32; dimension];

    for (v, _) in iter {
        let vector = T::as_f32(&v).map_err(|x| x.into())?;
        if vector.len() != dimension {
            return Err(ANNError::log_index_error(
                "Vector f32 dimension doesn't match input dim.",
            ));
        }
        for j in 0..vector.len() {
            result[j] += vector[j];
        }
    }

    if num_points == 0 {
        Ok(result)
    } else {
        Ok(result
            .into_iter()
            .map(|item| item / num_points as f32)
            .collect())
    }
}

/// Calculate the centroid of a dataset using sampling
///
/// # Arguments
/// * `path` - Path to the binary vector file
/// * `reader` - Storage provider reader
/// * `sampling_rate` - Rate at which to sample vectors (0.0-1.0)
/// * `rng` - Random number generator
///
/// # Returns
/// Returns the centroid vector as f32 values and the number of vectors used
fn calculate_centroid_with_sampling<T, Reader>(
    path: &str,
    reader: &Reader,
    sampling_rate: f64,
    rng: &mut impl Rng,
) -> ANNResult<Vec<f32>>
where
    T: VectorRepr,
    Reader: StorageReadProvider,
{
    // Create a sample vector reader with the appropriate sampling density
    let mut sample_reader = SampleVectorReader::<T, _>::new(
        path,
        SamplingDensity::from_sample_rate(sampling_rate),
        reader,
    )?;
    let (npts, dim) = sample_reader.get_dataset_headers();
    let dim = dim as usize;

    // Generate random indices based on sampling rate
    let distribution = StandardUniform;
    let indices = (0..npts).filter(|_| {
        let p: f64 = distribution.sample(rng);
        p < sampling_rate
    });

    // Calculate centroid from sampled vectors
    let mut centroid: Vec<f32> = vec![0.0f32; dim];
    let mut vectors_processed = 0;
    let mut centroid_initialized = false;

    sample_reader.read_vectors(indices, |vector| {
        if !centroid_initialized {
            // Reinitialize centroid with the dimension from the first vector
            let full_dim = T::full_dimension(vector).into_ann_result()?;
            centroid = vec![0.0f32; full_dim];
            centroid_initialized = true;
        }

        let f32_vector = T::as_f32(vector).into_ann_result()?;
        for j in 0..f32_vector.len() {
            centroid[j] += f32_vector[j];
        }
        vectors_processed += 1;
        Ok(())
    })?;

    // If no vectors were processed, return error
    if !centroid_initialized {
        Err(ANNError::log_index_error(
            "Trying to compute centroid on zero vectors",
        ))
    } else {
        // Normalize centroid
        for value in centroid.iter_mut() {
            *value /= vectors_processed as f32;
        }
        Ok(centroid)
    }
}

/// Find the vector closest to a given centroid
///
/// Uses squared L2 (Euclidean) distance to determine the closest vector.
///
pub fn find_nearest_vector_with_id<T, Iter>(
    iter: Iter,
    centroid: &[f32],
) -> ANNResult<Option<(Box<[T]>, usize)>>
where
    T: VectorRepr,
    Iter: Iterator<Item = (Box<[T]>, ())>,
{
    let mut min_dist: f32 = f32::MAX;
    let mut nearest = None;
    let mut min_id = 0;

    for (id, (v, _)) in iter.enumerate() {
        let vf32 = T::as_f32(&v).into_ann_result()?;
        let dist = SquaredL2::evaluate(centroid, vf32.as_ref());
        if dist < min_dist {
            min_dist = dist;
            nearest = Some(v.clone());
            min_id = id;
        }
    }

    Ok(nearest.map(|v| (v, min_id)))
}

/// Find the vector closest to the centroid (medoid) from a dataset
///
/// # Distance Metric
/// Uses squared L2 (Euclidean) distance to determine the closest vector to the centroid.
///
/// # Performance Note
/// This function iterates through the entire file twice:
/// 1. First pass: Calculates the centroid (mean vector) of all vectors
/// 2. Second pass: Finds the vector closest to the centroid
///
/// For large files, this can have significant performance impact.
/// Consider using a sampling approach for very large datasets.
///
/// # Returns
/// Returns a tuple containing the medoid vector and its position (index) in the file.
pub fn find_medoid_from_file<T, Reader>(path: &str, reader: &Reader) -> ANNResult<(Vec<T>, usize)>
where
    T: VectorRepr,
    Reader: StorageReadProvider,
{
    let iter: VectorDataIterator<Reader, AdHoc<T>> =
        VectorDataIterator::<Reader, AdHoc<T>>::new(path, None, reader)?;
    let num_points = iter.get_num_points();

    let mut iter = iter.peekable();
    if let Some((x, _)) = iter.peek() {
        let full_dimension = T::full_dimension(x).into_ann_result()?;
        // Calculate centroid
        let centroid = calculate_centroid(iter, full_dimension, num_points)?;

        // Find medoid (point closest to centroid)
        let iter = VectorDataIterator::<Reader, AdHoc<T>>::new(path, None, reader)?;
        let (medoid, medoid_id) = find_nearest_vector_with_id(iter, &centroid)?
            .ok_or_else(|| ANNError::log_index_error("medoid not found"))?;

        Ok((medoid.to_vec(), medoid_id))
    } else {
        Err(ANNError::log_index_error(
            "Medoid not calculable on zero length iterator",
        ))
    }
}

/// Find the vector closest to the centroid (medoid) from a dataset using sampling approach
///
/// # Distance Metric
/// Uses squared L2 (Euclidean) distance to determine the closest vector to the centroid.
///
/// # Arguments
/// * `path` - Path to the binary vector file
/// * `reader` - Storage provider reader
/// * `max_sample_size` - Maximum number of vectors to sample. Actual sample size may be smaller.
///   If set to 0, all vectors will be used (equivalent to no sampling).
///   For optimal performance, consider using `MAX_MEDOID_SAMPLE_SIZE`.
/// * `rng` - Random number generator
///
/// # Returns
/// Returns a tuple containing the medoid vector and its position (index) in the file.
pub fn find_medoid_with_sampling<T, Reader>(
    path: &str,
    reader: &Reader,
    max_sample_size: usize,
    rng: &mut impl Rng,
) -> ANNResult<(Vec<T>, usize)>
where
    T: VectorRepr,
    Reader: StorageReadProvider,
{
    let metadata = load_metadata_from_file(reader, path)?;

    // Calculate sampling rate based on max_sample_size
    let sampling_rate = if max_sample_size == 0 || max_sample_size >= metadata.npoints {
        1.0 // Use all points
    } else {
        max_sample_size as f64 / metadata.npoints as f64
    };

    info!(
        "Finding medoid from {} points with max max_sample_size: {}, sampling_rate: {:.2}",
        metadata.npoints, max_sample_size, sampling_rate
    );

    let centroid = calculate_centroid_with_sampling::<T, _>(path, reader, sampling_rate, rng)?;

    // Find medoid (point closest to centroid) from the full dataset
    let iter = VectorDataIterator::<Reader, AdHoc<T>>::new(path, None, reader)?;
    let (medoid, medoid_id) = find_nearest_vector_with_id(iter, &centroid)?
        .ok_or_else(|| ANNError::log_index_error("medoid not found"))?;

    Ok((medoid.to_vec(), medoid_id))
}

#[cfg(test)]
mod tests {
    use std::{io::Write, num::NonZeroUsize};

    use crate::storage::VirtualStorageProvider;
    use crate::utils::write_metadata;
    use diskann::utils::VectorRepr;
    use diskann_quantization::{
        CompressInto,
        algorithms::{Transform, transforms::NullTransform},
        minmax::{DataMutRef, MinMaxQuantizer},
        num::Positive,
    };
    use diskann_utils::ReborrowMut;
    use rand::{SeedableRng, rngs::StdRng};
    use vfs::{FileSystem, MemoryFS};

    use super::*;
    use crate::common::MinMaxElement;

    /// Helper function to create a binary vector file in memory for testing
    fn create_test_vector_file<T: VectorRepr>(
        filesystem: &MemoryFS,
        path: &str,
        vectors: &[Vec<T>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = filesystem.create_file(path)?;

        // Write header: num_points (u32), dimension (u32)
        let num_points = vectors.len();
        let dimension = if vectors.is_empty() {
            0
        } else {
            vectors[0].len()
        };

        write_metadata(&mut file, num_points, dimension)?;

        // Write vectors
        for vector in vectors {
            let bytes = bytemuck::cast_slice(vector);
            file.write_all(bytes)?;
        }

        Ok(())
    }

    /// Helper function to create f32 test vectors
    fn create_f32_test_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 3.0, 4.0],
        ]
    }

    /// Helper function to create compressed MinMax test vectors
    fn create_minmax_test_vectors() -> Result<Vec<Vec<MinMaxElement<8>>>, Box<dyn std::error::Error>>
    {
        let f32_vectors = create_f32_test_vectors();
        let mut minmax_vectors = Vec::new();

        for vector in f32_vectors {
            let transform =
                Transform::Null(NullTransform::new(NonZeroUsize::new(vector.len()).unwrap()));
            let quantizer = MinMaxQuantizer::new(transform, Positive::new(1.0).unwrap());

            let mut bytes =
                vec![
                    0_u8;
                    diskann_quantization::minmax::DataRef::<8>::canonical_bytes(vector.len())
                ];
            let mut compressed =
                DataMutRef::<8>::from_canonical_front_mut(&mut bytes, vector.len()).unwrap();
            quantizer
                .compress_into(vector.as_slice(), compressed.reborrow_mut())
                .unwrap();

            let minmax_vector: Vec<MinMaxElement<8>> = bytemuck::cast_slice(&bytes).to_vec();
            minmax_vectors.push(minmax_vector);
        }

        Ok(minmax_vectors)
    }

    #[test]
    fn test_calculate_centroid_basic() {
        let vectors = vec![
            (vec![1.0f32, 2.0, 3.0].into_boxed_slice(), ()),
            (vec![4.0f32, 5.0, 6.0].into_boxed_slice(), ()),
            (vec![7.0f32, 8.0, 9.0].into_boxed_slice(), ()),
        ];

        let centroid = calculate_centroid(vectors.into_iter(), 3, 3).unwrap();

        // Expected centroid: [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3] = [4.0, 5.0, 6.0]
        assert_eq!(centroid, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_calculate_centroid_empty_iterator() {
        let vectors: Vec<(Box<[f32]>, ())> = vec![];
        let centroid = calculate_centroid(vectors.into_iter(), 3, 0).unwrap();

        // With 0 points, should return zeros
        assert_eq!(centroid, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_calculate_centroid_with_minmax_success() {
        let minmax_vectors = create_minmax_test_vectors().unwrap();
        let vectors: Vec<(Box<[MinMaxElement<8>]>, ())> = minmax_vectors
            .into_iter()
            .map(|v| (v.into_boxed_slice(), ()))
            .collect();

        let dimension = MinMaxElement::full_dimension(&vectors[0].0).unwrap();
        let centroid = calculate_centroid(vectors.into_iter(), dimension, 4).unwrap();

        // Should compute centroid successfully
        assert_eq!(centroid.len(), 3);
        // Approximate centroid should be close to [3.5, 4.5, 5.5]
        assert!((centroid[0] - 3.5).abs() < 1e-2);
        assert!((centroid[1] - 4.5).abs() < 1e-2);
        assert!((centroid[2] - 5.5).abs() < 1e-2);
    }

    #[test]
    fn test_find_nearest_vector_with_id_basic() {
        let vectors = vec![
            (vec![1.0f32, 2.0, 3.0].into_boxed_slice(), ()),
            (vec![4.0f32, 5.0, 6.0].into_boxed_slice(), ()),
            (vec![7.0f32, 8.0, 9.0].into_boxed_slice(), ()),
        ];

        let centroid = vec![4.5, 5.5, 6.5];
        let result = find_nearest_vector_with_id(vectors.into_iter(), &centroid).unwrap();

        assert!(result.is_some());
        let (nearest_vector, nearest_id) = result.unwrap();
        // Vector [4.0, 5.0, 6.0] should be closest to [4.5, 5.5, 6.5]
        assert_eq!(nearest_id, 1);
        assert_eq!(nearest_vector.as_ref(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_find_nearest_vector_with_id_empty_iterator() {
        let vectors: Vec<(Box<[f32]>, ())> = vec![];
        let centroid = vec![1.0, 2.0, 3.0];
        let result = find_nearest_vector_with_id(vectors.into_iter(), &centroid).unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn test_find_nearest_vector_with_minmax_success() {
        let minmax_vectors = create_minmax_test_vectors().unwrap();
        let vectors: Vec<(Box<[MinMaxElement<8>]>, ())> = minmax_vectors
            .into_iter()
            .map(|v| (v.into_boxed_slice(), ()))
            .collect();

        let centroid = vec![3.5, 4.5, 5.5]; // Should be closest to one of the vectors
        let result = find_nearest_vector_with_id(vectors.into_iter(), &centroid).unwrap();

        assert!(result.is_some());
        let (_, nearest_id) = result.unwrap();
        // Should find a valid vector (any of the 4 vectors is acceptable)
        assert!(nearest_id == 1);
    }

    #[test]
    fn test_find_medoid_from_file_basic() {
        let filesystem = MemoryFS::new();
        let vectors = create_f32_test_vectors();
        create_test_vector_file(&filesystem, "/test_vectors.bin", &vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let result = find_medoid_from_file::<f32, _>("/test_vectors.bin", &storage_provider);

        assert!(result.is_ok());
        let (medoid, medoid_id) = result.unwrap();
        assert_eq!(medoid.len(), 3);
        assert!(medoid_id == 1); // Should be valid index
    }

    #[test]
    fn test_find_medoid_from_file_empty_file() {
        let filesystem = MemoryFS::new();
        let vectors: Vec<Vec<f32>> = vec![]; // Empty vector list
        create_test_vector_file(&filesystem, "/empty_vectors.bin", &vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let result = find_medoid_from_file::<f32, _>("/empty_vectors.bin", &storage_provider);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("zero length iterator")
        );
    }

    #[test]
    fn test_find_medoid_with_sampling_basic() {
        let filesystem = MemoryFS::new();
        let vectors = create_f32_test_vectors();
        create_test_vector_file(&filesystem, "/test_vectors.bin", &vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let mut rng = StdRng::seed_from_u64(12345);
        let result = find_medoid_with_sampling::<f32, _>(
            "/test_vectors.bin",
            &storage_provider,
            2, // Sample only 2 vectors
            &mut rng,
        );

        assert!(result.is_ok());
        let (medoid, medoid_id) = result.unwrap();
        assert_eq!(medoid.len(), 3);
        assert!(medoid_id < 4);
    }

    #[test]
    fn test_find_medoid_with_sampling_no_sampling() {
        let filesystem = MemoryFS::new();
        let vectors = create_f32_test_vectors();
        create_test_vector_file(&filesystem, "/test_vectors.bin", &vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let mut rng = StdRng::seed_from_u64(12345);
        let result = find_medoid_with_sampling::<f32, _>(
            "/test_vectors.bin",
            &storage_provider,
            0, // No sampling - use all vectors
            &mut rng,
        );

        assert!(result.is_ok());
        let (medoid, medoid_id) = result.unwrap();
        assert_eq!(medoid.len(), 3);
        assert!(medoid_id == 1);
    }

    #[test]
    fn test_calculate_centroid_with_sampling_empty_vectors() {
        let filesystem = MemoryFS::new();
        let vectors: Vec<Vec<f32>> = vec![];
        create_test_vector_file(&filesystem, "/empty_vectors.bin", &vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let mut rng = StdRng::seed_from_u64(12345);

        let result = calculate_centroid_with_sampling::<f32, _>(
            "/empty_vectors.bin",
            &storage_provider,
            1.0, // 100% sampling rate
            &mut rng,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("zero vectors"));
    }

    #[test]
    fn test_calculate_centroid_with_sampling_basic() {
        let filesystem = MemoryFS::new();
        let vectors = create_f32_test_vectors();
        create_test_vector_file(&filesystem, "/test_vectors.bin", &vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let mut rng = StdRng::seed_from_u64(12345);

        let result = calculate_centroid_with_sampling::<f32, _>(
            "/test_vectors.bin",
            &storage_provider,
            1.0, // 100% sampling rate
            &mut rng,
        );

        assert!(result.is_ok());
        let centroid = result.unwrap();
        assert_eq!(centroid.len(), 3);
        // Centroid should be reasonable values
        assert!(centroid.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_find_medoid_with_minmax_vectors() {
        let filesystem = MemoryFS::new();
        let minmax_vectors = create_minmax_test_vectors().unwrap();
        create_test_vector_file(&filesystem, "/minmax_vectors.bin", &minmax_vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let result =
            find_medoid_from_file::<MinMaxElement<8>, _>("/minmax_vectors.bin", &storage_provider);

        assert!(result.is_ok());
        let (medoid, medoid_id) = result.unwrap();
        assert!(!medoid.is_empty());
        assert_eq!(medoid_id, 1);
    }

    #[test]
    fn test_calculate_centroid_with_sampling_zero_sampling_rate() {
        let filesystem = MemoryFS::new();
        let vectors = create_f32_test_vectors();
        create_test_vector_file(&filesystem, "/test_vectors.bin", &vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let mut rng = StdRng::seed_from_u64(12345);

        // Use very low sampling rate that might result in no vectors being sampled
        let result = calculate_centroid_with_sampling::<f32, _>(
            "/test_vectors.bin",
            &storage_provider,
            0.001, // Very low sampling rate
            &mut rng,
        );

        // This might succeed or fail depending on random sampling
        // If it fails, it should be due to zero vectors
        if result.is_err() {
            assert!(result.unwrap_err().to_string().contains("zero vectors"));
        }
    }

    #[test]
    fn test_error_handling_with_corrupted_minmax_data() {
        // Create corrupted MinMax data (too short)
        let corrupted_vectors = vec![
            vec![MinMaxElement::<8>::default(); 2], // Too short to be valid MinMax data
        ];

        let filesystem = MemoryFS::new();
        create_test_vector_file(&filesystem, "/corrupted.bin", &corrupted_vectors).unwrap();

        let storage_provider = VirtualStorageProvider::new(filesystem);
        let result =
            find_medoid_from_file::<MinMaxElement<8>, _>("/corrupted.bin", &storage_provider);

        // Should fail due to invalid MinMax data
        assert!(result.is_err());
    }
}
