/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, marker::PhantomData};

use diskann::{utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::{
    model::MAX_PQ_TRAINING_SET_SIZE,
    storage::{StorageReadProvider, StorageWriteProvider},
    utils::{
        create_rnd_provider_from_optional_seed, gen_random_slice, BridgeErr, RayonThreadPoolRef,
    },
};
use diskann_quantization::{
    algorithms::transforms::{TargetDim, TransformKind},
    alloc::{GlobalAllocator, Poly, ScopedAllocator},
    spherical::{
        iface::{self, Quantizer},
        PreScale, SphericalQuantizer, SupportedMetric, UnsupportedMetric,
    },
};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::distance::Metric;

use super::{
    validate_data_generation_input, QuantCompressor, QuantDataGenerator, SearchQuantization,
};

pub(super) fn supported_metric(metric: Metric) -> Result<SupportedMetric, UnsupportedMetric> {
    match metric {
        Metric::CosineNormalized => Metric::Cosine,
        metric => metric,
    }
    .try_into()
}

struct SphericalGenerationContext<'a, Storage> {
    data_path: &'a str,
    quantizer_path: &'a str,
    storage: &'a Storage,
    metric: Metric,
    seed: Option<u64>,
    num_points: usize,
    source_dim: u32,
}

struct SphericalGeneration<'a, T, Storage> {
    plan: Box<dyn Quantizer>,
    phantom: PhantomData<(&'a Storage, T)>,
}

impl<'a, T, Storage> QuantCompressor<T> for SphericalGeneration<'a, T, Storage>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider + 'a,
{
    type CompressorContext = SphericalGenerationContext<'a, Storage>;

    fn new(context: &Self::CompressorContext) -> ANNResult<Self> {
        let metric = supported_metric(context.metric).bridge_err()?;
        let mut rng = create_rnd_provider_from_optional_seed(context.seed).create_rnd();
        let (training, count, dim) = gen_random_slice::<T, _>(
            context.data_path,
            MAX_PQ_TRAINING_SET_SIZE / context.num_points as f64,
            context.storage,
            &mut rng,
        )?;
        let transform = TransformKind::DoubleHadamard {
            target_dim: TargetDim::Natural,
        };
        let quantizer = if count > 0
            && metric != SupportedMetric::Cosine
            && training.iter().all(|&value| value == 0.0)
        {
            // Zero vectors can be encoded, but training requires a positive mean norm.
            // Unit scale and a zero centroid also work if later vectors are nonzero.
            SphericalQuantizer::generate(
                Poly::broadcast(0.0, dim, GlobalAllocator).map_err(ANNError::new)?,
                1.0,
                transform,
                metric,
                None,
                &mut rng,
                GlobalAllocator,
            )
        } else {
            SphericalQuantizer::train(
                rowmajor::Ref::try_from_data(&training, count, dim).bridge_err()?,
                transform,
                metric,
                PreScale::ReciprocalMeanNorm,
                &mut rng,
                GlobalAllocator,
            )
        }
        .map_err(ANNError::new)?;
        let plan: Box<dyn Quantizer> =
            Box::new(iface::Impl::<3>::new(quantizer).map_err(ANNError::new)?);
        let serialized = plan.serialize(GlobalAllocator).map_err(ANNError::new)?;
        let mut writer = context.storage.create_for_write(context.quantizer_path)?;
        writer.write_all(b"DSQ3")?;
        writer.write_all(&context.source_dim.to_le_bytes())?;
        writer.write_all(&serialized)?;
        writer.flush()?;
        Ok(Self {
            plan,
            phantom: PhantomData,
        })
    }

    fn compress(
        &self,
        vectors: rowmajor::Ref<f32>,
        mut output: rowmajor::Mut<u8>,
    ) -> ANNResult<()> {
        for index in 0..vectors.nrows() {
            self.plan
                .compress(
                    vectors.row(index),
                    iface::OpaqueMut::new(output.row_mut(index)),
                    ScopedAllocator::global(),
                )
                .map_err(ANNError::new)?;
        }
        Ok(())
    }

    fn compressed_bytes(&self) -> usize {
        self.plan.bytes()
    }
}

/// Generate Q3 search sidecars without rebuilding or changing the disk graph.
///
/// The dataset must use the same row order as the disk graph's internal IDs.
/// Existing Q3 sidecars are replaced. PQ sidecars are left untouched.
///
/// # Errors
/// Returns an error for invalid input, an unsupported metric, training or compression
/// failure, or storage errors. A failed generation may leave incomplete Q3 sidecars.
pub fn generate_spherical_q3<T, Storage>(
    prefix: &str,
    data_path: &str,
    metric: Metric,
    seed: Option<u64>,
    storage: &Storage,
    pool: RayonThreadPoolRef<'_>,
    max_block_size: usize,
) -> ANNResult<()>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider,
{
    let metadata = validate_data_generation_input(storage, data_path, max_block_size)?;
    if metadata.ndims() == 0 {
        return Err(ANNError::message(
            "Cannot generate spherical data with zero dimensions",
        ));
    }
    let [quantizer_path, codes_path] = SearchQuantization::SphericalQ3.paths(prefix);
    let context = SphericalGenerationContext {
        data_path,
        quantizer_path: &quantizer_path,
        storage,
        metric,
        seed,
        num_points: metadata.npoints(),
        source_dim: metadata.ndims_u32(),
    };
    QuantDataGenerator::<T, SphericalGeneration<'_, T, Storage>>::new(
        data_path.into(),
        codes_path,
        &context,
    )?
    .generate_data(storage, pool, max_block_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        search::quantization::QuantizedScratch,
        storage::quant::{SearchData, SearchDataInner},
    };
    use diskann_providers::{storage::VirtualStorageProvider, utils::create_thread_pool_for_test};
    use diskann_utils::io::write_bin;
    use diskann_vector::PreprocessedDistanceFunction;
    use rstest::rstest;

    fn dataset(storage: &impl StorageWriteProvider) -> rowmajor::Owned<f32> {
        let values = (0..37 * 13)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 7.0)
            .collect::<Vec<_>>();
        let data = rowmajor::Owned::try_from_data(values.into_boxed_slice(), 37, 13).unwrap();
        let mut writer = storage.create_for_write("/data.bin").unwrap();
        write_bin(data.as_view(), &mut writer).unwrap();
        writer.flush().unwrap();
        data
    }

    #[rstest]
    #[case(Metric::L2)]
    #[case(Metric::InnerProduct)]
    #[case(Metric::Cosine)]
    #[case(Metric::CosineNormalized)]
    fn spherical_q3_roundtrip_matches_inmem_distance(#[case] metric: Metric) {
        let storage = VirtualStorageProvider::new_memory();
        let vectors = dataset(&storage);
        let pool = create_thread_pool_for_test();
        generate_spherical_q3::<f32, _>(
            "/index",
            "/data.bin",
            metric,
            Some(7),
            &storage,
            pool.as_ref(),
            8,
        )
        .unwrap();
        let data = SearchData::load("/index", SearchQuantization::SphericalQ3, &storage).unwrap();
        data.validate(13, 37, metric).unwrap();
        assert!(data.validate(12, 37, metric).is_err());
        assert!(data.validate(13, 36, metric).is_err());
        let wrong_metric = if metric == Metric::L2 {
            Metric::InnerProduct
        } else {
            Metric::L2
        };
        assert!(data.validate(13, 37, wrong_metric).is_err());
        let SearchDataInner::SphericalQ3 { plan, .. } = &data.0 else {
            panic!("wrong backend")
        };
        let serialized = plan.serialize(GlobalAllocator).unwrap();
        let reference: Box<dyn Quantizer> =
            Box::new(iface::Impl::<3>::try_deserialize(&serialized, GlobalAllocator).unwrap());
        let mut scratch = QuantizedScratch::new(&data, 7).unwrap();
        assert!(scratch.distances(&data, &[0], |_, _| {}).is_err());
        for query_index in [0, 11, 0] {
            let query = vectors.row(query_index);
            scratch.prepare_query(&data, metric, query).unwrap();
            let QuantizedScratch::SphericalQ3 { computer, .. } = &scratch else {
                panic!("wrong scratch backend")
            };
            assert_eq!(
                computer.as_ref().unwrap().layout(),
                iface::QueryLayout::FullPrecision
            );
            let expected = reference
                .fused_query_computer(
                    query,
                    iface::QueryLayout::FullPrecision,
                    true,
                    GlobalAllocator,
                    ScopedAllocator::global(),
                )
                .unwrap();
            let ids = (0..37).collect::<Vec<u32>>();
            let mut actual = Vec::new();
            scratch
                .distances(&data, &ids, |distance, id| actual.push((id, distance)))
                .unwrap();
            for (id, distance) in actual {
                let mut code = vec![0; plan.bytes()];
                reference
                    .compress(
                        vectors.row(id as usize),
                        iface::OpaqueMut::new(&mut code),
                        ScopedAllocator::global(),
                    )
                    .unwrap();
                let expected_distance = expected
                    .evaluate_similarity(iface::Opaque::new(&code))
                    .unwrap();
                assert_eq!(distance, expected_distance);
            }
        }
        assert!(scratch.prepare_query(&data, metric, &[0.0; 12]).is_err());
        assert!(scratch.distances(&data, &[0], |_, _| {}).is_err());
        scratch
            .prepare_query(&data, metric, vectors.row(0))
            .unwrap();
        assert!(scratch.distances(&data, &[37], |_, _| {}).is_err());
        assert!(SearchData::load("/missing", SearchQuantization::SphericalQ3, &storage).is_err());
    }

    #[rstest]
    #[case(Metric::L2)]
    #[case(Metric::InnerProduct)]
    fn spherical_q3_zero_norm_roundtrip(#[case] metric: Metric) {
        let storage = VirtualStorageProvider::new_memory();
        let vectors = rowmajor::Owned::from_element(37, 13, 0.0f32);
        {
            let mut writer = storage.create_for_write("/zeros.bin").unwrap();
            write_bin(vectors.as_view(), &mut writer).unwrap();
            writer.flush().unwrap();
        }
        let pool = create_thread_pool_for_test();
        generate_spherical_q3::<f32, _>(
            "/index",
            "/zeros.bin",
            metric,
            Some(7),
            &storage,
            pool.as_ref(),
            8,
        )
        .unwrap();
        let data = SearchData::load("/index", SearchQuantization::SphericalQ3, &storage).unwrap();
        data.validate(13, 37, metric).unwrap();
        let mut scratch = QuantizedScratch::new(&data, 37).unwrap();
        for query in [[0.0; 13], [1.0; 13]] {
            scratch.prepare_query(&data, metric, &query).unwrap();
            let expected = if metric == Metric::L2 {
                query.iter().map(|value| value * value).sum::<f32>()
            } else {
                0.0
            };
            let mut count = 0;
            scratch
                .distances(&data, &(0..37).collect::<Vec<_>>(), |distance, _| {
                    assert!(distance.is_finite());
                    assert!((distance - expected).abs() < 0.02);
                    count += 1;
                })
                .unwrap();
            assert_eq!(count, 37);
        }
    }

    #[test]
    fn spherical_q3_zero_dimensions_preserve_sidecars() {
        use std::io::Read;

        fn check<T: VectorRepr>() {
            let storage = VirtualStorageProvider::new_memory();
            {
                let mut writer = storage.create_for_write("/zero-dim.bin").unwrap();
                diskann_utils::io::Metadata::new(37, 0)
                    .unwrap()
                    .write(&mut writer)
                    .unwrap();
                writer.flush().unwrap();
            }
            let paths = [SearchQuantization::PQ, SearchQuantization::SphericalQ3]
                .into_iter()
                .flat_map(|quantization| quantization.paths("/index"))
                .collect::<Vec<_>>();
            for path in &paths {
                let mut writer = storage.create_for_write(path).unwrap();
                writer.write_all(b"existing sidecar").unwrap();
                writer.flush().unwrap();
            }
            let pool = create_thread_pool_for_test();
            let error = generate_spherical_q3::<T, _>(
                "/index",
                "/zero-dim.bin",
                Metric::L2,
                Some(7),
                &storage,
                pool.as_ref(),
                8,
            )
            .unwrap_err();
            assert!(error.to_string().contains("zero dimensions"));
            for path in &paths {
                let mut bytes = Vec::new();
                storage
                    .open_reader(path)
                    .unwrap()
                    .read_to_end(&mut bytes)
                    .unwrap();
                assert_eq!(bytes, b"existing sidecar");
            }
        }
        check::<f32>();
        check::<i8>();
        check::<u8>();
    }

    #[rstest]
    #[case(f32::NAN)]
    #[case(f32::INFINITY)]
    #[case(1.0e-40)]
    fn spherical_q3_invalid_norm_is_not_treated_as_zero(#[case] value: f32) {
        let storage = VirtualStorageProvider::new_memory();
        let vectors = rowmajor::Owned::from_element(37, 13, value);
        {
            let mut writer = storage.create_for_write("/data.bin").unwrap();
            write_bin(vectors.as_view(), &mut writer).unwrap();
            writer.flush().unwrap();
        }
        let pool = create_thread_pool_for_test();
        assert!(generate_spherical_q3::<f32, _>(
            "/index",
            "/data.bin",
            Metric::L2,
            Some(7),
            &storage,
            pool.as_ref(),
            8,
        )
        .is_err());
        for path in SearchQuantization::SphericalQ3.paths("/index") {
            assert!(!storage.exists(&path));
        }
    }

    #[test]
    fn spherical_q3_generation_is_batch_independent_and_preserves_pq() {
        use std::io::Read;

        let storage = VirtualStorageProvider::new_memory();
        dataset(&storage);
        let pool = create_thread_pool_for_test();
        let pq_paths = SearchQuantization::PQ.paths("/index");
        for path in &pq_paths {
            let mut writer = storage.create_for_write(path).unwrap();
            writer.write_all(b"existing PQ").unwrap();
            writer.flush().unwrap();
        }
        let read = |path: &str| {
            let mut bytes = Vec::new();
            storage
                .open_reader(path)
                .unwrap()
                .read_to_end(&mut bytes)
                .unwrap();
            bytes
        };
        generate_spherical_q3::<f32, _>(
            "/index",
            "/data.bin",
            Metric::L2,
            Some(9),
            &storage,
            pool.as_ref(),
            8,
        )
        .unwrap();
        let first = SearchQuantization::SphericalQ3
            .paths("/index")
            .map(|path| read(&path));
        generate_spherical_q3::<f32, _>(
            "/index",
            "/data.bin",
            Metric::L2,
            Some(9),
            &storage,
            pool.as_ref(),
            37,
        )
        .unwrap();
        let second = SearchQuantization::SphericalQ3
            .paths("/index")
            .map(|path| read(&path));
        assert_eq!(first, second);
        for path in &pq_paths {
            assert_eq!(read(path), b"existing PQ");
        }
    }

    #[test]
    fn spherical_q3_rejects_corrupted_sidecars() {
        let storage = VirtualStorageProvider::new_memory();
        dataset(&storage);
        let pool = create_thread_pool_for_test();
        assert!(generate_spherical_q3::<f32, _>(
            "/index",
            "/data.bin",
            Metric::L2,
            Some(7),
            &storage,
            pool.as_ref(),
            0
        )
        .is_err());
        generate_spherical_q3::<f32, _>(
            "/index",
            "/data.bin",
            Metric::L2,
            Some(7),
            &storage,
            pool.as_ref(),
            8,
        )
        .unwrap();
        let [quantizer_path, codes_path] = SearchQuantization::SphericalQ3.paths("/index");
        let data = SearchData::load("/index", SearchQuantization::SphericalQ3, &storage).unwrap();
        let SearchDataInner::SphericalQ3 { plan, .. } = &data.0 else {
            panic!("wrong backend")
        };
        let valid_quantizer = plan.serialize(GlobalAllocator).unwrap();
        let mut wrong_shape = Vec::new();
        diskann_utils::io::Metadata::new(37, plan.bytes() + 1)
            .unwrap()
            .write(&mut wrong_shape)
            .unwrap();
        wrong_shape.resize(8 + 37 * plan.bytes(), 0);
        let mut truncated = Vec::new();
        diskann_utils::io::Metadata::new(37, plan.bytes())
            .unwrap()
            .write(&mut truncated)
            .unwrap();
        let mut trailing = truncated.clone();
        truncated.resize(8 + 37 * plan.bytes() - 1, 0);
        trailing.resize(8 + 37 * plan.bytes() + 1, 0);
        for bytes in [vec![0; 8], vec![0; 15], wrong_shape, truncated, trailing] {
            let mut writer = storage.create_for_write(&codes_path).unwrap();
            writer.write_all(&bytes).unwrap();
            writer.flush().unwrap();
            drop(writer);
            assert!(SearchData::load("/index", SearchQuantization::SphericalQ3, &storage).is_err());
        }
        let mut invalid_schema = b"DSQ3".to_vec();
        invalid_schema.extend_from_slice(&13u32.to_le_bytes());
        invalid_schema.extend_from_slice(&[0; 16]);
        let mut zero_dim = b"DSQ3".to_vec();
        zero_dim.extend_from_slice(&0u32.to_le_bytes());
        zero_dim.extend_from_slice(&valid_quantizer);
        for bytes in [vec![], vec![0; 16], invalid_schema, zero_dim] {
            let mut writer = storage.create_for_write(&quantizer_path).unwrap();
            writer.write_all(&bytes).unwrap();
            writer.flush().unwrap();
            drop(writer);
            assert!(SearchData::load("/index", SearchQuantization::SphericalQ3, &storage).is_err());
        }
    }
}
