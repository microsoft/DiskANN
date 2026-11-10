/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Version 1 stores counts, dimension, maximum degree, and metric in the record.
//! Artifact scalars are little-endian: external IDs are `u32`, vectors are row-major
//! `f32`, and each graph row contains a `u32` degree followed by that many `u32` targets.
//! Vector and graph rows include frozen points first; external IDs include only live nodes.

use diskann_record::{Version, load};

use super::Provider;
use crate::{
    counters::Counters,
    ids::IdMap,
    num::Capacity,
    repr::{Full, Representation, full},
    store,
};

pub(super) const VERSION: Version = Version::new(1, 0);

/// Runtime settings for loading a [`super::Provider`] with `f32` full-precision vectors
/// and `u32` external IDs.
///
/// Frozen points precede live nodes in saved artifacts. Increasing capacity only adds
/// empty writable slots and does not change saved graph IDs. Prefetch settings use
/// [`Full`]'s defaults.
#[derive(Debug, Default)]
pub struct FullProviderLoader {
    capacity: Option<Capacity>,
    store: store::Config,
}

impl FullProviderLoader {
    /// Create a loader whose writable capacity matches the saved live-node count.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override writable capacity, which must be at least the saved live-node count.
    pub fn capacity(mut self, capacity: Capacity) -> Self {
        self.capacity = Some(capacity);
        self
    }

    /// Override the store's concurrency settings.
    pub fn store(mut self, config: store::Config) -> Self {
        self.store = config;
        self
    }

    fn load_object(self, object: load::Object<'_>) -> load::Result<Provider<Full<f32>, u32>> {
        let live_count: u32 = object.field("live_count")?;
        let frozen_count: u32 = object.field("frozen_count")?;
        let mut external_ids = full::load::artifact(
            &object,
            "external_ids",
            u64::from(live_count) * size_of::<u32>() as u64,
        )?;
        let representation = Full::load(&object, self.capacity, self.store)?;
        let mapping = IdMap::new(representation.id_limit());
        for offset in 0..live_count {
            let external = full::load::read_u32(&mut external_ids)?;
            mapping
                .insert(external, frozen_count + offset)
                .map_err(load::Error::new)?;
        }
        Ok(Provider {
            representation,
            mapping,
            counters: Counters::new(),
        })
    }
}

impl<'a> load::Loader<'a> for FullProviderLoader {
    type Value = Provider<Full<f32>, u32>;

    fn load(self, context: load::Context<'a>) -> load::Result<Self::Value> {
        let object = context.as_object().ok_or(load::error::Kind::TypeMismatch)?;
        if object.version() != VERSION {
            return Err(load::error::Kind::UnknownVersion.into());
        }
        self.load_object(object).map_err(load::Error::new)
    }
}

impl<'a> load::Loadable<'a> for Provider<Full<f32>, u32> {
    fn load(context: load::Context<'a>) -> load::Result<Self> {
        load::Loader::load(FullProviderLoader::default(), context)
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write, path::Path};

    use diskann::{
        graph::AdjacencyList,
        provider::{Delete, SetElement},
    };
    use diskann_record::save;
    use diskann_vector::distance::Metric;

    use super::*;
    use crate::{
        num::{IdLimit, MaxDegree},
        repr::{Insert, Search},
    };

    struct Fixture {
        version: Version,
        live_count: u32,
        frozen_count: u32,
        dim: usize,
        max_degree: u32,
        metric: &'static str,
        vectors: Vec<u8>,
        graph: Vec<u8>,
        external_ids: Vec<u8>,
    }

    impl Default for Fixture {
        fn default() -> Self {
            Self {
                version: VERSION,
                live_count: 2,
                frozen_count: 1,
                dim: 2,
                max_degree: 2,
                metric: "l2",
                vectors: [0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
                graph: [1u32, 1, 2, 0, 2, 1, 1]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
                external_ids: [10u32, 20].into_iter().flat_map(u32::to_le_bytes).collect(),
            }
        }
    }

    impl save::Saveable for Fixture {
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Value<'_>> {
            let mut record = diskann_record::save_fields!(
                self,
                context,
                [live_count, frozen_count, dim, max_degree,]
            );
            record.insert("metric", save::Value::String(self.metric.into()))?;
            for (key, bytes) in [
                ("vectors", &self.vectors),
                ("graph", &self.graph),
                ("external_ids", &self.external_ids),
            ] {
                let mut writer = context.write(Some(key))?;
                writer.write_all(bytes).map_err(save::Error::new)?;
                record.insert(key, writer.finish()?)?;
            }
            Ok(record.into_value(self.version))
        }
    }

    impl Fixture {
        fn write(&self, dir: &Path) -> std::path::PathBuf {
            let metadata = dir.join("metadata.json");
            save::save_to_disk(self, dir, &metadata).unwrap();
            metadata
        }

        fn load(&self, loader: FullProviderLoader) -> load::Result<Provider<Full<f32>, u32>> {
            let dir = tempfile::tempdir().unwrap();
            let metadata = self.write(dir.path());
            load::load_from_disk_with(&metadata, dir.path(), loader)
        }
    }

    fn round_trip(provider: &Provider<Full<f32>, u32>) -> Provider<Full<f32>, u32> {
        let dir = tempfile::tempdir().unwrap();
        let metadata = dir.path().join("metadata.json");
        save::save_to_disk(provider, dir.path(), &metadata).unwrap();
        load::load_from_disk(&metadata, dir.path()).unwrap()
    }

    #[test]
    fn restores_vectors_graph_and_mapping_at_different_capacities() {
        let fixture = Fixture::default();
        for capacity in [None, Some(Capacity::new(2)), Some(Capacity::new(5))] {
            let loader = FullProviderLoader {
                capacity,
                ..FullProviderLoader::default()
            };
            let provider = fixture.load(loader).unwrap();
            let representation = &provider.representation;
            let capacity = capacity.unwrap_or(Capacity::new(2));
            assert_eq!(representation.capacity(), capacity);
            assert_eq!(
                representation.id_limit(),
                IdLimit::new(1 + capacity.value() as u32)
            );
            assert_eq!(representation.dim(), 2);
            assert_eq!(representation.metric(), Metric::L2);
            assert_eq!(provider.max_degree(), MaxDegree::new(2));

            let accessor = representation
                .prune_accessor(provider.local_counters())
                .unwrap();
            let mut neighbors = AdjacencyList::new();
            for (id, (vector, expected)) in [
                ([0.0, 0.0], vec![1]),
                ([1.0, 2.0], vec![0, 2]),
                ([3.0, 4.0], vec![1]),
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(&*representation.get(id as u32).unwrap(), &vector);
                assert_eq!(representation.is_readable(id as u32), Some(true));
                accessor.neighbors.get(id as u32, &mut neighbors).unwrap();
                assert_eq!(&neighbors[..], expected.as_slice());
            }
            assert_eq!(provider.mapping.to_internal(&10), Some(1));
            assert_eq!(provider.mapping.to_internal(&20), Some(2));
            assert_eq!(provider.mapping.to_external(0), None);
            assert_eq!(provider.mapping.to_external(1), Some(10));
            assert_eq!(provider.mapping.to_external(2), Some(20));
            assert!(representation.retire(0).is_err());
            for id in 3..representation.id_limit().value() {
                assert_eq!(representation.is_readable(id), Some(false));
                assert_eq!(provider.mapping.to_external(id), None);
                accessor.neighbors.get(id, &mut neighbors).unwrap();
                assert!(neighbors.is_empty());
            }
        }
    }

    #[test]
    fn ordinary_loading_uses_default_loader() {
        let dir = tempfile::tempdir().unwrap();
        let metadata = Fixture::default().write(dir.path());
        let provider: Provider<Full<f32>, u32> =
            load::load_from_disk(&metadata, dir.path()).unwrap();
        assert_eq!(provider.representation.capacity(), Capacity::new(2));
        assert_eq!(provider.mapping.to_internal(&20), Some(2));
    }

    #[test]
    fn save_round_trips_through_existing_loader() {
        let provider = Fixture::default()
            .load(FullProviderLoader::new().capacity(Capacity::new(5)))
            .unwrap();
        let restored = round_trip(&provider);

        assert_eq!(restored.representation.capacity(), Capacity::new(2));
        assert_eq!(restored.mapping.to_internal(&10), Some(1));
        assert_eq!(restored.mapping.to_internal(&20), Some(2));
        for id in 0..3 {
            assert_eq!(
                restored.representation.get(id).unwrap(),
                provider.representation.get(id).unwrap(),
            );
        }
        assert!(
            provider
                .representation
                .prune_accessor(provider.local_counters())
                .is_ok()
        );
    }

    #[test]
    fn save_rejects_conflicting_writer_exclusion() {
        let provider = Fixture::default()
            .load(FullProviderLoader::default())
            .unwrap();
        let exclusion = provider.representation.try_exclude_writers().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let metadata = dir.path().join("metadata.json");
        let error = save::save_to_disk(&provider, dir.path(), &metadata).unwrap_err();
        assert!(error.to_string().contains("could not exclude writers"));
        assert!(!metadata.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        drop(exclusion);
        round_trip(&provider);
    }

    #[test]
    fn save_waits_for_writers_and_allows_readers() {
        let provider = Fixture::default()
            .load(FullProviderLoader::new().capacity(Capacity::new(4)))
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let metadata = dir.path().join("metadata.json");
        std::thread::scope(|scope| {
            let reader = provider
                .representation
                .search_accessor(&[0.0, 0.0], &provider, provider.local_counters())
                .unwrap();
            let writer = provider
                .representation
                .prune_accessor(provider.local_counters())
                .unwrap();
            let saving = scope.spawn(|| save::save_to_disk(&provider, dir.path(), &metadata));
            let started = std::time::Instant::now();
            loop {
                if provider
                    .representation
                    .prune_accessor(provider.local_counters())
                    .is_err()
                {
                    break;
                }
                assert!(!saving.is_finished(), "save did not wait for the writer");
                assert!(
                    started.elapsed() < std::time::Duration::from_secs(10),
                    "save did not exclude new writers"
                );
                std::thread::yield_now();
            }

            assert!(!saving.is_finished());
            assert_eq!(reader.get_expand_beam().evaluate(0).unwrap(), Some(0.0));
            assert_eq!(&*provider.representation.get(1).unwrap(), &[1.0, 2.0]);
            assert!(provider.representation.retire(1).is_err());
            writer.neighbors.set(0, &[2]).unwrap();
            drop(writer);
            saving.join().unwrap().unwrap();
            assert_eq!(reader.get_expand_beam().evaluate(0).unwrap(), Some(0.0));
        });

        let restored: Provider<Full<f32>, u32> =
            load::load_from_disk(&metadata, dir.path()).unwrap();
        let writer = restored
            .representation
            .prune_accessor(restored.local_counters())
            .unwrap();
        let mut neighbors = AdjacencyList::new();
        writer.neighbors.get(0, &mut neighbors).unwrap();
        assert_eq!(&neighbors[..], &[2]);
        assert!(
            provider
                .representation
                .prune_accessor(provider.local_counters())
                .is_ok()
        );
    }

    #[tokio::test]
    async fn save_compacts_holes_and_streams_renamed_edges() {
        let provider = Fixture::default()
            .load(FullProviderLoader::new().capacity(Capacity::new(6)))
            .unwrap();
        let context = super::super::Context;
        for (external, vector) in [(30, [5.0, 6.0]), (40, [7.0, 8.0]), (50, [9.0, 10.0])] {
            provider
                .set_element(&context, &external, &vector[..])
                .await
                .unwrap();
        }
        provider.delete(&context, &10).await.unwrap();
        provider.delete(&context, &40).await.unwrap();
        let source_graph = [(0, [4, 5]), (2, [4, 5]), (3, [5, 2]), (5, [3, 0])];
        {
            let accessor = provider
                .representation
                .prune_accessor(provider.local_counters())
                .unwrap();
            for (id, neighbors) in source_graph {
                accessor.neighbors.set(id, &neighbors).unwrap();
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let metadata = dir.path().join("metadata.json");
        save::save_to_disk(&provider, dir.path(), &metadata).unwrap();
        for capacity in [None, Some(Capacity::new(8))] {
            let restored = load::load_from_disk_with(
                &metadata,
                dir.path(),
                FullProviderLoader {
                    capacity,
                    ..FullProviderLoader::default()
                },
            )
            .unwrap();
            let representation = &restored.representation;
            assert_eq!(
                representation.capacity(),
                capacity.unwrap_or(Capacity::new(3))
            );
            assert_eq!(restored.mapping.to_external(0), None);
            for (external, internal) in [(20, 1), (30, 2), (50, 3)] {
                assert_eq!(restored.mapping.to_internal(&external), Some(internal));
            }
            assert_eq!(restored.mapping.to_internal(&10), None);
            assert_eq!(restored.mapping.to_internal(&40), None);

            let accessor = representation
                .prune_accessor(restored.local_counters())
                .unwrap();
            let mut neighbors = AdjacencyList::new();
            for (id, (vector, expected)) in [
                ([0.0, 0.0], vec![3]),
                ([3.0, 4.0], vec![3]),
                ([5.0, 6.0], vec![3, 1]),
                ([9.0, 10.0], vec![2, 0]),
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(&*representation.get(id as u32).unwrap(), &vector);
                accessor.neighbors.get(id as u32, &mut neighbors).unwrap();
                assert_eq!(&neighbors[..], expected.as_slice());
            }
            assert!(representation.retire(0).is_err());
        }

        for (external, internal) in [(20, 2), (30, 3), (50, 5)] {
            assert_eq!(provider.mapping.to_internal(&external), Some(internal));
        }
        assert_eq!(provider.representation.capacity(), Capacity::new(6));
        let accessor = provider
            .representation
            .prune_accessor(provider.local_counters())
            .unwrap();
        let mut neighbors = AdjacencyList::new();
        for (id, expected) in source_graph {
            accessor.neighbors.get(id, &mut neighbors).unwrap();
            assert_eq!(&neighbors[..], &expected);
        }
    }

    #[test]
    fn save_rejects_inconsistent_slot_mapping_and_graph() {
        type FullProvider = Provider<Full<f32>, u32>;
        let corruptions: &[fn(&FullProvider)] = &[
            |provider| provider.mapping.occupied_entry(10).unwrap().delete(),
            |provider| provider.mapping.insert(30, 3).unwrap(),
            |provider| provider.mapping.insert(30, 0).unwrap(),
            |provider| {
                let accessor = provider
                    .representation
                    .prune_accessor(provider.local_counters())
                    .unwrap();
                accessor
                    .neighbors
                    .set(0, &[provider.representation.id_limit().value()])
                    .unwrap();
            },
        ];
        for corrupt in corruptions {
            let provider = Fixture::default()
                .load(FullProviderLoader::new().capacity(Capacity::new(4)))
                .unwrap();
            corrupt(&provider);
            let dir = tempfile::tempdir().unwrap();
            let metadata = dir.path().join("metadata.json");
            assert!(save::save_to_disk(&provider, dir.path(), &metadata).is_err());
            assert!(!metadata.exists());
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
            assert!(
                provider
                    .representation
                    .prune_accessor(provider.local_counters())
                    .is_ok()
            );
        }
    }

    #[tokio::test]
    async fn save_omits_all_deleted_live_nodes_and_their_edges() {
        let provider = Fixture::default()
            .load(FullProviderLoader::default())
            .unwrap();
        let context = super::super::Context;
        for external in [10, 20] {
            provider.delete(&context, &external).await.unwrap();
        }
        let restored = round_trip(&provider);
        assert_eq!(restored.representation.capacity(), Capacity::new(0));
        assert_eq!(restored.representation.id_limit(), IdLimit::new(1));
        assert_eq!(restored.mapping.len(), 0);
        assert_eq!(&*restored.representation.get(0).unwrap(), &[0.0, 0.0]);
        let accessor = restored
            .representation
            .prune_accessor(restored.local_counters())
            .unwrap();
        let mut neighbors = AdjacencyList::new();
        accessor.neighbors.get(0, &mut neighbors).unwrap();
        assert!(neighbors.is_empty());
    }

    #[tokio::test]
    async fn restored_provider_supports_insert_and_delete() {
        let mut store = store::Config::new();
        store.epoch_guard_slots(std::num::NonZeroUsize::new(1).unwrap());
        let provider = Fixture::default()
            .load(
                FullProviderLoader::new()
                    .capacity(Capacity::new(3))
                    .store(store),
            )
            .unwrap();
        let context = super::super::Context;
        provider
            .set_element(&context, &30, &[5.0f32, 6.0][..])
            .await
            .unwrap();
        assert_eq!(provider.mapping.to_internal(&30), Some(3));
        assert_eq!(&*provider.representation.get(3).unwrap(), &[5.0, 6.0]);
        provider.delete(&context, &10).await.unwrap();
        assert_eq!(provider.mapping.to_internal(&10), None);
        assert_eq!(provider.representation.is_readable(1), Some(false));
        assert_eq!(provider.mapping.to_internal(&20), Some(2));
    }

    #[test]
    fn rejects_invalid_runtime_capacity() {
        for capacity in [1, usize::MAX, u32::MAX as usize] {
            let err = Fixture::default()
                .load(FullProviderLoader::new().capacity(Capacity::new(capacity)))
                .unwrap_err();
            assert!(!err.is_recoverable());
        }
    }

    #[test]
    fn rejects_corrupt_metadata_and_artifacts() {
        let corruptions: &[fn(&mut Fixture)] = &[
            |fixture| fixture.version = Version::new(2, 0),
            |fixture| fixture.metric = "invalid",
            |fixture| fixture.dim = usize::MAX,
            |fixture| fixture.frozen_count = u32::MAX,
            |fixture| fixture.live_count = u32::MAX,
            |fixture| fixture.vectors.truncate(fixture.vectors.len() - 1),
            |fixture| fixture.vectors.push(0),
            |fixture| fixture.external_ids.truncate(4),
            |fixture| fixture.external_ids.extend_from_slice(&0u32.to_le_bytes()),
            |fixture| fixture.external_ids[4..].copy_from_slice(&10u32.to_le_bytes()),
            |fixture| fixture.graph.truncate(4),
            |fixture| fixture.graph.pop().map(|_| ()).unwrap(),
            |fixture| fixture.graph.extend_from_slice(&0u32.to_le_bytes()),
            |fixture| fixture.graph[..4].copy_from_slice(&3u32.to_le_bytes()),
            |fixture| fixture.graph[4..8].copy_from_slice(&3u32.to_le_bytes()),
        ];
        for (index, corrupt) in corruptions.iter().enumerate() {
            let mut fixture = Fixture::default();
            corrupt(&mut fixture);
            let err = fixture
                .load(FullProviderLoader::new().capacity(Capacity::new(8)))
                .unwrap_err();
            assert!(!err.is_recoverable(), "corruption {}: {}", index, err);
        }
    }

    #[test]
    fn restores_empty_provider() {
        let fixture = Fixture {
            live_count: 0,
            frozen_count: 0,
            vectors: Vec::new(),
            graph: Vec::new(),
            external_ids: Vec::new(),
            ..Fixture::default()
        };
        for capacity in [0, 2] {
            let provider = fixture
                .load(FullProviderLoader::new().capacity(Capacity::new(capacity)))
                .unwrap();
            assert_eq!(provider.representation.capacity(), Capacity::new(capacity));
            assert_eq!(
                provider.representation.id_limit(),
                IdLimit::new(capacity as u32)
            );
            let restored = round_trip(&provider);
            assert_eq!(restored.representation.capacity(), Capacity::new(0));
            assert_eq!(restored.representation.id_limit(), IdLimit::new(0));
        }
    }

    #[test]
    fn restores_frozen_only_and_no_frozen_providers() {
        for frozen_count in [0, 2, 3] {
            let live_count = 3 - frozen_count;
            let fixture = Fixture {
                live_count,
                frozen_count,
                external_ids: (0..live_count).flat_map(u32::to_le_bytes).collect(),
                ..Fixture::default()
            };
            let provider = round_trip(&fixture.load(FullProviderLoader::default()).unwrap());
            let representation = &provider.representation;
            let accessor = representation
                .search_accessor(&[0.0, 0.0], &provider, provider.local_counters())
                .unwrap();
            assert_eq!(accessor.start_points.start(), 0);
            assert_eq!(accessor.start_points.end(), frozen_count);
            for id in 0..frozen_count {
                assert_eq!(provider.mapping.to_external(id), None);
                assert!(representation.retire(id).is_err());
            }
            for offset in 0..live_count {
                assert_eq!(
                    provider.mapping.to_internal(&offset),
                    Some(frozen_count + offset)
                );
            }
            for (id, distance) in [0.0, 5.0, 25.0].into_iter().enumerate() {
                assert_eq!(
                    accessor.expand_beam.evaluate(id as u32).unwrap(),
                    Some(distance)
                );
            }
        }
    }

    #[test]
    fn preserves_metric_and_vector_bits() {
        let bits = [0x80000000u32, 0x7fc01234, 0x7f800000, 0xff800000, 1, 0];
        for metric in [
            Metric::L2,
            Metric::InnerProduct,
            Metric::Cosine,
            Metric::CosineNormalized,
        ] {
            let fixture = Fixture {
                metric: metric.as_str(),
                vectors: bits.into_iter().flat_map(u32::to_le_bytes).collect(),
                ..Fixture::default()
            };
            let provider = round_trip(&fixture.load(FullProviderLoader::default()).unwrap());
            assert_eq!(provider.representation.metric(), metric);
            let restored: Vec<_> = (0..3)
                .flat_map(|id| provider.representation.get(id).unwrap().into_vec())
                .map(f32::to_bits)
                .collect();
            assert_eq!(restored, bits);
        }
    }

    #[test]
    fn rejects_allocation_layout_overflow() {
        let fixture = Fixture {
            live_count: 0,
            frozen_count: 0,
            dim: isize::MAX as usize / size_of::<f32>(),
            vectors: Vec::new(),
            graph: Vec::new(),
            external_ids: Vec::new(),
            ..Fixture::default()
        };
        let err = fixture
            .load(FullProviderLoader::new().capacity(Capacity::new(2)))
            .unwrap_err();
        assert!(!err.is_recoverable());
    }

    #[test]
    #[ignore = "requires DISKANN_LOAD_MANIFEST pointing to a saved provider"]
    fn benchmark_load_saved_provider() {
        let metadata = std::env::var("DISKANN_LOAD_MANIFEST").unwrap();
        let metadata = Path::new(&metadata);
        let dir = metadata.parent().unwrap();
        let runs: usize = std::env::var("DISKANN_LOAD_RUNS")
            .unwrap_or_else(|_| "3".to_owned())
            .parse()
            .unwrap();
        assert!(runs > 0);
        let mut times = Vec::new();
        for run in 1..=runs {
            let started = std::time::Instant::now();
            let provider: Provider<Full<f32>, u32> = load::load_from_disk(metadata, dir).unwrap();
            let elapsed = started.elapsed();
            std::hint::black_box(&provider);
            println!("load {}: {:.3}s", run, elapsed.as_secs_f64());
            times.push(elapsed);
        }
        times.sort_unstable();
        println!("median: {:.3}s", times[times.len() / 2].as_secs_f64());
    }
}
