/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

struct Reservoir {
    hot: HotSlot,
    hashes: Vec<u16>,
    distances: Vec<u16>,
    neighbors: Vec<u32>,
    scan_lanes: usize,
    l_max: u8,
}

impl Reservoir {
    fn new(l_max: usize) -> Self {
        assert!(l_max <= MAX_RESERVOIR_LEN);
        let scan_lanes = round_up_to_32(l_max).max(32);
        Self {
            hot: HotSlot::new_empty(),
            hashes: vec![0; scan_lanes],
            distances: vec![0; scan_lanes],
            neighbors: vec![0; scan_lanes],
            scan_lanes,
            l_max: l_max as u8,
        }
    }

    fn cold(&self) -> ColdSlotPtrs {
        ColdSlotPtrs {
            hashes: self.hashes.as_ptr() as *mut u16,
            distances: self.distances.as_ptr() as *mut u16,
            neighbors: self.neighbors.as_ptr() as *mut u32,
            scan_lanes: self.scan_lanes,
        }
    }

    fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
        let cold = self.cold();
        // SAFETY: the test owns the reservoir and holds its only mutable reference.
        unsafe {
            insert_locked(
                &mut self.hot,
                cold,
                hash,
                neighbor,
                distance,
                self.l_max,
                select_find_hash(),
            )
        }
    }

    fn neighbors(&self) -> Vec<(u32, f32)> {
        let cold = self.cold();
        let mut scratch = Vec::new();
        // SAFETY: the test owns the reservoir; all cold slabs span scan_lanes entries.
        unsafe {
            collect_sorted_neighbors(
                &self.hot,
                cold.distances,
                cold.neighbors,
                usize::MAX,
                &mut scratch,
            )
        }
    }

    fn len(&self) -> usize {
        self.hot.len as usize
    }

    fn is_empty(&self) -> bool {
        self.hot.len == 0
    }
}

fn add_edge(hp: &HashPrune, src: usize, dst: usize, distance: f32) {
    let m = hp.sketches.num_planes();
    let sketches = hp.sketches.sketches();
    let hash = hp.relative_hash.call(RelativeHashArgs {
        src: sketches[src * m..(src + 1) * m].as_ptr(),
        dst: sketches[dst * m..(dst + 1) * m].as_ptr(),
        len: m,
    });
    let l_max = hp.l_max as u8;
    hp.with_locked(src, |hot, cold| {
        // SAFETY: with_locked guards the row and supplies valid cold-slab pointers.
        unsafe { insert_locked(hot, cold, hash, dst as u32, distance, l_max, hp.find_hash) };
    });
}

fn assert_sketch_source_type_matches_f32<T>(
    label: &str,
    convert: impl Fn(u8) -> T,
    reference: impl Fn(u8) -> f32,
) where
    T: VectorRepr + Send + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let points = 5;
    for dimensions in [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        let raw: Vec<u8> = (0..points * dimensions)
            .map(|index| ((index * 7 + index / dimensions * 3) % 23) as u8)
            .collect();
        let converted: Vec<T> = raw.iter().copied().map(&convert).collect();
        let f32_data: Vec<f32> = raw.iter().copied().map(&reference).collect();
        for planes in [1, 8, 16] {
            let (actual, expected) = pool.install(|| {
                (
                    sketches_from_data(&converted, points, dimensions, planes, 42).unwrap(),
                    sketches_from_data(&f32_data, points, dimensions, planes, 42).unwrap(),
                )
            });
            assert_eq!(
                actual.sketches(),
                expected.sketches(),
                "{label} dimensions={dimensions} planes={planes}"
            );
        }
    }
}

#[test]
fn test_f16_sketch_conversion_matches_f32_across_dimensions_and_planes() {
    assert_sketch_source_type_matches_f32(
        "f16",
        |value| half::f16::from_f32(value as f32),
        |value| value as f32,
    );
}

#[test]
fn test_u8_sketch_conversion_matches_f32_across_dimensions_and_planes() {
    assert_sketch_source_type_matches_f32("u8", |value| value, |value| value as f32);
}

#[test]
fn test_i8_sketch_conversion_matches_f32_across_dimensions_and_planes() {
    assert_sketch_source_type_matches_f32(
        "i8",
        |value| value as i8 - 11,
        |value| (value as i8 - 11) as f32,
    );
}

#[test]
fn test_relative_hash_matches_numeric_reference() {
    let dispatched = select_relative_hash();

    let src = [
        1.0, -2.0, 0.0, 7.5, -0.0, 3.25, -9.0, 4.0, 8.0, -1.5, 2.0, 0.0, 6.0, -3.0, 5.5, -7.25,
    ];
    let dst = [
        1.0, -3.0, 0.5, 7.0, 0.0, 3.25, -8.0, -4.0, 9.0, -1.5, -2.0, -0.0, 5.0, -2.0, 5.5, -8.0,
    ];

    for m in 0..=16 {
        let mut expected = 0u16;
        for j in 0..m {
            let diff: f32 = dst[j] - src[j];
            expected |= ((diff >= 0.0) as u16) << j;
        }

        let actual = dispatched.call(RelativeHashArgs {
            src: src.as_ptr(),
            dst: dst.as_ptr(),
            len: m,
        });
        assert_eq!(actual, expected, "m={m}");
    }
}

#[test]
fn test_relative_hash_defines_signed_zero_and_nan_buckets() {
    let src = [0.0; 4];
    let dst = [
        0.0,
        -0.0,
        f32::from_bits(0x7FC0_0000),
        f32::from_bits(0xFFC0_0000),
    ];

    assert_eq!(
        select_relative_hash().call(RelativeHashArgs {
            src: src.as_ptr(),
            dst: dst.as_ptr(),
            len: dst.len(),
        }),
        0b0011
    );
}

#[test]
fn test_find_hash_handles_padded_boundaries_and_all_bit_patterns() {
    let dispatched = select_find_hash();

    for target in [0, 0xF00D] {
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 254, 255] {
            let scan_lanes = round_up_to_32(len.max(1));
            let mut hashes = vec![target; scan_lanes];
            hashes[..len].fill(0x8001);
            let args = |hashes: &[u16]| FindHashArgs {
                hashes: hashes.as_ptr(),
                scan_lanes,
                len: len as u8,
                target,
            };

            assert_eq!(dispatched.call(args(&hashes)), None, "len={len}");
            for index in [0, len / 2, len.saturating_sub(1)] {
                if index < len {
                    hashes[index] = target;
                    assert_eq!(dispatched.call(args(&hashes)), Some(index), "len={len}");
                    hashes[index] = 0x8001;
                }
            }
        }
    }
}

#[test]
fn test_slab_is_zeroed_and_reports_its_bytes() {
    let slab = MmapSlab::<u32>::new_zeroed(4).unwrap();
    assert_eq!(slab.bytes(), 4 * std::mem::size_of::<u32>());
    assert_eq!(slab.len(), 4);
    assert!(!slab.as_ptr().is_null());
    assert_eq!(&*slab, &[0; 4]);
}

#[test]
fn test_round_up_to_32_boundaries() {
    assert_eq!(round_up_to_32(0), 0);
    assert_eq!(round_up_to_32(1), 32);
    assert_eq!(round_up_to_32(32), 32);
    assert_eq!(round_up_to_32(33), 64);
}

#[test]
fn test_hash_prune_accepts_structural_l_max_boundaries() {
    let data = [0.0_f32];
    let low = HashPrune::new(&data, 1, 1, 1, 1, 42).unwrap();
    assert_eq!(low.l_max, 1);
    assert_eq!(low.scan_lanes, 32);

    let high = HashPrune::new(&data, 1, 1, 1, MAX_RESERVOIR_LEN, 42).unwrap();
    assert_eq!(high.l_max, MAX_RESERVOIR_LEN);
    assert_eq!(high.scan_lanes, 256);
}

#[test]
fn test_hash_prune_rejects_l_max_outside_structural_boundaries() {
    for l_max in [0, MAX_RESERVOIR_LEN + 1] {
        let result = HashPrune::new(&[0.0_f32], 1, 1, 1, l_max, 42);
        let error = match result {
            Ok(_) => panic!("l_max={l_max} must be rejected"),
            Err(error) => error,
        };
        assert!(format!("{error:?}").contains(&format!("l_max ({l_max})")));
    }
}

#[test]
fn test_ordered_key_roundtrips_bf16_order_for_all_signs() {
    let values = [
        f32::NEG_INFINITY,
        -100.0,
        -0.0,
        0.0,
        0.25,
        100.0,
        f32::INFINITY,
    ];
    let keys: Vec<_> = values.iter().copied().map(ordered_key).collect();
    assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
    for (value, key) in values.into_iter().zip(keys) {
        assert_eq!(
            bf16_to_f32(key_to_bf16(key)),
            bf16_to_f32(f32_to_bf16(value))
        );
    }
}

#[test]
fn test_add_leaf_edges_matches_single_edge_reference() {
    let data = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let batched = HashPrune::new(&data, 4, 2, 8, 8, 42).unwrap();
    let reference = HashPrune::new(&data, 4, 2, 8, 8, 42).unwrap();
    let point_ids = [0, 1, 2, 3];
    let offsets = [0, 3, 6, 9, 12];
    let edges = [
        (1, 1.0),
        (2, 1.0),
        (3, 2.0),
        (0, 1.0),
        (2, 2.0),
        (3, 1.0),
        (0, 1.0),
        (1, 2.0),
        (3, 1.0),
        (0, 2.0),
        (1, 1.0),
        (2, 1.0),
    ];
    let mut scratch = Vec::new();

    batched.add_leaf_edges(&point_ids, &offsets, &edges, &mut scratch);
    let scratch_len = scratch.len();
    batched.add_leaf_edges(&point_ids[..2], &[0, 0, 0], &[], &mut scratch);
    assert_eq!(scratch.len(), scratch_len);

    for source in 0..point_ids.len() {
        for &(target, distance) in &edges[offsets[source] as usize..offsets[source + 1] as usize] {
            add_edge(&reference, source, target as usize, distance);
        }
    }
    let canonicalize = |rows: Vec<diskann::graph::AdjacencyList<u32>>| {
        rows.into_iter()
            .map(|row| {
                let mut ids = row.to_vec();
                ids.sort_unstable();
                ids
            })
            .collect::<Vec<_>>()
    };
    let actual = canonicalize(batched.into_candidate_lists());
    let expected = canonicalize(reference.into_candidate_lists());

    assert_eq!(actual, expected);
    assert!(actual.iter().all(|row| !row.is_empty()));
}

#[test]
fn test_reservoir_basic() {
    let mut reservoir = Reservoir::new(3);
    assert!(reservoir.is_empty());

    assert!(reservoir.insert(0, 1, 1.0));
    assert!(reservoir.insert(1, 2, 2.0));
    assert!(reservoir.insert(2, 3, 3.0));
    assert_eq!(reservoir.len(), 3);

    assert!(reservoir.insert(3, 4, 0.5));
    assert_eq!(reservoir.len(), 3);

    let neighbors = reservoir.neighbors();
    assert!(!neighbors.iter().any(|(id, _)| *id == 3));
    assert!(neighbors.iter().any(|(id, _)| *id == 4));
}

#[test]
fn test_reservoir_same_hash_keeps_closer() {
    let mut reservoir = Reservoir::new(10);

    assert!(reservoir.insert(0, 1, 2.0));
    assert_eq!(reservoir.len(), 1);

    assert!(reservoir.insert(0, 2, 1.0));
    assert_eq!(reservoir.len(), 1);

    let neighbors = reservoir.neighbors();
    assert_eq!(neighbors[0].0, 2);
    assert_eq!(neighbors[0].1, 1.0);

    assert!(!reservoir.insert(0, 3, 5.0));
    assert_eq!(reservoir.len(), 1);
}

#[test]
fn test_hash_prune_end_to_end() {
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let hp = HashPrune::new(&data, 4, 2, 4, 10, 42).unwrap();

    add_edge(&hp, 0, 1, 1.0);
    add_edge(&hp, 0, 2, 1.0);
    add_edge(&hp, 0, 3, 1.414);
    add_edge(&hp, 1, 0, 1.0);
    add_edge(&hp, 1, 3, 1.0);
    add_edge(&hp, 2, 0, 1.0);
    add_edge(&hp, 2, 3, 1.0);
    add_edge(&hp, 3, 1, 1.0);
    add_edge(&hp, 3, 2, 1.0);

    let graph = hp.into_nearest_lists(3);
    assert_eq!(graph.len(), 4);

    for (i, neighbors) in graph.iter().enumerate() {
        assert!(!neighbors.is_empty(), "point {} has no neighbors", i);
    }
}

#[test]
fn test_reservoir_lazy_allocation() {
    let mut res = Reservoir::new(5);
    assert!(res.is_empty());
    assert!(res.insert(0, 1, 1.0));
    assert_eq!(res.len(), 1);
}

#[test]
fn test_reservoir_insert_then_evict_cycle() {
    let mut res = Reservoir::new(3);
    res.insert(0, 10, 3.0);
    res.insert(1, 11, 2.0);
    res.insert(2, 12, 1.0);
    assert_eq!(res.len(), 3);
    assert!(res.insert(3, 13, 0.5));
    assert_eq!(res.len(), 3);
    let neighbors = res.neighbors();
    assert!(neighbors.iter().all(|&(_, d)| d <= 2.0));
}

#[test]
fn test_reservoir_all_same_hash() {
    let mut res = Reservoir::new(5);
    res.insert(0, 1, 3.0);
    res.insert(0, 2, 2.0);
    res.insert(0, 3, 1.0);
    assert_eq!(res.len(), 1);
    let neighbors = res.neighbors();
    assert_eq!(neighbors[0].0, 3);
    assert_eq!(neighbors[0].1, 1.0);
}

#[test]
fn test_reservoir_all_same_distance() {
    let mut res = Reservoir::new(5);
    res.insert(0, 1, 1.0);
    res.insert(1, 2, 1.0);
    res.insert(2, 3, 1.0);
    assert_eq!(res.len(), 3);
}

#[test]
#[allow(clippy::disallowed_methods)]
fn test_hash_prune_parallel_safety() {
    use rayon::prelude::*;
    let data = vec![0.0f32; 100 * 4];
    let hp = HashPrune::new(&data, 100, 4, 4, 10, 42).unwrap();
    (0..50).into_par_iter().for_each(|i| {
        add_edge(&hp, i, (i + 1) % 100, 1.0);
        add_edge(&hp, (i + 1) % 100, i, 1.0);
    });
    let graph = hp.into_nearest_lists(5);
    assert_eq!(graph.len(), 100);
}

#[test]
fn test_hash_prune_high_degree_limit() {
    let data = vec![0.0f32; 10 * 2];
    let hp = HashPrune::new(&data, 10, 2, 4, 10, 42).unwrap();
    for i in 0..10 {
        for j in 0..10 {
            if i != j {
                add_edge(&hp, i, j, (i as f32 - j as f32).abs());
            }
        }
    }
    let graph = hp.into_nearest_lists(1);
    for neighbors in &graph {
        assert!(
            neighbors.len() <= 1,
            "max_degree=1 should limit to 1 neighbor"
        );
    }
}

#[test]
fn test_hash_prune_extract_sorted() {
    let data = vec![0.0f32; 4 * 2];
    let hp = HashPrune::new(&data, 4, 2, 4, 10, 42).unwrap();
    add_edge(&hp, 0, 1, 3.0);
    add_edge(&hp, 0, 2, 1.0);
    add_edge(&hp, 0, 3, 2.0);
    let graph = hp.into_nearest_lists(3);
    assert!(!graph[0].is_empty());
}

#[test]
fn test_into_candidate_lists_returns_full_reservoir() {
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let hp = HashPrune::new(&data, 4, 2, 4, 10, 42).unwrap();
    add_edge(&hp, 0, 1, 1.0);
    add_edge(&hp, 0, 2, 1.0);
    add_edge(&hp, 0, 3, 1.414);
    add_edge(&hp, 1, 0, 1.0);
    add_edge(&hp, 2, 0, 1.0);
    add_edge(&hp, 3, 0, 1.414);

    let full = hp.into_candidate_lists();
    assert_eq!(full.len(), 4);
    assert!(!full[0].is_empty(), "node 0 should have neighbors");
    // ids-only, unsorted: every id is one of node 0's inserted neighbors
    // {1,2,3} with no duplicates (the LSH bucket may keep-closer-collapse a
    // colliding pair on this tiny 4-plane sketch, so we don't assert all 3).
    let mut n0 = full[0].to_vec();
    n0.sort_unstable();
    let deduped = {
        let mut d = n0.clone();
        d.dedup();
        d
    };
    assert_eq!(n0, deduped, "no duplicate ids in a reservoir row");
    assert!(
        n0.iter().all(|&id| (1..=3).contains(&id)),
        "node 0 ids must be a subset of its inserted neighbors {{1,2,3}}, got {:?}",
        n0
    );
}

#[test]
fn test_into_nearest_lists_truncates_to_max_degree() {
    let data = vec![0.0f32; 4 * 2];
    let hp = HashPrune::new(&data, 4, 2, 4, 10, 42).unwrap();
    add_edge(&hp, 0, 1, 1.0);
    add_edge(&hp, 0, 2, 2.0);
    add_edge(&hp, 0, 3, 3.0);

    let graph = hp.into_nearest_lists(2);
    assert!(
        graph[0].len() <= 2,
        "bounded graph extraction should truncate to max_degree"
    );
}

#[test]
fn test_reservoir_farthest_cache_after_eviction() {
    let mut res = Reservoir::new(3);
    res.insert(0, 10, 5.0);
    res.insert(1, 11, 4.0);
    res.insert(2, 12, 3.0);
    assert!(res.insert(3, 13, 2.0));
    assert!(res.insert(4, 14, 1.0));
    let neighbors = res.neighbors();
    assert_eq!(neighbors.len(), 3);
    for &(_, d) in &neighbors {
        assert!(d <= 3.1, "expected dist <= 3.0, got {}", d);
    }
}

#[test]
fn test_reservoir_farthest_insert_before_farthest_idx() {
    let mut res = Reservoir::new(4);
    res.insert(5, 1, 1.0);
    res.insert(10, 2, 3.0);
    res.insert(15, 3, 2.0);
    res.insert(3, 4, 0.5);
    let neighbors = res.neighbors();
    assert_eq!(neighbors.len(), 4);
    assert_eq!(neighbors[0].0, 4);
}
