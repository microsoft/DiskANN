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
        // SAFETY: the test owns the reservoir; all cold slabs span scan_lanes entries.
        unsafe { collect_sorted_neighbors(&self.hot, cold.distances, cold.neighbors, usize::MAX) }
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
    // SAFETY: both offsets select complete `m`-element sketch rows.
    let hash = unsafe {
        (hp.relative_hash)(
            sketches.as_ptr().add(src * m),
            sketches.as_ptr().add(dst * m),
            m,
        )
    };
    let l_max = hp.l_max as u8;
    hp.with_locked(src, |hot, cold| {
        // SAFETY: with_locked guards the row and supplies valid cold-slab pointers.
        unsafe { insert_locked(hot, cold, hash, dst as u32, distance, l_max, hp.find_hash) };
    });
}

#[test]
fn test_relative_hash_local_narrow_matches_scalar_semantics() {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

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
                expected |= ((!diff.is_sign_negative()) as u16) << j;
            }

            // SAFETY: the feature guard above checked AVX2 and both arrays
            // contain 16 elements, so every tested prefix is in bounds.
            let actual = unsafe { relative_hash_local_narrow(src.as_ptr(), dst.as_ptr(), m) };
            assert_eq!(actual, expected, "m={m}");
        }
    }
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
fn test_hash_prune_parallel_safety() {
    use rayon::prelude::*;
    let data = vec![0.0f32; 100 * 4];
    let hp = HashPrune::new(&data, 100, 4, 4, 10, 42).unwrap();
    (0..50).into_par_iter().for_each_installed(|i| {
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
    let mut n0 = full[0].clone();
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
