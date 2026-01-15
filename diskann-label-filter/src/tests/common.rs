/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Common test utilities and helper functions shared across multiple test modules

use crate::{
    attribute::{Attribute, AttributeValue},
    encoded_attribute_provider::roaring_attribute_store::RoaringAttributeStore,
    traits::attribute_store::AttributeStore,
};

use diskann_utils::future::AsyncFriendly;
use rand::{rng, Rng};
use roaring::RoaringTreemap;
use std::{
    collections::HashMap,
    sync::{Arc, Barrier},
    thread::{self, JoinHandle},
};

pub(crate) type TestIdType = u32;

/// Helper function to create a new RoaringAttributeStore for testing
pub(crate) fn create_test_store() -> RoaringAttributeStore<TestIdType> {
    RoaringAttributeStore::new()
}

/// Helper function to create test attributes
pub(crate) fn create_test_attributes(prefix: &str, num_attributes: usize) -> Vec<Attribute> {
    (0..num_attributes)
        .map(|i| {
            Attribute::from_value(
                format!("{}_field_{}", prefix, i),
                AttributeValue::String(format!("{}_value_{}", prefix, i)),
            )
        })
        .collect()
}

/// Creates a test dataset with random attributes
pub(crate) fn create_dataset(
    vec_count: u32,
    max_attrs_per_vec: u32,
    uniq_attr_count: u32,
) -> HashMap<u32, Vec<Attribute>> {
    let data: HashMap<TestIdType, Vec<Attribute>> = (0..vec_count)
        .map(|vec_id| {
            let attr_count = rng().random_range(1..max_attrs_per_vec);
            let mut attrs: Vec<Attribute> = Vec::with_capacity(attr_count as usize);
            for _i in 0..attr_count {
                let attr = Attribute::from_value(
                    format!("field_{}", rng().random_range(0..uniq_attr_count)),
                    AttributeValue::Integer(rng().random_range(1..20000)), //some random integer value.
                );
                attrs.push(attr);
            }
            (vec_id, attrs)
        })
        .collect();
    data
}

/// Creates write threads for concurrent testing
#[cfg(test)]
pub(crate) fn create_write_threads(
    store: Arc<RoaringAttributeStore<TestIdType>>,
    dataset: Arc<HashMap<TestIdType, Vec<Attribute>>>,
    added_points: Arc<scc::hash_set::HashSet<TestIdType>>,
    wt_count: usize,
    barrier: Arc<Barrier>,
) -> Vec<JoinHandle<()>> {
    if wt_count == 0 {
        return Vec::new();
    }

    // Decide on number of threads
    let num_threads = wt_count;

    eprintln!("Creating {} WRITE threads.", wt_count);
    // Convert the dataset keys to a vec for easier division
    let all_vector_ids: Vec<TestIdType> = dataset.keys().copied().collect();
    let chunk_size = all_vector_ids.len().div_ceil(num_threads);

    // Create threads
    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        // Get this thread's portion of vector IDs
        let start = thread_id * chunk_size;
        let end = std::cmp::min((thread_id + 1) * chunk_size, all_vector_ids.len());
        let thread_vectors: Vec<TestIdType> = all_vector_ids[start..end].to_vec();

        // Clone the dataset for this thread
        let thread_dataset = dataset.clone();
        let thread_store = store.clone();
        let thread_added_points = added_points.clone();
        let b = barrier.clone();

        // Spawn the thread
        let handle = thread::spawn(move || {
            b.wait(); //wait until all read/write threads are ready to start executing.
            for &vector_id in &thread_vectors {
                if let Some(attributes) = thread_dataset.get(&vector_id) {
                    let result = thread_store.set_element(&vector_id, attributes);
                    match result {
                        Ok(_) => (),
                        Err(e) => {
                            panic!("*** Error *** Thread id: W{}, failed to insert vector {} because {:#}", thread_id, vector_id, e.to_string());
                        }
                    }
                    let _ = thread_added_points.insert_sync(vector_id);
                }
            }
            eprintln!("Thread id: W{}, WRITER crossed finish line", thread_id);
        });

        handles.push(handle);
    }

    handles
}

/// Creates read threads for concurrent testing
#[cfg(test)]
pub(crate) fn create_read_threads<F>(
    store: Arc<RoaringAttributeStore<TestIdType>>,
    dataset: Arc<HashMap<TestIdType, Vec<Attribute>>>,
    rt_count: usize,
    barrier: Arc<Barrier>,
    func: F,
) -> Vec<JoinHandle<()>>
where
    F: Fn(TestIdType, &Arc<RoaringAttributeStore<TestIdType>>) -> Option<RoaringTreemap>
        + AsyncFriendly
        + Clone,
{
    if rt_count == 0 {
        return Vec::new();
    }

    let num_threads = rt_count;
    let all_vector_ids: Vec<TestIdType> = dataset.keys().copied().collect();
    let chunk_size = all_vector_ids.len().div_ceil(num_threads);

    let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(num_threads);

    eprintln!("Creating {} READ threads.", rt_count);

    for thread_id in 0..num_threads {
        let start = thread_id * chunk_size;
        let end = std::cmp::min((thread_id + 1) * chunk_size, all_vector_ids.len());
        let mut thread_vectors: Vec<TestIdType> = all_vector_ids[start..end].to_vec();

        // Clone the dataset for this thread
        let thread_dataset = dataset.clone();
        let thread_store = store.clone();
        let b = barrier.clone();
        let thread_func = func.clone();

        let handle = thread::spawn(move || {
            use std::time::Instant;

            let mut loop_iterations = 0;
            b.wait(); //do not start until both read and write threads are ready.
            eprintln!("Thread id: R{}, READER crossed barrier", thread_id);
            let start = Instant::now();
            while !thread_vectors.is_empty() && loop_iterations < 100 {
                use std::time::Duration;

                for id in thread_vectors.clone() {
                    let attr_map = thread_store.clone().attribute_map();
                    let attr_map_guard = attr_map.read().unwrap();

                    let mapped_attrs = match thread_func(id, &thread_store) {
                        Some(set) => set,
                        None => {
                            continue;
                        }
                    };

                    thread_vectors.pop_if(|x| *x == id);

                    //check if the attributes from the dataset are present in the store.
                    let mut all_attrs_exist = true;
                    for attr in thread_dataset.get(&id).unwrap() {
                        let attr_id = match attr_map_guard.get(attr) {
                            Some(id) => id,
                            None => {
                                all_attrs_exist = false;
                                break;
                            }
                        };
                        if !mapped_attrs.contains(attr_id) {
                            all_attrs_exist = false;
                            break;
                        }
                    }
                    if !all_attrs_exist {
                        panic!("*** Error *** Thread id: R{}, Attributes of Vector id {} not found in the store!", thread_id, id);
                    }
                }
                thread::sleep(Duration::from_millis(10));
                loop_iterations += 1;
            }
            let duration = start.elapsed();
            eprintln!(
                "Thread id: R{}, READER crossed finish line after {:?}",
                thread_id, duration
            );
        });

        handles.push(handle);
    }

    handles
}

/// Creates delete threads for concurrent testing
#[cfg(test)]
pub(crate) fn create_delete_threads(
    store: Arc<RoaringAttributeStore<TestIdType>>,
    dataset: Arc<HashMap<TestIdType, Vec<Attribute>>>,
    dt_count: usize,
    barrier: Arc<Barrier>,
) -> Vec<JoinHandle<()>> {
    if dt_count == 0 {
        return Vec::new();
    }

    const NUM_ITERATIONS: i32 = 1000;

    let num_threads = dt_count;
    let all_vector_ids: Vec<TestIdType> = dataset.keys().copied().collect();
    let chunk_size = all_vector_ids.len().div_ceil(num_threads);

    let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let start = thread_id * chunk_size;
        let end = std::cmp::min((thread_id + 1) * chunk_size, all_vector_ids.len());
        let mut thread_vectors: Vec<TestIdType> = all_vector_ids[start..end].to_vec();

        // Clone the dataset for this thread
        let thread_store = store.clone();
        let b = barrier.clone();

        let handle = thread::spawn(move || {
            use std::time::Instant;

            let mut loop_iterations = 0;
            b.wait(); //do not start until both sets of threads are ready.
            let start = Instant::now();
            while !thread_vectors.is_empty() {
                use std::time::Duration;

                for id in thread_vectors.clone() {
                    match thread_store.delete(&id) {
                        Ok(result) => {
                            if result {
                                //deleted successfully, so remove from our list
                                eprintln!("Deleted {} from store:", id);
                                thread_vectors.pop_if(|x| *x == id);
                            }
                        }
                        Err(e) => {
                            panic!("{:#}", e);
                        }
                    };
                }

                if loop_iterations > NUM_ITERATIONS {
                    eprintln!("*** Warning *** Thread id: D{}, There are still {} undeleted items in the thread_vectors list", thread_id, thread_vectors.len());
                    break;
                }
                thread::sleep(Duration::from_millis(10));
                loop_iterations += 1;
            }
            let duration = start.elapsed();
            eprintln!(
                "Thread id: D{}, DELETE crossed finish line after {:?}",
                thread_id, duration
            );
        });

        handles.push(handle);
    }

    handles
}
