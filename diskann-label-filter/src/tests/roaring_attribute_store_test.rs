/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    attribute::{Attribute, AttributeValue},
    set::traits::SetProvider,
    tests::common::{
        create_dataset, create_delete_threads, create_read_threads, create_test_attributes,
        create_test_store, create_write_threads, TestIdType,
    },
    traits::attribute_store::AttributeStore,
};

use std::sync::{Arc, Barrier};

#[test]
fn test_basic_set_and_retrieve() {
    let store = create_test_store();
    let vector_id: TestIdType = 42;

    // Test that ID doesn't exist initially
    assert!(!store.id_exists(&vector_id).unwrap());

    // Create and set attributes
    let attributes = vec![
        Attribute::from_value(
            "category",
            AttributeValue::String("electronics".to_string()),
        ),
        Attribute::from_value("price", AttributeValue::Real(299.99)),
        Attribute::from_value("available", AttributeValue::Bool(true)),
    ];

    // Set attributes
    let result = store.set_element(&vector_id, &attributes);
    assert!(result.is_ok(), "Failed to set attributes: {:?}", result);

    // Verify ID now exists
    assert!(store.id_exists(&vector_id).unwrap());

    // Get the index and verify data
    let index_arc = store.get_index();
    let index_guard = index_arc.read().unwrap();
    let attr_set = index_guard.get(&vector_id);

    // Should have 3 attributes
    let set = attr_set.unwrap().unwrap();
    assert_eq!(set.len(), 3);
    assert!(!set.is_empty());
}

#[test]
fn test_update_attributes() {
    let store = create_test_store();
    let vector_id: TestIdType = 100;

    // Initial attributes
    let initial_attrs = vec![
        Attribute::from_value("status", AttributeValue::String("draft".to_string())),
        Attribute::from_value("priority", AttributeValue::Integer(1)),
    ];

    let result = store.set_element(&vector_id, &initial_attrs);
    assert!(result.is_ok());
    assert!(store.id_exists(&vector_id).unwrap());

    // Verify initial count
    let index_arc = store.get_index();
    {
        let index_guard = index_arc.read().unwrap();
        let attr_set = index_guard.get(&vector_id);
        let set = attr_set.unwrap().unwrap();
        assert_eq!(set.len(), 2);
    }

    // Update with new attributes (should replace all)
    let updated_attrs = vec![
        Attribute::from_value("status", AttributeValue::String("published".to_string())),
        Attribute::from_value("priority", AttributeValue::Integer(5)),
        Attribute::from_value("views", AttributeValue::Integer(1000)),
    ];

    let result = store.set_element(&vector_id, &updated_attrs);
    assert!(result.is_ok());

    // Verify updated count
    {
        let index_guard = index_arc.read().unwrap();
        let attr_set = index_guard.get(&vector_id);
        let set = attr_set.unwrap().unwrap();
        assert_eq!(set.len(), 3);
    }
}

#[test]
fn test_delete_operations() {
    let store = create_test_store();
    let vector_id: TestIdType = 200;

    // Set some attributes
    let attributes = create_test_attributes("delete_test", 3);
    let result = store.set_element(&vector_id, &attributes);
    assert!(result.is_ok());
    assert!(store.id_exists(&vector_id).unwrap());

    // Delete the vector
    let result = store.delete(&vector_id);
    assert!(result.is_ok());

    // Verify the vector no longer exists
    assert!(!store.id_exists(&vector_id).unwrap());

    // Verify no attributes are associated with the ID
    let index_arc = store.get_index();
    let index_guard = index_arc.read().unwrap();
    let attr_set = index_guard.get(&vector_id);
    if let Some(set) = attr_set.unwrap() {
        assert_eq!(set.len(), 0);
    }
}

#[test]
fn test_delete_nonexistent_id() {
    let store = create_test_store();
    let nonexistent_id: TestIdType = 999;

    // Verify ID doesn't exist
    assert!(!store.id_exists(&nonexistent_id).unwrap());

    // Delete should not fail for non-existent ID
    let result = store.delete(&nonexistent_id);
    assert!(
        result.is_ok(),
        "Delete should succeed even for non-existent ID"
    );
}

#[test]
fn test_multiple_vectors_same_attributes() {
    let store = create_test_store();

    // Same attributes for different vectors
    let shared_attrs = vec![
        Attribute::from_value(
            "category",
            AttributeValue::String("electronics".to_string()),
        ),
        Attribute::from_value("brand", AttributeValue::String("apple".to_string())),
    ];

    let vector_ids = [1u32, 2u32, 3u32];

    // Set same attributes for multiple vectors
    for &vector_id in &vector_ids {
        let result = store.set_element(&vector_id, &shared_attrs);
        assert!(result.is_ok());
        assert!(store.id_exists(&vector_id).unwrap());
    }

    // Verify all vectors have the attributes
    let index_arc = store.get_index();
    let index_guard = index_arc.read().unwrap();

    for &vector_id in &vector_ids {
        let attr_set = index_guard.get(&vector_id);
        let set = attr_set.unwrap().unwrap();
        assert_eq!(set.len(), 2);
    }
}

#[test]
fn test_empty_attributes() {
    let store = create_test_store();
    let vector_id: TestIdType = 300;

    // Set empty attributes
    let empty_attrs: Vec<Attribute> = vec![];
    let result = store.set_element(&vector_id, &empty_attrs);
    assert!(result.is_err());
}

#[test]
fn test_large_number_of_attributes() {
    let store = create_test_store();
    let vector_id: TestIdType = 400;
    let attr_count: usize = 100;

    // Create many attributes
    let attributes = create_test_attributes("large_test", attr_count);
    let result = store.set_element(&vector_id, &attributes);
    assert!(result.is_ok());

    // Verify all attributes are stored
    let index_arc = store.get_index();
    let index_guard = index_arc.read().unwrap();
    let attr_set = index_guard.get(&vector_id);
    let set = attr_set.unwrap().unwrap();
    assert_eq!(set.len() as usize, attr_count);
}

#[test]
fn test_duplicate_attributes() {
    let store = create_test_store();
    let vector_id: TestIdType = 500;

    // Create attributes with duplicates
    let attributes = vec![
        Attribute::from_value(
            "category",
            AttributeValue::String("electronics".to_string()),
        ),
        Attribute::from_value(
            "category",
            AttributeValue::String("electronics".to_string()),
        ), // Duplicate
        Attribute::from_value("brand", AttributeValue::String("apple".to_string())),
    ];

    let result = store.set_element(&vector_id, &attributes);
    assert!(result.is_ok());

    let index_arc = store.get_index();
    let index_guard = index_arc.read().unwrap();
    let attr_set = index_guard.get(&vector_id);

    let set = attr_set.unwrap().unwrap();
    assert!(set.len() == 2);
}

#[ignore = "This is a rudimentary perf test to be run manually."]
#[test]
fn test_memory_usage_with_many_vectors() {
    let store = create_test_store();
    let num_vectors: TestIdType = 10000;
    let attrs_per_vector = 3;

    eprintln!(
        "Testing memory usage with {} vectors, {} attributes each",
        num_vectors, attrs_per_vector
    );

    let start_time = std::time::Instant::now();

    // Insert many vectors
    for vector_id in 0..num_vectors {
        let attributes = (0..attrs_per_vector)
            .map(|attr_id| {
                Attribute::from_value(
                    format!("field_{}", attr_id),
                    AttributeValue::String(format!("value_{}_{}", vector_id, attr_id)),
                )
            })
            .collect::<Vec<_>>();

        let result = store.set_element(&(vector_id as TestIdType), &attributes);
        assert!(result.is_ok(), "Failed to insert vector {}", vector_id);
    }

    let insert_time = start_time.elapsed();
    eprintln!("Insert time: {:?}", insert_time);

    // Verify all vectors exist
    let verification_start = std::time::Instant::now();
    let index_arc = store.get_index();
    let index_guard = index_arc.read().unwrap();

    for vector_id in 0..num_vectors {
        assert!(store.id_exists(&vector_id).unwrap());
        let attr_set = index_guard.get(&(vector_id as TestIdType));
        let set = attr_set.unwrap().unwrap();
        assert_eq!(set.len(), attrs_per_vector);
    }

    let verification_time = verification_start.elapsed();
    eprintln!("Verification time: {:?}", verification_time);

    // Test random access performance
    let random_access_start = std::time::Instant::now();
    let test_ids: [u32; 5] = [0, 1000, 5000, 7500, 9999];

    for &test_id in &test_ids {
        assert!(store.id_exists(&test_id).unwrap());
        let attr_set = index_guard.get(&(test_id as TestIdType));
        let set = attr_set.unwrap().unwrap();
        assert_eq!(set.len(), attrs_per_vector);
    }

    let random_access_time = random_access_start.elapsed();
    eprintln!("Random access time: {:?}", random_access_time);

    // Performance assertions (adjust thresholds as needed)
    assert!(
        insert_time.as_millis() < 3,
        "Insert took too long: {:?}",
        insert_time
    );
    assert!(
        verification_time.as_millis() < 3,
        "Verification took too long: {:?}",
        verification_time
    );
    assert!(
        random_access_time.as_micros() < 100,
        "Random access took too long: {:?}",
        random_access_time
    );
}

#[test]
fn test_concurrency() {
    let vec_count: u32 = 100;
    let uniq_attr_count: u32 = 10;
    let max_attrs_per_vec: u32 = 40;
    let rt_count: usize = 8;
    let wt_count: usize = 4;
    let data = create_dataset(vec_count, max_attrs_per_vec, uniq_attr_count);

    eprintln!("Created dataset.");
    let added_points = Arc::new(scc::hash_set::HashSet::new());
    let store = Arc::new(create_test_store());
    let dataset = Arc::new(data);
    let barrier = Arc::new(Barrier::new(rt_count + wt_count));

    let write_handles = create_write_threads(
        store.clone(),
        dataset.clone(),
        added_points.clone(),
        wt_count,
        barrier.clone(),
    );
    let read_handles = create_read_threads(
        store.clone(),
        dataset.clone(),
        rt_count,
        barrier.clone(),
        |id, thread_store| {
            let attr_index = thread_store.get_index();
            let index_guard = attr_index.read().unwrap();
            index_guard
                .get(&id)
                .unwrap()
                .map(|set_opt| set_opt.into_owned())
        },
    );

    // Wait for all threads to complete
    for handle in write_handles {
        handle.join().unwrap();
    }
    for handle in read_handles {
        handle.join().unwrap();
    }

    // Verify all vectors were inserted correctly
    for vector_id in dataset.keys() {
        assert!(
            store.clone().id_exists(vector_id).unwrap(),
            "Vector {} wasn't inserted",
            vector_id
        );
    }

    // Verify the count of added points matches
    assert_eq!(
        added_points.len() as u32,
        vec_count,
        "Not all vectors were added"
    );
}

#[test]
fn test_delete_concurrency() {
    let vec_count: u32 = 100;
    let uniq_attr_count: u32 = 10;
    let max_attrs_per_vec: u32 = 40;
    let wt_count = 4;
    let dt_count = 2;

    let dataset = Arc::new(create_dataset(
        vec_count,
        max_attrs_per_vec,
        uniq_attr_count,
    ));
    let store = Arc::new(create_test_store());
    let added_points = Arc::new(scc::hash_set::HashSet::new());
    let barrier = Arc::new(Barrier::new(wt_count + dt_count));

    let set_handles = create_write_threads(
        store.clone(),
        dataset.clone(),
        added_points.clone(),
        wt_count,
        barrier.clone(),
    );
    let delete_handles =
        create_delete_threads(store.clone(), dataset.clone(), dt_count, barrier.clone());

    for handle in set_handles {
        let _ = handle.join();
    }
    for handle in delete_handles {
        let _ = handle.join();
    }

    //check that the points were added
    assert_eq!(added_points.len() as u32, vec_count);

    //but the store is empty because delete threads have done their job.
    for vector_id in dataset.keys() {
        assert!(!store.clone().id_exists(vector_id).unwrap());
    }
}
