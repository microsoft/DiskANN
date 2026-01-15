/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, Barrier};

use crate::{
    tests::common::{
        create_dataset, create_read_threads, create_test_store, create_write_threads, TestIdType,
    },
    traits::{attribute_accessor::AttributeAccessor, attribute_store::AttributeStore},
};

#[test]
fn test_basic_attribute_accessor() {
    let vec_count: u32 = 100;
    let uniq_attr_count: u32 = 10;
    let max_attrs_per_vec: u32 = 40;
    let wt_count: usize = 4;
    let rt_count: usize = 8;
    let data = create_dataset(vec_count, max_attrs_per_vec, uniq_attr_count);

    eprintln!("Created dataset.");
    let added_points = Arc::new(scc::hash_set::HashSet::<TestIdType>::new());
    let store = Arc::new(create_test_store());
    let dataset = Arc::new(data);
    let barrier = Arc::new(Barrier::new(rt_count + wt_count));

    let write_threads = create_write_threads(
        store.clone(),
        dataset.clone(),
        added_points,
        wt_count,
        barrier.clone(),
    );

    let read_threads =
        create_read_threads(store, dataset, rt_count, barrier, |id, thread_store| {
            let mut attr_accessor = thread_store.attribute_accessor().unwrap();
            let labels_of_point = attr_accessor.visit_labels_of_point(id, |_, set_opt| {
                set_opt.map(|cow_set| cow_set.into_owned())
            });

            labels_of_point.unwrap() //ok if we panic here.
        });

    // Wait for all threads to complete
    for handle in write_threads {
        handle.join().unwrap();
    }
    for handle in read_threads {
        handle.join().unwrap();
    }
    //Verification of results is done inside the read threads.
}
