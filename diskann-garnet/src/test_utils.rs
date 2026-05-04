/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::garnet::{Callbacks, ReadDataCallback, RmwDataCallback, TERM_BITMASK};
use core::slice;
use dashmap::DashMap;
use std::ffi::c_void;

thread_local! {
    pub static STORE: DashMap<Vec<u8>, Vec<u8>> = DashMap::new();
}

/// No-op filter callback that accepts all candidates.
unsafe extern "C" fn noop_filter(_context: u64, _internal_id: u32) -> u8 {
    1
}

pub struct Store;

impl Store {
    pub fn callbacks(&self) -> Callbacks {
        Callbacks::new(test_read, test_write, test_delete, test_rmw, noop_filter)
    }

    pub fn clear(&self) {
        STORE.with(|s| s.clear());
    }

    pub fn set(&self, context: u64, key: &[u8], value: &[u8]) {
        let context = context & TERM_BITMASK;
        let mut k = Vec::new();
        k.extend_from_slice(bytemuck::bytes_of(&context));
        k.extend_from_slice(key);
        STORE.with(|s| s.insert(k, value.to_owned()));
    }

    pub fn get(&self, context: u64, key: &[u8]) -> Option<Vec<u8>> {
        let context = context & TERM_BITMASK;
        let mut k = Vec::new();
        k.extend_from_slice(bytemuck::bytes_of(&context));
        k.extend_from_slice(key);
        STORE.with(|s| s.get(&k).map(|v| v.to_owned()))
    }

    pub fn delete(&self, context: u64, key: &[u8]) {
        let context = context & TERM_BITMASK;
        let mut k = Vec::new();
        k.extend_from_slice(bytemuck::bytes_of(&context));
        k.extend_from_slice(key);
        STORE.with(|s| s.remove(&k));
    }
}

unsafe extern "C" fn test_read(
    ctx: u64,
    count: u32,
    id_bytes: *const u8,
    id_len: usize,
    cb: ReadDataCallback,
    cb_ctx: *mut c_void,
) {
    let ids = unsafe { slice::from_raw_parts(id_bytes, id_len) };

    let mut pos = 0usize;
    for idx in 0..count {
        let mut len = 0u32;
        let len_bytes = bytemuck::bytes_of_mut(&mut len);
        len_bytes.copy_from_slice(&ids[pos..pos + 4]);

        pos += 4;

        let id = &ids[pos..pos + len as usize];
        pos += len as usize;

        let store = Store;
        if let Some(v) = store.get(ctx, id) {
            unsafe {
                cb(idx, cb_ctx, v.as_ptr(), v.len());
            }
        }
    }
}

unsafe extern "C" fn test_write(
    ctx: u64,
    id_bytes: *const u8,
    id_len: usize,
    val_bytes: *const u8,
    val_len: usize,
) -> bool {
    let id = unsafe { slice::from_raw_parts(id_bytes, id_len) };
    let val = unsafe { slice::from_raw_parts(val_bytes, val_len) };

    let store = Store;
    store.set(ctx, id, val);
    true
}

unsafe extern "C" fn test_delete(ctx: u64, id_bytes: *const u8, id_len: usize) -> bool {
    let id = unsafe { slice::from_raw_parts(id_bytes, id_len) };

    let store = Store;
    store.delete(ctx, id);
    true
}

unsafe extern "C" fn test_rmw(
    ctx: u64,
    id_bytes: *const u8,
    id_len: usize,
    write_len: usize,
    cb: RmwDataCallback,
    cb_ctx: *mut c_void,
) -> bool {
    let id = unsafe { slice::from_raw_parts(id_bytes, id_len) };

    let store = Store;
    let mut val = if let Some(v) = store.get(ctx, id) {
        v
    } else {
        vec![0u8; write_len]
    };

    unsafe {
        cb(cb_ctx, val.as_mut_ptr(), val.len());
    }

    store.set(ctx, id, &val);

    true
}

mod tests {
    use std::collections::HashMap;

    use crate::{
        garnet::{Context, Term},
        test_utils::Store,
    };

    #[test]
    fn basic() {
        let store = Store;
        store.clear();
        let callbacks = store.callbacks();
        let ctx = Context(0);

        // Reading a non-existant key should fail.
        assert!(!callbacks.exists_iid(ctx, 0));

        // Round tripping a write should work.
        assert!(callbacks.write_iid(ctx, 0, b"test"));
        let mut val = vec![0u8; 4];
        assert!(callbacks.read_single_iid(ctx, 0, &mut val));
        assert_eq!(val, b"test");

        // Overwriting a key should work.
        assert!(callbacks.write_iid(ctx, 0, b"again"));
        let mut val = vec![0u8; 5];
        assert!(callbacks.read_single_iid(ctx, 0, &mut val));
        assert_eq!(val, b"again");

        // Exists and delete should work.
        assert!(callbacks.exists_iid(ctx, 0));
        assert!(callbacks.delete_iid(ctx, 0));
        assert!(!callbacks.exists_iid(ctx, 0));

        // Different contexts should stay separate.
        assert!(callbacks.write_iid(ctx.term(Term::Vector), 0, b"0000"));
        assert!(callbacks.write_iid(ctx.term(Term::Neighbors), 0, b"nnnn"));
        let mut val = vec![0u8; 4];
        assert!(callbacks.read_single_iid(ctx.term(Term::Vector), 0, &mut val));
        assert_eq!(val, b"0000");
        assert!(callbacks.read_single_iid(ctx.term(Term::Neighbors), 0, &mut val));
        assert_eq!(val, b"nnnn");

        // Multi-read should work.
        assert!(callbacks.write_iid(ctx.term(Term::Vector), 1, b"2222"));
        let ids = [4u32, 0, 4, 1, 4, 2];
        let mut results = HashMap::new();
        callbacks.read_multi_lpiid(ctx.term(Term::Vector), &ids, |i, v| {
            results.insert(i, v.to_owned());
        });
        assert_eq!(results.get(&0), Some(b"0000".to_vec()).as_ref());
        assert_eq!(results.get(&1), Some(b"2222".to_vec()).as_ref());
        assert_eq!(results.get(&2), None);

        // RMW should work.
        assert!(callbacks.rmw_iid(ctx.term(Term::Vector), 0, 4, |data| {
            data.copy_from_slice(b"0rmw");
        }));
        assert!(callbacks.read_single_iid(ctx.term(Term::Vector), 0, &mut val));
        assert_eq!(val, b"0rmw");
    }

    #[test]
    fn garnet_provider_with_store_callbacks() {
        use crate::provider::GarnetProvider;
        use diskann_vector::distance::Metric;

        let store: Store = Store;
        store.clear();
        let callbacks = store.callbacks();
        let ctx: Context = Context(0);

        // Create a u8 GarnetProvider with the test Store callbacks
        let dim = 8;
        let max_degree = 32;
        let provider = GarnetProvider::<u8>::new(dim, Metric::L2, max_degree, callbacks, ctx);

        // Provider should be created successfully
        assert!(provider.is_ok());
        let provider = provider.unwrap();

        // Verify basic provider properties
        assert_eq!(provider.max_internal_id(), 0); // No vectors yet

        // Create a u8 GarnetProvider with the test Store callbacks
        let dim = 8;
        let max_degree = 32;
        let provider = GarnetProvider::<f32>::new(dim, Metric::L2, max_degree, callbacks, ctx);

        // Provider should be created successfully
        assert!(provider.is_ok());
        let provider = provider.unwrap();

        // Verify basic provider properties
        assert_eq!(provider.max_internal_id(), 0); // No vectors yet
    }
}
