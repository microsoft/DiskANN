/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) use inner::{Counters, LocalCounters};

#[cfg(not(feature = "integration-test"))]
mod inner {
    use std::marker::PhantomData;

    #[derive(Debug, Default)]
    pub(crate) struct Counters;

    impl Counters {
        pub(crate) fn new() -> Self {
            Self
        }

        pub(crate) fn local(&self) -> LocalCounters<'_> {
            LocalCounters::new()
        }
    }

    #[derive(Debug)]
    pub(crate) struct LocalCounters<'a> {
        _marker: PhantomData<&'a ()>,
    }

    impl LocalCounters<'_> {
        fn new() -> Self {
            Self {
                _marker: PhantomData,
            }
        }

        pub(crate) fn fork(&self) -> Self {
            Self::new()
        }

        pub(crate) fn query_distance(&mut self, _i: u64) {}
        pub(crate) fn distance_ref(&self, _i: u64) {}
        pub(crate) fn get_vector(&mut self, _i: u64) {}
        pub(crate) fn get_vector_ref(&self, _i: u64) {}
        pub(crate) fn set_vector(&mut self, _i: u64) {}
        pub(crate) fn get_neighbors(&mut self, _i: u64) {}
        pub(crate) fn set_neighbors(&mut self, _i: u64) {}
        pub(crate) fn append_vector(&mut self, _i: u64) {}
    }
}

#[cfg(feature = "integration-test")]
mod inner {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    #[derive(Debug, Default)]
    pub(crate) struct Counters {
        query_distance: AtomicU64,
        distance: AtomicU64,
        get_vector: AtomicU64,
        set_vector: AtomicU64,
        get_neighbors: AtomicU64,
        set_neighbors: AtomicU64,
        append_neighbors: AtomicU64,
    }

    impl Counters {
        pub(crate) fn new() -> Self {
            Self::default()
        }

        pub(crate) fn local(&self) -> LocalCounters<'_> {
            LocalCounters::new(self)
        }

        pub(crate) fn snapshot(&self) -> crate::integration::counters::CounterSnapshot {
            crate::integration::counters::CounterSnapshot {
                query_distance: self.query_distance.load(Relaxed),
                distance: self.distance.load(Relaxed),
                get_vector: self.get_vector.load(Relaxed),
                set_vector: self.set_vector.load(Relaxed),
                get_neighbors: self.get_neighbors.load(Relaxed),
                set_neighbors: self.set_neighbors.load(Relaxed),
                append_neighbors: self.append_neighbors.load(Relaxed),
            }
        }
    }

    #[derive(Debug)]
    pub(crate) struct LocalCounters<'a> {
        query_distance: u64,
        // This fields needs to be `AtomicU64` because we increment in some loops where we
        // have to increment it behind a shared reference.
        distance: AtomicU64,
        // This fields needs to be `AtomicU64` because we increment in some loops where we
        // have to increment it behind a shared reference.
        get_vector: AtomicU64,
        set_vector: u64,
        get_neighbors: u64,
        set_neighbors: u64,
        append_neighbors: u64,
        parent: &'a Counters,
    }

    impl<'a> LocalCounters<'a> {
        fn new(parent: &'a Counters) -> Self {
            Self {
                query_distance: 0,
                distance: AtomicU64::new(0),
                get_vector: AtomicU64::new(0),
                set_vector: 0,
                get_neighbors: 0,
                set_neighbors: 0,
                append_neighbors: 0,
                parent,
            }
        }

        pub(crate) fn fork(&self) -> LocalCounters<'a> {
            Self::new(self.parent)
        }

        pub(crate) fn query_distance(&mut self, i: u64) {
            self.query_distance += i;
        }

        pub(crate) fn distance_ref(&self, i: u64) {
            self.distance.fetch_add(i, Relaxed);
        }

        pub(crate) fn get_vector(&mut self, i: u64) {
            *self.get_vector.get_mut() += i;
        }

        pub(crate) fn get_vector_ref(&self, i: u64) {
            self.get_vector.fetch_add(i, Relaxed);
        }

        pub(crate) fn set_vector(&mut self, i: u64) {
            self.set_vector += i;
        }

        pub(crate) fn get_neighbors(&mut self, i: u64) {
            self.get_neighbors += i;
        }

        pub(crate) fn set_neighbors(&mut self, i: u64) {
            self.set_neighbors += i;
        }

        pub(crate) fn append_vector(&mut self, i: u64) {
            self.append_neighbors += i;
        }
    }

    impl Drop for LocalCounters<'_> {
        fn drop(&mut self) {
            let parent = self.parent;

            fn update(dst: &AtomicU64, src: u64) {
                dst.fetch_add(src, Relaxed);
            }

            update(&parent.query_distance, self.query_distance);
            update(&parent.distance, *self.distance.get_mut());
            update(&parent.get_vector, *self.get_vector.get_mut());
            update(&parent.set_vector, self.set_vector);
            update(&parent.get_neighbors, self.get_neighbors);
            update(&parent.set_neighbors, self.set_neighbors);
            update(&parent.append_neighbors, self.append_neighbors);
        }
    }
}
