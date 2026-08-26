/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This module exposes "public" integration-test wrappers for the various internal store
//! mechanisms to drive larger concurrency tests.

pub mod checked;
pub mod invasive;

/// These implementations have a similar structure. A [`boilerplate`] macro is used to ensure
/// the capabilities exposed are mostly the same.
macro_rules! boilerplate {
    (
        $plugin:ty => $store:ident,
        for<$read_lt:lifetime> $read:ty => $reader:ident,
        for<$slot_lt:lifetime> $slot:ty => $writer:ident,
    ) => {
        /// A test store wraper.
        #[derive(Debug)]
        pub struct $store {
            store: $crate::store::Store<$plugin>,
        }

        impl $store {
            /// Return the total number of slots, including the frozen point.
            pub fn readable_slots(&self) -> usize {
                self.store.frozen().end as usize
            }

            /// Return the range of writable (non-frozen) slot indices.
            pub fn writable_slots(&self) -> usize {
                self.store.frozen().start as usize
            }

            /// Attempt to reclaim retired slots, returning the number reclaimed if any.
            pub fn reclaim(&self) -> Option<usize> {
                self.store.try_drain()
            }

            /// Acquire a slot, returning a [`Writer`]. Returns `None` if no slot is
            /// available for writing.
            pub fn acquire(&self) -> Option<$writer<'_>> {
                self.store.acquire().map(Writer::new)
            }

            /// Attempt to retire slot `i`. Returns `true` only if the slot was successfully
            /// retired.
            #[must_use = "result indicates success or failure"]
            pub fn retire(&self, i: usize) -> bool {
                self.store.retire(i).is_ok()
            }

            /// Attain a reader into the store. Returns `None` if all epoch guard slots
            /// are used.
            pub fn reader(&self) -> Option<$reader<'_>> {
                match <$plugin>::reader(&self.store) {
                    Ok(reader) => Some($reader::new(reader)),
                    Err($crate::epoch::Unavailable) => None,
                }
            }
        }

        /// A reader for the test store.
        #[derive(Debug)]
        pub struct $reader<$read_lt> {
            reader: $read,
        }

        impl<$read_lt> $reader<$read_lt> {
            fn new(reader: $read) -> Self {
                Self { reader }
            }
        }

        /// A writer for the test store.
        #[derive(Debug)]
        pub struct $writer<$slot_lt> {
            slot: $crate::store::Slot<$slot_lt, $slot>,
        }

        impl<$slot_lt> $writer<$slot_lt> {
            fn new(slot: $crate::store::Slot<$slot_lt, $slot>) -> Self {
                Self { slot }
            }

            /// Publish the slot - making it accessible to readers.
            pub fn publish(self) {
                self.slot.publish();
            }
        }
    };
}

use boilerplate;
