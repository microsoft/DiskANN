/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

macro_rules! representation {
    ($T:ty) => {
        $crate::repr::internal::macros::representation!({} $T where);
    };
    ({ $($generics:ident),* $(,)? } $T:ty where $($where:tt)*) => {
        impl<$($generics),*> $crate::repr::Representation for $T
        where
            $($where)*
        {
            fn max_degree(&self) -> MaxDegree {
                self.store.neighbors().max_degree()
            }

            fn retire(&self, i: u32) -> ANNResult<()> {
                Ok(self.store.retire(i.into_usize())?)
            }

            fn is_readable(&self, i: u32) -> Option<bool> {
                self.store.can_read_approximate(i.into_usize())
            }

            fn id_limit(&self) -> IdLimit {
                self.store.id_limit()
            }

            fn capacity(&self) -> Capacity {
                self.store.capacity()
            }
        }
    };
}

macro_rules! set_guard {
    ($(#[$doc:meta])* for<$lt:lifetime> $exclusive:ty) => {
        $(#[$doc])*
        #[derive(Debug)]
        pub struct Guard<$lt> {
            slot: $crate::store::Exclusive<$lt, $exclusive>,
        }

        impl<$lt> Guard<$lt> {
            fn new(slot: $crate::store::Exclusive<$lt, $exclusive>) -> Self {
                Self { slot }
            }
        }

        impl $crate::repr::Guard for Guard<'_> {
            fn publish(self) {
                self.slot.publish();
            }
            fn id(&self) -> u32 {
                self.slot.slot()
            }
        }
    }
}

pub(in crate::repr) use representation;
pub(in crate::repr) use set_guard;
