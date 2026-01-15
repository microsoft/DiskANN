/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub trait IgnoreLockPoison<'a, T: 'a> {
    type Guard;

    fn lock_or_panic(&'a self) -> Self::Guard;
}

impl<'a, T> IgnoreLockPoison<'a, T> for std::sync::Mutex<T>
where
    T: 'a,
{
    type Guard = std::sync::MutexGuard<'a, T>;

    fn lock_or_panic(&'a self) -> Self::Guard {
        #[allow(clippy::expect_used)]
        self.lock().expect("lock was poisoned")
    }
}
