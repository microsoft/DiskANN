/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

/// Create a generic multi-threaded pool with `num_threads`.
#[cfg(test)]
pub(crate) fn runtime(num_threads: NonZeroUsize) -> anyhow::Result<rayon::ThreadPool>
where
{
    runtime_with(num_threads, |x| x)
}

/// Create a generic multi-threaded pool with `num_threads`.
///
/// After initial setup, the [`rayon::ThreadPoolBuilder`] will be passed to the closure `f`
/// for customization. Note that the builder provided to the callback will already be
/// initialized to contain `num_threads` threads.
pub(crate) fn runtime_with<F>(num_threads: NonZeroUsize, f: F) -> anyhow::Result<rayon::ThreadPool>
where
    F: FnOnce(rayon::ThreadPoolBuilder) -> rayon::ThreadPoolBuilder,
{
    Ok(f(rayon::ThreadPoolBuilder::new().num_threads(num_threads.get())).build()?)
}
