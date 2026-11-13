/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod datatype;
pub(crate) mod filters;
pub(crate) mod io;

pub(crate) fn check_all<T, F>(
    old: &[T],
    new: &[T],
    key: &str,
    mut check: F,
) -> diskann_benchmark_runner::report::Report
where
    F: FnMut(&T, &T) -> diskann_benchmark_runner::report::Report,
{
    use diskann_benchmark_runner::report::Report;

    if old.len() != new.len() {
        Report::error(format_args!(
            "old {} has {} entries while the new has {}",
            key,
            old.len(),
            new.len()
        ))
    } else {
        let mut builder = Report::builder();
        for (i, (old, new)) in std::iter::zip(old.iter(), new.iter()).enumerate() {
            builder.push(i, check(old, new));
        }
        builder.finish()
    }
}
