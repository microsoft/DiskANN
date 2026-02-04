/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::utils::percentiles::AsF64Lossy;

pub(crate) fn average_all<I>(x: I) -> f64
where
    I: IntoIterator<Item: AsF64Lossy>,
{
    let (sum, count) = x.into_iter().fold((0.0, 0usize), |(sum, count), partial| {
        (sum + partial.as_f64_lossy(), count + 1)
    });

    if count == 0 { 0.0 } else { sum / count as f64 }
}
