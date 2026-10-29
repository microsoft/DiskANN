/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Print hardware cache capacities, logical sharers, and probe source.

use diskann_quantization::cache_diagnostics::HardwareCache;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(info) = HardwareCache::detect() else {
        eprintln!("Could not detect both L1d and L2 caches; no fallback values are shown.");
        return ExitCode::FAILURE;
    };
    println!("source={}", info.source);
    for (name, cache) in [("L1d", info.l1d), ("L2", info.l2)] {
        print!("{name}: capacity_bytes={}", cache.bytes);
        match cache.logical_sharers {
            Some(count) => println!(", logical_sharers={count}"),
            None => println!(", logical_sharers=unknown"),
        }
    }
    ExitCode::SUCCESS
}
