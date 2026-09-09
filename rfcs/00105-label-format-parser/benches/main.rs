/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use criterion::{criterion_main};
use benchmarks::parser_benches;
use benchmarks::evaluator_benches;

mod benchmarks;

criterion_main!(parser_benches, evaluator_benches);
