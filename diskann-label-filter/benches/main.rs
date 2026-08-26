/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks::{evaluator_benches, parser_benches};
use criterion::criterion_main;

mod benchmarks;

criterion_main!(parser_benches, evaluator_benches);
