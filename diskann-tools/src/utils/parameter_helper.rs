/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use num_cpus;

pub fn get_num_threads(num_threads: Option<usize>) -> usize {
    match num_threads {
        Some(n) => n,
        None => num_cpus::get(),
    }
}
