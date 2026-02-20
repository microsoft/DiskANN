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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_num_threads_with_some() {
        assert_eq!(get_num_threads(Some(4)), 4);
        assert_eq!(get_num_threads(Some(1)), 1);
        assert_eq!(get_num_threads(Some(16)), 16);
    }

    #[test]
    fn test_get_num_threads_with_none() {
        let result = get_num_threads(None);
        // Should return the number of CPUs, which is at least 1
        assert!(result >= 1);
        // Should match num_cpus::get()
        assert_eq!(result, num_cpus::get());
    }
}
