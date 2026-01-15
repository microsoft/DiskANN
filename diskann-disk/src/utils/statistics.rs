/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// A struct to gather statistics for a given disk query execution.
#[derive(Debug, Default, Clone)]
pub struct QueryStatistics {
    /// Total time to process the query in microseconds.
    pub total_execution_time_us: u128,

    /// Total time spent in IO operations in microseconds.
    pub io_time_us: u128,

    /// Total time spent in CPU operations in microseconds.
    pub cpu_time_us: u128,

    /// Time spent in query preprocessing for the PQ in microseconds.
    pub query_pq_preprocess_time_us: u128,

    /// Total number of IO operations issued.
    pub total_io_operations: u32,

    /// Number of saved comparisons (optimization metric).
    pub comparisons_saved: u32,

    /// Total number of comparisons performed.
    pub total_comparisons: u32,

    /// Total number of vertices loaded.
    pub total_vertices_loaded: u32,

    /// Number of hops performed during search.
    pub search_hops: u32,
}

/// Calculates the percentile value of a specific metric in a list of QueryStats.
pub fn get_percentile_stats<T: Ord + Copy>(
    stats: &[QueryStatistics],
    percentile: f32,
    member_fn: impl Fn(&QueryStatistics) -> T,
) -> T {
    let mut vals: Vec<T> = stats.iter().map(&member_fn).collect();
    vals.sort_unstable();
    let idx = ((percentile * stats.len() as f32) as usize).min(stats.len() - 1);
    vals[idx]
}

/// Calculates the mean value of a specific metric in a list of QueryStats.
pub fn get_mean_stats<T: Into<f64>>(
    stats: &[QueryStatistics],
    member_fn: impl Fn(&QueryStatistics) -> T,
) -> f64 {
    get_sum_stats(stats, member_fn) / (stats.len() as f64)
}

pub fn get_sum_stats<T: Into<f64>>(
    stats: &[QueryStatistics],
    member_fn: impl Fn(&QueryStatistics) -> T,
) -> f64 {
    stats.iter().map(&member_fn).map(|v| v.into()).sum()
}
#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn test_get_percentile_stats_batch() {
        test_get_percentile_stats(0.0f32);
        test_get_percentile_stats(0.5);
        test_get_percentile_stats(0.57);
        test_get_percentile_stats(0.85);
        test_get_percentile_stats(0.95);
        test_get_percentile_stats(0.99);
        test_get_percentile_stats(1.0);

        let mut rng = diskann_providers::utils::create_rnd_in_tests();
        let random_percentiles: Vec<f32> = (0..100).map(|_| rng.random_range(0f32..1f32)).collect();
        random_percentiles
            .iter()
            .for_each(|&p| test_get_percentile_stats(p));
    }

    fn test_get_percentile_stats(percentile: f32) {
        let mut rng = diskann_providers::utils::create_rnd_in_tests();
        let mut random_numbers: Vec<u32> = (0..1000).map(|_| rng.random_range(0..999999)).collect();

        let query_stats: Vec<QueryStatistics> = random_numbers
            .iter()
            .map(|&num| QueryStatistics {
                total_io_operations: num,
                ..Default::default()
            })
            .collect();

        let member_fn = |s: &QueryStatistics| s.total_io_operations;

        let result = get_percentile_stats(&query_stats, percentile, member_fn);

        let index =
            ((percentile * random_numbers.len() as f32) as usize).min(random_numbers.len() - 1);
        random_numbers.sort_unstable();
        let expected_result: u32 = random_numbers[index];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_get_mean_stats() {
        let numbers = [1, 2, 3, 4, 5];

        let query_stats: Vec<QueryStatistics> = numbers
            .iter()
            .map(|&num| QueryStatistics {
                total_io_operations: num,
                ..Default::default()
            })
            .collect();

        let member_fn = |s: &QueryStatistics| s.total_io_operations;
        let result = get_mean_stats(&query_stats, member_fn);

        let expected_result: f64 = 3.0; // (1 + 2 + 3 + 4 + 5) / 5 = 3

        assert!((result - expected_result).abs() <= 1e-3);
    }

    #[test]
    fn test_get_sum_stats() {
        let numbers = [1, 2, 3, 4, 5];

        let query_stats: Vec<QueryStatistics> = numbers
            .iter()
            .map(|&num| QueryStatistics {
                total_io_operations: num,
                ..Default::default()
            })
            .collect();

        let member_fn = |s: &QueryStatistics| s.total_io_operations;
        let result = get_sum_stats(&query_stats, member_fn);

        let expected_result: f64 = 15.0; // 1 + 2 + 3 + 4 + 5 = 15

        assert!((result - expected_result).abs() <= 1e-3);
    }
}
