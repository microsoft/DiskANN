/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{collections::HashSet, fmt, hash::Hash, io::Read, mem::size_of};

use bytemuck::cast_slice;
use diskann::{ANNError, ANNResult};
use diskann_providers::model::graph::traits::GraphDataType;
use diskann_providers::storage::StorageReadProvider;
use diskann_providers::utils::read_metadata;
use tracing::{error, info};

use crate::utils::CMDToolError;

pub struct TruthSet {
    pub index_nodes: Vec<u32>,
    pub distances: Option<Vec<f32>>,
    pub index_num_points: usize,
    pub index_dimension: usize,
}

pub struct TruthSetWithAssociatedData<Data: GraphDataType> {
    pub index_nodes: Vec<<Data as GraphDataType>::AssociatedDataType>,
    pub distances: Option<Vec<f32>>,
    pub index_num_points: usize,
    pub index_dimension: usize,
}

pub struct RangeSearchTruthSet {
    pub index_nodes: Vec<Vec<u32>>,
    pub distances: Option<Vec<Vec<f32>>>,
    pub index_num_points: usize,
    pub index_dimensions: Vec<u32>,
}

/// A struct used to indicate the bounds `k` and `n` for recall computation where:
///
/// * k: Is the number of ground truth neighbors to use.
/// * n: Is the number of retrieved neighbors.
///
/// Recall call is measured as the fraction of the `k` ground truth neighbors that are
/// present in the `n` retrieved neighbors.
///
/// We make a deliberate choice that the invariant `k <= n` must hold (don't search for
/// more ground truth neighbors than those actually retrieved.
///
/// Furthermore, both `n` and `k` must be non-zero.
///
/// The constructor `new` should be used instead of direct construction because it will
/// enforce this invariant.
#[derive(Debug, Clone, Copy)]
pub struct KRecallAtN {
    k: u32,
    n: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum RecallBoundsError {
    // Error when `k` is assigned a higher value than `n`.
    KGreaterThanN { k: u32, n: u32 },
    // Both arguments must be non-zero.
    // This alternative captures both values to provide a better error message in the case
    // that both arguments are zero.
    ArgumentIsZero { k: u32, n: u32 },
}
impl fmt::Display for RecallBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RecallBoundsError::KGreaterThanN { k, n } => {
                write!(
                    f,
                    "recall value k ({}) must be less than or equal to n ({})",
                    k, n
                )
            }
            // Match the various argument-is-zero cases.
            RecallBoundsError::ArgumentIsZero { k, n } => {
                if *k == 0 && *n == 0 {
                    write!(f, "recall values k and n must both be non-zero")
                } else if *k == 0 {
                    write!(f, "recall values k must be non-zero")
                } else {
                    write!(f, "recall values n must be non-zero")
                }
            }
        }
    }
}

// opt-in to error reporting.
impl std::error::Error for RecallBoundsError {}

// Allow conversion to `ANNError` for error propagation.
impl From<RecallBoundsError> for CMDToolError {
    fn from(err: RecallBoundsError) -> Self {
        CMDToolError {
            details: err.to_string(),
        }
    }
}

impl KRecallAtN {
    /// Construct a new instance of this class.
    ///
    /// If the invariant `k <= n` does not hold, than return the error type `KGreaterThanNError`.
    pub fn new(k: u32, n: u32) -> Result<Self, RecallBoundsError> {
        if k == 0 || n == 0 {
            Err(RecallBoundsError::ArgumentIsZero { k, n })
        } else if k > n {
            Err(RecallBoundsError::KGreaterThanN { k, n })
        } else {
            Ok(KRecallAtN { k, n })
        }
    }

    pub fn get_k(self) -> usize {
        self.k as usize
    }

    pub fn get_n(self) -> usize {
        self.n as usize
    }
}

/// Calculate the intersection between the top `k` ground truth elements and the top `n`
/// obtained results.
#[allow(clippy::too_many_arguments)]
pub fn calculate_recall<T: Eq + Hash + Copy>(
    num_queries: usize,
    ground_truth: &[T],
    gt_dist: Option<&Vec<f32>>,
    dim_gt: usize,
    our_results: &[T],
    dim_or: u32,
    recall_bounds: KRecallAtN,
) -> ANNResult<f64> {
    let mut total_recall: f64 = 0.0;
    let (mut gt, mut res): (HashSet<T>, HashSet<T>) = (HashSet::new(), HashSet::new());

    for i in 0..num_queries {
        gt.clear();
        res.clear();

        let gt_slice = &ground_truth[dim_gt * i..];
        let res_slice = &our_results[dim_or as usize * i..];
        let mut tie_breaker = recall_bounds.get_k();

        if let Some(gt_dist) = gt_dist {
            let gt_dist_vec = &gt_dist[dim_gt * i..];
            while tie_breaker < dim_gt
                && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_bounds.get_k() - 1]
            {
                tie_breaker += 1;
            }
        }

        (0..tie_breaker).for_each(|idx| {
            gt.insert(gt_slice[idx]);
        });

        (0..recall_bounds.get_n()).for_each(|idx| {
            res.insert(res_slice[idx]);
        });

        let mut cur_recall: u32 = 0;
        for v in gt.iter() {
            if res.contains(v) && cur_recall < recall_bounds.get_k() as u32 {
                cur_recall += 1;
            }
        }

        total_recall += cur_recall as f64;
    }

    Ok(total_recall / num_queries as f64 * (100.0 / recall_bounds.get_k() as f64))
}

pub fn calculate_range_search_recall(
    num_queries: u32,
    groundtruth: &[Vec<u32>],
    our_results: &[Vec<u32>],
) -> ANNResult<f64> {
    let mut total_recall = 0.0;
    for i in 0..num_queries as usize {
        let mut gt: HashSet<u32> = HashSet::new();
        let mut res: HashSet<u32> = HashSet::new();

        for &item in &groundtruth[i] {
            gt.insert(item);
        }

        for &item in &our_results[i] {
            res.insert(item);
        }

        let mut cur_recall = 0;
        for &v in &gt {
            if res.contains(&v) {
                cur_recall += 1;
            }
        }

        if !gt.is_empty() {
            total_recall += (100.0 * cur_recall as f64) / gt.len() as f64;
        } else {
            total_recall += 100.0;
        }
    }

    Ok(total_recall / num_queries as f64)
}

/// Calculates the filtered search recall for a set of queries.
///
/// This function computes the recall percentage for a filtered search scenario, where the recall
/// is calculated as the percentage of ground truth elements that are present in the retrieved
/// results, normalized by the maximum of min(k_recall, ground truth size) and retrieved results length.
///
/// # Arguments
///
/// * `num_queries` - The number of queries for which recall is being calculated.
/// * `gt_dist` - An optional vector of distances for the ground truth elements.
/// * `groundtruth` - A slice of vectors, where each vector contains the ground truth IDs for a query.
/// * `our_results` - A slice of vectors, where each vector contains the retrieved IDs for a query.
/// * `k_recall` - number of top results to consider from ground_truth for recall calculation. Must be greater than 0.
///
/// # Returns
///
/// Returns an `ANNResult<f64>` containing the average recall percentage across all queries.
///
/// # Assumptions
///
/// * The `groundtruth` and `our_results` slices must have the same length as `num_queries`.
/// * Each vector in 'groundtruth' must be the same length of the corresponding vector in 'gt_dist', if 'gt_dist' is provided.
///
/// # Behavior
///
/// - If the ground truth for a query is empty, the recall for that query is considered to be 100.0.
/// - For non-empty ground truth, the recall is calculated as:
///   `(100.0 * number_of_matches) / max(min(k_recall value, groundtruth size), our_results size)`
/// - When the vector in gt_dist is provided and k_recall value is less than the size of groundtruth, ties are broken.
///
/// # Differences from Range Search Recall
///
/// Unlike `calculate_range_search_recall`, this function normalizes the recall by the maximum
/// of the sizes of the k_recall/groundtruth value and retrieved results, which can lead to different recall
/// values in cases where the retrieved results contain more elements than the ground truth.
pub fn calculate_filtered_search_recall(
    num_queries: usize,
    gt_dist: Option<&[Vec<f32>]>,
    groundtruth: &[Vec<u32>],
    our_results: &[Vec<u32>],
    k_recall: u32,
) -> ANNResult<f64> {
    if k_recall == 0 {
        return Err(ANNError::log_index_error(format_args!(
            "k_recall value must be greater than 0, but got {}",
            k_recall
        )));
    }

    if groundtruth.len() != num_queries || our_results.len() != num_queries {
        return Err(ANNError::log_index_error(format_args!(
            "groundtruth length ({}) or our_results length ({}) does not match num_queries ({})",
            groundtruth.len(),
            our_results.len(),
            num_queries
        )));
    }

    let mut total_recall = 0.0;
    for i in 0..num_queries {
        let mut gt: HashSet<u32> = HashSet::new();
        let mut res: HashSet<u32> = HashSet::new();
        let gt_cutoff = (k_recall as usize).min(groundtruth[i].len());

        for &item in &groundtruth[i][..gt_cutoff] {
            //only insert k items from groundtruth
            gt.insert(item);
        }

        for &item in &our_results[i] {
            res.insert(item);
        }

        if gt_cutoff > 0 {
            //only break ties when groundtruth is not empty
            if let Some(gt_dist) = gt_dist {
                let gt_dist_vec = gt_dist[i].as_slice();

                if gt_dist_vec.len() != groundtruth[i].len() {
                    return Err(ANNError::log_index_error(format_args!(
                        "Ground truth distance for query ({}) vector length ({}) is not equal to groundtruth len ({})",
                        i,
                        gt_dist_vec.len(),
                        groundtruth[i].len(),
                    )));
                }

                let mut tie_breaker = gt_cutoff;

                while tie_breaker < gt_dist_vec.len() //while there are still ties, add them to gt
                        && gt_dist_vec[tie_breaker] == gt_dist_vec[gt_cutoff - 1]
                {
                    gt.insert(groundtruth[i][tie_breaker]);
                    tie_breaker += 1;
                }
            }
        }

        let mut cur_recall = 0;

        for &v in &gt {
            if res.contains(&v) {
                cur_recall += 1;
            }
        }

        if gt_cutoff > 0 {
            total_recall += (100.0 * cur_recall as f64) / gt_cutoff.max(res.len()) as f64;
        } else {
            total_recall += 100.0;
        }
    }

    Ok(total_recall / num_queries as f64)
}

pub fn get_graph_num_frozen_points(
    storage_provider: &impl StorageReadProvider,
    graph_file: &str,
) -> ANNResult<usize> {
    let mut file = storage_provider.open_reader(graph_file)?;
    let mut usize_buffer = [0; size_of::<usize>()];
    let mut u32_buffer = [0; size_of::<u32>()];

    file.read_exact(&mut usize_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    file.read_exact(&mut usize_buffer)?;
    let file_frozen_pts = usize::from_le_bytes(usize_buffer);

    Ok(file_frozen_pts)
}

pub fn get_graph_max_observed_degree(
    storage_provider: &impl StorageReadProvider,
    graph_file: &str,
) -> ANNResult<u32> {
    let mut file = storage_provider.open_reader(graph_file)?;
    let mut usize_buffer = [0; size_of::<usize>()];
    let mut u32_buffer = [0; size_of::<u32>()];

    file.read_exact(&mut usize_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    let max_observed_degree = u32::from_le_bytes(u32_buffer);

    Ok(max_observed_degree)
}

pub fn load_truthset(
    storage_provider: &impl StorageReadProvider,
    bin_file: &str,
) -> ANNResult<TruthSet> {
    let actual_file_size = storage_provider.get_length(bin_file)? as usize;
    let mut file = storage_provider.open_reader(bin_file)?;

    let metadata = read_metadata(&mut file)?;
    let (npts, dim) = (metadata.npoints, metadata.ndims);

    info!("Metadata: #pts = {npts}, #dims = {dim}... ");

    let expected_file_size_with_dists: usize =
        2 * npts * dim * size_of::<u32>() + 2 * size_of::<u32>();
    let expected_file_size_just_ids: usize = npts * dim * size_of::<u32>() + 2 * size_of::<u32>();

    // truthset_type: 1 = ids + distances, 2 = ids only
    let truthset_type : i32 = match actual_file_size {
        x if x == expected_file_size_with_dists => 1,
        x if x == expected_file_size_just_ids => 2,
        _ => return Err(ANNError::log_index_error(format_args!(
            "Error. File size mismatch. File should have bin format, with npts followed by ngt followed by npts*ngt ids and optionally followed by npts*ngt distance values; actual size: {}, expected: {} or {}",
            actual_file_size,
            expected_file_size_with_dists,
            expected_file_size_just_ids
        )))
    };

    let mut ids: Vec<u32> = vec![0; npts * dim];
    let mut buffer = vec![0; npts * dim * size_of::<u32>()];
    file.read_exact(&mut buffer)?;
    ids.clone_from_slice(cast_slice::<u8, u32>(&buffer));

    if truthset_type == 1 {
        let mut dists: Vec<f32> = vec![0.0; npts * dim];
        let mut buffer = vec![0; npts * dim * size_of::<f32>()];
        file.read_exact(&mut buffer)?;
        dists.clone_from_slice(cast_slice::<u8, f32>(&buffer));

        return Ok(TruthSet {
            index_nodes: ids,
            distances: Some(dists),
            index_num_points: npts,
            index_dimension: dim,
        });
    }

    Ok(TruthSet {
        index_nodes: ids,
        distances: None,
        index_num_points: npts,
        index_dimension: dim,
    })
}

pub fn load_truthset_with_associated_data<Data: GraphDataType>(
    storage_provider: &impl StorageReadProvider,
    bin_file: &str,
) -> ANNResult<TruthSetWithAssociatedData<Data>> {
    let mut file = storage_provider.open_reader(bin_file)?;

    let metadata = read_metadata(&mut file)?;
    let (npts, dim) = (metadata.npoints, metadata.ndims);

    info!("Metadata: #pts = {}, #dims = {}...", npts, dim);

    let mut associated_data: Vec<Data::AssociatedDataType> =
        vec![Data::AssociatedDataType::default(); npts * dim];

    for associated_datum in associated_data.iter_mut().take(npts * dim) {
        let mut associated_data_buf = vec![0u8; size_of::<Data::AssociatedDataType>()];
        file.read_exact(&mut associated_data_buf)
            .map_err(ANNError::log_io_error)?;

        match bincode::deserialize::<Data::AssociatedDataType>(&associated_data_buf) {
            Ok(datum) => {
                *associated_datum = datum;
            }
            Err(_) => {
                error!("Error deserializing associated data");
                return Err(ANNError::log_index_error("Error reading associated data"));
            }
        }
    }

    let mut dists: Vec<f32> = vec![0.0; npts * dim];
    let mut buffer = vec![0; npts * dim * size_of::<f32>()];
    file.read_exact(&mut buffer)?;
    dists.clone_from_slice(cast_slice::<u8, f32>(&buffer));

    Ok(TruthSetWithAssociatedData {
        index_nodes: associated_data,
        distances: Some(dists),
        index_num_points: npts,
        index_dimension: dim,
    })
}

// Load the range truthset from the file.
// The format of the file is as follows:
// 1. The first 4 bytes are the number of queries.
// 2. The next 4 bytes are the number of total vector ids.
// 3. The next (queries * 4) bytes are the numbers of vector ids for each query.
// 4. The next (total vector ids * 4) bytes are vector ids for each query.
pub fn load_range_truthset(
    storage_provider: &impl StorageReadProvider,
    bin_file: &str,
) -> ANNResult<RangeSearchTruthSet> {
    let mut file = storage_provider.open_reader(bin_file)?;

    let metadata = read_metadata(&mut file)?;
    let (npts, total_ids) = (metadata.npoints, metadata.ndims);
    let mut buffer = [0; size_of::<i32>()];

    info!("Metadata: #pts = {}, #totalIds = {}", npts, total_ids);

    let mut ids: Vec<Vec<u32>> = Vec::new();
    let mut counts: Vec<u32> = vec![0; npts];

    for count in counts.iter_mut() {
        file.read_exact(&mut buffer)?;
        *count = i32::from_le_bytes(buffer) as u32;
    }

    for &count in &counts {
        let mut point_ids: Vec<u32> = vec![0; count as usize];
        let mut buffer = vec![0; count as usize * size_of::<u32>()];
        file.read_exact(&mut buffer)?;
        point_ids.clone_from_slice(cast_slice::<u8, u32>(&buffer));
        ids.push(point_ids);
    }

    Ok(RangeSearchTruthSet {
        index_nodes: ids,
        distances: None,
        index_num_points: npts,
        index_dimensions: counts,
    })
}

// Load the vector filters from the file in the range truthset format.
pub fn load_vector_filters(
    storage_provider: &impl StorageReadProvider,
    bin_file: &str,
) -> ANNResult<Vec<HashSet<u32>>> {
    let range_truthset = load_range_truthset(storage_provider, bin_file)?;

    let query_filters: Vec<HashSet<u32>> = range_truthset
        .index_nodes
        .into_iter()
        .map(|filter| filter.into_iter().collect())
        .collect();

    Ok(query_filters)
}

#[cfg(test)]
mod test_search_index_utils {
    use super::*;

    struct ExpectedRecall {
        pub recall_k: usize,
        pub recall_n: usize,
        // Recall for each component.
        pub components: Vec<usize>,
    }

    impl ExpectedRecall {
        fn new(recall_k: usize, recall_n: usize, components: Vec<usize>) -> Self {
            assert!(recall_k <= recall_n);
            components.iter().for_each(|x| {
                assert!(*x <= recall_k);
            });
            Self {
                recall_k,
                recall_n,
                components,
            }
        }

        fn compute(&self) -> f64 {
            100.0 * (self.components.iter().sum::<usize>() as f64)
                / ((self.components.len() * self.recall_k) as f64)
        }
    }

    #[test]
    fn test_k_recall_at_n_struct() {
        // Happy paths should succeed.
        for k in 1..=10 {
            for n in k..=10 {
                let v = KRecallAtN::new(k, n).unwrap();
                assert_eq!(v.get_k(), k as usize);
                assert_eq!(v.get_n(), n as usize);
            }
        }

        // Error paths.
        // N.B.: Note the inversion of `k` and `n` in the loop bounds!
        for n in 1..=10 {
            for k in (n + 1)..=11 {
                let v = KRecallAtN::new(k, n).unwrap_err();
                match v {
                    RecallBoundsError::KGreaterThanN { k: k_err, n: n_err } => {
                        assert_eq!(k_err, k);
                        assert_eq!(n_err, n);
                    }
                    RecallBoundsError::ArgumentIsZero { .. } => {
                        panic!("unreachable reached");
                    }
                }
                let message = format!("{}", v);
                assert!(message.contains("recall value k"));
                assert!(message.contains("must be less than or equal to n"));
                assert!(message.contains(&format!("{}", k)));
                assert!(message.contains(&format!("{}", n)));
            }
        }

        // both zero
        let v = KRecallAtN::new(0, 0).unwrap_err();
        let message = format!("{}", v);
        assert!(message == "recall values k and n must both be non-zero");

        // k is zero
        let v = KRecallAtN::new(0, 10).unwrap_err();
        let message = format!("{}", v);
        assert!(message == "recall values k must be non-zero");

        // n is zero
        let v = KRecallAtN::new(10, 0).unwrap_err();
        let message = format!("{}", v);
        assert!(message == "recall values n must be non-zero");
    }

    #[test]
    fn test_compute_recall() {
        // Set up examples for ground truth and retrieved IDs.
        let groundtruth_dim = 10;
        let num_queries = 4;

        let groundtruth: Vec<u32> = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // row 0
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, // row 1
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // row 2
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // row 3
        ];

        assert_eq!(groundtruth.len(), num_queries * groundtruth_dim);

        let distances: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, // row 0
            2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // row 1
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // row 2
            0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, // row 3
        ];

        assert_eq!(distances.len(), groundtruth.len());

        // Shift row 0 by one and row 1 by two.
        let results_dim = 6;
        let our_results: Vec<u32> = vec![
            100, 0, 1, 2, 5, 6, // row 0
            100, 101, 7, 8, 9, 10, // row 1
            0, 1, 2, 3, 4, 5, // row 2
            0, 1, 2, 3, 4, 5, // row 3
        ];
        assert_eq!(our_results.len(), num_queries * results_dim);

        // No ties
        let expected_no_ties = vec![
            // Equal `k` and `n`
            ExpectedRecall::new(1, 1, vec![0, 0, 1, 1]),
            ExpectedRecall::new(2, 2, vec![1, 0, 2, 2]),
            ExpectedRecall::new(3, 3, vec![2, 1, 3, 3]),
            ExpectedRecall::new(4, 4, vec![3, 2, 4, 4]),
            ExpectedRecall::new(5, 5, vec![3, 3, 5, 5]),
            ExpectedRecall::new(6, 6, vec![4, 4, 6, 6]),
            // Unequal `k` and `n`.
            ExpectedRecall::new(1, 2, vec![1, 0, 1, 1]),
            ExpectedRecall::new(1, 3, vec![1, 0, 1, 1]),
            ExpectedRecall::new(2, 3, vec![2, 0, 2, 2]),
            ExpectedRecall::new(3, 5, vec![3, 1, 3, 3]),
        ];
        let epsilon = 1e-6; // Define a small tolerance
        for (i, expected) in expected_no_ties.iter().enumerate() {
            println!("No Ties: i = {i}");
            assert_eq!(expected.components.len(), num_queries);
            let recall = calculate_recall(
                num_queries,
                &groundtruth,
                None,
                groundtruth_dim,
                &our_results,
                results_dim as u32,
                KRecallAtN::new(expected.recall_k as u32, expected.recall_n as u32).unwrap(),
            );
            let left = recall.unwrap();
            let right = expected.compute();
            assert!(
                (left - right).abs() < epsilon,
                "left = {}, right = {}",
                left,
                right
            );
        }

        // With Ties
        let expected_with_ties = vec![
            // Equal `k` and `n`
            ExpectedRecall::new(1, 1, vec![0, 0, 1, 1]),
            ExpectedRecall::new(2, 2, vec![1, 0, 2, 2]),
            ExpectedRecall::new(3, 3, vec![2, 1, 3, 3]),
            ExpectedRecall::new(4, 4, vec![3, 2, 4, 4]),
            ExpectedRecall::new(5, 5, vec![4, 3, 5, 5]), // tie-breaker kicks in
            ExpectedRecall::new(6, 6, vec![5, 4, 6, 6]), // tie-breaker kicks in
            // Unequal `k` and `n`.
            ExpectedRecall::new(1, 2, vec![1, 0, 1, 1]),
            ExpectedRecall::new(1, 3, vec![1, 0, 1, 1]),
            ExpectedRecall::new(2, 3, vec![2, 1, 2, 2]),
            ExpectedRecall::new(4, 5, vec![4, 3, 4, 4]),
        ];

        for (i, expected) in expected_with_ties.iter().enumerate() {
            println!("With Ties: i = {i}");
            assert_eq!(expected.components.len(), num_queries);
            let recall = calculate_recall(
                num_queries,
                &groundtruth,
                Some(&distances),
                groundtruth_dim,
                &our_results,
                results_dim as u32,
                KRecallAtN::new(expected.recall_k as u32, expected.recall_n as u32).unwrap(),
            );
            let left = recall.unwrap();
            let right = expected.compute();
            assert!(
                (left - right).abs() < epsilon,
                "left = {}, right = {}",
                left,
                right
            );
        }
    }

    #[test]
    fn test_calculate_range_search_recall() {
        assert_eq!(
            calculate_range_search_recall(1, &[vec![5, 6],], &[vec![5, 6, 7, 8, 9],]).unwrap(),
            100.0,
            "Returned more results than ground truth"
        );

        assert_eq!(
            calculate_range_search_recall(1, &[vec![0, 1, 2, 3, 4],], &[vec![0, 1],]).unwrap(),
            40.0,
            "Returned less results than ground truth"
        );

        let groundtruth: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3, 4], vec![5, 6]];

        let our_results: Vec<Vec<u32>> = vec![vec![0, 1], vec![5, 6, 7, 8, 9]];

        assert_eq!(
            calculate_range_search_recall(2, &groundtruth, &our_results).unwrap(),
            70.0,
            "Combination of both cases"
        );

        assert_eq!(
            calculate_range_search_recall(1, &[vec![0, 1, 2, 3, 4],], &[vec![0, 1, 2, 3, 4],])
                .unwrap(),
            100.0,
            "The result matched the ground truth"
        );

        assert_eq!(
            calculate_range_search_recall(1, &[vec![0, 1, 2, 3, 4],], &[vec![0, 1, 12, 13, 14],])
                .unwrap(),
            40.0,
            "The result partially matched the ground truth"
        );

        assert_eq!(
            calculate_range_search_recall(1, &[vec![0; 0],], &[vec![0, 1, 2, 3, 4],]).unwrap(),
            100.0,
            "The empty ground truth"
        );
    }

    #[test]
    fn test_calculate_filtered_search_recall() {
        let filtered_search_recall =
            calculate_filtered_search_recall(1, None, &[vec![5, 6]], &[vec![5, 6, 7, 8, 9]], 1000)
                .unwrap();
        assert_eq!(
            filtered_search_recall, 40.0,
            "Returned more results than ground truth"
        );

        let range_search_recall =
            calculate_range_search_recall(1, &[vec![5, 6]], &[vec![5, 6, 7, 8, 9]]).unwrap();
        assert_eq!(
            range_search_recall, 100.0,
            "Returned more results than ground truth"
        );

        assert_ne!(
            filtered_search_recall, range_search_recall,
            "This test case showcases the difference between range and filtered search"
        );

        assert_eq!(
            calculate_filtered_search_recall(
                1,
                None,
                &[vec![0, 1, 2, 3, 4],],
                &[vec![0, 1],],
                1000
            )
            .unwrap(),
            40.0,
            "Returned less results than ground truth"
        );

        let groundtruth: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3, 4], vec![5, 6]];

        let our_results: Vec<Vec<u32>> = vec![vec![0, 1], vec![5, 6, 7, 8, 9]];

        assert_eq!(
            calculate_filtered_search_recall(2, None, &groundtruth, &our_results, 1000).unwrap(),
            40.0,
            "Combination of both cases"
        );

        assert_eq!(
            calculate_filtered_search_recall(
                1,
                None,
                &[vec![0, 1, 2, 3, 4],],
                &[vec![0, 1, 2, 3, 4],],
                1000
            )
            .unwrap(),
            100.0,
            "The result matched the ground truth"
        );

        assert_eq!(
            calculate_filtered_search_recall(
                1,
                None,
                &[vec![0, 1, 2, 3, 4],],
                &[vec![0, 1, 12, 13, 14],],
                1000
            )
            .unwrap(),
            40.0,
            "The result partially matched the ground truth"
        );

        assert_eq!(
            calculate_filtered_search_recall(
                1,
                None,
                &[vec![0; 0],],
                &[vec![0, 1, 2, 3, 4],],
                1000
            )
            .unwrap(),
            100.0,
            "The empty ground truth"
        );
    }

    #[test]
    fn test_calculate_filtered_search_recall_with_tie_breaking() {
        // Ground truth with distances
        let gt_distances: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3, 0.3, 0.3], // Ties at index 2, 3, 4
            vec![0.1, 0.2, 0.3, 0.4, 0.5], // No ties
        ];

        let groundtruth: Vec<Vec<u32>> = vec![
            vec![0, 1, 2, 3, 4], // Ground truth IDs
            vec![5, 6, 7, 8, 9],
        ];

        let our_results: Vec<Vec<u32>> = vec![
            vec![0, 1, 3, 2, 4], // Matches all ground truth
            vec![5, 6, 7, 8, 9], // Matches all ground truth
        ];

        // Test with tie-breaking
        assert_eq!(
            calculate_filtered_search_recall(
                2,
                Some(&gt_distances),
                &groundtruth,
                &our_results,
                3 // k_recall
            )
            .unwrap(),
            80.0, //query 0 has 5 matches including ties and 5 returned results => 100% recall
            // query 1 has 3 matches and 5 returned results => 60% recall
            "Tie-breaking should include all tied elements"
        );

        // Test without tie-breaking
        assert_eq!(
            calculate_filtered_search_recall(2, None, &groundtruth, &our_results, 3).unwrap(),
            60.0, // both queries have 3 matches and 5 returned results => 60% recall
            "Without tie-breaking, both queries should match on 3 of 5 elements"
        );

        // Test without tie-breaking and large k
        assert_eq!(
            calculate_filtered_search_recall(2, None, &groundtruth, &our_results, 10).unwrap(),
            100.0,
            "Without tie-breaking and with large k, both queries should match on all elements"
        );
    }

    #[test]
    fn test_calculate_filtered_search_recall_empty_ground_truth() {
        assert_eq!(
            calculate_filtered_search_recall(
                2,
                Some(&[vec![], vec![]]),
                &[vec![], vec![]],
                &[vec![0, 1, 2], vec![5, 6, 7],],
                1
            )
            .unwrap(),
            100.0,
            "Empty ground truth should result in 100% recall"
        );
    }
}
