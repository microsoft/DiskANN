/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Read, path::Path};

use anyhow::Context;
use bit_set::BitSet;
use diskann::utils::IntoUsize;
use diskann_benchmark_runner::utils::datatype::DataType;
use diskann_providers::storage::StorageReadProvider;
use diskann_utils::views::Matrix;
use serde::{Deserialize, Serialize};

pub(crate) struct BinFile<'a>(pub(crate) &'a Path);
pub(crate) struct RunbookFile<'a>(pub(crate) &'a Path);

/// Load a dataset or query set in `.bin` form from disk and return the result as a
/// row-major matrix.
#[inline(never)]
pub(crate) fn load_dataset<T>(path: BinFile<'_>) -> anyhow::Result<Matrix<T>>
where
    T: Copy + bytemuck::Pod,
{
    let (data, num_data, data_dim) = diskann_providers::utils::file_util::load_bin::<T, _>(
        &diskann_providers::storage::FileStorageProvider,
        &path.0.to_string_lossy(),
        0,
    )?;
    Ok(Matrix::try_from(data.into(), num_data, data_dim).map_err(|err| err.as_static())?)
}

/// Helper trait to load a `Matrix<Self>` from source files that potentially have a different
/// type.
pub(crate) trait ConvertingLoad: Sized {
    /// Return an error if the provided `data_type` cannot be loaded and converted to `Self`.
    fn check_converting_load(data_type: DataType) -> anyhow::Result<()>;

    /// Attempt to load the data at `path` as a `Matrix<Self>` assuming the on-disk
    /// representation has the encoding specified by `data_type`.
    ///
    /// If `data_type` is not compatible with `Self`, return an error.
    #[cfg(any(
        feature = "spherical-quantization",
        feature = "minmax-quantization",
        feature = "product-quantization"
    ))]
    fn converting_load(path: BinFile<'_>, data_type: DataType) -> anyhow::Result<Matrix<Self>>;
}

impl ConvertingLoad for f32 {
    fn check_converting_load(data_type: DataType) -> anyhow::Result<()> {
        let compatible = matches!(
            data_type,
            DataType::Float32 | DataType::Float16 | DataType::UInt8 | DataType::Int8
        );
        if compatible {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "data type {:?} is not supported for loading `f32` data",
                data_type
            ))
        }
    }

    #[inline(never)]
    #[cfg(any(
        feature = "spherical-quantization",
        feature = "minmax-quantization",
        feature = "product-quantization"
    ))]
    fn converting_load(path: BinFile<'_>, data_type: DataType) -> anyhow::Result<Matrix<f32>> {
        #[inline(never)]
        fn convert<T, U>(from: diskann_utils::views::MatrixView<T>) -> Matrix<U>
        where
            U: Default + Clone + From<T>,
            T: Copy,
        {
            let mut to = Matrix::new(U::default(), from.nrows(), from.ncols());
            std::iter::zip(to.as_mut_slice().iter_mut(), from.as_slice().iter())
                .for_each(|(t, f)| *t = (*f).into());
            to
        }
        match data_type {
            DataType::Float32 => load_dataset::<f32>(path),
            DataType::Float16 => Ok(convert(load_dataset::<half::f16>(path)?.as_view())),
            DataType::UInt8 => Ok(convert(load_dataset::<u8>(path)?.as_view())),
            DataType::Int8 => Ok(convert(load_dataset::<i8>(path)?.as_view())),
            _ => Err(anyhow::anyhow!(
                "data type {:?} is not supported for loading `f32` data",
                data_type
            )),
        }
    }
}

/// Load a groundtruth set from disk and return the  result as a row-major matrix.
pub(crate) fn load_groundtruth(path: BinFile<'_>) -> anyhow::Result<Matrix<u32>> {
    let provider = diskann_providers::storage::FileStorageProvider;
    let mut file = provider
        .open_reader(&path.0.to_string_lossy())
        .with_context(|| format!("while opening {}", path.0.display()))?;

    let (num_points, dim) = {
        let mut buffer = [0u8; std::mem::size_of::<u32>()];
        file.read_exact(&mut buffer)?;
        let num_points = u32::from_le_bytes(buffer).into_usize();

        file.read_exact(&mut buffer)?;
        let dim = u32::from_le_bytes(buffer).into_usize();
        (num_points, dim)
    };

    let mut groundtruth = Matrix::<u32>::new(0, num_points, dim);
    let groundtruth_slice: &mut [u8] = bytemuck::cast_slice_mut(groundtruth.as_mut_slice());
    file.read_exact(groundtruth_slice)?;
    Ok(groundtruth)
}

/// Load a range groundtruth set from disk
/// Range ground truth consists of a header with the number of points and
/// the total number of range results, then a `num_points` size array detailing
/// the number of results for each point, then the ground truth ids and distances
/// for all points in two contiguous arrays
/// We do not return groundtruth distances because there is no use for them in tie breaking
pub(crate) fn load_range_groundtruth(path: BinFile<'_>) -> anyhow::Result<Vec<Vec<u32>>> {
    let provider = diskann_providers::storage::FileStorageProvider;
    let mut file = provider
        .open_reader(&path.0.to_string_lossy())
        .with_context(|| format!("while opening {}", path.0.display()))?;

    let (num_points, total_results) = {
        let mut buffer = [0u8; std::mem::size_of::<u32>()];
        file.read_exact(&mut buffer)?;
        let num_points = u32::from_le_bytes(buffer).into_usize();

        file.read_exact(&mut buffer)?;
        let total_results = u32::from_le_bytes(buffer).into_usize();
        (num_points, total_results)
    };

    let mut sizes_and_ids: Vec<u32> = vec![0u32; num_points + total_results];
    let result_sizes_slice: &mut [u8] = bytemuck::cast_slice_mut(sizes_and_ids.as_mut_slice());
    file.read_exact(result_sizes_slice)?;

    let mut groundtruth_ids = Vec::<Vec<u32>>::with_capacity(num_points);
    let mut idx = 0;
    let sizes = &sizes_and_ids[..num_points];
    let ids = &sizes_and_ids[num_points..];
    for size in sizes {
        groundtruth_ids.push(ids[idx..idx + *size as usize].to_vec());
        idx += *size as usize;
    }
    Ok(groundtruth_ids)
}

/// The type of operation in a dynamic workload runbook
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum UpdateOperationType {
    Insert,
    Delete,
    Search,
    Replace,
}

/// A workload operation containing start/end range and operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UpdateStage {
    pub(crate) stage_idx: i64,
    #[serde(default)]
    pub(crate) start: Option<usize>,
    #[serde(default)]
    pub(crate) end: Option<usize>,
    #[serde(default)]
    pub(crate) ids_start: Option<usize>,
    #[serde(default)]
    pub(crate) ids_end: Option<usize>,
    #[serde(default)]
    pub(crate) tags_start: Option<usize>,
    #[serde(default)]
    pub(crate) tags_end: Option<usize>,
    pub(crate) operation: UpdateOperationType,
    #[serde(skip)]
    pub(crate) gt_filepath: Option<std::path::PathBuf>,
}
/// A complete dynamic runbook containing phases, max points, and dataset name
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DynamicRunbook {
    pub(crate) phases: Vec<UpdateStage>,
    pub(crate) max_pts: usize,
    pub(crate) dataset_name: String,
}

impl std::fmt::Display for UpdateStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Batch {} - {:?}", self.stage_idx, self.operation)?;

        match self.operation {
            UpdateOperationType::Replace => {
                // For Replace operations, show ids and tags ranges
                match (self.ids_start, self.ids_end, self.tags_start, self.tags_end) {
                    (Some(ids_start), Some(ids_end), Some(tags_start), Some(tags_end)) => {
                        write!(
                            f,
                            " [ids: {}..{}, tags: {}..{}]",
                            ids_start, ids_end, tags_start, tags_end
                        )
                    }
                    _ => write!(f, " [incomplete replace ranges]"),
                }
            }
            _ => {
                // For other operations, show start and end
                match (self.start, self.end) {
                    (Some(start), Some(end)) => write!(f, " [{}..{}]", start, end),
                    (Some(start), None) => write!(f, " [{}..∞]", start),
                    (None, Some(end)) => write!(f, " [0..{}]", end),
                    (None, None) => Ok(()),
                }
            }
        }
    }
}

impl DynamicRunbook {
    /// Create a dynamic workload runbook from a yaml file on disk and return the result as a `DynamicRunbook`.
    /// Compatible with big-ann-benchmark runbook format
    pub(crate) fn new_from_runbook_file(
        path: RunbookFile<'_>,
        dataset_name: String,
        gt_directory: Option<&str>,
    ) -> anyhow::Result<Self> {
        use serde_yaml::Value;

        let provider = diskann_providers::storage::FileStorageProvider;
        let mut file = provider
            .open_reader(&path.0.to_string_lossy())
            .with_context(|| format!("while opening {}", path.0.display()))?;

        // Read the raw content
        let mut content = String::new();
        file.read_to_string(&mut content)
            .with_context(|| format!("while reading file {}", path.0.display()))?;

        // Parse the YAML as a generic value first
        let yaml_data: Value = serde_yaml::from_str(&content)
            .with_context(|| format!("while parsing runbook from {}", path.0.display()))?;

        // Find the specific dataset by name
        let dataset_data = if let Value::Mapping(map) = yaml_data {
            map.into_iter()
                .find(|(key, _)| key.as_str() == Some(&dataset_name))
                .map(|(_, value)| value)
                .ok_or_else(|| anyhow::anyhow!("Dataset '{}' not found in runbook", dataset_name))?
        } else {
            return Err(anyhow::anyhow!("Invalid YAML structure"));
        };

        // Parse the dataset data to extract max_pts and phases
        let mut max_pts = 0usize;
        let mut phases: Vec<UpdateStage> = Vec::new();

        if let Value::Mapping(data_map) = dataset_data {
            for (key, value) in data_map {
                if let Some(key_str) = key.as_str() {
                    if key_str == "max_pts" {
                        max_pts = value.as_u64().unwrap_or(0) as usize;
                    } else if key_str == "gt_url" {
                        // Skip gt_url field - the dataset should be pre-downloaded.
                        continue;
                    } else {
                        return Err(anyhow::anyhow!(
                            "Unknown string key '{}' in dataset '{}'. Expected 'max_pts' or 'gt_url' for string keys.",
                            key_str, dataset_name
                        ));
                    }
                } else if let Some(phase_num) = key.as_i64() {
                    // This is a phase entry with integer key

                    #[derive(Deserialize)]
                    struct TempBatch {
                        #[serde(default)]
                        start: Option<usize>,
                        #[serde(default)]
                        end: Option<usize>,
                        #[serde(default)]
                        ids_start: Option<usize>,
                        #[serde(default)]
                        ids_end: Option<usize>,
                        #[serde(default)]
                        tags_start: Option<usize>,
                        #[serde(default)]
                        tags_end: Option<usize>,
                        operation: UpdateOperationType,
                    }

                    let temp: TempBatch = serde_yaml::from_value(value)
                        .with_context(|| format!("while parsing phase {}", phase_num))?;

                    // Validate that only Replace operations have ids_start, ids_end, tags_start, tags_end
                    if temp.operation != UpdateOperationType::Replace {
                        if temp.ids_start.is_some()
                            || temp.ids_end.is_some()
                            || temp.tags_start.is_some()
                            || temp.tags_end.is_some()
                        {
                            return Err(anyhow::anyhow!(
                                "Phase {}: Only Replace operations can have ids_start, ids_end, tags_start, or tags_end fields. Found {:?} operation with these fields.",
                                phase_num, temp.operation
                            ));
                        }
                    } else {
                        // For Replace operations, all four fields must be present
                        if temp.ids_start.is_none()
                            || temp.ids_end.is_none()
                            || temp.tags_start.is_none()
                            || temp.tags_end.is_none()
                        {
                            return Err(anyhow::anyhow!(
                                "Phase {}: Replace operations must have all four fields: ids_start, ids_end, tags_start, and tags_end. Missing fields detected.",
                                phase_num
                            ));
                        }
                    }

                    // Determine ground truth file path for search operations
                    let gt_filepath = if temp.operation == UpdateOperationType::Search {
                        if let Some(gt_dir) = gt_directory {
                            let gt_path = Self::find_gt_file_for_search_stage(gt_dir, phase_num)
                                .with_context(|| {
                                    format!(
                                        "while finding ground truth file for search stage {}",
                                        phase_num
                                    )
                                })?;
                            Some(gt_path)
                        } else {
                            return Err(anyhow::anyhow!(
                                "Ground truth directory must be provided for search stage {}",
                                phase_num
                            ));
                        }
                    } else {
                        None
                    };

                    let phase = UpdateStage {
                        stage_idx: phase_num,
                        start: temp.start,
                        end: temp.end,
                        ids_start: temp.ids_start,
                        ids_end: temp.ids_end,
                        tags_start: temp.tags_start,
                        tags_end: temp.tags_end,
                        operation: temp.operation,
                        gt_filepath,
                    };
                    phases.push(phase);
                }
            }
        }

        // Sort phases by their stage_idx and collect into a vector
        phases.sort_by_key(|phase| phase.stage_idx);

        let mut runbook = Self {
            phases,
            max_pts,
            dataset_name,
        };

        // Verify max_pts constraint is respected throughout the runbook
        runbook.verify_and_fix_max_pts_constraint()?;

        Ok(runbook)
    }

    /// Find groundtruth file for a specific step in the given directory
    /// Looks for files matching the pattern: step{stage_idx}.gt{\d}
    fn find_gt_file_for_search_stage(
        gt_directory: &str,
        stage_idx: i64,
    ) -> anyhow::Result<std::path::PathBuf> {
        let gt_dir = std::path::Path::new(gt_directory);

        if !gt_dir.exists() {
            return Err(anyhow::anyhow!(
                "Ground truth directory does not exist: {}",
                gt_directory
            ));
        }

        let step_pattern = format!("step{}.gt", stage_idx);

        // Read directory and find files matching the pattern
        let entries = std::fs::read_dir(gt_dir)
            .with_context(|| format!("Failed to read ground truth directory: {}", gt_directory))?;

        let mut matching_files = Vec::new();

        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            // Check if filename starts with step{stage_idx}.gt{integer}
            if file_name_str.starts_with(&step_pattern) {
                // Get the part after the pattern
                let suffix = &file_name_str[step_pattern.len()..];
                // Only match if suffix is empty or contains only digits, ignoring .data and .tags
                if suffix.is_empty() || suffix.chars().all(|c| c.is_ascii_digit()) {
                    matching_files.push(entry.path());
                }
            }
        }

        if matching_files.is_empty() {
            return Err(anyhow::anyhow!(
                "No ground truth file found for step {} in directory: {}. Expected file pattern: step{}.gt*",
                stage_idx,
                gt_directory,
                stage_idx
            ));
        }

        if matching_files.len() > 1 {
            return Err(anyhow::anyhow!(
                "Multiple ground truth files found for step {} in directory: {}. Found files: {:?}",
                stage_idx,
                gt_directory,
                matching_files
            ));
        }

        Ok(matching_files.into_iter().next().unwrap())
    }

    /// Verify that the max_pts constraint is respected throughout the runbook.
    /// For inserts, we add the range size to current points.
    /// For deletes, we subtract the range size from current points.
    /// For replaces, the number of points remains the same.
    /// If the current point count would exceed max_pts at any stage, print a warning and automatically
    /// adjust max_pts to the calculated maximum.
    /// Also verifies that
    /// 1. Replace operations only target tags that are currently in the index.
    /// 2. Replace operations does not insert vectors with the same ID under multiple tags
    fn verify_and_fix_max_pts_constraint(&mut self) -> anyhow::Result<()> {
        let mut current_pts = 0usize;
        let mut calculated_max_pts = 0usize;
        let mut active_id_ranges: Vec<(usize, usize)> = Vec::new(); // Track (start, end) ranges of active IDs
        let mut active_tag_ranges: Vec<(usize, usize)> = Vec::new(); // Track (start, end) ranges of active TAGs

        for phase in &self.phases {
            match phase.operation {
                UpdateOperationType::Insert => {
                    let (start, end) = self.get_operation_range(phase)?;
                    let range_size = end - start;
                    current_pts += range_size;
                    calculated_max_pts = calculated_max_pts.max(current_pts);

                    // Add this range to both active ID and TAG ranges (for inserts, they're the same)
                    active_id_ranges.push((start, end));
                    active_tag_ranges.push((start, end));
                }
                UpdateOperationType::Delete => {
                    let (start, end) = self.get_operation_range(phase)?;
                    let range_size = end - start;

                    if current_pts < range_size {
                        return Err(anyhow::anyhow!(
                            "Stage {}: Delete operation would delete {} points, but only {} points are currently in the index. Delete range: {}..{} (size: {})\n\
                            Currently active ID ranges: {:?}",
                            phase.stage_idx, range_size, current_pts, start, end, range_size, active_id_ranges
                        ));
                    }

                    current_pts -= range_size;

                    // Handle partial overlaps correctly when removing ID ranges
                    let mut new_active_id_ranges = Vec::new();
                    for &(active_start, active_end) in &active_id_ranges {
                        // Check for overlap
                        let overlap_start = start.max(active_start);
                        let overlap_end = end.min(active_end);

                        if overlap_start < overlap_end {
                            // There is an overlap, we need to split the range
                            // Add the part before the deleted range (if any)
                            if active_start < overlap_start {
                                new_active_id_ranges.push((active_start, overlap_start));
                            }
                            // Add the part after the deleted range (if any)
                            if overlap_end < active_end {
                                new_active_id_ranges.push((overlap_end, active_end));
                            }
                            // The overlapping part is deleted, so we don't add it
                        } else {
                            // No overlap, keep the range as is
                            new_active_id_ranges.push((active_start, active_end));
                        }
                    }
                    active_id_ranges = new_active_id_ranges;

                    // Handle partial overlaps correctly when removing TAG ranges (same logic)
                    let mut new_active_tag_ranges = Vec::new();
                    for &(active_start, active_end) in &active_tag_ranges {
                        // Check for overlap
                        let overlap_start = start.max(active_start);
                        let overlap_end = end.min(active_end);

                        if overlap_start < overlap_end {
                            // There is an overlap, we need to split the range
                            // Add the part before the deleted range (if any)
                            if active_start < overlap_start {
                                new_active_tag_ranges.push((active_start, overlap_start));
                            }
                            // Add the part after the deleted range (if any)
                            if overlap_end < active_end {
                                new_active_tag_ranges.push((overlap_end, active_end));
                            }
                            // The overlapping part is deleted, so we don't add it
                        } else {
                            // No overlap, keep the range as is
                            new_active_tag_ranges.push((active_start, active_end));
                        }
                    }
                    active_tag_ranges = new_active_tag_ranges;
                }
                UpdateOperationType::Replace => {
                    // Replace operations don't change the total number of points
                    // but we should verify the ranges are valid
                    let (ids_start, ids_end) = match (phase.ids_start, phase.ids_end) {
                        (Some(ids_start), Some(ids_end)) => (ids_start, ids_end),
                        _ => {
                            return Err(anyhow::anyhow!(
                                "Stage {}: Replace operation missing ids_start or ids_end",
                                phase.stage_idx
                            ))
                        }
                    };
                    let (tags_start, tags_end) = match (phase.tags_start, phase.tags_end) {
                        (Some(tags_start), Some(tags_end)) => (tags_start, tags_end),
                        _ => {
                            return Err(anyhow::anyhow!(
                                "Stage {}: Replace operation missing tags_start or tags_end",
                                phase.stage_idx
                            ))
                        }
                    };

                    let ids_range_size = ids_end - ids_start;
                    let tags_range_size = tags_end - tags_start;

                    if ids_range_size != tags_range_size {
                        return Err(anyhow::anyhow!(
                            "Stage {}: Replace operation has mismatched range sizes. IDs range: {}..{} (size: {}), Tags range: {}..{} (size: {})",
                            phase.stage_idx, ids_start, ids_end, ids_range_size, tags_start, tags_end, tags_range_size
                        ));
                    }

                    if tags_range_size > current_pts {
                        return Err(anyhow::anyhow!(
                            "Stage {}: Replace operation tries to replace {} points, but only {} points are currently in the index\n\
                            Currently active ID ranges: {:?}",
                            phase.stage_idx, tags_range_size, current_pts, active_id_ranges
                        ));
                    }

                    // Check that the IDs being used for replacement are NOT already in the index
                    let mut conflicting_ids = 0usize;
                    for &(active_start, active_end) in &active_id_ranges {
                        // Calculate overlap between the new IDs range and this active range
                        let overlap_start = ids_start.max(active_start);
                        let overlap_end = ids_end.min(active_end);
                        if overlap_start < overlap_end {
                            conflicting_ids += overlap_end - overlap_start;
                        }
                    }

                    if conflicting_ids > 0 {
                        return Err(anyhow::anyhow!(
                            "Stage {}: Replace operation uses ID range {}..{} (size: {}), but {} of these IDs are already in the index.\n\
                            Currently active ID ranges: {:?}\n\
                            Replace operation cannot use IDs that are already present in the index. Use different ID ranges that don't conflict with existing data.",
                            phase.stage_idx, ids_start, ids_end, ids_range_size, conflicting_ids, active_id_ranges
                        ));
                    }

                    // Check that the tags being replaced ARE actually in the index
                    let mut covered_tags = 0usize;
                    for &(active_start, active_end) in &active_tag_ranges {
                        // Calculate overlap between the tags range and this active range
                        let overlap_start = tags_start.max(active_start);
                        let overlap_end = tags_end.min(active_end);
                        if overlap_start < overlap_end {
                            covered_tags += overlap_end - overlap_start;
                        }
                    }

                    if covered_tags < tags_range_size {
                        let uncovered_tags = tags_range_size - covered_tags;
                        return Err(anyhow::anyhow!(
                            "Stage {}: Replace operation targets tag range {}..{} (size: {}), but {} of these tags are not currently in the index.\n\
                            Currently active TAG ranges: {:?}\n\
                            Replace operation can only replace tags that exist in the index.",
                            phase.stage_idx, tags_start, tags_end, tags_range_size, uncovered_tags, active_tag_ranges
                        ));
                    }

                    // Update active ID ranges: remove the old tags range and add the new IDs range
                    // Note: TAG ranges remain unchanged during replace operations - tags stay active!
                    // First, remove the tags being replaced from ID ranges
                    active_id_ranges.retain(|(active_start, active_end)| {
                        // Keep ranges that don't overlap with the tags being replaced
                        *active_end <= tags_start || *active_start >= tags_end
                    });

                    // Then add the new IDs range
                    active_id_ranges.push((ids_start, ids_end));
                }
                UpdateOperationType::Search => {
                    // Search operations don't affect the point count or ID ranges
                    continue;
                }
            }
        }

        // Update max_pts if we calculated a higher value than the original
        if calculated_max_pts > self.max_pts {
            eprintln!(
                "WARNING: Calculated max_pts ({}) exceeds original max_pts ({}). Automatically adjusting max_pts to {}.",
                calculated_max_pts, self.max_pts, calculated_max_pts
            );
            self.max_pts = calculated_max_pts;
        }

        Ok(())
    }

    /// Helper method to get operation range for Insert/Delete operations
    fn get_operation_range(&self, update_stage: &UpdateStage) -> anyhow::Result<(usize, usize)> {
        match update_stage.operation {
            UpdateOperationType::Insert | UpdateOperationType::Delete => {
                match (update_stage.start, update_stage.end) {
                    (Some(start), Some(end)) => Ok((start, end)),
                    _ => Err(anyhow::anyhow!(
                        "Stage {}: {:?} operation missing start or end",
                        update_stage.stage_idx,
                        update_stage.operation
                    )),
                }
            }
            UpdateOperationType::Replace => match (update_stage.ids_start, update_stage.ids_end) {
                (Some(ids_start), Some(ids_end)) => Ok((ids_start, ids_end)),
                _ => Err(anyhow::anyhow!(
                    "Stage {}: Replace operation missing ids_start or ids_end",
                    update_stage.stage_idx
                )),
            },
            UpdateOperationType::Search => Err(anyhow::anyhow!(
                "Stage {}: Search operations don't have ranges",
                update_stage.stage_idx
            )),
        }
    }
}

// Helper struct for serializing BitSet as Vec<u8> (raw storage)
#[derive(Serialize, Deserialize)]
struct SerializableBitSet(Vec<u8>);

impl From<&BitSet> for SerializableBitSet {
    fn from(bs: &BitSet) -> Self {
        SerializableBitSet(bs.get_ref().to_bytes())
    }
}

impl From<SerializableBitSet> for BitSet {
    fn from(val: SerializableBitSet) -> Self {
        BitSet::from_bytes(&val.0)
    }
}
