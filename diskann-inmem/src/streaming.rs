/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashSet,
    fs,
    io::{BufWriter, ErrorKind, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use thiserror::Error;

use crate::{
    Provider, Tag128,
    layers::Full,
    provider::{Config, Id},
};

/// External tag representation supported by C++ streaming snapshots.
pub trait StreamingTag: Id + Copy + Default {
    /// Serialized width in bytes.
    const WIDTH: usize;
    /// Decode one little-endian tag.
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, StreamingSnapshotError>;
    /// Encode one little-endian tag.
    fn write_le_bytes(self, bytes: &mut [u8]);
    /// Return whether this is the frozen-point placeholder.
    fn is_zero(self) -> bool;
}

impl StreamingTag for u32 {
    const WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, StreamingSnapshotError> {
        Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| {
            StreamingSnapshotError::Invalid("u32 tag width is invalid".into())
        })?))
    }
    fn is_zero(self) -> bool {
        self == 0
    }
    fn write_le_bytes(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
}

impl StreamingTag for u64 {
    const WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, StreamingSnapshotError> {
        Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
            StreamingSnapshotError::Invalid("u64 tag width is invalid".into())
        })?))
    }
    fn is_zero(self) -> bool {
        self == 0
    }
    fn write_le_bytes(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
}

impl StreamingTag for Tag128 {
    const WIDTH: usize = 16;
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, StreamingSnapshotError> {
        let low = bytes
            .get(..8)
            .and_then(|value| value.try_into().ok())
            .ok_or_else(|| StreamingSnapshotError::Invalid("u128 tag width is invalid".into()))?;
        let high = bytes
            .get(8..)
            .and_then(|value| value.try_into().ok())
            .ok_or_else(|| StreamingSnapshotError::Invalid("u128 tag width is invalid".into()))?;
        Ok(Self {
            low: u64::from_le_bytes(low),
            high: u64::from_le_bytes(high),
        })
    }
    fn is_zero(self) -> bool {
        self.low == 0 && self.high == 0
    }
    fn write_le_bytes(self, bytes: &mut [u8]) {
        bytes[..8].copy_from_slice(&self.low.to_le_bytes());
        bytes[8..].copy_from_slice(&self.high.to_le_bytes());
    }
}

/// Configuration for loading a mutable C++ streaming snapshot.
#[derive(Debug, Clone)]
pub struct StreamingSnapshotConfig {
    /// Expected UINT8 vector dimension.
    pub dim: usize,
    /// Additional writable capacity as a percentage of active points.
    pub max_insert_percentage: f32,
    /// Minimum graph degree requested by the caller.
    pub graph_degree: usize,
}

/// Loaded mutable provider and snapshot metadata.
#[derive(Debug)]
pub struct StreamingSnapshot<M: StreamingTag> {
    /// Provider owning vectors, adjacency, and typed external tags.
    pub provider: Provider<Full<u8>, M>,
    /// Effective maximum graph degree.
    pub max_degree: usize,
    /// Remapped frozen-point internal ID.
    pub frozen_internal_id: u32,
    /// Number of active external tags.
    pub active_count: usize,
    /// Writable provider capacity.
    pub capacity: usize,
}

/// Errors while parsing or restoring a streaming snapshot.
#[derive(Debug, Error)]
pub enum StreamingSnapshotError {
    /// A snapshot file could not be read.
    #[error("failed to read {kind} snapshot {path}: {source}")]
    Read {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Snapshot contents or loader configuration were invalid.
    #[error("{0}")]
    Invalid(String),
    /// The mutable provider could not be restored.
    #[error("failed to restore mutable provider: {0}")]
    Provider(#[from] crate::provider::ProviderError),
    /// A temporary snapshot file could not be written or published.
    #[error("failed to write {kind} snapshot {path}: {source}")]
    Write {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Snapshot publication failed; rollback details are included when restoration was incomplete.
    #[error("{0}")]
    Transaction(String),
}

struct ParsedGraph {
    adjacency: Vec<Vec<u32>>,
    max_degree: usize,
    frozen_index: usize,
}

/// Load C++ UINT8 streaming graph/data/tag files into a mutable typed provider.
pub fn load_streaming_snapshot<M: StreamingTag>(
    layer: Full<u8>,
    config: StreamingSnapshotConfig,
    graph_path: impl AsRef<Path>,
    data_path: impl AsRef<Path>,
    tag_path: impl AsRef<Path>,
) -> Result<StreamingSnapshot<M>, StreamingSnapshotError> {
    let vectors = parse_vectors(data_path.as_ref(), config.dim)?;
    if vectors.len() < 2 {
        return Err(StreamingSnapshotError::Invalid(
            "streaming snapshot requires a writable and a frozen point".into(),
        ));
    }
    let graph = parse_graph(graph_path.as_ref(), vectors.len())?;
    let tags = parse_tags::<M>(tag_path.as_ref(), vectors.len(), graph.frozen_index)?;
    let active_count = vectors.len() - 1;
    let additional =
        ((active_count as f64) * f64::from(config.max_insert_percentage) / 100.0).ceil() as usize;
    let capacity = active_count
        .checked_add(additional)
        .ok_or_else(|| StreamingSnapshotError::Invalid("streaming capacity overflowed".into()))?;
    if capacity > u32::MAX as usize {
        return Err(StreamingSnapshotError::Invalid(
            "streaming capacity exceeds u32".into(),
        ));
    }
    let max_degree = graph.max_degree.max(config.graph_degree);
    let provider = Provider::new_from_snapshot(
        layer,
        Config::new(capacity, max_degree),
        &vectors,
        &tags,
        &graph.adjacency,
        graph.frozen_index,
    )?;
    let frozen_internal_id = provider.frozen_ids().start;
    Ok(StreamingSnapshot {
        provider,
        max_degree,
        frozen_internal_id,
        active_count,
        capacity,
    })
}

static TEMP_ID: AtomicU64 = AtomicU64::new(1);
#[cfg(test)]
thread_local! {
    static FAIL_PUBLISH_AT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Save a compact C++-compatible streaming snapshot.
///
/// The active tag set is captured first. Inserts outside that set are excluded. If a selected
/// tag disappears while it is captured, this function fails before publication.
pub fn save_streaming_snapshot<M: StreamingTag>(
    provider: &Provider<Full<u8>, M>,
    graph_path: impl AsRef<Path>,
    data_path: impl AsRef<Path>,
    tag_path: impl AsRef<Path>,
) -> Result<(), StreamingSnapshotError> {
    let state = provider.snapshot_state()?;
    if state.rows.is_empty() {
        return Err(StreamingSnapshotError::Invalid(
            "frozen-only snapshots are not reloadable; at least one active row is required".into(),
        ));
    }

    let finals = [
        normalize_destination(graph_path.as_ref())?,
        normalize_destination(data_path.as_ref())?,
        normalize_destination(tag_path.as_ref())?,
    ];
    let mut used = HashSet::new();
    for path in &finals {
        if !used.insert(path_key(path)) {
            return Err(StreamingSnapshotError::Invalid(
                "snapshot output paths alias or collide".into(),
            ));
        }
    }
    let temps = [
        unique_helper(&finals[0], "tmp", &mut used)?,
        unique_helper(&finals[1], "tmp", &mut used)?,
        unique_helper(&finals[2], "tmp", &mut used)?,
    ];
    let backups = [
        unique_helper(&finals[0], "bak", &mut used)?,
        unique_helper(&finals[1], "bak", &mut used)?,
        unique_helper(&finals[2], "bak", &mut used)?,
    ];

    let write_result = (|| {
        write_data(&temps[1], &state)?;
        write_tags(&temps[2], &state)?;
        write_graph(&temps[0], provider.max_degree(), &state)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        cleanup_paths(&temps);
        return Err(error);
    }

    let mut backed_up = 0;
    for index in 0..finals.len() {
        if finals[index].exists() {
            if let Err(source) = fs::rename(&finals[index], &backups[index]) {
                return Err(rollback_error(
                    format!("failed to back up {}: {source}", finals[index].display()),
                    &finals,
                    &temps,
                    &backups,
                    backed_up,
                    0,
                ));
            }
            backed_up = index + 1;
        }
    }

    let mut published = 0;
    for index in 0..finals.len() {
        let result = injected_publish_failure(index)
            .and_then(|()| fs::rename(&temps[index], &finals[index]));
        if let Err(source) = result {
            return Err(rollback_error(
                format!("failed to publish {}: {source}", finals[index].display()),
                &finals,
                &temps,
                &backups,
                backed_up,
                published,
            ));
        }
        published += 1;
    }

    let mut cleanup_errors = Vec::new();
    for backup in backups.iter().take(backed_up) {
        remove_record(backup, "delete committed backup", &mut cleanup_errors);
    }
    if cleanup_errors.is_empty() {
        Ok(())
    } else {
        Err(StreamingSnapshotError::Transaction(format!(
            "snapshot published but backup cleanup failed: {}",
            cleanup_errors.join("; ")
        )))
    }
}

fn normalize_destination(path: &Path) -> Result<PathBuf, StreamingSnapshotError> {
    let file_name = path.file_name().ok_or_else(|| {
        StreamingSnapshotError::Invalid(format!(
            "snapshot path has no file name: {}",
            path.display()
        ))
    })?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|source| StreamingSnapshotError::Write {
        kind: "normalize",
        path: parent.to_path_buf(),
        source,
    })?;
    Ok(parent.join(file_name))
}

fn path_key(path: &Path) -> String {
    let value = path.to_string_lossy().into_owned();
    if cfg!(windows) {
        value.to_lowercase()
    } else {
        value
    }
}

fn unique_helper(
    final_path: &Path,
    kind: &str,
    used: &mut HashSet<String>,
) -> Result<PathBuf, StreamingSnapshotError> {
    let name = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| StreamingSnapshotError::Invalid("snapshot file name is not UTF-8".into()))?;
    for _ in 0..1024 {
        let nonce = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let candidate = final_path.with_file_name(format!(
            ".{name}.{}.{}.diskann.{kind}",
            std::process::id(),
            nonce
        ));
        if used.insert(path_key(&candidate)) && !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(StreamingSnapshotError::Invalid(
        "could not reserve a unique snapshot helper path".into(),
    ))
}

fn injected_publish_failure(index: usize) -> std::io::Result<()> {
    #[cfg(test)]
    if FAIL_PUBLISH_AT.get() == index {
        return Err(std::io::Error::other("injected publish failure"));
    }
    let _ = index;
    Ok(())
}

fn rollback_error(
    primary: String,
    finals: &[PathBuf; 3],
    temps: &[PathBuf; 3],
    backups: &[PathBuf; 3],
    backed_up: usize,
    published: usize,
) -> StreamingSnapshotError {
    let mut errors = Vec::new();
    for final_path in finals.iter().take(published) {
        remove_record(final_path, "remove partial publication", &mut errors);
    }
    for index in 0..backed_up {
        if backups[index].exists()
            && let Err(error) = fs::rename(&backups[index], &finals[index])
        {
            errors.push(format!(
                "restore {} from {}: {error}",
                finals[index].display(),
                backups[index].display()
            ));
        }
    }
    for temp in temps {
        remove_record(temp, "remove temporary", &mut errors);
    }
    if errors.is_empty() {
        StreamingSnapshotError::Transaction(primary)
    } else {
        StreamingSnapshotError::Transaction(format!(
            "{primary}; rollback errors: {}",
            errors.join("; ")
        ))
    }
}

fn remove_record(path: &Path, action: &str, errors: &mut Vec<String>) {
    if let Err(error) = fs::remove_file(path)
        && error.kind() != ErrorKind::NotFound
    {
        errors.push(format!("{action} {}: {error}", path.display()));
    }
}

fn cleanup_paths<P: AsRef<Path>>(paths: &[P]) {
    for path in paths {
        let _ = fs::remove_file(path.as_ref());
    }
}

fn writer(path: &Path, kind: &'static str) -> Result<BufWriter<fs::File>, StreamingSnapshotError> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map(BufWriter::new)
        .map_err(|source| StreamingSnapshotError::Write {
            kind,
            path: path.to_path_buf(),
            source,
        })
}

fn write_all(
    writer: &mut impl Write,
    bytes: &[u8],
    path: &Path,
    kind: &'static str,
) -> Result<(), StreamingSnapshotError> {
    writer
        .write_all(bytes)
        .map_err(|source| StreamingSnapshotError::Write {
            kind,
            path: path.to_path_buf(),
            source,
        })
}

fn write_data<M: StreamingTag>(
    path: &Path,
    state: &crate::provider::SnapshotState<M>,
) -> Result<(), StreamingSnapshotError> {
    let mut output = writer(path, "data")?;
    let count = u32::try_from(state.rows.len() + 1)
        .map_err(|_| StreamingSnapshotError::Invalid("too many snapshot points".into()))?;
    let dim = u32::try_from(state.frozen.vector.len())
        .map_err(|_| StreamingSnapshotError::Invalid("snapshot dimension exceeds u32".into()))?;
    write_all(&mut output, &count.to_le_bytes(), path, "data")?;
    write_all(&mut output, &dim.to_le_bytes(), path, "data")?;
    for row in &state.rows {
        if row.vector.len() != dim as usize {
            return Err(StreamingSnapshotError::Invalid(
                "snapshot vector dimensions changed".into(),
            ));
        }
        write_all(&mut output, &row.vector, path, "data")?;
    }
    write_all(&mut output, &state.frozen.vector, path, "data")?;
    output
        .flush()
        .map_err(|source| StreamingSnapshotError::Write {
            kind: "data",
            path: path.to_path_buf(),
            source,
        })
}

fn write_tags<M: StreamingTag>(
    path: &Path,
    state: &crate::provider::SnapshotState<M>,
) -> Result<(), StreamingSnapshotError> {
    let mut output = writer(path, "tag")?;
    let count = u32::try_from(state.rows.len() + 1)
        .map_err(|_| StreamingSnapshotError::Invalid("too many snapshot tags".into()))?;
    write_all(&mut output, &count.to_le_bytes(), path, "tag")?;
    write_all(&mut output, &1u32.to_le_bytes(), path, "tag")?;
    let mut bytes = vec![0u8; M::WIDTH];
    for row in &state.rows {
        row.tag.write_le_bytes(&mut bytes);
        write_all(&mut output, &bytes, path, "tag")?;
    }
    bytes.fill(0);
    write_all(&mut output, &bytes, path, "tag")?;
    output
        .flush()
        .map_err(|source| StreamingSnapshotError::Write {
            kind: "tag",
            path: path.to_path_buf(),
            source,
        })
}

fn write_graph<M: StreamingTag>(
    path: &Path,
    max_degree: usize,
    state: &crate::provider::SnapshotState<M>,
) -> Result<(), StreamingSnapshotError> {
    let rows = state
        .rows
        .iter()
        .map(|row| &row.neighbors)
        .chain(std::iter::once(&state.frozen.neighbors));
    let adjacency_size = rows.clone().try_fold(0usize, |total, neighbors| {
        neighbors
            .len()
            .checked_add(1)
            .and_then(|count| count.checked_mul(4))
            .and_then(|bytes| total.checked_add(bytes))
    });
    let total_size = adjacency_size
        .and_then(|size| size.checked_add(24))
        .ok_or_else(|| StreamingSnapshotError::Invalid("graph size overflow".into()))?;
    let frozen_index = u32::try_from(state.rows.len())
        .map_err(|_| StreamingSnapshotError::Invalid("too many snapshot points".into()))?;
    let max_degree = u32::try_from(max_degree)
        .map_err(|_| StreamingSnapshotError::Invalid("graph degree exceeds u32".into()))?;
    let mut output = writer(path, "graph")?;
    write_all(
        &mut output,
        &(total_size as u64).to_le_bytes(),
        path,
        "graph",
    )?;
    write_all(&mut output, &max_degree.to_le_bytes(), path, "graph")?;
    write_all(&mut output, &frozen_index.to_le_bytes(), path, "graph")?;
    write_all(&mut output, &1u64.to_le_bytes(), path, "graph")?;
    for neighbors in rows {
        let degree = u32::try_from(neighbors.len())
            .map_err(|_| StreamingSnapshotError::Invalid("graph degree exceeds u32".into()))?;
        write_all(&mut output, &degree.to_le_bytes(), path, "graph")?;
        for neighbor in neighbors {
            write_all(&mut output, &neighbor.to_le_bytes(), path, "graph")?;
        }
    }
    output
        .flush()
        .map_err(|source| StreamingSnapshotError::Write {
            kind: "graph",
            path: path.to_path_buf(),
            source,
        })
}
fn read_file(path: &Path, kind: &'static str) -> Result<Vec<u8>, StreamingSnapshotError> {
    fs::read(path).map_err(|source| StreamingSnapshotError::Read {
        kind,
        path: path.to_path_buf(),
        source,
    })
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, StreamingSnapshotError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| StreamingSnapshotError::Invalid("file offset overflow".into()))?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| StreamingSnapshotError::Invalid("truncated snapshot".into()))?;
    *offset = end;
    Ok(u32::from_le_bytes(value.try_into().map_err(|_| {
        StreamingSnapshotError::Invalid("invalid u32 field".into())
    })?))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, StreamingSnapshotError> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| StreamingSnapshotError::Invalid("file offset overflow".into()))?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| StreamingSnapshotError::Invalid("truncated snapshot".into()))?;
    *offset = end;
    Ok(u64::from_le_bytes(value.try_into().map_err(|_| {
        StreamingSnapshotError::Invalid("invalid u64 field".into())
    })?))
}

fn parse_vectors(path: &Path, expected_dim: usize) -> Result<Vec<Vec<u8>>, StreamingSnapshotError> {
    let bytes = read_file(path, "data")?;
    let mut offset = 0;
    let count = read_u32(&bytes, &mut offset)? as usize;
    let dim = read_u32(&bytes, &mut offset)? as usize;
    if dim != expected_dim {
        return Err(StreamingSnapshotError::Invalid(
            "configured dimension does not match the data snapshot".into(),
        ));
    }
    let payload = count
        .checked_mul(dim)
        .ok_or_else(|| StreamingSnapshotError::Invalid("data size overflow".into()))?;
    if bytes.len() != offset + payload {
        return Err(StreamingSnapshotError::Invalid(
            "data snapshot length does not match its header".into(),
        ));
    }
    Ok(bytes[offset..]
        .chunks_exact(dim)
        .map(<[u8]>::to_vec)
        .collect())
}

fn parse_graph(path: &Path, expected_points: usize) -> Result<ParsedGraph, StreamingSnapshotError> {
    let bytes = read_file(path, "graph")?;
    let mut offset = 0;
    let file_size = read_u64(&bytes, &mut offset)? as usize;
    let max_degree = read_u32(&bytes, &mut offset)? as usize;
    let frozen_index = read_u32(&bytes, &mut offset)? as usize;
    let frozen_count = read_u64(&bytes, &mut offset)? as usize;
    if file_size != bytes.len() {
        return Err(StreamingSnapshotError::Invalid(
            "graph snapshot size header is invalid".into(),
        ));
    }
    if frozen_count != 1 || frozen_index >= expected_points {
        return Err(StreamingSnapshotError::Invalid(
            "streaming snapshot must contain one valid frozen point".into(),
        ));
    }
    let mut adjacency = Vec::with_capacity(expected_points);
    while offset < bytes.len() {
        let degree = read_u32(&bytes, &mut offset)? as usize;
        if degree > max_degree {
            return Err(StreamingSnapshotError::Invalid(
                "graph adjacency exceeds the declared maximum degree".into(),
            ));
        }
        let mut neighbors = Vec::with_capacity(degree);
        for _ in 0..degree {
            let neighbor = read_u32(&bytes, &mut offset)?;
            if neighbor as usize >= expected_points {
                return Err(StreamingSnapshotError::Invalid(
                    "graph adjacency contains an out-of-range ID".into(),
                ));
            }
            neighbors.push(neighbor);
        }
        adjacency.push(neighbors);
    }
    if adjacency.len() != expected_points {
        return Err(StreamingSnapshotError::Invalid(
            "graph and data snapshot point counts differ".into(),
        ));
    }
    Ok(ParsedGraph {
        adjacency,
        max_degree,
        frozen_index,
    })
}

fn parse_tags<M: StreamingTag>(
    path: &Path,
    expected_points: usize,
    frozen_index: usize,
) -> Result<Vec<Option<M>>, StreamingSnapshotError> {
    let bytes = read_file(path, "tag")?;
    let mut offset = 0;
    let count = read_u32(&bytes, &mut offset)? as usize;
    let dimensions = read_u32(&bytes, &mut offset)? as usize;
    if dimensions != 1 {
        return Err(StreamingSnapshotError::Invalid(
            "tag snapshot dimension must be one".into(),
        ));
    }
    let payload = count
        .checked_mul(M::WIDTH)
        .ok_or_else(|| StreamingSnapshotError::Invalid("tag size overflow".into()))?;
    if bytes.len() != offset + payload {
        return Err(StreamingSnapshotError::Invalid(
            "tag snapshot length does not match its header".into(),
        ));
    }
    if count != expected_points && count + 1 != expected_points {
        return Err(StreamingSnapshotError::Invalid(
            "tag and data snapshot point counts differ".into(),
        ));
    }
    let raw: Vec<M> = bytes[offset..]
        .chunks_exact(M::WIDTH)
        .map(M::from_le_bytes)
        .collect::<Result<_, _>>()?;
    let mut tags = vec![None; expected_points];
    if count == expected_points {
        if !raw[frozen_index].is_zero() {
            return Err(StreamingSnapshotError::Invalid(
                "frozen-point tag placeholder must be zero".into(),
            ));
        }
        for (index, tag) in raw.into_iter().enumerate() {
            if index != frozen_index {
                tags[index] = Some(tag);
            }
        }
    } else {
        let mut source = raw.into_iter();
        for (index, target) in tags.iter_mut().enumerate() {
            if index != frozen_index {
                *target = source.next().map(Some).ok_or_else(|| {
                    StreamingSnapshotError::Invalid("missing snapshot tag".into())
                })?;
            }
        }
    }
    Ok(tags)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use diskann::provider::DataProvider;
    use diskann_vector::distance::Metric;

    use super::*;
    use crate::Context;

    #[test]
    fn loads_cpp_snapshot_with_frozen_placeholder_and_wide_tags() {
        let temp = tempfile::tempdir().unwrap();
        let data_path = temp.path().join("index.data");
        let graph_path = temp.path().join("index");
        let tag_path = temp.path().join("index.tags");

        let mut data = fs::File::create(&data_path).unwrap();
        data.write_all(&3u32.to_le_bytes()).unwrap();
        data.write_all(&4u32.to_le_bytes()).unwrap();
        data.write_all(&[0; 12]).unwrap();

        let adjacency = [[1u32, 2], [0, 2], [0, 1]];
        let size = 24 + adjacency.len() * 12;
        let mut graph = fs::File::create(&graph_path).unwrap();
        graph.write_all(&(size as u64).to_le_bytes()).unwrap();
        graph.write_all(&2u32.to_le_bytes()).unwrap();
        graph.write_all(&2u32.to_le_bytes()).unwrap();
        graph.write_all(&1u64.to_le_bytes()).unwrap();
        for neighbors in adjacency {
            graph.write_all(&2u32.to_le_bytes()).unwrap();
            for neighbor in neighbors {
                graph.write_all(&neighbor.to_le_bytes()).unwrap();
            }

        }

        let first = Tag128 {
            low: 0xaabb_ccdd,
            high: 1,
        };
        let second = Tag128 {
            low: 0xaabb_ccdd,
            high: 2,
        };
        let mut tags = fs::File::create(&tag_path).unwrap();
        tags.write_all(&3u32.to_le_bytes()).unwrap();
        tags.write_all(&1u32.to_le_bytes()).unwrap();
        for tag in [first, second, Tag128::default()] {
            tags.write_all(&tag.low.to_le_bytes()).unwrap();
            tags.write_all(&tag.high.to_le_bytes()).unwrap();
        }

        let loaded = load_streaming_snapshot::<Tag128>(
            Full::new(4, Metric::L2),
            StreamingSnapshotConfig {
                dim: 4,
                max_insert_percentage: 100.0,
                graph_degree: 2,
            },
            graph_path,
            data_path,
            tag_path,
        )
        .unwrap();
        assert_eq!(loaded.active_count, 2);
        assert_eq!(loaded.capacity, 4);
        assert_ne!(
            loaded.provider.to_internal_id(&Context, &first).unwrap(),
            loaded.provider.to_internal_id(&Context, &second).unwrap()
        );

        let saved_graph = temp.path().join("saved");
        let saved_data = temp.path().join("saved.data");
        let saved_tags = temp.path().join("saved.tags");
        save_streaming_snapshot(&loaded.provider, &saved_graph, &saved_data, &saved_tags).unwrap();
        let original = [
            fs::read(&saved_graph).unwrap(),
            fs::read(&saved_data).unwrap(),
            fs::read(&saved_tags).unwrap(),
        ];
        FAIL_PUBLISH_AT.set(1);
        assert!(
            save_streaming_snapshot(&loaded.provider, &saved_graph, &saved_data, &saved_tags)
                .is_err()
        );
        FAIL_PUBLISH_AT.set(usize::MAX);
        assert_eq!(fs::read(&saved_graph).unwrap(), original[0]);
        assert_eq!(fs::read(&saved_data).unwrap(), original[1]);
        assert_eq!(fs::read(&saved_tags).unwrap(), original[2]);

        let reloaded = load_streaming_snapshot::<Tag128>(
            Full::new(4, Metric::L2),
            StreamingSnapshotConfig {
                dim: 4,
                max_insert_percentage: 100.0,
                graph_degree: 2,
            },
            saved_graph,
            saved_data,
            saved_tags,
        )
        .unwrap();
        assert_eq!(reloaded.active_count, 2);
        assert!(reloaded.provider.to_internal_id(&Context, &first).is_ok());
        assert!(reloaded.provider.to_internal_id(&Context, &second).is_ok());

        let same = temp.path().join("same");
        assert!(save_streaming_snapshot(&reloaded.provider, &same, &same, &same,).is_err());
        assert!(!same.exists());

        let alias_dir = temp.path().join("alias-parent");
        fs::create_dir(&alias_dir).unwrap();
        let alias = temp.path().join("alias");
        let dotted_alias = alias_dir.join("..").join("alias");
        assert!(
            save_streaming_snapshot(
                &reloaded.provider,
                &alias,
                &dotted_alias,
                temp.path().join("alias.tags"),
            )
            .is_err()
        );
        assert!(!alias.exists());

        #[cfg(windows)]
        assert!(
            save_streaming_snapshot(
                &reloaded.provider,
                temp.path().join("CASE"),
                temp.path().join("case"),
                temp.path().join("case.tags"),
            )
            .is_err()
        );

        let frozen = [0u8; 4];
        let empty = Provider::<_, u32>::new(
            Full::new(4, Metric::L2),
            Config::new(4, 2),
            std::iter::once(frozen.as_slice()),
        )
        .unwrap();
        let empty_graph = temp.path().join("empty");
        assert!(
            save_streaming_snapshot(
                &empty,
                &empty_graph,
                temp.path().join("empty.data"),
                temp.path().join("empty.tags"),
            )
            .is_err()
        );
        assert!(!empty_graph.exists());
    }

    #[test]
    fn loads_cpp_snapshot_with_non_final_frozen_point() {
        let temp = tempfile::tempdir().unwrap();
        let data_path = temp.path().join("index.data");
        let graph_path = temp.path().join("index");
        let tag_path = temp.path().join("index.tags");

        let mut data = fs::File::create(&data_path).unwrap();
        data.write_all(&3u32.to_le_bytes()).unwrap();
        data.write_all(&4u32.to_le_bytes()).unwrap();
        data.write_all(&[0; 12]).unwrap();

        let adjacency = [[1u32, 2], [0, 2], [0, 1]];
        let size = 24 + adjacency.len() * 12;
        let mut graph = fs::File::create(&graph_path).unwrap();
        graph.write_all(&(size as u64).to_le_bytes()).unwrap();
        graph.write_all(&2u32.to_le_bytes()).unwrap();
        graph.write_all(&1u32.to_le_bytes()).unwrap();
        graph.write_all(&1u64.to_le_bytes()).unwrap();
        for neighbors in adjacency {
            graph.write_all(&2u32.to_le_bytes()).unwrap();
            for neighbor in neighbors {
                graph.write_all(&neighbor.to_le_bytes()).unwrap();
            }
        }

        let first = Tag128 { low: 1, high: 2 };
        let second = Tag128 { low: 3, high: 4 };
        let mut tags = fs::File::create(&tag_path).unwrap();
        tags.write_all(&3u32.to_le_bytes()).unwrap();
        tags.write_all(&1u32.to_le_bytes()).unwrap();
        for tag in [first, Tag128::default(), second] {
            tags.write_all(&tag.low.to_le_bytes()).unwrap();
            tags.write_all(&tag.high.to_le_bytes()).unwrap();
        }

        let loaded = load_streaming_snapshot::<Tag128>(
            Full::new(4, Metric::L2),
            StreamingSnapshotConfig {
                dim: 4,
                max_insert_percentage: 100.0,
                graph_degree: 2,
            },
            graph_path,
            data_path,
            tag_path,
        )
        .unwrap();

        assert_eq!(loaded.active_count, 2);
        assert!(loaded.provider.to_internal_id(&Context, &first).is_ok());
        assert!(loaded.provider.to_internal_id(&Context, &second).is_ok());
    }
}
