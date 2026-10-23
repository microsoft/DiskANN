/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Label-index loading and query serving.

use crate::{
    bloom::validate_bloom_config,
    error::EncodedLabelIndexError,
    format::{
        ensure_remaining, read_u32, read_u64, validate_label, validate_row_padding,
        LABEL_INDEX_FLAGS, LABEL_INDEX_MAGIC, LABEL_INDEX_VERSION, MAX_LABEL_COUNT,
        MAX_LABEL_LENGTH,
    },
};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    marker::PhantomData,
    path::Path,
    sync::Arc,
};

mod dense;
mod sparse;

use dense::compile_point_plan;
pub use sparse::BloomVisitScratch;

/// The Boolean normal form of a flat clause list passed to [`EncodedLabelIndex::query`].
#[allow(clippy::upper_case_acronyms)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterExpressionType {
    /// An outer OR of `&`-separated AND clauses.
    ///
    /// `["A&B", "C&D"]` represents `(A AND B) OR (C AND D)`.
    DNF = 0,
    /// An outer AND of `|`-separated OR clauses.
    ///
    /// `["A|B", "C|D"]` represents `(A OR B) AND (C OR D)`.
    CNF = 1,
}

struct LabelIndexPayload {
    words_per_row: usize,
    hash_count: usize,
    bits: Box<[u64]>,
    label_rows: Box<[u32]>,
    exact_row_by_label: Box<[u32]>,
    exact_label_count: usize,
    row_valid_counts: Box<[u32]>,
    nonzero_words: BloomWordSummary,
}

struct BloomWordSummary {
    words_per_row: usize,
    bits: Box<[u64]>,
    scan_useful: bool,
}

/// Query-time Bloom candidate enumeration configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BloomQueryConfig {
    summary_cache_bound_percent: f64,
    summary_scan_max_word_percent: f64,
    point_prefetch_lookahead: usize,
}

const DEFAULT_POINT_PREFETCH_LOOKAHEAD: usize = 32;

impl BloomQueryConfig {
    /// Create a query configuration with a 1% to 100% summary-cache bound.
    ///
    /// A value of 1 caches masks when the conservative match bound is at most
    /// 1% of the searchable vectors. The summary-scan word limit defaults to 10%, and exact
    /// payload point classification prefetches 32 IDs ahead on supported architectures.
    pub fn new(summary_cache_bound_percent: f64) -> Result<Self, EncodedLabelIndexError> {
        validate_percent(
            "Bloom summary cache bound percent",
            summary_cache_bound_percent,
        )?;
        Ok(Self {
            summary_cache_bound_percent,
            summary_scan_max_word_percent: 10.0,
            point_prefetch_lookahead: DEFAULT_POINT_PREFETCH_LOOKAHEAD,
        })
    }

    /// Set the maximum surviving payload-word percentage that uses word summaries.
    ///
    /// A value of 100 always uses available summaries instead of falling back
    /// to a full payload scan.
    pub fn with_summary_scan_max_word_percent(
        mut self,
        summary_scan_max_word_percent: f64,
    ) -> Result<Self, EncodedLabelIndexError> {
        validate_percent(
            "Bloom summary scan maximum word percent",
            summary_scan_max_word_percent,
        )?;
        self.summary_scan_max_word_percent = summary_scan_max_word_percent;
        Ok(self)
    }

    /// Set the payload point-classification prefetch lookahead.
    ///
    /// Zero disables prefetching. Nonzero values are used only on supported architectures and only
    /// by exact payload evaluators.
    pub fn with_point_prefetch_lookahead(mut self, lookahead: usize) -> Self {
        self.point_prefetch_lookahead = lookahead;
        self
    }

    /// Return the maximum conservative match-bound percentage eligible for mask caching.
    pub fn summary_cache_bound_percent(self) -> f64 {
        self.summary_cache_bound_percent
    }

    /// Return the maximum surviving payload-word percentage that uses word summaries.
    pub fn summary_scan_max_word_percent(self) -> f64 {
        self.summary_scan_max_word_percent
    }

    /// Return the payload point-classification prefetch lookahead.
    pub fn point_prefetch_lookahead(self) -> usize {
        self.point_prefetch_lookahead
    }

    pub(crate) fn should_scan_summarized_words(
        self,
        possible_word_count: usize,
        words_per_row: usize,
    ) -> bool {
        (possible_word_count as f64) * 100.0
            <= (words_per_row as f64) * self.summary_scan_max_word_percent
    }
}

impl Default for BloomQueryConfig {
    fn default() -> Self {
        Self {
            summary_cache_bound_percent: 1.0,
            summary_scan_max_word_percent: 10.0,
            point_prefetch_lookahead: DEFAULT_POINT_PREFETCH_LOOKAHEAD,
        }
    }
}

fn validate_percent(name: &str, value: f64) -> Result<(), EncodedLabelIndexError> {
    if !value.is_finite() || !(1.0..=100.0).contains(&value) {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "{name} must be in 1..=100, got {value}"
        )));
    }
    Ok(())
}

/// An immutable label index loaded from a versioned label-index file.
pub struct EncodedLabelIndex {
    labels: Box<[String]>,
    label_ids: HashMap<String, u32>,
    counts: Arc<[u32]>,
    summary_nonzero_words_per_row: Arc<[u32]>,
    num_vectors: u32,
    payload: Arc<LabelIndexPayload>,
}

impl std::fmt::Debug for EncodedLabelIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelIndex")
            .field("num_labels", &self.labels.len())
            .field("num_exact_labels", &self.num_exact_labels())
            .field("num_vectors", &self.num_vectors)
            .finish()
    }
}

#[derive(Debug)]
struct CompiledPlan {
    expression_type: FilterExpressionType,
    clause_offsets: Box<[usize]>,
    label_ids: Box<[Option<u32>]>,
}

#[derive(Debug)]
enum PointPlan {
    Dnf(Box<DnfPointPlan>),
    Cnf(Box<CnfPointPlan>),
}

#[derive(Debug)]
struct DnfPointPlan {
    clause_offsets: Box<[usize]>,
    payload_offsets: Box<[usize]>,
    summary_offsets: Box<[usize]>,
    use_summary: bool,
}

#[derive(Debug)]
struct CnfPointPlan {
    clause_offsets: Box<[usize]>,
    terminal_offsets: Box<[usize]>,
    payload_offsets: Box<[usize]>,
    summary_offsets: Box<[usize]>,
    use_summary: bool,
}

/// A point evaluator selected once from a compiled [`EncodedLabelQuery`].
///
/// Match this enum outside a repeated classification loop so each variant's concrete evaluator
/// can be statically dispatched without checking the query mode for every vector ID.
#[derive(Debug, Clone, Copy)]
pub enum EncodedLabelPointEvaluator<'a> {
    /// A DNF evaluator that reads payload rows directly.
    DnfPayload(DnfPayloadEvaluator<'a>),
    /// A DNF evaluator that checks summary rows before payload rows.
    DnfSummary(DnfSummaryEvaluator<'a>),
    /// A CNF evaluator that reads payload rows directly.
    CnfPayload(CnfPayloadEvaluator<'a>),
    /// A CNF evaluator that checks summary rows before payload rows.
    CnfSummary(CnfSummaryEvaluator<'a>),
}

/// A statically dispatched DNF evaluator for payload rows.
#[derive(Debug, Clone, Copy)]
pub struct DnfPayloadEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    clause_offsets: &'a [usize],
    payload_offsets: &'a [usize],
    prefetch_word_offset: Option<usize>,
    prefetch_lookahead: usize,
    cache_rejections: bool,
}

/// A statically dispatched DNF evaluator for summary-guarded payload rows.
#[derive(Debug, Clone, Copy)]
pub struct DnfSummaryEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    summary_bits: &'a [u64],
    clause_offsets: &'a [usize],
    payload_offsets: &'a [usize],
    summary_offsets: &'a [usize],
    cache_rejections: bool,
}

/// A statically dispatched CNF evaluator for payload rows.
#[derive(Debug, Clone, Copy)]
pub struct CnfPayloadEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    clause_offsets: &'a [usize],
    terminal_offsets: &'a [usize],
    payload_offsets: &'a [usize],
    prefetch_word_offset: Option<usize>,
    prefetch_lookahead: usize,
    cache_rejections: bool,
}

/// A statically dispatched CNF evaluator for summary-guarded payload rows.
#[derive(Debug, Clone, Copy)]
pub struct CnfSummaryEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    summary_bits: &'a [u64],
    clause_offsets: &'a [usize],
    terminal_offsets: &'a [usize],
    payload_offsets: &'a [usize],
    summary_offsets: &'a [usize],
    cache_rejections: bool,
}

const POINT_SUMMARY_MIN_ZERO_PERCENT: usize = 85;

/// Query-scoped evaluator compiled from an [`EncodedLabelIndex`].
///
/// Queries share the immutable index payload, so they remain usable after the source index is
/// dropped.
pub struct EncodedLabelQuery<'a> {
    num_vectors: u32,
    payload: Arc<LabelIndexPayload>,
    plan: CompiledPlan,
    point_plan: PointPlan,
    counts: Arc<[u32]>,
    config: BloomQueryConfig,
    _lifetime: PhantomData<&'a ()>,
}

impl std::fmt::Debug for EncodedLabelQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelQuery")
            .field("num_vectors", &self.num_vectors)
            .field("expression_type", &self.plan.expression_type)
            .field(
                "summary_cache_bound_percent",
                &self.config.summary_cache_bound_percent(),
            )
            .field(
                "summary_scan_max_word_percent",
                &self.config.summary_scan_max_word_percent(),
            )
            .field(
                "point_prefetch_lookahead",
                &self.config.point_prefetch_lookahead(),
            )
            .finish()
    }
}

impl EncodedLabelIndex {
    /// Load a unified exact/Bloom label index from `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EncodedLabelIndexError> {
        load_label_index(path.as_ref())
    }

    /// Return the number of vectors covered by this index.
    pub fn num_vectors(&self) -> u32 {
        self.num_vectors
    }

    /// Return the number of encoded labels.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// Return how many dictionary labels use exact Bitslice rows.
    pub fn num_exact_labels(&self) -> usize {
        self.payload.exact_label_count
    }

    /// Return valid-node counts for physical rows in exact-prefix then Bloom-row order.
    pub fn physical_row_valid_counts(&self) -> &[u32] {
        &self.payload.row_valid_counts
    }

    /// Return nonzero payload-word counts for physical rows.
    pub fn physical_row_nonzero_word_counts(&self) -> &[u32] {
        &self.summary_nonzero_words_per_row
    }

    /// Return whether this index contains an encoded label.
    pub fn contains_label(&self, label: &str) -> bool {
        self.label_ids.contains_key(label)
    }

    /// Compile a flat clause-list query using DNF or CNF semantics.
    pub fn query<S>(
        &self,
        clauses: &[S],
        expression_type: FilterExpressionType,
    ) -> Result<EncodedLabelQuery<'static>, EncodedLabelIndexError>
    where
        S: AsRef<str>,
    {
        self.query_with_config(clauses, expression_type, BloomQueryConfig::default())
    }

    /// Compile a query with explicit Bloom candidate-enumeration configuration.
    pub fn query_with_config<S>(
        &self,
        clauses: &[S],
        expression_type: FilterExpressionType,
        config: BloomQueryConfig,
    ) -> Result<EncodedLabelQuery<'static>, EncodedLabelIndexError>
    where
        S: AsRef<str>,
    {
        let plan = compile_plan(clauses, expression_type, &self.label_ids)?;
        let point_plan = compile_point_plan(
            &plan,
            &self.payload,
            &self.counts,
            &self.summary_nonzero_words_per_row,
        );
        Ok(EncodedLabelQuery {
            num_vectors: self.num_vectors,
            payload: Arc::clone(&self.payload),
            plan,
            point_plan,
            counts: self.counts.clone(),
            config,
            _lifetime: PhantomData,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct LabelIndexHeader {
    num_vectors: u32,
    num_labels: usize,
    exact_label_count: usize,
    hash_count: usize,
    physical_row_count: usize,
    words_per_row: usize,
    summary_words_per_row: usize,
}

struct LoadedLabelMappings {
    labels: Box<[String]>,
    label_ids: HashMap<String, u32>,
    counts: Arc<[u32]>,
    label_rows: Box<[u32]>,
    exact_row_by_label: Box<[u32]>,
}

struct LoadedLabelRows {
    row_valid_counts: Box<[u32]>,
    row_nonzero_word_counts: Arc<[u32]>,
    summary: BloomWordSummary,
    payload_bits: Box<[u64]>,
}

fn load_label_index(path: &Path) -> Result<EncodedLabelIndex, EncodedLabelIndexError> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let header = read_label_index_header(&mut reader)?;
    let mappings = read_label_mappings(&mut reader, file_len, header)?;
    let rows = read_label_rows(&mut reader, file_len, header, &mappings.counts)?;
    let payload = LabelIndexPayload {
        words_per_row: header.words_per_row,
        hash_count: header.hash_count,
        bits: rows.payload_bits,
        label_rows: mappings.label_rows,
        exact_row_by_label: mappings.exact_row_by_label,
        exact_label_count: header.exact_label_count,
        row_valid_counts: rows.row_valid_counts,
        nonzero_words: rows.summary,
    };

    Ok(EncodedLabelIndex {
        labels: mappings.labels,
        label_ids: mappings.label_ids,
        counts: mappings.counts,
        summary_nonzero_words_per_row: rows.row_nonzero_word_counts,
        num_vectors: header.num_vectors,
        payload: Arc::new(payload),
    })
}

fn read_label_index_header(
    reader: &mut BufReader<File>,
) -> Result<LabelIndexHeader, EncodedLabelIndexError> {
    let mut magic = [0u8; LABEL_INDEX_MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if magic != LABEL_INDEX_MAGIC {
        return Err(EncodedLabelIndexError::Invalid(
            "invalid label-index file magic".to_string(),
        ));
    }
    let version = read_u32(reader)?;
    if version != LABEL_INDEX_VERSION {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "unsupported label-index version {version}"
        )));
    }
    let flags = read_u32(reader)?;
    if flags != LABEL_INDEX_FLAGS {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "unsupported label-index flags {flags}"
        )));
    }

    let num_vectors = u32::try_from(read_u64(reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("label-index vector count exceeds u32".to_string())
    })?;
    if num_vectors == 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index vector count cannot be zero".to_string(),
        ));
    }
    let num_labels = usize::try_from(read_u64(reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("label-index label count exceeds usize".to_string())
    })?;
    if num_labels == 0 || num_labels > MAX_LABEL_COUNT {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label-index label count {num_labels} is outside 1..={MAX_LABEL_COUNT}"
        )));
    }
    let exact_label_count = usize::try_from(read_u32(reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("exact label count exceeds usize".to_string())
    })?;
    let bloom_row_count = usize::try_from(read_u32(reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("Bloom row count exceeds usize".to_string())
    })?;
    let hash_count = usize::try_from(read_u32(reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("Bloom hash count exceeds usize".to_string())
    })?;
    let reserved = read_u32(reader)?;
    if reserved != 0 {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label-index reserved field is {reserved}, expected 0"
        )));
    }
    if exact_label_count > num_labels {
        return Err(EncodedLabelIndexError::Invalid(
            "exact label count exceeds label count".to_string(),
        ));
    }
    if exact_label_count == num_labels {
        if bloom_row_count != 0 || hash_count != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "all-exact label-index cannot contain Bloom rows".to_string(),
            ));
        }
    } else {
        let bloom_rows = u32::try_from(bloom_row_count).map_err(|_| {
            EncodedLabelIndexError::Invalid("Bloom row count exceeds u32".to_string())
        })?;
        let hashes = u32::try_from(hash_count).map_err(|_| {
            EncodedLabelIndexError::Invalid("Bloom hash count exceeds u32".to_string())
        })?;
        validate_bloom_config(bloom_rows, hashes)?;
    }
    let physical_row_count = exact_label_count
        .checked_add(bloom_row_count)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("physical row count overflow".to_string())
        })?;
    if physical_row_count == 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index must contain at least one physical row".to_string(),
        ));
    }

    let words_per_row = usize::try_from(read_u64(reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("payload row length exceeds usize".to_string())
    })?;
    let expected_words_per_row = (num_vectors as usize).div_ceil(64);
    if words_per_row != expected_words_per_row {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "payload row has {words_per_row} words; expected {expected_words_per_row}"
        )));
    }
    let summary_words_per_row = usize::try_from(read_u64(reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("summary row length exceeds usize".to_string())
    })?;
    let expected_summary_words = words_per_row.div_ceil(64);
    if summary_words_per_row != expected_summary_words {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "summary row has {summary_words_per_row} words; expected {expected_summary_words}"
        )));
    }

    Ok(LabelIndexHeader {
        num_vectors,
        num_labels,
        exact_label_count,
        hash_count,
        physical_row_count,
        words_per_row,
        summary_words_per_row,
    })
}

fn read_label_mappings(
    reader: &mut BufReader<File>,
    file_len: u64,
    header: LabelIndexHeader,
) -> Result<LoadedLabelMappings, EncodedLabelIndexError> {
    let minimum_dictionary_bytes = header.num_labels.checked_mul(8).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("label dictionary size overflow".to_string())
    })?;
    ensure_remaining(
        reader,
        file_len,
        minimum_dictionary_bytes,
        "label dictionary",
    )?;
    let mut labels = Vec::new();
    let mut counts = Vec::new();
    let mut label_ids = HashMap::new();
    labels.try_reserve_exact(header.num_labels).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve label dictionary".to_string())
    })?;
    counts
        .try_reserve_exact(header.num_labels)
        .map_err(|_| EncodedLabelIndexError::Invalid("cannot reserve label counts".to_string()))?;
    label_ids.try_reserve(header.num_labels).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve label lookup map".to_string())
    })?;
    for id in 0..header.num_labels {
        let len = usize::try_from(read_u32(reader)?).map_err(|_| {
            EncodedLabelIndexError::Invalid("label length exceeds usize".to_string())
        })?;
        let count = read_u32(reader)?;
        if len > MAX_LABEL_LENGTH {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "label length {len} exceeds limit {MAX_LABEL_LENGTH}"
            )));
        }
        if count == 0 || count > header.num_vectors {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "label count {count} is outside 1..={}",
                header.num_vectors
            )));
        }
        ensure_remaining(reader, file_len, len, "label bytes")?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len).map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve label bytes".to_string())
        })?;
        bytes.resize(len, 0);
        reader.read_exact(&mut bytes)?;
        let label = String::from_utf8(bytes).map_err(|_| {
            EncodedLabelIndexError::Invalid("label-index contains invalid UTF-8".to_string())
        })?;
        validate_label(&label)?;
        let id = u32::try_from(id)
            .map_err(|_| EncodedLabelIndexError::Invalid("label count exceeds u32".to_string()))?;
        if label_ids.insert(label.clone(), id).is_some() {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "duplicate label '{label}' in label-index"
            )));
        }
        labels.push(label);
        counts.push(count);
    }

    let total_label_rows = header
        .num_labels
        .checked_mul(header.hash_count)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("label-row table size overflow".to_string())
        })?;
    let mut label_rows = Vec::new();
    label_rows
        .try_reserve_exact(total_label_rows)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve label-row table".to_string())
        })?;
    label_rows.resize(total_label_rows, u32::MAX);
    let mut exact_row_by_label = Vec::new();
    exact_row_by_label
        .try_reserve_exact(header.num_labels)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve exact-row table".to_string())
        })?;
    exact_row_by_label.resize(header.num_labels, u32::MAX);
    for (label_id, exact_row) in exact_row_by_label
        .iter_mut()
        .take(header.exact_label_count)
        .enumerate()
    {
        *exact_row = u32::try_from(label_id).map_err(|_| {
            EncodedLabelIndexError::Invalid("exact row index exceeds u32".to_string())
        })?;
    }
    let mapping_count = (header.num_labels - header.exact_label_count)
        .checked_mul(header.hash_count)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("Bloom mapping count overflow".to_string())
        })?;
    ensure_remaining(
        reader,
        file_len,
        mapping_count.checked_mul(4).ok_or_else(|| {
            EncodedLabelIndexError::Invalid("Bloom mapping byte size overflow".to_string())
        })?,
        "Bloom label-row mappings",
    )?;
    for label_id in header.exact_label_count..header.num_labels {
        let start = label_id * header.hash_count;
        for hash_index in 0..header.hash_count {
            let row = read_u32(reader)?;
            if row < header.exact_label_count as u32
                || row as usize >= header.physical_row_count
                || label_rows[start..start + hash_index].contains(&row)
            {
                return Err(EncodedLabelIndexError::Invalid(
                    "Bloom label rows must be distinct physical Bloom rows".to_string(),
                ));
            }
            label_rows[start + hash_index] = row;
        }
    }

    Ok(LoadedLabelMappings {
        labels: labels.into_boxed_slice(),
        label_ids,
        counts: Arc::from(counts),
        label_rows: label_rows.into_boxed_slice(),
        exact_row_by_label: exact_row_by_label.into_boxed_slice(),
    })
}

fn read_label_rows(
    reader: &mut BufReader<File>,
    file_len: u64,
    header: LabelIndexHeader,
    label_counts: &[u32],
) -> Result<LoadedLabelRows, EncodedLabelIndexError> {
    ensure_remaining(
        reader,
        file_len,
        header.physical_row_count.checked_mul(8).ok_or_else(|| {
            EncodedLabelIndexError::Invalid("row metadata byte size overflow".to_string())
        })?,
        "row metadata",
    )?;
    let mut row_valid_counts = Vec::new();
    let mut row_nonzero_word_counts = Vec::new();
    row_valid_counts
        .try_reserve_exact(header.physical_row_count)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve row valid counts".to_string())
        })?;
    row_nonzero_word_counts
        .try_reserve_exact(header.physical_row_count)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve row nonzero counts".to_string())
        })?;
    for _ in 0..header.physical_row_count {
        let valid_node_count = read_u32(reader)?;
        let nonzero_word_count = read_u32(reader)?;
        if valid_node_count > header.num_vectors
            || nonzero_word_count as usize > header.words_per_row
        {
            return Err(EncodedLabelIndexError::Invalid(
                "row metadata exceeds index dimensions".to_string(),
            ));
        }
        row_valid_counts.push(valid_node_count);
        row_nonzero_word_counts.push(nonzero_word_count);
    }
    if row_valid_counts[..header.exact_label_count] != label_counts[..header.exact_label_count] {
        return Err(EncodedLabelIndexError::Invalid(
            "exact row valid count does not match label count".to_string(),
        ));
    }

    let summary_bits = read_payload(
        reader,
        file_len,
        header.physical_row_count,
        header.summary_words_per_row,
        "label summary",
    )?;
    let summarized_words = u32::try_from(header.words_per_row).map_err(|_| {
        EncodedLabelIndexError::Invalid("payload row word count exceeds u32".to_string())
    })?;
    validate_row_padding(
        &summary_bits,
        header.physical_row_count,
        header.summary_words_per_row,
        summarized_words,
    )?;
    for (row, summary) in summary_bits
        .chunks_exact(header.summary_words_per_row)
        .enumerate()
    {
        let nonzero = summary.iter().map(|word| word.count_ones()).sum::<u32>();
        if nonzero != row_nonzero_word_counts[row] {
            return Err(EncodedLabelIndexError::Invalid(
                "summary popcount does not match row metadata".to_string(),
            ));
        }
    }

    let payload_bits = read_payload(
        reader,
        file_len,
        header.physical_row_count,
        header.words_per_row,
        "label payload",
    )?;
    validate_row_padding(
        &payload_bits,
        header.physical_row_count,
        header.words_per_row,
        header.num_vectors,
    )?;
    if reader.read(&mut [0u8; 1])? != 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index contains trailing bytes".to_string(),
        ));
    }

    let total_nonzero_words = row_nonzero_word_counts
        .iter()
        .try_fold(0usize, |sum, &count| sum.checked_add(count as usize))
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("row nonzero word count overflow".to_string())
        })?;
    let total_payload_words = header
        .physical_row_count
        .checked_mul(header.words_per_row)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("payload word count overflow".to_string())
        })?;
    let summary = BloomWordSummary {
        words_per_row: header.summary_words_per_row,
        bits: summary_bits.into_boxed_slice(),
        scan_useful: header.exact_label_count > 0
            || total_nonzero_words.saturating_mul(4) <= total_payload_words,
    };

    Ok(LoadedLabelRows {
        row_valid_counts: row_valid_counts.into_boxed_slice(),
        row_nonzero_word_counts: Arc::from(row_nonzero_word_counts),
        summary,
        payload_bits: payload_bits.into_boxed_slice(),
    })
}

fn read_payload(
    reader: &mut BufReader<File>,
    file_len: u64,
    row_count: usize,
    words_per_row: usize,
    format_name: &str,
) -> Result<Vec<u64>, EncodedLabelIndexError> {
    let total_words = row_count.checked_mul(words_per_row).ok_or_else(|| {
        EncodedLabelIndexError::Invalid(format!("{format_name} allocation size overflow"))
    })?;
    let payload_bytes = total_words.checked_mul(8).ok_or_else(|| {
        EncodedLabelIndexError::Invalid(format!("{format_name} byte size overflow"))
    })?;
    ensure_remaining(
        reader,
        file_len,
        payload_bytes,
        &format!("{format_name} payload"),
    )?;

    let mut bits = Vec::new();
    bits.try_reserve_exact(total_words).map_err(|_| {
        EncodedLabelIndexError::Invalid(format!("cannot reserve {format_name} payload"))
    })?;
    for _ in 0..total_words {
        bits.push(read_u64(reader)?);
    }
    Ok(bits)
}

fn compile_plan<S: AsRef<str>>(
    clauses: &[S],
    expression_type: FilterExpressionType,
    label_ids: &HashMap<String, u32>,
) -> Result<CompiledPlan, EncodedLabelIndexError> {
    if clauses.is_empty() {
        return Err(EncodedLabelIndexError::Invalid(
            "filter must contain at least one clause".to_string(),
        ));
    }

    let delimiter = match expression_type {
        FilterExpressionType::DNF => '&',
        FilterExpressionType::CNF => '|',
    };
    let mut clause_offsets = vec![0usize];
    let mut encoded = Vec::new();

    for clause in clauses {
        let clause = clause.as_ref();
        if clause.is_empty() {
            return Err(EncodedLabelIndexError::Invalid(
                "filter clauses cannot be empty".to_string(),
            ));
        }
        for terminal in clause.split(delimiter) {
            let terminal = terminal.trim();
            validate_label(terminal)?;
            encoded.push(label_ids.get(terminal).copied());
        }
        clause_offsets.push(encoded.len());
    }

    Ok(CompiledPlan {
        expression_type,
        clause_offsets: clause_offsets.into_boxed_slice(),
        label_ids: encoded.into_boxed_slice(),
    })
}
