/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Versioned on-disk label-index encoding, loading, and query evaluation for DiskANN.

use roaring::RoaringBitmap;
use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Seek, Write},
    marker::PhantomData,
    path::Path,
    sync::Arc,
};
use thiserror::Error;

const LABEL_INDEX_MAGIC: [u8; 8] = *b"DANLBL01";
const LABEL_INDEX_VERSION: u32 = 1;
const MAX_LABEL_COUNT: usize = 1_000_000;
const MAX_LABEL_LENGTH: usize = 1 << 20;
const MAX_POSTING_BYTES: usize = 512 << 20;
const MAX_DENSE_BITMAP_BYTES: usize = 256 << 20;
const MAX_BITMAP_VECTORS: u64 = (MAX_DENSE_BITMAP_BYTES as u64) * 8;
const MAX_LABEL_EXPRESSION_DEPTH: usize = 64;
const MAX_LABEL_EXPRESSION_NODES: usize = 4096;

/// The persisted label-index storage format.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelIndexFormat {
    /// One contiguous dense bit slice per encoded label.
    Bitslice = 0,
    /// One serialized Roaring posting list per encoded label.
    Bitmap = 1,
    /// Dense bit slices for frequent labels and contiguous postings for sparse labels.
    Hybrid = 2,
}

/// Build-time options for [`LabelIndexFormat::Hybrid`].
#[derive(Debug, Clone, Copy, Default)]
pub struct HybridBuildOptions {
    /// Minimum posting cardinality stored as a dense bit slice.
    ///
    /// When omitted, the encoder uses the memory break-even point against raw `u32` postings.
    pub dense_threshold: Option<u32>,
}

/// Storage summary returned by the hybrid encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridBuildStats {
    /// Cardinality at or above which labels were stored densely.
    pub dense_threshold: u32,
    /// Number of dense label rows.
    pub dense_labels: u32,
    /// Number of sparse posting lists.
    pub sparse_labels: u32,
    /// Dense payload bytes, excluding descriptors and dictionary data.
    pub dense_bytes: u64,
    /// Sparse posting and offset bytes, excluding descriptors and dictionary data.
    pub sparse_bytes: u64,
}

/// Persisted representation selected for a hybrid label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLabelRepresentation {
    /// Full vector-ID bit slice.
    Dense,
    /// Sorted vector-ID posting range.
    Sparse,
}

/// The outer Boolean structure of a clause list passed to [`EncodedLabelIndex::query`].
#[allow(clippy::upper_case_acronyms)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterExpressionType {
    /// Historical name for outer OR with `&`-separated labels inside each string:
    /// `["A&B", "C&D"]` means `(A AND B) OR (C AND D)`.
    ORMajor = 0,
    /// Historical name for outer AND with `|`-separated labels inside each string:
    /// `["A|B", "C|D"]` means `(A OR B) AND (C OR D)`.
    ANDMajor = 1,
}

impl FilterExpressionType {
    /// Preferred alias for [`FilterExpressionType::ORMajor`] using Boolean normal-form
    /// terminology.
    pub const DNF: Self = Self::ORMajor;

    /// Preferred alias for [`FilterExpressionType::ANDMajor`] using Boolean normal-form
    /// terminology.
    pub const CNF: Self = Self::ANDMajor;
}

/// A crate-owned Boolean label expression tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabelExpression {
    /// A terminal label.
    Label(String),
    /// A conjunction of child expressions.
    And(Vec<LabelExpression>),
    /// A disjunction of child expressions.
    Or(Vec<LabelExpression>),
    /// A negated child expression.
    Not(Box<LabelExpression>),
}

struct LabelExpressionBudget {
    remaining_nodes: usize,
}

impl LabelExpressionBudget {
    fn new() -> Self {
        Self {
            remaining_nodes: MAX_LABEL_EXPRESSION_NODES,
        }
    }

    fn reserve<E>(&mut self, depth: usize) -> Result<(), E>
    where
        E: de::Error,
    {
        if depth > MAX_LABEL_EXPRESSION_DEPTH {
            return Err(E::custom(format!(
                "label expression depth {depth} exceeds limit {MAX_LABEL_EXPRESSION_DEPTH}"
            )));
        }

        if self.remaining_nodes == 0 {
            let node_count = MAX_LABEL_EXPRESSION_NODES + 1;
            return Err(E::custom(format!(
                "label expression node count {node_count} exceeds limit {MAX_LABEL_EXPRESSION_NODES}"
            )));
        }

        self.remaining_nodes -= 1;
        Ok(())
    }
}

struct LabelExpressionSeed<'a> {
    budget: &'a mut LabelExpressionBudget,
    depth: usize,
}

impl<'a, 'de> DeserializeSeed<'de> for LabelExpressionSeed<'a> {
    type Value = LabelExpression;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(LabelExpressionVisitor {
            budget: self.budget,
            depth: self.depth,
        })
    }
}

struct LabelExpressionVisitor<'a> {
    budget: &'a mut LabelExpressionBudget,
    depth: usize,
}

impl<'de> Visitor<'de> for LabelExpressionVisitor<'_> {
    type Value = LabelExpression;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a label string or a one-key and/or/not object")
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_str(value)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.budget.reserve::<E>(self.depth)?;
        validate_label(value).map_err(E::custom)?;
        Ok(LabelExpression::Label(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.budget.reserve::<E>(self.depth)?;
        validate_label(&value).map_err(E::custom)?;
        Ok(LabelExpression::Label(value))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let LabelExpressionVisitor { budget, depth } = self;
        budget.reserve::<A::Error>(depth)?;

        let Some(operator) = map.next_key::<String>()? else {
            return Err(de::Error::custom(
                "label expression objects must contain exactly one operator",
            ));
        };

        let child_depth = depth + 1;
        let expression = match operator.as_str() {
            "and" | "$and" => {
                LabelExpression::And(map.next_value_seed(LabelExpressionArraySeed {
                    budget,
                    depth: child_depth,
                    operator: "and",
                })?)
            }
            "or" | "$or" => LabelExpression::Or(map.next_value_seed(LabelExpressionArraySeed {
                budget,
                depth: child_depth,
                operator: "or",
            })?),
            "not" | "$not" => {
                LabelExpression::Not(Box::new(map.next_value_seed(LabelExpressionSeed {
                    budget,
                    depth: child_depth,
                })?))
            }
            _ => {
                return Err(de::Error::custom(format!(
                    "unsupported label expression operator '{operator}'"
                )));
            }
        };

        if map.next_key::<String>()?.is_some() {
            return Err(de::Error::custom(
                "label expression objects must contain exactly one operator",
            ));
        }

        Ok(expression)
    }
}

struct LabelExpressionArraySeed<'a> {
    budget: &'a mut LabelExpressionBudget,
    depth: usize,
    operator: &'static str,
}

impl<'a, 'de> DeserializeSeed<'de> for LabelExpressionArraySeed<'a> {
    type Value = Vec<LabelExpression>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(LabelExpressionArrayVisitor {
            budget: self.budget,
            depth: self.depth,
            operator: self.operator,
        })
    }
}

struct LabelExpressionArrayVisitor<'a> {
    budget: &'a mut LabelExpressionBudget,
    depth: usize,
    operator: &'static str,
}

impl<'de> Visitor<'de> for LabelExpressionArrayVisitor<'_> {
    type Value = Vec<LabelExpression>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a non-empty array of label expressions")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let LabelExpressionArrayVisitor {
            budget,
            depth,
            operator,
        } = self;
        let Some(first) = seq.next_element_seed(LabelExpressionSeed { budget, depth })? else {
            return Err(de::Error::custom(format!(
                "label expression '{}' array cannot be empty",
                operator
            )));
        };

        let mut children = vec![first];
        while let Some(child) = seq.next_element_seed(LabelExpressionSeed { budget, depth })? {
            children.push(child);
        }

        Ok(children)
    }
}

#[derive(Debug, Error)]
pub enum EncodedLabelIndexError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
}

enum LabelStorage {
    Bitslice {
        words_per_label: usize,
        bits: Arc<[u64]>,
    },
    Bitmap {
        postings: Arc<[RoaringBitmap]>,
    },
    Hybrid {
        storage: Arc<HybridStorage>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HybridLabelKind {
    Dense,
    Sparse,
}

#[derive(Debug, Clone, Copy)]
struct HybridLabelDescriptor {
    kind: HybridLabelKind,
    ordinal: u32,
    cardinality: u32,
}

#[derive(Debug)]
struct HybridStorage {
    words_per_label: usize,
    descriptors: Box<[HybridLabelDescriptor]>,
    dense_bits: Box<[u64]>,
    sparse_offsets: Box<[u64]>,
    sparse_doc_ids: Box<[u32]>,
}

impl HybridStorage {
    fn contains(&self, label_id: u32, vec_id: u32) -> bool {
        let descriptor = self.descriptors[label_id as usize];
        match descriptor.kind {
            HybridLabelKind::Dense => {
                let word =
                    descriptor.ordinal as usize * self.words_per_label + vec_id as usize / 64;
                self.dense_bits[word] & (1u64 << (vec_id % 64)) != 0
            }
            HybridLabelKind::Sparse => {
                let ordinal = descriptor.ordinal as usize;
                let start = self.sparse_offsets[ordinal] as usize;
                let end = self.sparse_offsets[ordinal + 1] as usize;
                self.sparse_doc_ids[start..end]
                    .binary_search(&vec_id)
                    .is_ok()
            }
        }
    }

    fn descriptor(&self, label_id: Option<u32>) -> Option<HybridLabelDescriptor> {
        label_id.map(|id| self.descriptors[id as usize])
    }
}

/// An immutable encoded label index loaded from a versioned label-index file.
pub struct EncodedLabelIndex {
    labels: Box<[String]>,
    label_ids: HashMap<String, u32>,
    num_vectors: u32,
    storage: LabelStorage,
}

impl std::fmt::Debug for EncodedLabelIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelIndex")
            .field("num_labels", &self.labels.len())
            .field("num_vectors", &self.num_vectors)
            .field("format", &self.format())
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
enum PlanKind {
    Dnf,
    Cnf,
}

#[derive(Debug)]
struct CompiledPlan {
    kind: PlanKind,
    clause_offsets: Box<[usize]>,
    label_ids: Box<[Option<u32>]>,
}

#[derive(Debug)]
enum CompiledExpression {
    Flat(CompiledPlan),
    Label(Option<u32>),
    And(Box<[CompiledExpression]>),
    Or(Box<[CompiledExpression]>),
    Not(Box<CompiledExpression>),
}

enum QueryStorage {
    Bitslice {
        words_per_label: usize,
        bits: Arc<[u64]>,
        expression: CompiledExpression,
    },
    DenseBitmap {
        bits: Box<[u64]>,
    },
    Hybrid {
        storage: Arc<HybridStorage>,
        expression: CompiledExpression,
    },
}

/// Query-scoped evaluator compiled from an [`EncodedLabelIndex`].
///
/// Bitslice queries share the index payload through [`Arc`] so the compiled query remains usable
/// after the source [`EncodedLabelIndex`] is dropped.
pub struct EncodedLabelQuery<'a> {
    num_vectors: u32,
    storage: QueryStorage,
    _lifetime: PhantomData<&'a ()>,
}

impl std::fmt::Debug for EncodedLabelQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelQuery")
            .field("num_vectors", &self.num_vectors)
            .finish()
    }
}

impl EncodedLabelQuery<'_> {
    /// Return whether `vec_id` satisfies this compiled label query.
    pub fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }

        match &self.storage {
            QueryStorage::Bitslice {
                words_per_label,
                bits,
                expression,
            } => expression.matches_bitslice(*words_per_label, bits, vec_id),
            QueryStorage::DenseBitmap { bits } => dense_contains(bits, vec_id),
            QueryStorage::Hybrid {
                storage,
                expression,
            } => expression.matches_hybrid(storage, vec_id),
        }
    }
}

impl CompiledPlan {
    fn matches_bitslice(&self, words_per_label: usize, bits: &[u64], vec_id: u32) -> bool {
        let terminal_matches = |label_id: Option<u32>| {
            label_id.is_some_and(|label_id| {
                let label_id = label_id as usize;
                let vec_id = vec_id as usize;
                let word = bits[label_id * words_per_label + vec_id / 64];
                word & (1u64 << (vec_id % 64)) != 0
            })
        };

        match self.kind {
            PlanKind::Dnf => self.clause_offsets.windows(2).any(|clause| {
                self.label_ids[clause[0]..clause[1]]
                    .iter()
                    .copied()
                    .all(terminal_matches)
            }),
            PlanKind::Cnf => self.clause_offsets.windows(2).all(|clause| {
                self.label_ids[clause[0]..clause[1]]
                    .iter()
                    .copied()
                    .any(terminal_matches)
            }),
        }
    }
}

impl CompiledExpression {
    fn matches_bitslice(&self, words_per_label: usize, bits: &[u64], vec_id: u32) -> bool {
        match self {
            Self::Flat(plan) => plan.matches_bitslice(words_per_label, bits, vec_id),
            Self::Label(label_id) => label_id.is_some_and(|label_id| {
                let label_id = label_id as usize;
                let vec_id = vec_id as usize;
                let word = bits[label_id * words_per_label + vec_id / 64];
                word & (1u64 << (vec_id % 64)) != 0
            }),
            Self::And(children) => children
                .iter()
                .all(|child| child.matches_bitslice(words_per_label, bits, vec_id)),
            Self::Or(children) => children
                .iter()
                .any(|child| child.matches_bitslice(words_per_label, bits, vec_id)),
            Self::Not(child) => !child.matches_bitslice(words_per_label, bits, vec_id),
        }
    }

    fn matches_hybrid(&self, storage: &HybridStorage, vec_id: u32) -> bool {
        match self {
            Self::Flat(plan) => plan.matches_hybrid(storage, vec_id),
            Self::Label(label_id) => {
                label_id.is_some_and(|label_id| storage.contains(label_id, vec_id))
            }
            Self::And(children) => children
                .iter()
                .all(|child| child.matches_hybrid(storage, vec_id)),
            Self::Or(children) => children
                .iter()
                .any(|child| child.matches_hybrid(storage, vec_id)),
            Self::Not(child) => !child.matches_hybrid(storage, vec_id),
        }
    }

    fn optimize_hybrid(&mut self, storage: &HybridStorage) {
        match self {
            Self::Flat(plan) => plan.optimize_hybrid(storage),
            Self::And(children) => {
                children
                    .iter_mut()
                    .for_each(|child| child.optimize_hybrid(storage));
                if children.iter().all(|child| matches!(child, Self::Label(_))) {
                    children.sort_unstable_by(|left, right| {
                        hybrid_and_cmp(
                            storage.descriptor(expression_label_id(left)),
                            storage.descriptor(expression_label_id(right)),
                        )
                    });
                }
            }
            Self::Or(children) => {
                children
                    .iter_mut()
                    .for_each(|child| child.optimize_hybrid(storage));
                if children.iter().all(|child| matches!(child, Self::Label(_))) {
                    children.sort_unstable_by(|left, right| {
                        hybrid_or_cmp(
                            storage.descriptor(expression_label_id(left)),
                            storage.descriptor(expression_label_id(right)),
                        )
                    });
                }
            }
            Self::Not(child) => child.optimize_hybrid(storage),
            Self::Label(_) => {}
        }
    }
}

impl CompiledPlan {
    fn matches_hybrid(&self, storage: &HybridStorage, vec_id: u32) -> bool {
        let terminal_matches =
            |label_id: Option<u32>| label_id.is_some_and(|id| storage.contains(id, vec_id));

        match self.kind {
            PlanKind::Dnf => self.clause_offsets.windows(2).any(|clause| {
                self.label_ids[clause[0]..clause[1]]
                    .iter()
                    .copied()
                    .all(terminal_matches)
            }),
            PlanKind::Cnf => self.clause_offsets.windows(2).all(|clause| {
                self.label_ids[clause[0]..clause[1]]
                    .iter()
                    .copied()
                    .any(terminal_matches)
            }),
        }
    }

    fn optimize_hybrid(&mut self, storage: &HybridStorage) {
        for clause in self.clause_offsets.windows(2) {
            let labels = &mut self.label_ids[clause[0]..clause[1]];
            match self.kind {
                PlanKind::Dnf => labels.sort_unstable_by(|left, right| {
                    hybrid_and_cmp(storage.descriptor(*left), storage.descriptor(*right))
                }),
                PlanKind::Cnf => labels.sort_unstable_by(|left, right| {
                    hybrid_or_cmp(storage.descriptor(*left), storage.descriptor(*right))
                }),
            }
        }
    }
}

fn expression_label_id(expression: &CompiledExpression) -> Option<u32> {
    match expression {
        CompiledExpression::Label(label_id) => *label_id,
        _ => unreachable!("caller checks that every expression is a label"),
    }
}

fn hybrid_and_cmp(
    left: Option<HybridLabelDescriptor>,
    right: Option<HybridLabelDescriptor>,
) -> std::cmp::Ordering {
    hybrid_and_key(left).cmp(&hybrid_and_key(right))
}

fn hybrid_or_cmp(
    left: Option<HybridLabelDescriptor>,
    right: Option<HybridLabelDescriptor>,
) -> std::cmp::Ordering {
    hybrid_or_key(left).cmp(&hybrid_or_key(right))
}

fn hybrid_and_key(descriptor: Option<HybridLabelDescriptor>) -> (u8, u8, u32) {
    match descriptor {
        None => (0, 0, 0),
        Some(descriptor) => (
            1,
            match descriptor.kind {
                HybridLabelKind::Dense => 0,
                HybridLabelKind::Sparse => 1,
            },
            descriptor.cardinality,
        ),
    }
}

fn hybrid_or_key(descriptor: Option<HybridLabelDescriptor>) -> (u8, u8, std::cmp::Reverse<u32>) {
    match descriptor {
        Some(descriptor) => (
            0,
            match descriptor.kind {
                HybridLabelKind::Dense => 0,
                HybridLabelKind::Sparse => 1,
            },
            std::cmp::Reverse(descriptor.cardinality),
        ),
        None => (1, 0, std::cmp::Reverse(0)),
    }
}

impl EncodedLabelIndex {
    /// Return the number of vectors covered by this index.
    pub fn num_vectors(&self) -> u32 {
        self.num_vectors
    }

    /// Return the number of encoded labels.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// Return whether this index contains an encoded label.
    pub fn contains_label(&self, label: &str) -> bool {
        self.label_ids.contains_key(label)
    }

    /// Return hybrid representation and cardinality metadata for `label`.
    pub fn hybrid_label_metadata(&self, label: &str) -> Option<(HybridLabelRepresentation, u32)> {
        let label_id = *self.label_ids.get(label)? as usize;
        let LabelStorage::Hybrid { storage } = &self.storage else {
            return None;
        };
        let descriptor = storage.descriptors[label_id];
        Some((
            match descriptor.kind {
                HybridLabelKind::Dense => HybridLabelRepresentation::Dense,
                HybridLabelKind::Sparse => HybridLabelRepresentation::Sparse,
            },
            descriptor.cardinality,
        ))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, EncodedLabelIndexError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; LABEL_INDEX_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if magic != LABEL_INDEX_MAGIC {
            return Err(EncodedLabelIndexError::Invalid(
                "invalid label-index file magic".to_string(),
            ));
        }

        let version = read_u32(&mut reader)?;
        if version != LABEL_INDEX_VERSION {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "unsupported label-index version {version}"
            )));
        }

        let format = match read_u32(&mut reader)? {
            0 => LabelIndexFormat::Bitslice,
            1 => LabelIndexFormat::Bitmap,
            2 => LabelIndexFormat::Hybrid,
            value => {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "unsupported label-index format {value}"
                )));
            }
        };
        let num_vectors = u32::try_from(read_u64(&mut reader)?).map_err(|_| {
            EncodedLabelIndexError::Invalid("label-index vector count exceeds u32".to_string())
        })?;
        if num_vectors == 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "label-index vector count cannot be zero".to_string(),
            ));
        }
        if format == LabelIndexFormat::Bitmap && u64::from(num_vectors) > MAX_BITMAP_VECTORS {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "bitmap vector count {num_vectors} exceeds limit {MAX_BITMAP_VECTORS}"
            )));
        }

        let num_labels = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
            EncodedLabelIndexError::Invalid("label-index label count exceeds usize".to_string())
        })?;
        if num_labels > MAX_LABEL_COUNT {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "label-index label count {num_labels} exceeds limit {MAX_LABEL_COUNT}"
            )));
        }
        let minimum_dictionary_bytes = num_labels.checked_mul(4).ok_or_else(|| {
            EncodedLabelIndexError::Invalid("label dictionary size overflow".to_string())
        })?;
        ensure_remaining(
            &mut reader,
            file_len,
            minimum_dictionary_bytes,
            "label dictionary",
        )?;

        let mut labels = Vec::new();
        labels.try_reserve_exact(num_labels).map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve label dictionary".to_string())
        })?;

        let mut label_ids = HashMap::new();
        label_ids.try_reserve(num_labels).map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve label lookup map".to_string())
        })?;

        for id in 0..num_labels {
            let len = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
                EncodedLabelIndexError::Invalid("label length exceeds usize".to_string())
            })?;
            if len > MAX_LABEL_LENGTH {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "label length {len} exceeds limit {MAX_LABEL_LENGTH}"
                )));
            }
            ensure_remaining(&mut reader, file_len, len, "label bytes")?;
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

            let id = u32::try_from(id).map_err(|_| {
                EncodedLabelIndexError::Invalid("label count exceeds u32".to_string())
            })?;
            if label_ids.insert(label.clone(), id).is_some() {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "duplicate label '{label}' in label-index"
                )));
            }
            labels.push(label);
        }

        let storage = match format {
            LabelIndexFormat::Bitslice => {
                let words_per_label = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid("bitslice row length exceeds usize".to_string())
                })?;
                let expected_words = (num_vectors as usize).div_ceil(64);
                if words_per_label != expected_words {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "bitslice row has {words_per_label} words; expected {expected_words}"
                    )));
                }
                let total_words = num_labels.checked_mul(words_per_label).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid("bitslice allocation size overflow".to_string())
                })?;
                let payload_bytes = total_words.checked_mul(8).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid("bitslice byte size overflow".to_string())
                })?;
                ensure_remaining(&mut reader, file_len, payload_bytes, "bitslice payload")?;

                let mut bits = Vec::new();
                bits.try_reserve_exact(total_words).map_err(|_| {
                    EncodedLabelIndexError::Invalid("cannot reserve bitslice payload".to_string())
                })?;
                for _ in 0..total_words {
                    bits.push(read_u64(&mut reader)?);
                }
                validate_bitslice_padding(&bits, num_labels, words_per_label, num_vectors)?;

                LabelStorage::Bitslice {
                    words_per_label,
                    bits: Arc::from(bits),
                }
            }
            LabelIndexFormat::Bitmap => {
                let mut postings = Vec::new();
                postings.try_reserve_exact(num_labels).map_err(|_| {
                    EncodedLabelIndexError::Invalid("cannot reserve posting list table".to_string())
                })?;
                for _ in 0..num_labels {
                    let len = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "serialized posting length exceeds usize".to_string(),
                        )
                    })?;
                    if len > MAX_POSTING_BYTES {
                        return Err(EncodedLabelIndexError::Invalid(format!(
                            "serialized posting length {len} exceeds limit {MAX_POSTING_BYTES}"
                        )));
                    }
                    ensure_remaining(&mut reader, file_len, len, "serialized posting")?;

                    let mut limited = reader.by_ref().take(len as u64);
                    let posting = RoaringBitmap::deserialize_from(&mut limited)?;
                    if limited.limit() != 0 {
                        return Err(EncodedLabelIndexError::Invalid(
                            "serialized posting length does not match payload".to_string(),
                        ));
                    }
                    if posting.max().is_some_and(|max| max >= num_vectors) {
                        return Err(EncodedLabelIndexError::Invalid(
                            "serialized posting contains an out-of-range vector ID".to_string(),
                        ));
                    }
                    postings.push(posting);
                }

                LabelStorage::Bitmap {
                    postings: Arc::from(postings),
                }
            }
            LabelIndexFormat::Hybrid => {
                let words_per_label = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid(
                        "hybrid bitslice row length exceeds usize".to_string(),
                    )
                })?;
                let expected_words = (num_vectors as usize).div_ceil(64);
                if words_per_label != expected_words {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "hybrid bitslice row has {words_per_label} words; expected {expected_words}"
                    )));
                }

                let num_dense = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid(
                        "hybrid dense label count exceeds usize".to_string(),
                    )
                })?;
                let num_sparse = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid(
                        "hybrid sparse label count exceeds usize".to_string(),
                    )
                })?;
                if num_dense.checked_add(num_sparse) != Some(num_labels) {
                    return Err(EncodedLabelIndexError::Invalid(
                        "hybrid dense and sparse label counts do not match dictionary".to_string(),
                    ));
                }

                let descriptor_bytes = num_labels.checked_mul(12).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid(
                        "hybrid descriptor byte size overflow".to_string(),
                    )
                })?;
                ensure_remaining(
                    &mut reader,
                    file_len,
                    descriptor_bytes,
                    "hybrid descriptors",
                )?;

                let mut descriptors = Vec::new();
                descriptors.try_reserve_exact(num_labels).map_err(|_| {
                    EncodedLabelIndexError::Invalid("cannot reserve hybrid descriptors".to_string())
                })?;
                let mut seen_dense = vec![false; num_dense];
                let mut seen_sparse = vec![false; num_sparse];
                for _ in 0..num_labels {
                    let kind = match read_u32(&mut reader)? {
                        0 => HybridLabelKind::Dense,
                        1 => HybridLabelKind::Sparse,
                        value => {
                            return Err(EncodedLabelIndexError::Invalid(format!(
                                "unsupported hybrid label kind {value}"
                            )));
                        }
                    };
                    let ordinal = read_u32(&mut reader)?;
                    let cardinality = read_u32(&mut reader)?;
                    if cardinality > num_vectors {
                        return Err(EncodedLabelIndexError::Invalid(format!(
                            "hybrid label cardinality {cardinality} exceeds vector count {num_vectors}"
                        )));
                    }

                    let seen = match kind {
                        HybridLabelKind::Dense => seen_dense.get_mut(ordinal as usize),
                        HybridLabelKind::Sparse => seen_sparse.get_mut(ordinal as usize),
                    }
                    .ok_or_else(|| {
                        EncodedLabelIndexError::Invalid(
                            "hybrid label ordinal is out of range".to_string(),
                        )
                    })?;
                    if std::mem::replace(seen, true) {
                        return Err(EncodedLabelIndexError::Invalid(
                            "duplicate hybrid label ordinal".to_string(),
                        ));
                    }

                    descriptors.push(HybridLabelDescriptor {
                        kind,
                        ordinal,
                        cardinality,
                    });
                }
                if seen_dense.iter().any(|seen| !seen) || seen_sparse.iter().any(|seen| !seen) {
                    return Err(EncodedLabelIndexError::Invalid(
                        "hybrid label ordinals are incomplete".to_string(),
                    ));
                }

                let total_dense_words =
                    num_dense.checked_mul(words_per_label).ok_or_else(|| {
                        EncodedLabelIndexError::Invalid(
                            "hybrid dense allocation size overflow".to_string(),
                        )
                    })?;
                let dense_bytes = total_dense_words.checked_mul(8).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid("hybrid dense byte size overflow".to_string())
                })?;
                ensure_remaining(&mut reader, file_len, dense_bytes, "hybrid dense payload")?;
                let mut dense_bits = Vec::new();
                dense_bits
                    .try_reserve_exact(total_dense_words)
                    .map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "cannot reserve hybrid dense payload".to_string(),
                        )
                    })?;
                for _ in 0..total_dense_words {
                    dense_bits.push(read_u64(&mut reader)?);
                }
                validate_bitslice_padding(&dense_bits, num_dense, words_per_label, num_vectors)?;

                let offset_count = num_sparse.checked_add(1).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid(
                        "hybrid sparse offset count overflow".to_string(),
                    )
                })?;
                let offset_bytes = offset_count.checked_mul(8).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid(
                        "hybrid sparse offset byte size overflow".to_string(),
                    )
                })?;
                ensure_remaining(&mut reader, file_len, offset_bytes, "hybrid sparse offsets")?;
                let mut sparse_offsets = Vec::new();
                sparse_offsets
                    .try_reserve_exact(offset_count)
                    .map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "cannot reserve hybrid sparse offsets".to_string(),
                        )
                    })?;
                for _ in 0..offset_count {
                    sparse_offsets.push(read_u64(&mut reader)?);
                }
                if sparse_offsets.first().copied() != Some(0)
                    || sparse_offsets.windows(2).any(|pair| pair[0] > pair[1])
                {
                    return Err(EncodedLabelIndexError::Invalid(
                        "hybrid sparse offsets are not monotonic from zero".to_string(),
                    ));
                }

                let posting_count = usize::try_from(sparse_offsets.last().copied().unwrap_or(0))
                    .map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "hybrid sparse posting count exceeds usize".to_string(),
                        )
                    })?;
                let posting_bytes = posting_count.checked_mul(4).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid(
                        "hybrid sparse posting byte size overflow".to_string(),
                    )
                })?;
                ensure_remaining(
                    &mut reader,
                    file_len,
                    posting_bytes,
                    "hybrid sparse postings",
                )?;
                let mut sparse_doc_ids = Vec::new();
                sparse_doc_ids
                    .try_reserve_exact(posting_count)
                    .map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "cannot reserve hybrid sparse postings".to_string(),
                        )
                    })?;
                for _ in 0..posting_count {
                    sparse_doc_ids.push(read_u32(&mut reader)?);
                }

                for descriptor in &descriptors {
                    match descriptor.kind {
                        HybridLabelKind::Dense => {
                            let start = descriptor.ordinal as usize * words_per_label;
                            let end = start + words_per_label;
                            let cardinality = dense_bits[start..end]
                                .iter()
                                .map(|word| word.count_ones())
                                .sum::<u32>();
                            if cardinality != descriptor.cardinality {
                                return Err(EncodedLabelIndexError::Invalid(
                                    "hybrid dense cardinality does not match payload".to_string(),
                                ));
                            }
                        }
                        HybridLabelKind::Sparse => {
                            let ordinal = descriptor.ordinal as usize;
                            let start = sparse_offsets[ordinal] as usize;
                            let end = sparse_offsets[ordinal + 1] as usize;
                            let posting = &sparse_doc_ids[start..end];
                            if posting.len() != descriptor.cardinality as usize {
                                return Err(EncodedLabelIndexError::Invalid(
                                    "hybrid sparse cardinality does not match payload".to_string(),
                                ));
                            }
                            if posting.last().is_some_and(|id| *id >= num_vectors)
                                || posting.windows(2).any(|pair| pair[0] >= pair[1])
                            {
                                return Err(EncodedLabelIndexError::Invalid(
                                    "hybrid sparse posting is unsorted or out of range".to_string(),
                                ));
                            }
                        }
                    }
                }

                LabelStorage::Hybrid {
                    storage: Arc::new(HybridStorage {
                        words_per_label,
                        descriptors: descriptors.into_boxed_slice(),
                        dense_bits: dense_bits.into_boxed_slice(),
                        sparse_offsets: sparse_offsets.into_boxed_slice(),
                        sparse_doc_ids: sparse_doc_ids.into_boxed_slice(),
                    }),
                }
            }
        };

        if reader.read(&mut [0u8; 1])? != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "label-index contains trailing bytes".to_string(),
            ));
        }

        Ok(Self {
            labels: labels.into_boxed_slice(),
            label_ids,
            num_vectors,
            storage,
        })
    }

    /// Compile a clause-list query using DNF/CNF semantics.
    ///
    /// Use [`FilterExpressionType::DNF`] or [`FilterExpressionType::CNF`] for the preferred
    /// terminology. The legacy [`FilterExpressionType::ORMajor`] and
    /// [`FilterExpressionType::ANDMajor`] variants remain supported for compatibility. The
    /// returned [`EncodedLabelQuery`] owns its compiled expression and any shared storage handles.
    pub fn query<S>(
        &self,
        clauses: &[S],
        expression_type: FilterExpressionType,
    ) -> Result<EncodedLabelQuery<'static>, EncodedLabelIndexError>
    where
        S: AsRef<str>,
    {
        let expression = match expression_type {
            FilterExpressionType::ORMajor => {
                CompiledExpression::Flat(compile_plan(clauses, PlanKind::Dnf, &self.label_ids)?)
            }
            FilterExpressionType::ANDMajor => {
                CompiledExpression::Flat(compile_plan(clauses, PlanKind::Cnf, &self.label_ids)?)
            }
        };

        self.build_query(expression)
    }

    /// Compile an owned query from a recursive label expression.
    pub fn query_expression(
        &self,
        expression: &LabelExpression,
    ) -> Result<EncodedLabelQuery<'static>, EncodedLabelIndexError> {
        self.build_query(compile_label_expression(expression, &self.label_ids)?)
    }

    /// Compile an owned query from a JSON-encoded recursive label expression.
    pub fn query_ast_json(
        &self,
        expression_json: &str,
    ) -> Result<EncodedLabelQuery<'static>, EncodedLabelIndexError> {
        let expression = parse_label_expression_json(expression_json)?;
        self.query_expression(&expression)
    }

    fn build_query(
        &self,
        expression: CompiledExpression,
    ) -> Result<EncodedLabelQuery<'static>, EncodedLabelIndexError> {
        let storage = match &self.storage {
            LabelStorage::Bitslice {
                words_per_label,
                bits,
            } => QueryStorage::Bitslice {
                words_per_label: *words_per_label,
                bits: Arc::clone(bits),
                expression,
            },
            LabelStorage::Bitmap { postings } => QueryStorage::DenseBitmap {
                bits: materialize_bitmap(&expression, postings, self.num_vectors)?
                    .into_boxed_slice(),
            },
            LabelStorage::Hybrid { storage } => {
                let mut expression = expression;
                expression.optimize_hybrid(storage);
                QueryStorage::Hybrid {
                    storage: Arc::clone(storage),
                    expression,
                }
            }
        };

        Ok(EncodedLabelQuery {
            num_vectors: self.num_vectors,
            storage,
            _lifetime: PhantomData,
        })
    }

    /// Return the persisted storage format backing this index.
    pub fn format(&self) -> LabelIndexFormat {
        match &self.storage {
            LabelStorage::Bitslice { .. } => LabelIndexFormat::Bitslice,
            LabelStorage::Bitmap { .. } => LabelIndexFormat::Bitmap,
            LabelStorage::Hybrid { .. } => LabelIndexFormat::Hybrid,
        }
    }
}

/// Encode a JSONL label file into a versioned Bitslice, Roaring-bitmap, or hybrid label index.
///
/// Supported JSONL rows are:
///
/// - objects with optional `doc_id`; `true` fields use the field name, other scalar values use
///   `field=value`;
/// - objects with a `labels` string array;
/// - a raw string or string array, using the line number as the document ID.
pub fn encode_label_index_jsonl(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    format: LabelIndexFormat,
) -> Result<(), EncodedLabelIndexError> {
    encode_jsonl(input_path, output_path, format)
}

/// Encode a hybrid label index from a prebuilt dictionary and posting lists.
pub fn encode_hybrid_label_index_postings(
    output_path: impl AsRef<Path>,
    num_vectors: u32,
    labels: &[String],
    postings: &[RoaringBitmap],
    options: HybridBuildOptions,
) -> Result<HybridBuildStats, EncodedLabelIndexError> {
    write_label_index_with_options(
        output_path,
        LabelIndexFormat::Hybrid,
        num_vectors,
        labels,
        postings,
        options,
    )?
    .ok_or_else(|| {
        EncodedLabelIndexError::Invalid("hybrid encoder did not produce storage stats".to_string())
    })
}

/// Parse a JSON-encoded recursive label expression.
pub fn parse_label_expression_json(
    expression_json: &str,
) -> Result<LabelExpression, EncodedLabelIndexError> {
    let mut deserializer = serde_json::Deserializer::from_str(expression_json);
    let mut budget = LabelExpressionBudget::new();
    let expression = LabelExpressionSeed {
        budget: &mut budget,
        depth: 1,
    }
    .deserialize(&mut deserializer)?;
    deserializer.end()?;
    validate_label_expression(&expression)?;
    Ok(expression)
}

fn compile_plan<S: AsRef<str>>(
    clauses: &[S],
    kind: PlanKind,
    label_ids: &HashMap<String, u32>,
) -> Result<CompiledPlan, EncodedLabelIndexError> {
    if clauses.is_empty() {
        return Err(EncodedLabelIndexError::Invalid(
            "filter must contain at least one clause".to_string(),
        ));
    }

    let delimiter = match kind {
        PlanKind::Dnf => '&',
        PlanKind::Cnf => '|',
    };
    let mut clause_offsets = vec![0usize];
    let mut encoded = Vec::new();

    for clause in clauses {
        let mut terminal_count = 0usize;
        for terminal in clause.as_ref().split(delimiter) {
            let terminal = terminal.trim();
            validate_label(terminal)?;
            encoded.push(label_ids.get(terminal).copied());
            terminal_count += 1;
        }
        if terminal_count == 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "filter clauses cannot be empty".to_string(),
            ));
        }
        clause_offsets.push(encoded.len());
    }

    Ok(CompiledPlan {
        kind,
        clause_offsets: clause_offsets.into_boxed_slice(),
        label_ids: encoded.into_boxed_slice(),
    })
}

fn compile_label_expression(
    expression: &LabelExpression,
    label_ids: &HashMap<String, u32>,
) -> Result<CompiledExpression, EncodedLabelIndexError> {
    validate_label_expression(expression)?;
    Ok(compile_label_expression_inner(expression, label_ids))
}

fn compile_label_expression_inner(
    expression: &LabelExpression,
    label_ids: &HashMap<String, u32>,
) -> CompiledExpression {
    match expression {
        LabelExpression::Label(label) => CompiledExpression::Label(label_ids.get(label).copied()),
        LabelExpression::And(children) => CompiledExpression::And(
            children
                .iter()
                .map(|child| compile_label_expression_inner(child, label_ids))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        LabelExpression::Or(children) => CompiledExpression::Or(
            children
                .iter()
                .map(|child| compile_label_expression_inner(child, label_ids))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        LabelExpression::Not(child) => {
            CompiledExpression::Not(Box::new(compile_label_expression_inner(child, label_ids)))
        }
    }
}

fn validate_label_expression(expression: &LabelExpression) -> Result<(), EncodedLabelIndexError> {
    let mut node_count = 0usize;
    let mut stack = vec![(expression, 1usize)];

    while let Some((expression, depth)) = stack.pop() {
        if depth > MAX_LABEL_EXPRESSION_DEPTH {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "label expression depth {depth} exceeds limit {MAX_LABEL_EXPRESSION_DEPTH}"
            )));
        }

        node_count += 1;
        if node_count > MAX_LABEL_EXPRESSION_NODES {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "label expression node count {node_count} exceeds limit {MAX_LABEL_EXPRESSION_NODES}"
            )));
        }

        match expression {
            LabelExpression::Label(label) => validate_label(label)?,
            LabelExpression::And(children) => {
                if children.is_empty() {
                    return Err(EncodedLabelIndexError::Invalid(
                        "label expression 'and' array cannot be empty".to_string(),
                    ));
                }
                stack.extend(children.iter().rev().map(|child| (child, depth + 1)));
            }
            LabelExpression::Or(children) => {
                if children.is_empty() {
                    return Err(EncodedLabelIndexError::Invalid(
                        "label expression 'or' array cannot be empty".to_string(),
                    ));
                }
                stack.extend(children.iter().rev().map(|child| (child, depth + 1)));
            }
            LabelExpression::Not(child) => stack.push((child.as_ref(), depth + 1)),
        }
    }

    Ok(())
}

fn materialize_bitmap(
    expression: &CompiledExpression,
    postings: &[RoaringBitmap],
    num_vectors: u32,
) -> Result<Vec<u64>, EncodedLabelIndexError> {
    let result = materialize_bitmap_expression(expression, postings, num_vectors)?;
    densify(&result, num_vectors)
}

fn materialize_bitmap_expression(
    expression: &CompiledExpression,
    postings: &[RoaringBitmap],
    num_vectors: u32,
) -> Result<RoaringBitmap, EncodedLabelIndexError> {
    match expression {
        CompiledExpression::Flat(plan) => Ok(materialize_bitmap_plan(plan, postings)),
        CompiledExpression::Label(Some(label_id)) => Ok(postings[*label_id as usize].clone()),
        CompiledExpression::Label(None) => Ok(RoaringBitmap::new()),
        CompiledExpression::And(children) => {
            let Some((first, rest)) = children.split_first() else {
                return Err(EncodedLabelIndexError::Invalid(
                    "label expression 'and' array cannot be empty".to_string(),
                ));
            };
            let mut result = materialize_bitmap_expression(first, postings, num_vectors)?;
            for child in rest {
                result &= materialize_bitmap_expression(child, postings, num_vectors)?;
                if result.is_empty() {
                    break;
                }
            }
            Ok(result)
        }
        CompiledExpression::Or(children) => {
            if children.is_empty() {
                return Err(EncodedLabelIndexError::Invalid(
                    "label expression 'or' array cannot be empty".to_string(),
                ));
            }
            let mut result = RoaringBitmap::new();
            for child in children.iter() {
                result |= materialize_bitmap_expression(child, postings, num_vectors)?;
            }
            Ok(result)
        }
        CompiledExpression::Not(child) => {
            let mut result = full_bitmap(num_vectors);
            result -= materialize_bitmap_expression(child, postings, num_vectors)?;
            Ok(result)
        }
    }
}

fn materialize_bitmap_plan(plan: &CompiledPlan, postings: &[RoaringBitmap]) -> RoaringBitmap {
    match plan.kind {
        PlanKind::Dnf => {
            let mut result = RoaringBitmap::new();
            for clause in plan.clause_offsets.windows(2) {
                let labels = &plan.label_ids[clause[0]..clause[1]];
                if labels.iter().any(Option::is_none) {
                    continue;
                }
                let mut labels = labels.iter().filter_map(|label| *label);
                let Some(first) = labels.next() else {
                    continue;
                };
                let mut clause_result = postings[first as usize].clone();
                for label in labels {
                    clause_result &= &postings[label as usize];
                    if clause_result.is_empty() {
                        break;
                    }
                }
                result |= clause_result;
            }
            result
        }
        PlanKind::Cnf => {
            let mut result: Option<RoaringBitmap> = None;
            for clause in plan.clause_offsets.windows(2) {
                let mut clause_result = RoaringBitmap::new();
                for label in plan.label_ids[clause[0]..clause[1]]
                    .iter()
                    .filter_map(|label| *label)
                {
                    clause_result |= &postings[label as usize];
                }
                result = Some(match result {
                    None => clause_result,
                    Some(mut result) => {
                        result &= clause_result;
                        result
                    }
                });
                if result.as_ref().is_some_and(RoaringBitmap::is_empty) {
                    break;
                }
            }
            result.unwrap_or_default()
        }
    }
}

fn full_bitmap(num_vectors: u32) -> RoaringBitmap {
    let mut result = RoaringBitmap::new();
    result.insert_range(0..num_vectors);
    result
}

fn dense_contains(bits: &[u64], vec_id: u32) -> bool {
    let vec_id = vec_id as usize;
    bits[vec_id / 64] & (1u64 << (vec_id % 64)) != 0
}

fn densify(result: &RoaringBitmap, num_vectors: u32) -> Result<Vec<u64>, EncodedLabelIndexError> {
    let words = (num_vectors as usize).div_ceil(64);
    let bytes = words.checked_mul(8).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("dense bitmap byte size overflow".to_string())
    })?;
    if bytes > MAX_DENSE_BITMAP_BYTES {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "dense query bitmap requires {bytes} bytes; limit is {MAX_DENSE_BITMAP_BYTES}"
        )));
    }
    let mut bits = Vec::new();
    bits.try_reserve_exact(words).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve dense query bitmap".to_string())
    })?;
    bits.resize(words, 0);
    for vec_id in result {
        let vec_id = vec_id as usize;
        bits[vec_id / 64] |= 1u64 << (vec_id % 64);
    }
    Ok(bits)
}

fn encode_jsonl(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    format: LabelIndexFormat,
) -> Result<(), EncodedLabelIndexError> {
    let reader = BufReader::new(File::open(input_path)?);
    let mut labels = Vec::<String>::new();
    let mut label_ids = HashMap::<String, u32>::new();
    let mut postings = Vec::<RoaringBitmap>::new();
    let mut max_doc_id = None::<u32>;
    let mut seen_doc_ids = HashSet::<u32>::new();

    let mut record_number = 0u32;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let default_doc_id = record_number;
        record_number = record_number.checked_add(1).ok_or_else(|| {
            EncodedLabelIndexError::Invalid("JSONL record count exceeds u32".to_string())
        })?;
        let (doc_id, document_labels) = parse_document(&line, default_doc_id)?;
        if !seen_doc_ids.insert(doc_id) {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "duplicate doc_id {doc_id} in label JSONL"
            )));
        }
        max_doc_id = Some(max_doc_id.map_or(doc_id, |current| current.max(doc_id)));

        for label in document_labels {
            let label_id = if let Some(&label_id) = label_ids.get(&label) {
                label_id
            } else {
                if labels.len() >= MAX_LABEL_COUNT {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "label count exceeds limit {MAX_LABEL_COUNT}"
                    )));
                }
                let label_id = u32::try_from(labels.len()).map_err(|_| {
                    EncodedLabelIndexError::Invalid("label count exceeds u32".to_string())
                })?;
                label_ids.insert(label.clone(), label_id);
                labels.push(label);
                postings.push(RoaringBitmap::new());
                label_id
            };
            postings[label_id as usize].insert(doc_id);
        }
    }

    let num_vectors = max_doc_id
        .and_then(|doc_id| doc_id.checked_add(1))
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("label JSONL contains no documents".to_string())
        })?;
    write_label_index(output_path, format, num_vectors, &labels, &postings)
}

fn parse_document(
    line: &str,
    default_doc_id: u32,
) -> Result<(u32, Vec<String>), EncodedLabelIndexError> {
    let value: Value = serde_json::from_str(line)?;
    match value {
        Value::String(label) => {
            validate_label(&label)?;
            Ok((default_doc_id, vec![label]))
        }
        Value::Array(labels) => Ok((
            default_doc_id,
            parse_direct_label_array(&labels, "root label array")?,
        )),
        Value::Object(fields) => parse_object_document(fields, default_doc_id),
        _ => Err(EncodedLabelIndexError::Invalid(
            "each JSONL line must be an object, string, or string array".to_string(),
        )),
    }
}

fn parse_object_document(
    fields: Map<String, Value>,
    default_doc_id: u32,
) -> Result<(u32, Vec<String>), EncodedLabelIndexError> {
    let doc_id = match fields.get("doc_id") {
        Some(value) => {
            let value = value.as_u64().ok_or_else(|| {
                EncodedLabelIndexError::Invalid("doc_id must be a non-negative integer".to_string())
            })?;
            u32::try_from(value)
                .map_err(|_| EncodedLabelIndexError::Invalid("doc_id exceeds u32".to_string()))?
        }
        None => default_doc_id,
    };

    let mut labels = Vec::new();
    for (field, value) in fields {
        if field == "doc_id" {
            continue;
        }
        if field == "labels" {
            if let Value::Array(values) = &value {
                labels.extend(parse_direct_label_array(values, "labels field")?);
                continue;
            }
        }

        let label = match value {
            Value::Bool(true) => field,
            Value::Bool(false) => format!("{field}=false"),
            Value::Number(value) => format!("{field}={value}"),
            Value::String(value) => format!("{field}={value}"),
            Value::Null => format!("{field}=null"),
            Value::Array(_) | Value::Object(_) => {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "field '{field}' must contain a scalar value"
                )));
            }
        };
        validate_label(&label)?;
        labels.push(label);
    }

    Ok((doc_id, labels))
}

fn parse_direct_label_array(
    values: &[Value],
    context: &str,
) -> Result<Vec<String>, EncodedLabelIndexError> {
    values
        .iter()
        .map(|value| {
            let label = value.as_str().ok_or_else(|| {
                EncodedLabelIndexError::Invalid(format!("{context} must contain only strings"))
            })?;
            validate_label(label)?;
            Ok(label.to_string())
        })
        .collect()
}

fn validate_label(label: &str) -> Result<(), EncodedLabelIndexError> {
    if label.is_empty() {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot be empty".to_string(),
        ));
    }
    if label.contains(['&', '|']) {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label '{label}' contains a reserved filter delimiter"
        )));
    }
    if label.as_bytes().contains(&0) {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot contain embedded NUL bytes".to_string(),
        ));
    }
    if label.len() > MAX_LABEL_LENGTH {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label length {} exceeds limit {MAX_LABEL_LENGTH}",
            label.len()
        )));
    }
    if label.trim() != label {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot have leading or trailing whitespace".to_string(),
        ));
    }
    Ok(())
}

fn validate_bitslice_padding(
    bits: &[u64],
    num_labels: usize,
    words_per_label: usize,
    num_vectors: u32,
) -> Result<(), EncodedLabelIndexError> {
    let remainder = (num_vectors as usize) % 64;
    if remainder == 0 || words_per_label == 0 {
        return Ok(());
    }

    let valid_mask = (1u64 << remainder) - 1;
    for label_id in 0..num_labels {
        let last_word = bits[(label_id + 1) * words_per_label - 1];
        if last_word & !valid_mask != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "bitslice padding contains out-of-range vector IDs".to_string(),
            ));
        }
    }
    Ok(())
}

fn write_label_index(
    path: impl AsRef<Path>,
    format: LabelIndexFormat,
    num_vectors: u32,
    labels: &[String],
    postings: &[RoaringBitmap],
) -> Result<(), EncodedLabelIndexError> {
    write_label_index_with_options(
        path,
        format,
        num_vectors,
        labels,
        postings,
        HybridBuildOptions::default(),
    )
    .map(|_| ())
}

fn write_label_index_with_options(
    path: impl AsRef<Path>,
    format: LabelIndexFormat,
    num_vectors: u32,
    labels: &[String],
    postings: &[RoaringBitmap],
    hybrid_options: HybridBuildOptions,
) -> Result<Option<HybridBuildStats>, EncodedLabelIndexError> {
    if labels.len() > MAX_LABEL_COUNT {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label count {} exceeds limit {MAX_LABEL_COUNT}",
            labels.len()
        )));
    }
    if labels.len() != postings.len() {
        return Err(EncodedLabelIndexError::Invalid(
            "label dictionary and posting counts differ".to_string(),
        ));
    }
    if num_vectors == 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index vector count cannot be zero".to_string(),
        ));
    }
    if format == LabelIndexFormat::Bitmap && u64::from(num_vectors) > MAX_BITMAP_VECTORS {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "bitmap vector count {num_vectors} exceeds limit {MAX_BITMAP_VECTORS}"
        )));
    }
    if format == LabelIndexFormat::Hybrid
        && postings
            .iter()
            .any(|posting| posting.max().is_some_and(|id| id >= num_vectors))
    {
        return Err(EncodedLabelIndexError::Invalid(
            "posting contains an out-of-range vector ID".to_string(),
        ));
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&LABEL_INDEX_MAGIC)?;
    write_u32(&mut writer, LABEL_INDEX_VERSION)?;
    write_u32(&mut writer, format as u32)?;
    write_u64(&mut writer, u64::from(num_vectors))?;
    write_u64(&mut writer, labels.len() as u64)?;

    for label in labels {
        validate_label(label)?;
        let bytes = label.as_bytes();
        let len = u32::try_from(bytes.len()).map_err(|_| {
            EncodedLabelIndexError::Invalid(format!("label '{label}' is too long to encode"))
        })?;
        write_u32(&mut writer, len)?;
        writer.write_all(bytes)?;
    }

    let hybrid_stats = match format {
        LabelIndexFormat::Bitslice => {
            let words_per_label = (num_vectors as usize).div_ceil(64);
            write_u64(&mut writer, words_per_label as u64)?;

            let total_words = labels.len().checked_mul(words_per_label).ok_or_else(|| {
                EncodedLabelIndexError::Invalid("bitslice output size overflow".to_string())
            })?;
            let mut bits = Vec::new();
            bits.try_reserve_exact(total_words).map_err(|_| {
                EncodedLabelIndexError::Invalid("cannot reserve bitslice output".to_string())
            })?;
            bits.resize(total_words, 0);

            for (label_id, posting) in postings.iter().enumerate() {
                let row = &mut bits[label_id * words_per_label..(label_id + 1) * words_per_label];
                for vec_id in posting {
                    let vec_id = vec_id as usize;
                    row[vec_id / 64] |= 1u64 << (vec_id % 64);
                }
            }

            for word in bits {
                write_u64(&mut writer, word)?;
            }
            None
        }
        LabelIndexFormat::Bitmap => {
            for posting in postings {
                let serialized_size = posting.serialized_size();
                if serialized_size > MAX_POSTING_BYTES {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "serialized posting length {serialized_size} exceeds limit {MAX_POSTING_BYTES}"
                    )));
                }

                let mut bytes = Vec::new();
                bytes.try_reserve_exact(serialized_size).map_err(|_| {
                    EncodedLabelIndexError::Invalid("cannot reserve serialized posting".to_string())
                })?;
                posting.serialize_into(&mut bytes)?;
                write_u64(&mut writer, bytes.len() as u64)?;
                writer.write_all(&bytes)?;
            }
            None
        }
        LabelIndexFormat::Hybrid => Some(write_hybrid_payload(
            &mut writer,
            num_vectors,
            postings,
            hybrid_options,
        )?),
    };

    writer.flush()?;
    Ok(hybrid_stats)
}

fn write_hybrid_payload(
    writer: &mut impl Write,
    num_vectors: u32,
    postings: &[RoaringBitmap],
    options: HybridBuildOptions,
) -> Result<HybridBuildStats, EncodedLabelIndexError> {
    let words_per_label = (num_vectors as usize).div_ceil(64);
    let break_even = u32::try_from(words_per_label.saturating_mul(2))
        .unwrap_or(u32::MAX)
        .max(1);
    let dense_threshold = options.dense_threshold.unwrap_or(break_even).max(1);

    let dense_labels = postings
        .iter()
        .filter(|posting| posting.len() >= u64::from(dense_threshold))
        .count();
    let sparse_labels = postings.len() - dense_labels;
    let total_dense_words = dense_labels.checked_mul(words_per_label).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("hybrid dense allocation size overflow".to_string())
    })?;

    let mut descriptors = Vec::new();
    descriptors
        .try_reserve_exact(postings.len())
        .map_err(|_| EncodedLabelIndexError::Invalid("cannot reserve hybrid descriptors".into()))?;
    let mut dense_bits = Vec::new();
    dense_bits
        .try_reserve_exact(total_dense_words)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve hybrid dense payload".to_string())
        })?;
    dense_bits.resize(total_dense_words, 0);

    let sparse_posting_count = postings
        .iter()
        .filter(|posting| posting.len() < u64::from(dense_threshold))
        .try_fold(0usize, |total, posting| {
            let len = usize::try_from(posting.len()).map_err(|_| {
                EncodedLabelIndexError::Invalid(
                    "hybrid sparse posting length exceeds usize".to_string(),
                )
            })?;
            total.checked_add(len).ok_or_else(|| {
                EncodedLabelIndexError::Invalid("hybrid sparse posting count overflow".to_string())
            })
        })?;
    let mut sparse_offsets = Vec::new();
    sparse_offsets
        .try_reserve_exact(sparse_labels + 1)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve hybrid sparse offsets".to_string())
        })?;
    sparse_offsets.push(0);
    let mut sparse_doc_ids = Vec::new();
    sparse_doc_ids
        .try_reserve_exact(sparse_posting_count)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve hybrid sparse postings".to_string())
        })?;

    let mut dense_ordinal = 0u32;
    let mut sparse_ordinal = 0u32;
    for posting in postings {
        let cardinality = u32::try_from(posting.len()).map_err(|_| {
            EncodedLabelIndexError::Invalid("hybrid label cardinality exceeds u32".to_string())
        })?;
        if cardinality >= dense_threshold {
            let row_start = dense_ordinal as usize * words_per_label;
            let row = &mut dense_bits[row_start..row_start + words_per_label];
            for vec_id in posting {
                let vec_id = vec_id as usize;
                row[vec_id / 64] |= 1u64 << (vec_id % 64);
            }
            descriptors.push(HybridLabelDescriptor {
                kind: HybridLabelKind::Dense,
                ordinal: dense_ordinal,
                cardinality,
            });
            dense_ordinal += 1;
        } else {
            sparse_doc_ids.extend(posting.iter());
            sparse_offsets.push(sparse_doc_ids.len() as u64);
            descriptors.push(HybridLabelDescriptor {
                kind: HybridLabelKind::Sparse,
                ordinal: sparse_ordinal,
                cardinality,
            });
            sparse_ordinal += 1;
        }
    }

    write_u64(writer, words_per_label as u64)?;
    write_u64(writer, dense_labels as u64)?;
    write_u64(writer, sparse_labels as u64)?;
    for descriptor in &descriptors {
        write_u32(
            writer,
            match descriptor.kind {
                HybridLabelKind::Dense => 0,
                HybridLabelKind::Sparse => 1,
            },
        )?;
        write_u32(writer, descriptor.ordinal)?;
        write_u32(writer, descriptor.cardinality)?;
    }
    for word in dense_bits {
        write_u64(writer, word)?;
    }
    for offset in &sparse_offsets {
        write_u64(writer, *offset)?;
    }
    for doc_id in sparse_doc_ids {
        write_u32(writer, doc_id)?;
    }

    let dense_bytes = u64::try_from(total_dense_words)
        .ok()
        .and_then(|words| words.checked_mul(8))
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("hybrid dense byte size overflow".to_string())
        })?;
    let sparse_bytes = u64::try_from(sparse_posting_count)
        .ok()
        .and_then(|count| count.checked_mul(4))
        .and_then(|postings| {
            u64::try_from(sparse_offsets.len())
                .ok()
                .and_then(|offsets| offsets.checked_mul(8))
                .and_then(|offsets| postings.checked_add(offsets))
        })
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("hybrid sparse byte size overflow".to_string())
        })?;

    Ok(HybridBuildStats {
        dense_threshold,
        dense_labels: dense_ordinal,
        sparse_labels: sparse_ordinal,
        dense_bytes,
        sparse_bytes,
    })
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn ensure_remaining(
    reader: &mut BufReader<File>,
    file_len: u64,
    required: usize,
    context: &str,
) -> Result<(), EncodedLabelIndexError> {
    let position = reader.stream_position()?;
    let remaining = file_len.checked_sub(position).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("reader position exceeds file length".to_string())
    })?;
    if required as u64 > remaining {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "{context} requires {required} bytes but only {remaining} remain"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    fn sample_jsonl() -> &'static str {
        concat!(
            "{\"doc_id\":0,\"A\":true,\"group\":\"x\"}\n",
            "{\"doc_id\":1,\"B\":true,\"group\":\"x\"}\n",
            "{\"doc_id\":2,\"A\":true,\"B\":true,\"score\":2}\n",
            "{\"doc_id\":3,\"labels\":[\"C\",\"D\"]}\n",
        )
    }

    fn compile(
        index: &EncodedLabelIndex,
        clauses: &[&str],
        expression_type: FilterExpressionType,
    ) -> EncodedLabelQuery<'static> {
        index.query(clauses, expression_type).unwrap()
    }

    fn matching_ids(query: &EncodedLabelQuery, num_vectors: u32) -> Vec<u32> {
        (0..num_vectors)
            .filter(|&vec_id| query.is_match(vec_id))
            .collect()
    }

    fn assert_send_sync_static<T: Send + Sync + 'static>(_: &T) {}

    fn round_trip(format: LabelIndexFormat) -> EncodedLabelIndex {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, sample_jsonl()).unwrap();
        encode_label_index_jsonl(&input, &output, format).unwrap();
        EncodedLabelIndex::load(output).unwrap()
    }

    fn nested_not_expression(wrappings: usize) -> LabelExpression {
        let mut expression = LabelExpression::Label("A".to_string());
        for _ in 0..wrappings {
            expression = LabelExpression::Not(Box::new(expression));
        }
        expression
    }

    fn nested_not_json(wrappings: usize) -> String {
        let mut expression_json = "\"A\"".to_string();
        for _ in 0..wrappings {
            expression_json = format!(r#"{{"not":{expression_json}}}"#);
        }
        expression_json
    }

    #[test]
    fn hybrid_round_trip_mixes_dense_and_sparse_labels() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("labels.hybrid");
        let labels = vec!["dense".to_string(), "sparse".to_string()];
        let postings = vec![
            RoaringBitmap::from_iter([0, 1, 2]),
            RoaringBitmap::from_iter([2]),
        ];

        let stats = encode_hybrid_label_index_postings(
            &output,
            4,
            &labels,
            &postings,
            HybridBuildOptions {
                dense_threshold: Some(2),
            },
        )
        .unwrap();
        assert_eq!(
            stats,
            HybridBuildStats {
                dense_threshold: 2,
                dense_labels: 1,
                sparse_labels: 1,
                dense_bytes: 8,
                sparse_bytes: 20,
            }
        );

        let index = EncodedLabelIndex::load(output).unwrap();
        assert_eq!(index.format(), LabelIndexFormat::Hybrid);
        assert_eq!(
            matching_ids(&compile(&index, &["dense"], FilterExpressionType::DNF), 4),
            vec![0, 1, 2]
        );
        assert_eq!(
            matching_ids(&compile(&index, &["sparse"], FilterExpressionType::DNF), 4),
            vec![2]
        );
        assert_eq!(
            matching_ids(
                &compile(&index, &["dense&sparse"], FilterExpressionType::DNF),
                4
            ),
            vec![2]
        );
        let reversed = compile(&index, &["sparse&dense"], FilterExpressionType::DNF);
        assert_eq!(matching_ids(&reversed, 4), vec![2]);
        let QueryStorage::Hybrid {
            storage,
            expression: CompiledExpression::Flat(plan),
        } = &reversed.storage
        else {
            panic!("expected a flat hybrid query");
        };
        assert_eq!(
            storage.descriptor(plan.label_ids[0]).unwrap().kind,
            HybridLabelKind::Dense
        );
    }

    #[test]
    fn persisted_formats_match_for_dnf() {
        for format in [
            LabelIndexFormat::Bitslice,
            LabelIndexFormat::Bitmap,
            LabelIndexFormat::Hybrid,
        ] {
            let index = round_trip(format);
            let query = compile(&index, &["A&B", "C&D"], FilterExpressionType::DNF);
            assert!(!query.is_match(0));
            assert!(!query.is_match(1));
            assert!(query.is_match(2));
            assert!(query.is_match(3));
        }
    }

    #[test]
    fn persisted_formats_match_for_cnf() {
        for format in [
            LabelIndexFormat::Bitslice,
            LabelIndexFormat::Bitmap,
            LabelIndexFormat::Hybrid,
        ] {
            let index = round_trip(format);
            let query = compile(
                &index,
                &["A|B", "group=x|score=2"],
                FilterExpressionType::CNF,
            );
            assert!(query.is_match(0));
            assert!(query.is_match(1));
            assert!(query.is_match(2));
            assert!(!query.is_match(3));
        }
    }

    #[test]
    fn query_accepts_owned_strings() {
        let index = round_trip(LabelIndexFormat::Bitmap);
        let clauses = vec!["A&B".to_string(), "C&D".to_string()];
        let query = index.query(&clauses, FilterExpressionType::DNF).unwrap();
        assert!(!query.is_match(0));
        assert!(query.is_match(2));
        assert!(query.is_match(3));
    }

    #[test]
    fn unknown_labels_are_non_matches() {
        let index = round_trip(LabelIndexFormat::Bitslice);
        let query = compile(&index, &["missing"], FilterExpressionType::DNF);
        for id in 0..4 {
            assert!(!query.is_match(id));
        }
    }

    #[test]
    fn ast_matches_nested_expression_for_persisted_formats() {
        let expression = r#"{"or":[{"and":["A","B"]},{"and":["C","D"]}]}"#;

        for format in [
            LabelIndexFormat::Bitslice,
            LabelIndexFormat::Bitmap,
            LabelIndexFormat::Hybrid,
        ] {
            let index = round_trip(format);
            let query = index.query_ast_json(expression).unwrap();
            assert_eq!(matching_ids(&query, index.num_vectors), vec![2, 3]);
        }
    }

    #[test]
    fn ast_matches_equivalent_dnf_and_cnf_queries() {
        for format in [
            LabelIndexFormat::Bitslice,
            LabelIndexFormat::Bitmap,
            LabelIndexFormat::Hybrid,
        ] {
            let index = round_trip(format);

            let dnf = compile(&index, &["A&B", "C&D"], FilterExpressionType::DNF);
            let ast_dnf = index
                .query_ast_json(r#"{"or":[{"and":["A","B"]},{"and":["C","D"]}]}"#)
                .unwrap();
            assert_eq!(
                matching_ids(&dnf, index.num_vectors),
                matching_ids(&ast_dnf, index.num_vectors)
            );

            let cnf = compile(
                &index,
                &["A|B", "group=x|score=2"],
                FilterExpressionType::CNF,
            );
            let ast_cnf = index
                .query_expression(&LabelExpression::And(vec![
                    LabelExpression::Or(vec![
                        LabelExpression::Label("A".to_string()),
                        LabelExpression::Label("B".to_string()),
                    ]),
                    LabelExpression::Or(vec![
                        LabelExpression::Label("group=x".to_string()),
                        LabelExpression::Label("score=2".to_string()),
                    ]),
                ]))
                .unwrap();
            assert_eq!(
                matching_ids(&cnf, index.num_vectors),
                matching_ids(&ast_cnf, index.num_vectors)
            );
        }
    }

    #[test]
    fn ast_not_supports_known_and_unknown_labels() {
        for format in [
            LabelIndexFormat::Bitslice,
            LabelIndexFormat::Bitmap,
            LabelIndexFormat::Hybrid,
        ] {
            let index = round_trip(format);
            let not_a = index.query_ast_json(r#"{"not":"A"}"#).unwrap();
            assert_eq!(matching_ids(&not_a, index.num_vectors), vec![1, 3]);

            let not_missing = index.query_ast_json(r#"{"not":"missing"}"#).unwrap();
            assert_eq!(
                matching_ids(&not_missing, index.num_vectors),
                vec![0, 1, 2, 3]
            );
        }
    }

    #[test]
    fn compiled_queries_remain_usable_after_index_drop() {
        for format in [
            LabelIndexFormat::Bitslice,
            LabelIndexFormat::Bitmap,
            LabelIndexFormat::Hybrid,
        ] {
            let query = {
                let index = round_trip(format);
                Arc::new(
                    index
                        .query_ast_json(r#"{"or":[{"and":["A","B"]},{"and":["C","D"]}]}"#)
                        .unwrap(),
                )
            };
            assert_send_sync_static(query.as_ref());
            assert_eq!(matching_ids(&query, 4), vec![2, 3]);
        }
    }

    #[test]
    fn parse_label_expression_json_rejects_malformed_inputs() {
        assert!(parse_label_expression_json("{").is_err());
        assert!(parse_label_expression_json(r#"{"and":[]}"#).is_err());
        assert!(parse_label_expression_json(r#"{"or":[]}"#).is_err());
        assert!(parse_label_expression_json(r#"{"and":["A"],"or":["B"]}"#).is_err());
        assert!(parse_label_expression_json(r#"{"and":["A"],"and":["B"]}"#).is_err());
    }

    #[test]
    fn parse_label_expression_json_rejects_expressions_that_exceed_depth_limit() {
        let error =
            parse_label_expression_json(&nested_not_json(MAX_LABEL_EXPRESSION_DEPTH)).unwrap_err();
        assert!(error
            .to_string()
            .contains("label expression depth 65 exceeds limit 64"));
    }

    #[test]
    fn parse_label_expression_json_rejects_expressions_that_exceed_node_limit() {
        let labels = std::iter::repeat_n(r#""A""#, MAX_LABEL_EXPRESSION_NODES)
            .collect::<Vec<_>>()
            .join(",");
        let expression_json = format!(r#"{{"or":[{labels}]}}"#);
        let error = parse_label_expression_json(&expression_json).unwrap_err();
        assert!(error
            .to_string()
            .contains("label expression node count 4097 exceeds limit 4096"));
    }

    #[test]
    fn parse_label_expression_json_rejects_unknown_operator_before_consuming_its_value() {
        let error = parse_label_expression_json(r#"{"xor":"#).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported label expression operator 'xor'"));
    }

    #[test]
    fn parse_label_expression_json_rejects_extra_keys_before_consuming_their_values() {
        let error = parse_label_expression_json(r#"{"and":["A"],"or":"#).unwrap_err();
        assert!(error
            .to_string()
            .contains("label expression objects must contain exactly one operator"));
    }

    #[test]
    fn filter_expression_type_aliases_preserve_public_values() {
        assert_eq!(FilterExpressionType::ORMajor as u32, 0);
        assert_eq!(FilterExpressionType::ANDMajor as u32, 1);
        assert_eq!(FilterExpressionType::DNF, FilterExpressionType::ORMajor);
        assert_eq!(FilterExpressionType::CNF, FilterExpressionType::ANDMajor);
    }

    #[test]
    fn query_supports_legacy_filter_expression_variant_names() {
        let index = round_trip(LabelIndexFormat::Bitslice);

        let or_major = compile(&index, &["A&B", "C&D"], FilterExpressionType::ORMajor);
        let dnf = compile(&index, &["A&B", "C&D"], FilterExpressionType::DNF);
        assert_eq!(
            matching_ids(&or_major, index.num_vectors),
            matching_ids(&dnf, index.num_vectors)
        );

        let and_major = compile(
            &index,
            &["A|B", "group=x|score=2"],
            FilterExpressionType::ANDMajor,
        );
        let cnf = compile(
            &index,
            &["A|B", "group=x|score=2"],
            FilterExpressionType::CNF,
        );
        assert_eq!(
            matching_ids(&and_major, index.num_vectors),
            matching_ids(&cnf, index.num_vectors)
        );
    }

    #[test]
    fn query_expression_rejects_invalid_programmatic_expressions() {
        let index = round_trip(LabelIndexFormat::Bitslice);

        assert!(index
            .query_expression(&LabelExpression::Label(String::new()))
            .is_err());
        assert!(index
            .query_expression(&LabelExpression::And(vec![]))
            .is_err());
        assert!(index
            .query_expression(&LabelExpression::Or(vec![]))
            .is_err());
        assert!(index
            .query_expression(&nested_not_expression(MAX_LABEL_EXPRESSION_DEPTH))
            .is_err());
        assert!(index
            .query_expression(&LabelExpression::Or(vec![
                LabelExpression::Label(
                    "A".to_string()
                );
                MAX_LABEL_EXPRESSION_NODES
            ]))
            .is_err());
    }

    #[test]
    fn raw_and_object_jsonl_forms_are_supported() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(
            &input,
            concat!(
                "\"solo\"\n",
                "[\"left\",\"right\"]\n",
                "{\"doc_id\":4,\"enabled\":true,\"group\":\"g\",\"count\":2,\"deleted\":false,\"labels\":[\"inline\"]}\n",
            ),
        )
        .unwrap();

        encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).unwrap();
        let index = EncodedLabelIndex::load(output).unwrap();

        let raw = compile(&index, &["solo"], FilterExpressionType::DNF);
        assert!(raw.is_match(0));
        assert!(!raw.is_match(1));

        let array = compile(&index, &["left&right"], FilterExpressionType::DNF);
        assert!(array.is_match(1));
        assert!(!array.is_match(0));

        let object = compile(
            &index,
            &["enabled&group=g&count=2&deleted=false&inline"],
            FilterExpressionType::DNF,
        );
        assert!(object.is_match(4));
        assert!(!object.is_match(3));
    }

    #[test]
    fn blank_lines_do_not_shift_implicit_document_ids() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "\n\"A\"\n\n\"B\"\n").unwrap();
        encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).unwrap();
        let index = EncodedLabelIndex::load(output).unwrap();

        let a = compile(&index, &["A"], FilterExpressionType::DNF);
        let b = compile(&index, &["B"], FilterExpressionType::DNF);
        assert!(a.is_match(0));
        assert!(!a.is_match(1));
        assert!(!b.is_match(0));
        assert!(b.is_match(1));
    }

    #[test]
    fn reserved_delimiters_are_rejected_in_label_files() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "{\"doc_id\":0,\"labels\":[\"A&B\"]}\n").unwrap();
        assert!(encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).is_err());
    }

    #[test]
    fn embedded_nul_labels_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "{\"doc_id\":0,\"labels\":[\"A\\u0000B\"]}\n").unwrap();
        assert!(encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).is_err());
    }

    #[test]
    fn labels_with_outer_whitespace_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "{\"doc_id\":0,\"labels\":[\" A\"]}\n").unwrap();
        assert!(encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).is_err());
    }

    #[test]
    fn duplicate_document_ids_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(
            &input,
            "{\"doc_id\":0,\"A\":true}\n{\"doc_id\":0,\"B\":true}\n",
        )
        .unwrap();
        assert!(encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).is_err());
    }

    #[test]
    fn load_rejects_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        std::fs::write(&path, b"not-an-index").unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_invalid_hybrid_label_kind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Hybrid as u32).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u32(&mut writer, 1).unwrap();
        writer.write_all(b"A").unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 0).unwrap();
        write_u32(&mut writer, 2).unwrap();
        write_u32(&mut writer, 0).unwrap();
        write_u32(&mut writer, 1).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_out_of_range_posting_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut posting = RoaringBitmap::new();
        posting.insert(1);
        write_label_index(
            &path,
            LabelIndexFormat::Bitmap,
            1,
            &["A".to_string()],
            &[posting],
        )
        .unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_excessive_label_count_before_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Bitslice as u32).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, (MAX_LABEL_COUNT as u64) + 1).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_set_bits_in_bitslice_padding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Bitslice as u32).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u32(&mut writer, 1).unwrap();
        writer.write_all(b"A").unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 1u64 << 1).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_oversized_bitmap_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let oversized_vectors = u32::try_from((MAX_DENSE_BITMAP_BYTES / 8) * 64 + 1).unwrap();
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Bitmap as u32).unwrap();
        write_u64(&mut writer, u64::from(oversized_vectors)).unwrap();
        write_u64(&mut writer, 0).unwrap();
        writer.flush().unwrap();

        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_trailing_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, sample_jsonl()).unwrap();
        encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitmap).unwrap();

        let mut file = OpenOptions::new().append(true).open(&output).unwrap();
        file.write_all(&[0]).unwrap();
        drop(file);

        assert!(EncodedLabelIndex::load(output).is_err());
    }

    #[test]
    fn load_rejects_zero_vector_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Bitslice as u32).unwrap();
        write_u64(&mut writer, 0).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }
}
