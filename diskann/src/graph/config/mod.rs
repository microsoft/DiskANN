/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod defaults;
pub mod experimental;

use std::num::{NonZeroU32, NonZeroUsize};

use thiserror::Error;

use crate::utils::IntoUsize;

////////////////
// Prune Kind //
////////////////

/// The occlusion factor between a vector `i` and an candidate neighbor `k` tracks how close
/// the vectors `i` and `k` are relative to all other vectors and are directly comparable
/// to the build parameter `alpha`.
///
/// A higher occlusion factor means that `j` and `k` are "more similar" than `i` and `k`.
/// The pruning rules are heuristics established such that using higher values of `alpha`
/// yields sparser graphs.
///
/// ```text
/// i (candidate vector) ----------> k
///                                 /
///           j -------------------*
/// ```
///
/// Some details on the implementation are given below.
///
/// ## TriangleInequality (Euclidean and Cosine)
///
/// The occlusion factor is the highest ratio observed between `distance_ik` and
/// `distance_jk` for all candidates `j` that have been successfully added to `i`'s
/// adjacency list.
///
/// If vectors `j` and `k` are close, then the ratio `distance_ik / distance_jk` will be
/// high, potentially excluding `k` from consideration as a neighbor of `i`.
///
/// ## Occluding (Inner Product)
///
/// Remember that the similarity score for inner product is a negative value (for
/// sufficiently similar vectors).
///
/// The rule here states that if the similarity between `j` and `k` is `current_alpha`
/// times better than the similarity between `i` and `k`, then `k` should be removed
/// entirely as a neighbor candidate (achieved by settings its occlusion factor to
/// `f32::MAX`.
///
/// When alpha is greater, this requirement is more stringent, so higher alphas lead
/// to less dense graphs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PruneKind {
    TriangleInequality,
    Occluding,
}

const OCCLUDING_MASK: f32 = 0.01;

impl PruneKind {
    /// Construct a new [`PruneKind`] tailored for `metric`.
    ///
    /// For L2 and cosine variants, this returns [`Self::TriangleInequality`].
    /// For inner product variants, this returns [`Self:Occluding`].
    pub fn from_metric(metric: diskann_vector::distance::Metric) -> Self {
        use diskann_vector::distance::Metric;
        match metric {
            Metric::L2 | Metric::Cosine | Metric::CosineNormalized => Self::TriangleInequality,
            Metric::InnerProduct => Self::Occluding,
        }
    }

    /// Run the associate pruning rule.
    ///
    /// See the struct level documentation for [`PruneKind`] for details.
    pub fn update_occlude_factor(
        self,
        distance_ik: f32,
        distance_jk: f32,
        current_factor: f32,
        current_alpha: f32,
    ) -> f32 {
        match self {
            Self::TriangleInequality => {
                if distance_jk == 0.0 {
                    f32::MAX
                } else {
                    current_factor.max(distance_ik / distance_jk)
                }
            }
            Self::Occluding => {
                if distance_jk < current_alpha * distance_ik {
                    current_alpha + OCCLUDING_MASK
                } else {
                    current_factor
                }
            }
        }
    }
}

impl From<diskann_vector::distance::Metric> for PruneKind {
    fn from(metric: diskann_vector::distance::Metric) -> PruneKind {
        Self::from_metric(metric)
    }
}

/// Controls the number of edges considered among an insert batch for the multi-insert API.
///
/// When inserted data is temporally correlated (i.e., elements within a batch are likely
/// to be neighbors of each other), setting this to a high value can help prevent recall
/// degradation due to batching.
///
/// Note that high-levels of intra-batch candidates coupled with high batch sizes can
/// increase insertion time.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IntraBatchCandidates {
    /// No intra-batch candidates will be considered. This is useful when doing a bulk
    /// ingestion of non-temporally correlated data as neglecting edges within an insert
    /// batch is unlikely to affect recall.
    None,

    /// Consider up `max` candidates within a batch. Candidate generation for a batch item `i`
    /// will be taken from the position-wise neighboring items centered around `i`.
    ///
    /// In the example below, setting `max = 4` will result in the entries 2, 3, 5, and 6
    /// being considered as edge candidates for item `4`.
    ///
    /// ```text
    /// Intra Batch Candidates = 4
    ///
    ///              Intra-Batch Candidates of 4
    ///                   |---|-------|---|
    /// Batch Items:  1   2   3   4   5   6   7   8   9
    ///                           |
    ///                  Item Being Processed
    /// ```
    Max(NonZeroU32),

    /// All elements within a batch will be considered for canidates. This will generate the
    /// highest recall, but come at a performance penalty for large batches.
    #[default]
    All,
}

impl IntraBatchCandidates {
    /// If `value` is non-zero, return `Self::NonZero(value)`. Otherwise, return `Self::None`.
    pub const fn new(value: u32) -> Self {
        match NonZeroU32::new(value) {
            None => Self::None,
            Some(max) => Self::Max(max),
        }
    }

    /// Return the number of candidates to consider out of a batch of size `batch_size`.
    pub fn get(&self, batch_size: usize) -> usize {
        match self {
            Self::None => 0,
            Self::Max(max) => max.get().into_usize().min(batch_size),
            Self::All => batch_size,
        }
    }

    /// Return `true` if `self == IntraBatchParallelism::None`.
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

////////////
// Config //
////////////

/// Configuration state for index construction operations.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The degree that adjacency lists are pruned to.
    pruned_degree: NonZeroU32,

    /// The maximum degree in the graph. When adjacency lists exceed this degree, a prune
    /// is triggered.
    max_degree: NonZeroU32,

    /// The default search window size to use for build.
    l_build: NonZeroU32,

    /// The alpha value used for pruning.
    alpha: f32,

    /// The pruning occlusion approach to take.
    prune_kind: PruneKind,

    /// The upper-bound of occlusion list sizes.
    max_occlusion_size: NonZeroU32,

    /// Maximum number of backedges applied.
    max_backedges: NonZeroU32,

    /// Max minibatch insert parallelism. Set to Some(x) to use minibatch_insert.
    /// When minibatch_insert receives more than this number of vectors,
    /// it will split the insert into rounds each with at most this many parallel tasks.
    max_minibatch_par: NonZeroU32,

    /// The number of intra-batch candidates to consider during multi-insert.
    intra_batch_candidates: IntraBatchCandidates,

    /// Whether to attempt graph saturation after all prunes.
    saturate_after_prune: bool,

    /// Experiemntal fields.
    experimental_insert_retry: Option<experimental::InsertRetry>,
}

/// Allow conversion from `NonZeroU32` to `NonZeroUsize`.
///
/// LLVM can recognice when this conversion is infallible and will emit code that never
/// panics.
macro_rules! to_nonzero_usize {
    ($($x:tt)*) => {{
        let y: NonZeroU32 = $($x)*;

        const {
            assert!(std::mem::size_of::<NonZeroUsize>() >= std::mem::size_of::<NonZeroU32>())
        };

        // Lint: Infallible on 64-bit systems.
        #[allow(clippy::unwrap_used)]
        <NonZeroUsize as TryFrom<NonZeroU32>>::try_from(y).unwrap()
    }}
}
pub(super) use to_nonzero_usize;

impl Config {
    /// Attempt to construct a [`Config`] from a builder.
    ///
    /// See: [`Builder::build`].
    pub fn try_from_builder(builder: Builder) -> Result<Self, ConfigError> {
        let non_zero_error =
            |param: &'static str, val: usize| -> Result<NonZeroU32, ConfigErrorInner> {
                try_nonzero_u32(val).map_err(|err| ConfigErrorInner::Parameter(param, err))
            };

        // TODO: Error checking for alpha.
        let alpha = builder.alpha.unwrap_or(defaults::ALPHA);

        let pruned_degree = non_zero_error("pruned_degree", builder.pruned_degree)?;

        let max_degree = match builder.max_degree {
            MaxDegree::Value(max) => non_zero_error("max_degree", max)?,
            MaxDegree::Slack(slack) => {
                if slack < 1.0 || !slack.is_finite() {
                    return Err(ConfigErrorInner::Slack(InvalidSlack(slack)).into());
                }
                non_zero_error(
                    "max_degree (from slack)",
                    (slack * pruned_degree.get() as f32) as usize,
                )?
            }
            MaxDegree::Same => pruned_degree,
        };

        if max_degree < pruned_degree {
            return Err(ConfigErrorInner::Degrees(max_degree.get(), pruned_degree.get()).into());
        }

        let l_build = non_zero_error("l_build", builder.l_build)?;

        let max_occlusion_size = match builder.max_occlusion_size {
            Some(max) => non_zero_error("max_occlusion_size", max)?,
            None => defaults::MAX_OCCLUSION_SIZE,
        };

        let max_backedges = match builder.backedge_spec {
            Some(spec) => match spec {
                BackedgeSpec::Ratio(ratio) => {
                    if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 {
                        return Err(ConfigErrorInner::BackedgeRatio(ratio).into());
                    }
                    non_zero_error(
                        "backedge_ratio (from slack)",
                        (ratio * pruned_degree.get() as f32).ceil() as usize,
                    )?
                }
                BackedgeSpec::Amount(amount) => non_zero_error("max_backedges", amount)?,
            },
            None => pruned_degree,
        };

        if max_backedges > pruned_degree {
            return Err(
                ConfigErrorInner::Backedges(max_backedges.get(), pruned_degree.get()).into(),
            );
        }

        let max_minibatch_par = match builder.max_minibatch_par {
            Some(par) => non_zero_error("max_minibatch_par", par)?,
            None => defaults::MAX_MINIBATCH_PARALLELISM,
        };

        let intra_batch_candidates = builder
            .intra_batch_candidates
            .unwrap_or(defaults::INTRA_BATCH_CANDIDATES);

        let saturate_after_prune = builder.saturate_after_prune.unwrap_or(false);

        let config = Self {
            pruned_degree,
            max_degree,
            l_build,
            alpha,
            prune_kind: builder.prune_kind,
            max_occlusion_size,
            max_backedges,
            max_minibatch_par,
            intra_batch_candidates,
            saturate_after_prune,
            experimental_insert_retry: builder.insert_retry,
        };
        Ok(config)
    }

    //-----------//
    // Accessors //
    //-----------//

    pub fn pruned_degree(&self) -> NonZeroUsize {
        const { assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u32>()) };
        // Lint: Infallible on 64-bit systems.
        #[expect(clippy::unwrap_used)]
        self.pruned_degree_u32().try_into().unwrap()
    }

    pub fn max_degree(&self) -> NonZeroUsize {
        to_nonzero_usize!(self.max_degree_u32())
    }

    pub fn l_build(&self) -> NonZeroUsize {
        to_nonzero_usize!(self.l_build_u32())
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn prune_kind(&self) -> PruneKind {
        self.prune_kind
    }

    pub fn max_occlusion_size(&self) -> NonZeroUsize {
        to_nonzero_usize!(self.max_occlusion_size_u32())
    }

    pub fn max_backedges(&self) -> NonZeroUsize {
        to_nonzero_usize!(self.max_backedges_u32())
    }

    pub fn max_minibatch_par(&self) -> NonZeroUsize {
        to_nonzero_usize!(self.max_minibatch_par_u32())
    }

    pub fn intra_batch_candidates(&self) -> IntraBatchCandidates {
        self.intra_batch_candidates
    }

    pub fn saturate_after_prune(&self) -> bool {
        self.saturate_after_prune
    }

    pub fn experimental_insert_retry(&self) -> Option<&experimental::InsertRetry> {
        self.experimental_insert_retry.as_ref()
    }

    //---------------//
    // u32 accessors //
    //---------------//

    pub fn pruned_degree_u32(&self) -> NonZeroU32 {
        self.pruned_degree
    }

    pub fn max_degree_u32(&self) -> NonZeroU32 {
        self.max_degree
    }

    pub fn l_build_u32(&self) -> NonZeroU32 {
        self.l_build
    }

    pub fn max_occlusion_size_u32(&self) -> NonZeroU32 {
        self.max_occlusion_size
    }

    pub fn max_backedges_u32(&self) -> NonZeroU32 {
        self.max_backedges
    }

    pub fn max_minibatch_par_u32(&self) -> NonZeroU32 {
        self.max_minibatch_par
    }
}

/// Errors that can occur when building a [`Config`].
///
/// See [`Builder::build`] for possible failure modes.
#[derive(Debug, Clone, Error)]
#[error(transparent)]
pub struct ConfigError {
    #[from]
    inner: ConfigErrorInner,
}

impl From<ConfigError> for crate::ANNError {
    fn from(error: ConfigError) -> Self {
        crate::ANNError::new(crate::ANNErrorKind::IndexConfigError, error)
    }
}

#[derive(Debug, Clone, Error)]
enum ConfigErrorInner {
    #[error("parameter \"{0}\" invalid because {1}")]
    Parameter(&'static str, NotNonZeroU32),
    #[error("parameter \"max_degree\" invalid because {0}")]
    Slack(InvalidSlack),
    #[error("parameter \"max_degree\" ({0}) must not be less than \"pruned_degree\" ({1})")]
    Degrees(u32, u32),
    #[error("parameter \"max_backedges\" ({0}) must not be greater than \"pruned_degree\" ({1})")]
    Backedges(u32, u32),
    #[error("parameter \"backedge_ratio\" ({0}) as ratio invalid because must be in (0.0, 1.0]")]
    BackedgeRatio(f32),
}

fn try_nonzero_u32(x: usize) -> Result<NonZeroU32, NotNonZeroU32> {
    let y: u32 = x.try_into().map_err(|_| NotNonZeroU32(x))?;
    NonZeroU32::new(y).ok_or(NotNonZeroU32(x))
}

#[derive(Debug, Clone)]
struct NotNonZeroU32(usize);

impl std::fmt::Display for NotNonZeroU32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            write!(f, "it cannot be zero")
        } else if self.0 > (u32::MAX as usize) {
            write!(f, "its value ({}) exceeds u32::MAX", self.0)
        } else {
            // Shouldn't reach here.
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
struct InvalidSlack(f32);

impl std::fmt::Display for InvalidSlack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.0.is_finite() {
            write!(f, "it must be finite, not {}", self.0)
        } else if self.0 < 1.0 {
            write!(f, "it must be greater than 1.0 (instead, it is {})", self.0)
        } else {
            // Shouldn't reach here.
            Ok(())
        }
    }
}

/// Configuration setting for the maximum graph degree.
///
/// Graph degree is conrolled by two values, the "pruned degree" (what pruning aims to
/// achieve), and the "maximum degree" (the largest an adjacency list can get before pruning
/// is triggered).
///
/// Having this wiggle room greatly reduces the number of prunes required during backedge
/// insertion.
#[derive(Debug, Clone, Copy)]
pub enum MaxDegree {
    /// Specify the maximum degree as an absolute vlue.
    Value(usize),

    /// Specify the maximum degree as a value relative to the pruned degree. This requires
    /// that the contained scaling parameter be larger than or equal to 1.0.
    Slack(f32),

    /// Specify the maximum degree to be the same as the pruned degree. This is not
    /// recommended for insert workloads as it can lead to excessive pruning, but is
    /// suitable for search-only workloads.
    Same,
}

impl MaxDegree {
    /// Construct a new [`MaxDegree`] for the specified exact value. Note that this must
    /// be greater than or equal to the associated pruned degree.
    pub const fn new(value: usize) -> Self {
        Self::Value(value)
    }

    /// Construct a new [`MaxDegree`] as a value relative to the pruned degree. The true
    /// max degree will be calculated as
    /// ```math
    /// floor(pruned_degree * slack)
    /// ```
    /// Note that `slack` must be finite, non-zero, band must not cause the computed max
    /// degree to exceed `u32::MAX`.
    pub const fn slack(slack: f32) -> Self {
        Self::Slack(slack)
    }

    /// Construct a new [`MaxDegree`] using a heuristically reasonable default.
    pub const fn default_slack() -> Self {
        Self::slack(defaults::GRAPH_SLACK_FACTOR)
    }

    /// Construct a new [`MaxDegree`] that will be the same as the pruned degree.
    pub const fn same() -> Self {
        Self::Same
    }
}

// Allow the number of back edges to be specified as either a total amount, or an amount
// relative to the maximum graph degree.
enum BackedgeSpec {
    Ratio(f32),
    Amount(usize),
}

/// A builder for for [`Config`]. Necessary invariants among the fields will be checked
/// upon invoking [`Config::build`].
///
/// See [`Config::build`] for details.
pub struct Builder {
    pruned_degree: usize,
    max_degree: MaxDegree,
    l_build: usize,
    prune_kind: PruneKind,

    // optional //
    alpha: Option<f32>,
    max_occlusion_size: Option<usize>,
    backedge_spec: Option<BackedgeSpec>,
    max_minibatch_par: Option<usize>,
    intra_batch_candidates: Option<IntraBatchCandidates>,
    saturate_after_prune: Option<bool>,

    /// Experimental additions.
    insert_retry: Option<experimental::InsertRetry>,
}

impl Builder {
    /// Construct a new builder with the basic values.
    ///
    /// All other parameters will use their default values.
    pub fn new(
        pruned_degree: usize,
        max_degree: MaxDegree,
        l_build: usize,
        prune_kind: PruneKind,
    ) -> Self {
        Self {
            pruned_degree,
            max_degree,
            l_build,
            prune_kind,
            alpha: None,
            max_occlusion_size: None,
            backedge_spec: None,
            max_minibatch_par: None,
            intra_batch_candidates: None,
            saturate_after_prune: None,
            insert_retry: None,
        }
    }

    /// Construct a new builder with the basic values.
    ///
    /// All other parameters will use their default values.
    ///
    /// A closure `f` can be used to chain additional builder methods inline.
    pub fn new_with<F>(
        pruned_degree: usize,
        max_degree: MaxDegree,
        l_build: usize,
        prune_kind: PruneKind,
        f: F,
    ) -> Self
    where
        F: FnOnce(&mut Self),
    {
        let mut this = Self::new(pruned_degree, max_degree, l_build, prune_kind);
        f(&mut this);
        this
    }

    /// Configure the `alpha` parameter used during pruning.
    ///
    /// Values closer to 1.0 will yield denser graphs at the cost of increased build time.
    pub fn alpha(&mut self, alpha: f32) -> &mut Self {
        self.alpha = Some(alpha);
        self
    }

    /// Configure the search window size used during the search phase of index construction.
    ///
    /// Parameter `size` must be non-zero and not exceed `u32::MAX`.
    pub fn l_build(&mut self, size: usize) -> &mut Self {
        self.l_build = size;
        self
    }

    /// Configure the maximum number of candidates provided to prune.
    ///
    /// Parameter `size` must be non-zero and not exceed `u32::MAX`.
    pub fn max_occlusion_size(&mut self, size: usize) -> &mut Self {
        self.max_occlusion_size = Some(size);
        self
    }

    /// Configure the number of backedges as a ratio of the pruned degree.
    ///
    /// Parameter `ratio` must be in the interval `(0.0, 1.0]`.
    ///
    /// Setting this parameter will invalidate any previous value from [`Self::max_backedges`].
    pub fn backedge_ratio(&mut self, ratio: f32) -> &mut Self {
        self.backedge_spec = Some(BackedgeSpec::Ratio(ratio));
        self
    }

    /// Configure the number of backedges as an absolute value.
    ///
    /// Parameter `max` must be non-zero, not exceed `u32::MAX`, and must not be greater than
    /// `pruned_degree`.
    ///
    /// Setting this parameter will invalidate any previous value from [`Self::backedge_ratio`].
    pub fn max_backedges(&mut self, max: usize) -> &mut Self {
        self.backedge_spec = Some(BackedgeSpec::Amount(max));
        self
    }

    /// Configure that maximum parallelism used by the multi-insert APIs.
    ///
    /// Parameter `par` must be non-zero and not exceed `u32::MAX`.
    pub fn max_minibatch_par(&mut self, par: usize) -> &mut Self {
        self.max_minibatch_par = Some(par);
        self
    }

    /// Configure the intra-batch candidates.
    pub fn intra_batch_candidates(&mut self, candidates: IntraBatchCandidates) -> &mut Self {
        self.intra_batch_candidates = Some(candidates);
        self
    }

    /// Configure whether the adjacency lists are saturated after every pruning step.
    pub fn saturate_after_prune(&mut self, to_saturate: bool) -> &mut Self {
        self.saturate_after_prune = Some(to_saturate);
        self
    }

    /// Enable the experimental insert retry algorithm.
    pub fn insert_retry(&mut self, insert_retry: experimental::InsertRetry) -> &mut Self {
        self.insert_retry = Some(insert_retry);
        self
    }

    /// Attempt to build the config. Fails if:
    ///
    /// * The resolved `max_degree` is less than `pruned_degree`.
    /// * The resolved number of backedges is greater than `pruned_degree`.
    /// * Any parameter is either 0 or exceeds `u32::MAX`.
    pub fn build(self) -> Result<Config, ConfigError> {
        Config::try_from_builder(self)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    const SLACK: MaxDegree = MaxDegree::default_slack();
    const TOO_BIG: usize = 5_000_000_000;

    /// Utility to help check error messages.
    macro_rules! check_msg {
        ($msg:ident, $expected:literal $(,)?) => {
            assert_eq!($msg, $expected, "failed with: {}", $msg,);
        };
    }

    #[test]
    fn test_intra_batch_candidates() {
        assert_eq!(
            IntraBatchCandidates::default(),
            defaults::INTRA_BATCH_CANDIDATES
        );

        assert_eq!(IntraBatchCandidates::new(0), IntraBatchCandidates::None);
        assert_eq!(
            IntraBatchCandidates::new(10),
            IntraBatchCandidates::Max(NonZeroU32::new(10).unwrap())
        );

        // None
        {
            let c = IntraBatchCandidates::None;
            for i in 0..10 {
                assert_eq!(c.get(i), 0);
            }
        }

        // All
        {
            let c = IntraBatchCandidates::All;
            for i in 0..10 {
                assert_eq!(c.get(i), i);
            }
        }

        // Max
        {
            let c = IntraBatchCandidates::new(5);
            for i in 0..10 {
                if i <= 5 {
                    assert_eq!(c.get(i), i);
                } else {
                    assert_eq!(c.get(i), 5);
                }
            }
        }
    }

    #[test]
    fn test_defaults() {
        let prune_kind = PruneKind::TriangleInequality;
        let config = Builder::new(100, SLACK, 50, prune_kind).build().unwrap();

        assert_eq!(config.pruned_degree().get(), 100);
        assert_eq!(
            config.max_degree().get(),
            130,
            "default slack should be 1.3"
        );
        assert_eq!(config.l_build().get(), 50);
        assert_eq!(config.alpha(), defaults::ALPHA);
        assert_eq!(config.prune_kind(), prune_kind);
        assert_eq!(
            config.max_occlusion_size_u32(),
            defaults::MAX_OCCLUSION_SIZE
        );
        assert_eq!(
            config.max_backedges().get(),
            100,
            "backedges should equal pruned degree"
        );
        assert_eq!(
            config.max_minibatch_par_u32(),
            defaults::MAX_MINIBATCH_PARALLELISM
        );
        assert_eq!(
            config.intra_batch_candidates(),
            defaults::INTRA_BATCH_CANDIDATES
        );

        assert_eq!(
            config.saturate_after_prune(),
            defaults::SATURATE_AFTER_PRUNE
        );
        assert!(config.experimental_insert_retry().is_none());
    }

    #[test]
    fn test_pruned_degree() {
        let prune_kind = PruneKind::TriangleInequality;

        for i in [10, 20, 30] {
            let config = Builder::new(i, SLACK, 50, prune_kind).build().unwrap();
            assert_eq!(config.pruned_degree().get(), i);
        }

        let msg = Builder::new(0, SLACK, 50, prune_kind)
            .build()
            .unwrap_err()
            .to_string();
        check_msg!(
            msg,
            "parameter \"pruned_degree\" invalid because it cannot be zero",
        );

        let msg = Builder::new(TOO_BIG, SLACK, 50, prune_kind)
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"pruned_degree\" invalid because its value (5000000000) exceeds u32::MAX",
        );
    }

    #[test]
    fn test_max_degree() {
        let prune_kind = PruneKind::TriangleInequality;
        for i in [10, 20, 30] {
            let config = Builder::new(10, MaxDegree::new(i), 50, prune_kind)
                .build()
                .unwrap();

            assert_eq!(config.max_degree().get(), i);
        }

        for (slack, expected) in [(1.0, 11), (1.2, 13), (1.3, 14)] {
            let config = Builder::new(11, MaxDegree::slack(slack), 50, prune_kind)
                .build()
                .unwrap();

            assert_eq!(config.max_degree().get(), expected);
        }

        // Errors
        let msg = Builder::new(10, MaxDegree::new(0), 50, prune_kind)
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_degree\" invalid because it cannot be zero",
        );

        let msg = Builder::new(10, MaxDegree::new(9), 50, prune_kind)
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_degree\" (9) must not be less than \"pruned_degree\" (10)",
        );

        let msg = Builder::new(10, MaxDegree::slack(0.5), 50, prune_kind)
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_degree\" invalid because it must be greater than 1.0 (instead, it is 0.5)",
        );

        let msg = Builder::new(10, MaxDegree::slack(f32::NAN), 50, prune_kind)
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_degree\" invalid because it must be finite, not NaN",
        );

        let msg = Builder::new(10, MaxDegree::slack(5_000_000_000.0), 50, prune_kind)
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_degree (from slack)\" invalid because its value (49999998976) exceeds u32::MAX",
        );
    }

    #[test]
    fn test_alpha() {
        fn f(v: f32) -> impl FnOnce(&mut Builder) {
            move |b| {
                b.alpha(v);
            }
        }

        let prune_kind = PruneKind::TriangleInequality;
        for i in [1.0, 1.1, 1.2] {
            let config = Builder::new_with(10, SLACK, 10, prune_kind, f(i))
                .build()
                .unwrap();

            assert_eq!(config.alpha(), i);
        }
    }

    #[test]
    fn test_max_occlusion_size() {
        fn f(v: usize) -> impl FnOnce(&mut Builder) {
            move |b| {
                b.max_occlusion_size(v);
            }
        }

        let prune_kind = PruneKind::TriangleInequality;
        for i in [10, 20, 30] {
            let config = Builder::new_with(10, SLACK, 10, prune_kind, f(i))
                .build()
                .unwrap();

            assert_eq!(config.max_occlusion_size().get(), i);
        }

        let msg = Builder::new_with(10, SLACK, 10, prune_kind, f(0))
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_occlusion_size\" invalid because it cannot be zero",
        );

        let msg = Builder::new_with(10, SLACK, 10, prune_kind, f(TOO_BIG))
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_occlusion_size\" invalid because its value (5000000000) exceeds u32::MAX",
        );
    }

    #[test]
    fn test_l_build() {
        let prune_kind = PruneKind::TriangleInequality;
        for i in [10, 20, 30] {
            let config = Builder::new(10, SLACK, i, prune_kind).build().unwrap();

            assert_eq!(config.l_build().get(), i);
        }

        let msg = Builder::new(10, SLACK, 0, prune_kind)
            .build()
            .unwrap_err()
            .to_string();
        check_msg!(
            msg,
            "parameter \"l_build\" invalid because it cannot be zero",
        );

        let msg = Builder::new(10, SLACK, TOO_BIG, prune_kind)
            .build()
            .unwrap_err()
            .to_string();
        check_msg!(
            msg,
            "parameter \"l_build\" invalid because its value (5000000000) exceeds u32::MAX",
        );
    }

    #[test]
    fn test_backedge_ratio() {
        fn f(v: f32) -> impl FnOnce(&mut Builder) {
            move |b| {
                b.backedge_ratio(v);
            }
        }

        let prune_kind = PruneKind::TriangleInequality;

        // Check that the `ceil` operation is performed.
        for (ratio, expected) in [(0.5, 6), (0.8, 9), (1.0, 11)] {
            let config = Builder::new_with(11, SLACK, 10, prune_kind, f(ratio))
                .build()
                .unwrap();

            assert_eq!(config.max_backedges().get(), expected);
        }

        let msg = Builder::new_with(10, SLACK, 10, prune_kind, f(0.0))
            .build()
            .unwrap_err()
            .to_string();
        check_msg!(
            msg,
            "parameter \"backedge_ratio\" (0) as ratio invalid because must be in (0.0, 1.0]",
        );
    }

    #[test]
    fn test_max_backedges() {
        fn f(v: usize) -> impl FnOnce(&mut Builder) {
            move |b| {
                b.max_backedges(v);
            }
        }

        let prune_kind = PruneKind::TriangleInequality;

        // Check that the `ceil` operation is performed.
        for i in [1, 2, 11] {
            let config = Builder::new_with(11, SLACK, 10, prune_kind, f(i))
                .build()
                .unwrap();

            assert_eq!(config.max_backedges().get(), i);
        }

        let msg = Builder::new_with(10, SLACK, 10, prune_kind, f(0))
            .build()
            .unwrap_err()
            .to_string();
        check_msg!(
            msg,
            "parameter \"max_backedges\" invalid because it cannot be zero",
        );

        let msg = Builder::new_with(10, SLACK, 10, prune_kind, f(TOO_BIG))
            .build()
            .unwrap_err()
            .to_string();
        check_msg!(
            msg,
            "parameter \"max_backedges\" invalid because its value (5000000000) exceeds u32::MAX",
        );

        let msg = Builder::new_with(10, SLACK, 10, prune_kind, f(11))
            .build()
            .unwrap_err()
            .to_string();
        check_msg!(
            msg,
            "parameter \"max_backedges\" (11) must not be greater than \"pruned_degree\" (10)",
        );
    }

    #[test]
    fn test_max_minibatch_par() {
        fn f(v: usize) -> impl FnOnce(&mut Builder) {
            move |b| {
                b.max_minibatch_par(v);
            }
        }

        let prune_kind = PruneKind::TriangleInequality;
        for i in [10, 20, 30] {
            let config = Builder::new_with(10, SLACK, 10, prune_kind, f(i))
                .build()
                .unwrap();

            assert_eq!(config.max_minibatch_par().get(), i);
        }

        let msg = Builder::new_with(10, SLACK, 10, prune_kind, f(0))
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_minibatch_par\" invalid because it cannot be zero",
        );

        let msg = Builder::new_with(10, SLACK, 10, prune_kind, f(TOO_BIG))
            .build()
            .unwrap_err()
            .to_string();

        check_msg!(
            msg,
            "parameter \"max_minibatch_par\" invalid because its value (5000000000) exceeds u32::MAX",
        );
    }

    #[test]
    fn test_intra_batch_candidates_builder() {
        fn f(v: IntraBatchCandidates) -> impl FnOnce(&mut Builder) {
            move |b| {
                b.intra_batch_candidates(v);
            }
        }

        // None
        let config = Builder::new_with(
            10,
            SLACK,
            10,
            PruneKind::TriangleInequality,
            f(IntraBatchCandidates::None),
        )
        .build()
        .unwrap();

        assert_eq!(config.intra_batch_candidates(), IntraBatchCandidates::None);

        // All
        let config = Builder::new_with(
            10,
            SLACK,
            10,
            PruneKind::TriangleInequality,
            f(IntraBatchCandidates::All),
        )
        .build()
        .unwrap();

        assert_eq!(config.intra_batch_candidates(), IntraBatchCandidates::All);

        // Max
        for m in [1, 2, 10, 100] {
            let nz = NonZeroU32::new(m).unwrap();
            let config = Builder::new_with(
                10,
                SLACK,
                10,
                PruneKind::TriangleInequality,
                f(IntraBatchCandidates::Max(nz)),
            )
            .build()
            .unwrap();

            assert_eq!(
                config.intra_batch_candidates(),
                IntraBatchCandidates::Max(nz)
            );
        }
    }

    #[test]
    fn test_prune_kind() {
        let config = Builder::new(10, SLACK, 10, PruneKind::TriangleInequality)
            .build()
            .unwrap();

        assert_eq!(config.prune_kind(), PruneKind::TriangleInequality);

        let config = Builder::new(10, SLACK, 10, PruneKind::Occluding)
            .build()
            .unwrap();

        assert_eq!(config.prune_kind(), PruneKind::Occluding);
    }

    #[test]
    fn test_saturate_after_prune() {
        fn f(v: bool) -> impl FnOnce(&mut Builder) {
            move |b| {
                b.saturate_after_prune(v);
            }
        }

        let prune_kind = PruneKind::TriangleInequality;
        for i in [true, false] {
            let config = Builder::new_with(10, SLACK, 10, prune_kind, f(i))
                .build()
                .unwrap();

            assert_eq!(config.saturate_after_prune(), i);
        }
    }

    #[test]
    fn test_experimental() {
        let retry = experimental::InsertRetry::new(
            NonZeroU32::new(3).unwrap(),
            NonZeroU32::new(10).unwrap(),
            true,
        );
        let config = Builder::new_with(10, SLACK, 10, PruneKind::TriangleInequality, |b| {
            b.insert_retry(retry.clone());
        })
        .build()
        .unwrap();

        assert_eq!(config.experimental_insert_retry().unwrap(), &retry);
    }

    //------------//
    // Prune Kind //
    //------------//

    #[test]
    fn test_prune_kind_conversion() {
        let x: PruneKind = diskann_vector::distance::Metric::L2.into();
        assert_eq!(x, PruneKind::TriangleInequality);

        let x: PruneKind = diskann_vector::distance::Metric::Cosine.into();
        assert_eq!(x, PruneKind::TriangleInequality);

        let x: PruneKind = diskann_vector::distance::Metric::CosineNormalized.into();
        assert_eq!(x, PruneKind::TriangleInequality);

        let x: PruneKind = diskann_vector::distance::Metric::InnerProduct.into();
        assert_eq!(x, PruneKind::Occluding);
    }

    #[test]
    fn test_update_occlude_factor() {
        // Triangle Inequality
        let kind = PruneKind::TriangleInequality;

        // Regardless of `distance_ik`, we should yield `f32::MAX` when `distance_jk == 0`.
        for distance_ik in [f32::MIN, -1.2, 0.0, 0.123, 50.0, f32::MAX] {
            assert_eq!(
                kind.update_occlude_factor(distance_ik, 0.0, 1.0, 2.0),
                f32::MAX
            );
        }

        // Behavior of `update_occlude_factor` for these metrics should not depend
        // on `current_alpha`.
        for current_alpha in [1.0, 1.1, 1.2, 1.3] {
            // current factor less than `distance_ik / distance_jk`.
            assert_eq!(
                kind.update_occlude_factor(2.0, 1.0, 1.0, current_alpha),
                2.0
            );
            // current factor equal to `distance_ik / distance_jk`.
            assert_eq!(
                kind.update_occlude_factor(2.0, 1.0, 2.0, current_alpha),
                2.0
            );
            // current factor greater to `distance_ik / distance_jk`.
            assert_eq!(
                kind.update_occlude_factor(2.0, 1.0, 3.0, current_alpha),
                3.0
            );
        }

        // Occluding
        let kind = PruneKind::Occluding;

        // Test `distance_jk > current_alpha * distance_ik`.
        let current_factor = 0.0;
        assert_eq!(
            kind.update_occlude_factor(-2.0, -1.0, current_factor, 3.0),
            current_factor
        );
        assert_eq!(
            kind.update_occlude_factor(-3.0, -2.0, current_factor, 1.0),
            current_factor
        );

        // Test `distance_jk == current_alpha * distance_ik`.
        assert_eq!(
            kind.update_occlude_factor(-3.0, -3.0, current_factor, 1.0),
            current_factor
        );

        // Test `distance_jk < current_alpha * distance_ik`.
        assert_eq!(
            kind.update_occlude_factor(-3.0, -4.0, current_factor, 1.0),
            1.0 + OCCLUDING_MASK,
        );
    }
}
