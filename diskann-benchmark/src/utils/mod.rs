/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};

pub(crate) mod datafiles;
pub(crate) mod filters;
pub(crate) mod recall;
pub(crate) mod streaming;
pub(crate) mod tokio;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SimilarityMeasure {
    SquaredL2,
    InnerProduct,
    Cosine,
    CosineNormalized,
}

impl From<SimilarityMeasure> for diskann_vector::distance::Metric {
    fn from(value: SimilarityMeasure) -> Self {
        match value {
            SimilarityMeasure::SquaredL2 => Self::L2,
            SimilarityMeasure::InnerProduct => Self::InnerProduct,
            SimilarityMeasure::Cosine => Self::Cosine,
            SimilarityMeasure::CosineNormalized => Self::CosineNormalized,
        }
    }
}

impl std::fmt::Display for SimilarityMeasure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::SquaredL2 => "squared_l2",
            Self::InnerProduct => "inner_product",
            Self::Cosine => "cosine",
            Self::CosineNormalized => "cosine_normalized",
        };
        write!(f, "{}", st)
    }
}

/// A new-type wrapper to allow implementation of `std::fmt::Display` for non-local types.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub(crate) struct DisplayWrapper<'a, T>(pub(crate) &'a T)
where
    T: ?Sized;

impl<T> std::ops::Deref for DisplayWrapper<'_, T>
where
    T: ?Sized,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MaybeDisplay<T>(pub(crate) T, pub(crate) &'static str);

impl<T> std::fmt::Display for MaybeDisplay<Option<T>>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(v) => v.fmt(f),
            None => write!(f, "{}", self.1),
        }
    }
}

impl<T, E> std::fmt::Display for MaybeDisplay<Result<T, E>>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Ok(v) => v.fmt(f),
            Err(_) => write!(f, "{}", self.1),
        }
    }
}

/// Backend implementations are gated behind additive features to reduce compilation time
/// when only a subset of benchmarks are needed.
///
/// However, when benchmarks are feature gated, we want to provide a useful diagnostic when
/// users try to run a benchmark targeting the blocked out method.
///
/// To do this, we use stub implementations that use "dispatch matching" to match the same
/// `CentralDispatch` enum as the base benchmark, but return an error that describes
/// feature required to enable the backend benchmark.
macro_rules! stub_impl {
    ($feature:literal, $input:path $(,)?) => {
        #[cfg(not(feature = $feature))]
        mod imp {
            use diskann_benchmark_runner::{
                describeln,
                dispatcher::{FailureScore, MatchScore},
                output::Output,
                registry::Benchmarks,
                Benchmark, Checkpoint, Input,
            };

            use crate::inputs;

            pub(super) fn register(name: &str, registry: &mut Benchmarks) {
                registry.register(name, Stub);
            }

            /// An empty placeholder to provide a hint for the necessary feature.
            pub(super) struct Stub;

            impl Benchmark for Stub {
                type Input = $input;
                type Output = serde_json::Value;

                fn try_match(&self, _input: &$input) -> Result<MatchScore, FailureScore> {
                    Err(FailureScore(0))
                }

                fn description(
                    &self,
                    f: &mut std::fmt::Formatter<'_>,
                    _input: Option<&$input>,
                ) -> std::fmt::Result {
                    writeln!(f, "tag: \"{}\"", <$input as Input>::tag())?;
                    describeln!(f, "{}", concat!("Requires the \"", $feature, "\" feature"))
                }

                fn run(
                    &self,
                    _input: &$input,
                    _checkpoint: Checkpoint<'_>,
                    _output: &mut dyn Output,
                ) -> anyhow::Result<serde_json::Value> {
                    panic!("this function should not be called!");
                }
            }
        }
    };
}

pub(crate) use stub_impl;

/////////////////
// SmallBanner //
/////////////////

pub(crate) struct SmallBanner<'a>(pub(crate) &'a str);

impl std::fmt::Display for SmallBanner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = format!("| {} |", self.0);
        let len = st.len() - 2;
        writeln!(f, "+{:->len$}+", "")?;
        writeln!(f, "{}", st)?;
        writeln!(f, "+{:->len$}+", "")?;
        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_measure_conversion() {
        let x: diskann_vector::distance::Metric = SimilarityMeasure::SquaredL2.into();
        assert_eq!(x, diskann_vector::distance::Metric::L2);

        let x: diskann_vector::distance::Metric = SimilarityMeasure::InnerProduct.into();
        assert_eq!(x, diskann_vector::distance::Metric::InnerProduct);

        let x: diskann_vector::distance::Metric = SimilarityMeasure::Cosine.into();
        assert_eq!(x, diskann_vector::distance::Metric::Cosine);

        let x: diskann_vector::distance::Metric = SimilarityMeasure::CosineNormalized.into();
        assert_eq!(x, diskann_vector::distance::Metric::CosineNormalized);
    }

    // Display Wrapper Deref.
    #[test]
    fn display_wrapper_deref() {
        let s = "test string";
        let x = DisplayWrapper(s);
        let deref: &str = &x;
        assert_eq!(deref, s);
    }

    #[test]
    fn maybe_display() {
        for msg in ["a", "foo", "bar"] {
            let s = format!("{}", MaybeDisplay(Some("test"), msg));
            assert_eq!(s, "test");

            let s = format!("{}", MaybeDisplay(None::<&str>, msg));
            assert_eq!(s, msg);

            let s = format!("{}", MaybeDisplay(Result::<&str, &str>::Ok("test"), msg));
            assert_eq!(s, "test");

            let s = format!("{}", MaybeDisplay(Result::<&str, &str>::Err("test"), msg));
            assert_eq!(s, msg);
        }
    }
}
