/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::utils::fmt;

/// A simple list of features for annotating gated benchmarks/inputs.
///
/// Features supports one of the following four patterns:
///
/// A singular feature:
/// ```
/// use diskann_benchmark_runner::Features;
///
/// let features = Features::new("a");
/// assert_eq!(features.to_string(), "feature \"a\"");
/// ```
/// A conjunction:
/// ```
/// use diskann_benchmark_runner::Features;
///
/// let features = Features::all(["a", "b"]);
/// assert_eq!(features.to_string(), "features \"a\" and \"b\"");
/// ```
/// A disjunction:
/// ```
/// use diskann_benchmark_runner::Features;
///
/// let features = Features::any(["a", "b"]);
/// assert_eq!(features.to_string(), "features \"a\" or \"b\"");
/// ```
/// A custom layout"
/// ```
/// use diskann_benchmark_runner::Features;
///
/// let features = Features::custom("(\"a\" and \"b\") or \"c\"", true);
/// assert_eq!(features.to_string(), "features (\"a\" and \"b\") or \"c\"");
/// ```
///
/// See: [`crate::Registry::register_gated`] and [`crate::Registry::register_partially_gated`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Features(FeaturesInner);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FeaturesInner {
    Only(&'static str),
    Any(Vec<&'static str>),
    All(Vec<&'static str>),
    Custom { custom: &'static str, plural: bool },
}

impl Features {
    /// Construct a singular [`Features`].
    pub fn new(feature: &'static str) -> Self {
        Self(FeaturesInner::Only(feature))
    }

    /// Construct a [`Features`] as a disjunction of features.
    pub fn any<I>(itr: I) -> Self
    where
        I: IntoIterator<Item = &'static str>,
    {
        Self(FeaturesInner::Any(itr.into_iter().collect()))
    }

    /// Construct a [`Features`] as a conjunction of features.
    pub fn all<I>(itr: I) -> Self
    where
        I: IntoIterator<Item = &'static str>,
    {
        Self(FeaturesInner::All(itr.into_iter().collect()))
    }

    /// Construct a [`Features`] with a custom formatted string where `plural` can be used
    /// to control whether the returned struct uses "feature" or "features" in its formatting.
    pub fn custom(custom: &'static str, plural: bool) -> Self {
        Self(FeaturesInner::Custom { custom, plural })
    }

    /// Return `true` if this feature requirement is satisfied by the set of `enabled` feature
    /// names.
    ///
    /// [`FeaturesInner::Custom`] requirements are opaque to the registry and are therefore
    /// never considered satisfied.
    #[cfg(any(test, feature = "test-app"))]
    pub(crate) fn satisfied_by(&self, enabled: &std::collections::HashSet<String>) -> bool {
        match &self.0 {
            FeaturesInner::Only(feature) => enabled.contains(*feature),
            FeaturesInner::Any(any) => any.iter().any(|f| enabled.contains(*f)),
            FeaturesInner::All(all) => all.iter().all(|f| enabled.contains(*f)),
            FeaturesInner::Custom { .. } => false,
        }
    }
}

impl From<&'static str> for Features {
    fn from(feature: &'static str) -> Features {
        Features::new(feature)
    }
}

impl std::fmt::Display for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn feature(plural: bool) -> &'static str {
            if plural {
                "features"
            } else {
                "feature"
            }
        }

        match &self.0 {
            FeaturesInner::Only(feature) => write!(f, "feature \"{}\"", feature),
            FeaturesInner::Any(any) => {
                if any.is_empty() {
                    f.write_str("feature <missing>")
                } else {
                    write!(
                        f,
                        "{} {}",
                        feature(any.len() != 1),
                        fmt::Delimit::new(any.iter().map(fmt::Quote), ", ")
                            .with_last(", or ")
                            .with_pair(" or ")
                    )
                }
            }
            FeaturesInner::All(all) => {
                if all.is_empty() {
                    f.write_str("feature <missing>")
                } else {
                    write!(
                        f,
                        "{} {}",
                        feature(all.len() != 1),
                        fmt::Delimit::new(all.iter().map(fmt::Quote), ", ")
                            .with_last(", and ")
                            .with_pair(" and ")
                    )
                }
            }
            FeaturesInner::Custom { custom, plural } => {
                write!(f, "{} \"{}\"", feature(*plural), custom)
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rendering() {
        let list = [
            (Features::new("a"), "feature \"a\""),
            (Features::from("b"), "feature \"b\""),
            // Any
            (Features::any([]), "feature <missing>"),
            (Features::any(["a"]), "feature \"a\""),
            (Features::any(["a", "b"]), "features \"a\" or \"b\""),
            (
                Features::any(["a", "b", "c"]),
                "features \"a\", \"b\", or \"c\"",
            ),
            // All
            (Features::all([]), "feature <missing>"),
            (Features::all(["a"]), "feature \"a\""),
            (Features::all(["a", "b"]), "features \"a\" and \"b\""),
            (
                Features::all(["a", "b", "c"]),
                "features \"a\", \"b\", and \"c\"",
            ),
            // Custom
            (
                Features::custom("custom/stuff", false),
                "feature \"custom/stuff\"",
            ),
            (
                Features::custom("custom/stuff", true),
                "features \"custom/stuff\"",
            ),
        ];

        for (feature, expected) in list.iter() {
            assert_eq!(feature.to_string(), *expected, "Failed for {:?}", feature);
        }
    }
}
