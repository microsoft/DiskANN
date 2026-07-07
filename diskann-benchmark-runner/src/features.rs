/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::utils::fmt;

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
    pub fn new(feature: &'static str) -> Self {
        Self(FeaturesInner::Only(feature))
    }

    pub fn any<I>(itr: I) -> Self
    where
        I: IntoIterator<Item = &'static str>,
    {
        Self(FeaturesInner::Any(itr.into_iter().collect()))
    }

    pub fn all<I>(itr: I) -> Self
    where
        I: IntoIterator<Item = &'static str>,
    {
        Self(FeaturesInner::All(itr.into_iter().collect()))
    }

    pub fn custom(custom: &'static str, plural: bool) -> Self {
        Self(FeaturesInner::Custom { custom, plural })
    }
}

impl From<&'static str> for Features {
    fn from(feature: &'static str) -> Features {
        Features::new(feature)
    }
}

impl std::fmt::Display for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            FeaturesInner::Only(feature) => write!(f, "feature \"{}\"", feature),
            FeaturesInner::Any(any) => {
                let prefix = if any.len() == 1 {
                    "feature"
                } else {
                    "features"
                };
                write!(
                    f,
                    "{} {}",
                    prefix,
                    fmt::Delimit::new(any.iter().map(fmt::Quote), ", ")
                        .with_last(", or ")
                        .with_pair(" or ")
                )
            }
            FeaturesInner::All(all) => {
                let prefix = if all.len() == 1 {
                    "feature"
                } else {
                    "features"
                };
                write!(
                    f,
                    "{} {}",
                    prefix,
                    fmt::Delimit::new(all.iter().map(fmt::Quote), ", ")
                        .with_last(", and ")
                        .with_pair(" and ")
                )
            }
            FeaturesInner::Custom { custom, plural } => {
                let prefix = if *plural { "features" } else { "features" };
                write!(f, "{} \"{}\"", prefix, custom)
            }
        }
    }
}
