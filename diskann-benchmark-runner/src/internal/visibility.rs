/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::Features;

// Visibility of an item.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Visibility<'a> {
    Available,
    Gated { features: &'a Features },
}

impl Visibility<'_> {
    #[must_use = "this has no side-effects"]
    pub(crate) fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Filter {
    All,
    OnlyAvailable,
}

impl Filter {
    #[must_use = "this has no side-effects"]
    pub(crate) fn matches(self, vis: &Visibility<'_>) -> bool {
        match self {
            Self::All => true,
            Self::OnlyAvailable => matches!(vis, Visibility::Available),
        }
    }
}
