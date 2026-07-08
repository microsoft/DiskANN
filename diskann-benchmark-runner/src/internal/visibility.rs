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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        let features = Features::new("some-feature");
        assert!(Visibility::Available.is_available());
        assert!(!Visibility::Gated {
            features: &features
        }
        .is_available());
    }

    #[test]
    fn test_filter_matches() {
        let features = Features::new("some-feature");
        let available = Visibility::Available;
        let gated = Visibility::Gated {
            features: &features,
        };

        // `All` matches everything.
        assert!(Filter::All.matches(&available));
        assert!(Filter::All.matches(&gated));

        // `OnlyAvailable` matches available items exclusively.
        assert!(Filter::OnlyAvailable.matches(&available));
        assert!(!Filter::OnlyAvailable.matches(&gated));
    }

    // The listing sort relies on `Available` ordering before `Gated`, so lock in that
    // contract here.
    #[test]
    fn test_ordering() {
        let features = Features::new("some-feature");
        let available = Visibility::Available;
        let gated = Visibility::Gated {
            features: &features,
        };

        assert!(available < gated);
    }
}
