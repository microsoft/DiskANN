/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod sequencer;
pub(crate) use sequencer::Sequencer;

// Longer Running Tests
mod epoch;

macro_rules! assert_contains {
    ($haystack:expr, $needle:expr $(,)?) => {
        match (&$haystack, &$needle) {
            (haystack, needle) => {
                assert!(
                    haystack.contains(needle),
                    "{:?} does not contain {:?}",
                    haystack,
                    needle,
                );
            }
        }
    };
    ($haystack:expr, $needle:expr, $fmt:expr) => {
        match (&$haystack, &$needle) {
            (haystack, needle) => {
                assert!(
                    haystack.contains(needle),
                    concat!("{:?} does not contain {:?} -- ", $fmt),
                    haystack,
                    needle,
                );
            }
        }
    };
    ($haystack:expr, $needle:expr, $fmt:expr, $($args:tt)*) => {
        match (&$haystack, &$needle) {
            (haystack, needle) => {
                assert!(
                    haystack.contains(needle),
                    concat!("{:?} does not contain {:?} -- ", $fmt),
                    haystack,
                    needle,
                    $($args)*
                );
            }
        }
    }
}

pub(crate) use assert_contains;
