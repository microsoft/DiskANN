/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Return whether or not the `self` contains `item`.
pub trait AssertContains<T>
where
    T: ?Sized,
{
    #[must_use]
    fn contains_for_assert(&self, item: &T) -> bool;
}

#[doc(hidden)]
#[macro_export]
macro_rules! assert_contains {
    ($haystack:expr, $needle:expr $(,)?) => {
        match (&$haystack, &$needle) {
            (haystack, needle) => {
                use $crate::testing::AssertContains as _;

                assert!(
                    haystack.contains_for_assert(needle),
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
                use $crate::testing::AssertContains as _;

                assert!(
                    haystack.contains_for_assert(needle),
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
                use $crate::testing::AssertContains as _;

                assert!(
                    haystack.contains_for_assert(needle),
                    concat!("{:?} does not contain {:?} -- ", $fmt),
                    haystack,
                    needle,
                    $($args)*
                );
            }
        }
    }
}

impl AssertContains<str> for str {
    fn contains_for_assert(&self, item: &str) -> bool {
        self.contains(item)
    }
}

impl<T> AssertContains<T> for [T]
where
    T: PartialEq,
{
    fn contains_for_assert(&self, item: &T) -> bool {
        self.contains(item)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    #[test]
    fn test_happy_path() {
        assert_contains!("haystack and needle", "needle");
        assert_contains!("haystack and needle", "needle", "some context");
        assert_contains!("haystack and needle", "needle", "some context: {}", 10);
        assert_contains!(
            "haystack and needle",
            "needle",
            "some context: {}, {}",
            10,
            20
        );

        assert_contains!(String::from("haystack and needle"), "needle");
        assert_contains!(
            String::from("haystack and needle"),
            "needle",
            "some context"
        );
        assert_contains!(
            String::from("haystack and needle"),
            "needle",
            "some context: {}",
            10
        );
        assert_contains!(
            String::from("haystack and needle"),
            "needle",
            "some context: {}, {}",
            10,
            20
        );

        assert_contains!("haystack and needle", String::from("needle"));
        assert_contains!(
            "haystack and needle",
            String::from("needle"),
            "some context"
        );
        assert_contains!(
            "haystack and needle",
            String::from("needle"),
            "some context: {}",
            10
        );
        assert_contains!(
            "haystack and needle",
            String::from("needle"),
            "some context: {}, {}",
            10,
            20
        );

        assert_contains!([0, 1, 2, 3], 3);
        assert_contains!([0, 1, 2, 3], 3, "some context");
        assert_contains!([0, 1, 2, 3], 3, "some context: {}", 10);
        assert_contains!([0, 1, 2, 3], 3, "some context: {}, {}", 10, 20);
    }

    #[test]
    #[should_panic = "\"haystack\" does not contain \"needle\""]
    fn fail_str_1() {
        assert_contains!("haystack", "needle");
    }

    #[test]
    #[should_panic = "\"haystack\" does not contain \"needle\""]
    fn fail_str_2() {
        assert_contains!("haystack", "needle",);
    }

    #[test]
    #[should_panic = "\"haystack\" does not contain \"needle\" -- some context"]
    fn fail_str_3() {
        assert_contains!("haystack", "needle", "some context");
    }

    #[test]
    #[should_panic = "\"haystack\" does not contain \"needle\" -- some context"]
    fn fail_str_4() {
        assert_contains!("haystack", "needle", "some context",);
    }

    #[test]
    #[should_panic = "\"haystack\" does not contain \"needle\" -- some context: 10"]
    fn fail_str_5() {
        assert_contains!("haystack", "needle", "some context: {}", 10);
    }

    #[test]
    #[should_panic = "\"haystack\" does not contain \"needle\" -- some context: 10, 20"]
    fn fail_str_6() {
        assert_contains!("haystack", "needle", "some context: {}, {}", 10, 20);
    }

    #[test]
    #[should_panic = "[1, 2, 3] does not contain 5"]
    fn fail_container_1() {
        assert_contains!([1, 2, 3], 5);
    }

    #[test]
    #[should_panic = "[1, 2, 3] does not contain 5"]
    fn fail_container_2() {
        assert_contains!([1, 2, 3], 5,);
    }

    #[test]
    #[should_panic = "[1, 2, 3] does not contain 5 -- some context"]
    fn fail_container_3() {
        assert_contains!([1, 2, 3], 5, "some context");
    }

    #[test]
    #[should_panic = "[1, 2, 3] does not contain 5 -- some context"]
    fn fail_container_4() {
        assert_contains!([1, 2, 3], 5, "some context",);
    }

    #[test]
    #[should_panic = "[1, 2, 3] does not contain 5 -- some context: 10"]
    fn fail_container_5() {
        assert_contains!([1, 2, 3], 5, "some context: {}", 10);
    }

    #[test]
    #[should_panic = "[1, 2, 3] does not contain 5 -- some context: 10, 20"]
    fn fail_container_6() {
        assert_contains!([1, 2, 3], 5, "some context: {}, {}", 10, 20);
    }
}
