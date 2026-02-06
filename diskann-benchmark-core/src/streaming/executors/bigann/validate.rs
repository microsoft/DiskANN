/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ops::Range;

use super::Args;
use crate::streaming;

#[derive(Debug, Clone, Copy, PartialEq)]
struct SimpleRange {
    start: usize,
    end: usize,
}

impl SimpleRange {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end: end.max(start),
        }
    }

    fn len(&self) -> usize {
        self.end - self.start
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn overlaps(&self, other: SimpleRange) -> bool {
        self.contains(other.start) || other.contains(self.start)
    }

    fn remove(self, other: SimpleRange) -> Remaining {
        // If the other range is empty - there's nothing to remove.
        if other.is_empty() {
            if self.is_empty() {
                return Remaining::None;
            } else {
                return Remaining::One(self);
            }
        }

        // Five cases to consider:
        //
        // 1. The two ranges are disjoint - there's nothing to remove.
        // 2. `other` strictly contains `self - we remove all of `self`.
        // 3. `other` overlaps with the just the start of `self` - return the last part of `self`.
        // 4. `other` overlaps with the just the end of `self` - return the last part of `self`.
        // 5. `other` is inside of `self` - we must return two pieces.

        if !self.overlaps(other) {
            Remaining::One(self)
        } else if other.strictly_contains(self) {
            Remaining::None
        } else if other.start <= self.start {
            Remaining::One(Self::new(other.end, self.end))
        } else if other.end >= self.end {
            Remaining::One(Self::new(self.start, other.start))
        } else {
            assert!(self.strictly_contains(other));

            Remaining::Two(
                Self::new(self.start, other.start),
                Self::new(other.end, self.end),
            )
        }
    }

    fn contains(&self, i: usize) -> bool {
        self.as_range().contains(&i)
    }

    fn strictly_contains(&self, other: SimpleRange) -> bool {
        other.start >= self.start && other.end <= self.end
    }

    fn as_range(&self) -> Range<usize> {
        self.start..self.end
    }
}

impl std::fmt::Display for SimpleRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Remaining {
    None,
    One(SimpleRange),
    Two(SimpleRange, SimpleRange),
}

impl From<Range<usize>> for SimpleRange {
    fn from(range: Range<usize>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }
}

/// A test `stream` that ensures a Runbook is well formed.
pub(super) struct Validate {
    active: Vec<SimpleRange>,
    max_active: usize,
    currently_active: usize,
    max_tag: Option<usize>,
}

impl Validate {
    pub(super) fn new() -> Self {
        Self {
            active: Vec::new(),
            max_active: 0,
            currently_active: 0,
            max_tag: None,
        }
    }

    #[cfg(test)]
    fn new_test(active: Vec<SimpleRange>) -> Self {
        let currently_active = active.iter().map(|r| r.len()).sum();
        let max_tag = active.iter().filter_map(|r| r.end.checked_sub(1)).max();

        Self {
            active,
            max_active: currently_active,
            currently_active,
            max_tag,
        }
    }

    /// Return the maximum number of active points seen so far.
    pub(super) fn max_active(&self) -> usize {
        self.max_active
    }

    /// Return the maximum tag seen so far.
    pub(super) fn max_tag(&self) -> Option<usize> {
        self.max_tag
    }

    fn update_max(&mut self, tag: usize) {
        match self.max_tag.as_mut() {
            Some(max) => *max = (*max).max(tag),
            None => self.max_tag = Some(tag),
        }
    }

    /// Merge adjacent ranges where one range's end equals another's start.
    ///
    /// Assumes the ranges in `active` are already disjoint and ordered by start.
    /// After compaction, no two consecutive ranges will have `r1.end == r2.start`.
    fn compact(&mut self) {
        if self.active.len() <= 1 {
            return;
        }

        let mut write = 0;
        for read in 1..self.active.len() {
            let write_end = self.active[write].end;
            let read_start = self.active[read].start;

            if write_end == read_start {
                // Merge: extend the current range to include the next one
                self.active[write].end = self.active[read].end;
            } else {
                assert!(
                    read_start > write_end,
                    "internal invariant has been violated {} > {}",
                    read_start,
                    write_end,
                );

                // No merge: move to the next write position
                write += 1;
                if write != read {
                    self.active[write] = self.active[read];
                }
            }
        }

        self.active.truncate(write + 1);
    }

    fn find(&self, tags: SimpleRange) -> Slot {
        for (i, range) in self.active.iter().enumerate() {
            if tags.overlaps(*range) {
                return Slot::In(i);
            }
            if range.start >= tags.end {
                return Slot::Before(i);
            }
        }
        Slot::Back
    }

    fn insert(&mut self, tags: SimpleRange) -> anyhow::Result<()> {
        match self.find(tags) {
            Slot::Back => self.active.push(tags),
            Slot::Before(i) => self.active.insert(i, tags),
            Slot::In(_) => anyhow::bail!("tag range {} is already present for insertion", tags),
        }

        // We've successfully added tags, so this update is accurate.
        self.currently_active += tags.len();
        self.max_active = self.max_active.max(self.currently_active);
        if let Some(tag) = tags.end.checked_sub(1) {
            self.update_max(tag);
        }

        self.compact();
        Ok(())
    }

    fn replace(&mut self, tags: SimpleRange) -> anyhow::Result<()> {
        match self.find(tags) {
            Slot::Back | Slot::Before(_) => Err(anyhow::anyhow!(
                "tag range {} not valid for replacement",
                tags
            )),
            Slot::In(i) => {
                // Due to compaction, if `tags` is truly present in the list of tags, it
                // cannot be split across two ranges.
                let overlaps = self.active[i];
                if !overlaps.strictly_contains(tags) {
                    Err(anyhow::anyhow!("could not match the entire range {}", tags))
                } else {
                    Ok(())
                }
            }
        }
    }

    fn delete(&mut self, tags: SimpleRange) -> anyhow::Result<()> {
        match self.find(tags) {
            Slot::Back | Slot::Before(_) => {
                Err(anyhow::anyhow!("tag range {} not valid for deletion", tags))
            }
            Slot::In(i) => {
                let current = self.active[i];
                if !current.strictly_contains(tags) {
                    return Err(anyhow::anyhow!(
                        "could not match the entire range {} for deletion",
                        tags
                    ));
                }

                match current.remove(tags) {
                    Remaining::None => {
                        self.active.remove(i);
                    }
                    Remaining::One(remaining) => self.active[i] = remaining,
                    Remaining::Two(first, last) => {
                        self.active.splice(i..(i + 1), [first, last]);
                    }
                }

                // We've successfully remove tags, so this update is accurate.
                self.currently_active -= tags.len();
                Ok(())
            }
        }
    }
}

/// Location where a range would be located in the sorted ranges vector.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Slot {
    /// Range would be placed in the back. It is disjoint from existing ranges.
    Back,

    /// Range would be before this index. This implies that the range is disjoin from
    /// existing ranges.
    Before(usize),

    /// Range partial overlaps with this index. This should be the first such range that
    /// overlaps.
    In(usize),
}

impl streaming::Stream<Args> for Validate {
    type Output = ();

    fn search(&mut self, _args: super::Search<'_>) -> anyhow::Result<()> {
        Ok(())
    }

    fn insert(&mut self, args: super::Insert) -> anyhow::Result<()> {
        self.insert(args.ids.into())
    }

    fn replace(&mut self, args: super::Replace) -> anyhow::Result<()> {
        self.replace(args.ids.into())
    }

    fn delete(&mut self, args: super::Delete) -> anyhow::Result<()> {
        self.delete(args.ids.into())
    }

    fn maintain(&mut self, _args: ()) -> anyhow::Result<()> {
        Ok(())
    }

    fn needs_maintenance(&mut self) -> bool {
        false
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn range(start: usize, end: usize) -> SimpleRange {
        SimpleRange::new(start, end)
    }

    #[test]
    fn test_range() {
        let r = range(0, 1);
        assert_eq!(r.start, 0);
        assert_eq!(r.end, 1);
        assert_eq!(r.len(), 1);

        let r = range(10, 5);
        assert_eq!(r.start, 10);
        assert_eq!(r.end, 10);
        assert_eq!(r.len(), 0);

        let r = range(5, 100);
        assert_eq!(r.start, 5);
        assert_eq!(r.end, 100);
        assert_eq!(r.len(), 95);
    }

    #[test]
    fn simple_range() {
        // Disjoint ranges.
        {
            let a = range(0, 10);
            let b = range(20, 30);

            assert!(!a.overlaps(b), "a = {a}, b = {b}");
            assert!(!b.overlaps(a), "a = {a}, b = {b}");

            assert!(!a.strictly_contains(b));
            assert!(!b.strictly_contains(a));

            assert_eq!(a.remove(b), Remaining::One(a));
            assert_eq!(b.remove(a), Remaining::One(b));
        }

        // Disjoint ranges - b
        {
            let a = range(0, 10);
            let b = range(10, 30);

            assert!(!a.overlaps(b), "a = {a}, b = {b}");
            assert!(!b.overlaps(a), "a = {a}, b = {b}");

            assert!(!a.strictly_contains(b));
            assert!(!b.strictly_contains(a));

            assert_eq!(a.remove(b), Remaining::One(a));
            assert_eq!(b.remove(a), Remaining::One(b));
        }

        // Partial overlap.
        {
            let a = range(10, 20);
            let b = range(5, 15);

            assert!(a.overlaps(b), "a = {a}, b = {b}");
            assert!(b.overlaps(a), "a = {a}, b = {b}");

            assert!(!a.strictly_contains(b));
            assert!(!b.strictly_contains(a));

            assert_eq!(a.remove(b), Remaining::One(range(15, 20)));
            assert_eq!(b.remove(a), Remaining::One(range(5, 10)));
        }

        // Total overlap.
        {
            let a = range(10, 20);
            let b = range(10, 20);

            assert!(a.overlaps(b), "a = {a}, b = {b}");
            assert!(a.strictly_contains(b));
            assert_eq!(a.remove(b), Remaining::None);
        }

        // Totally inside.
        {
            let a = range(10, 20);
            let b = range(12, 18);

            assert!(a.overlaps(b), "a = {a}, b = {b}");

            assert!(a.strictly_contains(b));
            assert!(!b.strictly_contains(a));

            assert_eq!(a.remove(b), Remaining::Two(range(10, 12), range(18, 20)));
            assert_eq!(b.remove(a), Remaining::None);
        }

        // Totally inside - touching left.
        {
            let a = range(10, 20);
            let b = range(10, 15);

            assert!(a.overlaps(b), "a = {a}, b = {b}");

            assert!(a.strictly_contains(b));
            assert!(!b.strictly_contains(a));

            assert_eq!(a.remove(b), Remaining::One(range(15, 20)));
            assert_eq!(b.remove(a), Remaining::None);
        }

        // Totally inside - touching right.
        {
            let a = range(10, 20);
            let b = range(15, 20);

            assert!(a.overlaps(b), "a = {a}, b = {b}");

            assert!(a.strictly_contains(b));
            assert!(!b.strictly_contains(a));

            assert_eq!(a.remove(b), Remaining::One(range(10, 15)));
            assert_eq!(b.remove(a), Remaining::None);
        }

        // Empty ranges.
        {
            let a = range(10, 10);
            assert_eq!(a.remove(a), Remaining::None);

            let a = range(10, 20);
            for j in [5, 10, 15, 20, 25] {
                let b = range(j, j);
                assert_eq!(a.remove(b), Remaining::One(a));
            }
        }
    }

    #[test]
    fn test_compact() {
        // Empty
        {
            let mut v = Validate::new_test(vec![]);
            v.compact();
            assert_eq!(&v.active, &[]);
        }

        // One
        {
            let start = vec![range(10, 20)];
            let mut v = Validate::new_test(start.clone());
            v.compact();
            assert_eq!(v.active, start);
        }

        // No change.
        {
            let start = vec![range(10, 20), range(30, 40), range(50, 60)];

            let mut v = Validate::new_test(start.clone());
            v.compact();

            assert_eq!(v.active, start);
        }

        // Merge in multiple locations with empty elements.
        {
            let active = vec![
                range(0, 5),  // + Merged
                range(5, 5),  // |
                range(5, 5),  // |
                range(5, 10), // |
                range(20, 30),
                range(35, 40), // + Merged
                range(40, 41), // |
                range(41, 41), // |
                range(41, 50), // |
                range(55, 60),
                range(70, 70),  // + Merged
                range(70, 75),  // |
                range(75, 100), // |
            ];

            let mut v = Validate::new_test(active);
            v.compact();

            let expected = [
                range(0, 10),
                range(20, 30),
                range(35, 50),
                range(55, 60),
                range(70, 100),
            ];

            assert_eq!(&*v.active, &expected);
        }
    }

    #[test]
    fn test_find() {
        // Empty
        {
            let v = Validate::new_test(vec![]);
            assert_eq!(v.find(range(0, 10)), Slot::Back);
        }

        // One
        {
            let v = Validate::new_test(vec![range(10, 20)]);

            assert_eq!(v.find(range(0, 5)), Slot::Before(0));
            assert_eq!(v.find(range(5, 10)), Slot::Before(0));
            assert_eq!(v.find(range(8, 12)), Slot::In(0));
            assert_eq!(v.find(range(10, 15)), Slot::In(0));
            assert_eq!(v.find(range(15, 20)), Slot::In(0));
            assert_eq!(v.find(range(18, 22)), Slot::In(0));
            assert_eq!(v.find(range(20, 25)), Slot::Back);
        }

        // Multiple
        {
            let v = Validate::new_test(vec![range(10, 20), range(30, 40), range(50, 60)]);

            assert_eq!(v.find(range(0, 5)), Slot::Before(0));
            assert_eq!(v.find(range(5, 10)), Slot::Before(0));
            assert_eq!(v.find(range(15, 25)), Slot::In(0));

            assert_eq!(v.find(range(20, 25)), Slot::Before(1));
            assert_eq!(v.find(range(28, 35)), Slot::In(1));
            assert_eq!(v.find(range(35, 45)), Slot::In(1));

            assert_eq!(v.find(range(45, 50)), Slot::Before(2));
            assert_eq!(v.find(range(55, 65)), Slot::In(2));
            assert_eq!(v.find(range(65, 70)), Slot::Back);
        }
    }

    #[test]
    fn test_insert() {
        let mut v = Validate::new_test(vec![]);

        v.insert(range(10, 20)).unwrap();
        assert_eq!(&*v.active, &[range(10, 20)]);
        assert_eq!(v.max_active(), 10);
        assert_eq!(v.max_tag(), Some(19));

        v.insert(range(30, 40)).unwrap();
        assert_eq!(&*v.active, &[range(10, 20), range(30, 40)]);
        assert_eq!(v.max_active(), 20);
        assert_eq!(v.max_tag(), Some(39));

        v.insert(range(20, 30)).unwrap();
        assert_eq!(&*v.active, &[range(10, 40)]);
        assert!(v.max_active() == 30);
        assert_eq!(v.max_tag(), Some(39));

        let result = v.insert(range(15, 25));
        assert!(result.is_err());

        v.insert(range(0, 5)).unwrap();
        assert_eq!(&*v.active, &[range(0, 5), range(10, 40)]);
        assert_eq!(v.max_active(), 35);
        assert_eq!(v.max_tag(), Some(39));

        v.insert(range(7, 8)).unwrap();
        assert_eq!(&*v.active, &[range(0, 5), range(7, 8), range(10, 40)]);
        assert_eq!(v.max_active(), 36);
        assert_eq!(v.max_tag(), Some(39));

        v.insert(range(4, 8)).unwrap_err();
        v.insert(range(50, 60)).unwrap();
        assert_eq!(
            &*v.active,
            &[range(0, 5), range(7, 8), range(10, 40), range(50, 60)]
        );
        assert_eq!(v.max_active(), 46);
        assert_eq!(v.max_tag(), Some(59));

        v.insert(range(5, 7)).unwrap();
        assert_eq!(&*v.active, &[range(0, 8), range(10, 40), range(50, 60)]);
        assert_eq!(v.max_active(), 48);
        assert_eq!(v.max_tag(), Some(59));

        v.insert(range(40, 50)).unwrap();
        assert_eq!(&*v.active, &[range(0, 8), range(10, 60)]);
        assert_eq!(v.max_active(), 58);
        assert_eq!(v.max_tag(), Some(59));

        v.insert(range(8, 10)).unwrap();
        assert_eq!(&*v.active, &[range(0, 60)]);
        assert_eq!(v.max_active(), 60);
        assert_eq!(v.max_tag(), Some(59));
    }

    #[test]
    fn test_replace() {
        let mut v = Validate::new_test(vec![range(10, 20), range(30, 40)]);

        // Success
        v.replace(range(12, 18)).unwrap();
        v.replace(range(10, 20)).unwrap();
        v.replace(range(15, 15)).unwrap();
        v.replace(range(30, 35)).unwrap();

        // Failure
        v.replace(range(15, 25)).unwrap_err();
        v.replace(range(5, 15)).unwrap_err();
        v.replace(range(25, 35)).unwrap_err();
        v.replace(range(40, 50)).unwrap_err();
    }

    #[test]
    fn test_delete() {
        let mut v = Validate::new_test(vec![range(10, 20), range(30, 40)]);

        assert_eq!(v.max_active(), 20);

        // Partial deletions
        v.delete(range(15, 18)).unwrap();
        assert_eq!(&*v.active, &[range(10, 15), range(18, 20), range(30, 40)]);

        v.delete(range(35, 36)).unwrap();
        assert_eq!(
            &*v.active,
            &[range(10, 15), range(18, 20), range(30, 35), range(36, 40)]
        );

        // Partial at beginning
        v.delete(range(10, 12)).unwrap();
        assert_eq!(
            &*v.active,
            &[range(12, 15), range(18, 20), range(30, 35), range(36, 40)]
        );

        // Partial at end
        v.delete(range(32, 35)).unwrap();
        assert_eq!(
            &*v.active,
            &[range(12, 15), range(18, 20), range(30, 32), range(36, 40)]
        );

        // Full deletions
        v.delete(range(18, 20)).unwrap();
        assert_eq!(&*v.active, &[range(12, 15), range(30, 32), range(36, 40)]);

        v.delete(range(36, 40)).unwrap();
        assert_eq!(&*v.active, &[range(12, 15), range(30, 32)]);

        assert_eq!(v.max_active(), 20);
        assert_eq!(v.max_tag(), Some(39));

        // Failure cases.
        v.delete(range(20, 25)).unwrap_err();
        v.delete(range(0, 10)).unwrap_err();
        v.delete(range(10, 14)).unwrap_err();
        v.delete(range(31, 33)).unwrap_err();
        v.delete(range(40, 50)).unwrap_err();
    }
}
