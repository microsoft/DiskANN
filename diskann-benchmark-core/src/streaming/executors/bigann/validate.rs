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
        if other.is_empty() {
            if self.is_empty() {
                return Remaining::None;
            }
            return Remaining::One(self);
        }

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

    fn compact(&mut self) {
        if self.active.len() <= 1 {
            return;
        }

        let mut write = 0;
        for read in 1..self.active.len() {
            let write_end = self.active[write].end;
            let read_start = self.active[read].start;

            if write_end == read_start {
                self.active[write].end = self.active[read].end;
            } else {
                assert!(
                    read_start > write_end,
                    "internal invariant has been violated {} > {}",
                    read_start,
                    write_end,
                );

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

                self.currently_active -= tags.len();
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Slot {
    Back,
    Before(usize),
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
