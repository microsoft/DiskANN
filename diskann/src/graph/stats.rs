/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//--------------//
// Degree Stats //
//--------------//

#[derive(Debug, Default, Clone)]
pub struct DegreeStats {
    /// The maximum adjacency list length.
    maximum: usize,
    /// The minimum adjacency list length.
    minimum: usize,
    /// The total number of edges observed.
    sum: usize,
    /// The total number of adjacency lists observed.
    count: usize,
    /// Counts for adjacency lists of length 0, 1, and 2
    histogram: [usize; 3],
}

impl DegreeStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn maximum(&self) -> usize {
        self.maximum
    }

    pub fn minimum(&self) -> usize {
        self.minimum
    }

    pub fn sum(&self) -> usize {
        self.sum
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn average(&self) -> f64 {
        (self.sum as f64) / (self.count as f64)
    }

    pub fn count_less_than_two(&self) -> usize {
        self.histogram
            .into_iter()
            .take(3)
            .fold(0, |sum, i| sum.saturating_add(i))
    }

    pub fn count_empty(&self) -> usize {
        self.histogram[0]
    }

    pub fn push(&mut self, degree: usize) {
        let Self {
            maximum,
            minimum,
            sum,
            count,
            histogram,
        } = self;

        *maximum = (*maximum).max(degree);
        *minimum = (*minimum).max(degree);
        *sum = (*sum).saturating_add(degree);
        *count = (*count).saturating_add(1);

        match degree {
            0 => histogram[0] = histogram[0].saturating_add(1),
            1 => histogram[1] = histogram[1].saturating_add(1),
            2 => histogram[2] = histogram[2].saturating_add(1),
            _ => {}
        }
    }

    pub fn accum(&mut self, other: &DegreeStats) {
        let Self {
            maximum,
            minimum,
            sum,
            count,
            histogram,
        } = self;

        (*maximum) = (*maximum).max(other.maximum);
        (*minimum) = (*minimum).max(other.minimum);
        (*sum) = (*sum).saturating_add(other.sum);
        (*count) = (*count).saturating_add(other.count);

        std::iter::zip(histogram.iter_mut(), other.histogram.iter())
            .for_each(|(s, o)| (*s) = (*s).saturating_add(*o));
    }
}

impl Extend<usize> for DegreeStats {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = usize>,
    {
        iter.into_iter().for_each(|count| self.push(count))
    }
}

impl FromIterator<usize> for DegreeStats {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        let mut me = Self::default();
        me.extend(iter);
        me
    }
}
