/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    mem::size_of,
    num::NonZeroUsize,
    ops::{Deref, DerefMut, Index, IndexMut, Range},
};

use rand::{distr::Distribution, Rng};

use super::Static;

pub const PAGESIZE: usize = 4096;

/// Compute the offset from `current_position` to an alignment of `goal_offset` relative
/// to a page boundary.
fn align_to_offset<T>(current_position: usize, goal_offset: usize) -> usize {
    let current_offset = current_position % PAGESIZE;
    let step_bytes = if goal_offset > current_offset {
        goal_offset - current_offset
    } else {
        (goal_offset + 4096) - current_offset
    };
    assert_eq!(step_bytes % size_of::<T>(), 0);
    step_bytes / std::mem::size_of::<T>()
}

pub(crate) struct AlignedVector<T> {
    data: Vec<T>,
    start_offset: usize,
    dim: NonZeroUsize,
}

impl<T> AlignedVector<T> {
    pub(crate) fn new(dim: NonZeroUsize, page_alignment: usize) -> Self
    where
        T: Default + Clone,
    {
        assert_eq!(
            page_alignment % std::mem::size_of::<T>(),
            0,
            "alignment must be a multiple of the element size: {}",
            std::mem::size_of::<T>()
        );

        let data: Vec<T> = vec![T::default(); dim.get() + PAGESIZE / size_of::<T>()];
        let start_offset = align_to_offset::<T>(data.as_ptr() as usize, page_alignment);

        let slice = &data[start_offset..];
        assert!(slice.len() >= dim.get());
        assert_eq!(
            (slice.as_ptr() as usize) % PAGESIZE,
            page_alignment % PAGESIZE
        );

        Self {
            data,
            start_offset,
            dim,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.dim.get()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        let s = self.start_offset;
        &self.data[s..s + self.len()]
    }
}

impl<T> Deref for AlignedVector<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for AlignedVector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let s = self.start_offset;
        let range = s..s + self.len();
        &mut self.data[range]
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DatasetArgs {
    /// The offset inside a page where the first vector begins.
    alignment: usize,
    /// The dimension of each vector.
    dim: NonZeroUsize,
    /// The spacing between vectors. Must be greater than or equalt to `dim`.
    stride: NonZeroUsize,
    /// The number of vectors in the dataset.
    count: NonZeroUsize,
}

impl DatasetArgs {
    pub(crate) fn new(
        alignment: usize,
        dim: NonZeroUsize,
        stride: NonZeroUsize,
        count: NonZeroUsize,
    ) -> Self {
        assert!(
            stride.get() >= dim.get(),
            "Stride must be at least as large as dim"
        );
        DatasetArgs {
            alignment,
            dim,
            stride,
            count,
        }
    }

    pub(crate) fn dim(&self) -> NonZeroUsize {
        self.dim
    }

    pub(crate) fn alignment(&self) -> usize {
        self.alignment
    }
}

#[derive(Debug)]
pub(crate) struct AlignedDataset<T> {
    /// The raw underlying data.
    /// The actual start of the data begins at `start_offset` to allow arbitrary starting
    /// alignments.
    data: Vec<T>,
    /// The offset of the first vector.
    start_offset: usize,
    /// Parameters of the dataset.
    args: DatasetArgs,
}

impl<T> AlignedDataset<T> {
    pub(crate) fn new<R, D>(args: DatasetArgs, rng: &mut R, dist: D) -> Self
    where
        R: Rng,
        D: Distribution<T>,
        T: Default + Clone,
    {
        assert_eq!(
            args.alignment % std::mem::size_of::<T>(),
            0,
            "alignment must be a multiple of the element size: {}",
            std::mem::size_of::<T>()
        );

        // Figure out how big we need to make the underlying vector to ensure we can satisfy
        // the requested alignment.
        let alloc_size = args.stride.get() * args.count.get() + PAGESIZE / size_of::<T>();
        let data: Vec<T> = vec![T::default(); alloc_size];

        // Find the starting offset.
        let start_offset = align_to_offset::<T>(data.as_ptr() as usize, args.alignment);
        let slice = &data[start_offset..];
        assert!(slice.len() >= args.dim.get());
        assert_eq!(
            (slice.as_ptr() as usize) % PAGESIZE,
            args.alignment % PAGESIZE
        );

        // Create the dataset.
        let mut dataset = Self {
            data,
            start_offset,
            args,
        };

        // Populate the dataset.
        for i in 0..dataset.len() {
            let vector = &mut dataset[i];
            for v in vector.iter_mut() {
                *v = dist.sample(rng);
            }
        }

        dataset
    }

    /// Return the number of vectors in the dataset.
    pub(crate) fn len(&self) -> usize {
        self.args.count.get()
    }

    /// Return the dimension of each vector in the dataset.
    pub(crate) fn dim(&self) -> usize {
        self.args.dim.get()
    }

    /// Return an iterator over each vector.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &[T]> {
        self.data[self.start_offset..]
            .chunks_exact(self.args.stride.get())
            .take(self.len())
            .map(|v| &v[..self.dim()])
    }

    pub(crate) fn iter_sized<const N: usize>(
        &self,
        _dim: Static<N>,
    ) -> impl Iterator<Item = &[T; N]>
    where
        T: Copy,
    {
        assert_eq!(
            N,
            self.dim(),
            "static parameter N must be equal to the dynamic value `self.dim()`"
        );
        self.data[self.start_offset..]
            .chunks_exact(self.args.stride.get())
            .take(self.len())
            .map(|v| v[..N].try_into().unwrap())
    }

    /// Internal function to compute the range for an index operation.
    #[track_caller]
    fn index_range(&self, index: usize) -> Range<usize> {
        assert!(
            index < self.len(),
            "index {} must be less than the number of vectors: {}",
            index,
            self.len()
        );
        let start = self.start_offset + index * self.args.stride.get();
        start..(start + self.dim())
    }
}

impl<T> Index<usize> for AlignedDataset<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.index_range(index)]
    }
}

impl<T> IndexMut<usize> for AlignedDataset<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let range = self.index_range(index);
        &mut self.data[range]
    }
}
