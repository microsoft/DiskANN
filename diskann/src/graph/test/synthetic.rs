/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{Init, Matrix};

use crate::graph::AdjacencyList;

/// A synthetic data generator that generates points and neighbors in a regular latice with
/// the given number of dimensions.
///
/// * One: Create a 1-dimensional line of positive numbers beginning at 0.
/// * Three: Create a 3-dimensional cube with the bottom most corner at (0, 0, 0).
/// * Four: Create a 4-dimensional hypercube with the bottom most corner at (0, 0, 0, 0).
///
/// When generating neighbors with [`Self::neighbors`], vertices will be connected to their
/// neighbors along all cardinal axes. In general, an interior point in `N` dimensions will
/// have `2N` neighbors.
#[derive(Debug, Clone, Copy)]
pub enum Grid {
    One,
    Three,
    Four,
}

impl Grid {
    /// Return the generated grid with `f32` elements.
    ///
    /// See [`Self::data_as`] for documentation on the order of generation.
    pub fn data(self, size: usize) -> Matrix<f32> {
        Self::data_as(self, size, |i: usize| i as f32)
    }

    /// Return the generated grid where each point's value is its location in the grid.
    ///
    /// The closure `F` will be invoked on each dimensional value and should preserve the
    /// numeric value of the argument.
    ///
    /// Points will be generate such that the last coordinate varies the fastest, the second
    /// to last coordinate varies the second fastest etc.
    ///
    /// Coordinates are generated in ascending order.
    ///
    /// The returned matrix will have `self.dim()` columns and `self.dim() ^ size` rows.
    ///
    /// ```rust
    /// use diskann::graph::test::synthetic::Grid;
    ///
    /// fn identity(x: usize) -> usize {
    ///     x
    /// }
    ///
    /// // One dimensional data.
    /// let d = (Grid::One).data_as(3, identity);
    /// assert_eq!(d.row(0), &[0]);
    /// assert_eq!(d.row(1), &[1]);
    /// assert_eq!(d.row(2), &[2]);
    ///
    /// // Three dimensional data.
    /// let d = (Grid::Three).data_as(2, identity);
    ///
    /// assert_eq!(d.nrows(), 8);
    /// assert_eq!(d.ncols(), 3);
    ///
    /// assert_eq!(d.row(0), &[0, 0, 0]);
    /// assert_eq!(d.row(1), &[0, 0, 1]);
    ///
    /// assert_eq!(d.row(2), &[0, 1, 0]);
    /// assert_eq!(d.row(3), &[0, 1, 1]);
    ///
    /// assert_eq!(d.row(4), &[1, 0, 0]);
    /// assert_eq!(d.row(5), &[1, 0, 1]);
    ///
    /// assert_eq!(d.row(6), &[1, 1, 0]);
    /// assert_eq!(d.row(7), &[1, 1, 1]);
    ///
    /// // Four dimensional data.
    /// let d = (Grid::Four).data_as(2, identity);
    ///
    /// assert_eq!(d.nrows(), 16);
    /// assert_eq!(d.ncols(), 4);
    ///
    /// assert_eq!(d.row(0), &[0, 0, 0, 0]);
    /// assert_eq!(d.row(1), &[0, 0, 0, 1]);
    ///
    /// assert_eq!(d.row(2), &[0, 0, 1, 0]);
    /// assert_eq!(d.row(3), &[0, 0, 1, 1]);
    ///
    /// assert_eq!(d.row(4), &[0, 1, 0, 0]);
    /// assert_eq!(d.row(5), &[0, 1, 0, 1]);
    ///
    /// assert_eq!(d.row(6), &[0, 1, 1, 0]);
    /// assert_eq!(d.row(7), &[0, 1, 1, 1]);
    ///
    /// assert_eq!(d.row(8), &[1, 0, 0, 0]);
    /// assert_eq!(d.row(9), &[1, 0, 0, 1]);
    ///
    /// // etc.
    /// ```
    pub fn data_as<F, R>(self, size: usize, mut f: F) -> Matrix<R>
    where
        F: FnMut(usize) -> R,
    {
        match self {
            Self::One => {
                let mut i = 0;
                let init = Init(|| {
                    let this = f(i);
                    i += 1;
                    this
                });
                Matrix::new(init, size, 1)
            }
            Self::Three => {
                // The whole we do with the array here is to avoid a `Default` bound on `R`
                // by constructing the grid as we initialize the matrix.
                //
                // Is it overkill? Yes. Is it fun? Also yes!
                let mut v = [0; 3];
                let mut i = 0;
                let init = Init(|| {
                    let value = f(v[i]);
                    i += 1;
                    if i == 3 {
                        i = 0;
                        increment(&mut v, size);
                    }
                    value
                });

                Matrix::new(init, size.pow(self.dim().into()), 3)
            }
            Self::Four => {
                let mut v = [0; 4];
                let mut i = 0;
                let init = Init(|| {
                    let value = f(v[i]);
                    i += 1;
                    if i == 4 {
                        i = 0;
                        increment(&mut v, size);
                    }
                    value
                });

                Matrix::new(init, size.pow(self.dim().into()), 4)
            }
        }
    }

    /// Generate adjacency lists for a grid of the specified size. Nodes will be connected
    /// if they differ by exactly one coordinate, resulting in a hypercube lattice.
    ///
    /// This generation assumes elements are generated in the same order as that described
    /// by [`Self::data_as`] with elements numbered from `0` to `self.dim() ^ size`.
    ///
    /// The returned vector will have `self.dim() ^ size` rows and each adjacency list will
    /// have at most `2 * self.dim()` entries.
    ///
    /// # Note
    ///
    /// If `size` exceeds `u32::MAX` - the returned `Vec` will no longer have the correct
    /// length.
    pub fn neighbors(self, size: usize) -> Vec<AdjacencyList<u32>> {
        let mut lists: Vec<AdjacencyList<u32>> = Vec::new();

        let size: u32 = size as u32;
        match self {
            Self::One => {
                for i in 0..size {
                    let mut list = AdjacencyList::with_capacity(2);
                    if i > 0 {
                        list.push(i - 1);
                    }
                    if i < size - 1 {
                        list.push(i + 1);
                    }
                    lists.push(list);
                }
            }
            Self::Three => {
                let map = |i, j, k| (i * size * size) + (j * size) + k;
                for i in 0..size {
                    for j in 0..size {
                        for k in 0..size {
                            let mut list = AdjacencyList::new();
                            if i > 0 {
                                list.push(map(i - 1, j, k));
                            }
                            if i < size - 1 {
                                list.push(map(i + 1, j, k));
                            }
                            if j > 0 {
                                list.push(map(i, j - 1, k));
                            }
                            if j < size - 1 {
                                list.push(map(i, j + 1, k));
                            }
                            if k > 0 {
                                list.push(map(i, j, k - 1));
                            }
                            if k < size - 1 {
                                list.push(map(i, j, k + 1));
                            }
                            lists.push(list);
                        }
                    }
                }
            }
            Self::Four => {
                let map =
                    |i, j, k, l| (i * size * size * size) + (j * size * size) + (k * size) + l;

                for i in 0..size {
                    for j in 0..size {
                        for k in 0..size {
                            for l in 0..size {
                                let mut list = AdjacencyList::new();
                                if i > 0 {
                                    list.push(map(i - 1, j, k, l));
                                }
                                if i < size - 1 {
                                    list.push(map(i + 1, j, k, l));
                                }
                                if j > 0 {
                                    list.push(map(i, j - 1, k, l));
                                }
                                if j < size - 1 {
                                    list.push(map(i, j + 1, k, l));
                                }
                                if k > 0 {
                                    list.push(map(i, j, k - 1, l));
                                }
                                if k < size - 1 {
                                    list.push(map(i, j, k + 1, l));
                                }
                                if l > 0 {
                                    list.push(map(i, j, k, l - 1));
                                }
                                if l < size - 1 {
                                    list.push(map(i, j, k, l + 1));
                                }
                                lists.push(list);
                            }
                        }
                    }
                }
            }
        };

        lists
    }

    /// Return the graph start point for a grid of the given size.
    ///
    /// This returns a vector of length `self.dim()` populated with `size as f32`.
    pub fn start_point(self, size: usize) -> Vec<f32> {
        Self::start_point_as(self, size, |i: usize| i as f32)
    }

    /// Return the graph start point for a grid of the given size.
    ///
    /// This returns a vector of length `self.dim()` populated with repeated calls to `f(size)`.
    pub fn start_point_as<F, R>(self, size: usize, mut f: F) -> Vec<R>
    where
        F: FnMut(usize) -> R,
    {
        (0..self.dim()).map(|_| f(size)).collect()
    }

    /// Return the number of dimensions in the grid.
    pub fn dim(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Three => 3,
            Self::Four => 4,
        }
    }

    /// Return the number of points in a grid with the given edge size.
    pub fn num_points(self, size: usize) -> usize {
        size.pow(self.dim().into())
    }

    #[inline(never)]
    pub(super) fn setup(self, size: usize, start_id: u32) -> Setup {
        let num_points = self.num_points(size);

        Setup {
            start_point: self.start_point(size),
            start_id,
            start_neighbors: AdjacencyList::from_iter_unique(std::iter::once(
                num_points as u32 - 1,
            )),
            data: self.data(size),
            neighbors: self.neighbors(size),
        }
    }
}

fn increment<const N: usize>(array: &mut [usize; N], modulo: usize) {
    const {
        assert!(
            N != 0,
            "this algorithm doesn't work on 0 dimensional arrays"
        )
    };

    let mut i = N - 1;
    loop {
        let v = &mut array[i];
        *v += 1;
        if *v == modulo {
            *v = 0;
            match i.checked_sub(1) {
                Some(j) => i = j,
                None => return,
            }
        } else {
            return;
        }
    }
}

#[derive(Debug)]
pub(super) struct Setup {
    start_point: Vec<f32>,
    start_id: u32,
    start_neighbors: AdjacencyList<u32>,

    data: Matrix<f32>,
    neighbors: Vec<AdjacencyList<u32>>,
}

impl Setup {
    pub(super) fn start_point(&self) -> Vec<f32> {
        self.start_point.clone()
    }

    pub(super) fn start_id(&self) -> u32 {
        self.start_id
    }

    pub(super) fn start_neighbors(&self) -> impl Iterator<Item = (u32, AdjacencyList<u32>)> {
        std::iter::once((self.start_id(), self.start_neighbors.clone()))
    }

    pub(super) fn setup(&self) -> impl Iterator<Item = (u32, Vec<f32>, AdjacencyList<u32>)> {
        let mut i = 0u32;
        self.data
            .row_iter()
            .zip(self.neighbors.iter())
            .map(move |(data, neighbors)| {
                let id = i;
                i = i.wrapping_add(1);
                (id, data.into(), neighbors.clone())
            })
    }
}

//////////
// Test //
//////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increment_2() {
        let mut v = [0, 0];

        increment(&mut v, 3);
        assert_eq!(v, [0, 1]);

        increment(&mut v, 3);
        assert_eq!(v, [0, 2]);

        increment(&mut v, 3);
        assert_eq!(v, [1, 0]);

        increment(&mut v, 3);
        assert_eq!(v, [1, 1]);

        increment(&mut v, 3);
        assert_eq!(v, [1, 2]);

        increment(&mut v, 3);
        assert_eq!(v, [2, 0]);

        increment(&mut v, 3);
        assert_eq!(v, [2, 1]);

        increment(&mut v, 3);
        assert_eq!(v, [2, 2]);

        increment(&mut v, 3);
        assert_eq!(v, [0, 0]);
    }

    #[test]
    fn test_increment_3() {
        let mut v = [0, 0, 0];

        increment(&mut v, 2);
        assert_eq!(v, [0, 0, 1]);

        increment(&mut v, 2);
        assert_eq!(v, [0, 1, 0]);

        increment(&mut v, 2);
        assert_eq!(v, [0, 1, 1]);

        increment(&mut v, 2);
        assert_eq!(v, [1, 0, 0]);

        increment(&mut v, 2);
        assert_eq!(v, [1, 0, 1]);

        increment(&mut v, 2);
        assert_eq!(v, [1, 1, 0]);

        increment(&mut v, 2);
        assert_eq!(v, [1, 1, 1]);

        increment(&mut v, 2);
        assert_eq!(v, [0, 0, 0]);
    }
}
