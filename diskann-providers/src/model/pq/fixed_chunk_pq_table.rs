/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{ANNError, ANNResult, graph::test::debug_provider::DebugQuantizer, utils::IntoUsize};
use diskann_linalg::{self, Transpose};
use diskann_quantization::{
    CompressInto,
    product::{self, BasicTable},
    views::ChunkOffsetsBase,
};
use diskann_utils::views::{self, MatrixBase, MatrixView};
use diskann_vector::{PureDistanceFunction, distance};
use diskann_wide::ARCH;

use super::{NUM_PQ_CENTROIDS, distance as pq_distance};
use crate::utils::{Bridge, BridgeErr};

/// PQ Pivot table loading and calculate distance
///
/// The fields of this struct are public in the PQ crate to allow scoped computers direct
/// access to the internals.
#[derive(Debug, Clone)]
pub struct FixedChunkPQTable {
    /// The underlying table representation.
    table: BasicTable,

    /// centroid of each dimension
    centroids: Box<[f32]>,

    /// Optimized Product Quantization rotation matrix.  If not defined then OPQ is not
    /// used for this index
    opq_rotation_matrix: Option<Box<[f32]>>,
}

// These free functions use internals of the `FixedChunkPQTable`.
//
// We should clean up the API in the FFI.
pub fn direct_distance_impl<T>(
    pq_table: &[f32],
    chunk_offsets: &[usize],
    dim: usize,
    query_vec: &[f32],
    base_vec: &[u8],
) -> f32
where
    T: distance::simd::ResumableSIMDSchema<f32, f32, FinalReturn = f32>,
{
    let mut accumulator = distance::simd::Resumable::new(T::init(ARCH));
    let mut start = chunk_offsets[0];
    let num_pq_chunks = chunk_offsets.len() - 1;

    (0..num_pq_chunks).for_each(|chunk_index| {
        let stop = chunk_offsets[chunk_index + 1];
        let query = &query_vec[start..stop];
        let offset = base_vec[chunk_index] as usize;
        let chunk = &pq_table[(dim * offset + start)..(dim * offset + stop)];
        accumulator = distance::simd::simd_op(&accumulator, ARCH, query, chunk);
        start = stop;
    });
    accumulator.consume().sum()
}

/// Aggregate the pre-computed PQ distances for a collection of centroids.
///
/// # Arguments
/// * `pq_coordinates`: The coordinates of a PQ encoding used to access `precomputed_distances`
/// *
/// * `precomputed_distances`: Pre-computed distances between each chunk of a query vector
///   and each centroid with a row major layout:
///   ```ignore
///   d00 d01 d02 ... d0N // row0
///   d10 d11 d12 ... d1N // row1
///   ...
///   dK0 dK1 dK2 ... dKN // rowK
///   ```
///   where `dAB` is the distance between then `A`th chunk of the query vector and the `B`th
///   centroid for that chunk.
/// * num_centers: The dimension `N + 1` in the table above.
///
/// Currently, the following invariants must hold (checked in debug builds):
///
/// * `K` == 256 (there are exactly 256 centroids per chunk)
/// * `K * pq_coordinates.len() == precomputed_distances.len()`.
pub fn pq_dist_lookup_single(
    pq_coordinates: &[u8],
    precomputed_distances: &[f32],
    num_centers: usize,
) -> f32 {
    let num_pq_chunks = pq_coordinates.len();
    debug_assert_eq!(precomputed_distances.len(), num_centers * num_pq_chunks);
    let mut accum: f32 = 0.0;
    let iter = std::iter::zip(
        pq_coordinates.iter(),
        precomputed_distances.chunks(num_centers),
    );
    for (&value, distances) in iter {
        accum += distances[value.into_usize()];
    }
    accum
}

impl FixedChunkPQTable {
    /// Create a new `FixedChunkPQTable` for the provided PQ schema.
    ///
    /// # Parameters
    ///
    /// * `dim`: The number of dimension in the full precision data for this dataset.
    ///
    /// * `pq_table`: The raw pivot data for the PQ schema. This slice underlying this
    ///   representation must have a length exactly divisible by `dim`. The number of
    ///   pivots per chunk is this quotient.
    ///
    ///   Refer to the later section for the expected layout of this table.
    ///
    /// * `centroids`: The dimension-wise mean of the training data. The slice underlying
    ///   this representation must have length `dim`.
    ///
    /// * `chunk_offsets`: A vector marking the beginning and end of each chunk. That is,
    ///   the offsets of the start of chunk `i` is `chunk_offsets[i]` and the end is
    ///   `chunk_offsets[i+1]`.
    ///
    ///   The underlying slice must have the following properties:
    ///
    ///   1. Strict monotonicity. Values must be strictly increasing.
    ///   2. `chunk_offsets.len() >= 2`: There must be at least one start/end pair.
    ///   3. `chunk_offsets[0] == 0`: It must begin at 0.
    ///   4. `chunk_offsets.last().unwrap() == dim`: The last offset must match the
    ///      dimension of the full-precision data.
    ///
    /// * `opq_rotation_matrix`: An optional rotation matrix to apply to queries.
    ///   If given, the underlying slice must have length `dim * dim`.
    ///
    ///   NOTE: This feature is currently not fully supported.
    ///
    /// # PQ Table Layout
    ///
    /// The in-memory layout of the `pq_table` is shown in the table below in row-major form.
    /// ```text
    ///           | -- chunk 0 -- | -- chunk 1 -- | -- chunk 2 -- | .... | -- chunk N-1 -- |
    ///           +------------------------------------------------------------------------+
    ///  pivot 0  | c000 c001 ... | c010 c011 ... | c020 c021 ... | .... |       ...       |
    ///  pivot 1  | c100 c101 ... | c110 c111 ... | c120 c121 ... | .... |       ...       |
    ///    ...    |      ...      |      ...      |      ...      | .... |       ...       |
    ///  pivot K  | cK00 cK01 ... | cK10 cK11 ... | cK20 cK21 ... | .... |       ...       |
    /// ```
    pub fn new(
        dim: usize,
        pq_table: Box<[f32]>,
        centroids: Box<[f32]>,
        chunk_offsets: Box<[usize]>,
        opq_rotation_matrix: Option<Box<[f32]>>,
    ) -> ANNResult<Self> {
        let len = pq_table.len();
        let table = BasicTable::new(
            MatrixBase::try_from(pq_table, len / dim, dim).bridge_err()?,
            ChunkOffsetsBase::new(chunk_offsets).bridge_err()?,
        )
        .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?;

        if centroids.len() != dim {
            return Err(ANNError::log_pq_error(format_args!(
                "centroids slice has length {} but the expected dim is {}",
                centroids.len(),
                dim
            )));
        }

        if let Some(matrix) = opq_rotation_matrix.as_ref()
            && matrix.len() != dim * dim
        {
            return Err(ANNError::log_pq_error(format_args!(
                "opq rotation matrix should have length {}, instead is is {}",
                dim * dim,
                matrix.len()
            )));
        }

        Ok(Self {
            table,
            centroids,
            opq_rotation_matrix,
        })
    }

    /// Get chunk number.
    pub fn get_num_chunks(&self) -> usize {
        self.table.nchunks()
    }

    /// Shifting the query according to mean or the whole corpus. The output is a rotated query vector,
    /// which is later used to calculate the distance between each query chunk and each centroid using populate_chunk_distances.
    pub fn preprocess_query(&self, rotated_query_vec: &mut [f32]) {
        for (query, &centroid) in rotated_query_vec.iter_mut().zip(self.centroids.iter()) {
            *query -= centroid;
        }

        if let Some(rotation_matrix) = &self.opq_rotation_matrix {
            let read_only: &[f32] = rotated_query_vec;

            let read_dimension = self.get_dim();
            let mut temp_result = vec![0.0; self.get_dim()];
            // Multiply matrix 'rotated_query_vec' by matrix 'rotation_matrix'
            diskann_linalg::sgemm(
                Transpose::None,  // Do not transpose matrix 'a'
                Transpose::None,  // Do not transpose matrix 'b'
                1,                // m (number of rows in matrices 'a' and 'c')
                read_dimension,   // n (number of columns in matrices 'b' and 'c')
                read_dimension, // k (number of columns in matrix 'a', number of rows in matrix 'b')
                1.0,            // alpha (scaling factor for the product of matrices 'a' and 'b')
                read_only,      // matrix 'a'
                rotation_matrix, // matrix 'b'
                None,           // beta (scaling factor for matrix 'c')
                &mut temp_result, // matrix 'c' (result matrix)
            );

            rotated_query_vec[0..self.get_dim()].copy_from_slice(&temp_result);
        }
    }

    pub fn populate_chunk_distances_impl<T>(
        &self,
        rotated_query_vec: &[f32],
        aligned_pq_table_dist_scratch: &mut [f32],
    ) -> ANNResult<()>
    where
        T: for<'a, 'b> PureDistanceFunction<&'a [f32], &'b [f32]> + Default,
    {
        let num_centers = self.get_num_centers();
        let num_chunks = self.get_num_chunks();
        if aligned_pq_table_dist_scratch.len() < num_chunks * num_centers {
            return Err(ANNError::log_pq_error(
                "aligned_pq_table_dist_scratch.len() should at least be num_pq_chunks * num_centers",
            ));
        }

        let offsets: &[usize] = self.table.view_offsets().into();
        let table: &[f32] = self.table.view_pivots().into();
        let dim = self.get_dim();

        for centroid_index in 0..num_centers {
            let table_start = dim * centroid_index;
            for chunk_index in 0..num_chunks {
                let start = offsets[chunk_index];
                let stop = offsets[chunk_index + 1];

                let query = &rotated_query_vec[start..stop];
                let chunk = &table[(table_start + start)..(table_start + stop)];
                aligned_pq_table_dist_scratch[chunk_index * num_centers + centroid_index] =
                    T::evaluate(query, chunk);
            }
        }

        Ok(())
    }

    /// Pre-calculated the distance between each chunk in the query vector and each centroid
    /// by l2 distance.
    /// * `rotated_query_vec` - query vector: 1 * dim
    /// * `aligned_pq_table_dist_scratch` - pre-calculated the distance between query and
    ///   each centroid: chunk_size * num_centroids
    pub fn populate_chunk_distances(
        &self,
        rotated_query_vec: &[f32],
        aligned_pq_table_dist_scratch: &mut [f32],
    ) -> ANNResult<()> {
        self.populate_chunk_distances_impl::<distance::SquaredL2>(
            rotated_query_vec,
            aligned_pq_table_dist_scratch,
        )
    }

    /// Pre-calculated the distance between query and each centroid by inner product
    /// * `query_vec` - query vector: 1 * dim
    /// * `aligned_pq_table_dist_scratch` - pre-calculated the distance between query and
    ///   each centroid: chunk_size * num_centroids
    pub fn populate_chunk_inner_products(
        &self,
        query_vec: &[f32],
        aligned_pq_table_dist_scratch: &mut [f32],
    ) -> ANNResult<()> {
        self.populate_chunk_distances_impl::<distance::InnerProduct>(
            query_vec,
            aligned_pq_table_dist_scratch,
        )
    }

    /// Calculate the distance between query and given centroid by l2 distance
    /// * `query_vec` - query vector: 1 * dim
    /// * `base_vec` - quantized vector of size 1 * num_pq_chunks
    pub fn l2_distance(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        direct_distance_impl::<distance::simd::ResumableL2<diskann_wide::arch::Current>>(
            self.table.view_pivots().as_slice(),
            self.table.view_offsets().as_slice(),
            self.get_dim(),
            query_vec,
            base_vec,
        )
    }

    /// Calculate the distance between query and given centroid by cosine distance
    /// * `query_vec` - query vector: 1 * dim
    /// * `base_vec` - given centroid array: 1 * num_pq_chunks
    pub fn cosine_distance(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        // The SIMD kernel guarantees output in the range `[-1.0, 1.0]`, so this conversion
        // will be in `[0, 2]`.
        1.0 - direct_distance_impl::<distance::simd::ResumableCosine<diskann_wide::arch::Current>>(
            self.table.view_pivots().as_slice(),
            self.table.view_offsets().as_slice(),
            self.get_dim(),
            query_vec,
            base_vec,
        )
    }

    /// Calculate the distance between query and given centroid by cosine distance
    /// * `query_vec` - query vector: 1 * dim
    /// * `base_vec` - given centroid array: 1 * num_pq_chunks
    pub fn cosine_normalized_distance(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        self.cosine_distance(query_vec, base_vec)
    }

    /// Calculate the distance between query and given centroid by inner product
    /// * `query_vec` - query vector: 1 * dim
    /// * `base_vec` - given centroid array: 1 * num_pq_chunks
    pub fn inner_product_raw(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        direct_distance_impl::<distance::simd::ResumableIP<diskann_wide::arch::Current>>(
            self.table.view_pivots().as_slice(),
            self.table.view_offsets().as_slice(),
            self.get_dim(),
            query_vec,
            base_vec,
        )
    }

    /// Calculate the distance between query and given centroid by inner product
    ///
    /// # Parameters
    ///
    /// * `query_vec` - query vector: 1 * dim
    /// * `base_vec` - given centroid array: 1 * num_pq_chunks
    ///
    /// # Returns
    ///
    /// Returns negative value to simulate distances (max -> min conversion)
    pub fn inner_product(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        let res = self.inner_product_raw(query_vec, base_vec);
        -res
    }

    // Apply a resumable distance function between the PQ pivots pointed to the the left
    // and right hand compressed vectors.
    fn self_distance<T>(&self, left: &[u8], right: &[u8]) -> f32
    where
        T: distance::simd::ResumableSIMDSchema<f32, f32, FinalReturn = f32>,
    {
        assert_eq!(
            left.len(),
            self.get_num_chunks(),
            "pq vector must have length {}",
            self.get_num_chunks()
        );
        assert_eq!(
            right.len(),
            self.get_num_chunks(),
            "pq vector must have length {}",
            self.get_num_chunks()
        );

        let mut accumulator = distance::simd::Resumable::new(T::init(ARCH));

        let pq_table: &[f32] = self.table.view_pivots().into();
        let chunk_offsets: &[usize] = self.table.view_offsets().into();

        let mut start = chunk_offsets[0];
        let dim = self.get_dim();
        (0..self.get_num_chunks()).for_each(|chunk_index| {
            let stop = chunk_offsets[chunk_index + 1];

            let make_range = |offset: usize| (dim * offset + start)..(dim * offset + stop);

            let left_offset: usize = left[chunk_index].into();
            let right_offset: usize = right[chunk_index].into();

            let left_slice = &pq_table[make_range(left_offset)];
            let right_slice = &pq_table[make_range(right_offset)];

            accumulator = distance::simd::simd_op(&accumulator, ARCH, left_slice, right_slice);
            start = stop;
        });
        accumulator.consume().sum()
    }

    /// Compute the square L2 distance between two compressed vectors that use the same
    /// pivot table.
    ///
    /// Requires `left.len() == right.len()`.
    ///
    /// This function yields valid results both when zero centering is used and when it
    /// is not used.
    pub fn qq_l2_distance(&self, left: &[u8], right: &[u8]) -> f32 {
        self.self_distance::<distance::simd::ResumableL2<diskann_wide::arch::Current>>(left, right)
    }

    /// Compute the inner product between two compressed vectors that use the same
    /// pivot table.
    ///
    /// NOTE: This function returns the negated inner product as is common throughout the
    /// code base. This implies that **lower** values have **higher** similarity.
    ///
    /// Requires `left.len() == right.len()`.
    ///
    /// This function yields valid results only when zero centering is *NOT* used.
    pub fn qq_ip_distance(&self, left: &[u8], right: &[u8]) -> f32 {
        -self.self_distance::<distance::simd::ResumableIP<diskann_wide::arch::Current>>(left, right)
    }

    /// Compute the cosine similarity between two compressed vectors that use the same
    /// pivot table.
    ///
    /// NOTE: This function applies the transformation `1.0 - cosine_similarity` to yield
    /// a result between 0 and 2. This implies that **lower** values have **higher**
    /// similarity.
    ///
    /// Requires `left.len() == right.len()`.
    ///
    /// This function yields valid results only when zero centering is *NOT* used.
    pub fn qq_cosine_distance(&self, left: &[u8], right: &[u8]) -> f32 {
        1.0 - self.self_distance::<distance::simd::ResumableCosine<diskann_wide::arch::Current>>(
            left, right,
        )
    }

    // Miscellaneous helper methods.

    /// Revert vector by adding centroid
    /// * `base_vec` - given centroid array: 1 * num_pq_chunks
    /// * `out_vec` - reverted vector
    ///
    /// # Panics
    ///
    /// Panics under the following condition:
    /// * `base_vec.length() != self.get_dim()`.
    /// * Any entry in `base_vec` exceeds `self.get_centroids()`.
    pub fn inflate_vector(&self, base_vec: &[u8]) -> Vec<f32> {
        let mut out_vec: Vec<f32> = vec![0.0; self.get_dim()];
        self.inflate_vector_into(base_vec, &mut out_vec);
        out_vec
    }

    /// Same as [`Self::inflate_vector`], but accepts an output buffer instead of
    /// returning a fresh vector.
    pub fn inflate_vector_into(&self, base_vec: &[u8], out: &mut [f32]) {
        assert_eq!(base_vec.len(), self.get_num_chunks());
        assert_eq!(out.len(), self.get_dim());
        let chunk_offsets: &[usize] = self.table.view_offsets().into();
        let pq_table: &[f32] = self.table.view_pivots().into();
        let dim = self.get_dim();

        base_vec.iter().enumerate().for_each(|(i, b)| {
            let b = b.into_usize();
            let start = chunk_offsets[i];
            let stop = chunk_offsets[i + 1];
            let out_slice = &mut out[start..stop];
            let pivot = &pq_table[(dim * b + start)..(dim * b + stop)];
            let centroid = &self.centroids[start..stop];
            std::iter::zip(out_slice.iter_mut(), pivot.iter())
                .zip(centroid.iter())
                .for_each(|((o, p), c)| *o = *p + *c);
        });
    }

    /// Return the number of centers for each chunk.
    pub fn get_num_centers(&self) -> usize {
        self.table.ncenters()
    }

    /// Returns an immutable reference to the `pq_table`.
    pub fn get_pq_table(&self) -> &[f32] {
        self.table.view_pivots().into()
    }

    /// Returns an immutable reference to the `chunk_offsets`.
    pub fn get_chunk_offsets(&self) -> &[usize] {
        self.table.view_offsets().into()
    }

    /// Returns an immutable reference to the `centroids`.
    pub fn get_centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Returns the original dimension of the vectors.
    pub fn get_dim(&self) -> usize {
        self.table.dim()
    }

    /// Return whether or not this table is configured with OPQ.
    pub fn has_opq(&self) -> bool {
        self.opq_rotation_matrix.is_some()
    }

    /// Return the pivots as a `MatrixView`.
    pub fn view_pivots(&self) -> views::MatrixView<'_, f32> {
        self.table.view_pivots()
    }

    /// Return the chunk offsets as a `ChunkOffsetesView`.
    pub fn view_offsets(&self) -> diskann_quantization::views::ChunkOffsetsView<'_> {
        self.table.view_offsets()
    }
}

// This goes against Rust's Orphan rule, so we cannot implement it directly.
// However, we can use a wrapper type to implement the conversion.
// This is a workaround to allow the conversion from `product::TableCompressionError` to
// `ANNError` without violating the orphan rule.
impl From<Bridge<product::TableCompressionError>> for ANNError {
    fn from(value: Bridge<product::TableCompressionError>) -> ANNError {
        ANNError::log_pq_error(diskann_quantization::error::format(&value.into_inner()))
    }
}

impl<T> CompressInto<&[T], &mut [u8]> for FixedChunkPQTable
where
    T: Into<f32> + Copy,
{
    type Error = product::TableCompressionError;
    type Output = ();

    /// Perform PQ compression on `from` into `to`.
    ///
    /// Internally, this calls [`diskann_quantization::product::BasicTable::compress_into`].
    /// See the documentation for that method about the failure modes for this function.
    fn compress_into(&self, from: &[T], to: &mut [u8]) -> Result<(), Self::Error> {
        let translated: Vec<f32> = std::iter::zip(from.iter(), self.centroids.iter())
            .map(|(f, c)| {
                let f: f32 = (*f).into();
                f - *c
            })
            .collect();
        self.table.compress_into(&*translated, to)
    }
}

/// Given a slice of pq compressed vectors, fill the list of distances between the compressed pq vector to the query vector
/// in `dists_out` using the pre-calculated distances in `pq_dists`.
/// * `pq_coordinates` - batch nodes: n_pts * pq_nchunks
/// * `n_pts` - batch size
/// * `pq_nchunks` - number of pq chunks
/// * `pq_dists` - pre-calculated the distance between query and each centroid: chunk_size * num_centroids
/// * `dists_out` - ideally of size: n_pts, but for disk search usage - it can be larger than n_pts as well.
///
/// # Note to Maintainers
///
/// This function contains unsafe code but is safe to call. If you edit this function,
/// be **very** careful you keep the required invariants.
fn pq_dist_lookup(
    pq_coordinates: &[u8],
    n_pts: usize,
    pq_nchunks: usize,
    pq_dists: &[f32],
    dists_out: &mut [f32],
) -> ANNResult<()> {
    // Post-Monomorphization error checking.
    // This is a compile-time check that the number of centroids is 256.
    const {
        assert!(
            NUM_PQ_CENTROIDS == 256,
            "Global constant \"NUM_PQ_CENTROIDS\" must be 256 for safety requirements to hold"
        );
    }

    let coordinates = MatrixView::<u8>::try_from(pq_coordinates, n_pts, pq_nchunks).bridge_err()?;
    let distances = MatrixView::try_from(
        &pq_dists[..NUM_PQ_CENTROIDS * pq_nchunks],
        pq_nchunks,
        NUM_PQ_CENTROIDS,
    )
    .bridge_err()?;

    let dists_out = match dists_out.get_mut(..n_pts) {
        None => {
            return Err(ANNError::log_pq_error(format_args!(
                "ERROR: dists_out length: {} is less than n_pts: {}",
                dists_out.len(),
                n_pts
            )));
        }
        Some(slice) => {
            slice.fill(0.0);
            slice
        }
    };

    // Size of tile used for
    // [tiling optimization](https://www.intel.com/content/www/us/en/developer/articles/technical/efficient-use-of-tiling.html).
    // The tile size is chosen such that 16 * 256 * 4 = 16KB,
    // which fits well within typical L1 cache sizes, leaving room for other data.
    const TILE_SIZE: usize = 16;

    let full_tiles = distances.nrows() / TILE_SIZE;

    for tile in 0..full_tiles {
        unsafe {
            // SAFETY: It's safe to use `tile` as an `tile_index` since it's less than `full_tiles`
            // and each tile has a full size of `TILE_SIZE`.
            add_distance_for_a_tile(
                tile,
                TILE_SIZE,
                TILE_SIZE,
                dists_out,
                coordinates,
                distances,
            )
        };
    }

    let remainder = distances.nrows() - TILE_SIZE * full_tiles;
    if remainder != 0 {
        unsafe {
            // SAFETY: It's safe to use `full_tiles` as the `tile_index` for the last tile
            // with the tile size of `remainder`.
            add_distance_for_a_tile(
                full_tiles,
                TILE_SIZE,
                remainder,
                dists_out,
                coordinates,
                distances,
            )
        };
    }

    Ok(())
}

/// Given a tile index, this function calculates the distance between the query and each vector in the tile.
/// The result is directly added to the distance vector `dists_out`.
/// * `tile_idx` - index of the tile
/// * `tile_size` - size of the tile
/// * `cur_tile_size` - size of the current tile
/// * `dists_out` - output distance vector
/// * `coordinates` - batch nodes: n_pts * pq_nchunks
/// * `distances` - pre-calculated the distance between query and each centroid: chunk_size * num_centroids
///
/// # Safety
///
/// This function contains unsafe code because it's relying on the inputs like `tile_idx` and `tile_size`
/// to be safe with-respect to `coordinates` and `distances`. The following invariants must hold:
/// * `tile_idx` is less than equal to `distances.nrows() / tile_size`
/// * `tile_idx*tile_size+cur_tile_size-1` is less than `distances.nrows()`
/// * `distances.ncols()` is equal to `NUM_PQ_CENTROIDS`(256)
/// * `distances.nrows()` is equal to `coordinates.ncols()`
/// * `dists_out.len()` is equal to `coordinates.nrows()`
///
/// If you are using this function, be **very** careful to do argument validation as done in `pq_dist_lookup` function.
#[inline(always)]
unsafe fn add_distance_for_a_tile(
    tile_idx: usize,
    tile_size: usize,
    cur_tile_size: usize,
    dists_out: &mut [f32],
    coordinates: MatrixView<'_, u8>,
    distances: MatrixView<'_, f32>,
) {
    dists_out.iter_mut().enumerate().for_each(|(point, d)| {
        for offset in 0..cur_tile_size {
            let chunk = tile_idx * tile_size + offset;
            // SAFETY: It's safe to query `coordinates` with `point` and `chunk` since
            // `point` is less than `n_pts`(`coordinates.nrows()`) and
            // `chunk` is less than `pq_nchunks`(`coordinates.ncols()`)
            //  as validated in `pq_dist_lookup` function.
            let centroid: u8 = unsafe { *coordinates.get_unchecked(point, chunk) };

            // SAFETY: From above, `chunk` is less than `coordinatges.ncols()`, which must
            // be equal to `distances.nrows()` by the pre-conditions for this function.
            let row = unsafe { distances.get_row_unchecked(chunk) };

            // SAFETY: It's safe to query `row` with `centroid` since
            // it's less than 256(`NUM_PQ_CENTROIDS`) given that it's a u8.
            *d += unsafe { row.get_unchecked(centroid as usize) };
        }
    });
}

// Given a batch input vertex ids, this function calculates the coordinates of each vertex in the pq centroid table.
// i.e, find the closest centroid for each chunk in each vertex, and use the id of the closest centroid as a coordinate on that dimension.
fn aggregate_coords(
    ids: &[u32],
    all_coords: &[u8],
    num_pq_chunks: usize,
    pq_coordinate_scratch: &mut [u8],
) -> ANNResult<()> {
    if pq_coordinate_scratch.len() < ids.len() * num_pq_chunks {
        return Err(ANNError::log_pq_error(format_args!(
            "pq_coordinate_scratch doesn't have enough length. It has length {} but requires length {}",
            pq_coordinate_scratch.len(),
            ids.len() * num_pq_chunks
        )));
    }

    pq_coordinate_scratch[0..num_pq_chunks * ids.len()]
        .chunks_mut(num_pq_chunks)
        .enumerate()
        .for_each(|(index, chunk)| {
            let id_compressed_pivot = &all_coords[(ids[index] as usize * num_pq_chunks)
                ..(ids[index] as usize * num_pq_chunks + num_pq_chunks)];
            let temp_slice =
                unsafe { std::slice::from_raw_parts(id_compressed_pivot.as_ptr(), num_pq_chunks) };
            chunk.copy_from_slice(temp_slice);
        });

    Ok(())
}

// Compute the pq distance between the query and each vector in vector_ids.
// The pq distance is computed by first aggregating the pq coordinates for each vector in vector_ids, then do a pq lookup with the computed coordinates.
// The result is stored in pq_distance_scratch.
pub fn compute_pq_distance(
    vector_ids: &[u32],
    num_pq_chunks: usize,
    query_centroid_l2_distance: &[f32],
    pq_data: &[u8],
    pq_coordinate_scratch: &mut [u8],
    pq_distance_scratch: &mut [f32],
) -> ANNResult<()> {
    aggregate_coords(vector_ids, pq_data, num_pq_chunks, pq_coordinate_scratch)?;

    pq_dist_lookup(
        &pq_coordinate_scratch[..vector_ids.len() * num_pq_chunks],
        vector_ids.len(),
        num_pq_chunks,
        query_centroid_l2_distance,
        pq_distance_scratch,
    )?;

    Ok(())
}

/// Compute the pq distance between the query and the given single pq's coordinates.
pub fn compute_pq_distance_for_pq_coordinates(
    pq_coordinates: &[u8],
    num_pq_chunks: usize,
    query_centroid_l2_distance: &[f32],
    pq_distance_scratch: &mut [f32],
) -> ANNResult<()> {
    pq_dist_lookup(
        pq_coordinates,
        1, // n_pts = 1 since we are only computing distance for a single pq coordinate
        num_pq_chunks,
        query_centroid_l2_distance,
        pq_distance_scratch,
    )?;

    Ok(())
}

#[cfg(test)]
mod fixed_chunk_pq_table_test {
    use core::ops::Range;

    use crate::storage::{StorageReadProvider, VirtualStorageProvider};
    use approx::assert_relative_eq;
    use diskann_vector::{
        PureDistanceFunction,
        distance::{InnerProduct, SquaredL2},
    };
    use itertools::iproduct;

    use super::*;
    use crate::{
        common::AlignedBoxWithSlice,
        model::{NUM_PQ_CENTROIDS, pq::convert_types},
        utils::{file_exists, load_bin},
    };

    const DIM: usize = 128;

    #[test]
    fn constructor_errors() {
        // Test that we verify all the requirements in the constructor.
        type PreSchema = (usize, Box<[f32]>, Box<[f32]>, Box<[usize]>, Box<[f32]>);
        fn create_valid_schema() -> PreSchema {
            let dim = 5;
            (
                dim,
                vec![0.0; dim * 4].into(),
                vec![0.0; dim].into(),
                Box::new([0, 2, 3, dim]),
                vec![0.0; dim * dim].into(),
            )
        }

        // Check that our valid schema is indeed valid.
        {
            let (dim, pq_table, centroids, chunk_offsets, opq) = create_valid_schema();
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_ok()
            );
        }

        // `pq_table` length not evenly divisible by `dim`..
        {
            let (dim, _, centroids, chunk_offsets, opq) = create_valid_schema();
            let pq_table = vec![0.0; dim * 3 + 1].into();
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_err()
            );
        }

        // `centroids` length not equal to `dim`..
        {
            let (dim, pq_table, _, chunk_offsets, opq) = create_valid_schema();
            let centroids = vec![0.0; dim - 1].into();
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_err()
            );
        }

        // `offsets` does not begin at zero.
        {
            let (dim, pq_table, centroids, _, opq) = create_valid_schema();
            let chunk_offsets = Box::new([1, 2, dim]);
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_err()
            );
        }

        // `offsets` empty
        {
            let (dim, pq_table, centroids, _, opq) = create_valid_schema();
            let chunk_offsets = Box::new([]);
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_err()
            );
        }

        // `offsets` has length 1.
        {
            let (dim, pq_table, centroids, _, opq) = create_valid_schema();
            let chunk_offsets = Box::new([0]);
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_err()
            );
        }

        // `offsets` not strictly monotonic.
        {
            let (dim, pq_table, centroids, _, opq) = create_valid_schema();
            let chunk_offsets = Box::new([0, 1, 2, 2, dim]);
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_err()
            );
        }

        // `offsets` does not end at `dim`.
        {
            let (dim, pq_table, centroids, _, opq) = create_valid_schema();
            let chunk_offsets = Box::new([0, 1, 2, dim, dim + 1]);
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_err()
            );
        }

        // `opq` has the wrong length.
        {
            let (dim, pq_table, centroids, chunk_offsets, _) = create_valid_schema();
            let opq = vec![0.0; dim].into();
            assert!(
                FixedChunkPQTable::new(dim, pq_table, centroids, chunk_offsets, Some(opq)).is_err()
            );
        }
    }

    #[test]
    fn test_compute_pq_distance() {
        let num_pq_chunks = 17;
        let n_pts: usize = 10;
        let n_nbrs = 9;
        // mock neighbor index
        let neighbor_vector_ids: Vec<u32> = vec![3, 1, 5, 7, 6, 9, 6, 8, 2];

        // mock query_centroid_l2_distance, distance from query to each centroid `i` of chunk `j` as `j*NUM_PQ_CENTROIDS + i` for each chunk, just for simple calculation.
        let mut query_centroid_l2_distance =
            AlignedBoxWithSlice::new(NUM_PQ_CENTROIDS * num_pq_chunks, 256).unwrap();
        let distance_vec = (0..NUM_PQ_CENTROIDS * num_pq_chunks)
            .map(|i| i as f32)
            .collect::<Vec<f32>>();
        query_centroid_l2_distance.memcpy(&distance_vec).unwrap();

        // random nums, mock pq table, size = 17 * 10 = num_pq_chunks * n_pts
        let pq_data: Vec<u8> = vec![
            53, 88, 93, 231, 52, 96, 226, 207, 162, 177, 5, 76, 147, 20, 229, 0, 83, 252, 156, 52,
            141, 37, 242, 156, 136, 28, 205, 191, 96, 202, 120, 170, 170, 224, 127, 94, 241, 179,
            235, 223, 157, 45, 149, 185, 111, 141, 232, 68, 54, 104, 28, 191, 44, 244, 79, 15, 57,
            228, 66, 250, 211, 20, 152, 184, 12, 54, 197, 69, 143, 139, 71, 20, 180, 101, 210, 228,
            113, 98, 157, 16, 230, 24, 252, 49, 245, 24, 255, 44, 204, 92, 25, 136, 169, 22, 220,
            55, 109, 176, 175, 39, 199, 122, 3, 42, 54, 31, 92, 155, 194, 225, 23, 92, 225, 215,
            161, 36, 251, 139, 48, 228, 235, 247, 28, 151, 65, 58, 255, 238, 44, 149, 19, 121, 14,
            199, 72, 96, 37, 128, 238, 201, 162, 167, 235, 16, 91, 148, 227, 170, 208, 250, 19,
            186, 22, 141, 61, 188, 245, 23, 3, 95, 134, 192, 10, 188, 29, 232, 40, 222, 248, 24,
        ];

        let mut pq_distance_scratch: Vec<f32> = vec![0.0; n_nbrs];
        let mut pq_coordinate_scratch: Vec<u8> = vec![0; num_pq_chunks * neighbor_vector_ids.len()];

        // Call the function being tested
        compute_pq_distance(
            &neighbor_vector_ids,
            num_pq_chunks,
            &query_centroid_l2_distance,
            &pq_data,
            &mut pq_coordinate_scratch,
            &mut pq_distance_scratch,
        )
        .unwrap();

        // Calculate the expected output naively
        let pq_data = MatrixView::try_from(&pq_data, n_pts, num_pq_chunks).unwrap();
        let distances =
            MatrixView::try_from(&query_centroid_l2_distance, num_pq_chunks, NUM_PQ_CENTROIDS)
                .unwrap();
        let mut expected_pd_distance = vec![0.0; n_nbrs];
        expected_pd_distance
            .iter_mut()
            .enumerate()
            .for_each(|(i, d)| {
                for chunk in 0..num_pq_chunks {
                    let pq_coord = pq_data[(neighbor_vector_ids[i] as usize, chunk)];
                    *d += distances[(chunk, pq_coord as usize)];
                }
            });

        // Assert that the output is correct
        assert_eq!(pq_distance_scratch.len(), n_nbrs);
        assert_eq!(pq_distance_scratch, expected_pd_distance);
    }

    #[test]
    fn load_pivot_test() {
        let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let storage_provider = VirtualStorageProvider::new_overlay(workspace_root);
        let pq_pivots_path: &str = "/test_data/sift/siftsmall_learn_pq_pivots.bin";
        let (dim, pq_table, centroids, chunk_offsets) =
            load_pq_pivots_bin(pq_pivots_path, &1, &storage_provider).unwrap();
        let fixed_chunk_pq_table = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            None,
        )
        .unwrap();

        assert_eq!(dim, DIM);
        assert_eq!(fixed_chunk_pq_table.table.dim(), DIM);
        assert_eq!(fixed_chunk_pq_table.table.ncenters(), NUM_PQ_CENTROIDS);
        assert_eq!(fixed_chunk_pq_table.centroids.len(), DIM);

        assert_eq!(fixed_chunk_pq_table.get_chunk_offsets(), &[0, DIM]);
    }

    #[test]
    fn clone_pivot_table() {
        let dim = 128;
        let num_pq_centroids = 4;
        let pq_table = vec![1.0; dim * num_pq_centroids];
        let centroids = vec![1.0; dim];
        let chunk_offsets = vec![0, 7, 9, 11, 22, 34, 78, dim];

        let base = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            None,
        )
        .unwrap();

        let clone = base.clone();
        let FixedChunkPQTable {
            table,
            centroids,
            opq_rotation_matrix,
        } = clone;

        assert_eq!(table.view_pivots(), base.table.view_pivots());
        assert_eq!(table.view_offsets(), base.table.view_offsets());
        assert_eq!(centroids, base.centroids);
        assert_eq!(opq_rotation_matrix, base.opq_rotation_matrix);
    }

    #[test]
    fn get_num_chunks_test() {
        let num_chunks = 7;
        let pa_table = vec![0.0; DIM * NUM_PQ_CENTROIDS];
        let centroids = vec![0.0; DIM];
        let chunk_offsets = vec![0, 7, 9, 11, 22, 34, 78, 128];
        let fixed_chunk_pq_table = FixedChunkPQTable::new(
            DIM,
            pa_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            None,
        )
        .unwrap();
        let chunk: usize = fixed_chunk_pq_table.get_num_chunks();
        assert_eq!(chunk, num_chunks);
    }

    #[test]
    fn preprocess_query_test() {
        let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let storage_provider = VirtualStorageProvider::new_overlay(workspace_root);

        let pq_pivots_path: &str = "/test_data/sift/siftsmall_learn_pq_pivots.bin";
        let (dim, pq_table, centroids, chunk_offsets) =
            load_pq_pivots_bin(pq_pivots_path, &1, &storage_provider).unwrap();
        let fixed_chunk_pq_table = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            None,
        )
        .unwrap();

        let mut query_vec: Vec<f32> = vec![
            32.39f32, 78.57f32, 50.32f32, 80.46f32, 6.47f32, 69.76f32, 94.2f32, 83.36f32, 5.8f32,
            68.78f32, 42.32f32, 61.77f32, 90.26f32, 60.41f32, 3.86f32, 61.21f32, 16.6f32, 54.46f32,
            7.29f32, 54.24f32, 92.49f32, 30.18f32, 65.36f32, 99.09f32, 3.8f32, 36.4f32, 86.72f32,
            65.18f32, 29.87f32, 62.21f32, 58.32f32, 43.23f32, 94.3f32, 79.61f32, 39.67f32,
            11.18f32, 48.88f32, 38.19f32, 93.95f32, 10.46f32, 36.7f32, 14.75f32, 81.64f32,
            59.18f32, 99.03f32, 74.23f32, 1.26f32, 82.69f32, 35.7f32, 38.39f32, 46.17f32, 64.75f32,
            7.15f32, 36.55f32, 77.32f32, 18.65f32, 32.8f32, 74.84f32, 18.12f32, 20.19f32, 70.06f32,
            48.37f32, 40.18f32, 45.69f32, 88.3f32, 39.15f32, 60.97f32, 71.29f32, 61.79f32,
            47.23f32, 94.71f32, 58.04f32, 52.4f32, 34.66f32, 59.1f32, 47.11f32, 30.2f32, 58.72f32,
            74.35f32, 83.68f32, 66.8f32, 28.57f32, 29.45f32, 52.02f32, 91.95f32, 92.44f32,
            65.25f32, 38.3f32, 35.6f32, 41.67f32, 91.33f32, 76.81f32, 74.88f32, 33.17f32, 48.36f32,
            41.42f32, 23f32, 8.31f32, 81.69f32, 80.08f32, 50.55f32, 54.46f32, 23.79f32, 43.46f32,
            84.5f32, 10.42f32, 29.51f32, 19.73f32, 46.48f32, 35.01f32, 52.3f32, 66.97f32, 4.8f32,
            74.81f32, 2.82f32, 61.82f32, 25.06f32, 17.3f32, 17.29f32, 63.2f32, 64.1f32, 61.68f32,
            37.42f32, 3.39f32, 97.45f32, 5.32f32, 59.02f32, 35.6f32,
        ];
        fixed_chunk_pq_table.preprocess_query(&mut query_vec);
        assert_eq!(query_vec[0], 32.39f32 - fixed_chunk_pq_table.centroids[0]);
        assert_eq!(
            query_vec[127],
            35.6f32 - fixed_chunk_pq_table.centroids[127]
        );
    }

    #[test]
    fn preprocess_query_with_opq_test() {
        let dim = 10;
        let mut opq_rotation_matrix = Vec::with_capacity(100);
        for item in 0..100 {
            opq_rotation_matrix.push(item as f32 / 10.0);
        }
        let centroids = vec![1.0; 10];
        let chunk_offsets = vec![0, 10]; // one chunk

        // pq_table is not needed for the preprocess_query method
        let pq_table = vec![0.0; dim];

        let fixed_chunk_pq_table = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            Some(opq_rotation_matrix.into()),
        )
        .unwrap();

        let mut query_vec: Vec<f32> = vec![
            1.111f32,
            2.222f32,
            3.333f32,
            4.444f32,
            5.555f32,
            6.666f32,
            7.777f32,
            8.888f32,
            9.999f32,
            10.10101f32,
        ];

        fixed_chunk_pq_table.preprocess_query(&mut query_vec);

        // Round to four decimal places.  Different computers get slightly different results
        // after four decimal places so rounding makes the comparison easy
        let rounded_query_vec: Vec<f32> = query_vec
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect();

        let expected_result = vec![
            312.5491, 317.5587, 322.5683, 327.5779, 332.5875, 337.5971, 342.6067, 347.6163,
            352.6259, 357.6355,
        ];
        assert_eq!(
            rounded_query_vec, expected_result,
            "Actual result did not match either expected result"
        );
    }

    #[test]
    fn calculate_distances_tests() {
        let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let storage_provider = VirtualStorageProvider::new_overlay(workspace_root);

        let pq_pivots_path: &str = "/test_data/sift/siftsmall_learn_pq_pivots.bin";

        let (dim, pq_table, centroids, chunk_offsets) =
            load_pq_pivots_bin(pq_pivots_path, &1, &storage_provider).unwrap();
        let fixed_chunk_pq_table = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            None,
        )
        .unwrap();

        let query_vec: Vec<f32> = vec![
            32.39f32, 78.57f32, 50.32f32, 80.46f32, 6.47f32, 69.76f32, 94.2f32, 83.36f32, 5.8f32,
            68.78f32, 42.32f32, 61.77f32, 90.26f32, 60.41f32, 3.86f32, 61.21f32, 16.6f32, 54.46f32,
            7.29f32, 54.24f32, 92.49f32, 30.18f32, 65.36f32, 99.09f32, 3.8f32, 36.4f32, 86.72f32,
            65.18f32, 29.87f32, 62.21f32, 58.32f32, 43.23f32, 94.3f32, 79.61f32, 39.67f32,
            11.18f32, 48.88f32, 38.19f32, 93.95f32, 10.46f32, 36.7f32, 14.75f32, 81.64f32,
            59.18f32, 99.03f32, 74.23f32, 1.26f32, 82.69f32, 35.7f32, 38.39f32, 46.17f32, 64.75f32,
            7.15f32, 36.55f32, 77.32f32, 18.65f32, 32.8f32, 74.84f32, 18.12f32, 20.19f32, 70.06f32,
            48.37f32, 40.18f32, 45.69f32, 88.3f32, 39.15f32, 60.97f32, 71.29f32, 61.79f32,
            47.23f32, 94.71f32, 58.04f32, 52.4f32, 34.66f32, 59.1f32, 47.11f32, 30.2f32, 58.72f32,
            74.35f32, 83.68f32, 66.8f32, 28.57f32, 29.45f32, 52.02f32, 91.95f32, 92.44f32,
            65.25f32, 38.3f32, 35.6f32, 41.67f32, 91.33f32, 76.81f32, 74.88f32, 33.17f32, 48.36f32,
            41.42f32, 23f32, 8.31f32, 81.69f32, 80.08f32, 50.55f32, 54.46f32, 23.79f32, 43.46f32,
            84.5f32, 10.42f32, 29.51f32, 19.73f32, 46.48f32, 35.01f32, 52.3f32, 66.97f32, 4.8f32,
            74.81f32, 2.82f32, 61.82f32, 25.06f32, 17.3f32, 17.29f32, 63.2f32, 64.1f32, 61.68f32,
            37.42f32, 3.39f32, 97.45f32, 5.32f32, 59.02f32, 35.6f32,
        ];

        let mut aligned_pq_dist_scratch = vec![0.0; 256];
        fixed_chunk_pq_table
            .populate_chunk_distances(&query_vec, &mut aligned_pq_dist_scratch)
            .unwrap();
        assert_eq!(aligned_pq_dist_scratch.len(), 256);

        let pivots = fixed_chunk_pq_table.table.view_pivots();

        // populate_chunk_distances_test
        let sampled_output: f32 = SquaredL2::evaluate(pivots.row(0), query_vec.as_slice());
        assert_eq!(sampled_output, aligned_pq_dist_scratch[0]);

        // populate_chunk_inner_products_test
        fixed_chunk_pq_table
            .populate_chunk_inner_products(&query_vec, &mut aligned_pq_dist_scratch)
            .unwrap();

        let sampled_output: f32 = InnerProduct::evaluate(pivots.row(0), query_vec.as_slice());
        assert_relative_eq!(
            sampled_output,
            aligned_pq_dist_scratch[0],
            max_relative = 1e-6
        );

        // l2_distance_test
        let base_vec: Vec<u8> = vec![3u8];
        let dist = fixed_chunk_pq_table.l2_distance(&query_vec, &base_vec);
        let l2_output: f32 = SquaredL2::evaluate(pivots.row(3), query_vec.as_slice());
        assert_relative_eq!(l2_output, dist, max_relative = 1e-6);

        // inner_product_test
        let dist = fixed_chunk_pq_table.inner_product(&query_vec, &base_vec);
        let ip_output: f32 = InnerProduct::evaluate(pivots.row(3), query_vec.as_slice());
        assert_relative_eq!(ip_output, dist, max_relative = 1e-6);

        // inflate_vector_test
        let inflate_vector = fixed_chunk_pq_table.inflate_vector(&base_vec);
        assert_eq!(inflate_vector.len(), DIM);
        assert_eq!(
            inflate_vector[0],
            pivots[(3, 0)] + fixed_chunk_pq_table.centroids[0]
        );
        assert_eq!(
            inflate_vector[1],
            pivots[(3, 1)] + fixed_chunk_pq_table.centroids[1]
        );
        assert_eq!(
            inflate_vector[127],
            pivots[(3, 127)] + fixed_chunk_pq_table.centroids[127]
        );
    }

    #[test]
    fn test_self_distance() {
        // If this value changes - make sure to update the test loop below.
        let num_pq_chunks = 3;

        let num_centers = 3;
        let dim = 11;
        let offsets = vec![0, 4, 8, dim];
        let centroid = vec![0.0; dim];
        let pq_pivots_pre = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0], // c0
            vec![12.0, 13.0, 14.0, 15.0],
            vec![16.0, 17.0, 18.0, 19.0],
            vec![20.0, 21.0, 22.0], // c1
            vec![23.0, 24.0, 25.0, 26.0],
            vec![27.0, 28.0, 29.0, 30.0],
            vec![31.0, 32.0, 33.0], // c2
        ];

        // Validate the pre-layout.
        assert_eq!(pq_pivots_pre.len(), num_pq_chunks * num_centers);
        pq_pivots_pre
            .chunks(num_pq_chunks)
            .for_each(|inner: &[Vec<_>]| {
                for (i, v) in inner.iter().enumerate() {
                    let expected_length = offsets[i + 1] - offsets[i];
                    assert_eq!(v.len(), expected_length);
                }
            });

        // Merge the PQ pivots into a single dense table.
        let pq_table = pq_pivots_pre.iter().fold(Vec::<f32>::new(), |mut acc, x| {
            acc.extend(x.iter());
            acc
        });

        let table =
            FixedChunkPQTable::new(dim, pq_table.into(), centroid.into(), offsets.into(), None)
                .unwrap();

        let max_relative: f32 = 1.0e-7;
        let range: Range<u8> = 0..(num_centers as u8);
        for (a, b, c) in iproduct!(range.clone(), range.clone(), range.clone()) {
            let left = [a, b, c];
            for (d, e, f) in iproduct!(range.clone(), range.clone(), range.clone()) {
                let right = [d, e, f];

                // Reconstruct the full left and right vectors.
                let mut reconstructed_left = Vec::<f32>::new();
                let mut reconstructed_right = Vec::<f32>::new();

                std::iter::zip(left.iter(), right.iter())
                    .enumerate()
                    .for_each(|(index, (i, j))| {
                        let l = &pq_pivots_pre[num_pq_chunks * (*i as usize) + index];
                        let r = &pq_pivots_pre[num_pq_chunks * (*j as usize) + index];
                        reconstructed_left.extend_from_slice(l);
                        reconstructed_right.extend_from_slice(r);
                    });

                // L2 Distance
                let d = table.qq_l2_distance(left.as_slice(), right.as_slice());
                assert_relative_eq!(
                    d,
                    distance::SquaredL2::evaluate(&*reconstructed_left, &*reconstructed_right,),
                    max_relative = max_relative
                );

                // IP Distance
                let d = table.qq_ip_distance(left.as_slice(), right.as_slice());
                assert_relative_eq!(
                    d,
                    distance::InnerProduct::evaluate(&*reconstructed_left, &*reconstructed_right,),
                    max_relative = max_relative
                );

                // Cosine
                let d = table.qq_cosine_distance(left.as_slice(), right.as_slice());
                assert_relative_eq!(
                    d,
                    distance::Cosine::evaluate(&*reconstructed_left, &*reconstructed_right,),
                    max_relative = max_relative
                );
            }
        }
    }

    type LoadPQPivotResult = (usize, Vec<f32>, Vec<f32>, Vec<usize>);
    fn load_pq_pivots_bin<StorageProvider: StorageReadProvider>(
        pq_pivots_path: &str,
        num_pq_chunks: &usize,
        storage_provider: &StorageProvider,
    ) -> ANNResult<LoadPQPivotResult> {
        if !file_exists(storage_provider, pq_pivots_path) {
            return Err(ANNError::log_pq_error(
                "ERROR: PQ k-means pivot file not found.",
            ));
        }

        let (data, offset_num, offset_dim) =
            load_bin::<u64, _>(&mut storage_provider.open_reader(pq_pivots_path)?, 0)?;

        let file_offset_data =
            convert_types(&data, offset_num * offset_dim, |x: u64| x.into_usize());

        if offset_num != 4 {
            return Err(ANNError::log_pq_error(format_args!(
                "Error reading pq_pivots file {}. \
                 Offsets don't contain correct metadata, \
                 # offsets = {}, but expecting 4.",
                pq_pivots_path, offset_num
            )));
        }

        let (data, pq_center_num, dim) = load_bin::<f32, _>(
            &mut storage_provider.open_reader(pq_pivots_path).unwrap(),
            file_offset_data[0],
        )?;
        let pq_table = data.to_vec();
        if pq_center_num != NUM_PQ_CENTROIDS {
            return Err(ANNError::log_pq_error(format_args!(
                "Error reading pq_pivots file {}. file_num_centers = {}, but expecting {} centers.",
                pq_pivots_path, pq_center_num, NUM_PQ_CENTROIDS
            )));
        }

        let (data, centroid_dim, nc) = load_bin::<f32, _>(
            &mut storage_provider.open_reader(pq_pivots_path).unwrap(),
            file_offset_data[1],
        )?;
        let centroids = data.to_vec();
        if centroid_dim != dim || nc != 1 {
            return Err(ANNError::log_pq_error(format_args!(
                "Error reading pq_pivots file {}. file_dim = {}, \
                 file_cols = {} but expecting {} entries in 1 dimension.",
                pq_pivots_path, centroid_dim, nc, dim
            )));
        }

        let (data, chunk_offset_num, nc) = load_bin::<u32, _>(
            &mut storage_provider.open_reader(pq_pivots_path).unwrap(),
            file_offset_data[2],
        )?;
        let chunk_offsets = convert_types(&data, chunk_offset_num * nc, |x: u32| x.into_usize());
        if chunk_offset_num != num_pq_chunks + 1 || nc != 1 {
            return Err(ANNError::log_pq_error(format_args!(
                "Error reading pq_pivots file at chunk offsets; \
                 file has nr={}, nc={} but expecting nr={} and nc=1.",
                chunk_offset_num,
                nc,
                num_pq_chunks + 1
            )));
        }

        Ok((dim, pq_table, centroids, chunk_offsets))
    }

    #[test]
    fn test_populate_chunk_distances() {
        let dim = 8;
        let num_pq_chunks = 1;
        use rand::Rng;

        let mut rng = crate::utils::create_rnd_in_tests();
        let pq_table: Vec<f32> = (0..NUM_PQ_CENTROIDS * dim).map(|_| rng.random()).collect();
        let centroids: Vec<f32> = (0..dim).map(|_| rng.random()).collect();
        let chunk_offsets = vec![0, 8];
        let fixed_chunk_pq_table = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.clone().into(),
            None,
        )
        .unwrap();

        let rotated_query_vec: Vec<f32> = (0..dim).map(|_| rng.random()).collect();
        let mut aligned_pq_table_dist_scratch = vec![0.0; num_pq_chunks * NUM_PQ_CENTROIDS];

        fixed_chunk_pq_table
            .populate_chunk_distances(&rotated_query_vec, &mut aligned_pq_table_dist_scratch)
            .unwrap();

        assert_eq!(
            aligned_pq_table_dist_scratch.len(),
            num_pq_chunks * NUM_PQ_CENTROIDS
        );
        assert_eq!(fixed_chunk_pq_table.table.dim(), dim);
        assert_eq!(fixed_chunk_pq_table.table.ncenters(), NUM_PQ_CENTROIDS);

        // Assert the output vector is correct
        let expected_output: f32 = SquaredL2::evaluate(
            fixed_chunk_pq_table.table.view_pivots().row(0),
            &*rotated_query_vec,
        );
        assert_eq!(aligned_pq_table_dist_scratch[0], expected_output);
    }

    #[test]
    fn test_populate_chunk_distances_invalid_input() {
        let dim = 6;
        let pq_table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let centroids = vec![0.0; dim];
        let chunk_offsets = vec![0, 2, 4, 6];
        let pq_table = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            None,
        )
        .unwrap();

        let mut aligned_pq_table_dist_scratch = [0.0; 2];
        let rotated_query_vec = vec![0.0; dim];

        // Test when aligned_pq_table_dist_scratch is too short
        let result = pq_table
            .populate_chunk_distances(&rotated_query_vec, &mut aligned_pq_table_dist_scratch);
        assert!(result.is_err());
    }
}
#[cfg(test)]
mod pq_index_prune_query_test {

    use super::*;

    #[test]
    #[allow(clippy::identity_op)]
    fn pq_dist_lookup_test() {
        let pq_ids: Vec<u8> = vec![1u8, 3u8, 2u8, 2u8];
        let mut pq_dists: Vec<f32> = Vec::with_capacity(256 * 2);
        for _ in 0..pq_dists.capacity() {
            pq_dists.push(rand::random());
        }

        let mut dists_out = vec![0.0f32; 2];
        pq_dist_lookup(&pq_ids, 2, 2, &pq_dists, dists_out.as_mut_slice()).unwrap();
        assert_eq!(dists_out.len(), 2);
        assert_eq!(dists_out[0], pq_dists[0 + 1] + pq_dists[256 + 3]);
        assert_eq!(dists_out[1], pq_dists[0 + 2] + pq_dists[256 + 2]);
    }

    #[test]
    fn test_pq_dist_lookup_invalid_input() {
        let pq_coordinates = vec![0u8; 10];
        let n_pts = 5;
        let pq_nchunks = 2;
        let query_centroid_pq_dists = vec![0.0f32; 512];
        let mut pq_distance_scratch = vec![0.0f32; 4];

        assert!(
            pq_dist_lookup(
                &pq_coordinates,
                n_pts,
                pq_nchunks,
                &query_centroid_pq_dists,
                &mut pq_distance_scratch,
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod aggregate_coords_test {
    use super::*;

    #[test]
    fn test_aggregate_coords() {
        let ids = [0, 1, 2, 3];
        let all_coords = [
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33,
        ];
        let num_pq_chunks = 2;
        let mut pq_coordinate_scratch = vec![0; ids.len() * num_pq_chunks];

        aggregate_coords(&ids, &all_coords, num_pq_chunks, &mut pq_coordinate_scratch).unwrap();

        assert_eq!(
            pq_coordinate_scratch,
            vec![10, 11, 12, 13, 14, 15, 16, 17],
            "Aggregated coordinates are incorrect"
        );

        assert!(
            aggregate_coords(
                &ids,
                &all_coords,
                num_pq_chunks * 2,
                &mut pq_coordinate_scratch
            )
            .is_err()
        );
    }
}

impl DebugQuantizer for FixedChunkPQTable {
    type DistanceComputer = pq_distance::DistanceComputer<Arc<FixedChunkPQTable>>;
    type QueryComputer = pq_distance::QueryComputer<Arc<FixedChunkPQTable>>;

    fn num_chunks(&self) -> usize {
        self.get_num_chunks()
    }

    fn compress_into(&self, input: &[f32], output: &mut [u8]) -> ANNResult<()> {
        diskann_quantization::CompressInto::compress_into(self, input, output).bridge_err()?;
        Ok(())
    }

    fn build_distance_computer(
        &self,
        metric: diskann_vector::distance::Metric,
    ) -> ANNResult<Self::DistanceComputer> {
        pq_distance::DistanceComputer::new(Arc::new(self.clone()), metric).map_err(ANNError::from)
    }

    fn build_query_computer(
        &self,
        metric: diskann_vector::distance::Metric,
        query: &[f32],
    ) -> ANNResult<Self::QueryComputer> {
        pq_distance::QueryComputer::new(Arc::new(self.clone()), metric, query, None)
    }
}
