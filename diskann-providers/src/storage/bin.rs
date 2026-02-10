/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};

use super::{StorageReadProvider, StorageWriteProvider};
use byteorder::{LittleEndian, ReadBytesExt};
use diskann::{
    ANNError, ANNResult,
    utils::{IntoUsize, VectorRepr},
};

use crate::{
    model::graph::traits::AdHoc,
    utils::{load_metadata_from_file, write_metadata},
};

/// An simplified adaptor interface for allowing providers to use and [`load_graph`].
///
/// These traits are meant for IO purposes and are not meant as general access traits for
/// types that implement them.
///
/// Furthermore, these are meant for data structures that use the identity mapping between
/// internal and external IDs.
pub trait SetData {
    /// The element type of input slices.
    type Item;

    /// Set the data contained in `element` to position `i` in `self`.
    ///
    /// Upon failure, return an error.
    fn set_data(&mut self, i: usize, element: &[Self::Item]) -> ANNResult<()>;
}

/// An simplified adaptor interface for allowing providers to use [`save_to_bin`].
///
/// These traits are meant for IO purposes and are not meant as general access traits for
/// types that implement them.
///
/// Furthermore, these are meant for data structures that use the identity mapping between
/// internal and external IDs.
pub trait GetData {
    /// The element type of output slices.
    type Element;

    /// Types that implement this trait may return their data as slices directly, or may
    /// use some proxy object to hold the slice.
    ///
    /// The alias `Item` allows this heterogeneous behavior.
    type Item<'a>: std::ops::Deref<Target = [Self::Element]>
    where
        Self: 'a;

    /// Retrieve the data stored at index `i`.
    fn get_data(&self, i: usize) -> ANNResult<Self::Item<'_>>;

    /// Return the total number of elements contained in `self`.
    fn total(&self) -> usize;

    /// Return the dimension of each element in self.
    fn dim(&self) -> usize;
}

/// An simplified adaptor interface for allowing providers to use [`load_graph`].
///
/// These traits are meant for IO purposes and are not meant as general access traits for
/// types that implement them.
///
/// Furthermore, these are meant for data structures that use the identity mapping between
/// internal and external IDs.
pub(crate) trait SetAdjacencyList {
    /// The element type of input slices.
    type Item;

    /// Set the out-bound adjacency list for node `i`.
    ///
    /// Upon failure, return an error.
    fn set_adjacency_list(&mut self, i: usize, element: &[Self::Item]) -> ANNResult<()>;
}

/// An simplified adaptor interface for allowing providers to use [`save_graph`].
///
/// These traits are meant for IO purposes and are not meant as general access traits for
/// types that implement them.
///
/// Furthermore, these are meant for data structures that use the identity mapping between
/// internal and external IDs.
pub(crate) trait GetAdjacencyList {
    /// The element type of output slices.
    type Element;

    /// Types that implement this trait may return their data as slices directly, or may
    /// use some proxy object to hold the slice.
    ///
    /// The alias `Item` allows this heterogeneous behavior.
    type Item<'a>: std::ops::Deref<Target = [Self::Element]>
    where
        Self: 'a;

    /// Retrieve the data stored at index `i`.
    fn get_adjacency_list(&self, i: usize) -> ANNResult<Self::Item<'_>>;

    /// Return the total number of elements contained in `self`.
    fn total(&self) -> usize;

    /// Return number of additional points
    fn additional_points(&self) -> u64;

    /// Returns the maximum allowed degree of the graph.
    ///
    /// - None: Use observed maximum degree from the actual graph data.
    /// - Some(value): Use this configured value, which is critical for partial graphs
    ///   where observed degrees may be smaller than intended.
    fn max_degree(&self) -> Option<u32>;
}

//////////////////
// Data Loading //
//////////////////

/// Load data from a `.bin` formatted file at path `path` and use that data to initialize
/// a `SetData` compatible object `S.
///
/// The number of points and dimension of each vector will be determined from the file
/// metadata and passed to the closure `create` as `(num_points, dim)`.
///
/// After creation, this method will call `SetData::set_data` for all i in `0..num_points`
/// along with the vectors of length `dim`.
///
/// See also [`save_to_bin`] for a description of the `.bin` file format.
pub fn load_from_bin<T, P, F, S>(provider: &P, path: &str, create: F) -> ANNResult<S>
where
    P: StorageReadProvider,
    F: FnOnce(usize, usize) -> ANNResult<S>,
    S: SetData<Item = T>,
    T: VectorRepr,
{
    let metadata = load_metadata_from_file(provider, path).map_err(|err| {
        ANNError::log_index_error(format_args!(
            "failed to load data file \"{}\" due to the following error: {}",
            path, err
        ))
    })?;

    tracing::info!(
        "Loading {} vectors with dimension {} from storage system {} into dataset...",
        metadata.npoints,
        metadata.ndims,
        path
    );

    let mut data = create(metadata.npoints, metadata.ndims)?;
    let itr = crate::utils::VectorDataIterator::<_, AdHoc<T>>::new(path, None, provider)?;
    for (i, (vector, _)) in itr.enumerate() {
        data.set_data(i.into_usize(), &vector)?;
    }

    tracing::info!("Dataset loaded.");
    Ok(data)
}

/// Save `data` into a `.bin` file format at the filepath `path`.
///
/// After file creation, this method will call `GetData::get_data` for all `i` in
/// `0..data.total()`, serializing the data to disk.
///
/// See also: [`load_from_bin`].
///
/// # File Layout
///
/// The `.bin` layout consists of the following:
///
/// * `total_points` as `u32` in little-endian.
/// * `dim` as `u32` in little-endian.
/// * Vector data in a dense layout with each element store in its canonical binary
///   encoding using little-endian.
pub fn save_to_bin<S, T, P>(data: &S, provider: &P, path: &str) -> ANNResult<usize>
where
    S: GetData<Element = T>,
    T: bytemuck::Pod,
    P: StorageWriteProvider,
{
    let total = data.total();
    let dim = data.dim();

    let mut writer = provider.create_for_write(path)?;

    let mut points_written: u32 = 0;

    write_metadata(&mut writer, points_written, dim)?;
    for i in 0..total {
        // The binding provides a stable address for the return item of `get_data`,
        // regardless of if `get_data` returns a borrowed slice or a copy.
        let binding = data.get_data(i)?;
        let slice = &*binding;

        let len = slice.len();
        if len != dim {
            return Err(ANNError::log_index_error(
                "data provider returned a vector with a dimension other than advertised",
            ));
        }

        // The type `VectorElement` implements `bytemuck::Pod`, meaning
        // that we can safely inspect the contents through a byte lens.
        // Casting Pod type to bytes always succeeds (u8 has alignment of 1)
        let reinterpret: &[u8] = bytemuck::must_cast_slice(slice);

        writer.write_all(reinterpret)?;
        points_written += 1;
    }
    writer.seek(std::io::SeekFrom::Start(0_u64))?;
    writer.write_all(&points_written.to_le_bytes())?;
    writer.flush()?;
    let bytes_written = 2 * std::mem::size_of::<u32>()
        + points_written.into_usize() * dim * std::mem::size_of::<T>();
    Ok(bytes_written)
}

///////////////////
// Graph Loading //
///////////////////

/// Load data from a canonical graph formatted file at path `path` and use that data to
/// initialize a `SetAdjacencyList` compatible object `S.
///
/// The number of points and maximum degree of the stored graph will be determined from the
/// file and passed to the closure `create` as `(num_points, max_degree, num_start_points)`.
///
/// After creation, this method will call `SetAdjacencyList::set_adjacency_list` for all
/// i in `0..num_points`. The adjacency list give to `set_data` will vary in length
/// between `0` and `max_degree` and reflect the actual length of the saved adjacency list.
///
/// See also [`save_graph`] for a description of the file format.
pub fn load_graph<P, S, F>(provider: &P, path: &str, create: F) -> ANNResult<S>
where
    P: StorageReadProvider,
    S: SetAdjacencyList<Item = u32>,
    F: FnOnce(usize, usize, usize) -> ANNResult<S>,
{
    // The number of bytes consumed by the header.
    const METADATA_SIZE: usize = 24;

    // First, we open the file and read the metadata.
    let mut file = BufReader::new(provider.open_reader(path)?);

    let file_size = file.read_u64::<LittleEndian>()?.into_usize();
    let max_degree = file.read_u32::<LittleEndian>()?.into_usize();
    let start = file.read_u32::<LittleEndian>()?;
    let num_start_points = file.read_u64::<LittleEndian>()?.into_usize();

    // The position in the file after reading the metadata.
    let mut position = METADATA_SIZE;

    // Now that we've parsed the header, we need to figure out how many points are actually
    // in the saved dataset.
    //
    // Since this isn't stored directly in the header, we need to inspect the file directly.
    //
    // After we determine the number of points, we seek back to the beginning of the header
    // to actually load the data.
    let mut num_points: usize = 0;
    while position < file_size {
        num_points += 1;
        let num_neighbors: i64 = file.read_u32::<LittleEndian>()?.into();
        // Seek forward this amount.
        let seek_amount: i64 = num_neighbors * (std::mem::size_of::<u32>() as i64);

        file.seek_relative(seek_amount)?;
        position += std::mem::size_of::<u32>() + (seek_amount as usize);
    }

    tracing::info!("Num points: {}, max degree: {}", num_points, max_degree);

    // We've reached the end of the file.
    // Seek back to the just after the metadata, create the neighbor provider, and actually
    // begin populating adjacency lists.
    file.seek_relative(-((position - METADATA_SIZE) as i64))?;
    let mut graph = create(num_points, max_degree, num_start_points)?;

    position = METADATA_SIZE;
    let mut buffer: Vec<u32> = vec![0; max_degree];
    num_points = 0;

    let mut num_edges = 0;
    while position < file_size {
        let num_neighbors = file.read_u32::<LittleEndian>()?;
        if num_neighbors == 0 {
            tracing::debug!("Point found with no out-neighbors, point# {}", num_points);
        }

        let buffer = &mut buffer[..num_neighbors.into_usize()];
        file.read_u32_into::<LittleEndian>(buffer)?;
        graph.set_adjacency_list(num_points, buffer)?;

        position += std::mem::size_of::<u32>() * (1 + num_neighbors.into_usize());
        num_edges += num_neighbors.into_usize();
        num_points += 1;
    }

    tracing::info!(
        "Done. Index has {} nodes and {} out-edges, _start is set to {}",
        num_points,
        num_edges,
        start
    );

    Ok(graph)
}

/// Save `data` into a canonical graph layout at `path`.
///
/// After file creation, this method will call `GetAdjacencyList::get_adjacency_list` for
/// all `i` in `0..data.total()`, serializing the graph to disk.
///
/// See also: [`load_graph`].
///
/// # File Layout
///
/// The graph layout consists of a 24-byte header containing:
///
/// * The file size as a `u64` in little-endian.
/// * The maximum degree as `u32` in little-endian.
/// * The ID of the start point as `u32` little-endian.
/// * The number of start points as `u64` in little-endian.
///
/// After the header, each adjacency list in stored densely, consisting of a `u32` encoding
/// the length `L` of the adjacency list followed by `L` u32-values containing the
/// out-neighbors of this node. These adjacency lists are stored in-order.
pub fn save_graph<S, P>(
    graph: &S,
    provider: &P,
    start_point: u32,
    path: &str,
) -> ANNResult<usize>
where
    S: GetAdjacencyList<Element = u32>,
    P: StorageWriteProvider,
{
    let file = provider.create_for_write(path)?;

    let mut out = BufWriter::new(file);

    let mut index_size: u64 = 24;
    let mut observed_max_degree: u32 = 0;

    out.write_all(&index_size.to_le_bytes())?;
    out.write_all(&observed_max_degree.to_le_bytes())?; // Will be updated later with correct max_degree
    out.write_all(&start_point.to_le_bytes())?;

    out.write_all(&graph.additional_points().to_le_bytes())?;
    let total = graph.total();

    for i in 0..total {
        let binding = graph.get_adjacency_list(i)?;
        let neighbors: &[u32] = &binding;
        let num_neighbors: u32 = neighbors.len() as u32;

        // Write the number of neighbors as a `u32`.
        out.write_all(&num_neighbors.to_le_bytes())?;

        // Write all the neighbors, applying transformation if provided.
        neighbors
            .iter()
            .copied()
            .try_for_each(|n| out.write_all(&n.to_le_bytes()))?;

        observed_max_degree = observed_max_degree.max(num_neighbors);
        index_size += (std::mem::size_of::<u32>() * (1 + neighbors.len())) as u64;
    }

    // Use configured max degree if provided, otherwise use observed
    let max_degree = graph.max_degree().unwrap_or(observed_max_degree);

    // Finish up by writing the observed index size and max degree.
    out.seek(SeekFrom::Start(0))?;
    out.write_all(&index_size.to_le_bytes())?;
    out.write_all(&max_degree.to_le_bytes())?;
    out.flush()?;
    Ok(index_size.into_usize())
}
