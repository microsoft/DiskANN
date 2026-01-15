/* Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{
    io::{Read, Seek, Write},
    mem,
};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use diskann::{ANNError, ANNResult};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::graph::traits::GraphDataType,
    storage::{get_mem_index_file, path_utility::*},
    utils::{save_bytes, READ_WRITE_BLOCK_SIZE},
};
use tracing::info;

use crate::{
    data_model::{GraphHeader, GraphMetadata},
    storage::{CachedReader, CachedWriter},
};

// Struct DiskIndexWriterState maintains the state of the process of creating a disk
// layout using index and associated data. By moving the state to this struct, we
// can create it on stack in the create_disk_layout() function and then pass it
// other functions as data processing proceeds.
struct DiskIndexWriterState<StorageProvider>
where
    StorageProvider: StorageReadProvider,
{
    // If index data does not fit into memory, the reader will be created and used
    // to get data from the disk.
    muti_shard_index_reader: Option<StorageProvider::Reader>,

    // Reader to get associated data from disk.
    associated_data_reader: Option<CachedReader<StorageProvider>>,

    // Reader to get data from the disk.
    dataset_reader: Option<CachedReader<StorageProvider>>,

    // Parameters required for processing data. They are set before data processing starts.
    dims: u64,
    num_pts: u64,
    max_degree: u32,
    medoid: u32,
    vamana_frozen_num: u64,
    node_len: u64,
    associated_data_length: usize,
    read_blk_size: u64,
    write_blk_size: u64,
}

impl<StorageProvider> DiskIndexWriterState<StorageProvider>
where
    StorageProvider: StorageReadProvider,
{
    /// Create DiskIndexWriterState instance
    fn new() -> Self {
        DiskIndexWriterState {
            muti_shard_index_reader: None,
            associated_data_reader: None,
            dataset_reader: None,
            dims: 0,
            num_pts: 0,
            max_degree: 0,
            medoid: 0,
            vamana_frozen_num: 0,
            node_len: 0,
            associated_data_length: 0,
            read_blk_size: READ_WRITE_BLOCK_SIZE,
            write_blk_size: READ_WRITE_BLOCK_SIZE,
        }
    }
}

/// DiskIndexWriter is used to write disk index data to the storage system.
/// The storage system and data types are provided as parameters to methods that need them.
pub struct DiskIndexWriter {
    /// Dataset file
    dataset_file: String,

    /// Index file path prefix
    index_path_prefix: String,

    /// Optional associated data file
    associated_data_file: Option<String>,

    /// Block size (bytes) used when writing the disk index.
    block_size: usize,
}

impl DiskIndexWriter {
    /// Create DiskIndexWriter instance
    pub fn new(
        dataset_file: String,
        index_path_prefix: String,
        associated_data_file: Option<String>,
        block_size: usize,
    ) -> ANNResult<Self> {
        if block_size < GraphMetadata::get_size() {
            return Err(ANNError::log_index_config_error(
                "index_block_size".to_string(),
                format!(
                    "block_size should be greater than the size of GraphMetadata: {}",
                    GraphMetadata::get_size()
                ),
            ));
        }

        Ok(DiskIndexWriter {
            dataset_file,
            index_path_prefix,
            associated_data_file,
            block_size,
        })
    }

    pub fn dataset_file(&self) -> &String {
        &self.dataset_file
    }

    pub fn index_path_prefix(&self) -> &String {
        &self.index_path_prefix
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    fn get_neighbors_number<StorageProvider>(
        state: &mut DiskIndexWriterState<StorageProvider>,
    ) -> ANNResult<u32>
    where
        StorageProvider: StorageReadProvider,
    {
        let num_nbrs: u32;
        if let Some(vamana_reader) = state.muti_shard_index_reader.as_mut() {
            num_nbrs = vamana_reader.read_u32::<LittleEndian>()?;
        } else {
            return Err(ANNError::log_index_error("invalid index reader"));
        }

        Ok(num_nbrs)
    }

    fn copy_neighbors<StorageProvider>(
        state: &mut DiskIndexWriterState<StorageProvider>,
        nbrs_buf: &mut [u8],
    ) -> ANNResult<()>
    where
        StorageProvider: StorageReadProvider,
    {
        if let Some(vamana_reader) = state.muti_shard_index_reader.as_mut() {
            vamana_reader.read_exact(nbrs_buf)?;
        } else {
            return Err(ANNError::log_index_error("invalid index reader"));
        }

        Ok(())
    }

    fn open_vamana_reader<StorageProvider>(
        &self,
        state: &mut DiskIndexWriterState<StorageProvider>,
        storage_provider: &StorageProvider,
    ) -> ANNResult<()>
    where
        StorageProvider: StorageReadProvider,
    {
        let mem_index_file = self.get_mem_index_file();

        // Create cached reader + writer
        let actual_file_size = storage_provider.get_length(mem_index_file.as_str())?;
        info!("Vamana index file size={}", actual_file_size);

        state.muti_shard_index_reader =
            Some(storage_provider.open_reader(mem_index_file.as_str())?);

        if let Some(vamana_reader) = state.muti_shard_index_reader.as_mut() {
            let index_file_size = vamana_reader.read_u64::<LittleEndian>()?;
            if index_file_size != actual_file_size {
                info!(
                    "Vamana Index file size does not match expected size per meta-data. file size from file: {}, actual file size: {}",
                    index_file_size, actual_file_size
                );
            }

            state.max_degree = vamana_reader.read_u32::<LittleEndian>()?;
            state.medoid = vamana_reader.read_u32::<LittleEndian>()?;
            state.vamana_frozen_num = vamana_reader.read_u64::<LittleEndian>()?;

            return Ok(());
        }

        Err(ANNError::log_index_error("invalid index reader"))
    }

    fn open_associated_data_reader<StorageProvider>(
        &self,
        state: &mut DiskIndexWriterState<StorageProvider>,
        storage_provider: &StorageProvider,
    ) -> ANNResult<()>
    where
        StorageProvider: StorageReadProvider,
    {
        (state.associated_data_reader, state.associated_data_length) = match &self
            .associated_data_file
        {
            Some(associated_data_stream) => {
                let mut associated_data_reader = CachedReader::<StorageProvider>::new(
                    associated_data_stream.as_str(),
                    state.read_blk_size,
                    storage_provider,
                )?;

                let associated_data_num_pts = associated_data_reader.read_u32()? as u64;
                let length = associated_data_reader.read_u32()? as usize;

                if state.num_pts != associated_data_num_pts {
                    return Err(ANNError::log_index_error(format_args!(
                        "Number of points in dataset file ({}) does not match number of points in associated data file ({}).",
                        state.num_pts, associated_data_num_pts
                    )));
                }

                (Option::Some(associated_data_reader), length)
            }

            None => (Option::None, 0),
        };

        Ok(())
    }

    fn open_dataset_reader<StorageProvider>(
        &self,
        state: &mut DiskIndexWriterState<StorageProvider>,
        storage_provider: &StorageProvider,
    ) -> ANNResult<()>
    where
        StorageProvider: StorageReadProvider,
    {
        let dataset_reader = CachedReader::<StorageProvider>::new(
            self.dataset_file.as_str(),
            state.read_blk_size,
            storage_provider,
        )?;
        state.dataset_reader = Some(dataset_reader);

        if let Some(dataset_reader) = state.dataset_reader.as_mut() {
            state.num_pts = dataset_reader.read_u32()? as u64;
            state.dims = dataset_reader.read_u32()? as u64;
        }

        Ok(())
    }

    fn read_neighbors<Data, StorageProvider>(
        &self,
        state: &mut DiskIndexWriterState<StorageProvider>,
        block_buf: &mut [u8],
    ) -> ANNResult<()>
    where
        Data: GraphDataType<VectorIdType = u32>,
        StorageProvider: StorageReadProvider,
    {
        block_buf.fill(0);

        // write coords of node first
        if let Some(dataset_reader) = state.dataset_reader.as_mut() {
            let mut cur_node_coords =
                vec![0u8; (state.dims as usize) * mem::size_of::<Data::VectorDataType>()];

            dataset_reader.read(&mut cur_node_coords)?;
            block_buf[..cur_node_coords.len()].copy_from_slice(&cur_node_coords);
        }

        // read cur node's num_nbrs
        let num_nbrs: u32 = Self::get_neighbors_number(state)?;

        // sanity checks on num_nbrs
        debug_assert!(num_nbrs > 0);
        debug_assert!(num_nbrs <= state.max_degree);

        let num_nbrs_start = state.dims as usize * mem::size_of::<Data::VectorDataType>();
        let nbrs_buf_start = num_nbrs_start + mem::size_of::<u32>();

        // write num_nbrs
        LittleEndian::write_u32(
            &mut block_buf[num_nbrs_start..(num_nbrs_start + mem::size_of::<u32>())],
            num_nbrs,
        );

        // write neighbors
        let nbr_buf_end = nbrs_buf_start + (num_nbrs as usize) * mem::size_of::<u32>();

        let nbrs_buf = &mut block_buf[nbrs_buf_start..nbr_buf_end];

        Self::copy_neighbors(state, nbrs_buf)?;

        if let Some(associated_data_reader) = state.associated_data_reader.as_mut() {
            let cur_node_associated_data = &mut block_buf[(state.node_len as usize
                - state.associated_data_length * mem::size_of::<Data::AssociatedDataType>())
                ..(state.node_len as usize)];
            associated_data_reader.read(cur_node_associated_data)?;
        }

        Ok(())
    }

    fn write_header<Data, Writer, StorageProvider>(
        &self,
        state: &mut DiskIndexWriterState<StorageProvider>,
        block_size: usize,
        num_nodes_per_block: u64,
        disk_index_file_size: u64,
        writer: &mut Writer,
    ) -> ANNResult<()>
    where
        Data: GraphDataType<VectorIdType = u32>,
        Writer: Write + Seek,
        StorageProvider: StorageReadProvider,
    {
        let mut vamana_frozen_loc: u64 = 0;
        if state.vamana_frozen_num == 1 {
            vamana_frozen_loc = state.medoid as u64;
        }

        let graph_metadata = GraphMetadata::new(
            state.num_pts,
            state.dims as usize,
            state.medoid as u64,
            state.node_len,
            num_nodes_per_block,
            state.vamana_frozen_num,
            vamana_frozen_loc,
            disk_index_file_size,
            state.associated_data_length * mem::size_of::<Data::AssociatedDataType>(),
        );

        let header = GraphHeader::new(
            graph_metadata,
            block_size as u64,
            GraphHeader::CURRENT_LAYOUT_VERSION,
        );

        let bytes_header = header.to_bytes()?;
        save_bytes(
            writer,
            bytes_header.as_slice(),
            bytes_header.len(), // num_points is kept to make the graph compatible with C++ version
            1,                  // num_points is kept to make the graph compatible with C++ version
            0,
        )?;

        Ok(())
    }

    /// Create disk layout.
    /// Block #1: GraphMetadata.
    /// Block #2..#n: num_nodes_per_block nodes in each block
    ///
    /// GraphMetadata layout:
    /// |number_of_points (8 bytes)| dimensions (8 bytes) | medoid (8 bytes) |
    /// ...| node_len (8 bytes) | num_nodes_per_sector (8 bytes) | vamana_frozen_point_num (8 bytes) |
    /// ...| vamana_frozen_loc (8 bytes) | append_reorder_data (8 bytes) | disk_index_file_size (8 bytes) |
    /// ...| associated_data_length (8 bytes) | block_size (8 bytes) | layout_version (8 bytes) |
    ///
    /// The metadata layout is kept compatible with the C++ diskann codebase.
    ///
    /// After the metadata structure, the graph is laid out as a sequence of vertices and relevant data
    /// including the vector, #out neighbors, list of out neighbors, associated_data and appropriate padding.
    /// `| vector (dim * size_of<VectorDataType> bytes) | neighbor_count (4 bytes) | neighbors (neighbor_count * 4 bytes) |
    /// ... | filler | associated_data (associated_data_length * size_of<AssociatedDataType> bytes) |`
    ///
    /// The node_len and length of filler are calculated as follow：
    ///
    /// `node_length = (max_degree + 1) * size_of<u32> + dim * size_of<VectorDataType> + associated_data_length * size_of<AssociatedDataType>`
    ///
    /// `filler length = node_len -  (length of vector + 4 + 4 * neighbor_count + associated_data_length * size_of<AssociatedDataType>)`
    ///
    /// The filler is used to pad each node to node_len bytes on the disk, so it is possible to calculate the block number and offset of a node with its Id.
    ///
    /// When node_len < disk block size, we pack as many nodes as possible in to the disk sector without splitting a node across a sector
    /// For example, if node_len is 600B, we can pack 6 of these on a 4KB sector, and we leave 4096-3600 = 496B unused.
    ///
    /// When node_len > disk block size, we align start of node to block size.
    /// For example, if node_len is 6700B, then it would span two 4KB sectors beginning at the start of the first sector
    /// and end on the second sector and will be followed by 1492 bytes of padding to align the next node to block boundary.
    ///
    ///
    /// # Arguments
    /// * `storage_provider` - the storage provider for I/O operations
    pub fn create_disk_layout<Data, StorageProvider>(
        &self,
        storage_provider: &StorageProvider,
    ) -> ANNResult<()>
    where
        Data: GraphDataType<VectorIdType = u32>,
        StorageProvider: StorageReadProvider + StorageWriteProvider,
    {
        let block_size = self.block_size;
        let mut state: DiskIndexWriterState<StorageProvider> = DiskIndexWriterState::new();

        self.open_dataset_reader(&mut state, storage_provider)?;
        self.open_associated_data_reader(&mut state, storage_provider)?;
        self.open_vamana_reader(&mut state, storage_provider)?;

        let vector_size = state.dims * (mem::size_of::<Data::VectorDataType>() as u64);

        state.node_len = ((state.max_degree as u64 + 1) * (mem::size_of::<u32>() as u64))
            + vector_size
            + (state.associated_data_length * mem::size_of::<Data::AssociatedDataType>()) as u64;

        let num_nodes_per_block = (block_size as u64) / state.node_len; // 0 if node_len > block_size

        info!("block_size: {}B", block_size);
        info!("medoid: {}B", state.medoid);
        info!("node_len: {}B", state.node_len);
        info!("num_nodes_per_sector: {}B", num_nodes_per_block);
        info!(
            "associated_data_length: {}B",
            state.associated_data_length * mem::size_of::<Data::AssociatedDataType>()
        );

        // number of sectors (1 for meta data)
        let num_blocks = if num_nodes_per_block > 0 {
            state.num_pts.div_ceil(num_nodes_per_block)
        } else {
            let num_block_per_node = state.node_len.div_ceil(block_size as u64);
            info!("num_sector_per_node: {}B", num_block_per_node);
            state.num_pts * num_block_per_node
        };
        info!("num_blocks: {}B", num_blocks);

        let disk_layout_file = self.disk_index_file();
        {
            let storage_writer = storage_provider.create_for_write(disk_layout_file.as_str())?;
            let mut diskann_writer = CachedWriter::<StorageProvider>::new(
                disk_layout_file.as_str(),
                state.write_blk_size,
                storage_writer,
            )?;

            // Buffer of block_size bytes for each block.
            let mut block_buf = vec![0u8; block_size];
            diskann_writer.write(&block_buf)?;

            if num_nodes_per_block > 0 {
                let mut cur_node_id = 0u64;
                let mut node_buf = vec![0u8; state.node_len as usize];

                // Write multiple nodes per sector
                for sector in 0..num_blocks {
                    if sector % 100_000 == 0 {
                        info!("Sector #{} written", sector);
                    }
                    block_buf.fill(0);

                    for sector_node_id in 0..num_nodes_per_block {
                        if cur_node_id >= state.num_pts {
                            break;
                        }

                        self.read_neighbors::<Data, _>(&mut state, &mut node_buf)?;

                        // get offset into sector_buf
                        let sector_node_buf_start = (sector_node_id * state.node_len) as usize;
                        let sector_node_buf = &mut block_buf[sector_node_buf_start
                            ..(sector_node_buf_start + state.node_len as usize)];
                        sector_node_buf.copy_from_slice(&node_buf[..(state.node_len as usize)]);

                        cur_node_id += 1;
                    }

                    // flush sector to disk
                    diskann_writer.write(&block_buf)?;
                }
            } else {
                // Write multi-sector nodes
                let mut multi_block_buf =
                    vec![0u8; state.node_len.next_multiple_of(block_size as u64) as usize];
                let num_block_per_node = state.node_len.div_ceil(block_size as u64);

                for node_idx in 0..state.num_pts {
                    if (node_idx * num_block_per_node).is_multiple_of(100_000) {
                        info!("Sector #{} written", node_idx * num_block_per_node);
                    }

                    self.read_neighbors::<Data, _>(&mut state, &mut multi_block_buf)?;

                    // flush sector to disk
                    diskann_writer.write(&multi_block_buf)?;
                }
            }

            // Be sure to flush the writer before it goes out of scope so we can open a new one.
            diskann_writer.flush()?;
        }

        // Write the header.  Must re-open the file because the cached writer cannot seek to the start of the file.
        // CachedWriter owns the underlying writer so we must open a new writer.  A new scope ensures that the old
        // writer is out of scope.
        {
            let mut storage_writer = storage_provider.open_writer(disk_layout_file.as_str())?;
            let disk_index_file_size = (num_blocks + 1) * (block_size as u64);
            self.write_header::<Data, _, _>(
                &mut state,
                block_size,
                num_nodes_per_block,
                disk_index_file_size,
                &mut storage_writer,
            )?;

            storage_writer.flush()?;
            Ok(())
        }
    }

    pub fn index_build_cleanup<StorageProvider>(
        &self,
        storage_provider: &StorageProvider,
    ) -> ANNResult<()>
    where
        StorageProvider: StorageReadProvider + StorageWriteProvider,
    {
        // Clean up the in-memory index file if it exists.
        let inmem_index_identifier = self.get_mem_index_file();
        if storage_provider.exists(&inmem_index_identifier) {
            storage_provider.delete(&get_mem_index_file(&self.index_path_prefix))?;
        }

        Ok(())
    }

    pub fn disk_index_file(&self) -> String {
        get_disk_index_file(&self.index_path_prefix)
    }

    pub fn get_pq_pivot_file(&self) -> String {
        get_pq_pivot_file(&self.index_path_prefix)
    }

    pub fn get_compressed_pq_pivot_file(&self) -> String {
        get_compressed_pq_file(&self.index_path_prefix)
    }

    pub fn get_disk_index_pq_pivot_file(&self) -> String {
        get_disk_index_pq_pivot_file(&self.index_path_prefix)
    }

    pub fn get_disk_index_compressed_pq_file(&self) -> String {
        get_disk_index_compressed_pq_file(&self.index_path_prefix)
    }

    pub fn get_index_path_prefix(&self) -> String {
        self.index_path_prefix.clone()
    }

    pub fn get_dataset_file(&self) -> String {
        self.dataset_file.clone()
    }

    pub fn get_associated_data_file(&self) -> Option<String> {
        self.associated_data_file.clone()
    }

    pub fn get_mem_index_file(&self) -> String {
        get_mem_index_file(&self.index_path_prefix)
    }

    pub fn get_merged_index_prefix(&self) -> String {
        self.get_mem_index_file().clone() + "_tempFiles"
    }

    pub fn get_merged_index_subshard_id_map_file(prefix: &str, shard: usize) -> String {
        format!("{}_subshard-{}_ids_uint32.bin", prefix, shard)
    }

    pub fn get_merged_index_subshard_data_file(prefix: &str, shard: usize) -> String {
        format!("{}_subshard-{}.bin", prefix, shard)
    }

    pub fn get_merged_index_subshard_prefix(prefix: &str, shard: usize) -> String {
        format!("{}_subshard-{}", prefix, shard)
    }

    pub fn get_merged_index_subshard_mem_index_file(prefix: &str, shard: usize) -> String {
        format!("{}_subshard-{}_mem.index", prefix, shard)
    }

    pub fn get_merged_index_subshard_mem_dataset_file(subshard_mem_index_prefix: &str) -> String {
        get_mem_index_data_file(subshard_mem_index_prefix)
    }
}

#[cfg(test)]
mod disk_index_storage_test {
    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_providers::test_utils::graph_data_type_utils::GraphDataF32VectorU32Data;
    use diskann_utils::test_data_root;
    use vfs::OverlayFS;

    use super::*;

    const TRUTH_DISK_LAYOUT_METADATA_LENGTH: usize = 80;
    const DEFAULT_DISK_SECTOR_LEN: usize = 4096;

    #[test]
    fn create_disk_layout_test_low_dim() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());

        const TEST_DATA_FILE_LOW_DIM: &str = "/sift/siftsmall_learn_256pts.fbin";
        const DISK_INDEX_PATH_PREFIX_LOW_DIM: &str =
            "/disk_index_misc/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_index_writer";
        const TRUTH_DISK_LAYOUT_LOW_DIM: &str =
            "/disk_index_misc/truth_disk_index_siftsmall_learn_256pts_R4_L50_A1.2_disk.index";

        let disk_layout_file = {
            let index_writer = DiskIndexWriter::new(
                TEST_DATA_FILE_LOW_DIM.to_string(),
                DISK_INDEX_PATH_PREFIX_LOW_DIM.to_string(),
                Option::None,
                DEFAULT_DISK_SECTOR_LEN,
            )
            .unwrap();
            index_writer
                .create_disk_layout::<GraphDataF32VectorU32Data, _>(&storage_provider)
                .unwrap();

            let disk_layout_file = index_writer.disk_index_file();
            let mut rust_disk_layout: Vec<u8> = Vec::new();
            storage_provider
                .open_reader(disk_layout_file.as_str())
                .unwrap()
                .read_to_end(&mut rust_disk_layout)
                .unwrap();
            let mut truth_disk_layout: Vec<u8> = Vec::new();
            storage_provider
                .open_reader(TRUTH_DISK_LAYOUT_LOW_DIM)
                .unwrap()
                .read_to_end(&mut truth_disk_layout)
                .unwrap();

            // Assert that the metadata on rust disk index is compatible with the truth disk index.
            assert!(
                rust_disk_layout[8..TRUTH_DISK_LAYOUT_METADATA_LENGTH]
                    == truth_disk_layout[8..TRUTH_DISK_LAYOUT_METADATA_LENGTH]
            );

            // Assert that the rest of the disk index is identical with the truth disk index.
            assert!(
                rust_disk_layout[DEFAULT_DISK_SECTOR_LEN..]
                    == truth_disk_layout[DEFAULT_DISK_SECTOR_LEN..]
            );
            disk_layout_file
        };

        storage_provider
            .delete(disk_layout_file.as_str())
            .expect("Failed to delete file");
    }

    #[test]
    fn create_disk_layout_test_low_dim_with_associated_data() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());

        const TEST_DATA_FILE_LOW_DIM: &str = "/sift/siftsmall_learn_256pts.fbin";
        const DISK_INDEX_PATH_PREFIX_LOW_DIM_WITH_ASSOCIATED_DATA: &str = "/disk_index_misc/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_index_writer_with_associated_data";
        const TRUTH_DISK_LAYOUT_LOW_DIM: &str =
            "/disk_index_misc/truth_disk_index_siftsmall_learn_256pts_R4_L50_A1.2_disk.index";

        const ASSOCIATED_DATA_FILE: &str = "/sift/siftsmall_learn_256pts_u32_associated_data.fbin";

        let disk_layout_file = {
            let storage = DiskIndexWriter::new(
                TEST_DATA_FILE_LOW_DIM.to_string(),
                DISK_INDEX_PATH_PREFIX_LOW_DIM_WITH_ASSOCIATED_DATA.to_string(),
                Option::Some(ASSOCIATED_DATA_FILE.to_string()),
                DEFAULT_DISK_SECTOR_LEN,
            )
            .unwrap();

            storage
                .create_disk_layout::<GraphDataF32VectorU32Data, _>(&storage_provider)
                .unwrap();

            let disk_layout_file = storage.disk_index_file();

            let mut rust_disk_layout: Vec<u8> = Vec::new();
            storage_provider
                .open_reader(disk_layout_file.as_str())
                .unwrap()
                .read_to_end(&mut rust_disk_layout)
                .unwrap();
            let mut truth_disk_layout: Vec<u8> = Vec::new();
            storage_provider
                .open_reader(TRUTH_DISK_LAYOUT_LOW_DIM)
                .unwrap()
                .read_to_end(&mut truth_disk_layout)
                .unwrap();

            let mut associated_data: Vec<u8> = Vec::new();
            storage_provider
                .open_reader(ASSOCIATED_DATA_FILE)
                .unwrap()
                .read_to_end(&mut associated_data)
                .unwrap();

            compare_disk_index_graphs(&rust_disk_layout, &truth_disk_layout);
            compare_disk_index_graphs_associated_data::<u32>(&rust_disk_layout, &associated_data);
            disk_layout_file
        };

        storage_provider
            .delete(disk_layout_file.as_str())
            .expect("Failed to delete file");
    }

    #[test]
    fn create_disk_layout_test_high_dim() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());

        const TEST_DATA_FILE_HIGH_DIM: &str = "/disk_index_misc/rand_float_1024D_1K_norm1.0.bin";
        const DISK_INDEX_PATH_PREFIX_HIGH_DIM: &str =
            "/disk_index_misc/disk_index_rand_float_1024D_1Kpts_R4_L50_A1.2_index_writer";
        const TRUTH_DISK_LAYOUT_HIGH_DIM: &str =
            "/disk_index_misc/truth_disk_index_rand_float_1024D_1Kpts_R4_L50_A1.2_index_writer.index";

        let disk_layout_file = {
            let storage = DiskIndexWriter::new(
                TEST_DATA_FILE_HIGH_DIM.to_string(),
                DISK_INDEX_PATH_PREFIX_HIGH_DIM.to_string(),
                Option::None,
                DEFAULT_DISK_SECTOR_LEN,
            )
            .unwrap();

            storage
                .create_disk_layout::<GraphDataF32VectorU32Data, _>(&storage_provider)
                .unwrap();

            let disk_layout_file = storage.disk_index_file();
            let mut rust_disk_layout: Vec<u8> = Vec::new();
            storage_provider
                .open_reader(disk_layout_file.as_str())
                .unwrap()
                .read_to_end(&mut rust_disk_layout)
                .unwrap();
            let mut truth_disk_layout: Vec<u8> = Vec::new();
            storage_provider
                .open_reader(TRUTH_DISK_LAYOUT_HIGH_DIM)
                .unwrap()
                .read_to_end(&mut truth_disk_layout)
                .unwrap();

            // Assert that the metadata on rust disk index is compatible with the truth disk index.
            assert!(
                rust_disk_layout[8..TRUTH_DISK_LAYOUT_METADATA_LENGTH]
                    == truth_disk_layout[8..TRUTH_DISK_LAYOUT_METADATA_LENGTH]
            );

            // Assert that the rest of the disk index is identical with the truth disk index.
            assert!(
                rust_disk_layout[DEFAULT_DISK_SECTOR_LEN..]
                    == truth_disk_layout[DEFAULT_DISK_SECTOR_LEN..]
            );
            disk_layout_file
        };

        storage_provider
            .delete(disk_layout_file.as_str())
            .expect("Failed to delete file");
    }

    #[test]
    fn disk_index_writer_rejects_small_block_size() {
        let small_block_size = GraphMetadata::get_size() - 1;
        let result = DiskIndexWriter::new(
            "dataset".to_string(),
            "index".to_string(),
            Option::None,
            small_block_size,
        );

        assert!(result.is_err());
    }

    struct ExpectedWriter {
        dataset_file: String,
        index_path_prefix: String,
        pq_pivot_file: String,
        compressed_pq_file: String,
        disk_index_pq_pivot_file: String,
        disk_index_compressed_pq_file: String,
        associated_data_file: Option<String>,
    }

    fn assert_writer_eq_expected(writer: &DiskIndexWriter, expected: &ExpectedWriter) {
        assert_eq!(writer.dataset_file(), &expected.dataset_file);
        assert_eq!(writer.index_path_prefix(), &expected.index_path_prefix);

        assert_eq!(writer.get_pq_pivot_file(), expected.pq_pivot_file);
        assert_eq!(
            writer.get_compressed_pq_pivot_file(),
            expected.compressed_pq_file
        );
        assert_eq!(
            writer.get_disk_index_pq_pivot_file(),
            expected.disk_index_pq_pivot_file
        );
        assert_eq!(
            writer.get_disk_index_compressed_pq_file(),
            expected.disk_index_compressed_pq_file
        );
        assert_eq!(
            writer.get_associated_data_file(),
            expected.associated_data_file
        );
    }

    #[test]
    fn test_dataset_file_and_index_path_prefix() {
        let dataset_file_name = "dataset_file.txt";
        let index_path_prefix = "index_path_prefix";
        let associated_data_file = "associated_data_file.txt";
        let writer = DiskIndexWriter::new(
            dataset_file_name.to_string(),
            index_path_prefix.to_string(),
            Some(associated_data_file.to_string()),
            DEFAULT_DISK_SECTOR_LEN,
        )
        .unwrap();
        let expected = ExpectedWriter {
            dataset_file: dataset_file_name.to_string(),
            index_path_prefix: index_path_prefix.to_string(),
            pq_pivot_file: get_pq_pivot_file(index_path_prefix),
            compressed_pq_file: get_compressed_pq_file(index_path_prefix),
            disk_index_pq_pivot_file: get_disk_index_pq_pivot_file(index_path_prefix),
            disk_index_compressed_pq_file: get_disk_index_compressed_pq_file(index_path_prefix),
            associated_data_file: Some(associated_data_file.to_string()),
        };
        assert_writer_eq_expected(&writer, &expected);
    }

    #[test]
    fn test_get_mem_index_file() {
        let writer = DiskIndexWriter::new(
            "test_index".to_string(),
            "test_dataset".to_string(),
            Option::None,
            DEFAULT_DISK_SECTOR_LEN,
        )
        .unwrap();

        assert_eq!(writer.get_mem_index_file(), "test_dataset_mem.index");
    }

    #[test]
    fn test_get_merged_index_prefix() {
        let writer = DiskIndexWriter::new(
            "test_index".to_string(),
            "test_dataset".to_string(),
            Option::None,
            DEFAULT_DISK_SECTOR_LEN,
        )
        .unwrap();

        assert_eq!(
            writer.get_merged_index_prefix(),
            "test_dataset_mem.index_tempFiles"
        );
    }

    #[test]
    fn test_get_merged_index_subshard_id_map_file() {
        let prefix = "test_prefix";
        let shard = 5;
        assert_eq!(
            DiskIndexWriter::get_merged_index_subshard_id_map_file(prefix, shard),
            "test_prefix_subshard-5_ids_uint32.bin"
        );
    }

    #[test]
    fn test_get_merged_index_subshard_data_file() {
        let prefix = "test_prefix";
        let shard = 5;
        assert_eq!(
            DiskIndexWriter::get_merged_index_subshard_data_file(prefix, shard),
            "test_prefix_subshard-5.bin"
        );
    }

    #[test]
    fn test_get_merged_index_subshard_prefix() {
        let prefix = "test_prefix";
        let shard = 5;
        assert_eq!(
            DiskIndexWriter::get_merged_index_subshard_prefix(prefix, shard),
            "test_prefix_subshard-5"
        );
    }

    #[test]
    fn test_get_merged_index_subshard_mem_index_file() {
        let prefix = "test_prefix";
        let shard = 5;
        assert_eq!(
            DiskIndexWriter::get_merged_index_subshard_mem_index_file(prefix, shard),
            "test_prefix_subshard-5_mem.index"
        );
    }

    #[test]
    fn test_get_merged_index_subshard_mem_dataset_file() {
        let prefix = "test_prefix";
        assert_eq!(
            DiskIndexWriter::get_merged_index_subshard_mem_dataset_file(prefix),
            "test_prefix.data"
        );
    }

    #[test]
    fn test_disk_index_writer_state_uninitialized() {
        let mut state = DiskIndexWriterState::<VirtualStorageProvider<OverlayFS>>::new();
        let mut buf = vec![0u8; 16];
        assert!(DiskIndexWriter::get_neighbors_number(&mut state).is_err());
        assert!(DiskIndexWriter::copy_neighbors(&mut state, &mut buf).is_err());
    }

    // Compare that the index built in test is the same as the truth index. The truth index doesn't have associated data, we are only comparing the vector and neighbor data.
    pub fn compare_disk_index_graphs(graph_data: &[u8], truth_graph_data: &[u8]) {
        let graph_header = GraphHeader::try_from(&graph_data[8..]).unwrap();
        let truth_graph_header = GraphHeader::try_from(&truth_graph_data[8..]).unwrap();

        let test_node_per_block = graph_header.metadata().num_nodes_per_block;
        let test_max_node_length = graph_header.metadata().node_len;

        let truth_node_per_block = truth_graph_header.metadata().num_nodes_per_block;
        let truth_max_node_length = truth_graph_header.metadata().node_len;

        assert_eq!(
            graph_header.metadata().num_pts,
            truth_graph_header.metadata().num_pts
        );

        assert_eq!(
            graph_header.metadata().dims,
            truth_graph_header.metadata().dims
        );

        let num_pts = graph_header.metadata().num_pts as usize;
        let dim = graph_header.metadata().dims;

        for idx in 0..num_pts {
            let test_node_id_offset = node_data_offset(
                idx,
                test_max_node_length as usize,
                test_node_per_block as usize,
                DEFAULT_DISK_SECTOR_LEN,
            );

            let truth_node_id_offset = node_data_offset(
                idx,
                truth_max_node_length as usize,
                truth_node_per_block as usize,
                DEFAULT_DISK_SECTOR_LEN,
            );

            // Assert that the vector data is the same between the test and truth graphs for this node.
            assert_eq!(
                &graph_data
                    [test_node_id_offset..test_node_id_offset + dim * std::mem::size_of::<f32>()],
                &truth_graph_data
                    [truth_node_id_offset..truth_node_id_offset + dim * std::mem::size_of::<f32>()]
            );

            // Assert that the neighbor count is the same between the test and truth graphs for this node.
            let test_nbr_cnt_offset = test_node_id_offset + dim * std::mem::size_of::<f32>();
            let truth_nbr_cnt_offset = truth_node_id_offset + dim * std::mem::size_of::<f32>();

            let test_nbr_count = u32::from_le_bytes([
                graph_data[test_nbr_cnt_offset],
                graph_data[test_nbr_cnt_offset + 1],
                graph_data[test_nbr_cnt_offset + 2],
                graph_data[test_nbr_cnt_offset + 3],
            ]);

            let truth_nbr_count = u32::from_le_bytes([
                truth_graph_data[truth_nbr_cnt_offset],
                truth_graph_data[truth_nbr_cnt_offset + 1],
                truth_graph_data[truth_nbr_cnt_offset + 2],
                truth_graph_data[truth_nbr_cnt_offset + 3],
            ]);

            assert_eq!(test_nbr_count, truth_nbr_count);

            // Assert the neighbors (u32) are the same between the test and truth graphs for this node.
            let test_nbr_offset = test_nbr_cnt_offset + 4;
            let truth_nbr_offset = truth_nbr_cnt_offset + 4;
            assert_eq!(
                graph_data[test_nbr_offset..test_nbr_offset + test_nbr_count as usize * 4],
                truth_graph_data[truth_nbr_offset..truth_nbr_offset + truth_nbr_count as usize * 4]
            );
        }
    }

    // Compare that the associated data in the index graph built in test is the same as the associated data input.
    pub fn compare_disk_index_graphs_associated_data<AssociatedDataType>(
        graph_data: &[u8],
        associated_data: &[u8],
    ) {
        let graph_header = GraphHeader::try_from(&graph_data[8..]).unwrap();
        let test_node_per_block = graph_header.metadata().num_nodes_per_block;
        let test_max_node_length = graph_header.metadata().node_len as usize;

        let mut associated_data_offset = 0;
        let data_npts = u32::from_le_bytes([
            associated_data[associated_data_offset],
            associated_data[associated_data_offset + 1],
            associated_data[associated_data_offset + 2],
            associated_data[associated_data_offset + 3],
        ]) as usize;

        associated_data_offset = 4;
        let _associated_data_length = u32::from_le_bytes([
            associated_data[associated_data_offset],
            associated_data[associated_data_offset + 1],
            associated_data[associated_data_offset + 2],
            associated_data[associated_data_offset + 3],
        ]) as usize;

        let num_pts = graph_header.metadata().num_pts as usize;
        assert_eq!(num_pts, data_npts);

        associated_data_offset = 8;

        for idx in 0..num_pts {
            let test_node_id_offset = node_data_offset(
                idx,
                test_max_node_length,
                test_node_per_block as usize,
                DEFAULT_DISK_SECTOR_LEN,
            );

            let node_buf_end = test_node_id_offset + test_max_node_length;

            assert_eq!(
                graph_data[(node_buf_end - mem::size_of::<AssociatedDataType>())..node_buf_end],
                associated_data[associated_data_offset
                    ..associated_data_offset + mem::size_of::<AssociatedDataType>()]
            );

            associated_data_offset += mem::size_of::<AssociatedDataType>();
        }
    }

    pub fn node_data_offset(
        node_id: usize,
        node_length: usize,
        nodes_per_block: usize,
        block_size: usize,
    ) -> usize {
        let block_id = node_id / nodes_per_block;
        let node_id_in_block = node_id % nodes_per_block;
        let offset = block_id * block_size + node_id_in_block * node_length;
        offset + block_size
    }
}
