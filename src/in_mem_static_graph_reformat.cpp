#include "in_mem_static_graph_reformat_store.h"
#include "utils.h"

namespace diskann
{

std::tuple<uint32_t, uint32_t, size_t> InMemStaticGraphReformatStore::load_impl(const std::string& filename, size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;
    size_t num_points;
    size_t file_offset = 0; // will need this for single file format support

    FileReader file_reader;
    if (!file_reader.Open(filename))
    {
        throw diskann::ANNException("fail to open file ", -1, __FUNCSIG__,
            __FILE__, __LINE__);
    }

    // get file size
    size_t check_file_size = file_reader.GetFileSize();

    // Create fixed-size aligned buffer for 100,000 sectors
    const size_t SECTOR_SIZE = 4096;
    const size_t BUFFER_SECTORS = 100000;
    const size_t BUFFER_SIZE = BUFFER_SECTORS * SECTOR_SIZE; // ~409.6 MB
    
    char* aligned_buffer = nullptr;
    alloc_aligned((void**)&aligned_buffer, BUFFER_SIZE, SECTOR_SIZE);
    
    try {
        // Read metadata header (aligned to sector boundary)
        size_t metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t) + sizeof(size_t);
        size_t aligned_metadata_size = ROUND_UP(metadata_size, SECTOR_SIZE);
        
        if (!file_reader.Read(0, aligned_metadata_size, aligned_buffer)) {
            throw diskann::ANNException("Failed to read graph metadata", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        
        // Extract metadata from aligned buffer
        char* metadata_ptr = aligned_buffer;
        expected_file_size = *reinterpret_cast<size_t*>(metadata_ptr);
        metadata_ptr += sizeof(size_t);
        _max_observed_degree = *reinterpret_cast<uint32_t*>(metadata_ptr);
        metadata_ptr += sizeof(uint32_t);
        start = *reinterpret_cast<uint32_t*>(metadata_ptr);
        metadata_ptr += sizeof(uint32_t);
        file_frozen_pts = *reinterpret_cast<size_t*>(metadata_ptr);
        metadata_ptr += sizeof(size_t);
        num_points = *reinterpret_cast<size_t*>(metadata_ptr);
        
        file_offset = metadata_size;

    // max observed degree is the max degree of the graph
    _max_range_of_graph = _max_observed_degree;

    size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t) + sizeof(size_t);

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
        << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
        << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    if (check_file_size != expected_file_size)
    {
        std::stringstream stream;
        stream << "Vamana Index file size does not match expected size per "
                    "meta-data."
                << " file size from file: " << expected_file_size << " actual file size: " << check_file_size << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "Loading vamana graph " << filename << "..." << std::flush;

    // Load node index with chunked aligned reads
    _node_index.resize(num_points + 1);
    size_t node_index_size = _node_index.size() * sizeof(size_t);
    
    // Calculate aligned offset for node index data
    size_t aligned_node_offset = ROUND_DOWN(file_offset, SECTOR_SIZE);
    size_t node_offset_adjustment = file_offset - aligned_node_offset;
    size_t total_node_bytes_to_read = node_index_size + node_offset_adjustment;
    
    // Read node index in chunks
    size_t node_bytes_read = 0;
    size_t node_elements_copied = 0;
    
    while (node_bytes_read < total_node_bytes_to_read) {
        size_t chunk_size = std::min(BUFFER_SIZE, ROUND_UP(total_node_bytes_to_read - node_bytes_read, SECTOR_SIZE));
        size_t current_offset = aligned_node_offset + node_bytes_read;
        
        if (!file_reader.Read(current_offset, chunk_size, aligned_buffer)) {
            throw diskann::ANNException("Failed to read node index chunk", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        
        const char* chunk_data_start = aligned_buffer;
        if (node_bytes_read == 0) {
            // First chunk - account for offset
            chunk_data_start += node_offset_adjustment;
        }
        
        size_t available_data = chunk_size;
        if (node_bytes_read == 0) {
            available_data -= node_offset_adjustment;
        }
        
        size_t elements_in_chunk = std::min(available_data / sizeof(size_t), _node_index.size() - node_elements_copied);
        std::memcpy(_node_index.data() + node_elements_copied, chunk_data_start, elements_in_chunk * sizeof(size_t));
        
        node_elements_copied += elements_in_chunk;
        node_bytes_read += (elements_in_chunk * sizeof(size_t));
        if (node_bytes_read == 0 && node_offset_adjustment > 0) {
            node_bytes_read = node_offset_adjustment;
        }
    }
    
    file_offset += node_index_size;

    _graph_size = _node_index[num_points] * sizeof(std::uint32_t);
    
    size_t total_neighbors = _node_index[num_points];
    // add one more slot than actually need to avoid read invaild address
    // while the last point is no neighbor
    _graph.resize(total_neighbors + 1);
    
    // Load graph data with chunked aligned reads
    size_t aligned_graph_offset = ROUND_DOWN(file_offset, SECTOR_SIZE);
    size_t graph_offset_adjustment = file_offset - aligned_graph_offset;
    size_t total_graph_bytes_to_read = _graph_size + graph_offset_adjustment;
    
    // Read graph data in chunks
    size_t graph_bytes_read = 0;
    size_t graph_elements_copied = 0;
    
    while (graph_bytes_read < total_graph_bytes_to_read) {
        size_t chunk_size = std::min(BUFFER_SIZE, ROUND_UP(total_graph_bytes_to_read - graph_bytes_read, SECTOR_SIZE));
        size_t current_offset = aligned_graph_offset + graph_bytes_read;
        
        if (!file_reader.Read(current_offset, chunk_size, aligned_buffer)) {
            throw diskann::ANNException("Failed to read graph data chunk", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        
        const char* chunk_data_start = aligned_buffer;
        if (graph_bytes_read == 0) {
            // First chunk - account for offset
            chunk_data_start += graph_offset_adjustment;
        }
        
        size_t available_data = chunk_size;
        if (graph_bytes_read == 0) {
            available_data -= graph_offset_adjustment;
        }
        
        size_t elements_in_chunk = std::min(available_data / sizeof(std::uint32_t), _graph.size() - graph_elements_copied);
        std::memcpy(_graph.data() + graph_elements_copied, chunk_data_start, elements_in_chunk * sizeof(std::uint32_t));
        
        graph_elements_copied += elements_in_chunk;
        graph_bytes_read += (elements_in_chunk * sizeof(std::uint32_t));
        if (graph_bytes_read == 0 && graph_offset_adjustment > 0) {
            graph_bytes_read = graph_offset_adjustment;
        }
    }
    
    aligned_free(aligned_buffer);
    
    diskann::cout << "done. Index has " << num_points << " nodes and " << total_neighbors << " out-edges, _start is set to " << start
        << std::endl;
    } catch (...) {
        aligned_free(aligned_buffer);
        throw;
    }

    
    return std::make_tuple(static_cast<std::uint32_t>(num_points), start, file_frozen_pts);
}

}