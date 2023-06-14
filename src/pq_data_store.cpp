#include <exception>

#include "pq_data_store.h"
#include "pq.h"
#include "utils.h"
#include "distance.h"

namespace diskann {

//REFACTOR TODO: _use_opq should be removed and OPQ stuff added to a separate class.
template <typename data_t>
PQDataStore<data_t>::PQDataStore(uint32_t dim, uint32_t num_pq_chunks,
                                 std::shared_ptr<Distance<data_t>> distance_fn, std::shared_ptr<AbstractPQDistance<data_t>> pq_distance_fn)
    : AbstractDataStore(0, dim),
      _num_pq_chunks(num_pq_chunks),
      _quantized_data(nullptr),
      _distance_metric(distance_fn->get_metric()), 
      _pq_distance_fn(pq_distance_fn),
      _use_opq(false) {}

template <typename data_t>
location_t PQDataStore<data_t>::load(const std::string& filename) {
  return load_impl(filename);
}
template <typename data_t>
size_t PQDataStore<data_t>::save(const std::string& filename,
                         const location_t num_points) {
  diskann::save_bin(filename, _data, _num_points, _num_chunks, 0);
}

template<typename data_t>
size_t PQDataStore<data_t>::get_aligned_dim() const { 
  return _dim;
}

// Populate quantized data from regular data.
template<typename data_t>
void PQDataStore<data_t>::populate_data(const data_t* vectors, const location_t num_pts) {
  throw std::logic_error("Not implemented yet");
}

template<typename data_t>
 void PQDataStore<data_t>::populate_data(const std::string& filename, const size_t offset) {
  if (_quantized_data != nullptr) {
    aligned_free(_quantized_data);
  }
  double p_val = std::min(
      1.0, ((double)MAX_PQ_TRAINING_SET_SIZE / (double)file_num_points));

  auto pq_pivots_file =
      diskann::get_pq_pivots_filename(filename, _use_opq, _num_pq_chunks);
  auto pq_compressed_file = diskann::get_pq_compressed_vectors_filename(
      filename, _use_opq, _num_pq_chunks);

  // REFACTOR TODO: Split OPQ and PQ into classes.
  generate_quantized_data<T>(std::string(filename), pq_pivots_file,
                             pq_compressed_file, _distance_metric, p_val,
                             _num_pq_chunks, _use_opq);

  copy_aligned_data_from_file<uint8_t>(pq_compressed_file.c_str(),
                                       _quantized_data, file_num_points,
                                       _num_pq_chunks, _num_pq_chunks);
#ifdef EXEC_ENV_OLS
  throw ANNException(
      "load_pq_centroid_bin should not be called when "
      "EXEC_ENV_OLS is defined.",
      -1, __FUNCSIG__, __FILE__, __LINE__);
#else
  _pq_table.load_pq_centroid_bin(pq_pivots_file.c_str(), _num_pq_chunks);
#endif
}

 template<typename data_t>
 void PQDataStore<data_t>::extract_data_to_bin(const std::string& filename,
   const location_t num_pts) {
  throw std::logic_error("Not implemented yet");
}

 template<typename data_t>
 void PQDataStore<data_t>::get_vector(const location_t i, data_t* target) const {
   //REFACTOR TODO: Should we inflate the compressed vector here? 
  if (i < _num_points) {
    throw std::logic_error("Not implemented yet");
  } else {
    std::stringstream ss;
    ss << "Requested vector " << i << " but only  " << _num_points
       << " vectors are present";
    throw diskann::ANNException(ss.str(), -1);
  }
}
 template <typename data_t>
 void PQDataStore<data_t>::set_vector(const location_t i, const data_t* const vector) {
   //REFACTOR TODO: Should we accept a normal vector and compress here? 
   //memcpy (_data + i * _num_chunks, vector, _num_chunks * sizeof(data_t));
  throw std::logic_error("Not implemented yet");
 }

 template<typename data_t>
 void PQDataStore<data_t>::prefetch_vector(const location_t loc) {
   const uint8_t* ptr = _data + ((size_t) loc)*_num_chunks*sizeof(data_t);
   diskann::prefetch_vector(ptr, _num_chunks * sizeof(data_t));
 }

 template<typename data_t>
 void PQDataStore<data_t>::move_vectors(const location_t old_location_start,
   const location_t new_location_start,
   const location_t num_points) {
   //REFACTOR TODO: Moving vectors is only for in-mem fresh. 
   throw std::logic_error("Not implemented yet");
 }

 template<typename data_t>
 void PQDataStore<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc,
   const location_t num_points) {
   throw std::logic_error("Not implemented yet");
 }

 template<typename data_t>
 float PQDataStore<data_t>::get_distance(const data_t* query,
   const location_t loc) const
 {
    throw std::logic_error("Not implemented yet");
 }

 template<typename data_t>
 float PQDataStore<data_t>::get_distance(const location_t loc1,
   const location_t loc2) const {
    throw std::logic_error("Not implemented yet");
 }

 template<typename data_t>
 void PQDataStore<data_t>::get_distance(const data_t* query, const location_t* locations,
   const uint32_t location_count,
   float* distances, AbstractScratch* scratch_space) const {
    if (scratch_space == nullptr) {
      throw diskann::ANNException("Scratch space is null", -1);
    }
    PQScratch<data_t>* scratch = dynamic_cast<PQScratch<data_t>>(scratch_space);
    if (scratch == nullptr) {
      throw diskann::ANNException("Scratch space is not of type PQScratch", -1);
    }
    _pq_distance_fn->preprocess_query(query, get_dims(), scratch);
    diskann::aggregate_coords(locations, _quantized_data, this->_num_pq_chunks,
                              scratch->aligned_pq_coord_scratch);
    diskann::pq_dist_lookup(scratch->aligned_pq_coord_scratch, locations_count, this->_num_pq_chunks,
                            scratch->aligned_dist_scratch, dists_out);
 }


 location_t PQDataStore::calculate_medoid() const {
   throw std::logic_error("Not implemented yet");
}

 Distance<data_t>* PQDataStore::get_dist_fn() const {

}

 size_t get_alignment_factor() const { return 1; }


virtual location_t load_impl(const std::string& filename);
#ifdef EXEC_ENV_OLS
virtual location_t load_impl(AlignedFileReader& reader);
#endif
}  // namespace diskann