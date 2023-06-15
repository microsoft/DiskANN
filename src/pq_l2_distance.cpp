#include "pq.h"
#include "pq_l2_distance.h"


// block size for reading/processing large files and matrices in blocks
#define BLOCK_SIZE 5000000

namespace diskann
{

  template <typename data_t> PQL2Distance<data_t>::PQL2Distance(uint32_t num_chunks)
  {
  }

  template <typename data_t> PQL2Distance<data_t>::PQL2Distance()
  {
  #ifndef EXEC_ENV_OLS
      if (tables != nullptr)
          delete[] tables;
      if (chunk_offsets != nullptr)
          delete[] chunk_offsets;
      if (centroid != nullptr)
          delete[] centroid;
      if (rotmat_tr != nullptr)
          delete[] rotmat_tr;
  #endif
      if (tables_tr != nullptr)
          delete[] tables_tr;
  }

  template<typename data_t>
  bool PQL2Distance<data_t>::is_opq() const {
      return false;
  }
  

  //REFACTOR TODO: Undefined behavior if _num_chunks is not set.
  template<typename data_t>
  std::string PQL2Distance<data_t>::get_quantized_vectors_filename(const std::string &prefix) const
  {
      return diskann::get_quantized_vectors_filename(prefix, false, _num_chunks);
  }
  template <typename data_t> std::string PQL2Distance<data_t>::get_pivot_data_filename(const std::string &prefix) const
  {
      return diskann::get_table_data_filename(prefix, false, _num_chunks);
  }
  template <typename data_t>
  std::string PQL2Distance<data_t>::get_rotation_matrix_filename(const std::string &prefix) const
  {
    //REFACTOR TODO: Currently, we are assuming that PQ doesn't have a rotation matrix.
      return "";
  }



  #ifdef EXEC_ENV_OLS
  template <typename data_t>
  void PQL2Distance<data_t>::load_pivot_data(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks)
  {
  #else
  template <typename data_t> void PQL2Distance<data_t>::load_pivot_data(const char *pq_table_file, size_t num_chunks)
  {
  #endif
      uint64_t nr, nc;
      std::string rotmat_file = get_opq_rot_matrix_filename(pq_table_file);

  #ifdef EXEC_ENV_OLS
      size_t *file_offset_data; // since load_bin only sets the pointer, no need
      // to delete.
      diskann::load_bin<size_t>(files, pq_table_file, file_offset_data, nr, nc);
  #else
      std::unique_ptr<size_t[]> file_offset_data;
      diskann::load_bin<size_t>(pq_table_file, file_offset_data, nr, nc);
  #endif

      bool use_old_filetype = false;

      if (nr != 4 && nr != 5)
      {
          diskann::cout << "Error reading pq_pivots file " << pq_table_file
                        << ". Offsets dont contain correct metadata, # offsets = " << nr << ", but expecting " << 4
                        << " or " << 5;
          throw diskann::ANNException("Error reading pq_pivots file at offsets data.", -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
      }

      if (nr == 4)
      {
          diskann::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " " << file_offset_data[2]
                        << " " << file_offset_data[3] << std::endl;
      }
      else if (nr == 5)
      {
          use_old_filetype = true;
          diskann::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " " << file_offset_data[2]
                        << " " << file_offset_data[3] << file_offset_data[4] << std::endl;
      }
      else
      {
          throw diskann::ANNException("Wrong number of offsets in pq_pivots", -1, __FUNCSIG__, __FILE__, __LINE__);
      }

  #ifdef EXEC_ENV_OLS
      diskann::load_bin<float>(files, pq_table_file, tables, nr, nc, file_offset_data[0]);
  #else
      diskann::load_bin<float>(pq_table_file, tables, nr, nc, file_offset_data[0]);
  #endif

      if ((nr != NUM_PQ_CENTROIDS))
      {
          diskann::cout << "Error reading pq_pivots file " << pq_table_file << ". file_num_centers  = " << nr
                        << " but expecting " << NUM_PQ_CENTROIDS << " centers";
          throw diskann::ANNException("Error reading pq_pivots file at pivots data.", -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
      }

      this->ndims = nc;

  #ifdef EXEC_ENV_OLS
      diskann::load_bin<float>(files, pq_table_file, centroid, nr, nc, file_offset_data[1]);
  #else
      diskann::load_bin<float>(pq_table_file, centroid, nr, nc, file_offset_data[1]);
  #endif

      if ((nr != this->ndims) || (nc != 1))
      {
          diskann::cerr << "Error reading centroids from pq_pivots file " << pq_table_file << ". file_dim  = " << nr
                        << ", file_cols = " << nc << " but expecting " << this->ndims << " entries in 1 dimension.";
          throw diskann::ANNException("Error reading pq_pivots file at centroid data.", -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
      }

      int chunk_offsets_index = 2;
      if (use_old_filetype)
      {
          chunk_offsets_index = 3;
      }
  #ifdef EXEC_ENV_OLS
      diskann::load_bin<uint32_t>(files, pq_table_file, chunk_offsets, nr, nc, file_offset_data[chunk_offsets_index]);
  #else
      diskann::load_bin<uint32_t>(pq_table_file, chunk_offsets, nr, nc, file_offset_data[chunk_offsets_index]);
  #endif

      if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0))
      {
          diskann::cerr << "Error loading chunk offsets file. numc: " << nc << " (should be 1). numr: " << nr
                        << " (should be " << num_chunks + 1 << " or 0 if we need to infer)" << std::endl;
          throw diskann::ANNException("Error loading chunk offsets file", -1, __FUNCSIG__, __FILE__, __LINE__);
      }

      this->_num_chunks = nr - 1;
      diskann::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS << ", #dims: " << this->ndims
                    << ", #chunks: " << this->_num_chunks << std::endl;

      if (file_exists(rotmat_file))
      {
  #ifdef EXEC_ENV_OLS
          diskann::load_bin<float>(files, rotmat_file, (float *&)rotmat_tr, nr, nc);
  #else
          diskann::load_bin<float>(rotmat_file, rotmat_tr, nr, nc);
  #endif
          if (nr != this->ndims || nc != this->ndims)
          {
              diskann::cerr << "Error loading rotation matrix file" << std::endl;
              throw diskann::ANNException("Error loading rotation matrix file", -1, __FUNCSIG__, __FILE__, __LINE__);
          }
          use_rotation = true;
      }

      // alloc and compute transpose
      tables_tr = new float[256 * this->ndims];
      for (size_t i = 0; i < 256; i++)
      {
          for (size_t j = 0; j < this->ndims; j++)
          {
              tables_tr[j * 256 + i] = tables[i * this->ndims + j];
          }
      }
  }

  template <typename data_t> uint32_t PQL2Distance<data_t>::get_num_chunks()
  {
      return static_cast<uint32_t>(_num_chunks);
  }

  template <typename data_t> void PQL2Distance<data_t>::preprocess_query(float *query_vec, uint32_t ndims)
  {
      for (uint32_t d = 0; d < ndims; d++)
      {
          query_vec[d] -= centroid[d];
      }
      std::vector<float> tmp(ndims, 0);
      if (use_rotation)
      {
          for (uint32_t d = 0; d < ndims; d++)
          {
              for (uint32_t d1 = 0; d1 < ndims; d1++)
              {
                  tmp[d] += query_vec[d1] * rotmat_tr[d1 * ndims + d];
              }
          }
          std::memcpy(query_vec, tmp.data(), ndims * sizeof(float));
      }
  }

  template<typename data_t>
  float PQL2Distance<data_t>::preprocessed_distance(PQScratch<data_t> &pq_scratch,
                                                    const uint32_t n_ids,                 
                                                    float* dists_out)
  {
      pq_dist_lookup(pq_scratch->aligned_pq_coord_scratch, n_ids, _num_chunks,
                     pq_scratch->aligned_pqtable_dist_scratch, dists_out);
  }

  template <typename data_t> float PQL2Distance<data_t>::brute_force_distance(const float *query_vec, uint8_t *base_vec)
  {
      float res = 0;
      for (size_t chunk = 0; chunk < n_chunks; chunk++)
      {
          for (size_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++)
          {
              const float *centers_dim_vec = tables_tr + (256 * j);
              float diff = centers_dim_vec[base_vec[chunk]] - (query_vec[j]);
              res += diff * diff;
          }
      }
      return res;
  }
}