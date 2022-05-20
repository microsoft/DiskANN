// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
// #include <sync_index.h>
#include <future>
#include <time.h>
#include <timer.h>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

// load_aligned_bin modified to read pieces of the file, but using ifstream instead of cached_ifstream.
template<typename T>
inline void load_aligned_bin_part(const std::string& bin_file, T* data, size_t offset_points, size_t points_to_read) {
    diskann::Timer timer;
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    reader.seekg(0, std::ios::beg);

    int npts_i32, dim_i32;
    reader.read((char*)&npts_i32, sizeof(int));
    reader.read((char*)&dim_i32, sizeof(int));
    size_t npts = (unsigned)npts_i32;
    size_t dim = (unsigned)dim_i32;

    size_t expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
        std::stringstream stream;
        stream << "Error. File size mismatch. Actual size is " << actual_file_size
            << " while expected size is  " << expected_actual_file_size
            << " npts = " << npts << " dim = " << dim
            << " size of <T>= " << sizeof(T) << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
            __LINE__);
    }

    if (offset_points + points_to_read > npts) {
        std::stringstream stream;
        stream << "Error. Not enough points in file. Requested " << offset_points
            << "  offset and " << points_to_read << " points, but have only "
            << npts << " points" << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
            __LINE__);
    }

    reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

    const size_t rounded_dim = ROUND_UP(dim, 8);;

    for (size_t i = 0; i < points_to_read; i++) {
        reader.read((char*)(data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }

    const double elapsedSeconds = timer.elapsed() / 1000000.0;
    std::cout << "Read " << points_to_read << " points using non-cached reads in " << elapsedSeconds << std::endl;
}



template<typename T>
int build_incremental_index(const std::string& data_path, const unsigned L,
                            const unsigned R, const unsigned C,
                            const unsigned num_rnds, const float alpha,
                            const std::string& save_path,
                            const unsigned     num_frozen,
                            const unsigned      threads) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", false);
  paras.Set<unsigned>("num_rnds", num_rnds);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  typedef int TagT;

  std::cout << "loading complete" << std::endl; 

  // Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
  //                       const bool dynamic_index,
  //                       const bool enable_tags, const bool support_eager_delete)

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, true, true,
                                 false);
  int64_t deletes_offset = (uint64_t) (.5*num_points);
  // int64_t deletes_start = (int64_t) (0);
  int64_t deletes_size = (int64_t) (.25*num_points);
  // int64_t deletes_restart = (int64_t) (.75*num_points);
  {
    //tags need to start at .5*data_size, increase to .75*data_size, then 0-.25*data_size
    std::vector<TagT> tags((int64_t) num_points*.5);
    // std::iota(tags.begin(), tags.end(), 0);
    std::iota(tags.begin(), tags.begin()+deletes_size, deletes_offset);
    std::iota(tags.begin()+deletes_size, tags.end(), 0);
    // std::iota(tags.begin()+deletes_restart, tags.end(), deletes_restart);

    // std::cout << tags[0] << " " << tags[deletes_pause-1] << " " << tags[deletes_pause] << " " << tags[deletes_restart-1] << " "
    //   << tags[deletes_restart] << " " << tags[num_points-1] << std::endl;     

    diskann::Timer timer;
    index.build(data_path.c_str(), (int64_t) num_points*.5, paras, tags);
    std::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";
  }

    

  std::vector<unsigned> delete_vector(deletes_size);
  std::iota(delete_vector.begin(), delete_vector.end(), deletes_offset);


  // int threads = 16;
  int sub_threads = (threads + 1) / 2;

  int64_t inserts_size = (int64_t) (.25*num_points);
  int64_t inserts_start = (int64_t) (.5*num_points);
  int64_t inserts_end = (int64_t) (.75*num_points);
  std::vector<TagT> insert_tags(inserts_size);
  std::iota(insert_tags.begin(), insert_tags.end(), inserts_size);

  {
    diskann::Timer timer;
    index.enable_delete();

    auto inserts = std::async(std::launch::async, [&] (){
      #pragma omp        parallel for num_threads(sub_threads)
        for (int64_t j = inserts_start; j < inserts_end; ++j) {
          index.insert_point(data_load + j * aligned_dim, paras, insert_tags[j-inserts_start]);
        }
    });



    auto deletes = std::async(std::launch::async, [&] (){
      #pragma omp parallel for num_threads(sub_threads)
        for (size_t i = 0; i < delete_vector.size(); i++) {
          unsigned p = delete_vector[i];
          if (index.lazy_delete(p) != 0)
            std::cerr << "Delete tag " << p << " not found" << std::endl;
        }
      if (index.disable_delete_2(paras, true) != 0) {
        std::cerr << "Disable delete failed" << std::endl;
      }
    });

    inserts.wait(); 
    deletes.wait();

  
    std::cout << "Time Elapsed" << timer.elapsed() / 1000 << "ms\n";

  }

      index.save(save_path.c_str());


  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 11) {
    std::cout << "Correct usage: " << argv[0]
              << " type[int8/uint8/float] data_file L R C alpha "
                 "num_rounds "
              << "save_graph_file  #frozen_points #threads" << std::endl;
    exit(-1);
  }

  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  float       alpha = (float) std::atof(argv[6]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[7]);
  std::string save_path(argv[8]);
  // unsigned    num_incr = (unsigned) atoi(argv[9]);
  unsigned    num_frozen = (unsigned) atoi(argv[9]);
  unsigned    num_threads = (unsigned) atoi(argv[10]);

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t>(argv[2], L, R, C, num_rnds, alpha,
                                    save_path, num_frozen, num_threads);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t>(argv[2], L, R, C, num_rnds, alpha,
                                     save_path, num_frozen, num_threads);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float>(argv[2], L, R, C, num_rnds, alpha, save_path,
                                   num_frozen, num_threads);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
