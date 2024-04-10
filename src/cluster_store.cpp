// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <memory>
#include "abstract_scratch.h"
#include "cluster_store.h"
#include "math_utils.h"
#include "utils.h"

namespace diskann
{

template <typename data_t>
InMemClusterStore<data_t>::InMemClusterStore(const size_t dim)
    : AbstractClusterStore<data_t>(dim)
{
}

template <typename data_t> InMemClusterStore<data_t>::~InMemClusterStore() {
    delete[] this->_cluster_centroids;
}

template <typename data_t> uint32_t InMemClusterStore<data_t>::load(const std::string &filename) {
    std::string centers_file(filename);
    std::string posting_file(filename);
    centers_file += "_centers.bin";
    posting_file += "_posting.bin";

    if (!file_exists(posting_file))
     return 0;

    diskann::load_bin<float>(centers_file, this->_cluster_centroids,
                                   this->_num_clusters, this->_dim);

    std::ifstream in(posting_file, std::ios::binary | std::ios::in);
    uint64_t          total_count = 0;

    _posting_lists.resize(this->_num_clusters);
    for (unsigned i = 0; i < this->_num_clusters; i++) {
      unsigned cur_count;
      in.read((char *) &cur_count, sizeof(unsigned));
      uint32_t* vals = new uint32_t[cur_count];
      in.read((char *) vals, (uint64_t)cur_count * sizeof(unsigned));
      roaring_bitmap_add_many((roaring_bitmap_t*)_posting_lists[i].get_bitmap(), cur_count, vals);
      delete[] vals;
      total_count += cur_count;
    }
    in.close();

    uint64_t chksum = 0;
    uint32_t idx = 0;
    for (auto &x : _posting_lists) {
        if (x.size() ==0)
        std::cout<< idx << " ";
        idx++;
    }

    std::cout << "Read a total of " << total_count
                << " points from inverted index file." << std::endl;
    return this->_num_clusters;
}

template <typename data_t> size_t InMemClusterStore<data_t>::save(const std::string &filename) {
    std::string centers_file(filename);
    std::string posting_file(filename);
    centers_file += "_centers.bin";
    posting_file += "_posting.bin";
    diskann::save_bin<float>(centers_file, this->_cluster_centroids, this->_num_clusters,
                           this->_dim);

    std::ofstream out(posting_file, std::ios::binary | std::ios::out);
    uint64_t          total_count = 0;

    for (unsigned i = 0; i < this->_num_clusters; i++) {
      uint32_t count = (uint32_t) _posting_lists[i].size();
      out.write((char *) &count, sizeof(unsigned));

      uint32_t* arr = new uint32_t[count];
      auto x = (roaring_bitmap_t*) _posting_lists[i].get_bitmap();
      roaring_bitmap_to_uint32_array(x, arr);
      out.write((char *) arr, sizeof(unsigned)*(uint64_t)count);
      total_count += count;
      delete[] arr;
    }
    out.close();

    std::cout << "Written a total of " << total_count
                << " points to file." << std::endl;
    return total_count;
}

template <typename data_t> void InMemClusterStore<data_t>::add_cetroids(float *clusters, uint32_t num_clusters) {
    std::cout<<"Inside centroid addition: dim is " << this->_dim << std::endl;
    this->_num_clusters = num_clusters;
    this->_cluster_centroids = new float[(uint64_t)num_clusters*this->_dim];
    std::memcpy(this->_cluster_centroids, clusters, (uint64_t)num_clusters*this->_dim);
    _posting_lists.clear();
    _posting_lists.resize(num_clusters);
}

template <typename data_t> void InMemClusterStore<data_t>::assign_data_to_clusters(const data_t *vectors, std::vector<uint32_t> &ids) {
    uint64_t num_pts = ids.size();
    float* vectors_float = new float[num_pts*this->_dim];
    diskann::convert_types<data_t, float>(vectors, vectors_float, num_pts, this->_dim);

    uint32_t* closest_centers = new uint32_t[num_pts];
    math_utils::compute_closest_centers(vectors_float, num_pts, this->_dim, this->_cluster_centroids, this->_num_clusters, 1,
                                            closest_centers);

    for (uint64_t pos = 0; pos < num_pts; pos++) {
        _posting_lists[closest_centers[pos]].add(ids[pos]);
    }
    delete[] closest_centers;
    delete[] vectors_float;
}

template <typename data_t> void InMemClusterStore<data_t>::get_closest_clusters(const data_t *const query, const uint32_t num_closest, std::vector<uint32_t> &closest_clusters) {

    float* query_float = new float[this->_dim];
    diskann::convert_types<data_t, float>(query, query_float, 1, this->_dim);

    closest_clusters.resize(num_closest);
    math_utils::compute_closest_centers(query_float, 1, this->_dim, this->_cluster_centroids, this->_num_clusters, num_closest,
                                            closest_clusters.data());

    delete[] query_float;
}

// todo: do we copy the roaring bitmap inside the roaring list?
template <typename data_t> void InMemClusterStore<data_t>::get_cluster_members(const uint32_t cluster_id, AbstractIdList &output_list) {
//    output_list = _posting_lists[cluster_id];
    output_list.create_from(_posting_lists[cluster_id]);
    //std::cout<<"*"<<_posting_lists[cluster_id].size()<<"*";
}

template DISKANN_DLLEXPORT class InMemClusterStore<float>;
template DISKANN_DLLEXPORT class InMemClusterStore<int8_t>;
template DISKANN_DLLEXPORT class InMemClusterStore<uint8_t>;
}