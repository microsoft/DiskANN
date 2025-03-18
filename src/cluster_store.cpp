// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <memory>
#include "abstract_scratch.h"
#include "cluster_store.h"
#include "math_utils.h"
#include "utils.h"
#include <mkl.h>

namespace diskann
{

template <typename data_t>
InMemClusterStore<data_t>::InMemClusterStore(const size_t dim) : AbstractClusterStore<data_t>(dim)
{
}

template <typename data_t> InMemClusterStore<data_t>::~InMemClusterStore()
{
    delete[] this->_cluster_centroids;
}

template <typename data_t> uint32_t InMemClusterStore<data_t>::load(const std::string &filename)
{
    std::string centers_file(filename);
    std::string posting_file(filename);
    centers_file += "_centers.bin";
    posting_file += "_posting.bin";

    if (!file_exists(posting_file))
        return 0;

    diskann::load_bin<float>(centers_file, this->_cluster_centroids, this->_num_clusters, this->_dim);

    this->_cluster_norms = new float[this->_num_clusters];
    math_utils::compute_vecs_l2sq(this->_cluster_norms, this->_cluster_centroids, this->_num_clusters, this->_dim);

    std::vector<uint32_t> non_empty_clusters;

    std::ifstream in(posting_file, std::ios::binary | std::ios::in);
    uint64_t total_count = 0;

    _posting_lists.resize(this->_num_clusters);
    for (unsigned i = 0; i < this->_num_clusters; i++)
    {
        unsigned cur_count = 0;

        in.read((char *)&cur_count, sizeof(unsigned));

        if (cur_count > 0)
        {
            non_empty_clusters.emplace_back(i);
            uint32_t *vals = new uint32_t[cur_count];
            in.read((char *)vals, (uint64_t)cur_count * sizeof(unsigned));
            _posting_lists[i] = RoaringIdList(cur_count, vals);
            delete[] vals;
        }

        //      roaring_bitmap_add_many((roaring_bitmap_t*)_posting_lists[i].get_bitmap(), cur_count, vals);
        total_count += cur_count;
    }
    in.close();

    for (uint32_t i = 0; i < non_empty_clusters.size(); i++)
    {
        if (i != non_empty_clusters[i])
        {
            std::memcpy(this->_cluster_centroids + i * this->_dim,
                        this->_cluster_centroids + ((uint64_t)non_empty_clusters[i]) * this->_dim,
                        this->_dim * sizeof(float));
            _posting_lists[i] = _posting_lists[non_empty_clusters[i]];
        }
    }

    this->_num_clusters = non_empty_clusters.size();

    std::cout << "Read a total of " << total_count << " points from inverted index file." << std::endl;
    std::cout << " Resized to " << this->_num_clusters << " clusters after removing empty clusters." << std::endl;
    return this->_num_clusters;
}

template <typename data_t> size_t InMemClusterStore<data_t>::save(const std::string &filename)
{
    std::string centers_file(filename);
    std::string posting_file(filename);
    centers_file += "_centers.bin";
    posting_file += "_posting.bin";
    diskann::save_bin<float>(centers_file, this->_cluster_centroids, this->_num_clusters, this->_dim);

    std::ofstream out(posting_file, std::ios::binary | std::ios::out);
    uint64_t total_count = 0;

    for (unsigned i = 0; i < this->_num_clusters; i++)
    {
        uint32_t count = (uint32_t)_posting_lists[i].size();
        out.write((char *)&count, sizeof(unsigned));

        uint32_t *arr = new uint32_t[count];

        roaring::Roaring x = _posting_lists[i].list;
        x.toUint32Array(arr);

        out.write((char *)arr, sizeof(unsigned) * (uint64_t)count);
        total_count += count;
        delete[] arr;
    }
    out.close();

    std::cout << "Written a total of " << total_count << " points to file." << std::endl;
    return total_count;
}

template <typename data_t> void InMemClusterStore<data_t>::add_cetroids(float *clusters, uint32_t num_clusters)
{
    this->_num_clusters = num_clusters;

    diskann::cout << "Set num clusters to " << num_clusters << ", and dim to " << this->_dim << std::endl;
    this->_cluster_centroids = new float[(uint64_t)num_clusters * this->_dim];
    std::memcpy(this->_cluster_centroids, clusters, (uint64_t)num_clusters * this->_dim * sizeof(float));

    _posting_lists.clear();
    _posting_lists.resize(num_clusters);
}

template <typename data_t>
void InMemClusterStore<data_t>::assign_data_to_clusters(data_t *vectors, std::vector<uint32_t> &ids)
{
    uint64_t num_pts = ids.size();
    float *vectors_float;
    if (sizeof(data_t) != sizeof(float))
    {
        vectors_float = new float[num_pts * this->_dim];
        diskann::convert_types<data_t, float>(vectors, vectors_float, num_pts, this->_dim);
    }
    else
    {
        vectors_float = (float *)vectors;
    }

    uint32_t *closest_centers = new uint32_t[num_pts];
    math_utils::compute_closest_centers(vectors_float, num_pts, this->_dim, this->_cluster_centroids,
                                        this->_num_clusters, 1, closest_centers);

    for (uint64_t pos = 0; pos < num_pts; pos++)
    {
        _posting_lists[closest_centers[pos]].add(ids[pos]);
    }
    delete[] closest_centers;
    delete[] vectors_float;
}

template <typename data_t>
void InMemClusterStore<data_t>::get_closest_clusters(data_t *query, const uint32_t num_closest,
                                                     InMemQueryScratch<data_t> *scratch)
{
    auto &closest_clusters = scratch->closest_clusters();
    float *query_float; // = new float[this->_dim];
    if (sizeof(data_t) == sizeof(float))
    {
        query_float = (float *)query;
    }
    else
    {
        query_float = scratch->get_query_float();
        diskann::convert_types<data_t, float>(query, query_float, 1, this->_dim);
    }

    auto &x = scratch->closest_clusters();
    closest_clusters.resize(num_closest);
    auto &cluster_distances = scratch->get_cluster_distance_vector();

    std::memcpy(cluster_distances.data(), this->_cluster_norms, this->_num_clusters * sizeof(float));

    cblas_sgemv(CblasRowMajor, CblasNoTrans, (MKL_INT)this->_num_clusters, (MKL_INT)this->_dim, -2.0f,
                this->_cluster_centroids, (MKL_INT)this->_dim, query_float, (MKL_INT)1, 1, cluster_distances.data(),
                (MKL_INT)1);

    std::priority_queue<PivotContainer> top_k_queue;
    for (size_t j = 0; j < this->_num_clusters; j++)
    {
        PivotContainer this_piv(j, cluster_distances[j]);
        top_k_queue.push(this_piv);
    }
    for (size_t j = 0; j < num_closest; j++)
    {
        PivotContainer this_piv = top_k_queue.top();
        closest_clusters[j] = (uint32_t)this_piv.piv_id;
        top_k_queue.pop();
    }

    //    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)num_points, (MKL_INT)num_centers, (MKL_INT)dim,
    //    -2.0f,
    //                data, (MKL_INT)dim, centers, (MKL_INT)dim, 1.0f, dist_matrix, (MKL_INT)num_centers);

    //    math_utils::compute_closest_centers(query_float, 1, this->_dim, this->_cluster_centroids, this->_num_clusters,
    //    num_closest,
    //                                            closest_clusters.data());

    // delete[] query_float;
}

// todo: do we copy the roaring bitmap inside the roaring list?
template <typename data_t>
void InMemClusterStore<data_t>::get_cluster_members(const uint32_t cluster_id, AbstractIdList &output_list)
{
    //    std::cout<<"*" << _posting_lists[cluster_id].size() <<"*";
    output_list.copy_from(_posting_lists[cluster_id]);
}

template DISKANN_DLLEXPORT class InMemClusterStore<float>;
template DISKANN_DLLEXPORT class InMemClusterStore<int8_t>;
template DISKANN_DLLEXPORT class InMemClusterStore<uint8_t>;
} // namespace diskann