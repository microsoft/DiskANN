// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <string>
#include <memory>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#include "disk_utils.h"
#include "index.h"
#include "pq_flash_index.h"

PYBIND11_MAKE_OPAQUE(std::vector<unsigned>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;
using namespace diskann;

template <class T> struct DiskANNIndex
{
    PQFlashIndex<T> *pq_flash_index;
    std::shared_ptr<AlignedFileReader> reader;

    DiskANNIndex(diskann::Metric metric)
    {
#ifdef _WINDOWS
        reader = std::make_shared<WindowsAlignedFileReader>();
#else
        reader = std::make_shared<LinuxAlignedFileReader>();
#endif
        pq_flash_index = new PQFlashIndex<T>(reader, metric);
    }

    ~DiskANNIndex()
    {
        delete pq_flash_index;
    }

    auto get_metric()
    {
        return pq_flash_index->get_metric();
    }

    void cache_bfs_levels(size_t num_nodes_to_cache)
    {
        std::vector<uint32_t> node_list;
        pq_flash_index->cache_bfs_levels(num_nodes_to_cache, node_list);
        pq_flash_index->load_cache_list(node_list);
    }

    void cache_sample_paths(size_t num_nodes_to_cache, const std::string &warmup_query_file, uint32_t num_threads)
    {
        if (!file_exists(warmup_query_file))
        {
            return;
        }

        std::vector<uint32_t> node_list;
        pq_flash_index->generate_cache_list_from_sample_queries(warmup_query_file, 15, 4, num_nodes_to_cache,
                                                                num_threads, node_list);
        pq_flash_index->load_cache_list(node_list);
    }

    int load_index(const std::string &index_path_prefix, const int num_threads, const size_t num_nodes_to_cache,
                   int cache_mechanism)
    {
        int load_success = pq_flash_index->load(num_threads, index_path_prefix.c_str());
        if (load_success != 0)
        {
            throw std::runtime_error("load_index failed.");
        }
        if (cache_mechanism == 0)
        {
            // Nothing to do
        }
        else if (cache_mechanism == 1)
        {
            std::string sample_file = index_path_prefix + std::string("_sample_data.bin");
            cache_sample_paths(num_nodes_to_cache, sample_file, num_threads);
        }
        else if (cache_mechanism == 2)
        {
            cache_bfs_levels(num_nodes_to_cache);
        }
        return 0;
    }

    auto search(py::array_t<T, py::array::c_style | py::array::forcecast> &query, const uint64_t knn,
                const uint64_t l_search, const uint64_t beam_width)
    {
        py::array_t<unsigned> ids(knn);
        py::array_t<float> dists(knn);

        std::vector<unsigned> u32_ids(knn);
        std::vector<uint64_t> u64_ids(knn);
        QueryStats stats;

        pq_flash_index->cached_beam_search(query.data(), knn, l_search, u64_ids.data(), dists.mutable_data(),
                                           beam_width, false, &stats);

        auto r = ids.mutable_unchecked<1>();
        for (uint64_t i = 0; i < knn; ++i)
            r(i) = (unsigned)u64_ids[i];

        return std::make_pair(ids, dists);
    }

    auto batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries,
                      const uint64_t knn, const uint64_t l_search, const uint64_t beam_width, const int num_threads)
    {
        py::array_t<unsigned> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        omp_set_num_threads(num_threads);

        std::vector<uint64_t> u64_ids(knn * num_queries);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)num_queries; i++)
        {
            pq_flash_index->cached_beam_search(queries.data(i), knn, l_search, u64_ids.data() + i * knn,
                                               dists.mutable_data(i), beam_width);
        }

        auto r = ids.mutable_unchecked();
        for (uint64_t i = 0; i < num_queries; ++i)
            for (uint64_t j = 0; j < knn; ++j)
                r(i, j) = (unsigned)u64_ids[i * knn + j];

        return std::make_pair(ids, dists);
    }
};

typedef uint32_t IdT;
typedef uint32_t filterT;

template <class T> struct DynamicInMemIndex
{
    Index<T, IdT, filterT> *_index;
    const IndexWriteParameters write_params;

    DynamicInMemIndex(Metric m, const size_t dim, const size_t max_points, const IndexWriteParameters &index_parameters,
                      const uint32_t initial_search_list_size, const uint32_t search_threads,
                      const bool concurrent_consolidate)
        : write_params(index_parameters)
    {
        _index = new Index<T>(m, dim, max_points,
                              true,                     // dynamic_index
                              index_parameters,         // used for insert
                              initial_search_list_size, // used to prepare the scratch space for searching. can / may be
                                                        // expanded if the search asks for a larger L.
                              search_threads,           // also used for the scratch space
                              true,                     // enable_tags
                              concurrent_consolidate,
                              false,  // pq_dist_build
                              0,      // num_pq_chunks
                              false); // use_opq = false
    }

    ~DynamicInMemIndex()
    {
        delete _index;
    }

    int insert(py::array_t<T, py::array::c_style | py::array::forcecast> &vector, const IdT id)
    {
        return _index->insert_point(vector.data(), id);
    }

    int mark_deleted(const IdT id)
    {
        return _index->lazy_delete(id);
    }

    auto search(py::array_t<T, py::array::c_style | py::array::forcecast> &query, const uint64_t knn,
                const uint64_t l_search)
    {
        py::array_t<IdT> ids(knn);
        py::array_t<float> dists(knn);
        std::vector<T *> empty_vector;
        _index->search_with_tags(query.data(), knn, l_search, ids.mutable_data(), dists.mutable_data(), empty_vector);
        return std::make_pair(ids, dists);
    }

    auto batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries,
                      const uint64_t knn, const uint64_t l_search, const int num_threads)
    {
        py::array_t<unsigned> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});
        std::vector<T *> empty_vector;

        omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)num_queries; i++)
        {
            _index->search_with_tags(queries.data(i), knn, l_search, ids.mutable_data(i), dists.mutable_data(i),
                                     empty_vector);
        }

        return std::make_pair(ids, dists);
    }

    auto consolidate_delete()
    {
        return _index->consolidate_deletes(write_params);
    }
};

template <class T> struct StaticInMemIndex
{
    Index<T, IdT, filterT> *_index;

    StaticInMemIndex(Metric m, const std::string &data_path, IndexWriteParameters &index_parameters)
    {
        size_t ndims, npoints;
        diskann::get_bin_metadata(data_path, npoints, ndims);
        _index = new Index<T>(m, ndims, npoints,
                              false, // not a dynamic_index
                              false, // no enable_tags/ids
                              false, // no concurrent_consolidate,
                              false, // pq_dist_build
                              0,     // num_pq_chunks
                              false, // use_opq = false
                              0);    // num_frozen_pts = 0
        _index->build(data_path.c_str(), npoints, index_parameters);
    }

    ~StaticInMemIndex()
    {
        delete _index;
    }

    auto search(py::array_t<T, py::array::c_style | py::array::forcecast> &query, const uint64_t knn,
                const uint64_t l_search)
    {
        py::array_t<IdT> ids(knn);
        py::array_t<float> dists(knn);
        std::vector<T *> empty_vector;
        _index->search(query.data(), knn, l_search, ids.mutable_data(), dists.mutable_data());
        return std::make_pair(ids, dists);
    }

    auto batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries,
                      const uint64_t knn, const uint64_t l_search, const int num_threads)
    {
        py::array_t<unsigned> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});
        std::vector<T *> empty_vector;

        omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)num_queries; i++)
        {
            _index->search(queries.data(i), knn, l_search, ids.mutable_data(i), dists.mutable_data(i));
        }

        return std::make_pair(ids, dists);
    }
};

PYBIND11_MODULE(_diskannpy, m)
{
    m.doc() = "DiskANN Python Bindings";
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

    py::enum_<Metric>(m, "Metric")
        .value("L2", Metric::L2)
        .value("INNER_PRODUCT", Metric::INNER_PRODUCT)
        .export_values();

    py::class_<StaticInMemIndex<float>>(m, "DiskANNStaticInMemFloatIndex")
        .def(py::init([](diskann::Metric metric, const std::string &data_path, IndexWriteParameters &index_parameters) {
            return std::unique_ptr<StaticInMemIndex<float>>(
                new StaticInMemIndex<float>(metric, data_path, index_parameters));
        }))
        .def("search", &StaticInMemIndex<float>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"))
        .def("batch_search", &StaticInMemIndex<float>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("num_threads"));

    py::class_<StaticInMemIndex<int8_t>>(m, "DiskANNStaticInMemInt8Index")
        .def(py::init([](diskann::Metric metric, const std::string &data_path, IndexWriteParameters &index_parameters) {
            return std::unique_ptr<StaticInMemIndex<int8_t>>(
                new StaticInMemIndex<int8_t>(metric, data_path, index_parameters));
        }))
        .def("search", &StaticInMemIndex<int8_t>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"))
        .def("batch_search", &StaticInMemIndex<int8_t>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("num_threads"));

    py::class_<StaticInMemIndex<uint8_t>>(m, "DiskANNStaticInMemUint8Index")
        .def(py::init([](diskann::Metric metric, const std::string &data_path, IndexWriteParameters &index_parameters) {
            return std::unique_ptr<StaticInMemIndex<uint8_t>>(
                new StaticInMemIndex<uint8_t>(metric, data_path, index_parameters));
        }))
        .def("search", &StaticInMemIndex<uint8_t>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"))
        .def("batch_search", &StaticInMemIndex<uint8_t>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("num_threads"));

    py::class_<DynamicInMemIndex<float>>(m, "DiskANNDynamicInMemFloatIndex")
        .def(py::init([](diskann::Metric metric, const size_t dim, const size_t max_points,
                         const IndexWriteParameters &index_parameters, const uint32_t initial_search_list_size,
                         const uint32_t search_threads, const bool concurrent_consolidate) {
            return std::unique_ptr<DynamicInMemIndex<float>>(
                new DynamicInMemIndex<float>(metric, dim, max_points, index_parameters, initial_search_list_size,
                                             search_threads, concurrent_consolidate));
        }))
        .def("search", &DynamicInMemIndex<float>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"))
        .def("batch_search", &DynamicInMemIndex<float>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("num_threads"))
        .def("insert", &DynamicInMemIndex<float>::insert, py::arg("vector"), py::arg("id"))
        .def("mark_deleted", &DynamicInMemIndex<float>::mark_deleted, py::arg("id"))
        .def("consolidate_delete", &DynamicInMemIndex<float>::consolidate_delete);

    py::class_<DynamicInMemIndex<int8_t>>(m, "DiskANNDynamicInMemInt8Index")
        .def(py::init([](diskann::Metric metric, const size_t dim, const size_t max_points,
                         const IndexWriteParameters &index_parameters, const uint32_t initial_search_list_size,
                         const uint32_t search_threads, const bool concurrent_consolidate) {
            return std::unique_ptr<DynamicInMemIndex<int8_t>>(
                new DynamicInMemIndex<int8_t>(metric, dim, max_points, index_parameters, initial_search_list_size,
                                              search_threads, concurrent_consolidate));
        }))
        .def("search", &DynamicInMemIndex<int8_t>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"))
        .def("batch_search", &DynamicInMemIndex<int8_t>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("num_threads"))
        .def("insert", &DynamicInMemIndex<int8_t>::insert, py::arg("vector"), py::arg("id"))
        .def("mark_deleted", &DynamicInMemIndex<int8_t>::mark_deleted, py::arg("id"))
        .def("consolidate_delete", &DynamicInMemIndex<int8_t>::consolidate_delete);

    py::class_<DynamicInMemIndex<uint8_t>>(m, "DiskANNDynamicInMemUint8Index")
        .def(py::init([](diskann::Metric metric, const size_t dim, const size_t max_points,
                         const IndexWriteParameters &index_parameters, const uint32_t initial_search_list_size,
                         const uint32_t search_threads, const bool concurrent_consolidate) {
            return std::unique_ptr<DynamicInMemIndex<uint8_t>>(
                new DynamicInMemIndex<uint8_t>(metric, dim, max_points, index_parameters, initial_search_list_size,
                                               search_threads, concurrent_consolidate));
        }))
        .def("search", &DynamicInMemIndex<uint8_t>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"))
        .def("batch_search", &DynamicInMemIndex<uint8_t>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("num_threads"))
        .def("insert", &DynamicInMemIndex<uint8_t>::insert, py::arg("vector"), py::arg("id"))
        .def("mark_deleted", &DynamicInMemIndex<uint8_t>::mark_deleted, py::arg("id"))
        .def("consolidate_delete", &DynamicInMemIndex<uint8_t>::consolidate_delete);

    py::class_<DiskANNIndex<float>>(m, "DiskANNFloatIndex")
        .def(py::init([](diskann::Metric metric) {
            return std::unique_ptr<DiskANNIndex<float>>(new DiskANNIndex<float>(metric));
        }))
        .def("cache_bfs_levels", &DiskANNIndex<float>::cache_bfs_levels, py::arg("num_nodes_to_cache"))
        .def("load_index", &DiskANNIndex<float>::load_index, py::arg("index_path_prefix"), py::arg("num_threads"),
             py::arg("num_nodes_to_cache"), py::arg("cache_mechanism") = 1)
        .def("search", &DiskANNIndex<float>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"),
             py::arg("beam_width"))
        .def("batch_search", &DiskANNIndex<float>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("num_threads"))
        .def(
            "build",
            [](DiskANNIndex<float> &self, const char *data_file_path, const char *index_prefix_path, unsigned R,
               unsigned L, double final_index_ram_limit, double indexing_ram_budget, unsigned num_threads,
               unsigned pq_disk_bytes) {
                std::string params = std::to_string(R) + " " + std::to_string(L) + " " +
                                     std::to_string(final_index_ram_limit) + " " + std::to_string(indexing_ram_budget) +
                                     " " + std::to_string(num_threads);
                if (pq_disk_bytes > 0)
                {
                    params = params + " " + std::to_string(pq_disk_bytes);
                }
                diskann::build_disk_index<float>(data_file_path, index_prefix_path, params.c_str(), self.get_metric());
            },
            py::arg("data_file_path"), py::arg("index_prefix_path"), py::arg("R"), py::arg("L"),
            py::arg("final_index_ram_limit"), py::arg("indexing_ram_limit"), py::arg("num_threads"),
            py::arg("pq_disk_bytes") = 0);

    py::class_<DiskANNIndex<int8_t>>(m, "DiskANNInt8Index")
        .def(py::init([](diskann::Metric metric) {
            return std::unique_ptr<DiskANNIndex<int8_t>>(new DiskANNIndex<int8_t>(metric));
        }))
        .def("cache_bfs_levels", &DiskANNIndex<int8_t>::cache_bfs_levels, py::arg("num_nodes_to_cache"))
        .def("load_index", &DiskANNIndex<int8_t>::load_index, py::arg("index_path_prefix"), py::arg("num_threads"),
             py::arg("num_nodes_to_cache"), py::arg("cache_mechanism") = 1)
        .def("search", &DiskANNIndex<int8_t>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"),
             py::arg("beam_width"))
        .def("batch_search", &DiskANNIndex<int8_t>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("num_threads"))
        .def(
            "build",
            [](DiskANNIndex<int8_t> &self, const char *data_file_path, const char *index_prefix_path, unsigned R,
               unsigned L, double final_index_ram_limit, double indexing_ram_budget, unsigned num_threads,
               unsigned pq_disk_bytes) {
                std::string params = std::to_string(R) + " " + std::to_string(L) + " " +
                                     std::to_string(final_index_ram_limit) + " " + std::to_string(indexing_ram_budget) +
                                     " " + std::to_string(num_threads);
                if (pq_disk_bytes > 0)
                    params = params + " " + std::to_string(pq_disk_bytes);
                diskann::build_disk_index<int8_t>(data_file_path, index_prefix_path, params.c_str(), self.get_metric());
            },
            py::arg("data_file_path"), py::arg("index_prefix_path"), py::arg("R"), py::arg("L"),
            py::arg("final_index_ram_limit"), py::arg("indexing_ram_limit"), py::arg("num_threads"),
            py::arg("pq_disk_bytes") = 0);

    py::class_<DiskANNIndex<uint8_t>>(m, "DiskANNUInt8Index")
        .def(py::init([](diskann::Metric metric) {
            return std::unique_ptr<DiskANNIndex<uint8_t>>(new DiskANNIndex<uint8_t>(metric));
        }))
        .def("cache_bfs_levels", &DiskANNIndex<uint8_t>::cache_bfs_levels, py::arg("num_nodes_to_cache"))
        .def("load_index", &DiskANNIndex<uint8_t>::load_index, py::arg("index_path_prefix"), py::arg("num_threads"),
             py::arg("num_nodes_to_cache"), py::arg("cache_mechanism") = 1)
        .def("search", &DiskANNIndex<uint8_t>::search, py::arg("query"), py::arg("knn"), py::arg("l_search"),
             py::arg("beam_width"))
        .def("batch_search", &DiskANNIndex<uint8_t>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("num_threads"))
        .def(
            "build",
            [](DiskANNIndex<uint8_t> &self, const char *data_file_path, const char *index_prefix_path, unsigned R,
               unsigned L, double final_index_ram_limit, double indexing_ram_budget, unsigned num_threads,
               unsigned pq_disk_bytes) {
                std::string params = std::to_string(R) + " " + std::to_string(L) + " " +
                                     std::to_string(final_index_ram_limit) + " " + std::to_string(indexing_ram_budget) +
                                     " " + std::to_string(num_threads);
                if (pq_disk_bytes > 0)
                    params = params + " " + std::to_string(pq_disk_bytes);
                diskann::build_disk_index<uint8_t>(data_file_path, index_prefix_path, params.c_str(),
                                                   self.get_metric());
            },
            py::arg("data_file_path"), py::arg("index_prefix_path"), py::arg("R"), py::arg("L"),
            py::arg("final_index_ram_limit"), py::arg("indexing_ram_limit"), py::arg("num_threads"),
            py::arg("pq_disk_bytes") = 0);
}
