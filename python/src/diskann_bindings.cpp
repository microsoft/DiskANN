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
#include "utils.h"

PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;
using namespace diskann;

#ifdef _WINDOWS
typedef WindowsAlignedFileReader PlatformSpecificAlignedFileReader;
#else
typedef LinuxAlignedFileReader PlatformSpecificAlignedFileReader;
#endif

template <class T> struct DiskIndex
{
    PQFlashIndex<T> *_pq_flash_index;
    std::shared_ptr<AlignedFileReader> reader;

    DiskIndex(const diskann::Metric metric, const std::string &index_path_prefix, const uint32_t num_threads,
              const size_t num_nodes_to_cache, const uint32_t cache_mechanism)
    {
        reader = std::make_shared<PlatformSpecificAlignedFileReader>();
        _pq_flash_index = new PQFlashIndex<T>(reader, metric);
        int load_success = _pq_flash_index->load(num_threads, index_path_prefix.c_str());
        if (load_success != 0)
        {
            throw std::runtime_error("index load failed.");
        }
        if (cache_mechanism == 1)
        {
            std::string sample_file = index_path_prefix + std::string("_sample_data.bin");
            cache_sample_paths(num_nodes_to_cache, sample_file, num_threads);
        }
        else if (cache_mechanism == 2)
        {
            cache_bfs_levels(num_nodes_to_cache);
        }
    }

    ~DiskIndex()
    {
        delete _pq_flash_index;
    }

    void cache_bfs_levels(const size_t num_nodes_to_cache)
    {
        std::vector<uint32_t> node_list;
        _pq_flash_index->cache_bfs_levels(num_nodes_to_cache, node_list);
        _pq_flash_index->load_cache_list(node_list);
    }

    void cache_sample_paths(const size_t num_nodes_to_cache, const std::string &warmup_query_file,
                            const uint32_t num_threads)
    {
        if (!file_exists(warmup_query_file))
        {
            return;
        }

        std::vector<uint32_t> node_list;
        _pq_flash_index->generate_cache_list_from_sample_queries(warmup_query_file, 15, 4, num_nodes_to_cache,
                                                                 num_threads, node_list);
        _pq_flash_index->load_cache_list(node_list);
    }

    auto search(py::array_t<T, py::array::c_style | py::array::forcecast> &query, const uint64_t knn,
                const uint64_t complexity, const uint64_t beam_width)
    {
        py::array_t<uint32_t> ids(knn);
        py::array_t<float> dists(knn);

        std::vector<uint32_t> u32_ids(knn);
        std::vector<uint64_t> u64_ids(knn);
        QueryStats stats;

        _pq_flash_index->cached_beam_search(query.data(), knn, complexity, u64_ids.data(), dists.mutable_data(),
                                            beam_width, false, &stats);

        auto r = ids.mutable_unchecked<1>();
        for (uint64_t i = 0; i < knn; ++i)
            r(i) = (unsigned)u64_ids[i];

        return std::make_pair(ids, dists);
    }

    auto batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries,
                      const uint64_t knn, const uint64_t complexity, const uint64_t beam_width, const int num_threads)
    {
        py::array_t<uint32_t> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        omp_set_num_threads(num_threads);

        std::vector<uint64_t> u64_ids(knn * num_queries);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)num_queries; i++)
        {
            _pq_flash_index->cached_beam_search(queries.data(i), knn, complexity, u64_ids.data() + i * knn,
                                                dists.mutable_data(i), beam_width);
        }

        auto r = ids.mutable_unchecked();
        for (uint64_t i = 0; i < num_queries; ++i)
            for (uint64_t j = 0; j < knn; ++j)
                r(i, j) = (uint32_t)u64_ids[i * knn + j];

        return std::make_pair(ids, dists);
    }
};

typedef uint32_t IdT;
typedef uint32_t filterT;

template <class T> struct DynamicInMemIndex
{
    Index<T, IdT, filterT> *_index;
    IndexWriteParameters _write_params;
    const std::string &_index_path;

    DynamicInMemIndex(const Metric m, const size_t dim, const size_t max_points, const uint32_t complexity,
                      const uint32_t graph_degree, const bool saturate_graph, const uint32_t max_occlusion_size,
                      const float alpha, const uint32_t num_threads, const uint32_t filter_complexity,
                      const uint32_t num_frozen_points, const uint32_t initial_search_complexity,
                      const uint32_t initial_search_threads, const bool concurrent_consolidation,
                      const std::string &index_path = "")
        : _write_params(IndexWriteParametersBuilder(complexity, graph_degree)
                            .with_saturate_graph(saturate_graph)
                            .with_max_occlusion_size(max_occlusion_size)
                            .with_alpha(alpha)
                            .with_num_threads(num_threads)
                            .with_filter_list_size(filter_complexity)
                            .with_num_frozen_points(num_frozen_points)
                            .build()),
          _index_path(index_path)
    {
        const uint32_t _initial_search_complexity =
            initial_search_complexity != 0 ? initial_search_complexity : complexity;
        const uint32_t _initial_search_threads =
            initial_search_threads != 0 ? initial_search_threads : omp_get_num_threads();

        _index = new Index<T>(m, dim, max_points,
                              true,                       // dynamic_index
                              _write_params,              // used for insert
                              _initial_search_complexity, // used to prepare the scratch space for searching. can / may
                                                          // be expanded if the search asks for a larger L.
                              _initial_search_threads,    // also used for the scratch space
                              true,                       // enable_tags
                              concurrent_consolidation,
                              false,  // pq_dist_build
                              0,      // num_pq_chunks
                              false); // use_opq = false
        if (!index_path.empty())
        {
            _index->load(index_path.c_str(), _write_params.num_threads, complexity);
        }
        _index->enable_delete();
    }

    ~DynamicInMemIndex()
    {
        delete _index;
    }

    int insert(py::array_t<T, py::array::c_style | py::array::forcecast> &vector, const IdT id)
    {
        return _index->insert_point(vector.data(), id);
    }

    auto batch_insert(py::array_t<T, py::array::c_style | py::array::forcecast> &vectors,
                      py::array_t<IdT, py::array::c_style | py::array::forcecast> &ids, const int32_t num_inserts,
                      const int num_threads = 0)
    {
        if (num_threads == 0)
            omp_set_num_threads(omp_get_num_procs());
        else
            omp_set_num_threads(num_threads);
        py::array_t<int> insert_retvals(num_inserts);

#pragma omp parallel for schedule(dynamic, 1)
        for (int32_t i = 0; i < num_inserts; i++)
        {
            insert_retvals.mutable_data()[i] = _index->insert_point(vectors.data(i), *(ids.data(i)));
        }

        return insert_retvals;
    }

    int mark_deleted(const IdT id)
    {
        return _index->lazy_delete(id);
    }

    void save(const std::string &save_path = "", const bool compact_before_save = false)
    {
        const std::string path = !save_path.empty() ? save_path : _index_path;
        if (path.empty())
        {
            throw std::runtime_error(
                "A save_path must be provided if a starting index was not provided in the DynamicMemoryIndex "
                "constructor via the index_path parameter");
        }
        _index->save(path.c_str(), compact_before_save);
    }

    auto search(py::array_t<T, py::array::c_style | py::array::forcecast> &query, const uint64_t knn,
                const uint64_t complexity)
    {
        py::array_t<IdT> ids(knn);
        py::array_t<float> dists(knn);
        std::vector<T *> empty_vector;
        _index->search_with_tags(query.data(), knn, complexity, ids.mutable_data(), dists.mutable_data(), empty_vector);
        return std::make_pair(ids, dists);
    }

    auto batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries,
                      const uint64_t knn, const uint64_t complexity, const int num_threads)
    {
        py::array_t<unsigned> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});
        std::vector<T *> empty_vector;

        if (num_threads == 0)
            omp_set_num_threads(omp_get_num_procs());
        else
            omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)num_queries; i++)
        {
            _index->search_with_tags(queries.data(i), knn, complexity, ids.mutable_data(i), dists.mutable_data(i),
                                     empty_vector);
        }

        return std::make_pair(ids, dists);
    }

    auto consolidate_delete()
    {
        _index->consolidate_deletes(_write_params);
    }
};

template <class T> struct StaticInMemIndex
{
    Index<T, IdT, filterT> *_index;

    StaticInMemIndex(const Metric m, const std::string &data_path, const std::string &index_prefix,
                     const uint32_t num_threads, const uint32_t initial_search_complexity)
    {
        const uint32_t _num_threads = num_threads != 0 ? num_threads : omp_get_num_threads();
        if (initial_search_complexity == 0)
        {
            throw std::runtime_error("initial_search_complexity must be a positive uint32_t");
        }

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
        _index->load(index_prefix.c_str(), _num_threads, initial_search_complexity);
    }

    ~StaticInMemIndex()
    {
        delete _index;
    }

    auto search(py::array_t<T, py::array::c_style | py::array::forcecast> &query, const uint64_t knn,
                const uint64_t complexity)
    {
        py::array_t<IdT> ids(knn);
        py::array_t<float> dists(knn);
        std::vector<T *> empty_vector;
        _index->search(query.data(), knn, complexity, ids.mutable_data(), dists.mutable_data());
        return std::make_pair(ids, dists);
    }

    auto batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries,
                      const uint64_t knn, const uint64_t complexity, const int num_threads)
    {
        const uint32_t _num_threads = num_threads != 0 ? num_threads : omp_get_num_threads();
        py::array_t<unsigned> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});
        std::vector<T *> empty_vector;

        omp_set_num_threads(_num_threads);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)num_queries; i++)
        {
            _index->search(queries.data(i), knn, complexity, ids.mutable_data(i), dists.mutable_data(i));
        }

        return std::make_pair(ids, dists);
    }
};

template <typename T>
void build_disk_index2(const diskann::Metric metric, const std::string &data_file_path,
                       const std::string &index_prefix_path, const uint32_t complexity, const uint32_t graph_degree,
                       const double final_index_ram_limit, const double indexing_ram_budget, const uint32_t num_threads,
                       const uint32_t pq_disk_bytes)
{
    std::string params = std::to_string(graph_degree) + " " + std::to_string(complexity) + " " +
                         std::to_string(final_index_ram_limit) + " " + std::to_string(indexing_ram_budget) + " " +
                         std::to_string(num_threads);
    if (pq_disk_bytes > 0)
        params = params + " " + std::to_string(pq_disk_bytes);
    diskann::build_disk_index<T>(data_file_path.c_str(), index_prefix_path.c_str(), params.c_str(), metric);
}

template <typename T, typename TagT = IdT, typename LabelT = filterT>
void build_in_memory_index(const diskann::Metric &metric, const std::string &vector_bin_path,
                           const std::string &index_output_path, const uint32_t graph_degree, const uint32_t complexity,
                           const float alpha, const uint32_t num_threads, const bool use_pq_build,
                           const size_t num_pq_bytes, const bool use_opq, const std::string &label_file,
                           const std::string &universal_label, const uint32_t filter_complexity,
                           const bool use_tags = false)
{
    diskann::IndexWriteParameters index_build_params = diskann::IndexWriteParametersBuilder(complexity, graph_degree)
                                                           .with_filter_list_size(filter_complexity)
                                                           .with_alpha(alpha)
                                                           .with_saturate_graph(false)
                                                           .with_num_threads(num_threads)
                                                           .build();
    size_t data_num, data_dim;
    diskann::get_bin_metadata(vector_bin_path, data_num, data_dim);
    diskann::Index<T, TagT, LabelT> index(metric, data_dim, data_num, false, use_tags, false, use_pq_build,
                                          num_pq_bytes, use_opq);
    if (label_file == "")
    {
        index.build(vector_bin_path.c_str(), data_num, index_build_params);
    }
    else
    {
        std::string labels_file_to_use = index_output_path + "_label_formatted.txt";
        std::string mem_labels_int_map_file = index_output_path + "_labels_map.txt";
        convert_labels_string_to_int(label_file, labels_file_to_use, mem_labels_int_map_file, universal_label);
        if (universal_label != "")
        {
            filterT unv_label_as_num = 0;
            index.set_universal_label(unv_label_as_num);
        }
        index.build_filtered_index(index_output_path.c_str(), labels_file_to_use, data_num, index_build_params);
    }
    index.save(index_output_path.c_str());
}

template <typename T>
inline void add_variant(py::module_ &m, const std::string &build_name, const std::string &class_name)
{
    const std::string build_disk_name = "build_disk_" + build_name + "_index";
    m.def(build_disk_name.c_str(), &build_disk_index2<T>, py::arg("metric"), py::arg("data_file_path"),
          py::arg("index_prefix_path"), py::arg("complexity"), py::arg("graph_degree"),
          py::arg("final_index_ram_limit"), py::arg("indexing_ram_budget"), py::arg("num_threads"),
          py::arg("pq_disk_bytes"));

    const std::string build_in_memory_name = "build_in_memory_" + build_name + "_index";
    m.def(build_in_memory_name.c_str(), &build_in_memory_index<T>, py::arg("metric"), py::arg("data_file_path"),
          py::arg("index_output_path"), py::arg("graph_degree"), py::arg("complexity"), py::arg("alpha"),
          py::arg("num_threads"), py::arg("use_pq_build"), py::arg("num_pq_bytes"), py::arg("use_opq"),
          py::arg("label_file") = "", py::arg("universal_label") = "", py::arg("filter_complexity") = 0,
          py::arg("use_tags") = false);

    const std::string static_index = "StaticMemory" + class_name + "Index";
    py::class_<StaticInMemIndex<T>>(m, static_index.c_str())
        .def(py::init([](const diskann::Metric metric, const std::string &data_path, const std::string &index_path,
                         const uint32_t num_threads, const uint32_t initial_search_complexity) {
                 return std::unique_ptr<StaticInMemIndex<T>>(
                     new StaticInMemIndex<T>(metric, data_path, index_path, num_threads, initial_search_complexity));
             }),
             py::arg("metric"), py::arg("data_path"), py::arg("index_path"), py::arg("num_threads"),
             py::arg("initial_search_complexity"))
        .def("search", &StaticInMemIndex<T>::search, py::arg("query"), py::arg("knn"), py::arg("complexity"))
        .def("batch_search", &StaticInMemIndex<T>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("complexity"), py::arg("num_threads"));

    const std::string dynamic_index = "DynamicMemory" + class_name + "Index";
    py::class_<DynamicInMemIndex<T>>(m, dynamic_index.c_str())
        .def(py::init([](const diskann::Metric metric, const size_t dim, const size_t max_points,
                         const uint32_t complexity, const uint32_t graph_degree, const bool saturate_graph,
                         const uint32_t max_occlusion_size, const float alpha, const uint32_t num_threads,
                         const uint32_t filter_complexity, const uint32_t num_frozen_points,
                         const uint32_t initial_search_complexity, const uint32_t search_threads,
                         const bool concurrent_consolidation, const std::string &index_path) {
                 return std::unique_ptr<DynamicInMemIndex<T>>(new DynamicInMemIndex<T>(
                     metric, dim, max_points, complexity, graph_degree, saturate_graph, max_occlusion_size, alpha,
                     num_threads, filter_complexity, num_frozen_points, initial_search_complexity, search_threads,
                     concurrent_consolidation, index_path));
             }),
             py::arg("metric"), py::arg("dim"), py::arg("max_points"), py::arg("complexity"), py::arg("graph_degree"),
             py::arg("saturate_graph") = diskann::defaults::SATURATE_GRAPH,
             py::arg("max_occlusion_size") = diskann::defaults::MAX_OCCLUSION_SIZE,
             py::arg("alpha") = diskann::defaults::ALPHA, py::arg("num_threads") = diskann::defaults::NUM_THREADS,
             py::arg("filter_complexity") = diskann::defaults::FILTER_LIST_SIZE,
             py::arg("num_frozen_points") = diskann::defaults::NUM_FROZEN_POINTS_DYNAMIC,
             py::arg("initial_search_complexity") = 0, py::arg("search_threads") = 0,
             py::arg("concurrent_consolidation") = true, py::arg("index_path") = "")
        .def("search", &DynamicInMemIndex<T>::search, py::arg("query"), py::arg("knn"), py::arg("complexity"))
        .def("batch_search", &DynamicInMemIndex<T>::batch_search, py::arg("queries"), py::arg("num_queries"),
             py::arg("knn"), py::arg("complexity"), py::arg("num_threads"))
        .def("batch_insert", &DynamicInMemIndex<T>::batch_insert, py::arg("vectors"), py::arg("ids"),
             py::arg("num_inserts"), py::arg("num_threads"))
        .def("save", &DynamicInMemIndex<T>::save, py::arg("save_path") = "", py::arg("compact_before_save") = false)
        .def("insert", &DynamicInMemIndex<T>::insert, py::arg("vector"), py::arg("id"))
        .def("mark_deleted", &DynamicInMemIndex<T>::mark_deleted, py::arg("id"))
        .def("consolidate_delete", &DynamicInMemIndex<T>::consolidate_delete);

    const std::string disk_name = "Disk" + class_name + "Index";
    py::class_<DiskIndex<T>>(m, disk_name.c_str())
        .def(py::init([](const diskann::Metric metric, const std::string &index_path_prefix, const uint32_t num_threads,
                         const size_t num_nodes_to_cache, const uint32_t cache_mechanism) {
                 return std::unique_ptr<DiskIndex<T>>(
                     new DiskIndex<T>(metric, index_path_prefix, num_threads, num_nodes_to_cache, cache_mechanism));
             }),
             py::arg("metric"), py::arg("index_path_prefix"), py::arg("num_threads"), py::arg("num_nodes_to_cache"),
             py::arg("cache_mechanism") = 1)
        .def("cache_bfs_levels", &DiskIndex<T>::cache_bfs_levels, py::arg("num_nodes_to_cache"))
        .def("search", &DiskIndex<T>::search, py::arg("query"), py::arg("knn"), py::arg("complexity"),
             py::arg("beam_width"))
        .def("batch_search", &DiskIndex<T>::batch_search, py::arg("queries"), py::arg("num_queries"), py::arg("knn"),
             py::arg("complexity"), py::arg("beam_width"), py::arg("num_threads"));
}

PYBIND11_MODULE(_diskannpy, m)
{
    m.doc() = "DiskANN Python Bindings";
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

    // let's re-export our defaults
    py::module_ default_values = m.def_submodule(
        "defaults",
        "A collection of the default values used for common diskann operations. `GRAPH_DEGREE` and `COMPLEXITY` are not"
        " set as defaults, but some semi-reasonable default values are selected for your convenience. We urge you to "
        "investigate their meaning and adjust them for your use cases.");

    default_values.attr("ALPHA") = diskann::defaults::ALPHA;
    default_values.attr("NUM_THREADS") = diskann::defaults::NUM_THREADS;
    default_values.attr("MAX_OCCLUSION_SIZE") = diskann::defaults::MAX_OCCLUSION_SIZE;
    default_values.attr("FILTER_COMPLEXITY") = diskann::defaults::FILTER_LIST_SIZE;
    default_values.attr("NUM_FROZEN_POINTS_STATIC") = diskann::defaults::NUM_FROZEN_POINTS_STATIC;
    default_values.attr("NUM_FROZEN_POINTS_DYNAMIC") = diskann::defaults::NUM_FROZEN_POINTS_DYNAMIC;
    default_values.attr("SATURATE_GRAPH") = diskann::defaults::SATURATE_GRAPH;
    default_values.attr("GRAPH_DEGREE") = diskann::defaults::MAX_DEGREE;
    default_values.attr("COMPLEXITY") = diskann::defaults::BUILD_LIST_SIZE;
    default_values.attr("PQ_DISK_BYTES") = (uint32_t)0;
    default_values.attr("USE_PQ_BUILD") = false;
    default_values.attr("NUM_PQ_BYTES") = (uint32_t)0;
    default_values.attr("USE_OPQ") = false;

    add_variant<float>(m, "float", "Float");
    add_variant<uint8_t>(m, "uint8", "UInt8");
    add_variant<int8_t>(m, "int8", "Int8");

    py::enum_<Metric>(m, "Metric")
        .value("L2", Metric::L2)
        .value("INNER_PRODUCT", Metric::INNER_PRODUCT)
        .export_values();
}
