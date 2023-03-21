// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <string>
#include <memory>

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
        const std::string index_path = index_path_prefix + std::string("_disk.index");
        int load_success = pq_flash_index->load(num_threads, index_path.c_str());
        if (load_success != 0)
        {
            return load_success;
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

    void search(std::vector<T> &query, const _u64 query_idx, const _u64 dim, const _u64 num_queries, const _u64 knn,
                const _u64 l_search, const _u64 beam_width, std::vector<unsigned> &ids, std::vector<float> &dists)
    {
        QueryStats stats;
        if (ids.size() < knn * num_queries)
        {
            ids.resize(knn * num_queries);
            dists.resize(knn * num_queries);
        }
        std::vector<_u64> _u64_ids(knn);
        pq_flash_index->cached_beam_search(query.data() + (query_idx * dim), knn, l_search, _u64_ids.data(),
                                           dists.data() + (query_idx * knn), beam_width, &stats);
        for (_u64 i = 0; i < knn; i++)
            ids[(query_idx * knn) + i] = _u64_ids[i];
    }

    void batch_search(std::vector<T> &queries, const _u64 dim, const _u64 num_queries, const _u64 knn,
                      const _u64 l_search, const _u64 beam_width, std::vector<unsigned> &ids, std::vector<float> &dists,
                      const int num_threads)
    {
        if (ids.size() < knn * num_queries)
        {
            ids.resize(knn * num_queries);
            dists.resize(knn * num_queries);
        }
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t q = 0; q < num_queries; ++q)
        {
            std::vector<_u64> u64_ids(knn);

            pq_flash_index->cached_beam_search(queries.data() + q * dim, knn, l_search, u64_ids.data(),
                                               dists.data() + q * knn, beam_width);
            for (_u64 i = 0; i < knn; i++)
                ids[(q * knn) + i] = u64_ids[i];
        }
    }

    auto search_numpy_input(py::array_t<T, py::array::c_style | py::array::forcecast> &query, const _u64 dim,
                            const _u64 knn, const _u64 l_search, const _u64 beam_width)
    {
        py::array_t<unsigned> ids(knn);
        py::array_t<float> dists(knn);

        std::vector<unsigned> u32_ids(knn);
        std::vector<_u64> u64_ids(knn);
        QueryStats stats;

        pq_flash_index->cached_beam_search(query.data(), knn, l_search, u64_ids.data(), dists.mutable_data(),
                                           beam_width, &stats);

        auto r = ids.mutable_unchecked<1>();
        for (_u64 i = 0; i < knn; ++i)
            r(i) = (unsigned)u64_ids[i];

        return std::make_pair(ids, dists);
    }

    auto batch_search_numpy_input(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, const _u64 dim,
                                  const _u64 num_queries, const _u64 knn, const _u64 l_search, const _u64 beam_width,
                                  const int num_threads)
    {
        py::array_t<unsigned> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        std::vector<_u64> u64_ids(knn * num_queries);
        diskann::QueryStats *stats = new diskann::QueryStats[num_queries];

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < num_queries; i++)
        {
            pq_flash_index->cached_beam_search(queries.data(i), knn, l_search, u64_ids.data() + i * knn,
                                               dists.mutable_data(i), beam_width, stats + i);
        }

        auto r = ids.mutable_unchecked();
        for (_u64 i = 0; i < num_queries; ++i)
            for (_u64 j = 0; j < knn; ++j)
                r(i, j) = (unsigned)u64_ids[i * knn + j];

        std::unordered_map<std::string, double> collective_stats;
        collective_stats["mean_latency"] = diskann::get_mean_stats<double>(
            stats, num_queries, [](const diskann::QueryStats &stats) { return stats.total_us; });
        collective_stats["latency_999"] = diskann::get_percentile_stats<double>(
            stats, num_queries, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });
        collective_stats["mean_ssd_ios"] = diskann::get_mean_stats<double>(
            stats, num_queries, [](const diskann::QueryStats &stats) { return stats.n_ios; });
        collective_stats["mean_dist_comps"] = diskann::get_mean_stats<double>(
            stats, num_queries, [](const diskann::QueryStats &stats) { return stats.n_cmps; });
        delete[] stats;
        return std::make_pair(std::make_pair(ids, dists), collective_stats);
    }

    auto batch_range_search_numpy_input(py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
                                        const _u64 dim, const _u64 num_queries, const double range,
                                        const _u64 min_list_size, const _u64 max_list_size, const _u64 beam_width,
                                        const int num_threads)
    {
        py::array_t<unsigned> offsets(num_queries + 1);

        std::vector<std::vector<_u64>> u64_ids(num_queries);
        std::vector<std::vector<float>> dists(num_queries);

        auto offsets_mutable = offsets.mutable_unchecked();
        offsets_mutable(0) = 0;

        diskann::QueryStats *stats = new diskann::QueryStats[num_queries];

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < num_queries; i++)
        {
            _u32 res_count = pq_flash_index->range_search(queries.data(i), range, min_list_size, max_list_size,
                                                          u64_ids[i], dists[i], beam_width, stats + i);
            offsets_mutable(i + 1) = res_count;
        }

        uint64_t total_res_count = 0;
        for (_u64 i = 0; i < num_queries; ++i)
        {
            total_res_count += offsets_mutable(i + 1);
        }

        py::array_t<unsigned> ids(total_res_count);
        py::array_t<float> res_dists(total_res_count);

        auto ids_mutable = ids.mutable_unchecked();
        auto res_dists_mutable = res_dists.mutable_unchecked();
        size_t pos = 0;
        for (_u64 i = 0; i < num_queries; ++i)
        {
            for (_u64 j = 0; j < offsets_mutable(i + 1); ++j)
            {
                ids_mutable(pos) = (unsigned)u64_ids[i][j];
                res_dists_mutable(pos++) = dists[i][j];
            }
            offsets_mutable(i + 1) = offsets_mutable(i) + offsets_mutable(i + 1);
        }

        std::unordered_map<std::string, double> collective_stats;
        collective_stats["mean_latency"] = diskann::get_mean_stats<double>(
            stats, num_queries, [](const diskann::QueryStats &stats) { return stats.total_us; });
        collective_stats["latency_999"] = diskann::get_percentile_stats<double>(
            stats, num_queries, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });
        collective_stats["mean_ssd_ios"] = diskann::get_mean_stats<double>(
            stats, num_queries, [](const diskann::QueryStats &stats) { return stats.n_ios; });
        collective_stats["mean_dist_comps"] = diskann::get_mean_stats<double>(
            stats, num_queries, [](const diskann::QueryStats &stats) { return stats.n_cmps; });
        delete[] stats;
        return std::make_pair(std::make_pair(offsets, std::make_pair(ids, res_dists)), collective_stats);
    }
};

PYBIND11_MODULE(diskannpy, m)
{
    m.doc() = "DiskANN Python Bindings";
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

    py::bind_vector<std::vector<unsigned>>(m, "VectorUnsigned");
    py::bind_vector<std::vector<float>>(m, "VectorFloat");
    py::bind_vector<std::vector<int8_t>>(m, "VectorInt8");
    py::bind_vector<std::vector<uint8_t>>(m, "VectorUInt8");

    py::enum_<Metric>(m, "Metric")
        .value("L2", Metric::L2)
        .value("INNER_PRODUCT", Metric::INNER_PRODUCT)
        .export_values();

    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def(
            "set",
            [](Parameters &self, const std::string &name, py::object value) {
                if (py::isinstance<py::bool_>(value))
                {
                    return self.Set(name, py::cast<bool>(value));
                }
                else if (py::isinstance<py::int_>(value))
                {
                    return self.Set(name, py::cast<unsigned>(value));
                }
                else if (py::isinstance<py::float_>(value))
                {
                    return self.Set(name, py::cast<float>(value));
                }
            },
            py::arg("name"), py::arg("value"));

    py::class_<Neighbor>(m, "Neighbor")
        .def(py::init<>())
        .def(py::init<unsigned, float>())
        .def(py::self < py::self)
        .def(py::self == py::self);

    py::class_<AlignedFileReader>(m, "AlignedFileReader");

#ifdef _WINDOWS
    py::class_<WindowsAlignedFileReader>(m, "WindowsAlignedFileReader").def(py::init<>());
#else
    py::class_<LinuxAlignedFileReader>(m, "LinuxAlignedFileReader").def(py::init<>());
#endif

    m.def(
        "omp_set_num_threads", [](const size_t num_threads) { omp_set_num_threads(num_threads); },
        py::arg("num_threads") = 1);

    m.def("omp_get_max_threads", []() { return omp_get_max_threads(); });

    m.def(
        "load_aligned_bin_float",
        [](const std::string &path, std::vector<float> &data) {
            float *data_ptr = nullptr;
            size_t num, dims, aligned_dims;
            load_aligned_bin<float>(path, data_ptr, num, dims, aligned_dims);
            data.assign(data_ptr, data_ptr + num * aligned_dims);
            auto l = py::list(3);
            l[0] = py::int_(num);
            l[1] = py::int_(dims);
            l[2] = py::int_(aligned_dims);
            aligned_free(data_ptr);
            return l;
        },
        py::arg("path"), py::arg("data"));

    m.def(
        "load_truthset",
        [](const std::string &path, std::vector<unsigned> &ids, std::vector<float> &distances) {
            unsigned *id_ptr = nullptr;
            float *dist_ptr = nullptr;
            size_t num, dims;
            load_truthset(path, id_ptr, dist_ptr, num, dims);
            // TODO: Remove redundant copies.
            ids.assign(id_ptr, id_ptr + num * dims);
            distances.assign(dist_ptr, dist_ptr + num * dims);
            auto l = py::list(2);
            l[0] = py::int_(num);
            l[1] = py::int_(dims);
            delete[] id_ptr;
            delete[] dist_ptr;
            return l;
        },
        py::arg("path"), py::arg("ids"), py::arg("distances"));

    m.def(
        "calculate_recall",
        [](const unsigned num_queries, std::vector<unsigned> &ground_truth_ids, std::vector<float> &ground_truth_dists,
           const unsigned ground_truth_dims, std::vector<unsigned> &results, const unsigned result_dims,
           const unsigned recall_at) {
            unsigned *gti_ptr = ground_truth_ids.data();
            float *gtd_ptr = ground_truth_dists.data();
            unsigned *r_ptr = results.data();

            double total_recall = 0;
            std::set<unsigned> gt, res;
            for (size_t i = 0; i < num_queries; i++)
            {
                gt.clear();
                res.clear();
                size_t tie_breaker = recall_at;
                if (gtd_ptr != nullptr)
                {
                    tie_breaker = recall_at - 1;
                    float *gt_dist_vec = gtd_ptr + ground_truth_dims * i;
                    while (tie_breaker < ground_truth_dims && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                        tie_breaker++;
                }

                gt.insert(gti_ptr + ground_truth_dims * i, gti_ptr + ground_truth_dims * i + tie_breaker);
                res.insert(r_ptr + result_dims * i, r_ptr + result_dims * i + recall_at);
                unsigned cur_recall = 0;
                for (auto &v : gt)
                {
                    if (res.find(v) != res.end())
                    {
                        cur_recall++;
                    }
                }
                total_recall += cur_recall;
            }
            return py::float_(total_recall / (num_queries) * (100.0 / recall_at));
        },
        py::arg("num_queries"), py::arg("ground_truth_ids"), py::arg("ground_truth_dists"),
        py::arg("ground_truth_dims"), py::arg("results"), py::arg("result_dims"), py::arg("recall_at"));

    m.def(
        "calculate_recall_numpy_input",
        [](const unsigned num_queries, std::vector<unsigned> &ground_truth_ids, std::vector<float> &ground_truth_dists,
           const unsigned ground_truth_dims, py::array_t<unsigned, py::array::c_style | py::array::forcecast> &results,
           const unsigned result_dims, const unsigned recall_at) {
            unsigned *gti_ptr = ground_truth_ids.data();
            float *gtd_ptr = ground_truth_dists.data();
            unsigned *r_ptr = results.mutable_data();

            double total_recall = 0;
            std::set<unsigned> gt, res;
            for (size_t i = 0; i < num_queries; i++)
            {
                gt.clear();
                res.clear();
                size_t tie_breaker = recall_at;
                if (gtd_ptr != nullptr)
                {
                    tie_breaker = recall_at - 1;
                    float *gt_dist_vec = gtd_ptr + ground_truth_dims * i;
                    while (tie_breaker < ground_truth_dims && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                        tie_breaker++;
                }

                gt.insert(gti_ptr + ground_truth_dims * i, gti_ptr + ground_truth_dims * i + tie_breaker);
                res.insert(r_ptr + result_dims * i, r_ptr + result_dims * i + recall_at);
                unsigned cur_recall = 0;
                for (auto &v : gt)
                {
                    if (res.find(v) != res.end())
                    {
                        cur_recall++;
                    }
                }
                total_recall += cur_recall;
            }
            return py::float_(total_recall / (num_queries) * (100.0 / recall_at));
        },
        py::arg("num_queries"), py::arg("ground_truth_ids"), py::arg("ground_truth_dists"),
        py::arg("ground_truth_dims"), py::arg("results"), py::arg("result_dims"), py::arg("recall_at"));

    m.def(
        "save_bin_u32",
        [](const std::string &file_name, std::vector<unsigned> &data, size_t npts, size_t dims) {
            save_bin<_u32>(file_name, data.data(), npts, dims);
        },
        py::arg("file_name"), py::arg("data"), py::arg("npts"), py::arg("dims"));

    py::class_<DiskANNIndex<float>>(m, "DiskANNFloatIndex")
        .def(py::init([](diskann::Metric metric) {
            return std::unique_ptr<DiskANNIndex<float>>(new DiskANNIndex<float>(metric));
        }))
        .def("cache_bfs_levels", &DiskANNIndex<float>::cache_bfs_levels, py::arg("num_nodes_to_cache"))
        .def("load_index", &DiskANNIndex<float>::load_index, py::arg("index_path_prefix"), py::arg("num_threads"),
             py::arg("num_nodes_to_cache"), py::arg("cache_mechanism") = 1)
        .def("search", &DiskANNIndex<float>::search, py::arg("query"), py::arg("query_idx"), py::arg("dim"),
             py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("ids"),
             py::arg("dists"))
        .def("batch_search", &DiskANNIndex<float>::batch_search, py::arg("queries"), py::arg("dim"),
             py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("ids"),
             py::arg("dists"), py::arg("num_threads"))
        .def("search_numpy_input", &DiskANNIndex<float>::search_numpy_input, py::arg("query"), py::arg("dim"),
             py::arg("knn"), py::arg("l_search"), py::arg("beam_width"))
        .def("batch_search_numpy_input", &DiskANNIndex<float>::batch_search_numpy_input, py::arg("queries"),
             py::arg("dim"), py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"),
             py::arg("num_threads"))
        .def("batch_range_search_numpy_input", &DiskANNIndex<float>::batch_range_search_numpy_input, py::arg("queries"),
             py::arg("dim"), py::arg("num_queries"), py::arg("range"), py::arg("min_list_size"),
             py::arg("max_list_size"), py::arg("beam_width"), py::arg("num_threads"))
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
        .def("search", &DiskANNIndex<int8_t>::search, py::arg("query"), py::arg("query_idx"), py::arg("dim"),
             py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("ids"),
             py::arg("dists"))
        .def("batch_search", &DiskANNIndex<int8_t>::batch_search, py::arg("queries"), py::arg("dim"),
             py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("ids"),
             py::arg("dists"), py::arg("num_threads"))
        .def("search_numpy_input", &DiskANNIndex<int8_t>::search_numpy_input, py::arg("query"), py::arg("dim"),
             py::arg("knn"), py::arg("l_search"), py::arg("beam_width"))
        .def("batch_search_numpy_input", &DiskANNIndex<int8_t>::batch_search_numpy_input, py::arg("queries"),
             py::arg("dim"), py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"),
             py::arg("num_threads"))
        .def("batch_range_search_numpy_input", &DiskANNIndex<int8_t>::batch_range_search_numpy_input,
             py::arg("queries"), py::arg("dim"), py::arg("num_queries"), py::arg("range"), py::arg("min_list_size"),
             py::arg("max_list_size"), py::arg("beam_width"), py::arg("num_threads"))
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
        .def("search", &DiskANNIndex<uint8_t>::search, py::arg("query"), py::arg("query_idx"), py::arg("dim"),
             py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("ids"),
             py::arg("dists"))
        .def("batch_search", &DiskANNIndex<uint8_t>::batch_search, py::arg("queries"), py::arg("dim"),
             py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"), py::arg("ids"),
             py::arg("dists"), py::arg("num_threads"))
        .def("search_numpy_input", &DiskANNIndex<uint8_t>::search_numpy_input, py::arg("query"), py::arg("dim"),
             py::arg("knn"), py::arg("l_search"), py::arg("beam_width"))
        .def("batch_search_numpy_input", &DiskANNIndex<uint8_t>::batch_search_numpy_input, py::arg("queries"),
             py::arg("dim"), py::arg("num_queries"), py::arg("knn"), py::arg("l_search"), py::arg("beam_width"),
             py::arg("num_threads"))
        .def("batch_range_search_numpy_input", &DiskANNIndex<uint8_t>::batch_range_search_numpy_input,
             py::arg("queries"), py::arg("dim"), py::arg("num_queries"), py::arg("range"), py::arg("min_list_size"),
             py::arg("max_list_size"), py::arg("beam_width"), py::arg("num_threads"))
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
