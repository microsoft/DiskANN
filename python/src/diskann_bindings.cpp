// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>

#include "linux_aligned_file_reader.h"
#include "pq_flash_index.h"


PYBIND11_MAKE_OPAQUE(std::vector<unsigned>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);

namespace py = pybind11;
using namespace diskann;

std::unique_ptr<PQFlashIndex<float>> FloatPQFlashIndexCreator() {
  std::shared_ptr<AlignedFileReader>   reader(new LinuxAlignedFileReader());
  auto                                 index = new PQFlashIndex<float>(reader);
  std::unique_ptr<PQFlashIndex<float>> unique_ptr_index(index);
  return unique_ptr_index;
}

PYBIND11_MODULE(diskannpy, m) {
  m.doc() = "DiskANN Python Bindings";
  m.attr("__version__") = "0.1.0";

  py::bind_vector<std::vector<unsigned>>(m, "VectorUnsigned");
  py::bind_vector<std::vector<float>>(m, "VectorFloat");

  py::enum_<Metric>(m, "Metric").value("L2", Metric::L2).export_values();

  py::class_<Parameters>(m, "Parameters")
      .def(py::init<>())
      .def(
          "set",
          [](Parameters &self, const std::string &name, py::object value) {
            if (py::isinstance<py::bool_>(value)) {
              return self.Set(name, py::cast<bool>(value));
            } else if (py::isinstance<py::int_>(value)) {
              return self.Set(name, py::cast<unsigned>(value));
            } else if (py::isinstance<py::float_>(value)) {
              return self.Set(name, py::cast<float>(value));
            }
          },
          py::arg("name"), py::arg("value"));

  py::class_<Neighbor>(m, "Neighbor")
      .def(py::init<>())
      .def(py::init<unsigned, float, bool>())
      .def(py::self < py::self)
      .def(py::self == py::self);

  py::class_<SimpleNeighbor>(m, "SimpleNeighbor")
      .def(py::init<>())
      .def(py::init<unsigned, float>())
      .def(py::self < py::self)
      .def(py::self == py::self);

  py::class_<AlignedFileReader>(m, "AlignedFileReader");

  py::class_<LinuxAlignedFileReader>(m, "LinuxAlignedFileReader")
      .def(py::init<>());
  //    .def("get_ctx", &LinuxAlignedFileReader::get_ctx)
  //    .def("register_thread", &LinuxAlignedFileReader::register_thread)
  //    .def("open", &LinuxAlignedFileReader::open)
  //    .def("close", &LinuxAlignedFileReader::close)
  //    .def("read", &LinuxAlignedFileReader::read);

  m.def(
      "set_num_threads",
      [](const size_t num_threads) { omp_set_num_threads(num_threads); },
      py::arg("num_threads") = 1);

  m.def(
      "load_aligned_bin_float",
      [](const std::string &path, std::vector<float> &data) {
        float *data_ptr = nullptr;
        size_t num, dims, aligned_dims;
        load_aligned_bin<float>(path, data_ptr, num, dims, aligned_dims);
        // TODO: Remove redundant copy.
        data.assign(data_ptr, data_ptr + num * dims);
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
      [](const std::string &path, std::vector<unsigned> &ids,
         std::vector<float> &distances) {
        unsigned *id_ptr = nullptr;
        float *   dist_ptr = nullptr;
        size_t    num, dims;
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
      [](const unsigned num_queries, std::vector<unsigned> &ground_truth_ids,
         std::vector<float> &ground_truth_dists,
         const unsigned ground_truth_dims, std::vector<unsigned> &results,
         const unsigned result_dims, const unsigned recall_at) {
        unsigned *gti_ptr = ground_truth_ids.data();
        float *   gtd_ptr = ground_truth_dists.data();
        unsigned *r_ptr = results.data();

        double             total_recall = 0;
        std::set<unsigned> gt, res;
        for (size_t i = 0; i < num_queries; i++) {
          gt.clear();
          res.clear();
          size_t tie_breaker = recall_at;
          if (gtd_ptr != nullptr) {
            tie_breaker = recall_at - 1;
            float *gt_dist_vec = gtd_ptr + ground_truth_dims * i;
            while (tie_breaker < ground_truth_dims &&
                   gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
              tie_breaker++;
          }

          gt.insert(gti_ptr + ground_truth_dims * i,
                    gti_ptr + ground_truth_dims * i + tie_breaker);
          res.insert(r_ptr + result_dims * i,
                     r_ptr + result_dims * i + recall_at);
          unsigned cur_recall = 0;
          for (auto &v : gt) {
            if (res.find(v) != res.end()) {
              cur_recall++;
            }
          }
          total_recall += cur_recall;
        }
        return py::float_(total_recall / (num_queries) * (100.0 / recall_at));
      },
      py::arg("num_queries"), py::arg("ground_truth_ids"),
      py::arg("ground_truth_dists"), py::arg("ground_truth_dims"),
      py::arg("results"), py::arg("result_dims"), py::arg("recall_at"));

  m.def(
      "save_bin_u32",
      [](const std::string &file_name, std::vector<unsigned> &data, size_t npts,
         size_t dims) { save_bin<_u32>(file_name, data.data(), npts, dims); },
      py::arg("file_name"), py::arg("data"), py::arg("npts"), py::arg("dims"));

  py::class_<PQFlashIndex<float>>(m, "DiskANNFloatIndex")
      .def(py::init(&FloatPQFlashIndexCreator))
      .def(
          "load_index",
          [](PQFlashIndex<float> &self, const std::string &index_path_prefix) {
            const std::string pq_path = index_path_prefix + std::string("_pq");
            const std::string index_path =
                index_path_prefix + std::string("_disk.index");
            self.load(1, pq_path.c_str(), index_path.c_str());
            std::vector<uint32_t> node_list;
            _u64                  num_nodes_to_cache = 100000;
            self.cache_bfs_levels(num_nodes_to_cache, node_list);
            std::cout << "loaded index, cached " << node_list.size()
                      << " nodes based on BFS" << std::endl;
          },
          py::arg("index_path_prefix"))
      .def(
          "search",
          [](PQFlashIndex<float> &self, const float *query, const _u64 dim,
             const _u64 knn, const _u64 l_search, const _u64 beam_width,
             _u64 *ids, float *dists) {
            QueryStats stats;
            self.cached_beam_search(query, knn, l_search, ids, dists,
                                    beam_width, &stats);
          },
          py::arg("query"), py::arg("dim"), py::arg("knn") = 10,
          py::arg("l_search"), py::arg("beam_width"), py::arg("ids"),
          py::arg("dists"))
      .def(
          "batch_search",
          [](PQFlashIndex<float> &self, const float *query_data,
             const _u64 nqueries, const _u64 dim, const _u64 knn,
             const _u64 l_search, const _u64 beam_width, _u64 *ids,
             float *dists) {
#pragma omp parallel for schedule(dynamic, 1)
            for (_u64 i = 0; i < nqueries; ++i)
              self.cached_beam_search(query_data + i * dim, knn, l_search,
                                      ids + i * knn, dists + i * knn,
                                      beam_width);
          },
          py::arg("query_data"), py::arg("nqueries"), py::arg("dim"),
          py::arg("knn") = 10, py::arg("l_search"), py::arg("beam_width"),
          py::arg("ids"), py::arg("dists"));
      //.def(
      //  "build",
      //     [](PQFlashIndex<float> &          self,
      //        const char *dataFilePath, const std::string &index_prefix_path,
      //        unsigned R, unsigned L, double final_index_ram_limit,
      //        double indexing_ram_budget, unsigned num_threads) {
      //       std::string pq_pivots_path = index_prefix_path + "_pq_pivots";
      //       std::string pq_compressed_vectors_path =
      //           index_prefix_path + "_pq_compressed.bin";
      //       std::string mem_index_path = index_prefix_path + "_mem.index";
      //       std::string disk_index_path = index_prefix_path + "_disk.index";
      //       std::string medoids_path = disk_index_path + "_medoids.bin";
      //       std::string centroids_path = disk_index_path + "_centroids.bin";
      //       std::string sample_base_prefix = index_prefix_path + "_sample";

      //       if (num_threads != 0) {
      //         omp_set_num_threads(num_threads);
      //         mkl_set_num_threads(num_threads);
      //       }

      //       cout << "Starting index build: R=" << R << " L=" << L
      //            << " Query RAM budget: " << final_index_ram_limit
      //            << " Indexing RAM budget: " << indexing_ram_budget
      //            << " T: " << num_threads << std::endl;

      //       auto s = std::chrono::high_resolution_clock::now();

      //       size_t points_num, dim;

      //       get_bin_metadata(dataFilePath, points_num, dim);

      //       size_t num_pq_chunks =
      //           (size_t)(std::floor)(_u64(final_index_ram_limit / points_num));

      //       num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
      //       num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
      //       num_pq_chunks =
      //           num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

      //       cout << "Compressing " << dim << "-dimensional data into "
      //            << num_pq_chunks << " bytes per vector." << std::endl;

      //       size_t train_size, train_dim;
      //       float *train_data;

      //       double p_val = ((double) TRAINING_SET_SIZE / (double) points_num);
      //       // generates random sample and sets it to train_data and updates
      //       // train_size
      //       gen_random_slice<T>(dataFilePath, p_val, train_data, train_size,
      //                           train_dim);

      //       cout << "Training data loaded of size " << train_size << std::endl;

      //       generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256,
      //                          (uint32_t) num_pq_chunks, 15, pq_pivots_path);
      //       generate_pq_data_from_pivots<T>(
      //           dataFilePath, 256, (uint32_t) num_pq_chunks, pq_pivots_path,
      //           pq_compressed_vectors_path);

      //       delete[] train_data;

      //       build_merged_vamana_index<T>(
      //           dataFilePath, _compareMetric, L, R, p_val, indexing_ram_budget,
      //           mem_index_path, medoids_path, centroids_path);

      //       create_disk_layout<T>(dataFilePath, mem_index_path,
      //                             disk_index_path);

      //       double sample_sampling_rate = (150000.0 / points_num);
      //       gen_random_slice<T>(dataFilePath, sample_base_prefix,
      //                           sample_sampling_rate);

      //       std::remove(mem_index_path.c_str());

      //       auto e = std::chrono::high_resolution_clock::now();
      //       std::chrono::duration<double> diff = e - s;
      //       cout << "Indexing time: " << diff.count() << std::endl;
      //     });
}
