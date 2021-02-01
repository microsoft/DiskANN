// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>

#include "index.h"
#include "utils.h"
#include "memory_mapper.h"

PYBIND11_MAKE_OPAQUE(std::vector<unsigned>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);

namespace py = pybind11;
using namespace diskann;

PYBIND11_MODULE(vamanapy, m) {
  m.doc() = "Vamana Python Bindings";
  m.attr("__version__") = "0.1.0";

  py::bind_vector<std::vector<unsigned>>(m, "VectorUnsigned");
  py::bind_vector<std::vector<float>>(m, "VectorFloat");

  py::enum_<Metric>(m, "Metric")
    .value("L2", Metric::L2)
    .value("INNER_PRODUCT", Metric::INNER_PRODUCT)
    .value("FAST_L2", Metric::FAST_L2)
    .value("PQ", Metric::PQ)
    .export_values();

  py::class_<Parameters>(m, "Parameters")
    .def(py::init<>())
    .def("set", [](Parameters &self, const std::string &name, py::object value) {
        if (py::isinstance<py::bool_>(value)) {
          return self.Set(name, py::cast<bool>(value));
        } else if (py::isinstance<py::int_>(value)) {
          return self.Set(name, py::cast<unsigned>(value));
        } else if (py::isinstance<py::float_>(value)){
          return self.Set(name, py::cast<float>(value));
        }
      }, py::arg("name"), py::arg("value"));

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

  m.def("set_num_threads", [](const size_t num_threads) {
      omp_set_num_threads(num_threads);
    }, py::arg("num_threads") = 1);

  m.def("load_aligned_bin_float", [](const std::string &path,
                                     std::vector<float> &data) {
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
    }, py::arg("path"), py::arg("data"));

  m.def("load_truthset", [](const std::string &path,
                            std::vector<unsigned> &ids,
                            std::vector<float> &distances) {
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
    }, py::arg("path"), py::arg("ids"), py::arg("distances"));

  m.def("calculate_recall", [](const unsigned num_queries,
                               std::vector<unsigned> &ground_truth_ids,
                               std::vector<float> &ground_truth_dists,
                               const unsigned ground_truth_dims,
                               std::vector<unsigned> &results,
                               const unsigned result_dims,
                               const unsigned recall_at) {
      unsigned *gti_ptr = ground_truth_ids.data();
      float *gtd_ptr = ground_truth_dists.data();
      unsigned *r_ptr = results.data();

      double total_recall = 0;
      std::set<unsigned> gt, res;
      for (size_t i = 0; i < num_queries; i++) {
        gt.clear();
        res.clear();
        size_t    tie_breaker = recall_at;
        if (gtd_ptr != nullptr) {
          tie_breaker = recall_at - 1;
          float *gt_dist_vec = gtd_ptr + ground_truth_dims * i;
          while (tie_breaker < ground_truth_dims &&
                  gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
            tie_breaker++;
        }

        gt.insert(gti_ptr + ground_truth_dims * i, gti_ptr + ground_truth_dims * i + tie_breaker);
        res.insert(r_ptr + result_dims * i, r_ptr + result_dims * i + recall_at);
        unsigned cur_recall = 0;
        for (auto &v : gt) {
          if (res.find(v) != res.end()) {
            cur_recall++;
          }
        }
        total_recall += cur_recall;
      }
      return py::float_(total_recall / (num_queries) * (100.0 / recall_at));
    }, py::arg("num_queries"), py::arg("ground_truth_ids"),
    py::arg("ground_truth_dists"), py::arg("ground_truth_dims"),
    py::arg("results"), py::arg("result_dims"), py::arg("recall_at"));

  m.def("save_bin_u32", [](const std::string& file_name,
                           std::vector<unsigned> &data, size_t npts,
                           size_t dims) {
      save_bin<_u32>(file_name, data.data(), npts, dims);
    }, py::arg("file_name"), py::arg("data"), py::arg("npts"),
    py::arg("dims"));

  py::class_<Index<float, int>>(m, "SinglePrecisionIndex")
    .def(py::init<Metric, const char *, const size_t, const size_t,
      const size_t, const bool, const bool, const bool>(),
      py::arg("m"), py::arg("filename"),
      py::arg("max_points") = 0, py::arg("nd") = 0,
      py::arg("num_frozen_pts") = 0, py::arg("enable_tags") = false,
      py::arg("store_data") = true, py::arg("support_eager_delete") = false)
    .def("save", [](Index<float, int> &self,
                    const std::string file_name) {
        return self.save(file_name.c_str());
      }, py::arg("file_name"))
    .def("load", [](Index<float, int> &self,
                    const std::string file_name, bool load_tags,
                    const std::string tag_file_name) {
        if (tag_file_name == "") {
          return self.load(file_name.c_str(), load_tags, NULL);
        } else {
          return self.load(file_name.c_str(), load_tags, tag_file_name.c_str());
        }
      }, py::arg("file_name"), py::arg("load_tags") = false,
      py::arg("tag_file_name") = "")
    .def("generate_random_frozen_points", [](Index<float, int> &self,
                                             const std::string file_name) {
        if (file_name == "") {
          return self.generate_random_frozen_points(NULL);
        } else {
          return self.generate_random_frozen_points(file_name.c_str());
        }
      }, py::arg("file_name") = "")
    .def("build", [](Index<float, int> &self, Parameters &parameters,
                     const std::vector<int> &tags) {
        if (tags.size() == 0) {
          return self.build(parameters);
        } else {
          return self.build(parameters, tags);
        }
      }, py::arg("parameters"), py::arg("tags"))
    .def("search", [](Index<float, int> &self, std::vector<float> &query,
                      const size_t query_index, const size_t knn,
                      const size_t num_queries, const size_t l_search,
                      std::vector<unsigned> &ids, const size_t id_index) {
        if (ids.size() == 0) {
          ids.resize(knn * num_queries);
        }

        self.search(query.data() + query_index, knn, l_search,
                    ids.data() + id_index);
      }, py::arg("query"), py::arg("query_index"), py::arg("knn") = 10,
      py::arg("num_queries"), py::arg("l_search"), py::arg("ids"),
      py::arg("id_index"))
    .def("search_with_tags", [](Index<float, int> &self,
                                std::vector<float> &query, size_t knn,
                                size_t l_search, std::vector<int> &tags,
                                unsigned num_frozen_pts,
                                std::vector<unsigned> &indices_buffer) {
        if (indices_buffer.size() == 0) {
          return self.search_with_tags(query.data(), knn, l_search,
                                       tags.data(), num_frozen_pts, NULL);
        } else {
          return self.search_with_tags(query.data(), knn, l_search,
                                       tags.data(), num_frozen_pts,
                                       indices_buffer.data());
        }
      }, py::arg("query"), py::arg("knn") = 10, py::arg("l_search"),
      py::arg("tags"), py::arg("num_frozen_pts"), py::arg("indices_buffer"))
    .def("read_just_data", &Index<float, int>::readjust_data)
    .def("insert_point", [](Index<float, int> &self,
                            const std::vector<float> &point,
                            const Parameters &parameter,
                            std::vector<Neighbor> &pool,
                            std::vector<Neighbor> &tmp,
                            tsl::robin_set<unsigned> &visited,
                            std::vector<SimpleNeighbor> &cut_graph,
                            const int tag) {
        return self.insert_point(point.data(), parameter, pool, tmp, visited,
                                 cut_graph, tag);
      }, py::arg("point"), py::arg("parameter"), py::arg("pool"),
      py::arg("tmp"), py::arg("visited"), py::arg("cut_graph"), py::arg("tag"))
    .def("enable_delete", &Index<float, int>::enable_delete)
    .def("disable_delete", [](Index<float, int> &self,
                              const Parameters &parameters,
                              const bool consolidate) {
        return self.disable_delete(parameters, consolidate);
      }, py::arg("parameters"), py::arg("consolidate") = false)
    .def("delete_point", &Index<float, int>::delete_point)
    .def("eager_delete", [](Index<float, int> &self, const int tag,
                            const Parameters &parameters) {
        return self.eager_delete(tag, parameters);
      }, py::arg("tag"), py::arg("parameters"))
    .def("optimize_graph", &Index<float, int>::optimize_graph)
    .def("search_with_optimized_graph", [](Index<float, int> &self,
                                           std::vector<float> &query,
                                           const size_t query_index,
                                           const size_t knn,
                                           const size_t num_queries,
                                           const size_t l_search,
                                           std::vector<unsigned> &ids,
                                           const size_t id_index) {
        if (ids.size() == 0) {
          ids.resize(knn * num_queries);
        }

        self.search_with_opt_graph(query.data() + query_index, knn,
                                   l_search, ids.data() + id_index);
      }, py::arg("query"), py::arg("query_index"), py::arg("knn") = 10,
      py::arg("num_queries"), py::arg("l_search"), py::arg("ids"),
      py::arg("id_index"))
    .def("single_numpy_query", [](Index<float, int> &self,
                                  py::array_t<float,
                                              py::array::c_style |
                                              py::array::forcecast> &query,
                                  const size_t knn,
                                  const size_t l_search) {
        py::array_t<unsigned> ids(knn);
        self.search_with_opt_graph(query.data(), knn,
                                   l_search, ids.mutable_data());
        return ids;
      }, py::arg("query"), py::arg("knn") = 10, py::arg("l_search"))
    .def("batch_numpy_query", [](Index<float, int> &self,
                                 py::array_t<float,
                                             py::array::c_style |
                                             py::array::forcecast> &queries,
                                 const size_t knn,
                                 const size_t num_queries,
                                 const size_t l_search) {
        py::array_t<unsigned> ids(knn * num_queries);
        for (unsigned i = 0; i < num_queries; i++) {
          self.search_with_opt_graph(queries.data(i), knn,
                                     l_search, ids.mutable_data(i * knn));
        }
        return ids;
      }, py::arg("queries"), py::arg("knn") = 10, py::arg("num_queries"),
      py::arg("l_search"));
}
