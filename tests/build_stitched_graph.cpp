#include <random>
#include "index.h"

size_t random(size_t range_from, size_t range_to) {
  std::random_device                    rand_dev;
  std::mt19937                          generator(rand_dev());
  std::uniform_int_distribution<size_t> distr(range_from, range_to);
  return distr(generator);
}

template<typename T>
void prune_and_save(std::string data_file, std::string stitched_index,
                    std::string pruned_index, _u32 R,
                    std::map<std::string, size_t> labels_dist,
                    std::mt19937                  labels_rng_gen) {
  diskann::Index<T> index(diskann::L2, data_file.c_str());
  index.load(stitched_index.c_str());  // to load NSG

  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", 1.2);
  paras.Set<bool>("saturate_graph", 1);
  paras.Set<std::map<std::string, size_t>>("labels_counts", labels_dist);

  index.prune_all_nodes(paras);

  index.save(pruned_index.c_str());
}

template<typename T>
void save_in_memory_index(const char*                           filename,
                          std::vector<tsl::robin_set<unsigned>> full_graph,
                          tsl::robin_map<std::string, _u64>     eps,
                          bool use_universal_label, const char* universal_label,
                          const char* map_file) {
  std::ifstream source(map_file, std::ios::binary);
  std::ofstream dest((std::string) filename + "_labels.txt", std::ios::binary);

  dest << source.rdbuf();

  source.close();
  dest.close();
  std::ofstream medoid_writer(std::string(filename) + "_labels_to_medoids.txt");
  for (auto iter : eps) {
    medoid_writer << iter.first << ", " << iter.second << std::endl;
    std::cout << iter.first << ", " << iter.second << std::endl;
  }
  medoid_writer.close();

  if (use_universal_label) {
    std::ofstream universal_label_writer(std::string(filename) +
                                         "_universal_label.txt");
    universal_label_writer << universal_label << std::endl;
    universal_label_writer.close();
  }
  std::cout << "Universal label written" << std::endl;

  size_t   index_size = 0;
  unsigned max_deg = 0, ep = 0;
  _u64     total_gr_edges = 0;
  for (auto& z : full_graph) {
    max_deg = std::max(max_deg, (_u32) z.size());
  }
  std::cout << "Max degree is " << max_deg << std::endl;

  std::ofstream out(std::string(filename), std::ios::binary | std::ios::out);
  out.write((char*) &index_size, sizeof(uint64_t));
  out.write((char*) &max_deg, sizeof(unsigned));
  out.write((char*) &ep, sizeof(unsigned));
  for (unsigned i = 0; i < full_graph.size(); i++) {
    unsigned GK = (unsigned) full_graph[i].size();
    out.write((char*) &GK, sizeof(unsigned));
    for (auto& e : full_graph[i])
      out.write((char*) &e, sizeof(unsigned));
    total_gr_edges += GK;
  }
  index_size = out.tellp();
  out.seekp(0, std::ios::beg);
  out.write((char*) &index_size, sizeof(uint64_t));
  out.close();

  std::cout << "Full graph written" << std::endl;

  std::cout << "Avg degree: "
            << ((float) total_gr_edges) / ((float) (full_graph.size()))
            << std::endl;
}

std::vector<std::vector<unsigned>> load_in_memory_index(const char* filename,
                                                        _u64&       file_size,
                                                        _u64&       ep) {
  std::ifstream in(filename, std::ios::binary);
  size_t        expected_file_size, width;
  in.read((char*) &expected_file_size, sizeof(_u64));
  file_size += expected_file_size;
  in.read((char*) &width, sizeof(unsigned));
  in.read((char*) &ep, sizeof(unsigned));
  std::cout << "Loading vamana index " << filename << "..." << std::flush;
  std::vector<std::vector<unsigned>> filt_graph;

  size_t   cc = 0;
  unsigned nodes = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char*) &k, sizeof(unsigned));
    if (in.eof())
      break;
    cc += k;
    ++nodes;
    std::vector<unsigned> tmp(k);
    in.read((char*) tmp.data(), k * sizeof(unsigned));

    filt_graph.emplace_back(tmp);
    if (nodes % 10000000 == 0)
      std::cout << "." << std::flush;
  }
  /*if (_final_graph.size() != _nd) {
    std::cout << "ERROR. mismatch in number of points. Graph has "
              << _final_graph.size() << " points and loaded dataset has " << _nd
              << " points. " << std::endl;
    return;
  }*/

  std::cout << "..done. Index has " << nodes << " nodes and " << cc
            << " out-edges" << std::endl;
  return (filt_graph);
}

template<typename T>
int build_in_memory_index2(const std::string& data_path, const unsigned R,
                           const unsigned Lr, const unsigned Lf,
                           const float alpha, const std::string& save_path,
                           const unsigned num_threads) {
  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("Lr", Lr);
  paras.Set<unsigned>("Lf", Lf);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  diskann::Index<T> index(diskann::L2, data_path.c_str());
  auto              s = std::chrono::high_resolution_clock::now();
  index.set_universal_label("0");

  index.build(paras);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

template<typename T>
tsl::robin_map<std::string, tsl::robin_map<_u64, _u64>> filter_base_faster(
    const std::string data_type, const std::string& coord_file,
    const std::string& map_file, const bool use_universal_label,
    const std::string universal_label, tsl::robin_set<std::string> label_set) {
  std::cout << "Loading base file " << coord_file << "...\n";
  tsl::robin_map<std::string, tsl::robin_map<_u64, _u64>> rev_map;
  std::map<std::string, std::ofstream>                    outputs;
  tsl::robin_map<std::string, _u32>                       num_filt;
  tsl::robin_map<std::string, _u32>                       line_c;

  std::string   line, token;
  unsigned      line_cnt = 0;
  unsigned      num_pts, num_dims;
  std::ifstream coord_stream(coord_file, std::ios::binary);
  std::ifstream map_stream(map_file);
  coord_stream.read((char*) &num_pts, sizeof(num_pts));
  coord_stream.read((char*) &num_dims, sizeof(num_dims));
  std::vector<T> pt(num_dims);
  // tsl::robin_map<std::string, _u32>                       line_c;

  for (auto& filter_label : label_set) {
    if (use_universal_label && filter_label == universal_label)
      continue;
    outputs[filter_label] = std::ofstream(coord_file + filter_label);
    num_filt[filter_label] = 0;
    line_c[filter_label] = 0;
    outputs[filter_label].write((char*) &num_filt[filter_label],
                                sizeof(num_filt[filter_label]));
    outputs[filter_label].write((char*) &num_dims, sizeof(num_dims));
  }

  while (std::getline(map_stream, line)) {
    std::istringstream iss(line);

    coord_stream.read((char*) pt.data(), sizeof(T) * num_dims);
    while (std::getline(iss, token, ',')) {
      token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
      token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
      if (use_universal_label && token == universal_label) {
        for (auto& filter_label : label_set) {
          if (filter_label == universal_label)
            continue;
          // rev_map[token][num_filt[token]] = line_cnt;
          outputs[filter_label].write((char*) pt.data(), sizeof(T) * num_dims);
          num_filt[token]++;
        }
      } else {
        // rev_map[token][num_filt[token]] = line_cnt;
        outputs[token].write((char*) pt.data(), sizeof(T) * num_dims);
        num_filt[token]++;
      }
    }
    line_cnt++;
  }

  for (auto& filter_label : label_set) {
    if (use_universal_label && filter_label == universal_label)
      continue;
    outputs[filter_label].seekp(0, std::ios::beg);
    outputs[filter_label].write((char*) &num_filt[filter_label],
                                sizeof(num_filt[filter_label]));
    outputs[filter_label].close();
  }

  return (rev_map);
}

template<typename T>
tsl::robin_map<std::string, tsl::robin_map<_u64, _u64>> filter_base(
    const std::string data_type, const std::string& coord_file,
    const std::string& map_file, const bool use_universal_label,
    const std::string universal_label, tsl::robin_set<std::string> label_set) {
  std::cout << "Loading base file " << coord_file << "...\n";
  tsl::robin_map<std::string, tsl::robin_map<_u64, _u64>> rev_map;

  for (auto& filter_label : label_set) {
    std::ifstream coord_stream(coord_file, std::ios::binary);
    std::ifstream map_stream(map_file);
    unsigned      num_pts, num_dims;
    coord_stream.read((char*) &num_pts, sizeof(num_pts));
    coord_stream.read((char*) &num_dims, sizeof(num_dims));
    unsigned num_filt_pts = 0;
    if (filter_label == universal_label)
      continue;
    std::string   filtered_base(coord_file + filter_label);
    std::ofstream out_coord_stream = std::ofstream(filtered_base);

    out_coord_stream.write((char*) &num_filt_pts, sizeof(num_filt_pts));
    out_coord_stream.write((char*) &num_dims, sizeof(num_dims));

    std::vector<T>        pt(num_dims);
    std::vector<unsigned> filtered_pts_index(0);
    std::string           line, token;
    unsigned              line_cnt = 0;
    // int max_label=INT_MIN;

    /*std::ifstream temp;
    temp.open(filtered_base);
    if (temp) {
      std::cout << filtered_base << " exists" << std::endl;
      temp.close();
      continue;
    }
    temp.close();*/
    std::cout << "Writing base file " << filtered_base << "...\n";

    while (std::getline(map_stream, line)) {
      std::istringstream iss(line);

      coord_stream.read((char*) pt.data(), sizeof(T) * num_dims);
      while (std::getline(iss, token, ',')) {
        token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
        if (token == filter_label ||
            (use_universal_label && token == universal_label)) {
          rev_map[filter_label][num_filt_pts] = line_cnt;
          filtered_pts_index.push_back(line_cnt);
          out_coord_stream.write((char*) pt.data(), sizeof(T) * num_dims);
          num_filt_pts++;
          break;
        }
      }
      line_cnt++;
    }
    std::cout << "Num filt points for label " << filter_label << " is "
              << num_filt_pts << std::endl;
    if (num_pts != line_cnt) {
      std::cout
          << "Error: Number of labels (ignoring unlabeled errors)- expected: "
          << num_pts << ", found:" << line_cnt << ")\n";
    }
    out_coord_stream.seekp(0, std::ios::beg);
    out_coord_stream.write((char*) &num_filt_pts, sizeof(num_filt_pts));
    out_coord_stream.close();
    /*coord_stream.clear();
    map_stream.clear();
    coord_stream.seekg(0, std::ios::beg);
    map_stream.seekg(0, std::ios::beg);*/
  }
  return (rev_map);
}

tsl::robin_set<std::string> parse_label_file2(
    std::string map_file, std::map<std::string, size_t>& labels_dist,
    std::mt19937& labels_rng_gen) {
  //_filtered_ann = 1;
  std::ifstream                         infile(map_file);
  std::string                           line, token;
  unsigned                              line_cnt = 0;
  tsl::robin_set<std::string>           _labels2;
  std::vector<std::vector<std::string>> _pts_to_labels2;

  std::map<std::string, size_t> _labels_counts;
  std::map<std::string, float>  _labels_distribution;
  std::mt19937                  _labels_rng_gen;

  while (std::getline(infile, line)) {
    std::istringstream       iss(line);
    std::vector<std::string> lbls(0);
    // long int              val;
    while (getline(iss, token, ',')) {
      token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
      token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
      lbls.push_back(token);
      _labels2.insert(token);
      _labels_counts[token]++;
      /*  if (_filter_to_medoid_id.find(token) == _filter_to_medoid_id.end())
        { _filter_to_medoid_id[token] = line_cnt;
        }*/
    }
    if (lbls.size() <= 0) {
      std::cout << "No label found";
      exit(-1);
    }
    std::sort(lbls.begin(), lbls.end());
    _pts_to_labels2.push_back(lbls);
    line_cnt++;
  }

  std::cout << "Getting labels distribution" << std::endl;
  const std::size_t total_label_occurrences =
      std::accumulate(std::begin(_labels_counts), std::end(_labels_counts), 0,
                      [](const std::size_t previous,
                         const std::pair<const std::string, std::size_t>& p) {
                        return previous + p.second;
                      });

  std::random_device dev;
  _labels_rng_gen = std::mt19937(dev());

  for (auto const& x : _labels_counts) {
    _labels_distribution[x.first] = x.second / line_cnt;
  }

  std::cout << "Identified " << _labels2.size() << " distinct label(s)"
            << std::endl;

  labels_dist = _labels_counts;
  labels_rng_gen = _labels_rng_gen;
  return (_labels2);
  /*    _u32 ctr = 0;
          for (const auto &x : _labels) {
            std::cout << ctr << ": " << x << " " << x.size() << std::endl;
            ctr++;
          } */
}

int main(int argc, char** argv) {
  if (argc != 12) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>]  [data_file.bin]  "
                 "[output_index_file]  "
              << "[R]  [L]  [alpha] [stitched_R]"
              << "  [num_threads_to_use] [label_file (use \"null\" for regular "
                 "unfiltered build)]  [use_universal_label] [universal_label]. "
                 "See README for more "
                 "information on "
                 "parameters."
              << std::endl;
    exit(-1);
  }

  const std::string data_type(argv[1]);
  const std::string data_path(argv[2]);
  const std::string save_path(argv[3]);
  const unsigned    R = (unsigned) atoi(argv[4]);
  const unsigned    L = (unsigned) atoi(argv[5]);
  const float       alpha = (float) atof(argv[6]);
  const unsigned    stitched_R = atoi(argv[7]);
  const unsigned    num_threads = (unsigned) atoi(argv[8]);
  std::string       label_file(argv[9]);
  bool              use_universal_label = (bool) atoi(argv[10]);
  std::string       universal_label = "";
  if (use_universal_label)
    universal_label = std::string(argv[11]);

  std::map<std::string, size_t> label_dist;
  std::mt19937                  label_rng_gen;

  tsl::robin_set<std::string> label_set =
      parse_label_file2(label_file, label_dist, label_rng_gen);
  tsl::robin_map<std::string, tsl::robin_map<_u64, _u64>> rev_map;

  if (data_type == "uint8")
    rev_map =
        filter_base<uint8_t>(data_type, data_path, label_file,
                             use_universal_label, universal_label, label_set);
  else if (data_type == "int8")
    rev_map =
        filter_base<int8_t>(data_type, data_path, label_file,
                            use_universal_label, universal_label, label_set);
  else if (data_type == "float")
    rev_map =
        filter_base<float>(data_type, data_path, label_file,
                           use_universal_label, universal_label, label_set);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;

  std::vector<tsl::robin_set<unsigned>> full_graph(1000000);
  tsl::robin_map<std::string, _u64>     eps;
  _u64                                  dup = 0, file_size = 0, ep;

  for (auto& filter_label : label_set) {
    if (filter_label == "0") {
      continue;
    }
    std::string save_path_label = save_path + filter_label;
    std::string data_path_label = data_path + filter_label;

    if (data_type == std::string("int8"))
      build_in_memory_index2<int8_t>(data_path_label, R, L, 0, alpha,
                                     save_path_label, num_threads);
    else if (data_type == std::string("uint8"))
      build_in_memory_index2<uint8_t>(data_path_label, R, L, 0, alpha,
                                      save_path_label, num_threads);
    else if (data_type == std::string("float"))
      build_in_memory_index2<float>(data_path_label, R, L, 0, alpha,
                                    save_path_label, num_threads);
    else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;

    std::vector<std::vector<unsigned>> filt_graph =
        load_in_memory_index(save_path_label.c_str(), file_size, ep);
    ep = random(0, filt_graph.size());
    eps[filter_label] = rev_map[filter_label][ep];
    std::cout << "Load is done for filter label " << filter_label << std::endl;

    /*for (auto& it : rev_map[filter_label]) {
      std::cout << it.first << " " << it.second << std::endl;
    }*/

    for (_u32 i = 0; i < filt_graph.size(); i++) {
      for (auto& nbr : filt_graph[i]) {
        /*for (auto& z : full_graph[rev_map[filter_label][i]])
          std::cout << "fgrmfi " << z << std::endl;*/
        // std::cout << nbr << std::endl;
        if (!full_graph[rev_map[filter_label][i]].count(
                rev_map[filter_label][nbr]))
          full_graph[rev_map[filter_label][i]].insert(
              rev_map[filter_label][nbr]);
        else
          dup++;
      }
    }
  }
  std::cout << "Full graph is done" << std::endl;

  if (data_type == std::string("int8"))
    file_size -= dup * sizeof(int8_t);
  else if (data_type == std::string("uint8"))
    file_size -= dup * sizeof(uint8_t);
  else if (data_type == std::string("float"))
    file_size -= dup * sizeof(float);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;

  if (data_type == std::string("int8")) {
    save_in_memory_index<int8_t>(save_path.c_str(), full_graph, eps,
                                 use_universal_label, universal_label.c_str(),
                                 label_file.c_str());
    prune_and_save<int8_t>(data_path, save_path, save_path + "_pruned",
                           stitched_R, label_dist, label_rng_gen);
  } else if (data_type == std::string("uint8")) {
    save_in_memory_index<uint8_t>(save_path.c_str(), full_graph, eps,
                                  use_universal_label, universal_label.c_str(),
                                  label_file.c_str());
    prune_and_save<uint8_t>(data_path, save_path, save_path + "_pruned",
                            stitched_R, label_dist, label_rng_gen);
  } else if (data_type == std::string("float")) {
    save_in_memory_index<float>(save_path.c_str(), full_graph, eps,
                                use_universal_label, universal_label.c_str(),
                                label_file.c_str());
    prune_and_save<float>(data_path, save_path, save_path + "_pruned",
                          stitched_R, label_dist, label_rng_gen);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
