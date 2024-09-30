#include <vector>
#include <string_view>
#include <in_mem_filter_store.h>
#include <multi_filter/abstract_predicate.h>
#include <multi_filter/simple_boolean_predicate.h>

namespace diskann {
  template<typename LabelT>
  const std::vector<LabelT> &InMemFilterStore<LabelT>::get_filters_for_point(
      location_t point) const {
  }

  template<typename LabelT>
  void InMemFilterStore<LabelT>::add_filters_for_point(
      location_t point, const std::vector<LabelT> &filters) {
  }

  template<typename LabelT>
  float InMemFilterStore<LabelT>::get_predicate_selectivity(
      const AbstractPredicate &pred) const {
    return 0.0f;
  }

  template<typename LabelT>
  const std::unordered_map<LabelT, std::vector<location_t>>
      &InMemFilterStore<LabelT>::get_label_to_medoids() const {
  }

  template<typename LabelT>
  const std::vector<location_t> &InMemFilterStore<LabelT>::get_medoids_of_label(
      const LabelT label) const {
  }

  template<typename LabelT>
  void InMemFilterStore<LabelT>::set_universal_label(const LabelT univ_label) {
    _universal_filter_label = univ_label;
    _use_universal_label = true;
  }

  template<typename LabelT>
  inline bool InMemFilterStore<LabelT>::point_has_label(
      location_t point_id, const LabelT label_id) const {
    uint32_t start_vec = _pts_to_label_offsets[point_id];
    uint32_t num_lbls = _pts_to_label_counts[point_id];
    bool     ret_val = false;
    for (uint32_t i = 0; i < num_lbls; i++) {
      if (_pts_to_labels[start_vec + i] == label_id) {
        ret_val = true;
        break;
      }
    }
    return ret_val;
  }

  template<typename LabelT>
  inline bool InMemFilterStore<LabelT>::is_dummy_point(location_t id) const {
    return _dummy_pts.find(id) != _dummy_pts.end();
  }

  template<typename LabelT>
  inline bool InMemFilterStore<LabelT>::point_has_label_or_universal_label(
      location_t id, const LabelT filter_label) const {
    return point_has_label(id, filter_label) || 
            (_use_universal_label && point_has_label(id, _universal_filter_label));
  }

  template<typename LabelT>
  inline LabelT InMemFilterStore<LabelT>::get_converted_label(
      const std::string &filter_label) const {
    if (_label_map.find(filter_label) != _label_map.end()) {
      return _label_map[filter_label];
    }
    if (_use_universal_label) {
      return _universal_filter_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }

  // Load functions for SEARCH START
  template<typename LabelT>
  bool InMemFilterStore<LabelT>::load(const std::string &disk_index_file) {
    std::string labels_file = disk_index_file + "_labels.txt";
    std::string labels_to_medoids = disk_index_file + "_labels_to_medoids.txt";
    std::string dummy_map_file = disk_index_file + "_dummy_map.txt";
    std::string labels_map_file = disk_index_file + "_labels_map.txt";
    std::string univ_label_file = disk_index_file + "_universal_label.txt";

    size_t num_pts_in_label_file = 0;

    // TODO: Check for encoding issues here. We are opening files as binary and
    // reading them as bytes, not sure if that can cause an issue with UTF
    // encodings.
    bool has_filters = true;
    if (false == load_file_and_parse(labels_file, &load_label_file)) {
      diskann::cout << "Index does not have filter data. " << std::endl;
      return false;
    }
    if (false == load_file_and_parse(labels_map_file, &load_label_map)) {
      diskann::cerr << "Failed to find file: " << labels_map_file
                    << " while labels_file exists." << std::endl;
      return false;
    }
    if (false ==
        load_file_and_parse(labels_to_medoids, &load_labels_to_medoids)) {
      diskann::cerr << "Failed to find file: " << labels_to_medoids
                    << " while labels file exists." << std::endl;
      return false;
    }
    // missing universal label file is NOT an error.
    load_file_and_parse(univ_label_file,
                        [this](const std::string_view &content) {
                          label_as_num = (LabelT) std::strtoul(univ_label);
                          this->set_universal_label(label_as_num);
                        });

    // missing dummy map file is also NOT an error.
    load_file_and_parse(dummy_map_file, &load_dummy_map);
    return true;
  }


// TODO: Improve this to not load the entire file in memory
template<typename LabelT>
void InMemFilterStore<LabelT>::load_label_file(
    const std::string_view &label_file_content) {
  std::string line;
  uint32_t    line_cnt = 0;

  uint32_t num_pts_in_label_file;
  uint32_t num_total_labels;
  get_label_file_metadata(label_file_content, num_pts_in_label_file,
                          num_total_labels);

  _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
  _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
  _pts_to_labels = new LabelT[num_total_labels];
  uint32_t labels_seen_so_far = 0;

  std::string label_str;
  size_t      cur_pos = 0;
  size_t      next_pos = 0;
  while (cur_pos < file_size && cur_pos != std::string::npos) {
    next_pos = label_file_content.find('\n', cur_pos);
    if (next_pos == std::string::npos) {
      break;
    }

    _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
    uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
    num_lbls_in_cur_pt = 0;

    size_t lbl_pos = cur_pos;
    size_t next_lbl_pos = 0;
    while (lbl_pos < next_pos && lbl_pos != std::string::npos) {
      next_lbl_pos = label_file_content.find(',', lbl_pos);
      if (next_lbl_pos ==
          std::string::npos)  // the last label in the whole file
      {
        next_lbl_pos = next_pos;
      }

      if (next_lbl_pos >
          next_pos)  // the last label in one line, just read to the end
      {
        next_lbl_pos = next_pos;
      }

      label_str.assign(label_file_content.c_str() + lbl_pos,
                       next_lbl_pos - lbl_pos);
      if (label_str[label_str.length() - 1] ==
          '\t')  // '\t' won't exist in label file?
      {
        label_str.erase(label_str.length() - 1);
      }

      LabelT token_as_num = (LabelT) std::stoul(label_str);
      _pts_to_labels[labels_seen_so_far++] = (LabelT) token_as_num;
      num_lbls_in_cur_pt++;

      // move to next label
      lbl_pos = next_lbl_pos + 1;
    }

    // move to next line
    cur_pos = next_pos + 1;

    if (num_lbls_in_cur_pt == 0) {
      diskann::cout << "No label found for point " << line_cnt << std::endl;
      exit(-1);
    }

    line_cnt++;
  }

  num_points_labels = line_cnt;
}

template<typename LabelT>
void InMemFilterStore<LabelT>::load_labels_to_medoids(std::basic_istream<char>& medoid_stream) {
  std::string line, token;

  _filter_to_medoid_ids.clear();
  try {
    while (std::getline(medoid_stream, line)) {
      std::istringstream    iss(line);
      uint32_t              cnt = 0;
      std::vector<uint32_t> medoids;
      LabelT                label;
      while (std::getline(iss, token, ',')) {
        if (cnt == 0)
          label = (LabelT) std::stoul(token);
        else
          medoids.push_back((uint32_t) stoul(token));
        cnt++;
      }
      _filter_to_medoid_ids[label].swap(medoids);
    }
  } catch (std::system_error &e) {
    throw FileException(labels_to_medoids, e, __FUNCSIG__, __FILE__, __LINE__);
  }
}

template<typename LabelT>
void InMemFilterStore<LabelT>::load_label_map(
    std::basic_istream<char> &map_reader) {
  std::string line, token;
  LabelT      token_as_num;
  std::string label_str;
  while (std::getline(map_reader, line)) {
    std::istringstream iss(line);
    getline(iss, token, '\t');
    label_str = token;
    getline(iss, token, '\t');
    token_as_num = (LabelT) std::stoul(token);
    _label_map[label_str] = token_as_num;
  }
  return _label_map;
}

template<typename LabelT>
void InMemFilterStore<LabelT>::load_dummy_map(
    std::basic_istream<char> &dummy_map_stream) {
  std::string line, token;

  while (std::getline(dummy_map_stream, line)) {
    std::istringstream iss(line);
    uint32_t           cnt = 0;
    uint32_t           dummy_id;
    uint32_t           real_id;
    while (std::getline(iss, token, ',')) {
      if (cnt == 0)
        dummy_id = (uint32_t) stoul(token);
      else
        real_id = (uint32_t) stoul(token);
      cnt++;
    }
    _dummy_pts.insert(dummy_id);
    _has_dummy_pts.insert(real_id);
    _dummy_to_real_map[dummy_id] = real_id;

    if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
      _real_to_dummy_map[real_id] = std::vector<uint32_t>();

    _real_to_dummy_map[real_id].emplace_back(dummy_id);
  }
  diskann::cout << "Loaded dummy map" << std::endl;
}


template <typename LabelT>
bool InMemFilterStore<LabelT>::load_file_and_parse(const std::string &filename,
                    void (*parse_fn)(std::basic_istream<char> &stream)) {
  if (file_exists(filename)) {
    std::basic_istream<char> stream(filename);
    if (false == stream.fail()) {
      parse_fn(stream);
      return true;
    } else {
      std::stringstream ss; 
      ss << "Could not open file: " << filename << std::endl;
      throw diskann::ANNException(ss.str(), -1);
    }
  } else {
    return false;
  }
}

template<typename LabelT> 
bool InMemFilterStore<LabelT>::load_file_and_parse(
    const std::string &filename,
    void (*parse_fn)(const std::string_view &content)) {
  if (file_exists(filename)) {
    size_t           file_size = 0;
    auto             file_content_ptr = get_file_content(filename, file_size);
    std::string_view content_as_str(file_content_ptr.get(), file_size);
    parse_fn(content_as_str);
    return true;
  } else {
    return false;
  }
}

template<typename LabelT>
std::unique_ptr<char[]> get_file_content(const std::string &filename,
                                         uint64_t &    file_size) {
  std::ifstream infile(filename, std::ios::binary);
  if (infile.fail()) {
    throw diskann::ANNException(std::string("Failed to open file ") + filename,
                                -1);
  }
  infile.seekg(0, std::ios::end);
  file_size = infile.tellg();

  buffer = new char[file_size];
  infile.seekg(0, std::ios::beg);
  infile.read(buffer, file_size);

  return std::unique_ptr<char[]>(buffer);
}

// Load functions for SEARCH END
}


/*
  template<typename LabelT>
#ifdef EXEC_ENV_OLS
  bool InMemFilterStore<LabelT>::load(MemoryMappedFiles &files,
                                      const std::string &disk_index_prefix) {
#else
  bool InMemFilterStore<LabelT>::load(const std::string &label_files_prefix) {
#endif
    std::string labels_file = std ::string(_disk_index_file) + "_labels.txt";
    std::string labels_to_medoids =
        std ::string(_disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file =
        std ::string(_disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file =
        std ::string(_disk_index_file) + "_labels_map.txt";
    size_t num_pts_in_label_file = 0;

    // TODO: Ideally we don't want to read entire data files into memory for
    // processing them. Fortunately for us, the most restrictive client in terms
    // of runtime memory already loads the data into blobs. So we'll go ahead
    // and do the same. But this needs to be fixed, maybe with separate code
    // paths.
    // TODO: Check for encoding issues here. We are opening files as binary and
    // reading them as bytes, not sure if that can cause an issue with UTF
    // encodings.
    bool has_filters = true;
#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_file)) {
      FileContent &     content_labels = files.getContent(labels_file);
      std::stringstream infile(std::string(
          (const char *) content_labels._content, content_labels._size));
#else
    if (file_exists(labels_file)) {
      size_t file_size;
      auto   label_file_content = get_file_content(labels_file, file_size);
      std::string_view content_as_str(label_file_content.get(), file_size);
#endif
      load_label_file(content_as_str);
      assert(num_pts_in_label_file == this->_num_points);
    } else {
      diskann::cerr << "Index does not have filter data." << std::endl;
      return false;
    }

    // If we have come here, it means that the labels_file exists. This means
    // the other files must also exist, and them missing is a bug.
#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_map_file)) {
      FileContent &     content_labels_map = files.getContent(labels_map_file);
      std::stringstream map_reader(
          std::string((const char *) content_labels_map._content,
                      content_labels_map._size));
#else
    if (file_exists(labels_map_file)) {
      std::ifstream map_reader(labels_map_file);
#endif
      _label_map = load_label_map(map_reader);

#ifndef EXEC_ENV_OLS
      map_reader.close();
#endif
    } else {
      std::stringstream ss;
      ss << "Index is filter enabled (labels file exists) but label map file: "
         << labels_map_file << " could not be opened";
      diskann::cerr << ss.str() << std::endl;
      throw diskann::ANNException(ss.str(), -1);
    }

#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_to_medoids)) {
      FileContent &content_labels_to_meoids =
          files.getContent(labels_to_medoids);
      std::stringstream medoid_stream(
          std::string((const char *) content_labels_to_meoids._content,
                      content_labels_to_meoids._size));
#else
    if (file_exists(labels_to_medoids)) {
      std::ifstream medoid_stream(labels_to_medoids);
      assert(medoid_stream.is_open());
#endif
      load_labels_to_medoids(medoid_stream);
    }
    std::string univ_label_file =
        std ::string(_disk_index_file) + "_universal_label.txt";

#ifdef EXEC_ENV_OLS
    if (files.fileExists(univ_label_file)) {
      FileContent &     content_univ_label = files.getContent(univ_label_file);
      std::stringstream universal_label_reader(
          std::string((const char *) content_univ_label._content,
                      content_univ_label._size));
#else
    if (file_exists(univ_label_file)) {
      std::ifstream universal_label_reader(univ_label_file);
      assert(universal_label_reader.is_open());
#endif
      std::string univ_label;
      universal_label_reader >> univ_label;
#ifndef EXEC_ENV_OLS
      universal_label_reader.close();
#endif
      LabelT label_as_num = (LabelT) std::stoul(univ_label);
      set_universal_label(label_as_num);
    }

#ifdef EXEC_ENV_OLS
    if (files.fileExists(dummy_map_file)) {
      FileContent &     content_dummy_map = files.getContent(dummy_map_file);
      std::stringstream dummy_map_stream(std::string(
          (const char *) content_dummy_map._content, content_dummy_map._size));
#else
    if (file_exists(dummy_map_file)) {
      std::ifstream dummy_map_stream(dummy_map_file);
      assert(dummy_map_stream.is_open());
#endif
      std::string line, token;

      while (std::getline(dummy_map_stream, line)) {
        std::istringstream iss(line);
        uint32_t           cnt = 0;
        uint32_t           dummy_id;
        uint32_t           real_id;
        while (std::getline(iss, token, ',')) {
          if (cnt == 0)
            dummy_id = (uint32_t) stoul(token);
          else
            real_id = (uint32_t) stoul(token);
          cnt++;
        }
        _dummy_pts.insert(dummy_id);
        _has_dummy_pts.insert(real_id);
        _dummy_to_real_map[dummy_id] = real_id;

        if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
          _real_to_dummy_map[real_id] = std::vector<uint32_t>();

        _real_to_dummy_map[real_id].emplace_back(dummy_id);
      }
#ifndef EXEC_ENV_OLS
      dummy_map_stream.close();
#endif
      diskann::cout << "Loaded dummy map" << std::endl;
    }
  }

}
*/