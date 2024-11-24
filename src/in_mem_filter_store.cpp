#include "in_mem_filter_store.h"
#include "ann_exception.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include <exception>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace diskann
{
// TODO: Move to utils.h
DISKANN_DLLEXPORT std::unique_ptr<char[]> get_file_content(const std::string &filename, uint64_t &file_size);

template <typename LabelT> InMemFilterStore<LabelT>::~InMemFilterStore()
{
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
        _pts_to_label_offsets = nullptr;
    }
    if (_pts_to_label_counts != nullptr)
    {
        delete[] _pts_to_label_counts;
        _pts_to_label_counts = nullptr;
    }
    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
        _pts_to_labels = nullptr;
    }
}

template <typename LabelT>
const std::unordered_map<LabelT, std::vector<location_t>> &InMemFilterStore<LabelT>::get_label_to_medoids() const
{
    return this->_filter_to_medoid_ids;
}

template <typename LabelT>
const std::vector<location_t> &InMemFilterStore<LabelT>::get_medoids_of_label(const LabelT label)
{
    if (_filter_to_medoid_ids.find(label) != _filter_to_medoid_ids.end())
    {
        return this->_filter_to_medoid_ids[label];
    }
    else
    {
        std::stringstream ss;
        ss << "Could not find " << label << " in filters_to_medoid_ids map." << std::endl;
        diskann::cerr << ss.str();
        throw ANNException(ss.str(), -1);
    }
}

template <typename LabelT> void InMemFilterStore<LabelT>::set_universal_label(const LabelT univ_label)
{
    _universal_filter_label = univ_label;
    _use_universal_label = true;
}

// Load functions for SEARCH START
template <typename LabelT> bool InMemFilterStore<LabelT>::load(const std::string &disk_index_file)
{
    std::string labels_file = disk_index_file + "_labels.txt";
    std::string labels_to_medoids = disk_index_file + "_labels_to_medoids.txt";
    std::string dummy_map_file = disk_index_file + "_dummy_map.txt";
    std::string labels_map_file = disk_index_file + "_labels_map.txt";
    std::string univ_label_file = disk_index_file + "_universal_label.txt";

    return load(labels_file, labels_to_medoids, labels_map_file, univ_label_file, dummy_map_file);
}

template <typename LabelT> bool InMemFilterStore<LabelT>::load(
    const std::string& labels_filepath,
    const std::string& labels_to_medoids_filepath,
    const std::string& labels_map_filepath,
    const std::string& unv_label_filepath,
    const std::string& dummy_map_filepath)
{
    // TODO: Check for encoding issues here. We are opening files as binary and
    // reading them as bytes, not sure if that can cause an issue with UTF
    // encodings.
    bool has_filters = true;
    if (false == load_file_and_parse(labels_filepath, &InMemFilterStore<LabelT>::load_label_file))
    {
        diskann::cout << "Index does not have filter data. " << std::endl;
        return false;
    }
    if (false == parse_stream(labels_map_filepath, &InMemFilterStore<LabelT>::load_label_map))
    {
        diskann::cerr << "Failed to find file: " << labels_map_filepath << " while labels_file exists." << std::endl;
        return false;
    }

    if (false == parse_stream(labels_to_medoids_filepath, &InMemFilterStore<LabelT>::load_labels_to_medoids))
    {
        diskann::cerr << "Failed to find file: " << labels_to_medoids_filepath << " while labels file exists." << std::endl;
        return false;
    }
    // missing universal label file is NOT an error.
    load_file_and_parse(unv_label_filepath, &InMemFilterStore::parse_universal_label);

    // missing dummy map file is also NOT an error.
    parse_stream(dummy_map_filepath, &InMemFilterStore<LabelT>::load_dummy_map);
    _is_valid = true;
    return _is_valid;
}

template <typename LabelT> bool InMemFilterStore<LabelT>::has_filter_support() const
{
    return _is_valid;
}

template <typename LabelT>  bool InMemFilterStore<LabelT>::is_label_valid(const std::string& filter_label) const
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return true;
    }

    return false;
}

// TODO: Improve this to not load the entire file in memory
template <typename LabelT> void InMemFilterStore<LabelT>::load_label_file(const std::string_view &label_file_content)
{
    std::string line;
    uint32_t line_cnt = 0;

    uint32_t num_pts_in_label_file;
    uint32_t num_total_labels;
    get_label_file_metadata(label_file_content, num_pts_in_label_file, num_total_labels);

    _num_points = num_pts_in_label_file;

    _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
    _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
    _pts_to_labels = new LabelT[num_total_labels];
    uint32_t labels_seen_so_far = 0;

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    size_t file_size = label_file_content.size();

    while (cur_pos < file_size && cur_pos != std::string_view::npos)
    {
        next_pos = label_file_content.find('\n', cur_pos);
        if (next_pos == std::string_view::npos)
        {
            break;
        }

        _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
        uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
        num_lbls_in_cur_pt = 0;

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string_view::npos)
        {
            next_lbl_pos = search_string_range(label_file_content, ',', lbl_pos, next_pos);
            if (next_lbl_pos == std::string_view::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }

            // TODO: SHOULD NOT EXPECT label_file_content TO BE NULL_TERMINATED
            label_str.assign(label_file_content.data() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }

            LabelT token_as_num = (LabelT)std::stoul(label_str);
            _pts_to_labels[labels_seen_so_far++] = (LabelT)token_as_num;
            num_lbls_in_cur_pt++;

            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }

        // move to next line
        cur_pos = next_pos + 1;

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }

        line_cnt++;
    }

    // TODO: We need to check if the number of labels and the number of points
    // is as expected. Maybe add the check in PQFlashIndex?
    // num_points_labels = line_cnt;
}

template <typename LabelT>
void InMemFilterStore<LabelT>::load_labels_to_medoids(std::basic_istream<char> &medoid_stream)
{
    std::string line, token;

    _filter_to_medoid_ids.clear();
    while (std::getline(medoid_stream, line))
    {
        std::istringstream iss(line);
        uint32_t cnt = 0;
        std::vector<uint32_t> medoids;
        LabelT label;
        while (std::getline(iss, token, ','))
        {
            if (cnt == 0)
                label = (LabelT)std::stoul(token);
            else
                medoids.push_back((uint32_t)stoul(token));
            cnt++;
        }
        _filter_to_medoid_ids[label].swap(medoids);
    }
}

template <typename LabelT> void InMemFilterStore<LabelT>::load_label_map(std::basic_istream<char> &map_reader)
{
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        _label_map[label_str] = token_as_num;
    }
}

template <typename LabelT> void InMemFilterStore<LabelT>::parse_universal_label(const std::string_view &content)
{
    LabelT label_as_num = (LabelT)std::stoul(std::string(content));
    this->set_universal_label(label_as_num);
}

template <typename LabelT> void InMemFilterStore<LabelT>::load_dummy_map(std::basic_istream<char> &dummy_map_stream)
{
    std::string line, token;

    while (std::getline(dummy_map_stream, line))
    {
        std::istringstream iss(line);
        uint32_t cnt = 0;
        uint32_t dummy_id;
        uint32_t real_id;
        while (std::getline(iss, token, ','))
        {
            if (cnt == 0)
                dummy_id = (uint32_t)stoul(token);
            else
                real_id = (uint32_t)stoul(token);
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
void InMemFilterStore<LabelT>::generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                      const uint32_t nthreads)
{
    std::random_device rd;
    labels.clear();
    labels.resize(num_labels);

    uint64_t num_total_labels = _pts_to_label_offsets[_num_points - 1] + _pts_to_label_counts[_num_points - 1];
    std::mt19937 gen(rd());
    if (num_total_labels == 0)
    {
        std::stringstream stream;
        stream << "No labels found in data. Not sampling random labels ";
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::uniform_int_distribution<uint64_t> dis(0, num_total_labels - 1);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < num_labels; i++)
    {
        uint64_t rnd_loc = dis(gen);
        labels[i] = (LabelT)_pts_to_labels[rnd_loc];
    }
}

template <typename LabelT> void InMemFilterStore<LabelT>::reset_stream_for_reading(std::basic_istream<char> &infile)
{
    infile.clear();
    infile.seekg(0);
}

template <typename LabelT>
void InMemFilterStore<LabelT>::get_label_file_metadata(const std::string_view &fileContent, uint32_t &num_pts,
                                                       uint32_t &num_total_labels)
{
    num_pts = 0;
    num_total_labels = 0;

    size_t file_size = fileContent.length();

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = fileContent.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = search_string_range(fileContent, ',', lbl_pos, next_pos);
            if (next_lbl_pos == std::string::npos) // the last label
            {
                next_lbl_pos = next_pos;
            }

            num_total_labels++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
}

template <typename LabelT>
bool InMemFilterStore<LabelT>::parse_stream(const std::string &filename,
                                            void (InMemFilterStore::*parse_fn)(std::basic_istream<char> &stream))
{
    if (file_exists(filename))
    {
        std::ifstream stream(filename);
        if (false == stream.fail())
        {
            std::invoke(parse_fn, this, stream);
            return true;
        }
        else
        {
            std::stringstream ss;
            ss << "Could not open file: " << filename << std::endl;
            throw diskann::ANNException(ss.str(), -1);
        }
    }
    else
    {
        return false;
    }
}

template <typename LabelT>
bool InMemFilterStore<LabelT>::load_file_and_parse(const std::string &filename,
                                                   void (InMemFilterStore::*parse_fn)(const std::string_view &content))
{
    if (file_exists(filename))
    {
        size_t file_size = 0;
        auto file_content_ptr = get_file_content(filename, file_size);
        std::string_view content_as_str(file_content_ptr.get(), file_size);
        std::invoke(parse_fn, this, content_as_str);
        return true;
    }
    else
    {
        return false;
    }
}

template <typename LabelT>
size_t InMemFilterStore<LabelT>::search_string_range(const std::string_view& str, char ch, size_t start, size_t end)
{
    for (; start != end; start++)
    {
        if (str[start] == ch)
        {
            return start;
        }
    }

    return std::string::npos;

}

std::unique_ptr<char[]> get_file_content(const std::string &filename, uint64_t &file_size)
{
    std::ifstream infile(filename, std::ios::binary);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + filename, -1);
    }
    infile.seekg(0, std::ios::end);
    file_size = infile.tellg();

    auto buffer = new char[file_size];
    infile.seekg(0, std::ios::beg);
    infile.read(buffer, file_size);

    return std::unique_ptr<char[]>(buffer);
}

// Load functions for SEARCH END
template class InMemFilterStore<uint16_t>;
template class InMemFilterStore<uint32_t>;
template class InMemFilterStore<uint64_t>;

} // namespace diskann
