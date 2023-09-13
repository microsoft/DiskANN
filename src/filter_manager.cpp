#include <filter_manager.h>

namespace diskann
{

template <typename label_type>
FilterManager<label_type>::FilterManager(const size_t num_points) : _num_points(num_points)
{
    _pts_to_labels.resize(num_points);
}

template <typename label_type>
bool FilterManager<label_type>::detect_common_filters(uint32_t point_id, bool search_invocation,
                                                      const std::vector<label_type> &incoming_labels)
{
    auto &curr_node_labels = _pts_to_labels[point_id];
    std::vector<label_type> common_filters;
    std::set_intersection(incoming_labels.begin(), incoming_labels.end(), curr_node_labels.begin(),
                          curr_node_labels.end(), std::back_inserter(common_filters));
    if (common_filters.size() > 0)
    {
        // This is to reduce the repetitive calls. If common_filters size is > 0 ,
        // we dont need to check further for universal label
        return true;
    }
    if (_use_universal_label)
    {
        if (!search_invocation)
        {
            if (std::find(incoming_labels.begin(), incoming_labels.end(), _universal_label) != incoming_labels.end() ||
                std::find(curr_node_labels.begin(), curr_node_labels.end(), _universal_label) != curr_node_labels.end())
                common_filters.push_back(_universal_label);
        }
        else
        {
            if (std::find(curr_node_labels.begin(), curr_node_labels.end(), _universal_label) != curr_node_labels.end())
                common_filters.push_back(_universal_label);
        }
    }
    return (common_filters.size() > 0);
}

template <typename label_type>
const std::vector<label_type> &FilterManager<label_type>::get_labels_by_point_id(const location_t point_id)
{
    return _pts_to_labels[point_id];
}

template <typename label_type> const tsl::robin_set<label_type> &FilterManager<label_type>::get_all_label_set()
{
    return _labels;
}

template <typename label_type>
void FilterManager<label_type>::update_medoid_by_label(const label_type &label, const uint32_t new_medoid)
{
    _label_to_medoid_id[label] = new_medoid;
}

template <typename label_type> const uint32_t &FilterManager<label_type>::get_medoid_by_label(const label_type &label)
{
    return _label_to_medoid_id[label];
}

template <typename label_type> bool FilterManager<label_type>::label_has_medoid(const label_type &label)
{
    return _label_to_medoid_id.find(label) != _label_to_medoid_id.end();
}

template <typename label_type>
void FilterManager<label_type>::add_label_to_point(const location_t point_id, label_type label)
{
    _pts_to_labels[point_id].emplace_back(label);
}

template <typename label_type> void FilterManager<label_type>::set_universal_label(label_type universal_label)
{
    _universal_label = universal_label; // remove this when multiple labels are supported
    _use_universal_label = true;
    //_universal_labels_set.insert(universal_label); // when we support multiple universal labels
}

template <typename label_type> const label_type FilterManager<label_type>::get_universal_label() const
{
    // for now there is only one universal label, so return the first one
    // return *_universal_labels_set.begin();
    return _universal_label;
}

template <typename label_type> size_t FilterManager<label_type>::load_labels(const std::string &labels_file)
{
    // parse the generated label file
    return parse_label_file(labels_file);
}

template <typename label_type> size_t FilterManager<label_type>::load_medoids(const std::string &labels_to_medoid_file)
{
    if (file_exists(labels_to_medoid_file))
    {
        std::ifstream medoid_stream(labels_to_medoid_file);
        std::string line, token;
        uint32_t line_cnt = 0;

        _label_to_medoid_id.clear();
        while (std::getline(medoid_stream, line))
        {
            std::istringstream iss(line);
            uint32_t cnt = 0;
            uint32_t medoid = 0;
            label_type label;
            while (std::getline(iss, token, ','))
            {
                token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                label_type token_as_num = (label_type)std::stoul(token);
                if (cnt == 0)
                    label = token_as_num;
                else
                    medoid = token_as_num;
                cnt++;
            }
            _label_to_medoid_id[label] = medoid;
            line_cnt++;
        }
        return (size_t)line_cnt;
    }
    throw ANNException("ERROR: can not load medoids file does not exist", -1);
}

template <typename label_type> void FilterManager<label_type>::load_label_map(const std::string &labels_map_file)
{
    std::ifstream map_reader(labels_map_file);
    std::string line, token;
    label_type token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (label_type)std::stoul(token);
        _label_map[label_str] = token_as_num;
    }
}

template <typename label_type>
void FilterManager<label_type>::save_labels(const std::string &save_path_prefix, const size_t total_points)
{

    if (_pts_to_labels.size() > 0)
    {
        std::ofstream label_writer(save_path_prefix + "_labels.txt");
        assert(label_writer.is_open());
        for (uint32_t i = 0; i < total_points; i++)
        {
            for (uint32_t j = 0; j < (_pts_to_labels[i].size() - 1); j++)
            {
                label_writer << _pts_to_labels[i][j] << ",";
            }
            if (_pts_to_labels[i].size() != 0)
                label_writer << _pts_to_labels[i][_pts_to_labels[i].size() - 1];
            label_writer << std::endl;
        }
        label_writer.close();
    }

    if (_use_universal_label)
    {
        std::ofstream universal_label_writer(save_path_prefix + "_universal_label.txt");
        assert(universal_label_writer.is_open());
        universal_label_writer << _universal_label << std::endl;
        /*for (auto label : _universal_labels_set)
        {
            universal_label_writer << label << std::endl;
        }*/
        universal_label_writer.close();
    }
}

template <typename label_type> void FilterManager<label_type>::save_medoids(const std::string &save_path)
{
    if (_label_to_medoid_id.size() > 0)
    {
        std::ofstream medoid_writer(save_path);
        if (medoid_writer.fail())
        {
            throw diskann::ANNException(std::string("Failed to open medoid file ") + save_path, -1);
        }
        for (auto iter : _label_to_medoid_id)
        {
            medoid_writer << iter.first << ", " << iter.second << std::endl;
        }
        medoid_writer.close();
    }
}

template <typename label_type> label_type FilterManager<label_type>::get_converted_label(const std::string &raw_label)
{
    if (_label_map.empty())
    {
        throw diskann::ANNException("Error: Label map is empty, please load the map before hand", -1);
    }
    if (_label_map.find(raw_label) != _label_map.end())
    {
        return _label_map[raw_label];
    }
    if (_use_universal_label)
    {
        return _universal_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename label_type>
void FilterManager<label_type>::calculate_best_medoids(const size_t num_points_to_load, const uint32_t num_candidates)
{
    std::unordered_map<label_type, std::vector<uint32_t>> label_to_points;

    for (uint32_t point_id = 0; point_id < num_points_to_load; point_id++)
    {
        for (auto label : _pts_to_labels[point_id])
        {
            if (label != _universal_label)
            {
                label_to_points[label].emplace_back(point_id);
            }
            else
            {
                for (typename tsl::robin_set<label_type>::size_type lbl = 0; lbl < _labels.size(); lbl++)
                {
                    auto itr = _labels.begin();
                    std::advance(itr, lbl);
                    auto &x = *itr;
                    label_to_points[x].emplace_back(point_id);
                }
            }
        }
    }

    uint32_t num_cands = num_candidates;
    for (auto itr = _labels.begin(); itr != _labels.end(); itr++)
    {
        uint32_t best_medoid_count = std::numeric_limits<uint32_t>::max();
        auto &curr_label = *itr;
        uint32_t best_medoid;
        auto labeled_points = label_to_points[curr_label];
        for (uint32_t cnd = 0; cnd < num_cands; cnd++)
        {
            uint32_t cur_cnd = labeled_points[rand() % labeled_points.size()];
            uint32_t cur_cnt = std::numeric_limits<uint32_t>::max();
            if (_medoid_counts.find(cur_cnd) == _medoid_counts.end())
            {
                _medoid_counts[cur_cnd] = 0;
                cur_cnt = 0;
            }
            else
            {
                cur_cnt = _medoid_counts[cur_cnd];
            }
            if (cur_cnt < best_medoid_count)
            {
                best_medoid_count = cur_cnt;
                best_medoid = cur_cnd;
            }
        }
        _label_to_medoid_id[curr_label] = best_medoid;
        _medoid_counts[best_medoid]++;
    }
}

template <typename label_type> size_t FilterManager<label_type>::parse_label_file(const std::string &label_file)
{
    // Format of Label txt file: filters with comma separators
    std::ifstream infile(label_file);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
    }

    std::string line, token;
    uint32_t line_cnt = 0;

    while (std::getline(infile, line))
    {
        line_cnt++;
    }
    _pts_to_labels.resize(line_cnt, std::vector<label_type>());

    infile.clear();
    infile.seekg(0, std::ios::beg);
    line_cnt = 0;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<label_type> lbls(0);
        getline(iss, token, '\t');
        std::istringstream new_iss(token);
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            label_type token_as_num = (label_type)std::stoul(token);
            lbls.push_back(token_as_num);
            _labels.insert(token_as_num);
        }
        if (lbls.size() <= 0)
        {
            diskann::cout << "No label found on line: " << line << std::endl;
            exit(-1);
        }
        std::sort(lbls.begin(), lbls.end());
        _pts_to_labels[line_cnt] = lbls;
        line_cnt++;
    }
    diskann::cout << "Identified " << _labels.size() << " distinct label(s)" << std::endl;
    return (size_t)line_cnt;
}

template DISKANN_DLLEXPORT class FilterManager<uint32_t>;
template DISKANN_DLLEXPORT class FilterManager<uint16_t>;

} // namespace diskann